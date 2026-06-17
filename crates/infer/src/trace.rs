//! The memoised type/phase trace over a module's binding DAG.
//!
//! Bindings are visited in source order; references recurse (FlatPPL is
//! order-irrelevant) with the side-tables doubling as the memo. A reference
//! cycle is an error (the module is not a DAG); the offending binding gets a
//! `(%failed …)` type so the gap is visible in annotated output.

use std::collections::{HashMap, HashSet};

use flatppl_core::{
    BindingId, Call, CallHead, Module, NamedKind, Node, NodeId, Phase, RefNs, Scalar, Symbol, Type,
    ValueSet,
};

use crate::modules::InferSession;
use crate::ops;
use crate::{Diagnostic, Level};

pub(crate) struct Inferencer<'m, 's> {
    pub(crate) module: &'m mut Module,
    pub(crate) level: Level,
    /// Spans the dependency bundle; read by the `RefNs::Module` arm to resolve
    /// cross-module references.
    pub(crate) session: &'s InferSession<'s>,
    pub(crate) diags: Vec<Diagnostic>,
    /// Inferred types/phases/value-sets, local until the final level-aware
    /// flush (a `Level::Phase` run computes types internally but never
    /// annotates them).
    tys: HashMap<NodeId, Type>,
    phases: HashMap<NodeId, Phase>,
    vsets: HashMap<NodeId, ValueSet>,
    /// Pre-seeded annotations for substituted input nodes: (type, phase,
    /// valueset). Applied at the top of `infer_node` before any other logic,
    /// making the substituted input authoritative for everything downstream.
    seeds: HashMap<NodeId, (Type, Phase, ValueSet)>,
    /// Bindings on the active resolution path (cycle detection).
    in_progress: Vec<BindingId>,
    /// Ops already reported as catalogue gaps (one note per op).
    noted_gaps: HashSet<Symbol>,
    /// For a cross-module callable reference (`helpers.obs_kernel`), the
    /// dependency's inferred body-result type, keyed by the importer ref node.
    /// `reified_result_type` consults this so applying a cross-module callable
    /// reaches its body type — the body lives in the dependency's interner and
    /// cannot be looked up by node here. Populated in the `RefNs::Module` arm.
    module_callable_results: HashMap<NodeId, Type>,
    /// For a §09 standard-module reference (`hepphys.CrystalBall`), the
    /// catalogue signature of the referenced binding, keyed by the importer ref
    /// node. The user-call path consults this so that applying the reference
    /// (`hepphys.CrystalBall(args)`) lowers the catalogue sig with the concrete
    /// call args — the bare ref node itself types as `Type::Any` (matching a
    /// bare base name). Populated in the `RefNs::Module` arm.
    module_catalogue_refs: HashMap<NodeId, crate::modules::CatalogueRef>,
}

impl<'m, 's> Inferencer<'m, 's> {
    pub(crate) fn new(module: &'m mut Module, level: Level, session: &'s InferSession<'s>) -> Self {
        Inferencer {
            module,
            level,
            session,
            diags: Vec::new(),
            tys: HashMap::new(),
            phases: HashMap::new(),
            vsets: HashMap::new(),
            seeds: HashMap::new(),
            in_progress: Vec::new(),
            noted_gaps: HashSet::new(),
            module_callable_results: HashMap::new(),
            module_catalogue_refs: HashMap::new(),
        }
    }

    /// Like `new`, but pre-seeds the given node annotations so that
    /// substituted input nodes carry their importer-context types into the
    /// dependency walk.
    pub(crate) fn new_seeded(
        module: &'m mut Module,
        level: Level,
        session: &'s InferSession<'s>,
        seeds: &[(NodeId, crate::modules::Resolved)],
    ) -> Self {
        let seed_map = seeds
            .iter()
            .map(|(id, r)| (*id, (r.ty.clone(), r.phase, r.vset.clone())))
            .collect();
        Inferencer {
            module,
            level,
            session,
            diags: Vec::new(),
            tys: HashMap::new(),
            phases: HashMap::new(),
            vsets: HashMap::new(),
            seeds: seed_map,
            in_progress: Vec::new(),
            noted_gaps: HashSet::new(),
            module_callable_results: HashMap::new(),
            module_catalogue_refs: HashMap::new(),
        }
    }

    pub(crate) fn run(mut self) -> Vec<Diagnostic> {
        let ids: Vec<BindingId> = self.module.bindings().map(|(id, _)| id).collect();
        for id in ids {
            self.infer_binding(id);
        }
        // Level-aware flush into the module's annotation side-tables.
        for (&id, phase) in &self.phases {
            self.module.set_phase(id, *phase);
        }
        if self.level >= Level::Type {
            for (&id, ty) in &self.tys {
                self.module.set_type(id, ty.clone());
            }
        }
        if self.level >= Level::Valueset {
            // Total discipline (spec §11): a value-typed node's set is at
            // least the type's natural extent — fall back where no producer
            // established anything finer. One chokepoint; producers stay
            // refinement-only.
            for (&id, ty) in &self.tys {
                let stored = self.vsets.get(&id);
                if stored.is_none() || stored == Some(&ValueSet::Unknown) {
                    let natural = ValueSet::natural_of(ty);
                    if natural != ValueSet::Unknown {
                        self.vsets.insert(id, natural);
                    }
                }
            }
            for (&id, set) in &self.vsets {
                self.module.set_valueset(id, set.clone());
            }
        }
        // Drain dependency diagnostics accumulated during this run (cross-module
        // cycle errors and other dep-level errors). `infer_dep` stores child
        // diagnostics here after each dependency walk; draining ensures they
        // propagate up through every level of the import chain.
        self.diags.extend(self.session.drain_dep_diags());
        self.diags
    }

    /// The inferred type of an already-visited node (ops rules use this to
    /// look through reified bodies).
    pub(crate) fn lookup_type(&self, id: NodeId) -> Option<&Type> {
        self.tys.get(&id)
    }

    /// The cross-module callable body-result type recorded for `id`, if `id`
    /// is a `RefNs::Module` reference to a reified-callable binding. `None`
    /// for local callables (their body is looked up by node instead).
    pub(crate) fn module_callable_result(&self, id: NodeId) -> Option<&Type> {
        self.module_callable_results.get(&id)
    }

    /// The catalogue signature recorded for `id`, if `id` is a `RefNs::Module`
    /// reference resolved to a §09 standard-module binding. `None` otherwise.
    /// The user-call path reads this to lower the sig with concrete call args.
    pub(crate) fn module_catalogue_ref(&self, id: NodeId) -> Option<&crate::modules::CatalogueRef> {
        self.module_catalogue_refs.get(&id)
    }

    /// The inferred value set of an already-visited node (`Unknown` when the
    /// walk has not established one).
    pub(crate) fn lookup_valueset(&self, id: NodeId) -> ValueSet {
        self.vsets.get(&id).cloned().unwrap_or(ValueSet::Unknown)
    }

    /// Record a node's value set (no-op below `Level::Valueset`).
    pub(crate) fn set_vset(&mut self, id: NodeId, set: ValueSet) {
        if self.level >= Level::Valueset {
            self.vsets.insert(id, set);
        }
    }

    /// Type + phase of a binding's RHS (memoised).
    pub(crate) fn infer_binding(&mut self, id: BindingId) -> (Type, Phase) {
        let rhs = self.module.binding(id).rhs;
        if let (Some(ty), Some(phase)) = (self.tys.get(&rhs), self.phases.get(&rhs)) {
            return (ty.clone(), *phase);
        }
        if self.in_progress.contains(&id) {
            let path: Vec<&str> = self
                .in_progress
                .iter()
                .map(|&b| self.module.resolve(self.module.binding(b).name))
                .collect();
            let name = self
                .module
                .resolve(self.module.binding(id).name)
                .to_string();
            self.diags.push(Diagnostic::error(format!(
                "binding `{name}` is part of a reference cycle ({})",
                path.join(" → ")
            )));
            let ty = Type::Failed("reference cycle".into());
            self.tys.insert(rhs, ty.clone());
            self.phases.insert(rhs, Phase::Fixed);
            return (ty, Phase::Fixed);
        }
        self.in_progress.push(id);
        let result = self.infer_node(rhs);
        self.in_progress.pop();
        result
    }

    /// Type + phase of a node (memoised). Every node is traced; the flush
    /// annotates the module per level, and the FlatPIR writer projects
    /// `%meta` at call positions only (spec §11).
    pub(crate) fn infer_node(&mut self, id: NodeId) -> (Type, Phase) {
        // Substitution seed: if this node was pre-seeded by a %assign
        // substitution, write its annotation directly and return — making the
        // input authoritative for everything downstream that references it.
        if let Some((ty, phase, vset)) = self.seeds.get(&id).cloned() {
            self.tys.insert(id, ty.clone());
            self.phases.insert(id, phase);
            if self.level >= Level::Valueset {
                self.vsets.insert(id, vset);
            }
            return (ty, phase);
        }
        if let (Some(ty), Some(phase)) = (self.tys.get(&id), self.phases.get(&id)) {
            return (ty.clone(), *phase);
        }
        // Clone the node to release the module borrow during recursion; nodes
        // are small (boxed slices of ids).
        let node = self.module.node(id).clone();
        let (ty, phase) = match &node {
            Node::Lit(s) => {
                self.set_vset(id, ops::literal_valueset(s));
                (ops::literal_type(s), Phase::Fixed)
            }
            Node::Const(sym) => {
                let name = self.module.resolve(*sym).to_string();
                self.set_vset(id, ops::const_valueset(&name));
                (ops::const_type(&name), Phase::Fixed)
            }
            Node::Ref(r) => match r.ns {
                RefNs::SelfMod => match self.module.binding_by_name(r.name) {
                    Some(b) => {
                        let result = self.infer_binding(b);
                        let rhs = self.module.binding(b).rhs;
                        let set = self.lookup_valueset(rhs);
                        self.set_vset(id, set);
                        result
                    }
                    None => {
                        let name = self.module.resolve(r.name).to_string();
                        self.diags.push(Diagnostic::error_at(
                            id,
                            format!("unresolved reference `{name}`"),
                        ));
                        (Type::Failed("unresolved reference".into()), Phase::Fixed)
                    }
                },
                // A placeholder is implicitly `elementof(anything)` (spec §04
                // "Placeholder variables") — unconstrained, parameterized.
                RefNs::Local => {
                    self.set_vset(id, ValueSet::Anything);
                    (Type::Any, Phase::Parameterized)
                }
                RefNs::Module(alias) => {
                    let binding_name = self.module.resolve(r.name).to_string();
                    let subst_annos = self.subst_annos_for(alias);
                    match self.session.resolve(
                        &*self.module,
                        alias,
                        &binding_name,
                        &subst_annos,
                        self.level,
                    ) {
                        Ok(res) => {
                            // Stash the dependency's callable body-result type
                            // (if any) keyed by this ref node, so applying the
                            // callable (`reified_result_type`) can reach it
                            // across the interner boundary.
                            if let Some(result) = res.result {
                                self.module_callable_results.insert(id, result);
                            }
                            // Stash the §09 catalogue sig (if any) keyed by this
                            // ref node, so the user-call path can lower it with
                            // the concrete call args when the ref is applied.
                            if let Some(catalogue) = res.catalogue {
                                self.module_catalogue_refs.insert(id, catalogue);
                            }
                            self.set_vset(id, res.vset);
                            (res.ty, res.phase)
                        }
                        Err(message) => {
                            self.diags.push(crate::Diagnostic::error_at(id, message));
                            (Type::Failed("cross-module resolution".into()), Phase::Fixed)
                        }
                    }
                }
            },
            Node::Hole | Node::Axis(_) => {
                self.set_vset(id, ValueSet::Unknown);
                (Type::Any, Phase::Fixed)
            }
            Node::Call(call) => self.infer_call(id, call),
        };
        // A cycle marker may have landed on this node while the walk was in
        // flight (see infer_binding); it is authoritative — don't clobber it.
        if let (Some(t), Some(p)) = (self.tys.get(&id), self.phases.get(&id)) {
            return (t.clone(), *p);
        }
        self.tys.insert(id, ty.clone());
        self.phases.insert(id, phase);
        (ty, phase)
    }

    fn infer_call(&mut self, id: NodeId, call: &Call) -> (Type, Phase) {
        // Children first: callee (user calls), positional, named.
        let callee = match call.head {
            CallHead::User(callee) => Some((callee, self.infer_node(callee))),
            CallHead::Builtin(_) => None,
        };
        let args: Vec<(NodeId, Type, Phase)> = call
            .args
            .iter()
            .map(|&a| {
                let (t, p) = self.infer_node(a);
                (a, t, p)
            })
            .collect();
        let named: Vec<(Symbol, NodeId, Type, Phase)> = call
            .named
            .iter()
            .map(|n| {
                let (t, p) = self.infer_node(n.value);
                (n.name, n.value, t, p)
            })
            .collect();

        self.validate_load_assigns(call);

        // The §04 ancestor rule: a call's phase is the join of its inputs'
        // phases, except where the op itself introduces a phase.
        let joined = callee
            .iter()
            .map(|(_, (_, p))| *p)
            .chain(args.iter().map(|(_, _, p)| *p))
            .chain(named.iter().map(|(_, _, _, p)| *p))
            .fold(Phase::Fixed, join_phase);

        let callee = callee.map(|(n, tp)| (n, tp.0));
        let (ty, phase) = ops::call_rule(self, id, call, callee.clone(), &args, &named, joined);
        if self.level >= Level::Valueset {
            let set = ops::call_valueset(self, call, callee.as_ref(), &args, &named, &ty);
            self.set_vset(id, set);
        }
        let ty = if self.level >= Level::Normalization {
            ops::fill_mass(self, id, call, callee.as_ref(), ty, &args, &named)
        } else {
            ty
        };
        (ty, phase)
    }

    /// Validate `%assign` substitution names on a `load_module` /
    /// `standard_module` call. For each named `%assign` arg whose name is not
    /// a known binding in the dependency, emits an anchored error on the
    /// argument value node. No-op for every other call head or builtin op.
    ///
    /// The data needed from `call` is cloned out first so the borrow is
    /// released before `self.diags.push` or `self.module.resolve` are called.
    fn validate_load_assigns(&mut self, call: &Call) {
        let load_check: Option<(String, Vec<(Symbol, NodeId)>)> =
            if let CallHead::Builtin(head) = call.head {
                let head_name = self.module.resolve(head).to_string();
                if matches!(head_name.as_str(), "load_module" | "standard_module") {
                    let path = call.args.first().and_then(|&a| {
                        if let Node::Lit(Scalar::Str(s)) = self.module.node(a) {
                            Some(s.to_string())
                        } else {
                            None
                        }
                    });
                    let assigns: Vec<(Symbol, NodeId)> = call
                        .named
                        .iter()
                        .filter(|n| n.kind == NamedKind::Assign)
                        .map(|n| (n.name, n.value))
                        .collect();
                    path.map(|p| (p, assigns))
                } else {
                    None
                }
            } else {
                None
            };
        // `call` borrow is fully released — safe to push diagnostics.
        if let Some((path, assigns)) = load_check {
            if let Some(dep) = self.session.bundle.get(&path) {
                for (name_sym, value_node) in assigns {
                    let input = self.module.resolve(name_sym).to_string();
                    let known = dep.bindings().any(|(_, b)| dep.resolve(b.name) == input);
                    if !known {
                        self.diags.push(Diagnostic::error_at(
                            value_node,
                            format!("module `{path}` has no input `{input}`"),
                        ));
                    }
                }
            }
        }
    }

    /// Record a catalogue gap for `op`, once. Phase-only runs skip these —
    /// type gaps are irrelevant when types are not requested.
    pub(crate) fn note_gap(&mut self, op: Symbol) {
        if self.level == Level::Phase {
            return;
        }
        if self.noted_gaps.insert(op) {
            let name = self.module.resolve(op).to_string();
            self.diags.push(Diagnostic::note(format!(
                "no type rule for `{name}` yet — its calls are left %deferred"
            )));
        }
    }

    /// Record a note, once per distinct message.
    pub(crate) fn note_once_str(&mut self, message: &str) {
        if !self.diags.iter().any(|d| d.message == message) {
            self.diags.push(Diagnostic::note(message));
        }
    }

    /// Infer the substitution value nodes of `alias`'s load directive in this
    /// (importer) module's context, returning `(input-name, Resolved)` pairs.
    fn subst_annos_for(&mut self, alias: Symbol) -> Vec<(String, crate::modules::Resolved)> {
        let subs = self.session.substitutions_of(self.module, alias);
        subs.into_iter()
            .map(|(name, value)| {
                let (ty, phase) = self.infer_node(value);
                let vset = self.lookup_valueset(value);
                (
                    name,
                    crate::modules::Resolved {
                        ty,
                        phase,
                        vset,
                        result: None,
                        catalogue: None,
                    },
                )
            })
            .collect()
    }
}

/// `stochastic > parameterized > fixed` (spec §04 phases).
pub(crate) fn join_phase(a: Phase, b: Phase) -> Phase {
    use Phase::*;
    match (a, b) {
        (Stochastic, _) | (_, Stochastic) => Stochastic,
        (Parameterized, _) | (_, Parameterized) => Parameterized,
        (Fixed, Fixed) => Fixed,
    }
}
