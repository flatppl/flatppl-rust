//! The memoised type/phase trace over a module's binding DAG.
//!
//! Bindings are visited in source order; an explicit work stack resolves
//! dependencies with the side-tables doubling as the memo. A reference
//! cycle is an error (the module is not a DAG); the offending binding gets a
//! `(%failed …)` type so the gap is visible in annotated output.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

use flatppl_core::{
    BindingId, Call, CallHead, Module, NamedKind, Node, NodeId, Phase, Ref, RefNs, Scalar, Symbol,
    Type, ValueSet,
};

use crate::modules::{Dependency, InferSession, Resolution, Resolved};
use crate::ops;
use crate::rule::{LocalTargets, Resume, RuleStep};
use crate::{Diagnostic, Level};

type AutoInputScope = HashMap<NodeId, Box<[(Symbol, Ref)]>>;

enum Work {
    Binding(BindingId),
    Enter(NodeId),
    Finish(NodeId),
    LeaveBinding(BindingId),
    ResumeBody(Box<BodyFrame>),
    ResumeModule(Box<ModuleFrame>),
}

/// Mutable annotations belong to one substitution context, not to the graph.
struct Scope {
    diags: Vec<Diagnostic>,
    inferred: HashMap<NodeId, (Type, Phase)>,
    vsets: HashMap<NodeId, ValueSet>,
    seeds: HashMap<NodeId, (Type, Phase, ValueSet)>,
    in_progress: Vec<BindingId>,
    active_bindings: HashSet<BindingId>,
    noted_gaps: HashSet<Symbol>,
    module_callable_results: HashMap<NodeId, Type>,
    module_catalogue_refs: HashMap<NodeId, crate::modules::CatalogueRef>,
}

struct CallFrame {
    id: NodeId,
    call: Call,
    callee: Option<(NodeId, Type)>,
    args: Vec<(NodeId, Type, Phase)>,
    named: Vec<(Symbol, NodeId, Type, Phase)>,
    body_result: Option<Box<InferredBody>>,
}

/// One completed type-stage request, consumed by the same call's value-set rule.
/// Child annotations and auto-input discoveries never leave their original scope.
struct InferredBody {
    body: NodeId,
    seeds: Vec<(NodeId, Resolved)>,
    targets: Option<Box<LocalTargets>>,
    result: (Type, ValueSet),
}

impl InferredBody {
    fn matches(&self, body: NodeId, seeds: &[(NodeId, Resolved)]) -> bool {
        self.body == body
            && self.seeds.len() == seeds.len()
            && self.seeds.iter().zip(seeds).all(|((a, x), (b, y))| {
                a == b
                    && x.ty == y.ty
                    && x.phase == y.phase
                    && x.vset == y.vset
                    && x.result.is_none()
                    && y.result.is_none()
                    && x.catalogue.is_none()
                    && y.catalogue.is_none()
            })
    }
}

enum CallResume {
    Type(CallFrame, Resume<(Type, Phase)>),
    Valueset(CallFrame, Type, Phase, Resume<ValueSet>),
}

struct BodyFrame {
    body: NodeId,
    seeds: Vec<(NodeId, Resolved)>,
    targets: Option<Box<LocalTargets>>,
    scope: Scope,
    resume: CallResume,
}

struct ModuleFrame {
    node: NodeId,
    module: Module,
    scope: Scope,
    kernel_tag_nodes: Rc<HashSet<NodeId>>,
    auto_inputs: Vec<AutoInputScope>,
    key: (String, String),
    path: String,
    binding: String,
}

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
    inferred: HashMap<NodeId, (Type, Phase)>,
    vsets: HashMap<NodeId, ValueSet>,
    /// Pre-seeded annotations for substituted input nodes: (type, phase,
    /// valueset). Applied when entering a node before any other logic,
    /// making the substituted input authoritative for everything downstream.
    seeds: HashMap<NodeId, (Type, Phase, ValueSet)>,
    /// Bindings on the active resolution path (cycle detection).
    in_progress: Vec<BindingId>,
    active_bindings: HashSet<BindingId>,
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
    /// Argument positions of a `builtin_*` primitive or a `broadcast`, where a
    /// bare distribution-constructor atom is a kernel TAG rather than a value
    /// reference. See [`collect_kernel_tag_nodes`].
    kernel_tag_nodes: Rc<HashSet<NodeId>>,
    /// Nested substitutions inherit known boundaries but discard discoveries
    /// made under their argument phases. Only the root scope is flushed.
    auto_inputs: Vec<AutoInputScope>,
}

/// The exact node ids sitting in a kernel-TAG SLOT, where a bare
/// distribution-constructor atom is a tag rather than a variable reference.
///
/// One node per tag-bearing call, decided by
/// [`crate::builtins::kernel_tag_node`] — the single table of which argument of
/// which call is the tag, shared with the determiniser's conformance scan so the
/// two cannot disagree.
///
/// This exists because inference runs over determiniser OUTPUT as well as user
/// source: the determiniser resolves `broadcast(hepphys.ContinuedPoisson, rates)`
/// to a BARE `Const(ContinuedPoisson)` tag — both engines key the registry bare —
/// and the driver re-runs `infer` on its own output each iteration. Without the
/// slot, the §04 bare-name gate could not tell that tag from a user writing
/// `add(CrystalBall, 1.0)`, which must still be rejected because §09 gives a
/// member no unqualified spelling.
///
/// The slot is exact, not per-call: an earlier cut exempted every argument of any
/// `builtin_*` / `broadcast` call and narrowed only by name, which let a §09
/// constructor pass in the observed-value, params and rngstate slots
/// (`builtin_logdensityof(Normal, record(…), CrystalBall)` and friends lowered at
/// exit 0). Only the tag slot is a tag.
fn collect_kernel_tag_nodes(m: &Module) -> (HashSet<NodeId>, usize) {
    let mut out = HashSet::new();
    let mut visited = HashSet::new();
    let mut pending: Vec<NodeId> = m.bindings().map(|(_, binding)| binding.rhs).collect();
    while let Some(id) = pending.pop() {
        if !visited.insert(id) {
            continue;
        }
        if let Node::Call(c) = m.node(id) {
            out.extend(crate::builtins::kernel_tag_node(m, c));
        }
        m.node(id).for_each_child(|child| pending.push(child));
    }
    (out, visited.len())
}

impl<'m, 's> Inferencer<'m, 's> {
    pub(crate) fn new(module: &'m mut Module, level: Level, session: &'s InferSession<'s>) -> Self {
        let (kernel_tag_nodes, reachable) = collect_kernel_tag_nodes(module);
        let mut inf = Self::with_kernel_tags(module, level, session, Rc::new(kernel_tag_nodes));
        // A root walk can reach every binding. Substituted body walks stay sparse.
        inf.inferred.reserve(reachable);
        if level >= Level::Valueset {
            inf.vsets.reserve(reachable);
        }
        inf
    }

    fn with_kernel_tags(
        module: &'m mut Module,
        level: Level,
        session: &'s InferSession<'s>,
        kernel_tag_nodes: Rc<HashSet<NodeId>>,
    ) -> Self {
        Inferencer {
            module,
            level,
            session,
            diags: Vec::new(),
            inferred: HashMap::new(),
            vsets: HashMap::new(),
            seeds: HashMap::new(),
            in_progress: Vec::new(),
            active_bindings: HashSet::new(),
            noted_gaps: HashSet::new(),
            module_callable_results: HashMap::new(),
            module_catalogue_refs: HashMap::new(),
            kernel_tag_nodes,
            auto_inputs: vec![HashMap::new()],
        }
    }

    fn seed_inputs(&mut self, seeds: &[(NodeId, crate::modules::Resolved)]) {
        self.seeds = seeds
            .iter()
            .map(|(id, r)| (*id, (r.ty.clone(), r.phase, r.vset.clone())))
            .collect();
    }

    fn suspend_scope(&mut self) -> Scope {
        Scope {
            diags: std::mem::take(&mut self.diags),
            inferred: std::mem::take(&mut self.inferred),
            vsets: std::mem::take(&mut self.vsets),
            seeds: std::mem::take(&mut self.seeds),
            in_progress: std::mem::take(&mut self.in_progress),
            active_bindings: std::mem::take(&mut self.active_bindings),
            noted_gaps: std::mem::take(&mut self.noted_gaps),
            module_callable_results: std::mem::take(&mut self.module_callable_results),
            module_catalogue_refs: std::mem::take(&mut self.module_catalogue_refs),
        }
    }

    fn restore_scope(&mut self, scope: Scope) {
        self.diags = scope.diags;
        self.inferred = scope.inferred;
        self.vsets = scope.vsets;
        self.seeds = scope.seeds;
        self.in_progress = scope.in_progress;
        self.active_bindings = scope.active_bindings;
        self.noted_gaps = scope.noted_gaps;
        self.module_callable_results = scope.module_callable_results;
        self.module_catalogue_refs = scope.module_catalogue_refs;
    }

    pub(crate) fn auto_inputs_of(&self, id: NodeId) -> Option<&[(Symbol, Ref)]> {
        self.auto_inputs
            .iter()
            .rev()
            .find_map(|scope| scope.get(&id).map(AsRef::as_ref))
            .or_else(|| self.module.auto_inputs_of(id))
    }

    pub(crate) fn set_auto_inputs(&mut self, id: NodeId, entries: Box<[(Symbol, Ref)]>) {
        self.auto_inputs.last_mut().unwrap().insert(id, entries);
    }

    pub(crate) fn run(mut self) -> Vec<Diagnostic> {
        self.initialize_module();
        let mut work = Vec::new();
        self.schedule_bindings(&mut work);
        self.drain_work(&mut work);
        self.finish_module()
    }

    fn schedule_bindings(&self, work: &mut Vec<Work>) {
        let ids: Vec<_> = self.module.bindings().map(|(id, _)| id).collect();
        work.extend(ids.into_iter().rev().map(Work::Binding));
    }

    fn initialize_module(&mut self) {
        // Spec §05 axis positions, before the trace: the rules are positional,
        // and seeding the offending node `Failed` keeps a refused bracket to one
        // error instead of a cascade of type complaints about it.
        for (node, diag) in crate::axes::position_errors(self.module) {
            self.inferred.insert(
                node,
                (Type::Failed("axis out of position".into()), Phase::Fixed),
            );
            self.diags.push(diag);
        }
    }

    fn finish_module(&mut self) -> Vec<Diagnostic> {
        // Level-aware flush into the module's annotation side-tables.
        for (id, entries) in self.auto_inputs.pop().unwrap() {
            self.module.set_auto_inputs(id, entries);
        }
        // Move completed annotations into the module as natural value sets
        // are derived, without cloning either table.
        for (id, (ty, phase)) in std::mem::take(&mut self.inferred) {
            if self.level >= Level::Valueset {
                // Total discipline (spec §11): a value-typed node's set is at
                // least the type's natural extent — fall back where no producer
                // established anything finer. One chokepoint; producers stay
                // refinement-only.
                let stored = self.vsets.get(&id);
                if stored.is_none() || stored == Some(&ValueSet::Unknown) {
                    let natural = ValueSet::natural_of(&ty);
                    if natural != ValueSet::Unknown {
                        self.vsets.insert(id, natural);
                    }
                }
            }
            self.module.set_phase(id, phase);
            if self.level >= Level::Type {
                self.module.set_type(id, ty);
            }
        }
        if self.level >= Level::Valueset {
            for (id, set) in std::mem::take(&mut self.vsets) {
                self.module.set_valueset(id, set);
            }
        }
        // Drain dependency diagnostics accumulated during this run (cross-module
        // cycle errors and other dep-level errors). Module frames store child
        // diagnostics here after each dependency walk; draining ensures they
        // propagate up through every level of the import chain.
        self.diags.extend(self.session.drain_dep_diags());
        std::mem::take(&mut self.diags)
    }

    /// The inferred type of an already-visited node (ops rules use this to
    /// look through reified bodies).
    pub(crate) fn lookup_type(&self, id: NodeId) -> Option<&Type> {
        self.inferred.get(&id).map(|(ty, _)| ty)
    }

    /// The inferred phase of an already-visited node. `None` while the walk has
    /// not reached it — the module's own annotations are not written until the
    /// final flush, so an ops rule must read the live table, not `Module`.
    /// A caller pruning on this must treat `None` as "do not prune".
    pub(crate) fn lookup_phase(&self, id: NodeId) -> Option<Phase> {
        self.inferred.get(&id).map(|(_, phase)| *phase)
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

    /// §04 auto-trace for a boundary-less `functionof` / `kernelof`: its inputs
    /// are the `elementof` parametric-phase leaves of the body's ancestor
    /// subgraph, in **canonical order (sorted by name)** so a converter's
    /// incidental build order (e.g. pyhf JSON sample/modifier order) never leaks
    /// into the input list. Fixed-phase ancestors (`external` / `load_data` /
    /// literals) are closed over — pruned via the phase table. Every ancestor's
    /// phase is already memoised when this runs (the reification's children are
    /// traced before its op rule), so the walk reads, never recurses, inference.
    /// Returns the local `elementof` leaf inputs and a flag set if the trace
    /// reached a parametric **cross-module** dependency. Reification is
    /// module-local (spec §04): a parameterized value reached through a
    /// loaded-module reference cannot become an input, so the caller turns the
    /// flag into a static error.
    pub(crate) fn collect_auto_inputs(&self, body: NodeId) -> (Vec<(Symbol, Ref)>, bool) {
        // Keyed by leaf name (dedup + canonical sort).
        let mut leaves: BTreeMap<String, Ref> = BTreeMap::new();
        let mut cross_module = false;
        let mut visited: HashSet<NodeId> = HashSet::new();
        self.walk_auto(body, &mut leaves, &mut cross_module, &mut visited);
        (
            leaves.into_values().map(|r| (r.name, r)).collect(),
            cross_module,
        )
    }

    /// Iterative ancestor walk for [`collect_auto_inputs`], following `self`
    /// references across binding boundaries and recording each `elementof`
    /// leaf binding's name once (`leaves`, keyed by name → sorted + deduped);
    /// sets `cross_module` if it reaches a parametric loaded-module reference.
    fn walk_auto(
        &self,
        id: NodeId,
        leaves: &mut BTreeMap<String, Ref>,
        cross_module: &mut bool,
        visited: &mut HashSet<NodeId>,
    ) {
        let mut pending = vec![id];
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            // A fixed-phase subgraph has no `elementof` ancestor (spec §04) — prune.
            // (Absent phase ⇒ don't prune: soundly does extra work, never skips.)
            if self.lookup_phase(id) == Some(Phase::Fixed) {
                continue;
            }
            // Cached reference chains can be arbitrarily deep even when inference
            // itself never recurses. Keep their traversal on the heap.
            let children: Vec<NodeId> = match self.module.node(id) {
                Node::Ref(r) if r.ns == RefNs::SelfMod => match self.module.binding_by_name(r.name)
                {
                    Some(b) => {
                        let rhs = self.module.binding(b).rhs;
                        if is_elementof(self.module, rhs) {
                            // A parametric leaf: record the binding (by name); do not
                            // descend into the `elementof` set argument.
                            leaves.insert(
                                self.module.resolve(r.name).to_string(),
                                Ref {
                                    ns: RefNs::SelfMod,
                                    name: r.name,
                                },
                            );
                            Vec::new()
                        } else {
                            vec![rhs]
                        }
                    }
                    None => Vec::new(),
                },
                // A parametric cross-module dependency. Reification is module-local
                // (spec §04): such a value cannot become a reified input — flag it so
                // the caller errors. (A FIXED cross-module ref was pruned above, so
                // reaching here means the referenced binding is parameterized.)
                Node::Ref(r) if matches!(r.ns, RefNs::Module(_)) => {
                    *cross_module = true;
                    Vec::new()
                }
                Node::Call(call) => {
                    let mut c = Vec::new();
                    if let CallHead::User(callee) = call.head {
                        c.push(callee);
                    }
                    c.extend(call.args.iter().copied());
                    c.extend(call.named.iter().map(|n| n.value));
                    c
                }
                // Lit / Const / Hole / Axis / Ref(%local) carry no parametric leaves
                // (a `%local` placeholder is a Spec-boundary input, never auto).
                _ => Vec::new(),
            };
            pending.extend(children.into_iter().rev());
        }
    }

    fn enter_binding(&mut self, id: BindingId) -> bool {
        if !self.active_bindings.insert(id) {
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
            let rhs = self.module.binding(id).rhs;
            self.inferred.insert(rhs, (ty.clone(), Phase::Fixed));
            return false;
        }
        self.in_progress.push(id);
        true
    }

    /// Rules run after their dependencies finish. Reading an annotation must
    /// never launch another trace or cross a substituted boundary.
    pub(crate) fn node_annotation(&self, id: NodeId) -> (Type, Phase) {
        self.inferred
            .get(&id)
            .expect("inference dependency must be complete")
            .clone()
    }

    pub(crate) fn is_substituted(&self, id: NodeId) -> bool {
        self.seeds.contains_key(&id)
    }

    fn drain_work(&mut self, work: &mut Vec<Work>) {
        while let Some(step) = work.pop() {
            match step {
                Work::Binding(binding) => {
                    let rhs = self.module.binding(binding).rhs;
                    if !self.inferred.contains_key(&rhs) && self.enter_binding(binding) {
                        work.push(Work::LeaveBinding(binding));
                        work.push(Work::Enter(rhs));
                    }
                }
                Work::LeaveBinding(binding) => {
                    self.in_progress.pop();
                    self.active_bindings.remove(&binding);
                }
                Work::Finish(node) => {
                    // A cycle marker keeps its failed type, but the operation
                    // still runs its independent checks (e.g. arity).
                    self.finish_node(node, work);
                }
                Work::ResumeBody(frame) => {
                    let ty = self.inferred[&frame.body].0.clone();
                    let vset = self.lookup_valueset(frame.body);
                    self.auto_inputs.pop();
                    self.restore_scope(frame.scope);
                    match frame.resume {
                        CallResume::Type(mut call, resume) => {
                            if self.level >= Level::Valueset {
                                call.body_result = Some(Box::new(InferredBody {
                                    body: frame.body,
                                    seeds: frame.seeds,
                                    targets: frame.targets,
                                    result: (ty.clone(), vset.clone()),
                                }));
                            }
                            let step = resume(self, (ty, vset));
                            self.finish_call_type(call, step, work);
                        }
                        CallResume::Valueset(call, result, phase, resume) => {
                            let step = resume(self, (ty, vset));
                            self.finish_call_valueset(call, result, phase, step, work);
                        }
                    }
                }
                Work::ResumeModule(frame) => {
                    let diags = self.finish_module();
                    self.session.push_dep_diags(diags);
                    let dependency = std::mem::replace(self.module, frame.module);
                    self.kernel_tag_nodes = frame.kernel_tag_nodes;
                    self.auto_inputs = frame.auto_inputs;
                    self.restore_scope(frame.scope);
                    self.session
                        .finish_dependency(frame.key.clone(), dependency);
                    let result = self.session.resolved_binding(
                        self.module,
                        &frame.key,
                        &frame.path,
                        &frame.binding,
                    );
                    self.record_resolution(frame.node, result);
                }
                Work::Enter(node) => {
                    if let Some((ty, phase, vset)) = self.seeds.get(&node).cloned() {
                        self.inferred.insert(node, (ty, phase));
                        self.set_vset(node, vset);
                        continue;
                    }
                    if self.inferred.contains_key(&node) {
                        continue;
                    }
                    work.push(Work::Finish(node));
                    match self.module.node(node) {
                        Node::Ref(r) if r.ns == RefNs::SelfMod => {
                            if let Some(binding) = self.module.binding_by_name(r.name) {
                                let rhs = self.module.binding(binding).rhs;
                                if !self.inferred.contains_key(&rhs) && self.enter_binding(binding)
                                {
                                    work.push(Work::LeaveBinding(binding));
                                    work.push(Work::Enter(rhs));
                                }
                            }
                        }
                        Node::Ref(r) if matches!(r.ns, RefNs::Module(_)) => {
                            let RefNs::Module(alias) = r.ns else {
                                unreachable!()
                            };
                            for (_, value) in self
                                .session
                                .substitutions_of(self.module, alias)
                                .into_iter()
                                .rev()
                            {
                                work.push(Work::Enter(value));
                            }
                        }
                        node => {
                            let start = work.len();
                            node.for_each_child(|child| work.push(Work::Enter(child)));
                            work[start..].reverse();
                        }
                    }
                }
            }
        }
    }

    fn finish_node(&mut self, id: NodeId, work: &mut Vec<Work>) {
        match self.module.node(id).clone() {
            Node::Call(call) => self.start_call(id, call, work),
            Node::Ref(Ref {
                ns: RefNs::Module(alias),
                name,
            }) => {
                let binding = self.module.resolve(name).to_string();
                let seeds = self.subst_annos_for(alias);
                match self.session.resolve(self.module, alias, &binding, &seeds) {
                    Ok(Resolution::Ready(result)) => self.record_resolution(id, Ok(result)),
                    Err(error) => self.record_resolution(id, Err(error)),
                    Ok(Resolution::Infer(dependency)) => self.start_module(id, *dependency, work),
                }
            }
            _ => {
                self.infer_node_inner(id);
            }
        }
    }

    fn start_module(&mut self, node: NodeId, dependency: Dependency, work: &mut Vec<Work>) {
        let Dependency {
            module,
            seeds,
            key,
            path,
            binding,
        } = dependency;
        let scope = self.suspend_scope();
        let module = std::mem::replace(self.module, module);
        let (tags, reachable) = collect_kernel_tag_nodes(self.module);
        let kernel_tag_nodes = std::mem::replace(&mut self.kernel_tag_nodes, Rc::new(tags));
        let auto_inputs = std::mem::replace(&mut self.auto_inputs, vec![HashMap::new()]);
        self.inferred.reserve(reachable);
        if self.level >= Level::Valueset {
            self.vsets.reserve(reachable);
        }
        self.seed_inputs(&seeds);
        self.initialize_module();
        work.push(Work::ResumeModule(Box::new(ModuleFrame {
            node,
            module,
            scope,
            kernel_tag_nodes,
            auto_inputs,
            key,
            path,
            binding,
        })));
        self.schedule_bindings(work);
    }

    fn record_resolution(&mut self, node: NodeId, result: Result<Resolved, String>) {
        let result = match result {
            Ok(res) => {
                if let Some(result) = res.result {
                    self.module_callable_results.insert(node, result);
                }
                if let Some(catalogue) = res.catalogue {
                    self.module_catalogue_refs.insert(node, catalogue);
                }
                self.set_vset(node, res.vset);
                (res.ty, res.phase)
            }
            Err(message) => {
                self.diags.push(Diagnostic::error_at(node, message));
                (Type::Failed("cross-module resolution".into()), Phase::Fixed)
            }
        };
        self.inferred.insert(node, result);
    }

    fn infer_node_inner(&mut self, id: NodeId) -> (Type, Phase) {
        // Clone the node to release the module borrow while annotating; nodes
        // are small (boxed slices of ids).
        let node = self.module.node(id).clone();
        let (ty, phase) = match &node {
            Node::Lit(s) => {
                self.set_vset(id, ops::literal_valueset(s));
                (ops::literal_type(s), Phase::Fixed)
            }
            Node::Const(sym) => {
                let name = self.module.resolve(*sym).to_string();
                // A bare atom is the `base` namespace: the parser emits one for
                // every unqualified name it did not find among the module's
                // bindings (`syntax::parser::resolve_bare_name`). If it is not a
                // built-in either, the name resolves nowhere and §04 "Name
                // resolution" makes that a static error — without this arm
                // `const_type` hands the unknown name `Type::Any`, inference
                // absorbs it as a `%fixed` scalar, and the determiniser lowers a
                // FREE VARIABLE into FlatPDL. The `self.`-qualified spelling of
                // the same mistake has always errored ("unresolved reference").
                // One exemption: a distribution constructor in kernel-TAG
                // position. §09 members are not in the `base` namespace, but the
                // determiniser emits one BARE as a tag and re-runs inference over
                // its own output — see `collect_kernel_tag_nodes`.
                let is_tag = self.kernel_tag_nodes.contains(&id)
                    && crate::builtins::is_kernel_tag_name(&name);
                if !is_tag && !crate::builtins::is_base_name(&name) {
                    self.diags.push(Diagnostic::error_at(
                        id,
                        format!(
                            "unresolvable name `{name}`: not a binding in this module and not a \
                             FlatPPL built-in (spec §04 \"Name resolution\")"
                        ),
                    ));
                    (Type::Failed("unresolvable name".into()), Phase::Fixed)
                } else {
                    self.set_vset(id, ops::const_valueset(&name));
                    (ops::const_type(&name), Phase::Fixed)
                }
            }
            Node::Ref(r) => match r.ns {
                RefNs::SelfMod => match self.module.binding_by_name(r.name) {
                    Some(b) => {
                        let rhs = self.module.binding(b).rhs;
                        let result = self.inferred[&rhs].clone();
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
                RefNs::Module(_) => unreachable!("module references use scheduler frames"),
            },
            Node::Hole | Node::Axis(_) => {
                self.set_vset(id, ValueSet::Unknown);
                (Type::Any, Phase::Fixed)
            }
            Node::Call(_) => unreachable!("calls use scheduler frames"),
        };
        // A cycle marker may have landed on this node while the walk was in
        // flight (see enter_binding); it is authoritative — don't clobber it.
        if let Some(result) = self.inferred.get(&id) {
            return result.clone();
        }
        self.inferred.insert(id, (ty.clone(), phase));
        (ty, phase)
    }

    fn start_call(&mut self, id: NodeId, call: Call, work: &mut Vec<Work>) {
        // Children first: callee (user calls), positional, named.
        let callee = match call.head {
            CallHead::User(callee) => Some((callee, self.node_annotation(callee))),
            CallHead::Builtin(_) => None,
        };
        let args: Vec<(NodeId, Type, Phase)> = call
            .args
            .iter()
            .map(|&a| {
                let (t, p) = self.node_annotation(a);
                (a, t, p)
            })
            .collect();
        let named: Vec<(Symbol, NodeId, Type, Phase)> = call
            .named
            .iter()
            .map(|n| {
                let (t, p) = self.node_annotation(n.value);
                (n.name, n.value, t, p)
            })
            .collect();

        self.validate_load_assigns(&call);

        // The §04 ancestor rule: a call's phase is the join of its inputs'
        // phases, except where the op itself introduces a phase.
        let joined = callee
            .iter()
            .map(|(_, (_, p))| *p)
            .chain(args.iter().map(|(_, _, p)| *p))
            .chain(named.iter().map(|(_, _, _, p)| *p))
            .fold(Phase::Fixed, join_phase);

        let callee = callee.map(|(n, tp)| (n, tp.0));
        let step = ops::call_rule(self, id, &call, callee.clone(), &args, &named, joined);
        let frame = CallFrame {
            id,
            call,
            callee,
            args,
            named,
            body_result: None,
        };
        self.finish_call_type(frame, step, work);
    }

    fn start_body(
        &mut self,
        body: NodeId,
        seeds: Vec<(NodeId, Resolved)>,
        targets: Option<Box<LocalTargets>>,
        resume: CallResume,
        work: &mut Vec<Work>,
    ) {
        let scope = self.suspend_scope();
        self.seed_inputs(&seeds);
        self.auto_inputs.push(HashMap::new());
        let seeds = if self.level >= Level::Valueset && matches!(&resume, CallResume::Type(..)) {
            seeds
        } else {
            Vec::new()
        };
        work.push(Work::ResumeBody(Box::new(BodyFrame {
            body,
            seeds,
            targets,
            scope,
            resume,
        })));
        work.push(Work::Enter(body));
    }

    fn finish_call_type(
        &mut self,
        mut call: CallFrame,
        step: RuleStep<(Type, Phase)>,
        work: &mut Vec<Work>,
    ) {
        match step {
            RuleStep::Ready((ty, phase)) => {
                let valueset = if self.level >= Level::Valueset {
                    ops::call_valueset(
                        self,
                        &call.call,
                        call.callee.as_ref(),
                        &call.args,
                        &call.named,
                        &ty,
                        call.body_result
                            .as_ref()
                            .and_then(|result| result.targets.as_deref()),
                    )
                } else {
                    RuleStep::Ready(ValueSet::Unknown)
                };
                if let Some(result) = &mut call.body_result {
                    result.targets = None;
                }
                self.finish_call_valueset(call, ty, phase, valueset, work);
            }
            RuleStep::InferBody {
                body,
                seeds,
                targets,
                resume,
            } => {
                call.body_result = None;
                self.start_body(body, seeds, targets, CallResume::Type(call, resume), work);
            }
        }
    }

    fn finish_call_valueset(
        &mut self,
        mut call: CallFrame,
        ty: Type,
        phase: Phase,
        step: RuleStep<ValueSet>,
        work: &mut Vec<Work>,
    ) {
        match step {
            RuleStep::Ready(set) => {
                self.set_vset(call.id, set);
                let ty = if self.level >= Level::Normalization {
                    ops::fill_mass(
                        self,
                        call.id,
                        &call.call,
                        call.callee.as_ref(),
                        ty,
                        &call.args,
                        &call.named,
                    )
                } else {
                    ty
                };
                self.inferred.entry(call.id).or_insert((ty, phase));
            }
            RuleStep::InferBody {
                body,
                seeds,
                targets: _,
                resume,
            } => {
                // Type and value-set rules can request the same substituted body.
                // Retain their separate continuations, including broadcast lifting.
                if let Some(previous) = call.body_result.take()
                    && previous.matches(body, &seeds)
                {
                    let step = resume(self, previous.result);
                    self.finish_call_valueset(call, ty, phase, step, work);
                    return;
                }
                self.start_body(
                    body,
                    seeds,
                    None,
                    CallResume::Valueset(call, ty, phase, resume),
                    work,
                );
            }
        }
    }

    /// Validate `%assign` substitutions on a `load_module` call against spec §04
    /// (Module composition → Load-time substitution): each `%assign` name must
    /// designate an `external`/`elementof` **input** of the dependency, and the
    /// substitution value's phase must match the input's kind — `external` ←
    /// fixed, `elementof` ← parameterized. Emits anchored errors on the offending
    /// value node. No-op for non-load calls and for `standard_module` (its
    /// dependency never enters the bundle, so the inner block is skipped).
    ///
    /// The data needed from `call` is cloned out first so the borrow is
    /// released before `self.diags.push` or `self.module.resolve` are called.
    fn validate_load_assigns(&mut self, call: &Call) {
        let load_check: Option<(String, Vec<(Symbol, NodeId)>)> =
            if let CallHead::Builtin(head) = call.head {
                if matches!(self.module.resolve(head), "load_module" | "standard_module") {
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
        if let Some((path, assigns)) = load_check
            && let Some(dep) = self.session.dep_for_literal(&path)
        {
            for (name_sym, value_node) in assigns {
                let input = self.module.resolve(name_sym).to_string();
                match dep_input_kind(dep, &input) {
                    // Spec §04: the LHS must name an *input* of the loaded
                    // module. No binding by that name → "has no input".
                    None => self.diags.push(Diagnostic::error_at(
                        value_node,
                        format!("module `{path}` has no input `{input}`"),
                    )),
                    // A binding exists but is an ordinary computed value, not
                    // an `external`/`elementof` input ("No other kinds of
                    // nodes … may be bound").
                    Some(InputKind::Other) => self.diags.push(Diagnostic::error_at(
                        value_node,
                        format!(
                            "`{input}` is not an input (`external`/`elementof`) of module `{path}`"
                        ),
                    )),
                    // An input: its phase governs what may bind to it —
                    // `external` ← fixed, `elementof` ← parameterized — and
                    // (spec §04) the value's set must lie within the input's
                    // declared set.
                    Some(kind) => {
                        // Phase. The substitution value was inferred just
                        // above, so its phase is recorded; skip if not (never
                        // false-positive on a missing phase).
                        if let Some(value_phase) = self.lookup_phase(value_node) {
                            let required = kind.required_phase();
                            if value_phase != required {
                                self.diags.push(Diagnostic::error_at(
                                    value_node,
                                    format!(
                                        "{} `{input}` of module `{path}` may only be bound to \
                                             a {required} value (got {value_phase})",
                                        kind.describe()
                                    ),
                                ));
                            }
                        }
                        // Value set (only at `Level::Valueset`+). Conservative:
                        // flag only a *proven strict superset* — the value
                        // admits points outside the declared domain. Pairs the
                        // checker can't prove (incomparable / `Unknown` sets)
                        // are left alone, so no valid model is rejected.
                        if self.level >= Level::Valueset {
                            let declared = declared_input_set(dep, &input);
                            let value_set = self.lookup_valueset(value_node);
                            if is_concrete_set(&value_set)
                                && declared.subset_of(&value_set)
                                && !value_set.subset_of(&declared)
                            {
                                self.diags.push(Diagnostic::error_at(
                                    value_node,
                                    format!(
                                        "the substitution value set `{}` is wider than input \
                                             `{input}`'s declared set `{declared}` in module \
                                             `{path}` (spec §04: value sets must be compatible)",
                                        self.module.display_valueset(&value_set)
                                    ),
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    // (free helpers `InputKind` / `dep_input_kind` live at module scope below)

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
                let (ty, phase) = self.node_annotation(value);
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

/// Is the node at `id` a bare `elementof(...)` call — a module parameter, the
/// parametric leaf of [`Inferencer::collect_auto_inputs`]?
fn is_elementof(module: &Module, id: NodeId) -> bool {
    match module.node(id) {
        Node::Call(c) => match c.head {
            CallHead::Builtin(h) => module.resolve(h) == "elementof",
            CallHead::User(_) => false,
        },
        _ => false,
    }
}

/// How a loaded-module binding may be used as a substitution target (spec §04).
enum InputKind {
    /// `external(...)` — a load-time hyperparameter; bind a **fixed** value.
    External,
    /// `elementof(...)` — a model parameter; bind a **parameterized** value.
    Elementof,
    /// An ordinary computed binding, not an input — may not be substituted.
    Other,
}

impl InputKind {
    /// The phase a substitution value must have to bind to this input.
    fn required_phase(&self) -> Phase {
        match self {
            InputKind::External => Phase::Fixed,
            InputKind::Elementof => Phase::Parameterized,
            InputKind::Other => Phase::Fixed, // unreachable: Other is rejected earlier
        }
    }

    /// Human-readable label for the diagnostic ("external input" / "parameter …").
    fn describe(&self) -> &'static str {
        match self {
            InputKind::External => "external input",
            InputKind::Elementof => "parameter (`elementof`) input",
            InputKind::Other => "binding",
        }
    }
}

/// Classify the binding named `name` in dependency module `dep` for §04
/// substitution rules. `None` when no such binding exists ("has no input");
/// `Some(External|Elementof)` for a bare `external(...)`/`elementof(...)` input;
/// `Some(Other)` for any other binding (a non-input that may not be substituted).
fn dep_input_kind(dep: &Module, name: &str) -> Option<InputKind> {
    let (_, b) = dep.bindings().find(|(_, b)| dep.resolve(b.name) == name)?;
    let Node::Call(c) = dep.node(b.rhs) else {
        return Some(InputKind::Other);
    };
    let CallHead::Builtin(h) = c.head else {
        return Some(InputKind::Other);
    };
    Some(match dep.resolve(h) {
        "external" => InputKind::External,
        "elementof" => InputKind::Elementof,
        _ => InputKind::Other,
    })
}

/// The declared value-set of dependency input `name` — the set argument of its
/// `elementof(S)` / `external(S)` binding, read structurally (no inference, the
/// dep may be uninferred here). Covers the set-constant vocabulary and
/// literal-bound `interval`; anything else → `Unknown` (so the §04 value-set
/// check simply does not fire — conservative).
fn declared_input_set(dep: &Module, name: &str) -> ValueSet {
    let Some((_, b)) = dep.bindings().find(|(_, b)| dep.resolve(b.name) == name) else {
        return ValueSet::Unknown;
    };
    let Node::Call(c) = dep.node(b.rhs) else {
        return ValueSet::Unknown;
    };
    let CallHead::Builtin(h) = c.head else {
        return ValueSet::Unknown;
    };
    match dep.resolve(h) {
        "elementof" | "external" => set_const_node(dep, c.args.first().copied()),
        _ => ValueSet::Unknown,
    }
}

/// Read a constant set expression node (`reals`, `posreals`, …, or
/// `interval(lo, hi)` with literal bounds) into a [`ValueSet`]. `Unknown` for
/// anything non-constant — the foreign dep is not inferred here, so only the
/// statically-evident set vocabulary is recognised.
fn set_const_node(dep: &Module, node: Option<NodeId>) -> ValueSet {
    let Some(node) = node else {
        return ValueSet::Unknown;
    };
    match dep.node(node) {
        Node::Const(sym) => match dep.resolve(*sym) {
            "reals" => ValueSet::Reals,
            "posreals" => ValueSet::PosReals,
            "nonnegreals" => ValueSet::NonNegReals,
            "unitinterval" => ValueSet::UnitInterval,
            "integers" => ValueSet::Integers,
            "posintegers" => ValueSet::PosIntegers,
            "nonnegintegers" => ValueSet::NonNegIntegers,
            "booleans" => ValueSet::Booleans,
            "complexes" => ValueSet::Complexes,
            "rngstates" => ValueSet::RngStates,
            "anything" => ValueSet::Anything,
            _ => ValueSet::Unknown,
        },
        Node::Call(c) if matches!(c.head, CallHead::Builtin(h) if dep.resolve(h) == "interval") => {
            let bound = |n: Option<NodeId>| match n.map(|n| dep.node(n)) {
                Some(Node::Lit(Scalar::Real(r))) => Some(*r),
                Some(Node::Lit(Scalar::Int(i))) => Some(*i as f64),
                Some(Node::Const(s)) if dep.resolve(*s) == "inf" => Some(f64::INFINITY),
                _ => None,
            };
            match (
                bound(c.args.first().copied()),
                bound(c.args.get(1).copied()),
            ) {
                (Some(lo), Some(hi)) => ValueSet::Interval(lo, hi),
                _ => ValueSet::Unknown,
            }
        }
        _ => ValueSet::Unknown,
    }
}

/// A set concrete enough to ground a §04 value-set comparison (not a
/// "don't-know" marker). `Anything` is excluded so an unconstrained value is
/// never flagged as a violation on its own.
fn is_concrete_set(vs: &ValueSet) -> bool {
    !matches!(
        vs,
        ValueSet::Deferred | ValueSet::Unknown | ValueSet::Anything
    )
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
