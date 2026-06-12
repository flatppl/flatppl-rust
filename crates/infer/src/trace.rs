//! The memoised type/phase trace over a module's binding DAG.
//!
//! Bindings are visited in source order; references recurse (FlatPPL is
//! order-irrelevant) with the side-tables doubling as the memo. A reference
//! cycle is an error (the module is not a DAG); the offending binding gets a
//! `(%failed …)` type so the gap is visible in annotated output.

use std::collections::HashSet;

use flatppl_core::{BindingId, Call, CallHead, Module, Node, NodeId, Phase, RefNs, Symbol, Type};

use crate::Diagnostic;
use crate::ops;

pub(crate) struct Inferencer<'m> {
    pub(crate) module: &'m mut Module,
    pub(crate) diags: Vec<Diagnostic>,
    /// Bindings on the active resolution path (cycle detection).
    in_progress: Vec<BindingId>,
    /// Ops already reported as catalogue gaps (one note per op).
    noted_gaps: HashSet<Symbol>,
}

impl<'m> Inferencer<'m> {
    pub(crate) fn new(module: &'m mut Module) -> Self {
        Inferencer {
            module,
            diags: Vec::new(),
            in_progress: Vec::new(),
            noted_gaps: HashSet::new(),
        }
    }

    pub(crate) fn run(mut self) -> Vec<Diagnostic> {
        let ids: Vec<BindingId> = self.module.bindings().map(|(id, _)| id).collect();
        for id in ids {
            self.infer_binding(id);
        }
        self.diags
    }

    /// Type + phase of a binding's RHS (memoised via the side-tables).
    pub(crate) fn infer_binding(&mut self, id: BindingId) -> (Type, Phase) {
        let rhs = self.module.binding(id).rhs;
        if let (Some(ty), Some(phase)) = (self.module.type_of(rhs), self.module.phase_of(rhs)) {
            return (ty.clone(), phase);
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
            self.module.set_type(rhs, ty.clone());
            self.module.set_phase(rhs, Phase::Fixed);
            return (ty, Phase::Fixed);
        }
        self.in_progress.push(id);
        let result = self.infer_node(rhs);
        self.in_progress.pop();
        result
    }

    /// Type + phase of a node (memoised via the side-tables). Every node gets
    /// annotated; the FlatPIR writer projects `%meta` at call positions only
    /// (spec §11), the rest is internal to the trace.
    pub(crate) fn infer_node(&mut self, id: NodeId) -> (Type, Phase) {
        if let (Some(ty), Some(phase)) = (self.module.type_of(id), self.module.phase_of(id)) {
            return (ty.clone(), phase);
        }
        // Clone the node to release the module borrow during recursion; nodes
        // are small (boxed slices of ids).
        let node = self.module.node(id).clone();
        let (ty, phase) = match &node {
            Node::Lit(s) => (ops::literal_type(s), Phase::Fixed),
            Node::Const(sym) => (ops::const_type(self.module.resolve(*sym)), Phase::Fixed),
            Node::Ref(r) => match r.ns {
                RefNs::SelfMod => match self.module.binding_by_name(r.name) {
                    Some(b) => self.infer_binding(b),
                    None => {
                        let name = self.module.resolve(r.name).to_string();
                        self.diags
                            .push(Diagnostic::error(format!("unresolved reference `{name}`")));
                        (Type::Failed("unresolved reference".into()), Phase::Fixed)
                    }
                },
                // A placeholder is implicitly `elementof(anything)` (spec §04
                // "Placeholder variables") — unconstrained, parameterized.
                RefNs::Local => (Type::Any, Phase::Parameterized),
                // Cross-module inference rides on load_module support, which
                // is deferred until multi-file fixtures exist (see TODO).
                RefNs::Module(_) => {
                    self.note_once_str(
                        "cross-module references are not inferred yet \
                         (load_module support is deferred) — types left %deferred",
                    );
                    (Type::Deferred, Phase::Fixed)
                }
            },
            Node::Hole | Node::Axis(_) => (Type::Any, Phase::Fixed),
            Node::Call(call) => self.infer_call(id, call),
        };
        // A cycle marker may have landed on this node while the walk was in
        // flight (see infer_binding); it is authoritative — don't clobber it.
        if let (Some(t), Some(p)) = (self.module.type_of(id), self.module.phase_of(id)) {
            return (t.clone(), p);
        }
        self.module.set_type(id, ty.clone());
        self.module.set_phase(id, phase);
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

        // The §04 ancestor rule: a call's phase is the join of its inputs'
        // phases, except where the op itself introduces a phase.
        let joined = callee
            .iter()
            .map(|(_, (_, p))| *p)
            .chain(args.iter().map(|(_, _, p)| *p))
            .chain(named.iter().map(|(_, _, _, p)| *p))
            .fold(Phase::Fixed, join_phase);

        ops::call_rule(
            self,
            id,
            call,
            callee.map(|(n, tp)| (n, tp.0)),
            &args,
            &named,
            joined,
        )
    }

    /// Record a catalogue gap for `op`, once.
    pub(crate) fn note_gap(&mut self, op: Symbol) {
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
