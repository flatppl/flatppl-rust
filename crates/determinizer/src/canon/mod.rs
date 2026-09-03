//! FlatPDL → FlatPDL canonicalization: post-measure-elimination normalization
//! passes that reduce/canonicalize the determiniser's output while preserving
//! `flatpdl.flatprof` conformance AND exact semantics (Buffy #263). Each pass
//! is idempotent and refuse-free; the driver runs them to a combined fixpoint.

use std::collections::HashSet;

use flatppl_core::{BindingId, CallHead, Module, Node, NodeId, Ref, RefNs, Symbol};

mod dce;
mod flatten;
mod fold;
mod inline;

/// Whether `bid` is the reserved `inputs` binding (spec §13
/// `sec:determinization-signature`).
///
/// `inputs` is a pure declaration: every element must name a binding to
/// promote, so rewriting one into a value can only destroy it. §13's phase
/// table maps a *derived* fixed binding listed in `inputs` to "function
/// argument, replacing the computed value", and folding `inputs = c` for
/// `c = 2.0 * 3.0` down to `inputs = 6.0` erases the argument the caller must
/// supply. Every canonicalization pass therefore leaves this binding's RHS
/// untouched.
///
/// `outputs` is deliberately NOT exempt. Its elements are ordinary
/// expressions — a model can write the whole density inline as `outputs =
/// logdensityof(…)` — and they need folding like any other value: a
/// statically-foldable `Uniform` bound must reach the emitter as a literal
/// interval, or the support check refuses.
pub(crate) fn is_reserved_abi_binding(m: &Module, bid: BindingId) -> bool {
    m.resolve(m.binding(bid).name) == "inputs"
}

/// The binding names the reserved `inputs` declaration lists — the promoted
/// arguments of the compiled function.
///
/// §13's phase table replaces such a binding's *computed value* with a function
/// argument, so its value must not be substituted into the consumers that read
/// it: `inputs = c` for `c = 2.0 * 3.0` compiles to a one-argument function, not
/// to a `6.0` baked at every use of `c`. [`resolve_alias_refs`] therefore never
/// inlines a reference to one of these.
///
/// [`resolve_alias_refs`]: fold::resolve_alias_refs
pub(crate) fn promoted_input_names(m: &Module) -> HashSet<Symbol> {
    let Some((_, binding)) = m.bindings().find(|(_, b)| m.resolve(b.name) == "inputs") else {
        return HashSet::new();
    };
    abi_tuple_elems(m, binding.rhs)
        .into_iter()
        .filter_map(|elem| match m.node(elem) {
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) => Some(*name),
            _ => None,
        })
        .collect()
}

/// Normalize a reserved binding's RHS — a single value, or the `(v1, v2, …)`
/// surface sugar's `tuple(...)` call — to its element nodes in declared order.
/// Mirrors `flatppl_stablehlo::modes::tuple_elems`, which reads the same shape
/// back off the determinized module.
fn abi_tuple_elems(m: &Module, rhs: NodeId) -> Vec<NodeId> {
    if let Node::Call(c) = m.node(rhs) {
        if matches!(c.head, CallHead::Builtin(sym) if m.resolve(sym) == "tuple") {
            return c.args.to_vec();
        }
    }
    vec![rhs]
}

/// The `FLATPPL_DETERMINIZE_NO_CANON` env escape hatch: when set (to any value),
/// `canonicalize` is a no-op. Used to determinize a model both ways for the
/// numeric det-js equivalence gate; NOT a supported production toggle.
fn no_canon() -> bool {
    std::env::var_os("FLATPPL_DETERMINIZE_NO_CANON").is_some()
}

/// Run every canonicalization pass to a combined fixpoint, then — if `roots` is
/// given — root-based dead-code elimination (Pass 4-A, Buffy #263): only
/// bindings reachable from the requested-output `roots` survive. Re-infers
/// between sweeps because a reduction can shift inferred types/phases that a
/// later pass reads. A no-op if `FLATPPL_DETERMINIZE_NO_CANON` is set.
///
/// DCE runs ONCE, after the fixpoint — the dead-set is only stable once
/// inline/fold/flatten have converged; running it mid-fixpoint could drop a
/// binding a later pass would still have rewritten through. `roots = None`
/// preserves keep-all (backward-compatible).
pub(crate) fn canonicalize(m: &mut Module, roots: Option<&[Symbol]>) {
    if no_canon() {
        return;
    }
    loop {
        let mut changed = false;
        changed |= fold::const_fold(m);
        changed |= fold::resolve_alias_refs(m);
        changed |= fold::sweep_dead_bindings(m);
        changed |= inline::inline_user_calls(m);
        changed |= flatten::flatten_structural(m);
        if !changed {
            break;
        }
        let _ = flatppl_infer::infer(m);
    }
    if let Some(roots) = roots {
        dce::retain_reachable(m, roots);
    }
}
