//! FlatPDL → FlatPDL canonicalization: post-measure-elimination normalization
//! passes that reduce/canonicalize the determiniser's output while preserving
//! `flatpdl.flatprof` conformance AND exact semantics. Each pass
//! is idempotent and refuse-free; the driver runs them to a combined fixpoint.

pub(crate) use flatppl_core::query::{
    is_inputs_binding as is_reserved_abi_binding, promoted_input_names,
};
use flatppl_core::{Module, Symbol};

pub(crate) use flatppl_core::graph::retain_reachable;
mod flatten;
pub(crate) use flatten::{project, project_tuple};
mod fold;
mod inline;
pub(crate) use inline::{
    expand_composed_call, inline_expression, inline_user_calls, normalize_call_wrappers,
};
mod integrals;
pub(crate) use integrals::{is_closed as integral_is_closed, is_integral};
mod sharing;

/// The `FLATPPL_DETERMINIZE_NO_CANON` env escape hatch: when set (to any value),
/// `canonicalize` is a no-op. Used to determinize a model both ways for the
/// numeric det-js equivalence gate; NOT a supported production toggle.
fn no_canon() -> bool {
    std::env::var_os("FLATPPL_DETERMINIZE_NO_CANON").is_some()
}

/// Run every canonicalization pass to a combined fixpoint, then — if `roots` is
/// given — root-based dead-code elimination (Pass 4-A): only
/// bindings reachable from the requested-output `roots` survive. Re-infers
/// between sweeps because a reduction can shift inferred types/phases that a
/// later pass reads. A no-op if `FLATPPL_DETERMINIZE_NO_CANON` is set.
///
/// DCE runs after the fixpoint — the dead-set is only stable once
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
        // Inlining may expose a scan's concrete callable argument only now.
        // Refresh types before specializing it, then let the next sweep
        // canonicalize the specialized body.
        if changed {
            let _ = flatppl_infer::infer(m);
        }
        changed |= crate::density::broadcast::specialize_callable_sequences(m);
        changed |= flatten::flatten_structural(m);
        if !changed {
            break;
        }
        let _ = flatppl_infer::infer(m);
    }
    if let Some(roots) = roots {
        retain_reachable(m, roots);
    }
    let integrals_changed = integrals::hoist_integrals(m);
    if sharing::share_broadcasts(m, false) || integrals_changed {
        let _ = flatppl_infer::infer(m);
    }
}

/// Whether `m` reads a bare-name alias or literal binding that
/// [`resolve_alias_chains`] would point at the end of its chain. False for a
/// canonicalized module.
pub fn has_alias_chains(m: &Module) -> bool {
    fold::has_alias_refs(m)
}

/// Point every reference to a bare-name alias or literal binding at the end of
/// its chain, as canonicalization does, and re-infer. Promoted `inputs` keep
/// their references. Runs even under `FLATPPL_DETERMINIZE_NO_CANON`: a consumer
/// that reads a fixed operand through one ref hop needs it. Returns whether
/// anything changed.
pub fn resolve_alias_chains(m: &mut Module) -> bool {
    let mut changed = false;
    // Each sweep advances every chain one link; a chain is no longer than the
    // binding count.
    for _ in 0..=m.binding_count() {
        if !fold::resolve_alias_refs(m) {
            break;
        }
        changed = true;
    }
    if changed {
        let _ = flatppl_infer::infer(m);
    }
    changed
}

/// Prepare an already-determinized module for FlatPPL or FlatPIR serialization.
///
/// Text expands a DAG into a tree. Name repeated cell-local arrays as separate
/// broadcasts so their expressions are printed once, without cloning per point.
/// Native backends consume the shared DAG directly and do not need this pass.
/// `roots` has the same meaning as in `determinize_with_roots`.
pub fn prepare_serialization(m: &mut Module, roots: Option<&[Symbol]>) {
    if !no_canon() && sharing::share_broadcasts(m, true) {
        if let Some(roots) = roots {
            retain_reachable(m, roots);
        }
        let _ = flatppl_infer::infer(m);
    }
}
