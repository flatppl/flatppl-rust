//! FlatPDL → FlatPDL canonicalization: post-measure-elimination normalization
//! passes that reduce/canonicalize the determiniser's output while preserving
//! `flatpdl.flatprof` conformance AND exact semantics (Buffy #263). Each pass
//! is idempotent and refuse-free; the driver runs them to a combined fixpoint.

use flatppl_core::Module;

mod flatten;
mod fold;
mod inline;

/// The `FLATPPL_DETERMINIZE_NO_CANON` env escape hatch: when set (to any value),
/// `canonicalize` is a no-op. Used to determinize a model both ways for the
/// numeric det-js equivalence gate; NOT a supported production toggle.
fn no_canon() -> bool {
    std::env::var_os("FLATPPL_DETERMINIZE_NO_CANON").is_some()
}

/// Run every canonicalization pass to a combined fixpoint. Re-infers between
/// sweeps because a reduction can shift inferred types/phases that a later pass
/// reads. A no-op if `FLATPPL_DETERMINIZE_NO_CANON` is set.
pub(crate) fn canonicalize(m: &mut Module) {
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
}
