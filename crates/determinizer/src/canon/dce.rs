//! Root-based dead-code elimination (Buffy #263 Pass 4-A): keep only the
//! bindings reachable from a requested-output root set; remove the rest. Runs
//! after the `canonicalize` fixpoint, when the dead-set is stable. Subsumes the
//! `synthetic`-only sweep when roots are given: it removes synthetic
//! scaffolding, zeroed measure stubs, AND dead value bindings uniformly.

use std::collections::HashSet;

use flatppl_core::{BindingId, Module, Symbol};

use crate::driver::collect_referenced_names;

/// Keep exactly the bindings reachable from `roots` (transitively via `%ref self`
/// leaves and reification `Inputs`); remove the rest. A root named in `roots` is
/// always kept, even if unreferenced (it is the requested output — this is what
/// closes the Pass-1 `__score__` trap). A root name not present in the module is
/// ignored (no panic).
pub(crate) fn retain_reachable(m: &mut Module, roots: &[Symbol]) {
    // Seed the worklist with the root bindings that exist.
    let mut keep: HashSet<BindingId> = HashSet::new();
    let mut work: Vec<BindingId> = Vec::new();
    for &r in roots {
        if let Some(bid) = m.binding_by_name(r) {
            if keep.insert(bid) {
                work.push(bid);
            }
        }
    }
    // Transitive closure: for each kept binding, mark every binding it references.
    while let Some(bid) = work.pop() {
        let rhs = m.binding(bid).rhs;
        let mut names: HashSet<Symbol> = HashSet::new();
        collect_referenced_names(m, rhs, &mut names);
        for name in names {
            if let Some(dep) = m.binding_by_name(name) {
                if keep.insert(dep) {
                    work.push(dep);
                }
            }
        }
    }
    m.retain_bindings(&keep);
}
