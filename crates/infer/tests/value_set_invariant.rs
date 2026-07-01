//! The spec-§11 value-set refinement invariant, enforced as a regression guard.
//!
//! §11: a value-typed node's value-set is "a sound statically inferred set
//! containing the node's possible values … a subset of the type's natural
//! extent, defaulting to that extent when nothing tighter is known." So
//! `natural_of` (in `flatppl-core`) is THE canonical type→value-set mapping, and
//! every producer (`elementof`, `cartprod`, `load_data`, `iid`, `softmax`, …)
//! must only *refine* it. trace.rs makes `natural_of` the single fallback
//! chokepoint, but nothing checked that producers actually stay refinements —
//! and that is exactly the gap that let the tuple / table value-sets drift.
//!
//! This guard asserts `valueset ⊆ natural_of(type)` for every value-typed
//! binding across a corpus chosen to exercise the producers most prone to drift.
//! A producer that disagrees with the natural extent (as `load_data` did with
//! the pre-fix `natural_of(Table)`) fails here.

use flatppl_core::ValueSet;
use flatppl_infer::infer;

fn check_invariant(src: &str) {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = infer(&mut m);
    let ids: Vec<_> = m.bindings().map(|(id, _)| id).collect();
    for id in ids {
        let name = m.resolve(m.binding(id).name).to_string();
        let node = m.binding(id).rhs;
        let (Some(ty), Some(vs)) = (m.type_of(node), m.valueset_of(node)) else {
            continue;
        };
        let natural = ValueSet::natural_of(ty);
        // A claim only exists for a value-typed node (natural extent known) with
        // a concrete value-set.
        if matches!(natural, ValueSet::Unknown)
            || matches!(vs, ValueSet::Deferred | ValueSet::Unknown)
        {
            continue;
        }
        let vs_s = m.display_valueset(vs);
        let nat_s = m.display_valueset(&natural);
        assert!(
            vs.subset_of(&natural),
            "binding `{name}`: value-set `{vs_s}` is not a subset of the type's natural \
             extent `{nat_s}` — a producer has drifted from `natural_of` (spec §11)"
        );
    }
}

/// Every producer's value-set is a refinement of the type's natural extent.
#[test]
fn value_sets_are_refinements_of_natural_extent() {
    for src in [
        // refinements tighter than the type's natural extent
        "x = elementof(posreals)",
        "v = elementof(cartpow(reals, 3))",
        "s = elementof(stdsimplex(4))",
        "w = [0.2, 0.3, 0.5]\nsm = softmax(w)",
        // positional cartprod: CartProd value-set vs an array type (CartPow nat)
        "p = elementof(cartprod(reals, integers))",
        // all-vector cartprod: members `cat`, so blocks concatenate into a flat
        // 5-vector (spec §03) — its CartProd([cartpow 2, cartpow 3]) value-set
        // must still prove ⊆ cartpow(reals, 5) (block lengths sum, not count)
        "pv = elementof(cartprod(cartpow(reals, 2), cartpow(reals, 3)))",
        // heterogeneous block element sets: reals ++ integers, still a real vec
        "ph = elementof(cartprod(cartpow(reals, 2), cartpow(integers, 3)))",
        // keyword cartprod / records
        "r = elementof(cartprod(a = reals, b = posreals))",
        "rec = record(a = 1.0, b = record(c = 2))",
        // tables (natural fallback) + load_data (a producer whose value-set must
        // agree with natural_of(Table) — the drift the §03 fix removed)
        "t = table(a = [1.0, 2.0], b = [3, 4])",
        "tn = table(id = [1, 2], hits = table(x = [1.0, 2.0]))",
        "ld = load_data(source = \"d.csv\", valueset = cartprod(a = reals, b = posreals))",
        "lc = load_data(source = \"d.csv\", valueset = cartpow(posreals, 3))",
        // measures: the support is a refinement of the domain extent
        "m = truncate(Normal(mu = 0.0, sigma = 1.0), interval(0.0, 1.0))",
        "obs = iid(Normal(mu = 0.0, sigma = 1.0), 5)",
    ] {
        check_invariant(src);
    }
}
