//! `Type::Failed` conformance backstop (design doc "Component 5a"): any node
//! that `flatppl-infer` could not type ("inference attempted but failed; the
//! module is ill-formed", `flatppl_core::ty`) must never reach `Ok(())`
//! from `is_flatpdl` — it is not valid FlatPDL. `visit` already rejects
//! `Measure`/`Likelihood`/`Kernel`-typed nodes and a `Stochastic` phase, but
//! previously let a `Type::Failed` node fall through the wildcard arm. This is
//! the generic net; targeted ad-hoc refusals upstream (e.g. the cross-module
//! kernel-application argument check in `density.rs`) still fire earlier and
//! are unaffected.

use flatppl_determinizer::{NonConformKind, determinize, is_flatpdl};

fn infer_module(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

/// `cartpow(S, size)` requires the size argument (spec §03 "Cartesian
/// power"); the 1-arg form `cartpow(reals)` infers to `Type::Failed` (see
/// `infer/src/ops.rs`'s `cartpow` type arm). Nested inside
/// `elementof(cartpow(reals))`, the failed node survives as a child of the
/// `elementof` binding's RHS — exactly the kind of residual ill-formed node
/// the backstop must catch.
#[test]
fn is_flatpdl_rejects_residual_type_failed() {
    let m = infer_module("p = elementof(cartpow(reals))");
    let v = is_flatpdl(&m).unwrap_err();
    assert!(
        v.iter().any(|n| matches!(n.kind, NonConformKind::Failed)),
        "a residual Type::Failed node must be reported as NonConformKind::Failed; got: {v:?}"
    );
}

/// Regression guard for the tightening: a normal model, fully lowered by the
/// determiniser (no residual measure/likelihood/kernel/stochastic/failed
/// node), must still pass `is_flatpdl` after the backstop is added.
#[test]
fn is_flatpdl_accepts_valid_flatpdl() {
    let m = infer_module(
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         lp = logdensityof(lawof(record(a = a)), record(a = 0.5))\n",
    );
    let out = determinize(&m).expect("gaussian model must determinize");
    assert!(
        is_flatpdl(&out).is_ok(),
        "a validly lowered FlatPDL module must pass is_flatpdl; got: {:?}",
        is_flatpdl(&out)
    );
}
