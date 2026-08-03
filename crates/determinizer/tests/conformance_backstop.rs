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

/// `canon::inline` beta-reduces every user call it CAN reduce and leaves the
/// rest in place refuse-free, so an arity mismatch at the call site survived to
/// exit as a live `(%call (%ref self scale) 1.5 3.0)`. FlatPDL admits
/// deterministic ops and the six `builtin_*` primitives (§07 "Measure kernel
/// evaluation primitives"); an application of a user-defined callable is
/// neither, so the gate must reject it — which turns the case from exit 0 with
/// output no engine can evaluate into a refusal naming the construct.
#[test]
fn residual_user_call_from_arity_mismatch_is_rejected_and_refused() {
    let m = infer_module("scale(x) = mul(x, 2.0)\ns = scale(1.5, 3.0)\n");
    let v = is_flatpdl(&m).expect_err("a residual user call must be non-conformant");
    assert!(
        v.iter()
            .any(|n| matches!(n.kind, NonConformKind::ResidualUserCall)),
        "expected a ResidualUserCall violation; got: {v:?}"
    );
    let e = determinize(&m).expect_err("determinize must refuse rather than emit the residual");
    assert_eq!(e.construct, "ResidualUserCall", "refusal: {e:?}");
}

/// The other way `canon::inline` bails: the callee resolves to a non-callable
/// binding, so there is no body to substitute. The surviving `(%call (%ref self
/// f) 1.5)` is the same defect class as the arity-mismatch case above.
#[test]
fn residual_user_call_from_unresolved_callee_is_rejected_and_refused() {
    let m = infer_module("f = elementof(reals)\ns = f(1.5)\n");
    let v = is_flatpdl(&m).expect_err("a residual user call must be non-conformant");
    assert!(
        v.iter()
            .any(|n| matches!(n.kind, NonConformKind::ResidualUserCall)),
        "expected a ResidualUserCall violation; got: {v:?}"
    );
    let e = determinize(&m).expect_err("determinize must refuse rather than emit the residual");
    assert_eq!(e.construct, "ResidualUserCall", "refusal: {e:?}");
}

/// `builtin_logdensityof(kernel, kernel_input, x)` takes exactly three arguments
/// (§07 "Measure kernel evaluation primitives"). `flatppl-infer` has no arity
/// rule for the primitive — it types the two-argument call `Scalar(Real)`, not
/// `Type::Failed` — so the generic `Failed` backstop never fires and the arity
/// check is what must catch this. The second assertion pins that: a `Failed`
/// violation here would mean the check is redundant with inference.
#[test]
fn is_flatpdl_rejects_wrong_arity_builtin_primitive() {
    let m = infer_module("y = builtin_logdensityof(1.0, 2.0)\n");
    let v = is_flatpdl(&m).expect_err("a two-argument builtin_logdensityof must be non-conformant");
    assert!(
        v.iter()
            .any(|n| matches!(n.kind, NonConformKind::BuiltinArity)),
        "expected a BuiltinArity violation; got: {v:?}"
    );
    assert!(
        !v.iter().any(|n| matches!(n.kind, NonConformKind::Failed)),
        "inference must NOT already mark a mis-arity primitive Failed, else this \
         check duplicates it; got: {v:?}"
    );
}

/// `builtin_sample(rngstate, kernel, kernel_input, n, m, ...)` is the one
/// variadic primitive — the trailing sample shape is optional ("or a scalar `X`
/// if no `n, m, ...` are given", §07) — so three arguments is a minimum, not a
/// count. Neither the bare form nor a shaped one may be flagged.
#[test]
fn is_flatpdl_accepts_builtin_sample_with_and_without_a_shape() {
    for src in [
        "y = builtin_sample(1.0, 2.0, 3.0)\n",
        "y = builtin_sample(1.0, 2.0, 3.0, 4)\n",
    ] {
        let m = infer_module(src);
        let flagged = is_flatpdl(&m).err().is_some_and(|v| {
            v.iter()
                .any(|n| matches!(n.kind, NonConformKind::BuiltinArity))
        });
        assert!(!flagged, "builtin_sample must not be arity-flagged: {src}");
    }
    let m = infer_module("y = builtin_sample(1.0, 2.0)\n");
    let v = is_flatpdl(&m).expect_err("builtin_sample below three arguments is non-conformant");
    assert!(
        v.iter()
            .any(|n| matches!(n.kind, NonConformKind::BuiltinArity)),
        "expected a BuiltinArity violation; got: {v:?}"
    );
}

/// The arity count is positional PLUS named, so a keyword spelling is counted the
/// same as the positional one and is not falsely flagged at three arguments. No
/// determiniser emission site produces a named argument on a `builtin_*` primitive,
/// so this is the only cover the named half of the count has.
#[test]
fn builtin_primitive_arity_counts_named_arguments() {
    let ok = infer_module("y = builtin_logdensityof(kernel = 1.0, kernel_input = 2.0, x = 3.0)\n");
    assert!(
        !is_flatpdl(&ok).err().is_some_and(|v| v
            .iter()
            .any(|n| matches!(n.kind, NonConformKind::BuiltinArity))),
        "three named arguments must not be arity-flagged; got: {:?}",
        is_flatpdl(&ok)
    );

    let bad = infer_module(
        "y = builtin_logdensityof(kernel = 1.0, kernel_input = 2.0, x = 3.0, extra = 4.0)\n",
    );
    let v = is_flatpdl(&bad).expect_err("a fourth named argument is non-conformant");
    assert!(
        v.iter()
            .any(|n| matches!(n.kind, NonConformKind::BuiltinArity) && n.reason.contains("got 4")),
        "expected a BuiltinArity violation reporting four arguments; got: {v:?}"
    );
}

/// The surface printer spells a builtin call and a user call the same way —
/// `f(x)` either way — so surface syntax cannot show an external gate whether a
/// determinised module carries a residual user call. FlatPIR (§11) keeps the head
/// distinction, which is what `determinize --emit flatpir` exposes. This pins
/// that the rendering is readable back: written FlatPIR parses, and re-writing
/// the parsed module reproduces the same text.
#[test]
fn flatpir_rendering_of_flatpdl_round_trips() {
    let m = infer_module(
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         lp = logdensityof(lawof(record(a = a)), record(a = 0.5))\n",
    );
    let out = determinize(&m).expect("gaussian model must determinize");
    let pir = flatppl_flatpir::write(&out);
    let reparsed = flatppl_flatpir::read(&pir).expect("determinised FlatPIR must parse");
    assert_eq!(
        pir,
        flatppl_flatpir::write(&reparsed),
        "FlatPIR rendering of FlatPDL must round-trip through the reader"
    );
}
