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
/// rest in place refuse-free, so an arity mismatch at the call site survives to
/// exit as a live `(%call (%ref self scale) 1.5 3.0)`. FlatPDL admits
/// deterministic ops and the six `builtin_*` primitives (§07 "Measure kernel
/// evaluation primitives"); an application of a user-defined callable is
/// neither, so the gate reports it.
///
/// `flatppl-infer` now also marks the mis-arity application `Type::Failed`
/// (`scale` declares one parameter), so `is_flatpdl` reports both violations and
/// the refusal names the `Failed` node — inference reaches the defect first.
#[test]
fn residual_user_call_from_arity_mismatch_is_rejected_and_refused() {
    let m = infer_module("scale(x) = mul(x, 2.0)\ns = scale(1.5, 3.0)\n");
    let v = is_flatpdl(&m).expect_err("a residual user call must be non-conformant");
    assert!(
        v.iter()
            .any(|n| matches!(n.kind, NonConformKind::ResidualUserCall)),
        "expected a ResidualUserCall violation; got: {v:?}"
    );
    assert!(
        v.iter().any(|n| matches!(n.kind, NonConformKind::Failed)
            && n.reason.contains("declares 1 parameter, got 2")),
        "inference must mark the mis-arity application Failed; got: {v:?}"
    );
    let e = determinize(&m).expect_err("determinize must refuse rather than emit the residual");
    assert_eq!(e.construct, "Failed", "refusal: {e:?}");
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
/// (§07 "Measure kernel evaluation primitives"), and both nets now catch a
/// two-argument call: `flatppl-infer`'s catalogue arity rule marks it
/// `Type::Failed`, and this structural check reports `BuiltinArity`.
///
/// The structural check stays despite the overlap: `is_flatpdl` reads the
/// inferred side-tables, so on a module whose annotations came from somewhere
/// else — FlatPIR `%meta` read straight off disk, or an older inference run —
/// the `Failed` net is not there and only the call shape shows the defect.
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
        v.iter().any(|n| matches!(n.kind, NonConformKind::Failed)),
        "inference's arity rule must also mark the primitive Failed; got: {v:?}"
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

/// A FREE VARIABLE — a bare atom naming nothing in the `base` namespace and no
/// binding — must never reach `Ok(())`. `flatppl-infer` now rejects one at its
/// source (spec §04 "Name resolution"), so in practice the module below never
/// gets built; this arm is the structural backstop for a future path that
/// synthesises or re-admits one, which is why the module is assembled BY HAND
/// and inference is never run. With no type table, the `Type::Failed` arm cannot
/// fire, so `FreeBareName` is the only kind that can report it.
#[test]
fn is_flatpdl_rejects_a_free_bare_name_without_inference() {
    use flatppl_core::{Binding, Module, Node};

    let mut m = Module::new();
    let f1 = m.intern("f1");
    let free = m.alloc(Node::Const(f1));
    let name = m.intern("q");
    m.add_binding(Binding {
        name,
        rhs: free,
        doc: None,
        public: true,
        synthetic: false,
    });

    let v = is_flatpdl(&m).expect_err("a free bare name is non-conformant");
    assert!(
        v.iter()
            .any(|n| matches!(n.kind, NonConformKind::FreeBareName)
                && n.reason.contains("free variable `f1`")),
        "expected a FreeBareName violation naming `f1`; got: {v:?}"
    );
    assert!(
        !v.iter().any(|n| matches!(n.kind, NonConformKind::Failed)),
        "the check must not depend on the type table; got: {v:?}"
    );
}

/// The companion direction: a bare atom that IS a built-in (`pi`, and `sum` used
/// as a value) is conformant, so the gate does not reject ordinary FlatPDL.
#[test]
fn is_flatpdl_accepts_bare_builtin_atoms() {
    let m = infer_module("v = [1.0, 2.0]\ny = mul(pi, reduce(sum, v))\n");
    let out = determinize(&m).expect("a deterministic model must determinize");
    assert!(
        is_flatpdl(&out).is_ok(),
        "bare built-in atoms must stay conformant; got: {:?}",
        is_flatpdl(&out)
    );
}

/// End-to-end: the determiniser's exit gate turns the violation into a refusal
/// rather than emitting the free variable. This is the contract the reproducer
/// broke — `determinize` exited 0 emitting `builtin_logdensityof(Normal,
/// record(mu = f1, sigma = 1.5), 0.7)` with nothing binding `f1`.
#[test]
fn determinize_refuses_a_model_with_an_unbound_component_name() {
    let m = infer_module(
        "z = draw(Normal(mu = 0.4, sigma = 1.0))\n\
         q = logdensityof(joint(f1 = Normal(mu = z, sigma = 0.5), \
         f2 = Normal(mu = f1, sigma = 1.5)), record(f1 = 0.3, f2 = 0.7))\n",
    );
    // Both arms fire on the same node. `visit` reads the type table first, so
    // the `Type::Failed` backstop is `violations[0]` and names the refusal; the
    // CLI has already printed inference's own located diagnostic by then. The
    // structural arm is asserted separately below so a future path that stops
    // typing the node `Failed` still leaves the contract enforced.
    let err = determinize(&m).expect_err("an unbound component name must refuse, not lower");
    assert_eq!(err.construct, "Failed");
    assert_eq!(err.reason, "unresolvable name");

    let v = is_flatpdl(&m).expect_err("the inferred module is non-conformant");
    assert!(
        v.iter()
            .any(|n| matches!(n.kind, NonConformKind::FreeBareName)
                && n.reason.contains("free variable `f1`")),
        "the structural arm must also report `f1`; got: {v:?}"
    );
}
