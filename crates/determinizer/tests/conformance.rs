use flatppl_determinizer::{determinize, is_flatpdl};

fn infer_module(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

#[test]
fn deterministic_model_is_flatpdl() {
    // pure deterministic arithmetic — no measure, no draw
    let m = infer_module("x = elementof(reals)\ny = add(x, 1.0)");
    assert!(is_flatpdl(&m).is_ok(), "{:?}", is_flatpdl(&m));
}

#[test]
fn a_draw_is_not_flatpdl() {
    // a stochastic node violates the predicate (Stochastic phase / Measure-typed law)
    let m = infer_module("z = draw(Normal(mu = 0.0, sigma = 1.0))");
    let v = is_flatpdl(&m).unwrap_err();
    assert!(
        v.iter().any(|n| matches!(
            n.kind,
            flatppl_determinizer::NonConformKind::StochasticPhase
                | flatppl_determinizer::NonConformKind::MeasureTyped
        )),
        "expected a stochastic/measure violation; got: {v:?}"
    );
}

// --- Task 2 tests ---

#[test]
fn already_deterministic_determinizes_to_conformant() {
    let m = infer_module("x = elementof(reals)\ny = add(x, 1.0)");
    let out = determinize(&m).expect("deterministic model must not refuse");
    assert!(is_flatpdl(&out).is_ok());
}

#[test]
fn unhandled_measure_algebra_refuses_clearly() {
    // joint of two laws — not yet lowered => a clear refusal naming the construct
    let m = infer_module(
        "a = Normal(mu = 0.0, sigma = 1.0)\nb = Normal(mu = 1.0, sigma = 1.0)\nj = joint(a, b)",
    );
    let e = determinize(&m).unwrap_err();
    assert!(
        e.construct.contains("joint"),
        "refusal must name the construct; got: {e:?}"
    );
}

// A bare `likelihoodof(kernelof(...), obs)` binding — inferred but NOT
// determinized — is non-conformant on TWO counts that `is_flatpdl` reads straight
// off the `flatppl-infer` side-tables:
//   * the `likelihoodof` node itself infers to `Type::Likelihood` — a measure-
//     layer type that is OUT of FlatPDL (the `Type::Likelihood` arm of `visit`),
//     so a `LikelihoodTyped` violation must be reported; and
//   * the `kernelof(...)` node infers to `Type::Kernel` but sits OUTSIDE any
//     `builtin_*` call argument (it is a plain top-level binding RHS), so the
//     "kernel only as a builtin_* arg" rule flags it as `KernelNotBuiltinArg`.
// This asserts BOTH kinds appear in the violation vector (there will also be
// `StochasticPhase` / `MeasureTyped` violations from the surviving draw and the
// inner Normal law — we do not require their absence, only that the two
// measure-algebra-type arms fired).
#[test]
fn bare_likelihoodof_of_kernel_reports_likelihood_and_kernel_violations() {
    let m = infer_module(
        "mu = elementof(reals)\n\
         k = kernelof(record(y = draw(Normal(mu = mu, sigma = 1.0))), mu = mu)\n\
         L = likelihoodof(k, record(y = 0.5))",
    );
    let v = is_flatpdl(&m).unwrap_err();
    assert!(
        v.iter().any(|n| matches!(
            n.kind,
            flatppl_determinizer::NonConformKind::LikelihoodTyped
        )),
        "the likelihoodof node is Likelihood-typed => LikelihoodTyped; got: {v:?}"
    );
    assert!(
        v.iter().any(|n| matches!(
            n.kind,
            flatppl_determinizer::NonConformKind::KernelNotBuiltinArg
        )),
        "the kernelof node is Kernel-typed outside a builtin_* arg => KernelNotBuiltinArg; got: {v:?}"
    );
}

// --- Pass 4 Task A review Fix 3: dangling-ref self-check ---
//
// Root-based DCE (Buffy #263 Pass 4-A, `canon::dce::retain_reachable`) is the
// FIRST capability that removes bindings outright. A latent miss in its
// reachability walk (`driver::collect_referenced_names`) could drop a binding
// something else still points at, leaving a `(%ref self <name>)` — as an
// ordinary body sub-node or a `functionof`/`kernelof` reification `Inputs`
// boundary entry — dangling. `is_flatpdl` is the conformance gate every
// determinized module passes through, so it is the permanent, always-on place
// to catch this: these tests hand-build a `Module` (bypassing the parser,
// which cannot produce a dangling ref by construction) with each dangling-ref
// shape and assert `is_flatpdl` reports `NonConformKind::DanglingSelfRef`.

#[test]
fn dangling_self_ref_in_body_is_flagged() {
    use flatppl_core::{Binding, Module, Node, Ref, RefNs};

    let mut m = Module::new();
    let missing = m.intern("missing");
    // `x = (%ref self missing)` — `missing` names no binding in the module.
    let dangling = m.alloc(Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name: missing,
    }));
    let x_name = m.intern("x");
    m.add_binding(Binding {
        name: x_name,
        rhs: dangling,
        doc: None,
        public: true,
        synthetic: false,
    });

    let v = is_flatpdl(&m).expect_err("a dangling body self-ref must be non-conformant");
    assert!(
        v.iter().any(|n| matches!(
            n.kind,
            flatppl_determinizer::NonConformKind::DanglingSelfRef
        )),
        "expected a DanglingSelfRef violation; got: {v:?}"
    );
}

#[test]
fn dangling_self_ref_in_reification_input_is_flagged() {
    use flatppl_core::{Binding, Call, CallHead, Inputs, Module, Node, Ref, RefNs};

    let mut m = Module::new();
    // k = functionof(2.0, g = g) — `g` names no binding in the module. This is
    // exactly the shape `canon::dce::retain_reachable` must keep alive when `g`
    // IS present (see `canon_dce_golden.rs`'s
    // `dce_keeps_binding_referenced_only_via_reification_input`); here `g` is
    // simply absent from the start, standing in for "DCE dropped it".
    let two = m.alloc(Node::Lit(flatppl_core::Scalar::Real(2.0)));
    let g_name = m.intern("g");
    let functionof = m.intern("functionof");
    let k_rhs = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(functionof),
        args: vec![two].into(),
        named: Vec::new().into(),
        inputs: Some(Inputs::Spec(
            vec![(
                g_name,
                Ref {
                    ns: RefNs::SelfMod,
                    name: g_name,
                },
            )]
            .into(),
        )),
    }));
    let k_name = m.intern("k");
    m.add_binding(Binding {
        name: k_name,
        rhs: k_rhs,
        doc: None,
        public: true,
        synthetic: false,
    });

    let v =
        is_flatpdl(&m).expect_err("a dangling reification Inputs self-ref must be non-conformant");
    assert!(
        v.iter().any(|n| matches!(
            n.kind,
            flatppl_determinizer::NonConformKind::DanglingSelfRef
        )),
        "expected a DanglingSelfRef violation for the dangling reification input; got: {v:?}"
    );
}
