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
