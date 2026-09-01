//! Batched density (`builtin_logdensityof.(D, broadcast(record, …), vec)`) for
//! the two distributions whose scalar builder is not rank-agnostic on its own.
//!
//! `registry::lower_logdensityof_batched` routes an `iid` fan-out either to a
//! rank-agnostic builder (`registry::is_batch_safe`) or to a dedicated batched
//! one (`registry::batched_logpdf`). These tests pin both routes and the refusal
//! that guards the third case.
//!
//! `Uniform` takes the first route: §08's "outside the support the density is
//! zero" mask carries the variate's shape, so `iid(Uniform(S), n)` yields the
//! rank-1 vector the caller's `sum` reduces. Before the mask the builder returned
//! one scalar and `sum` counted a single term instead of `n`.
//!
//! `Categorical`/`Categorical0` take the second: `p` arrives wrapped by the batch
//! record and the variate is a vector, so the scalar single-slice lookup cannot
//! serve the call. The batched lowering is one `stablehlo.gather` into `log(p)`.

use flatppl_core::Module;

fn determinize_src(src: &str) -> Module {
    let mut m = flatppl_syntax::parse(src).expect("parse");
    // Notes are kept: `stdsimplex` has no type rule yet, so every fixture with a
    // simplex parameter carries one, and it is honest rather than a failure.
    let diags: Vec<_> = flatppl_infer::infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .collect();
    assert!(diags.is_empty(), "infer diagnostics: {diags:?}");
    flatppl_determinizer::determinize(&m).expect("must determinize, not refuse")
}

fn emit(src: &str) -> String {
    flatppl_stablehlo::emit(
        &determinize_src(src),
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect("must emit @logdensity")
}

fn refusal(src: &str) -> String {
    flatppl_stablehlo::emit(
        &determinize_src(src),
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect_err("must refuse")
    .to_string()
}

const IID_UNIFORM: &str = "\
flatppl_compat = \"0.1\"
square = iid(Uniform(interval(0.0, 1.0)), 2)
lp = logdensityof(square, [0.5, 0.8])
outputs = (lp)
";

/// `iid(Uniform(S), 2)`: the mask must be rank-1 over the two-element variate,
/// so the reduce that follows sums TWO `-log(1.0)` terms. A rank-0 mask (the
/// pre-mask builder's shape) would leave `sum` a scalar reduce and halve the
/// log-likelihood, which no structural check on the constant alone would catch.
#[test]
fn a_batched_uniform_density_masks_at_the_variates_rank() {
    let out = emit(IID_UNIFORM);
    assert!(
        out.contains("tensor<2xi1>"),
        "the support mask must carry the variate's rank-1 shape, in:\n{out}"
    );
    assert!(
        out.contains("stablehlo.reduce"),
        "the per-element densities must be reduced, in:\n{out}"
    );
}

/// §08's shared rule "outside the support the density is zero" — the scalar
/// path. `1.5` is outside `interval(0, 1)`, so the density is `-inf`, not the
/// finite `-log(1.0) = 0` the unmasked builder returned.
#[test]
fn a_uniform_density_is_neg_inf_off_its_support() {
    let out = emit(
        "\
flatppl_compat = \"0.1\"
m = Uniform(interval(0.0, 1.0))
lp = logdensityof(m, 1.5)
outputs = (lp)
",
    );
    // 0x7F800000 is f32 +inf; the mask negates it for the off-support branch.
    assert!(
        out.contains("dense<0x7F800000>") && out.contains("stablehlo.negate"),
        "expected a -inf off-support branch, in:\n{out}"
    );
    assert!(
        out.contains("compare GE") && out.contains("compare LE"),
        "§03 makes `interval(lo, hi)` CLOSED, so the guard is GE/LE, in:\n{out}"
    );
}

const IID_CATEGORICAL: &str = "\
flatppl_compat = \"0.1\"
obs = [1, 3, 2, 1]
p = elementof(stdsimplex(3))
lp = logdensityof(iid(Categorical(p = p), 4), obs)
inputs = (p)
outputs = (lp)
";

/// `iid(Categorical(p), 4)`: ONE `stablehlo.gather` into `log(p)` covers all
/// four mass terms. The scalar builder emits one `slice` + `reshape` + `log` per
/// observation, so a fan-out that fell back to it would emit four `log`s of four
/// slices — the count is what separates the two lowerings.
#[test]
fn a_batched_categorical_density_is_one_gather_into_log_p() {
    let out = emit(IID_CATEGORICAL);
    // The op name is quoted because the gather has no pretty form; matching the
    // bare name would also hit the `#stablehlo.gather<…>` attribute on the same
    // line and double every count.
    assert_eq!(
        out.matches("\"stablehlo.gather\"").count(),
        1,
        "expected exactly one gather, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.log ").count(),
        1,
        "expected `log(p)` once, taken before the gather, in:\n{out}"
    );
    assert!(
        out.contains("dense<1> : tensor<i32>"),
        "Categorical is 1-based, so the gather must subtract 1, in:\n{out}"
    );
}

/// `Categorical0` takes the same route with `base = 0`: the 0-based `k` is
/// already the array position, so nothing is subtracted off the index. The two
/// entries' numeric agreement is gated in `flatppl-testsuite`, not here — this
/// pins only that the 0-based entry reaches the batched gather at all.
#[test]
fn categorical0_batched_takes_the_same_gather_route_at_base_zero() {
    let out = emit(
        "\
flatppl_compat = \"0.1\"
obs = [0, 2, 1, 0]
p = elementof(stdsimplex(3))
lp = logdensityof(iid(Categorical0(p = p), 4), obs)
inputs = (p)
outputs = (lp)
",
    );
    assert_eq!(
        out.matches("\"stablehlo.gather\"").count(),
        1,
        "expected exactly one gather, in:\n{out}"
    );
    assert!(
        out.contains("dense<0> : tensor<i32>"),
        "Categorical0 is 0-based, so the gather must subtract 0, in:\n{out}"
    );
}

/// `stablehlo.gather` CLAMPS an out-of-range index rather than failing, so an
/// unchecked category would silently score the nearest one. `4` is past the end
/// of a length-3 `p`.
#[test]
fn an_out_of_range_batched_category_refuses() {
    let msg = refusal(
        "\
flatppl_compat = \"0.1\"
obs = [1, 4]
p = elementof(stdsimplex(3))
lp = logdensityof(iid(Categorical(p = p), 2), obs)
inputs = (p)
outputs = (lp)
",
    );
    assert!(
        msg.contains("category index out of range"),
        "expected the out-of-range refusal, got: {msg}"
    );
}

/// The default-deny gate still holds for a dist with neither route:
/// `NegativeBinomial2`'s builder uses `get0`/`reshape`, and it carries no batched
/// builder.
#[test]
fn a_dist_with_neither_route_still_refuses_under_broadcast() {
    let msg = refusal(
        "\
flatppl_compat = \"0.1\"
obs = [1, 3, 2, 1]
mu = elementof(posreals)
lp = logdensityof(iid(NegativeBinomial2(mu = mu, psi = 2.0), 4), obs)
inputs = (mu)
outputs = (lp)
",
    );
    assert!(
        msg.contains("is not rank-agnostic"),
        "expected the batch-safety refusal, got: {msg}"
    );
}
