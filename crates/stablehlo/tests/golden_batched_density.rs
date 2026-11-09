//! Batched density (`builtin_logdensityof.(D, broadcast(record, …), vec)`) for
//! distributions whose scalar builder is not rank-agnostic on its own.
//!
//! `registry::lower_logdensityof_batched` routes an `iid` fan-out by each
//! distribution's `registry::Broadcast`: an elementwise builder, a dedicated batched
//! one, or one vector or matrix cell per element. These tests pin the routes.
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

const IID_UNIFORM: &str = "\
flatppl_compat = \"0.1\"
x = elementof(cartpow(reals, 2))
square = iid(Uniform(interval(0.0, 1.0)), 2)
lp = logdensityof(square, x)
inputs = (x)
outputs = (lp)
";

/// `iid(Uniform(S), 2)` at a run-time point (a fixed point folds the mask): the
/// mask must be rank-1 over the two-element variate,
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
        !out.contains("stablehlo.select"),
        "the constant off-support mask folds to the -inf branch, in:\n{out}"
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

/// `iid(Categorical(p), 4)`: one gather followed by log covers all four mass terms. The scalar builder emits one `slice` + `reshape` + `log` per
/// observation, so a fan-out that fell back to it would emit four `log`s of four
/// slices — the count is what separates the two lowerings.
#[test]
fn a_batched_categorical_density_uses_one_gather_and_log() {
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
        "expected one log after the gather, in:\n{out}"
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
        out.contains("dense<[0, 2, 1, 0]> : tensor<4xi32>"),
        "Categorical0 already supplies 0-based gather indices:\n{out}"
    );
}

/// `NegativeBinomial2`'s density is arithmetic and `lgamma` in `mu`, `psi` and
/// the count, so a broadcast scores every observation at once.
#[test]
fn a_negative_binomial2_broadcast_scores_each_count() {
    let out = emit(
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
        out.contains("applies stablehlo.add across dimensions = [0] : (tensor<4xf32>"),
        "expected one sum over the four per-count terms, in:\n{out}"
    );
}

/// A multivariate density scores one vector cell per element: `Dirichlet.(A)` pairs
/// each row of `A` with the matching observation, in one vectorized program with no
/// `while`. Values are checked against scipy in the testsuite.
#[test]
fn a_vector_density_scores_each_cell_under_broadcast() {
    let out = emit(
        "\
flatppl_compat = \"0.1\"
a = elementof(posreals)
lp = logdensityof(Dirichlet.([[a, 2.0], [2.0, a]]), [[0.3, 0.7], [0.6, 0.4]])
inputs = (a)
outputs = (lp)
",
    );
    assert!(
        out.contains("tensor<2x2xf32>"),
        "the cells stay one batch, in:\n{out}"
    );
    assert!(
        !out.contains("stablehlo.while"),
        "no per-cell loop, in:\n{out}"
    );
}

/// One probability vector per observation: `rows[idx]` selects a vector for
/// each variate cell, so the batched route must pair cell `i` of `p` with
/// observation `i`. The fixed index vector made `p` a `[3, 2]` array of cells,
/// which the shared-vector route read as a single rank-2 cell and refused.
/// Each observation reads one probability of its source row, never a whole row.
#[test]
fn a_categorical_broadcast_over_gathered_rows_pairs_each_row_with_its_observation() {
    let out = emit(
        "\
flatppl_compat = \"0.1\"
a = elementof(unitinterval)
rows = [[a, 1 - a], [1 - a, a]]
lp = logdensityof(Categorical.(rows[[1, 2, 2]]), [1, 1, 2])
inputs = (a)
outputs = (lp)
",
    );
    let scores = out
        .lines()
        .filter(|line| {
            line.contains("\"stablehlo.gather\"")
                && line.contains("(tensor<4xf32>, tensor<3x1xi32>) -> tensor<3xf32>")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        scores.len(),
        1,
        "expected one gather of the three category probabilities from the flattened rows, in:\n{out}"
    );
    assert!(
        !out.contains("tensor<3x2xf32>"),
        "no row is gathered per observation, in:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.log ").count(),
        1,
        "expected one log after the gather, in:\n{out}"
    );
}

/// A function that returns a probability vector, broadcast over an index
/// vector, takes the same per-cell route.
#[test]
fn a_categorical_broadcast_over_computed_probability_vectors_emits() {
    let out = emit(
        "\
flatppl_compat = \"0.1\"
a = elementof(reals)
probs(s, i) = softmax([s * i, 0.0, -s])
lp = logdensityof(Categorical.(probs.(a, [1, 2, 3, 4])), [1, 3, 2, 1])
inputs = (a)
outputs = (lp)
",
    );
    assert!(
        out.contains("operand_batching_dims = [0]") && out.contains("tensor<4x3xf32>"),
        "each observation must index its own probability vector, in:\n{out}"
    );
}

/// Every §08 distribution scores under broadcast. The matrix ones score one matrix
/// cell per element through the same per-cell route, whether the parameters are
/// shared (`iid`) or one per cell (a dotted constructor). Values are checked against
/// scipy and closed forms in the testsuite.
#[test]
fn matrix_and_vector_densities_score_each_cell() {
    let corr = "[rowstack([[1.0, 0.3], [0.3, 1.0]]), rowstack([[1.0, -0.2], [-0.2, 1.0]])]";
    let chol = "[rowstack([[1.0, 0.0], [0.3, 0.9539392014169457]]), \
                rowstack([[1.0, 0.0], [-0.2, 0.9797958971132712]])]";
    let spd = "[rowstack([[1.0, 0.2], [0.2, 2.0]]), rowstack([[3.0, -0.5], [-0.5, 1.5]])]";
    let scale = "rowstack([[2.0, 0.3], [0.3, 1.0]])";
    for density in [
        format!("logdensityof(iid(LKJ(2, a), 2), {corr})"),
        format!("logdensityof(LKJ.(2, [a, 2.0 * a]), {corr})"),
        format!("logdensityof(iid(LKJCholesky(2, a), 2), {chol})"),
        format!("logdensityof(LKJCholesky.(2, [a, 2.0 * a]), {chol})"),
        format!("logdensityof(iid(Wishart(a + 3.0, {scale}), 2), {spd})"),
        format!("logdensityof(Wishart.([a + 3.0, a + 4.0], fill({scale}, 2)), {spd})"),
        format!("logdensityof(iid(InverseWishart(a + 3.0, {scale}), 2), {spd})"),
        format!("logdensityof(InverseWishart.([a + 3.0, a + 4.0], fill({scale}, 2)), {spd})"),
        "logdensityof(Multinomial.(5, [[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]), [[1, 2, 2], [0, 1, 4]])"
            .to_string(),
    ] {
        let out = emit(&format!(
            "flatppl_compat = \"0.1\"\na = elementof(posreals)\nlp = {density}\n\
             inputs = (a)\noutputs = (lp)\n"
        ));
        assert!(
            !out.contains("stablehlo.while"),
            "{density}: no per-cell loop, in:\n{out}"
        );
    }
}
