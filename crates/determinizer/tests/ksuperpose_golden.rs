//! Density lowering for an applied `ksuperpose` — spec §06 "Density of composed
//! measures":
//!
//! > `ksuperpose` (weighted measure addition over the parameter family):
//! > `logdensityof(ksuperpose(κ, w)(θ), x) = logsumexp_i(log wᵢ +
//! > logdensityof(κ(θᵢ), x))`, so a zero weight contributes −∞ and drops out.
//!
//! For a SCALAR parameter family the emitted FlatPDL is AXIS-NATIVE — one
//! `logsumexp` over one broadcast, never `N` unrolled terms — because §06 makes
//! `N` "the length of `weights`, which need not be statically known". A
//! MULTIVARIATE family has no `broadcast(record, …)` form and takes the unrolled
//! per-component slice form instead, which is why it needs a static `N`. Numeric
//! verification of these shapes lives in
//! `crates/stablehlo/tests/golden_ksuperpose.rs`, which executes them, and — for
//! the multivariate family the StableHLO emitter cannot yet index — in
//! `flatppl-testsuite`'s `corpora/coverage/mv_mixture` against a frozen scipy
//! vector.

use flatppl_determinizer::{determinize, is_flatpdl};

fn lower(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = flatppl_infer::infer(&mut m);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "inference errors: {errors:?}");
    let out = determinize(&m).expect("must lower, not refuse");
    assert!(is_flatpdl(&out).is_ok(), "not FlatPDL-conformant");
    flatppl_flatpir::write(&out)
}

fn refusal(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let err = determinize(&m).expect_err("must refuse");
    format!("{} :: {}", err.construct, err.reason)
}

const TWO_NORMALS: &str = "\
w = [0.3, 1.2]
mus = [-1.0, 2.0]
sigmas = [1.0, 0.5]
mix = ksuperpose(Normal, w)(mu = mus, sigma = sigmas)
lp = logdensityof(mix, 0.5)
";

/// The §06 density rule, emitted as one axis-level expression: the log-weights
/// added to the per-component densities, contracted by `logsumexp`. One
/// `builtin_logdensityof` for any `N` — the broadcast IS the per-component
/// evaluation — and no measure layer left.
#[test]
fn the_mixture_density_is_one_logsumexp_over_one_broadcast() {
    let pir = lower(TWO_NORMALS);
    assert!(
        pir.contains("(logsumexp "),
        "logsumexp contracts the family:\n{pir}"
    );
    assert!(
        pir.contains("(broadcast log (%ref self w))"),
        "the log-weights are broadcast over the weight vector:\n{pir}"
    );
    assert!(
        pir.contains("(broadcast add "),
        "log wᵢ is added elementwise to the component densities:\n{pir}"
    );
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "axis-native: ONE density term for any N, not N unrolled terms:\n{pir}"
    );
    assert!(
        !pir.contains("ksuperpose") && !pir.contains("(logdensityof "),
        "the measure layer is gone:\n{pir}"
    );
}

/// The family axis is contracted with `logsumexp`, where `iid`'s independent
/// product over the same broadcast shape contracts with `sum`. Pinned as a pair
/// because that single op is the whole difference between a mixture and a
/// product, and swapping it is a silent, plausible-looking mislowering.
#[test]
fn the_mixture_contracts_with_logsumexp_where_iid_contracts_with_sum() {
    let mixture = lower(TWO_NORMALS);
    assert!(
        mixture.contains("(logsumexp ") && !mixture.contains("(sum (%meta"),
        "a mixture reduces with logsumexp, never sum:\n{mixture}"
    );
    let product = lower(
        "obs = [0.5, 0.7, 0.9]\n\
         m = iid(Normal(mu = 0.0, sigma = 1.0), 3)\n\
         lp = logdensityof(m, obs)\n",
    );
    assert!(
        product.contains("(sum ") && !product.contains("(logsumexp "),
        "an independent product reduces with sum, never logsumexp:\n{product}"
    );
}

/// §06 puts the lift's total mass at "$\sum_i w_i$ for a Markov `kernel`", so the
/// spec's own `normalize(ksuperpose(Normal, weights)(…))` spelling has a
/// closed-form normalizer `sum(w)` and must LOWER rather than refuse for want of
/// `totalmass`.
#[test]
fn normalize_of_a_markov_mixture_divides_by_the_weight_sum() {
    let pir = lower(
        "w = [0.3, 1.2]\n\
         mus = [-1.0, 2.0]\n\
         sigmas = [1.0, 0.5]\n\
         mix = normalize(ksuperpose(Normal, w)(mu = mus, sigma = sigmas))\n\
         lp = logdensityof(mix, 0.5)\n",
    );
    assert!(
        pir.contains("(log (%meta ((%scalar real) %fixed reals) (sum (%ref self w))))"),
        "Z = sum(w), so logZ = log(sum(w)):\n{pir}"
    );
    assert!(
        pir.contains("(sub ") && pir.contains("(logsumexp "),
        "the normalized density subtracts logZ from the mixture density:\n{pir}"
    );
    assert!(
        !pir.contains("totalmass"),
        "`totalmass` is not FlatPDL and must not appear:\n{pir}"
    );
}

/// §06: "Non-collection arguments are held constant across the components." The
/// scalar rides the broadcast whole; only the weights and the per-component `mu`
/// carry the axis.
#[test]
fn a_held_constant_scalar_parameter_rides_the_broadcast() {
    let pir = lower(
        "w = [0.3, 1.2]\n\
         mus = [-1.0, 2.0]\n\
         mix = ksuperpose(Normal, w)(mu = mus, sigma = 1.0)\n\
         lp = logdensityof(mix, 0.5)\n",
    );
    assert!(
        pir.contains("(%kwarg sigma 1.0)"),
        "the scalar sigma enters the per-component record unchanged:\n{pir}"
    );
    assert!(pir.contains("(logsumexp "), "still a mixture:\n{pir}");
}

/// §08's categorical over arbitrary values:
/// `normalize(ksuperpose(Dirac, p)(value = labels))`. `Dirac` is a §06
/// fundamental measure, so this also pins that the lowering finds its parameter
/// name outside the §08 distribution catalogue.
#[test]
fn a_dirac_superposition_lowers_to_the_same_mixture_shape() {
    let pir = lower(
        "p = [0.2, 0.8]\n\
         labels = [0.0, 1.5]\n\
         c = normalize(ksuperpose(Dirac, p)(value = labels))\n\
         lp = logdensityof(c, 1.5)\n",
    );
    assert!(
        pir.contains("broadcast builtin_logdensityof Dirac"),
        "the component kernel tag is the bare `Dirac` constructor:\n{pir}"
    );
    assert!(
        pir.contains("(%kwarg value (%ref self labels))"),
        "`Dirac`'s own `value` parameter carries the family:\n{pir}"
    );
    assert!(pir.contains("(logsumexp "), "still a mixture:\n{pir}");
}

/// §04 "Calling conventions": positional family arguments bind to the component
/// constructor's ordered parameter names, so the two spellings lower alike.
#[test]
fn positional_family_arguments_bind_to_the_component_parameter_names() {
    let positional = lower(
        "w = [0.3, 1.2]\n\
         mix = ksuperpose(Normal, w)([-1.0, 2.0], [1.0, 0.5])\n\
         lp = logdensityof(mix, 0.5)\n",
    );
    assert!(
        positional.contains("%kwarg mu") && positional.contains("%kwarg sigma"),
        "positional args are named by §08's parameter order:\n{positional}"
    );
}

/// §06: `N` "need not be statically known" — so weights that are not a literal
/// must lower through the same axis-native form, with no static size read.
#[test]
fn runtime_weights_lower_without_a_static_component_count() {
    let pir = lower(
        "w = external(cartpow(nonnegreals, 2))\n\
         mus = external(cartpow(reals, 2))\n\
         mix = ksuperpose(Normal, w)(mu = mus, sigma = 1.0)\n\
         lp = logdensityof(mix, 0.5)\n",
    );
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "one density term regardless of where the weights come from:\n{pir}"
    );
    assert!(pir.contains("(logsumexp "), "still a mixture:\n{pir}");
}

/// A CURRIED `ksuperpose` has no variate to score: §06 gives it a density only
/// "applied to a parameter family". A located refusal, not a panic and not a
/// density for the lift.
#[test]
fn an_unapplied_lift_refuses_with_a_located_message() {
    let msg = refusal(
        "w = [0.3, 1.2]\n\
         k = ksuperpose(Normal, w)\n\
         lp = logdensityof(k, 0.5)\n",
    );
    assert!(
        msg.contains("ksuperpose") && msg.contains("until it is applied"),
        "got: {msg}"
    );
}

/// A REIFIED component needs a per-component body evaluation with no backend
/// form (the gap `lower_iid` documents for its own `functionof`-broadcast
/// candidate). Refuse-don't-mislower.
#[test]
fn a_reified_component_refuses_rather_than_mislowering() {
    let msg = refusal(
        "w = [0.3, 1.2]\n\
         mus = [-1.0, 2.0]\n\
         k = kernelof(draw(Normal(mu = _m_, sigma = 1.0)), m = _m_)\n\
         mix = ksuperpose(k, w)(m = mus)\n\
         lp = logdensityof(mix, 0.5)\n",
    );
    assert!(msg.contains("not a bare measure constructor"), "got: {msg}");
}

/// A TABLE parameter family types (§06 counts its rows as the one axis) but the
/// per-column extraction is not built, so the lowering refuses with a message
/// naming the working spelling.
#[test]
fn a_table_family_refuses_and_names_the_working_spelling() {
    let msg = refusal(
        "w = [0.3, 1.2]\n\
         params = table(mu = [-1.0, 2.0], sigma = [1.0, 0.5])\n\
         mix = ksuperpose(Normal, w)(params)\n\
         lp = logdensityof(mix, 0.5)\n",
    );
    assert!(
        msg.contains("TABLE parameter family") && msg.contains("keyword vectors"),
        "got: {msg}"
    );
}

const MULTIVARIATE: &str = "\
w = [0.2, 0.8]
mus = rowstack([[0.0, 0.0], [3.0, 3.0]])
c1 = rowstack([[1.0, 0.2], [0.2, 1.0]])
covs = [c1, c1]
mix = ksuperpose(MvNormal, w)(mu = mus, cov = covs)
y = elementof(cartpow(reals, 2))
lp = logdensityof(mix, y)
";

/// §06's family rule reads a family argument's axes against the rank of the
/// parameter it feeds, so an $N \times d$ `mu` beside an $N \times d \times d$
/// `cov` is one legal mixture. It has no `broadcast(record, …)` form (§04
/// *Collection arguments* strips every axis to reach a cell), so it lowers to the
/// per-component slice form instead — the SAME §06 density rule, unrolled.
#[test]
fn a_multivariate_family_lowers_to_per_component_slices() {
    let pir = lower(MULTIVARIATE);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "one density term per component, N = 2 here:\n{pir}"
    );
    assert!(
        pir.contains("(logsumexp ") && !pir.contains("(sum "),
        "the components are still contracted with logsumexp, never sum:\n{pir}"
    );
    assert!(
        !pir.contains("ksuperpose") && !pir.contains("(logdensityof "),
        "the measure layer is gone:\n{pir}"
    );
}

/// The selector count is the argument's OUTER rank, not its total axis count.
/// §07 `get`: "the keyword `all` selects an entire axis: `get(M, i, all)` returns
/// row i". So the FLAT $N \times d$ `mus` takes one `all` for its remaining outer
/// axis, while the NESTED `covs = [c1, c1]` — a 2-vector of 2×2 matrices, a
/// distinct §03 type — needs no `all`, because its one index already lands on the
/// whole matrix. Pinned per argument because emitting one `all` too many for the
/// nested spelling is a silent, plausible-looking mislowering.
#[test]
fn each_family_argument_is_sliced_by_its_own_outer_rank() {
    let pir = lower(MULTIVARIATE);
    assert!(
        pir.contains("(get (%ref self mus) 1 all)") && pir.contains("(get (%ref self mus) 2 all)"),
        "the flat N x d `mus` slices a row per component:\n{pir}"
    );
    assert!(
        pir.contains("(get (%ref self covs) 1)") && pir.contains("(get (%ref self covs) 2)"),
        "the nested N-vector of matrices takes one selector, no `all`:\n{pir}"
    );
    assert!(
        !pir.contains("(get (%ref self covs) 1 all"),
        "an `all` for an axis the ELEMENT carries would index into the matrix:\n{pir}"
    );
    assert!(
        pir.contains("(get (%ref self w) 1)") && pir.contains("(get (%ref self w) 2)"),
        "each component reads its own weight, 1-based per §07:\n{pir}"
    );
}

/// §06: "a collection whose leading axis has size one … is shared by every
/// component", so a singular `cov` reads row 1 for both components while `mu`
/// keeps advancing.
#[test]
fn a_singular_family_axis_is_shared_by_every_component() {
    let pir = lower(
        "w = [0.2, 0.8]\n\
         mus = rowstack([[0.0, 0.0], [3.0, 3.0]])\n\
         c1 = rowstack([[1.0, 0.2], [0.2, 1.0]])\n\
         covs = [c1]\n\
         mix = ksuperpose(MvNormal, w)(mu = mus, cov = covs)\n\
         y = elementof(cartpow(reals, 2))\n\
         lp = logdensityof(mix, y)\n",
    );
    assert_eq!(
        pir.matches("(get (%ref self covs) 1)").count(),
        2,
        "the singular cov is read at row 1 by both components:\n{pir}"
    );
    assert!(
        pir.contains("(get (%ref self mus) 2 all)"),
        "the size-N `mus` still advances:\n{pir}"
    );
}

/// §06 makes `N` "not necessarily statically known" and the axis-native scalar
/// form honours that, but the per-component slice extraction emits one `get` per
/// component, so it needs a static `N`. Refuse rather than guess a count.
#[test]
fn a_multivariate_family_with_a_dynamic_n_refuses() {
    let msg = refusal(
        "n = elementof(posintegers)\n\
         w = external(cartpow(nonnegreals, n))\n\
         mus = external(cartpow(cartpow(reals, 2), n))\n\
         c1 = rowstack([[1.0, 0.2], [0.2, 1.0]])\n\
         covs = [c1, c1]\n\
         mix = ksuperpose(MvNormal, w)(mu = mus, cov = covs)\n\
         y = elementof(cartpow(reals, 2))\n\
         lp = logdensityof(mix, y)\n",
    );
    assert!(
        msg.contains("statically known component count"),
        "got: {msg}"
    );
}

/// §06 says the mixture "is sampleable whenever `kernel` is", so the refusal must
/// read as UNIMPLEMENTED — a component-index draw that is not built — and must not
/// be confused with `weighted`'s genuine intractability.
#[test]
fn sampling_a_mixture_refuses_as_unimplemented_not_intractable() {
    let msg = refusal(
        "s = rnginit(0)\n\
         w = [0.3, 1.2]\n\
         mix = normalize(ksuperpose(Normal, w)(mu = [-1.0, 2.0], sigma = 1.0))\n\
         x = draw(mix)\n\
         draws = rand(s, lawof(x))\n",
    );
    assert!(
        msg.contains("ksuperpose") && msg.contains("not implemented"),
        "got: {msg}"
    );
    assert!(
        !msg.contains("intractable"),
        "§06 makes the mixture sampleable in principle; the message must not say \
         otherwise: {msg}"
    );
}

/// A single zero weight contributes `log 0 = −∞` and drops out of the
/// `logsumexp` — §06's own words. Nothing special is emitted for it; the shape is
/// the ordinary one, and the arithmetic does the work.
#[test]
fn a_zero_weight_needs_no_special_shape() {
    let pir = lower(
        "w = [0.0, 1.2]\n\
         mus = [-1.0, 2.0]\n\
         sigmas = [1.0, 0.5]\n\
         mix = ksuperpose(Normal, w)(mu = mus, sigma = sigmas)\n\
         lp = logdensityof(mix, 0.5)\n",
    );
    assert!(
        pir.contains("(broadcast log (%ref self w))") && pir.contains("(logsumexp "),
        "the zero weight rides the ordinary log-weight path:\n{pir}"
    );
}
