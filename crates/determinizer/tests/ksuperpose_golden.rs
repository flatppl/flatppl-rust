//! Density lowering for an applied `ksuperpose` — spec §06 "Density of composed
//! measures":
//!
//! > `ksuperpose` (weighted measure addition over the parameter family):
//! > `logdensityof(ksuperpose(κ, w)(θ), x) = logsumexp_i(log wᵢ +
//! > logdensityof(κ(θᵢ), x))`, so a zero weight contributes −∞ and drops out.
//!
//! The emitted FlatPDL is AXIS-NATIVE — one `logsumexp` over one broadcast, never
//! `N` unrolled terms — because §06 makes `N` "the length of `weights`, which need
//! not be statically known". Numeric verification of these shapes lives in
//! `crates/stablehlo/tests/golden_ksuperpose.rs`, which executes them.

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
