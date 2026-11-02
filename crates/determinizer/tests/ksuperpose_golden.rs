//! Density lowering for direct `ksuperpose` — spec §06 "Density of composed
//! measures":
//!
//! > `ksuperpose` (weighted measure addition over the parameter family):
//! > `logdensityof(ksuperpose(κ, w, θ), x) = logsumexp_i(log wᵢ +
//! > logdensityof(κ(θᵢ), x))`, so a zero weight contributes −∞ and drops out.
//!
//! Scalar families retain the axis-native density expression; nested cells,
//! tables and multiple axes use one shared mapped component body.

mod common;

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
mix = ksuperpose(Normal, w, mu = mus, sigma = sigmas)
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
/// spec's own `normalize(ksuperpose(Normal, weights, …))` spelling has a
/// closed-form normalizer `sum(w)` and must LOWER rather than refuse for want of
/// `totalmass`.
#[test]
fn normalize_of_a_markov_mixture_divides_by_the_weight_sum() {
    let pir = lower(
        "w = [0.3, 1.2]\n\
         mus = [-1.0, 2.0]\n\
         sigmas = [1.0, 0.5]\n\
         mix = normalize(ksuperpose(Normal, w, mu = mus, sigma = sigmas))\n\
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
         mix = ksuperpose(Normal, w, mu = mus, sigma = 1.0)\n\
         lp = logdensityof(mix, 0.5)\n",
    );
    assert!(
        pir.contains("(%kwarg sigma 1.0)"),
        "the scalar sigma enters the per-component record unchanged:\n{pir}"
    );
    assert!(pir.contains("(logsumexp "), "still a mixture:\n{pir}");
}

/// §08's categorical over arbitrary values:
/// `normalize(ksuperpose(Dirac, p, value = labels))`. `Dirac` is a §06
/// fundamental measure, so this also pins that the lowering finds its parameter
/// name outside the §08 distribution catalogue.
#[test]
fn a_dirac_superposition_lowers_to_the_same_mixture_shape() {
    let pir = lower(
        "p = [0.2, 0.8]\n\
         labels = [0.0, 1.5]\n\
         c = normalize(ksuperpose(Dirac, p, value = labels))\n\
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
         mix = ksuperpose(Normal, w, [-1.0, 2.0], [1.0, 0.5])\n\
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
         mix = ksuperpose(Normal, w, mu = mus, sigma = 1.0)\n\
         lp = logdensityof(mix, 0.5)\n",
    );
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "one density term regardless of where the weights come from:\n{pir}"
    );
    assert!(pir.contains("(logsumexp "), "still a mixture:\n{pir}");
}

#[test]
fn reified_product_mixtures_share_one_density_body_at_any_component_count() {
    for n in [2, 500] {
        let pir = lower(&format!(
            "p = external(unitinterval)\n\
             counts(n) = iid(Binomial(n, p), 5)\n\
             sizes = indicesof(zeros({n}))\nweights = ones({n})\n\
             mix = normalize(ksuperpose(counts, weights, sizes))\n\
             lp = logdensityof(mix, [1, 1, 2, 2, 2])\n"
        ));
        assert_eq!(pir.matches("builtin_logdensityof").count(), 1, "{pir}");
        assert_eq!(pir.matches("(logsumexp ").count(), 1, "{pir}");
    }
}

const MULTIVARIATE: &str = "\
w = [0.2, 0.8]
mus = [[0.0, 0.0], [3.0, 3.0]]
c1 = rowstack([[1.0, 0.2], [0.2, 1.0]])
covs = [c1, c1]
mix = ksuperpose(MvNormal, w, mu = mus, cov = covs)
y = elementof(cartpow(reals, 2))
lp = logdensityof(mix, y)
";

/// Nested vectors and matrices feed one mapped component density body.
#[test]
fn a_multivariate_family_lowers_to_per_component_slices() {
    let pir = lower(MULTIVARIATE);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "one shared density body:\n{pir}"
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

/// A mixture draw is a component-index draw, then one draw of that component.
#[test]
fn sampling_a_mixture_draws_an_index_then_its_component() {
    for draw in [
        "x = draw(mix)\ndraws = rand(lawof(x), s)",
        "draws = rand(mix, s)",
        "draws = rand(iid(mix, 5), s)",
    ] {
        let pir = lower(&format!(
            "s = rnginit(0)\n\
             w = [0.3, 1.2]\n\
             mix = normalize(ksuperpose(Normal, w, mu = [-1.0, 2.0], sigma = 1.0))\n\
             {draw}\n"
        ));
        assert_eq!(pir.matches("(builtin_sample ").count(), 2, "{draw}:\n{pir}");
        assert!(pir.contains("Categorical0"), "{draw}:\n{pir}");
        assert!(!pir.contains("ksuperpose"), "{draw}:\n{pir}");
    }
}

/// §04 draws only from a probability measure, and §06 makes normalization
/// explicit. A mixture whose unit mass is not proven needs `normalize`.
#[test]
fn an_unnormalized_mixture_refuses_to_sample() {
    let msg = refusal(
        "s = rnginit(0)\n\
         x = draw(ksuperpose(Normal, [0.3, 1.2], mu = [-1.0, 2.0], sigma = 1.0))\n\
         draws = rand(lawof(x), s)\n",
    );
    assert!(
        msg.contains("probability measure") && msg.contains("normalize"),
        "got: {msg}"
    );
}

/// §06 "Provably normalized superpositions": unexpanded simplex weights and
/// literal weights that sum to one need no `normalize`.
#[test]
fn a_provably_normalized_mixture_samples_without_normalize() {
    for weights in ["p", "[0.25, 0.75]"] {
        let pir = lower(&format!(
            "p = external(stdsimplex(2))\n\
             x = draw(ksuperpose(Normal, {weights}, mu = [-1.0, 2.0], sigma = 1.0))\n\
             draws = rand(lawof(x), rnginit(0))\n"
        ));
        assert!(pir.contains("Categorical0"), "{weights}:\n{pir}");
    }
}

/// All-zero weights give the zero measure, which §06 `normalize` leaves undefined.
#[test]
fn a_zero_mass_mixture_refuses_to_sample() {
    let msg = refusal(
        "x = draw(normalize(ksuperpose(Normal, [0.0, 0.0], mu = [-1.0, 2.0], sigma = 1.0)))\n\
         draws = rand(lawof(x), rnginit(0))\n",
    );
    assert!(msg.contains("zero measure"), "got: {msg}");
}

/// Tables supply rows through both the splat and the unary record interface.
#[test]
fn table_rows_sample_through_both_kernel_interfaces() {
    for kernel in ["Normal", "(r -> Normal(r.mu, r.sigma))"] {
        for draw in ["rand(mix, s)", "rand(iid(mix, 4), s)"] {
            let pir = lower(&format!(
                "s = rnginit(0)\n\
                 p = table(mu = [-1., 2.], sigma = [1., 0.5])\n\
                 mix = normalize(ksuperpose({kernel}, [0.3, 1.2], p))\n\
                 draws = {draw}\n"
            ));
            assert_eq!(
                pir.matches("(builtin_sample ").count(),
                2,
                "{kernel}:\n{pir}"
            );
        }
    }
}

#[test]
fn iid_mixtures_of_vector_components_refuse() {
    let msg = refusal(
        "mix = normalize(ksuperpose(MvNormal, [1.0, 3.0], mu = [[0., 0.], [3., 3.]], cov = [eye(2)]))\n\
         draws = rand(iid(mix, 4), rnginit(0))\n",
    );
    assert!(msg.contains("scalar components only"), "got: {msg}");
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
         mix = ksuperpose(Normal, w, mu = mus, sigma = sigmas)\n\
         lp = logdensityof(mix, 0.5)\n",
    );
    assert!(
        pir.contains("(broadcast log (%ref self w))") && pir.contains("(logsumexp "),
        "the zero weight rides the ordinary log-weight path:\n{pir}"
    );
}

#[test]
fn tables_follow_both_splat_and_unary_record_interfaces() {
    for kernel in ["Normal", "(r -> Normal(r.mu, r.sigma))"] {
        let pir = lower(&format!(
            "p = table(mu = [-1.,2.], sigma = [1.,0.5])\nm = normalize(ksuperpose({kernel}, [0.3,1.2], p))\nlp = logdensityof(m, 0.5)"
        ));
        assert_eq!(pir.matches("builtin_logdensityof").count(), 1, "{pir}");
        assert!(pir.contains("(logsumexp "), "{pir}");
    }
}

#[test]
fn expanding_weights_are_summed_over_all_broadcast_cells() {
    for (weights, means) in [
        ("[2.]", "[-1.,2.]"),
        ("2.", "[-1.,2.]"),
        ("rowstack([[2.],[3.]])", "rowstack([[-1.,2.,3.]])"),
    ] {
        let pir = lower(&format!(
            "w = {weights}\nmeans = {means}\nm = ksuperpose(Normal, w, means, sigma = 1.)\nz = totalmass(m)\nlp = logdensityof(normalize(m), 0.5)"
        ));
        assert!(
            pir.contains("(sum ") && pir.contains("(logsumexp "),
            "{pir}"
        );
        assert!(!pir.contains("totalmass"), "{pir}");
    }
}
