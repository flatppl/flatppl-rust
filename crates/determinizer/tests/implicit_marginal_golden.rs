//! `lawof(y)` over a `y ~ Dist(param = z, …)` with `z` latent is the conjugate marginal,
//! not a refusal.
//!
//! §04 *Reification to measures* makes `lawof(x)` the total law of `x`, and §04 *Kernels
//! and `kernelof`* says which ancestors go: a `prior_predictive = lawof(record(obs =
//! obs))` is "obtained by marginalizing over `theta1` and `theta2` — they are internal
//! stochastic nodes in the traced sub-DAG, not boundary inputs, so `lawof` integrates them
//! out. `prior_predictive` is equivalent to `kchain(prior, forward_kernel)`." The guard
//! that stopped the CONDITIONAL density escaping refused this whole shape, including the
//! pairs `CONJUGATE_TABLE` already answers in closed form. It now tries the table first.
//!
//! The maths of each row, its test point, and the wrong answer that point discriminates
//! against are in `src/marginal.md`. The numbers quoted below are verified there against
//! Distributions.jl plus quadrature of the same integral; nothing here computes one —
//! flatppl-rust is not a density engine, so every assertion is on the emitted FlatPDL,
//! passed through `is_flatpdl`.
use flatppl_determinizer::{determinize, is_flatpdl};

mod common;
use common::pir_binding;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

/// The FlatPIR of a module that must lower, conformance-checked on the way through.
fn pir(src: &str) -> String {
    let out = determinize(&parse_infer(src)).expect("must lower, not refuse");
    let text = flatppl_flatpir::write(&out);
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl failed:\n{text}");
    text
}

fn refusal(src: &str) -> String {
    determinize(&parse_infer(src))
        .expect_err("no conjugate row applies — refuse, do not score the conditional")
        .reason
}

// Row 1 (Normal prior on a Normal mean), in both spellings and both statement orders —
// four routes to the same integral, and each reaches the table differently. `lp_y` first
// finds `z` still a `draw`; `lp_z` first finds it a pinned literal, and the marginal is
// then only buildable from the pin's recorded provenance. The bare spelling routes through
// the value-law guard, the record spelling through the record-law guard.
//
// The marginal is `Normal(0, √2)`: `sqrt(1² + 1²)` const-folds to 1.4142135623730951.
// logdensity at y = 0.5 is -1.3280121234846454. The wrong answer this pins out is the
// conditional `Normal(0.3, 1)` at 0.5 = -0.9389385332046728, the density that escaped as a
// finished number when a sibling query had pinned `z = 0.3` — 0.389 nats off.
#[test]
fn normal_prior_on_a_mean_marginalizes_in_both_spellings_and_orders() {
    let model = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
";
    for query in [
        "lp_z = logdensityof(lawof(z), 0.3)\nlp_y = logdensityof(lawof(y), 0.5)",
        "lp_y = logdensityof(lawof(y), 0.5)\nlp_z = logdensityof(lawof(z), 0.3)",
        "lp_z = logdensityof(lawof(record(z = z)), record(z = 0.3))\n\
         lp_y = logdensityof(lawof(record(y = y)), record(y = 0.5))",
        "lp_y = logdensityof(lawof(record(y = y)), record(y = 0.5))\n\
         lp_z = logdensityof(lawof(record(z = z)), record(z = 0.3))",
    ] {
        let text = pir(&format!("{model}{query}"));
        let lp_y = pir_binding(&text, "lp_y");
        assert_eq!(
            lp_y.matches("builtin_logdensityof").count(),
            1,
            "the closed-form marginal is a single density term:\n{lp_y}"
        );
        assert!(
            lp_y.contains("(builtin_logdensityof Normal "),
            "the Normal–Normal marginal is a Normal:\n{lp_y}"
        );
        // sqrt(σ0² + σ²) = sqrt(2), const-folded. A variance DIFFERENCE folds to 0.0 and
        // either term alone to 1.0, so this literal pins the variance sum by itself.
        assert!(
            lp_y.contains("(%field sigma 1.4142135623730951)"),
            "marginal sigma is sqrt(σ0² + σ²) = sqrt(2):\n{lp_y}"
        );
        assert!(
            lp_y.contains("(%field mu 0.0)"),
            "marginal mean is the prior mu:\n{lp_y}"
        );
        // The wrong answer, structurally: the conditional would carry the likelihood's own
        // sigma at the pinned latent.
        assert!(
            !lp_y.contains("(%field mu 0.3)") && !lp_y.contains("(%field sigma 1.0)"),
            "not the conditional at the pinned latent:\n{lp_y}"
        );
        // `lp_z` scores z's own PRIOR, unchanged by the marginalization of `y`.
        assert!(
            pir_binding(&text, "lp_z").contains("(%field sigma 1.0)"),
            "the latent's own query still scores its prior:\n{text}"
        );
        assert!(
            !text.contains("lawof") && !text.contains("(draw ") && !text.contains("kchain"),
            "measure layer gone:\n{text}"
        );
    }
}

// The marginal does not need a second query to exist. With `lawof(y)` the only query the
// latent is never pinned, so this route reaches the table through the still-present `draw`
// rather than through the pin provenance.
#[test]
fn a_marginal_lowers_with_no_second_query_to_pin_the_latent() {
    let lp_y = pir_binding(
        &pir("\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
lp_y = logdensityof(lawof(y), 0.5)"),
        "lp_y",
    );
    assert!(
        lp_y.contains("(%field sigma 1.4142135623730951)"),
        "marginal sigma is sqrt(2):\n{lp_y}"
    );
}

// Row 2 (Gamma prior on a Poisson rate), both spellings. §08's
// `NegativeBinomial(alpha, beta)` IS the Gamma(shape = α, rate = β)–Poisson mixture, so
// the parameter map is the identity and the emission carries no arithmetic.
//
// Prior `Gamma(shape = 2, rate = 1/3)` (scale 3) → marginal
// `NegativeBinomial(alpha = 2, beta = 1/3)`, i.e. p = 1/(1+3) = 1/4. logdensity at k = 5 is
// -2.419239615270632. The wrong answer this pins out is the plug-in `Poisson(6)` at k = 5 =
// -1.8286943966417715 — the likelihood at the prior MEAN rate α/β = 6, which shares the
// marginal's mean, so only the dispersion separates them (0.591 nats).
#[test]
fn gamma_prior_on_a_rate_marginalizes_in_both_spellings() {
    let model = "\
rate = draw(Gamma(shape = 2.0, rate = 0.3333333333333333))
k = draw(Poisson(rate = rate))
";
    for query in [
        "lp_k = logdensityof(lawof(k), 5)",
        "lp_k = logdensityof(lawof(record(k = k)), record(k = 5))",
    ] {
        let text = pir(&format!("{model}{query}"));
        let lp_k = pir_binding(&text, "lp_k");
        assert!(
            lp_k.contains("(builtin_logdensityof NegativeBinomial (%meta"),
            "the Gamma–Poisson marginal is a NegativeBinomial:\n{lp_k}"
        );
        // Identity map: alpha ← prior shape, beta ← prior rate, no arithmetic.
        assert!(
            lp_k.contains("(%field alpha 2.0)")
                && lp_k.contains("(%field beta 0.3333333333333333)"),
            "identity parameter map from the prior:\n{lp_k}"
        );
        assert!(
            !lp_k.contains("(add ") && !lp_k.contains("(pow ") && !lp_k.contains("(sqrt "),
            "the identity map performs no arithmetic:\n{lp_k}"
        );
        // The plug-in wrong answer would name the LIKELIHOOD family at a rate.
        assert!(
            !lp_k.contains("Poisson"),
            "not the plug-in Poisson at the prior mean rate:\n{lp_k}"
        );
    }
}

// One field marginalizes, an independent sibling does not. The routing hands the whole
// record back to the product lowering with just that field's measure replaced, so the
// sibling keeps its own factor and its own pin — a marginal must not cost the product.
#[test]
fn a_marginalized_field_keeps_its_independent_sibling_factor() {
    let text = pir("\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
q = draw(Normal(mu = 4.0, sigma = 3.0))
lp = logdensityof(lawof(record(y = y, q = q)), record(y = 0.5, q = 0.7))");
    let lp = pir_binding(&text, "lp");
    assert_eq!(
        lp.matches("builtin_logdensityof").count(),
        2,
        "the joint is y's marginal times q's own law:\n{lp}"
    );
    assert!(
        lp.contains("(%field sigma 1.4142135623730951)"),
        "y's factor is the marginal:\n{lp}"
    );
    assert!(
        lp.contains("(%field mu 4.0)") && lp.contains("(%field sigma 3.0)"),
        "q's factor is q's own law, unchanged:\n{lp}"
    );
}

// Refusal is the fallback, and each row's nearest non-conjugate neighbour must keep hitting
// it. A latent on a SCALE is the one that matters most: emitting Row 1's mean-conjugate
// marginal for a scale mixture is the most likely way to get this table wrong.
#[test]
fn a_latent_with_no_conjugate_row_still_refuses() {
    for (label, src) in [
        // The latent feeds `sigma`, not the conjugating `mu`.
        (
            "normal prior on a scale",
            "\
s = draw(Normal(mu = 0.0, sigma = 2.0))
y = draw(Normal(mu = 1.0, sigma = s))
lp_y = logdensityof(lawof(y), 0.5)",
        ),
        // The families do not pair.
        (
            "gamma prior on a normal mean",
            "\
z = draw(Gamma(shape = 2.0, rate = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
lp_y = logdensityof(lawof(y), 0.5)",
        ),
        // Two latents is not a single-prior integral.
        (
            "two latents",
            "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
w = draw(Gamma(shape = 2.0, rate = 1.0))
y = draw(Normal(mu = z, sigma = w))
lp_y = logdensityof(lawof(y), 0.5)",
        ),
        // A DERIVED parameter has a closed-form marginal that is not this row's, so the
        // exact-ref check must refuse rather than reuse Row 1's map.
        (
            "derived mean",
            "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = 2.0 * z, sigma = 1.0))
lp_y = logdensityof(lawof(y), 0.5)",
        ),
        // A two-level hierarchy: the row integrates ONE prior, and `z`'s prior still
        // conditions on `w`.
        (
            "two-level hierarchy",
            "\
w = draw(Normal(mu = 0.0, sigma = 1.0))
z = draw(Normal(mu = w, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
lp_y = logdensityof(lawof(y), 0.5)",
        ),
        // A TRANSFORM of the value needs the pushforward of the marginal.
        (
            "transformed value",
            "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
b = exp(y)
lp_b = logdensityof(lawof(b), 1.6487212707001282)",
        ),
        // Same, through the record spelling's own guard.
        (
            "transformed field",
            "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = z, sigma = 1.0))
b = exp(y)
lp_b = logdensityof(lawof(record(b = b)), record(b = 1.6487212707001282))",
        ),
    ] {
        let reason = refusal(src);
        assert!(
            reason.contains("no conjugate closed-form applies"),
            "{label} must refuse as an unanswerable marginal: {reason}"
        );
    }
}
