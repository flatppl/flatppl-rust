//! `lawof(y)` over a `y ~ Dist(param = z, …)` with `z` latent is the conjugate marginal,
//! not a refusal.
//!
//! §04 *Reification to measures* makes `lawof(x)` the total law of `x`, and §04 *Kernels
//! and `kernelof`* says which ancestors go: internal stochastic nodes of the traced
//! sub-DAG "are not boundary inputs, so `lawof` integrates them out", equivalently
//! `kchain(prior, forward_kernel)`. The guard against the CONDITIONAL escaping refused
//! this whole shape, including pairs `CONJUGATE_TABLE` answers in closed form; it now
//! tries the table first.
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

// Row 3 (Beta prior on a Binomial `p`), both spellings. §08 names no `BetaBinomial`
// constructor, so the row emits the log-pmf from §07 builtins — `loggamma` is a §07
// "Elementary functions" entry. A row does not have to name a distribution: determinised
// output is a deterministic expression.
//
// Prior `Beta(alpha = 2, beta = 3)`, likelihood `Binomial(n = 10, p = p)`, marginal
// `BetaBinomial(10, 2, 3)`. logdensity at k = 7 is -2.526728144641337. The wrong answer this
// pins out is the plug-in `Binomial(10, 0.4)` at k = 7 = -3.1590202516350088 — the
// conditional at the prior MEAN `α/(α+β) = 0.4` (0.632 nats).
#[test]
fn beta_prior_on_a_binomial_p_marginalizes_in_both_spellings() {
    let model = "\
p = draw(Beta(alpha = 2.0, beta = 3.0))
k = draw(Binomial(n = 10, p = p))
";
    for query in [
        "lp_k = logdensityof(lawof(k), 7)",
        "lp_k = logdensityof(lawof(record(k = k)), record(k = 7))",
    ] {
        let lp_k = pir_binding(&pir(&format!("{model}{query}")), "lp_k");
        // No constructor to score: the row's answer IS the expression.
        assert!(
            !lp_k.contains("builtin_logdensityof"),
            "the beta-binomial marginal is an expression, not a scored constructor:\n{lp_k}"
        );
        // log C(10, 7) = loggamma(11) − loggamma(8) − loggamma(4). Every argument
        // const-folds, so the three literals pin the coefficient by themselves.
        assert!(
            lp_k.contains("(loggamma 11.0)")
                && lp_k.contains("(loggamma 8.0)")
                && lp_k.contains("(loggamma 4.0)"),
            "log C(n, k) is loggamma(n+1) − loggamma(k+1) − loggamma(n−k+1):\n{lp_k}"
        );
        // log B(k+α, n−k+β) = log B(9, 6) and log B(α, β) = log B(2, 3). The POSTERIOR
        // shapes are what a plug-in has no analogue of, so 9.0/6.0/15.0 are the row.
        assert!(
            lp_k.contains("(loggamma 9.0)")
                && lp_k.contains("(loggamma 6.0)")
                && lp_k.contains("(loggamma 15.0)")
                && lp_k.contains("(loggamma 5.0)"),
            "log B(k+α, n−k+β) − log B(α, β) at α = 2, β = 3, n = 10, k = 7:\n{lp_k}"
        );
        // The plug-in wrong answer would score the LIKELIHOOD family at the prior mean.
        assert!(
            !lp_k.contains("Binomial") && !lp_k.contains("0.4"),
            "not the plug-in Binomial(10, 0.4) at the prior mean p:\n{lp_k}"
        );
    }
}

// Row 4 (Exponential prior on the VARIANCE → Laplace). The latent reaches the conjugating
// parameter through a `sqrt`, because §08's `Normal` takes `sigma` and the prior is on the
// variance. §08 parameterizes `Exponential` by RATE, so a prior of mean 2b² is
// `rate = 1/(2b²)` and the map inverts that: `scale = 1/sqrt(2λ)`.
//
// Prior `Exponential(rate = 0.5)` (mean 2, so b = 1) → marginal `Laplace(0, 1)`. logdensity
// at y = 4.0 is -4.693147180559945. The wrong answer this pins out is the plug-in
// `Normal(0, sqrt 2)` at the prior mean variance = -5.265512123484645 (0.572 nats).
#[test]
fn exponential_prior_on_a_variance_marginalizes_to_a_laplace() {
    let model = "\
v = draw(Exponential(rate = 0.5))
y = draw(Normal(mu = 0.0, sigma = sqrt(v)))
";
    for query in [
        "lp_y = logdensityof(lawof(y), 4.0)",
        "lp_y = logdensityof(lawof(record(y = y)), record(y = 4.0))",
    ] {
        let lp_y = pir_binding(&pir(&format!("{model}{query}")), "lp_y");
        assert!(
            lp_y.contains("(builtin_logdensityof Laplace "),
            "the Exponential-variance marginal is a Laplace:\n{lp_y}"
        );
        assert!(
            lp_y.contains("(%field location 0.0)") && lp_y.contains("(%field scale 1.0)"),
            "location is the likelihood's mu, scale is 1/sqrt(2λ) = 1:\n{lp_y}"
        );
        // The plug-in wrong answer would keep the likelihood's family.
        assert!(
            !lp_y.contains("Normal"),
            "not the plug-in Normal at the prior mean variance:\n{lp_y}"
        );
    }

    // `scale = 1` cannot tell `1/sqrt(2λ)` from a `λ` passed through, so a second shape
    // with a DISTINCT answer pins the arithmetic. `rate = 0.125` is mean 8 = 2b² with
    // b = 2, and 1/sqrt(2 · 0.125) = 2. Structural only — no density number is claimed.
    let lp_y = pir_binding(
        &pir("\
v = draw(Exponential(rate = 0.125))
y = draw(Normal(mu = 0.0, sigma = sqrt(v)))
lp_y = logdensityof(lawof(y), 4.0)"),
        "lp_y",
    );
    assert!(
        lp_y.contains("(%field scale 2.0)"),
        "scale is 1/sqrt(2λ) = 2 at λ = 0.125, not λ itself:\n{lp_y}"
    );

    // The likelihood's `mu` becomes the marginal's LOCATION: `y = μ + s·ε` for a symmetric
    // mixture, so the marginal is the same law shifted. At `mu = 1.5`, `rate = 0.5`,
    // logdensity at y = 4.0 is -3.1931471805599454 (`src/marginal.md`).
    let lp_y = pir_binding(
        &pir("\
v = draw(Exponential(rate = 0.5))
y = draw(Normal(mu = 1.5, sigma = sqrt(v)))
lp_y = logdensityof(lawof(y), 4.0)"),
        "lp_y",
    );
    assert!(
        lp_y.contains("(%field location 1.5)") && lp_y.contains("(%field scale 1.0)"),
        "the likelihood's mu is the marginal's location, the scale unchanged by it:\n{lp_y}"
    );
}

// Row 5 (InverseGamma prior on the VARIANCE → scaled Student t), both spellings. §08's
// `StudentT(nu)` is the standard form only — "The location-scale form is obtained via
// `pushfwd(fn(mu + sigma * _), StudentT(nu))`" — and a `pushfwd` is not a bare constructor,
// so this row emits the log-density.
//
// Prior `InverseGamma(shape = 2.5, scale = 3.0)` → location 0, scale sqrt(β/α) =
// 1.0954451150103321, ν = 2α = 5. logdensity at y = 5.0 is -5.986463573222975. The wrong
// answer this pins out is the plug-in `Normal(0, sqrt(β/(α−1)))` at the prior mean variance
// = -7.515512123484645 (1.529 nats).
#[test]
fn inverse_gamma_prior_on_a_variance_marginalizes_to_a_scaled_student_t() {
    let model = "\
v = draw(InverseGamma(shape = 2.5, scale = 3.0))
y = draw(Normal(mu = 0.0, sigma = sqrt(v)))
";
    for query in [
        "lp_y = logdensityof(lawof(y), 5.0)",
        "lp_y = logdensityof(lawof(record(y = y)), record(y = 5.0))",
    ] {
        let lp_y = pir_binding(&pir(&format!("{model}{query}")), "lp_y");
        assert!(
            !lp_y.contains("builtin_logdensityof"),
            "the scaled Student t marginal is an expression, not a scored constructor:\n{lp_y}"
        );
        // scale = sqrt(β/α) = sqrt(1.2), const-folded. Reading β as a MULTIPLICATIVE scale
        // rather than as Gamma's rate would fold to a different constant.
        assert!(
            lp_y.contains("(log 1.0954451150103321)"),
            "the log-normalizer carries log(scale) with scale = sqrt(β/α) = sqrt(1.2):\n{lp_y}"
        );
        // log sqrt(ν) with ν = 2α = 5: sqrt folds to 2.23606797749979.
        assert!(
            lp_y.contains("(log 2.23606797749979)"),
            "the log-normalizer carries log(sqrt ν) with ν = 2α = 5:\n{lp_y}"
        );
        // log B(ν/2, 1/2) = loggamma(2.5) + loggamma(0.5) − loggamma(3.0), which absorbs
        // the Γ(1/2) = sqrt(π) so no `pi` constant appears.
        assert!(
            lp_y.contains("(loggamma 2.5)")
                && lp_y.contains("(loggamma 0.5)")
                && lp_y.contains("(loggamma 3.0)"),
            "the log-normalizer carries log B(ν/2, 1/2):\n{lp_y}"
        );
        // ((ν+1)/2) · log1p(z²/ν) with z = 5/sqrt(1.2): both factors const-fold.
        assert!(
            lp_y.contains("(mul 3.0 ") && lp_y.contains("(log1p 4.166666666666666)"),
            "the tail is ((ν+1)/2) · log1p(z²/ν) = 3 · log1p(25/1.2/5):\n{lp_y}"
        );
        // The plug-in wrong answer would keep the likelihood's family.
        assert!(
            !lp_y.contains("Normal"),
            "not the plug-in Normal at the prior mean variance:\n{lp_y}"
        );
    }

    // `mu` enters through `z = (y − μ)/s`, so a nonzero one changes the TAIL, not the
    // normalizer. At `mu = 1.5` the tail argument becomes (5 − 1.5)²/1.2/5 =
    // 2.041666666666667 and logdensity at y = 5.0 is -4.396997199853038
    // (`src/marginal.md`). A row that dropped `mu` would keep 4.166666666666666.
    let lp_y = pir_binding(
        &pir("\
v = draw(InverseGamma(shape = 2.5, scale = 3.0))
y = draw(Normal(mu = 1.5, sigma = sqrt(v)))
lp_y = logdensityof(lawof(y), 5.0)"),
        "lp_y",
    );
    assert!(
        lp_y.contains("(log1p 2.041666666666667)") && !lp_y.contains("(log1p 4.166666666666666)"),
        "mu enters through z = (y − μ)/s, so the tail argument shifts with it:\n{lp_y}"
    );
    // The normalizer is independent of `mu`: same scale, same ν, same log-beta.
    assert!(
        lp_y.contains("(log 1.0954451150103321)")
            && lp_y.contains("(log 2.23606797749979)")
            && lp_y.contains("(loggamma 2.5)"),
        "the log-normalizer does not depend on mu:\n{lp_y}"
    );
}

// The `Sqrt` path's ref resolution is unreachable, and this pins WHY — so a later reader
// does not hunt for the coverage, and a change that makes the shape lower is forced to
// re-examine the row. A named intermediate (`s = sqrt(v)`; `sigma = s`) refuses EARLIER
// than the table: `s` keeps referencing `v`, so the driver never sweeps `v = draw(…)` and
// the residual `draw` reaches exit. The refusal is the driver's, not the row's — the
// construct is `draw`, not `kchain`.
#[test]
fn a_named_sqrt_intermediate_refuses_upstream_not_in_the_row() {
    for src in [
        "\
v = draw(Exponential(rate = 0.5))
s = sqrt(v)
y = draw(Normal(mu = 0.0, sigma = s))
lp = logdensityof(lawof(y), 4.0)",
        "\
v = draw(Exponential(rate = 0.5))
s = sqrt(v)
kk = kernelof(record(y = draw(Normal(mu = 0.0, sigma = s))), v = v)
pp = kchain(lawof(record(v = v)), kk)
lp = logdensityof(pp, record(y = 4.0))",
    ] {
        let err = determinize(&parse_infer(src))
            .expect_err("a named sqrt intermediate leaves the latent's draw live — refuse");
        assert_eq!(
            err.construct, "draw",
            "the refusal is the driver's residual-draw scan, not the conjugate row's: {err:?}"
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

    // Two fields over DIFFERENT latents is a genuine product and must keep lowering, so the
    // shared-latent refusal below cannot be a blanket ban on marginalizing two fields.
    let two = pir_binding(
        &pir("\
z1 = draw(Normal(mu = 0.0, sigma = 1.0))
z2 = draw(Normal(mu = 0.0, sigma = 3.0))
y1 = draw(Normal(mu = z1, sigma = 1.0))
y2 = draw(Normal(mu = z2, sigma = 1.0))
lp = logdensityof(lawof(record(y1 = y1, y2 = y2)), record(y1 = 0.5, y2 = 0.7))"),
        "lp",
    );
    assert!(
        two.contains("(%field sigma 1.4142135623730951)")
            && two.contains("(%field sigma 3.1622776601683795)"),
        "each field marginalizes its own latent: sqrt(1+1) and sqrt(9+1):\n{two}"
    );
}

// **Fields sharing ONE latent are CORRELATED, so their joint is not the product of their
// marginals** — and for one shape the joint itself is closed-form. For
// `yᵢ ~ Normal(mu = z, sigma = σᵢ)` over `z ~ Normal(μ₀, s₀)` each field's marginal is
// `Normal(μ₀, sqrt(s₀² + σᵢ²))` and each is right, but `Cov(yᵢ, yⱼ) = Var(z) = s₀²`, so the
// joint is `MvNormal(μ₀·1, s₀²·J + diag(σᵢ²))`. Σ is diagonal plus rank one, so
// Sherman–Morrison and the matrix determinant lemma give the log-density as §07 builtins —
// `src/marginal.md`, *The shared-latent record law*, has the derivation and every number.
//
// Every per-field answer being right is why this cannot live in a conjugate row: the row is
// asked for y1's law and returns it correctly. `conjugate_marginal_measure` reports the
// latent it integrated, the record path sees the collision, and only then is the joint law
// tried.
//
// `log`/`log1p` are not const-folded (`canon::fold` excludes transcendentals), so for an
// all-literal model the emission keeps three checkable parts: the folded `N·log 2π`, the
// residual `log`/`log1p` terms, and the quadratic form as ONE folded literal. That last
// literal is the Sherman–Morrison result and the strongest thing assertable here — pairing
// point C's σ with the wrong fields moves it from 5.851661943957181 to 4.2324472630774075,
// and point B's from 0.747121951219512 to 2.042975609756093.
//
// Every truth below is verified in `marginal.md` three ways — the closed form, `MvNormal` in
// Distributions.jl, and quadrature of the mixture integral — and nothing here computes one.
#[test]
fn a_shared_latent_record_lowers_the_correlated_joint_not_the_product() {
    // Point A. μ₀ = 0, s₀ = 1, σ = (1, 1) at (0.5, 0.7): truth -2.5171832107434002, against
    // the product of the marginals -2.716024246969291, a 0.199-nat gap.
    let model_a = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y1 = draw(Normal(mu = z, sigma = 1.0))
y2 = draw(Normal(mu = z, sigma = 1.0))
";
    for query in [
        "lp = logdensityof(lawof(record(y1 = y1, y2 = y2)), record(y1 = 0.5, y2 = 0.7))",
        // Reversed field order. The `log(vᵢ)` terms permute with it, so the emission is not
        // byte-identical; the folded literals are, and they are what carries the pairing.
        "lp = logdensityof(lawof(record(y2 = y2, y1 = y1)), record(y2 = 0.7, y1 = 0.5))",
        // The other provenance path: an earlier query PINNED `z`, so the record query finds
        // a literal where the latent was. This is the shape whose conditional escaped before
        // the pin's provenance was recorded, and it must still reach the joint.
        "lp_z = logdensityof(lawof(z), 0.3)\n\
         lp = logdensityof(lawof(record(y1 = y1, y2 = y2)), record(y1 = 0.5, y2 = 0.7))",
    ] {
        let lp = pir_binding(&pir(&format!("{model_a}{query}")), "lp");
        assert!(
            lp.contains("(mul -0.5 ") && lp.contains("3.6757541328186907"),
            "−½ over the flat sum, opening with 2·log 2π:\n{lp}"
        );
        assert_eq!(
            lp.matches("(log 1.0)").count(),
            2,
            "one log σᵢ² per field, both variances 1:\n{lp}"
        );
        assert!(
            lp.contains("(log1p 2.0)"),
            "the rank-one log-det term log(1 + s₀²Σdᵢ) at k = 2:\n{lp}"
        );
        assert!(
            lp.contains(" 0.26)"),
            "the Sherman–Morrison quadratic form 0.74 − 1.44/3 = 0.26:\n{lp}"
        );
        // The product of the marginals is what this replaces, so its signature must be gone.
        assert!(
            !lp.contains("builtin_logdensityof"),
            "the joint is one expression, not a product of scored marginals:\n{lp}"
        );
        assert!(
            !lp.contains("1.4142135623730951"),
            "no per-field Normal(0, √2) marginal survives:\n{lp}"
        );
    }

    // Point B. THREE fields with UNEQUAL σ = (0.5, 1, 2), s₀ = 1.5, at (0.9, 1.2, 2):
    // truth -4.405587203673088. Unequal σ is what makes the point discriminate at all —
    // with σ equal the fields are exchangeable and a row that permuted them would pass.
    // Wrong answers: the product of the marginals -5.424117657134536, the conditional at
    // z = μ₀ -5.596815599614018, σ reversed -5.053514032941378, and the two half-applied
    // corrections -6.872026228063332 (no Sherman–Morrison) and -3.1303765752237744 (no
    // log-det term).
    let model_b = "\
z = draw(Normal(mu = 0.0, sigma = 1.5))
y1 = draw(Normal(mu = z, sigma = 0.5))
y2 = draw(Normal(mu = z, sigma = 1.0))
y3 = draw(Normal(mu = z, sigma = 2.0))
";
    for query in [
        "lp = logdensityof(lawof(record(y1 = y1, y2 = y2, y3 = y3)), \
         record(y1 = 0.9, y2 = 1.2, y3 = 2.0))",
        "lp = logdensityof(lawof(record(y3 = y3, y2 = y2, y1 = y1)), \
         record(y3 = 2.0, y2 = 1.2, y1 = 0.9))",
    ] {
        let lp = pir_binding(&pir(&format!("{model_b}{query}")), "lp");
        assert!(
            lp.contains("5.513631199228036"),
            "3·log 2π — the field COUNT is folded in, so a miscount shows here:\n{lp}"
        );
        for arg in ["(log 0.25)", "(log 1.0)", "(log 4.0)"] {
            assert!(
                lp.contains(arg),
                "log σᵢ² for each of 0.5, 1, 2 — {arg}:\n{lp}"
            );
        }
        assert!(
            lp.contains("(log1p 11.8125)"),
            "k = 1.5²·(4 + 1 + 0.25) = 11.8125:\n{lp}"
        );
        assert!(
            lp.contains(" 0.747121951219512)"),
            "the quadratic form; a σ/field mispairing gives 2.042975609756093:\n{lp}"
        );
    }

    // Point C. NONZERO μ₀ = 1.5, s₀ = 0.8, σ = (0.7, 1.3) at (3.5, 0.5): truth
    // -5.163204327709579. Carried for the reason Rows 4 and 5 carry a nonzero-location
    // point — μ₀ enters only through rᵢ = xᵢ − μ₀, so a row that dropped it keeps every
    // other literal and passes every μ₀ = 0 point. Dropped, it gives -8.216098673951652,
    // a 3.053-nat gap. This is also a SPREAD point (one field high, one low), so its gap
    // against the product of the marginals is -0.857 — the opposite sign from point B's,
    // which pins the sign of the correlation term and not just its magnitude.
    let model_c = "\
z = draw(Normal(mu = 1.5, sigma = 0.8))
y1 = draw(Normal(mu = z, sigma = 0.7))
y2 = draw(Normal(mu = z, sigma = 1.3))
";
    for query in [
        "lp = logdensityof(lawof(record(y1 = y1, y2 = y2)), record(y1 = 3.5, y2 = 0.5))",
        "lp = logdensityof(lawof(record(y2 = y2, y1 = y1)), record(y2 = 0.5, y1 = 3.5))",
    ] {
        let lp = pir_binding(&pir(&format!("{model_c}{query}")), "lp");
        assert!(
            lp.contains("(log 0.48999999999999994)") && lp.contains("(log 1.6900000000000002)"),
            "log σᵢ² for 0.7 and 1.3:\n{lp}"
        );
        assert!(
            lp.contains("(log1p 1.6848206738316633)"),
            "k = 0.8²·(1/0.49 + 1/1.69):\n{lp}"
        );
        assert!(
            lp.contains(" 5.851661943957181)"),
            "the quadratic form, which is where μ₀ enters; mispaired σ gives \
             4.2324472630774075:\n{lp}"
        );
    }
}

// The joint law must be reachable through a REIFIED spelling of the record too. §06's
// *Uniform kernel extension* identifies a measure with a nullary kernel, so a CLOSED
// `functionof` means its body, and the `lawof` routing unwraps it to a fixpoint. That unwrap
// and this law are independent changes to the same dispatch area, so the composition is
// pinned rather than assumed.
//
// The named spelling is compared on the `lp` binding, not the whole module: its
// `M = functionof(record(…))` binding survives as a dead `Function`-typed one, which the
// measure-binding sweep does not reach. That residual is byte-identical at `482d26f` with
// this branch's changes stashed, so it is pre-existing and not this law's doing.
#[test]
fn a_reified_shared_latent_record_reaches_the_same_joint_law() {
    let model = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y1 = draw(Normal(mu = z, sigma = 1.0))
y2 = draw(Normal(mu = z, sigma = 1.0))
";
    let plain = pir_binding(
        &pir(&format!(
            "{model}lp = logdensityof(lawof(record(y1 = y1, y2 = y2)), \
             record(y1 = 0.5, y2 = 0.7))"
        )),
        "lp",
    );
    assert!(
        plain.contains(" 0.26)"),
        "the plain spelling must be the joint, or the parity below proves nothing:\n{plain}"
    );
    for query in [
        // Named closed reification.
        "M = functionof(record(y1 = y1, y2 = y2))\n\
         lp = logdensityof(lawof(M), record(y1 = 0.5, y2 = 0.7))",
        // Inline.
        "lp = logdensityof(lawof(functionof(record(y1 = y1, y2 = y2))), \
         record(y1 = 0.5, y2 = 0.7))",
        // Nested — §04's rationale applies at every level, so the unwrap is a fixpoint.
        "lp = logdensityof(lawof(functionof(functionof(record(y1 = y1, y2 = y2)))), \
         record(y1 = 0.5, y2 = 0.7))",
    ] {
        let lp = pir_binding(&pir(&format!("{model}{query}")), "lp");
        assert_eq!(
            lp, plain,
            "a reified record spelling must reach the joint law identically"
        );
    }
}

// Refusal stays the fallback for every shared-latent shape the law does not cover. The
// blocks below are the ones that REACH the law's recogniser — each field's own conjugate row
// matches, so the repeated latent is detected — and are turned away there. The rest of the
// non-matching shapes (a scale latent under a Normal prior, a non-Normal prior, two shared
// latents, a derived mean, a two-level hierarchy, a transformed field) never get that far:
// no per-field row matches them, so they refuse upstream with the per-field reason, and
// `a_shared_latent_shape_with_no_per_field_row_refuses_upstream` pins that instead.
//
// These probes do NOT isolate one recogniser check each. Verified by mutation: each is
// caught by two or three of them at once, so removing any single check reddens nothing here
// (`shared_latent_record_law` records which checks that leaves unreachable-as-sole-cause).
// The partly-shared record below is the one with a demonstrated floor — removing BOTH the
// mean-path and the latent-agreement check lets it mislower, which is what makes it a real
// guard rather than a restatement of the caller's filtering.
#[test]
fn a_shared_latent_shape_outside_the_record_law_still_refuses() {
    for (label, src) in [
        // Rows 4 and 5: a shared VARIANCE. Each field's marginal is right (`Laplace(0, 1)`,
        // and the scaled t) and the joint is a correlated SCALE mixture, not a Gaussian with
        // a rank-one Σ — the law's Normal-prior and mean-position checks both refuse it.
        (
            "shared variance, laplace row",
            "\
v = draw(Exponential(rate = 0.5))
y1 = draw(Normal(mu = 0.0, sigma = sqrt(v)))
y2 = draw(Normal(mu = 0.0, sigma = sqrt(v)))
lp = logdensityof(lawof(record(y1 = y1, y2 = y2)), record(y1 = 0.5, y2 = 4.0))",
        ),
        (
            "shared variance, scaled-t row",
            "\
v = draw(InverseGamma(shape = 2.5, scale = 3.0))
y1 = draw(Normal(mu = 0.0, sigma = sqrt(v)))
y2 = draw(Normal(mu = 0.0, sigma = sqrt(v)))
lp = logdensityof(lawof(record(y1 = y1, y2 = y2)), record(y1 = 0.5, y2 = 5.0))",
        ),
        // A shared latent under a different FAMILY. Row 2 answers each field, the fields are
        // correlated, and the joint is no Gaussian at all.
        (
            "shared rate, gamma-poisson rows",
            "\
z = draw(Gamma(shape = 2.0, rate = 1.0))
k1 = draw(Poisson(rate = z))
k2 = draw(Poisson(rate = z))
lp = logdensityof(lawof(record(k1 = k1, k2 = k2)), record(k1 = 3, k2 = 5))",
        ),
        // PARTLY shared: y1 and y2 share `z`, y3 integrates `w`. The joint is this law's
        // form times `w`'s own marginal — correct in principle, outside the decided scope,
        // and refused rather than approximated.
        (
            "two fields share a latent, a third does not",
            "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
w = draw(Normal(mu = 0.0, sigma = 1.0))
y1 = draw(Normal(mu = z, sigma = 1.0))
y2 = draw(Normal(mu = z, sigma = 1.0))
y3 = draw(Normal(mu = w, sigma = 1.0))
lp = logdensityof(lawof(record(y1 = y1, y2 = y2, y3 = y3)), \
             record(y1 = 0.5, y2 = 0.7, y3 = 0.2))",
        ),
        // A TRANSFORMED field, written AFTER the second shared field — the position the
        // per-field screen never reaches, because the repeat is detected on the second shared
        // field and the loop returns there. Without the whole-record gate this lowered and
        // scored the query's value of `b` as the untransformed draw: emitted
        // `-4.033712780173963` against the truth `-3.985439088559615` for `exp` (0.048 nats)
        // and `-4.319047460733908` for the affine map (0.285 nats), both from quadrature of
        // the mixture with the change of variables applied. The two maps emitted IDENTICALLY,
        // which is what proves the map was ignored rather than mis-applied.
        (
            "a transformed field AFTER the second shared field, exp",
            "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y1 = draw(Normal(mu = z, sigma = 1.0))
y2 = draw(Normal(mu = z, sigma = 1.0))
y3 = draw(Normal(mu = z, sigma = 1.0))
b = exp(y3)
lp = logdensityof(lawof(record(y1 = y1, y2 = y2, b = b)), \
             record(y1 = 0.5, y2 = 0.7, b = 1.5))",
        ),
        (
            "a transformed field AFTER the second shared field, affine",
            "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y1 = draw(Normal(mu = z, sigma = 1.0))
y2 = draw(Normal(mu = z, sigma = 1.0))
y3 = draw(Normal(mu = z, sigma = 1.0))
b = 2.0 * y3
lp = logdensityof(lawof(record(y1 = y1, y2 = y2, b = b)), \
             record(y1 = 0.5, y2 = 0.7, b = 1.5))",
        ),
    ] {
        let reason = refusal(src);
        assert!(
            reason.contains("marginalize over the SAME latent")
                && reason.contains("no closed form covers this shape"),
            "{label}: a correlated joint outside the law must refuse, not emit a \
             product of marginals: {reason}"
        );
    }
}

// The shared-latent shapes the record law does not reach, because no per-field conjugate row
// matches them either: they refuse at the per-field guard, one step earlier. Pinned so that
// widening a conjugate row later cannot silently route one of them into the record law —
// each would then hit the law's own checks, and this test would show the reason moved.
#[test]
fn a_shared_latent_shape_with_no_per_field_row_refuses_upstream() {
    for (label, src) in [
        (
            "shared latent on a SCALE",
            "\
z = draw(Normal(mu = 0.0, sigma = 2.0))
y1 = draw(Normal(mu = 1.0, sigma = z))
y2 = draw(Normal(mu = 1.0, sigma = z))
lp = logdensityof(lawof(record(y1 = y1, y2 = y2)), record(y1 = 0.5, y2 = 0.7))",
        ),
        (
            "non-Normal shared prior on the mean",
            "\
z = draw(Exponential(rate = 1.0))
y1 = draw(Normal(mu = z, sigma = 1.0))
y2 = draw(Normal(mu = z, sigma = 1.0))
lp = logdensityof(lawof(record(y1 = y1, y2 = y2)), record(y1 = 0.5, y2 = 0.7))",
        ),
        (
            "two shared latents in the mean",
            "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
w = draw(Normal(mu = 0.0, sigma = 1.0))
y1 = draw(Normal(mu = add(z, w), sigma = 1.0))
y2 = draw(Normal(mu = add(z, w), sigma = 1.0))
lp = logdensityof(lawof(record(y1 = y1, y2 = y2)), record(y1 = 0.5, y2 = 0.7))",
        ),
        (
            "derived shared mean",
            "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y1 = draw(Normal(mu = 2.0 * z, sigma = 1.0))
y2 = draw(Normal(mu = 2.0 * z, sigma = 1.0))
lp = logdensityof(lawof(record(y1 = y1, y2 = y2)), record(y1 = 0.5, y2 = 0.7))",
        ),
        (
            "two-level hierarchy over the shared latent",
            "\
w = draw(Normal(mu = 0.0, sigma = 1.0))
z = draw(Normal(mu = w, sigma = 1.0))
y1 = draw(Normal(mu = z, sigma = 1.0))
y2 = draw(Normal(mu = z, sigma = 1.0))
lp = logdensityof(lawof(record(y1 = y1, y2 = y2)), record(y1 = 0.5, y2 = 0.7))",
        ),
        (
            "a transformed field over the shared latent",
            "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y1 = draw(Normal(mu = z, sigma = 1.0))
y2 = draw(Normal(mu = z, sigma = 1.0))
b = exp(y1)
lp = logdensityof(lawof(record(b = b, y2 = y2)), record(b = 1.5, y2 = 0.7))",
        ),
    ] {
        let reason = refusal(src);
        assert!(
            reason.contains("parameterized by an uncarried draw")
                && reason.contains("no conjugate row covers it"),
            "{label}: expected the per-field refusal, one step before the record law: \
             {reason}"
        );
    }
}

// A `sigma` over a SIBLING field is admitted, because the law needs σᵢ latent-INDEPENDENT and
// that is not the same as constant. `σ₂ = y1` emits the sibling's own query value, pinned by
// the chain rule the record path already applies to sibling draws.
//
// Truth `-2.2381096204634274` — quadrature of `∫ φ(z) N(0.5; z, 1) N(0.7; z, 0.5) dz`.
//
// **The value is right and the Gaussian reading is not.** With `σ₂ = y1` the fields are not
// conditionally independent given `z`, so this model is not jointly Gaussian and `Σ` is not
// its covariance. What holds is what the emission needs: at a fixed query point `σ₂` is the
// constant `x₁`, so `Πᵢ N(xᵢ; z, σᵢ) = p(x₁ | z)·p(x₂ | z, y1 = x₁)` by the chain rule, and
// integrating that against the prior is the joint density there. `marginal.md`, *A σ over a
// sibling field*, says the same and warns against reading Σ back out.
#[test]
fn a_sigma_over_a_sibling_field_is_admitted_at_the_pinned_sibling() {
    let lp = pir_binding(
        &pir("\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y1 = draw(Normal(mu = z, sigma = 1.0))
y2 = draw(Normal(mu = z, sigma = y1))
lp = logdensityof(lawof(record(y1 = y1, y2 = y2)), record(y1 = 0.5, y2 = 0.7))"),
        "lp",
    );
    assert!(
        lp.contains("(log 1.0)") && lp.contains("(log 0.25)"),
        "σ₂² is the PINNED sibling 0.5 squared, not a residual ref:\n{lp}"
    );
    assert!(lp.contains("(log1p 5.0)"), "k = 1²·(1 + 1/0.25) = 5:\n{lp}");
    assert!(
        lp.contains(" 0.39500000000000024)"),
        "the quadratic form at the pinned sibling:\n{lp}"
    );
}

// ONE field over a shared-latent-shaped model is Row 1, not the record law, and must keep
// emitting Row 1's `Normal(μ₀, sqrt(s₀² + σ²))`. The law requires two fields precisely so
// this output does not change.
#[test]
fn a_single_field_record_stays_on_the_per_field_row() {
    let lp = pir_binding(
        &pir("\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y1 = draw(Normal(mu = z, sigma = 1.0))
lp = logdensityof(lawof(record(y1 = y1)), record(y1 = 0.5))"),
        "lp",
    );
    assert!(
        lp.contains("builtin_logdensityof") && lp.contains("(%field sigma 1.4142135623730951)"),
        "one field is Row 1's marginal Normal(0, √2), scored as a measure:\n{lp}"
    );
}

// The `iid`/`joint` combinators over the SAME correlated shape correctly emit the product,
// and must keep doing so: §06 defines `joint(M1, M2, …)` as the "independent product
// measure" `(M1 ⊗ M2)(A × B) = M1(A) · M2(B)`, so `joint(a = lawof(y1), b = lawof(y2))` asks
// for the product of the two marginals — a DIFFERENT measure from
// `lawof(record(y1 = y1, y2 = y2))`, which is the law of the traced sub-DAG. The
// shared-latent refusal above must therefore be sited in the record path, not in the row.
#[test]
fn joint_and_iid_over_a_shared_latent_are_products_by_definition() {
    for src in [
        "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y1 = draw(Normal(mu = z, sigma = 1.0))
y2 = draw(Normal(mu = z, sigma = 1.0))
lp = logdensityof(joint(a = lawof(y1), b = lawof(y2)), record(a = 0.5, b = 0.7))",
        "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
y1 = draw(Normal(mu = z, sigma = 1.0))
lp = logdensityof(iid(lawof(y1), 2), [0.5, 0.7])",
    ] {
        let lp = pir_binding(&pir(src), "lp");
        assert_eq!(
            lp.matches("builtin_logdensityof").count(),
            2,
            "the independent product is two factors:\n{lp}"
        );
        assert_eq!(
            lp.matches("(%field sigma 1.4142135623730951)").count(),
            2,
            "each factor is the marginal Normal(0, sqrt 2):\n{lp}"
        );
    }
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
        // Rows 4 and 5 are on the VARIANCE, and this is the neighbour that matters for
        // them: the same prior feeding the MEAN is a LOCATION mixture, whose marginal is
        // the exponentially modified Gaussian — and at y = 0.5 that differs from Row 4's
        // Laplace by only 0.017 nats, so a row matching a bare ref where it wants `sqrt`
        // would look almost right. Its density needs `erfc`, a §09 standard-module member
        // the determiniser emits no call to (`src/marginal.md`), so it must refuse.
        (
            "exponential prior on a location",
            "\
v = draw(Exponential(rate = 0.5))
y = draw(Normal(mu = v, sigma = 1.0))
lp_y = logdensityof(lawof(y), 0.5)",
        ),
        (
            "inverse gamma prior on a location",
            "\
v = draw(InverseGamma(shape = 2.5, scale = 3.0))
y = draw(Normal(mu = v, sigma = 1.0))
lp_y = logdensityof(lawof(y), 0.5)",
        ),
        // A prior on the STANDARD DEVIATION, not the variance: the latent reaches `sigma`
        // as a bare ref, so the `sqrt` path must reject it. Rows 4 and 5 hold only for a
        // variance mixture; reusing them here would score the wrong mixture entirely.
        (
            "exponential prior on a standard deviation",
            "\
v = draw(Exponential(rate = 0.5))
y = draw(Normal(mu = 0.0, sigma = v))
lp_y = logdensityof(lawof(y), 4.0)",
        ),
        (
            "inverse gamma prior on a standard deviation",
            "\
v = draw(InverseGamma(shape = 2.5, scale = 3.0))
y = draw(Normal(mu = 0.0, sigma = v))
lp_y = logdensityof(lawof(y), 5.0)",
        ),
        // Row 3's families do not pair with a Normal likelihood.
        (
            "beta prior on a normal mean",
            "\
p = draw(Beta(alpha = 2.0, beta = 3.0))
y = draw(Normal(mu = p, sigma = 1.0))
lp_y = logdensityof(lawof(y), 0.5)",
        ),
    ] {
        let reason = refusal(src);
        assert!(
            reason.contains("no conjugate row"),
            "{label} must refuse as an unanswerable marginal: {reason}"
        );
    }
}
