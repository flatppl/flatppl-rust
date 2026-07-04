use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn determinize_src(src: &str) -> flatppl_core::Module {
    determinize(&parse_infer(src)).expect("must lower, not refuse")
}

// kchain(M, K) with a CONTINUOUS latent that forms a Normal–Normal conjugate
// pair marginalizes IN CLOSED FORM (no enumeration, no quadrature):
//
//   ∫ N(y; μ, σ)·N(μ; μ0, σ0) dμ = N(y; μ0, sqrt(σ0² + σ²))
//
// The prior `z ~ Normal(mu = 0.0, sigma = 2.0)` feeds the likelihood mean
// `y ~ Normal(mu = z, sigma = 1.0)`, so the marginal over `y` is
// `Normal(mu = 0.0, sigma = sqrt(2.0² + 1.0²))`. The determiniser must emit
// exactly ONE `builtin_logdensityof(Normal, {mu, sigma}, obs)` scoring that
// marginal at the scalar observation `0.5` — and no `kchain` / `lawof` /
// `draw` / `kernelof` may survive.
//
// Emitted FlatPIR (density term):
//   (builtin_logdensityof Normal
//     (record (%field mu 0.0)
//             (%field sigma (sqrt (add (pow 2.0 2.0) (pow 1.0 2.0)))))
//     0.5)
#[test]
fn normal_normal_conjugate_marginal() {
    let src = "\
z = draw(Normal(mu = 0.0, sigma = 2.0))
k = kernelof(record(y = draw(Normal(mu = z, sigma = 1.0))), z = z)
pp = kchain(lawof(record(z = z)), k)
lp = logdensityof(pp, record(y = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);

    // Exactly ONE density term: the closed-form marginal Normal scored at obs.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "closed-form marginal is a single density term:\n{pir}"
    );
    // The marginal is a Normal.
    assert!(
        pir.contains("(builtin_logdensityof Normal "),
        "marginal is a Normal:\n{pir}"
    );
    // Marginal mean = the prior's mu (0.0), passed through unchanged.
    assert!(
        pir.contains("(%field mu 0.0)"),
        "marginal mean is the prior mu (0.0):\n{pir}"
    );
    // Marginal sigma = sqrt(add(pow(σ0, 2), pow(σ, 2))) with σ0 = 2.0 (prior),
    // σ = 1.0 (likelihood). The exact `pow` bases pin the variance sum σ0² + σ².
    assert!(pir.contains("(sqrt "), "marginal sigma uses sqrt:\n{pir}");
    assert!(
        pir.contains("(add "),
        "marginal variance sums the two variances:\n{pir}"
    );
    assert_eq!(
        pir.matches("(pow ").count(),
        2,
        "each stddev is squared via pow:\n{pir}"
    );
    assert!(
        pir.contains("(pow 2.0 2.0)"),
        "prior variance is σ0² = 2.0²:\n{pir}"
    );
    assert!(
        pir.contains("(pow 1.0 2.0)"),
        "likelihood variance is σ² = 1.0²:\n{pir}"
    );
    // Scored at the SCALAR observation 0.5 (the record's `y` field), not the
    // record itself — the `y` field was consumed by the descent.
    assert!(
        pir.contains(" 0.5))") && !pir.contains("(%field y"),
        "marginal is scored at the scalar obs 0.5, record variate consumed:\n{pir}"
    );
    // The measure layer is fully gone.
    assert!(
        !pir.contains("kchain")
            && !pir.contains("lawof")
            && !pir.contains("(draw ")
            && !pir.contains("kernelof"),
        "measure layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

// Refuse-don't-mislower (detection contract (b)): the SAME Normal prior +
// Normal likelihood, but the latent feeds `sigma` — not the conjugating mean
// parameter `mu`. A Normal prior on a standard deviation is NOT the Normal–Normal
// (mean) conjugacy, so no row's conjugating-parameter check passes and the
// determiniser must REFUSE, never emit the mean-conjugate marginal.
#[test]
fn conjugate_marginal_refuses_when_latent_feeds_sigma() {
    let src = "\
z = draw(Normal(mu = 0.0, sigma = 2.0))
k = kernelof(record(y = draw(Normal(mu = 1.0, sigma = z))), z = z)
pp = kchain(lawof(record(z = z)), k)
lp = logdensityof(pp, record(y = 0.5))";
    let m = parse_infer(src);
    let err = determinize(&m)
        .expect_err("latent feeding sigma is not the Normal–Normal mean conjugacy — refuse");
    assert!(
        err.construct.contains("kchain"),
        "refusal names kchain: {err:?}"
    );
    assert!(
        err.reason.contains("non-enumerable"),
        "refusal explains the non-enumerable marginal: {err:?}"
    );
}

// Refuse-don't-mislower (detection contract (a)): a continuous latent whose
// prior family has NO conjugate row for the likelihood. A Gamma prior feeding a
// Normal likelihood mean is not a table entry, so no row matches and the
// determiniser must REFUSE rather than fabricate a marginal.
#[test]
fn conjugate_marginal_refuses_non_conjugate_family() {
    let src = "\
z = draw(Gamma(shape = 2.0, rate = 1.0))
k = kernelof(record(y = draw(Normal(mu = z, sigma = 1.0))), z = z)
pp = kchain(lawof(record(z = z)), k)
lp = logdensityof(pp, record(y = 0.5))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("a Gamma–Normal pair has no conjugate row — refuse");
    assert!(
        err.construct.contains("kchain"),
        "refusal names kchain: {err:?}"
    );
    assert!(
        err.reason.contains("non-enumerable"),
        "refusal explains the non-enumerable marginal: {err:?}"
    );
}
