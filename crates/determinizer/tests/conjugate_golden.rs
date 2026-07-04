use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn determinize_src(src: &str) -> flatppl_core::Module {
    determinize(&parse_infer(src)).expect("must lower, not refuse")
}

/// Strip the transparent `(%meta (<triple>) <inner>)` type annotations the
/// FlatPIR writer interleaves, leaving the bare S-expression — so a test can
/// assert an exact nested structure without the type noise between nodes.
fn strip_meta(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut drop_close: Vec<bool> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if s[i..].starts_with("(%meta ") {
            // Drop `(%meta ` + the balanced `(<triple>)` group + one space; the
            // matching close paren is dropped when we reach it (drop_close = true).
            i += "(%meta ".len();
            let mut depth = 0i32;
            loop {
                match b[i] {
                    b'(' => depth += 1,
                    b')' => depth -= 1,
                    _ => {}
                }
                i += 1;
                if depth == 0 {
                    break;
                }
            }
            if i < b.len() && b[i] == b' ' {
                i += 1;
            }
            drop_close.push(true);
        } else {
            match b[i] {
                b'(' => {
                    out.push('(');
                    drop_close.push(false);
                }
                b')' => {
                    if !drop_close.pop().unwrap_or(false) {
                        out.push(')');
                    }
                }
                c => out.push(c as char),
            }
            i += 1;
        }
    }
    out
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

// kchain(M, K) with a CONTINUOUS latent that forms a Gamma–Poisson conjugate
// pair marginalizes IN CLOSED FORM: NegativeBinomial(alpha, beta) (§08) IS the
// Gamma(shape=α, rate=β)–Poisson(rate=λ) mixture
//
//   ∫ Poisson(N; λ)·Gamma(λ; α, β) dλ = NegativeBinomial(α, β)
//
// so the marginal is an IDENTITY parameter map — no arithmetic. The prior
// `z ~ Gamma(shape = 2.0, rate = 3.0)` feeds the likelihood rate
// `y ~ Poisson(rate = z)`, so the marginal over `y` is
// `NegativeBinomial(alpha = 2.0, beta = 3.0)`. The determiniser must emit
// exactly ONE `builtin_logdensityof(NegativeBinomial, {alpha, beta}, obs)`
// scoring that marginal at the scalar observation `5` — and no `kchain` /
// `lawof` / `draw` / `kernelof` may survive.
//
// Emitted FlatPIR (density term):
//   (builtin_logdensityof NegativeBinomial
//     (record (%field alpha 2.0)
//             (%field beta 3.0))
//     5)
#[test]
fn gamma_poisson_conjugate_marginal() {
    let src = "\
z = draw(Gamma(shape = 2.0, rate = 3.0))
k = kernelof(record(y = draw(Poisson(rate = z))), z = z)
pp = kchain(lawof(record(z = z)), k)
lp = logdensityof(pp, record(y = 5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);

    // Exactly ONE density term: the closed-form marginal NegativeBinomial scored at obs.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "closed-form marginal is a single density term:\n{pir}"
    );
    // The marginal is a NegativeBinomial.
    assert!(
        pir.contains("(builtin_logdensityof NegativeBinomial "),
        "marginal is a NegativeBinomial:\n{pir}"
    );
    // Identity parameter map: alpha = prior shape (2.0), beta = prior rate (3.0).
    assert!(
        pir.contains("(%field alpha 2.0)"),
        "marginal alpha is the prior shape (2.0):\n{pir}"
    );
    assert!(
        pir.contains("(%field beta 3.0)"),
        "marginal beta is the prior rate (3.0):\n{pir}"
    );
    // No arithmetic: the identity map reuses the prior's value nodes unchanged.
    assert!(
        !pir.contains("add") && !pir.contains("pow") && !pir.contains("sqrt"),
        "identity map performs no arithmetic:\n{pir}"
    );
    // The exact marginal kwarg record: alpha then beta, in that order.
    assert!(
        pir.contains("(record (%field alpha 2.0) (%field beta 3.0))"),
        "exact marginal kwarg shape:\n{pir}"
    );
    // Scored at the SCALAR observation 5 (the record's `y` field), not the
    // record itself — the `y` field was consumed by the descent.
    assert!(
        pir.contains(" 5))") && !pir.contains("(%field y"),
        "marginal is scored at the scalar obs 5, record variate consumed:\n{pir}"
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

// `normalize(logweighted(x -> logdensityof(g2, x), g1))` with `g1`, `g2` both
// Normal is a POINTWISE PRODUCT OF TWO GAUSSIANS, which normalizes IN CLOSED
// FORM (no quadrature): the overlap integral is itself a Gaussian
//
//   Z = ∫ N(x; μ1, σ1)·N(x; μ2, σ2) dx = N(μ1; μ2, sqrt(σ1² + σ2²)),
//
// so `logdensityof(prod, x)` = logdensityof(g1, x) + logdensityof(g2, x) − logZ,
// with logZ = logdensityof(Normal(mu = μ2, sigma = sqrt(σ1² + σ2²)), μ1). Here
// g1 = Normal(0.0, 1.0), g2 = Normal(1.0, 2.0), so the emitted density (scored
// at the scalar variate 0.5) is three Normal density terms:
//
//   (sub (add (builtin_logdensityof Normal {mu 0.0, sigma 1.0} 0.5)
//             (builtin_logdensityof Normal {mu 1.0, sigma 2.0} 0.5))
//        (builtin_logdensityof Normal
//           {mu 1.0, sigma (sqrt (add (pow 1.0 2.0) (pow 2.0 2.0)))} 0.0))
//
// and no `normalize` / `logweighted` / `functionof` / `draw` / measure may
// survive.
#[test]
fn product_of_gaussians_normalize() {
    let src = "\
g1 = Normal(mu = 0.0, sigma = 1.0)
g2 = Normal(mu = 1.0, sigma = 2.0)
prod = normalize(logweighted(x -> logdensityof(g2, x), g1))
lp = logdensityof(prod, 0.5)";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    let bare = strip_meta(&pir);

    // Three density terms: g1 at the variate, g2 at the variate, and the overlap
    // Z (a Normal scored at μ1). Each is a Normal `builtin_logdensityof`.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        3,
        "product-of-Gaussians density is g1 + g2 − logZ (three Normal terms):\n{pir}"
    );

    // The two data terms score g1 / g2 at the SCALAR variate 0.5 (their exact
    // Normal kwargs, pinned at 0.5) — the closed-form pointwise product.
    assert!(
        bare.contains(
            "(builtin_logdensityof Normal (record (%field mu 0.0) (%field sigma 1.0)) 0.5)"
        ),
        "g1 (Normal 0.0, 1.0) scored at the scalar variate 0.5:\n{bare}"
    );
    assert!(
        bare.contains(
            "(builtin_logdensityof Normal (record (%field mu 1.0) (%field sigma 2.0)) 0.5)"
        ),
        "g2 (Normal 1.0, 2.0) scored at the scalar variate 0.5:\n{bare}"
    );

    // logZ = the Gaussian overlap: Normal(mu = μ2 = 1.0, sigma = sqrt(σ1² + σ2²) =
    // sqrt(add(pow(1.0, 2), pow(2.0, 2)))) scored at μ1 = 0.0. The exact `pow`
    // bases pin the variance SUM (not difference), and the scoring point μ1 = 0.0.
    assert!(
        bare.contains(
            "(builtin_logdensityof Normal (record (%field mu 1.0) (%field sigma (sqrt (add (pow 1.0 2.0) (pow 2.0 2.0))))) 0.0)"
        ),
        "logZ is Normal(mu = μ2, sigma = sqrt(σ1² + σ2²)) scored at μ1 = 0.0:\n{bare}"
    );

    // Overall shape: sub(add(t1, t2), logZ) — the two data terms summed, minus the
    // overlap logZ.
    assert!(
        bare.contains("(sub (add (builtin_logdensityof Normal (record (%field mu 0.0)"),
        "closed form is sub(add(g1, g2), logZ):\n{bare}"
    );

    // The measure layer is fully gone.
    assert!(
        !pir.contains("normalize")
            && !pir.contains("logweighted")
            && !pir.contains("functionof")
            && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl failed:\n{pir}"
    );
}

// Refuse-don't-mislower: a `normalize(logweighted(…))` whose base is NOT a
// Gaussian product must keep refusing (the closed-form Gaussian-overlap Z is not
// valid there), never mislower to the product formula. Two shapes exercise the
// two recognizer gates:
//   (a) the base is not a Normal (`Exponential`), and
//   (b) ℓ is a reified function but not `logdensityof(Normal, x)`
//       (here `logdensityof(Exponential, x)`).
#[test]
fn product_normalize_refuses_non_gaussian_base() {
    // (a) base is Exponential, ℓ scores a Normal — not a product of two Gaussians.
    let src_a = "\
g1 = Exponential(rate = 1.0)
g2 = Normal(mu = 1.0, sigma = 2.0)
prod = normalize(logweighted(x -> logdensityof(g2, x), g1))
lp = logdensityof(prod, 0.5)";
    let m_a = parse_infer(src_a);
    let err_a =
        determinize(&m_a).expect_err("logweighted base is not a product of two Gaussians — refuse");
    assert!(
        err_a.construct.contains("normalize"),
        "refusal names normalize: {err_a:?}"
    );

    // (b) base is Normal but ℓ scores an Exponential — not two Gaussians.
    let src_b = "\
g1 = Normal(mu = 0.0, sigma = 1.0)
g2 = Exponential(rate = 1.0)
prod = normalize(logweighted(x -> logdensityof(g2, x), g1))
lp = logdensityof(prod, 0.5)";
    let m_b = parse_infer(src_b);
    let err_b =
        determinize(&m_b).expect_err("logweighted weight is not a Gaussian logdensityof — refuse");
    assert!(
        err_b.construct.contains("normalize"),
        "refusal names normalize: {err_b:?}"
    );
}

// g2's `mu`/`sigma` must not reference the lambda argument `x` itself: scoring
// `Normal(mu = x, sigma = 1.0)` at `x` is `N(x; x, 1)`, a constant — not a second
// Gaussian *factor* of `x` — so this is not a Gaussian-product overlap at all and
// must refuse rather than emit a dangling `%local` ref to the vanished binder.
#[test]
fn product_normalize_refuses_g2_param_referencing_lambda_arg() {
    // (a) mu2 = x.
    let src_mu = "\
g1 = Normal(mu = 0.0, sigma = 1.0)
prod = normalize(logweighted(x -> logdensityof(Normal(mu = x, sigma = 1.0), x), g1))
lp = logdensityof(prod, 0.5)";
    let m_mu = parse_infer(src_mu);
    let err_mu = determinize(&m_mu).expect_err("g2's mu references the lambda argument — refuse");
    assert!(
        err_mu.construct.contains("normalize"),
        "refusal names normalize: {err_mu:?}"
    );

    // (b) sigma2 = x.
    let src_sigma = "\
g1 = Normal(mu = 0.0, sigma = 1.0)
prod = normalize(logweighted(x -> logdensityof(Normal(mu = 0.0, sigma = x), x), g1))
lp = logdensityof(prod, 0.5)";
    let m_sigma = parse_infer(src_sigma);
    let err_sigma =
        determinize(&m_sigma).expect_err("g2's sigma references the lambda argument — refuse");
    assert!(
        err_sigma.construct.contains("normalize"),
        "refusal names normalize: {err_sigma:?}"
    );
}
