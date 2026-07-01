// Structural conformance gate for the determiniser.
//
// This file holds STRUCTURAL determinizer tests only: they check that a
// handful of rosetta models (single Gaussian, product of Gaussians, iid,
// joint, likelihoodof) determinize into a FlatPDL-conformant module with the
// expected number of `builtin_logdensityof` calls and no residual
// measure-layer ops (`lawof`, `draw`, `iid`, `joint`, `likelihoodof`,
// `logdensityof`). Each test also sanity-checks a closed-form oracle value in
// pure Rust — no external engine is involved.
//
// Numeric value verification (scoring the emitted FlatPDL surface syntax
// through the flatppl-js engine and comparing to a frozen oracle) lives in
// `flatppl-testsuite`, not here: `flatppl-rust` is not a density engine.

use std::f64::consts::PI;

use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

/// Closed-form Gaussian log-density: log N(x; mu, sigma).
fn gaussian_logpdf(x: f64, mu: f64, sigma: f64) -> f64 {
    -0.5 * (2.0 * PI).ln() - sigma.ln() - 0.5 * ((x - mu) / sigma).powi(2)
}

// ── Structural oracle checks (always run) ────────────────────────────────────
//
// These verify that:
// 1. The determinizer produces a FlatPDL-conformant module.
// 2. The emitted surface syntax encodes the correct `builtin_logdensityof`
//    call(s), with no residual measure layer.
// The closed-form oracle values below are pure-Rust arithmetic sanity checks;
// they are not compared against any engine here.

#[test]
fn single_gaussian_oracle_agrees_with_flatpdl_structure() {
    // Model: a ~ Normal(0, 1); score a=0.5.
    // Oracle: log N(0.5; 0.0, 1.0) = -1.0439385332046727…
    let oracle = gaussian_logpdf(0.5, 0.0, 1.0);
    assert!(
        (oracle - (-1.043_938_533_204_672_7_f64)).abs() < 1e-12,
        "closed-form oracle sanity: {oracle}"
    );

    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("single-gaussian must lower");

    // FlatPDL conformance.
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "emitted FlatPDL must be conformant"
    );

    // Surface syntax encodes exactly one builtin_logdensityof call.
    let src_out = flatppl_syntax::print(&out);
    assert!(
        src_out.contains("builtin_logdensityof"),
        "emitted FlatPDL contains builtin_logdensityof:\n{src_out}"
    );
    // Use the FlatPIR form to check for residual measure-layer ops: FlatPIR
    // spells the measure-layer op as `(logdensityof `, while the FlatPDL
    // primitive is `(builtin_logdensityof ` — they don't overlap.
    let pir_out = flatppl_flatpir::write(&out);
    assert!(
        !pir_out.contains("(logdensityof ")
            && !pir_out.contains("lawof")
            && !pir_out.contains("(draw "),
        "measure layer eliminated:\n{pir_out}"
    );
    // The determinized module binds `lp` to a deterministic real — `a` is
    // pinned to the scored value (0.5) and no stochastic nodes remain.
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "single density term:\n{pir}"
    );
}

#[test]
fn product_gaussians_oracle_agrees_with_flatpdl_structure() {
    // Model: a ~ N(0,1), b ~ N(1,2); score a=0.5, b=0.5.
    // Oracle: log N(0.5;0,1) + log N(0.5;1,2)
    let oracle = gaussian_logpdf(0.5, 0.0, 1.0) + gaussian_logpdf(0.5, 1.0, 2.0);
    let expected = -1.043_938_533_204_672_7_f64 + (-1.643_335_713_764_618_f64);
    assert!(
        (oracle - expected).abs() < 1e-12,
        "closed-form oracle sanity: {oracle}"
    );

    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
b = draw(Normal(mu = 1.0, sigma = 2.0))
lp = logdensityof(lawof(record(a = a, b = b)), record(a = 0.5, b = 0.5))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("product must lower");

    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "emitted FlatPDL must be conformant"
    );

    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "two density terms:\n{pir}"
    );
    assert!(
        !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer eliminated:\n{pir}"
    );
}

#[test]
fn iid_normal_sum_oracle() {
    // logdensityof(iid(Normal(0,1), 3), [0.5, -0.3, 1.2]) = Σ log N(xᵢ;0,1)
    let xs = [0.5_f64, -0.3, 1.2];
    let oracle: f64 = xs.iter().map(|&x| gaussian_logpdf(x, 0.0, 1.0)).sum();
    let src = "\
d = iid(Normal(mu = 0.0, sigma = 1.0), 3)
lp = logdensityof(d, [0.5, -0.3, 1.2])";
    let m = parse_infer(src);
    let out = determinize(&m).expect("iid must lower");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL"
    );
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        3,
        "3 iid terms:\n{pir}"
    );
    assert!(
        !pir.contains("(iid ") && !pir.contains("(logdensityof "),
        "no measure layer:\n{pir}"
    );
    // Closed-form oracle sanity (pure Rust arithmetic, no engine).
    assert!(
        (oracle
            - (gaussian_logpdf(0.5, 0.0, 1.0)
                + gaussian_logpdf(-0.3, 0.0, 1.0)
                + gaussian_logpdf(1.2, 0.0, 1.0)))
        .abs()
            < 1e-12
    );
}

#[test]
fn joint_two_gaussians_oracle() {
    // logdensityof(joint(Normal(0,1), Normal(1,2)), [0.5, 0.5]) = logN(0.5;0,1)+logN(0.5;1,2)
    let oracle = gaussian_logpdf(0.5, 0.0, 1.0) + gaussian_logpdf(0.5, 1.0, 2.0);
    let src = "\
d = joint(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 1.0, sigma = 2.0))
lp = logdensityof(d, [0.5, 0.5])";
    let m = parse_infer(src);
    let out = determinize(&m).expect("joint must lower");
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "2 joint terms:\n{pir}"
    );
    assert!(!pir.contains("(joint "), "no joint:\n{pir}");
    assert!(oracle.is_finite());
}

#[test]
fn weighted_function_weight_oracle() {
    // logdensityof(weighted(x -> exp(x), g), 0.5) = log(exp(0.5)) + logdensityof(g, 0.5)
    //   = 0.5 + logN(0.5;0,1)   (g = N(0,1))
    // §06:469 — the weight may be a function of the variate; it is applied at v.
    let oracle = 0.5 + gaussian_logpdf(0.5, 0.0, 1.0);
    let src = "\
g = Normal(mu = 0.0, sigma = 1.0)
d = weighted(x -> exp(x), g)
lp = logdensityof(d, 0.5)";
    let m = parse_infer(src);
    let out = determinize(&m).expect("function-weighted weighted must lower");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL"
    );
    let pir = flatppl_flatpir::write(&out);
    // One density term (g); the applied weight is `log((%call w v))`, not a
    // `builtin_logdensityof`.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "single density term:\n{pir}"
    );
    assert!(
        !pir.contains("(weighted ") && !pir.contains("(logdensityof "),
        "measure layer gone:\n{pir}"
    );
    assert!(oracle.is_finite());
}

#[test]
fn logweighted_function_weight_oracle() {
    // logdensityof(logweighted(x -> logdensityof(g2, x), g1), 0.5)
    //   = logdensityof(g2, 0.5) + logdensityof(g1, 0.5)
    //   = logN(0.5;1,2) + logN(0.5;0,1)   (g1=N(0,1), g2=N(1,2))
    let oracle = gaussian_logpdf(0.5, 1.0, 2.0) + gaussian_logpdf(0.5, 0.0, 1.0);
    let src = "\
g1 = Normal(mu = 0.0, sigma = 1.0)
g2 = Normal(mu = 1.0, sigma = 2.0)
d = logweighted(x -> logdensityof(g2, x), g1)
lp = logdensityof(d, 0.5)";
    let m = parse_infer(src);
    let out = determinize(&m).expect("function-weighted logweighted must lower");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL"
    );
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "g1 + g2 terms:\n{pir}"
    );
    assert!(
        !pir.contains("(logweighted ") && !pir.contains("(logdensityof "),
        "measure layer gone:\n{pir}"
    );
    assert!(oracle.is_finite());
}

#[test]
fn normalize_truncated_normal_oracle() {
    // normalize(truncate(Normal(0,1), interval(-1,1))) scored at 0.5:
    //   = logN(0.5;0,1) - log(Φ(1) - Φ(-1))
    // Φ(1)-Φ(-1) = 0.6826894921370859
    let z = 0.6826894921370859_f64;
    let oracle = gaussian_logpdf(0.5, 0.0, 1.0) - z.ln();
    let src = "\
g = Normal(mu = 0.0, sigma = 1.0)
d = normalize(truncate(g, interval(-1.0, 1.0)))
lp = logdensityof(d, 0.5)";
    let m = parse_infer(src);
    let out = determinize(&m).expect("normalize(truncate) must lower");
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("builtin_touniform"),
        "closed-form Z via touniform:\n{pir}"
    );
    assert!(
        !pir.contains("(normalize ") && !pir.contains("totalmass"),
        "no normalize/totalmass:\n{pir}"
    );
    assert!(oracle.is_finite());
}

#[test]
fn likelihoodof_gaussian_oracle() {
    // obs = likelihoodof(iid(Normal(mu,sigma), 1), [1.27])
    // logdensityof(obs, record(mu=0, sigma=1)) = log N(1.27; 0, 1)
    let oracle = gaussian_logpdf(1.27, 0.0, 1.0);
    let src = "\
mu = elementof(reals)
sigma = elementof(posreals)
gauss_x = Normal(mu = mu, sigma = sigma)
obs = likelihoodof(iid(gauss_x, 1), [1.27])
lp = logdensityof(obs, record(mu = 0.0, sigma = 1.0))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("likelihood must lower");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL"
    );
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "1 term:\n{pir}"
    );
    assert!(
        !pir.contains("(likelihoodof ") && !pir.contains("(iid "),
        "measure layer gone:\n{pir}"
    );
    assert!(oracle.is_finite());
}

// Regression for review finding C1 (cross-query parameter leak). TWO likelihood
// queries over the SAME shared params (`mu`, `sigma`) at DISTINCT θ points must
// each score at its OWN θ. Each θ is inlined into that query's density subtree;
// the shared `mu`/`sigma` bindings are NOT mutated (which would clobber both
// terms to the last θ written — a silent mislowering that `is_flatpdl` passes).
#[test]
fn two_likelihood_queries_do_not_leak_theta_across_each_other() {
    let src = "\
mu = elementof(reals)
sigma = elementof(posreals)
gauss_x = Normal(mu = mu, sigma = sigma)
obs = likelihoodof(iid(gauss_x, 1), [1.27])
lp = logdensityof(obs, record(mu = 0.0, sigma = 1.0))
lp2 = logdensityof(obs, record(mu = 5.0, sigma = 2.0))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("two-query likelihood must lower");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL:\n{}",
        flatppl_flatpir::write(&out)
    );
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "one density term per query:\n{pir}"
    );

    // The two terms must be DISTINCT and each carry its OWN θ, inlined as
    // literals (not a shared `(%ref self mu/sigma)` that resolves to the last
    // θ). Inspect each query's binding line independently.
    let lp_line = pir
        .lines()
        .find(|l| l.contains("(%bind lp "))
        .expect("lp binding present");
    let lp2_line = pir
        .lines()
        .find(|l| l.contains("(%bind lp2 "))
        .expect("lp2 binding present");

    assert!(
        lp_line.contains("(%field mu 0.0)") && lp_line.contains("(%field sigma 1.0)"),
        "lp must score at ITS θ (mu=0.0, sigma=1.0):\n{lp_line}"
    );
    assert!(
        lp2_line.contains("(%field mu 5.0)") && lp2_line.contains("(%field sigma 2.0)"),
        "lp2 must score at ITS θ (mu=5.0, sigma=2.0):\n{lp2_line}"
    );

    // No θ leaked the other way: the two terms carry different values.
    assert!(
        !lp_line.contains("5.0") && !lp2_line.contains("0.0"),
        "θ leaked across the two queries:\nlp:  {lp_line}\nlp2: {lp2_line}"
    );

    // The shared params must NOT have been mutated to a θ literal — they stay
    // `elementof` free-param declarations (valid FlatPDL). A `(%bind mu 5.0)` /
    // `(%bind sigma 2.0)` is the smoking gun of the mutate-shared-bindings bug.
    assert!(
        pir.contains("(%bind mu (") && pir.contains("elementof reals"),
        "mu stays an elementof param decl (not clobbered to a θ literal):\n{pir}"
    );
    assert!(
        !pir.contains("(%bind mu 5.0)") && !pir.contains("(%bind sigma 2.0)"),
        "shared params must not be mutated to a query's θ:\n{pir}"
    );

    // And no residual self-ref to the (now-unused) params survives in either
    // scored density subtree.
    assert!(
        !lp_line.contains("(%ref self mu)") && !lp2_line.contains("(%ref self mu)"),
        "θ must be inlined, not left as a shared self-ref:\nlp:  {lp_line}\nlp2: {lp2_line}"
    );
}

// Regression fixture for transitive pinning (audit finding H3): a variate reached
// through a derived binding (`a = 2·theta`, `theta = draw(M)`) must score at
// the pinned `theta` and propagate transitively — no stochastic `draw` may
// survive, even though `a` is unreferenced by `lp` and depends on `theta`.
#[test]
fn derived_binding_pins_transitively() {
    // theta ~ Normal(0,1); a = 2*theta (derived). Score the joint at theta=0.5.
    // density should be log N(0.5; 0, 1), scored at the pinned theta.
    let oracle = gaussian_logpdf(0.5, 0.0, 1.0);
    let src = "\
theta = draw(Normal(mu = 0.0, sigma = 1.0))
a = mul(2.0, theta)
lp = logdensityof(lawof(record(theta = theta)), record(theta = 0.5))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("must lower");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "no stochastic draw survives (a's dep):\n{}",
        flatppl_flatpir::write(&out)
    );
    assert_eq!(
        flatppl_flatpir::write(&out)
            .matches("builtin_logdensityof")
            .count(),
        1
    );
    assert!(oracle.is_finite());
}

#[test]
fn empty_record_is_zero() {
    let src = "lp = logdensityof(lawof(record()), record())";
    let m = parse_infer(src);
    let out = determinize(&m).expect("empty record must lower to 0");
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
    let pir = flatppl_flatpir::write(&out);
    assert!(
        !pir.contains("builtin_logdensityof"),
        "no density terms:\n{pir}"
    );
}
