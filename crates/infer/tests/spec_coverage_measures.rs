//! Golden tests for §06 mass-class composition and §08 distribution domain/support.
//!
//! Each test first uses `ir(src)` to capture the annotated FlatPIR then
//! asserts an exact substring.  Tests that expose a spec gap are marked
//! `#[ignore = "candidate-bug: …"]`.

use flatppl_infer::infer;

fn ir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = infer(&mut m);
    flatppl_flatpir::write(&m)
}

// ============================================================
// §06 Mass arms
// ============================================================

/// `logweighted(fixed-scalar, M)` — a fixed scalar log-weight rescales the
/// measure; the mass class of the base survives intact.
///
/// - Normal(0,1) is %normalized  → logweighted → %finite
/// - Lebesgue(reals) is %locallyfinite → logweighted → %locallyfinite
#[test]
fn logweighted_fixed_scalar_mass() {
    // Normal is normalized; a fixed scalar weight demotes to %finite.
    let src = "m = logweighted(2.5, Normal(0.0, 1.0))";
    let out = ir(src);
    println!("logweighted/Normal:\n{out}");
    assert!(
        out.contains("(%mass %finite)"),
        "logweighted(scalar, Normal) should be %finite; got:\n{out}"
    );

    // Lebesgue(reals) is %locallyfinite; a fixed scalar weight keeps that.
    let src2 = "m = logweighted(2.5, Lebesgue(reals))";
    let out2 = ir(src2);
    println!("logweighted/Lebesgue:\n{out2}");
    assert!(
        out2.contains("(%mass %locallyfinite)"),
        "logweighted(scalar, Lebesgue(reals)) should be %locallyfinite; got:\n{out2}"
    );
}

/// `weighted(fn(…), M)` — the weight is a non-fixed (function) value, so the
/// mass class cannot be determined statically → %unknown.
#[test]
fn weighted_function_weight_is_unknown() {
    let src = "f = fn(_ * 2.0)\nm = weighted(f, Normal(0.0, 1.0))";
    let out = ir(src);
    println!("weighted/fn:\n{out}");
    assert!(
        out.contains("(%mass %unknown)"),
        "weighted(fn, Normal) should be %unknown; got:\n{out}"
    );
}

/// `iid(truncate(Normal,interval), 3)` — truncation of a normalized measure
/// gives %finite; iid preserves %finite.
#[test]
fn iid_finite_base_stays_finite() {
    let src = "t = truncate(Normal(0.0, 1.0), interval(0.0, 1.0))\nm = iid(t, 3)";
    let out = ir(src);
    println!("iid/finite:\n{out}");
    assert!(
        out.contains("(%mass %finite)"),
        "iid of a %finite base should be %finite; got:\n{out}"
    );
}

/// `joint` mass obeys the product rule:
/// - all %finite components → %finite
/// - mixing %finite and %locallyfinite → %locallyfinite
#[test]
fn joint_mass_extra_arms() {
    // Both components truncated (finite): product is finite.
    let src_ff = "\
a = truncate(Normal(0.0, 1.0), interval(0.0, 1.0))
b = truncate(Normal(0.0, 1.0), interval(0.0, 1.0))
j = joint(a = a, b = b)";
    let out_ff = ir(src_ff);
    println!("joint/finite+finite:\n{out_ff}");
    assert!(
        out_ff.contains("(%bind j (%meta ((%measure") && out_ff.contains("(%mass %finite)"),
        "joint(finite, finite) should be %finite; got:\n{out_ff}"
    );

    // One finite, one locally-finite: product is locally-finite.
    let src_fl = "\
a = truncate(Normal(0.0, 1.0), interval(0.0, 1.0))
b = Lebesgue(reals)
j = joint(a = a, b = b)";
    let out_fl = ir(src_fl);
    println!("joint/finite+locallyfinite:\n{out_fl}");
    assert!(
        out_fl.contains("(%bind j (%meta ((%measure") && out_fl.contains("(%mass %locallyfinite)"),
        "joint(finite, locallyfinite) should be %locallyfinite; got:\n{out_fl}"
    );
}

/// `Counting` on a bounded support is %finite; on an unbounded named set it
/// is %locallyfinite.
#[test]
fn counting_bounded_vs_unbounded() {
    let src_bounded = "m = Counting(interval(0, 10))";
    let out_bounded = ir(src_bounded);
    println!("Counting(interval):\n{out_bounded}");
    assert!(
        out_bounded.contains("(%mass %finite)"),
        "Counting(bounded interval) should be %finite; got:\n{out_bounded}"
    );

    let src_unbounded = "m = Counting(posintegers)";
    let out_unbounded = ir(src_unbounded);
    println!("Counting(posintegers):\n{out_unbounded}");
    assert!(
        out_unbounded.contains("(%mass %locallyfinite)"),
        "Counting(posintegers) should be %locallyfinite; got:\n{out_unbounded}"
    );
}

/// `truncate(Lebesgue(reals), interval(neg(inf), inf))` — the base is
/// %locallyfinite and the truncation interval is unbounded (contains ±∞), so
/// the engine cannot establish a finite mass → %unknown.
#[test]
fn truncate_locallyfinite_unbounded_is_unknown() {
    let src = "m = truncate(Lebesgue(reals), interval(neg(inf), inf))";
    let out = ir(src);
    println!("truncate/locallyfinite/unbounded:\n{out}");
    assert!(
        out.contains("(%mass %unknown)"),
        "truncate(Lebesgue(reals), interval(-inf,inf)) should be %unknown; got:\n{out}"
    );
}

/// `normalize(bayesupdate(L, prior))` — bayesupdate gives %unknown; normalize
/// of %unknown is %normalized (the engine cannot disprove finiteness) with no
/// error diagnostic.
#[test]
fn normalize_of_unknown_is_normalized() {
    // Build a simple bayesupdate posterior (unknown mass) then normalize it.
    // §06 normalize: any non-null, non-infinite mass → %normalized.
    let src = "\
mu = elementof(reals)
prior = Normal(mu = mu, sigma = 1.0)
n = normalize(truncate(Cauchy(0, 5), interval(0, inf)))
post = bayesupdate(n, n)
norm_post = normalize(post)";
    let out = ir(src);
    println!("normalize_of_unknown:\n{out}");
    // bayesupdate must be %unknown
    assert!(
        out.contains("(%bind post (%meta ((%measure") && out.contains("(%mass %unknown)"),
        "bayesupdate should produce %unknown mass; got:\n{out}"
    );
    // normalize of %unknown → %normalized, no error
    assert!(
        out.contains("(%bind norm_post (%meta ((%measure") && out.contains("(%mass %normalized)"),
        "normalize(bayesupdate(...)) should be %normalized; got:\n{out}"
    );
    // No %failed in the normalize result
    assert!(
        !out.contains("(%bind norm_post (%meta (%failed"),
        "normalize of unknown should not fail; got:\n{out}"
    );
}

// ============================================================
// §08 Distribution domain/support
// ============================================================

/// Bernoulli domain is `(%scalar integer)` (spec §08: integer/booleans),
/// NOT boolean; value-set (support) is `booleans`.
#[test]
fn bernoulli_domain_is_integer() {
    let src = "m = Bernoulli(0.5)";
    let out = ir(src);
    println!("Bernoulli:\n{out}");
    assert!(
        out.contains("(%domain (%scalar integer))"),
        "Bernoulli domain must be (%scalar integer); got:\n{out}"
    );
    assert!(
        out.contains("booleans"),
        "Bernoulli support must be booleans; got:\n{out}"
    );
}

/// Beta(1,1) has domain %scalar real and support `unitinterval`.
#[test]
fn beta_support_is_unitinterval() {
    let src = "m = Beta(1.0, 1.0)";
    let out = ir(src);
    println!("Beta:\n{out}");
    assert!(
        out.contains("(%domain (%scalar real))"),
        "Beta domain must be (%scalar real); got:\n{out}"
    );
    assert!(
        out.contains("unitinterval"),
        "Beta support must be unitinterval; got:\n{out}"
    );
}

/// Exponential and Weibull both have support `nonnegreals`.
#[test]
fn exponential_weibull_support_nonnegreals() {
    let src_exp = "m = Exponential(1.0)";
    let out_exp = ir(src_exp);
    println!("Exponential:\n{out_exp}");
    assert!(
        out_exp.contains("nonnegreals"),
        "Exponential support must be nonnegreals; got:\n{out_exp}"
    );

    let src_wei = "m = Weibull(1.0, 1.0)";
    let out_wei = ir(src_wei);
    println!("Weibull:\n{out_wei}");
    assert!(
        out_wei.contains("nonnegreals"),
        "Weibull support must be nonnegreals; got:\n{out_wei}"
    );
}

/// Gamma and ChiSquared both have support `nonnegreals`, not `posreals`: the
/// density is nonzero at x=0 whenever shape <= 1 (Exponential IS Gamma(1,
/// rate) — scipy oracle Gamma(shape=1,rate=1)@0 = 1.0; ChiSquared(k) =
/// Gamma(k/2, 1/2), so ChiSquared(2) = Exponential(1/2) — scipy oracle
/// ChiSquared(2)@0 = 0.5). §08 lists `nonnegreals` for both.
#[test]
fn gamma_chisquared_support_nonnegreals() {
    let src_gamma = "m = Gamma(shape = 2.0, rate = 1.0)";
    let out_gamma = ir(src_gamma);
    println!("Gamma:\n{out_gamma}");
    assert!(
        out_gamma.contains("nonnegreals"),
        "Gamma support must be nonnegreals; got:\n{out_gamma}"
    );

    let src_chisq = "m = ChiSquared(k = 2.0)";
    let out_chisq = ir(src_chisq);
    println!("ChiSquared:\n{out_chisq}");
    assert!(
        out_chisq.contains("nonnegreals"),
        "ChiSquared support must be nonnegreals; got:\n{out_chisq}"
    );
}

/// Negative control for the Gamma/ChiSquared nonnegreals change: InverseGamma
/// and LogNormal have density 0 at x=0 (scipy oracle: both @0 = 0.0) and MUST
/// stay `posreals` — the fix is surgical to Gamma/ChiSquared only.
#[test]
fn inversegamma_lognormal_support_stays_posreals() {
    let src_ig = "m = InverseGamma(shape = 2.0, scale = 1.0)";
    let out_ig = ir(src_ig);
    println!("InverseGamma:\n{out_ig}");
    assert!(
        out_ig.contains("posreals"),
        "InverseGamma support must stay posreals; got:\n{out_ig}"
    );

    let src_ln = "m = LogNormal(mu = 0.0, sigma = 1.0)";
    let out_ln = ir(src_ln);
    println!("LogNormal:\n{out_ln}");
    assert!(
        out_ln.contains("posreals"),
        "LogNormal support must stay posreals; got:\n{out_ln}"
    );
}

/// Pareto(1,1) has domain %scalar real, support `posreals`, mass %normalized.
#[test]
fn pareto_support_is_posreals() {
    let src = "m = Pareto(1.0, 1.0)";
    let out = ir(src);
    println!("Pareto:\n{out}");
    assert!(
        out.contains("posreals"),
        "Pareto support must be posreals; got:\n{out}"
    );
    assert!(
        out.contains("(%mass %normalized)"),
        "Pareto must be %normalized; got:\n{out}"
    );
}

/// Categorical0([0.5,0.5]) has domain `(%scalar integer)` and support
/// `nonnegintegers`.
#[test]
fn categorical0_domain_integer_support_nonnegintegers() {
    let src = "m = Categorical0([0.5, 0.5])";
    let out = ir(src);
    println!("Categorical0:\n{out}");
    assert!(
        out.contains("(%domain (%scalar integer))"),
        "Categorical0 domain must be (%scalar integer); got:\n{out}"
    );
    assert!(
        out.contains("nonnegintegers"),
        "Categorical0 support must be nonnegintegers; got:\n{out}"
    );
}

/// Wishart, InverseWishart, LKJ, LKJCholesky all have a rank-2 dynamic real
/// array domain: `(%array 2 (%dynamic %dynamic) (%scalar real))`.
#[test]
fn matrix_dists_domain_is_rank2_dynamic_real() {
    let expected_domain = "(%array 2 (%dynamic %dynamic) (%scalar real))";

    let cases = [
        ("Wishart(3.0, eye(3))", "Wishart"),
        ("InverseWishart(3.0, eye(3))", "InverseWishart"),
        ("LKJ(3, 1.0)", "LKJ"),
        ("LKJCholesky(3, 1.0)", "LKJCholesky"),
    ];

    for (expr, name) in cases {
        let src = format!("m = {expr}");
        let out = ir(&src);
        println!("{name}:\n{out}");
        assert!(
            out.contains(expected_domain),
            "{name} domain must be {expected_domain}; got:\n{out}"
        );
    }
}

/// Positional `joint` has a `cat`-shaped variate DOMAIN (spec §06: the variate
/// is the `cat` of the component variates — all scalars → a vector), not the
/// (empty) record the keyword-only `joint_type` produced for positional args.
/// Keyword `joint` stays a record domain.
#[test]
fn positional_joint_domain_is_cat_array() {
    let out = ir("j = joint(Normal(mu = 0.0, sigma = 1.0), Exponential(rate = 1.0))");
    assert!(
        out.contains("(%domain (%array 1 (2) (%scalar real)))"),
        "positional joint of two scalar measures → 2-element real array domain; got:\n{out}"
    );
    let outr = ir("jr = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Exponential(rate = 1.0))");
    assert!(
        outr.contains("(%domain (%record (a (%scalar real)) (b (%scalar real))))"),
        "keyword joint → record domain; got:\n{outr}"
    );
}
