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

// ============================================================
// Measure-argument gates on `lawof` and `kernelof`
// ============================================================
//
// Two §04 rules that nothing enforced before. `lawof(m)` requires `m`'s `%mass`
// to be `%normalized` (flatppl-design#73 @ 9d9a91c, pending owner review);
// `kernelof(x)` requires `x` to be a value node — "`x` must not be a measure",
// already normative in §04.

/// Diagnostics from a full infer pass, as `Debug` strings.
fn diags_of(src: &str) -> Vec<String> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m).iter().map(|d| format!("{d:?}")).collect()
}

fn rejects(src: &str, needle: &str) -> bool {
    diags_of(src).iter().any(|d| d.contains(needle))
}

/// §04 (#73): "`lawof(m)` requires `m`'s `%mass` to be `%normalized` …; anything
/// else is a static error, since an unnormalized measure is not its own law".
/// One case per rejected mass class the checker can produce.
#[test]
fn lawof_rejects_a_measure_that_is_not_normalized() {
    // %finite — a truncation is a restriction, mass ≤ 1 and not provably 1.
    assert!(rejects(
        "n = Normal(mu = 0.0, sigma = 1.0)\nt = truncate(n, interval(0.0, 1.0))\nz = lawof(t)",
        "requires a `%normalized` measure"
    ));
    // %locallyfinite — infinite total mass.
    assert!(rejects(
        "z = lawof(Lebesgue(reals))",
        "total mass is `%locallyfinite`"
    ));
    // The message names the escape hatch §04 names.
    assert!(rejects(
        "n = Normal(mu = 0.0, sigma = 1.0)\nt = truncate(n, interval(0.0, 1.0))\nz = lawof(t)",
        "normalize(...)"
    ));
}

/// The gate still quantifies over the mass CLASS — a measure it holds as
/// `%finite` is rejected however normalized it looks — but a `superpose` whose
/// weights provably sum to one is no longer `%finite` at all, so `lawof` of it is
/// legal. Both halves are pinned here because the distinction is the whole design:
/// the PROOF moved into the mass rule, the gate did not soften.
#[test]
fn lawof_takes_a_proven_mixture_and_still_refuses_a_finite_measure() {
    let proven = "\
n = Normal(mu = 0.0, sigma = 1.0)
sp = superpose(weighted(0.5, n), weighted(0.5, n))
z = lawof(sp)";
    assert!(diags_of(proven).is_empty(), "{:?}", diags_of(proven));
    let finite = "\
n = Normal(mu = 0.0, sigma = 1.0)
z = lawof(weighted(0.5, n))";
    assert!(rejects(finite, "total mass is `%finite`"));
    // `normalize` is the stated escape and must clear it.
    let fixed = "\
n = Normal(mu = 0.0, sigma = 1.0)
z = lawof(normalize(weighted(0.5, n)))";
    assert!(diags_of(fixed).is_empty(), "{:?}", diags_of(fixed));
}

/// The gate must not fire where §04 says `lawof` is fine: a `%normalized`
/// measure, a VALUE (the overwhelmingly common spelling), and a record of values.
#[test]
fn lawof_accepts_normalized_measures_and_values() {
    for src in [
        "z = lawof(Normal(mu = 0.0, sigma = 1.0))",
        "y ~ Normal(mu = 0.0, sigma = 1.0)\nz = lawof(y)",
        "y ~ Normal(mu = 0.0, sigma = 1.0)\nz = lawof(record(y = y))",
        "n = Normal(mu = 0.0, sigma = 1.0)\nz = lawof(normalize(truncate(n, interval(0.0, 1.0))))",
    ] {
        assert!(
            diags_of(src).is_empty(),
            "{src} must infer clean: {:?}",
            diags_of(src)
        );
    }
}

/// §04: `kernelof` "reifies (typically stochastic) value nodes"; "`x` must not be
/// a measure". Both measure-layer argument shapes are rejected, and the
/// diagnostics differ so a user with a kernel is not told they have a measure.
#[test]
fn kernelof_rejects_a_measure_layer_argument() {
    let measure = "mu = elementof(reals)\nz = kernelof(Normal(mu = mu, sigma = 1.0), mu = mu)";
    assert!(rejects(measure, "this argument is a measure"));
    assert!(
        rejects(measure, "functionof"),
        "point at the construct §04 names"
    );

    let kernel = "\
mu = elementof(reals)
y ~ Normal(mu = mu, sigma = 1.0)
k = kernelof(record(y = y), mu = mu)
z = kernelof(k, mu = mu)";
    assert!(rejects(kernel, "this argument is a kernel"));
}

/// `kernelof` of a VALUE stays legal, and so does `functionof` of a measure —
/// §04 (#73) makes `functionof` the construct that reifies a measure node to a
/// kernel directly, which is what the rejection message points at.
#[test]
fn kernelof_of_a_value_and_functionof_of_a_measure_stay_legal() {
    for src in [
        "mu = elementof(reals)\ny ~ Normal(mu = mu, sigma = 1.0)\nz = kernelof(record(y = y), mu = mu)",
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\nz = kernelof(a)",
        "mu = elementof(reals)\nz = functionof(Normal(mu = mu, sigma = 1.0), mu = mu)",
    ] {
        assert!(
            diags_of(src).is_empty(),
            "{src} must infer clean: {:?}",
            diags_of(src)
        );
    }
}

/// Both gates read the inferred TYPE, so an alias hop cannot defeat them — unlike
/// the determiniser's structural `resolve_ref_one` walks, whose one-hop defeat is
/// a known adjacent defect. A ref's type IS its target's type, at any depth.
#[test]
fn both_gates_survive_alias_hops() {
    assert!(rejects(
        "n = Normal(mu = 0.0, sigma = 1.0)\nt = truncate(n, interval(0.0, 1.0))\nt2 = t\nt3 = t2\nz = lawof(t3)",
        "requires a `%normalized` measure"
    ));
    // Three hops on the kernelof side too — the same depth the lawof case pins, so
    // neither gate is left resting on a single-hop claim.
    assert!(rejects(
        "mu = elementof(reals)\nn = Normal(mu = mu, sigma = 1.0)\nn2 = n\nz = kernelof(n2, mu = mu)",
        "this argument is a measure"
    ));
    assert!(rejects(
        "mu = elementof(reals)\nn = Normal(mu = mu, sigma = 1.0)\nn2 = n\nn3 = n2\nz = kernelof(n3, mu = mu)",
        "this argument is a measure"
    ));
    // And for the KERNEL body, which reaches the gate by a different arm.
    assert!(rejects(
        "\
mu = elementof(reals)
y ~ Normal(mu = mu, sigma = 1.0)
k = kernelof(record(y = y), mu = mu)
k2 = k
k3 = k2
z = kernelof(k3, mu = mu)",
        "this argument is a kernel"
    ));
}

// ============================================================
// `lawof` of a measure is the IDENTITY (#73's definition half)
// ============================================================

/// §04 as amended by flatppl-design#73 (@ `9d9a91c`, pending owner review):
/// "`lawof(m)` is `lawof(draw(m))`, the law of a draw from `m`" and "A probability
/// measure of fixed or parameterized phase is its own law, so `lawof(m)` is
/// equivalent to `m` and `lawof` is idempotent."
///
/// So the result type is the ARGUMENT's type, not a measure wrapping it. Before
/// this, `lawof(Normal(…))` typed as a measure whose DOMAIN was a measure.
#[test]
fn lawof_of_a_normalized_measure_is_the_identity() {
    let out = ir("n = Normal(mu = 0.0, sigma = 1.0)\nz = lawof(n)");
    let n = out.lines().find(|l| l.contains("%bind n")).unwrap_or("");
    let z = out.lines().find(|l| l.contains("%bind z")).unwrap_or("");
    let ty = "((%measure (%domain (%scalar real)) (%mass %normalized))";
    assert!(n.contains(ty), "the measure itself:\n{out}");
    assert!(z.contains(ty), "lawof of it must type identically:\n{out}");
    // No measure-over-measure anywhere.
    assert!(
        !z.contains("(%domain (%measure"),
        "must not wrap the argument as the domain:\n{out}"
    );
}

/// §04's idempotence, which falls out of the identity: `lawof(lawof(m))` types as
/// `m`. Worth its own pin — it is the property that would break first if the arm
/// ever went back to wrapping.
#[test]
fn lawof_is_idempotent_on_a_measure() {
    let out = ir("n = Normal(mu = 0.0, sigma = 1.0)\nz = lawof(lawof(lawof(n)))");
    assert!(
        out.lines()
            .find(|l| l.contains("%bind z"))
            .unwrap_or("")
            .contains("((%measure (%domain (%scalar real)) (%mass %normalized))"),
        "three lawof layers type as one:\n{out}"
    );
}

/// A `normalize(...)`d restriction is the spelling §04 names as the escape, and
/// `normalize` always stamps `%normalized` (never `%deferred`), so this case
/// stays a theorem, not the gate's `%deferred` admission route — see
/// `lawof_of_a_deferred_mass_measure_stays_deferred` for that one.
#[test]
fn lawof_of_a_normalized_restriction_types_over_the_element_domain() {
    let out = ir("n = Normal(mu = 0.0, sigma = 1.0)\n\
         z = lawof(normalize(truncate(n, interval(0.0, 1.0))))");
    assert!(
        out.lines()
            .find(|l| l.contains("%bind z"))
            .unwrap_or("")
            .contains("((%measure (%domain (%scalar real)) (%mass %normalized))"),
        "a measure over the ELEMENT domain, not over the measure:\n{out}"
    );
}

/// §04: "On a non-nullary kernel, `lawof` lifts pointwise, as the uniform kernel
/// extension does for measure-algebra operations." So the result is a KERNEL over
/// the same inputs, not a measure whose domain is a kernel.
#[test]
fn lawof_of_a_kernel_lifts_pointwise() {
    let out = ir("mu = elementof(reals)\n\
         y ~ Normal(mu = mu, sigma = 1.0)\n\
         k = kernelof(record(y = y), mu = mu)\n\
         z = lawof(k)");
    let z = out.lines().find(|l| l.contains("%bind z")).unwrap_or("");
    assert!(
        z.contains("((%kernel (%inputs mu) (%mass %normalized))"),
        "a kernel over the same inputs:\n{out}"
    );
    assert!(
        !z.contains("(%domain (%kernel"),
        "must not wrap the kernel as a domain:\n{out}"
    );
}

/// `lawof`'s gate extends to a KERNEL argument by the same three rules as the
/// measure case, pointwise (owner ruling, `lawof-kernel-mass-maths.md`,
/// 2026-08-19): §04's "On a non-nullary kernel, `lawof` lifts pointwise"
/// composes the whole measure-argument mass clause — requirement, settled-class
/// error, and no-laundering rider alike — onto each output measure the kernel
/// generates. Wherever `lawof(K)` is defined at all, every output measure has
/// mass 1, so a kernel with a SETTLED non-`%normalized` mass (here `%finite`,
/// from `functionof` over a `weighted(...)` body) types an expression that has
/// no value and must be a static error, exactly like `lawof` of a `%finite`
/// measure. Before this fix the kernel branch was ungated and stamped the
/// result `%normalized` regardless.
#[test]
fn lawof_rejects_a_kernel_that_is_not_normalized() {
    assert!(rejects(
        "k = functionof(weighted(0.5, Normal(mu = 0.0, sigma = 1.0)))\nq = lawof(k)",
        "total mass is `%finite`"
    ));
    // The parameterized-input shape from the maths doc's probe 2, so a
    // regression cannot narrow the fix to the nullary case alone.
    assert!(rejects(
        "z = elementof(reals)\n\
         k = functionof(weighted(0.5, Normal(mu = z, sigma = 1.0)), z = z)\n\
         q = lawof(k)",
        "total mass is `%finite`"
    ));
}

/// The kernel gate's `%deferred` arm mirrors the measure arm's no-laundering
/// rider exactly: an admitted `%deferred`-mass kernel must come out `%deferred`,
/// never `%normalized`. `functionof(joint())` is the executed producer (a
/// zero-component `joint` body, the one source of a genuinely `%deferred`-mass
/// measure reachable from source).
#[test]
fn lawof_of_a_deferred_mass_kernel_stays_deferred() {
    let out = ir("k = functionof(joint())\nq = lawof(k)");
    let k = out.lines().find(|l| l.contains("%bind k")).unwrap_or("");
    let q = out.lines().find(|l| l.contains("%bind q")).unwrap_or("");
    assert!(
        k.contains("(%kernel (%inputs ) (%mass %deferred))"),
        "functionof(joint()) must itself be a %deferred-mass kernel:\n{out}"
    );
    assert!(
        q.contains("(%kernel (%inputs ) (%mass %deferred))"),
        "lawof of a %deferred-mass kernel must stay %deferred, not launder to \
         %normalized:\n{out}"
    );
    assert!(
        !q.contains("%normalized"),
        "must not stamp the unproven assumption as known:\n{out}"
    );
}

/// Design-PR #73 option C's no-laundering rider (owner ruling, decisions-log
/// 2026-08-18): an engine admitting a `%deferred`-mass argument to `lawof` must
/// leave the RESULT's mass `%deferred`, never stamp `%normalized` — stamping
/// `%normalized` would record an unproven assumption as §11's "statically KNOWN"
/// class. `joint()` (zero components) is the one measure reachable from source
/// whose mass is genuinely `%deferred` (`product_mass`'s empty-list arm), so it
/// is the red case: `lawof`'s gate admits it (deferred passes, per
/// `unprovable_normalization`), and the result must carry `%deferred` onward.
#[test]
fn lawof_of_a_deferred_mass_measure_stays_deferred() {
    let out = ir("e = joint()\nq = lawof(e)");
    let e = out.lines().find(|l| l.contains("%bind e")).unwrap_or("");
    let q = out.lines().find(|l| l.contains("%bind q")).unwrap_or("");
    assert!(
        e.contains("(%mass %deferred)"),
        "joint() must itself be %deferred mass (the gate's admission case):\n{out}"
    );
    assert!(
        q.contains("(%mass %deferred)"),
        "lawof of a %deferred-mass measure must stay %deferred, not launder to \
         %normalized:\n{out}"
    );
    assert!(
        !q.contains("(%mass %normalized)"),
        "must not stamp the unproven assumption as known:\n{out}"
    );
}

/// The VALUE spelling — the only one the corpus uses — is unchanged: the law of a
/// value is a measure over that value's type.
#[test]
fn lawof_of_a_value_still_types_over_the_value() {
    for (src, want) in [
        (
            "y ~ Normal(mu = 0.0, sigma = 1.0)\nz = lawof(y)",
            "((%measure (%domain (%scalar real)) (%mass %normalized))",
        ),
        (
            "y ~ Normal(mu = 0.0, sigma = 1.0)\nz = lawof(record(y = y))",
            "((%measure (%domain (%record (y (%scalar real)))) (%mass %normalized))",
        ),
    ] {
        let out = ir(src);
        assert!(
            out.lines()
                .find(|l| l.contains("%bind z"))
                .unwrap_or("")
                .contains(want),
            "{src}\nwant {want}\n{out}"
        );
    }
}

/// Alias-transparent, like the gate: the identity is read off the inferred type, so
/// hops cannot change it.
#[test]
fn the_lawof_identity_survives_alias_hops() {
    let out = ir("n = Normal(mu = 0.0, sigma = 1.0)\nn2 = n\nn3 = n2\nz = lawof(n3)");
    assert!(
        out.lines()
            .find(|l| l.contains("%bind z"))
            .unwrap_or("")
            .contains("((%measure (%domain (%scalar real)) (%mass %normalized))"),
        "three alias hops must not change the identity typing:\n{out}"
    );
}

// ============================================================
// `draw` requires a probability measure (owner ruling; spec PR to follow)
// ============================================================
//
// No implicit normalization: drawing from a measure whose mass is not a
// probability is a static error, and `normalize(m)` is the escape. Derived from
// #73's equation read right-to-left — `lawof(m)` = `lawof(draw(m))` and `lawof`
// requires `%normalized`, so a draw from an unnormalized measure has no law.
// Mirrors `lawof`'s gate exactly; both route through `unprovable_normalization`.

/// Every mass class the checker can prove is not a probability is rejected, and the
/// message names `normalize(...)`.
#[test]
fn draw_rejects_a_measure_that_is_not_a_probability() {
    let n = "n = Normal(mu = 0.0, sigma = 1.0)\n";
    // %finite by restriction.
    assert!(rejects(
        &format!("{n}y = draw(truncate(n, interval(0.0, 1.0)))"),
        "`draw` requires a probability measure"
    ));
    // %finite by reweighting.
    assert!(rejects(
        &format!("{n}y = draw(weighted(2.0, n))"),
        "total mass is `%finite`"
    ));
    // %locallyfinite — a reference measure.
    assert!(rejects(
        "y = draw(Lebesgue(reals))",
        "total mass is `%locallyfinite`"
    ));
    // The escape is named.
    assert!(rejects(
        &format!("{n}y = draw(truncate(n, interval(0.0, 1.0)))"),
        "normalize(...)"
    ));
}

/// A mixture whose weights PROVABLY sum to one is `%normalized`, so the gate lets
/// it through — the mass rule proves the normalization upstream rather than the
/// gate relaxing. Both readings, and the `normalize` spelling §06 recommends stays
/// valid (it is then a no-op on an already-normalized measure).
///
/// This reverses an earlier decision: the same `superpose(weighted(0.5, n),
/// weighted(0.5, n))` was pinned as REJECTED when the only rule was
/// class-quantified. What changed is not the gate but how much the mass rule
/// proves; the gate still rejects every measure it is handed as `%finite`, which
/// [`draw_still_rejects_a_finite_measure`] pins.
#[test]
fn draw_accepts_a_mixture_whose_weights_provably_sum_to_one() {
    for (label, src) in [
        (
            "literal halves",
            "\
n = Normal(mu = 0.0, sigma = 1.0)
sp = superpose(weighted(0.5, n), weighted(0.5, n))
y = draw(sp)",
        ),
        (
            "literal decimals that no float width adds to one",
            "\
n = Normal(mu = 0.0, sigma = 1.0)
sp = superpose(weighted(0.1, n), weighted(0.2, n), weighted(0.7, n))
y = draw(sp)",
        ),
        (
            "the complement pattern with a stochastic weight — the zero-inflated shape",
            "\
psi ~ Beta(1.5, 1.5)
sp = superpose(weighted(psi, Binomial(20, 0.4)), weighted(1 - psi, Dirac(0)))
y = draw(sp)",
        ),
        (
            "the complement pattern with a parameterized weight",
            "\
w = elementof(unitinterval)
n = Normal(mu = 0.0, sigma = 1.0)
sp = superpose(weighted(w, n), weighted(1 - w, Dirac(0.0)))
y = draw(sp)",
        ),
        (
            "components bound separately, so the proof looks through references",
            "\
psi ~ Beta(1.5, 1.5)
hit = weighted(psi, Binomial(20, 0.4))
miss = weighted(1 - psi, Dirac(0))
y = draw(superpose(hit, miss))",
        ),
        (
            "the explicit normalize spelling stays valid on top of the proof",
            "\
psi ~ Beta(1.5, 1.5)
sp = superpose(weighted(psi, Binomial(20, 0.4)), weighted(1 - psi, Dirac(0)))
y = draw(normalize(sp))",
        ),
        (
            "iid of the mixture, which is how a model actually draws it",
            "\
psi ~ Beta(1.5, 1.5)
sp = superpose(weighted(psi, Binomial(20, 0.4)), weighted(1 - psi, Dirac(0)))
y = draw(iid(sp, 10))",
        ),
    ] {
        assert!(
            diags_of(src).is_empty(),
            "{label} must infer clean: {:?}",
            diags_of(src)
        );
    }
}

/// The sum-to-one proof is exactly as wide as its two readings. Each of these is
/// a mixture a human would call normalized, and none of them is PROVEN so, which
/// is the point: no arithmetic prover, no value-level reasoning about weights.
#[test]
fn the_sum_to_one_proof_rejects_everything_it_cannot_decide() {
    for (label, src) in [
        (
            "not a complement: 1 - 2e",
            "\
psi ~ Beta(1.5, 1.5)
sp = superpose(weighted(psi, Binomial(20, 0.4)), weighted(1 - 2 * psi, Dirac(0)))
y = draw(sp)",
        ),
        (
            "two DIFFERENT subtrees, e1 and 1 - e2, even with identical laws",
            "\
psi ~ Beta(1.5, 1.5)
phi ~ Beta(1.5, 1.5)
sp = superpose(weighted(psi, Binomial(20, 0.4)), weighted(1 - phi, Dirac(0)))
y = draw(sp)",
        ),
        (
            "a complement whose part is not proven to lie in [0, 1]",
            "\
w = elementof(reals)
n = Normal(mu = 0.0, sigma = 1.0)
sp = superpose(weighted(w, n), weighted(1 - w, Dirac(0.0)))
y = draw(sp)",
        ),
        (
            "literals summing to 0.999",
            "\
n = Normal(mu = 0.0, sigma = 1.0)
sp = superpose(weighted(0.5, n), weighted(0.499, n))
y = draw(sp)",
        ),
        (
            // f64 addition of these three IS exactly 1.0 — two roundings land on
            // it — so an engine that folded the weights in f64 would wrongly
            // prove this. The declared decimals sum to 0.9999999999999999.
            "literals that only f64 addition would call one",
            "\
n = Normal(mu = 0.0, sigma = 1.0)
sp = superpose(
    weighted(0.3333333333333333, n),
    weighted(0.3333333333333333, n),
    weighted(0.3333333333333333, n))
y = draw(sp)",
        ),
        (
            "a negative weight, so the components are not both measures",
            "\
n = Normal(mu = 0.0, sigma = 1.0)
sp = superpose(weighted(-0.5, n), weighted(1.5, n))
y = draw(sp)",
        ),
        (
            "weights that sum to one over an UNNORMALIZED component",
            "\
n = Normal(mu = 0.0, sigma = 1.0)
u = weighted(3.0, n)
sp = superpose(weighted(0.5, u), weighted(0.5, n))
y = draw(sp)",
        ),
        (
            "a bare component that is not `weighted` at all",
            "\
n = Normal(mu = 0.0, sigma = 1.0)
sp = superpose(n, weighted(0.5, n))
y = draw(sp)",
        ),
        (
            // THE soundness control. These two weight subtrees are syntactically
            // identical and are two INDEPENDENT coordinates (#73: "each draw from
            // `m` is a fresh coordinate"), so the masses sum to one only on a
            // probability-zero event. Structural equality alone accepted this and
            // typed it `%normalized` — an almost-surely-non-probability lowered as
            // a law with no normalizer, i.e. a silently wrong number.
            "two INLINE draws that are structurally identical but independent",
            "\
n1 = Normal(mu = 0.0, sigma = 1.0)
n2 = Normal(mu = 5.0, sigma = 1.0)
mix = superpose(
    weighted(draw(Uniform(interval(0.0, 1.0))), n1),
    weighted(1 - draw(Uniform(interval(0.0, 1.0))), n2))
y = draw(mix)",
        ),
        (
            // The same hole one phase over: §04 says each `elementof` LEAF becomes
            // an input of the reified callable, so two occurrences are two
            // parameters. Pinned to prove the exclusion set is a property and not
            // the single name `draw`.
            "two INLINE elementof parameters, structurally identical but distinct",
            "\
n1 = Normal(mu = 0.0, sigma = 1.0)
n2 = Normal(mu = 5.0, sigma = 1.0)
mix = superpose(
    weighted(elementof(unitinterval), n1),
    weighted(1 - elementof(unitinterval), n2))
y = draw(mix)",
        ),
    ] {
        assert!(
            !diags_of(src).is_empty(),
            "{label} must NOT be proven normalized, but inferred clean"
        );
        assert!(
            rejects(src, "`draw` requires a probability measure"),
            "{label} must be rejected by the draw gate: {:?}",
            diags_of(src)
        );
    }
}

/// The gate's class-quantified rejection, unchanged by the sum-to-one proof: a
/// measure the checker holds as `%finite` is refused whatever produced it.
#[test]
fn draw_still_rejects_a_finite_measure() {
    let src = "\
n = Normal(mu = 0.0, sigma = 1.0)
y = draw(weighted(0.5, n))";
    assert!(rejects(src, "total mass is `%finite`"));
    let fixed = "\
n = Normal(mu = 0.0, sigma = 1.0)
y = draw(normalize(weighted(0.5, n)))";
    assert!(diags_of(fixed).is_empty(), "{:?}", diags_of(fixed));
}

/// `lawof` shares the improvement, because the proof lives in the `superpose`
/// MASS rule and not in either gate: the mixture is `%normalized`, so it is its
/// own law.
#[test]
fn lawof_accepts_a_mixture_whose_weights_provably_sum_to_one() {
    let src = "\
psi ~ Beta(1.5, 1.5)
sp = superpose(weighted(psi, Binomial(20, 0.4)), weighted(1 - psi, Dirac(0)))
m = lawof(sp)";
    assert!(diags_of(src).is_empty(), "{:?}", diags_of(src));
    // Unproven weights still fail `lawof`'s gate, with its own message.
    let unproven = "\
psi ~ Beta(1.5, 1.5)
phi ~ Beta(1.5, 1.5)
sp = superpose(weighted(psi, Binomial(20, 0.4)), weighted(1 - phi, Dirac(0)))
m = lawof(sp)";
    assert!(rejects(unproven, "`%unknown`"));
}

/// What the gate must NOT touch: the ordinary spellings every model uses, and the
/// explicit `normalize` escape. `draw` of a KERNEL is also left alone — §06's
/// uniform kernel extension scopes itself to "measure-to-measure operations", and
/// `draw` is measure-to-value, so nothing licenses a pointwise reading to gate.
#[test]
fn draw_accepts_probabilities_the_escape_and_a_kernel() {
    for src in [
        "y = draw(Normal(mu = 0.0, sigma = 1.0))",
        "y ~ Normal(mu = 0.0, sigma = 1.0)",
        "y = draw(iid(Normal(mu = 0.0, sigma = 1.0), 3))",
        "n = Normal(mu = 0.0, sigma = 1.0)\ny = draw(normalize(truncate(n, interval(0.0, 1.0))))",
        "n = Normal(mu = 0.0, sigma = 1.0)\ny = draw(normalize(weighted(2.0, n)))",
        "mu = elementof(reals)\ny ~ Normal(mu = mu, sigma = 1.0)\n\
         k = kernelof(record(y = y), mu = mu)\nz = draw(k)",
    ] {
        assert!(
            diags_of(src).is_empty(),
            "{src} must infer clean: {:?}",
            diags_of(src)
        );
    }
}

/// Alias-transparent at three hops on BOTH arms, like the `lawof`/`kernelof` gates:
/// the rule reads the inferred type, and a ref's type is its target's.
#[test]
fn the_draw_gate_survives_alias_hops() {
    assert!(rejects(
        "n = Normal(mu = 0.0, sigma = 1.0)\nt = truncate(n, interval(0.0, 1.0))\n\
         t2 = t\nt3 = t2\ny = draw(t3)",
        "`draw` requires a probability measure"
    ));
    // The accepting side must be alias-transparent too, or the gate could be
    // defeated into REFUSING a legitimate model by an alias hop.
    let ok = "n = Normal(mu = 0.0, sigma = 1.0)\nm = normalize(truncate(n, interval(0.0, 1.0)))\n\
              m2 = m\nm3 = m2\ny = draw(m3)";
    assert!(diags_of(ok).is_empty(), "{:?}", diags_of(ok));
}

/// The shape that motivated the ruling: `draw(truncate(lawof(…), S))` used to lower
/// to the marginal density gated on `S` with no normalizer — an unnormalized
/// measure silently presented as a law. Both gates now speak, and `lawof`'s fires
/// on the inner argument only when it is itself unnormalized, so this reports the
/// `draw`.
#[test]
fn the_ldid_unnormalized_shape_is_now_rejected() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(truncate(lawof(Normal(mu = a, sigma = 1.0)), interval(0.0, 3.0)))
lp = logdensityof(lawof(y), 0.5)";
    assert!(rejects(src, "`draw` requires a probability measure"));
    // And the explicit escape clears it.
    let fixed = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(normalize(truncate(lawof(Normal(mu = a, sigma = 1.0)), interval(0.0, 3.0))))
lp = logdensityof(lawof(y), 0.5)";
    assert!(diags_of(fixed).is_empty(), "{:?}", diags_of(fixed));
}

// ============================================================
// `rand` requires a probability measure (#425 class; mirrors the `draw` gate)
// ============================================================
//
// §07: `rand(rstate, m)` draws a variate from a normalized measure `m`. Same
// owner ruling as `draw`, same `unprovable_normalization` gate, adapted to
// `rand`'s second argument (the first is the rngstate).

/// The unprovable classes are rejected and the message names the mass class
/// and the `normalize(...)` escape, exactly like the `draw` gate.
#[test]
fn rand_rejects_a_measure_that_is_not_a_probability() {
    let src = "s = rnginit(0)\ny = rand(s, Lebesgue(support = reals))";
    assert!(rejects(src, "`rand` requires a probability measure"));
    assert!(rejects(src, "total mass is `%locallyfinite`"));
    assert!(rejects(src, "normalize(...)"));
}

/// The ordinary spelling — `rand` of a plain probability distribution — still
/// infers the `(domain, %rngstate)` tuple untouched by the gate.
#[test]
fn rand_still_accepts_a_probability_measure() {
    let out = ir("s = rnginit(0)\ny = rand(s, Normal(0.0, 1.0))");
    assert!(
        out.lines()
            .find(|l| l.contains("%bind y"))
            .unwrap_or("")
            .contains("(%tuple (%scalar real) %rngstate)"),
        "{out}"
    );
}

/// `normalize(M)` of a `%finite` M is provably `Mass::Normalized` and passes
/// the gate for free, same as it does for `draw`.
#[test]
fn rand_accepts_normalize_of_a_finite_measure() {
    let src = "s = rnginit(0)\n\
               y = rand(s, normalize(truncate(Lebesgue(support = reals), interval(0.0, 1.0))))";
    assert!(diags_of(src).is_empty(), "{:?}", diags_of(src));
}

/// `%deferred` mass is "not yet inferred", not a proven-unnormalized verdict,
/// so the gate lets it through honestly rather than forcing a false refusal.
/// Stopping inference at `Level::Valueset` (before the Normalization-level
/// mass pass fills it in) is the one way to hand the gate a measure whose
/// mass is genuinely still `%deferred`, per `level_valueset_vs_normalization`
/// in `golden.rs`.
#[test]
fn rand_accepts_a_deferred_mass_measure() {
    let src = "s = rnginit(0)\ny = rand(s, Normal(0.0, 1.0))";
    let mut module = flatppl_syntax::parse(src).unwrap();
    let diags = flatppl_infer::infer_with(&mut module, flatppl_infer::Level::Valueset);
    let out = flatppl_flatpir::write(&module);
    assert!(
        out.contains("%mass %deferred"),
        "the mass must genuinely still be deferred at this level:\n{out}"
    );
    let diags: Vec<String> = diags.iter().map(|d| format!("{d:?}")).collect();
    assert!(
        !diags
            .iter()
            .any(|d| d.contains("`rand` requires a probability measure")),
        "{diags:?}"
    );
}

// ============================================================
// `joint` mass class (kernel-joint-q4-maths.md §7)
// ============================================================

/// All-normalized components stay `%normalized` regardless of arity spelling
/// (kernel-joint-q4-maths.md §7: "the record law of probability components
/// has mass 1") — the positional all-Normal control case.
#[test]
fn joint_mass_positional_all_normalized_stays_normalized() {
    let src = "m = joint(Normal(0.0, 1.0), Normal(1.0, 2.0))";
    let out = ir(src);
    assert!(
        out.contains("(%mass %normalized)"),
        "positional all-normalized joint must be %normalized; got:\n{out}"
    );
}

/// Two components sharing a stochastic ancestor (`z1`, `z2` both trace back
/// to `mu`) and both non-normalized after `truncate`: the product rule is
/// unsound here in general (kernel-joint-q4-maths.md §7, Student-t/`y^2`
/// counterexample). Both components are `lawof`-reified, so neither is
/// provably trace-clean (`joint_component_is_trace_clean` — reification is
/// one of the two channels spec §06's `joint` entry names for sharing a
/// stochastic node) and the composed class degrades to `%unknown`.
///
/// This does NOT prove the degrade fires because of the shared `mu` — a pair
/// of `lawof`-reified components with no shared ancestor at all would hit the
/// same branch and read the same `%unknown` (no ancestor-identity oracle
/// exists to tell the two apart, by the wave brief's design). What the
/// carve-out DOES discriminate is reified vs bare-constructor components: see
/// `positional_joint_mass_matches_keyword_joint_mass` in `golden.rs`, whose
/// two bare `Lebesgue(reals)` components are provably trace-clean and stay
/// exact at `%locallyfinite` under the identical two-non-normalized shape.
#[test]
fn joint_mass_two_nonnormalized_components_degrade_to_unknown() {
    let src = "mu ~ Normal(0.0, 1.0)\n\
               z1 ~ Normal(mu, 1.0)\n\
               z2 ~ Normal(mu, 1.0)\n\
               lz1 = lawof(z1)\n\
               lz2 = lawof(z2)\n\
               j = joint(truncate(lz1, interval(0.0, 5.0)), truncate(lz2, interval(0.0, 5.0)))";
    let out = ir(src);
    let bind = out
        .find("(%bind j ")
        .unwrap_or_else(|| panic!("binding j not found in:\n{out}"));
    assert!(
        out[bind..(bind + 120).min(out.len())].contains("%unknown"),
        "shared-ancestor joint of two non-normalized components must be \
         %unknown; got:\n{out}"
    );
}

/// A single non-normalized component sharing ancestry with normalized peers
/// stays exact (kernel-joint-q4-maths.md §7: "exactly one non-normalized
/// component… its class carries over. Sound.") — no degrade when only one
/// member deviates from `%normalized`.
#[test]
fn joint_mass_one_nonnormalized_component_stays_exact() {
    let src = "mu ~ Normal(0.0, 1.0)\n\
               z1 ~ Normal(mu, 1.0)\n\
               z2 ~ Normal(mu, 1.0)\n\
               lz2 = lawof(z2)\n\
               j = joint(truncate(lawof(z1), interval(0.0, 5.0)), lz2)";
    let out = ir(src);
    let bind = out
        .find("(%bind j ")
        .unwrap_or_else(|| panic!("binding j not found in:\n{out}"));
    assert!(
        out[bind..(bind + 120).min(out.len())].contains("%finite"),
        "one non-normalized (truncated) + one normalized component must stay \
         %finite; got:\n{out}"
    );
}

/// `draw`/`rand` gate integration: a positional `joint` of two non-normalized
/// components must be refused, same as the keyword form always was — this is
/// the draw/rand-gate consumer of the mass fix above, not a new gate rule.
/// Refuses at `%locallyfinite` (the trace-clean carve-out keeps this shape
/// exact; see `positional_joint_mass_matches_keyword_joint_mass`), same as it
/// did once `#159` landed — the gate itself is untouched by this wave.
#[test]
fn draw_of_positional_joint_two_lebesgue_refuses() {
    assert!(rejects(
        "y = draw(joint(Lebesgue(reals), Lebesgue(reals)))",
        "requires a probability measure"
    ));
}

/// `rand` shares `draw`'s gate (`unprovable_normalization`) but a different
/// argument offset (measure at index 1) — pin the pair so the two cannot
/// drift.
#[test]
fn rand_of_positional_joint_two_lebesgue_refuses() {
    assert!(rejects(
        "s = rnginit(0)\ny = rand(s, joint(Lebesgue(reals), Lebesgue(reals)))",
        "requires a probability measure"
    ));
}

/// Regression (review Critical 1): `normalize` is the one consumer that
/// distinguishes `%locallyfinite` from `%unknown` — a `%locallyfinite`
/// argument is a static error (spec §06, undefined normalization), while
/// `%unknown` silently passes through as `%normalized`. Before the trace-clean
/// carve-out, this exact shape's class widened from `%locallyfinite` to
/// `%unknown` and this error disappeared, reopening (via `normalize`) the
/// hole the wave exists to close: `draw(normalize(joint(Lebesgue(reals),
/// Lebesgue(reals))))` would type `%normalized` and pass the draw gate. Both
/// components are provably trace-clean bare constructors, so this must stay
/// a static error.
#[test]
fn normalize_of_two_lebesgue_joint_is_a_static_error() {
    assert!(rejects(
        "n = normalize(joint(Lebesgue(reals), Lebesgue(reals)))",
        "infinite total mass is undefined"
    ));
    assert!(rejects(
        "n = normalize(joint(a = Lebesgue(reals), b = Lebesgue(reals)))",
        "infinite total mass is undefined"
    ));
}

/// Regression (re-review Important 3): the disqualifier catalogue must not
/// disqualify a component just because it depends on an `elementof`
/// parameter — §04 "Phases" classifies `elementof` inputs as *parameterized*,
/// not stochastic, and kernel-joint-q4-maths.md §8 is explicit that "a shared
/// input name is a shared value, not a shared stochastic node". Neither
/// component below has a `draw` anywhere, so there is no stochastic node for
/// them to share, and the joint must stay exact and trigger the same static
/// error as the all-literal version — every HS3-converted model
/// parameterizes exactly this way (`rf304_uncorrprod/model.flatppl`:
/// `mean1 = elementof(reals)`, …), so a wrong answer here would be silent on
/// every converted fixture.
#[test]
fn normalize_of_elementof_parameterized_joint_is_a_static_error() {
    assert!(rejects(
        "a = elementof(reals)\n\
         c1 = locscale(Lebesgue(reals), a, 2.0)\n\
         c2 = locscale(Lebesgue(reals), a, 3.0)\n\
         n = normalize(joint(c1, c2))",
        "infinite total mass is undefined"
    ));
}

/// Same regression, milder instance: `truncate` of a `%locallyfinite` measure
/// to a bounded set is `%finite` (`ops.rs` truncate/restrict arm); at
/// `%unknown` it would fall through to `%unknown` instead, losing a provable
/// class with no diagnostic to mark the loss.
#[test]
fn truncate_of_two_lebesgue_joint_is_finite() {
    let src = "j = truncate(joint(Lebesgue(reals), Lebesgue(reals)), interval(0.0, 1.0))";
    let out = ir(src);
    assert!(
        out.contains("(%mass %finite)"),
        "truncated two-Lebesgue joint must be %finite; got:\n{out}"
    );
}

/// `joint()` (arity zero): `product_mass(&[])`'s `all()` was vacuously
/// `%normalized` regardless of arity — the same root cause the brief names
/// for the positional bug, reachable through arity zero instead of the
/// named/positional split. The domain arm already leaves a zero-component
/// `joint`'s domain `%deferred` (nothing resolves the variate shape), so the
/// mass side now matches that honestly instead of claiming a definite class
/// the domain does not support. `joint()`'s legality is a separate,
/// unaddressed question — not decided here.
#[test]
fn joint_of_no_components_is_deferred_not_normalized() {
    let src = "e = joint()";
    let out = ir(src);
    assert!(
        out.contains("(%mass %deferred)"),
        "empty joint must be %deferred, not a definite class; got:\n{out}"
    );
}

// ============================================================
// `joint` over KERNEL components — the fan-out kernel
// (spec §06 `joint` entry per flatppl-design#85; Q1–Q5 derived in
// flatppl-dev/kernel-joint-q4-maths.md)
// ============================================================

/// The probe of `kernel-joint-q4-maths.md` §2: `u` is an internal latent shared
/// by both components' traces, `z` is the boundary both reify against.
const SHARED_LATENT_PROBE: &str = "z  = elementof(reals)\n\
     u  ~ Normal(mu = z, sigma = 1.0)\n\
     a1 ~ Normal(mu = u, sigma = 1.0)\n\
     a2 ~ Normal(mu = u, sigma = 1.0)\n";

/// The `(%bind <name> …)` line of `src`'s FlatPIR.
fn bind_line(out: &str, name: &str) -> String {
    let needle = format!("(%bind {name} ");
    let at = out
        .find(&needle)
        .unwrap_or_else(|| panic!("binding {name} not found in:\n{out}"));
    out[at..]
        .lines()
        .next()
        .unwrap_or_else(|| panic!("binding {name} has no line in:\n{out}"))
        .to_string()
}

/// Q1: "The result's inputs are the union of the component kernels' inputs by
/// name". Disjoint signatures coexist — this is grounding-report probe p4, which
/// the union rule makes legal with inputs `{z, w}` rather than a signature
/// mismatch.
#[test]
fn kernel_joint_inputs_are_the_union_of_the_component_inputs_by_name() {
    let out = ir("z  = elementof(reals)\n\
                  w  = elementof(reals)\n\
                  b1 ~ Normal(mu = z, sigma = 1.0)\n\
                  b2 ~ Normal(mu = w, sigma = 1.0)\n\
                  K1 = kernelof(b1, z = z)\n\
                  K2 = kernelof(b2, w = w)\n\
                  KJ = joint(p = K1, q = K2)");
    assert!(
        bind_line(&out, "KJ").contains("(%kernel (%inputs z w)"),
        "disjoint component inputs must union, not conflict; got:\n{out}"
    );
}

/// Q1, the fan half of the same rule: "a name declared by several components
/// binds once in the result and every declaring component receives that one
/// value". Both spellings, so the keyword and positional arms cannot drift.
#[test]
fn kernel_joint_shared_input_name_binds_once_and_fans() {
    for spelling in ["joint(p = K1, q = K2)", "joint(K1, K2)"] {
        let out = ir(&format!(
            "{SHARED_LATENT_PROBE}\
             K1 = kernelof(a1, z = z)\n\
             K2 = kernelof(a2, z = z)\n\
             KJ = {spelling}"
        ));
        assert!(
            bind_line(&out, "KJ").contains("(%kernel (%inputs z)"),
            "a shared input name must bind once ({spelling}); got:\n{out}"
        );
    }
}

/// Q2: "The keyword form applies unchanged, producing a kernel whose output
/// variate is a record." §11's `(%kernel (%inputs …) (%mass …))` has no slot for
/// the variate, so the record surfaces on APPLICATION — the applied fan-out is a
/// measure over `record{p, q}`.
#[test]
fn applied_keyword_kernel_joint_is_a_measure_over_a_record() {
    let out = ir(&format!(
        "{SHARED_LATENT_PROBE}\
         K1 = kernelof(a1, z = z)\n\
         K2 = kernelof(a2, z = z)\n\
         KJ = joint(p = K1, q = K2)\n\
         M  = KJ(z = 0.0)"
    ));
    assert!(
        bind_line(&out, "M").contains(
            "(%measure (%domain (%record (p (%scalar real)) (q (%scalar real)))) \
             (%mass %normalized))"
        ),
        "the applied keyword fan-out is a record-variate measure; got:\n{out}"
    );
}

/// The positional counterpart: §06 forms the output variate by `cat`, so two
/// scalar components applied give a length-2 vector, not a record.
#[test]
fn applied_positional_kernel_joint_is_a_measure_over_the_cat_variate() {
    let out = ir(&format!(
        "{SHARED_LATENT_PROBE}\
         K1 = kernelof(a1, z = z)\n\
         K2 = kernelof(a2, z = z)\n\
         KJ = joint(K1, K2)\n\
         M  = KJ(z = 0.0)"
    ));
    assert!(
        bind_line(&out, "M")
            .contains("(%measure (%domain (%array 1 (2) (%scalar real))) (%mass %normalized))"),
        "the applied positional fan-out cats its component variates; got:\n{out}"
    );
}

/// Q3: "Measure components are permitted and are the nullary case: they ignore
/// the input." So a mixed `joint` is legal and the measure contributes nothing to
/// the input union — grounding-report probe p3.
#[test]
fn mixed_measure_and_kernel_joint_is_a_kernel_over_the_kernel_inputs_alone() {
    let out = ir("z  = elementof(reals)\n\
                  b1 ~ Normal(mu = z, sigma = 1.0)\n\
                  K1 = kernelof(b1, z = z)\n\
                  M2 = Normal(mu = 0.0, sigma = 2.0)\n\
                  KJ = joint(p = K1, q = M2)");
    assert!(
        bind_line(&out, "KJ").contains("(%kernel (%inputs z) (%mass %normalized))"),
        "a measure component is the nullary case, contributing no input; got:\n{out}"
    );
}

/// Q5: the mass rule is the measure case applied pointwise, so a fan-out of
/// Markov kernels is a Markov kernel (§06: "A fan-out of Markov kernels is a
/// Markov kernel"). Covered by the tests above; pinned here against the mixed
/// form too, where the fold has to read a measure and a kernel component with the
/// same rule.
#[test]
fn kernel_joint_of_normalized_components_is_markov_in_every_spelling() {
    for spelling in [
        "joint(p = K1, q = K2)",
        "joint(K1, K2)",
        "joint(p = K1, q = Normal(0.0, 1.0))",
        "joint(K1, Normal(0.0, 1.0))",
    ] {
        let out = ir(&format!(
            "{SHARED_LATENT_PROBE}\
             K1 = kernelof(a1, z = z)\n\
             K2 = kernelof(a2, z = z)\n\
             KJ = {spelling}"
        ));
        assert!(
            bind_line(&out, "KJ").contains("(%mass %normalized)"),
            "a fan-out of Markov kernels is Markov ({spelling}); got:\n{out}"
        );
    }
}

/// Q5's qualification reaches through kernels, because the fold is the SAME one
/// the measure case uses ([`joint_mass`]): two non-normalized components that may
/// share a stochastic node degrade to `%unknown` rather than folding
/// `%locallyfinite`×`%locallyfinite` (`kernel-joint-q4-maths.md` §7, the
/// Student-t/`y^2` counterexample). One non-normalized member still folds
/// exactly, which is the second half here.
///
/// Every KERNEL component is a reification, and `functionof`/`kernelof` are
/// disqualifiers of `joint_component_is_trace_clean`, so no kernel component is
/// ever provably trace-clean and 2+ non-normalized kernel components always
/// degrade. Sound, and the same disclosed conservatism the measure case carries
/// for two `lawof` components.
#[test]
fn kernel_joint_mass_degrades_on_two_nonnormalized_components_and_not_on_one() {
    let two = ir(&format!(
        "{SHARED_LATENT_PROBE}\
         L1 = functionof(Lebesgue(reals), z = z)\n\
         L2 = functionof(Lebesgue(interval(0.0, 1.0)), z = z)\n\
         KJ = joint(p = L1, q = L2)"
    ));
    assert!(
        bind_line(&two, "KJ").contains("(%mass %unknown)"),
        "two non-normalized components cannot fold to a definite class; got:\n{two}"
    );
    let one = ir(&format!(
        "{SHARED_LATENT_PROBE}\
         L1 = functionof(Lebesgue(reals), z = z)\n\
         K2 = kernelof(a2, z = z)\n\
         KJ = joint(p = L1, q = K2)"
    ));
    assert!(
        bind_line(&one, "KJ").contains("(%mass %locallyfinite)"),
        "one non-normalized member folds exactly; got:\n{one}"
    );
}

/// The Q1 consistency clause: "Components that share a stochastic node must bind
/// every boundary ancestor of that node under the same input name; a `joint`
/// whose sharing components disagree on that name is a static error." `u` is
/// shared and its boundary ancestor `z` is `s` in one component and `t` in the
/// other, so `u`'s parent has no value where `s != t`.
#[test]
fn kernel_joint_sharing_components_must_agree_on_a_shared_nodes_input_name() {
    assert!(rejects(
        &format!(
            "{SHARED_LATENT_PROBE}\
             K1 = kernelof(a1, s = z)\n\
             K2 = kernelof(a2, t = z)\n\
             KJ = joint(p = K1, q = K2)"
        ),
        "bound under different input names"
    ));
}

/// The clause is restricted to boundary ANCESTORS of the shared node, so a
/// disagreement off that ancestry is legal — widening it to any commonly-bound
/// target would reject this well-formed program. Here the shared `u` descends
/// from `w`, which both components bind as `c`; `z` is bound under `s` and `t`
/// but is not an ancestor of `u`, so the retained node's parent stays
/// single-valued and the inputs union to `{s, c, t}`.
#[test]
fn kernel_joint_name_disagreement_off_the_shared_ancestry_is_legal() {
    let src = "z  = elementof(reals)\n\
               w  = elementof(reals)\n\
               u  ~ Normal(mu = w, sigma = 1.0)\n\
               a1 ~ Normal(mu = u, sigma = 1.0)\n\
               a2 ~ Normal(mu = u, sigma = 1.0)\n\
               K1 = kernelof(a1, s = z, c = w)\n\
               K2 = kernelof(a2, t = z, c = w)\n\
               KJ = joint(p = K1, q = K2)";
    assert!(
        !rejects(src, "bound under different input names"),
        "a disagreement that is not a shared node's ancestor is legal"
    );
    assert!(
        bind_line(&ir(src), "KJ").contains("(%kernel (%inputs s c t)"),
        "and the inputs still union by name"
    );
}

/// Trace-DISJOINT components need no clause at all (`kernel-joint-q4-maths.md`
/// §4: "Trace-disjoint components need no clause"), so differing input names for
/// the same boundary node are legal when nothing is shared.
#[test]
fn trace_disjoint_kernel_joint_may_disagree_on_an_input_name() {
    let src = "z  = elementof(reals)\n\
               b1 ~ Normal(mu = z, sigma = 1.0)\n\
               b2 ~ Normal(mu = z, sigma = 1.0)\n\
               K1 = kernelof(b1, s = z)\n\
               K2 = kernelof(b2, t = z)\n\
               KJ = joint(p = K1, q = K2)";
    assert!(
        !rejects(src, "bound under different input names"),
        "disjoint traces share no node, so no name has to agree"
    );
    assert!(
        bind_line(&ir(src), "KJ").contains("(%kernel (%inputs s t)"),
        "and both names are inputs of the fan-out"
    );
}

/// The W1 shape: a MEASURE component sharing a stochastic node whose ancestor the
/// kernel component binds as a boundary input.
///
/// `kernel-joint-w1-maths.md` §3 shows the shape denotes nothing: inside `K1(a)` the
/// shared `u` carries `Normal(a, 1)` while inside `M` it carries `Normal(v, 1)` for the
/// ambient `v`, and one node cannot carry two laws. #85's clause covers it under "does
/// not bind it at all — in particular a measure component, which binds nothing".
///
/// The number the shape used to score, −2.0878770664093453, is reading A's — a
/// different question (the `joint` of `K1` and the KERNELIZATION of `M`), which the
/// next test spells explicitly and legally.
#[test]
fn a_measure_component_sharing_a_boundary_bound_ancestor_is_a_static_error() {
    let src = format!(
        "{SHARED_LATENT_PROBE}\
         K1 = kernelof(a1, z = z)\n\
         M  = lawof(u)\n\
         KJ = joint(p = K1, q = M)"
    );
    assert!(
        rejects(&src, "binds it under no name"),
        "a measure component sharing a boundary-descended node is a static error"
    );
    assert!(
        rejects(
            &src,
            "a measure component binds it under no name (measure components are nullary"
        ),
        "and the diagnostic names the MEASURE component as the non-binder, with the reason \
         AFTER the verb rather than interrupting it; got: {:?}",
        diags_of(&src)
    );
}

/// Reading E (`kernel-joint-w1-maths.md` §4): the same law is one explicit reification
/// away, and THAT is legal — both components bind the shared node's ancestor `z` under
/// the same input name.
#[test]
fn both_components_binding_the_shared_ancestor_is_legal() {
    let src = format!(
        "{SHARED_LATENT_PROBE}\
         K1 = kernelof(a1, z = z)\n\
         K2 = kernelof(u, z = z)\n\
         KJ = joint(p = K1, q = K2)"
    );
    assert!(
        !rejects(&src, "shared stochastic node"),
        "both components bind the shared ancestor under one name; got: {:?}",
        diags_of(&src)
    );
    assert!(
        bind_line(&ir(&src), "KJ").contains("(%kernel (%inputs z)"),
        "and the fan-out takes the one shared input"
    );
}

/// `kernel-joint-w1-maths.md` §5, first bullet — the CLOSED shared node. No ancestor of
/// the shared `u` is anyone's boundary, so `M` is a closed measure, Q3's "they ignore
/// the input" holds verbatim, and the shape is legal.
#[test]
fn a_measure_component_sharing_a_closed_node_is_legal() {
    let src = "z  = elementof(reals)\n\
               u  ~ Normal(mu = 0.0, sigma = 1.0)\n\
               a1 ~ Normal(mu = u + z, sigma = 1.0)\n\
               K1 = kernelof(a1, z = z)\n\
               M  = lawof(u)\n\
               KJ = joint(p = K1, q = M)";
    assert!(
        !rejects(src, "shared stochastic node"),
        "a closed shared node has no boundary-bound ancestor; got: {:?}",
        diags_of(src)
    );
    assert!(
        bind_line(&ir(src), "KJ").contains("(%kernel (%inputs z)"),
        "and the measure component contributes no input (Q3)"
    );
}

/// `kernel-joint-w1-maths.md` §5, third bullet — no shared stochastic node at all. `M`
/// is parameterized by the ambient `z` and `K1`'s reified graph holds a decoupled input,
/// so no retained node forces two views onto one parent slot.
#[test]
fn a_measure_component_sharing_no_stochastic_node_is_legal() {
    let src = "z  = elementof(reals)\n\
               u  ~ Normal(mu = z, sigma = 1.0)\n\
               a1 ~ Normal(mu = u, sigma = 1.0)\n\
               w  ~ Normal(mu = z, sigma = 2.0)\n\
               K1 = kernelof(a1, z = z)\n\
               M  = lawof(w)\n\
               KJ = joint(p = K1, q = M)";
    assert!(
        !rejects(src, "shared stochastic node"),
        "nothing is shared, so the clause does not apply; got: {:?}",
        diags_of(src)
    );
}

/// The component's OWN boundary severs its trace (§04: a boundary node "can be thought
/// of as being substituted with a new node … in the reified graph"), so `K2`, which
/// binds `u` itself, shares nothing with `K1` and must not be read as a non-binder of
/// `u`'s ancestor `z`.
///
/// Without the cut in `component_draw_nodes` this well-formed program is rejected: the
/// syntactic walk reaches `u`'s draw through `K2`'s body, `z` is an ancestor of it, and
/// `K2` declares `u` but not `z`.
#[test]
fn a_component_binding_the_shared_node_itself_shares_nothing() {
    let src = format!(
        "{SHARED_LATENT_PROBE}\
         K1 = kernelof(a1, z = z)\n\
         K2 = kernelof(a2, u = u)\n\
         KJ = joint(p = K1, q = K2)"
    );
    assert!(
        !rejects(&src, "shared stochastic node"),
        "a boundary-severed node is not shared; got: {:?}",
        diags_of(&src)
    );
    assert!(
        bind_line(&ir(&src), "KJ").contains("(%kernel (%inputs z u)"),
        "and the inputs union by name"
    );
}

/// The kernel-side analogue of W1 (`kernel-joint-w1-maths.md` §6, "sharing variant"):
/// the non-binder is a KERNEL, not a measure. One clause covers both, so the error
/// wording is the same "under no name" case.
///
/// `K2` reifies against `w`, which is not an ancestor of the shared `u`; `u`'s ancestor
/// `z` is bound by `K1` alone.
#[test]
fn a_kernel_component_binding_a_shared_ancestor_under_no_name_is_a_static_error() {
    let src = "z  = elementof(reals)\n\
               w  = elementof(reals)\n\
               u  ~ Normal(mu = z, sigma = 1.0)\n\
               a1 ~ Normal(mu = u, sigma = 1.0)\n\
               a2 ~ Normal(mu = u + w, sigma = 1.0)\n\
               K1 = kernelof(a1, z = z)\n\
               K2 = kernelof(a2, c = w)\n\
               KJ = joint(p = K1, q = K2)";
    assert!(
        rejects(src, "binds it under no name"),
        "a sharing KERNEL that binds the ancestor under no name is the same error; got: {:?}",
        diags_of(src)
    );
    assert!(
        rejects(
            src,
            "another kernel component binds it under no name (its own boundary omits that \
             ancestor)"
        ),
        "and the diagnostic must name a KERNEL non-binder, not a measure component that is \
         not there; got: {:?}",
        diags_of(src)
    );
    assert!(
        !rejects(src, "a measure component"),
        "there is no measure component in this program; got: {:?}",
        diags_of(src)
    );
}

/// An all-measure `joint` must be untouched by the kernel arm: still a measure,
/// still the record/`cat` domain, still the same mass fold.
#[test]
fn an_all_measure_joint_is_unchanged_by_the_kernel_arm() {
    let kw = ir("j = joint(a = Normal(0.0, 1.0), b = Exponential(1.0))");
    assert!(
        bind_line(&kw, "j").contains(
            "(%measure (%domain (%record (a (%scalar real)) (b (%scalar real)))) \
             (%mass %normalized))"
        ),
        "keyword measure joint unchanged; got:\n{kw}"
    );
    let pos = ir("j = joint(Lebesgue(reals), Lebesgue(reals))");
    assert!(
        bind_line(&pos, "j").contains("(%mass %locallyfinite)"),
        "positional measure joint's exact fold unchanged; got:\n{pos}"
    );
}

/// §04 "Calling conventions": "A call with field or column names that do not
/// match the callable's argument names is a static error." A fan-out kernel
/// declares its inputs in its TYPE rather than on a boundary node, and the
/// user-call arity/name check read only the boundary — so every ill-formed
/// application of one typed as a closed `%measure` with the declared input never
/// bound, and `draw(KJ())` as a concrete `%record`. Each spelling must now be
/// refused exactly as the plain-`kernelof` control is.
#[test]
fn an_ill_formed_application_of_a_fan_out_kernel_is_a_static_error() {
    let model = |app: &str| {
        format!(
            "{SHARED_LATENT_PROBE}\
             K1 = kernelof(a1, z = z)\n\
             K2 = kernelof(a2, z = z)\n\
             KJ = joint(p = K1, q = K2)\n\
             M  = {app}"
        )
    };
    assert!(
        rejects(&model("KJ()"), "`KJ` declares 1 parameter, got 0 arguments"),
        "an unbound declared input is not a closed measure"
    );
    assert!(
        rejects(
            &model("KJ(nope = 0.0)"),
            "`KJ` has no parameter `nope` (declares: `z`)"
        ),
        "an undeclared keyword is a static error"
    );
    assert!(
        rejects(
            &model("KJ(z = 0.0, extra = 1.0)"),
            "`KJ` declares 1 parameter, got 2 arguments"
        ),
        "a surplus argument is a static error"
    );
    // The well-formed application still types, and still types as the record.
    let ok = model("KJ(z = 0.0)");
    assert!(
        !rejects(&ok, "declares 1 parameter"),
        "the well-formed application must not be caught by the arity check"
    );
    assert!(
        bind_line(&ir(&ok), "M").contains("(%measure (%domain (%record"),
        "and it is still a measure over the record variate"
    );
}

/// The fold must not consume a component mass that its own type rule never set.
/// `fill_mass` returns early unless the type is `Type::Measure`, so
/// `truncate(kernelof(…), …)` keeps the base's `%normalized` where the measure
/// version reads `%finite` (a separate, carded defect — asserted here as the
/// premise, not as correct). Folding that unchanged published a wrong STRONGER
/// class: the `joint` read `%normalized` instead of Q5's `%unknown`, and `kchain`
/// carried it onto a MEASURE, past the gates an unnormalized measure must not
/// pass.
#[test]
fn a_fan_out_does_not_inherit_an_unlifted_component_mass() {
    let out = ir("z  = elementof(reals)\n\
                  a1 ~ Normal(mu = z, sigma = 1.0)\n\
                  a2 ~ Normal(mu = z, sigma = 1.0)\n\
                  T1 = truncate(kernelof(a1, z = z), interval(0.0, 5.0))\n\
                  T2 = truncate(kernelof(a2, z = z), interval(0.0, 5.0))\n\
                  KJ = joint(p = T1, q = T2)\n\
                  C  = kchain(Normal(mu = 0.0, sigma = 1.0), KJ)");
    assert!(
        bind_line(&out, "T1").contains("(%mass %normalized)"),
        "premise: the component's mass is un-lifted and reads %normalized; if \
         this line fails the `fill_mass` lift landed and this test should read \
         the true class instead:\n{out}"
    );
    assert!(
        bind_line(&out, "KJ").contains("(%mass %unknown)"),
        "the fan-out must not fold an un-lifted component class; got:\n{out}"
    );
    assert!(
        bind_line(&out, "C").contains("(%mass %unknown)"),
        "and `kchain` must not carry a claimed %normalized onto a measure; \
         got:\n{out}"
    );
}

/// The Q1 narrowing's load-bearing half. `component_draw_nodes` walks straight
/// past the reification boundary, so both components here report `w`'s draw and
/// the intersection is non-empty — yet boundary substitution SEVERS `w` and the
/// components are genuinely trace-disjoint. Only the ancestor test stops the
/// false fire: `w`'s own subtree is `Normal(0.0, 1.0)` and never reaches
/// `(%ref self z)`. Without it, a well-formed program would be rejected.
#[test]
fn a_shared_draw_upstream_of_the_boundary_does_not_fire_the_q1_error() {
    let src = "w  ~ Normal(mu = 0.0, sigma = 1.0)\n\
               z  = 2.0 * w\n\
               a1 ~ Normal(mu = z, sigma = 1.0)\n\
               a2 ~ Normal(mu = z, sigma = 1.0)\n\
               K1 = kernelof(a1, s = z)\n\
               K2 = kernelof(a2, t = z)\n\
               KJ = joint(p = K1, q = K2)";
    assert!(
        !rejects(src, "bound under different input names"),
        "a draw the boundary severs is not shared between the components"
    );
    assert!(
        bind_line(&ir(src), "KJ").contains("(%kernel (%inputs s t)"),
        "and both names are inputs of the fan-out"
    );
}

/// The `%autoinputs` boundary spelling reaches the Q1 check too: a bare
/// `kernelof(a2)` auto-traces `z` as its input name, so pairing it with an
/// explicit `s = z` is the same disagreement, and two bare `kernelof`s agree by
/// construction. Every other test here uses an explicit boundary, so this pins
/// the side-table path `input_entries` reads for `Inputs::Auto`.
#[test]
fn the_q1_error_reads_an_auto_inputs_boundary_too() {
    assert!(
        rejects(
            &format!(
                "{SHARED_LATENT_PROBE}\
                 K1 = kernelof(a1, s = z)\n\
                 K2 = kernelof(a2)\n\
                 KJ = joint(p = K1, q = K2)"
            ),
            "bound under different input names"
        ),
        "an auto-traced input name disagrees with an explicit one"
    );
    assert!(
        !rejects(
            &format!(
                "{SHARED_LATENT_PROBE}\
                 K1 = kernelof(a1)\n\
                 K2 = kernelof(a2)\n\
                 KJ = joint(p = K1, q = K2)"
            ),
            "bound under different input names"
        ),
        "two auto-traced boundaries agree by construction"
    );
}

/// `disintegrate_type` uses an EMPTY kernel-inputs list as its documented
/// don't-know sentinel ("Falls back to empty kernel inputs … when the joint isn't
/// a record measure or the selector isn't a static field-name set"). §04 forbids
/// what a genuine empty list would mean — "No callables may have nullary inputs,
/// as this would make them equivalent to known values" — so the arity check must
/// read it as "unknown", not as `want = 0`. Reading it as a declaration blamed the
/// call site for the arity of a kernel whose inputs the engine could not
/// determine, on programs that previously inferred and deferred honestly.
///
/// The element reaches the check through a `get`, which IS a local builtin call
/// with no boundary, so the fallback fires for it — the sentinel is what has to be
/// declined, not the shape.
#[test]
fn an_unresolvable_disintegrate_element_can_still_be_applied() {
    // The selector names a field the joint does not have.
    let missing = "a ~ Normal(mu = 0.0, sigma = 1.0)\n\
                   b ~ Normal(mu = a, sigma = 1.0)\n\
                   J = lawof(record(a = a, b = b))\n\
                   fk, prior = disintegrate([\"nosuchfield\"], J)\n\
                   M = fk(a = 0.5)";
    assert!(
        !rejects(missing, "declares 0 parameters"),
        "an unknown input list is not a zero-parameter declaration: {:?}",
        diags_of(missing)
    );
    assert!(
        bind_line(&ir(missing), "fk").contains("(%kernel (%inputs )"),
        "premise: the sentinel really is the empty list"
    );
    // The joint is not a record measure at all.
    let not_a_record = "a ~ Normal(mu = 0.0, sigma = 1.0)\n\
                        Jsc = lawof(a)\n\
                        fk, prior = disintegrate([\"a\"], Jsc)\n\
                        M = fk(a = 0.5)";
    assert!(
        !rejects(not_a_record, "declares 0 parameters"),
        "same sentinel through the non-record fallback: {:?}",
        diags_of(not_a_record)
    );
    // And the fix is not a blanket opt-out: a RESOLVED element is still checked.
    let resolved = "a ~ Normal(mu = 0.0, sigma = 1.0)\n\
                    b ~ Normal(mu = a, sigma = 1.0)\n\
                    J = lawof(record(a = a, b = b))\n\
                    fk, prior = disintegrate([\"b\"], J)\n\
                    M = fk(nope = 0.5)";
    assert!(
        rejects(resolved, "`fk` has no parameter `nope` (declares: `a`)"),
        "a resolved input list is a real declaration and §04 still applies"
    );
}

/// A `joint` mixing positional and keyword components is a static error. §06
/// spells two forms and no third, and reading only `named` whenever it is
/// non-empty silently DROPS every positional component — for a kernel that drops
/// its inputs from the union, contradicting #85's "the union of the component
/// kernels' inputs by name", which carries no qualification by spelling. Both
/// routes to an answer agree: rewriting the call by §06's own relabel equivalence
/// leaves a positional `joint` mixing a bare variate with a relabelled record,
/// which "Mixing shape classes is a static error" rejects; and the determiniser's
/// `lower_joint` already refuses the shape in the same words.
///
/// Before this, the drop was silent AND enforced: `KJ` typed `(%inputs w)` with
/// `z` gone, and applying it with both real inputs failed the arity check against
/// the truncated list.
#[test]
fn a_joint_mixing_positional_and_keyword_kernel_components_is_a_static_error() {
    let src = "z  = elementof(reals)\n\
               w  = elementof(reals)\n\
               b1 ~ Normal(mu = z, sigma = 1.0)\n\
               b2 ~ Normal(mu = w, sigma = 1.0)\n\
               K1 = kernelof(b1, z = z)\n\
               K2 = kernelof(b2, w = w)\n\
               KJ = joint(K1, q = K2)\n\
               M  = KJ(z = 0.0, w = 0.0)";
    assert!(
        rejects(src, "mixes positional and keyword components"),
        "the mixed spelling is refused on its own terms: {:?}",
        diags_of(src)
    );
    assert!(
        !rejects(src, "declares 1 parameter, got 2 arguments"),
        "and NOT reported as a call-site arity error against the dropped list: {:?}",
        diags_of(src)
    );
}

/// The MEASURE arms of `joint` drop a positional component the same way the
/// kernel arm did before its fix: reading only `named` whenever it is
/// non-empty silently drops every positional component. Before this,
/// `joint(Normal(0.0, 1.0), b = Exponential(1.0))` typed over `record{b}`
/// alone, with `Normal(0.0, 1.0)` gone and no diagnostic.
#[test]
fn a_joint_mixing_positional_and_keyword_measure_components_is_a_static_error() {
    let src = "j = joint(Normal(0.0, 1.0), b = Exponential(1.0))";
    assert!(
        rejects(src, "mixes positional and keyword components"),
        "the mixed spelling is refused on its own terms: {:?}",
        diags_of(src)
    );
    let out = ir(src);
    assert!(
        !out.contains("(%record (b"),
        "must not silently type over the record with the positional dropped; got:\n{out}"
    );
}

/// `superpose`'s argument must be a measure (spec §06: measure addition).
/// Before this, `fresh_measure`'s `Some(other) => other.clone()` arm passed a
/// non-measure argument straight through unchanged: `superpose(record(m1 =
/// n1, m2 = n2))` typed as `(%record (m1 (%measure …)) (m2 (%measure …)))` —
/// a record where a measure belongs — with no diagnostic.
#[test]
fn superpose_of_a_record_is_rejected() {
    let src = "n1 = Normal(0.0, 1.0)\n\
               n2 = Exponential(1.0)\n\
               s = superpose(record(m1 = n1, m2 = n2))";
    assert!(
        rejects(src, "must be a measure"),
        "superpose(record(...)) must be a located static error, got: {:?}",
        diags_of(src)
    );
    let out = ir(src);
    assert!(
        !out.contains("(%bind s (%meta ((%record"),
        "must not silently type `s` as a record; got:\n{out}"
    );
}

/// The same check applies to EVERY `superpose` argument, not just the first.
/// Before this, a bad argument in a later position was silently DROPPED from
/// the type entirely — worse than position 0's silent pass-through:
/// `superpose(n, record(m1 = n1, m2 = n2))` typed as `n`'s own plain measure
/// (`%unknown` mass), the record gone, with no diagnostic.
#[test]
fn superpose_of_a_measure_and_a_record_is_rejected() {
    let src = "n1 = Normal(0.0, 1.0)\n\
               n2 = Exponential(1.0)\n\
               s = superpose(n1, record(m1 = n1, m2 = n2))";
    assert!(
        rejects(src, "must be a measure"),
        "a bad argument in position 2 must be a located static error, got: {:?}",
        diags_of(src)
    );
    let out = ir(src);
    assert!(
        !out.contains("(%bind s (%meta ((%measure (%domain (%scalar real)) (%mass %unknown))"),
        "must not silently drop the bad argument and type `s` as n1's plain measure; got:\n{out}"
    );
}

// ============================================================
// `iid` over a record-valued measure has a `%table` variate (spec §11)
// ============================================================

/// A SCALAR-size `iid` over a record-valued measure types `(%table (%columns
/// …) (%nrows N))`, not `(%array 1 (N) (%record …))` — design PR #83 (owner
/// ruling, decisions-log 2026-08-18): "the text is correct (§11 gives `%table`
/// its own form; js conforms); rust types `%array` instead." §11's `%table`
/// form and §03's Cartesian power ("the power is the set of tables" for a
/// record set) both back the reading; flatppl-js already conforms
/// (`designpass2-report.md`, "PR 83"). The value-set slot already agreed
/// (`cartpow (record …) 5`, an n-row table per §03) — only the TYPE slot was
/// wrong, so this pins that the two slots now agree on the shape.
#[test]
fn scalar_iid_over_a_record_measure_types_a_table() {
    let src = "M = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Beta(alpha = 1.0, beta = 1.0))\n\
               q = iid(M, 5)";
    assert!(diags_of(src).is_empty(), "{:?}", diags_of(src));
    let out = ir(src);
    let q = out.lines().find(|l| l.contains("%bind q")).unwrap_or("");
    assert!(
        q.contains(
            "(%measure (%domain (%table (%columns (a (%scalar real)) (b (%scalar real))) \
             (%nrows 5)))"
        ),
        "scalar record-iid must type a %table, not an array of records:\n{out}"
    );
    assert!(
        !q.contains("(%array"),
        "must not also carry the old %array-of-records spelling:\n{out}"
    );

    // The annotated FlatPIR round-trips through a re-read (§11 form is legal to
    // parse back, not just to print).
    let reread = flatppl_flatpir::read(&out)
        .unwrap_or_else(|e| panic!("annotated output unreadable: {e}\n{out}"));
    assert_eq!(
        flatppl_flatpir::write(&reread),
        out,
        "annotated FlatPIR is not a write fixpoint"
    );
}

/// The MULTI-axis case (`iid(M, [2, 3])`) has no `%table` reading — a table
/// has one row axis (§03: "a multi-axis power of a record set has no table
/// reading at all") — and stays array-of-records, untouched by #83's fix.
#[test]
fn multi_axis_iid_over_a_record_measure_stays_array_of_records() {
    let src = "M = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Beta(alpha = 1.0, beta = 1.0))\n\
               q = iid(M, [2, 3])";
    assert!(diags_of(src).is_empty(), "{:?}", diags_of(src));
    let out = ir(src);
    let q = out.lines().find(|l| l.contains("%bind q")).unwrap_or("");
    assert!(
        q.contains(
            "(%measure (%domain (%array 2 (2 3) (%record (a (%scalar real)) (b (%scalar real))))"
        ),
        "multi-axis record-iid must stay an array of records:\n{out}"
    );
    assert!(
        !q.contains("%table"),
        "must not gain a %table reading:\n{out}"
    );
}

/// A record whose SCALAR count is not statically known still gets the
/// `%table` shape (a `%dynamic` row count), not the array reading — the
/// table/array choice is driven by the count's RANK, which is always 1 for a
/// scalar count regardless of whether its value is fixed (see `count_dims`).
#[test]
fn scalar_iid_over_a_record_measure_with_a_dynamic_count_still_tables() {
    let src = "n = elementof(posintegers)\n\
               M = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Beta(alpha = 1.0, beta = 1.0))\n\
               q = iid(M, n)";
    assert!(diags_of(src).is_empty(), "{:?}", diags_of(src));
    let out = ir(src);
    let q = out.lines().find(|l| l.contains("%bind q")).unwrap_or("");
    assert!(
        q.contains("(%table (%columns (a (%scalar real)) (b (%scalar real))) (%nrows %dynamic))"),
        "a dynamic scalar count must still type a %table with a %dynamic row count:\n{out}"
    );
}
