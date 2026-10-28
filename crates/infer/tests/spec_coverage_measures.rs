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

/// Matrix distributions retain dynamic dimensions without a resolved size.
#[test]
fn matrix_dists_domain_is_rank2_dynamic_real() {
    let expected_domain = "(%array 2 (%dynamic %dynamic) (%scalar real))";

    let cases = [
        ("Wishart(3.0, eye(3))", "Wishart"),
        ("InverseWishart(3.0, eye(3))", "InverseWishart"),
        ("LKJ(external(posintegers), 1.0)", "LKJ"),
        ("LKJCholesky(external(posintegers), 1.0)", "LKJCholesky"),
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
// Both reification operations take value nodes.

/// Diagnostics from a full infer pass, as `Debug` strings.
fn diags_of(src: &str) -> Vec<String> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m).iter().map(|d| format!("{d:?}")).collect()
}

fn rejects(src: &str, needle: &str) -> bool {
    diags_of(src).iter().any(|d| d.contains(needle))
}

#[test]
fn lawof_requires_a_value_node() {
    for source in [
        "M = Normal(0, 1)\nz = lawof(M)",
        "K(x) = Normal(x, 1)\nz = lawof(K)",
        "f(x) = x + 1\nz = lawof(f)",
        "z = lawof((1, 2))",
    ] {
        let mut module = flatppl_syntax::parse(source).unwrap();
        assert!(
            infer(&mut module)
                .iter()
                .any(|d| d.severity == flatppl_infer::Severity::Error)
        );
    }
    assert!(diags_of("z = lawof(draw(Normal(0, 1)))").is_empty());
}

/// §04: `kernelof` "reifies (typically stochastic) value nodes"; "`x` must not be
/// a measure". Both measure-layer argument shapes are rejected, and the
/// diagnostics differ so a user with a kernel is not told they have a measure.
#[test]
fn kernelof_rejects_a_measure_layer_argument() {
    for source in ["f(x)=x*x\nbad=kernelof(f)", "bad=kernelof((1,2))"] {
        assert!(rejects(source, "non-value object"));
    }
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
        "requires a value node"
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

// ============================================================
// `draw` requires a probability measure (owner ruling; spec PR to follow)
// ============================================================
//
// No implicit normalization: drawing from a measure whose mass is not a
// probability is a static error, and `normalize(m)` is the escape. Derived from
// Draw mass is checked by `unprovable_normalization`.

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
            "the complement pattern with a parametric weight",
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
y = draw(truncate(lawof(draw(Normal(mu = a, sigma = 1.0))), interval(0.0, 3.0)))
lp = logdensityof(lawof(y), 0.5)";
    assert!(rejects(src, "`draw` requires a probability measure"));
    // And the explicit escape clears it.
    let fixed = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(normalize(truncate(lawof(draw(Normal(mu = a, sigma = 1.0))), interval(0.0, 3.0))))
lp = logdensityof(lawof(y), 0.5)";
    assert!(diags_of(fixed).is_empty(), "{:?}", diags_of(fixed));
}

// ============================================================
// `rand` requires a probability measure (#425 class; mirrors the `draw` gate)
// ============================================================
//
// §07: `rand(m, rstate)` draws a variate from a normalized measure `m`. Same
// owner ruling as `draw`, same `unprovable_normalization` gate, adapted to
// `rand`'s first argument (the second is the rngstate).

/// The unprovable classes are rejected and the message names the mass class
/// and the `normalize(...)` escape, exactly like the `draw` gate.
#[test]
fn rand_rejects_a_measure_that_is_not_a_probability() {
    let src = "s = rnginit(0)\ny = rand(Lebesgue(support = reals), s)";
    assert!(rejects(src, "`rand` requires a probability measure"));
    assert!(rejects(src, "total mass is `%locallyfinite`"));
    assert!(rejects(src, "normalize(...)"));
}

/// The ordinary spelling — `rand` of a plain probability distribution — still
/// infers the `(domain, %rngstate)` tuple untouched by the gate.
#[test]
fn rand_still_accepts_a_probability_measure() {
    let out = ir("s = rnginit(0)\ny = rand(Normal(0.0, 1.0), s)");
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
               y = rand(normalize(truncate(Lebesgue(support = reals), interval(0.0, 1.0))), s)";
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
    let src = "s = rnginit(0)\ny = rand(Normal(0.0, 1.0), s)";
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
// §06 `joint` product mass
// ============================================================

/// A product of probability measures has mass one in either arity spelling.
#[test]
fn joint_mass_positional_all_normalized_stays_normalized() {
    let src = "m = joint(Normal(0.0, 1.0), Normal(1.0, 2.0))";
    let out = ir(src);
    assert!(
        out.contains("(%mass %normalized)"),
        "positional all-normalized joint must be %normalized; got:\n{out}"
    );
}

/// Each law absorbs its own copy of `mu`. Truncating both probability laws
/// gives finite components, whose independent product is still finite.
#[test]
fn joint_mass_two_truncated_reified_laws_is_finite() {
    let src = "mu ~ Normal(0.0, 1.0)\n\
               z1 ~ Normal(mu, 1.0)\n\
               z2 ~ Normal(mu, 1.0)\n\
               lz1 = lawof(z1)\n\
               lz2 = lawof(z2)\n\
               j = joint(truncate(lz1, interval(0.0, 5.0)), truncate(lz2, interval(0.0, 5.0)))";
    assert!(diags_of(src).is_empty(), "{:?}", diags_of(src));
    let out = ir(src);
    assert!(
        bind_line(&out, "j").contains("(%mass %finite)"),
        "the product of two truncated laws must be finite; got:\n{out}"
    );
}

/// A finite component times a probability component remains finite.
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

/// A product of two Lebesgue measures is locally finite, so it cannot be drawn
/// as a probability measure.
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
        "s = rnginit(0)\ny = rand(joint(Lebesgue(reals), Lebesgue(reals)), s)",
        "requires a probability measure"
    ));
}

/// The exact locally finite product class must reach the normalization gate.
/// Widening it to unknown would incorrectly permit normalization.
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

/// Shared free constructor parameters do not weaken the product mass class.
/// The normalization gate still rejects this parametric Lebesgue product.
#[test]
fn normalize_of_elementof_parametric_joint_is_a_static_error() {
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

// ============================================================
// §06 `joint` over kernel components — the fan-out kernel
// ============================================================

/// Each kernel absorbs its own copy of `u`; `z` supplies a boundary input.
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

/// The result's inputs are the union of the component kernels' inputs by name.
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

/// A name declared by several components binds once and is supplied to each
/// component. Check both keyword and positional products.
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

/// The keyword product has a record variate. §11's kernel type has no slot for
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

/// A measure component contributes no declared inputs to a mixed product.
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

/// The mass rule is the measure case applied pointwise, so a fan-out of
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

/// Locally finite times finite, or times normalized, remains locally finite.
/// Kernel reification does not change the independent-product mass rule.
#[test]
fn kernel_joint_mass_is_the_pointwise_product_class() {
    let two = ir(&format!(
        "{SHARED_LATENT_PROBE}\
         L1 = functionof(Lebesgue(reals), z = z)\n\
         L2 = functionof(Lebesgue(interval(0.0, 1.0)), z = z)\n\
         KJ = joint(p = L1, q = L2)\n\
         M = KJ(z = 0.0)"
    ));
    assert!(
        bind_line(&two, "KJ").contains("(%mass %locallyfinite)"),
        "locally finite times finite must remain locally finite; got:\n{two}"
    );
    assert!(
        bind_line(&two, "M").contains("(%mass %locallyfinite)"),
        "applying the product kernel must preserve its mass class; got:\n{two}"
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

/// Each component owns its boundary substitution even when both source graphs
/// reach `u`. The two public input names survive in the product interface.
#[test]
fn kernel_joint_components_keep_distinct_names_for_the_same_boundary_source() {
    let src = format!(
        "{SHARED_LATENT_PROBE}\
         K1 = kernelof(a1, s = z)\n\
         K2 = kernelof(a2, t = z)\n\
         KJ = joint(p = K1, q = K2)\n\
         M = KJ(s = 0.0, t = 2.0)"
    );
    assert!(diags_of(&src).is_empty(), "{:?}", diags_of(&src));
    let out = ir(&src);
    assert!(
        bind_line(&out, "KJ").contains("(%kernel (%inputs s t) (%mass %normalized))"),
        "both component inputs must survive; got:\n{out}"
    );
    assert!(
        bind_line(&out, "M").contains(
            "(%measure (%domain (%record (p (%scalar real)) (q (%scalar real)))) \
             (%mass %normalized))"
        ),
        "a fully applied product must be a normalized record measure; got:\n{out}"
    );
}

/// Unused declared inputs remain in the interface, and a common input name is
/// included only once. Here `s` and `t` are unused while both kernels use `c`.
#[test]
fn kernel_joint_keeps_unused_inputs_and_deduplicates_common_names() {
    let src = "z  = elementof(reals)\n\
               w  = elementof(reals)\n\
               u  ~ Normal(mu = w, sigma = 1.0)\n\
               a1 ~ Normal(mu = u, sigma = 1.0)\n\
               a2 ~ Normal(mu = u, sigma = 1.0)\n\
               K1 = kernelof(a1, s = z, c = w)\n\
               K2 = kernelof(a2, t = z, c = w)\n\
               KJ = joint(p = K1, q = K2)";
    assert!(diags_of(src).is_empty(), "{:?}", diags_of(src));
    assert!(
        bind_line(&ir(src), "KJ").contains("(%kernel (%inputs s c t)"),
        "and the inputs still union by name"
    );
}

/// Distinct components may expose the same source parameter under distinct names.
#[test]
fn kernel_joint_unions_distinct_names_for_a_common_source_parameter() {
    let src = "z  = elementof(reals)\n\
               b1 ~ Normal(mu = z, sigma = 1.0)\n\
               b2 ~ Normal(mu = z, sigma = 1.0)\n\
               K1 = kernelof(b1, s = z)\n\
               K2 = kernelof(b2, t = z)\n\
               KJ = joint(p = K1, q = K2)";
    assert!(diags_of(src).is_empty(), "{:?}", diags_of(src));
    assert!(
        bind_line(&ir(src), "KJ").contains("(%kernel (%inputs s t)"),
        "and both names are inputs of the fan-out"
    );
}

/// Reifying outputs at different depths can still yield the same input signature.
#[test]
fn kernel_components_with_different_outputs_share_one_declared_input() {
    let src = format!(
        "{SHARED_LATENT_PROBE}\
         K1 = kernelof(a1, z = z)\n\
         K2 = kernelof(u, z = z)\n\
         KJ = joint(p = K1, q = K2)"
    );
    assert!(diags_of(&src).is_empty(), "{:?}", diags_of(&src));
    assert!(
        bind_line(&ir(&src), "KJ").contains("(%kernel (%inputs z)"),
        "and the fan-out takes the one shared input"
    );
}

/// A closed reified law contributes no inputs to a mixed kernel product.
#[test]
fn a_closed_reified_measure_adds_no_kernel_inputs() {
    let src = "z  = elementof(reals)\n\
               u  ~ Normal(mu = 0.0, sigma = 1.0)\n\
               a1 ~ Normal(mu = u + z, sigma = 1.0)\n\
               K1 = kernelof(a1, z = z)\n\
               M  = lawof(u)\n\
               KJ = joint(p = K1, q = M)";
    assert!(diags_of(src).is_empty(), "{:?}", diags_of(src));
    assert!(
        bind_line(&ir(src), "KJ").contains("(%kernel (%inputs z)"),
        "and the measure component contributes no input"
    );
}

/// Components can cut their source graphs at different depths. Both boundary
/// names remain available to the product's caller.
#[test]
fn kernel_joint_keeps_boundaries_at_different_source_depths() {
    let src = format!(
        "{SHARED_LATENT_PROBE}\
         K1 = kernelof(a1, z = z)\n\
         K2 = kernelof(a2, u = u)\n\
         KJ = joint(p = K1, q = K2)"
    );
    assert!(diags_of(&src).is_empty(), "{:?}", diags_of(&src));
    assert!(
        bind_line(&ir(&src), "KJ").contains("(%kernel (%inputs z u)"),
        "and the inputs union by name"
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
        flatppl_syntax::parse(&model("KJ()")).is_err(),
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
/// class: the `joint` read `%normalized` instead of `%unknown`, and `kchain`
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
                  C  = kchain(joint(z = Normal(mu = 0.0, sigma = 1.0)), KJ)");
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

/// A derived stochastic value can be a component boundary. Substitution cuts
/// its ancestors, and each component keeps its own public input name.
#[test]
fn kernel_joint_accepts_distinct_inputs_for_a_derived_stochastic_boundary() {
    let src = "w  ~ Normal(mu = 0.0, sigma = 1.0)\n\
               z  = 2.0 * w\n\
               a1 ~ Normal(mu = z, sigma = 1.0)\n\
               a2 ~ Normal(mu = z, sigma = 1.0)\n\
               K1 = kernelof(a1, s = z)\n\
               K2 = kernelof(a2, t = z)\n\
               KJ = joint(p = K1, q = K2)";
    assert!(diags_of(src).is_empty(), "{:?}", diags_of(src));
    assert!(
        bind_line(&ir(src), "KJ").contains("(%kernel (%inputs s t)"),
        "and both names are inputs of the fan-out"
    );
}

/// Auto-discovered and explicit input names participate in the same union.
/// Distinct names remain distinct; matching auto-input names appear once.
#[test]
fn kernel_joint_unions_auto_and_explicit_inputs() {
    for (first, inputs, application) in [
        ("kernelof(a1, s = z)", "s z", "KJ(s = 0.0, z = 2.0)"),
        ("kernelof(a1)", "z", "KJ(z = 2.0)"),
    ] {
        let src = format!(
            "{SHARED_LATENT_PROBE}\
             K1 = {first}\n\
             K2 = kernelof(a2)\n\
             KJ = joint(p = K1, q = K2)\n\
             M = {application}"
        );
        assert!(diags_of(&src).is_empty(), "{:?}", diags_of(&src));
        let out = ir(&src);
        assert!(
            bind_line(&out, "KJ")
                .contains(&format!("(%kernel (%inputs {inputs}) (%mass %normalized))")),
            "component input names must union; got:\n{out}"
        );
        assert!(
            bind_line(&out, "M").contains(
                "(%measure (%domain (%record (p (%scalar real)) (q (%scalar real)))) \
                 (%mass %normalized))"
            ),
            "all declared inputs are bound in the applied record measure; got:\n{out}"
        );
    }
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
        rejects(src, "either positional or keyword"),
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
        rejects(src, "either positional or keyword"),
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

/// A scalar count DERIVED to 0 over a record-valued measure is an error, not
/// a `(%table … (%nrows 0))` — a table has no zero-row form (spec §06 iid,
/// PR #83 amendment a131108). §06 already requires a positive WRITTEN size,
/// so a zero here only ever comes from a derived size the const-eval table
/// resolved (here `lengthof(xs) - lengthof(xs)`), the analysis-time
/// counterpart of a size that resolves to 0 at data-load time.
///
/// The derived size types `%dynamic`, not `(%nrows 0)` — the empty-arrays
/// ruling's sub-ruling 3 — so the refusal reads the resolved value rather
/// than the dim. The fixture used `lengthof(zeros(0))`, which the written-size
/// positivity rule now refuses first for its literal `0`.
#[test]
fn scalar_iid_over_a_record_measure_with_a_derived_zero_count_is_refused() {
    let src = "xs = [1.0, 2.0]\n\
               n = sub(lengthof(xs), lengthof(xs))\n\
               M = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Beta(alpha = 1.0, beta = 1.0))\n\
               q = iid(M, n)";
    assert!(
        rejects(src, "§06 iid"),
        "a derived-zero record-iid size must be refused citing spec §06 iid: {:?}",
        diags_of(src)
    );
    let out = ir(src);
    assert!(
        !out.contains("%nrows 0"),
        "must not type a zero-row table:\n{out}"
    );
}
