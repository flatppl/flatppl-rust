//! Closed-form normalizer for a POLYNOMIAL density over a bounded interval:
//!
//! ```text
//! normalize(truncate(weighted(x -> polynomial(c, x), Lebesgue(reals)), interval(lo, hi)))
//! ```
//!
//! This is the shape a HS3 `polynomial_dist` imports to. §07 *Approximation
//! functions* fixes the coefficient order — `polynomial(c, x)` is
//! `Σ_{i=0}^{n-1} c_{i+1} x^i`, "the first element is the constant term" — so
//! `Z = ∫_lo^hi p(x) dx = Σ_i c_{i+1} (hi^{i+1} − lo^{i+1}) / (i+1)`.
//!
//! Every `Z` asserted here is integrated by hand as an exact rational, in the
//! test that asserts it. The engines are not the oracle.

mod common;

use flatppl_determinizer::{LoweringOptions, determinize, determinize_with_options, is_flatpdl};

fn determinize_pir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diagnostics = flatppl_infer::infer(&mut m);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let out = determinize(&m).expect("must lower, not refuse");
    let pir = flatppl_flatpir::write(&out);
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl:\n{pir}");
    pir
}

fn refusal(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diagnostics = flatppl_infer::infer(&mut m);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let err = determinize(&m).expect_err("must refuse");
    format!("{err:?}")
}

fn normalizer_mass(pir: &str) -> f64 {
    let score = common::pir_binding(pir, "lp");
    let log_mass = common::call_arg(&score, "isfinite", 0);
    common::call_arg(&log_mass, "log", 0).parse().unwrap()
}

/// The rf203_ranges shape. `∫_{-10}^{10} (1 + a x) dx = [x + a x²/2] = 20 + 0`,
/// so `Z = 20` exactly and the free coefficient `a` drops out — its odd power
/// integrates to zero over a symmetric interval.
#[test]
fn linear_polynomial_over_symmetric_interval() {
    let pir = determinize_pir(
        "\
a = 0.04
px = normalize(
  truncate(weighted((x,p) -> polynomial([1.0, p], x), Lebesgue(reals), a), interval(-10.0, 10.0)),
)
lp = logdensityof(px, 0.5)",
    );
    assert_eq!(normalizer_mass(&pir), 20.0);
    assert!(
        !pir.contains("totalmass"),
        "totalmass is not FlatPDL:\n{pir}"
    );
    // The numerator keeps the weight applied at the variate and the truncation
    // gate, so the density is that lowering minus log Z.
    assert!(pir.contains("polynomial"), "weight applied at v:\n{pir}");
    assert!(pir.contains("(sub "), "density − log Z:\n{pir}");
}

/// A cubic with all four powers present, on an asymmetric interval, so every
/// per-degree factor is exercised and none cancels.
///
/// `p(x) = 1 + x/2 − x²/4 + x³/8` on `[-2, 3]`, integrated by hand as an exact
/// rational:
///
/// ```text
/// ∫ p = [x + x²/4 − x³/12 + x⁴/32]
///     = (3 + 9/4 − 27/12 + 81/32) − (−2 + 4/4 + 8/12 + 16/32)
///     = 531/96 − 16/96 = 515/96
/// ```
///
/// The term-wise power rule must reproduce that:
/// `1·5 + 0.5·2.5 − 0.25·(35/3) + 0.125·(65/4) = 515/96`.
#[test]
fn cubic_polynomial_over_asymmetric_interval() {
    let exact = 515.0_f64 / 96.0;
    let folded = 5.0 + 0.5 * 2.5 + -0.25 * (35.0 / 3.0) + 0.125 * (65.0 / 4.0);
    assert!(
        (folded - exact).abs() <= 4.0 * f64::EPSILON * exact,
        "the term-wise power rule must equal the exact integral 515/96: {folded} vs {exact}"
    );
    for weight in [
        "weighted(x -> polynomial([1.0, 0.5, -0.25, 0.125], x), Lebesgue(reals))",
        "weighted((x,p) -> polynomial([1.0, p, -0.25, 0.125], x), Lebesgue(reals), 0.5)",
    ] {
        let pir = determinize_pir(&format!(
            "\
px = normalize(
  truncate(
    {weight},
    interval(-2.0, 3.0),
  ),
)
lp = logdensityof(px, 1.0)",
        ));
        assert!(
            pir.contains(&folded.to_string()),
            "Z = 515/96 = {folded}:\n{pir}"
        );
    }
}

/// A polynomial's mass over an unbounded interval is infinite, so §06's
/// "Z … finite and nonzero" precondition fails — refuse, with a message naming
/// the endpoints rather than the generic no-closed-form-mass one.
#[test]
fn unbounded_interval_refuses_on_the_endpoints() {
    let msg = refusal(
        "\
a = elementof(reals)
px = normalize(
  truncate(weighted((x,p) -> polynomial([1.0, p], x), Lebesgue(reals), a), interval(-inf, inf)),
)
lp = logdensityof(px, 0.5)",
    );
    assert!(
        msg.contains("both interval endpoints static and finite"),
        "endpoint-specific refusal:\n{msg}"
    );
}

/// The alias fix and the mass rule compose: the HS3 importer binds the pdf to
/// `__M__` by name, so the rule has to be reached through a bare-name alias too.
#[test]
fn mass_rule_is_reached_through_an_alias() {
    let pir = determinize_pir(
        "\
a = 0.04
px = normalize(
  truncate(weighted((x,p) -> polynomial([1.0, p], x), Lebesgue(reals), a), interval(-10.0, 10.0)),
)
mm = px
lp = logdensityof(mm, 0.5)",
    );
    assert_eq!(normalizer_mass(&pir), 20.0);
}

#[test]
fn bounded_reference_and_truncation_have_the_same_polynomial_mass() {
    // Integral of 1 + 2x on [0, 1] is 2, including a wider outer truncation.
    for intensity in [
        "weighted(x -> polynomial([1.0, 2.0], x), Lebesgue(interval(0, 1)))",
        "truncate(weighted(x -> polynomial([1.0, 2.0], x), Lebesgue(reals)), interval(0, 1))",
        "truncate(weighted(x -> polynomial([1.0, 2.0], x), Lebesgue(interval(0, 1))), interval(-1, 2))",
    ] {
        let pir = determinize_pir(&format!(
            "I = {intensity}\nM = normalize(I)\nlp = logdensityof(M, 0.5)\nprocess = logdensityof(PoissonProcess(I), [0.5])\n"
        ));
        let score = common::pir_binding(&pir, "lp");
        assert!(score.contains("(log 2.0)"), "{score}");
        assert!(!pir.contains("%log_integral"), "{pir}");
    }
}

#[test]
fn polynomial_mass_intersects_all_supports() {
    // Integral of 1 + 2x on [1, 3] is 10.
    let pir = determinize_pir(
        "I = truncate(truncate(weighted(x -> polynomial([1.0, 2.0], x), Lebesgue(interval(0, 4))), interval(-1, 3)), interval(1, 5))\nlp = logdensityof(normalize(I), 1.5)",
    );
    assert_eq!(normalizer_mass(&pir), 10.0);
}

#[test]
fn reified_coefficient_aliases_remain_inside_the_integral() {
    // f(t) = 2t, so normalization cannot depend on the ambient query input x.
    let mut module = flatppl_syntax::parse(
        "x = elementof(reals)\na = x\nf = functionof(polynomial([a, 1.0], x), x=x)\nI = weighted(f, Lebesgue(interval(0, 1)))\nlp = logdensityof(normalize(I), 0.5)\ninputs = x\noutputs = lp",
    )
    .unwrap();
    assert!(flatppl_infer::infer(&mut module).is_empty());
    let output = determinize_with_options(
        &module,
        &flatppl_infer::ModuleBundle::new(),
        None,
        &LoweringOptions {
            numerical_integrals: true,
            ..LoweringOptions::default()
        },
    )
    .unwrap();
    let pir = flatppl_flatpir::write(&output);
    assert!(
        !common::pir_binding(&pir, "lp").contains("(%ref self x)"),
        "the normalizer must integrate the aliased coordinate:\n{pir}"
    );
}

#[test]
fn extra_argument_keeps_the_ambient_value_of_the_boundary_name() {
    let pir = determinize_pir(
        "x = 1.0\na = elementof(reals)\nf = functionof(polynomial([a, 1.0], x), x=x, a=a)\nI = weighted(f, Lebesgue(interval(0, 1)), x)\nlp = logdensityof(normalize(I), 0.25)",
    );
    // The supplied coefficient is x = 1, independently of f's first argument.
    assert_eq!(normalizer_mass(&pir), 1.5);
}

#[test]
fn polynomial_mass_avoids_endpoint_power_overflow_and_cancellation() {
    for (coefficients, lo, hi, point, mass) in [
        ("[0.0, 1e-200]", "1e200", "2e200", "1.5e200", 1.5e200_f64),
        ("[0.0, 1e300]", "0", "1e-200", "5e-201", 5e-101),
        ("[0.0, 1e-16]", "1e16", "10000000000000002", "1e16", 2.0),
        (
            "[1e16, -1.0]",
            "9999999999999998",
            "1e16",
            "9999999999999998",
            2.0,
        ),
    ] {
        let pir = determinize_pir(&format!(
            "I = weighted(x -> polynomial({coefficients}, x), Lebesgue(interval({lo}, {hi})))\nlp = logdensityof(normalize(I), {point})"
        ));
        let actual = normalizer_mass(&pir);
        assert!(
            (actual - mass).abs() <= 4.0 * f64::EPSILON * mass,
            "{actual} vs {mass}"
        );
    }
}
