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

use flatppl_determinizer::{determinize, is_flatpdl};

fn determinize_pir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let out = determinize(&m).expect("must lower, not refuse");
    let pir = flatppl_flatpir::write(&out);
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl:\n{pir}");
    pir
}

fn refusal(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let err = determinize(&m).expect_err("must refuse");
    format!("{err:?}")
}

/// The rf203_ranges shape. `∫_{-10}^{10} (1 + a x) dx = [x + a x²/2] = 20 + 0`,
/// so `Z = 20` exactly and the free coefficient `a` drops out — its odd power
/// integrates to zero over a symmetric interval.
#[test]
fn linear_polynomial_over_symmetric_interval() {
    let pir = determinize_pir(
        "\
a = elementof(reals)
px = normalize(
  truncate(weighted(x -> polynomial([1.0, a], x), Lebesgue(reals)), interval(-10.0, 10.0)),
)
lp = logdensityof(px, 0.5)",
    );
    // Z = 1·20 + a·0.
    assert!(
        pir.contains("(add 20.0 ") && pir.contains("(mul (%ref self a) 0.0)"),
        "Z = 20 + a·0:\n{pir}"
    );
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
    let pir = determinize_pir(
        "\
px = normalize(
  truncate(
    weighted(x -> polynomial([1.0, 0.5, -0.25, 0.125], x), Lebesgue(reals)),
    interval(-2.0, 3.0),
  ),
)
lp = logdensityof(px, 1.0)",
    );
    let exact = 515.0_f64 / 96.0;
    let folded = 5.0 + 0.5 * 2.5 + -0.25 * (35.0 / 3.0) + 0.125 * (65.0 / 4.0);
    assert!(
        (folded - exact).abs() <= 4.0 * f64::EPSILON * exact,
        "the term-wise power rule must equal the exact integral 515/96: {folded} vs {exact}"
    );
    assert!(
        pir.contains(&folded.to_string()),
        "Z = 515/96 = {folded}:\n{pir}"
    );
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
  truncate(weighted(x -> polynomial([1.0, a], x), Lebesgue(reals)), interval(-inf, inf)),
)
lp = logdensityof(px, 0.5)",
    );
    assert!(
        msg.contains("both interval endpoints static and finite"),
        "endpoint-specific refusal:\n{msg}"
    );
}

/// A coefficient vector whose LENGTH is not static leaves the polynomial's degree
/// unknown, so the antiderivative is not statically expressible. Refuse rather
/// than guess a degree.
#[test]
fn dynamic_coefficient_vector_refuses() {
    let msg = refusal(
        "\
c = elementof(cartpow(reals, 3))
px = normalize(
  truncate(weighted(x -> polynomial(c, x), Lebesgue(reals)), interval(-1.0, 1.0)),
)
lp = logdensityof(px, 0.5)",
    );
    assert!(
        msg.contains("closed-form mass rule"),
        "generic unnormalized-measure refusal:\n{msg}"
    );
}

/// The polynomial must be applied to the weight's OWN argument. A polynomial of
/// something else is a constant weight in the variate, not this rule's shape, and
/// its mass is a different integral.
#[test]
fn polynomial_of_another_quantity_refuses() {
    let msg = refusal(
        "\
a = elementof(reals)
px = normalize(
  truncate(weighted(x -> polynomial([1.0, 2.0], a), Lebesgue(reals)), interval(-1.0, 1.0)),
)
lp = logdensityof(px, 0.5)",
    );
    assert!(
        msg.contains("closed-form mass rule"),
        "generic unnormalized-measure refusal:\n{msg}"
    );
}

/// The reference measure must be Lebesgue over the whole line, so the truncation
/// set alone is the domain of integration. A `Lebesgue(interval(...))` base makes
/// it the intersection instead.
#[test]
fn bounded_lebesgue_reference_refuses() {
    let msg = refusal(
        "\
px = normalize(
  truncate(
    weighted(x -> polynomial([1.0, 2.0], x), Lebesgue(interval(0.0, 5.0))),
    interval(-1.0, 1.0),
  ),
)
lp = logdensityof(px, 0.5)",
    );
    assert!(
        msg.contains("closed-form mass rule"),
        "generic unnormalized-measure refusal:\n{msg}"
    );
}

/// The alias fix and the mass rule compose: the HS3 importer binds the pdf to
/// `__M__` by name, so the rule has to be reached through a bare-name alias too.
#[test]
fn mass_rule_is_reached_through_an_alias() {
    let pir = determinize_pir(
        "\
a = elementof(reals)
px = normalize(
  truncate(weighted(x -> polynomial([1.0, a], x), Lebesgue(reals)), interval(-10.0, 10.0)),
)
mm = px
lp = logdensityof(mm, 0.5)",
    );
    assert!(
        pir.contains("(add 20.0 ") && pir.contains("(mul (%ref self a) 0.0)"),
        "Z = 20 + a·0 through the alias:\n{pir}"
    );
}
