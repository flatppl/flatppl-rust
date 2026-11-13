//! §08 count densities have the domain `integers`: a real-typed point with a
//! fractional part lies outside the support, so the density is zero and the
//! log-density is `-inf` (scipy's `logpmf` gives `-inf` there too). A real point
//! that equals a whole number scores like that integer, so a converter that
//! writes counts as `70.0` keeps scoring. §09 `ContinuedPoisson` is continuous
//! by design and keeps the lgamma extension (`golden_hep_dists.rs`).
//!
//! The executed values at fractional, whole-real and integer points, and their
//! parameter gradients, are checked against scipy through Enzyme-JAX.

use flatppl_stablehlo::{Dtype, EmitOptions, Mode};

fn emit(source: &str) -> String {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap();
    flatppl_stablehlo::emit(
        &model,
        Mode::LogDensity,
        &EmitOptions {
            dtype: Dtype::F64,
            restrict_enzyme_compatible: false,
            ..Default::default()
        },
    )
    .unwrap()
}

/// The right-hand side defining `ssa`.
fn definition<'a>(hlo: &'a str, ssa: &str) -> &'a str {
    hlo.lines()
        .find_map(|line| line.trim().strip_prefix(&format!("{ssa} = ")))
        .unwrap_or_else(|| panic!("no definition of {ssa}:\n{hlo}"))
}

/// Whether the first value the `@logdensity` entry returns is a fixed `-inf`:
/// the `-inf` constant, or the negation of the `+inf` constant.
fn returns_minus_inf(hlo: &str) -> bool {
    let ret = hlo
        .lines()
        .find_map(|line| line.trim().strip_prefix("return "))
        .expect("a return");
    let ret = ret.split(',').next().unwrap().trim();
    let value = definition(hlo, ret);
    value.starts_with("stablehlo.constant dense<0xFFF0000000000000>")
        || value
            .strip_prefix("stablehlo.negate ")
            .and_then(|rest| rest.split(' ').next())
            .is_some_and(|x| {
                definition(hlo, x).starts_with("stablehlo.constant dense<0x7FF0000000000000>")
            })
}

const FAMILIES: &[(&str, &str, &str)] = &[
    ("Poisson(rate = t)", "2.5", "3.0"),
    ("Binomial(n = 10, p = t)", "2.5", "3.0"),
    ("Geometric(p = t)", "2.5", "3.0"),
    ("NegativeBinomial(alpha = t, beta = 0.5)", "2.5", "3.0"),
    ("NegativeBinomial2(mu = t, psi = 2.0)", "2.5", "3.0"),
    ("Categorical(p = [t, 0.3, 0.7 - t])", "2.5", "2.0"),
    ("Categorical0(p = [t, 0.3, 0.7 - t])", "1.5", "1.0"),
];

fn score_at(density: &str, point: &str) -> String {
    emit(&format!(
        "t = elementof(posreals)\nvalue = logdensityof({density}, {point})\ninputs = t\noutputs = value\n"
    ))
}

#[test]
fn a_fractional_count_scores_minus_inf() {
    for (density, fractional, _) in FAMILIES {
        let hlo = score_at(density, fractional);
        assert!(
            returns_minus_inf(&hlo),
            "{density} at {fractional} must score a fixed -inf:\n{hlo}"
        );
    }
}

#[test]
fn a_whole_real_count_still_scores() {
    for (density, _, whole) in FAMILIES {
        let hlo = score_at(density, whole);
        assert!(
            !returns_minus_inf(&hlo),
            "{density} at {whole} must keep its score:\n{hlo}"
        );
    }
}

/// A runtime real point gets the whole-number check; an integer-typed point is
/// whole by type and gets none. ContinuedPoisson's goldens pin that it gets none.
#[test]
fn only_a_real_count_is_checked_for_a_fractional_part() {
    let source = |density: &str, set: &str| {
        emit(&format!(
            "t = elementof(posreals)\nx = elementof({set})\nvalue = logdensityof({density}, x)\ninputs = (t, x)\noutputs = value\n"
        ))
    };
    let checks_x = |hlo: &str| hlo.contains("stablehlo.floor %arg1");
    for (density, _, _) in FAMILIES {
        assert!(checks_x(&source(density, "reals")), "{density} over reals");
        assert!(
            !checks_x(&source(density, "integers")),
            "{density} over integers"
        );
    }
}
