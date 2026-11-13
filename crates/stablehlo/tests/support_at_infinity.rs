//! §03 "Note on infinities": `reals` admits `±inf`, but the Lebesgue reference
//! measure covers the finite-real part and "the points ±∞ carry zero Lebesgue
//! mass". So `±inf` lies outside the support of every elementwise density, the
//! density tends to 0 there, and its log-density is `-inf`. Formulas such as
//! `(k - 1) log(x) - rate x` give the IEEE artifact `inf - inf = NaN`, so every
//! elementwise density guards the variate. The executed values are pinned by
//! the testsuite row `stablehlo/support_at_infinity`.
//!
//! The Normal sampler's `sqrt(2)` is emitted at full precision: an 8-digit
//! literal moved every f64 draw by about 4.4e-8 relative.

use flatppl_stablehlo::{Dtype, EmitOptions, Mode};

fn emit(source: &str, mode: Mode, dtype: Dtype) -> String {
    emit_with(source, mode, dtype, true)
}

/// `sufficient_statistics` selects the shared likelihood-summary path or the
/// plain per-factor score, at both the determiniser and the emitter.
fn emit_with(source: &str, mode: Mode, dtype: Dtype, sufficient_statistics: bool) -> String {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let options = EmitOptions {
        dtype,
        restrict_enzyme_compatible: false,
        sufficient_statistics,
        ..Default::default()
    };
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &options.lowering_options(),
    )
    .unwrap();
    flatppl_stablehlo::emit(&model, mode, &options).unwrap()
}

/// The guard compares `|x|` with `inf` and selects `-inf` there.
fn guards_the_variate(hlo: &str) -> bool {
    let abs = hlo.lines().find_map(|line| {
        let (ssa, rest) = line.trim().split_once(" = ")?;
        rest.starts_with("stablehlo.abs %arg0")
            .then_some(ssa.to_owned())
    });
    abs.is_some_and(|abs| {
        hlo.lines()
            .any(|line| line.contains("stablehlo.compare EQ,") && line.contains(&abs))
    })
}

#[test]
fn every_half_line_density_guards_an_infinite_variate() {
    for density in [
        "Gamma(shape = 2.5, rate = 1.5)",
        "InverseGamma(shape = 2.5, scale = 1.5)",
        "Weibull(shape = 2.5, scale = 1.5)",
        "Rayleigh(scale = 1.5)",
        "ChiSquared(k = 3.0)",
        "Exponential(rate = 1.5)",
        "LogNormal(mu = 0.0, sigma = 1.0)",
        "Pareto(shape = 2.5, scale = 1.5)",
        "Logistic(mu = 0.0, s = 1.0)",
        "Normal(mu = 0.0, sigma = 1.0)",
    ] {
        let hlo = emit(
            &format!("x = elementof(reals)\ninputs = x\noutputs = logdensityof({density}, x)\n"),
            Mode::LogDensity,
            Dtype::F64,
        );
        assert!(guards_the_variate(&hlo), "{density}\n{hlo}");
    }
}

/// The right-hand side defining `ssa`.
fn definition<'a>(hlo: &'a str, ssa: &str) -> &'a str {
    hlo.lines()
        .find_map(|line| line.trim().strip_prefix(&format!("{ssa} = ")))
        .unwrap_or_else(|| panic!("no definition of {ssa}:\n{hlo}"))
}

/// The `@logdensity` result is `-inf` by construction: the folded `-inf`
/// constant, or the guard's select of that constant on `|x| == inf`.
fn scores_minus_inf(hlo: &str, literal: &str) -> bool {
    let ret = hlo
        .lines()
        .find_map(|line| line.trim().strip_prefix("return "))
        .expect("a return");
    let ret = ret.split(',').next().unwrap().trim();
    let is_minus_inf = |ssa: &str| {
        definition(hlo, ssa).starts_with(&format!("stablehlo.constant dense<{literal}>"))
    };
    if is_minus_inf(ret) {
        return true;
    }
    let Some(operands) = definition(hlo, ret).strip_prefix("stablehlo.select ") else {
        return false;
    };
    let operands: Vec<&str> = operands.split(" :").next().unwrap().split(", ").collect();
    let [predicate, chosen, _] = operands[..] else {
        return false;
    };
    let Some(compared) = definition(hlo, predicate).strip_prefix("stablehlo.compare EQ, ") else {
        return false;
    };
    let magnitude = compared.split(", ").next().unwrap();
    is_minus_inf(chosen) && definition(hlo, magnitude).starts_with("stablehlo.abs ")
}

/// §03: a literal `±inf` variate scores `-inf` on both the summary and the
/// plain path. A fixed `inf` folds the score outright; `-inf` is written
/// `neg(inf)`, which does not fold, so the guard selects `-inf` at run time.
/// The density formula alone would give `inf - inf = NaN`.
#[test]
fn a_literal_infinite_variate_scores_minus_inf() {
    for (density, x) in [
        ("Weibull(shape = k, scale = 1.5)", "inf"),
        ("Gamma(shape = k, rate = 1.5)", "inf"),
        ("ChiSquared(k = k)", "inf"),
        ("Rayleigh(scale = k)", "inf"),
        ("Logistic(mu = 0.0, s = k)", "-inf"),
    ] {
        for (dtype, literal) in [
            (Dtype::F32, "0xFF800000"),
            (Dtype::F64, "0xFFF0000000000000"),
        ] {
            for summaries in [true, false] {
                let hlo = emit_with(
                    &format!(
                        "k = elementof(posreals)\nvalue = logdensityof({density}, {x})\ninputs = k\noutputs = value\n"
                    ),
                    Mode::LogDensity,
                    dtype,
                    summaries,
                );
                assert!(
                    scores_minus_inf(&hlo, literal),
                    "{density} at {x} (summaries {summaries}) must score -inf:\n{hlo}"
                );
            }
        }
    }
}

#[test]
fn the_normal_sampler_uses_sqrt2_at_full_precision() {
    let source =
        "s = rnginit(0)\nx, s2 = rand(Normal(0.0, 1.0), s)\ninputs = s\noutputs = (x, s2)\n";
    for (dtype, name) in [(Dtype::F64, "f64"), (Dtype::F32, "f32")] {
        let hlo = emit(source, Mode::Sample, dtype);
        let literals: Vec<f64> = hlo
            .lines()
            .filter(|line| line.contains("stablehlo.constant dense<1.414"))
            .map(|line| {
                let start = line.find("dense<").unwrap() + "dense<".len();
                let end = start + line[start..].find('>').unwrap();
                line[start..end].parse().unwrap()
            })
            .collect();
        assert!(!literals.is_empty(), "{hlo}");
        for value in literals {
            assert_eq!(value, std::f64::consts::SQRT_2, "{name}");
        }
    }
}
