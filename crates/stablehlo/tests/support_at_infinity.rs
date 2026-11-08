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
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
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
        mode,
        &EmitOptions {
            dtype,
            restrict_enzyme_compatible: false,
            ..Default::default()
        },
    )
    .unwrap()
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
