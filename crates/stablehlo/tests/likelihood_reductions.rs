use flatppl_core::Module;
use flatppl_stablehlo::{Dtype, EmitOptions, Mode};

fn model(source: &str) -> Module {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let errors = flatppl_infer::infer(&mut module);
    assert!(errors.is_empty(), "{errors:?}");
    let roots = [module.intern("inputs"), module.intern("outputs")];
    flatppl_determinizer::determinize_with_roots(
        &module,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
    )
    .unwrap()
}

fn emit(source: &str, dtype: Dtype) -> String {
    flatppl_stablehlo::emit(
        &model(source),
        Mode::LogDensity,
        &EmitOptions {
            dtype,
            ..Default::default()
        },
    )
    .unwrap()
}

fn returned_constant(ir: &str, dtype: Dtype) -> f64 {
    let returned = ir
        .lines()
        .find_map(|line| line.trim().strip_prefix("return "))
        .unwrap();
    let mut name = returned.split_whitespace().next().unwrap();
    let reduction_prefix = format!("{name} = stablehlo.reduce(");
    if let Some(input) = ir
        .lines()
        .find_map(|line| line.trim().strip_prefix(&reduction_prefix))
    {
        name = input.split_whitespace().next().unwrap();
    }
    let prefix = format!("{name} = stablehlo.constant dense<");
    let literal = ir
        .lines()
        .find_map(|line| line.trim().strip_prefix(&prefix))
        .expect("fixed reduced density must fold to a returned scalar");
    let literal = literal.split('>').next().unwrap();
    if let Some(hex) = literal
        .strip_prefix("\"0x")
        .and_then(|x| x.strip_suffix('"'))
    {
        let bytes = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect::<Vec<_>>();
        return match dtype {
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| f32::from_le_bytes(*x) as f64)
                .sum(),
            Dtype::F64 => bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|x| f64::from_le_bytes(*x))
                .sum(),
        };
    }
    literal.parse().unwrap()
}

fn assert_fixed(distribution: &str, data: &str, count: usize, expected: f64) {
    let source = format!("outputs = logdensityof(iid({distribution}, {count}), {data})");
    for dtype in [Dtype::F32, Dtype::F64] {
        let ir = emit(&source, dtype);
        let actual = returned_constant(&ir, dtype);
        let tolerance = match dtype {
            Dtype::F32 => 2e-6,
            Dtype::F64 => 2e-14,
        };
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{distribution}: {actual} != {expected}\n{ir}"
        );
    }
}

#[test]
fn fixed_iid_scores_keep_the_full_normalizing_constants() {
    let log_two_pi = (2.0 * std::f64::consts::PI).ln();
    assert_fixed(
        "Normal(1, 2)",
        "[0.0, 1.0, 2.0]",
        3,
        -3.0 * (2.0_f64.ln() + log_two_pi / 2.0) - 0.25,
    );
    assert_fixed(
        "Gamma(2, 3)",
        "[1.0, 2.0]",
        2,
        4.0 * 3.0_f64.ln() + 2.0_f64.ln() - 9.0,
    );
    assert_fixed("Beta(2, 3)", "[0.25, 0.5]", 2, 2.53125_f64.ln());
    assert_fixed(
        "Exponential(2)",
        "[0.0, 1.0, 2.0]",
        3,
        3.0 * 2.0_f64.ln() - 6.0,
    );
    assert_fixed("Poisson(2)", "[0, 1, 2]", 3, 2.0 * 2.0_f64.ln() - 6.0);
    assert_fixed(
        "Binomial(4, 0.25)",
        "[0, 1, 4]",
        3,
        (4.0 * 0.25_f64.powi(5) * 0.75_f64.powi(7)).ln(),
    );
}

#[test]
fn normal_centered_moments_keep_large_locations() {
    let source = "outputs = logdensityof(iid(Normal(100000008, 4), 3), [100000000.0, 100000008.0, 100000016.0])";
    let expected = -3.0 * (4.0_f64.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln()) - 4.0;
    for dtype in [Dtype::F32, Dtype::F64] {
        let actual = returned_constant(&emit(source, dtype), dtype);
        assert!((actual - expected).abs() < 3e-6);
    }
    let source = "outputs = logdensityof(iid(Normal(1000000000000.0, 2), 3), [1000000000000.25, 1000000000000.5, 1000000000001.0])";
    let expected = -3.0 * (2.0_f64.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln()) - 1.3125 / 8.0;
    assert!((returned_constant(&emit(source, Dtype::F64), Dtype::F64) - expected).abs() < 1e-14);
    let source = "outputs = logdensityof(iid(Normal(100000016.0,10.0),4), [100000000.0,100000008.0,100000016.0,100000024.0])";
    let expected = -4.0 * (10.0_f64.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln()) - 1.92;
    assert!((returned_constant(&emit(source, Dtype::F32), Dtype::F32) - expected).abs() < 2e-6);
    assert_fixed(
        "Normal(0.7,1.3)",
        "[0.17,0.31,1.19,2.73]",
        4,
        -4.0 * (1.3_f64.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln())
            - (0.53_f64.powi(2) + 0.39_f64.powi(2) + 0.49_f64.powi(2) + 2.03_f64.powi(2))
                / (2.0 * 1.3_f64.powi(2)),
    );
}

#[test]
fn integer_normal_observations_keep_subtraction_before_float_conversion() {
    let source = "outputs = logdensityof(iid(Normal(100000001,1),2), [100000000,100000002])";
    let ir = emit(source, Dtype::F32);
    let expected = -(2.0 * std::f64::consts::PI).ln() - 1.0;
    assert!(
        (returned_constant(&ir, Dtype::F32) - expected).abs() < 3e-7,
        "{ir}"
    );
}

#[test]
fn live_data_and_varying_parameters_retain_observation_scoring() {
    let source = "mu = external(reals)\ny = [1.0, 2.0, 3.0]\ninputs = (mu, y)\noutputs = logdensityof(iid(Normal(mu, 1), 3), y)";
    let ir = emit(source, Dtype::F64);
    assert!(
        ir.contains("%arg1: tensor<3xf64>") && ir.contains("stablehlo.reduce"),
        "{ir}"
    );
    let ir = emit(
        "mu = external(reals)\ninputs = mu\noutputs = logdensityof(Normal.([mu, mu + 1, mu + 2], 1), [1.0, 2.0, 3.0])",
        Dtype::F64,
    );
    assert!(
        ir.contains("stablehlo.reduce") && ir.contains("tensor<3xf64>"),
        "{ir}"
    );
}

#[test]
fn reductions_preserve_valid_discrete_endpoints() {
    assert_fixed("Poisson(0)", "[0, 0, 0]", 3, 0.0);
    assert_fixed("Binomial(4, 0)", "[0, 0, 0]", 3, 0.0);
    assert_fixed("Binomial(4, 1)", "[4, 4, 4]", 3, 0.0);
}

#[test]
fn unsupported_data_keeps_the_existing_numerical_and_support_paths() {
    for (distribution, data) in [
        ("Poisson(r)", "[64, 65]"),
        ("Gamma(2, r)", "[0.0, 1.0]"),
        ("Beta(2, r)", "[0.0, 1.0]"),
    ] {
        let ir = emit(
            &format!(
                "r = external(posreals)\ninputs = r\noutputs = logdensityof(iid({distribution}, 2), {data})"
            ),
            Dtype::F64,
        );
        assert!(ir.contains("stablehlo.reduce"), "{distribution}: {ir}");
    }
    let ir = emit(
        "n = external(posintegers)\np = external(unitinterval)\ninputs = (n, p)\noutputs = logdensityof(iid(Binomial(n,p), 2), [1,2])",
        Dtype::F64,
    );
    assert!(ir.contains("stablehlo.reduce"), "{ir}");
    let ir = emit(
        "outputs = logdensityof(iid(Normal(0.0,1e-200),2), [0.0,1e-200])",
        Dtype::F64,
    );
    let expected = -2.0 * 1e-200_f64.ln() - (2.0 * std::f64::consts::PI).ln() - 0.5;
    assert!(
        (returned_constant(&ir, Dtype::F64) - expected).abs() < 2e-12,
        "{ir}"
    );
}

#[test]
fn shared_runtime_parameters_reduce_to_scalar_arithmetic() {
    let source = "mu = external(reals)\nsigma = external(posreals)\ninputs = (mu,sigma)\noutputs = logdensityof(iid(Normal(mu,sigma),3), [0.0,1.0,2.0])";
    let ir = emit(source, Dtype::F64);
    assert!(
        !ir.contains("tensor<3xf64>") && !ir.contains("stablehlo.reduce"),
        "{ir}"
    );
    let ir = emit(
        "sigma = external(posreals)\ninputs = sigma\noutputs = logdensityof(Normal.([0.0,0.0,0.0],sigma), [0.0,1.0,2.0])",
        Dtype::F64,
    );
    assert!(
        !ir.contains("tensor<3xf64>") && !ir.contains("stablehlo.reduce"),
        "{ir}"
    );
    let source = "means = external(cartpow(reals, 2))\nscore(mu) = logdensityof(iid(Normal(mu,2),3), [0.0,1.0,2.0])\ninputs = means\noutputs = score.(means)";
    let ir = emit(source, Dtype::F64);
    assert!(ir.contains("-> tensor<2xf64>"), "{ir}");
    assert!(
        !ir.contains("tensor<2x3xf64>") && !ir.contains("stablehlo.reduce"),
        "{ir}"
    );
}
