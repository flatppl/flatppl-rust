//! Distribution parameters outside their declared value sets report status 9.
//! A parameter whose set is known at emit time needs no runtime check.

use flatppl_stablehlo::{Dtype, EmitOptions, Mode, Target};

fn emit_mode(source: &str, sample: bool) -> Result<String, String> {
    let mut module = flatppl_syntax::parse(source).expect("parse");
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [module.intern("inputs"), module.intern("outputs")];
    let lowered = flatppl_determinizer::determinize_with_options(
        &module,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .expect("determinize");
    flatppl_stablehlo::emit(
        &lowered,
        if sample {
            Mode::Sample
        } else {
            Mode::LogDensity
        },
        &EmitOptions {
            restrict_enzyme_compatible: false,
            target: Target::Cpu,
            dtype: Dtype::F32,
            ..Default::default()
        },
    )
    .map_err(|error| error.to_string())
}

fn emit(source: &str) -> String {
    emit_mode(source, false).expect("emit")
}

fn checked(ir: &str) -> bool {
    ir.contains("flatppl.runtime_status") && ir.contains("dense<9>")
}

fn density(params: &str, measure: &str, variate: &str) -> String {
    format!(
        "{params}\nx = elementof({variate})\ninputs = ({inputs}x)\noutputs = logdensityof({measure}, x)\n",
        inputs = params
            .lines()
            .filter_map(|line| line
                .split_once(" = elementof")
                .map(|(name, _)| format!("{name}, ")))
            .collect::<String>()
    )
}

#[test]
fn runtime_scale_reports_status_9() {
    let ir = emit(&density(
        "mu = elementof(reals)\nsigma = elementof(reals)",
        "Normal(mu, sigma)",
        "reals",
    ));
    assert!(checked(&ir), "{ir}");
    assert!(ir.contains("stablehlo.compare LE, %arg1"), "{ir}");
    // NaN propagates, so `mu` over `reals` has no test.
    assert_eq!(ir.matches("stablehlo.compare").count(), 1, "{ir}");
}

#[test]
fn a_declared_input_set_is_still_checked() {
    let ir = emit(&density(
        "sigma = elementof(posreals)",
        "Normal(0.0, sigma)",
        "reals",
    ));
    assert!(checked(&ir), "{ir}");
}

#[test]
fn constants_inside_their_sets_need_no_check() {
    let ir = emit(&density("", "Normal(0.0, 2.0)", "reals"));
    assert!(!ir.contains("flatppl.runtime_status"), "{ir}");
    let ir = emit(&density("", "Dirichlet([1.0, 2.0, 3.0])", "stdsimplex(3)"));
    assert!(!ir.contains("flatppl.runtime_status"), "{ir}");
    let ir = emit(&density(
        "",
        "MvNormal([0.0, 0.0], rowstack([[2.0, 1.0], [1.0, 2.0]]))",
        "cartpow(reals, 2)",
    ));
    assert!(!ir.contains("flatppl.runtime_status"), "{ir}");
}

#[test]
fn constants_outside_their_sets_refuse() {
    let source = density("", "Dirichlet([1.0, -1.0])", "stdsimplex(2)");
    let error = emit_mode(&source, false).unwrap_err();
    assert!(error.contains("`Dirichlet`'s `alpha`"), "{error}");
    let source = density("", "Categorical([0.5, 0.6])", "posintegers");
    let error = emit_mode(&source, false).unwrap_err();
    assert!(error.contains("stdsimplex(n)"), "{error}");
    let source = density(
        "",
        "MvNormal([0.0, 0.0], rowstack([[1.0, 2.0], [2.0, 1.0]]))",
        "cartpow(reals, 2)",
    );
    let error = emit_mode(&source, false).unwrap_err();
    assert!(error.contains("positive definite"), "{error}");
}

#[test]
fn transforms_that_guarantee_the_set_elide_the_check() {
    let cases = [
        ("a = elementof(reals)", "Exponential(exp(a))", "reals"),
        ("a = elementof(reals)", "Bernoulli(invlogit(a))", "integers"),
        (
            "a = elementof(reals)",
            "Gamma(abs(a) + 1.0, exp(a))",
            "reals",
        ),
        (
            "v = elementof(cartpow(reals, 3))",
            "Categorical(softmax(v))",
            "posintegers",
        ),
        (
            "v = elementof(cartpow(reals, 3))",
            "Dirichlet(exp.(v))",
            "stdsimplex(3)",
        ),
    ];
    for (params, measure, variate) in cases {
        let ir = emit(&density(params, measure, variate));
        // `abs(a) + 1.0` has no §07 result set, so only the shape keeps a test.
        if measure.starts_with("Gamma") {
            assert_eq!(
                ir.matches("stablehlo.compare LE").count(),
                1,
                "{measure}: {ir}"
            );
            continue;
        }
        assert!(!ir.contains("flatppl.runtime_status"), "{measure}: {ir}");
    }
}

#[test]
fn every_scalar_set_has_a_runtime_test() {
    let cases = [
        (
            "p = elementof(reals)",
            "Bernoulli(p)",
            "integers",
            ["LT", "GT"].as_slice(),
        ),
        (
            "r = elementof(reals)",
            "Poisson(r)",
            "integers",
            ["LT"].as_slice(),
        ),
        (
            "n = elementof(integers)\np = elementof(reals)",
            "Binomial(n, p)",
            "integers",
            ["LT", "GT"].as_slice(),
        ),
    ];
    for (params, measure, variate, directions) in cases {
        let ir = emit(&density(params, measure, variate));
        assert!(checked(&ir), "{measure}: {ir}");
        for direction in directions {
            assert!(
                ir.contains(&format!("stablehlo.compare {direction}")),
                "{measure} lacks {direction}: {ir}"
            );
        }
    }
}

#[test]
fn a_vector_parameter_is_checked_per_element() {
    let ir = emit(&density(
        "alpha = elementof(cartpow(reals, 3))",
        "Dirichlet(alpha)",
        "stdsimplex(3)",
    ));
    assert!(checked(&ir), "{ir}");
    // The vector reduces to its minimum, which stays in the surrounding fusion.
    assert!(
        ir.contains("stablehlo.reduce(%arg0 init: %0) applies stablehlo.minimum"),
        "{ir}"
    );
    assert!(!ir.contains("applies stablehlo.or"), "{ir}");
}

#[test]
fn a_runtime_simplex_checks_cells_and_total() {
    let ir = emit(&density(
        "p = elementof(cartpow(reals, 3))",
        "Categorical(p)",
        "posintegers",
    ));
    assert!(checked(&ir), "{ir}");
    assert!(ir.contains("applies stablehlo.minimum"), "{ir}");
    // The f32 allowance for three cells, 8 * 3 * 2^-23.
    assert!(ir.contains("dense<0.00000286102294921875>"), "{ir}");
    // One status merge covers the cells and the total.
    assert_eq!(ir.matches("dense<9>").count(), 1, "{ir}");
}

#[test]
fn a_runtime_covariance_reuses_its_cholesky_factor() {
    let ir = emit(&density(
        "cov = elementof(cartpow(reals, [2, 2]))",
        "MvNormal([0.0, 0.0], cov)",
        "cartpow(reals, 2)",
    ));
    assert!(checked(&ir), "{ir}");
    assert_eq!(ir.matches("stablehlo.cholesky").count(), 1, "{ir}");
}

#[test]
fn a_sampler_checks_its_parameters() {
    let source = "sigma = external(reals)\nvalue, state = rand(Normal(0.0, sigma), rnginit(23))\n\
        inputs = sigma\noutputs = (value, state)\n";
    let ir = emit_mode(source, true).expect("emit");
    assert!(checked(&ir), "{ir}");
}

#[test]
fn fixed_data_summaries_check_their_parameters() {
    let source = "sigma = external(reals)\n\
        inputs = sigma\noutputs = logdensityof(iid(Normal(0.0, sigma), 3), [1.0, 2.0, 4.0])";
    let ir = emit(source);
    assert!(checked(&ir), "{ir}");
}
