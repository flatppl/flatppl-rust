//! Enzyme-restricted `prod` lowers by target. One multiply reduction, whose
//! fork adjoint stays finite at zero factors, serves the GPU profile and narrow
//! CPU rows. Wide CPU rows pair factors with extent-one dots.

fn emit(source: &str, target: flatppl_stablehlo::Target) -> String {
    let mut module = flatppl_syntax::parse(source).expect("parse");
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots: Vec<_> = ["inputs", "outputs"].map(|name| module.intern(name)).into();
    let lowered = flatppl_determinizer::determinize_with_roots(
        &module,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
    )
    .expect("determinize");
    flatppl_stablehlo::emit(
        &lowered,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions {
            target,
            ..Default::default()
        },
    )
    .expect("prod must emit under Enzyme restrictions")
}

fn rows(width: usize) -> String {
    format!(
        "theta = elementof(cartpow(reals, [3, {width}]))\n\
         rows = [theta[1, :], theta[2, :], theta[3, :]]\n\
         inputs = theta\noutputs = prod.(rows)\n"
    )
}

#[test]
fn gpu_prod_is_one_multiply_reduction() {
    for width in [5, 64] {
        let ir = emit(&rows(width), flatppl_stablehlo::Target::Gpu);
        assert_eq!(ir.matches("stablehlo.reduce(").count(), 1, "{ir}");
        assert!(ir.contains("applies stablehlo.multiply"), "{ir}");
        assert!(!ir.contains("stablehlo.dot_general"), "{ir}");
    }
}

#[test]
fn cpu_prod_pairs_only_rows_of_at_least_64_factors() {
    let narrow = emit(&rows(63), flatppl_stablehlo::Target::Cpu);
    assert!(narrow.contains("applies stablehlo.multiply"), "{narrow}");
    assert!(!narrow.contains("stablehlo.dot_general"), "{narrow}");
    let wide = emit(&rows(64), flatppl_stablehlo::Target::Cpu);
    assert!(!wide.contains("applies stablehlo.multiply"), "{wide}");
    // Sixty-four factors pair in six levels.
    assert_eq!(wide.matches("stablehlo.dot_general").count(), 6, "{wide}");
}
