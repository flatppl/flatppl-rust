//! Enzyme-restricted `prod` lowers to one multiply reduction. The fork's
//! `ProductReductionTree` adjoint keeps its derivatives finite at zero factors.

fn emit(source: &str) -> String {
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
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect("prod must emit under Enzyme restrictions")
}

#[test]
fn enzyme_prod_is_one_multiply_reduction() {
    let ir = emit(
        "theta = elementof(cartpow(reals, [3, 5]))\n\
         rows = [theta[1, :], theta[2, :], theta[3, :]]\n\
         inputs = theta\noutputs = prod.(rows)\n",
    );
    assert_eq!(ir.matches("stablehlo.reduce(").count(), 1, "{ir}");
    assert!(ir.contains("applies stablehlo.multiply"), "{ir}");
    assert!(!ir.contains("stablehlo.dot_general"), "{ir}");
}
