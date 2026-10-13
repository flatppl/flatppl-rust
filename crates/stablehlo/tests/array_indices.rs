#[test]
fn index_axis_tuples_export_as_separate_results() {
    let mut module = flatppl_syntax::parse("outputs = indicesof0(zeros([2, 3]))").unwrap();
    assert!(flatppl_infer::infer(&mut module).is_empty());
    let roots = [module.intern("outputs")];
    let module = flatppl_determinizer::determinize_with_roots(
        &module,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
    )
    .unwrap();
    let ir = flatppl_stablehlo::emit(
        &module,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap();
    assert_eq!(
        ir.lines().nth(1).unwrap().trim(),
        "func.func @logdensity() -> (tensor<2xi32>, tensor<3xi32>) {"
    );
}
