fn emit(support: &str) -> String {
    let source = format!(
        "reg = standard_module(\"regression\", \"0.1\")\n\
         x = external(reals)\n\
         inputs = x\n\
         outputs = logdensityof(reg.Flat({support}), x)"
    );
    let mut module = flatppl_syntax::parse(&source).unwrap();
    assert!(flatppl_infer::infer(&mut module).is_empty());
    let roots = [module.intern("inputs"), module.intern("outputs")];
    let module = flatppl_determinizer::determinize_with_roots(
        &module,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
    )
    .unwrap();
    flatppl_stablehlo::emit(
        &module,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap()
}

#[test]
fn flat_unitinterval_is_the_unnormalized_closed_unit_interval() {
    assert_eq!(emit("unitinterval"), emit("interval(0.0, 1.0)"));
}
