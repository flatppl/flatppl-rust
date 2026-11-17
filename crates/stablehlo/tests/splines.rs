use flatppl_stablehlo::{Dtype, EmitOptions, Mode};

#[test]
fn bspline_degree_is_bounded_by_knot_count() {
    let mut module = flatppl_syntax::parse(
        r#"
        sp = standard_module("splines", "0.1")
        x = external(reals)
        knots = external(cartpow(reals, 36))
        degree = 17
        inputs = (x, knots)
        outputs = sp.bspline_basis(x, knots, degree)
    "#,
    )
    .unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [module.intern("inputs"), module.intern("outputs")];
    let module = flatppl_determinizer::determinize_with_options(
        &module,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap();
    let ir = flatppl_stablehlo::emit(
        &module,
        Mode::LogDensity,
        &EmitOptions {
            dtype: Dtype::F64,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(ir.contains("tensor<18xf64>"));
}

#[test]
fn serialized_default_flatpdl_keeps_spline_domain_status() {
    for grid in ["[0, 1, 2]", "[0, 0, 2]"] {
        let source = format!(
            r#"
            sp = standard_module("splines", "0.1")
            x = external(reals)
            inputs = x
            outputs = sp.interp_linear(x, {grid}, [0, 1, 4])
        "#
        );
        let mut module = flatppl_syntax::parse(&source).unwrap();
        let diagnostics = flatppl_infer::infer(&mut module);
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
        let module = flatppl_determinizer::determinize(&module).unwrap();
        let serialized = flatppl_flatpir::write(&module);
        let module = flatppl_flatpir::read(&serialized).unwrap();
        let ir =
            flatppl_stablehlo::emit(&module, Mode::LogDensity, &EmitOptions::default()).unwrap();
        assert!(ir.contains("flatppl.runtime_status"), "{ir}");
        assert!(ir.contains("dense<12> : tensor<i32>"), "{ir}");
    }
}
