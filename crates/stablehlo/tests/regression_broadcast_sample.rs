use flatppl_infer::ModuleBundle;

fn emit(source: &str) -> String {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let (diagnostics, bundle) = flatppl_infer::infer_module_with_inputs(
        &mut module,
        ModuleBundle::new(),
        flatppl_infer::Level::Shape,
    );
    assert!(
        !diagnostics
            .iter()
            .any(|d| d.severity == flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    let roots = [module.intern("inputs"), module.intern("outputs")];
    let flat = flatppl_determinizer::determinize_with_options(
        &module,
        &bundle,
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap();
    flatppl_stablehlo::emit(
        &flat,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions {
            restrict_enzyme_compatible: false,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn outer_regression_samples_keep_each_matrix_cell_and_threaded_key() {
    for law in [
        "BernoulliLogitGLM.([x,x],0.1,bs)",
        "BinomialLogitGLM.([x,x],4,0.1,bs)",
        "CategoricalLogitGLM.([x,x],[[0.1,-0.2]],[rowstack([[0.2,0.1],[-0.1,0.3]]),rowstack([[0.1,-0.3],[0.3,0.1]])])",
        "NormalGLM.([x,x],0.1,bs,0.7)",
        "PoissonLogGLM.([x,x],0.1,bs)",
        "NegativeBinomial2LogGLM.([x,x],0.1,bs,2.0)",
        "StudentTGLM.([x,x],3.0,0.1,bs,0.7)",
        "GammaLogGLM.([x,x],0.1,bs,2.0)",
        "OrderedLogitGLM.([x,x],bs,[[-1.0,1.0]])",
        "BetaLogitGLM.([x,x],0.1,bs,4.0)",
    ] {
        let source = format!(
            "reg=standard_module(\"regression\",\"0.1\")\nx=external(cartpow(reals,[3,2]))\nbs=[[0.2,-0.3],[-0.1,0.4]]\ny ~ reg.{law}\nr=rnginit(41)\ninputs=(x,r)\noutputs=rand(lawof(y),r)\n"
        );
        let mlir = emit(&source);
        assert!(mlir.contains("tensor<2x3x"), "{law}\n{mlir}");
        assert!(
            mlir.matches("stablehlo.rng_bit_generator").count() >= 2,
            "{law}\n{mlir}"
        );
    }
}
