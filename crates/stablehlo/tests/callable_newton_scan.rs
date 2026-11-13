use flatppl_stablehlo::{Dtype, EmitOptions, Mode, Target};

#[test]
fn callable_newton_with_nested_scans_emits_for_both_targets_and_precisions() {
    let src = include_str!("fixtures/callable_newton_scan.flatppl");
    let mut model = flatppl_syntax::parse(src).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let lowered = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .expect("determinize");
    for target in [Target::Cpu, Target::Gpu] {
        for dtype in [Dtype::F32, Dtype::F64] {
            let ir = flatppl_stablehlo::emit(
                &lowered,
                Mode::LogDensity,
                &EmitOptions {
                    target,
                    dtype,
                    ..Default::default()
                },
            )
            .expect("emit nested solve");
            assert!(ir.contains("stablehlo.while"));
        }
    }
}
