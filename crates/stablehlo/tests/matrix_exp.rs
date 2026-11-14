use flatppl_stablehlo::{Dtype, EmitOptions, Mode, Target};

fn emit(shape: &str, call: &str, dtype: Dtype, target: Target) -> String {
    let source = format!(
        "e = standard_module(\"ext-linear-algebra\", \"0.1\")\n\
         A = elementof({shape})\ninputs = A\noutputs = {call}"
    );
    let mut module = flatppl_syntax::parse(&source).unwrap();
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
    flatppl_stablehlo::emit(
        &module,
        Mode::LogDensity,
        &EmitOptions {
            dtype,
            target,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn matrix_exponential_preserves_matrix_cells_and_named_arguments() {
    for dtype in [Dtype::F32, Dtype::F64] {
        for target in [Target::Cpu, Target::Gpu] {
            let plain = emit("cartpow(reals, [2, 2])", "e.matexp(A)", dtype, target);
            let named = emit("cartpow(reals, [2, 2])", "e.matexp(A = A)", dtype, target);
            assert_eq!(plain, named);
            let ty = match dtype {
                Dtype::F32 => "f32",
                Dtype::F64 => "f64",
            };
            assert!(
                plain.contains(&format!("-> (tensor<2x2x{ty}>, tensor<i")),
                "{plain}"
            );
            assert!(!plain.contains("custom_call"), "{plain}");
        }
    }
}

#[test]
fn matrix_exponential_native_broadcast_preserves_batch_and_matrix_axes() {
    let output = emit(
        "cartpow(cartpow(reals, [2, 2]), 3)",
        "e.matexp.(A)",
        Dtype::F32,
        Target::Gpu,
    );
    assert!(
        output.contains("-> (tensor<3x2x2xf32>, tensor<i32>)"),
        "{output}"
    );
}
