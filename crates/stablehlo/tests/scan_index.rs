use flatppl_stablehlo::{Dtype, EmitOptions, Mode, Target};

fn emit(source: &str, target: Target, dtype: Dtype) -> String {
    let mut model = flatppl_syntax::parse(source).expect("parse");
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_roots(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
    )
    .expect("determinize");
    flatppl_stablehlo::emit(
        &model,
        Mode::LogDensity,
        &EmitOptions {
            dtype,
            target,
            ..Default::default()
        },
    )
    .expect("emit")
}

fn source(indices: &str, lookup: &str, extent: u64, inputs: &str) -> String {
    format!(
        "x = elementof(cartpow(reals, {extent}))\n\
         positions = {indices}\n\
         alias = positions\n\
         step(s, j, data) = tanh(s + {lookup})\n\
         states = scan(step, 0.125, alias, x)\n\
         inputs = {inputs}\n\
         outputs = sum(states)"
    )
}

fn signature(ir: &str) -> &str {
    ir.lines().find(|line| line.contains("func.func")).unwrap()
}

fn reads_position_vector(ir: &str, size: u64, index_type: &str) -> bool {
    ir.lines().any(|line| {
        line.contains("stablehlo.dynamic_slice")
            && line.contains(&format!("(tensor<{size}x{index_type}>,"))
    })
}

fn positions(values: impl IntoIterator<Item = i64>) -> String {
    let values = values
        .into_iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}

#[test]
fn in_bounds_scan_positions_keep_the_status_signature_and_skip_the_lookup_check() {
    for (dtype, index_type) in [(Dtype::F32, "i32"), (Dtype::F64, "i64")] {
        for (indices, lookup) in [
            (positions(2..=36), "get(data, j)"),
            (positions(0..35), "get0(data, j)"),
        ] {
            let model = source(&indices, lookup, 36, "x");
            let cpu = emit(&model, Target::Cpu, dtype);
            let gpu = emit(&model, Target::Gpu, dtype);
            assert_eq!(signature(&cpu), signature(&gpu));
            assert_eq!(cpu.lines().next(), gpu.lines().next());
            assert!(gpu.contains("flatppl.runtime_status = 1"), "{gpu}");
            assert!(reads_position_vector(&cpu, 35, index_type), "{cpu}");
            assert!(!reads_position_vector(&gpu, 35, index_type), "{gpu}");
            assert!(cpu.contains("stablehlo.compare GE"), "{cpu}");
            assert!(!gpu.contains("stablehlo.compare GE"), "{gpu}");
        }
    }
}

#[test]
fn a_scan_without_dynamic_lookup_keeps_its_status_free_signature() {
    let model = source(&positions(1..=35), "get(data, 1) + 0.01 * j", 36, "x");
    let cpu = emit(&model, Target::Cpu, Dtype::F64);
    let gpu = emit(&model, Target::Gpu, Dtype::F64);
    assert_eq!(signature(&cpu), signature(&gpu));
    assert!(!gpu.contains("flatppl.runtime_status"), "{gpu}");
    assert!(!reads_position_vector(&gpu, 35, "i64"), "{gpu}");
}

#[test]
fn unproved_positions_and_single_step_scans_keep_the_position_reads() {
    for (indices, length, inputs) in [
        (positions((2..=36).rev()), 35, "x"),
        (positions((1..=36).filter(|i| *i != 2)), 35, "x"),
        (positions(std::iter::once(1).chain(1..=34)), 35, "x"),
        ("[1, 2, 3]".to_string(), 3, "x"),
        ("[1]".to_string(), 1, "x"),
        (
            "elementof(cartpow(integers, 35))".to_string(),
            35,
            "(x, positions)",
        ),
    ] {
        let model = source(&indices, "get(data, j)", 36, inputs);
        let gpu = emit(&model, Target::Gpu, Dtype::F64);
        assert!(reads_position_vector(&gpu, length, "i64"), "{gpu}");
        assert!(gpu.contains("stablehlo.compare GE"), "{gpu}");
    }
}

#[test]
fn out_of_bounds_scan_positions_remain_runtime_errors_in_the_tail() {
    for indices in [positions(-2..33), positions(3..=37)] {
        let model = source(&indices, "get(data, j)", 36, "x");
        let cpu = emit(&model, Target::Cpu, Dtype::F64);
        let gpu = emit(&model, Target::Gpu, Dtype::F64);
        assert_eq!(signature(&cpu), signature(&gpu));
        assert_eq!(cpu.lines().next(), gpu.lines().next());
        assert!(gpu.contains("stablehlo.compare GE"), "{gpu}");
        assert!(gpu.contains("stablehlo.compare LE"), "{gpu}");
    }
}
