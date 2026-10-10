//! Converter-form `pyhf_helpers` yields lower as each sample's multiply chain.

use flatppl_infer::ModuleBundle;

fn emit(source: &str) -> String {
    emit_for(source, flatppl_stablehlo::Target::Cpu)
}

fn emit_for(source: &str, target: flatppl_stablehlo::Target) -> String {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let (_, bundle) = flatppl_infer::infer_module_with_inputs(
        &mut model,
        ModuleBundle::new(),
        flatppl_infer::Level::Shape,
    );
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &bundle,
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap();
    flatppl_stablehlo::emit(
        &model,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions {
            target,
            ..Default::default()
        },
    )
    .unwrap()
}

const PREFIX: &str = "pyhf = standard_module(\"pyhf_helpers\", \"0.1\")\n\
    theta = elementof(cartpow(reals, 4))\ninputs = theta\n\
    a = theta[1]\ng = theta[[2, 3]]\ns = theta[4]\n\
    n1 = [1.0, 2.0]\nn2 = [3.0, 4.0]\ndelta = [0.5, 0.25] .* s\n";

#[test]
fn converter_form_yields_emit_the_per_sample_chain() {
    let helper = format!(
        "{PREFIX}shifts = array(cat(delta, fill(0.0, 2)), [2, 1, 2], [1, 2, 3])\n\
         factors = array(cat(fill(a, 2), g, fill(a, 2), fill(1.0, 2)), [2, 2, 2], [1, 2, 3])\n\
         outputs = pyhf.expected_counts(pyhf.sample_yields(rowstack([n1, n2]), shifts, factors))"
    );
    let chain = format!(
        "{PREFIX}outputs = pyhf.expected_counts(rowstack([(n1 .+ delta) .* a .* g, n2 .* a]))"
    );
    assert_eq!(emit(&helper), emit(&chain));
}

#[test]
fn a_one_sample_sum_is_its_chain() {
    let helper = format!(
        "{PREFIX}factors = array(cat(fill(a, 2)), [1, 1, 2], [1, 2, 3])\n\
         outputs = pyhf.expected_counts(pyhf.sample_yields(rowstack([n1]), \
         fill(0.0, [1, 1, 2]), factors))"
    );
    assert_eq!(emit(&helper), emit(&format!("{PREFIX}outputs = n1 .* a")));
}

#[test]
fn another_factor_spelling_keeps_the_block_reduction() {
    let source = format!(
        "{PREFIX}factors = fill(a, [2, 1, 2])\n\
         outputs = pyhf.expected_counts(pyhf.sample_yields(rowstack([n1, n2]), \
         fill(0.0, [2, 1, 2]), factors))"
    );
    assert!(emit(&source).contains("applies stablehlo.multiply"));
}

/// Two samples over two bins with the given scalar factors each. The first
/// sample also carries the per-bin factor `g`; the second pads.
fn scalar_block(scalars: &[&str]) -> String {
    let scalars: String = scalars.iter().map(|x| format!("fill({x}, 2), ")).collect();
    let width = scalars.matches("fill(").count() + 1;
    format!(
        "{PREFIX}factors = array(cat({scalars}g, {scalars}fill(1.0, 2)), [2, {width}, 2], [1, 2, 3])\n\
         outputs = pyhf.expected_counts(pyhf.sample_yields(rowstack([n1, n2]), \
         fill(0.0, [2, 1, 2]), factors))"
    )
}

#[test]
fn gpu_target_scales_the_stacked_templates_once() {
    let source = scalar_block(&["a", "s"]);
    let gpu = emit_for(&source, flatppl_stablehlo::Target::Gpu);
    // One extent-one dot multiplies the [2, 2] stack by the scale column. The
    // per-bin factor meets a constant nominal, so it multiplies natively.
    assert_eq!(gpu.matches("stablehlo.dot_general").count(), 1, "{gpu}");
    assert!(gpu.contains("tensor<2x2x1xf32>"), "{gpu}");
    assert_ne!(gpu, emit(&source));
}

#[test]
fn gpu_target_keeps_per_sample_rows_below_two_scalars_per_sample() {
    let gpu = emit_for(&scalar_block(&["a"]), flatppl_stablehlo::Target::Gpu);
    assert!(!gpu.contains("tensor<2x2x1xf32>"), "{gpu}");
}

#[test]
fn gpu_target_keeps_per_sample_rows_for_one_bin() {
    let source = "pyhf = standard_module(\"pyhf_helpers\", \"0.1\")\n\
        theta = elementof(cartpow(reals, 2))\ninputs = theta\n\
        a = theta[1]\ns = theta[2]\n\
        factors = array(cat(fill(a, 1), fill(s, 1), fill(s, 1), fill(a, 1)), [2, 2, 1], [1, 2, 3])\n\
        outputs = pyhf.expected_counts(pyhf.sample_yields(rowstack([[1.0], [2.0]]), \
        fill(0.0, [2, 1, 1]), factors))";
    let gpu = emit_for(source, flatppl_stablehlo::Target::Gpu);
    assert!(!gpu.contains("tensor<2x1x1xf32>"), "{gpu}");
}

#[test]
fn a_folded_scalar_run_multiplies_factor_by_factor_on_cpu() {
    let helper = format!(
        "{PREFIX}factors = array(cat(fill(mul(a, s), 2), g), [1, 2, 2], [1, 2, 3])\n\
         outputs = pyhf.expected_counts(pyhf.sample_yields(rowstack([n1]), \
         fill(0.0, [1, 1, 2]), factors))"
    );
    assert_eq!(
        emit(&helper),
        emit(&format!("{PREFIX}outputs = n1 .* a .* s .* g"))
    );
}

#[test]
fn class_rows_sum_in_sample_order_as_the_chain() {
    let helper = format!(
        "{PREFIX}y1 = pyhf.sample_yields(rowstack([n1]), fill(0.0, [1, 1, 2]), \
         array(cat(fill(a, 2), g), [1, 2, 2], [1, 2, 3]))\n\
         y2 = pyhf.sample_yields(rowstack([n2]), fill(0.0, [1, 1, 2]), \
         array(cat(fill(s, 2)), [1, 1, 2], [1, 2, 3]))\n\
         outputs = pyhf.expected_counts(rowstack([get(y2, 1, all), get(y1, 1, all)]))"
    );
    let chain =
        format!("{PREFIX}outputs = pyhf.expected_counts(rowstack([n2 .* s, n1 .* a .* g]))");
    assert_eq!(emit(&helper), emit(&chain));
}

#[test]
fn gpu_target_scales_the_rows_of_several_classes_once() {
    let source = format!(
        "{PREFIX}y1 = pyhf.sample_yields(rowstack([n1]), fill(0.0, [1, 1, 2]), \
         array(cat(fill(mul(a, s), 2), g), [1, 2, 2], [1, 2, 3]))\n\
         y2 = pyhf.sample_yields(rowstack([n2]), fill(0.0, [1, 1, 2]), \
         array(cat(fill(mul(s, a), 2)), [1, 1, 2], [1, 2, 3]))\n\
         outputs = pyhf.expected_counts(rowstack([get(y2, 1, all), get(y1, 1, all)]))"
    );
    let gpu = emit_for(&source, flatppl_stablehlo::Target::Gpu);
    assert!(gpu.contains("tensor<2x2x1xf32>"), "{gpu}");
}
