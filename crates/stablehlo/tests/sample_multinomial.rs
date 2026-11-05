use flatppl_stablehlo::{EmitOptions, Mode, Target};

fn emit(source: &str, target: Target) -> String {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap();
    flatppl_stablehlo::emit(
        &model,
        Mode::Sample,
        &EmitOptions {
            restrict_enzyme_compatible: false,
            target,
            ..Default::default()
        },
    )
    .unwrap()
}

const SOURCE: &str = "p = external(stdsimplex(3))\ninputs = p\n\
                      outputs = rand(Multinomial(n = 7, p = p), rnginit(0))";

#[test]
fn gpu_multinomial_counts_all_trials_without_a_loop() {
    let module = emit(SOURCE, Target::Gpu);
    assert_eq!(module.matches("stablehlo.while").count(), 0, "{module}");
    assert_eq!(
        module.matches("stablehlo.rng_bit_generator").count(),
        1,
        "{module}"
    );
    // One trial per row, one category per column.
    assert!(module.contains("tensor<7x3xi1>"), "{module}");
}

#[test]
fn cpu_multinomial_keeps_its_trial_loop() {
    let module = emit(SOURCE, Target::Cpu);
    assert_eq!(module.matches("stablehlo.while").count(), 1, "{module}");
}
