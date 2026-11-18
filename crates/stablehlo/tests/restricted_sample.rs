use flatppl_stablehlo::{EmitOptions, Mode};

fn lower(source: &str) -> flatppl_core::Module {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap()
}

fn emit(model: &flatppl_core::Module) -> String {
    flatppl_stablehlo::emit(
        model,
        Mode::LogDensity,
        &EmitOptions {
            restrict_enzyme_compatible: false,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn normal_restriction_admits_iid_and_threads_rng_between_samples() {
    let model = lower(
        "lo = external(reals)\nhi = external(reals)\n\
         law = normalize(truncate(Normal(1, 2), interval(lo, hi)))\n\
         first, state1 = rand(law, rnginit(109))\n\
         second, _ = rand(iid(law, [2, 3]), state1)\n\
         inputs = (lo, hi)\noutputs = (first, second)",
    );
    let module = emit(&model);
    assert!(
        module.contains("-> (tensor<f32>, tensor<2x3xf32>"),
        "{module}"
    );
    assert_eq!(module.matches("stablehlo.while").count(), 2);
}

#[test]
fn hurdle_count_normalization_preserves_integer_outcomes_and_row_axes() {
    for law in ["Poisson(rate)", "NegativeBinomial(0.2, 0.2 / rate)"] {
        let model = lower(&format!(
            "rate = external(cartpow(posreals, [2, 3]))\n\
             gate = external(cartpow(unitinterval, [2, 3]))\n\
             row(rate, h) = normalize(superpose(\n\
                 weighted(h, Dirac(0)),\n\
                 weighted(1 - h, normalize(truncate({law}, interval(1, inf))))))\n\
             result, _ = rand(row.(rate, gate), rnginit(7))\n\
             inputs = (rate, gate)\noutputs = result"
        ));
        let module = emit(&model);
        assert!(module.contains("-> (tensor<2x3xi32>"), "{module}");
        assert!(flatppl_determinizer::is_flatpdl(&model).is_err());
    }
}
