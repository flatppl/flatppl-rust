use flatppl_stablehlo::{EmitOptions, Mode};

fn emit(source: &str) -> String {
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
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn lkj_sampling_matches_cholesky_row_gram_and_successor_state() {
    for n in [1, 3] {
        let prefix = "eta = external(posreals)\n";
        let suffix = "next_value, next_state = rand(Normal(0, 1), state)\n\
                      inputs = eta\noutputs = (value, next_value, next_state)";
        let direct = format!("{prefix}value, state = rand(LKJ({n}, eta), rnginit(23))\n{suffix}");
        let composed = format!(
            "{prefix}factor, state = rand(LKJCholesky({n}, eta), rnginit(23))\n\
             value = row_gram(factor)\n{suffix}"
        );
        assert_eq!(emit(&direct), emit(&composed), "n = {n}");
    }
}

#[test]
fn iid_lkj_keeps_matrix_cells_and_cholesky_rng_consumption() {
    let source = |constructor| {
        format!(
            "eta = external(posreals)\ninputs = eta\n\
             outputs = rand(iid({constructor}(3, eta), [2, 3]), rnginit(23))"
        )
    };
    let direct = emit(&source("LKJ"));
    let cholesky = emit(&source("LKJCholesky"));
    assert!(direct.contains("tensor<2x3x3x3xf32>"), "{direct}");
    assert!(
        direct.contains("batching_dims = [0] x [0], contracting_dims = [2] x [1]"),
        "each matrix contracts within its independent batch lane: {direct}"
    );
    assert_eq!(
        direct.matches("stablehlo.rng_bit_generator").count(),
        cholesky.matches("stablehlo.rng_bit_generator").count(),
        "the Gram adds no random draws"
    );
}
