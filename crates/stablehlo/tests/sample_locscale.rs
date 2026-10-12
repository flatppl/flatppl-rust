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
fn scalar_locscale_samples_match_the_affine_pushforward_and_successor_state() {
    for base in ["Normal(0, 1)", "StudentT(5)", "Bernoulli(0.3)"] {
        let source = format!(
            "shift = external(reals)\nscale = external(reals)\n\
             value, state = rand(locscale({base}, shift, scale), rnginit(23))\n\
             next_value, next_state = rand(Normal(0, 1), state)\n\
             inputs = (shift, scale)\noutputs = (value, next_value, next_state)"
        );
        let expected = source.replace(
            &format!("locscale({base}, shift, scale)"),
            &format!("pushfwd(z -> scale*z + shift, {base})"),
        );
        assert_eq!(emit(&source), emit(&expected), "{base}");
    }
}

#[test]
fn shared_locscale_draws_reuse_the_base_sample() {
    let source = "shift ~ Normal(0, 1)\n\
                  row = l -> locscale(StudentT(5), l, -2)\nvalue ~ row(shift)\n\
                  child ~ Normal(value, 1)\n\
                  prior = lawof(record(shift = shift, value = value, shared = value, child = child))\n\
                  sample, state = rand(prior, rnginit(23))\noutputs = (sample, state)";
    let expected = "shift, state1 = rand(Normal(0, 1), rnginit(23))\n\
                    z, state2 = rand(StudentT(5), state1)\nvalue = -2*z + shift\n\
                    child, state = rand(Normal(value, 1), state2)\n\
                    outputs = (record(shift = shift, value = value, shared = value, child = child), state)";
    assert_eq!(emit(source), emit(expected));
}

#[test]
fn locscale_sampling_retains_callback_row_inputs() {
    let source = "nu = external(posreals)\nshift = external(reals)\nscale = external(reals)\n\
                  row = (d, l, s) -> locscale(StudentT(d), l, s)\n\
                  value, state = rand(row(nu, shift, scale), rnginit(23))\n\
                  inputs = (nu, shift, scale)\noutputs = (value, state)";
    let expected = source.replace(
        "row(nu, shift, scale)",
        "locscale(StudentT(nu), shift, scale)",
    );
    assert_eq!(emit(source), emit(&expected));
}

#[test]
fn scalar_sampling_keeps_shape_and_fresh_draw_boundaries() {
    for measure in [
        "locscale(Normal(0, 1), [1, 2], 2)",
        "locscale(Normal(0, 1), 1, [1, 2])",
        "locscale(MvNormal([0, 0], eye(2)), [1, 2], eye(2))",
    ] {
        let mut model =
            flatppl_syntax::parse(&format!("outputs = rand({measure}, rnginit(23))")).unwrap();
        flatppl_infer::infer(&mut model);
        assert!(flatppl_determinizer::determinize(&model).is_err());
    }
    let mut model = flatppl_syntax::parse(
        "x ~ Normal(0, 1)\n\
         y ~ locscale(pushfwd(exp, lawof(x)), 1, -2)\n\
         outputs = rand(lawof(record(x = x, y = y)), rnginit(23))",
    )
    .unwrap();
    flatppl_infer::infer(&mut model);
    assert!(flatppl_determinizer::determinize(&model).is_err());
}
