use flatppl_stablehlo::{Dtype, EmitOptions, Mode};

fn emit_as(source: &str, dtype: Dtype) -> String {
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
            dtype,
            ..Default::default()
        },
    )
    .unwrap()
}

fn emit(source: &str) -> String {
    emit_as(source, Dtype::F32)
}

#[test]
fn positional_named_and_mixed_parameters_emit_one_mixture() {
    let prefix = "w = external(cartpow(nonnegreals, 2))\n\
                  mu = external(cartpow(reals, 2))\n\
                  sigma = external(cartpow(posreals, 2))\ninputs = (w, mu, sigma)\n";
    let source = |parameters| {
        format!(
            "{prefix}outputs = rand(iid(normalize(ksuperpose(Normal, w, {parameters})), 8), \
             rnginit(5))"
        )
    };
    let named = emit(&source("mu = mu, sigma = sigma"));
    assert_eq!(emit(&source("mu, sigma")), named);
    assert_eq!(emit(&source("mu, sigma = sigma")), named);
}

#[test]
fn lane_parameters_are_gathered_without_a_component_axis() {
    let source = |n| {
        format!(
            "w = [1.0, 2.0, 3.0]\n\
             outputs = rand(iid(normalize(ksuperpose(Normal, w, mu = [-3.0, 0.0, 3.0], \
             sigma = [1.0, 0.5, 2.0])), {n}), rnginit(5))"
        )
    };
    let large = emit(&source(1000));
    assert!(!large.contains("tensor<1000x3x"), "{large}");
    assert_eq!(
        emit(&source(10)).lines().count(),
        large.lines().count(),
        "the lane count does not unroll"
    );
}

#[test]
fn single_and_iid_mixtures_emit_in_both_precisions() {
    for law in [
        "normalize(ksuperpose(Normal, w, mu = mu, sigma = 1.0))",
        "iid(normalize(ksuperpose(Normal, w, mu = mu, sigma = 1.0)), [2, 3])",
        "normalize(ksuperpose(Poisson, w, rate = exp.(mu)))",
        "normalize(ksuperpose(MvNormal, w, mu = [[0.0, 0.0], [3.0, 3.0]], cov = [eye(2)]))",
        "iid(normalize(ksuperpose(Dirac, w, value = mu)), 4)",
    ] {
        let source = format!(
            "w = external(cartpow(nonnegreals, 2))\nmu = external(cartpow(reals, 2))\n\
             inputs = (w, mu)\noutputs = rand({law}, rnginit(5))"
        );
        for (dtype, real) in [(Dtype::F32, "f32"), (Dtype::F64, "f64")] {
            let module = emit_as(&source, dtype);
            assert!(module.contains(&format!("x{real}>")), "{law}: {module}");
        }
    }
}

/// §08 draws only from `p` in `stdsimplex(n)`. Runtime weights that leave the
/// simplex report status 7 rather than a plausible index.
#[test]
fn runtime_weights_report_a_status_and_constant_probabilities_do_not() {
    let runtime = emit(
        "w = external(cartpow(reals, 2))\ninputs = w\n\
         outputs = rand(normalize(ksuperpose(Normal, w, mu = [-3.0, 3.0], sigma = 1.0)), \
         rnginit(5))",
    );
    assert!(runtime.contains("flatppl.runtime_status"), "{runtime}");
    assert!(runtime.contains("dense<7>"), "{runtime}");
    let constant = emit("outputs = rand(Categorical(p = [0.25, 0.75]), rnginit(5))");
    assert!(!constant.contains("flatppl.runtime_status"), "{constant}");
}
