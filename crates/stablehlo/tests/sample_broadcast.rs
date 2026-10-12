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
fn scalar_kernel_broadcast_binds_positional_and_keyword_rows_identically() {
    for (law, constructor, parameters) in [
        ("Normal.(x, 2)", "Normal", "mu = x, sigma = 2"),
        ("StudentT.(x)", "StudentT", "nu = x"),
        ("Gamma.(x, 2)", "Gamma", "shape = x, rate = 2"),
        ("Bernoulli.(x)", "Bernoulli", "p = x"),
    ] {
        let prefix = "x = external(cartpow(reals, 3))\ninputs = x\n";
        let source = format!("{prefix}outputs = rand({law}, rnginit(19))");
        let expected =
            format!("{prefix}outputs = rand(broadcast({constructor}, {parameters}), rnginit(19))");
        assert_eq!(emit(&source), emit(&expected));
    }
}

#[test]
fn primitive_rejection_sampling_does_not_expand_the_row_count() {
    let source = |n| {
        format!(
            "nu = external(cartpow(posreals, {n}))\ninputs = nu\n\
         outputs = rand(StudentT.(nu), rnginit(19))"
        )
    };
    assert_eq!(
        emit(&source(4)).lines().count(),
        emit(&source(400)).lines().count()
    );
}

#[test]
fn guarded_callback_sampling_retains_product_axes() {
    let prefix = "mu = external(cartpow(reals, [2, 1]))\n\
                  sigma = external(cartpow(reals, [1, 3]))\n\
                  nu = external(cartpow(posreals, [2, 3]))\n\
                  inputs = (mu, sigma, nu)\n";
    let source = format!(
        "{prefix}row = (l, s, d) -> normalize(weighted(ifelse(s > 0 && isfinite(s), 1, 0), \
         locscale(StudentT(d), l, ifelse(s > 0 && isfinite(s), s, 1))))\n\
         outputs = rand(row.(mu, sigma, nu), rnginit(19))"
    );
    let inline = source.replace(
        "row.(mu, sigma, nu)",
        "broadcast((l, s, d) -> normalize(weighted(ifelse(s > 0 && isfinite(s), 1, 0), \
         locscale(StudentT(d), l, ifelse(s > 0 && isfinite(s), s, 1)))), mu, sigma, nu)",
    );
    assert_eq!(emit(&source), emit(&inline));
}

#[test]
fn shared_broadcast_law_reuses_the_declared_draw() {
    let prefix = "p = external(cartpow(reals, 2))\ninputs = p\n";
    let source = format!(
        "{prefix}x ~ Normal.(fill(p[1], 3), fill(p[2], 3))\n\
         outputs = rand(lawof(record(a = x, b = x)), rnginit(19))"
    );
    let explicit = format!(
        "{prefix}value, state = rand(Normal.(fill(p[1], 3), fill(p[2], 3)), rnginit(19))\n\
         outputs = (value, value, state)"
    );
    assert_eq!(emit(&source), emit(&explicit));
}

#[test]
fn scalar_row_sampling_does_not_flatten_parameter_event_axes() {
    for (ctor, params) in [
        (
            "MvNormal",
            "mu = [[0, 0], [1, 1]], cov = [[[1, 0], [0, 1]]]",
        ),
        ("Categorical", "p = [[0.2, 0.8], [0.6, 0.4]]"),
        ("Binomial", "n = 3, p = [0.2, 0.8]"),
    ] {
        let mut model = flatppl_syntax::parse(&format!(
            "p = external(reals)\ninputs = p\n\
             outputs = builtin_sample(rnginit(19), broadcasted({ctor}), record({params}))"
        ))
        .unwrap();
        flatppl_infer::infer(&mut model);
        let error = flatppl_stablehlo::emit(
            &model,
            Mode::Sample,
            &EmitOptions {
                restrict_enzyme_compatible: false,
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("broadcast"), "{ctor}: {error}");
    }
}
