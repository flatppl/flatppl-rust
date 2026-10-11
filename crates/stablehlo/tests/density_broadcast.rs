use flatppl_core::Module;

fn emit(source: &str) -> String {
    let mut model: Module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_roots(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
    )
    .unwrap();
    flatppl_stablehlo::emit(
        &model,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .unwrap()
}

#[test]
fn fixed_measure_broadcast_preserves_each_density_and_its_sum() {
    let source = "x = external(cartpow(reals, 3))\n\
                  terms = logdensityof.(Normal(0, 1), x)\n\
                  inputs = x\noutputs = (terms, sum(terms))";
    assert_eq!(
        emit(source),
        emit(&source.replace(
            "logdensityof.(Normal(0, 1), x)",
            "fn(logdensityof(Normal(0, 1), _)).(x)",
        )),
    );
}

#[test]
fn density_broadcast_keeps_mixed_scalar_and_collection_operands() {
    let source = "x = external(cartpow(reals, 3))\n\
                  shift = external(reals)\n\
                  inputs = (x, shift)\n\
                  outputs = densityof.(Normal(shift, 2), x .+ shift)";
    assert_eq!(
        emit(source),
        emit(&source.replace(
            "densityof.(Normal(shift, 2), x .+ shift)",
            "fn(densityof(Normal(shift, 2), _)).(x .+ shift)",
        )),
    );
}

#[test]
fn ordered_prior_applies_the_sum_to_the_whole_variate() {
    let source = "x = external(cartpow(reals, 8))\n\
                  prior = logweighted(fn(sum(logdensityof.(Normal(0, 5), _))), \
                      Lebesgue(support = ordered(8)))\n\
                  inputs = x\noutputs = logdensityof(prior, x)";
    assert_eq!(
        emit(source),
        emit(&source.replace(
            "logdensityof.(Normal(0, 5), _)",
            "broadcast(v -> logdensityof(Normal(0, 5), v), _)",
        )),
    );
}

#[test]
fn broadcast_affine_pushforward_retains_shared_runtime_parameters() {
    let source = "mu = external(reals)\n\
                  tau = external(posreals)\n\
                  x = external(cartpow(reals, 8))\n\
                  M = pushfwd(z -> (z .* tau) .+ mu, iid(Normal(0, 1), 8))\n\
                  inputs = (mu, tau, x)\noutputs = logdensityof(M, x)";
    let expected = emit(source);
    assert_eq!(
        expected,
        emit(&source.replace(
            "(z .* tau) .+ mu",
            "broadcast(add, broadcast(mul, z, tau), mu)"
        )),
    );
    assert_eq!(
        expected,
        emit(&source.replace("(z .* tau) .+ mu", "mu .+ (tau .* z)"))
    );
}
