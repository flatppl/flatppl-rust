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
fn shared_broadcast_draws_keep_one_sample_and_successor_key() {
    let prefix = "p = external(cartpow(reals, 2))\n\
                  row = (l, s) -> locscale(StudentT(8), l, s)\n\
                  inputs = p\n";
    let source = format!(
        "{prefix}mu ~ Normal(0, 1)\nsigma ~ Normal(0, 1)\n\
         x ~ row.(fill(mu, 3), fill(sigma, 3))\n\
         K = kernelof(record(a = x, b = x), mu = mu, sigma = sigma)\n\
         outputs = rand(K(p[1], p[2]), rnginit(19))"
    );
    let explicit = format!(
        "{prefix}value, state = rand(row.(fill(p[1], 3), fill(p[2], 3)), rnginit(19))\n\
         outputs = (value, value, state)"
    );
    assert_eq!(emit(&source), emit(&explicit));
}

#[test]
fn applied_kernel_reuses_shared_ancestors_and_keeps_fresh_calls() {
    let prefix = "p = external(cartpow(reals, 2))\ninputs = p\n";
    let source = format!(
        "{prefix}mu ~ Normal(0, 1)\nsigma ~ Normal(0, 1)\n\
         x ~ Normal(mu, sigma)\ny ~ Normal(x, 1)\n\
         K = kernelof(record(a = x, b = x, y = y), mu = mu, sigma = sigma)\n\
         first, state1 = rand(K(p[1], p[2]), rnginit(19))\n\
         second, state2 = rand(K(p[1] + 1, p[2]), state1)\n\
         third, state = rand(K(p[1], p[2]), state2)\n\
         outputs = (first, second, third, state)"
    );
    let explicit = format!(
        "{prefix}x1, state1 = rand(Normal(p[1], p[2]), rnginit(19))\n\
         y1, state2 = rand(Normal(x1, 1), state1)\n\
         x2, state3 = rand(Normal(p[1] + 1, p[2]), state2)\n\
         y2, state4 = rand(Normal(x2, 1), state3)\n\
         x3, state5 = rand(Normal(p[1], p[2]), state4)\n\
         y3, state = rand(Normal(x3, 1), state5)\n\
         outputs = (record(a = x1, b = x1, y = y1), record(a = x2, b = x2, y = y2),\n\
                    record(a = x3, b = x3, y = y3), state)"
    );
    assert_eq!(emit(&source), emit(&explicit));
}

#[test]
fn local_kernel_wrapper_preserves_shared_draws() {
    let prefix = "p = external(reals)\ninputs = p\n";
    let source = format!(
        "{prefix}mu ~ Normal(0, 1)\nx ~ Normal(mu, 1)\n\
         K = kernelof(record(a = x, b = x), mu = mu)\n\
         F(m) = K(m)\n\
         outputs = rand(F(p), rnginit(19))"
    );
    let explicit = format!(
        "{prefix}value, state = rand(Normal(p, 1), rnginit(19))\n\
         outputs = (value, value, state)"
    );
    assert_eq!(emit(&source), emit(&explicit));
}
