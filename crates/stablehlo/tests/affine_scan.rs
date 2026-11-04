use flatppl_stablehlo::{Dtype, EmitOptions, Mode, Target};

fn emit(source: &str, target: Target) -> String {
    let mut module = flatppl_syntax::parse(source).expect("parse");
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots: Vec<_> = ["inputs", "outputs"].map(|name| module.intern(name)).into();
    let lowered = flatppl_determinizer::determinize_with_roots(
        &module,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
    )
    .expect("determinize");
    flatppl_stablehlo::emit(
        &lowered,
        Mode::LogDensity,
        &EmitOptions {
            dtype: Dtype::F64,
            target,
            ..Default::default()
        },
    )
    .expect("emit")
}

fn arma11(t: usize) -> String {
    format!(
        "y = elementof(cartpow(reals, {t}))\n\
         mu = elementof(reals)\n\
         phi = elementof(reals)\n\
         theta = elementof(reals)\n\
         sigma = elementof(posreals)\n\
         prediction(state, mu, phi, theta) = mu + phi * state.previous + theta * state.error\n\
         step(state, observation, mu, phi, theta) = record(\n    \
             previous = observation, error = observation - prediction(state, mu, phi, theta))\n\
         states = scan(step, record(previous = mu, error = 0.0), y, mu, phi, theta)\n\
         state_error(state) = state.error\n\
         errors = state_error.(states)\n\
         inputs = (mu, phi, theta, sigma, y)\n\
         outputs = sum(-0.5 .* (errors ./ sigma) .^ 2) - {t} * log(sigma)\n"
    )
}

/// A scalar-carry scan over `y` with the given step body in `s` and `x`.
fn scalar_scan(step: &str) -> String {
    format!(
        "y = elementof(cartpow(reals, 6))\n\
         a = elementof(reals)\n\
         step(s, x, a) = {step}\n\
         states = scan(step, 0.5, y, a)\n\
         inputs = (a, y)\n\
         outputs = sum(states)\n"
    )
}

#[test]
fn gpu_arma11_residuals_use_a_doubling_scan() {
    let ir = emit(&arma11(200), Target::Gpu);
    assert!(!ir.contains("stablehlo.while"), "{ir}");
    // ceil(log2 200) = 8 levels, one shift of the residuals each, plus the
    // shifted observations that feed the moving-average term.
    assert_eq!(ir.matches("stablehlo.concatenate").count(), 9, "{ir}");
}

#[test]
fn cpu_arma11_keeps_the_loop() {
    let ir = emit(&arma11(200), Target::Cpu);
    assert_eq!(ir.matches("stablehlo.while").count(), 1, "{ir}");
    assert!(!ir.contains("stablehlo.concatenate"), "{ir}");
}

#[test]
fn gpu_affine_scalar_steps_use_a_doubling_scan() {
    for step in ["s + x", "a * s - x", "x - s / a", "-(s * a) + 2.0 * x"] {
        let ir = emit(&scalar_scan(step), Target::Gpu);
        assert!(!ir.contains("stablehlo.while"), "{step}: {ir}");
    }
}

#[test]
fn gpu_non_affine_steps_keep_the_loop() {
    for step in ["s * s + x", "x * exp(s)", "s * (a + s)", "x / s"] {
        let ir = emit(&scalar_scan(step), Target::Gpu);
        assert_eq!(ir.matches("stablehlo.while").count(), 1, "{step}: {ir}");
        assert!(!ir.contains("stablehlo.concatenate"), "{step}: {ir}");
    }
}

#[test]
fn gpu_coupled_record_fields_keep_the_loop() {
    // Each field is affine, but together they form a 2x2 linear map.
    let ir = emit(
        "y = elementof(cartpow(reals, 6))\n\
         step(s, x) = record(u = s.v + x, v = 0.5 * s.u)\n\
         states = scan(step, record(u = 0.0, v = 1.0), y)\n\
         field_u(s) = s.u\n\
         inputs = (y)\n\
         outputs = sum(field_u.(states))\n",
        Target::Gpu,
    );
    assert_eq!(ir.matches("stablehlo.while").count(), 1, "{ir}");
}
