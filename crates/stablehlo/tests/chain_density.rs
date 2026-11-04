use flatppl_stablehlo::{EmitOptions, Mode};

fn emit(source: &str) -> String {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_roots(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
    )
    .unwrap();
    flatppl_stablehlo::emit(&model, Mode::LogDensity, &EmitOptions::default()).unwrap()
}

fn markov_scalar(n: usize) -> String {
    format!(
        "d = elementof(reals)\n\
         traj = elementof(cartpow(reals, {n}))\n\
         step(x, d) = Normal(x + d, 1.0)\n\
         inputs = (d, traj)\n\
         outputs = logdensityof(markovchain(step, 0.0, {n}, d), traj)"
    )
}

fn kscan_mvnormal(n: usize) -> String {
    format!(
        "a = elementof(reals)\n\
         traj = elementof(cartpow(cartpow(reals, 2), {n}))\n\
         xs = elementof(cartpow(reals, {n}))\n\
         step(x, u, a) = MvNormal(x .+ a * u, rowstack([[1.0, 0.2], [0.2, 1.0]]))\n\
         inputs = (a, traj, xs)\n\
         outputs = logdensityof(kscan(step, fill(0.0, 2), xs, a), traj)"
    )
}

#[test]
fn chain_density_program_size_does_not_grow_with_length() {
    for model in [markov_scalar, kscan_mvnormal] {
        let short = emit(&model(10));
        let long = emit(&model(1000));
        assert!(!long.contains("stablehlo.while"), "{long}");
        assert_eq!(
            short.lines().count(),
            long.lines().count(),
            "{short}\n{long}"
        );
    }
}
