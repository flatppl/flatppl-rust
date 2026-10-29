use flatppl_stablehlo::{EmitOptions, Mode, Target};

fn normal(n: usize, target: Target) -> String {
    let source = format!(
        "mu = elementof(cartpow(reals, {n}))\n\
         cov = elementof(cartpow(reals, [{n}, {n}]))\n\
         x = elementof(cartpow(reals, {n}))\n\
         a = draw(MvNormal(mu = mu, cov = cov))\n\
         lp = logdensityof(lawof(record(a = a)), record(a = x))\n\
         inputs = (mu, cov, x)\noutputs = (lp)\n"
    );
    emit(&source, target)
}

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
            target,
            ..Default::default()
        },
    )
    .expect("emit")
}

#[test]
fn gpu_unrolls_small_matrix_loops_only() {
    for n in [2, 8] {
        let gpu = normal(n, Target::Gpu);
        assert!(!gpu.contains("stablehlo.while"), "{gpu}");
        let cpu = normal(n, Target::Cpu);
        assert_eq!(cpu.matches("stablehlo.while").count(), 2, "{cpu}");
    }
    let gpu = normal(9, Target::Gpu);
    assert_eq!(gpu.matches("stablehlo.while").count(), 2, "{gpu}");
}

#[test]
fn pivoted_inverse_keeps_a_tighter_gpu_unroll_limit() {
    for (n, loops) in [(4, 0), (5, 2)] {
        let source =
            format!("a = external(cartpow(reals, [{n}, {n}]))\ninputs = a\noutputs = inv(a)\n");
        let gpu = emit(&source, Target::Gpu);
        assert_eq!(gpu.matches("stablehlo.while").count(), loops, "{gpu}");
    }
}
