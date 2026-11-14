use flatppl_stablehlo::{EmitOptions, Mode, Target};

#[test]
fn thin_qr_keeps_rectangular_workspace_in_native_batches() {
    for (target, n) in [(Target::Cpu, 9), (Target::Gpu, 4)] {
        let mut module = flatppl_syntax::parse(&format!(
            "A = external(cartpow(cartpow(reals, [96, {n}]), 3))\n\
             getq(a) = qr(a).Q\ngetr(a) = qr(a).R\n\
             inputs = A\noutputs = (getq.(A), getr.(A))",
        ))
        .unwrap();
        assert!(flatppl_infer::infer(&mut module).is_empty());
        let roots = [module.intern("inputs"), module.intern("outputs")];
        let lowered = flatppl_determinizer::determinize_with_roots(
            &module,
            &flatppl_infer::ModuleBundle::new(),
            Some(&roots),
        )
        .unwrap();
        let ir = flatppl_stablehlo::emit(
            &lowered,
            Mode::LogDensity,
            &EmitOptions {
                target,
                ..EmitOptions::default()
            },
        )
        .unwrap();
        assert!(ir.contains(&format!("tensor<3x96x{n}xf32>")), "{ir}");
        assert!(ir.contains(&format!("tensor<3x{n}x{n}xf32>")), "{ir}");
        assert!(!ir.contains("96x96xf32>"), "{ir}");
    }
}
