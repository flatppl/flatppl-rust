use flatppl_stablehlo::{Dtype, EmitOptions, Mode, Target};

#[test]
fn callable_newton_with_nested_scans_emits_for_both_targets_and_precisions() {
    let src = include_str!("fixtures/callable_newton_scan.flatppl");
    let mut model = flatppl_syntax::parse(src).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let lowered = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .expect("determinize");
    for target in [Target::Cpu, Target::Gpu] {
        for dtype in [Dtype::F32, Dtype::F64] {
            let ir = flatppl_stablehlo::emit(
                &lowered,
                Mode::LogDensity,
                &EmitOptions {
                    target,
                    dtype,
                    ..Default::default()
                },
            )
            .expect("emit nested solve");
            assert!(ir.contains("stablehlo.while"));
        }
    }
}

fn lower_and_emit(src: &str) -> Result<String, String> {
    let mut model = flatppl_syntax::parse(src).map_err(|e| format!("{e:?}"))?;
    let _ = flatppl_infer::infer(&mut model);
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let lowered = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .map_err(|e| e.reason)?;
    flatppl_stablehlo::emit(&lowered, Mode::LogDensity, &EmitOptions::default())
        .map_err(|e| e.to_string())
}

#[test]
fn rowstack_of_callable_scan_selects_a_matrix_column() {
    let src = include_str!("fixtures/callable_newton_scan.flatppl")
        .replace("outputs = solve(", "trajectory = solve(")
        + "\noutputs = rowstack(trajectory)[:, 1]\n";
    // §07 rowstack turns the nested scan vectors into one 3-by-1 matrix;
    // selecting its only column must preserve all three trajectory entries.
    let ir = lower_and_emit(&src).expect("select the stacked scan column");
    assert!(ir.contains("[0:3, 0:1]"), "{ir}");
    assert!(ir.contains("-> tensor<3xf32>"), "{ir}");
}

#[test]
fn value_only_tuple_with_two_live_leaves_terminates() {
    let src = "x=elementof(cartpow(reals,2))\n\
        step(a,i,p)=a+i*p[1]+p[2]\n\
        inputs=x\noutputs=scan(step,0.0,[1.0,2.0],(x[1],x[2]))";
    let ir = lower_and_emit(src).expect("emit a value-only parameter tuple");
    assert!(ir.contains("stablehlo.while"), "{ir}");
}

#[test]
fn callback_cannot_change_state_shape() {
    let src = "x=elementof(reals)\n\
        f(t,y,p)=[p,p]\n\
        step(a,i,q)=q[1](i,a,q[2])\n\
        solve(f,y,p)=scan(step,y,[1.0,2.0],(f,p))\n\
        inputs=x\noutputs=solve(f,[x],x)";
    let err = lower_and_emit(src).expect_err("different state shapes must be refused");
    assert!(err.contains("scan: step changes the state type"), "{err}");
}
