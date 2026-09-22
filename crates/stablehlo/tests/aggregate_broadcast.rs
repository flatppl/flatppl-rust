//! Aggregation contracts model axes, never the enclosing callable's batch axes.

fn emit(source: &str) -> String {
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
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect("aggregate inside a callable broadcast must emit")
}

#[test]
fn batched_aggregation_keeps_the_batch_and_output_axes() {
    let ir = emit(
        "flatppl_compat = \"0.1\"\n\
         matrices = elementof(cartpow(cartpow(reals, [2, 3]), 4))\n\
         columns(a) = aggregate(sum, [.j], a[.i, .j])\n\
         inputs = (matrices)\noutputs = columns.(matrices)\n",
    );
    assert!(ir.contains("-> tensor<4x3xf32>"), "{ir}");
    assert_eq!(ir.matches("stablehlo.reduce(").count(), 1, "{ir}");
    assert!(ir.contains("across dimensions = [2]"), "{ir}");
}

#[test]
fn nested_batched_variance_keeps_fixed_selectors_and_captured_operands() {
    let ir = emit(
        "flatppl_compat = \"0.1\"\n\
         matrices = elementof(cartpow(cartpow(cartpow(reals, [2, 3]), 4), 5))\n\
         weights = elementof(cartpow(reals, 3))\n\
         variance(a) = aggregate(var, [], a[2, .j] * weights[.j])\n\
         group(xs) = variance.(xs)\n\
         inputs = (matrices, weights)\noutputs = group.(matrices)\n",
    );
    assert!(ir.contains("-> tensor<5x4xf32>"), "{ir}");
    assert_eq!(ir.matches("stablehlo.reduce(").count(), 2, "{ir}");
}

#[test]
fn stack_constructors_change_cell_axes_not_batch_axes() {
    for (constructor, expected) in [("rowstack", "4x2x3"), ("colstack", "4x3x2")] {
        let ir = emit(&format!(
            "flatppl_compat = \"0.1\"\n\
             rows = elementof(cartpow(cartpow(reals, 3), 4))\n\
             stack(x) = {constructor}([x, x .+ 1.0])\n\
             inputs = (rows)\noutputs = stack.(rows)\n"
        ));
        assert!(ir.contains(&format!("-> tensor<{expected}xf32>")), "{ir}");
    }
}

#[test]
fn addaxes_inserts_leading_axes_after_the_batch_prefix() {
    let ir = emit(
        "flatppl_compat = \"0.1\"\n\
         rows = elementof(cartpow(cartpow(reals, 3), 4))\n\
         as_row(x) = addaxes(x, 1, 0)\n\
         inputs = (rows)\noutputs = as_row.(rows)\n",
    );
    assert!(ir.contains("-> tensor<4x1x3xf32>"), "{ir}");
}

#[test]
fn a_shared_invariant_broadcast_keeps_each_callers_batch_shape() {
    let ir = emit(
        r#"
x = elementof(reals)
a = external(reals)
rates = broadcast(z -> exp(z), [x, x + 1.0])
L = likelihoodof(Normal(rates[1] + rates[2], 1.0), 0.0)
f = t -> logdensityof(L, record(x = a)) + t
o1 = broadcast(f, [1.0, 2.0, 3.0])
o2 = broadcast(f, [1.0, 2.0, 3.0, 4.0])
inputs = a
outputs = (o1, o2)
"#,
    );
    assert!(ir.contains("-> (tensor<3xf32>, tensor<4xf32>)"), "{ir}");
}
