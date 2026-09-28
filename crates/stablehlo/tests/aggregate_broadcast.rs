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
fn independent_reductions_pack_without_reducing_the_batch_axis() {
    let ir = emit(
        r#"
flatppl_compat = "0.1"
a = elementof(cartpow(cartpow(reals, 4), 3))
b = elementof(cartpow(cartpow(reals, 4), 3))
mapped(x) = exp.(sin.(x) .+ cos.(x)) ./ (1.0 .+ exp.(-x))
combined(x, y) = sum(mapped(x)) + sum(mapped(y))
inputs = (a, b)
outputs = combined.(a, b)
"#,
    );
    assert_eq!(ir.matches("stablehlo.reduce(").count(), 1, "{ir}");
    assert!(ir.contains("across dimensions = [2]"), "{ir}");
    assert!(ir.contains("tensor<2x3x4xf32>"), "{ir}");
    assert!(ir.contains("-> tensor<3xf32>"), "{ir}");
}

#[test]
fn batched_input_vector_uses_one_ordered_gather() {
    let ir = emit(
        r#"
select(x) = [x[4], x[1], x[4], x[2], x[1]]
mapped(x) = select.(x)
points = elementof(cartpow(cartpow(cartpow(reals, 7), 3), 2))
inputs = points
outputs = mapped.(points)
"#,
    );
    assert_eq!(ir.matches("\"stablehlo.gather\"").count(), 1, "{ir}");
    assert!(!ir.contains("stablehlo.concatenate"), "{ir}");
    assert!(ir.contains("-> tensor<2x3x5xf32>"), "{ir}");
}

#[test]
fn large_static_input_selections_do_not_truncate_to_i32() {
    let ir = emit(
        r#"
select(x) = [x[2147483649], x[1]]
points = elementof(cartpow(cartpow(reals, 2147483649), 2))
inputs = points
outputs = select.(points)
"#,
    );
    assert!(ir.contains("[0:2, 2147483648:2147483649]"), "{ir}");
    assert!(!ir.contains("\"stablehlo.gather\""), "{ir}");
}

#[test]
fn unequal_segment_sums_share_one_reduction() {
    let ir = emit(
        r#"
flatppl_compat = "0.1"
planes(a) = aggregate(sum, [.k, .j], a[.i, .j, .k])
sums(x) = [planes(x[[1, 3, 1], :, :]), planes(x[[7, 2], :, :]), planes(x[[4], :, :])]
points = elementof(cartpow(cartpow(reals, [7, 2, 3]), 3))
inputs = points
outputs = sums.(points)
"#,
    );
    assert_eq!(ir.matches("stablehlo.reduce(").count(), 1, "{ir}");
    assert_eq!(ir.matches("\"stablehlo.gather\"").count(), 1, "{ir}");
}

#[test]
fn computed_selection_sums_and_products_keep_distinct_identities() {
    let ir = emit(
        r#"
segments(x) = [prod([x[1], x[3], x[1]]), prod([x[7], x[2]]),
               sum([x[4], x[5]]), sum([x[6], x[2], x[6]])]
mapped(x) = segments(-x)
points = elementof(cartpow(cartpow(reals, 7), 3))
inputs = points
outputs = mapped.(points)
"#,
    );
    assert_eq!(ir.matches("applies stablehlo.add").count(), 1, "{ir}");
    assert_eq!(ir.matches("applies stablehlo.multiply").count(), 1, "{ir}");
    assert_eq!(ir.matches("stablehlo.slice").count(), 4, "{ir}");
}

#[test]
fn segment_packing_retains_dynamic_shared_splat_and_overpadded_selections() {
    for (outputs, count) in [
        ("(sum(x[index]), sum(x[[2, 3]]))", 2),
        ("(sum(selected), sum(x[[2, 3]]), selected)", 2),
        ("(sum(x[fill(1, 3)]), sum(x[[2, 3]]))", 2),
        (
            "(sum(x[[1]]), sum(x[[2, 3]]), sum(x[[1, 2, 3, 4, 5, 6, 7]]))",
            3,
        ),
    ] {
        let ir = emit(&format!(
            "x = elementof(cartpow(reals, 7))\n\
             index = elementof(cartpow(posintegers, 3))\n\
             selected = x[[1, 4, 1]]\n\
             inputs = (x, index)\noutputs = {outputs}\n"
        ));
        assert_eq!(ir.matches("stablehlo.reduce(").count(), count, "{ir}");
    }
}

#[test]
fn sliced_consumers_pack_within_each_source_tensor() {
    let ir = emit(
        r#"
flatppl_compat = "0.1"
x = elementof(cartpow(reals, 4))
y = elementof(cartpow(reals, 5))
a = exp.(x)
b = exp.(y)
f(z) = ((z * z + 1.0) * z + 2.0) * z + 3.0
inputs = (x, y)
outputs = (f(a[1]) + f(a[2]) + f(a[3]) + f(a[4]), f(b[1]) + f(b[2]) + f(b[3]) + f(b[4]))
"#,
    );
    let multiplies: Vec<_> = ir
        .lines()
        .filter(|line| line.contains("stablehlo.multiply"))
        .collect();
    assert_eq!(multiplies.len(), 6, "{ir}");
    assert!(
        multiplies
            .iter()
            .all(|line| line.ends_with("tensor<4xf32>")),
        "{ir}"
    );
}

#[test]
fn singleton_broadcasts_share_irregular_gathers() {
    let source = r#"
score(t) = sum(
    2.0 * [t[26]] * exp(t[1]) * exp(t[3])
  + 3.0 * [t[26]] * exp(t[3]) * exp(t[8])
  + 4.0 * [t[26]] * exp(t[7]) * exp(t[11])
  + 5.0 * [t[26]] * exp(t[10]) * exp(t[13])
  + 6.0 * [t[26]] * exp(t[17]) * exp(t[10])
  + 7.0 * [t[26]] * exp(t[20]) * exp(t[3])
  + 8.0 * [t[26]] * exp(t[22]) * exp(t[24])
  + 9.0 * [t[26]] * exp(t[25]) * exp(t[23])
)
points = elementof(cartpow(cartpow(reals, 26), 3))
inputs = points
outputs = score.(points)
"#;
    let ir = emit(source);
    // One gather selects exponential inputs. The second shares both irregular
    // selections, including repeated and out-of-order lanes, before broadcasting.
    assert_eq!(ir.matches("\"stablehlo.gather\"").count(), 2, "{ir}");
    // Arithmetic consumers do not meet the view-only profitability condition.
    let scalar = emit(&source.replace("[t[26]]", "t[26]"));
    assert_eq!(
        scalar.matches("\"stablehlo.gather\"").count(),
        3,
        "{scalar}"
    );
}

#[test]
fn singleton_broadcasts_share_short_selections() {
    let source = r#"
score(t) = sum(
    2.0 * [t[12]] * exp(t[1]) * exp(t[3])
  + 3.0 * [t[12]] * exp(t[3]) * exp(t[8])
  + 4.0 * [t[12]] * exp(t[7]) * exp(t[11])
)
points = elementof(cartpow(cartpow(reals, 12), 3))
inputs = points
outputs = score.(points)
"#;
    let ir = emit(source);
    assert_eq!(ir.matches("\"stablehlo.gather\"").count(), 2, "{ir}");
    let scalar = emit(&source.replace("[t[12]]", "t[12]"));
    assert_eq!(
        scalar.matches("\"stablehlo.gather\"").count(),
        1,
        "{scalar}"
    );
}

#[test]
fn short_selection_sharing_keeps_large_static_indices() {
    let source = r#"
score(t) = sum(
    2.0 * [t[2]] * t[2147483649] * t[3]
  + 3.0 * [t[2]] * t[3] * t[8]
  + 4.0 * [t[2]] * t[7] * t[11]
)
points = elementof(cartpow(cartpow(reals, 2147483649), 3))
inputs = points
outputs = score.(points)
"#;
    let small = emit(&source.replace("2147483649", "12"));
    assert_eq!(small.matches("\"stablehlo.gather\"").count(), 1, "{small}");
    let ir = emit(source);
    assert!(ir.contains("2147483648:2147483649"), "{ir}");
    assert!(!ir.contains("\"stablehlo.gather\""), "{ir}");
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
