//! Numeric multi-axis subset selection keeps array axes and callable batches.

fn emit(source: &str) -> Result<String, flatppl_stablehlo::EmitError> {
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
        &flatppl_stablehlo::EmitOptions {
            restrict_enzyme_compatible: false,
            ..Default::default()
        },
    )
}

#[test]
fn row_index_vector_and_all_select_matrix_rows_in_order() {
    let ir = emit(
        "flatppl_compat = \"0.1\"\n\
         matrix = elementof(cartpow(reals, [4, 3]))\n\
         selected = get(matrix, [3, 1, 3], all)\n\
         inputs = (matrix)\noutputs = (selected)\n",
    )
    .expect("matrix subset selection must emit");

    assert!(ir.contains("-> tensor<3x3xf32>"), "{ir}");
    assert_eq!(ir.matches("\"stablehlo.gather\"").count(), 1, "{ir}");
    assert!(ir.contains("dense<[2, 0, 2]> : tensor<3xi32>"), "{ir}");
    assert!(
        ir.contains("offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0]"),
        "{ir}"
    );
    assert!(ir.contains("slice_sizes = array<i64: 1, 3>"), "{ir}");
}

#[test]
fn rowstack_binding_is_indexed_as_a_flat_matrix() {
    let ir = emit(
        "flatppl_compat = \"0.1\"\n\
         first = elementof(cartpow(reals, 3))\n\
         second = elementof(cartpow(reals, 3))\n\
         shifts = rowstack([first, second, first, second])\n\
         selected = get(shifts, [3, 1, 3], all)\n\
         inputs = (first, second)\noutputs = (selected)\n",
    )
    .expect("rowstack establishes one flat matrix cell");

    assert!(ir.contains("-> tensor<3x3xf32>"), "{ir}");
    assert_eq!(ir.matches("\"stablehlo.gather\"").count(), 1, "{ir}");
}

#[test]
fn matrix_selection_flows_through_batched_aggregation() {
    let ir = emit(
        "flatppl_compat = \"0.1\"\n\
         matrices = elementof(cartpow(cartpow(reals, [4, 3]), 2))\n\
         columns(matrix) = aggregate(sum, [.col], get(matrix, [3, 1, 3], all)[.row, .col])\n\
         inputs = (matrices)\noutputs = columns.(matrices)\n",
    )
    .expect("selection inside a callable aggregation must emit");

    assert!(ir.contains("-> tensor<2x3xf32>"), "{ir}");
    assert_eq!(ir.matches("\"stablehlo.gather\"").count(), 1, "{ir}");
    assert!(
        ir.contains("offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1]"),
        "{ir}"
    );
    assert!(ir.contains("slice_sizes = array<i64: 2, 1, 3>"), "{ir}");
    assert!(ir.contains("across dimensions = [2]"), "{ir}");
}

#[test]
fn vector_selector_can_replace_a_later_axis() {
    let ir = emit(
        "flatppl_compat = \"0.1\"\n\
         matrix = elementof(cartpow(reals, [4, 3]))\n\
         selected = get(matrix, all, [3, 1])\n\
         inputs = (matrix)\noutputs = (selected)\n",
    )
    .expect("column subset selection must emit");

    assert!(ir.contains("-> tensor<4x2xf32>"), "{ir}");
    assert!(
        ir.contains("offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1]"),
        "{ir}"
    );
    assert!(ir.contains("slice_sizes = array<i64: 4, 1>"), "{ir}");
}

#[test]
fn regular_selection_uses_a_slice_with_exact_last_element() {
    let source = "matrices = elementof(cartpow(cartpow(reals, [4, 7]), 2))\n\
         pick(matrix) = get(matrix, all, [1, 4])\n\
         inputs = matrices\noutputs = pick.(matrices)\n";
    let ir = emit(source).expect("regular column selection must emit");

    assert_eq!(ir.matches("stablehlo.slice").count(), 1, "{ir}");
    assert!(ir.contains("[0:2, 0:4, 0:4:3]"), "{ir}");
    assert!(ir.contains("-> tensor<2x4x2xf32>"), "{ir}");
    assert!(!ir.contains("\"stablehlo.gather\""), "{ir}");

    let irregular = emit(&source.replace("[1, 4]", "[1, 3, 6]")).unwrap();
    assert_eq!(
        irregular.matches("\"stablehlo.gather\"").count(),
        1,
        "{irregular}"
    );
    assert!(!irregular.contains("stablehlo.slice"), "{irregular}");
}

#[test]
fn repeated_contiguous_selection_broadcasts_whole_blocks() {
    let source = "matrices = elementof(cartpow(cartpow(reals, [5, 3]), 2))\n\
         pick(matrix) = get(matrix, [2, 3, 2, 3], all)\n\
         inputs = matrices\noutputs = pick.(matrices)\n";
    let ir = emit(source).expect("repeated row selection must emit");

    assert!(ir.contains("[0:2, 1:3, 0:3]"), "{ir}");
    assert!(ir.contains("dims = [0, 2, 3]"), "{ir}");
    assert!(ir.contains("-> tensor<2x2x2x3xf32>"), "{ir}");
    assert!(ir.contains("-> tensor<2x4x3xf32>"), "{ir}");
    assert!(!ir.contains("\"stablehlo.gather\""), "{ir}");

    let irregular = emit(&source.replace("[2, 3, 2, 3]", "[2, 3, 2, 4]")).unwrap();
    assert_eq!(
        irregular.matches("\"stablehlo.gather\"").count(),
        1,
        "{irregular}"
    );
}

#[test]
fn literal_only_and_all_slice_and_collapse_axes() {
    let ir = emit(
        "flatppl_compat = \"0.1\"\n\
         tensor = elementof(cartpow(reals, [2, 1, 3]))\n\
         selected = get(tensor, 2, only, all)\n\
         inputs = (tensor)\noutputs = (selected)\n",
    )
    .expect("literal/only/all selection must emit");

    assert!(ir.contains("stablehlo.slice %arg0 [1:2, 0:1, 0:3]"), "{ir}");
    assert!(ir.contains("-> tensor<3xf32>"), "{ir}");
    assert!(!ir.contains("\"stablehlo.gather\""), "{ir}");
}

#[test]
fn unsupported_advanced_selectors_refuse() {
    let multiple = emit(
        "flatppl_compat = \"0.1\"\n\
         matrix = elementof(cartpow(reals, [4, 3]))\n\
         selected = get(matrix, [1, 2], [2, 1])\n\
         inputs = (matrix)\noutputs = (selected)\n",
    )
    .expect_err("multiple vector selectors remain outside this lowering");
    assert!(multiple.msg.contains("multiple integer-vector selectors"));

    let non_numeric = emit(
        "flatppl_compat = \"0.1\"\n\
         matrix = elementof(cartpow(booleans, [4, 3]))\n\
         selected = get(matrix, [1, 2], all)\n\
         inputs = (matrix)\noutputs = (selected)\n",
    )
    .expect_err("boolean tensors are not numeric indexing operands");
    assert!(non_numeric.msg.contains("requires a numeric tensor"));

    let nested = emit(
        "flatppl_compat = \"0.1\"\n\
         a = elementof(cartpow(reals, 3))\n\
         nested = [a, a]\n\
         selected = get(nested, [1, 2], all)\n\
         inputs = (a)\noutputs = (selected)\n",
    )
    .expect_err("nested array cells are not flat matrices");
    assert!(nested.msg.contains("one flat numeric tensor cell"));

    let batch_varying = emit(
        "flatppl_compat = \"0.1\"\n\
         matrices = elementof(cartpow(cartpow(reals, [4, 3]), 2))\n\
         indices = [[1, 2], [2, 1]]\n\
         pick(matrix, rows) = get(matrix, rows, all)\n\
         inputs = (matrices)\noutputs = pick.(matrices, indices)\n",
    )
    .expect_err("each query point cannot carry a different index vector");
    assert!(
        batch_varying
            .msg
            .contains("one shared rank-1 integer vector")
    );
}
