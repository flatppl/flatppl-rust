//! `sum` over a boolean array lowers to a COUNT, not to parity.
//!
//! §03 "Bool": "In arithmetic contexts, `false` is promoted to zero and `true` to one,
//! permitting expressions such as `true + true`, `3 * false`, and `sum(mask)` to count
//! true entries." No `i1` combine computes that: `stablehlo.add` on `i1` is a wrapping
//! 1-bit add. [`flatppl_stablehlo`]'s `Emitter::reduce_axis` therefore emits
//! `stablehlo.convert` to `i32` first, and the result carries `Int` so the ABI return
//! type agrees with the `integers` type `flatppl_infer` gives `sum(bool_array)`.
//!
//! Executed under IREE 3.11.0 (`llvm-cpu` / `local-task`) against numpy, on the modules
//! these tests pin:
//!
//! | model | executed | numpy |
//! |---|---|---|
//! | `sum([true, true, false])` | 2 | 2 |
//! | `sum(x .> 0.0)`, `x = [1, -2, 3, 4]` | 3 | 3 |
//! | `sum(x .> 0.0)`, `x = [-1, -2, -3, -4]` | 0 | 0 |
//! | `sum(x .> 0.0)`, `x = [0, 0, 1, 2]` | 2 | 2 |
//! | `sum(x .> 0.0)`, `x = [5, 5, 5, 5]` | 4 | 4 |
//! | `sum(rowstack([[T,F,T],[T,T,F]]))` | 4 | 4 |
//! | `sum(x .> 0.0)`, `x` a `[2, 3]` argument | 4 / 0 / 6 / 1 | same |
//!
//! Before the fix the same rank-1 masks executed to `false` / `true` / `true` / `false` —
//! the parity of the count.
//!
//! New tests live here rather than in `golden.rs` because two sibling branches are
//! landing changes in that file.

use flatppl_core::Module;

fn determinize_abi_roots(src: &str, roots: &[&str]) -> Module {
    let mut m = flatppl_syntax::parse(src).expect("parse");
    let diags = flatppl_infer::infer(&mut m);
    assert!(diags.is_empty(), "infer diagnostics: {diags:?}");
    let syms: Vec<flatppl_core::Symbol> = roots.iter().map(|r| m.intern(r)).collect();
    flatppl_determinizer::determinize_with_roots(
        &m,
        &flatppl_infer::ModuleBundle::new(),
        Some(&syms),
    )
    .expect("must determinize, not refuse")
}

fn emit(src: &str, roots: &[&str]) -> String {
    let d = determinize_abi_roots(src, roots);
    flatppl_stablehlo::emit(&d, flatppl_stablehlo::Mode::LogDensity, &Default::default())
        .expect("must emit, not refuse")
}

fn emit_err(src: &str, roots: &[&str]) -> String {
    let d = determinize_abi_roots(src, roots);
    match flatppl_stablehlo::emit(&d, flatppl_stablehlo::Mode::LogDensity, &Default::default()) {
        Ok(out) => panic!("expected a refusal, got:\n{out}"),
        Err(e) => format!("{e:?}"),
    }
}

const LITERAL_MASK: &str = "mask = [true, true, false]\nc = sum(mask)\noutputs = (c)\n";

/// The `i1` batch is converted to `i32` BEFORE the reduce, the reduce's operand/init/
/// result element types all agree at `i32`, and the additive identity is the integer
/// `0` — not the boolean `false` the old code emitted.
#[test]
fn boolean_sum_converts_to_integer_before_reducing() {
    let mlir = emit(LITERAL_MASK, &["outputs"]);
    assert!(
        mlir.contains("stablehlo.convert %6 : (tensor<3xi1>) -> tensor<3xi32>"),
        "the i1 batch must be converted before reducing:\n{mlir}"
    );
    assert!(
        mlir.contains("stablehlo.constant dense<0> : tensor<i32>"),
        "the additive identity must be the integer 0:\n{mlir}"
    );
    assert!(
        mlir.contains(
            "applies stablehlo.add across dimensions = [0] : (tensor<3xi32>, tensor<i32>) \
             -> tensor<i32>"
        ),
        "the reduce must run entirely at i32:\n{mlir}"
    );
    assert!(
        !mlir.contains("(tensor<3xi1>, tensor<i1>) -> tensor<i1>"),
        "no 1-bit reduce may survive — that is the parity bug:\n{mlir}"
    );
}

/// The ABI return type comes from the node's INFERRED type
/// (`flatppl_stablehlo::mlir_type_of`), so the emitter's promotion and the `infer` rule
/// have to name the same type or the emitted function is ill-typed. Both say `i32`.
#[test]
fn boolean_sum_abi_return_type_is_integer() {
    let mlir = emit(LITERAL_MASK, &["outputs"]);
    assert!(
        mlir.contains("func.func @logdensity() -> tensor<i32>"),
        "the ABI must return i32, matching the inferred `integers`:\n{mlir}"
    );
    assert!(
        !mlir.contains("-> tensor<i1> {"),
        "the ABI must not return i1:\n{mlir}"
    );
}

/// The §03 use `sum(mask)` names — counting a comparison mask, here over a runtime
/// argument. This is the module executed against numpy for the four vectors in this
/// file's table.
#[test]
fn counting_a_runtime_comparison_mask_lowers_to_an_integer_reduce() {
    let src = "x = elementof(cartpow(reals, 4))\n\
               mask = x .> 0.0\n\
               c = sum(mask)\n\
               inputs = (x)\n\
               outputs = (c)\n";
    let mlir = emit(src, &["inputs", "outputs"]);
    assert!(
        mlir.contains("func.func @logdensity(%arg0: tensor<4xf32>) -> tensor<i32>"),
        "a real batch in, a count out:\n{mlir}"
    );
    assert!(
        mlir.contains("stablehlo.compare GT") && mlir.contains("-> tensor<4xi1>"),
        "the mask is an i1 batch:\n{mlir}"
    );
    assert!(
        mlir.contains("stablehlo.convert") && mlir.contains("-> tensor<4xi32>"),
        "which is promoted before the reduce:\n{mlir}"
    );
}

/// The promotion is guarded on `Bool`, so the `Real` and `Int` reduction paths emit
/// exactly what they did before: no `stablehlo.convert`, and their own identity literal.
/// Reduce-over-`Real` is every density lowering in the emitter, so a stray convert here
/// would move every existing golden.
#[test]
fn real_and_integer_sums_emit_no_conversion() {
    let real = emit(
        "v = [1.5, -0.5, 2.0]\nc = sum(v)\noutputs = (c)\n",
        &["outputs"],
    );
    assert!(
        !real.contains("stablehlo.convert"),
        "a real sum must not convert:\n{real}"
    );
    assert!(
        real.contains("stablehlo.constant dense<0.000000e+00> : tensor<f32>")
            && real.contains("(tensor<3xf32>, tensor<f32>) -> tensor<f32>"),
        "and keeps the float identity and f32 reduce:\n{real}"
    );

    let int = emit("v = [1, 2, 3]\nc = sum(v)\noutputs = (c)\n", &["outputs"]);
    assert!(
        !int.contains("stablehlo.convert"),
        "an integer sum was already at i32 and must not convert:\n{int}"
    );
    assert!(
        int.contains("(tensor<3xi32>, tensor<i32>) -> tensor<i32>"),
        "and reduces at i32:\n{int}"
    );
}

/// A RANK-2 boolean `sum` converts ONCE and then reduces axis by axis.
/// `Emitter::reduce_full` calls `reduce_axis` once per rank, and `reduce_axis` promotes at
/// its top — so a naive reading would expect one `stablehlo.convert` per axis. There is
/// exactly one: the first pass converts `i1` → `i32`, and every later pass sees an `i32`
/// operand, for which `Emitter::convert` is a no-op that emits no line.
///
/// This case became emittable only when `rowstack` gained a lowering (#177); before that
/// the rank-2 boolean reduce could not be executed at all. Executed under IREE against
/// numpy: `sum(rowstack([[true, false, true], [true, true, false]]))` = 4 = `np.sum`, and
/// the runtime-mask form `sum(x .> 0.0)` over a `[2, 3]` argument = 4 / 0 / 6 / 1 for
/// `[[1,-2,3],[4,-5,6]]` / all-negative / all-ones / `[[0,0,0],[0,0,7]]`, each matching.
#[test]
fn a_rank_two_boolean_sum_converts_once_then_reduces_each_axis() {
    let src = "m = rowstack([[true, false, true], [true, true, false]])\n\
               c = sum(m)\n\
               outputs = (c)\n";
    let mlir = emit(src, &["outputs"]);
    assert_eq!(
        mlir.matches("stablehlo.convert").count(),
        1,
        "exactly one convert, not one per axis:\n{mlir}"
    );
    assert!(
        mlir.contains("stablehlo.convert %16 : (tensor<2x3xi1>) -> tensor<2x3xi32>"),
        "the whole rank-2 batch is promoted up front:\n{mlir}"
    );
    // Two reduces, each at i32: [2, 3] -> [3] -> scalar.
    assert_eq!(
        mlir.matches("stablehlo.reduce").count(),
        2,
        "one reduce per axis:\n{mlir}"
    );
    assert!(
        mlir.contains("(tensor<2x3xi32>, tensor<i32>) -> tensor<3xi32>")
            && mlir.contains("(tensor<3xi32>, tensor<i32>) -> tensor<i32>"),
        "both passes run at i32:\n{mlir}"
    );
    assert!(
        mlir.contains("func.func @logdensity() -> tensor<i32>"),
        "and the ABI returns a count:\n{mlir}"
    );
}

/// `maximum`/`minimum` over a boolean array stay REFUSED, and that is consistent rather
/// than an oversight: §03's promotion covers arithmetic contexts, while §07's
/// $\max_i x_i$ selects an element, so `infer` keeps `maximum(bool_array)` at `booleans`
/// and this emitter has no `i1` `stablehlo.reduce` identity to lower it with. Refusing
/// beats emitting a value whose type disagrees with the inferred one.
#[test]
fn boolean_extrema_are_still_refused() {
    let msg = emit_err(
        "mask = [true, true, false]\nc = maximum(mask)\noutputs = (c)\n",
        &["outputs"],
    );
    assert!(
        msg.contains("maximum/minimum: only a real array is supported"),
        "unexpected refusal: {msg}"
    );
}
