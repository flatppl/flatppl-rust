//! Golden tests for the §07 shape constructors `rowstack`, `colstack` and
//! `addaxes` (`crate::ops`'s `lower_stack` / `lower_addaxes`).
//!
//! Deliberately a separate file from `tests/golden.rs`: these lowerings landed
//! alongside another emitter wave that appends there, and splitting the file
//! keeps the textual conflict surface at zero. The helpers below are local
//! copies for the same reason — `golden.rs`'s are private to it.
//!
//! The three frozen `.mlir` goldens this file pins (`stack_matmul_literals`,
//! `stack_colstack`, `stack_addaxes_broadcast`) were compiled with
//! `iree-base-compiler` (llvm-cpu) and EXECUTED, their results matched against
//! §07's/§04's own printed values and a numpy oracle — read off disk, so the
//! pinned text and the verified numbers cannot diverge. See
//! `.superpowers/sdd/2026-08-05-joint-constructs-the-joint/wave-hlostack-report.md`.

use flatppl_core::Module;

fn determinize_src(src: &str) -> Module {
    let mut m = flatppl_syntax::parse(src).expect("parse");
    let diags = flatppl_infer::infer(&mut m);
    assert!(diags.is_empty(), "infer diagnostics: {diags:?}");
    flatppl_determinizer::determinize(&m).expect("must determinize, not refuse")
}

/// Determinize with the `inputs`/`outputs` ABI bindings as the roots — the
/// shape a model with runtime (`elementof`) operands takes.
fn determinize_abi(src: &str) -> Module {
    let mut m = flatppl_syntax::parse(src).expect("parse");
    let diags = flatppl_infer::infer(&mut m);
    assert!(diags.is_empty(), "infer diagnostics: {diags:?}");
    let syms: Vec<flatppl_core::Symbol> =
        ["inputs", "outputs"].iter().map(|r| m.intern(r)).collect();
    flatppl_determinizer::determinize_with_roots(
        &m,
        &flatppl_infer::ModuleBundle::new(),
        Some(&syms),
    )
    .expect("must determinize, not refuse")
}

fn emit_logdensity(m: &Module) -> String {
    flatppl_stablehlo::emit(
        m,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect("must emit @logdensity")
}

fn emit_with_dtype(m: &Module, dtype: flatppl_stablehlo::Dtype) -> String {
    flatppl_stablehlo::emit(
        m,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions { dtype },
    )
    .expect("must emit @logdensity")
}

/// The emitter's refusal message for a source it declines to lower. `infer` and
/// the determiniser must both be clean, so the refusal is this crate's.
fn emit_err(src: &str) -> String {
    let m = determinize_src(src);
    flatppl_stablehlo::emit(
        &m,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect_err("must refuse in the emitter")
    .msg
}

/// §04 "Multi-axis aggregation"'s own example prelude, verbatim (integer
/// literals included), with the matrix product §04 prints as `[[6, 8], [10, 6]]`.
///
/// The frozen golden is the EXECUTED artifact.
const MATMUL_LITERALS_SRC: &str = "\
A = rowstack([[1, 3, 5], [9, 5, 1]])
B = rowstack([[1, 0], [0, 1], [1, 1]])
C = A * B
outputs = (C)
";

/// §04's prelude emits and matches its frozen golden. `rowstack` itself emits
/// NO op: the literal vector-of-vectors already lowers to the `[2, 3]` /
/// `[3, 2]` tensors, so the only structural op is the product's
/// `stablehlo.dot_general`.
#[test]
fn rowstack_literals_matmul_matches_frozen_golden() {
    let out = emit_logdensity(&determinize_src(MATMUL_LITERALS_SRC));
    assert_eq!(
        out,
        include_str!("goldens/stack_matmul_literals.mlir"),
        "emitted @logdensity drifted from tests/goldens/stack_matmul_literals.mlir"
    );
}

/// An all-integer matrix product stays `i32` end to end. §04's prelude is
/// written with integer literals and §03 makes `integers ⊂ reals`, so the
/// operands lower as `i32`; `Emitter::dot_contract` used to render the RESULT
/// `f32` unconditionally, emitting `(tensor<2x3xi32>, tensor<3x2xi32>) ->
/// tensor<2x2xf32>`. That crosses the base type category StableHLO's
/// `isPromotableElementType` requires lhs, rhs and result to share, and
/// disagrees with `infer`'s `mul_type` (which types the product `integers`) —
/// but it is NOT caught by execution: IREE parses, verifies, compiles and runs
/// that module, silently returning `f32`. A pre-existing bug with pre-existing
/// reachable spellings (`fill(1, [2, 3]) * fill(1, [3, 2])`, an integer matrix
/// ABI input); `rowstack` of integer literals is the door that surfaced it.
#[test]
fn integer_matrix_product_keeps_its_element_type() {
    let out = emit_logdensity(&determinize_src(MATMUL_LITERALS_SRC));
    assert!(
        out.contains(
            "stablehlo.dot_general %16, %35, contracting_dims = [1] x [0], precision = \
             [DEFAULT, DEFAULT] : (tensor<2x3xi32>, tensor<3x2xi32>) -> tensor<2x2xi32>"
        ),
        "integer operands must give an integer product:\n{out}"
    );
    assert!(
        !out.contains("xf32>"),
        "no f32 anywhere in an all-integer module:\n{out}"
    );
}

/// `rowstack` emits nothing at all — no `transpose`, no `broadcast_in_dim`.
/// The vector-of-vectors and the matrix it stacks into share one tensor form
/// (`types::mlir_type_of` flattens the nested element chain), so the row order
/// §07 fixes is already the emitted one.
#[test]
fn rowstack_of_runtime_vectors_emits_no_op() {
    let src = "\
v1 = elementof(cartpow(reals, 3))
v2 = elementof(cartpow(reals, 3))
A = rowstack([v1, v2])
inputs = (v1, v2)
outputs = (A)
";
    let out = emit_logdensity(&determinize_abi(src));
    assert!(
        out.contains(
            "func.func @logdensity(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<2x3xf32>"
        ),
        "two length-3 args stack into a 2x3 matrix:\n{out}"
    );
    assert!(
        !out.contains("stablehlo.transpose"),
        "rowstack must not transpose:\n{out}"
    );
    assert!(
        !out.contains("stablehlo.broadcast_in_dim"),
        "rowstack must not broadcast:\n{out}"
    );
    // The only ops are `Emitter::vector`'s own stacking pair.
    assert_eq!(
        out.matches("stablehlo.concatenate").count(),
        1,
        "exactly one concatenate:\n{out}"
    );
}

/// `colstack` is the row stack TRANSPOSED — §07's example makes
/// `colstack([[1, 2, 3], [4, 5, 6]])` the 3x2 `[[1, 4], [2, 5], [3, 6]]`.
/// The frozen golden is the EXECUTED artifact.
#[test]
fn colstack_matches_frozen_golden() {
    let src = "\
M = colstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
outputs = (M)
";
    let out = emit_logdensity(&determinize_src(src));
    assert_eq!(
        out,
        include_str!("goldens/stack_colstack.mlir"),
        "emitted @logdensity drifted from tests/goldens/stack_colstack.mlir"
    );
    assert!(
        out.contains(
            "stablehlo.transpose %16, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>"
        ),
        "colstack transposes the row stack:\n{out}"
    );
}

/// The same operand through both constructors: `rowstack` gives `[2, 3]`,
/// `colstack` the transposed `[3, 2]`. One test so the two cannot drift into
/// agreeing.
#[test]
fn rowstack_and_colstack_are_transposes_of_each_other() {
    let src = "\
v1 = elementof(cartpow(reals, 3))
v2 = elementof(cartpow(reals, 3))
R = rowstack([v1, v2])
C = colstack([v1, v2])
inputs = (v1, v2)
outputs = (R, C)
";
    let out = emit_logdensity(&determinize_abi(src));
    assert!(
        out.contains("-> (tensor<2x3xf32>, tensor<3x2xf32>)"),
        "rowstack 2x3, colstack 3x2:\n{out}"
    );
    assert_eq!(
        out.matches("stablehlo.transpose").count(),
        1,
        "only colstack transposes:\n{out}"
    );
}

/// §04 "Relationship to broadcasting"'s `B = addaxes(v, 0, 1)` feeding an
/// elementwise product against a matrix — one `stablehlo.reshape` to `[3, 1]`,
/// then §04's size-one axis expansion. The frozen golden is the EXECUTED
/// artifact.
#[test]
fn addaxes_broadcast_matches_frozen_golden() {
    let src = "\
v = [1.0, 2.0, 3.0]
A = rowstack([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
B = addaxes(v, 0, 1)
C = A .* B
outputs = (C)
";
    let out = emit_logdensity(&determinize_src(src));
    assert_eq!(
        out,
        include_str!("goldens/stack_addaxes_broadcast.mlir"),
        "emitted @logdensity drifted from tests/goldens/stack_addaxes_broadcast.mlir"
    );
}

/// §04 "Broadcasting" pins the argument convention through two named
/// behaviours: `addaxes(b, 1, 0)` "behaves like NumPy-style broadcasting"
/// (the singleton axis LEADS, so the vector aligns with the trailing axis) and
/// `addaxes(b, 0, 1)` "like Julia-style" (it TRAILS, aligning with the leading
/// axis). Both were executed against numpy; this pins the two shapes.
#[test]
fn addaxes_argument_convention_matches_the_two_named_styles() {
    let src = "\
A = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b3 = [10.0, 100.0, 1000.0]
b2 = [10.0, 100.0]
np_style = A .* addaxes(b3, 1, 0)
jl_style = A .* addaxes(b2, 0, 1)
outputs = (np_style, jl_style)
";
    let out = emit_logdensity(&determinize_src(src));
    assert!(
        out.contains("stablehlo.reshape %23 : (tensor<3xf32>) -> tensor<1x3xf32>"),
        "addaxes(b3, 1, 0) is a LEADING singleton axis (NumPy-style):\n{out}"
    );
    assert!(
        out.contains("stablehlo.reshape %31 : (tensor<2xf32>) -> tensor<2x1xf32>"),
        "addaxes(b2, 0, 1) is a TRAILING singleton axis (Julia-style):\n{out}"
    );
}

/// §07's own worked shape: "Given an array `A` of size `(3, 4, 5)`,
/// `addaxes(A, 2, 3)` will return an array of size `(1, 1, 3, 4, 5, 1, 1, 1)`."
/// Checked on the rank-2 operand this crate can build, `(2, 3)` →
/// `(1, 1, 2, 3, 1, 1, 1)` — same 2-leading / 3-trailing pattern.
#[test]
fn addaxes_inserts_both_axis_runs() {
    let src = "\
M = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
X = addaxes(M, 2, 3)
outputs = (X)
";
    let out = emit_logdensity(&determinize_src(src));
    assert!(
        out.contains("stablehlo.reshape %16 : (tensor<2x3xf32>) -> tensor<1x1x2x3x1x1x1xf32>"),
        "2 leading and 3 trailing singleton axes:\n{out}"
    );
}

/// `addaxes(A, 0, 0)` is the identity and emits NOTHING — the same
/// no-op-rather-than-noise choice `lower_transpose` makes for a vector.
#[test]
fn addaxes_with_zero_counts_emits_nothing() {
    let src = "\
v = elementof(cartpow(reals, 3))
x = addaxes(v, 0, 0)
inputs = (v)
outputs = (x)
";
    let out = emit_logdensity(&determinize_abi(src));
    assert!(
        !out.contains("stablehlo.reshape"),
        "addaxes(A, 0, 0) must emit no reshape:\n{out}"
    );
    assert!(
        out.contains("func.func @logdensity(%arg0: tensor<3xf32>) -> tensor<3xf32>"),
        "the operand passes straight through:\n{out}"
    );
}

/// A container of uniformly TRANSPOSED vectors is accepted: §03 "Arrays" says
/// "The term vector will represent both non-transposed vectors
/// (one-dimensional arrays) and transposed vectors in the following, unless
/// noted otherwise", and §07 fixes where the argument's vectors go whatever
/// their own orientation. Emits the same module as the non-transposed spelling
/// (a vector's transpose emits nothing either), which is the point.
#[test]
fn rowstack_of_transposed_vectors_is_accepted() {
    let rows = "\
v1 = elementof(cartpow(reals, 3))
v2 = elementof(cartpow(reals, 3))
A = rowstack([transpose(v1), transpose(v2)])
inputs = (v1, v2)
outputs = (A)
";
    let plain = "\
v1 = elementof(cartpow(reals, 3))
v2 = elementof(cartpow(reals, 3))
A = rowstack([v1, v2])
inputs = (v1, v2)
outputs = (A)
";
    assert_eq!(
        emit_logdensity(&determinize_abi(rows)),
        emit_logdensity(&determinize_abi(plain)),
        "a row container and a column container stack to the same matrix"
    );
}

/// The lowerings are dtype-parameterized like their neighbours: nothing bakes
/// `f32`. (`Dtype::F64` was NOT executed — see the wave report.)
#[test]
fn stack_lowerings_are_dtype_parameterized() {
    let src = "\
M = colstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
X = addaxes(M, 1, 0)
outputs = (X)
";
    let out = emit_with_dtype(&determinize_src(src), flatppl_stablehlo::Dtype::F64);
    assert!(
        out.contains("(tensor<2x3xf64>) -> tensor<3x2xf64>")
            && out.contains("(tensor<3x2xf64>) -> tensor<1x3x2xf64>"),
        "f64 must thread through transpose and reshape:\n{out}"
    );
    assert!(!out.contains("f32"), "no f32 in an F64 module:\n{out}");
}

// ---- refusals ---------------------------------------------------------------

/// A RAGGED container refuses one level down, in `ops::lower_vector`:
/// inference types `[[1.0, 2.0], [3.0]]` as an array of `%any` rather than
/// reporting it, so the emitter is the first layer to see it. Pinned from
/// `rowstack` so the constructor cannot start accepting one.
#[test]
fn rowstack_refuses_ragged_rows() {
    let err = emit_err("x = rowstack([[1.0, 2.0], [3.0]])\noutputs = (x)\n");
    assert!(
        err.contains("ragged vector-of-vectors has no tensor form"),
        "got: {err}"
    );
}

/// A vector of SCALARS is not §07's "vector of equal-length vectors".
#[test]
fn rowstack_refuses_a_rank_1_operand() {
    let err = emit_err("x = rowstack([1.0, 2.0])\noutputs = (x)\n");
    assert!(err.contains("`rowstack` has no lowering for"), "got: {err}");
    assert!(
        err.contains("vector of equal-length vectors"),
        "the refusal must quote §07's domain, got: {err}"
    );
}

/// A vector of MATRICES is not either — rank 3 has no matrix to build.
#[test]
fn rowstack_refuses_a_rank_3_operand() {
    let err = emit_err(
        "A = rowstack([[1.0, 2.0], [3.0, 4.0]])\n\
         B = rowstack([[5.0, 6.0], [7.0, 8.0]])\n\
         x = rowstack([A, B])\n\
         outputs = (x)\n",
    );
    assert!(
        err.contains("`rowstack` has no lowering for Ranked([Some(2), Some(2), Some(2)])"),
        "got: {err}"
    );
}

/// `colstack` refuses the same shapes, with its own name in the message.
#[test]
fn colstack_refuses_a_rank_1_operand() {
    let err = emit_err("x = colstack([1.0, 2.0])\noutputs = (x)\n");
    assert!(err.contains("`colstack` has no lowering for"), "got: {err}");
}

/// A MATRIX argument refuses. §03 "Arrays" says vectors of vectors "are not
/// interpreted as matrices implicitly, but can be turned into matrices explicitly
/// using `rowstack` or `colstack`", and the converse is no more sanctioned — but
/// the two have the SAME lowered tensor, so `lower_stack`'s rank-2 check cannot
/// see it and this would otherwise be a silent identity (`rowstack`) or a silent
/// transpose (`colstack`) of an operand §07 gives the constructor no meaning for.
/// Decided on the inferred type, where they still differ:
/// `(%array 1 (2) (%array 1 (2) …))` against `(%array 2 (2 2) …)`.
#[test]
fn stack_refuses_a_matrix_argument() {
    for head in ["rowstack", "colstack"] {
        let err = emit_err(&format!(
            "A = rowstack([[1.0, 2.0], [3.0, 4.0]])\nx = {head}(A)\noutputs = (x)\n"
        ));
        assert!(
            err.contains(&format!(
                "`{head}`'s argument is a rank-2 array, not a vector of vectors"
            )),
            "got: {err}"
        );
        assert!(
            err.contains("not interpreted as matrices implicitly"),
            "the refusal must quote §03, got: {err}"
        );
    }
}

/// The same guard on a rank-3 argument, and on a matrix arriving as an ABI input
/// rather than from a literal — the inferred type is the discriminator either way.
#[test]
fn stack_refuses_higher_rank_and_abi_matrix_arguments() {
    let err = emit_err(
        "A = rowstack([[1.0, 2.0], [3.0, 4.0]])\nT = addaxes(A, 1, 0)\nx = colstack(T)\noutputs = (x)\n",
    );
    assert!(
        err.contains("`colstack`'s argument is a rank-3 array"),
        "got: {err}"
    );
    let m = determinize_abi(
        "B = elementof(cartpow(reals, [2, 3]))\nx = colstack(B)\ninputs = (B)\noutputs = (x)\n",
    );
    let err = flatppl_stablehlo::emit(
        &m,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect_err("a matrix ABI input must refuse")
    .msg;
    assert!(
        err.contains("`colstack`'s argument is a rank-2 array"),
        "got: {err}"
    );
}

/// The guard must NOT catch a genuine vector-of-vectors ABI input. §03's nested
/// `cartpow(cartpow(reals, 3), 2)` valueset infers `(%array 1 (2) (%array 1 (3)
/// …))` — rank-1 container, legal argument — where `cartpow(reals, [2, 3])`
/// infers the rank-2 `(%array 2 (2 3) …)` the test above refuses. Same tensor,
/// opposite verdicts, which is the whole point of deciding on the inferred type.
#[test]
fn stack_accepts_a_nested_cartpow_container() {
    let src = "\
A = elementof(cartpow(cartpow(reals, 3), 2))
x = colstack(A)
inputs = (A)
outputs = (x)
";
    let out = emit_logdensity(&determinize_abi(src));
    assert!(
        out.contains("func.func @logdensity(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32>"),
        "a nested-cartpow container must still stack:\n{out}"
    );
}

/// A container that MIXES orientations — a column and a row in one vector — is
/// not a well-typed FlatPPL array (§03 gives an array a single element type,
/// and a transposed vector is a distinct type). Both elements lower to the same
/// `tensor<nxf32>`, so without this guard the stack lowered silently.
#[test]
fn rowstack_refuses_mixed_orientation() {
    let err = emit_err(
        "v1 = [1.0, 2.0]\nv2 = [3.0, 4.0]\nx = rowstack([v1, transpose(v2)])\noutputs = (x)\n",
    );
    assert!(err.contains("mixes vector orientations"), "got: {err}");
}

/// §07 gives `addaxes` the domain "array, …", not "vector", so §03's
/// both-orientations blanket does not widen it — and the widening would change
/// the answer: a row's tensor form is `[n]`, so `addaxes(transpose(v), 0, 1)`
/// would emit the COLUMN `[n, 1]`.
#[test]
fn addaxes_refuses_a_transposed_vector() {
    let err = emit_err("v = [1.0, 2.0, 3.0]\nx = addaxes(transpose(v), 0, 1)\noutputs = (x)\n");
    assert!(
        err.contains("addaxes: `A` is a transposed vector"),
        "got: {err}"
    );
}

/// §07: "`n_leading` and `n_trailing` must be non-negative fixed integers."
/// Surface `-1` arrives as `neg(1)`, so it is a non-literal here, not a
/// negative literal.
#[test]
fn addaxes_refuses_a_negative_count() {
    let err = emit_err("v = [1.0, 2.0, 3.0]\nx = addaxes(v, -1, 0)\noutputs = (x)\n");
    assert!(
        err.contains("addaxes: `n_leading` must be a non-negative fixed integer literal"),
        "got: {err}"
    );
}

/// §07's `A` is an array. A scalar has no axes to pad.
#[test]
fn addaxes_refuses_a_scalar_operand() {
    let err = emit_err("x = addaxes(1.0, 1, 0)\noutputs = (x)\n");
    assert!(
        err.contains("addaxes: `A` must be a statically-shaped array, got Scalar"),
        "got: {err}"
    );
}

/// Wrong arity is INFERENCE's refusal, not the emitter's — `addaxes(v, 1)`
/// never reaches `lower_addaxes`, so its shared `args_exact` guard stays
/// defensive. Pinned here so a later inference change cannot silently move the
/// gate without this file noticing.
#[test]
fn addaxes_wrong_arity_is_caught_before_the_emitter() {
    let mut m = flatppl_syntax::parse("v = [1.0, 2.0, 3.0]\nx = addaxes(v, 1)\noutputs = (x)\n")
        .expect("parse");
    let diags = flatppl_infer::infer(&mut m);
    assert!(
        diags
            .iter()
            .any(|d| d.message.contains("`addaxes` takes 3 arguments")),
        "inference must report the arity, got: {diags:?}"
    );
}
