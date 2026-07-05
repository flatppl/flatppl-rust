//! Smoke test: the Task-1 stub `emit` accepts a determinized (FlatPDL) module
//! and returns a minimal valid StableHLO module string. Later tasks replace
//! this with real golden output comparisons.
//!
//! Task 2 adds `mlir_type_of`: the `Type`/`Dim` → MLIR `tensor<…>` mapping.
//! These tests build tiny `Module`s by hand (`alloc` a placeholder node,
//! `set_type` the type under test) rather than parsing + inferring source,
//! since only the type side-table matters for this mapping.

use flatppl_core::{Dim, Mass, Module, Node, Scalar, ScalarType, Type};
use flatppl_stablehlo::{Dtype, Emitter, MlirTy, mlir_type_of};

/// Every physical `{`/`(`/`[` in `s` has a matching close, and vice versa —
/// a cheap structural well-formedness check for hand-assembled MLIR text
/// (it does not parse the text, just counts bracket nesting).
fn is_delimiter_balanced(s: &str) -> bool {
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => depth -= 1,
            _ => {}
        }
        if depth < 0 {
            return false;
        }
    }
    depth == 0
}

#[test]
fn emit_stub_on_flatpdl_returns_module() {
    let src = "flatppl_compat = \"0.1\"\nx = 1.0\n";
    let m = flatppl_syntax::parse(src).unwrap();
    let d = flatppl_determinizer::determinize(&m).unwrap();
    let out = flatppl_stablehlo::emit(&d, flatppl_stablehlo::Mode::LogDensity, &Default::default())
        .unwrap();
    assert!(out.contains("module {"));
}

/// A placeholder node to hang a `set_type` on — its `Node` payload is
/// irrelevant to `mlir_type_of`, which only reads the type side-table.
fn placeholder(m: &mut Module, ty: Type) -> flatppl_core::NodeId {
    let id = m.alloc(Node::Lit(Scalar::Real(0.0)));
    m.set_type(id, ty);
    id
}

#[test]
fn mlir_type_of_scalar_renders_rank0_tensor() {
    let mut m = Module::new();
    let id = placeholder(&mut m, Type::Scalar(ScalarType::Real));
    let ty = mlir_type_of(&m, id, Dtype::F32).unwrap();
    assert_eq!(ty, MlirTy::Scalar);
    assert_eq!(ty.render(Dtype::F32), "tensor<f32>");
}

#[test]
fn mlir_type_of_flat_array_renders_ranked_tensor() {
    let mut m = Module::new();
    let id = placeholder(
        &mut m,
        Type::Array {
            shape: Box::new([Dim::Static(2), Dim::Static(3)]),
            elem: Box::new(Type::Scalar(ScalarType::Real)),
        },
    );
    let ty = mlir_type_of(&m, id, Dtype::F32).unwrap();
    assert_eq!(ty, MlirTy::Ranked(vec![Some(2), Some(3)]));
    assert_eq!(ty.render(Dtype::F32), "tensor<2x3xf32>");
}

#[test]
fn mlir_type_of_nested_array_flattens_to_one_tensor_shape() {
    // Array{shape:[2], elem:Array{shape:[3], elem:real}} — vec-of-vec (spec
    // §03 nesting) — must flatten to the SAME single tensor shape as the flat
    // rank-2 array above, not a nested/tuple MLIR type.
    let mut m = Module::new();
    let id = placeholder(
        &mut m,
        Type::Array {
            shape: Box::new([Dim::Static(2)]),
            elem: Box::new(Type::Array {
                shape: Box::new([Dim::Static(3)]),
                elem: Box::new(Type::Scalar(ScalarType::Real)),
            }),
        },
    );
    let ty = mlir_type_of(&m, id, Dtype::F32).unwrap();
    assert_eq!(ty, MlirTy::Ranked(vec![Some(2), Some(3)]));
    assert_eq!(ty.render(Dtype::F32), "tensor<2x3xf32>");
}

#[test]
fn mlir_type_of_dynamic_dim_renders_question_mark() {
    let mut m = Module::new();
    let id = placeholder(
        &mut m,
        Type::Array {
            shape: Box::new([Dim::Dynamic, Dim::Static(3)]),
            elem: Box::new(Type::Scalar(ScalarType::Real)),
        },
    );
    let ty = mlir_type_of(&m, id, Dtype::F32).unwrap();
    assert_eq!(ty, MlirTy::Ranked(vec![None, Some(3)]));
    assert_eq!(ty.render(Dtype::F32), "tensor<?x3xf32>");
}

#[test]
fn mlir_type_of_tvector_renders_ranked_tensor() {
    let mut m = Module::new();
    let id = placeholder(
        &mut m,
        Type::TVector {
            len: Dim::Static(4),
            elem: Box::new(Type::Scalar(ScalarType::Real)),
        },
    );
    let ty = mlir_type_of(&m, id, Dtype::F32).unwrap();
    assert_eq!(ty, MlirTy::Ranked(vec![Some(4)]));
    assert_eq!(ty.render(Dtype::F32), "tensor<4xf32>");
}

#[test]
fn mlir_type_of_dtype_is_configurable_not_hardcoded() {
    let mut m = Module::new();
    let id = placeholder(&mut m, Type::Scalar(ScalarType::Real));
    let ty = mlir_type_of(&m, id, Dtype::F64).unwrap();
    assert_eq!(ty.render(Dtype::F64), "tensor<f64>");
}

#[test]
fn mlir_type_of_refuses_aggregate_types() {
    let mut m = Module::new();
    let field = m.intern("x");
    let id = placeholder(
        &mut m,
        Type::Record(Box::new([(field, Type::Scalar(ScalarType::Real))])),
    );
    let err = mlir_type_of(&m, id, Dtype::F32).unwrap_err();
    assert!(err.msg.contains("aggregate"));
    assert!(err.msg.contains("destructured"));
    assert_eq!(err.node, Some(id));
}

#[test]
fn mlir_type_of_refuses_residual_measure_layer_types() {
    let mut m = Module::new();
    let id = placeholder(
        &mut m,
        Type::Measure {
            domain: Box::new(Type::Scalar(ScalarType::Real)),
            mass: Mass::Normalized,
        },
    );
    let err = mlir_type_of(&m, id, Dtype::F32).unwrap_err();
    assert!(err.msg.contains("residual measure-layer type in FlatPDL"));
    assert_eq!(err.node, Some(id));
}

#[test]
fn mlir_type_of_refuses_other_types_naming_the_type() {
    // `RngState` hits the catch-all arm (neither aggregate nor
    // measure-layer) — the refusal must name the offending type via its
    // `Debug` form, not just say "no MLIR tensor form" with no detail.
    let mut m = Module::new();
    let id = placeholder(&mut m, Type::RngState);
    let err = mlir_type_of(&m, id, Dtype::F32).unwrap_err();
    assert!(err.msg.contains("type has no MLIR tensor form"));
    assert!(err.msg.contains("RngState"));
    assert_eq!(err.node, Some(id));
}

// ---- Task 3: Emitter core -------------------------------------------------

/// The brief's Step-2 example, verbatim: a hand-built two-scalar-add graph,
/// `finish`ed with no arguments.
#[test]
fn emitter_scalar_add_produces_well_formed_module() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let a = e.scalar(2.0);
    let b = e.scalar(3.0);
    let c = e.add(&a, &b);
    let out = e.finish("logdensity", &[], &c);

    assert!(out.contains("stablehlo.add"));
    assert!(out.contains("func.func @logdensity"));
    assert!(out.contains("return"));
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emitter_finish_wraps_args_and_return_type() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F64);
    let arg = flatppl_stablehlo::Value {
        ssa: "%arg0".to_string(),
        ty: MlirTy::Scalar,
    };
    let doubled = e.add(&arg, &arg);
    let out = e.finish("f", &[("%arg0".to_string(), MlirTy::Scalar)], &doubled);
    assert!(out.starts_with("module {\n"));
    assert!(out.contains("func.func @f(%arg0: tensor<f64>) -> tensor<f64> {"));
    assert!(out.trim_end().ends_with('}'));
    assert!(is_delimiter_balanced(&out));
}

/// Every named elementary wrapper dispatches through `unary`/`binary` to the
/// StableHLO op its doc comment promises, and preserves the operand's shape.
#[test]
fn emitter_elementary_wrappers_emit_expected_ops() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let a = e.scalar(1.0);
    let b = e.scalar(2.0);

    let cases: Vec<(flatppl_stablehlo::Value, &str)> = vec![
        (e.sub(&a, &b), "stablehlo.subtract"),
        (e.mul(&a, &b), "stablehlo.multiply"),
        (e.div(&a, &b), "stablehlo.divide"),
        (e.pow(&a, &b), "stablehlo.power"),
        (e.neg(&a), "stablehlo.negate"),
        (e.log(&a), "stablehlo.log"),
        (e.exp(&a), "stablehlo.exponential"),
        (e.sqrt(&a), "stablehlo.sqrt"),
        (e.abs(&a), "stablehlo.abs"),
        (e.cos(&a), "stablehlo.cosine"),
    ];
    let out = e.finish("f", &[], &cases[0].0);
    for (_, op) in &cases {
        assert!(out.contains(op), "missing {op} in:\n{out}");
    }
    assert!(is_delimiter_balanced(&out));
}

/// `chlo.lgamma` is a function-type op (`in_ty -> out_ty`), not the plain
/// `: ty` form the `stablehlo.*` elementary unaries use — the real
/// StableHLO+CHLO MLIR parser rejects `chlo.lgamma %a : ty` ("expected
/// '->'"). Pin both the op name and the `->` so a regression back to the
/// single-type form is caught.
#[test]
fn emitter_lgamma_emits_function_type_form() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let a = e.scalar(1.0);
    let r = e.lgamma(&a);
    let out = e.finish("f", &[], &r);

    assert!(
        out.contains("chlo.lgamma %"),
        "missing chlo.lgamma in:\n{out}"
    );
    let lgamma_line = out.lines().find(|l| l.contains("chlo.lgamma")).unwrap();
    assert!(
        lgamma_line.contains(" -> "),
        "chlo.lgamma line missing '->' function-type arrow:\n{lgamma_line}"
    );
    assert!(is_delimiter_balanced(&out));
}

#[test]
fn emitter_compare_and_select_type_check() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let a = e.scalar(1.0);
    let b = e.scalar(2.0);
    let pred = e.compare("LT", &a, &b);
    let picked = e.select(&pred, &a, &b);
    let out = e.finish("f", &[], &picked);

    assert!(out.contains("stablehlo.compare LT"));
    assert!(out.contains("tensor<i1>"));
    assert!(out.contains("stablehlo.select"));
    assert!(is_delimiter_balanced(&out));
}

/// `reduce_sum`/`reduce_max` must emit `stablehlo.reduce`'s *pretty* form
/// (`stablehlo.reduce(%in init: %init) applies stablehlo.OP across
/// dimensions = [D] : (...) -> ...`) — the real parser rejects the generic
/// `"stablehlo.reduce"(...) <{dimensions=...}> ({region})` form this crate
/// used to emit. `reduce_max`'s identity must be real dtype-exact negative
/// infinity (`0xFF800000` for f32), not a finite `-1e30` stand-in that is
/// silently wrong for inputs at or below it.
#[test]
fn emitter_reduce_sum_and_max_reduce_to_scalar() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let v = e.constant(1.0, MlirTy::Ranked(vec![Some(3)]));
    let s = e.reduce_sum(&v);
    assert_eq!(s.ty, MlirTy::Scalar);

    let mx = e.reduce_max(&v);
    assert_eq!(mx.ty, MlirTy::Scalar);

    let out = e.finish("f", &[], &mx);
    assert!(out.contains("stablehlo.reduce("));
    assert!(
        out.contains("applies stablehlo.add across dimensions"),
        "missing pretty-form add reduce in:\n{out}"
    );
    assert!(
        out.contains("applies stablehlo.maximum across dimensions"),
        "missing pretty-form maximum reduce in:\n{out}"
    );
    assert!(
        out.contains("dense<0xFF800000>"),
        "missing dtype-exact -inf reduce_max identity in:\n{out}"
    );
    assert!(
        !out.contains("stablehlo.return"),
        "no region form expected:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// Same dtype-exact `-inf` identity check, pinned for `f64` too — the bit
/// pattern (not just the presence of *a* hex literal) is dtype-dependent.
#[test]
fn emitter_reduce_max_f64_identity_is_dtype_exact_neg_inf() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F64);
    let v = e.constant(1.0, MlirTy::Ranked(vec![Some(3)]));
    let mx = e.reduce_max(&v);
    let out = e.finish("f", &[], &mx);
    assert!(
        out.contains("dense<0xFFF0000000000000>"),
        "missing f64 -inf identity in:\n{out}"
    );
}

#[test]
fn emitter_reduce_sum_on_scalar_is_a_noop() {
    // A rank-0 operand has no axes to reduce: `reduce_sum` should hand back
    // the same value without emitting a spurious reduce op.
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let s = e.scalar(1.0);
    let summed = e.reduce_sum(&s);
    assert_eq!(summed.ssa, s.ssa);
    assert_eq!(summed.ty, MlirTy::Scalar);
}

#[test]
fn emitter_matrix_helpers_emit_expected_ops() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let mat = e.constant(1.0, MlirTy::Ranked(vec![Some(3), Some(3)]));
    let vec3 = e.constant(1.0, MlirTy::Ranked(vec![Some(3)]));

    let l = e.cholesky(&mat);
    assert_eq!(l.ty, mat.ty);
    let d = e.diag(&l);
    assert_eq!(d.ty, MlirTy::Ranked(vec![Some(3)]));
    let mv = e.matvec(&l, &vec3);
    assert_eq!(mv.ty, vec3.ty);
    let y = e.tri_solve(&l, &vec3);
    assert_eq!(y.ty, vec3.ty);

    let out = e.finish("f", &[], &y);
    assert!(out.contains("stablehlo.cholesky"));
    assert!(out.contains("stablehlo.iota"));
    assert!(out.contains("stablehlo.dot_general"));
    assert!(
        out.contains("contracting_dims = [1] x [0]"),
        "missing dot_general pretty-form contracting_dims in:\n{out}"
    );
    assert!(out.contains("precision = [DEFAULT, DEFAULT]"));
    assert!(
        out.contains("\"stablehlo.triangular_solve\"("),
        "missing triangular_solve generic-form head in:\n{out}"
    );
    assert!(is_delimiter_balanced(&out));
}

/// `matvec`'s result type must be `a`'s leading dimension, not `b`'s type —
/// those only coincide in the square case the other test exercises. A
/// rectangular `[m, n] @ [n]` product must produce `[m]`.
#[test]
fn emitter_matvec_result_type_is_lhs_leading_dim() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let mat = e.constant(1.0, MlirTy::Ranked(vec![Some(5), Some(3)]));
    let vec3 = e.constant(1.0, MlirTy::Ranked(vec![Some(3)]));

    let mv = e.matvec(&mat, &vec3);
    assert_eq!(mv.ty, MlirTy::Ranked(vec![Some(5)]));
    assert_ne!(mv.ty, vec3.ty);
}

/// `matvec` panics rather than mis-lowering when the shapes don't line up
/// (mirrors the panic-on-bad-shape discipline `diag`/`reduce_axis` already
/// follow) — a mismatched trailing dim is an internal invariant violation
/// upstream type-checking should have already ruled out.
#[test]
#[should_panic(expected = "does not match rhs length")]
fn emitter_matvec_panics_on_shape_mismatch() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let mat = e.constant(1.0, MlirTy::Ranked(vec![Some(5), Some(3)]));
    let vec4 = e.constant(1.0, MlirTy::Ranked(vec![Some(4)]));
    e.matvec(&mat, &vec4);
}

#[test]
fn emitter_fresh_ssa_names_never_repeat() {
    let m = Module::new();
    let mut e = Emitter::new(&m, Dtype::F32);
    let a = e.scalar(1.0);
    let b = e.scalar(2.0);
    let c = e.add(&a, &b);
    // Three distinct ops so far (two scalars + one add) must have three
    // distinct SSA names.
    let mut names = vec![a.ssa.clone(), b.ssa.clone(), c.ssa.clone()];
    names.sort();
    names.dedup();
    assert_eq!(names.len(), 3);
}
