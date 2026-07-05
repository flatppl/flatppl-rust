//! Smoke test: the Task-1 stub `emit` accepts a determinized (FlatPDL) module
//! and returns a minimal valid StableHLO module string. Later tasks replace
//! this with real golden output comparisons.
//!
//! Task 2 adds `mlir_type_of`: the `Type`/`Dim` → MLIR `tensor<…>` mapping.
//! These tests build tiny `Module`s by hand (`alloc` a placeholder node,
//! `set_type` the type under test) rather than parsing + inferring source,
//! since only the type side-table matters for this mapping.

use flatppl_core::{Dim, Mass, Module, Node, Scalar, ScalarType, Type};
use flatppl_stablehlo::{Dtype, MlirTy, mlir_type_of};

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
