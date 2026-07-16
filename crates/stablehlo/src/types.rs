//! FlatPDL `Type` → [`MlirTy`] mapping (spec §03/§11 grounding) — the
//! foundation the emitter uses for every SSA value's type.
//!
//! Every FlatPDL value type maps onto some MLIR tensor rank: scalars become
//! rank-0 tensors, arrays/row-vectors become ranked tensors with `Dim` →
//! static-or-`?` dims. FlatPDL preserves array *nesting* (`Array.elem` may
//! itself be an `Array` — vec-of-vec stays distinct from a matrix, spec §03),
//! but MLIR's `tensor<...>` has no such nesting: [`mlir_type_of`] flattens a
//! nested `Array`/`TVector` element chain into ONE tensor shape.
//!
//! Types with no tensor form are refused, never mis-lowered: aggregates
//! (`Record`/`Tuple`/`Table`) must be destructured by an earlier pass, and a
//! residual measure-layer type (`Measure`/`Kernel`/`Likelihood`) reaching
//! here is an invariant violation — `emit` only ever runs on already-
//! determinized FlatPDL (`flatppl_determinizer::is_flatpdl`).

use flatppl_core::{Dim, Module, NodeId, ScalarType, Type};

use crate::Dtype;
use crate::mlir::{ElemKind, MlirTy};
use crate::refuse::EmitError;

/// Map the FlatPDL type of `id` (looked up via [`Module::type_of`]) to its
/// MLIR tensor form and element kind.
///
/// `dtype` is accepted for signature symmetry with the rest of the emitter's
/// `Dtype` threading, but is unused here: the tensor *shape* never depends on
/// the element dtype — that is only applied at [`MlirTy::render`] time, never
/// baked into a FlatPPL `Type` (FlatPPL never mandates a precision). The
/// returned [`ElemKind`] reflects the node's actual inferred scalar kind
/// (spec §03 boolean/integer/real) via [`scalar_kind`]; a `Complex` scalar
/// has no tensor form and is refused rather than mis-lowered.
pub fn mlir_type_of(
    m: &Module,
    id: NodeId,
    _dtype: Dtype,
) -> Result<(MlirTy, ElemKind), EmitError> {
    let ty = m
        .type_of(id)
        .ok_or_else(|| EmitError::at(id, "node has no inferred type"))?;
    match ty {
        Type::Scalar(st) => Ok((MlirTy::Scalar, scalar_kind(id, *st)?)),
        Type::Array { shape, elem } => {
            let mut dims: Vec<Option<u64>> = shape.iter().map(dim_to_mlir).collect();
            let kind = flatten_elem(id, elem, &mut dims)?;
            Ok((MlirTy::Ranked(dims), kind))
        }
        Type::TVector { len, elem } => {
            let mut dims = vec![dim_to_mlir(len)];
            let kind = flatten_elem(id, elem, &mut dims)?;
            Ok((MlirTy::Ranked(dims), kind))
        }
        Type::RngState => Ok((MlirTy::Key, ElemKind::Real)),
        _ => Err(refuse_non_tensor(id, ty)),
    }
}

/// Map a scalar leaf's spec §03 category to its [`ElemKind`]. `Complex` has
/// no tensor form (StableHLO's tensor element types are real/integer/boolean
/// only) — refused here, localized to `id`, rather than mis-lowered.
fn scalar_kind(id: NodeId, st: ScalarType) -> Result<ElemKind, EmitError> {
    match st {
        ScalarType::Real => Ok(ElemKind::Real),
        ScalarType::Integer => Ok(ElemKind::Int),
        ScalarType::Boolean => Ok(ElemKind::Bool),
        ScalarType::Complex => Err(EmitError::at(id, "complex scalar has no tensor form")),
    }
}

/// Recursively flatten a nested `Array`/`TVector` element chain into `dims`,
/// stopping at a `Scalar` leaf (spec §03: nesting collapses to one tensor
/// shape), and return that leaf's [`ElemKind`] — the array/vector's own
/// element kind. Any other leaf refuses with the same diagnostics as the
/// top-level match in [`mlir_type_of`], localized to the original node `id`.
fn flatten_elem(
    id: NodeId,
    elem: &Type,
    dims: &mut Vec<Option<u64>>,
) -> Result<ElemKind, EmitError> {
    match elem {
        Type::Scalar(st) => scalar_kind(id, *st),
        Type::Array { shape, elem } => {
            dims.extend(shape.iter().map(dim_to_mlir));
            flatten_elem(id, elem, dims)
        }
        Type::TVector { len, elem } => {
            dims.push(dim_to_mlir(len));
            flatten_elem(id, elem, dims)
        }
        _ => Err(refuse_non_tensor(id, elem)),
    }
}

/// Shared refusal for any FlatPDL `Type` with no MLIR tensor form, localized
/// to `id`. Used by both [`mlir_type_of`] and [`flatten_elem`] so the three
/// diagnostics (aggregate / residual measure-layer / catch-all) stay in one
/// place. The catch-all names the offending type via `Debug` — `Type`
/// carries no interner-backed names, so this is the only precise identifier
/// available without a `Module` reference.
fn refuse_non_tensor(id: NodeId, ty: &Type) -> EmitError {
    match ty {
        Type::Record(_) | Type::Tuple(_) | Type::Table { .. } => EmitError::at(
            id,
            "aggregate type has no tensor form; must be destructured",
        ),
        Type::Measure { .. } | Type::Kernel { .. } | Type::Likelihood { .. } => {
            EmitError::at(id, "residual measure-layer type in FlatPDL")
        }
        _ => EmitError::at(id, format!("type has no MLIR tensor form: {ty:?}")),
    }
}

fn dim_to_mlir(d: &Dim) -> Option<u64> {
    match d {
        Dim::Static(n) => Some(*n as u64),
        Dim::Dynamic => None,
    }
}
