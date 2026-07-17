//! The MLIR-side type model + textual rendering the StableHLO emitter needs
//! for every SSA value: a shaped-tensor type and its `render` to text.
//!
//! `MlirTy` deliberately carries no element dtype — FlatPPL never mandates a
//! precision (spec grounding), so the element type (`f32`/`f64`) is supplied
//! by the caller's [`crate::Dtype`] only at [`MlirTy::render`] time. The
//! FlatPDL `Type`/`Dim` → `MlirTy` mapping itself lives in `types.rs`.
//!
//! [`ElemKind`] is the orthogonal axis `MlirTy`/`Dtype` don't carry: which
//! element *family* (boolean/integer/real, spec §03) a value's tensor holds.
//! Every [`Value`] carries one; [`MlirTy::render`] combines it with `Dtype`
//! to pick the concrete element string.

use crate::Dtype;

/// An MLIR tensor type, shape-only (no baked-in element dtype).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MlirTy {
    /// A rank-0 tensor (`tensor<f32>`) — FlatPDL scalars.
    Scalar,
    /// A ranked tensor with a (possibly partially dynamic) static shape.
    /// `None` entries render as MLIR's `?` (a dynamic dimension).
    Ranked(Vec<Option<u64>>),
    /// A tuple of MLIR types (`tuple<...>`).
    Tuple(Vec<MlirTy>),
    /// The rng-state key tensor (spec §07 rng ABI) — always `tensor<2xui64>`,
    /// independent of [`Dtype`] (the key is never a float).
    Key,
}

/// The element family of a value's tensor, orthogonal to shape (`MlirTy`)
/// and float precision (`Dtype`). Resolved from a value's inferred scalar
/// kind (spec §03 boolean/integer/real categories); `Complex` has no tensor
/// form and is refused before reaching here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElemKind {
    Real,
    Int,
    Bool,
}

/// One SSA value: its name (`%0`, `%arg0`, …), MLIR type, and element kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value {
    pub ssa: String,
    pub ty: MlirTy,
    pub elem: ElemKind,
}

impl MlirTy {
    /// Render as textual MLIR: `tensor<f32>` (rank-0), `tensor<3xf32>`,
    /// `tensor<2x3xf32>`, `tensor<?x3xf32>` for a dynamic leading dim, and
    /// `tuple<...>` for a tuple of MLIR types. `dtype` supplies the float
    /// precision, `elem` the element family (`ElemKind::Real` renders
    /// `f32`/`f64`, `Int` renders `i32`/`i64`, `Bool` renders `i1`
    /// regardless of `dtype`); an empty `Ranked` shape renders the same as
    /// `Scalar` (both are rank-0).
    pub fn render(&self, dtype: Dtype, elem: ElemKind) -> String {
        let e = match elem {
            ElemKind::Real => match dtype {
                Dtype::F32 => "f32",
                Dtype::F64 => "f64",
            },
            ElemKind::Int => match dtype {
                Dtype::F32 => "i32",
                Dtype::F64 => "i64",
            },
            ElemKind::Bool => "i1",
        };
        match self {
            MlirTy::Scalar => format!("tensor<{e}>"),
            MlirTy::Ranked(dims) => {
                let mut out = String::from("tensor<");
                for dim in dims {
                    match dim {
                        Some(n) => out.push_str(&n.to_string()),
                        None => out.push('?'),
                    }
                    out.push('x');
                }
                out.push_str(e);
                out.push('>');
                out
            }
            MlirTy::Tuple(parts) => {
                let inner: Vec<String> = parts.iter().map(|p| p.render(dtype, elem)).collect();
                format!("tuple<{}>", inner.join(", "))
            }
            // Pinned in the rng-threaded-rand plan's Task-1 spike: dtype-
            // independent, always `ui64` (never `f32`/`f64`).
            MlirTy::Key => "tensor<2xui64>".to_string(),
        }
    }
}
