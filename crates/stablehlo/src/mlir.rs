//! The MLIR-side type model + textual rendering the StableHLO emitter needs
//! for every SSA value: a shaped-tensor type and its `render` to text.
//!
//! `MlirTy` deliberately carries no element dtype — FlatPPL never mandates a
//! precision (spec grounding), so the element type (`f32`/`f64`) is supplied
//! by the caller's [`crate::Dtype`] only at [`MlirTy::render`] time. The
//! FlatPDL `Type`/`Dim` → `MlirTy` mapping itself lives in `types.rs`.

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

/// One SSA value: its name (`%0`, `%arg0`, …) and MLIR type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value {
    pub ssa: String,
    pub ty: MlirTy,
}

impl MlirTy {
    /// Render as textual MLIR: `tensor<f32>` (rank-0), `tensor<3xf32>`,
    /// `tensor<2x3xf32>`, `tensor<?x3xf32>` for a dynamic leading dim, and
    /// `tuple<...>` for a tuple of MLIR types. `dtype` supplies the element
    /// string (`f32`/`f64`); an empty `Ranked` shape renders the same as
    /// `Scalar` (both are rank-0).
    pub fn render(&self, dtype: Dtype) -> String {
        let elem = match dtype {
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
        };
        match self {
            MlirTy::Scalar => format!("tensor<{elem}>"),
            MlirTy::Ranked(dims) => {
                let mut out = String::from("tensor<");
                for dim in dims {
                    match dim {
                        Some(n) => out.push_str(&n.to_string()),
                        None => out.push('?'),
                    }
                    out.push('x');
                }
                out.push_str(elem);
                out.push('>');
                out
            }
            MlirTy::Tuple(parts) => {
                let inner: Vec<String> = parts.iter().map(|p| p.render(dtype)).collect();
                format!("tuple<{}>", inner.join(", "))
            }
            // Pinned in the rng-threaded-rand plan's Task-1 spike: dtype-
            // independent, always `ui64` (never `f32`/`f64`).
            MlirTy::Key => "tensor<2xui64>".to_string(),
        }
    }
}
