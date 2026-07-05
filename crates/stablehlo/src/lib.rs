//! `flatppl-stablehlo` — prints textual StableHLO from post-`determinize`
//! FlatPDL.
//!
//! This crate only *emits*: it takes a FlatPDL [`Module`] (the deterministic
//! profile produced by `flatppl-determinizer`, which has already eliminated
//! the measure layer) and lowers it to StableHLO's textual MLIR dialect. It
//! never determinizes on its own — callers run `flatppl_determinizer::determinize`
//! first, and `emit` refuses (never mis-lowers) if the input still carries
//! measure-layer constructs.
//!
//! This is Task 1 of the backend: crate scaffold only. `emit` is a stub that
//! asserts FlatPDL-conformance and returns a minimal empty module; later tasks
//! fill in the real emitter and op registry.
//!
//! Task 2 adds the `Type`/`Dim` → MLIR `tensor<…>` mapping ([`mlir_type_of`],
//! [`MlirTy`]) that every later emitter task builds SSA values on top of.
//!
//! Task 3 adds [`Emitter`]: SSA bookkeeping, the `NodeId` → `Value` memo map,
//! and the typed op-helper API (elementary ops, CHLO special functions,
//! reductions, matrix helpers, and `finish`'s module/func assembly) that
//! Task 4's node-dispatch lowering is built on top of.
//!
//! Task 4 adds [`Emitter::lower_node`] (leaf/call dispatch + memoization)
//! and `ops::lower_builtin` (the deterministic builtin-head → op map for
//! every non-distribution `Call`): together they turn any post-determinize
//! FlatPDL expression graph into a [`Value`], composing only Task 3's
//! op-helper API.

mod emitter;
mod mlir;
mod ops;
mod refuse;
mod types;

pub use emitter::Emitter;
pub use mlir::{MlirTy, Value};
pub use refuse::EmitError;
pub use types::mlir_type_of;

use flatppl_core::Module;

/// Which computation to emit: the model's log-density, or a sampling program.
pub enum Mode {
    LogDensity,
    Sample,
}

/// Floating-point element type for emitted StableHLO tensors.
#[derive(Clone, Copy)]
pub enum Dtype {
    F32,
    F64,
}

/// Emitter configuration. `dtype` defaults to [`Dtype::F32`] — the emitter
/// never assumes/hardcodes 64-bit floats.
pub struct EmitOptions {
    pub dtype: Dtype,
}

impl Default for EmitOptions {
    fn default() -> Self {
        Self { dtype: Dtype::F32 }
    }
}

/// Emit textual StableHLO for `m`, which must already be FlatPDL-conformant
/// (i.e. the output of `flatppl_determinizer::determinize`). Refuses (never
/// mis-lowers) if `m` still carries measure-layer constructs.
///
/// Stub (Task 1): once FlatPDL-conformance is confirmed, returns a minimal
/// valid empty module. `_mode` / `_opts` are threaded through the signature
/// for later tasks and are not yet used.
pub fn emit(m: &Module, _mode: Mode, _opts: &EmitOptions) -> Result<String, EmitError> {
    flatppl_determinizer::is_flatpdl(m)
        .map_err(|_| EmitError::whole("input is not FlatPDL (determinize first)"))?;
    Ok("module {\n}\n".to_string())
}
