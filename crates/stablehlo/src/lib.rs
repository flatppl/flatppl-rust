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
//! The crate is a walk-and-print emitter: [`mlir_type_of`]/[`MlirTy`] map
//! FlatPDL `Type`/`Dim` to MLIR `tensor<…>` types, [`Emitter`] does SSA
//! bookkeeping (the `NodeId` → `Value` memo map, `stablehlo.rng`, and a typed
//! op-helper API — elementary ops, CHLO special functions, reductions, matrix
//! helpers, and `finish`'s module/func assembly), `ops::lower_builtin` maps
//! every non-distribution `Call` head to its op sequence, and `registry.rs`
//! holds the distribution registry: a ctor-name-keyed table from a
//! distribution constructor (e.g. `Normal`) to its closed-form `logpdf` and
//! `sample` builders, covering every distribution in the spec's base catalogue.
//!
//! `modes.rs` builds the two emitted programs `emit` routes to:
//! `emit_logdensity` (`Mode::LogDensity`, the `@logdensity` function) and
//! `emit_sample` (`Mode::Sample`, the `@sample` function, `mu + sigma * Z`-style
//! reparameterised sampling seeded via `Emitter::rng`).

mod emitter;
mod mlir;
mod modes;
mod ops;
mod refuse;
mod registry;
mod types;

pub use emitter::Emitter;
pub use mlir::{ElemKind, MlirTy, Value};
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
    /// The public binding to emit as the query, designated by name. FlatPDL
    /// carries no query marker (see [`modes`]); the host (the CLI verb / the
    /// testsuite harness) picks the query binding and names it here. `None`
    /// falls back to the "last public binding" convention — correct for a
    /// self-contained hand-written scoring model, but ambiguous once
    /// cross-module grafting (`load_module`) splices a foreign model's inert
    /// data/residue bindings in *after* the query in source order. Naming the
    /// query keeps it identifiable regardless of grafted trailing bindings.
    pub query: Option<String>,
    /// Compile-time shape pins for fixed-phase ABI inputs whose FlatPDL type
    /// carries a dynamic dim — `load_data(...)` (typed `CartPow(set,
    /// Dim::Dynamic)`, so `tensor<?×f32>`) and a shaped `external(...)` — keyed
    /// by the input binding's name. The design doc (`docs/superpowers/specs/
    /// 2026-07-17-inputs-outputs-abi-design.md`, "`load_data` — shape, not
    /// values") pins the length from a compile-time read of the resolved file
    /// *only* for its shape; the values remain the runtime argument (never
    /// baked). The CLI (`stablehlo_cmd`) reads each `load_data` ABI-input file's
    /// length and populates this map; [`modes::emit_logdensity_abi`] uses it to
    /// override the dynamic dim so the argument types `tensor<N×f32>` — a `?`
    /// dim would be unusable downstream (the executor requires a static shape
    /// for the density's reduce/broadcast). Empty (the default) means no pin:
    /// an input with a dynamic dim then types as-is (a `?` arg).
    pub input_shapes: std::collections::HashMap<String, Vec<u64>>,
}

impl Default for EmitOptions {
    fn default() -> Self {
        Self {
            dtype: Dtype::F32,
            query: None,
            input_shapes: std::collections::HashMap::new(),
        }
    }
}

/// Emit textual StableHLO for `m`, which must already be FlatPDL-conformant
/// (i.e. the output of `flatppl_determinizer::determinize`). Refuses (never
/// mis-lowers) if `m` still carries measure-layer constructs.
///
/// Routes to the mode builder for `mode`: [`Mode::LogDensity`] → the
/// `inputs`/`outputs` ABI path ([`modes::emit_logdensity_abi`], PR-1) when
/// `m` declares the ABI ([`modes::read_abi`] is `Some`), else the legacy
/// last-public-binding/source-order path ([`modes::emit_logdensity`]);
/// [`Mode::Sample`] → [`modes::emit_sample`] (the ABI is PR-1
/// `LogDensity`-mode-only — a `Sample`-mode module carrying `inputs`/
/// `outputs` is not specially routed and falls through to the existing
/// query-finding convention, which refuses if that convention's guard is not
/// met rather than mis-lowering).
pub fn emit(m: &Module, mode: Mode, opts: &EmitOptions) -> Result<String, EmitError> {
    flatppl_determinizer::is_flatpdl(m)
        .map_err(|_| EmitError::whole("input is not FlatPDL (determinize first)"))?;
    match mode {
        Mode::LogDensity => match modes::read_abi(m) {
            Some(abi) => modes::emit_logdensity_abi(m, &abi, opts),
            None => modes::emit_logdensity(m, opts),
        },
        Mode::Sample => modes::emit_sample(m, opts),
    }
}
