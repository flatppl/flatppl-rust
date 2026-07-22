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
//! `modes::emit_logdensity_abi` (`Mode::LogDensity`, the `@logdensity`
//! function) and `modes::emit_sample_abi` (`Mode::Sample`, the `@sample`
//! function, `mu + sigma * Z`-style reparameterised sampling seeded via
//! `Emitter::rng`).
//!
//! # The `inputs` / `outputs` compilation ABI
//!
//! FlatPDL carries no marker for "which binding is a function argument" or
//! "which is the result", so a model designates them explicitly with two
//! reserved top-level bindings — `inputs = …` and `outputs = …`, each a single
//! value or a tuple. **Tuple order is the ABI order** of the emitted function.
//! The host (`flatppl stablehlo`) roots dead-code elimination on
//! `{inputs, outputs}` (keeping the outputs' backward cone plus the declared
//! inputs) and the emitter reads the ABI off the determinized module
//! (`modes::read_abi` → `modes::emit_logdensity_abi` / `emit_sample_abi`). The
//! ABI is **required for both modes**: a module declaring neither reserved
//! binding is refused — the older last-public-binding / source-order query
//! heuristic has been removed, so there is no fallback.
//!
//! Each `inputs` entry becomes a `func.func` argument, typed by phase (spec
//! §04):
//! - an `elementof` parameter → an argument in its inferred element kind (an
//!   integer-domain parameter such as `posintegers` arrives as a `tensor<i32>`
//!   argument and is converted to float only where it enters real-valued math);
//! - an `external(S)` input → an argument typed from `S`;
//! - a `load_data(…)` input → a `tensor<N×f32>` argument whose length `N` is
//!   pinned from a compile-time read of the file's row count (`.csv` / `.wsv`;
//!   other formats refuse) — the **values are never baked**, they are the
//!   runtime argument (the CLI supplies the lengths via
//!   [`EmitOptions::input_shapes`]).
//!
//! `inputs` is authoritative and exhaustive: every `elementof` parameter in
//! the module must be listed, or emission refuses. Each `outputs` entry is a
//! deterministic FlatPDL result in declared order — a `logdensityof(M, point)`
//! density query, or (`Mode::Sample`) a sampled value: the emitted `@sample`
//! threads a leading `%key : tensor<2xui64>` rng argument (spec §07 `rand`) and
//! returns the `(value, new_key)` pair, so its single `outputs` entry names the
//! sampled binding rather than a density.
//!
//! ## Worked examples
//!
//! Ordered arguments, single result — `inputs` names the three parameters, so
//! they become `%arg0..%arg2` in that order:
//!
//! ```text
//! alpha = elementof(reals)
//! beta  = elementof(reals)
//! sigma = elementof(posreals)
//! mu    = alpha .+ beta .* [1.0, 2.0]
//! y     = draw(Normal.(mu, sigma))
//! inputs  = (alpha, beta, sigma)
//! outputs = logdensityof(lawof(record(y = y)), record(y = [1.1, 2.2]))
//! ```
//! →
//! ```mlir
//! func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32>
//! ```
//!
//! Multiple outputs — the function returns a tuple in `outputs` order (here the
//! likelihood and the posterior, both a function of the one input `mu`):
//!
//! ```text
//! mu = elementof(reals)
//! inputs  = mu
//! outputs = (logdensityof(L, record(a = mu)), logdensityof(post, record(a = 0.5)))
//! ```
//! →
//! ```mlir
//! func.func @logdensity(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>)
//! ```
//!
//! Data as a runtime argument — a `load_data` input listed in `inputs` is a
//! shape-pinned tensor (here `d.csv` has 3 data rows), so one compiled module
//! scores any 3-vector without re-emitting; its values are not constants:
//!
//! ```text
//! mu = elementof(reals)
//! y  = load_data("d.csv", reals)
//! inputs  = (mu, y)
//! outputs = logdensityof(post, record(a = mu))
//! ```
//! →
//! ```mlir
//! func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<3xf32>) -> tensor<f32>
//! ```
//!
//! Caveat for `load_module` query modules: an `inputs` parameter's *binding*
//! name must not shadow a binding of the loaded model (a bare `alpha` scoring a
//! model that itself binds `alpha` collides across the independent namespaces
//! and refuses) — give the free parameters distinct names (e.g. `t_alpha`) and
//! map them onto the variate field names in the `outputs` `record(…)`.
//!
//! See `crates/stablehlo/docs/inputs-outputs-abi.md` for the full reference —
//! every refusal rule and query-module usage.

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
            input_shapes: std::collections::HashMap::new(),
        }
    }
}

/// Emit textual StableHLO for `m`, which must already be FlatPDL-conformant
/// (i.e. the output of `flatppl_determinizer::determinize`). Refuses (never
/// mis-lowers) if `m` still carries measure-layer constructs.
///
/// Requires the `inputs`/`outputs` compilation ABI (`modes::read_abi` is
/// `Some`) and routes to the mode builder for `mode`: [`Mode::LogDensity`] →
/// `modes::emit_logdensity_abi`, [`Mode::Sample`] → `modes::emit_sample_abi`.
/// A module declaring neither reserved binding is refused — the older
/// last-public-binding/source-order query heuristic has been removed, so there
/// is no fallback.
pub fn emit(m: &Module, mode: Mode, opts: &EmitOptions) -> Result<String, EmitError> {
    flatppl_determinizer::is_flatpdl(m)
        .map_err(|_| EmitError::whole("input is not FlatPDL (determinize first)"))?;
    match modes::read_abi(m) {
        Some(abi) => match mode {
            Mode::LogDensity => modes::emit_logdensity_abi(m, &abi, opts),
            Mode::Sample => modes::emit_sample_abi(m, &abi, opts),
        },
        None => Err(EmitError::whole(
            "no inputs/outputs ABI declared; the last-public-binding query \
             heuristic has been removed — declare `inputs = (…)` and \
             `outputs = (…)` (typically in a query module) to designate the \
             compiled function's arguments and results",
        )),
    }
}
