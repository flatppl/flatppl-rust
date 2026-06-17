//! `flatppl-core` — the in-memory model for FlatPPL.
//!
//! The foundational crate every other `flatppl-*` crate depends on. It holds a
//! single **multi-level IR** (MLIR-style): sugar-stripped, construct-preserving
//! FlatPIR that can represent high-level constructs (`metricsum`, measures,
//! distributions) *and* their lowered forms — lowering between levels is a
//! separate, deliberate pass. No parsing or serialization lives here (see
//! `flatppl-syntax` for canonical FlatPPL and `flatppl-flatpir` for FlatPIR
//! S-expressions).
//!
//! Shape: a [`Module`] owns an [`Arena`] of [`Node`]s addressed by integer
//! handles ([`NodeId`] / [`BindingId`]), with analysis results (types, phases,
//! spans) in [`SecondaryMap`] side-tables. See `ARCHITECTURE.md` for the
//! rationale behind these choices.

pub mod id;
mod module;
pub mod node;
pub mod ty;

pub use id::{Arena, BindingId, Idx, Interner, NodeId, SecondaryMap, Symbol};
pub use module::{Binding, Doc, Markup, Module, Span};
pub use node::{
    Axis, Call, CallHead, Inputs, NamedArg, NamedKind, Node, Ref, RefNs, Scalar, Variance,
};
pub use ty::{Dim, Mass, Phase, ScalarType, Type, ValueSet};

/// The FlatPPL language version this toolchain targets — the string stamped into
/// a generated module's `flatppl_compat` binding (spec §11). The whole ecosystem
/// is pinned to `0.1` pre-release (see `flatppl-dev/CONVENTIONS.md`, "Version
/// state"); this constant is the single source of truth for that value.
pub const FLATPPL_COMPAT: &str = "0.1";
