//! Types, shapes, and phases — the inference domain.
//!
//! Mirrors the FlatPIR type categories (spec §11). Two design points from
//! `ARCHITECTURE.md`: array *nesting* is preserved (`Array.elem` may itself be an
//! `Array`, so vec-of-vec ≠ matrix, spec §03), and a dimension is either a static
//! size or `%dynamic` — never a sentinel.

use crate::id::Symbol;

/// The structural category of a value / object (the FlatPIR `%meta` type slot).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    /// `%deferred` — not yet inferred.
    Deferred,
    /// `(%failed "reason")` — inference attempted but failed; the module is ill-formed.
    Failed(Box<str>),
    /// `%any` — no concrete constraint (e.g. the input of `fn(sum(_))`).
    Any,
    /// `(%scalar real|integer|boolean|complex)`.
    Scalar(ScalarType),
    /// `(%array <ndims> <shape> <elem>)`. `<ndims>` (the number of dimensions /
    /// axes) is `shape.len()`, so it is not stored here — it is recomputed on
    /// write and validated against `shape.len()` on read. `elem` may itself be an
    /// `Array`: nested arrays (vec-of-vec) stay distinct from higher-dimensional
    /// arrays / matrices (spec §03).
    Array { shape: Box<[Dim]>, elem: Box<Type> },
    /// `(%tvector len elem)` — a transposed (row) vector; distinct from a
    /// 1-dimensional `Array`.
    TVector { len: Dim, elem: Box<Type> },
    /// `(%record (name ty) …)` — ordered named fields.
    Record(Box<[(Symbol, Type)]>),
    /// `(%tuple ty …)` — ≥ 2 positional components.
    Tuple(Box<[Type]>),
    /// `(%table (%columns …) (%nrows N))` — column element types + row count.
    Table {
        columns: Box<[(Symbol, Type)]>,
        nrows: Dim,
    },
    /// `(%measure (%domain ty))` — a closed measure over `domain`.
    // TODO: the additive sample/batch/event shape triple (engine-concepts §20.10)
    // attaches here when fusion/dispatch needs it.
    Measure { domain: Box<Type> },
    /// `(%kernel (%inputs …))` — a user-defined transition kernel.
    Kernel { inputs: Box<[Symbol]> },
    /// `(%function (%inputs …))` — a user-defined function.
    Function { inputs: Box<[Symbol]> },
    /// `(%likelihood (%inputs …) (%obstype ty))`.
    Likelihood {
        inputs: Box<[Symbol]>,
        obstype: Box<Type>,
    },
    /// `rngstate`.
    RngState,
    /// `%module` — a loaded-module reference (not a first-class value).
    Module,
    /// An inference type variable.
    Var(u32),
}

/// The four scalar value types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarType {
    Real,
    Integer,
    Boolean,
    Complex,
}

/// An array dimension: a concrete size, or `%dynamic` (resolved at load / run time).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    Static(u32),
    Dynamic,
}

/// Phase classification (spec §04): governs life-cycle and closure behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Fixed,
    Parameterized,
    Stochastic,
}
