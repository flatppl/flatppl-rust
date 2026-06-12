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
    /// `(%measure (%domain ty) (%mass m))` — a closed measure over `domain`
    /// with total-mass class `mass`.
    // TODO: the additive sample/batch/event shape triple (engine-concepts §20.10)
    // attaches here when fusion/dispatch needs it.
    Measure { domain: Box<Type>, mass: Mass },
    /// `(%kernel (%inputs …) (%mass m))` — a user-defined transition kernel;
    /// `mass` is the total-mass class of the output measure, uniform over all
    /// inputs (`Normalized` ⇔ a Markov kernel).
    Kernel { inputs: Box<[Symbol]>, mass: Mass },
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

/// The total-mass class of a measure (spec §11 "Total-mass classes"): the
/// strongest statically known class, in a strict hierarchy — `LocallyFinite`
/// implies *infinite* total mass (a locally finite measure with finite total
/// mass is `Finite`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mass {
    /// `%deferred` — not yet inferred.
    Deferred,
    /// `%null` — the zero measure.
    Null,
    /// `%normalized` — total mass one (a probability measure).
    Normalized,
    /// `%finite` — finite total mass (possibly zero).
    Finite,
    /// `%locallyfinite` — infinite total mass, but finite mass on every
    /// bounded set (e.g. `Lebesgue(reals)`, `Counting(integers)`).
    LocallyFinite,
    /// `%unknown` — nothing beyond the ambient s-finiteness is known.
    Unknown,
}

/// A statically known value set (spec §03), the `valueset` annotation domain:
/// the strongest known set containing a node's value. For measure-typed nodes
/// this is the measure's support. Engines may be conservative — the set
/// vocabulary is not intersection-closed, so `Unknown` is always sound.
#[derive(Clone, Debug, PartialEq)]
pub enum ValueSet {
    /// `%deferred` — not yet inferred.
    Deferred,
    /// `%unknown` — inferred, but no constraint is known.
    Unknown,
    /// `reals` — ℝ (±∞ admitted).
    Reals,
    /// `posreals` — (0, +∞].
    PosReals,
    /// `nonnegreals` — [0, +∞].
    NonNegReals,
    /// `unitinterval` — [0, 1].
    UnitInterval,
    /// `integers` — ℤ.
    Integers,
    /// `posintegers` — {1, 2, …}.
    PosIntegers,
    /// `nonnegintegers` — {0, 1, …}.
    NonNegIntegers,
    /// `booleans`.
    Booleans,
    /// `complexes` — ℂ.
    Complexes,
    /// `rngstates`.
    RngStates,
    /// `anything` — no constraint by construction.
    Anything,
    /// `stdsimplex(n)` — the standard probability simplex.
    StdSimplex(Dim),
    /// `interval(lo, hi)` with static literal bounds.
    Interval(f64, f64),
    /// `cartpow(set, n)` — arrays with every element in `set`.
    CartPow(Box<ValueSet>, Dim),
}

impl ValueSet {
    /// Is the set bounded? `None` when not statically known. (Drives the
    /// `Lebesgue`/`truncate` total-mass rules.)
    pub fn is_bounded(&self) -> Option<bool> {
        use ValueSet::*;
        match self {
            UnitInterval | Booleans | StdSimplex(_) => Some(true),
            Interval(lo, hi) => Some(lo.is_finite() && hi.is_finite()),
            Reals | PosReals | NonNegReals | Integers | PosIntegers | NonNegIntegers
            | Complexes => Some(false),
            CartPow(elem, Dim::Static(_)) => elem.is_bounded(),
            CartPow(_, Dim::Dynamic) => None,
            Deferred | Unknown | Anything | RngStates => None,
        }
    }

    /// Conservative subset check: `true` means `self ⊆ other` is proven;
    /// `false` means unproven (NOT disproven).
    pub fn subset_of(&self, other: &ValueSet) -> bool {
        use ValueSet::*;
        if self == other {
            return !matches!(self, Deferred | Unknown);
        }
        match (self, other) {
            (_, Anything) => !matches!(self, Deferred | Unknown),
            (PosReals | NonNegReals | UnitInterval, Reals) => true,
            (PosReals | UnitInterval, NonNegReals) => true,
            (PosIntegers | NonNegIntegers, Integers | Reals) => true,
            (PosIntegers, NonNegIntegers | PosReals | NonNegReals) => true,
            (NonNegIntegers, NonNegReals) => true,
            (Integers, Reals) => true,
            (Interval(lo, hi), other) => match other {
                Reals => true,
                NonNegReals => *lo >= 0.0,
                PosReals => *lo > 0.0,
                UnitInterval => *lo >= 0.0 && *hi <= 1.0,
                _ => false,
            },
            (StdSimplex(n), CartPow(elem, d)) => {
                (n == d || matches!(d, Dim::Dynamic))
                    && UnitInterval.subset_of(elem.as_ref())
            }
            (CartPow(a, n), CartPow(b, d)) => {
                (n == d || matches!(d, Dim::Dynamic)) && a.subset_of(b)
            }
            _ => false,
        }
    }
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
