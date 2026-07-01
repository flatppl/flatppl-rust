//! Types, shapes, and phases — the inference domain.
//!
//! Mirrors the FlatPIR type categories (spec §11). Two design points from
//! `ARCHITECTURE.md`: array *nesting* is preserved (`Array.elem` may itself be an
//! `Array`, so vec-of-vec ≠ matrix, spec §03), and a dimension is either a static
//! size or `%dynamic` — never a sentinel.

use crate::id::Symbol;
use std::fmt;

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
    /// `cartprod(S1, S2, …)` — heterogeneous product: the set of `cat`s of one
    /// element per component set (spec §03, "Cartesian product", positional
    /// form). A set of ARRAYS, not tuples — scalar components occupy one position
    /// each, vector components concatenate; the array element type is the common
    /// type of the components.
    CartProd(Box<[ValueSet]>),
    /// `cartprod(a = S1, b = S2, …)` — a record / table-column set: field `a`
    /// in `S1`, … (spec §03, keyword form). Field names are interned
    /// `Symbol`s, so this renders via `Module::display_valueset`, not the
    /// symbol-free `Display` impl.
    RecordSet(Box<[(Symbol, ValueSet)]>),
}

impl ValueSet {
    /// The **natural extent** of a type: the maximal set its values inhabit
    /// (`(%scalar real)` → `reals`, a 1-D real array → `cartpow(reals, n)`,
    /// a measure → its domain's extent). `Unknown` for non-value types
    /// (callables, modules) and for types without a set vocabulary. The
    /// value-set slot of a value-typed call is at least this (spec §11).
    pub fn natural_of(ty: &Type) -> ValueSet {
        match ty {
            Type::Scalar(ScalarType::Real) => ValueSet::Reals,
            Type::Scalar(ScalarType::Integer) => ValueSet::Integers,
            Type::Scalar(ScalarType::Boolean) => ValueSet::Booleans,
            Type::Scalar(ScalarType::Complex) => ValueSet::Complexes,
            Type::Array { shape, elem } => match ValueSet::natural_of(elem) {
                ValueSet::Unknown => ValueSet::Unknown,
                inner => cartpow_over(inner, shape),
            },
            Type::TVector { len, elem } => match ValueSet::natural_of(elem) {
                ValueSet::Unknown => ValueSet::Unknown,
                inner => ValueSet::CartPow(Box::new(inner), *len),
            },
            // Tuples are objects, not values (spec §04): a tuple has no
            // value-set. Its components carry their own value-sets through their
            // own types; the tuple itself contributes none.
            Type::Tuple(_) => ValueSet::Unknown,
            Type::Record(fields) => record_natural(fields),
            // A table's value-set is the `cartpow` of its row-record set over the
            // row count (spec §03: `cartpow(recordset, n)` is "the set of n-row
            // tables"). The row record is `Record(columns)` — each column stores
            // its per-row element type (scalar, array, or — for a table-valued
            // column — a record). This matches `load_data`'s table value-set
            // (`cartpow(recordset, %dynamic)`), so both representations agree.
            Type::Table { columns, nrows } => match record_natural(columns) {
                ValueSet::Unknown => ValueSet::Unknown,
                rec => ValueSet::CartPow(Box::new(rec), *nrows),
            },
            Type::Measure { domain, .. } => ValueSet::natural_of(domain),
            Type::RngState => ValueSet::RngStates,
            Type::Any => ValueSet::Anything,
            _ => ValueSet::Unknown,
        }
    }

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
            CartProd(parts) => product_bounded(parts.iter()),
            RecordSet(fields) => product_bounded(fields.iter().map(|(_, s)| s)),
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
                (n == d || matches!(d, Dim::Dynamic)) && UnitInterval.subset_of(elem.as_ref())
            }
            (CartPow(a, n), CartPow(b, d)) => {
                (n == d || matches!(d, Dim::Dynamic)) && a.subset_of(b)
            }
            // A heterogeneous product is a subset of a homogeneous power when the
            // flattened lengths agree and every component's element set refines
            // the power's element — e.g. `cartprod(reals, integers) ⊆
            // cartpow(reals, 2)`. This is what lets a positional `cartprod`'s
            // value-set be a refinement of its array type's natural extent (§11
            // refinement invariant). A `cartprod` member is the `cat` of one
            // element per component (spec §03), so a component contributes its
            // own flattened BLOCK of positions: a rank-0 scalar set one position
            // of that set, a rank-1 power `cartpow(e, n)` n positions of `e`, a
            // `stdsimplex(n)` n unit-interval positions — so the total length is
            // the SUM of block lengths, not the component count (all-scalar is
            // the special case where every block is length one).
            (CartProd(parts), CartPow(elem, d)) => {
                let mut total: u32 = 0;
                let elements_refine = parts.iter().all(|p| match p {
                    CartPow(e, Dim::Static(n)) => {
                        total += *n;
                        e.subset_of(elem)
                    }
                    StdSimplex(Dim::Static(n)) => {
                        total += *n;
                        UnitInterval.subset_of(elem)
                    }
                    // A block whose flattened length isn't statically known
                    // (a dynamic power / simplex) can't prove a length match.
                    CartPow(_, Dim::Dynamic) | StdSimplex(Dim::Dynamic) => false,
                    // Otherwise treat the component as a single rank-0 position:
                    // scalar sets refine directly; nested products / record sets
                    // fail `subset_of(elem)` (they don't cat into a scalar power).
                    other => {
                        total += 1;
                        other.subset_of(elem)
                    }
                });
                elements_refine && (matches!(d, Dim::Dynamic) || *d == Dim::Static(total))
            }
            (CartProd(a), CartProd(b)) if a.len() == b.len() => {
                a.iter().zip(b.iter()).all(|(x, y)| x.subset_of(y))
            }
            (RecordSet(a), RecordSet(b))
                if a.len() == b.len() && a.iter().zip(b.iter()).all(|((n, _), (m, _))| n == m) =>
            {
                a.iter()
                    .zip(b.iter())
                    .all(|((_, x), (_, y))| x.subset_of(y))
            }
            _ => false,
        }
    }
}

/// Right-nested Cartesian power over a multi-axis shape:
/// `[d0, d1, …, dk]` → `CartPow(… CartPow(elem, dk) …, d0)`. An empty shape
/// returns `elem` unchanged (rank-0 = a scalar set).
pub fn cartpow_over(elem: ValueSet, shape: &[Dim]) -> ValueSet {
    shape
        .iter()
        .rev()
        .fold(elem, |acc, &d| ValueSet::CartPow(Box::new(acc), d))
}

fn record_natural(fields: &[(Symbol, Type)]) -> ValueSet {
    let sets: Vec<(Symbol, ValueSet)> = fields
        .iter()
        .map(|(n, t)| (*n, ValueSet::natural_of(t)))
        .collect();
    if sets.iter().any(|(_, s)| matches!(s, ValueSet::Unknown)) {
        ValueSet::Unknown
    } else {
        ValueSet::RecordSet(sets.into())
    }
}

fn product_bounded<'a>(parts: impl Iterator<Item = &'a ValueSet>) -> Option<bool> {
    // bounded ⇔ every component bounded; an unbounded component short-circuits
    // to Some(false); an unknown component (and no unbounded one) is None; the
    // empty product is Some(true).
    let mut saw_none = false;
    for p in parts {
        match p.is_bounded() {
            Some(true) => {}
            Some(false) => return Some(false),
            None => saw_none = true,
        }
    }
    if saw_none { None } else { Some(true) }
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

// ── Human-facing rendering (concise, code-like surface notation) ─────────────
//
// These `Display` impls render the inference domain in a compact, readable
// notation for IDE surfaces, unlike the derived `Debug` (which prints Rust
// struct syntax). `Type` itself is rendered by `Module::display_type`, because
// naming its interned fields/inputs needs the module interner. Symbol-free
// domains (value-sets, phases, masses, scalars, dims) need no interner, so they
// get plain `Display` impls here.

impl ScalarType {
    /// The spec §11 scalar keyword (`real` / `integer` / `boolean` / `complex`).
    pub fn name(self) -> &'static str {
        match self {
            ScalarType::Real => "real",
            ScalarType::Integer => "integer",
            ScalarType::Boolean => "boolean",
            ScalarType::Complex => "complex",
        }
    }
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl Mass {
    /// Friendly total-mass word (`normalized`, `finite`, `locally-finite`, …).
    pub fn name(self) -> &'static str {
        match self {
            Mass::Deferred => "deferred",
            Mass::Null => "null",
            Mass::Normalized => "normalized",
            Mass::Finite => "finite",
            Mass::LocallyFinite => "locally-finite",
            Mass::Unknown => "unknown",
        }
    }
}

impl fmt::Display for Mass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Static(n) => write!(f, "{n}"),
            Dim::Dynamic => f.write_str("?"),
        }
    }
}

impl Phase {
    /// The spec §04 phase keyword (`fixed` / `parameterized` / `stochastic`).
    pub fn name(self) -> &'static str {
        match self {
            Phase::Fixed => "fixed",
            Phase::Parameterized => "parameterized",
            Phase::Stochastic => "stochastic",
        }
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl fmt::Display for ValueSet {
    /// The value-set surface vocabulary (`reals`, `cartpow(set, n)`, …).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ValueSet::*;
        match self {
            Deferred => f.write_str("deferred"),
            Unknown => f.write_str("unknown"),
            Reals => f.write_str("reals"),
            PosReals => f.write_str("posreals"),
            NonNegReals => f.write_str("nonnegreals"),
            UnitInterval => f.write_str("unitinterval"),
            Integers => f.write_str("integers"),
            PosIntegers => f.write_str("posintegers"),
            NonNegIntegers => f.write_str("nonnegintegers"),
            Booleans => f.write_str("booleans"),
            Complexes => f.write_str("complexes"),
            RngStates => f.write_str("rngstates"),
            Anything => f.write_str("anything"),
            StdSimplex(d) => write!(f, "stdsimplex({d})"),
            Interval(lo, hi) => write!(f, "interval({lo}, {hi})"),
            CartPow(set, d) => write!(f, "cartpow({set}, {d})"),
            CartProd(parts) => {
                let inner: Vec<String> = parts.iter().map(ToString::to_string).collect();
                write!(f, "cartprod({})", inner.join(", "))
            }
            RecordSet(fields) => {
                let inner: Vec<String> = fields.iter().map(|(_, s)| format!("_: {s}")).collect();
                write!(f, "record({})", inner.join(", "))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cartpow_over_nests_multi_axis() {
        use ValueSet::*;
        // [2, 3] → CartPow(CartPow(reals, 3), 2)
        let got = cartpow_over(Reals, &[Dim::Static(2), Dim::Static(3)]);
        assert_eq!(
            got,
            CartPow(
                Box::new(CartPow(Box::new(Reals), Dim::Static(3))),
                Dim::Static(2)
            )
        );
        // empty shape returns the element unchanged
        assert_eq!(cartpow_over(Reals, &[]), Reals);
        // single axis is a plain rank-1 power
        assert_eq!(
            cartpow_over(Integers, &[Dim::Static(4)]),
            CartPow(Box::new(Integers), Dim::Static(4))
        );
    }

    #[test]
    fn natural_of_lifts_rank2_and_products() {
        use crate::ty::{ScalarType, Type};
        use ValueSet::*;
        // flat rank-2 real array → nested CartPow
        let flat = Type::Array {
            shape: Box::new([Dim::Static(2), Dim::Static(3)]),
            elem: Box::new(Type::Scalar(ScalarType::Real)),
        };
        assert_eq!(
            ValueSet::natural_of(&flat),
            CartPow(
                Box::new(CartPow(Box::new(Reals), Dim::Static(3))),
                Dim::Static(2)
            )
        );
        // nested rank-1 array (matrix-as-vec-of-vec) → same nested CartPow
        let nested = Type::Array {
            shape: Box::new([Dim::Static(2)]),
            elem: Box::new(Type::Array {
                shape: Box::new([Dim::Static(3)]),
                elem: Box::new(Type::Scalar(ScalarType::Real)),
            }),
        };
        assert_eq!(ValueSet::natural_of(&nested), ValueSet::natural_of(&flat));
        // tuple → no value-set (spec §04: tuples are objects, not values)
        let tup = Type::Tuple(Box::new([
            Type::Scalar(ScalarType::Real),
            Type::Scalar(ScalarType::Integer),
        ]));
        assert_eq!(ValueSet::natural_of(&tup), Unknown);
    }

    #[test]
    fn natural_of_table_nests_array_and_table_columns() {
        use crate::ty::{ScalarType, Type};
        use ValueSet::*;
        let mut m = crate::module::Module::new();
        let a = m.intern("a");
        let b = m.intern("b");
        let hits = m.intern("hits");
        let x = m.intern("x");
        // A table whose columns store per-row elements: a scalar, a 3-vector
        // (array-valued column), and a record (a table-valued column).
        let table = Type::Table {
            columns: Box::new([
                (a, Type::Scalar(ScalarType::Real)),
                (
                    b,
                    Type::Array {
                        shape: Box::new([Dim::Static(3)]),
                        elem: Box::new(Type::Scalar(ScalarType::Real)),
                    },
                ),
                (
                    hits,
                    Type::Record(Box::new([(x, Type::Scalar(ScalarType::Real))])),
                ),
            ]),
            nrows: Dim::Static(2),
        };
        // The table value-set is `cartpow(rowRecord, nrows)` (spec §03 "set of
        // n-row tables"). The row record carries the 3-vector verbatim (NOT
        // stripped to a scalar) and nests the sub-table's record.
        assert_eq!(
            ValueSet::natural_of(&table),
            CartPow(
                Box::new(RecordSet(Box::new([
                    (a, Reals),
                    (b, CartPow(Box::new(Reals), Dim::Static(3))),
                    (hits, RecordSet(Box::new([(x, Reals)]))),
                ]))),
                Dim::Static(2)
            )
        );
    }

    #[test]
    fn product_subset_and_bounded() {
        use ValueSet::*;
        // componentwise subset, same length
        assert!(
            CartProd(Box::new([PosReals, UnitInterval]))
                .subset_of(&CartProd(Box::new([Reals, Reals])))
        );
        // length mismatch is not a subset
        assert!(!CartProd(Box::new([Reals])).subset_of(&CartProd(Box::new([Reals, Reals]))));
        // a heterogeneous product is a subset of a homogeneous power when the
        // lengths agree and each component lies in the element set
        assert!(
            CartProd(Box::new([Reals, Integers]))
                .subset_of(&CartPow(Box::new(Reals), Dim::Static(2)))
        );
        assert!(
            CartProd(Box::new([PosReals, UnitInterval]))
                .subset_of(&CartPow(Box::new(Reals), Dim::Dynamic))
        );
        // wrong length or an out-of-set component is not a subset
        assert!(
            !CartProd(Box::new([Reals, Integers]))
                .subset_of(&CartPow(Box::new(Reals), Dim::Static(3)))
        );
        assert!(
            !CartProd(Box::new([Reals, Integers]))
                .subset_of(&CartPow(Box::new(PosReals), Dim::Static(2)))
        );
        // bounded iff every component bounded
        assert_eq!(
            CartProd(Box::new([UnitInterval, Booleans])).is_bounded(),
            Some(true)
        );
        assert_eq!(
            CartProd(Box::new([UnitInterval, Reals])).is_bounded(),
            Some(false)
        );
        // multi-axis power bounded follows the element
        assert_eq!(
            cartpow_over(UnitInterval, &[Dim::Static(2), Dim::Static(2)]).is_bounded(),
            Some(true)
        );
        assert_eq!(
            cartpow_over(Reals, &[Dim::Static(2), Dim::Static(2)]).is_bounded(),
            Some(false)
        );
    }

    #[test]
    fn product_of_vector_components_subset_of_flat_power() {
        use ValueSet::*;
        // Spec §03 (clarified): a `cartprod` member is the `cat` of one element
        // per component, so VECTOR components concatenate —
        // `cartprod(cartpow(reals,2), cartpow(reals,3))` is the set of flat
        // 5-vectors. Its value-set must be provable ⊆ the type's natural extent
        // `cartpow(reals, 5)` (§11 refinement invariant): the flattened length is
        // the SUM of block lengths (5), not the component count (2), and every
        // block's element set must refine the power's element.
        assert!(
            CartProd(Box::new([
                CartPow(Box::new(Reals), Dim::Static(2)),
                CartPow(Box::new(Reals), Dim::Static(3)),
            ]))
            .subset_of(&CartPow(Box::new(Reals), Dim::Static(5)))
        );
        // Heterogeneous block element sets: a reals block ++ an integers block is
        // still a real 5-vector (integers ⊂ reals) — matches the §03 element-type
        // rule (`cartprod(reals, integers)` is real-valued).
        assert!(
            CartProd(Box::new([
                CartPow(Box::new(Reals), Dim::Static(2)),
                CartPow(Box::new(Integers), Dim::Static(3)),
            ]))
            .subset_of(&CartPow(Box::new(Reals), Dim::Static(5)))
        );
        // A scalar block contributes one position; mixing with a vector block is
        // still sound set algebra (the scalar occupies one flattened slot).
        assert!(
            CartProd(Box::new([
                Reals,
                CartPow(Box::new(Integers), Dim::Static(3))
            ]))
            .subset_of(&CartPow(Box::new(Reals), Dim::Static(4)))
        );
        // A `stdsimplex(n)` block contributes n unit-interval positions.
        assert!(
            CartProd(Box::new([StdSimplex(Dim::Static(3)), Reals]))
                .subset_of(&CartPow(Box::new(Reals), Dim::Static(4)))
        );
        // Wrong total length is not a subset (2 + 3 = 5, not 6).
        assert!(
            !CartProd(Box::new([
                CartPow(Box::new(Reals), Dim::Static(2)),
                CartPow(Box::new(Reals), Dim::Static(3)),
            ]))
            .subset_of(&CartPow(Box::new(Reals), Dim::Static(6)))
        );
        // A dynamic block length can't prove the total → not a provable subset.
        assert!(
            !CartProd(Box::new([
                CartPow(Box::new(Reals), Dim::Dynamic),
                CartPow(Box::new(Reals), Dim::Static(3)),
            ]))
            .subset_of(&CartPow(Box::new(Reals), Dim::Static(5)))
        );
        // An out-of-set block element is not a subset (an integers block ⊄ a
        // posreals power).
        assert!(
            !CartProd(Box::new([
                CartPow(Box::new(Reals), Dim::Static(2)),
                CartPow(Box::new(Integers), Dim::Static(3)),
            ]))
            .subset_of(&CartPow(Box::new(PosReals), Dim::Static(5)))
        );
    }

    #[test]
    fn recordset_subset_matches_fields() {
        use ValueSet::*;
        let mut m = crate::module::Module::new();
        let a = m.intern("a");
        let b = m.intern("b");
        let lhs = RecordSet(Box::new([(a, PosReals), (b, UnitInterval)]));
        let rhs = RecordSet(Box::new([(a, Reals), (b, Reals)]));
        assert!(lhs.subset_of(&rhs));
        // mismatched field order / names is not a subset
        let swapped = RecordSet(Box::new([(b, Reals), (a, Reals)]));
        assert!(!lhs.subset_of(&swapped));
    }

    #[test]
    fn subset_chains() {
        use ValueSet::*;
        assert!(PosReals.subset_of(&Reals));
        assert!(PosIntegers.subset_of(&NonNegIntegers));
        assert!(PosIntegers.subset_of(&NonNegReals));
        assert!(UnitInterval.subset_of(&NonNegReals));
        assert!(!UnitInterval.subset_of(&PosReals)); // contains 0
        assert!(!NonNegReals.subset_of(&PosReals));
        assert!(Interval(0.25, 0.75).subset_of(&UnitInterval));
        assert!(Interval(0.0, f64::INFINITY).subset_of(&NonNegReals));
        assert!(!Interval(-1.0, 1.0).subset_of(&NonNegReals));
        // Deferred/Unknown prove nothing, not even reflexively.
        assert!(!Unknown.subset_of(&Unknown));
        assert!(!Deferred.subset_of(&Anything));
        // The simplex sits inside nonnegative unit vectors.
        let nn = CartPow(Box::new(NonNegReals), Dim::Dynamic);
        assert!(StdSimplex(Dim::Static(3)).subset_of(&nn));
        assert!(
            CartPow(Box::new(UnitInterval), Dim::Static(3))
                .subset_of(&CartPow(Box::new(Reals), Dim::Static(3)))
        );
        assert!(
            !CartPow(Box::new(Reals), Dim::Static(3))
                .subset_of(&CartPow(Box::new(Reals), Dim::Static(4)))
        );
    }

    #[test]
    fn boundedness() {
        use ValueSet::*;
        assert_eq!(UnitInterval.is_bounded(), Some(true));
        assert_eq!(StdSimplex(Dim::Dynamic).is_bounded(), Some(true));
        assert_eq!(Interval(0.0, 1.0).is_bounded(), Some(true));
        assert_eq!(Interval(0.0, f64::INFINITY).is_bounded(), Some(false));
        assert_eq!(Reals.is_bounded(), Some(false));
        assert_eq!(
            CartPow(Box::new(UnitInterval), Dim::Static(3)).is_bounded(),
            Some(true)
        );
        assert_eq!(
            CartPow(Box::new(UnitInterval), Dim::Dynamic).is_bounded(),
            None
        );
        assert_eq!(Unknown.is_bounded(), None);
    }

    #[test]
    fn display_surface_vocab() {
        use ValueSet::*;
        assert_eq!(ScalarType::Real.to_string(), "real");
        assert_eq!(Mass::Normalized.to_string(), "normalized");
        assert_eq!(Mass::LocallyFinite.to_string(), "locally-finite");
        assert_eq!(Phase::Stochastic.to_string(), "stochastic");
        assert_eq!(Dim::Static(3).to_string(), "3");
        assert_eq!(Dim::Dynamic.to_string(), "?");
        assert_eq!(Reals.to_string(), "reals");
        assert_eq!(StdSimplex(Dim::Static(4)).to_string(), "stdsimplex(4)");
        assert_eq!(Interval(0.0, 1.0).to_string(), "interval(0, 1)");
        assert_eq!(
            CartPow(Box::new(Reals), Dim::Dynamic).to_string(),
            "cartpow(reals, ?)"
        );
    }

    #[test]
    fn natural_extents() {
        assert_eq!(
            ValueSet::natural_of(&Type::Scalar(ScalarType::Real)),
            ValueSet::Reals
        );
        let vec3 = Type::Array {
            shape: Box::new([Dim::Static(3)]),
            elem: Box::new(Type::Scalar(ScalarType::Integer)),
        };
        assert_eq!(
            ValueSet::natural_of(&vec3),
            ValueSet::CartPow(Box::new(ValueSet::Integers), Dim::Static(3))
        );
        // A measure's extent is its domain's; callables have none.
        let m = Type::Measure {
            domain: Box::new(Type::Scalar(ScalarType::Real)),
            mass: Mass::Normalized,
        };
        assert_eq!(ValueSet::natural_of(&m), ValueSet::Reals);
        let k = Type::Kernel {
            inputs: Box::new([]),
            mass: Mass::Normalized,
        };
        assert_eq!(ValueSet::natural_of(&k), ValueSet::Unknown);
    }
}
