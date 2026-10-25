//! The declarative builtin/standard-module signature catalogue: RON data
//! (built-in + host-supplied external) deserialized into `Sig` rows, plus the
//! `lower` bridge to core inference types. Per-name *signatures* only;
//! structural inference stays in `ops.rs`/`trace.rs`.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Deserialize;

mod lower;
pub(crate) use lower::{LowerCtx, bin_count_shape, lower, no_intern};

#[derive(Debug, Clone, Deserialize)]
pub struct Catalogue {
    pub(crate) base: Vec<Builtin>,
    pub(crate) modules: Vec<Module>,
    /// Base rows are immutable after parsing. Keep their declaration order
    /// while sharing the name lookup across inference passes.
    #[serde(skip)]
    base_index: OnceLock<HashMap<String, usize>>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Builtin {
    pub(crate) name: String,
    pub(crate) sig: Sig,
    /// Honest-degrade note (design policy): set when this row's support/shape
    /// is a sound approximation of the spec §08 entry that the type system
    /// cannot express exactly (e.g. param-dependent integer-bounded supports).
    ///
    /// Parsed from RON for schema fidelity, but base-builtin degraded notes have
    /// no runtime surfacing path (only standard-module notes are reported via
    /// `module`), so the field is deserialized and never read.
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) degraded: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Module {
    pub(crate) name: String,
    pub(crate) version: String,
    pub(crate) bindings: Vec<Binding>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Binding {
    pub(crate) name: String,
    pub(crate) sig: Sig,
    #[serde(default)]
    pub(crate) degraded: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) enum Sig {
    Distribution {
        domain: DomainSig,
        support: SupportTag,
        mass: MassTag,
        /// Ordered constructor parameter names (spec §08/§09 "Parameters"
        /// column), e.g. `Normal` → `["mu", "sigma"]`. Consumed by the
        /// determiniser to build a density record's named fields from
        /// positional call arguments, and by [`Catalogue::base_arity`] for the
        /// call-arity rule. `#[serde(default)]` so parsing does not
        /// break mid-migration — every row is expected to fill it in.
        #[serde(default)]
        params: Vec<String>,
        /// Declared RANK (number of axes) of each parameter, parallel to
        /// `params` — `MvNormal` → `[1, 2]`, since §08 gives `mu` as a "mean
        /// vector" and `cov` as an "$n \times n$" matrix. Read by
        /// [`Self::distribution_param_ranks`] for §06's `ksuperpose` family-axis
        /// rule, which measures a family argument's axes against the rank of the
        /// parameter it feeds.
        ///
        /// **An empty list means every parameter is a scalar (rank 0).** That is
        /// the shape of all but the seven shaped rows below, and of every §09
        /// module row, so the default states the common case instead of leaving
        /// it unread. `param_ranks_are_parallel_to_params` pins the length.
        #[serde(default)]
        param_ranks: Vec<usize>,
        /// Declared VALUE SET of each parameter, parallel to `params` — §08's
        /// "Parameters" bullets, e.g. `Normal` → `[Reals, PosReals]` for
        /// `mu = elementof(reals)` / `sigma = elementof(posreals)`. Read by
        /// [`Catalogue::base_param_sets`] for the constant-argument check.
        ///
        /// **An empty list means no parameter is checked**, and a shorter list
        /// checks only the prefix it covers, so a row states exactly the sets the
        /// spec states and nothing is invented for the rest.
        #[serde(default)]
        param_sets: Vec<ParamSet>,
    },
    Function {
        /// Declared parameter list. Its length is read by [`Arity::of`] for the
        /// call-arity rule, and each tag's shape class by [`sig_param_shapes`]
        /// for the argument shape check.
        params: Vec<ParamSig>,
        /// Ordered parameter NAMES, from §07's "Arguments" column (or §06's prose
        /// for a measure operation), e.g. `atan2` → `["y", "x"]`. Parallel to
        /// `params`, which carries only type tags.
        ///
        /// Read by [`Catalogue::base_param_names`] so §04's name-binding rule
        /// applies to a splatted call on this row, exactly as it already does for
        /// a `Distribution`. **`#[serde(default)]` is load-bearing, not just a
        /// migration escape:** an empty list means "this row documents no names",
        /// and `base_param_names` then answers `None`, which the arity check reads
        /// as "accept" — so an undocumented row stays permissive instead of
        /// refusing every splat. See that method for which rows are deliberately
        /// left empty.
        #[serde(default)]
        names: Vec<String>,
        result: ResultSig,
        /// The result's value-set, when tighter than the result type's natural
        /// extent (e.g. `sqrt → nonnegreals`, `invlogit → unitinterval`).
        /// Defaults to `Natural` (= `ValueSet::natural_of(result_type)`), so a
        /// row that does not constrain its range needs no entry.
        #[serde(default)]
        result_set: ResultSet,
        /// Declared VALUE SET of each parameter, parallel to `params` — §08's
        /// "Parameters" bullets, e.g. `Normal` → `[Reals, PosReals]` for
        /// `mu = elementof(reals)` / `sigma = elementof(posreals)`. Read by
        /// [`Catalogue::base_param_sets`] for the constant-argument check.
        ///
        /// **An empty list means no parameter is checked**, and a shorter list
        /// checks only the prefix it covers, so a row states exactly the sets the
        /// spec states and nothing is invented for the rest.
        #[serde(default)]
        param_sets: Vec<ParamSet>,
    },
    /// A builtin whose §07 parameter list is declared here but whose result type
    /// is computed structurally in `ops.rs` (operand promotion, container shape
    /// construction, measure-algebra threading). `ResultSig` cannot express
    /// those, so the row carries the arity alone: `lower` never sees it and
    /// `function_result` returns `None` for it, leaving the `ops.rs` arm
    /// authoritative for the type.
    Structural {
        params: Vec<ParamSig>,
        /// Ordered parameter NAMES — see [`Sig::Function::names`]; identical
        /// contract, including the permissive-when-empty default.
        #[serde(default)]
        names: Vec<String>,
        /// Declared VALUE SET of each parameter, parallel to `params` — §08's
        /// "Parameters" bullets, e.g. `Normal` → `[Reals, PosReals]` for
        /// `mu = elementof(reals)` / `sigma = elementof(posreals)`. Read by
        /// [`Catalogue::base_param_sets`] for the constant-argument check.
        ///
        /// **An empty list means no parameter is checked**, and a shorter list
        /// checks only the prefix it covers, so a row states exactly the sets the
        /// spec states and nothing is invented for the rest.
        #[serde(default)]
        param_sets: Vec<ParamSet>,
    },
}

/// The admissible positional-argument count of a catalogue row: `min` required
/// parameters, and `max` = `None` for a trailing [`ParamSig::Variadic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arity {
    pub min: usize,
    pub max: Option<usize>,
}

impl Arity {
    /// Derive the arity of a declared parameter list. Required parameters count
    /// toward `min`; a trailing `Optional` raises `max` without raising `min`;
    /// a trailing `Variadic` removes the upper bound.
    fn of(params: &[ParamSig]) -> Arity {
        let mut min = 0;
        let mut max = Some(0);
        for p in params {
            match p {
                ParamSig::Variadic(_) => max = None,
                ParamSig::Optional(_) => max = max.map(|m| m + 1),
                _ => {
                    min += 1;
                    max = max.map(|m| m + 1);
                }
            }
        }
        Arity { min, max }
    }

    /// True when `got` arguments satisfy this arity.
    pub fn admits(&self, got: usize) -> bool {
        got >= self.min && self.max.is_none_or(|m| got <= m)
    }

    /// The declared count as it reads in a diagnostic: `1 argument`,
    /// `2 arguments`, `1 or 2 arguments`, `at least 3 arguments`. The noun agrees
    /// with the last number in the phrase, so `at least 1` is singular.
    pub fn describe(&self) -> String {
        let (count, last) = match self.max {
            Some(max) if max == self.min => (format!("{max}"), max),
            Some(max) if max == self.min + 1 => (format!("{} or {max}", self.min), max),
            Some(max) => (format!("{} to {max}", self.min), max),
            None => (format!("at least {}", self.min), self.min),
        };
        let noun = if last == 1 { "argument" } else { "arguments" };
        format!("{count} {noun}")
    }
}

/// The value-set of a function result, tighter than its type's natural extent.
/// Applied by [`lower`] only when the result's scalar kind matches the tag's
/// domain (a real-range tag on a complex result falls back to the natural set),
/// and lifted over a rank-1 array result into a `CartPow`. `Natural` (the
/// default) is exactly `ValueSet::natural_of(result_type)`.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub(crate) enum ResultSet {
    #[default]
    Natural,
    Reals,
    PosReals,
    NonNegReals,
    UnitInterval,
    Integers,
    PosIntegers,
    NonNegIntegers,
    Booleans,
    Complexes,
    /// A closed real range `[lo, hi]` (infinities allowed for half-bounded
    /// ranges): `tanh → interval(-1, 1)`, `erfc → interval(0, 2)`.
    Interval(f64, f64),
}

/// The declared VALUE SET of one call parameter, for the constant-argument
/// check. Read off §08's "Parameters" bullets (`sigma = elementof(posreals)`) or
/// §07's Domains column (`quantile`'s `p` over `interval(0, 1)`).
///
/// This is a VALUE constraint, not a type constraint: `ParamSig` carries the
/// parameter's type tag, whose shape class is checked for every argument, while
/// a `ParamSet` is checked against a statically-known constant argument by
/// `ops::refuse_constant_outside_declared_set`.
///
/// **`Unconstrained` is the default and means "no check".** A row lists a set
/// only where the spec states one, so an undocumented parameter stays permissive
/// — the same prove-it-is-wrong discipline `base_param_names` uses. `Reals` is
/// worth declaring even though no real constant can fall outside it: it records
/// that the spec DID state the set, which is what stops a later reader from
/// mistaking silence for an omission.
///
/// Deliberately NOT `SupportTag`: that enum carries dimension-aware entries
/// (`StdSimplex`, `CartPowReals`) and a `Structural` escape that belong to a
/// distribution's variate support, none of which a scalar constant can be tested
/// against. A vector or matrix parameter is left `Unconstrained` here for the same
/// reason — §08 gives `Dirichlet`'s `alpha` as an "array of positive reals", which
/// is a per-element claim this check does not reach.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq)]
pub(crate) enum ParamSet {
    #[default]
    Unconstrained,
    Reals,
    PosReals,
    NonNegReals,
    UnitInterval,
    Integers,
    PosIntegers,
    NonNegIntegers,
    Booleans,
    /// A closed real range `[lo, hi]` — §03 "Interval": "`interval(lo, hi)`
    /// denotes the closed interval $[lo, hi]$."
    Interval(f64, f64),
}

impl ParamSet {
    /// Does this set contain `v`? `None` when the set states no constraint, so a
    /// caller can tell "admitted" from "not checked".
    ///
    /// Every bound is CLOSED where the spec closes it: §03 gives `nonnegreals` as
    /// $[0, +\infty]$, `posreals` as $(0, +\infty]$, and `unitinterval` as
    /// $[0, 1]$, and §03 "Interval" makes `interval(lo, hi)` the closed
    /// $[lo, hi]$. That closure is what keeps `Bernoulli(p = true)` legal: `1` is
    /// in $[0, 1]$.
    ///
    /// §03's infinity note is load-bearing here — "`posreals`, `nonnegreals`, and
    /// `reals` admit `inf`" — so `+inf` passes those three rather than being
    /// refused as out of range. A NaN constant satisfies no comparison and so
    /// answers `false` for every set; nothing constructs one statically today.
    pub(crate) fn contains(&self, v: f64) -> Option<bool> {
        use ParamSet::*;
        let integral = v.fract() == 0.0 && v.is_finite();
        Some(match self {
            Unconstrained => return None,
            Reals => !v.is_nan(),
            PosReals => v > 0.0,
            NonNegReals => v >= 0.0,
            UnitInterval => (0.0..=1.0).contains(&v),
            Integers => integral,
            PosIntegers => integral && v >= 1.0,
            NonNegIntegers => integral && v >= 0.0,
            Booleans => v == 0.0 || v == 1.0,
            Interval(lo, hi) => v >= *lo && v <= *hi,
        })
    }

    /// The set as the spec spells it, for the diagnostic.
    pub(crate) fn describe(&self) -> String {
        use ParamSet::*;
        match self {
            Unconstrained => "anything".to_string(),
            Reals => "reals".to_string(),
            PosReals => "posreals".to_string(),
            NonNegReals => "nonnegreals".to_string(),
            UnitInterval => "unitinterval".to_string(),
            Integers => "integers".to_string(),
            PosIntegers => "posintegers".to_string(),
            NonNegIntegers => "nonnegintegers".to_string(),
            Booleans => "booleans".to_string(),
            Interval(lo, hi) => format!("interval({lo}, {hi})"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) enum DomainSig {
    Scalar(ScalarTag),
    VectorFromParam {
        elem: ScalarTag,
        param: String,
    },
    DynMatrix,
    /// An integer array of counts, one count per bin of the edges `param` holds:
    /// §07 `bincounts` gives `n + 1` edges `n` bins, and a record of edge vectors
    /// one axis per field, in field order. `BinnedPoissonProcess`'s domain.
    BinCounts {
        param: String,
    },
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) enum ScalarTag {
    Real,
    Integer,
    Boolean,
    Complex,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) enum SupportTag {
    Reals,
    PosReals,
    NonNegReals,
    UnitInterval,
    Integers,
    PosIntegers,
    NonNegIntegers,
    Booleans,
    Complexes,
    Anything,
    /// Dimension-aware simplex: `ValueSet::StdSimplex(param_dim)`. Meaningful
    /// only for `VectorFromParam` domain entries; `param_dim` is read from the
    /// same named parameter as the `DomainSig::VectorFromParam.param` field.
    StdSimplex,
    /// Dimension-aware real Cartesian power: `ValueSet::CartPow(Reals, param_dim)`.
    CartPowReals,
    /// Dimension-aware non-negative integer Cartesian power: `ValueSet::CartPow(NonNegIntegers, param_dim)`.
    CartPowNonNegIntegers,
    /// Support not representable as a fixed tag (e.g. arg-dependent or matrix
    /// distributions). Lowers to `ValueSet::Unknown`.
    Unknown,
    /// The support is computed structurally from a call argument at inference
    /// time, not from a static tag.  The catalogue row carries the domain; the
    /// support MUST remain on the code path (ops.rs `distribution_support`).
    /// Task 4 dispatch MUST fall back to `distribution_support` for any row
    /// with this tag rather than reading the catalogue support.
    /// Lowers to `ValueSet::Unknown` (the static approximation; the live path
    /// gives the real support).
    Structural,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) enum MassTag {
    Normalized,
    Finite,
    LocallyFinite,
    Unknown,
}

// Parameter-type tags. The list's length is read by `Arity::of`, and each tag's
// SHAPE class (scalar, vector, matrix) by `sig_param_shapes` for the argument
// shape check. The `ScalarTag` payloads stay documentation: §03 admits the
// canonical inclusions "where specified by the language", and §07's Domains
// column does not say per row which it admits. `Optional` and `Variadic` are
// trailing markers (guarded by `trailing_markers_only`).
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub(crate) enum ParamSig {
    Scalar(ScalarTag),
    Vector(ScalarTag),
    Matrix,
    Callable,
    Any,
    /// May be omitted; the spec states the default (`diag(A)` → `k = 0`).
    Optional(Box<ParamSig>),
    /// Zero or more further arguments of this kind.
    Variadic(Box<ParamSig>),
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) enum ResultSig {
    Scalar(ScalarTag),
    /// Result is a real scalar for a real (or integer, or boolean) arg `i` and a
    /// complex scalar for a complex one — the codomain of §07's elementary
    /// functions, whose "Domains" column lists `reals`/`complexes` and never
    /// `integers`. An integer argument is admitted only through §03's
    /// `integers ⊂ reals`, so the real-domain function applies and the result is
    /// real: `exp(2)` is `real`, not `integer`.
    RealOrComplexOfArg(usize),
    DomainMap {
        arg: usize,
        map: Vec<(ScalarTag, ScalarTag)>,
    },
    Matrix {
        rows: DimExpr,
        cols: DimExpr,
    },
    /// Result is exactly arg `i`'s type — shape and element preserved. For
    /// identity-like, order-permuting, and cumulative ops whose output mirrors
    /// the input (`identity`, `reverse`, `cumsum`, `cumprod`).
    SameAsArg(usize),
    /// Result has arg `i`'s shape but a real element type (`real`, `imag`):
    /// a real scalar for a scalar argument, a real array of the same shape for
    /// an array argument.
    RealOfArgShape(usize),
    /// Result is the common type of args `i` and `j`: identical argument types
    /// pass through unchanged, otherwise the scalar promotion of the two
    /// (`integers ⊂ reals ⊂ complexes`). For `ifelse`'s two branches.
    CommonOf(usize, usize),
    /// Result is a scalar whose kind is arg `i`'s element kind, drilling array
    /// nesting (`det`, `trace`): a real matrix yields a real scalar, a complex
    /// matrix a complex scalar.
    ElemScalarKind(usize),
    /// Result is a rank-1 array (vector) of the given length and element type
    /// (`linspace`/`extlinspace` → real, `sizeof` → integer, `diag` → the
    /// argument's element kind via `ElemSig::OfArg`).
    Vector {
        len: DimExpr,
        elem: ElemSig,
    },
    /// Result is a rank-2 array (matrix) whose element type follows `elem`
    /// rather than being forced real — for element-preserving matrix maps
    /// (`inv`, `lower_cholesky`, `diagmat`): a complex matrix inverts to a
    /// complex matrix. (`ResultSig::Matrix` stays for always-real results.)
    MatrixElem {
        rows: DimExpr,
        cols: DimExpr,
        elem: ElemSig,
    },
    /// Transpose of arg `i`, preserving rank and element kind (`transpose`,
    /// `adjoint`): a rank-2 array's two dims are swapped; a vector's transpose
    /// is a transposed vector (spec §07: "the transpose of a vector is a
    /// transposed vector, not a single-row matrix") — same rank-1 array type.
    TransposeOf(usize),
    /// A record result with named fields, each field's type given by its own
    /// `ResultSig` (`qr` → `record(Q, R)`; more record-valued functions to
    /// come). Field names are interned via the lowering context's interner.
    Record(Vec<(String, ResultSig)>),
}

/// The element-type source of a `Vector` / `MatrixElem` result.
#[derive(Debug, Clone, Deserialize)]
pub(crate) enum ElemSig {
    Real,
    Integer,
    Boolean,
    Complex,
    /// The (array-drilled) element kind of positional arg `i`; defaults to real
    /// when that arg's kind is not statically known.
    OfArg(usize),
}

/// A result-shape dimension expression (for `Matrix` / `MatrixElem` / `Vector`
/// result sigs), resolved against the call args by `lower::dim_of`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) enum DimExpr {
    /// Unknown at this level → `%dynamic`.
    Dyn,
    /// Leading dim of positional arg `i` (rank-1 args).
    OfParam(usize),
    /// Flattened dim `axis` of positional arg `i` — drills array nesting, so a
    /// matrix's rows/cols are `Axis(i, 0)` / `Axis(i, 1)` whether it is a flat
    /// rank-2 array or a nested vector-of-vectors.
    Axis(usize, usize),
    /// Product of two dim expressions (e.g. `kron`'s pq × mn). Static only when
    /// both operands are static (overflow falls back to dynamic).
    Mul(Box<DimExpr>, Box<DimExpr>),
    /// A fixed number of appended entries, as in a spline's affine columns.
    Add(Box<DimExpr>, u32),
    /// The positive integer literal written at positional arg `i`, as in
    /// `linspace`'s point count `n`.
    WrittenSize(usize),
}

/// Parse a catalogue from RON source.
pub fn parse_catalogue(src: &str) -> Result<Catalogue, ron::error::SpannedError> {
    ron::from_str(src)
}

static BUILTIN: OnceLock<Catalogue> = OnceLock::new();

/// The process-global built-in catalogue (parsed once from `catalogue.ron`).
pub(crate) fn builtin() -> &'static Catalogue {
    BUILTIN.get_or_init(|| {
        let mut cat = parse_catalogue(include_str!("../../catalogue.ron"))
            .expect("built-in catalogue.ron must parse");
        // Standard modules (spec §09) each live in their own RON file under
        // `catalogues/`; parse and merge them into the base catalogue here.
        cat.modules = STD_MODULE_SRCS
            .iter()
            .map(|src| {
                ron::from_str::<Module>(src).expect("built-in standard-module .ron must parse")
            })
            .collect();
        cat
    })
}

/// Per-module RON sources for the spec-§09 standard modules, embedded at build.
const STD_MODULE_SRCS: &[&str] = &[
    include_str!("../../catalogues/particle-physics.ron"),
    include_str!("../../catalogues/pyhf-helpers.ron"),
    include_str!("../../catalogues/ext-linear-algebra.ron"),
    include_str!("../../catalogues/special-functions.ron"),
    include_str!("../../catalogues/polynomials.ron"),
    include_str!("../../catalogues/splines.ron"),
    include_str!("../../catalogues/distances.ron"),
];

/// The declared call arity of any catalogue row, base or §09 module member, or
/// `None` when the row declares no parameter list.
///
/// A distribution row's arity is exact: §08 "Built-in distributions" states that
/// "the names and order of the distribution parameters specified below define the
/// names and positional order of the kernel arguments", and no §08 entry gives a
/// parameter a default, so every one is required. An empty list stays unspecified.
///
/// Keyed on the `Sig` rather than on a name so a §09 member can be checked from
/// the resolution side-table, which holds the sig and not the catalogue index.
pub(crate) fn sig_arity(sig: &Sig) -> Option<Arity> {
    match sig {
        Sig::Function { params, .. } | Sig::Structural { params, .. } => Some(Arity::of(params)),
        Sig::Distribution { params, .. } if params.is_empty() => None,
        Sig::Distribution { params, .. } => Some(Arity {
            min: params.len(),
            max: Some(params.len()),
        }),
    }
}

/// The declared parameter VALUE SETS of any catalogue row, base or §09 module
/// member. Empty when the row states none, and a caller reads position `i` only
/// when the list reaches it — see [`Sig::Distribution::param_sets`].
pub(crate) fn sig_param_sets(sig: &Sig) -> &[ParamSet] {
    match sig {
        Sig::Distribution { param_sets, .. }
        | Sig::Function { param_sets, .. }
        | Sig::Structural { param_sets, .. } => param_sets,
    }
}

/// The shape class a parameter's declared domain fixes: §07's "scalars",
/// "vectors" and "matrices" Domains cells, and §08's `elementof(<scalar set>)`,
/// vector and matrix parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParamShape {
    Scalar,
    Vector,
    Matrix,
}

impl ParamShape {
    pub(crate) fn describe(self) -> &'static str {
        match self {
            ParamShape::Scalar => "a scalar",
            ParamShape::Vector => "a vector",
            ParamShape::Matrix => "a matrix",
        }
    }
}

/// The declared shape class of each parameter of any catalogue row, base or §09
/// module member, parallel to its positional parameters. `None` at a position
/// whose domain fixes no single shape class (`any`, "vectors, tables", a set,
/// a callable), so that position is not checked.
///
/// A distribution row reads its existing columns. A rank-1 or rank-2
/// `param_ranks` entry is a vector or matrix parameter. A rank-0 parameter with
/// a declared value set is a scalar: §08 writes every such parameter as
/// `elementof(<set>)` of a scalar set (`mu = elementof(reals)`). A rank-0
/// parameter without a set is `Uniform`'s `support`, a set, and stays unchecked.
/// So does the edges parameter of a `BinCounts` domain, which may be a record.
///
/// A function row reads its `ParamSig` tags, and a trailing `Variadic` tag
/// covers every further position, so the list may be longer than `params`.
pub(crate) fn sig_param_shapes(sig: &Sig, count: usize) -> Vec<Option<ParamShape>> {
    fn of(p: &ParamSig) -> Option<ParamShape> {
        match p {
            ParamSig::Scalar(_) => Some(ParamShape::Scalar),
            ParamSig::Vector(_) => Some(ParamShape::Vector),
            ParamSig::Matrix => Some(ParamShape::Matrix),
            ParamSig::Optional(inner) | ParamSig::Variadic(inner) => of(inner),
            ParamSig::Callable | ParamSig::Any => None,
        }
    }
    match sig {
        Sig::Distribution {
            domain,
            params,
            param_ranks,
            param_sets,
            ..
        } => (0..params.len().min(count))
            .map(|i| match param_ranks.get(i).copied().unwrap_or(0) {
                // §07 `bincounts` also takes a record of edge vectors.
                _ if matches!(domain, DomainSig::BinCounts { param } if *param == params[i]) => {
                    None
                }
                1 => Some(ParamShape::Vector),
                2 => Some(ParamShape::Matrix),
                0 if param_sets
                    .get(i)
                    .is_some_and(|s| *s != ParamSet::Unconstrained) =>
                {
                    Some(ParamShape::Scalar)
                }
                _ => None,
            })
            .collect(),
        Sig::Function { params, .. } | Sig::Structural { params, .. } => (0..count)
            .map(|i| match params.get(i) {
                Some(p) => of(p),
                None => match params.last() {
                    Some(ParamSig::Variadic(inner)) => of(inner),
                    _ => None,
                },
            })
            .collect(),
    }
}

/// The declared parameter NAMES of any catalogue row, or `None` when the row
/// declares none. See [`Catalogue::base_param_names`] for which rows are
/// deliberately nameless and why an empty list means "accept".
pub(crate) fn sig_param_names(sig: &Sig) -> Option<&[String]> {
    match sig {
        Sig::Distribution { params, .. } if !params.is_empty() => Some(params),
        Sig::Function { names, .. } | Sig::Structural { names, .. } if !names.is_empty() => {
            Some(names)
        }
        _ => None,
    }
}

impl Catalogue {
    /// Look up a base (built-in) distribution signature by name.
    pub(crate) fn base(&self, name: &str) -> Option<&Sig> {
        let index = self.base_index.get_or_init(|| {
            let mut index = HashMap::with_capacity(self.base.len());
            for (i, row) in self.base.iter().enumerate() {
                // Preserve the first-row precedence of the former linear scan.
                index.entry(row.name.clone()).or_insert(i);
            }
            index
        });
        index.get(name).map(|&i| &self.base[i].sig)
    }

    /// The declared call arity of base builtin `name`, or `None` when the
    /// catalogue declares no parameter list for it.
    ///
    /// A distribution row's arity is exact: §08 "Built-in distributions" states
    /// that "the names and order of the distribution parameters specified below
    /// define the names and positional order of the kernel arguments", and no
    /// §08 entry gives a parameter a default, so every one is required.
    pub fn base_arity(&self, name: &str) -> Option<Arity> {
        sig_arity(self.base(name)?)
    }

    /// A sole positional aggregate is kept whole when the declared interface
    /// admits one positional argument, before checking its domain.
    pub(crate) fn base_takes_aggregate_whole(&self, name: &str) -> bool {
        self.base_arity(name).is_some_and(|arity| arity.admits(1))
    }

    /// Declared input names for ordinary keyword binding and aggregate spreading.
    /// A missing name list cannot name-bind a splatted aggregate; an interface
    /// that admits one positional input keeps that aggregate whole instead.
    pub fn base_param_names(&self, name: &str) -> Option<&[String]> {
        sig_param_names(self.base(name)?)
    }

    /// The declared parameter value sets of base builtin `name`, empty when the
    /// row states none. See [`ParamSet`] for why silence means "not checked".
    pub(crate) fn base_param_sets(&self, name: &str) -> &[ParamSet] {
        self.base(name).map_or(&[], sig_param_sets)
    }

    /// Whether a variadic row has no declared names to bind splatted fields.
    /// This is consulted only after one-positional-input admission has failed.
    pub(crate) fn base_has_unnamed_variadic(&self, name: &str) -> bool {
        /// §04 rows whose variadic inputs may be NAMED — a splat can bind on these.
        const NAMED_VARIADIC: &[&str] = &[
            "record",
            "table",
            "functionof",
            "kernelof",
            "broadcast",
            "load_module",
        ];
        // A row documenting names for its fixed inputs is not an "unnamed" row: the ordinary
        // name check decides it. This is what keeps `builtin_sample` out of the guard.
        if self.base_param_names(name).is_some() {
            return false;
        }
        // §07's `vector` takes its ELEMENTS as arguments, so a sole aggregate is an element
        // question for §03, not a binding question for §04.
        if name == "vector" || NAMED_VARIADIC.contains(&name) {
            return false;
        }
        let params = match self.base(name) {
            Some(Sig::Function { params, .. }) | Some(Sig::Structural { params, .. }) => params,
            _ => return false,
        };
        params.iter().any(|p| matches!(p, ParamSig::Variadic(_)))
    }

    /// The spec section that documents `name`'s parameter names, for a diagnostic
    /// that cites where the reader should check. §08 for a distribution
    /// constructor, §06 for a measure operation, §07 for everything else.
    ///
    /// The measure-operation set is listed rather than inferred: `densityof`,
    /// `logdensityof` and `bijection` are `Sig::Function` rows like any §07 builtin,
    /// so nothing in the signature distinguishes them, and citing §07 for them would
    /// send a reader to a table they do not appear in — `bijection` occurs zero times
    /// in §07 and has its row and prose entry in §06 ("annotate `f` with inverse and
    /// log-volume for density evaluation", arguments `f`, `f_inv`, `logvolume`).
    ///
    /// **Only four entries below are live.** The other 20 are not base catalogue rows
    /// at all — `ops.rs` types them structurally, so `base_param_names` never reaches
    /// them and their §06 citation cannot fire. They are listed anyway so that a
    /// future catalogue row for any of them is sectioned correctly on arrival rather
    /// than silently citing §07. The live four are `densityof`, `logdensityof`,
    /// `totalmass` and `bijection`.
    pub fn base_param_section(&self, name: &str) -> &'static str {
        const MEASURE_OPS: &[&str] = &[
            // Live — these four are catalogue rows today.
            "densityof",
            "logdensityof",
            "totalmass",
            "bijection",
            // Not catalogue rows (typed in `ops.rs`); listed for future arrivals.
            "kchain",
            "jointchain",
            "locscale",
            "pushfwd",
            "restrict",
            "truncate",
            "weighted",
            "logweighted",
            "normalize",
            "superpose",
            "mixture",
            "iid",
            "bayesupdate",
            "likelihoodof",
            "disintegrate",
            "kernelof",
            "functionof",
            "lawof",
            "joint",
            "relabel",
        ];
        if self.base_is_distribution(name) {
            "§08"
        } else if MEASURE_OPS.contains(&name) {
            "§06"
        } else {
            "§07"
        }
    }

    /// True iff base builtin `name` exists and has a distribution signature.
    /// Used by the LSP to bias completion ordering after a `~` binding.
    pub fn base_is_distribution(&self, name: &str) -> bool {
        matches!(self.base(name), Some(Sig::Distribution { .. }))
    }

    /// Look up a standard-module binding.  Returns `(sig, degraded_note)` or
    /// `None` if the module or binding is not in the catalogue.
    pub(crate) fn module(&self, module: &str, binding: &str) -> Option<(&Sig, Option<&str>)> {
        self.modules
            .iter()
            .find(|m| m.name == module)
            .and_then(|m| {
                m.bindings
                    .iter()
                    .find(|b| b.name == binding)
                    .map(|b| (&b.sig, b.degraded.as_deref()))
            })
    }

    /// Look up a standard module's version string.
    pub(crate) fn module_version(&self, module: &str) -> Option<&str> {
        self.modules
            .iter()
            .find(|m| m.name == module)
            .map(|m| m.version.as_str())
    }

    /// All base (§07/§08) builtin names.
    pub fn base_names(&self) -> impl Iterator<Item = &str> {
        self.base.iter().map(|b| b.name.as_str())
    }

    /// True iff `name` is a binding of ANY standard module (spec §09).
    ///
    /// A §09 member is reachable only through its module alias
    /// (`pp.kallen(...)`) — §09 "Standard modules" gives no way to call one
    /// unqualified, and the `base` namespace does not contain it. So this is a
    /// name that must NOT resolve bare; see `crate::builtins::is_base_name`,
    /// which consults a base row first so a name that is somehow both still
    /// resolves.
    pub fn is_module_member(&self, name: &str) -> bool {
        self.modules
            .iter()
            .any(|m| m.bindings.iter().any(|b| b.name == name))
    }

    /// The public binding names of a standard module, if present.
    pub fn module_binding_names(&self, module: &str) -> Option<impl Iterator<Item = &str>> {
        self.modules
            .iter()
            .find(|m| m.name == module)
            .map(|m| m.bindings.iter().map(|b| b.name.as_str()))
    }

    /// Ordered parameter names of a standard-module member (spec §09), e.g.
    /// module `pp` member `kallen`. `None` when the module or the member is
    /// absent, or when the row documents no names — the same
    /// "empty list means undocumented" contract as [`Self::base_param_names`].
    ///
    /// The module counterpart of `base_param_names`, so a caller that needs the
    /// roster for `alias.member(…)` reads it from the catalogue instead of
    /// carrying its own copy.
    pub fn module_param_names(&self, module: &str, binding: &str) -> Option<&[String]> {
        match self.module(module, binding)?.0 {
            Sig::Distribution { params, .. } if !params.is_empty() => Some(params),
            Sig::Function { names, .. } | Sig::Structural { names, .. } if !names.is_empty() => {
                Some(names)
            }
            _ => None,
        }
    }

    /// The declared call arity of a standard-module member (spec §09), or `None`
    /// when the row declares no parameter list. The module counterpart of
    /// [`Self::base_arity`], with the same distribution-is-exact rule.
    pub fn module_arity(&self, module: &str, binding: &str) -> Option<Arity> {
        sig_arity(self.module(module, binding)?.0)
    }

    /// Ordered constructor parameter names for a built-in distribution (spec
    /// §08/§09 "Parameters" column), e.g. `Normal` → `["mu", "sigma"]`.
    /// Checks base builtins first, then every standard module's bindings.
    /// `None` if `name` isn't a distribution, or isn't found at all.
    pub fn distribution_param_names(&self, name: &str) -> Option<Vec<String>> {
        if let Some(Sig::Distribution { params, .. }) = self.base(name) {
            return Some(params.clone());
        }
        self.modules.iter().find_map(|m| {
            m.bindings
                .iter()
                .find(|b| b.name == name)
                .and_then(|b| match &b.sig {
                    Sig::Distribution { params, .. } => Some(params.clone()),
                    Sig::Function { .. } | Sig::Structural { .. } => None,
                })
        })
    }

    /// Declared mass of a distribution constructor, including standard modules.
    pub fn distribution_mass(&self, name: &str) -> Option<flatppl_core::Mass> {
        let sig = self.base(name).or_else(|| {
            self.modules.iter().find_map(|module| {
                module
                    .bindings
                    .iter()
                    .find(|binding| binding.name == name)
                    .map(|b| &b.sig)
            })
        })?;
        match sig {
            Sig::Distribution { mass, .. } => Some(lower::mass(*mass)),
            _ => None,
        }
    }

    /// Declared parameter RANKS for a built-in distribution, parallel to
    /// [`Self::distribution_param_names`] — `MvNormal` → `[1, 2]`. A row that
    /// declares no ranks reports every parameter as rank 0 (see
    /// [`Sig::Distribution::param_ranks`]). `None` under the same conditions as
    /// `distribution_param_names`.
    pub fn distribution_param_ranks(&self, name: &str) -> Option<Vec<usize>> {
        let ranks = |params: &Vec<String>, param_ranks: &Vec<usize>| {
            if param_ranks.is_empty() {
                vec![0; params.len()]
            } else {
                param_ranks.clone()
            }
        };
        if let Some(Sig::Distribution {
            params,
            param_ranks,
            ..
        }) = self.base(name)
        {
            return Some(ranks(params, param_ranks));
        }
        self.modules.iter().find_map(|m| {
            m.bindings
                .iter()
                .find(|b| b.name == name)
                .and_then(|b| match &b.sig {
                    Sig::Distribution {
                        params,
                        param_ranks,
                        ..
                    } => Some(ranks(params, param_ranks)),
                    Sig::Function { .. } | Sig::Structural { .. } => None,
                })
        })
    }
}

/// A merged view of the built-in catalogue plus zero or more host-supplied
/// external catalogues.  Built-in is always consulted first; external
/// catalogues are consulted in slice order.
///
/// Used by `InferSession` to resolve `standard_module` references: existing
/// callers pass `external: &[]` and see identical behaviour to before.
pub(crate) struct CatalogueSet<'a> {
    pub(crate) builtin: &'static Catalogue,
    pub(crate) external: &'a [Catalogue],
}

impl<'a> CatalogueSet<'a> {
    /// Build a set backed by only the built-in catalogue (no external sources).
    pub(crate) fn builtin_only() -> Self {
        CatalogueSet {
            builtin: builtin(),
            external: &[],
        }
    }

    /// Build a set with host-supplied external catalogues.
    pub(crate) fn with_external(external: &'a [Catalogue]) -> Self {
        CatalogueSet {
            builtin: builtin(),
            external,
        }
    }

    /// Look up a standard-module binding across all sources (built-in first,
    /// then external in order).  Returns `(sig, degraded_note)` or `None`.
    pub(crate) fn module(&self, module: &str, binding: &str) -> Option<(&Sig, Option<&str>)> {
        self.builtin
            .module(module, binding)
            .or_else(|| self.external.iter().find_map(|c| c.module(module, binding)))
    }

    /// Look up a module's version string across all sources (built-in first).
    pub(crate) fn module_version(&self, module: &str) -> Option<&str> {
        self.builtin
            .module_version(module)
            .or_else(|| self.external.iter().find_map(|c| c.module_version(module)))
    }

    /// Check for duplicate module names across all sources.  A name that
    /// appears in more than one source (built-in vs external, or two
    /// externals) is an error, as is a name appearing twice within a single
    /// external catalogue.  Returns `Err("duplicate standard module 'NAME'")`;
    /// multiple collisions are reported as a newline-joined string.
    pub(crate) fn check_collisions(&self) -> Result<(), String> {
        let mut errors: Vec<String> = Vec::new();

        for ext_cat in self.external {
            // Within a single external catalogue, flag names that appear more
            // than once (untrusted third-party sources may contain duplicates).
            let mut seen_in_cat: std::collections::HashSet<&str> = std::collections::HashSet::new();
            for ext_mod in &ext_cat.modules {
                if !seen_in_cat.insert(ext_mod.name.as_str()) {
                    errors.push(format!("duplicate standard module '{}'", ext_mod.name));
                    continue; // one error per name is enough
                }
                // Collides with built-in?
                if self.builtin.module_version(&ext_mod.name).is_some() {
                    errors.push(format!("duplicate standard module '{}'", ext_mod.name));
                    continue; // one error per name is enough
                }
                // Collides with an earlier external?
                let earlier_dup = self
                    .external
                    .iter()
                    .take_while(|c| !std::ptr::eq(*c, ext_cat))
                    .any(|c| c.module_version(&ext_mod.name).is_some());
                if earlier_dup {
                    errors.push(format!("duplicate standard module '{}'", ext_mod.name));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("\n"))
        }
    }
}

#[cfg(test)]
mod tests {
    use flatppl_core::{Dim, Mass, ScalarType, Type};

    use super::*;
    use crate::catalogue::lower::{LowerCtx, lower};
    use crate::ops::{
        distribution_domain_static, distribution_support_static, function_type_static,
    };

    #[test]
    fn parses_a_minimal_catalogue() {
        let src = r#"Catalogue(
            base: [ Builtin(name: "Normal", sig: Distribution(domain: Scalar(Real), support: Reals, mass: Normalized)) ],
            modules: [ Module(name: "m", version: "0.1", bindings: [
                Binding(name: "f", sig: Function(params: [Scalar(Real)], result: RealOrComplexOfArg(0))),
            ]) ],
        )"#;
        let cat = parse_catalogue(src).expect("parses");
        assert_eq!(cat.base.len(), 1);
        assert_eq!(cat.modules[0].bindings[0].name, "f");
    }

    /// Every distribution in the catalogue must lower to exactly the same
    /// (domain Type, mass, support ValueSet) that the legacy ops.rs rules
    /// produce.  `param_dim` is set to `Dynamic` (the pre-Shape-level default)
    /// matching the production behavior when no concrete argument is available.
    #[test]
    fn catalogue_faithful_to_legacy_ops() {
        const NAMES: &[&str] = &[
            // Continuous univariate
            "Normal",
            "GeneralizedNormal",
            "Cauchy",
            "StudentT",
            "Logistic",
            "VonMises",
            "Laplace",
            "LogNormal",
            "Gamma",
            "InverseGamma",
            "ChiSquared",
            "Exponential",
            "Weibull",
            "Beta",
            "Pareto",
            // Uniform is intentionally excluded from NAMES: its support is
            // SupportTag::Structural, which lowers to Unknown (the static
            // approximation). The faithfulness test's support comparison is not
            // meaningful for structural supports; the live arg-dependent behavior
            // is guarded by `uniform_support_is_the_argument_set`.  The domain
            // comparison (Scalar(Real)) would still pass, but including Uniform
            // here would imply static-support faithfulness which is a false
            // guarantee for structural distributions.
            // Discrete univariate
            "Bernoulli",
            "Categorical",
            "Categorical0",
            "Binomial",
            "Geometric",
            "NegativeBinomial",
            "NegativeBinomial2",
            "Poisson",
            // Multivariate
            "MvNormal",
            "Dirichlet",
            "Multinomial",
            // Matrix
            "Wishart",
            "InverseWishart",
            "LKJ",
            "LKJCholesky",
        ];

        // Use Dynamic for all param dims — matches pre-Shape-level inference.
        let param_dim_fn: &dyn Fn(&str) -> Dim = &|_| Dim::Dynamic;

        let cat = builtin();

        for name in NAMES {
            // --- Legacy oracle ---
            let legacy_domain = distribution_domain_static(name, param_dim_fn)
                .unwrap_or_else(|| panic!("{name}: not in legacy distribution_domain_static"));
            let legacy_support = distribution_support_static(name, param_dim_fn);

            // --- Catalogue lower ---
            let sig = cat
                .base(name)
                .unwrap_or_else(|| panic!("{name}: missing from built-in catalogue"));

            let ctx = LowerCtx {
                arg_scalar: &|_| Some(ScalarType::Real),
                param_dim: param_dim_fn,
                arg_dim: &|_| Dim::Dynamic,
                arg_type: &|_| None,
                written_size: &|_| Dim::Dynamic,
                intern: &no_intern,
            };
            let (cat_ty, cat_support) = lower(sig, &ctx);

            // The catalogue type must be Measure(domain, Normalized).
            let (cat_domain, cat_mass) = match cat_ty {
                Type::Measure { domain, mass } => (*domain, mass),
                other => panic!("{name}: catalogue lowered to {other:?}, expected Measure"),
            };

            // Domain comparison.
            assert_eq!(
                cat_domain, legacy_domain,
                "{name}: catalogue domain {cat_domain:?} != legacy {legacy_domain:?}"
            );

            // Mass: every §08 distribution is Normalized.
            assert_eq!(
                cat_mass,
                Mass::Normalized,
                "{name}: catalogue mass {cat_mass:?} != Normalized"
            );

            // Support comparison.
            assert_eq!(
                cat_support, legacy_support,
                "{name}: catalogue support {cat_support:?} != legacy {legacy_support:?}"
            );
        }
    }

    /// Every migrated per-name function in the catalogue must lower to exactly
    /// the same result type that the old per-name call_rule arms produced.
    /// `function_type_static` is the static oracle (encodes the old arm logic);
    /// `function_result` (via `lower`) is the catalogue path.
    ///
    /// Two argument-scalar scenarios are tested for each function:
    ///   - arg0 = `None` (no concrete type — default behaviour)
    ///   - arg0 = `Some(Complex)` (complex-in path, relevant for RealOrComplexOfArg)
    #[test]
    fn catalogue_functions_faithful_to_legacy_ops() {
        // (name, arg0_scalar) pairs to exercise.
        // For RealOrComplexOfArg fns the complex path matters; for fixed-output fns both
        // should return the same constant type.
        let cases: &[(&str, Option<ScalarType>)] = &[
            // scalar-integer output
            ("floor", None),
            ("floor", Some(ScalarType::Complex)),
            ("ceil", None),
            ("round", None),
            ("integer", None),
            ("div", None),
            ("mod", None),
            ("lengthof", None),
            // scalar-real output
            // (divide and mean are structural, not catalogue rows — covered by
            // golden tests divide_promotes_complex_operands / mean_reduces_to_element_type)
            ("logdensityof", None),
            ("densityof", None),
            ("l1norm", None),
            ("l2norm", None),
            ("logsumexp", None),
            // scalar-real output whatever the argument: §07 gives `log2`,
            // `log10`, `gamma` and `loggamma` the `posreals` domain with no
            // `complexes` beside it, so the complex-in path is real too.
            ("log2", None),
            ("log2", Some(ScalarType::Complex)),
            ("log2", Some(ScalarType::Integer)),
            ("log10", None),
            ("log10", Some(ScalarType::Complex)),
            ("gamma", None),
            ("gamma", Some(ScalarType::Complex)),
            ("loggamma", None),
            ("loggamma", Some(ScalarType::Complex)),
            ("loggamma", Some(ScalarType::Integer)),
            // scalar-complex output
            ("cis", None),
            ("complex", None),
            // scalar-boolean output
            ("equal", None),
            ("unequal", None),
            ("lt", None),
            ("le", None),
            ("gt", None),
            ("ge", None),
            ("in", None),
            ("land", None),
            ("lor", None),
            ("lnot", None),
            ("isfinite", None),
            ("isinf", None),
            ("isnan", None),
            ("iszero", None),
            // RealOrComplexOfArg(0): real→real
            ("exp", None),
            ("exp", Some(ScalarType::Real)),
            ("log", None),
            ("sqrt", None),
            ("sin", None),
            ("cos", None),
            ("tan", None),
            ("asin", None),
            ("acos", None),
            ("atan", None),
            ("sinh", None),
            ("cosh", None),
            ("tanh", None),
            ("asinh", None),
            ("acosh", None),
            ("atanh", None),
            ("log1p", None),
            ("expm1", None),
            ("logit", None),
            ("invlogit", None),
            ("probit", None),
            ("invprobit", None),
            // RealOrComplexOfArg(0): complex→complex
            ("exp", Some(ScalarType::Complex)),
            ("log", Some(ScalarType::Complex)),
            ("sqrt", Some(ScalarType::Complex)),
            ("conj", None),
            ("conj", Some(ScalarType::Complex)),
            // RealOrComplexOfArg(0): integer→REAL. §07's "Domains" column for the
            // elementary functions never lists `integers`, so an integer argument
            // is admitted only via §03's `integers ⊂ reals` and the result is real.
            ("exp", Some(ScalarType::Integer)),
            ("log", Some(ScalarType::Integer)),
            ("sqrt", Some(ScalarType::Integer)),
            ("sin", Some(ScalarType::Integer)),
            ("conj", Some(ScalarType::Integer)),
            ("invlogit", Some(ScalarType::Integer)),
            // abs / abs2: complex→real (DomainMap)
            ("abs", None),
            ("abs", Some(ScalarType::Real)),
            ("abs", Some(ScalarType::Complex)),
            ("abs2", None),
            ("abs2", Some(ScalarType::Complex)),
        ];

        let cat = builtin();

        for &(name, arg0_scalar) in cases {
            // Oracle: what the old per-name arm returned.
            let legacy = function_type_static(name, arg0_scalar)
                .unwrap_or_else(|| panic!("{name}: not in function_type_static oracle"));

            // Catalogue path.
            let sig = cat
                .base(name)
                .unwrap_or_else(|| panic!("{name}: missing from built-in catalogue"));
            let ctx = LowerCtx {
                arg_scalar: &|i| if i == 0 { arg0_scalar } else { None },
                param_dim: &|_| Dim::Dynamic,
                arg_dim: &|_| Dim::Dynamic,
                arg_type: &|_| None,
                written_size: &|_| Dim::Dynamic,
                intern: &no_intern,
            };
            let (cat_ty, _) = lower(sig, &ctx);

            assert_eq!(
                cat_ty, legacy,
                "{name}(arg0={arg0_scalar:?}): catalogue {cat_ty:?} != legacy {legacy:?}"
            );
        }
    }

    /// An external catalogue that lists the same module name twice is a
    /// duplicate-within-source collision and must produce an error.
    #[test]
    fn internal_duplicate_in_external_catalogue_errors() {
        let dup_ron = r#"Catalogue(
            base: [],
            modules: [
                Module(name: "dup", version: "0.1", bindings: [
                    Binding(name: "Foo", sig: Distribution(domain: Scalar(Real), support: Reals, mass: Normalized)),
                ]),
                Module(name: "dup", version: "0.2", bindings: [
                    Binding(name: "Bar", sig: Distribution(domain: Scalar(Real), support: Reals, mass: Normalized)),
                ]),
            ],
        )"#;
        let dup_cat = parse_catalogue(dup_ron).expect("dup_ron parses");
        let set = CatalogueSet::with_external(std::slice::from_ref(&dup_cat));
        let result = set.check_collisions();
        assert!(result.is_err(), "expected a collision error; got Ok(())");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("duplicate standard module 'dup'"),
            "error should name 'dup'; got {msg:?}"
        );
    }

    /// Completeness guard for the six §09 standard modules.
    ///
    /// Asserts that every public binding listed in §09 is present in the
    /// built-in catalogue, and that every row with a `degraded` note has a
    /// non-empty string (so the notes don't silently rot to empty strings).
    #[test]
    fn std_modules_complete() {
        // (module_name, version, [binding names])
        let modules: &[(&str, &str, &[&str])] = &[
            (
                "particle-physics",
                "0.1",
                &[
                    // distributions
                    "CrystalBall",
                    "DoubleSidedCrystalBall",
                    "Argus",
                    "RelativisticBreitWigner",
                    "Voigtian",
                    "BifurcatedNormal",
                    "ContinuedPoisson",
                    // interpolation functions
                    "interp_pwlin",
                    "interp_pwexp",
                    "interp_poly2_lin",
                    "interp_poly6_lin",
                    "interp_poly6_exp",
                    // resonance functions
                    "resonance_breitwigner",
                    // kinematics functions
                    "kallen",
                    "breakup_momentum",
                    "blatt_weisskopf",
                    // Wigner rotation functions
                    "wignerd",
                    "wignerD",
                    "wignerd_doublearg",
                    "wignerD_doublearg",
                ],
            ),
            (
                "ext-linear-algebra",
                "0.1",
                &[
                    "lu", "svd", "eigen", "eigmax", "eigmin", "matexp", "kron", "lstsq", "rank",
                ],
            ),
            (
                "special-functions",
                "0.1",
                &[
                    "erf",
                    "erfc",
                    "bessel_j",
                    "bessel_y",
                    "bessel_i",
                    "bessel_k",
                    "digamma",
                    "polygamma",
                    "gammainc",
                    "betainc",
                    "airy",
                ],
            ),
            (
                "polynomials",
                "0.1",
                &["legendre", "hermite", "laguerre", "chebyshev"],
            ),
            (
                "distances",
                "0.1",
                &[
                    "pairwise_distance",
                    "cross_distance",
                    "euclidean",
                    "squared_euclidean",
                    "cosine",
                    "manhattan",
                    "chebyshev",
                    "minkowski",
                    "jensenshannon",
                ],
            ),
        ];

        let cat = builtin();

        for &(mod_name, expected_version, bindings) in modules {
            // Version present.
            let version = cat
                .module_version(mod_name)
                .unwrap_or_else(|| panic!("module '{mod_name}' missing from catalogue"));
            assert_eq!(
                version, expected_version,
                "module '{mod_name}' version mismatch"
            );

            // Every binding present; degraded notes (if any) are non-empty.
            for &binding_name in bindings {
                let (_, degraded) = cat.module(mod_name, binding_name).unwrap_or_else(|| {
                    panic!("module '{mod_name}': binding '{binding_name}' missing")
                });
                if let Some(note) = degraded {
                    assert!(
                        !note.is_empty(),
                        "module '{mod_name}' binding '{binding_name}': degraded note is empty string"
                    );
                }
            }
        }
    }

    #[test]
    fn enumerates_base_and_module_binding_names() {
        let cat = builtin();
        assert!(cat.base_names().any(|n| n == "Normal"));
        let pp: Vec<&str> = cat
            .module_binding_names("particle-physics")
            .unwrap()
            .collect();
        assert!(pp.contains(&"CrystalBall"));
        assert!(cat.module_binding_names("no-such-module").is_none());
    }

    /// `distribution_param_names` looks up the ordered constructor parameter
    /// names (spec §08/§09 "Parameters" column) across both base builtins and
    /// standard-module bindings, and returns `None` for non-distributions and
    /// unknown names.
    #[test]
    fn distribution_param_names_looks_up_base_and_module() {
        let cat = builtin();
        // Base distribution.
        assert_eq!(
            cat.distribution_param_names("Normal"),
            Some(vec!["mu".to_string(), "sigma".to_string()])
        );
        // Module distribution (particle-physics).
        assert_eq!(
            cat.distribution_param_names("CrystalBall"),
            Some(vec![
                "m0".to_string(),
                "sigma".to_string(),
                "alpha".to_string(),
                "n".to_string(),
            ])
        );
        // A base function (not a distribution) → None.
        assert_eq!(cat.distribution_param_names("sqrt"), None);
        // Unknown name → None.
        assert_eq!(cat.distribution_param_names("NotARealBuiltin"), None);
    }

    /// Every `Sig::Distribution` in the base catalogue and every standard
    /// module must carry its `params` list (spec §08/§09 "Parameters"
    /// column) — mirrors `catalogue_faithful_to_legacy_ops`'s enumeration, but
    /// scoped to param-name completeness rather than domain/support/mass
    /// faithfulness.
    #[test]
    fn distribution_param_names_are_complete() {
        let cat = builtin();

        for b in &cat.base {
            if let Sig::Distribution { params, .. } = &b.sig {
                assert!(
                    !params.is_empty(),
                    "base distribution '{}' has empty params",
                    b.name
                );
            }
        }

        for m in &cat.modules {
            for binding in &m.bindings {
                if let Sig::Distribution { params, .. } = &binding.sig {
                    assert!(
                        !params.is_empty(),
                        "module '{}' distribution '{}' has empty params",
                        m.name,
                        binding.name
                    );
                }
            }
        }
    }

    /// A declared `param_ranks` list must cover every parameter, so the
    /// family-axis rule never reads a rank off the wrong parameter. An empty
    /// list is the all-scalar default and is admitted.
    #[test]
    fn param_ranks_are_parallel_to_params() {
        let cat = builtin();
        let check = |what: &str, params: &Vec<String>, param_ranks: &Vec<usize>| {
            assert!(
                param_ranks.is_empty() || param_ranks.len() == params.len(),
                "{what}: {} param_ranks for {} params",
                param_ranks.len(),
                params.len()
            );
        };
        for b in &cat.base {
            if let Sig::Distribution {
                params,
                param_ranks,
                ..
            } = &b.sig
            {
                check(&b.name, params, param_ranks);
            }
        }
        for m in &cat.modules {
            for binding in &m.bindings {
                if let Sig::Distribution {
                    params,
                    param_ranks,
                    ..
                } = &binding.sig
                {
                    check(&format!("{}.{}", m.name, binding.name), params, param_ranks);
                }
            }
        }
    }

    /// The shaped §08 rows: `MvNormal` `mu` is a vector and `cov` a matrix,
    /// `Multinomial` `n` a scalar and `p` a vector. An unshaped row reports
    /// rank 0 for every parameter.
    #[test]
    fn distribution_param_ranks_read_the_declared_shapes() {
        let cat = builtin();
        assert_eq!(cat.distribution_param_ranks("MvNormal"), Some(vec![1, 2]));
        assert_eq!(
            cat.distribution_param_ranks("Multinomial"),
            Some(vec![0, 1])
        );
        assert_eq!(cat.distribution_param_ranks("Dirichlet"), Some(vec![1]));
        assert_eq!(cat.distribution_param_ranks("Wishart"), Some(vec![0, 2]));
        assert_eq!(cat.distribution_param_ranks("Normal"), Some(vec![0, 0]));
        // A §09 module distribution: every parameter is a scalar.
        assert_eq!(
            cat.distribution_param_ranks("CrystalBall"),
            Some(vec![0, 0, 0, 0])
        );
        assert_eq!(cat.distribution_param_ranks("sqrt"), None);
    }

    #[test]
    fn base_is_distribution_classifies_builtins() {
        let cat = builtin();
        // "Normal" is a §08 distribution in the base catalogue.
        assert!(
            cat.base_is_distribution("Normal"),
            "Normal must be a distribution"
        );
        // A base function (not a distribution) must be false.
        assert!(
            !cat.base_is_distribution("sqrt"),
            "sqrt is a function, not a distribution"
        );
        // An unknown name is false (not present in base).
        assert!(
            !cat.base_is_distribution("NotARealBuiltin"),
            "unknown name must be false"
        );
    }

    /// `Optional` and `Variadic` are trailing markers: a required parameter
    /// after one of them would make `Arity::of`'s `min` wrong.
    #[test]
    fn arity_markers_are_trailing_only() {
        fn check(what: &str, params: &[ParamSig]) {
            let mut seen_marker = false;
            for p in params {
                match p {
                    ParamSig::Optional(_) => seen_marker = true,
                    ParamSig::Variadic(_) => seen_marker = true,
                    _ => assert!(
                        !seen_marker,
                        "{what}: required parameter after an Optional/Variadic marker"
                    ),
                }
            }
            // A `Variadic` subsumes anything after it.
            let variadic_at = params
                .iter()
                .position(|p| matches!(p, ParamSig::Variadic(_)));
            if let Some(i) = variadic_at {
                assert_eq!(
                    i,
                    params.len() - 1,
                    "{what}: Variadic must be the last parameter"
                );
            }
        }

        let cat = builtin();
        for b in &cat.base {
            match &b.sig {
                Sig::Function { params, .. } | Sig::Structural { params, .. } => {
                    check(&b.name, params)
                }
                Sig::Distribution { .. } => {}
            }
        }
        for m in &cat.modules {
            for b in &m.bindings {
                match &b.sig {
                    Sig::Function { params, .. } | Sig::Structural { params, .. } => {
                        check(&format!("{}.{}", m.name, b.name), params)
                    }
                    Sig::Distribution { .. } => {}
                }
            }
        }
    }

    #[test]
    fn parsed_base_catalogue_preserves_first_row() {
        let cat = parse_catalogue(
            r#"Catalogue(base: [
                Builtin(name: "f", sig: Structural(params: [Any])),
                Builtin(name: "g", sig: Structural(params: [Any, Any])),
                Builtin(name: "f", sig: Structural(params: [Any, Any, Any])),
            ], modules: [])"#,
        )
        .unwrap();
        assert_eq!(
            cat.base_arity("f"),
            Some(Arity {
                min: 1,
                max: Some(1)
            })
        );
        assert_eq!(cat.base_names().collect::<Vec<_>>(), ["f", "g", "f"]);
    }

    /// The §07/§08 argument counts the arity rule enforces, including the two
    /// shapes a plain `params.len()` would get wrong: `diag`'s optional `k` and
    /// `builtin_sample`'s variadic sample shape.
    #[test]
    fn base_arity_reads_the_declared_parameter_list() {
        let cat = builtin();
        let fixed = |min: usize| {
            Some(Arity {
                min,
                max: Some(min),
            })
        };
        assert_eq!(cat.base_arity("exp"), fixed(1));
        assert_eq!(cat.base_arity("add"), fixed(2));
        assert_eq!(cat.base_arity("ifelse"), fixed(3));
        assert_eq!(cat.base_arity("bijection"), fixed(3));
        // §07: "when called as `diag(A)`, `k` defaults to `0`".
        assert_eq!(
            cat.base_arity("diag"),
            Some(Arity {
                min: 1,
                max: Some(2)
            })
        );
        // §07: `builtin_sample | rngstate, kernel, kernel_input, n, m, ...`.
        assert_eq!(
            cat.base_arity("builtin_sample"),
            Some(Arity { min: 3, max: None })
        );
        assert_eq!(
            cat.base_arity("builtin_logdensityof"),
            fixed(3),
            "the five fixed-arity primitives take kernel, kernel_input, x"
        );
        // §07: `get | container, selectors...` — a container and at least one selector.
        assert_eq!(cat.base_arity("get"), Some(Arity { min: 2, max: None }));
        // §08 distribution rows: every parameter is required, so the arity is
        // exactly the declared count.
        assert_eq!(cat.base_arity("Normal"), fixed(2));
        assert_eq!(cat.base_arity("StudentT"), fixed(1));
        assert_eq!(cat.base_arity("GeneralizedNormal"), fixed(3));
        assert_eq!(cat.base_arity("NotARealBuiltin"), None);
    }

    /// The §09 module counterparts of `base_param_names` / `base_arity`, which
    /// the LSP's signature help reads so the argument roster lives in one place.
    ///
    /// A distribution row carries its §09 "Parameters" names and an exact arity,
    /// and so — since every module `Function` row gained §09's Arguments column —
    /// does a function row. The "empty list means undocumented" contract
    /// `base_param_names` states is still live for a row that declares no names;
    /// no shipped module row is in that state today, so the case is asserted
    /// against a synthetic sig in `sig_param_names_reads_an_undocumented_row`.
    #[test]
    fn module_param_names_and_arity_read_the_declared_row() {
        let cat = builtin();
        assert_eq!(
            cat.module_param_names("particle-physics", "RelativisticBreitWigner"),
            Some(["mean".to_string(), "width".to_string()].as_slice())
        );
        assert_eq!(
            cat.module_arity("particle-physics", "RelativisticBreitWigner"),
            Some(Arity {
                min: 2,
                max: Some(2)
            })
        );
        assert_eq!(
            cat.module_param_names("special-functions", "bessel_j"),
            Some(["v".to_string(), "z".to_string()].as_slice()),
            "the row carries §09's Arguments column (`v`, `z`)"
        );
        assert_eq!(
            cat.module_arity("special-functions", "bessel_j"),
            Some(Arity {
                min: 2,
                max: Some(2)
            }),
            "the arity is still declared by the parameter-type list"
        );
        assert_eq!(
            cat.module_param_names("particle-physics", "NotAMember"),
            None
        );
        assert_eq!(cat.module_arity("no-such-module", "whatever"), None);
    }

    /// The permissive default: a row with an EMPTY `names` list answers `None`,
    /// which the arity check reads as "accept", so an undocumented row does not
    /// refuse every keyword call to it. No shipped module row is undocumented any
    /// more — every one carries §09's Arguments column — so the contract is
    /// asserted against a synthetic sig rather than a live row that could later
    /// gain names and silently stop covering this case.
    #[test]
    fn sig_param_names_reads_an_undocumented_row() {
        let undocumented = Sig::Function {
            params: vec![ParamSig::Scalar(ScalarTag::Real)],
            names: Vec::new(),
            result: ResultSig::Scalar(ScalarTag::Real),
            result_set: ResultSet::Natural,
            param_sets: Vec::new(),
        };
        assert_eq!(sig_param_names(&undocumented), None);
        assert_eq!(
            sig_arity(&undocumented),
            Some(Arity {
                min: 1,
                max: Some(1)
            }),
            "the arity is declared by the parameter-type list, names or not"
        );
    }

    #[test]
    fn arity_admits_and_describes() {
        let one_or_two = Arity {
            min: 1,
            max: Some(2),
        };
        assert!(!one_or_two.admits(0));
        assert!(one_or_two.admits(1));
        assert!(one_or_two.admits(2));
        assert!(!one_or_two.admits(3));
        assert_eq!(one_or_two.describe(), "1 or 2 arguments");

        let at_least_three = Arity { min: 3, max: None };
        assert!(!at_least_three.admits(2));
        assert!(at_least_three.admits(300));
        assert_eq!(at_least_three.describe(), "at least 3 arguments");

        assert_eq!(
            Arity {
                min: 2,
                max: Some(2)
            }
            .describe(),
            "2 arguments"
        );
        assert_eq!(
            Arity {
                min: 1,
                max: Some(3)
            }
            .describe(),
            "1 to 3 arguments"
        );
        // The noun agrees with the last number in the phrase, so both of these
        // are singular.
        assert_eq!(
            Arity {
                min: 1,
                max: Some(1)
            }
            .describe(),
            "1 argument"
        );
        assert_eq!(
            Arity { min: 1, max: None }.describe(),
            "at least 1 argument"
        );
    }
}
