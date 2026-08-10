//! The declarative builtin/standard-module signature catalogue: RON data
//! (built-in + host-supplied external) deserialized into `Sig` rows, plus the
//! `lower` bridge to core inference types. Per-name *signatures* only;
//! structural inference stays in `ops.rs`/`trace.rs`.

use std::sync::OnceLock;

use serde::Deserialize;

mod lower;
pub(crate) use lower::{LowerCtx, lower, no_intern};

#[derive(Debug, Clone, Deserialize)]
pub struct Catalogue {
    pub(crate) base: Vec<Builtin>,
    pub(crate) modules: Vec<Module>,
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
    },
    Function {
        /// Declared parameter list. `lower` does not type-check arguments
        /// (result inference is structural over the call), so only the list's
        /// *shape* is read — by [`Arity::of`], for the call-arity rule.
        params: Vec<ParamSig>,
        result: ResultSig,
        /// The result's value-set, when tighter than the result type's natural
        /// extent (e.g. `sqrt → nonnegreals`, `invlogit → unitinterval`).
        /// Defaults to `Natural` (= `ValueSet::natural_of(result_type)`), so a
        /// row that does not constrain its range needs no entry.
        #[serde(default)]
        result_set: ResultSet,
    },
    /// A builtin whose §07 parameter list is declared here but whose result type
    /// is computed structurally in `ops.rs` (operand promotion, container shape
    /// construction, measure-algebra threading). `ResultSig` cannot express
    /// those, so the row carries the arity alone: `lower` never sees it and
    /// `function_result` returns `None` for it, leaving the `ops.rs` arm
    /// authoritative for the type.
    Structural { params: Vec<ParamSig> },
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

#[derive(Debug, Clone, Deserialize)]
pub(crate) enum DomainSig {
    Scalar(ScalarTag),
    VectorFromParam { elem: ScalarTag, param: String },
    DynMatrix,
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

// Parameter-type tags. The *payloads* are documentation of the spec's declared
// domains — `lower` does not consult them, so they are never read; the list's
// shape is read by `Arity::of`. `Optional` and `Variadic` are trailing markers
// (guarded by `trailing_markers_only`).
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
    include_str!("../../catalogues/ext-linear-algebra.ron"),
    include_str!("../../catalogues/special-functions.ron"),
    include_str!("../../catalogues/polynomials.ron"),
    include_str!("../../catalogues/distances.ron"),
];

impl Catalogue {
    /// Look up a base (built-in) distribution signature by name.
    pub(crate) fn base(&self, name: &str) -> Option<&Sig> {
        self.base.iter().find(|b| b.name == name).map(|b| &b.sig)
    }

    /// The declared call arity of base builtin `name`, or `None` when the
    /// catalogue declares no parameter list for it.
    ///
    /// A distribution row's arity is exact: §08 "Built-in distributions" states
    /// that "the names and order of the distribution parameters specified below
    /// define the names and positional order of the kernel arguments", and no
    /// §08 entry gives a parameter a default, so every one is required. An
    /// unfilled `params` (the `#[serde(default)]` migration escape) yields
    /// `None` rather than an invented arity of zero.
    pub fn base_arity(&self, name: &str) -> Option<Arity> {
        match self.base(name)? {
            Sig::Function { params, .. } | Sig::Structural { params } => Some(Arity::of(params)),
            Sig::Distribution { params, .. } if params.is_empty() => None,
            Sig::Distribution { params, .. } => Some(Arity {
                min: params.len(),
                max: Some(params.len()),
            }),
        }
    }

    /// Base builtins whose documented domain admits a record or a TABLE, and so
    /// take a sole positional aggregate whole instead of auto-splatting it —
    /// §04's single-input carve-out (flatppl-design#78, pending owner review):
    /// "A callable with exactly one input whose documented domain admits records
    /// or tables is exempt and receives a sole positional record or table whole,
    /// so that `sum(t)` and `lengthof(t)` reduce over the table rather than
    /// splatting."
    ///
    /// Derived by classifying all 96 single-input base builtins against §07's
    /// Domains column and prose, not by taking #78's two examples as the list. Each
    /// member, with where its aggregate domain is documented:
    ///
    /// - `lengthof` ("vectors, tables"), `reverse` ("vectors, tables"), `indicesof`
    ///   and `indicesof0` ("vectors, arrays, tables") — the Domains cell names
    ///   tables outright. §03 "Tables" backs `lengthof`: "`lengthof(t)` returns the
    ///   number of table rows."
    /// - `identity` ("any") — an unrestricted domain admits records and tables, and
    ///   a function returning its argument unchanged must not restructure it.
    /// - `sum`, `mean`, `var`, `std` — their Domains cells say only "real/complex
    ///   arrays" / "real arrays"; the table domain lives in §07's **Table
    ///   reductions** paragraph ("When `sum`, `mean`, `var`, or `std` is applied to a
    ///   table, the reduction operates column-wise and returns a record whose fields
    ///   are the column names"). #78 names `sum(t)` normatively for exactly this
    ///   reason. `std` was added to that paragraph by an owner ruling on 2026-08-10
    ///   (flatppl-design `4c93237`, onto #77) after this guard first shipped without
    ///   it — it is $\sqrt{\mathrm{var}}$, so a column-wise `var` implies a
    ///   column-wise `std`.
    ///
    /// Deliberately ABSENT, each checked against its own row rather than assumed:
    ///
    /// - `boolean`, `integer`, `real` — "any **scalar** numeric". The word "any" is
    ///   qualified, so these do not admit aggregates.
    /// - `sizeof` ("vectors, arrays"), `prod` ("real/complex arrays"), and every
    ///   other reduction/norm/stack row — arrays only.
    /// - `qr` — RETURNS `record(Q, R)`, but its domain is "$m \times n$ matrices".
    ///   The carve-out is about the domain, not the result.
    /// - `totalmass` — §06, and its input is a measure, not an aggregate.
    /// - `length` and `log2` — catalogue rows with no §07 entry at all, so no
    ///   documented domain to admit anything.
    /// - Every single-input §08 constructor (`Poisson`, `Dirichlet`, `Categorical`,
    ///   `Exponential`, …) — scalar or vector domains, never aggregates. This is
    ///   what keeps `Poisson(record(zzz = 0.5))` a static error.
    /// - `get`, `get0` ("records, arrays, tables, tuples") and `filter` ("function,
    ///   array or table") DO admit aggregates in their cells, but they are
    ///   MULTI-input, so #78's "exactly one input" half excludes them and they never
    ///   reach this list. They are also absent from the single-input arity set, so
    ///   the exclusion holds twice over.
    ///
    /// The caller pairs this with the arity half of #78's condition, so a row that
    /// later gains a second parameter stops being exempt without this list changing
    /// — see [`Catalogue::base_takes_aggregate_whole`].
    const AGGREGATE_DOMAIN_BUILTINS: &[&str] = &[
        "identity",
        "indicesof",
        "indicesof0",
        "lengthof",
        "mean",
        "reverse",
        "std",
        "sum",
        "var",
    ];

    /// True iff base builtin `name` satisfies BOTH halves of §04's single-input
    /// carve-out: exactly one declared input, and a documented domain admitting
    /// records or tables ([`Self::AGGREGATE_DOMAIN_BUILTINS`]).
    ///
    /// Both halves are read off the CALLEE's signature, never the caller's field
    /// names, which is what keeps the rule decidable at the call site and leaves
    /// every multi-input splat alone. Arity is checked here rather than trusted of
    /// the list: `Exponential(record(rate = 1.0))` is one input but a `reals`
    /// domain, and a hypothetical two-input `sum` would stop being exempt on its
    /// own.
    pub(crate) fn base_takes_aggregate_whole(&self, name: &str) -> bool {
        let single_input = matches!(
            self.base_arity(name),
            Some(Arity {
                min: 1,
                max: Some(1)
            })
        );
        single_input && Self::AGGREGATE_DOMAIN_BUILTINS.contains(&name)
    }

    /// The declared parameter NAMES of base builtin `name`, for the §04
    /// name-binding rule. Only distribution rows have them: a `Sig::Function` /
    /// `Sig::Structural` row declares `ParamSig` type tags, whose §07 names live
    /// in the row's comment rather than in the data.
    pub fn base_param_names(&self, name: &str) -> Option<&[String]> {
        match self.base(name)? {
            Sig::Distribution { params, .. } if !params.is_empty() => Some(params),
            _ => None,
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

    /// The public binding names of a standard module, if present.
    pub fn module_binding_names(&self, module: &str) -> Option<impl Iterator<Item = &str>> {
        self.modules
            .iter()
            .find(|m| m.name == module)
            .map(|m| m.bindings.iter().map(|b| b.name.as_str()))
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
            ("length", None),
            // scalar-real output
            // (divide and mean are structural, not catalogue rows — covered by
            // golden tests divide_promotes_complex_operands / mean_reduces_to_element_type)
            ("logdensityof", None),
            ("densityof", None),
            ("l1norm", None),
            ("l2norm", None),
            ("logsumexp", None),
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
            // log2: §07 divergence — not in spec but kept for compatibility.
            ("log2", None),
            ("log10", None),
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
            ("gamma", None),
            ("loggamma", None),
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
            ("loggamma", Some(ScalarType::Integer)),
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
    /// module must carry a non-empty `params` list (spec §08/§09 "Parameters"
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
                Sig::Function { params, .. } | Sig::Structural { params } => check(&b.name, params),
                Sig::Distribution { .. } => {}
            }
        }
        for m in &cat.modules {
            for b in &m.bindings {
                match &b.sig {
                    Sig::Function { params, .. } | Sig::Structural { params } => {
                        check(&format!("{}.{}", m.name, b.name), params)
                    }
                    Sig::Distribution { .. } => {}
                }
            }
        }
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
