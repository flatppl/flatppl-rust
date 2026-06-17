//! The declarative builtin/standard-module signature catalogue: RON data
//! (built-in + host-supplied external) deserialized into `Sig` rows, plus the
//! `lower` bridge to core inference types. Per-name *signatures* only;
//! structural inference stays in `ops.rs`/`trace.rs`.

use std::sync::OnceLock;

use serde::Deserialize;

mod lower;
pub(crate) use lower::{LowerCtx, lower};

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
    },
    Function {
        // Declared parameter types: parsed from RON for schema fidelity, but
        // `lower` does not type-check function arguments (result inference is
        // structural over the call), so the field is never read.
        #[allow(dead_code)]
        params: Vec<ParamSig>,
        result: ResultSig,
    },
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

// Parameter-type tags: parsed alongside `Sig::Function.params` for RON schema
// fidelity, but `lower` does not consult them, so the payloads are never read.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub(crate) enum ParamSig {
    Scalar(ScalarTag),
    Vector(ScalarTag),
    Matrix,
    Callable,
    Any,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) enum ResultSig {
    Scalar(ScalarTag),
    SameScalarKind(usize),
    DomainMap {
        arg: usize,
        map: Vec<(ScalarTag, ScalarTag)>,
    },
    Matrix {
        rows: DimExpr,
        cols: DimExpr,
    },
}

// Matrix dimension expressions: parsed for RON schema fidelity. Lowering maps
// every `ResultSig::Matrix` to a dynamic-dim matrix (the type system has no
// shape arithmetic), so the parameter indices are never read.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub(crate) enum DimExpr {
    Dyn,
    OfParam(usize),
    MulDims(usize, usize),
}

/// Parse a catalogue from RON source.
pub fn parse_catalogue(src: &str) -> Result<Catalogue, ron::error::SpannedError> {
    ron::from_str(src)
}

static BUILTIN: OnceLock<Catalogue> = OnceLock::new();

/// The process-global built-in catalogue (parsed once from `catalogue.ron`).
pub(crate) fn builtin() -> &'static Catalogue {
    BUILTIN.get_or_init(|| {
        parse_catalogue(include_str!("../../catalogue.ron"))
            .expect("built-in catalogue.ron must parse")
    })
}

impl Catalogue {
    /// Look up a base (built-in) distribution signature by name.
    pub(crate) fn base(&self, name: &str) -> Option<&Sig> {
        self.base.iter().find(|b| b.name == name).map(|b| &b.sig)
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
                Binding(name: "f", sig: Function(params: [Scalar(Real)], result: SameScalarKind(0))),
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
    ///   - arg0 = `Some(Complex)` (complex-in path, relevant for SameScalarKind)
    #[test]
    fn catalogue_functions_faithful_to_legacy_ops() {
        // (name, arg0_scalar) pairs to exercise.
        // For SameScalarKind fns the complex path matters; for fixed-output fns both
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
            // SameScalarKind(0): real→real
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
            // SameScalarKind(0): complex→complex
            ("exp", Some(ScalarType::Complex)),
            ("log", Some(ScalarType::Complex)),
            ("sqrt", Some(ScalarType::Complex)),
            ("conj", None),
            ("conj", Some(ScalarType::Complex)),
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
}
