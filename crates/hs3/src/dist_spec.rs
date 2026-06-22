//! Single source of truth for per-distribution-kind metadata.
//!
//! HS3 distribution `type` strings carry static, kind-keyed metadata that the
//! importer consults in several places:
//!   - the variate field name (which `extra` key holds the observed-variable
//!     name) and whether it is a scalar or an array of names,
//!   - whether the kind carries no variate of its own (composites whose variate
//!     comes from sub-distributions, or expression-based kinds that embed it),
//!   - whether the lowering needs the `hepphys` standard module in scope,
//!   - the `% …`-provenance doc line for non-1:1 lowerings.
//!
//! Previously this metadata was duplicated across `distribution.rs`
//! (`variate_field`, `has_no_own_variate`, `needs_hepphys`, `variate_name`) and
//! `convert.rs` (the `variate_fields` match, `dist_doc_line`). This module is
//! the one table both consume; the parameter-domain resolution
//! (kind + field → set name) also lives here so domain defaults are declared
//! once.

/// How a distribution kind names its variate (observed variable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variate {
    /// A single observed-variable name in the given `extra` field.
    Scalar(&'static str),
    /// An array of observed-variable names in the given `extra` field
    /// (`multivariate_normal_dist`'s `x`).
    MultiArray(&'static str),
    /// No variate of its own: composites (variate comes from sub-distributions),
    /// expression-based kinds (variate embedded in the formula), Poisson-process
    /// kinds (variate is the inner distribution / count space), and kinds that
    /// emit their own `relabel` (`barlow_beeston_lite_poisson_constraint_dist`).
    None,
}

impl Variate {
    /// The `extra` field key that holds the variate name(s), if this kind has a
    /// variate field. `None` for [`Variate::None`].
    pub fn field(&self) -> Option<&'static str> {
        match self {
            Variate::Scalar(f) | Variate::MultiArray(f) => Some(f),
            Variate::None => None,
        }
    }
}

/// Static metadata for one HS3 distribution `type`.
#[derive(Debug)]
pub struct DistSpec {
    /// The HS3 `type` string this entry describes.
    pub kind: &'static str,
    /// How the kind names its variate.
    pub variate: Variate,
    /// Whether the lowering needs the `hepphys` standard module in scope.
    pub needs_hepphys: bool,
    /// `% …`-provenance line for non-1:1 lowerings; `None` for 1:1 mappings.
    pub doc_line: Option<&'static str>,
    /// Every `extra` key the `emit_distribution` lowering reads for this kind,
    /// INCLUDING the variate field. The converter's free-parameter promotion
    /// uses this allowlist: a string-valued field NOT listed here is an
    /// unrecognized field and must be rejected rather than silently promoted to
    /// a free parameter. Empty for the default (untabulated) kind, where the
    /// only recognized fields are the variate `x` and whatever string fields the
    /// generic gaussian-style emitter reads — callers should treat an empty
    /// slice as "no allowlist available, fall back to permissive behavior".
    pub known_fields: &'static [&'static str],
}

/// The descriptor table. `known_fields` lists every `extra` key the
/// corresponding `emit_distribution` arm reads (variate field included). Kinds
/// absent here fall to [`DEFAULT`] (1:1 scalar-`x`, empty allowlist).
static SPECS: &[DistSpec] = &[
    // ---- 1:1 fundamental distributions ----
    DistSpec {
        kind: "gaussian_dist",
        variate: Variate::Scalar("x"),
        needs_hepphys: false,
        doc_line: None,
        known_fields: &["mean", "sigma", "x"],
    },
    DistSpec {
        kind: "normal_dist",
        variate: Variate::Scalar("x"),
        needs_hepphys: false,
        doc_line: None,
        known_fields: &["mean", "sigma", "x"],
    },
    DistSpec {
        kind: "poisson_dist",
        variate: Variate::Scalar("x"),
        needs_hepphys: false,
        doc_line: None,
        known_fields: &["mean", "x"],
    },
    DistSpec {
        kind: "exponential_dist",
        variate: Variate::Scalar("x"),
        needs_hepphys: false,
        doc_line: None,
        known_fields: &["c", "x"],
    },
    DistSpec {
        kind: "lognormal_dist",
        variate: Variate::Scalar("x"),
        needs_hepphys: false,
        doc_line: None,
        known_fields: &["mu", "sigma", "x"],
    },
    DistSpec {
        kind: "uniform_dist",
        variate: Variate::Scalar("x"),
        needs_hepphys: false,
        doc_line: None,
        // min/max come from doc.domains, not `extra`.
        known_fields: &["x", "min", "max"],
    },
    DistSpec {
        kind: "generalized_normal_dist",
        variate: Variate::Scalar("x"),
        needs_hepphys: false,
        doc_line: None,
        known_fields: &["mean", "alpha", "beta", "x"],
    },
    // ---- hepphys distributions ----
    DistSpec {
        kind: "crystalball_dist",
        variate: Variate::Scalar("m"),
        needs_hepphys: true,
        doc_line: None,
        known_fields: &[
            "m", "m0", "sigma", "alpha", "n", "sigma_L", "sigma_R", "alpha_L", "n_L", "alpha_R",
            "n_R",
        ],
    },
    DistSpec {
        kind: "argus_dist",
        variate: Variate::Scalar("mass"),
        needs_hepphys: true,
        doc_line: None,
        known_fields: &["mass", "resonance", "slope", "power"],
    },
    // ---- multivariate ----
    DistSpec {
        kind: "multivariate_normal_dist",
        variate: Variate::MultiArray("x"),
        needs_hepphys: false,
        doc_line: None,
        known_fields: &["mean", "covariances", "x"],
    },
    // ---- composite / expression / Poisson-process kinds ----
    DistSpec {
        kind: "barlow_beeston_lite_poisson_constraint_dist",
        // Emits its own relabel(..., [x names]); the convert.rs caller must NOT
        // wrap it again, so it carries no variate from the table's point of view.
        variate: Variate::None,
        needs_hepphys: false,
        doc_line: Some(
            "HS3 barlow_beeston_lite_poisson_constraint_dist → per-bin broadcast(Poisson, expected)",
        ),
        known_fields: &["x", "expected"],
    },
    DistSpec {
        kind: "product_dist",
        variate: Variate::None,
        needs_hepphys: false,
        doc_line: Some("HS3 product_dist → joint over factor distributions"),
        known_fields: &["factors"],
    },
    DistSpec {
        kind: "mixture_dist",
        variate: Variate::None,
        needs_hepphys: false,
        doc_line: Some("HS3 mixture_dist → normalize(superpose(weighted(coeff, summand)…))"),
        known_fields: &["summands", "coefficients", "extended"],
    },
    DistSpec {
        kind: "generic_dist",
        variate: Variate::None,
        needs_hepphys: false,
        doc_line: Some(
            "HS3 generic_dist → normalize(weighted(functionof(<expr>), Lebesgue(reals)))",
        ),
        known_fields: &["expression"],
    },
    DistSpec {
        kind: "density_function_dist",
        variate: Variate::None,
        needs_hepphys: false,
        doc_line: Some(
            "HS3 density_function_dist → normalize(weighted(<function>, Lebesgue(reals)))",
        ),
        known_fields: &["function"],
    },
    DistSpec {
        kind: "log_density_function_dist",
        variate: Variate::None,
        needs_hepphys: false,
        doc_line: Some(
            "HS3 log_density_function_dist → normalize(logweighted(<function>, Lebesgue(reals)))",
        ),
        known_fields: &["function"],
    },
    DistSpec {
        kind: "efficiency_product_pdf_dist",
        variate: Variate::None,
        needs_hepphys: false,
        doc_line: Some("HS3 efficiency_product_pdf_dist → weighted(<eff>, <pdf>)"),
        known_fields: &["eff", "pdf"],
    },
    DistSpec {
        kind: "rate_extended_dist",
        variate: Variate::None,
        needs_hepphys: false,
        doc_line: Some("HS3 rate_extended_dist → PoissonProcess(weighted(rate, shape))"),
        known_fields: &["rate", "distribution"],
    },
    DistSpec {
        kind: "rate_density_dist",
        variate: Variate::None,
        needs_hepphys: false,
        doc_line: Some(
            "HS3 rate_density_dist → PoissonProcess(weighted(<function>, Lebesgue(reals)))",
        ),
        known_fields: &["function"],
    },
    DistSpec {
        kind: "bincounts_extended_dist",
        variate: Variate::None,
        needs_hepphys: false,
        doc_line: Some(
            "HS3 bincounts_extended_dist → BinnedPoissonProcess(bins, weighted(rate, shape))",
        ),
        known_fields: &["rate", "distribution", "axes"],
    },
    DistSpec {
        kind: "bincounts_density_dist",
        variate: Variate::None,
        needs_hepphys: false,
        doc_line: Some(
            "HS3 bincounts_density_dist → BinnedPoissonProcess(bins, weighted(<function>, Lebesgue(reals)))",
        ),
        known_fields: &["function", "axes"],
    },
    DistSpec {
        kind: "polynomial_dist",
        // Variate is the `x` field (scalar), but the lowering also needs a doc line.
        variate: Variate::Scalar("x"),
        needs_hepphys: false,
        doc_line: Some(
            "HS3 polynomial_dist → normalize(weighted(functionof(polynomial(coefficients)), Lebesgue(reals)))",
        ),
        known_fields: &["coefficients", "x"],
    },
    DistSpec {
        kind: "chebychev_dist",
        // Variate is the `x` field (scalar observable); domain required for truncation.
        variate: Variate::Scalar("x"),
        needs_hepphys: false,
        doc_line: Some(
            "HS3 chebychev_dist → normalize(truncate(weighted(functionof(1+Σ a_k·T_k(t)), Lebesgue(reals)), interval))",
        ),
        known_fields: &["coefficients", "x"],
    },
    // Deliberately Unsupported: `emit_distribution` returns Err(Unsupported) for
    // relativistic_breit_wigner_dist (HS3's multi-channel parameterization has no
    // 1:1 FlatPPL map), so conversion aborts before any of this metadata is read.
    // The row exists only so the table mirrors the emit match — there is no
    // variate, no doc line, and an empty allowlist (no field promotion ever runs).
    DistSpec {
        kind: "relativistic_breit_wigner_dist",
        variate: Variate::None,
        needs_hepphys: false,
        doc_line: None,
        known_fields: &[],
    },
];

/// The default spec for kinds absent from [`SPECS`]: a 1:1 scalar-`x` mapping
/// with no hepphys requirement, no doc line, and no field allowlist (an empty
/// `known_fields` signals callers to fall back to permissive behavior).
static DEFAULT: DistSpec = DistSpec {
    kind: "",
    variate: Variate::Scalar("x"),
    needs_hepphys: false,
    doc_line: None,
    known_fields: &[],
};

/// Look up the explicit descriptor for `kind`, if one is tabulated.
///
/// Returns `None` for kinds that fall back to the default 1:1 scalar-`x`
/// behavior. Callers that always want a spec should use [`spec_or_default`].
pub(crate) fn spec(kind: &str) -> Option<&'static DistSpec> {
    SPECS.iter().find(|s| s.kind == kind)
}

/// Like [`spec`] but returns the shared default descriptor for untabulated
/// kinds, so callers never branch on `None`.
pub(crate) fn spec_or_default(kind: &str) -> &'static DistSpec {
    spec(kind).unwrap_or(&DEFAULT)
}

/// The variate descriptor for `kind`.
pub(crate) fn variate(kind: &str) -> Variate {
    spec_or_default(kind).variate
}

/// Whether `kind` carries no variate of its own.
///
/// Kept as a named companion to [`variate`] for table coherence; consumers may
/// also match on [`Variate::None`] directly (which is what `convert.rs` does),
/// so this is `#[allow(dead_code)]` outside tests.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn has_no_own_variate(kind: &str) -> bool {
    matches!(variate(kind), Variate::None)
}

/// The scalar variate field name for `kind` (the `extra` key whose value is the
/// observed-variable name). Falls back to `"x"`.
pub(crate) fn variate_field(kind: &str) -> &'static str {
    variate(kind).field().unwrap_or("x")
}

/// Whether `kind`'s lowering needs the `hepphys` standard module in scope.
pub(crate) fn needs_hepphys(kind: &str) -> bool {
    spec_or_default(kind).needs_hepphys
}

/// The `extra` keys the lowering for `kind` recognizes (variate field included).
/// Empty for untabulated kinds, which signals "no allowlist — be permissive".
pub(crate) fn known_fields(kind: &str) -> &'static [&'static str] {
    spec_or_default(kind).known_fields
}

/// Whether `field` is a recognized `extra` key for `kind`. Always `true` when
/// `kind` has no allowlist (empty `known_fields`), so untabulated kinds keep the
/// old permissive behavior rather than spuriously rejecting fields.
pub(crate) fn is_known_field(kind: &str, field: &str) -> bool {
    let fields = known_fields(kind);
    fields.is_empty() || fields.contains(&field)
}

/// The `% …`-provenance doc line for `kind`'s lowering, or `None` for 1:1
/// mappings that need no annotation.
pub(crate) fn doc_line(kind: &str) -> Option<&'static str> {
    spec_or_default(kind).doc_line
}

/// The natural-domain set name for a free parameter appearing in field `field`
/// of a distribution of the given `kind`. Returns a bare set constant name
/// (`"reals"`, `"posreals"`, …); HS3 `domains` declarations override this.
pub(crate) fn param_domain(kind: &str, field: &str) -> &'static str {
    match (kind, field) {
        // Scale-like params always positive
        (_, "sigma") | (_, "sigma_L") | (_, "sigma_R") => "posreals",
        (_, "n") | (_, "n_L") | (_, "n_R") => "posreals",
        (_, "beta") => "posreals",
        // ARGUS slope is typically negative; only power is strictly positive.
        ("argus_dist", "slope") => "reals",
        (_, "power") => "posreals",
        // alpha is a scale only for generalized_normal_dist; for crystalball it is a tail cut (reals)
        ("generalized_normal_dist", "alpha") => "posreals",
        // Poisson mean (rate) is positive.
        ("poisson_dist", "mean") => "posreals",
        // exponential_dist `c` is NOT positive: emit lowers it to rate = neg(c) and
        // FlatPPL Exponential requires rate > 0, so `c` must be NEGATIVE. There is
        // no `negreals` set constant, so `c` falls through to `reals` (the HS3
        // `domains` block is expected to pin the actual sign when present).
        // Poisson-process rate (expected count ≥ 0)
        (_, "rate") => "posreals",
        // Barlow-Beeston expected counts are ≥ 0
        ("barlow_beeston_lite_poisson_constraint_dist", "expected") => "posreals",
        // Chebyshev coefficients are unrestricted real numbers
        ("chebychev_dist", "coefficients") => "reals",
        _ => "reals",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_kind_is_scalar_x() {
        assert_eq!(variate_field("gaussian_dist"), "x");
        assert_eq!(variate("gaussian_dist"), Variate::Scalar("x"));
        assert!(!has_no_own_variate("gaussian_dist"));
        assert!(!needs_hepphys("gaussian_dist"));
        assert_eq!(doc_line("gaussian_dist"), None);
    }

    #[test]
    fn crystalball_uses_m_and_needs_hepphys() {
        assert_eq!(variate_field("crystalball_dist"), "m");
        assert!(needs_hepphys("crystalball_dist"));
    }

    #[test]
    fn argus_uses_mass_and_needs_hepphys() {
        assert_eq!(variate_field("argus_dist"), "mass");
        assert!(needs_hepphys("argus_dist"));
        assert_eq!(param_domain("argus_dist", "slope"), "reals");
    }

    #[test]
    fn mvnormal_is_multiarray_x() {
        assert_eq!(
            variate("multivariate_normal_dist"),
            Variate::MultiArray("x")
        );
        assert!(!has_no_own_variate("multivariate_normal_dist"));
    }

    #[test]
    fn composites_have_no_own_variate() {
        for kind in [
            "mixture_dist",
            "product_dist",
            "generic_dist",
            "density_function_dist",
            "log_density_function_dist",
            "rate_extended_dist",
            "rate_density_dist",
            "bincounts_extended_dist",
            "bincounts_density_dist",
        ] {
            assert!(
                has_no_own_variate(kind),
                "{kind} should have no own variate"
            );
        }
    }

    #[test]
    fn barlow_beeston_self_relabels_so_no_variate() {
        assert!(has_no_own_variate(
            "barlow_beeston_lite_poisson_constraint_dist"
        ));
    }

    #[test]
    fn polynomial_has_scalar_x_and_doc_line() {
        assert_eq!(variate_field("polynomial_dist"), "x");
        assert!(doc_line("polynomial_dist").is_some());
    }

    #[test]
    fn param_domain_defaults() {
        assert_eq!(param_domain("gaussian_dist", "mean"), "reals");
        assert_eq!(param_domain("gaussian_dist", "sigma"), "posreals");
        assert_eq!(param_domain("poisson_dist", "mean"), "posreals");
    }

    #[test]
    fn exponential_c_is_reals_not_posreals() {
        // emit lowers `c` to rate = neg(c); Exponential needs rate > 0 so `c` < 0.
        // No negreals set exists, so `c` must fall through to reals (NOT posreals,
        // which would force rate < 0).
        assert_eq!(param_domain("exponential_dist", "c"), "reals");
    }

    #[test]
    fn known_fields_allowlist() {
        assert!(is_known_field("gaussian_dist", "mean"));
        assert!(is_known_field("gaussian_dist", "sigma"));
        assert!(is_known_field("gaussian_dist", "x"));
        assert!(!is_known_field("gaussian_dist", "bogus"));
        // crystalball double-sided variant fields are recognized.
        assert!(is_known_field("crystalball_dist", "sigma_L"));
        assert!(is_known_field("crystalball_dist", "m"));
        assert!(!is_known_field("crystalball_dist", "junk"));
    }

    #[test]
    fn untabulated_kind_is_permissive() {
        // No allowlist → every field is "known" (permissive fallback).
        assert!(known_fields("no_such_dist").is_empty());
        assert!(is_known_field("no_such_dist", "anything"));
    }

    #[test]
    fn variate_field_is_in_known_fields() {
        // The variate field must itself be a recognized field for each tabulated kind.
        for s in SPECS {
            if let Some(f) = s.variate.field() {
                assert!(
                    s.known_fields.contains(&f),
                    "{}: variate field {f:?} missing from known_fields",
                    s.kind
                );
            }
        }
    }
}
