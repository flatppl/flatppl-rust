//! The `base` namespace — every name FlatPPL resolves without a module binding.
//!
//! Two consumers: spec-§04 name resolution ([`is_base_name`], called from the
//! trace's bare-atom arm and from the determiniser's FlatPDL conformance scan)
//! and the lint `shadows-builtin` rule.
//!
//! [`BUILTINS`] is generated from `flatppl-grammars/keyword-lists.json`; kept in
//! sync by `crates/lint/tests/builtins_sync.rs`. It is the highlighter's word
//! list, so it is neither a superset nor a subset of the `base` namespace, and
//! [`is_base_name`] corrects it in both directions:
//!
//! - **Added.** `catalogue.ron` carries three rows the keyword list omits (`in`,
//!   `length`, `log2`).
//! - **Removed.** The keyword list carries the 21 §09 standard-module members of
//!   `particle-physics` — 8 distribution constructors (`CrystalBall`, `Argus`,
//!   `Voigtian`, `Landau`, `DoubleSidedCrystalBall`, `RelativisticBreitWigner`,
//!   `BifurcatedNormal`, `ContinuedPoisson`) and 13 functions (`kallen`,
//!   `wignerd`, `blatt_weisskopf`, the `interp_*` family, …). Each has a row in
//!   `catalogues/particle-physics.ron` and is reachable ONLY behind its module
//!   alias, so a bare occurrence is unresolvable.
//!
//! What the keyword list uniquely and correctly carries: the §03 set names, the
//! constants, the measure ops and structural constructs `ops.rs` types by hand,
//! and the five §08 distributions with no catalogue row at all — `Dirac`,
//! `Lebesgue`, `Counting`, `PoissonProcess`, `BinnedPoissonProcess`.

/// Names FlatPPL treats as built-ins (functions, distributions, constants,
/// sets). A user binding with one of these names shadows the built-in.
///
/// The lint roster, synced to the grammars keyword list — **not** the `base`
/// namespace. Use [`is_base_name`] for name resolution.
pub const BUILTINS: &[&str] = &[
    "Argus",
    "Bernoulli",
    "Beta",
    "BifurcatedNormal",
    "BinnedPoissonProcess",
    "Binomial",
    "Categorical",
    "Categorical0",
    "Cauchy",
    "ChiSquared",
    "ContinuedPoisson",
    "Counting",
    "CrystalBall",
    "Dirac",
    "Dirichlet",
    "DoubleSidedCrystalBall",
    "Exponential",
    "Gamma",
    "GeneralizedNormal",
    "Geometric",
    "InverseGamma",
    "InverseWishart",
    "LKJ",
    "LKJCholesky",
    "Landau",
    "Laplace",
    "Lebesgue",
    "LogNormal",
    "Logistic",
    "Multinomial",
    "MvNormal",
    "NegativeBinomial",
    "NegativeBinomial2",
    "Normal",
    "Pareto",
    "Poisson",
    "PoissonProcess",
    "RelativisticBreitWigner",
    "StudentT",
    "Uniform",
    "Voigtian",
    "VonMises",
    "Weibull",
    "Wishart",
    "abs",
    "abs2",
    "acos",
    "acosh",
    "add",
    "addaxes",
    "adjoint",
    "aggregate",
    "all",
    "anything",
    "array",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bandedmat",
    "base",
    "bayesupdate",
    "bernstein",
    "bijection",
    "bincounts",
    "blatt_weisskopf",
    "blockdiagmat",
    "boolean",
    "booleans",
    "breakup_momentum",
    "broadcast",
    "broadcasted",
    "builtin_fromnormal",
    "builtin_fromuniform",
    "builtin_logdensityof",
    "builtin_sample",
    "builtin_tonormal",
    "builtin_touniform",
    "cartpow",
    "cartprod",
    "cat",
    "ceil",
    "checked",
    "cis",
    "col_gram",
    "colstack",
    "complex",
    "complexes",
    "conj",
    "conv",
    "cos",
    "cosh",
    "cross",
    "crosscorr",
    "cumprod",
    "cumsum",
    "densityof",
    "det",
    "diag",
    "diagmat",
    "disintegrate",
    "div",
    "divide",
    "draw",
    "elementof",
    "equal",
    "exp",
    "expm1",
    "external",
    "extlinspace",
    "eye",
    "false",
    "fchain",
    "fill",
    "filter",
    "fixed",
    "flatppl_compat",
    "floor",
    "fn",
    "functionof",
    "gamma",
    "ge",
    "get",
    "get0",
    "gt",
    "identity",
    "ifelse",
    "iid",
    "im",
    "imag",
    "indicesof",
    "indicesof0",
    "inf",
    "integer",
    "integers",
    "interp_poly2_lin",
    "interp_poly6_exp",
    "interp_poly6_lin",
    "interp_pwexp",
    "interp_pwlin",
    "interval",
    "inv",
    "invlogit",
    "invprobit",
    "isfinite",
    "isinf",
    "isnan",
    "iszero",
    "joinblocks",
    "joint",
    "joint_likelihood",
    "jointchain",
    "kallen",
    "kchain",
    "kernelof",
    "kscan",
    "l1norm",
    "l1unit",
    "l2norm",
    "l2unit",
    "land",
    "lawof",
    "le",
    "lengthof",
    "likelihoodof",
    "linsolve",
    "linspace",
    "lnot",
    "load_data",
    "load_module",
    "locscale",
    "log",
    "log10",
    "log1p",
    "logabsdet",
    "logdensityof",
    "loggamma",
    "logit",
    "logsoftmax",
    "logsumexp",
    "logweighted",
    "lor",
    "lower_cholesky",
    "lt",
    "lxor",
    "markovchain",
    "max",
    "maximum",
    "mean",
    "metricsum",
    "min",
    "minimum",
    "mod",
    "mul",
    "neg",
    "nonnegintegers",
    "nonnegreals",
    "normalize",
    "onehot",
    "ones",
    "only",
    "partition",
    "pi",
    "polynomial",
    "posintegers",
    "posreals",
    "pow",
    "probit",
    "prod",
    "pushfwd",
    "qr",
    "quadform",
    "rand",
    "real",
    "reals",
    "record",
    "reduce",
    "relabel",
    "resonance_breitwigner",
    "restrict",
    "reverse",
    "rnginit",
    "rngstate",
    "rngstates",
    "round",
    "row_gram",
    "rowstack",
    "scan",
    "selectbins",
    "self",
    "self_outer",
    "sin",
    "sinh",
    "sizeof",
    "softmax",
    "splitblocks",
    "sqrt",
    "standard_module",
    "std",
    "stdsimplex",
    "stepwise",
    "string",
    "sub",
    "sum",
    "superpose",
    "table",
    "tan",
    "tanh",
    "tile",
    "totalmass",
    "trace",
    "transpose",
    "true",
    "truncate",
    "tuple",
    "unequal",
    "unitinterval",
    "valueset",
    "var",
    "vector",
    "weighted",
    "wignerD",
    "wignerD_doublearg",
    "wignerd",
    "wignerd_doublearg",
    "zeros",
];

/// True iff `name` resolves in the `base` namespace — spec §04 "Name resolution"
/// step 2, "Otherwise, it resolves to the FlatPPL built-in of that name". A name
/// that is neither a current-module binding nor a base built-in is unresolvable,
/// and §04 makes that a static error.
///
/// Bare names only. A `§09` member behind its alias is a `RefNs::Module` ref
/// resolved against the module catalogue and never reaches here.
pub fn is_base_name(name: &str) -> bool {
    let cat = crate::catalogue::builtin();
    // A base row wins outright, so a name that is somehow BOTH a base row and a
    // §09 member keeps resolving. Nothing collides today
    // (`no_base_row_is_also_a_module_member` pins that), but the order makes the
    // §09 exclusion below unable to shadow a real builtin if one ever does.
    if cat.base(name).is_some() {
        return true;
    }
    // `BUILTINS` is strictly sorted (pinned by `builtins_are_sorted_and_unique`).
    BUILTINS.binary_search(&name).is_ok() && !cat.is_module_member(name)
}

/// True iff `name` is a distribution constructor — base (§08) or standard-module
/// (§09).
///
/// Such a name legitimately appears BARE in emitted FlatPDL as a kernel TAG: the
/// determiniser resolves `broadcast(hepphys.ContinuedPoisson, rates)` to
/// `broadcast(builtin_logdensityof, ContinuedPoisson, …)`, dropping the module
/// qualification because both engines' registries key the constructor bare (see
/// `determinizer/tests/broadcast_golden.rs`, the histfactory
/// `hepphys.ContinuedPoisson.(…)` shape). A tag is not a variable reference —
/// every evaluator resolves it — so it is not a free name.
///
/// This is NOT a resolution rule: [`is_base_name`] still rejects a bare §09
/// constructor in source, because §09 gives it no unqualified spelling. This
/// predicate is only for judging already-lowered FlatPDL.
pub fn is_kernel_tag_name(name: &str) -> bool {
    crate::catalogue::builtin()
        .distribution_param_names(name)
        .is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `is_base_name` reaches `BUILTINS` by `binary_search`. Pinned here rather
    /// than only in `crates/lint/tests/builtins_sync.rs`, whose sibling-repo
    /// guard is skipped in a git worktree (see that file).
    #[test]
    fn builtins_are_sorted_and_unique() {
        assert!(
            BUILTINS.windows(2).all(|w| w[0] < w[1]),
            "BUILTINS must be strictly sorted for binary_search"
        );
    }

    /// The §09 exclusion in `is_base_name` is subtractive, so it would silently
    /// remove a real builtin if a base row and a module member ever shared a
    /// name. They do not; the base-row-first order handles it if they ever do.
    #[test]
    fn no_base_row_is_also_a_module_member() {
        let cat = crate::catalogue::builtin();
        let both: Vec<&str> = cat
            .base_names()
            .filter(|n| cat.is_module_member(n))
            .collect();
        assert!(
            both.is_empty(),
            "a base row that is also a §09 member needs the ordering argument re-checked: {both:?}"
        );
    }

    /// Spec §09: a standard-module member is reachable only behind its alias, so
    /// a bare occurrence is not in the `base` namespace.
    #[test]
    fn bare_module_members_are_not_base_names() {
        for name in [
            "kallen",
            "wignerd",
            "blatt_weisskopf",
            "interp_pwlin",
            "CrystalBall",
            "Argus",
            "Voigtian",
            "Landau",
        ] {
            assert!(
                !is_base_name(name),
                "`{name}` is a §09 member and must not resolve bare"
            );
        }
    }

    /// The five §08 distributions with no catalogue row of any kind still
    /// resolve — they are base builtins the keyword list is the only record of.
    #[test]
    fn rowless_base_distributions_are_base_names() {
        for name in [
            "Dirac",
            "Lebesgue",
            "Counting",
            "PoissonProcess",
            "BinnedPoissonProcess",
        ] {
            assert!(is_base_name(name), "`{name}` is a §08 distribution");
        }
    }

    /// The three `catalogue.ron` rows the keyword list omits.
    #[test]
    fn catalogue_only_rows_are_base_names() {
        for name in ["in", "length", "log2"] {
            assert!(is_base_name(name), "`{name}` has a catalogue.ron row");
        }
    }
}
