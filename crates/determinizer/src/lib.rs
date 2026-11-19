//! FlatPPL → FlatPDL determiniser: a greedy directional legalizer that eliminates
//! the measure layer, leaving deterministic ops + the six `builtin_*` primitives.
//! Type-level transform — flatppl-rust does not evaluate densities.
mod canon;
mod chain;
mod conformance;
mod crossmodule;
mod density;
mod disintegrate;
mod driver;
mod extras;
mod free_draw;
mod invert;
mod jointchain;
mod kernel;
mod log_mass;
mod marginal;
mod preparation;
mod primitives;
mod query_analysis;
mod refuse;
mod sample;
mod stdfn;
pub use canon::{has_alias_chains, prepare_serialization, resolve_alias_chains};
pub use conformance::{is_flatpdl, is_flatpdl_with_options};
pub use disintegrate::{rewrite_restrict, split_disintegrate};
pub use driver::{
    SamplingObserver, desugar_restrict_and_disintegrate, determinize, determinize_with,
    determinize_with_options, determinize_with_options_and_report, determinize_with_roots,
    determinize_with_sampling_report,
};
pub use flatppl_model_analysis::{
    AnalysisError, ConjugateMatch, ConjugateRowId, CoordinatePrecision, GroupPartition,
    IncrementalScorePlan, IncrementalTerm, InputArray, InputCell, InputPath, ModelAnalysis,
    NormalCoordinate, NormalCoordinateQueries, NormalCoordinatesQueries, OutputTerm,
    PreparationOptions, PreparedLikelihood, ReductionDecision, ReductionOutcome, ReductionReport,
    sufficient_statistics, sufficient_statistics_with_report,
};

pub use flatppl_model_analysis::posterior::{conjugate_product, conjugate_update};
pub use marginal::{exponentially_modified_normal_logpdf, match_conjugate_marginal};
pub use preparation::{
    prepare_likelihood, prepare_likelihood_with_options, prepare_likelihood_with_report,
};
pub use query_analysis::{
    analyze_model, prepare_incremental_score, prepare_normal_coordinate,
    prepare_normal_coordinate_at_path, prepare_normal_coordinates, projected_child,
};
pub use refuse::{NonConformKind, NonConformance, RefuseError};
pub use stdfn::{expand_spline, standard_function};

/// Deterministic standard functions a backend lowers without portable expansion.
/// The default keeps the backend-independent FlatPDL form.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoweringOptions<'a> {
    /// Leave independent likelihood factors unreduced.
    pub disable_sufficient_statistics: bool,
    pub retain_standard_functions: &'a [(&'a str, &'a str)],
    /// Keep log-CDF subtraction opaque for fixed one-sided interval lanes.
    pub compact_log_cdf_difference: bool,
    /// Keep rand's (value, state) result even without a tuple projection.
    /// False preserves the legacy value-terminal convention.
    pub preserve_rand_tuple: bool,
    /// Admit compiler-only numerical integrals. Exact FlatPDL remains the default.
    pub numerical_integrals: bool,
    /// Admit the compiler-only ascending vector sort used by ordered sampling.
    pub vector_sort: bool,
    /// Admit compiler-only samplers for normalized restrictions and hurdle laws.
    pub restricted_sampling: bool,
    /// Target CDF support for scalar continuous constructors. None keeps the
    /// portable CDF rule. Missing CDFs use quadrature only with numerical_integrals.
    pub supports_cdf: Option<fn(&str) -> bool>,
}

impl LoweringOptions<'_> {
    pub(crate) fn retains(&self, module: &str, member: &str) -> bool {
        self.retain_standard_functions.contains(&(module, member))
    }
}
