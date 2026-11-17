//! Statistical analysis of typed FlatPPL modules. Structural factor planning uses
//! the target's explicit input boundary. Scalar conjugacy uses the query's observed
//! draws. Query analysis and preparation take FlatPDL.

mod ancestry;
mod conjugacy;
mod dependencies;
mod incremental;
mod normal_coordinates;
mod sufficient_statistics;

pub use ancestry::{StochasticAncestor, measure_stochastic_ancestors, sole_named_ancestor};
pub use conjugacy::{ConjugateMatch, ConjugateRowId, match_conjugate_marginal};
pub use dependencies::{
    AnalysisError, InputArray, InputCell, InputPath, ModelAnalysis, NormalCoordinate, OutputTerm,
    analyze_lowered_query, references_latent,
};

pub use incremental::{IncrementalScorePlan, IncrementalTerm, prepare_incremental_score};
pub use normal_coordinates::{
    CoordinatePrecision, NormalCoordinateQueries, NormalCoordinatesQueries,
    prepare_normal_coordinate, prepare_normal_coordinate_at_path, prepare_normal_coordinates,
};

pub use sufficient_statistics::{
    GroupPartition, IidNormalPlan, PreparationError, PreparationOptions, PreparedLikelihood,
    ReductionDecision, ReductionOutcome, ReductionReport, plan_iid_normal, prepare_likelihood,
    prepare_likelihood_with_options, prepare_likelihood_with_report, sufficient_statistics,
    sufficient_statistics_with_report,
};
