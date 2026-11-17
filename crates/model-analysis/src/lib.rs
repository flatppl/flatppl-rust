//! Statistical analysis of typed FlatPDL queries. Callers establish FlatPDL
//! conformance before analyzing dependencies or building companion queries.

mod dependencies;
mod incremental;
mod normal_coordinates;
mod sufficient_statistics;

pub use dependencies::{
    AnalysisError, InputArray, InputCell, InputPath, ModelAnalysis, NormalCoordinate, OutputTerm,
    analyze_lowered_query,
};

pub use incremental::{IncrementalScorePlan, IncrementalTerm, prepare_incremental_score};
pub use normal_coordinates::{
    CoordinatePrecision, NormalCoordinateQueries, NormalCoordinatesQueries,
    prepare_normal_coordinate, prepare_normal_coordinate_at_path, prepare_normal_coordinates,
};

pub use sufficient_statistics::{
    GroupPartition, PreparationError, PreparationOptions, PreparedLikelihood, ReductionDecision,
    ReductionOutcome, ReductionReport, prepare_likelihood, prepare_likelihood_with_options,
    prepare_likelihood_with_report, sufficient_statistics, sufficient_statistics_with_report,
};
