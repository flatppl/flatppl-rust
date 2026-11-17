//! Statistical analysis of typed FlatPPL modules. Structural factor planning uses
//! the target's explicit input boundary. Query analysis and preparation take FlatPDL.

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
    GroupPartition, IidNormalPlan, PreparationError, PreparationOptions, PreparedLikelihood,
    ReductionDecision, ReductionOutcome, ReductionReport, plan_iid_normal, prepare_likelihood,
    prepare_likelihood_with_options, prepare_likelihood_with_report, sufficient_statistics,
    sufficient_statistics_with_report,
};
