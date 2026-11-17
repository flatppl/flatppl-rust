//! Statistical analysis of typed FlatPDL queries. Callers establish FlatPDL
//! conformance before analyzing dependencies or building companion queries.

mod dependencies;
pub use dependencies::{
    AnalysisError, InputArray, InputCell, InputPath, ModelAnalysis, NormalCoordinate, OutputTerm,
    analyze_lowered_query,
};
mod incremental;
mod normal_coordinates;

pub use incremental::{IncrementalScorePlan, IncrementalTerm, prepare_incremental_score};
pub use normal_coordinates::{
    CoordinatePrecision, NormalCoordinateQueries, NormalCoordinatesQueries,
    prepare_normal_coordinate, prepare_normal_coordinate_at_path, prepare_normal_coordinates,
};
