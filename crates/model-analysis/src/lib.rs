//! Statistical analysis of FlatPPL query graphs.

mod dependencies;
pub use dependencies::{
    AnalysisError, InputArray, InputCell, InputPath, ModelAnalysis, NormalCoordinate, OutputTerm,
    analyze_lowered_query,
};
