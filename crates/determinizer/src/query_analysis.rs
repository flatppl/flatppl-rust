use flatppl_core::Module;
use flatppl_model_analysis::{AnalysisError, ModelAnalysis, analyze_lowered_query};

/// Check FlatPDL conformance and analyze a query's declared parameter and data inputs.
pub fn analyze_model(m: &Module, data_inputs: &[String]) -> Result<ModelAnalysis, AnalysisError> {
    crate::is_flatpdl(m)
        .map_err(|e| AnalysisError(format!("model analysis requires FlatPDL: {e:?}")))?;
    analyze_lowered_query(m, data_inputs)
}
