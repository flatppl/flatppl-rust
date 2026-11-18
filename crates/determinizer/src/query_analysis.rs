use flatppl_core::Module;
use flatppl_core::{NodeId, Symbol};
use flatppl_model_analysis as analysis;
use std::collections::HashSet;

/// Follow a statically selected tuple slot, record field or vector cell without
/// reading through a promoted input. Analysis and lowering use the same projections.
pub fn projected_child(m: &Module, node: NodeId, inputs: &HashSet<Symbol>) -> Option<NodeId> {
    crate::canon::project_tuple(m, node, inputs).or_else(|| crate::canon::project(m, node, inputs))
}

use analysis::{
    AnalysisError, CoordinatePrecision, IncrementalScorePlan, InputPath, ModelAnalysis,
    NormalCoordinateQueries, NormalCoordinatesQueries,
};

fn require_flatpdl(m: &Module) -> Result<(), AnalysisError> {
    crate::is_flatpdl(m)
        .map_err(|e| AnalysisError(format!("model analysis requires FlatPDL: {e:?}")))
}

/// Check FlatPDL conformance and analyze a query's declared parameter and data inputs.
pub fn analyze_model(m: &Module, data_inputs: &[String]) -> Result<ModelAnalysis, AnalysisError> {
    require_flatpdl(m)?;
    analysis::analyze_lowered_query(m, data_inputs)
}

/// Split a lowered scalar query without changing its original ABI or output.
/// Freeze file-backed constants before calling this function. Re-evaluate a term
/// after any named dependency changes, including in-place edits to array inputs.
pub fn prepare_incremental_score(
    m: &Module,
    data_inputs: &[String],
) -> Result<IncrementalScorePlan, AnalysisError> {
    require_flatpdl(m)?;
    analysis::prepare_incremental_score(m, data_inputs)
}

/// Build one conditional map, holding every other declared input fixed. No
/// joint composition or automatic sampler change occurs. Location and scale
/// must exclude the selected input. Multiple direct Normal factors are ambiguous.
/// Freeze fixed file data before calling this function.
pub fn prepare_normal_coordinate(
    m: &Module,
    data_inputs: &[String],
    parameter: &str,
    precision: CoordinatePrecision,
    logdensity: bool,
) -> Result<NormalCoordinateQueries, AnalysisError> {
    require_flatpdl(m)?;
    analysis::prepare_normal_coordinate(m, data_inputs, parameter, precision, logdensity)
}

/// Select one real record field while retaining the original record ABI.
/// The selected field contains z in score/forward/Jacobian inputs. Other fields
/// retain their supplied values. Forward and inverse return the whole record.
pub fn prepare_normal_coordinate_at_path(
    m: &Module,
    data_inputs: &[String],
    path: &InputPath,
    precision: CoordinatePrecision,
    logdensity: bool,
) -> Result<NormalCoordinateQueries, AnalysisError> {
    require_flatpdl(m)?;
    analysis::prepare_normal_coordinate_at_path(m, data_inputs, path, precision, logdensity)
}

/// Compose a selected acyclic set of scalar Normal maps. The selected induced
/// dependency graph must be triangular; unselected inputs stay fixed. Validity
/// includes both complete floating-point round trips, not just local maps.
pub fn prepare_normal_coordinates(
    m: &Module,
    data_inputs: &[String],
    paths: &[InputPath],
    precision: CoordinatePrecision,
    logdensity: bool,
) -> Result<NormalCoordinatesQueries, AnalysisError> {
    require_flatpdl(m)?;
    analysis::prepare_normal_coordinates(m, data_inputs, paths, precision, logdensity)
}
