use flatppl_core::Module;
use flatppl_model_analysis::{PreparationOptions, PreparedLikelihood, ReductionReport};

use crate::RefuseError;

/// Split a FlatPDL query into preparation and scoring without changing ordinary
/// emission. Explicit data inputs may be fixed or parametric in the source.
pub fn prepare_likelihood(
    m: &Module,
    data_inputs: &[String],
) -> Result<PreparedLikelihood, RefuseError> {
    prepare_likelihood_with_options(m, data_inputs, &PreparationOptions::default(), None)
}

/// Append committed decisions, including when preparation finds no reduction.
pub fn prepare_likelihood_with_report(
    m: &Module,
    data_inputs: &[String],
    report: &mut ReductionReport,
) -> Result<PreparedLikelihood, RefuseError> {
    prepare_likelihood_with_options(m, data_inputs, &PreparationOptions::default(), Some(report))
}

/// Prepare with an explicit layout and an optional reduction report.
pub fn prepare_likelihood_with_options(
    m: &Module,
    data_inputs: &[String],
    options: &PreparationOptions,
    report: Option<&mut ReductionReport>,
) -> Result<PreparedLikelihood, RefuseError> {
    flatppl_model_analysis::prepare_likelihood_with_options(m, data_inputs, options, report)
        .map_err(|e| RefuseError {
            node: e.node,
            construct: "prepare_likelihood".into(),
            reason: e.reason,
        })
}
