//! Read-only reports from the same passes used by compilation and preparation.

use flatppl_determinizer::{ModelAnalysis, ReductionReport};

use crate::{Context, Diagnostic, LoadedModule};

/// Structural facts and decisions from a selected compilation/preparation pipeline.
/// A preparation refusal is report data, not an invented empty preparation.
pub struct QueryAnalysis {
    pub model: ModelAnalysis,
    pub reductions: ReductionReport,
    pub summaries: Vec<String>,
    pub preparation_refusal: Option<String>,
}

impl Context {
    /// Inspect a query without changing its source, ABI or later compilation.
    /// `preparing` selects explicit data preparation instead of ordinary
    /// compilation. Fixed file data may be read to establish reduction proofs.
    pub fn analyze_query(
        &self,
        module: &LoadedModule,
        data_inputs: &[String],
        preparing: bool,
        sufficient_statistics: bool,
    ) -> Result<QueryAnalysis, Diagnostic> {
        let (mut lowered, model) = self.analyze_model(module, data_inputs)?;
        let mut reductions = ReductionReport::default();
        let mut summaries = Vec::new();
        let mut preparation_refusal = None;
        if !preparing {
            reductions = module.compilation_reductions(sufficient_statistics)?;
        } else {
            crate::load_fixed_data(&mut lowered, |source| module.read_data(source)).map_err(
                |error| Diagnostic::new("analyze_query", module.source_name(), error.to_string()),
            )?;
            if !sufficient_statistics {
                reductions.record_disabled(&lowered);
            } else {
                match flatppl_determinizer::prepare_likelihood_with_report(
                    &lowered,
                    data_inputs,
                    &mut reductions,
                ) {
                    Ok(prepared) => summaries = prepared.summaries,
                    Err(error) => preparation_refusal = Some(error.reason),
                }
            }
        }
        Ok(QueryAnalysis {
            model,
            reductions,
            summaries,
            preparation_refusal,
        })
    }
}
