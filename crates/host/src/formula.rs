use crate::{Constant, Context, Diagnostic, LoadedModule};
use flatppl_formula::{DataMode, DataValue, ModelInfo, Recipe, Spec, Table};
use std::sync::Arc;

/// The training snapshot and the unbound template retained for new data.
pub struct FormulaModel {
    pub module: Arc<LoadedModule>,
    pub template: Arc<LoadedModule>,
    pub recipe: Recipe,
    pub info: ModelInfo,
}

impl Context {
    /// Import a Gaussian formula model, retaining its Rust preprocessing recipe.
    /// Data placement in the specification controls printed exports only.
    pub fn import_formula(
        &mut self,
        spec: &Spec,
        data: &Table,
        name: Option<&str>,
        source_path: Option<&str>,
    ) -> Result<FormulaModel, Diagnostic> {
        let mut spec = spec.clone();
        spec.options.data_placement = flatppl_formula::DataPlacement::External;
        let converted = flatppl_formula::build(&spec, data).map_err(|error| {
            Diagnostic::new(
                "formula",
                source_path.or(name).unwrap_or("<formula>"),
                error,
            )
        })?;
        let source = flatppl_syntax::print(&converted.module);
        let template = self.parse(&source, name, source_path)?;
        let module = self.set(&template, &constants(converted.data))?;
        Ok(FormulaModel {
            module,
            template,
            recipe: converted.recipe,
            info: converted.info,
        })
    }
}

impl FormulaModel {
    /// Bind a fresh snapshot in the same context. Prediction needs no response.
    pub fn prepare(
        &self,
        context: &mut Context,
        data: &Table,
        mode: DataMode,
    ) -> Result<Arc<LoadedModule>, Diagnostic> {
        let prepared = flatppl_formula::prepare_data(&self.recipe, data, mode)
            .map_err(|error| Diagnostic::new("formula", self.template.source_name(), error))?;
        context.set(&self.template, &constants(prepared))
    }
}

fn constants(data: flatppl_formula::DataDocument) -> Vec<(String, Constant)> {
    data.constants
        .into_iter()
        .map(|(name, value)| {
            let values = match value {
                DataValue::Integer(value) => return (name, Constant::Integer(value)),
                DataValue::Real32(values) => {
                    values.into_iter().map(Constant::Real32).collect::<Vec<_>>()
                }
                DataValue::Real64(values) => values.into_iter().map(Constant::Real).collect(),
                DataValue::Integers(values) => values.into_iter().map(Constant::Integer).collect(),
            };
            (
                name,
                Constant::Array {
                    shape: vec![values.len() as u32],
                    data: values,
                },
            )
        })
        .collect()
}
