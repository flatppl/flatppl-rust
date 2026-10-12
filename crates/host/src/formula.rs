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
    /// Import a JSON specification with an optional Arrow IPC table override.
    /// With no IPC bytes, the specification must contain its typed data table.
    pub fn import_formula_json(
        &mut self,
        source: &str,
        data_ipc: Option<&[u8]>,
        name: Option<&str>,
        source_path: Option<&str>,
    ) -> Result<FormulaModel, Diagnostic> {
        let fail = |error: String| {
            Diagnostic::new(
                "formula",
                source_path.or(name).unwrap_or("<formula>"),
                error,
            )
        };
        let spec = flatppl_formula::read(source).map_err(|e| fail(e.to_string()))?;
        let data = data_ipc
            .map(Table::from_arrow_ipc)
            .transpose()
            .map_err(|e| fail(e.to_string()))?;
        let table = data.as_ref().or(spec.data.as_ref()).ok_or_else(|| {
            fail("formula import needs data in the specification or Arrow IPC bytes".into())
        })?;
        self.import_formula(&spec, table, name, source_path)
    }

    /// Import a formula model, retaining its Rust preprocessing recipe.
    /// The source template keeps data external; the loaded snapshot binds it.
    pub fn import_formula(
        &mut self,
        spec: &Spec,
        data: &Table,
        name: Option<&str>,
        source_path: Option<&str>,
    ) -> Result<FormulaModel, Diagnostic> {
        let converted = flatppl_formula::build(spec, data).map_err(|error| {
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
        self.prepare_with_rows(context, data, mode)
            .map(|(module, _)| module)
    }

    /// Prepare Arrow IPC data and return the retained input row indices.
    pub fn prepare_arrow(
        &self,
        context: &mut Context,
        data_ipc: &[u8],
        mode: DataMode,
    ) -> Result<(Arc<LoadedModule>, Vec<usize>), Diagnostic> {
        let data = Table::from_arrow_ipc(data_ipc)
            .map_err(|error| Diagnostic::new("formula", self.template.source_name(), error))?;
        self.prepare_with_rows(context, &data, mode)
    }

    /// Bind a fresh snapshot and retain the model-frame row mapping.
    pub fn prepare_with_rows(
        &self,
        context: &mut Context,
        data: &Table,
        mode: DataMode,
    ) -> Result<(Arc<LoadedModule>, Vec<usize>), Diagnostic> {
        let mut prepared = flatppl_formula::prepare_data(&self.recipe, data, mode)
            .map_err(|error| Diagnostic::new("formula", self.template.source_name(), error))?;
        let rows = std::mem::take(&mut prepared.rows);
        let module = context.set(&self.template, &constants(prepared))?;
        Ok((module, rows))
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
