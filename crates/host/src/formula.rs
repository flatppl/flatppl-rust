use crate::{Constant, Context, Diagnostic, LoadedModule};
use flatppl_fileaccess::Cache;
use flatppl_formula::{DataDocument, DataMode, DataValue, ModelInfo, Recipe, Spec, Table};
use std::{path::Path, sync::Arc};

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
        self.formula_table(spec, data, name, source_path, false)
    }

    /// Import a local data file, retaining prepared columns in the shared cache.
    /// Generated source refers to those immutable columns through `load_data`.
    pub fn import_formula_path_json(
        &mut self,
        source: &str,
        path: &Path,
        name: Option<&str>,
        source_path: Option<&str>,
    ) -> Result<FormulaModel, Diagnostic> {
        let spec = flatppl_formula::read(source)
            .map_err(|e| Diagnostic::new("formula", &path.to_string_lossy(), e))?;
        self.import_formula_path(&spec, path, name, source_path)
    }

    /// Import a typed formula specification and a local data file.
    pub fn import_formula_path(
        &mut self,
        spec: &Spec,
        path: &Path,
        name: Option<&str>,
        source_path: Option<&str>,
    ) -> Result<FormulaModel, Diagnostic> {
        self.formula_table(spec, &read_table(path)?, name, source_path, true)
    }

    fn formula_table(
        &mut self,
        spec: &Spec,
        data: &Table,
        name: Option<&str>,
        source_path: Option<&str>,
        file_backed: bool,
    ) -> Result<FormulaModel, Diagnostic> {
        let converted = flatppl_formula::build(spec, data).map_err(|error| {
            Diagnostic::new(
                "formula",
                source_path.or(name).unwrap_or("<formula>"),
                error,
            )
        })?;
        let source = flatppl_syntax::print(&converted.module);
        let template = self.parse(&source, if file_backed { None } else { name }, source_path)?;
        let module = if file_backed {
            self.formula_source(&template, &converted.data, name, source_path)?
        } else {
            self.set(&template, &constants(converted.data))?
        };
        Ok(FormulaModel {
            module,
            template,
            recipe: converted.recipe,
            info: converted.info,
        })
    }

    fn formula_source(
        &mut self,
        template: &LoadedModule,
        data: &DataDocument,
        name: Option<&str>,
        source_path: Option<&str>,
    ) -> Result<Arc<LoadedModule>, Diagnostic> {
        let fail = |error: String| Diagnostic::new("formula", template.source_name(), error);
        if !self.owns(template) {
            return Err(fail("the module belongs to a different context".into()));
        }
        let columns = data.columns_json().map_err(|e| fail(e.to_string()))?;
        let bytes = serde_json::to_vec(&columns).map_err(|e| fail(e.to_string()))?;
        let path = Cache::from_env()
            .store_prepared_json(&bytes)
            .map_err(|e| fail(e.to_string()))?;
        let path = path
            .to_str()
            .ok_or_else(|| fail("prepared data path must be UTF-8".into()))?;
        let module = data
            .with_data_source(template.parsed().clone(), path)
            .map_err(|e| fail(e.to_string()))?;
        self.parse(&flatppl_syntax::print(&module), name, source_path)
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

    /// Prepare a local table file using the training recipe and a durable sidecar.
    pub fn prepare_path(
        &self,
        context: &mut Context,
        path: &Path,
        mode: DataMode,
    ) -> Result<(Arc<LoadedModule>, Vec<usize>), Diagnostic> {
        let data = read_table(path)?;
        let mut prepared = flatppl_formula::prepare_data(&self.recipe, &data, mode)
            .map_err(|error| Diagnostic::new("formula", self.template.source_name(), error))?;
        let rows = std::mem::take(&mut prepared.rows);
        let module = context.formula_source(&self.template, &prepared, None, None)?;
        Ok((module, rows))
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

fn read_table(path: &Path) -> Result<Table, Diagnostic> {
    let fail = |error: String| Diagnostic::new("formula", &path.to_string_lossy(), error);
    if !std::fs::metadata(path)
        .map_err(|e| fail(e.to_string()))?
        .is_file()
    {
        return Err(fail("formula data must be a regular file".into()));
    }
    let bytes = std::fs::read(path).map_err(|e| fail(e.to_string()))?;
    let table =
        match path.extension().and_then(|s| s.to_str()) {
            Some("csv") => Table::from_csv(bytes.as_slice()),
            Some("json") => {
                Table::from_json(std::str::from_utf8(&bytes).map_err(|e| fail(e.to_string()))?)
            }
            Some("arrow" | "arrows" | "ipc" | "feather") => Table::from_arrow_ipc(&bytes),
            _ => return Err(fail(
                "formula data must be .csv, .json, or Arrow IPC (.arrow, .arrows, .ipc, .feather)"
                    .into(),
            )),
        };
    table.map_err(|e| fail(e.to_string()))
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
