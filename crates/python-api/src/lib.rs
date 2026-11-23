//! Python bindings shared by the `flatppl._native` extension entry point.
//!
//! Enable `extension` to register these bindings in a PyO3 module. Maturin
//! configures extension linking when building the Python wheel.

#![cfg(feature = "extension")]

use std::{path::Path, sync::Arc};

use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyInt};

use flatppl_host::flatppl_formula::DataMode;
use flatppl_host::{
    Auxiliary, BatchSpec, Constant, Context, Diagnostic, Dtype, EmitOptions, FormulaModel,
    FormulaTableSource, IntegrationOptions, LoadedModule, MathFormat, Observations,
    PreparedFormulaPrediction, PyhfOptions, Target,
};

pyo3::create_exception!(_native, Error, PyException);

type BatchArguments = (Vec<u64>, Vec<Vec<Option<usize>>>);

type NativeCell = (Vec<String>, Vec<u64>);
type NativeCellArguments<'py> = Vec<(Vec<String>, Vec<Bound<'py, PyAny>>)>;
type NativeTerm = (
    NativeModule,
    Vec<String>,
    String,
    Vec<Vec<String>>,
    Vec<NativeCell>,
);
type NativeNormalCoordinates = (
    NativeModule,
    NativeModule,
    NativeModule,
    NativeModule,
    Vec<String>,
    Vec<Vec<String>>,
);

fn input_path(mut path: Vec<String>) -> PyResult<flatppl_host::InputPath> {
    if path.is_empty() {
        return Err(PyValueError::new_err(
            "an input path requires a root input name",
        ));
    }
    Ok(flatppl_host::InputPath {
        input: path.remove(0),
        fields: path,
    })
}

fn path_parts(path: &flatppl_host::InputPath) -> Vec<String> {
    std::iter::once(path.input.clone())
        .chain(path.fields.iter().cloned())
        .collect()
}

fn cell_parts(cell: &flatppl_host::InputCell) -> NativeCell {
    (path_parts(&cell.path), cell.indices.clone())
}

fn input_cells(changed: NativeCellArguments<'_>) -> PyResult<Vec<flatppl_host::InputCell>> {
    changed
        .into_iter()
        .map(|(path, indices)| {
            let indices = indices
                .into_iter()
                .map(|index| {
                    if index.is_instance_of::<PyBool>() || !index.is_instance_of::<PyInt>() {
                        return Err(PyValueError::new_err(
                            "array cell indices must be nonnegative integers",
                        ));
                    }
                    index.extract::<u64>()
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(flatppl_host::InputCell {
                path: input_path(path)?,
                indices,
            })
        })
        .collect()
}

fn error(diagnostic: Diagnostic) -> PyErr {
    Error::new_err(
        serde_json::to_string(&diagnostic).expect("diagnostics contain only JSON values"),
    )
}

fn formula_mode(mode: &str, allow_new_levels: bool) -> PyResult<DataMode> {
    match mode {
        "prediction" => Ok(DataMode::Prediction),
        "training" if allow_new_levels => Err(PyValueError::new_err(
            "allow_new_levels is only supported in prediction mode",
        )),
        "training" => Ok(DataMode::Training),
        _ => Err(PyValueError::new_err("mode must be prediction or training")),
    }
}

fn formula_prediction(snapshot: PreparedFormulaPrediction) -> (NativeModule, String) {
    let metadata = serde_json::json!({
        "rows": snapshot.rows,
        "new_group_shapes": snapshot.new_group_shapes,
    });
    (
        NativeModule {
            inner: snapshot.module,
        },
        metadata.to_string(),
    )
}

fn formula_table_source<'a>(
    bytes: Option<&'a [u8]>,
    path: Option<&'a str>,
) -> PyResult<FormulaTableSource<'a>> {
    match (bytes, path) {
        (Some(bytes), None) => Ok(FormulaTableSource::Arrow(bytes)),
        (None, Some(path)) => Ok(FormulaTableSource::Path(Path::new(path))),
        _ => Err(PyValueError::new_err(
            "expected either Arrow IPC bytes or a table path",
        )),
    }
}

#[pyclass(name = "Context", module = "flatppl._native")]
struct NativeContext {
    inner: Context,
}

#[pymethods]
impl NativeContext {
    #[new]
    fn new() -> Self {
        Self {
            inner: Context::default(),
        }
    }

    #[pyo3(signature = (source, name=None, source_path=None))]
    fn parse(
        &mut self,
        source: &str,
        name: Option<&str>,
        source_path: Option<&str>,
    ) -> PyResult<NativeModule> {
        self.inner
            .parse(source, name, source_path)
            .map(|inner| NativeModule { inner })
            .map_err(error)
    }

    #[pyo3(signature = (
        source, name=None, source_path=None, measurement=None, observations="bound",
        auxiliary="bound",
    ))]
    fn import_pyhf(
        &mut self,
        source: &str,
        name: Option<&str>,
        source_path: Option<&str>,
        measurement: Option<String>,
        observations: &str,
        auxiliary: &str,
    ) -> PyResult<NativeModule> {
        let observations = match observations {
            "bound" => Observations::Bound,
            "external" => Observations::External,
            _ => {
                return Err(PyValueError::new_err(
                    "observations must be bound or external",
                ));
            }
        };
        let auxiliary = match auxiliary {
            "bound" => Auxiliary::Bound,
            "external" => Auxiliary::External,
            _ => return Err(PyValueError::new_err("auxiliary must be bound or external")),
        };
        let options = PyhfOptions {
            measurement,
            observations,
            auxiliary,
        };
        self.inner
            .import_pyhf(source, &options, name, source_path)
            .map(|inner| NativeModule { inner })
            .map_err(error)
    }

    #[pyo3(signature = (spec, data_ipc=None, name=None, source_path=None))]
    fn import_formula(
        &mut self,
        spec: &str,
        data_ipc: Option<&[u8]>,
        name: Option<&str>,
        source_path: Option<&str>,
    ) -> PyResult<NativeFormulaModel> {
        self.inner
            .import_formula_json(spec, data_ipc, name, source_path)
            .map(|inner| NativeFormulaModel { inner })
            .map_err(error)
    }

    #[pyo3(signature = (spec, path, name=None, source_path=None))]
    fn import_formula_path(
        &mut self,
        spec: &str,
        path: &str,
        name: Option<&str>,
        source_path: Option<&str>,
    ) -> PyResult<NativeFormulaModel> {
        self.inner
            .import_formula_path_json(spec, Path::new(path), name, source_path)
            .map(|inner| NativeFormulaModel { inner })
            .map_err(error)
    }

    #[pyo3(signature = (model, data_ipc, mode="prediction", *, allow_new_levels=false))]
    fn prepare_formula(
        &mut self,
        model: &NativeFormulaModel,
        data_ipc: &[u8],
        mode: &str,
        allow_new_levels: bool,
    ) -> PyResult<(NativeModule, String)> {
        let mode = formula_mode(mode, allow_new_levels)?;
        if allow_new_levels {
            return model
                .inner
                .prepare_arrow_with_new_groups(&mut self.inner, data_ipc)
                .map(formula_prediction)
                .map_err(error);
        }
        let (inner, rows) = model
            .inner
            .prepare_arrow(&mut self.inner, data_ipc, mode)
            .map_err(error)?;
        Ok((
            NativeModule { inner },
            serde_json::to_string(&rows).expect("row indices are JSON integers"),
        ))
    }

    #[pyo3(signature = (model, path, mode="prediction", *, allow_new_levels=false))]
    fn prepare_formula_path(
        &mut self,
        model: &NativeFormulaModel,
        path: &str,
        mode: &str,
        allow_new_levels: bool,
    ) -> PyResult<(NativeModule, String)> {
        let mode = formula_mode(mode, allow_new_levels)?;
        if allow_new_levels {
            return model
                .inner
                .prepare_path_with_new_groups(&mut self.inner, Path::new(path))
                .map(formula_prediction)
                .map_err(error);
        }
        let (inner, rows) = model
            .inner
            .prepare_path(&mut self.inner, Path::new(path), mode)
            .map_err(error)?;
        Ok((
            NativeModule { inner },
            serde_json::to_string(&rows).expect("row indices are JSON integers"),
        ))
    }

    #[pyo3(signature = (model, data_ipc, path, history_ipc, history_path, *, allow_new_levels=false))]
    fn prepare_formula_forecast(
        &mut self,
        model: &NativeFormulaModel,
        data_ipc: Option<&[u8]>,
        path: Option<&str>,
        history_ipc: Option<&[u8]>,
        history_path: Option<&str>,
        allow_new_levels: bool,
    ) -> PyResult<(NativeModule, String)> {
        let data = formula_table_source(data_ipc, path)?;
        let history = formula_table_source(history_ipc, history_path)?;
        model
            .inner
            .prepare_forecast(&mut self.inner, data, history, allow_new_levels)
            .map(formula_prediction)
            .map_err(error)
    }

    fn load(&mut self, path: &str) -> PyResult<NativeModule> {
        self.inner
            .load(path)
            .map(|inner| NativeModule { inner })
            .map_err(error)
    }

    fn load_bundle(&mut self, name: &str, data: &[u8]) -> PyResult<NativeModule> {
        self.inner
            .load_bundle(name, data)
            .map(|inner| NativeModule { inner })
            .map_err(error)
    }

    fn register(&mut self, name: &str, module: &NativeModule) -> PyResult<()> {
        self.inner.register(name, &module.inner).map_err(error)
    }

    fn contains(&self, name: &str) -> bool {
        self.inner.contains(name)
    }

    fn pack<'py>(&self, py: Python<'py>, module: &NativeModule) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.pack(&module.inner).map_err(error)?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn fingerprint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.fingerprint().map_err(error)?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn set(&mut self, module: &NativeModule, constants: &str) -> PyResult<NativeModule> {
        let constants: Vec<(String, Constant)> =
            serde_json::from_str(constants).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner
            .set(&module.inner, &constants)
            .map(|inner| NativeModule { inner })
            .map_err(error)
    }

    #[pyo3(signature = (module, data_inputs, *, pack_summaries=true))]
    fn prepare_likelihood(
        &mut self,
        module: &NativeModule,
        data_inputs: Vec<String>,
        pack_summaries: bool,
    ) -> PyResult<(NativeModule, NativeModule, Vec<String>)> {
        let prepared = self
            .inner
            .prepare_likelihood_with_options(
                &module.inner,
                &data_inputs,
                &flatppl_host::PreparationOptions { pack_summaries },
            )
            .map_err(error)?;
        Ok((
            NativeModule {
                inner: prepared.preparation,
            },
            NativeModule {
                inner: prepared.score,
            },
            prepared.summaries,
        ))
    }

    fn slice(&mut self, module: &NativeModule, outputs: Vec<String>) -> PyResult<NativeModule> {
        let outputs: Vec<&str> = outputs.iter().map(String::as_str).collect();
        self.inner
            .slice(&module.inner, &outputs)
            .map(|inner| NativeModule { inner })
            .map_err(error)
    }

    fn prepare_incremental_score(
        &mut self,
        module: &NativeModule,
        data_inputs: Vec<String>,
    ) -> PyResult<NativeIncrementalPlan> {
        self.inner
            .prepare_incremental_score(&module.inner, &data_inputs)
            .map(|inner| NativeIncrementalPlan { inner })
            .map_err(error)
    }

    fn analyze(
        &self,
        module: &NativeModule,
        data_inputs: Vec<String>,
        preparing: bool,
        sufficient_statistics: bool,
    ) -> PyResult<String> {
        use serde_json::json;
        let report = self
            .inner
            .analyze_query(
                &module.inner,
                &data_inputs,
                preparing,
                sufficient_statistics,
            )
            .map_err(error)?;
        let model = report.model;
        let paths =
            |items: &[flatppl_host::InputPath]| items.iter().map(path_parts).collect::<Vec<_>>();
        let terms = model
            .terms
            .iter()
            .map(|term| {
                json!({
                    "sign": term.sign, "multiplicity": term.multiplicity,
                    "parameters": term.parameters, "data": term.data,
                    "input_paths": paths(&term.input_paths),
                    "input_cells": term.input_cells.iter().map(cell_parts).collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>();
        let coordinates = model.normal_coordinates.iter().map(|coordinate| json!({
            "path": path_parts(&flatppl_host::InputPath { input: coordinate.parameter.clone(), fields: coordinate.fields.clone() }),
            "location_paths": paths(&coordinate.location_paths),
            "scale_paths": paths(&coordinate.scale_paths),
            "finite_location_proven": coordinate.finite_location_proven,
            "positive_finite_scale_proven": coordinate.positive_finite_scale_proven,
            "cyclic": coordinate.cyclic,
        })).collect::<Vec<_>>();
        let reductions = report.reductions.decisions.iter().map(|decision| {
            let mut value = json!({"binding": decision.binding, "family": decision.family,
                "rule": null, "reason": null, "partition": null});
            match &decision.outcome {
                flatppl_host::ReductionOutcome::Reduced { rule, partition } => {
                    value["outcome"] = json!("reduced");
                    value["rule"] = json!(rule);
                    value["partition"] = match partition {
                        Some(p) => json!({"observations": p.observations, "groups": p.groups, "singletons": p.singletons}),
                        None => json!(null),
                    };
                }
                flatppl_host::ReductionOutcome::Retained { reason } => {
                    value["outcome"] = json!("retained");
                    value["reason"] = json!(reason);
                }
                flatppl_host::ReductionOutcome::Disabled => value["outcome"] = json!("disabled"),
            }
            value
        }).collect::<Vec<_>>();
        Ok(json!({
            "parameters": model.parameters, "data": model.data,
            "components": model.components, "unused_parameters": model.unused_parameters,
            "input_paths": paths(&model.input_paths), "terms": terms,
            "input_arrays": model.input_arrays.iter().map(|array| json!({
                "path": path_parts(&array.path), "shape": array.shape,
            })).collect::<Vec<_>>(),
            "normal_coordinates": coordinates, "reductions": reductions,
            "posterior_updates": report.posterior_updates,
            "summaries": report.summaries, "preparation_refusal": report.preparation_refusal,
            "preparing": preparing, "sufficient_statistics": sufficient_statistics,
        })
        .to_string())
    }

    fn normal_coordinate(
        &mut self,
        module: &NativeModule,
        data_inputs: Vec<String>,
        parameter: Vec<String>,
        logdensity: bool,
        dtype: &str,
    ) -> PyResult<(
        NativeModule,
        NativeModule,
        NativeModule,
        NativeModule,
        String,
    )> {
        let dtype = match dtype {
            "float32" => Dtype::F32,
            "float64" => Dtype::F64,
            _ => return Err(PyValueError::new_err("dtype must be float32 or float64")),
        };
        let plan = self
            .inner
            .prepare_normal_coordinate_queries_at_path(
                &module.inner,
                &data_inputs,
                &input_path(parameter)?,
                logdensity,
                dtype,
            )
            .map_err(error)?;
        Ok((
            NativeModule { inner: plan.score },
            NativeModule {
                inner: plan.forward,
            },
            NativeModule {
                inner: plan.inverse,
            },
            NativeModule {
                inner: plan.log_jacobian,
            },
            plan.coordinate_input,
        ))
    }

    fn normal_coordinates(
        &mut self,
        module: &NativeModule,
        data_inputs: Vec<String>,
        parameters: Vec<Vec<String>>,
        logdensity: bool,
        dtype: &str,
    ) -> PyResult<NativeNormalCoordinates> {
        let dtype = match dtype {
            "float32" => Dtype::F32,
            "float64" => Dtype::F64,
            _ => return Err(PyValueError::new_err("dtype must be float32 or float64")),
        };
        let paths = parameters
            .into_iter()
            .map(input_path)
            .collect::<PyResult<Vec<_>>>()?;
        let plan = self
            .inner
            .prepare_normal_coordinates_queries(
                &module.inner,
                &data_inputs,
                &paths,
                logdensity,
                dtype,
            )
            .map_err(error)?;
        Ok((
            NativeModule { inner: plan.score },
            NativeModule {
                inner: plan.forward,
            },
            NativeModule {
                inner: plan.inverse,
            },
            NativeModule {
                inner: plan.log_jacobian,
            },
            plan.coordinate_inputs,
            plan.paths.iter().map(path_parts).collect(),
        ))
    }

    #[pyo3(signature = (module, logdensity, correlation_factors=None))]
    fn transform(
        &mut self,
        module: &NativeModule,
        logdensity: bool,
        correlation_factors: Option<Vec<String>>,
    ) -> PyResult<(NativeModule, NativeModule, NativeModule)> {
        let queries = self
            .inner
            .transform_with_correlation_factors(
                &module.inner,
                logdensity,
                correlation_factors.as_deref().unwrap_or_default(),
            )
            .map_err(error)?;
        Ok((
            NativeModule {
                inner: queries.unconstrained,
            },
            NativeModule {
                inner: queries.constrain,
            },
            NativeModule {
                inner: queries.unconstrain,
            },
        ))
    }
}

#[pyclass(name = "IncrementalPlan", module = "flatppl._native")]
struct NativeIncrementalPlan {
    inner: flatppl_host::IncrementalScoreQueries,
}

#[pymethods]
impl NativeIncrementalPlan {
    fn terms(&self) -> Vec<NativeTerm> {
        self.inner
            .terms()
            .iter()
            .map(|term| {
                (
                    NativeModule {
                        inner: term.query.clone(),
                    },
                    term.dependencies.clone(),
                    term.cache_input.clone(),
                    term.input_paths.iter().map(path_parts).collect(),
                    term.input_cells.iter().map(cell_parts).collect(),
                )
            })
            .collect()
    }
    fn combine(&self) -> NativeModule {
        NativeModule {
            inner: self.inner.combine().clone(),
        }
    }
    fn inputs(&self) -> Vec<String> {
        self.inner.inputs().to_vec()
    }
    fn affected_terms(&self, changed: Vec<String>) -> PyResult<Vec<usize>> {
        self.inner
            .affected_terms(&changed)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn affected_paths(&self, changed: Vec<Vec<String>>) -> PyResult<Vec<usize>> {
        let paths = changed
            .into_iter()
            .map(input_path)
            .collect::<PyResult<Vec<_>>>()?;
        self.inner
            .affected_paths(&paths)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn affected_cells(&self, changed: NativeCellArguments<'_>) -> PyResult<Vec<usize>> {
        self.inner
            .affected_cells(&input_cells(changed)?)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn revisions(&self) -> NativeScoreRevisions {
        NativeScoreRevisions {
            inner: self.inner.revisions(),
        }
    }
}

#[pyclass(name = "ScoreRevisions", module = "flatppl._native")]
struct NativeScoreRevisions {
    inner: flatppl_host::ScoreRevisions,
}

#[pyclass(name = "ScoreTicket", module = "flatppl._native")]
struct NativeScoreTicket {
    inner: Option<flatppl_host::ScoreTicket>,
}

#[pymethods]
impl NativeScoreRevisions {
    fn invalidate(&mut self, changed: Vec<String>) -> PyResult<Vec<usize>> {
        self.inner
            .invalidate(&changed)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn ticket(&self, term: usize) -> PyResult<NativeScoreTicket> {
        self.inner
            .ticket(term)
            .map(|ticket| NativeScoreTicket {
                inner: Some(ticket),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn invalidate_paths(&mut self, changed: Vec<Vec<String>>) -> PyResult<Vec<usize>> {
        let paths = changed
            .into_iter()
            .map(input_path)
            .collect::<PyResult<Vec<_>>>()?;
        self.inner
            .invalidate_paths(&paths)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn invalidate_cells(&mut self, changed: NativeCellArguments<'_>) -> PyResult<Vec<usize>> {
        self.inner
            .invalidate_cells(&input_cells(changed)?)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn accept(&mut self, ticket: &mut NativeScoreTicket) -> bool {
        ticket
            .inner
            .take()
            .is_some_and(|ticket| self.inner.accept(ticket))
    }
    fn is_current(&self, term: usize) -> PyResult<bool> {
        self.inner
            .is_current(term)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "FormulaModel", module = "flatppl._native", frozen)]
struct NativeFormulaModel {
    inner: FormulaModel,
}

#[pymethods]
impl NativeFormulaModel {
    #[getter]
    fn module(&self) -> NativeModule {
        NativeModule {
            inner: self.inner.module.clone(),
        }
    }

    fn recipe(&self) -> String {
        serde_json::to_string(&self.inner.recipe).expect("formula recipes contain only JSON values")
    }

    fn info(&self) -> String {
        serde_json::to_string(&self.inner.info).expect("formula metadata contains only JSON values")
    }
}

#[pyclass(name = "Module", module = "flatppl._native", frozen)]
struct NativeModule {
    inner: Arc<LoadedModule>,
}

#[pymethods]
impl NativeModule {
    fn fingerprint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.fingerprint().map_err(error)?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn pack<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.inner
            .pack()
            .map(|data| PyBytes::new(py, &data))
            .map_err(error)
    }

    #[getter]
    fn source(&self) -> &str {
        self.inner.source()
    }

    #[getter]
    fn source_name(&self) -> &str {
        self.inner.source_name()
    }

    fn bindings(&self) -> String {
        serde_json::to_string(&self.inner.bindings())
            .expect("binding metadata contains only JSON values")
    }

    fn constant(&self, name: &str) -> PyResult<String> {
        let value = self.inner.constant(name).map_err(error)?;
        serde_json::to_string(&value).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn math(&self, format: &str) -> PyResult<String> {
        let format = match format {
            "tex" => MathFormat::Tex,
            "mathml" => MathFormat::MathMl,
            "typst" => MathFormat::Typst,
            _ => return Err(PyValueError::new_err("format must be tex, mathml or typst")),
        };
        Ok(serde_json::to_string(&self.inner.math(format)).expect("rows contain only JSON values"))
    }

    /// A prepared export may carry a zero-input `preparation` program. Pass
    /// its checked tensor results after the declared arguments of the score.
    #[pyo3(signature = (dtype, autodiff, integration=None, batch=None, target="cpu", sufficient_statistics=true, prepare_fixed_data=false))]
    #[allow(clippy::too_many_arguments)]
    fn export(
        &self,
        py: Python<'_>,
        dtype: &str,
        autodiff: bool,
        integration: Option<(f64, f64, u32)>,
        batch: Option<BatchArguments>,
        target: &str,
        sufficient_statistics: bool,
        prepare_fixed_data: bool,
    ) -> PyResult<String> {
        let dtype = match dtype {
            "float32" => Dtype::F32,
            "float64" => Dtype::F64,
            _ => return Err(PyValueError::new_err("dtype must be float32 or float64")),
        };
        let target = match target {
            "cpu" => Target::Cpu,
            "gpu" => Target::Gpu,
            _ => return Err(PyValueError::new_err("target must be cpu or gpu")),
        };
        let options = EmitOptions {
            dtype,
            restrict_enzyme_compatible: autodiff,
            integration: integration.map(|(rtol, atol, max_intervals)| IntegrationOptions {
                rtol,
                atol,
                max_intervals,
            }),
            target,
            check_parameters: true,
            sufficient_statistics,
        };
        let batch = batch.map(|(shape, input_axes)| BatchSpec { shape, input_axes });
        let exported = py
            .detach(|| match (batch.as_ref(), prepare_fixed_data) {
                (Some(batch), true) => self.inner.compile_batched_prepared(&options, batch),
                (Some(batch), false) => self.inner.compile_batched(&options, batch),
                (None, true) => self.inner.compile_prepared(&options),
                (None, false) => self.inner.compile(&options),
            })
            .map_err(error)?;
        Ok(serde_json::to_string(&exported).expect("exports contain only JSON values"))
    }
}

/// Register the native classes and structured compiler exception.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeContext>()?;
    m.add_class::<NativeModule>()?;
    m.add_class::<NativeFormulaModel>()?;
    m.add_class::<NativeIncrementalPlan>()?;
    m.add_class::<NativeScoreRevisions>()?;
    m.add_class::<NativeScoreTicket>()?;
    m.add("Error", m.py().get_type::<Error>())?;
    Ok(())
}
