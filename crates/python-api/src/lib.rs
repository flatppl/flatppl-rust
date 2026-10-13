//! Python bindings shared by the `flatppl._native` extension entry point.
//!
//! Enable `extension` to register these bindings in a PyO3 module. Maturin
//! configures extension linking when building the Python wheel.

#![cfg(feature = "extension")]

use std::{path::Path, sync::Arc};

use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

use flatppl_host::flatppl_formula::DataMode;
use flatppl_host::{
    Auxiliary, BatchSpec, Constant, Context, Diagnostic, Dtype, EmitOptions, FormulaModel,
    IntegrationOptions, LoadedModule, MathFormat, Observations, PyhfOptions, Target,
};

pyo3::create_exception!(_native, Error, PyException);

type BatchArguments = (Vec<u64>, Vec<Vec<Option<usize>>>);

fn error(diagnostic: Diagnostic) -> PyErr {
    Error::new_err(
        serde_json::to_string(&diagnostic).expect("diagnostics contain only JSON values"),
    )
}

fn formula_mode(mode: &str) -> PyResult<DataMode> {
    match mode {
        "prediction" => Ok(DataMode::Prediction),
        "training" => Ok(DataMode::Training),
        _ => Err(PyValueError::new_err("mode must be prediction or training")),
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

    #[pyo3(signature = (model, data_ipc, mode="prediction"))]
    fn prepare_formula(
        &mut self,
        model: &NativeFormulaModel,
        data_ipc: &[u8],
        mode: &str,
    ) -> PyResult<(NativeModule, String)> {
        let mode = formula_mode(mode)?;
        let (inner, rows) = model
            .inner
            .prepare_arrow(&mut self.inner, data_ipc, mode)
            .map_err(error)?;
        Ok((
            NativeModule { inner },
            serde_json::to_string(&rows).expect("row indices are JSON integers"),
        ))
    }

    #[pyo3(signature = (model, path, mode="prediction"))]
    fn prepare_formula_path(
        &mut self,
        model: &NativeFormulaModel,
        path: &str,
        mode: &str,
    ) -> PyResult<(NativeModule, String)> {
        let (inner, rows) = model
            .inner
            .prepare_path(&mut self.inner, Path::new(path), formula_mode(mode)?)
            .map_err(error)?;
        Ok((
            NativeModule { inner },
            serde_json::to_string(&rows).expect("row indices are JSON integers"),
        ))
    }

    fn load(&mut self, path: &str) -> PyResult<NativeModule> {
        self.inner
            .load(path)
            .map(|inner| NativeModule { inner })
            .map_err(error)
    }

    fn register(&mut self, name: &str, module: &NativeModule) -> PyResult<()> {
        self.inner.register(name, &module.inner).map_err(error)
    }

    fn set(&mut self, module: &NativeModule, constants: &str) -> PyResult<NativeModule> {
        let constants: Vec<(String, Constant)> =
            serde_json::from_str(constants).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner
            .set(&module.inner, &constants)
            .map(|inner| NativeModule { inner })
            .map_err(error)
    }

    fn slice(&mut self, module: &NativeModule, outputs: Vec<String>) -> PyResult<NativeModule> {
        let outputs: Vec<&str> = outputs.iter().map(String::as_str).collect();
        self.inner
            .slice(&module.inner, &outputs)
            .map(|inner| NativeModule { inner })
            .map_err(error)
    }

    fn transform(
        &mut self,
        module: &NativeModule,
        logdensity: bool,
    ) -> PyResult<(NativeModule, NativeModule, NativeModule)> {
        let queries = self
            .inner
            .transform(&module.inner, logdensity)
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

    #[pyo3(signature = (dtype, autodiff, integration=None, batch=None, target="cpu"))]
    fn export(
        &self,
        py: Python<'_>,
        dtype: &str,
        autodiff: bool,
        integration: Option<(f64, f64, u32)>,
        batch: Option<BatchArguments>,
        target: &str,
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
        };
        let exported = py
            .detach(|| match batch {
                Some((shape, input_axes)) => self
                    .inner
                    .compile_batched(&options, &BatchSpec { shape, input_axes }),
                None => self.inner.compile(&options),
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
    m.add("Error", m.py().get_type::<Error>())?;
    Ok(())
}
