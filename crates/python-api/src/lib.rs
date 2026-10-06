//! Python bindings shared by the `flatppl._native` extension entry point.
//!
//! Enable `extension` to register these bindings in a PyO3 module. Maturin
//! configures extension linking when building the Python wheel.

#![cfg(feature = "extension")]

use std::sync::Arc;

use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

use flatppl_host::{
    Constant, Context, Diagnostic, Dtype, EmitOptions, IntegrationOptions, LoadedModule,
};

pyo3::create_exception!(_native, Error, PyException);

fn error(diagnostic: Diagnostic) -> PyErr {
    Error::new_err(
        serde_json::to_string(&diagnostic).expect("diagnostics contain only JSON values"),
    )
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

    #[pyo3(signature = (dtype, autodiff, integration=None))]
    fn export(
        &self,
        py: Python<'_>,
        dtype: &str,
        autodiff: bool,
        integration: Option<(f64, f64, u32)>,
    ) -> PyResult<String> {
        let dtype = match dtype {
            "float32" => Dtype::F32,
            "float64" => Dtype::F64,
            _ => return Err(PyValueError::new_err("dtype must be float32 or float64")),
        };
        let options = EmitOptions {
            dtype,
            restrict_enzyme_compatible: autodiff,
            integration: integration.map(|(rtol, atol, max_intervals)| IntegrationOptions {
                rtol,
                atol,
                max_intervals,
            }),
        };
        let exported = py.detach(|| self.inner.compile(&options)).map_err(error)?;
        Ok(serde_json::to_string(&exported).expect("exports contain only JSON values"))
    }
}

/// Register the native classes and structured compiler exception.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeContext>()?;
    m.add_class::<NativeModule>()?;
    m.add("Error", m.py().get_type::<Error>())?;
    Ok(())
}
