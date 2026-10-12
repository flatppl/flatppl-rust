# FlatPPL Python bindings

This crate holds the PyO3 adapter for [`flatppl-host`](../host). Enable the `extension` feature and call `register_module` from a PyO3 module entry point.

The [Python package](https://github.com/flatppl/flatppl-python) owns wheel packaging, the Python API, and JAX execution. Its small `flatppl._native` entry point delegates to this crate. The shared compiler and query ABI stay in `flatppl-host` for other host languages.

The default build has no dependencies. Workspace `--all-features` checks require a Python interpreter. Maturin configures extension linking during wheel builds, so this crate does not enable PyO3's `extension-module` feature.

Native `Context.import_formula(spec, data_ipc=None, name=None, source_path=None)`
returns a frozen `FormulaModel`. Its `module` property is the training snapshot;
`recipe()` and `info()` return JSON metadata. `Context.prepare_formula(model,
data_ipc, mode="prediction")` returns `(module, rows_json)` for a new snapshot
in the same context. Arrow decoding, categories, missing-row selection and
prediction preparation stay in Rust. The Python package supplies the table
interchange transport and its public `FormulaView`.
