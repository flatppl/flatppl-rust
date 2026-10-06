# FlatPPL host API

`flatppl-host` provides source contexts and typed StableHLO query exports for native host languages. It has no Python, JAX, or execution dependency.

```rust
use flatppl_host::{Context, EmitOptions};

let mut context = Context::default();
let model = context.parse("distribution = Normal(0.0, 1.0)", None, None)?;
context.register("model.flatppl", &model)?;
let query = context.parse(
    r#"
    m = load_module("model.flatppl")
    x = elementof(reals)
    inputs = x
    outputs = logdensityof(m.distribution, x)
    "#,
    None,
    None,
)?;
let exported = query.compile(&EmitOptions::default())?;
```

`Context::load` loads a file or directory bundle. `Context::parse` accepts inline source, an optional logical name, and an optional source path for relative imports. `Context::register` gives a loaded module an explicit import name within that context. Names and source snapshots are append-only. Use a new context to replace definitions or reload files.

`LoadedModule` retains its resolved imports independently of the context's lifetime. Compilation requires explicit FlatPPL `inputs` and `outputs`. It returns StableHLO text, the entry point, input and output schemas, and authored output names. Records, tables, and tuples flatten into ordered tensor leaves. Each schema's `index` identifies a tensor argument or result.

`EmitOptions` defaults to float32 and Enzyme-compatible lowering. Set `enzyme_compatible` to false for unrestricted emission. The host owns value packing, execution, and differentiation. Compilation does not run Enzyme or a sampler.

The separate [`flatppl-python-api`](../python-api) crate maps this API into PyO3 classes for the [Python package](https://github.com/flatppl/flatppl-python).
