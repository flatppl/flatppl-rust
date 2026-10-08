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

`EmitOptions` defaults to float32 and restrictions for known Enzyme limitations. Set `restrict_enzyme_compatible` to false for unrestricted emission. The host owns value packing, execution, and differentiation. Compilation does not run Enzyme or a sampler, and neither setting guarantees or rules out Enzyme compatibility.

The separate [`flatppl-python-api`](../python-api) crate maps this API into PyO3 classes for the [Python package](https://github.com/flatppl/flatppl-python).

`Context::import_pyhf(json, name, source_path)` converts a pyhf model or workspace
using `flatppl-hs3` and returns an ordinary context-owned module. The host supplies
JSON text and handles any file IO. `LoadedModule::source()` exposes the generated
FlatPPL. Register the module and write explicit queries for its expected counts
or likelihood. Model-only documents expose observations as external inputs.
Conversion errors use the `pyhf` diagnostic stage.

## Batch independent calls

`LoadedModule::compile_batched` accepts a `BatchSpec` from the StableHLO crate.
Any host can use this API without Python or JAX:

```rust
use flatppl_host::BatchSpec;

let batch = BatchSpec {
    shape: vec![64],
    input_axes: vec![vec![Some(0)]],
};
let exported = query.compile_batched(&EmitOptions::default(), &batch)?;
```

This compiles 64 independent calls to the scalar density above. Each row of
`input_axes` describes one flattened tensor input, in schema index order.
Each entry identifies the physical axis for that batch dimension. `None` shares
the input across that dimension. Axes must be distinct and in range.

Multiple batch dimensions form a Cartesian frame. For a vector cell of length 3,
`shape: vec![2, 4]` and axes `[Some(1), Some(0)]` require an input of shape
`[4, 2, 3]`. Every output receives the leading batch prefix `[2, 4]`.
Tensor schemas describe these physical shapes. `batch_shape` records the frame.
Table row counts and `value_type` still describe each authored cell.
FlatPPL reductions and `lengthof` retain their cell meaning.

The emitter tensorizes supported pointwise operations, cell reductions, and
fixed-shape scans whose steps use supported operations. A batched scan retains
one time loop and updates all lanes together. Its time axis stays separate from
host batch axes, including shared inputs and record states.
Other operations use a StableHLO loop around the scalar program, preserving
control flow and explicit random states. All execution remains on the host's
selected device. Batch sizes are static and their product must fit an i32 index.
Zero-size batches produce empty outputs. An empty frame is the ordinary export.

Set `EmitOptions.integration` to `Some(IntegrationOptions::default())` to allow
adaptive numerical integration when exact scalar marginal or normalizer rules
do not apply. The default remains `None`. The emitted program evaluates the
integral on the execution device. Read the
[integration contract](../stablehlo/docs/integration.md) before enabling it.
