# Formula conversion

This crate converts a small brms formula subset into ordinary FlatPPL.
It performs no inference and has no R or Python dependency.

The supported subset uses Gaussian identity-link models, numeric, logical and
factor predictors, and random intercepts such as `y ~ x * treatment + (1 | g)`.
Formula terms support `+`, `-`, `:`, `*`, `/`, `%in%`, powers and parentheses.
R term expansion and brms fixed-term ordering determine the design columns.
Numeric and explicit factor grouping columns retain their training levels.
Grouping terms are ordered with C collation for stable brms parameter names.
Missing model rows are omitted. Unused columns do not affect row selection.
Offsets, expression calls, random slopes, other families and coefficient-specific
priors are not yet supported. The response cannot also be a fixed predictor.
Unsupported terms and options fail explicitly.

Unordered factors use treatment contrasts. Ordered factors use polynomial
contrasts. Set `options.contrasts` per column, for example
`{"treatment": "sum"}`. Accepted values are `treatment`, `sum`, `helmert` and
`polynomial`. Factor values carry explicit `levels`, zero-based `codes` and
an optional `ordered` flag. Training drops unused factor levels and preserves
the remaining order. Logical predictors use the levels `FALSE`, `TRUE`.
No-intercept formulas use R's cell coding, including full columns for `0 + f`.
Polynomial contrasts support at most 16 observed levels. Higher-degree R
contrasts become sensitive to QR rank decisions and rounding; use treatment,
sum or Helmert contrasts for larger factors.
Indicator-only terms such as `0 + o` do not need polynomial contrasts and have
no such level limit.

Defaults match the Gaussian brms rules: a centred Student-t intercept,
half-Student-t scales, and improper `regression.Flat()` coefficients.
Supply `Normal(mean, sd)` or `normal(mean, sd)` priors by class (`b`,
`Intercept`, `sd`, `sigma`). Group-specific `sd` entries override the class prior.
Scale priors are normalized restrictions to positive values. A flat prior cannot
be sampled. An improper prior does not imply an improper posterior.

```rust
let spec = flatppl_formula::read(spec_json)?;
let converted = flatppl_formula::build(&spec, &table)?;
let prediction = flatppl_formula::prepare_data(
    &converted.recipe, &new_table, flatppl_formula::DataMode::Prediction,
)?;
```

`Conversion` contains a module, named data constants, a serializable training
recipe, and parameter metadata. The recipe retains training means, factor
contrasts and levels, design terms, and group levels. Numeric source columns
remain raw. The emitted FlatPPL selects contrast values and computes interaction
products. Metadata records the default-prior rule and group collation. Generated
default priors carry short comments identifying the response summary rules.
Prediction requires no response column. It reuses the training contrast matrices
and remaps factor labels even when new data uses a different level order.
It rejects unseen factor and group levels.
Build a new recipe for a new fit or cross-validation fold.

The optional `arrow` feature adds `Table::from_arrow_ipc(bytes)` for uncompressed
IPC files and streams. It preserves float32, float64, integers, logical values,
dictionary levels and orderedness, and null validity. Plain string columns use
sorted levels. Dictionary indices are remapped across batches; ordered
dictionaries must retain a consistent level order. Missing values remain in the
table until model-frame construction, so nulls in unused columns do not drop rows.

`flatppl_host::Context::import_formula` returns a `FormulaModel` with a bound
training module and an unbound template. Its `prepare` method binds a fresh
snapshot in the same context. It never rebinds the training module.
`Context::import_formula_json` accepts specification text and optional Arrow
bytes; `FormulaModel::prepare_arrow` returns a fresh snapshot and retained input
row indices. Python and wasm use this same Rust preprocessing.

With the CLI `formula` feature:

```sh
flatppl convert model.formula.json model.flatppl
```

The JSON specification embeds a typed table. See
[the Gaussian fixture](tests/fixtures/gaussian.formula.json).
The command writes `model.data.json`, `model.recipe.json` and `model.model.json`
alongside the source. Consumers bind the named data constants explicitly.
Set `options.data_placement` to `inline` for a self-contained source snapshot.
The data document preserves float32 columns. StableHLO precision remains a
compiler option.

The factor and term fixtures come from base R, with the fixed formula
reconstruction used by brms 2.23.2. The Gaussian default-prior test values are
hand-derived from brms rules. [The R script](tests/fixtures/brms-oracle.R) checks
those defaults with brms 2.23.2.
[The factor oracle script](tests/fixtures/factor-matrix.R) regenerates the
design matrices, prediction matrices and brms coefficient names using base R.
[The Gaussian oracle](tests/fixtures/gaussian-oracle.stan) independently defines
the target and its normalized score. A successful print round trip alone does
not establish numerical correctness.
