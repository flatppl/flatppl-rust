# Formula conversion

This crate converts a small brms formula subset into ordinary FlatPPL.
It performs no inference and has no R or Python dependency.

The first slice supports Gaussian identity-link models, additive numeric
predictors, intercept controls, and random intercepts such as `y ~ x + (1 | g)`.
Numeric and explicit factor grouping columns retain their training levels.
Grouping terms are ordered with C collation for stable brms parameter names.
Missing model rows are omitted. Unused columns do not affect row selection.
Factor predictors, interactions, offsets, other families and coefficient-specific
priors are not yet supported. The response cannot also be a fixed predictor.
Unsupported terms and options fail explicitly.

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
recipe, and parameter metadata. The recipe retains training means and group
levels. Metadata records the default-prior rule and group collation. Generated
default priors carry short comments identifying the response summary rules.
Prediction requires no response column. It rejects unseen groups.
Build a new recipe for a new fit or cross-validation fold.

`flatppl_host::Context::import_formula` returns a `FormulaModel` with a bound
training module and an unbound template. Its `prepare` method binds a fresh
snapshot in the same context. It never rebinds the training module.

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

The design and default-prior test values are hand-derived from R/brms rules.
[The R script](tests/fixtures/brms-oracle.R) checks them with brms 2.23.2.
[The Gaussian oracle](tests/fixtures/gaussian-oracle.stan) independently defines
the target and its normalized score. A successful print round trip alone does
not establish numerical correctness.
