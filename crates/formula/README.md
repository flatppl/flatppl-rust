# Formula conversion

This crate converts a small brms formula subset into ordinary FlatPPL.
It performs no inference and has no R or Python dependency.

The supported subset uses scalar response families, numeric, logical and factor
predictors, and group effects such as `y ~ x * treatment + (1 + x | g)`.
Formula terms support `+`, `-`, `:`, `*`, `/`, `%in%`, powers and parentheses.
R term expansion and brms fixed-term ordering determine the design columns.
Numeric and explicit factor grouping columns retain their training levels.
Grouping terms are ordered with C collation for stable brms parameter names.
Missing model rows are omitted. Unused columns do not affect row selection.
Numeric expressions and offsets are prepared in Rust and saved in the recipe.
Group terms support random slopes, nested groups and independent `||` effects.
The response cannot also be a fixed predictor.
Unsupported terms and options fail explicitly.

Set `family` to `gaussian`, `student`, `lognormal`, `bernoulli`, `binomial`, `poisson`,
`negbinomial`, `geometric`, `Gamma`, `exponential`, `weibull`, `beta`, `von_mises` or `cumulative`.
`link` is optional:

| Family | Default link | Other links |
| --- | --- | --- |
| Gaussian, Student | identity | log, inverse, softplus, squareplus, logit, probit, probit_approx, cloglog, cauchit, softit |
| Lognormal | identity | inverse |
| Bernoulli, binomial, beta | logit | probit, probit_approx, cloglog, cauchit, softit, identity, log |
| Poisson, negative binomial, geometric | log | identity, sqrt, softplus, squareplus |
| Gamma, exponential, Weibull | log | identity, inverse, softplus, squareplus |
| von Mises | tan_half | identity |
| Cumulative ordinal | logit | — |

Student models use direct `sigma` and `nu` parameters, with `nu >= 1`.
Gamma and negative-binomial models use direct `shape`. Their mean is `mu`;
the emitted FlatPPL rate is `shape / mu`. Exponential models use rate `1 / mu`.
Geometric models count failures before the first success and use
`NegativeBinomial(1, 1 / mu)`, so `mu` is the expected count.
Weibull models use a direct `shape` and mean `mu`, with scale
`mu / gamma(1 + 1 / shape)` and a Gamma(0.01, 0.01) shape prior.
Beta models use mean `mu` and direct precision `phi`, with a Gamma(0.01, 0.01)
prior and response values strictly between zero and one.
von Mises responses use radians in `[-pi, pi]`, a direct `kappa` with a
Gamma(2, 0.01) prior, and inverse tan-half link `2 * atan(eta)`.
The tan-half intercept prior is Student-t(1, 0, 1); the identity link uses the
ordinary response-based Student-t prior.
Binomial formulas require `y | trials(n) ~ ...`, where `n` is a nonnegative
integer or a numeric column of counts. Trials remain required for prediction.
Counts must be nonnegative integers, and binomial responses cannot exceed
their trials. Gamma, exponential, lognormal and Weibull responses must be strictly positive.
Bernoulli responses may be numeric, logical or factors; the recipe retains
the training event coding when preparing new data.

Cumulative responses use ordered factors or positive integer category codes.
Unused factor levels are dropped; integer codes retain gaps up to their maximum.
The recipe saves that category order for prediction. Ordered `cuts` replace the
ordinary intercept, so removing the intercept is not supported. `Intercept[1]`,
`Intercept[2]`, etc. map to the centred thresholds; derived `b_Intercept[i]` adds
the training predictor-mean contribution to report the original design scale.
Thresholds share a continuous prior, by default Student-t(3, 0, 2.5), conditioned
on their order with `tilt`. This normalized law adds `log(m!)` to brms's target
for `m` thresholds; `score_offset` records the difference. Class `Intercept`
sets their common prior. Coefficient selectors `"1"`, `"2"`, etc. are accepted
when they match after alias expansion and exact numeric-literal comparison.
Different parameterizations of the same law may still be refused, such as
`cauchy(0, 2)` versus native `Cauchy(0, 2)`. Unequal threshold
priors and other ordinal links or threshold parameterizations are not supported.

Response weights use `y | weights(w) ~ ...`, or
`y | trials(n) + weights(w) ~ ...` for binomial data. `w` names a numeric
column. After omitting incomplete model rows, retained weights must be finite
and nonnegative. Zero weights, including an all-zero vector, are allowed.
Weights are not rescaled. Weight expressions, constants and `scale = TRUE`
are outside this subset; unlike brms, infinite weights are refused under the
converter's finite numeric input policy.

The generated `K_model` and `L` keep the ordinary unweighted observation law.
`log_lik` returns unweighted pointwise log likelihoods for the retained training
rows, with the same parameter arguments as `K_model`. `weights_data` stores
the aligned weights. `Pi_post` adds `sum(weights_data .* log_lik(...))` once
to the prior log density. Prediction requires neither responses nor weights;
it uses the existing predictor-row filtering and saved training recipe.

Gaussian and Student formulas accept `y | se(s) ~ ...`, where `s` is a numeric
column, constant or supported column expression. The supplied values replace
residual `sigma`; its prior is unavailable. `se(s, sigma=TRUE)` retains `sigma`
and uses `sqrt(s^2 + sigma^2)`. Student models interpret this as scale, without a
degrees-of-freedom correction. Values must be finite and nonnegative. Conversion
rejects zero when `sigma=FALSE`, because the total likelihood scale would be zero;
with `sigma=TRUE`, zero known error is allowed. Prediction requires the columns used in `se()` and
reuses the saved expression. Separate response additions with `+`.

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

Defaults follow brms: a centred Student-t intercept, half-Student-t scales,
and improper `regression.Flat()` coefficients. Numeric response statistics use
the selected link for the intercept and group scales; direct `sigma` uses the
identity-link response scale. Lognormal uses log-response statistics for the
intercept, group scales and direct `sigma`, including with its inverse link.
The default Student degrees of freedom has a
Gamma(2, 0.1) prior normalized above 1. Gamma shape has Gamma(0.01, 0.01),
and negative-binomial shape has inverse-Gamma(0.4, 0.3).
Prior text is a FlatPPL measure expression. The brms aliases `normal`, `student_t`,
`cauchy`, `exponential`, `gamma`, `inv_gamma`, `beta`, `lognormal`, `logistic`,
`weibull`, `double_exponential`, `chi_square` and `uniform` accept their usual
positional arguments. `lkj`, `lkj_corr` and `lkj_corr_cholesky` require a group
correlation target. `Flat()` abbreviates `reg.Flat()` from `regression`.

Select priors by `class` (`b`, `Intercept`, `sd`, `cor`, `L`, `sigma`, `nu`, `shape`, `phi`, `kappa`) with
optional `coef`, `group` and `dpar`. Selectors must identify existing parameters. A coefficient
prior overrides its class prior; a group prior overrides the global group-scale
prior. A coefficient-specific `sd` prior also needs `group`.

Group scales resolve each coefficient's prior separately, including scalar
constants. `cor` and `L` are equivalent selectors for a correlated group's
Cholesky factor: `lkj(2)`, `lkj_corr(2)` and `lkj_corr_cholesky(2)` all select
`LKJCholesky(n, 2)` with the group's dimension. A group-specific correlation
prior overrides the class prior. Scalar bounds and matrix constants are not
supported for correlation priors.

```json
[
  {"class": "b", "prior": "normal(0, 2)", "lb": 0, "ub": 2},
  {"class": "b", "coef": "x", "prior": "normal(1, 1.5)"},
  {"class": "b", "coef": "z", "prior": "constant(0.5)"}
]
```

Coefficient overrides inherit the class row's bounds; bounds cannot be attached
to coefficient rows. Bounds can be numbers or scalar FlatPPL expressions using
the generated parameter names. Proper restrictions retain their normalization,
including parameter-dependent normalizers. Flat restrictions retain unit density.
An explicit open Flat support with a finite upper bound is currently refused:
the required `truncate(Flat(...), ...)` draw is not yet admitted by the engines.
Use `Flat(interval(...))` when a closed interval is intended.

`constant(c)` requires a finite numeric literal and creates a fixed coefficient
without a sampled coordinate. It requires fixed bounds when bounds are supplied;
the model must retain at least one sampled parameter.
Expressions use the ordinary FlatPPL type and phase validation. Shared coefficient
priors use `iid`; differing priors use scalar declarations and a combined vector.
A Flat prior cannot be sampled. An improper prior does not imply an improper
posterior.

```rust
let spec = flatppl_formula::read(spec_json)?;
let converted = flatppl_formula::build(&spec, &table)?;
let prediction = flatppl_formula::prepare_data(
    &converted.recipe, &new_table, flatppl_formula::DataMode::Prediction,
)?;
```

`Conversion` contains a module, named data constants, a serializable training
recipe, and parameter metadata. Generated source keeps observations and prepared
columns in external bindings; the data document supplies their values separately.
For a file export, `conversion.with_data_source("model.data.json")` returns a
module using `load_data` and its prepared numeric JSON sidecar. The caller writes
both files. The sidecar path resolves from the generated module; it references
prepared rows and encoded columns, not the original raw table. Data columns are
separate; shape and fitted model constants remain in source. The original
conversion remains external for dataframe binding and prediction. JSON has no
dtype; float32 columns retain their exact
stored values when written as JSON numbers.
The recipe retains family and link, response
coding, training means, factor contrasts and levels, design terms, and group
levels. Numeric source columns remain raw. The emitted FlatPPL selects contrast values and computes interaction
products. Metadata records the default-prior rule and group collation.
Prediction requires no response column. It reuses the training contrast matrices
and remaps factor labels even when new data uses a different level order.
It rejects unseen factor and group levels.
Build a new recipe for a new fit or cross-validation fold.

`Table::from_json(text)` accepts the typed table format, an array of row objects,
or an object of column arrays. Missing record fields and `null` cells remain
missing until model-frame construction. String columns become unordered factors
with sorted levels. Boolean, numeric and string values cannot be mixed in a
column. Integer columns preserve signed 64-bit values. Mixing integers with
decimal or exponent numbers produces float64 only when each integer remains
exactly representable; overflow and lossy promotion are errors. Typed tables
retain explicit float32, factor levels and orderedness. All-null columns remain
nullable numeric columns, so unused missing columns do not remove model rows.

`Table::from_csv(reader)` accepts a headered CSV. Empty fields and the exact token
`NA` are null, including quoted fields. Other whitespace and text are preserved.
Columns infer `true`/`false` or `TRUE`/`FALSE` booleans, signed 64-bit integers,
finite float64 numbers, then unordered string factors with sorted levels.
Numeric columns reject integer overflow and lossy integer-to-float promotion.
CSV quoting supports commas, quotes and newlines in fields. Use typed JSON or
Arrow when inference or these null tokens are unsuitable for a column.

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
flatppl convert model.formula.json model.flatppl --data observations.csv
flatppl convert model.formula.json model.flatppl --data observations.json
```

Embed a typed `data` table in the specification, as in
[the Gaussian fixture](tests/fixtures/gaussian.formula.json), or provide `--data`;
using both is an error. External JSON accepts all three table formats above.
Paths are local regular files and relative paths resolve from the current working
directory. Build the CLI with `formula-arrow` to read Arrow IPC files or streams
with `.arrow`, `.arrows`, `.ipc` or `.feather` extensions through `--data`.
The command writes `model.data.json`, `model.recipe.json` and `model.model.json`
alongside the source. Its `load_data` declaration reads the prepared numeric
sidecar relative to `model.flatppl`; keep these files together. Row selection is
recorded in model metadata. Training data is never embedded in generated source.
For migration, the old JSON `data_placement` option is temporarily accepted and
ignored; the CLI flag has been removed. StableHLO precision remains a compiler
option.

The factor and term fixtures come from base R, with the fixed formula
reconstruction used by brms 2.23.2. The Gaussian default-prior test values are
hand-derived from brms rules. [The R script](tests/fixtures/brms-oracle.R) checks
those defaults with brms 2.23.2.
[The family prior fixture](tests/fixtures/family-priors.json) records executed
brms defaults for different links and response types; its adjacent R script
regenerates the source values.
[The factor oracle script](tests/fixtures/factor-matrix.R) regenerates the
design matrices, prediction matrices and brms coefficient names using base R.
[The Gaussian oracle](tests/fixtures/gaussian-oracle.stan) independently defines
the target and its normalized score. A successful print round trip alone does
not establish numerical correctness.
[The ordinal oracle](tests/fixtures/cumulative-oracle.json) freezes actual
brms-generated Stan scores and gradients, including the ordering offset and a
float32 tail case. Its [R script](tests/fixtures/cumulative-oracle.R) reproduces
the category coding, generated Stan, and independent density values.

[The standard-error fixture](tests/fixtures/standard-errors.json) records actual
normalized brms-generated Stan scores and gradients for Gaussian and Student
models with both `se()` residual-scale settings; its adjacent R script regenerates
the models.
