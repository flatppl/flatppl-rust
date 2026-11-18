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

Additive `s(x)` and `s(x, k=6)` terms use the one-dimensional mgcv/brms
thin-plate basis. `x` must name a numeric column. The default rank is 10;
the requested rank must be at least 3 and no greater than the number of
distinct retained training values. This first implementation supports at most
2,000 distinct values; mgcv's larger-data knot subsampling is not implemented.
Other bases for `s`, custom knots and smooth interactions need separate
support.

Each smooth is centred over the retained training rows. Its unpenalized slope
uses class `b` (coefficient `sx_1` for `s(x)`); its scale uses class `sds`,
optionally selected by the canonical term, such as `"s(x, k = 6)"`.
The defaults are an improper Lebesgue slope prior and half-Student-t scale. Penalized coefficients
are independent standard Normals. The fitted recipe stores the basis and
penalty transform, so prediction needs only predictor columns and never refits
or recentres the smooth. This also applies within distributional formulas.
Numerical setup runs in Rust; emitted models use ordinary matrix operations.

`s(x, k=6, by=f)` fits one curve per retained level of the factor `f`, with
its own slope and scale. An ordered factor with multiple retained levels
omits its first level. Add `s(x)` to give the first level a reference curve
and model departures at other levels.
Add `f` as a main effect for separate intercepts. This main effect uses its
ordinary contrasts. Smooths select curves by factor level.

For level `A`, select the slope prior with class `b` and coefficient
`sx:fA_1`. Select the scale prior with class `sds` and coefficient
`s(x, k = 6, by = f):fA`. The basis and original level order persist in the
recipe. New data needs `x` and `f` with levels from the training data.

`t2(x, z)` makes a surface from two numeric columns using natural cubic
regression spline margins. Each margin defaults to rank 5. Set `k=4` for
equal ranks or `k=c(4, 6)` for separate ranks. The product of the ranks
can be at most 4,096. The surface has three centred unpenalized columns
and disjoint rough-by-rough, affine-by-rough and rough-by-affine blocks.
Each penalized block has its own scale. A term-specific `sds` prior
such as `coef = "t2(x, z)"` applies separately to all three scales.
The unpenalized columns use class `b` and coefficients `t2xz_1` through
`t2xz_3`.

The knots, spline spaces, and block penalties follow mgcv's cubic `t2`
construction. The affine margins use a constant and a centred, unit-RMS
linear column. This fixes the basis in the two-dimensional penalty
nullspace. Coefficient scales can differ from mgcv's eigensolver-dependent
basis. Prediction reuses the training basis and extends each margin
linearly past its boundary knots.

Distributional formulas use the same terms for a family parameter:

```json
{
  "formula": "y ~ x",
  "family": "gaussian",
  "dpars": {"sigma": {"formula": "sigma ~ g", "link": "log"}},
  "priors": [{"class": "b", "dpar": "sigma", "prior": "normal(0, 1)"}]
}
```

This is the typed equivalent of `bf(y ~ x, sigma ~ g)`; `bf(...)` strings are
not yet accepted. Each predictor saves its own contrasts, centring, offsets
and group effects. Training omits incomplete rows jointly across the response
and every predictor. Prediction reuses all fitted recipes without a response.
For scalar families, priors without `dpar` apply to the mean predictor. Set `dpar` for auxiliary
coefficients, intercepts, group scales or correlations. A direct `sigma` prior
is unavailable when `sigma` has a predictor; its priors belong to the linear
predictor before the inverse link.

The same interface supports `sigma`, `nu`, `shape`, `phi`, `kappa`, `zi` and `hu`
where present in the family. Positive parameters default to the log link;
Student `nu` uses `logm1`, whose inverse is `1 + exp(eta)`, and `zi`/`hu` use logit.
Identity is also available; positive parameters other than `nu` additionally
accept softplus and squareplus. Auxiliary priors follow brms's link-dependent
defaults. `ModelInfo.dpars` records each formula, link and design columns.
For example, `temp_Intercept_sigma` is the centred scale intercept and
`b_Intercept_sigma` reports `b_sigma_Intercept` on the original design scale.

### Nonlinear predictors

Supply `nlpars` to treat the right side as arithmetic over named parameter
predictors and numeric data columns. Each named predictor uses ordinary
formula terms. All listed families with a scalar location or mean
predictor support nonlinear expressions. Cumulative models use the
expression on the logit scale with the usual ordered thresholds.
Categorical models use separate category predictors.

```json
{
  "formula": "y ~ a * exp(b * x)",
  "nlpars": {"a": "a ~ 1", "b": "b ~ 1"},
  "priors": [
    {"class": "b", "nlpar": "a", "prior": "normal(1, 2)"},
    {"class": "b", "nlpar": "b", "prior": "normal(0, 2)"}
  ]
}
```

The whole expression supplies the predictor on the link scale. Use `y ~ exp(ld) * (theta - difficulty)`
with the default Bernoulli logit link for a 2PL model. Arithmetic supports
`+`, `-`, `*`, `/`, `^`, parentheses, `exp` and `log`. Each population
coefficient needs an explicit `b` prior scoped by `nlpar`, including the
uncentred intercept (coefficient `Intercept`). Group effects are independent
across nonlinear predictors. The recipe saves every parameter design and
prepares the response expression’s data columns on new rows.

### Monotonic predictors

Use an additive term such as `y ~ mo(dose) + (1 | site)` for ordered factors
or integer scores. A real slope sets the direction and average change across
steps. A simplex distributes that change over the steps. The lowest level
contributes zero. For example, with three levels, slope 2 and simplex
`[0.25, 0.75]`, the contributions are `[0, 1, 4]`.

Prediction reuses the retained training order of ordered factors and the
training range of integer scores. Unused factor levels are dropped. Gaps between integer scores retain their
steps, and predictions can use those intermediate scores. A term supports up
to 4,096 steps. To set priors,
select the slope with `class = "b", coef = "modose"` and the simplex with
`class = "simo", coef = "modose1"`. The simplex defaults to a uniform
Dirichlet. Use `dirichlet(2)` for a symmetric prior or
`dirichlet(c(2, 1))` to specify each concentration for a two-step term.
With one step, the simplex is fixed at `[1]`.

Monotonic terms can interact with numeric predictors, factors, and other
monotonic terms. For example, `y ~ mo(dose) * arm` adds an arm effect and
an arm contrast on the monotonic effect. Each resulting coefficient has
its own simplex per monotonic factor by default. Use
`mo(dose, id="dose_steps") * arm` to share one simplex across the main
effect and arm contrasts. Give shared simplexes a prior with
`class = "simo", coef = "dose_steps"`. A shared ID requires the same
number of steps in every occurrence.

### Families

Set `family` to `gaussian`, `student`, `lognormal`, `bernoulli`, `binomial`, `poisson`,
`negbinomial`, `geometric`, `Gamma`, `exponential`, `weibull`, `beta`, `von_mises`,
`cumulative`, `categorical`, `zero_inflated_poisson`, `zero_inflated_binomial`,
`zero_inflated_negbinomial`, `zero_inflated_beta`, `hurdle_poisson`,
`hurdle_negbinomial`, `hurdle_gamma` or `hurdle_lognormal`.
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
| Categorical | logit | — |

Student models use direct `sigma` and `nu` parameters, with `nu >= 1`.
Gamma and negative-binomial models use direct `shape`. Their mean is `mu`;
the emitted FlatPPL rate is `shape / mu`. Exponential models use rate `1 / mu`.
Geometric models count failures before the first success and use
`NegativeBinomial(1, 1 / mu)`, so `mu` is the expected count.
Weibull models use a direct `shape` and mean `mu`, with scale
`mu / gamma(1 + 1 / shape)` and a Gamma(0.01, 0.01) shape prior.
Beta models use mean `mu` and direct precision `phi`, with a Gamma(0.01, 0.01)
prior and response values strictly between zero and one.
`zero_inflated_beta` adds a probability `zi` of an exact zero; positive values
follow the same beta law. Its `phi` prior is unchanged and `zi` defaults to
Beta(1, 1). Responses lie in `[0, 1)`. The observation is an ordinary
superposition of a fixed atom and a continuous law, with their common reference.
`hurdle_poisson` assigns probability `hu` to zero and otherwise uses Poisson
conditioned on a positive count. It uses the Poisson links and a Beta(1, 1)
prior for direct `hu`; a predictor for `hu` defaults to logit with a
Logistic(0, 1) intercept prior.
`hurdle_negbinomial` uses the same zero probability and the negative-binomial
law conditioned on a positive count, retaining its InverseGamma(0.4, 0.3)
prior on `shape` and its mean/rate parameterization.
Sampling the zero-truncated count component is not yet supported by the Rust
sampler lowering; its density and derivatives are available.
`hurdle_gamma` combines an exact zero with the ordinary Gamma mean/shape law,
with `hu` defined as above and the same Gamma(0.01, 0.01) shape prior.
Its responses are nonnegative; the continuous component contributes no mass
at zero.
`hurdle_lognormal` instead uses the ordinary lognormal law for positive values,
with `mu` on the log-response scale and a direct `sigma`. The mean-link options
are identity and inverse. Its response-based priors use log responses, replacing
zeros by 0.1 before taking logs, just as brms does.
von Mises responses use radians in `[-pi, pi]`, a direct `kappa` with a
Gamma(2, 0.01) prior, and inverse tan-half link `2 * atan(eta)`.
The tan-half intercept prior is Student-t(1, 0, 1); the identity link uses the
ordinary response-based Student-t prior.
Binomial formulas require `y | trials(n) ~ ...`, where `n` is an integer or a
numeric column of counts. Trials remain required for prediction. Use positive
trial counts for execution. The converter accepts zero trials, but the emitted
FlatPPL `Binomial` law requires positive trials. Counts must be nonnegative
integers, and binomial responses cannot exceed their trials. Gamma,
exponential, lognormal and Weibull responses must be strictly positive.
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

Categorical responses accept factors, logical values or numeric labels. The first
retained level is the reference category, with logit zero. Every other category
gets a predictor from the main formula, named `mu` followed by its brms category
name. For levels `red`, `blue`, `green`, the predictors are `mublue` and `mugreen`:

```json
{
  "formula": "y ~ x",
  "family": "categorical",
  "dpars": {"mublue": {"formula": "mublue ~ 0 + z"}},
  "priors": [
    {"class": "b", "dpar": "mublue", "prior": "normal(0, 1)"},
    {"class": "b", "dpar": "mugreen", "prior": "normal(0, 1)"}
  ]
}
```

Each category retains its own design, centring, offsets, groups and priors.
The observation law is `Categorical(softmax([0, eta_blue, eta_green]))` for each
row. Coefficients default to Lebesgue measure and category intercepts to Student-t(3, 0, 2.5).
As in brms, `b`, `Intercept` and `sd` priors require `dpar`. Correlation priors
use global or group selectors across all categories, without `dpar`.
The recipe preserves training response levels and their predictor order;
`ModelInfo.dpars` lists the generated names. Names follow brms with C collation
and default numeric formatting; nonreference labels must be ASCII and ambiguous
names are refused.
An alternative reference category and a missing-row omission that removes an
entire response level are currently refused. Filter those rows and refit with an
explicit factor level order. Prediction needs no response or weights.

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

Gaussian, Exponential and Weibull formulas accept `y | cens(status, y2) ~ ...`.
Status is `left`, `none`, `right` or `interval`, or the corresponding integer
`-1`, `0`, `1` or `2`. Exponential and Weibull models require positive exact
times and allow zero as a censoring bound.
The response gives the exact value, the upper boundary for left censoring, or
the lower boundary for right and interval censoring. For interval rows, `y2`
gives the upper end of `(y, y2]`. Other rows ignore `y2`, including missing
values. Censored rows score the observed event mass.

Use `y | trunc(lb = 0, ub = limit) ~ ...` for a Gaussian response drawn from
a known range. Bounds can be constants or finite numeric columns. Omit one
bound for one-sided truncation. `K_model` conditions the response law on this
range. If `cens` and `trunc` both appear, censored rows score the event's
intersection with the range, divided by the range's probability.
`L` and `log_lik` score the observations before any `weights` apply.

`expected_response` takes the same parameter arguments as `K_model`. It
emits the conditional mean for each new row and response probabilities in
training order for categorical and ordinal models. Binomial means count
successes. Censoring predicts the latent response and truncation conditions
on the supplied bounds. Student models with one degree of freedom return
`nan`. Von Mises models use circular summaries and omit this function.
Prediction requires truncation bound columns and omits censoring inputs.
The StableHLO sampler currently refuses truncated Normal draws.

Use `y | ar(time = visit, group = subject) ~ x` for Gaussian residual AR(1).
Group and integer time columns are required. Times must be unique within each
group and can have gaps. Rows are ordered by group and time, with their input
indices saved in `rows`. Each group starts from the stationary residual
distribution. Across a gap of `d` time steps the residual has conditional
mean `ar^d` times the last residual and innovation variance
`sigma^2 * (1 + ar^2 + ... + ar^(2*d-2))`. The `ar` prior defaults to uniform
on `(-1, 1)`, and `sigma` is a scalar innovation standard deviation.
`log_lik` reports each conditional log density in model row order.
Prediction requires both columns and starts a fresh stationary trajectory
in each group, including groups that appeared in training.

Unordered factors use treatment contrasts. Ordered factors use polynomial
contrasts. Set `options.contrasts` per column, for example
`{"treatment": "sum"}`. Accepted values are `treatment`, `sum`, `helmert` and
`polynomial`. Factor values carry explicit `levels`, zero-based `codes` and
an optional `ordered` flag. Training drops unused factor levels and preserves
the remaining order. Logical predictors use the levels `FALSE`, `TRUE`.
No-intercept formulas use R's cell coding, including full columns for `0 + f`.
Use `0 + Intercept + x` for an ordinary, uncentred intercept coefficient.
Its prior has class `b`, optionally `coef = "Intercept"`, and defaults to
`Lebesgue(reals)`. The reserved column is generated as ones for training and prediction;
any supplied `Intercept` column must contain only ones. As in brms, this form
keeps contrast coding for population-level factors (`0 + Intercept + f`).
Polynomial contrasts support at most 16 observed levels. Higher-degree R
contrasts become sensitive to QR rank decisions and rounding; use treatment,
sum or Helmert contrasts for larger factors.
Indicator-only terms such as `0 + o` do not need polynomial contrasts and have
no such level limit.

Defaults follow brms: a centred Student-t intercept, half-Student-t scales,
and improper `Lebesgue(reals)` coefficients. Numeric response statistics use
the selected link for the intercept and group scales; direct `sigma` uses the
identity-link response scale. Lognormal uses log-response statistics for the
intercept, group scales and direct `sigma`, including with its inverse link.
The default Student degrees of freedom has a
Gamma(2, 0.1) prior normalized above 1. Gamma shape has Gamma(0.01, 0.01),
and negative-binomial shape has inverse-Gamma(0.4, 0.3).
Prior text is a FlatPPL measure expression. The brms aliases `normal`, `student_t`,
`cauchy`, `exponential`, `gamma`, `inv_gamma`, `beta`, `lognormal`, `logistic`,
`weibull`, `rayleigh`, `double_exponential`, `chi_square` and `uniform` accept their usual
positional arguments. `lkj`, `lkj_corr` and `lkj_corr_cholesky` require a group
correlation target.

Select priors by `class` (`b`, `Intercept`, `sd`, `sds`, `cor`, `L`, `sigma`, `nu`, `shape`, `phi`, `kappa`) with
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
including parameter-dependent normalizers. Lebesgue restrictions retain unit density,
even when an inferred parameter controls the interval width. Some intersections
of open supports with additional bounds cannot be represented as an interval.

`constant(c)` requires a finite numeric literal and creates a fixed coefficient
without a sampled coordinate. It requires fixed bounds when bounds are supplied;
the model must retain at least one sampled parameter.
Expressions use the ordinary FlatPPL type and phase validation. Shared coefficient
priors use `iid`; differing priors use scalar declarations and a combined vector.
Improper parameters use `elementof` and explicit measure factors. Proper parameters
remain draws with a combined `lawof`; `joint` combines independent groups, and
`jointchain` binds conditional priors explicitly. An improper prior cannot be
sampled, but its posterior can be proper.

`ModelInfo.parameters` labels each coordinate as `sampled`, `fixed` or `derived`;
the legacy `derived` flag remains true for both fixed and derived values.
`prior_factors` records each inference coordinate's prior mass.
`prior_sampling` is `unavailable` for nonnormalized factors and otherwise `not_checked`:
normalized mass does not establish an engine's sampler support.

The generated `formula_parameter_names` maps bindings and one-based coordinates
to the same parameter metadata. `formula_model` includes that map, design names
and retained factor, group and response levels. These module constants project
JSON sequences to `record(length = n, _1 = ..., _2 = ...)`, including empty
sequences. Integer level values use exact decimal strings and retain their
`kind = "integer"` tag. Host JSON keeps ordinary arrays and integer values.

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
The recipe stores the response formula, family, link and response coding separately
from its `predictors` map. Each entry (`mu`, `sigma`, and so on) retains its own
formula, link, training means, factor contrasts and levels, design terms, group
levels and priors. Numeric source columns remain raw. The emitted FlatPPL selects contrast values and computes interaction
products. Metadata records the default-prior rule and group collation.
Prediction requires no response column. It reuses the training contrast matrices
and remaps factor labels even when new data uses a different level order.
It rejects unseen factor and group levels.
Build a new recipe for a new fit or cross-validation fold.

`flatppl_formula::read_json_table(text)` accepts the typed table format, an array of row objects,
or an object of column arrays. Missing record fields and `null` cells remain
missing until model-frame construction. String columns become unordered factors
with sorted levels. Boolean, numeric and string values cannot be mixed in a
column. Integer columns preserve signed 64-bit values. Mixing integers with
decimal or exponent numbers produces float64 only when each integer remains
exactly representable; overflow and lossy promotion are errors. Typed tables
retain explicit float32, factor levels and orderedness. All-null columns remain
nullable numeric columns, so unused missing columns do not remove model rows.

`flatppl_formula::read_csv_table(bytes)` accepts a headered CSV, read by
`flatppl_data::read_csv`: fields are trimmed, lines starting with `#` are
skipped, and every row must have one field per column. Empty fields and the
exact token `NA` are null, including quoted fields.
Columns infer `true`/`false` or `TRUE`/`FALSE` booleans, signed 64-bit integers,
finite float64 numbers, then unordered string factors with sorted levels.
Numeric columns reject integer overflow and lossy integer-to-float promotion.
CSV quoting supports commas, quotes and newlines in fields. Use typed JSON or
Arrow when inference or these null tokens are unsuitable for a column.

The optional `arrow` feature adds `Table::from_arrow_ipc(bytes)` (from `flatppl-data`) for
IPC files and streams. Compressed buffers read only with `flatppl-data`'s
`arrow-compression` feature, which the host enables and the wasm build does not. It preserves float32, float64, integers, logical values,
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

[The distributional reproducer](tests/fixtures/distributional-oracle.R) generates
normalized brms models and independent density/gradient oracles for Gaussian
mean/scale predictors, Student mean/scale/degrees-of-freedom predictors, and
group effects on both mean and scale. The adjacent JSON fixtures retain the
executed Stan comparisons and parameter ordering.

[The spline fixture](tests/fixtures/splines.json) records mgcv 1.9.4 and
brms 2.23.2 training and prediction bases. Its adjacent R script regenerates
the values. The native basis preserves the fixed slope's scale and the full
penalized covariance; independent eigensolvers may rotate the standard Normal
coefficients without changing the induced smooth or prior.

[The categorical reproducer](tests/fixtures/categorical-oracle.R) checks saved
response coding, selector rules and numeric category names against brms, and
generates default and weighted category-specific Stan models. The adjacent JSON
retains their normalized scores, coefficient gradients and source hashes beside
independent softmax calculations.
