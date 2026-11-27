# FlatPPL — Rust implementation

Rust ecosystem for FlatPPL, the Flat Portable Probabilistic Language.

## About FlatPPL

FlatPPL is a minimal, inference-agnostic stochastic language for specifying
probabilistic models.

## Components

This monorepo is a Cargo workspace; member crates live under `crates/`:

* [`flatppl-core`](crates/core) — the in-memory IR (extended-FlatPIR model)
* [`flatppl-syntax`](crates/syntax) — canonical FlatPPL surface syntax: parse + print
* [`flatppl-flatpir`](crates/flatpir) — FlatPIR S-expression reader + writer
* [`flatppl-infer`](crates/infer) — type, shape, and phase inference
* [`flatppl-optimise`](crates/optimise) — output-driven pruning and egglog rewrites for the typed model DAG
* [`flatppl-model-analysis`](crates/model-analysis) — dependency analysis, scalar conjugacy, sufficient statistics, incremental scores and Normal coordinate maps
* [`flatppl-autodiff`](crates/autodiff) — source-to-source AD with egglog rules
* [`flatppl-hs3`](crates/hs3) — HS3 / pyhf → FlatPPL importer
* [`flatppl-formula`](crates/formula) — Gaussian regression formulas → FlatPPL, with reusable data recipes
* [`flatppl-convert`](crates/convert) — shared IR construction helpers for converters
* [`flatppl-fileaccess`](crates/fileaccess) — resolve a `source` (local path or `http`/`https` URL) to a local file, with the shared remote-content cache (native host layer)
* [`flatppl-host`](crates/host) — immutable source contexts and typed StableHLO query exports for native hosts
* [`flatppl-python-api`](crates/python-api) — PyO3 bindings for the [Python package](https://github.com/flatppl/flatppl-python)
* [`flatppl-mathdoc`](crates/mathdoc) — FlatPPL rendered as mathematics: a math AST over the typed module, printed as MathML (HTML pages, the viewer's Math pane); notation in [`NOTATION.md`](crates/mathdoc/NOTATION.md)
* [`flatppl-cli`](crates/cli) — the `flatppl` command-line driver
* [`flatppl-lsp`](crates/lsp) — FlatPPL language server (diagnostics, hover, go-to-definition, completion)

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the design and the road ahead.

## Installation (early users)

Requires a Rust toolchain (≥ 1.85, e.g. via [rustup](https://rustup.rs)).
The crates are not published to crates.io yet; install the CLI straight from
the repository:

```sh
cargo install --git https://github.com/flatppl/flatppl-rust flatppl-cli
```

This places a `flatppl` binary on your Cargo bin path:

```sh
flatppl convert model.flatppl model.flatpir   # FlatPPL → FlatPIR
flatppl convert model.flatpir model.flatppl   # FlatPIR → FlatPPL
flatppl convert messy.flatppl tidy.flatppl    # canonicalize (same format)
flatppl infer model.flatppl typed.flatpir    # emit type/phase-annotated FlatPIR
flatppl infer --level=phase m.flatppl m.flatpir  # or: type, valueset, normalization, shape
flatppl optimise model.flatppl --emit flatpir -o optimised.flatpir
flatppl prepare model.flatppl                  # fetch the model's remote deps into the cache
flatppl convert model.flatppl model.html       # the model as a page of mathematics (MathML)
flatppl stablehlo query.flatppl --dtype f64 -o query.mlir # float64 compilation; default is f32
```

StableHLO queries must declare `inputs` and `outputs` to define their compiled ABI.
The f32 dtype uses i32 integer arithmetic and f64 uses i64: an integer literal
outside that width refuses, and a computed integer beyond it wraps.

### Source differentiation

Build the CLI with `--features autodiff` and request the derivative explicitly.

```sh
flatppl autodiff query.flatppl --wrt theta -o gradient.flatppl
```

Gradient mode requires one real scalar output. Repeat `--wrt` for each selected
query input. Inputs can be real scalars, fixed-shape real arrays, records,
fixed-row tables, or promoted fixed tuples of these. Each selected tuple
position must support a derivative. Record and table derivatives contain the
real fields in declaration order. Arrays retain their shape and tables retain
their row count. The result lists the original value followed by gradients in
the requested order.
Use `--empty-wrt` for an empty selection, including queries with only discrete
inputs. The gradient query then returns the primal alone.

Use `--mode jvp` or `--mode vjp` for structured outputs. Both modes preserve
the original values and split nested tuple outputs into derivative components.
JVP takes one tangent per selected input and returns the output tangents.
VJP takes one cotangent per differentiable output component and returns one
cotangent per selected input.

AD runs after determinization and before optimization. Egglog rules cover
pointwise arithmetic and conditionals, scalar broadcast callbacks, reductions,
indexing and stacking, matrix operations, selected distribution transports,
and log densities including Beta, Gamma, StudentT, Dirichlet, and MvNormal.
The transform also handles fixed record and array scans, log-space Markov
updates, and fixed-grid solver steps with active initial states and parameters.
Active operations without a rule produce an error. Derivatives apply on the
smooth, finite domain of the query.

Each rule has a key: the catalog (`Pullback` or `Pushforward`), the operation,
its operand count and the operand slot from 0. `flatppl autodiff --list-rules`
prints every key, with `*` for a value that the rule leaves open.
`--explain-rules` prints the keys one derivative loads to stderr, as `rule`
lines, before the transform runs.

Rust callers can use `flatppl_autodiff::determinize_query` followed by
`gradient`, `jvp`, or `vjp`. The result is one FlatPDL module with the primal
and derivative outputs. To print standalone FlatPPL, call
`flatppl_autodiff::prepare_source` before `flatppl_syntax::print`.

Use `--numerical-integrals` to differentiate an integral and print the
result as FlatPPL. The generated query uses weighted integrals and requires
the same opt-in when compiled.

```sh
flatppl autodiff query.flatppl --numerical-integrals --wrt theta -o gradient.flatppl
flatppl stablehlo gradient.flatppl --numerical-integrals
```

Numerical AD integrates the derivative of the integrand and adds terms for
moving bounds and supported scalar cutoffs. Reverse differentiation supports
scalar, fixed-vector, fixed-matrix, and nested record captures. The Rust
`*_with_options` entry points accept `NUMERICAL_SOURCE_OPTIONS`.

Shape derivatives of the regularized incomplete gamma functions and their log
forms also require `--numerical-integrals`. The real projection of Bessel J
supports order derivatives when the declared bounds imply an order in
`[0, 16]` and an argument in `[1/8, 4]`.
Call `prepare_source_with_options` before printing. To compile the
derivative directly to StableHLO, add `--ad-wrt` to the `stablehlo` command
and optionally select `--ad-mode jvp` or `--ad-mode vjp`.

### Lowering and emission

The Rust determinizer lowers affine log weights to closed distribution laws for
Normal, MvNormal, Gamma, Exponential, ChiSquared, Bernoulli, Binomial, Poisson,
Geometric and both NegativeBinomial parameterizations. This applies to
`normalize(logweighted(...))` and equivalent `tilt` expressions with a
unit-density source reference. Runtime parameters remain live in density and
sampling queries. Nonlinear weights and restricted references retain their
existing lowering paths.

Positive-count `Poisson` and `NegativeBinomial` laws expressed as
`normalize(truncate(M, interval(1, inf)))` have exact density normalizers,
including live parameter gradients. Sampling these truncated laws is not yet
implemented and is explicitly refused.

StableHLO emission restricts lowering to avoid known Enzyme limitations by
default. This mode preserves zero-factor product and cumulative product derivatives and avoids known
selection-adjoint failures. It retains source tensor operations but skips
pointwise packet packing.
It shares identical gathers and their slice lowerings within each region to
avoid rebuilding the same tensor.
It requires static tensor shapes and deterministic queries, and refuses `probit`
and complex `cumprod`. Pass `--restrict-enzyme-compatible=false` for unrestricted emission,
including sampling queries. The Rust API exposes the same choice through
`EmitOptions::restrict_enzyme_compatible`, which defaults to `true`.
`--target cpu|gpu` (`EmitOptions::target`, default `cpu`) picks a lowering
profile for the executing backend. It changes only how equivalent operations
are lowered: the output keeps the same entry point, inputs and values, and runs
on any backend.
Qualification targets first-order real-input CPU derivatives with Enzyme-AD 0.0.15
and JAX 0.10.2. The flag does not invoke Enzyme or guarantee every composition
or compiler version supports differentiation. Unrestricted output may also work
with Enzyme; disabling the restrictions does not imply incompatibility.

Formats are inferred from the file extensions. FlatPPL output uses the full
surface syntax (operators, indexing, lambdas, `:=`); pass `--syntax minimal`
for the lowered function-call form instead.

**Inputs are local files.** A model may `load_module` *remote* (`http`/`https`)
dependencies, but the command input itself is always a local path — like
`cargo build` on a local crate. `convert` and `infer` are **local and offline**:
they resolve dependencies from a shared local cache only and never touch the
network. To populate that cache, run **`flatppl prepare <model>`** — the one
command that downloads a model's transitive remote dependencies (recursively,
relative URLs resolved against their importing file). `flatppl prepare --update`
refreshes already-cached deps. Cache location and trust are env-controlled
(`$FLATPPL_CACHEDIR`; an untrusted URL prompts interactively and is refused in
non-interactive use unless `FLATPPL_TRUST=1`).

## Building and testing (developers)

`flatppl optimise` accepts FlatPPL or FlatPIR and writes FlatPPL by default.
For example, selecting a field from a private record can skip the other fields' computations when they have no observable effects.
Portable rewrites also run during determinization. StableHLO applies its rules to the validated query before emission, using the existing CPU or GPU target.
Stan can select native matrix contractions. The rule files cover each category separately. [Rule catalogue](crates/optimise/src/rules)

Independent likelihood factors use shared data summaries where supported.
`flatppl stan --reduce sufficient` prepares observed IID Normal summaries in Stan's
`transformed data` block; without it, the program states Stan's vectorized Normal
density. A change in the observation count also selects that density. For
`determinize` and `stablehlo`, disable the reduction with
`--sufficient-statistics=false`. For runtime data,
emit a preparation query and a scoring query in either portable or StableHLO form:

```sh
flatppl determinize query.flatppl --data-input y --prepare-data prepare.flatppl -o score.flatppl
flatppl stablehlo query.flatppl --dtype f64 --data-input y --prepare-data prepare.mlir -o score.mlir
```

`flatppl analyze query.flatppl --data-input y` reports structural parameter
components, additive terms and advisory Normal coordinates without changing
the query. Data classification is explicit. The report lists required map checks.

The same command reports conjugate posterior updates reachable from `outputs`.
For example, a `Beta(1, 1)` prior and one observed Bernoulli success give
`Beta(2, 1)`. For `rand(normalize(P), state)` the report names the matched
families and identifies the exact IID draw selected by Rust StableHLO lowering.
`logdensityof(normalize(P), point)` scores the matched posterior law.
`totalmass(P)` and `log(totalmass(P))` give the evidence and log evidence for
each matched update. Count evidence retains each observation’s count coefficient.
Ordinary `densityof` and `logdensityof` queries reuse fixed observation summaries.
The same updates supply normalized scores, evidence and seeded draws. For live
IID observations, prepare the query once per dataset to reuse its summaries.
Binomial evidence with changing trial counts still needs the row allocations.

This optional Rust capability covers named parameters with Beta–Bernoulli,
Beta–Binomial, Gamma–Poisson, Normal–Normal (mean), InverseGamma–Normal
(variance), or Dirichlet–Categorical, Categorical0 or Multinomial updates.
For a named joint prior, each likelihood factor must use one prior field and
each field must have a matching update. Analysis reports the fields in prior order.
“No matching rule” leaves conjugacy
undetermined. Runtime data can determine the posterior parameters and its
validity. Other FlatPPL engines choose their own sampling methods.

Named Normal coefficients with affine Normal observation means can use one
exact joint Gaussian update. The rule covers explicit scalar observation
factors with known noise, up to 64 rows including priors. It supplies normalized
density, evidence and seeded draws through the same queries. Analysis reports
the block as `MvNormal`.

MvNormal, Dirichlet and Wishart support vector/matrix summaries. MvNormal and
Wishart require fixed finite data or explicitly bounded runtime inputs, for
example `y = external(cartpow(cartpow(interval(-100, 100), 3), 1000))`.
Unbounded observations keep ordinary scoring. Use float64 for covariance
matrices close to singularity.

Categorical and Categorical0 likelihoods can reuse one count per category
when observations outnumber categories. Bernoulli accepts Boolean or numeric
observations. Binomial likelihoods can prepare live observations for fixed
trial counts from 1 through 63.

InverseWishart and small fixed-trial Multinomial likelihoods also support
summaries. Fixed-design Gaussian regression supports matrix-vector means and
affine scalar features, intercepts and offsets with shared sigma. Prepare both
the design and observations, for example `--data-input X --data-input y`.
The reduced score uses a scaled QR factor and a data-derived coefficient
reference. Runtime regression and InverseWishart data require explicit
preparation to amortize QR setup. Parameter-dependent designs and varying sigmas retain
ordinary scoring.

InverseGamma, Pareto, fixed-shape Weibull and fixed-dispersion NegativeBinomial
likelihoods also support summaries. Real observations need positive finite
values or explicit positive finite bounds. Count reductions require dispersion
at most 64 or fixed counts below 64. Include runtime shape/dispersion inputs in
preparation with another `--data-input` option. These families require explicit
preparation for runtime data so setup is reused between scores.

ChiSquared, bounded VonMises, LKJ and LKJCholesky also support preparation.
VonMises requires location bounds within `[-pi, pi]` and concentration bounds
within `[0, 64]`. Grouped scalar families reuse the same support and numerical
guards as their shared-parameter reductions. They support fixed index vectors,
for example `Normal.(mu[group], sigma)` and `Poisson.(rate[group])`.
Prepare observations with `--data-input y`. Runtime groups require explicit
preparation of their indices with `--data-input group` and a supported family.
Indexed lambdas such as `f(i,m)=Normal(m[i],1.0)` with `f.(group,[mu])` also
support grouping. Deterministic derived observations can use several preparation
inputs. Include each dependency with `--data-input`.

Runtime Exponential, Gamma, Beta, Poisson and Geometric groups prepare bounded
statistics. Real families require finite support bounds and positive parameter
bounds. Gamma/Beta concentrations stay at most 64. Poisson requires a positive
rate lower bound; Geometric requires probabilities strictly between zero and one.
Binomial groups prepare native outcome counts for fixed trials from 1 through 63.
Unsupported ranges retain ordinary scoring. Preparation must run again after
observations or indices change.

Runtime group predictors using `exp`, `invlogit` and affine real arithmetic also
support preparation, for example `Bernoulli.(invlogit.(a[group] .+ b))`.
The score evaluates links once per group while `a` and `b` remain live inputs.
Bounds must prove finite forward and reverse arithmetic in both precisions.
Unproved ranges, saturated probabilities and other links retain ordinary scoring.

Add `--explain-reductions` to either command to report committed rules and
fallbacks on stderr. Reports cover recognized independent density sums and
distinguish lowering, materialized data, and explicit preparation.
Reporting leaves output files and the query ABI unchanged. Each grouped report
counts observations, occupied parameter tuples, and singleton groups.

Run preparation once per data binding, then append its results to the scoring
arguments in preparation-output order. Other scoring arguments keep their
original order. Data needed outside reduced factors remains a scoring argument.
Both queries must use the same precision. See [the shared pass](ARCHITECTURE.md#shared-likelihood-summaries)
for coverage and the Rust preparation API.

Preparation packs scalar summaries into vectors of matching real, integer, or
boolean type. Other summary shapes stay unchanged. Use the exported schemas to
bind each result as one score argument. Disable packing with
`--pack-summaries=false` or `PreparationOptions { pack_summaries: false }`.

Rust hosts can call `Context::prepare_incremental_score` to obtain ordinary
term queries and an original-order combiner. `affected_terms` identifies work
after named inputs change. `revisions` supplies explicit invalidation and
stale-result tickets. Notify every change, including in-place array edits and
rollbacks, and check native status before accepting a result. Each compiled
plan, precision and target needs a separate coordinator. Cache term gradients
too when a full gradient is required.

```sh
git clone https://github.com/flatppl/flatppl-rust
cd flatppl-rust
cargo build --workspace        # build all crates
cargo test --workspace         # run all test suites
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
cargo run -p flatppl-cli -- convert model.flatppl model.flatpir
```

Rust hosts can opt into `Context::prepare_normal_coordinate` for one scalar
Normal parameter. It compiles a score, forward map, inverse map and log Jacobian
at the requested precision. Every companion returns `(value, valid)` and may
also return native runtime status. Check both. The original query stays unchanged,
and only the selected coordinate's input slot changes in the companion score.

## Funding

This work was supported by Germany's Federal Ministry of Research, Technology
and Space (BMFTR) within the ErUM-Data programme under grant FKZ 05D25PC1
(DEMOS consortium).

## License

[MIT](LICENSE)
