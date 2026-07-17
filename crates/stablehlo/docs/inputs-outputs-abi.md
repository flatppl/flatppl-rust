# The `inputs` / `outputs` compilation ABI

`flatppl-stablehlo` emits a StableHLO `func.func` from a determinized FlatPDL
module. FlatPDL has no marker for *which* binding is a function argument or
*which* is the result, so the emitter needs to be told. This document describes
the explicit ABI a model declares with two reserved bindings, `inputs` and
`outputs`, and the default convention used when they are absent.

## Default convention (no `inputs` / `outputs`)

With neither binding present, `@logdensity` is emitted from a convention:

- every `elementof` parameter becomes an argument, in source order;
- the **last public binding** is the single result (it must contain a density
  term, else emission refuses).

This is adequate for a self-contained scoring model whose density is the last
binding. It is fragile once cross-module grafting (a `load_module` query
scoring a foreign `posterior`) reorders bindings, and it offers no way to make
observed data a runtime argument — data reached by the query is emitted as
`stablehlo.constant`. The CLI prints a one-line deprecation warning on this
path and encourages declaring the ABI explicitly.

## Declaring the ABI

```
inputs  = v | (v1, v2, …)
outputs = v | (v1, v2, …)
```

Both are reserved top-level binding names; each is a single value or a tuple,
and **tuple order is the ABI order** of the emitted function. When a model
declares either binding:

- dead-code elimination is rooted on `{inputs, outputs}` — the emitted module
  keeps the backward cone feeding the outputs plus the declared inputs, and
  drops everything else (a constant that feeds an output but descends from no
  input is kept — it is needed to compute the result);
- the emitter reads the ABI off the determinized module directly, rather than
  guessing positionally.

The ABI applies to `--mode logdensity` only; `--mode sample` ignores it.

### `inputs` — the arguments

Each `inputs` entry becomes a `func.func` argument, typed by its FlatPPL phase:

| Entry | Emitted argument |
|---|---|
| `elementof(S)` parameter | argument (inferred element kind: real / int / bool) |
| `external(S)` input | argument typed from `S` |
| `load_data(source, S)` input | `tensor<N×f32>`, `N` pinned from a compile-time read of the file's row count (`.csv` / `.wsv`) — **values are never baked**, they are the runtime argument |

`inputs` is **authoritative and exhaustive**: every `elementof` parameter in
the module must appear in it, or emission refuses. A binding that is neither an
`elementof` parameter nor a fixed `external`/`load_data` input (e.g. a literal
or a derived computation) cannot be an argument and refuses. A `load_data`
whose file format is unsupported (`.json`, Arrow) refuses rather than
mis-shaping the argument. A fixed-phase input the query reaches but that is not
listed in `inputs` refuses, pointing at the ABI.

### `outputs` — the results

Each `outputs` entry is a `logdensityof(M, point)` query (spec §06 — a measure
and the point to evaluate it at). Determinization reduces each to a
deterministic density expression before emission. The function returns the
outputs in declared order (a single value, or a tuple for multiple outputs).

## Worked examples

Ordered arguments, single result — `inputs` names the three parameters, so they
become `%arg0..%arg2` in that order:

```
alpha = elementof(reals)
beta  = elementof(reals)
sigma = elementof(posreals)
mu    = alpha .+ beta .* [1.0, 2.0]
y     = draw(Normal.(mu, sigma))
inputs  = (alpha, beta, sigma)
outputs = logdensityof(lawof(record(y = y)), record(y = [1.1, 2.2]))
```

```mlir
func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32>
```

Multiple outputs — the function returns a tuple in `outputs` order (here a
likelihood and a posterior, both functions of the one input `mu`):

```
mu = elementof(reals)
inputs  = mu
outputs = (logdensityof(L, record(a = mu)), logdensityof(post, record(a = 0.5)))
```

```mlir
func.func @logdensity(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>)
```

Data as a runtime argument — a `load_data` input listed in `inputs` is a
shape-pinned tensor (here `d.csv` has 3 data rows), so one compiled module
scores any length-3 data vector without re-emitting; the data is not a constant:

```
mu = elementof(reals)
y  = load_data("d.csv", reals)
inputs  = (mu, y)
outputs = logdensityof(post, record(a = mu))
```

```mlir
func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<3xf32>) -> tensor<f32>
```

Fallback — a model with neither binding still emits, from the last-public-binding
convention, with a deprecation warning:

```
mu = elementof(reals)
lp = logdensityof(lawof(record(a = draw(Normal(mu = mu, sigma = 1.0)))), record(a = 0.5))
```

```
warning: no inputs/outputs bindings; using the legacy last-public-binding query — declare inputs/outputs for an explicit ABI
```

```mlir
func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32>
```

## Query modules (`load_module`)

A common use is a small query module that composes a pristine model and scores
it:

```
m = load_module("model.flatppl")
t_alpha = elementof(reals)
t_beta  = elementof(reals)
inputs  = (t_alpha, t_beta)
outputs = logdensityof(m.posterior, record(alpha = t_alpha, beta = t_beta))
```

Note the free-parameter **binding** names are prefixed (`t_alpha`, not
`alpha`): a bare `alpha` would collide with the loaded model's own `alpha`
binding across the independent module namespaces and refuse. The record
**field** names stay the variate names the model itself uses.

## Where this lives in the code

- `modes::read_abi` — reads the `inputs`/`outputs` bindings into an `Abi`
  (input binding symbols in order; output query nodes in order); `None` when
  neither binding is present (the fallback signal).
- `modes::emit_logdensity_abi` — the ABI emission path: exhaustiveness check,
  ordered arguments, shape-pinned `load_data`, ordered results.
- `EmitOptions::input_shapes` — the compile-time `load_data` length map
  (binding name → shape) the host supplies; the CLI reads each `load_data`
  file's row count into it.
- The CLI verb (`flatppl stablehlo`) recognizes `inputs`/`outputs` on the
  surface model, roots determinization DCE on them, and populates
  `input_shapes`.
