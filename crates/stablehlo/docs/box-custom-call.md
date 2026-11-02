# The `flatppl.box` custom call

A FlatPPL model can use a black box through flatppl-bridge. The box is an
ordinary module whose members are `external(anything)`, and the host binds the
module's `load_module` alias to a box. The emitter turns each box query into one
portable `stablehlo.custom_call` that the executor binds. The call names no
server, transport or library.

## Binding a box

The host supplies the box's describe reply (flatppl-bridge protocol,
"Describe") at compile time:

- CLI: `flatppl stablehlo model.flatppl --box bb=normal:describe.json`, where
  `bb` is a top-level `load_module` alias of the model, `normal` the box name on
  its server and `describe.json` the reply.
- Rust: `flatppl_host::parse_describe` and `LoadedModule::compile_with_boxes`.
- Python: `Module.compile(boxes={"bb": flatppl.Box(url, "normal")})`.

Every public binding of the interface module must be `external(anything)` and a
member of the box. Inference types each member from its describe entry: a
measure member is a measure on its `variate` set, a kernel member a kernel from
its `input` set, and a function member a function to its `output` set. A record
input set gives one interface input per field. Any other input set gives one
input named `input`.

## Lowering

The determiniser lowers model-side queries to the protocol's primitive forms:

| Model | FlatPDL |
|---|---|
| `logdensityof(bb.M, x)` | `builtin_logdensityof(bb.M, x)` |
| `logdensityof(bb.K(a = u, b = v), x)` | `builtin_logdensityof(bb.K, record(a = u, b = v), x)` |
| `densityof(...)` | `exp` of the log density |
| `likelihoodof(bb.K, x)` at `theta` | `builtin_logdensityof(bb.K, theta, x)` |
| `y ~ bb.K(theta)` in a density | the kernel density at `y` |
| `rand(bb.M, s)`, `y ~ bb.M` when sampled | `builtin_sample(s, bb.M)` |
| `iid(bb.K(theta), n)` | `sum(broadcast(builtin_logdensityof, bb.K, singleton, x))` |
| `rand(iid(bb.K(theta), n), s)` | `builtin_sample(s, bb.K, theta, n)` |
| `bb.f(x)`, `bb.f.(xs)` | the application itself |

A measure member is a nullary kernel, so its primitives take no input argument.
The ordinary rules handle `joint`, `weighted`, `logweighted`, `truncate` and
`pushfwd` with a bijection around these leaves. A form without a primitive
reduction, such as `normalize` of an unnormalized member, refuses. `~` and
`rand` refuse on a member whose describe entry has `normalized: false`, because
only a probability measure has samples.

## The operation

```mlir
%r = stablehlo.custom_call @flatppl.box(%operands...) {
  api_version = 4 : i32,
  backend_config = {
    box = "normal", member = "K", operation = "logdensity",
    query = "broadcast(builtin_logdensityof, B.K, input, x)",
    arguments = [
      {name = "input", valueset = "cartprod(mu = reals, sigma = reals)", rowwise = false, fields = ["mu", "sigma"]},
      {name = "x", valueset = "reals", rowwise = true}],
    rows = [3], sizes = []}
} : (tensor<f32>, tensor<f32>, tensor<3xf32>) -> tensor<3xf32>
```

| Attribute | Meaning |
|---|---|
| `box` | The box name. |
| `member` | The member name. |
| `operation` | `logdensity`, `sample` or `apply`. |
| `query` | The protocol query term, with `B` the box and each argument a binding. |
| `arguments` | The query arguments in order (see below). |
| `rows` | The broadcast row shape. Empty for a single query. |
| `sizes` | The sample sizes of `builtin_sample`. |

Each argument has a `name` (`input` or `x`), the `valueset` of one row as
FlatPPL text, and `rowwise`. A row-wise argument has the row axes first and
iterates over them. Any other argument is held constant across the rows. A
record argument lists its `fields` and contributes one operand per field, in
that order. Otherwise an argument is one operand.

The value sets come from the inferred types, so they carry shape and element
kind but not bounds. The box server checks every value against the member's own
declared sets.

### Operands and results

- `logdensity`: the argument operands; one real result of shape `rows`.
- `apply`: the input operands; the output of each row, of shape `rows` plus the
  output shape.
- `sample`: the RNG state `tensor<2xui64>` first, then the input operands; the
  draws, of shape `sizes` plus the variate shape.

The box has its own RNG. The executor seeds it with `rnginit(bytes)`, where
`bytes` are the 16 little-endian bytes of the state operand. The program then
advances its own state with one `rng_bit_generator`, so later draws stay
independent of the box's.

A record output or variate, a dynamic shape, and a box query inside a loop or a
broadcast function body refuse. A box query is always a top-level operation of
the function, which lets the executor split the program around it. A program
compiled over host batch axes refuses box queries: the executor batches them
itself.

## Executing

The executor must replace each `@flatppl.box` call before it compiles the
program. flatppl-python splits the function at the calls. It runs the pieces
between them through Enzyme-JAX and binds each call to `jax.pure_callback`,
which sends one protocol request. A custom VJP sends the derivative request:
`gradient` for a scalar result per row, and `vjp` otherwise. Under `jax.vmap`, a
call sends one `broadcast(...)` query for the whole batch, with one RNG state
per row for a sample.

## Where this lives

- `flatppl_core::BoxBinding`: a box bound to a `load_module` alias of a module.
- `flatppl_infer::boxes`: types members from the describe entries.
- `flatppl_determinizer::boxes`: the density and sample leaves, and the
  refusals.
- `crate::boxes` (this crate): the custom call.
- `flatppl_host::boxes`: parses the describe reply and binds aliases.
