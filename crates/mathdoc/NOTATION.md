# FlatPPL as mathematics — the notation

How `flatppl-mathdoc` renders a FlatPPL module as mathematical notation. The
conversion is one way (FlatPPL → math), built from the typed and phased module,
and printed as MathML (the viewer's Math pane, HTML documents) or Typst
(documents). The structure below is target-independent; printers only choose
glyphs. Semantics follow spec §04 and §06; where a construct has an established
mathematical notation it is used, where it has none the construct keeps its
FlatPPL name in roman type. Nothing is inferred about the author's intent beyond
the rules stated here.

## Principles

- **One binding, one row.** A module renders as its bindings in **source order**,
  each as one relation `lhs rel rhs`. No topological sort: structuring the text
  is the author's job, as it is in a paper.
- **What you name is what you get.** Symbols come from the binding names by the
  fixed rules below. An author who wants `a_{g_i}` writes `g = group_data`; one
  who wants `CrystalBall` writes `CrystalBall = hepphys.CrystalBall`.
- **Measures appear bare, densities appear as `p`.** A measure-valued expression
  prints as the measure; a density value prints as `p_M(x)` with the measure in
  the subscript. The distinction FlatPPL makes at the language level is kept.
- **Overload notation, never semantics.** Math may overload operators more freely
  than code (`M^{⊗n}` for `iid`, `f_* M` for `pushfwd`), but every rendering is
  the definition of the construct it stands for, never a guess at what the
  author meant.
- **A row never goes blank.** A right-hand side the lowering cannot render
  prints as its canonical FlatPPL text in monospace, with a diagnostic saying
  why (see "Fallback").

## Document structure

- The doc-comment on `flatppl_compat` is the module documentation: its first
  Markdown heading is the title, the rest the abstract. `flatppl_compat` has no
  row.
- A **multi-line** (`%%%`) doc-comment is prose: a paragraph before its row,
  which also ends the current equation block. Headings inside it structure the
  document. A **one-line** (`%`) doc-comment is an annotation in the right
  column of its row. Whether the comment led or trailed the binding in the
  source is irrelevant.
- Consecutive rows with no prose between them form one aligned block (aligned on
  the relation symbol). Each row is labelled with its binding name.
- Plain `#` comments do not exist after parsing and never appear.
- Parser-generated bindings (`__0x…`) have no row of their own. A decomposition
  folds into the statement it serves: `a, b, c ~ M` is one row `(a, b, c) ∼ M`;
  a `disintegrate` decomposition is one row `(K, ν) = disintegrate_{sel}(M)`.
  A partial decomposition `p, _ = v` inlines the source: `p = v_1`. A discarded
  value `_ = e` renders nothing. Author-named `_private` bindings are ordinary
  rows.
- Arrays beyond twelve entries print as `x ∈ ℝ^{n}` with an annotation and the
  values in a data appendix. A `table(…)` literal prints as `table(c = …)` with
  its columns (typesetting it as a table is planned).
- A generated notation appendix lists parameters (`elementof`), external inputs,
  random variables (`draw`) and the parametrisation of every distribution used,
  read off the module.

## Names

Split the binding name on `_`. The **head** is the first segment, with any
trailing digits split off.

| Head | Renders as | Examples |
|---|---|---|
| a spelled-out Greek letter (`alpha` … `omega`, `Alpha` … `Omega`, `var` forms) | the Greek letter | `mu` → μ, `Gamma` → Γ |
| a single Latin letter | that letter in italics | `x`, `J`, `L` |
| anything else | the **whole name** as upright text, underscores kept | `prior`, `alphaxy`, `forward_kernel`, `rcp_max`, `yS` |

Only a Greek or single-letter head takes subscripts: the head's trailing digits
first, then each further segment (Greek, letter, word or digits by the same
rule), comma-separated.

| Name | Renders as |
|---|---|
| `theta1`, `s12`, `c0` | θ₁, s₁₂, c₀ |
| `mu_a`, `sigma_B`, `S_mu`, `nu_B` | μ_a, σ_B, S_μ, ν_B |
| `y_data`, `x_init`, `L_input` | y_{data}, x_{init}, L_{input} (upright subscript words) |
| `E1_data`, `Z0_12` | E_{1,data}, Z_{0,12} |
| `sigma2` | σ₂ (the rule cannot know it means σ²; name it so it reads right) |

Heads are resolved by role: `Gamma` in value position is Γ, `Gamma(shape, rate)`
in call position is the distribution, printed roman. Distribution and builtin
names, lambda parameters that are words, and module-qualified references
(`common.f_a`, `hepphys.CrystalBall`) print as upright text. Axis names follow
the same rules (`.mu` → μ).

## Statement forms

| FlatPPL | Row |
|---|---|
| `x = expr` | x = expr |
| `x ~ M`, `x = draw(M)` | x ∼ M |
| `x = elementof(S)` | x ∈ S |
| `x = external(S)` | x ∈ S, annotation "external input" |
| `f(a, b) = expr`, `f = (a, b) -> expr`, `f = fn(… _ …)` | f(a, b) = expr |
| `F = functionof(e, p = a, q = d)` | F(p, q) = e with a, d read as p, q |
| `F = functionof(y)` (inputs from inference) | F(inputs) = y, by reference to y's own row |
| `K = kernelof(x, p = a)` | K(p) = Law(x \| p) |
| `L = likelihoodof(K, data)` | L(inputs) = p_K(data \| inputs) |
| `C[.i, .k] := body` | C_{ik} = Σ_{j} body |
| `g: s[] := body` | s = body with upper/lower indices, annotation "indices lowered with g" |
| `a, b ~ M` | (a, b) ∼ M |
| `K, nu = disintegrate(["obs"], M)` | (K, ν) = disintegrate_{obs}(M) |

## Values and sets

| FlatPPL | Math |
|---|---|
| `3`, `1.5`, `1e-6`, `true`, `"s"` | 3, 1.5, 1·10⁻⁶, true, "s" (powers of ten below 10⁻⁴ and from 10¹⁶) |
| `pi`, `inf`, `im` | π, ∞, i (upright) |
| `[a, b, c]`, `vector(…)` | (a, b, c) |
| `record(a = 1, b = x)` | (a = 1, b = x) |
| `(a, b)` tuple | (a, b) |
| `rowstack([[1, 2], [3, 4]])` | a bracketed matrix |
| `complex(a, b)`, `cis(t)`, `conj(z)`, `abs2(z)`, `real(z)`, `imag(z)` | a + b i, e^{i t}, z̄, \|z\|², Re z, Im z |
| `reals`, `posreals`, `nonnegreals`, `unitinterval` | ℝ, (0, ∞], [0, ∞], [0, 1] |
| `integers`, `posintegers`, `nonnegintegers`, `booleans`, `complexes` | ℤ, ℤ_{>0}, ℕ₀, 𝔹, ℂ |
| `interval(a, b)` | [a, b] (closed, as §03 defines it; `inf` ends stay closed) |
| `cartpow(S, n)`, `cartpow(S, [m, n])` | Sⁿ, S^{m×n} |
| `cartprod(S, T)`, `cartprod(a = S, b = T)` | S × T, {a ∈ S, b ∈ T} |
| `stdsimplex(n)` | Δ^{n−1} |
| `c = fixed(x)` | c = x, annotation "fixed"; inside a record `fixed(x)` stays in roman |

## Operators and functions

| FlatPPL | Math |
|---|---|
| `a + b`, `a - b`, `-a` | a + b, a − b, −a |
| `a * b` | ab by juxtaposition; a · b when the right operand is a number or both are |
| `a / b` | a fraction |
| `a ^ b`, `sqrt(a)`, `abs(a)` | a^{b}, √a, \|a\| |
| `==`, `!=`, `<`, `<=`, `>`, `>=`, `in` | =, ≠, <, ≤, >, ≥, ∈ |
| `&&`, `\|\|`, `!` | ∧, ∨, ¬ |
| `ifelse(c, a, b)` | cases: a if c, b otherwise |
| `exp`, `log`, `floor`, `ceil`, `min`, `max` … | exp(x), log(x), ⌊x⌋, ⌈x⌉, min(a, b), max(a, b) |
| `transpose(A)`, `adjoint(A)`, `inv(A)`, `det(A)`, `trace(A)` | Aᵀ, A†, A⁻¹, det A, tr A |
| `eye(n)`, `diagmat(v)`, `quadform(A, x)`, `cross(a, b)` | I_n, diag(v), xᵀ A x, a × b |
| `sum(v)`, `prod(v)`, `maximum(v)`, `minimum(v)` | Σ_i v_i, Π_i v_i, max_i v_i, min_i v_i |
| `l1norm(v)`, `l2norm(v)`, `linfnorm(v)` | ‖v‖₁, ‖v‖₂, ‖v‖_∞ |
| `v[i]`, `A[i, j]`, `A[:, j]`, `t[1]` | v_i, A_{i,j}, A_{·,j}, t_1 |
| `r.a`, `m.x` | r.a, m.x |
| `get(x, ["a", "c"])`, `get(v, [1, 3])` | x_{\{a, c\}}, v_{(1, 3)} |
| `fchain(f, g)` | g ∘ f |
| any other builtin `f(args)` | f(args) in roman |

Keyword arguments to a builtin with a declared parameter order print
positionally in that order (`Normal(mu = m, sigma = s)` → Normal(m, s)); on a
callable without one they print as `name = value`.

## Collections, broadcasting, aggregation

| FlatPPL | Math |
|---|---|
| `iid(M, n)`, `iid(M, [m, n])` | M^{⊗n}, M^{⊗(m×n)} |
| `K.(xs, ys)`, `broadcast(K, xs, ys)` with K a kernel | ⨂_{i=1}^{n} K(xs_i, ys_i) |
| `f.(xs, c)`, `xs .+ c`, `broadcast(f, xs, c)` with f a function | (f(xs_i, c))_{i=1}^{n}, (xs_i + c)_{i=1}^{n} |
| nested dotted expressions | one family, one index: (invlogit(a_{g_i} + b x_i))_{i} |
| `a[idx]` with an array of indices | a_{idx} and, under an index i, a_{idx_i} |
| `aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])` | Σ_{j} A_{ij} B_{jk}, other reductions as var_{j}(…) in roman |
| `metricsum(g, [.mu^], r[.mu^] * r[.mu_])` | r^{μ} r_{μ}, upper and lower indices as written |

Index letters are fresh (i, j, k, … skipping names bound in the module). The
range comes from the typed module: a named size where the source gives one
(`iid(M, J)`), else the static length, else a bare index.

## Measures, kernels, likelihoods (§06, §04)

| FlatPPL | Math |
|---|---|
| `Normal(0, 1)`, `Gamma(shape = a, rate = b)` | Normal(0, 1), Gamma(a, b) — §08 names, §08 argument order |
| `Uniform(S)` | Uniform(S) with the set |
| `Lebesgue(support = S)`, `Counting(S)`, `Dirac(v)` | λ_S (λ for ℝ), Counting(S), δ_v |
| `weighted(w, M)`, `logweighted(l, M)` | w · M, e^{l} · M |
| `superpose(M1, M2)` | M₁ + M₂ |
| `normalize(M)`, `totalmass(M)` | normalize(M), totalmass(M) |
| `truncate(M, S)` | M\|_S |
| `pushfwd(f, M)`, `locscale(M, a, b)` | f_* M, a + b · M |
| `joint(M1, M2)`, `joint(a = M1, b = M2)` | M₁ ⊗ M₂, M₁(da) ⊗ M₂(db) — only when no component is stochastic or reifies a draw (§06: `joint` retains shared stochastic ancestors and is then not a product); otherwise joint(…) in roman |
| `relabel(M, ["x"])` | M(dx) |
| `lawof(x)`, `lawof(record(a = a, b = b))` | Law(x), Law(a, b) |
| `kernelof(x, p = a)` as an expression | p ↦ Law(x \| p); with no inputs, Law(x) |
| `functionof(e, p = a)` as an expression | p ↦ e; with no inputs, e |
| `F = functionof(e)` with no inputs | F() = e |
| `densityof(M, x)`, `logdensityof(M, x)` | p_M(x), log p_M(x) |
| `likelihoodof(K, data)` as an expression | p_K(data \| inputs); with no inputs, p_K(data) |
| `joint_likelihood(L1, L2)` | L₁ · L₂ |
| `bayesupdate(L, prior)` | L · prior |
| `restrict(M, record(a = v))` | M(· \| a = v) — the unnormalised conditional, distinct from `truncate`'s M\|_S |
| a keyword spelling of a measure operator (`bayesupdate(prior = …, L = …)`) | the construct name in roman with `name = value` arguments |
| `kchain`, `jointchain`, `markovchain`, `kscan`, `ksuperpose`, `disintegrate`, `bijection`, `PoissonProcess`, … | the construct name in roman with its arguments |

A record whose fields are references to bindings of the same name prints as the
list of those symbols (`Law(μ, τ, θ)`); any other field prints as `name = value`.

## Modules, data, randomness

| FlatPPL | Math |
|---|---|
| `m = load_module("f.flatppl", c = v)` | m = load_module("f.flatppl", c = v) |
| `h = standard_module("particle-physics", "0.1")` | h = standard_module("particle-physics", "0.1") |
| `x = load_data("d.csv", S)` | x = load_data("d.csv", S) |
| `rnginit`, `rand`, `rngstate` | roman |

## Fallback

Every construct has a rendering (unknown builtins print in roman), so the
fallback is reached on two conditions only: a right-hand side nested deeper than
`flatppl_core::DEFAULT_MAX_DEPTH` levels before or after lowering (a
`superpose` of 200 terms folds into a 200-deep chain), or a bare hole `_`
outside `fn(…)`. The row then shows the binding's canonical FlatPPL text in
monospace and carries a diagnostic saying why. A binding whose inference failed
still renders structurally; only the typed features (index ranges, kernel input
lists, the `joint` independence test, the notation appendix) degrade.
