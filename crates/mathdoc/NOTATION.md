# FlatPPL as mathematics — the notation

How `flatppl-mathdoc` renders a FlatPPL module as mathematical notation. The
conversion is one way (FlatPPL → math), built from the typed and phased module,
and printed as MathML (the viewer's Math pane, HTML documents), plain TeX,
GitHub Markdown with TeX math, or native Typst. The structure below is target-independent; printers only choose
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
  than code (`Mⁿ` for `iid`, `f_* M` for `pushfwd`), but every rendering is
  the definition of the construct it stands for, never a guess at what the
  author meant.
- **A row never goes blank.** A right-hand side the lowering cannot render
  prints as its FlatPPL text in monospace, with a diagnostic saying why (see
  "Fallback").

## Document structure

- The doc-comment on `flatppl_compat` is the module documentation: its first
  Markdown heading is the title, the rest the abstract. `flatppl_compat` has no
  row.
- A **multi-line** (`%%%`) doc-comment is prose: a paragraph before its row,
  which also ends the current equation block. Headings inside it structure the
  document. A **one-line** (`%`) doc-comment is an annotation in the right
  column of its row. Whether the comment led or trailed the binding in the
  source is irrelevant.
- Document exports group consecutive rows with no prose between them into one block,
  aligned on the relation symbol. Each row is labelled with its binding name.
- Plain `#` comments do not exist after parsing and never appear.
- Parser-generated bindings (`__0x…`) have no row of their own. A decomposition
  folds into the statement it serves: `a, b, c ~ M` is one row `(a, b, c) ∼ M`;
  a `disintegrate` decomposition is one row `(K, ν) = disintegrate_{sel}(M)`.
  A partial decomposition `p, _ = v` inlines the source: `p = v_1`. A discarded
  value `_ = e` renders nothing. Author-named `_private` bindings are ordinary
  rows.
- Literal arrays beyond twelve entries, or wider than the shared 72-glyph
  estimate, print as `x ∈ ℝ^{n}` with an annotation and all values in an
  indexed data appendix. Wide rectangular tables of literal columns also
  move there, with their column names and row order intact; the model row
  shows a `data table` placeholder and its shape. Small or computed tables
  keep their `table(c = …)` expression. No values are rounded for layout.
- Long sums with at least three terms use aligned continuation lines in all
  printers. Only the left-associated sum chain splits; right-hand groups,
  enclosing fences, powers and source references stay intact. The shared
  width estimate is deterministic, not a font or viewport measurement.
  HTML equation blocks scroll horizontally when an expression still cannot fit.
- A generated notation key explains the symbols and distribution conventions
  used in the module. The HTML appendix also lists parameters (`elementof`),
  external inputs, and random variables (`draw`).

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
| --- | --- |
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
| --- | --- |
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
| --- | --- |
| `3`, `1.5`, `1e-6`, `true`, `"s"` | 3, 1.5, 1·10⁻⁶, true, "s" (powers of ten below 10⁻⁴ and from 10¹⁶) |
| `pi`, `inf`, `im` | π, ∞, i (upright) |
| `[a, b, c]`, `vector(…)` | (a, b, c) |
| `record(a = 1, b = x)` | (a = 1, b = x) |
| `(a, b)` tuple | (a, b) |
| `rowstack([[1, 2], [3, 4]])` | a bracketed matrix |
| `complex(a, b)`, `cis(t)`, `conj(z)`, `abs2(z)`, `real(z)`, `imag(z)` | a + b i, e^{i t}, z̄, \|z\|², Re z, Im z |
| `reals`, `posreals`, `nonnegreals`, `unitinterval` | ℝ̄, (0, ∞], [0, ∞], [0, 1] |
| `integers`, `posintegers`, `nonnegintegers`, `booleans`, `complexes` | ℤ, ℤ_{>0}, ℕ₀, 𝔹, ℂ |
| `interval(a, b)` | [a, b] (closed, as §03 defines it; `inf` ends stay closed) |
| `cartpow(S, n)`, `cartpow(S, [m, n])` | Sⁿ, S^{m×n} |
| `cartprod(S, T)`, `cartprod(a = S, b = T)` | S × T, {a ∈ S, b ∈ T} |
| `stdsimplex(n)` | Δ^{n−1} |
| `c = fixed(x)` | c = x, annotation "fixed"; inside a record `fixed(x)` stays in roman |

## Operators and functions

| FlatPPL | Math |
| --- | --- |
| `a + b`, `a - b`, `-a` | a + b, a − b, −a |
| `a * b` | ab by juxtaposition; a numeric literal coefficient reads first (`x * 3` → 3x), with a dot between numeric factors |
| `a / b` | a fraction |
| `a * (1 / b)` | a / b as one fraction, without cancelling factors |
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
positionally in that order (`Normal(mu = m, sigma = s)` → 𝒩(m, s²)); on a
callable without one they print as `name = value`.

The arithmetic tree applies these display rules as it is built, so MathML,
TeX and Typst agree. Signed numeric coefficients read first too
(`cis(-1.2)` → e^{−1.2i}). Symbolic factors keep their order, including matrix
products. No constants or powers are evaluated, and no factors cancel:
`Normal(0, sigma * 3)` remains 𝒩(0, (3σ)²), and `x * (1 / x)` remains x/x.
The source graph and its numeric evaluation are unchanged.
Negative right-hand factors keep parentheses, including nested coefficients:
`a * (b * -3)` reads as a(−3b), not a − 3b.

## Collections, broadcasting, aggregation

| FlatPPL | Math |
| --- | --- |
| `iid(M, n)`, `iid(M, [m, n])` | Mⁿ, M^{m×n} — the n-fold product measure as a bare power (van der Vaart's Pⁿ), matching Sⁿ for `cartpow`; the `⊗` is kept for products of different factors |
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
| --- | --- |
| `Normal(mu, sigma)` | 𝒩(μ, σ²); `Normal(0, 2)` stays 𝒩(0, 2²), never 𝒩(0, 4) |
| `MvNormal(mu, cov)` | 𝒩(μ, cov), with the covariance unchanged |
| `StudentT(nu)`, `ChiSquared(k)` | t_ν, χ²_k |
| `Gamma(shape, rate)`, `Exponential(rate)` | Gamma(shape, rate), Exp(rate) |
| `InverseGamma(shape, scale)`, `Weibull(shape, scale)` | InverseGamma(shape, scale), Weibull(shape, scale) |
| `Uniform(S)` | 𝒰(S) with the set |
| `Lebesgue(support = S)`, `Counting(S)`, `Dirac(v)` | λ_S (λ for ℝ̄), Counting(S), δ_v |
| `weighted(w, M)`, `logweighted(l, M)` | as the set function §06 defines: ν(A) = ∫_A w(x) dM(x), ν(A) = ∫_A e^{l(x)} dM(x); the bound variable is the weight's own parameter (a lambda's body is written out), else a fresh letter; a constant weight is c · M |
| `superpose(M1, M2)` | M₁ + M₂ |
| `normalize(M)`, `totalmass(M)` | normalize(M), totalmass(M) |
| `truncate(M, S)` | M\|_S |
| `pushfwd(f, M)`, `locscale(M, a, b)` | f_* M, a + b · M |
| `joint(M1, M2)`, `joint(a = M1, b = M2)` | M₁ ⊗ M₂, M₁(da) ⊗ M₂(db) — only when no component is stochastic, reifies a draw, or reaches into a loaded module (§06: `joint` retains shared stochastic ancestors and is then not a product; a loaded module's binding may hold a draw this module cannot see, a standard module holds none); otherwise, and for a spelling that mixes positional and keyword components, joint(…) in roman |
| `relabel(M, ["x"])` | M(dx) |
| `lawof(x)`, `lawof(record(a = a, b = b))` | Law(x), Law(a, b) — upright, as in current probability writing; the script ℒ stays free for an author's likelihood |
| `kernelof(x, p = a)` as an expression | p ↦ Law(x \| p); with no inputs, Law(x) |
| `functionof(e, p = a)` as an expression | p ↦ e; with no inputs, e |
| `F = functionof(e)` with no inputs | F() = e |
| `densityof(M, x)`, `logdensityof(M, x)` | p_M(x), log p_M(x) |
| `likelihoodof(K, data)` as an expression | p_K(data \| inputs); with no inputs, p_K(data) |
| `joint_likelihood(L1, L2)` | L₁ · L₂ |
| `bayesupdate(L, prior)` | ν(A) = ∫_A L(θ) d prior(θ) with L's inputs as the bound variables (an inline `likelihoodof` writes p_K(data \| θ)); L · prior when L has no inputs |
| any of the three in expression position | the set slot is the placeholder: normalize(∫_· L(θ) d prior(θ)) |
| `restrict(M, record(a = v))` | M(· \| a = v) — the unnormalised conditional, distinct from `truncate`'s M\|_S |
| a keyword spelling of a measure operator (`bayesupdate(prior = …, L = …)`) | the construct name in roman with `name = value` arguments |
| `kchain`, `jointchain`, `markovchain`, `kscan`, `ksuperpose`, `disintegrate`, `bijection`, `PoissonProcess`, … | the construct name in roman with its arguments |

A record whose fields are references to bindings of the same name prints as the
list of those symbols (`Law(μ, τ, θ)`); any other field prints as `name = value`.

The Normal variance keeps the source scale expression, grouped before squaring.
Rate and scale conventions appear in the legend, not as argument labels. Other
distributions retain their upright FlatPPL names and declared argument order.
Unresolved keyword arguments retain the named call rather than guessing roles.
`reals` means the extended reals ℝ̄, including both infinities (§03). The ℝ in
the shorthand for an elided array of finite literals remains the finite reals.

## Export formats

The CLI selects the document format from the output suffix:

```sh
flatppl convert model.flatppl model.html
flatppl convert model.flatppl model.md
flatppl convert model.flatppl model.tex
flatppl convert model.flatppl model.typ
```

All formats use the same lowered expressions and notation key. HTML retains
MathML, source links, and rendered documentation. GitHub Markdown uses fenced
`math` blocks with `aligned` equations and preserves Markdown documentation.
The fences keep TeX backslashes and underscores out of Markdown parsing.
The emitted `\operatorname` and `aligned` constructs belong to MathJax's AMS
support. Author documentation is preserved, not translated to a restricted TeX
dialect. Foreign Typst documentation
appears as fenced source. TeX and Typst exports preserve documentation as
escaped literal text, not executable author commands. They include diagnostics
and the full data appendix. TeX documents require LuaLaTeX or XeLaTeX with
`unicode-math` for Unicode identifiers and prose.

The Rust JSON API also accepts `formats: ["mathml", "tex", "typst"]`. Each
binding and notation entry contains only the requested forms. TeX and Typst
forms are native math source without delimiters or browser annotations.
MathML remains the default. The viewer shows the shared notation key in a
collapsed disclosure after the equations.

## Modules, data, randomness

| FlatPPL | Math |
| --- | --- |
| `m = load_module("f.flatppl", c = v)` | m = load_module("f.flatppl", c = v) |
| `h = standard_module("particle-physics", "0.1")` | h = standard_module("particle-physics", "0.1") |
| `x = load_data("d.csv", S)` | x = load_data("d.csv", S) |
| `rnginit`, `rand`, `rngstate` | roman |

## Fallback

Every construct has a rendering (unknown builtins print in roman), so the
fallback is reached on two conditions only: a right-hand side nested deeper than
`flatppl_core::DEFAULT_MAX_DEPTH` levels before or after lowering (a
`superpose` of 200 terms folds into a 200-deep chain), or a bare hole `_`
outside `fn(…)`. The row then shows the binding's FlatPPL text in monospace (as
written when the source is at hand, its canonical print otherwise; a projection
of a decomposition shows the source with its component index, `p = v[1]`) and
carries a diagnostic saying why. A binding whose inference failed
still renders structurally; only the typed features (index ranges, kernel input
lists, the `joint` independence test, the notation appendix) degrade.
