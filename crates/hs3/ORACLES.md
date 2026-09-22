# Independent oracle values for the HS3 / pyhf importer

Concrete, per-construct checkpoints for the lowerings in this crate: one
evaluated point (or a small set) for each HS3 kind, derived **independently of
this code** — from the HS3 standard and pyHS3, from ROOT/RooFit, or in closed
form. They exist because the end-to-end gate (`flatppl-testsuite`) compares a
whole model's frozen ΔNLL vector: when that vector moves, these values say
*which* construct moved.

An oracle is only an oracle if it was obtained without consulting the converter.
"The tests pass", "the comment reads correctly" and "both engines agree" are not
evidence here. Where no independent oracle was reachable, the construct is
listed under [Still unverified](#still-unverified) rather than assumed correct.

Ground truth for a whole HS3 document is ROOT/RooFit. pyHS3 is a second backend
and can itself diverge from ROOT — the `shapesys` row below is a case where it
does.

## Fundamental distributions

| Construct | Point | Value | Obtained from |
|---|---|---|---|
| `Normal(1, 2)` | `logpdf(0.5)` | `-1.6433357137646178` | Distributions.jl, cross-checked in closed form |
| `Exponential(rate = 1.54)` | `logpdf(1.0)` | `-1.1082175835744623` | closed form; pins `rate = c`, **not** `neg(c)` |
| `LogNormal(0.5, 0.8)` | `logpdf(2.0)` | `-1.4180873447615459` | closed form |
| `Uniform(0, 10)` | `logpdf(3.5)` | `-2.302585092994046` (= `-log 10`) | closed form |
| `GeneralizedNormal(1, 2, 3)` | `logpdf(0.5)` | `-1.288727719379548` | closed form from `β/(2αΓ(1/β))·exp(−(\|x−μ\|/α)^β)`; numeric normalization gives `∫ pdf = 1.0000000013` |
| `Poisson(5)` | `pmf(3)` | `0.14037389581428056` | closed form |
| `MvNormal([1, -2], [[2, .5], [.5, 1]])` | `pdf([0, 0])` | `0.005192489090464107` | closed form reading the matrix as a **covariance** (density uses `Σ⁻¹`) |
| — same, read as a precision matrix | `pdf([0, 0])` | `0.02849378822612621` | the discriminating wrong reading — the two differ by 5.5×, so this row is the regression sentinel |

## HEP shapes

| Construct | Point | Value | Obtained from |
|---|---|---|---|
| Crystal Ball tail/core join | `A(B+α)^{-n}` | `0.32465246735834974` = `exp(-1.5²/2)` | closed form: continuity of the piecewise density at the join |
| `Argus(5.2, -20, 0.5)` | `f(4.0)` | `0.0007264926315388241` | closed form |
| `Argus(5.2, -20, 0.5)` | `f(4.5)` | `0.014860551800001563` | closed form |
| `Argus(5.2, -20, 0.5)` | `f(0)`, `f(5.2)` | `0` (both) | support endpoints, `interval(0, resonance)` |

## Counting and extended likelihoods

| Construct | Point | Value | Obtained from |
|---|---|---|---|
| `rate_extended`: `N = 100`, shape `Normal(0, 1)`, 3 events at `[0.5, -0.3, 1.2]` | logdensity | `-89.831` | the HS3 extended likelihood `N^k e^{−N} ∏ pdf_D(t_i)` in closed form; confirms `weighted(weight = N, base = D)` argument order |
| barlow-beeston-lite: expected `[10, 8]`, observed `[12, 7]` | joint pmf | `0.013230057574642698` (log `-4.3253`) | product of independent per-bin Poissons; confirms reference measure Counting and `expected` domain `posreals` |

## Composition

| Construct | Point | Value | Obtained from |
|---|---|---|---|
| `product_dist`, distinct variates | `N(0.5; 0, 1) · N(1; 2, 3)` | `0.04428784979103323` | closed form |
| `product_dist`, shared variate (renormalized) | `x = 0.3` | `0.38151055652198734`, with `Z = 0.14246520430275` | closed form product-of-Gaussians (precision-weighted Normal); matches RooProdPdf over a shared observable |
| `mixture_dist`, non-extended | `x = 0.7` | HS3 `0.11269233693718823` vs FlatPPL `normalize(...)` `0.11269233693707821` | pyHS3 against the emitted form — agreement to ~1e-12 |
| `mixture_dist`, extended | `x = 0.7` | bare `superpose` `0.7060056821356362`; pyHS3 `0.14120113642712723` | pyHS3; the ratio is exactly `5.0 = Σ cᵢ`, i.e. the two differ by the missing normalization |

## Generic expressions, density functions, polynomials

| Construct | Point | Value | Obtained from |
|---|---|---|---|
| `generic_dist`, `w = 1 + 0.1·\|x\| + sin(√\|5x + 0.1\|)` on `[-20, 20]` | normalization, density | `Z = 86.27727286774481`; `density(3) = 0.007215784669634544` | numeric quadrature of the expression |
| `density_function_dist` / `log_density_function_dist`, `g = -0.5(x-1)²` on `[-5, 5]` | `normalize(...)` at `x = 2` | `0.24197838851393727` | quadrature; equals the truncated `Normal(1, 1)` pdf at 2 (`0.24197838851393724`), with `Z = 2.5065488841277204` |
| `polynomial_dist`, `1 + 0.1x` on `[-10, 10]` | mass, `pdf(5)` | `M = 20`, `pdf(5) = 0.075` | closed form over the **declared** range. Without the range truncation the mass diverges (`M = ∞`, `pdf(5) → 0`) — that pair is the regression sentinel for the domain-truncation step |
| `chebychev_dist`, `a0 = 0.5`, `a1 = 0.2` on `[0, 10]` | `x = 2 / 5 / 8` | `0.644 / 0.800 / 1.244` | pyROOT `RooChebychev`, cross-checked with scipy. Integer (floor) coefficient division instead of real division yields `0.700 / 0.800 / 0.800` — the regression sentinel |

The Chebyshev convention these values pin is `1 + Σ aₖ·Tₖ`, with
`t = (2x − lo − hi)/(hi − lo)` and `coefficients[0] → a₁`; an independent
pyROOT + scipy adjudication puts the engine within ~2e-9 of ROOT on it.

## HistFactory

| Modifier / piece | Point | Value | Obtained from |
|---|---|---|---|
| `normsys`, default interpolation `poly6_exp` (code 4) | `α = 0.5 / 1.5 / -0.7` | `1.04931492 / 1.15368973 / 0.92909316` | closed form from the interpolation definition |
| `histosys`, default `poly6_lin` (code 4p), nominal 20, hi 24, lo 17 | `α = 0.4` | `21.534768`; delta-add equals absolute form to `0e+00` | closed form |
| `shapesys`, per-bin `gamma` with `ContinuedPoisson(gamma·tau)`, `tau = (nom/σ)²` | `tau`; `logL` | `tau = [277.778, 55.184]`; `logL(γ = 1) = -6.658431`; `logL(γ = [1.2, 0.9]) = -11.864923` | closed form |
| `staterror`, Poisson form `tau = Σnom²/Σσ²`; Gauss form `Normal(gamma, δ)`, `δ = √(Σσ²)/Σnom` at 1 | `tau`, `δ`, `logL` | `tau = [400, 100]`, `δ = [0.05, 0.1]`, `logL(1) = 3.460440` | closed form |
| interpolation code 1 vs code 4 (the nested `data.interpolation` lift) | `α = 0.5` | `1.04880884817` vs `1.04931491542` (Δ 5.06e-4) | closed form — the separation is small, so a test that does not use these points cannot tell the codes apart |
| interpolation code 1 vs code 4 | `α = -0.7` | `0.92890170` vs `0.92909316` | closed form |
| native `shapesys` `vals` read as relative vs absolute, nominal `[50, 52]`, vals `[3, 7]` | `tau` | this crate: `[0.111, 0.0204]`; pyHS3: `[277.8, 55.2]` | ROOT is the authority and agrees with this crate — a case where pyHS3 diverges from ROOT, **not** a converter bug |

## Fixture parameter signs

Upstream fixtures store the exponential slope already inverted, so the HS3 `c`
that reaches the importer is positive: `rf703` has `tau = -1.54 → c = +1.54`,
`rf207` has `alpha = -1.0 → c = +1.0`. This is what makes `rate = c` (no
negation) the right lowering, and it is the concrete reason HS3 `c` belongs in
`posreals`.

## Still unverified

No independent oracle has been reached for these; they are open, not assumed
correct.

- Extended-mixture normalization against ROOT's extended NLL (only the pyHS3
  comparison above exists).
- `MvNormal`'s HS3 wire form — the field name `covariances` and the 2-D array
  shape come from the profile table alone; the covariance *interpretation* is
  verified, the wire shape is not. No fixture; pyHS3 lacks the kind.
- Crystal Ball end to end, and whether ROOT's HS3 writer emits `_L`/`_R` fields
  on `crystalball_dist` or a distinct `crystalball_doublesided_dist` type
  string. No fixture.
- `rate_density_dist`, `bincounts_extended_dist`, `bincounts_density_dist` —
  derived in closed form from their intensity measures, but no HS3 fixture and
  no pyHS3 implementation, so never compared end to end.
