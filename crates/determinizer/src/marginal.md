# The conjugate marginal table — the maths

Sidecar to `marginal.rs`. Every row of `CONJUGATE_TABLE` is one closed-form solution of
the `kchain` marginal integral. This file states each integral, its closed form, the §08
constructor that names the answer, and the test point that pins it — so a reader can
check the maths without reverse-engineering a `build_*_marginal`.

## What the table solves

§06 *Density of composed measures*: "`kchain` marginalizes the intermediate variate, so
its density is the marginal integral $\int \mathrm{densityof}(K(a), x)\,\mathrm{d}M(a)$.
This is generally intractable; an engine evaluates it in closed form, or by enumeration
of a discrete latent, and otherwise reports a static error."

A table row IS the "in closed form" branch, for one recognised prior/likelihood pair:

$$\int p_K(x \mid a)\, p_M(a)\, \mathrm{d}a = p_{M'}(x)$$

for a single §08 distribution $M'$. The row supplies $M'$'s parameters as functions of
$M$'s and $K$'s. It is not general integration: a row matches only when the two
constructors line up exactly, the latent feeds exactly the conjugating parameter, and
every other likelihood parameter is latent-independent.

## Two spellings, one table

**Explicit.** `kchain(lawof(record(z = z)), kernelof(record(y = y), z = z))` names the
integral in the measure algebra. `lower_kchain_marginal` handles it.

**Implicit.** `lawof(y)` over a `y ~ Dist(param = z, …)` with `z` latent asks for the
same integral. §04 *Reification to measures*: `lawof(x)` "reifies the ancestor sub-DAG of
`x` as the **probability measure** that is the total law of x". §04 *Kernels and
`kernelof`* spells out which ancestors are integrated: a `prior_predictive =
lawof(record(obs = obs))` is "obtained by marginalizing over `theta1` and `theta2` — they
are internal stochastic nodes in the traced sub-DAG, not boundary inputs, so `lawof`
integrates them out. `prior_predictive` is equivalent to `kchain(prior,
forward_kernel)`." `conjugate_marginal_measure` routes this spelling to the same rows.

Both spellings reach the table, in both statement orders, whether the latent is still
latent or an earlier query already pinned it.

## Row 1 — Normal prior on a Normal mean

| | |
|---|---|
| model | `z ~ Normal(mu = μ₀, sigma = σ₀)`; `y ~ Normal(mu = z, sigma = σ)` |
| latent | `z`, feeding the likelihood's `mu` |
| §08 answer | `Normal(mu = μ₀, sigma = sqrt(σ₀² + σ²))` |
| builder | `build_normal_normal_marginal` |

The integral, with §08's `Normal(mu, sigma)` density
$\frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$:

$$\int \mathcal{N}(y;\, z,\, \sigma)\, \mathcal{N}(z;\, \mu_0,\, \sigma_0)\, \mathrm{d}z
= \mathcal{N}\!\left(y;\, \mu_0,\, \sqrt{\sigma_0^2 + \sigma^2}\right)$$

$y = z + \varepsilon$ with $z \sim \mathcal{N}(\mu_0, \sigma_0^2)$ and
$\varepsilon \sim \mathcal{N}(0, \sigma^2)$ independent, so the sum is Gaussian with the
means and the **variances** added. The marginal is wider than either factor — a builder
that subtracted the variances, or dropped one, would be narrower.

The parameter map is arithmetic, emitted as
`Normal(mu = μ₀, sigma = sqrt(add(pow(σ₀, 2), pow(σ, 2))))`. With literal σ it const-folds.

### Test point

| | |
|---|---|
| model | `z ~ Normal(0, 1)`; `y ~ Normal(mu = z, sigma = 1)` |
| point | `y = 0.5` |
| marginal | `Normal(0, √2)`, emitted `sigma` folds to `1.4142135623730951` |
| truth | `-1.3280121234846454` |
| wrong answer | `-0.9389385332046728` |
| gap | `0.389` nats |

The wrong answer is the **conditional** `Normal(0.3, 1)` at `y = 0.5` — the density the
determiniser emitted when a sibling query had pinned `z = 0.3` and nothing recorded that
the literal had been a latent. That is the defect the implicit routing closes, so it is
the discriminator this point is chosen for.

**What the point discriminates against.** The truth-minus-wrong gap crosses zero twice
over `y`, at `y ≈ -0.6515` and `y ≈ +1.8515`: the marginal is flatter, so it loses in the
centre and wins in the tails. A test at either crossing is blind by construction. `y = 0.5`
sits next to the gap's central extremum (at `y = 0.6`, where
$\frac{d}{dy}\left[-\frac{y^2}{4} + \frac{(y-0.3)^2}{2}\right] = 0$), `1.15` away from the
nearer crossing.

Structurally the emitted `sigma` pins the variance sum by itself: `sqrt(1² + 1²)` folds to
`1.4142135623730951`, where a variance **difference** folds to `0.0` and either term alone
to `1.0`.

**A caution this point does not cover.** With `σ₀ = σ = 1` the two sigmas are
interchangeable, so a builder that swapped prior and likelihood sigma passes here. The
formula is symmetric in them, so that is not a defect — but a row whose map is *not*
symmetric needs a point with distinct parameters. `conjugate_golden.rs`'s explicit-spelling
golden uses `σ₀ = 2`, `σ = 1` → `sqrt(5)` for that reason; keep both.

## Row 2 — Gamma prior on a Poisson rate

| | |
|---|---|
| model | `rate ~ Gamma(shape = α, rate = β)`; `k ~ Poisson(rate = rate)` |
| latent | `rate`, feeding the likelihood's `rate` |
| §08 answer | `NegativeBinomial(alpha = α, beta = β)` |
| builder | `build_gamma_poisson_marginal` |

The integral, with §08's `Poisson(rate)` density $\frac{\lambda^k e^{-\lambda}}{k!}$ and
`Gamma(shape, rate)` density $\frac{\beta^\alpha}{\Gamma(\alpha)}\lambda^{\alpha-1}
e^{-\beta\lambda}$:

$$\int_0^\infty \frac{\lambda^k e^{-\lambda}}{k!} \cdot
\frac{\beta^\alpha}{\Gamma(\alpha)} \lambda^{\alpha-1} e^{-\beta\lambda}\, \mathrm{d}\lambda
= \frac{\beta^\alpha}{\Gamma(\alpha)\, k!} \int_0^\infty \lambda^{k+\alpha-1}
e^{-(1+\beta)\lambda}\, \mathrm{d}\lambda
= \frac{\beta^\alpha\, \Gamma(k+\alpha)}{\Gamma(\alpha)\, k!\, (1+\beta)^{k+\alpha}}$$

which regroups to

$$\binom{k+\alpha-1}{\alpha-1}
\left(\frac{\beta}{\beta+1}\right)^{\alpha}
\left(\frac{1}{\beta+1}\right)^{k}$$

— **exactly** §08's `NegativeBinomial(alpha, beta)` density, term for term. So the
parameter map is the identity: `alpha ← shape`, `beta ← rate`, no arithmetic. §08's own
`beta` is the rate, not a probability; the success probability is $p = \beta/(\beta+1)$.

Note the map is only the identity because §08 parameterizes `Gamma` by **rate** and
`NegativeBinomial` by the matching rate. A `scale`-parameterized prior would need
`beta = 1/scale`.

### Test point

| | |
|---|---|
| model | `rate ~ Gamma(shape = 2, rate = 1/3)` (scale 3); `k ~ Poisson(rate = rate)` |
| point | `k = 5` |
| marginal | `NegativeBinomial(alpha = 2, beta = 1/3)`, i.e. $p = 1/(1+3) = 1/4$ |
| truth | `-2.419239615270632` |
| wrong answer | `-1.8286943966417715` |
| gap | `0.591` nats |

The wrong answer is `Poisson(6)` at `k = 5` — the likelihood evaluated at the prior MEAN
rate $\alpha/\beta = 6$, i.e. the conditional with the integral replaced by a plug-in.
That is the plausible mislowering here: the marginal and the plug-in share a mean, so only
the **dispersion** tells them apart.

**What the point discriminates against.** The gap crosses zero twice over `k`, between
`k = 3` and `k = 4` and between `k = 10` and `k = 11`: the negative binomial is
overdispersed relative to the Poisson with the same mean, so it loses mass in the centre
and gains it in both tails. A test in either tail, or at a crossing, would not see the
plug-in error. `k = 5` is one step from the gap's extremum (`k = 6`, `k = 7`, gap `0.724`)
and two steps inside the lower crossing.

Structurally, the emitted constructor NAME already separates the two answers
(`NegativeBinomial` vs `Poisson`), and the identity map means the emission must carry no
arithmetic at all.

## Refusal is the fallback, and stays load-bearing

No row applying is a refusal, never a licence to score the conditional. The refusals worth
naming, each with a test:

- **Two record fields marginalizing the SAME latent.** This one is different in kind from
  the rest, and it is the only place where every per-row answer is right and the assembled
  result is still wrong — so the row cannot catch it. For
  `y1, y2 ~ Normal(mu = z, sigma = 1)` over `z ~ Normal(0, 1)`, each marginal is
  `Normal(0, √2)` and each is correct, but the fields are **correlated** through the shared
  ancestor: `Cov(y1, y2) = Var(z) = 1`. So

  | | |
  |---|---|
  | truth at `(0.5, 0.7)` | `MvNormal([0,0], [2 1; 1 2])` = `-2.5171832107434002` |
  | product of the marginals | `-2.716024246969291` |
  | gap | `0.199` nats |

  §04 *Kernels and `kernelof`* works this exact shape and makes the joint
  `kchain(prior, forward_kernel)`, which is not a product of the fields' marginals for any
  prior. `conjugate_marginal_measure` therefore returns the latent it integrated
  (`ImplicitMarginal::latent`), and the record path refuses when two marginalized fields
  report the same one. Two fields over DIFFERENT latents is a genuine product and lowers.

  **`iid` and `joint` over the same model are NOT this case and correctly emit the
  product.** §06 defines `joint(M1, M2, …)` as the "independent product measure"
  `(M1 ⊗ M2)(A × B) = M1(A) · M2(B)`, so `joint(a = lawof(y1), b = lawof(y2))` asks for the
  product of the two marginals — a different measure from `lawof(record(y1 = y1, y2 = y2))`,
  which is the law of the traced sub-DAG. This is why the check is sited in the record path
  rather than in the row or the combinator.

- **The latent feeds a scale, not the mean.** `z ~ Normal(0, 2)`; `y ~ Normal(mu = 1, sigma = z)`
  is a Normal prior on a standard deviation. That is not the Normal–Normal (mean)
  conjugacy — no closed form here — and emitting Row 1's marginal for it would be the most
  likely way to get this whole table wrong. Detection is the row's
  `conjugating_param` check: the latent's ref must be the value of exactly `mu`.
- **The families do not pair.** A `Gamma` prior on a `Normal` mean is no table entry.
- **Two latents.** `y ~ Normal(mu = z, sigma = w)` with both latent is not a
  single-prior integral; check (c) and `sole_named_ancestor` both reject it.
- **A derived parameter.** `y ~ Normal(mu = 2 * z, …)` has a closed-form marginal
  (`Normal(2μ₀, sqrt(4σ₀² + σ²))`) that is NOT this row's, so the exact-ref check
  refuses it.
- **A two-level hierarchy.** `w` latent, `z ~ Normal(mu = w, …)`, `y ~ Normal(mu = z, …)`:
  the row integrates one prior, and a marginal built from `z`'s prior would still condition
  on `w`.
- **A transformed value.** `b = exp(y)` over an ancestor-parameterized `y` needs the
  pushforward of the marginal, which the row does not return.
- **A reification.** `lawof(functionof(Normal(mu = z, …)))` still refuses: the row needs a
  bare distribution constructor, and the reification wrapper is not one. A coverage gap,
  not a correctness one.

## Adding a row

1. Do the integral and state it here first, with the §08 constructor that names the answer
   and the parameter map.
2. Pick a test point by **scanning the truth-minus-wrong gap** over the variate, not by
   picking a round number. Both rows above have gaps that change sign; a point at a
   crossing proves nothing. Record what the point discriminates against.
3. Verify the truth against an independent oracle — closed form by hand, or
   Distributions.jl plus quadrature of the same integral. The determiniser emits no
   numbers, so a green golden test proves the emitted STRUCTURE only; the number is the
   engines' gate.
4. Add the refusal test for the nearest non-conjugate neighbour, so the row cannot widen
   silently.
5. **Check what happens when two variates share the row's latent.** A correct marginal is
   correct for ONE variate; summing two of them asserts independence the shared ancestor
   denies. The record path already refuses on a repeated `ImplicitMarginal::latent`, so a new
   row inherits that — but a new *caller* that assembles a product does not.
