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

The row supplies $p_{M'}$. It is not general integration: a row matches only when the two
constructors line up exactly, the latent reaches exactly the conjugating parameter along
the path the row expects, and every other likelihood parameter is latent-independent.

## A row need not name a §08 distribution

Determinised output is a deterministic expression, so a row may return its answer in
either of two forms (`MarginalForm`):

- **`Measure`** — a §08 distribution-constructor node, scored by the ordinary density path
  into one `builtin_logdensityof`. Rows 1, 2 and 4.
- **`LogDensity`** — the closed-form log-density itself, built from §07 builtins at the
  variate. Rows 3 and 5, because §08 names no `BetaBinomial` and its `StudentT(nu)` is the
  standard form only (its location-scale form is a `pushfwd`, not a bare constructor).

Both rows' expressions use `loggamma`, a §07 builtin ("Elementary functions", domain
`posreals`) — **not** a §09 standard-module member, so no `load_module` is involved. That
distinction is what separates them from the deferred row at the end of this file.

Nothing about a row being closed-form requires a constructor to exist. Rows 3 and 5 were
once recorded as blocked on a §08 addition on that false premise; do not reintroduce it.

## The latent's path to the conjugating parameter

§08 parameterizes each distribution by the quantity it names, and that is not always the
quantity a conjugacy is stated in. `Normal` takes `sigma`, so a prior on the **variance**
reaches the conjugating parameter through a `sqrt`: `Normal(mu = 0, sigma = sqrt(v))`. Each
row therefore records a `LatentPath` — `Direct` (the parameter's value IS the latent's ref)
or `Sqrt` — and a row matches only along its own path.

**Why the row must record it.** A location mixture and a scale mixture over the same base
agree closely. For Row 4's own prior, the location mixture `y = m + ε` with
`m ~ Exponential(rate = 1)`, `ε ~ Normal(0, 1)` has log-density `-1.1759117615936188` at
`y = 0.5`, while Row 4's scale mixture `Laplace(0, 1)` gives `-1.1931471805599454` — a gap
of **0.017 nats**. A row that accepted a bare ref where it wanted `sqrt` would score one as
the other, and no plausible test point near the origin would notice.

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

## Row 3 — Beta prior on a Binomial `p`

| | |
|---|---|
| model | `p ~ Beta(alpha = α, beta = β)`; `k ~ Binomial(n = n, p = p)` |
| latent | `p`, feeding the likelihood's `p`, `LatentPath::Direct` |
| §08 answer | none — §08 names no `BetaBinomial`; the row emits the log-pmf |
| builder | `build_beta_binomial_marginal` / `build_beta_binomial_logpmf` |

The integral, with §08's `Binomial(n, p)` pmf $\binom{n}{k}p^k(1-p)^{n-k}$ and `Beta(alpha,
beta)` density $\frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)}$:

$$\int_0^1 \binom{n}{k} p^k (1-p)^{n-k}
\frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)}\, \mathrm{d}p
= \binom{n}{k} \frac{1}{B(\alpha,\beta)} \int_0^1 p^{k+\alpha-1}(1-p)^{n-k+\beta-1}\,
\mathrm{d}p
= \binom{n}{k}\frac{B(k+\alpha,\, n-k+\beta)}{B(\alpha,\beta)}$$

— the beta-binomial pmf, the integral being the definition of the beta function. In log
space, with `loggamma` the §07 builtin and $\log B(x,y) = \log\Gamma(x) + \log\Gamma(y) -
\log\Gamma(x+y)$:

```text
log C(n, k) = loggamma(n+1) − loggamma(k+1) − loggamma(n−k+1)
logpmf      = log C(n, k) + log B(k+α, n−k+β) − log B(α, β)
```

**Parameter map, re-derived against §08's own parameterisation.** §08's `Beta` takes two
SHAPES `alpha`/`beta` (not a mean and a concentration), and §08's `Binomial` takes `n` and a
`p` that is a **probability** (not a rate or a logit). So α, β come from the prior verbatim
and `n` from the likelihood verbatim — no reparameterisation. `n` is read from the
likelihood because the trial count is the likelihood's, and check (c) has already proven it
latent-independent.

### Test point

| | |
|---|---|
| model | `p ~ Beta(2, 3)`; `k ~ Binomial(n = 10, p = p)` |
| point | `k = 7` |
| marginal | `BetaBinomial(10, 2, 3)` |
| truth | `-2.526728144641337` |
| wrong answer | `-3.1590202516350088` |
| gap | `0.632` nats |

The wrong answer is the plug-in `Binomial(10, 0.4)` at `k = 7` — the conditional at the
prior MEAN $\alpha/(\alpha+\beta) = 0.4$. The two share a mean, so only the dispersion tells
them apart, exactly as in Row 2.

**What the point discriminates against.** The gap crosses zero twice over `k`, between
`k = 2` and `k = 3` and between `k = 6` and `k = 7`: the beta-binomial is overdispersed, so
it loses mass in the centre and gains it in both tails. `k = 7` is the first integer above
the upper crossing, where the gap is already `0.632`; the mirror point is the lower
extremum `k = 4` (gap `-0.584`). A test at `k = 6` (gap `-0.061`) would be nearly blind.

Structurally the emitted expression pins the map by itself: every `loggamma` argument
const-folds, so the row shows as `loggamma` at `11.0, 8.0, 4.0` (the coefficient) and
`9.0, 6.0, 15.0` against `2.0, 3.0, 5.0` (the posterior-over-prior beta ratio). A plug-in
would name `Binomial` and carry a `0.4`, and neither survives.

## Rows 4 and 5 — a prior on the VARIANCE, reached through `sqrt`

Both are Gaussian **scale** mixtures: `y | v ~ Normal(mu = μ, sigma = sqrt(v))` with a prior
on `v`. Both therefore use `LatentPath::Sqrt`, and the 0.017-nat location/scale
near-agreement above is why that path is recorded rather than assumed.

Both rows pass the likelihood's `mu` through as the marginal's location. `y = μ + s·ε` for a
symmetric mixture `ε`, so the marginal is the same law shifted by `μ` — an elementary
location shift, and check (c) has already proven `mu` latent-independent. The test points
below all sit at `μ = 0`, so the **numeric** verification covers `μ = 0` only; the shift
itself is the derivation just given, not a checked number.

### Row 4 — Exponential prior on the variance → Laplace

| | |
|---|---|
| model | `v ~ Exponential(rate = λ)`; `y ~ Normal(mu = μ, sigma = sqrt(v))` |
| latent | `v`, feeding the likelihood's `sigma` under a `sqrt` |
| §08 answer | `Laplace(location = μ, scale = 1/sqrt(2λ))` |
| builder | `build_exponential_variance_marginal` |

With §08's `Exponential(rate)` density $\lambda e^{-\lambda v}$, `Normal(mu, sigma)` density
$\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ and `Laplace(location, scale)` density
$\frac{1}{2b}e^{-|x-\mu|/b}$, at $\mu = 0$:

$$\int_0^\infty \frac{1}{\sqrt{2\pi v}}e^{-y^2/2v}\, \lambda e^{-\lambda v}\, \mathrm{d}v
= \frac{\lambda}{\sqrt{2\pi}} \int_0^\infty v^{-1/2}
e^{-\left(\frac{y^2}{2v} + \lambda v\right)}\, \mathrm{d}v
= \sqrt{\frac{\lambda}{2}}\; e^{-\sqrt{2\lambda}\,|y|}$$

(the inner integral is the standard $\int_0^\infty v^{-1/2}e^{-a/v - cv}\mathrm{d}v =
\sqrt{\pi/c}\,e^{-2\sqrt{ac}}$ with $a = y^2/2$, $c = \lambda$). That is
$\frac{1}{2b}e^{-|y|/b}$ with $b = 1/\sqrt{2\lambda}$ — `Laplace(0, b)` exactly.

**Parameter map, re-derived against §08's own parameterisation.** §08 parameterizes
`Exponential` by **rate** $\lambda$ (`rate = elementof(posreals)`: "the decay rate"), not by
mean or scale; and `Laplace` by **scale** $b$, not by rate. So the map is
$b = 1/\sqrt{2\lambda}$, emitted `divide(1.0, sqrt(mul(2.0, λ)))`. Stated the other way, a
prior of MEAN $2b^2$ is `rate = 1/(2b^2)`, and the map inverts that. A row that passed
$\lambda$ through, or read it as a scale, would give a different constant.

| | |
|---|---|
| model | `v ~ Exponential(rate = 0.5)` (mean 2, so `b = 1`); `y ~ Normal(0, sqrt(v))` |
| point | `y = 4.0` |
| marginal | `Laplace(0, 1)`, emitted `scale` folds to `1.0` |
| truth | `-4.693147180559945` |
| wrong answer | `-5.265512123484645` |
| gap | `0.572` nats |

The wrong answer is the plug-in `Normal(0, sqrt 2)` — the likelihood at the prior MEAN
variance $1/\lambda = 2$.

**Do not move this test point without re-scanning the gap.** The gap is only `0.135` at
`y = 0.5`, and it changes sign twice over `y ∈ [0.5, 4.0]` (`-0.178`, `-0.428`, `-0.178` at
`y` = 1, 2, 3), so it crosses zero near `y ≈ 1` and `y ≈ 3` and a test there is blind by
construction. Worse, at `y = 0.5` the wrong answer is `-1.3280121234846454`, numerically
identical to **Row 1's truth** — so a row mix-up at that point would look correct.

Structurally `scale = 1.0` alone cannot tell $1/\sqrt{2\lambda}$ from a $\lambda$ passed
through, since both are 1 at $\lambda = 0.5$. A second shape with `rate = 0.125` → `scale
2.0` pins the arithmetic; it asserts structure only and claims no density number.

### Row 5 — InverseGamma prior on the variance → scaled Student t

| | |
|---|---|
| model | `v ~ InverseGamma(shape = α, scale = β)`; `y ~ Normal(mu = μ, sigma = sqrt(v))` |
| latent | `v`, feeding the likelihood's `sigma` under a `sqrt` |
| §08 answer | none as a constructor — the location-scale t is a `pushfwd`; the row emits the log-density |
| builder | `build_inverse_gamma_variance_marginal` / `build_scaled_t_logpdf` |

With §08's `InverseGamma(shape, scale)` density
$\frac{\beta^\alpha}{\Gamma(\alpha)}v^{-\alpha-1}e^{-\beta/v}$, at $\mu = 0$:

$$\int_0^\infty \frac{1}{\sqrt{2\pi v}}e^{-y^2/2v}\,
\frac{\beta^\alpha}{\Gamma(\alpha)} v^{-\alpha-1} e^{-\beta/v}\, \mathrm{d}v
= \frac{\beta^\alpha}{\Gamma(\alpha)\sqrt{2\pi}} \int_0^\infty
v^{-\alpha-\frac{3}{2}} e^{-\left(\beta + \frac{y^2}{2}\right)/v}\, \mathrm{d}v
= \frac{\Gamma\!\left(\alpha+\frac12\right)\beta^\alpha}
{\Gamma(\alpha)\sqrt{2\pi}\left(\beta + \frac{y^2}{2}\right)^{\alpha + 1/2}}$$

which regroups to the Student t with $\nu = 2\alpha$ scaled by $s = \sqrt{\beta/\alpha}$:
substituting $\beta = \alpha s^2$ and $\nu = 2\alpha$ gives
$\frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)
\sqrt{\nu\pi}\, s}\left(1 + \frac{(y/s)^2}{\nu}\right)^{-(\nu+1)/2}$, i.e. §08's
`StudentT(nu)` density in $y/s$ divided by $s$ — the density of
`pushfwd(fn(mu + s * _), StudentT(nu))`.

The emitted form writes the normalizer with a log-beta rather than the gamma ratio and
$\log(\nu\pi)/2$, because $B(\nu/2, 1/2) = \Gamma(\nu/2)\Gamma(1/2)/\Gamma((\nu+1)/2)$
absorbs the $\Gamma(1/2) = \sqrt{\pi}$ — so no `pi` constant is needed and
`build_logbeta` is reused:

```text
z    = (y − μ)/s,   s = sqrt(β/α),   ν = 2α
logZ = log(s) + log(sqrt(ν)) + log B(ν/2, 1/2)
out  = −[ logZ + ((ν+1)/2) · log1p(z²/ν) ]
```

**Parameter map, re-derived against §08's own parameterisation.** §08 calls
`InverseGamma`'s second parameter `scale`, but it is **not** a multiplicative scale: §08
states "The `scale` parameter of `InverseGamma` plays the same numerical role as the `rate`
parameter of `Gamma`", and the density has it as the $\beta$ in $e^{-\beta/v}$. So the map
reads `scale` as that $\beta$, giving $s = \sqrt{\beta/\alpha}$ and $\nu = 2\alpha$. Reading
it as a multiplicative scale would put $\beta$ in the wrong place and fold to a different
constant.

| | |
|---|---|
| model | `v ~ InverseGamma(shape = 2.5, scale = 3.0)`; `y ~ Normal(0, sqrt(v))` |
| point | `y = 5.0` |
| marginal | location 0, `s = sqrt(3/2.5) = 1.0954451150103321`, `ν = 5` |
| truth | `-5.986463573222975` |
| wrong answer | `-7.515512123484645` |
| gap | `1.529` nats |

The wrong answer is the plug-in `Normal(0, sqrt(β/(α−1)))` — the likelihood at the prior
MEAN variance $\beta/(\alpha-1) = 2$.

**What the point discriminates against.** The t is heavier-tailed than the Gaussian with the
same central scale, so the gap is positive near the origin (`0.146` at `y = 0.5`), negative
through the shoulder (`-0.007`, `-0.152`, `-0.327`, `-0.293` at `y` = 1, 1.4, 2, 3) and
strongly positive in the tail. It therefore crosses zero just above `y = 1` and again
between `y = 3` and `y = 5`. The point was moved from `y = 1.4`, where the gap is only
`0.152`, out to `y = 5.0` where it is `1.529`.

Structurally the folded literals pin the map: `log 1.0954451150103321` is $\log s$ with $s$
read as $\sqrt{\beta/\alpha}$, `log 2.23606797749979` is $\log\sqrt{\nu}$ at $\nu = 2\alpha
= 5$, `loggamma` at `2.5`/`0.5`/`3.0` is $\log B(\nu/2, 1/2)$, and `mul 3.0` with
`log1p 4.166666666666666` is $((\nu+1)/2)\log(1 + z^2/\nu)$.

## Deferred by decision — Exponential prior on the MEAN

`v ~ Exponential(rate = λ)`; `y ~ Normal(mu = v, sigma = σ)` is a **location** mixture, whose
marginal is the exponentially modified Gaussian — verified `-1.1759117615936188` at
`y = 0.5` with `λ = 1`, `σ = 1`, against quadrature of its own integral. It is not in the
table, and it must not be added.

Its density needs `erfc`, which is a **§09 standard-module member**
(`09-standard-modules.md`, `special-functions`), not a §07 builtin. The determiniser emits
no §09 module call anywhere, and such a call would have to carry a `load_module` whose name
**and version** match the catalogue in *both* engines. **Module loading in determinised
output is deferred pending consortium discussion** — a settled decision, not an open
question.

§13 gives nothing to appeal to: it describes the target only as "a deterministic DAG", never
uses the name "FlatPDL", and states that determinization "is preliminary and subject to
change. It is not part of FlatPPL semantic versioning yet." So the admissible vocabulary of
determinised output is defined solely by `is_flatpdl`, and widening it would set profile
policy by accident.

Do not reimplement `erfc` from §07 builtins to route around this. If any other row you
consider needs a §09 member, stop and report it rather than reaching for the same
workaround. This row is recorded here so a later reader does not rediscover it and assume
nobody had checked.

The refusal test for it is `a_latent_with_no_conjugate_row_still_refuses`'s
"exponential prior on a location" case, which is also Row 4's nearest neighbour.

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
