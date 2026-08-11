# The conjugate marginal table — the maths

Sidecar to `marginal.rs`. Every row of `CONJUGATE_TABLE` is one closed-form solution of
the `kchain` marginal integral. This file states each integral, its closed form, the §08
constructor that names the answer, and the test point that pins it — so a reader can
check the maths without reverse-engineering a `build_*_marginal`.

**Two families live here.** `CONJUGATE_TABLE`'s rows each marginalize ONE variate's law.
The **shared-latent record law** below is a second family, deliberately outside the table:
it is the law of N variates jointly, so no per-variate row can express it, and no product
of per-variate rows is equal to it.

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

This is a spec fact, not an implementer's convenience. §13 *Signature: `inputs` and
`outputs`* lists what an output may be — a density, a sampled value, and "any other
**deterministic expression** over the inputs". So a row may return its answer in either of
two forms (`MarginalForm`):

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
symmetric mixture `ε`, so the marginal is the same law shifted by `μ`, and check (c) has
already proven `mu` latent-independent. **Verified at nonzero `μ` as well as at `μ = 0`** —
each row carries a second point below, its truth derived by quadrature of the row's own
mixture integral at `μ = 1.5`.

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

**Nonzero location.** Same prior, `mu = 1.5`:

| | |
|---|---|
| model | `v ~ Exponential(rate = 0.5)`; `y ~ Normal(mu = 1.5, sigma = sqrt(v))` |
| point | `y = 4.0` |
| marginal | `Laplace(1.5, 1)` |
| truth | `-3.1931471805599454` |

$-\log 2 - |4 - 1.5| = -\log 2 - 2.5$ exactly, and quadrature of the mixture integral at
`μ = 1.5` agrees. The location travels; the scale does not depend on it.

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

**Nonzero location.** Same prior, `mu = 1.5`:

| | |
|---|---|
| model | `v ~ InverseGamma(shape = 2.5, scale = 3.0)`; `y ~ Normal(mu = 1.5, sigma = sqrt(v))` |
| point | `y = 5.0` |
| truth | `-4.396997199853038` |

`mu` enters only through $z = (y - \mu)/s$, so the **tail** argument moves and the
normalizer does not: the emitted `log1p` argument goes from `4.166666666666666` to
`2.041666666666667` = $(5 - 1.5)^2/1.2/5$, while `log 1.0954451150103321`,
`log 2.23606797749979` and the log-beta are unchanged. A row that dropped `mu` would keep
`4.166666666666666`, which is what the golden asserts against.

## The shared-latent record law — one prior, N correlated fields

| | |
|---|---|
| model | `z ~ Normal(mu = μ₀, sigma = s₀)`; `yᵢ ~ Normal(mu = z, sigma = σᵢ)` for `i = 1…N` |
| query | `logdensityof(lawof(record(y₁ = y₁, …, y_N = y_N)), record(y₁ = x₁, …))` |
| latent | `z`, feeding every field's `mu`, and carried by NO field of the record |
| answer | `MvNormal(μ₀·1, Σ)` with `Σ = s₀²·J + diag(σ₁²…σ_N²)`, `J` all-ones |
| builder | `build_shared_latent_normal_logpdf` |
| recogniser | `shared_latent_record_law` |

This is the shape the record path used to refuse. Every per-field marginal is right
(`Normal(μ₀, sqrt(s₀² + σᵢ²))`, Row 1) and their product is still the wrong measure,
because the fields are correlated through `z`. `N = 1` is not this family: it is Row 1,
and it keeps lowering there.

**`σᵢ` must be latent-independent, which is not the same as constant.** A `σᵢ` that
references a SIBLING field's draw is admitted, and is correct — see *A σ over a sibling
field* below. What a `σᵢ` may not do is reference the shared latent, because then the
integral is not this one.

**Every field must be a BARE draw.** A transformed field (`b = exp(y3)`) is refused, and the
refusal has to be tested over the WHOLE record rather than per field: the record path detects
the repeated latent on the SECOND shared field, so a transform written after it is never
screened by the per-field arm. Missing that gate scored the query's value of `b` as the
untransformed draw — no inverse, no log-volume — and `exp(y3)` and `2.0·y3` emitted
identically, which is the proof the map was ignored rather than mis-applied.

**Three spellings reach this one law.** §06 *Joint composition* has a `joint` retain a
stochastic node shared between its component traces, so `joint` rewrites to the record above
rather than carrying its own closed form. `density::joint_component_coordinate` decides which
components join the rewrite and what each contributes as its coordinate:

| spelling | each component's coordinate |
|---|---|
| `lawof(record(y₁ = y₁, …))` | the field value as written |
| `joint(a = lawof(y₁), b = lawof(y₂))` | the reified value `yᵢ` |
| `joint(a = Normal(mu = z, sigma = σₐ), b = Normal(mu = z, sigma = σᵦ))` | a FRESH `draw` of the constructor |

The third is the constructor-parameter route, and the fresh draw is the whole point: §06 has
each component contribute "a fresh coordinate" while the shared node enters the composed
trace once, so `joint(m, m)` over one stochastic `m` is TWO conditionally independent draws
over ONE `z` — the correlated law, not the singular diagonal joint that two reified laws of
one draw give. A component reaching no stochastic node joins nothing and stays an independent
factor. The positional forms are the same rewrite over `cat`-sliced values.

### The integral

`z` is integrated out of the product of the conditionals (§04 *Reification to measures*
makes `lawof(record(…))` the total law of the traced sub-DAG; §04 *Kernels and `kernelof`*
identifies it with `kchain(prior, forward_kernel)`):

$$p(x_1,\dots,x_N) = \int \mathcal{N}(z;\, \mu_0,\, s_0)
\prod_{i=1}^{N} \mathcal{N}(x_i;\, z,\, \sigma_i)\; \mathrm{d}z$$

Solve it structurally rather than by integrating. Write
$y = \mu_0\mathbf{1} + s_0\xi\mathbf{1} + \varepsilon$ with $\xi \sim \mathcal{N}(0,1)$ and
$\varepsilon \sim \mathcal{N}(0, D)$, $D = \mathrm{diag}(\sigma_1^2,\dots,\sigma_N^2)$,
independent. An affine image of a Gaussian is Gaussian, so $y$ is Gaussian, and its first
two moments are read off directly:

$$\mathbb{E}[y] = \mu_0\mathbf{1}, \qquad
\mathrm{Cov}(y) = s_0^2\,\mathbf{1}\mathbf{1}^{\mathsf{T}} + D \;=\; \Sigma$$

so $\mathrm{Var}(y_i) = s_0^2 + \sigma_i^2$ — each **diagonal** entry is Row 1's marginal
variance, which is why the per-field rows are individually correct — and
$\mathrm{Cov}(y_i, y_j) = s_0^2 = \mathrm{Var}(z)$ for $i \neq j$, which is the part a
product of those rows throws away.

### Σ⁻¹ and log det Σ without matrix ops

$\Sigma$ is diagonal plus rank one, so both pieces of the Gaussian log-density are scalar
expressions. Write $d_i = 1/\sigma_i^2$, $r_i = x_i - \mu_0$, and

$$S = \sum_i d_i = \mathbf{1}^{\mathsf{T}}D^{-1}\mathbf{1}, \qquad
T = \sum_i d_i r_i = \mathbf{1}^{\mathsf{T}}D^{-1}r, \qquad
k = s_0^2 S$$

**Sherman–Morrison** on $\Sigma = D + s_0^2\,\mathbf{1}\mathbf{1}^{\mathsf{T}}$:

$$\Sigma^{-1} = D^{-1} - \frac{s_0^2\,D^{-1}\mathbf{1}\mathbf{1}^{\mathsf{T}}D^{-1}}
{1 + s_0^2\,\mathbf{1}^{\mathsf{T}}D^{-1}\mathbf{1}}
\quad\Longrightarrow\quad
r^{\mathsf{T}}\Sigma^{-1}r = \sum_i d_i r_i^2 \;-\; \frac{s_0^2\,T^2}{1 + k}$$

**The matrix determinant lemma** on the same split:

$$\det\Sigma = \det(D)\,\bigl(1 + s_0^2\,\mathbf{1}^{\mathsf{T}}D^{-1}\mathbf{1}\bigr)
\quad\Longrightarrow\quad
\log\det\Sigma = \sum_i \log \sigma_i^2 \;+\; \log(1 + k)$$

Both corrections carry the SAME $1 + k$, and both vanish at $s_0 = 0$ — where the fields
really are independent and the law collapses to the product of the conditionals.

### The emitted expression

$$\log p = -\tfrac{1}{2}\Bigl[\,N\log 2\pi \;+\; \sum_i \log \sigma_i^2
\;+\; \log(1+k) \;+\; \sum_i d_i r_i^2 \;-\; \frac{s_0^2 T^2}{1+k}\,\Bigr]$$

emitted as one flat §07 sum, in this order:

```text
vᵢ   = pow(σᵢ, 2.0)                      dᵢ = divide(1.0, vᵢ)      rᵢ = sub(xᵢ, μ₀)
S    = add-fold over dᵢ                  k  = mul(pow(s₀, 2.0), S)
T    = add-fold over mul(dᵢ, rᵢ)
quad = sub(add-fold over mul(dᵢ, pow(rᵢ, 2.0)),
           divide(mul(pow(s₀, 2.0), pow(T, 2.0)), add(1.0, k)))
out  = mul(-0.5, add-fold [ mul(N, log2π) , log(v₁) … log(v_N) , log1p(k) , quad ])
```

`log2π` is the literal `1.8378770664093453`; `N` is the field count, so `mul(N, log2π)`
const-folds and a wrong field count shows in the folded literal. Everything is §07
("Elementary functions"): `add`, `sub`, `mul`, `divide`, `pow`, `log`, `log1p`. **No matrix
op, and no §08 `MvNormal` constructor** — a constructor would force the record variate to
be converted to a vector, which is the variate-kind seam, and `is_flatpdl` admits the
expression as it stands.

`log`/`log1p` are deliberately not const-folded (`canon::fold` excludes transcendentals to
keep the det-js equivalence bit-identical), so the emitted sum keeps three legible parts
even for an all-literal model: the folded `N log 2π`, the residual `log`/`log1p` terms, and
`quad` as ONE folded literal. That literal is the Sherman–Morrison result, so it is the
strongest thing a golden can assert.

**Field order.** The row pairs `σᵢ` with `xᵢ` by field NAME — the variates come from
`match_independent_record`, which looks each field up in the query record — and emits the
sum in the record's WRITTEN order. So reordering the record's fields permutes the `log(vᵢ)`
terms in the emitted sum: the output is not byte-identical, and only the pairing is
invariant.

What IS invariant is every folded literal: `N log 2π`, the `log1p` argument `k`, and `quad`.
`quad` is where a mispairing would show, and loudly — pairing point C's σ with the wrong
fields moves it from `5.851661943957181` to `4.2324472630774075`, and point B's from
`0.747121951219512` to `2.042975609756093`. All three points below are additionally verified
to evaluate bit-identically under reversal in the emitted summation order, though that is a
property of these points, not a guarantee: a permuted floating-point sum need not be exact
for `N ≥ 3`.

### Test points

All four truths agree to full double precision along THREE independent routes: this
section's closed form, `MvNormal` in Distributions.jl, and Gauss–Kronrod quadrature
(`QuadGK`, `rtol = 1e-13`) of the mixture integral above. The quadrature is the one that
does not share this derivation's algebra.

**Point A — the pinned two-field case.** `μ₀ = 0`, `s₀ = 1`, `σ = (1, 1)`, `x = (0.5, 0.7)`.

| | |
|---|---|
| truth | `-2.5171832107434002` |
| product of the marginals | `-2.716024246969291` |
| gap | `0.199` nats |

The wrong answer is the product of two correct `Normal(0, √2)` marginals — the number the
determiniser emitted before the record path refused this shape. Folded literals:
`mul(2, log2π)` → `3.6757541328186907`, `log1p` argument `2.0`, `quad` → `0.26`.

**Point B — three fields, UNEQUAL σ.** `μ₀ = 0`, `s₀ = 1.5`, `σ = (0.5, 1, 2)`,
`x = (0.9, 1.2, 2)`.

| | |
|---|---|
| truth | `-4.405587203673088` |
| product of the marginals | `-5.424117657134536` (gap `1.019`) |
| conditional at `z = μ₀` | `-5.596815599614018` (gap `1.191`) |
| σ in reversed order | `-5.053514032941378` (gap `0.648`) |
| Sherman–Morrison term dropped | `-6.872026228063332` (gap `2.466`) |
| the `log(1+k)` det term dropped | `-3.1303765752237744` (gap `−1.275`) |

Unequal σ is what makes this point discriminate at all: with σ equal, the fields are
exchangeable and a row that permuted them would pass. The last two rows are the two ways to
half-apply the rank-one correction — apply it to the quadratic form only, or to the log-det
only — and both are caught. Folded literals: `mul(3, log2π)` → `5.513631199228036`, `log`
arguments `0.25`, `1.0`, `4.0`, `log1p` argument `11.8125`, `quad` → `0.747121951219512`.

**Gap scan.** The gap against the product of the marginals does NOT change sign along the
all-fields-together direction `x = t·(1,1,1)`: it is positive everywhere, with its minimum
`0.689` at the origin, because moving the fields together is exactly the correlation the
product misses. It DOES change sign along a spreading direction — along `t·(1,0,−1)` it
crosses zero near `t ≈ ±1.05`, and along `t·(1,0,0)` near `t ≈ ±1.3` — so a point chosen on
a spread axis can be blind. This point sits off those axes; shifting all three fields by
`±0.4` moves the gap monotonically over `0.801 … 1.309`, with no crossing nearby.

**Point C — nonzero μ₀.** `μ₀ = 1.5`, `s₀ = 0.8`, `σ = (0.7, 1.3)`, `x = (3.5, 0.5)`.

| | |
|---|---|
| truth | `-5.163204327709579` |
| μ₀ dropped (read as 0) | `-8.216098673951652` (gap `3.053`) |
| conditional at `z = μ₀` | `-6.121057028165009` (gap `0.958`) |
| product of the marginals | `-4.306423795663165` (gap `−0.857`) |
| `s₀` and `σ₁` swapped | `-4.893335214229616` (gap `−0.270`) |

Carried for the same reason Rows 4 and 5 carry a nonzero-location point: μ₀ enters only
through `rᵢ = xᵢ − μ₀`, so a row that dropped it would keep every other literal and pass
every `μ₀ = 0` point. The point is a SPREAD one (one field high, one low), which is why its
gap against the product of the marginals is negative — that is the opposite sign from
Point B's, so the two together pin the sign of the correlation term rather than just its
magnitude. All four gaps exceed `0.27` nats; shifting both fields by `±0.4` leaves the
product gap flat near `−0.85`, with no crossing. Folded literals: `log` arguments
`0.48999999999999994` and `1.6900000000000002`, `log1p` argument `1.6848206738316633`,
`quad` → `5.851661943957181`.

**Point D — the constructor-joint spelling.** `μ₀ = 0.5`, `s₀ = 2`, `σ = (0.6, 0.8)`,
`x = (2.5, −1.0)`, so `Σ = [[4.36, 4], [4, 4.64]]`.

| | |
|---|---|
| truth | `-8.748747354129808` |
| product of the marginals | `-4.0426427710908985` (gap `−4.706`) |
| conditional at `z = μ₀` | `-8.417275946884702` (gap `−0.331`) |
| σ paired to the wrong fields | `-8.690833208895306` (gap `−0.058`) |
| μ₀ dropped (read as 0) | `-8.86575756593011` (gap `0.117`) |
| Sherman–Morrison term dropped | `-9.872393397582488` (gap `1.124`) |
| the `log(1+k)` det term dropped | `-7.293629903432019` (gap `−1.455`) |

Carried for the constructor-parameter spelling, where the coordinates are fresh draws the
model never named — so it is the point that proves the rewrite integrates `z` ONCE over two
coordinates rather than reusing one. A SPREAD point (one field well above μ₀, one below)
under strong correlation (`s₀` over three times either `σᵢ`), which is what makes the product
gap `4.7` nats rather than Point A's `0.2`. `s₀ ≫ σᵢ` is also the regime where the two
half-applied corrections separate most, and both are caught above. Folded literals:
`mul(2, log2π)` → `3.6757541328186907`, `log` arguments `0.36` and `0.6400000000000001`,
`log1p` argument `17.36111111111111`, `quad` → `12.379444024205748`.

### A σ over a sibling field — admitted, and why the Σ reading stops applying

```flatppl
z  = draw(Normal(mu = 0.0, sigma = 1.0))
y1 = draw(Normal(mu = z, sigma = 1.0))
y2 = draw(Normal(mu = z, sigma = y1))
```

`σ₂` is the sibling field `y1`, which is latent-independent, so the recogniser admits it and
emits `σ₂² = 0.25` at the query point `(0.5, 0.7)` — the sibling's own query value, pinned by
the chain rule the record path already applies to sibling draws (that is what `exempt` /
`siblings` is for). Truth `-2.2381096204634274`, verified by quadrature of
`∫ φ(z) N(0.5; z, 1) N(0.7; z, 0.5) dz`; emitted `log` arguments `1.0` and `0.25`, `log1p`
argument `5.0`, `quad` `0.39500000000000024`.

**The value is right and the Gaussian READING is not.** With `σ₂ = y1` the fields are not
conditionally independent given `z`, so this model is not jointly Gaussian and `Σ` is not its
covariance. What still holds is the only thing the emission needs: at a FIXED query point
`σ₂` is the constant `x₁`, so

$$\prod_i \mathcal{N}(x_i;\, z,\, \sigma_i)
= p(x_1 \mid z)\; p(x_2 \mid z, y_1 = x_1)$$

by the chain rule, and integrating that against the prior is the joint density at that point.
So the expression evaluates correctly while the `MvNormal(μ₀·1, Σ)` sentence at the top of
this section does not describe this model. Do not read Σ back out of it.

Support is the query's problem, not the lowering's: a query putting `y1 ≤ 0` asks for a
`Normal` with a non-positive scale, exactly as any model with a data-dependent `sigma` does.

### What keeps refusing

The family is deliberately narrow: N fields, each a BARE `draw` of `Normal(mu = z, …)`
directly referencing ONE shared latent whose own prior is `Normal` and ancestor-free. Each
of these keeps the shared-latent refusal, and each is pinned by a test. All three spellings
inherit the same list — widening which spellings REACH the law never widens what it answers,
and `a_constructor_joint_outside_the_record_law_refuses` pins the constructor route's rows
against the `lawof` route's:

- **A scale latent.** `yᵢ ~ Normal(mu = 1, sigma = z)`. Not a Gaussian marginal at all, so
  no rank-one Σ exists.
- **A non-Normal shared prior.** `z ~ Exponential(…)` feeding N Normal means. The mixture
  is Gaussian only conditionally; the joint is not.
- **A non-Normal field family.** `yᵢ ~ Poisson(rate = z)` — the fields are correlated, and
  the joint is not in this closed form.
- **Two shared latents.** `yᵢ ~ Normal(mu = add(z, w), …)`. Σ is then rank TWO plus
  diagonal, and neither Sherman–Morrison as written nor the one-prior integral covers it.
- **A DERIVED field mean.** `yᵢ ~ Normal(mu = mul(2.0, z), …)` has a closed-form joint that
  is NOT this one (the loadings differ per field), so the exact-ref check refuses it, the
  same way Row 1's does.
- **A transformed field**, in ANY position. `b = exp(y1)` needs the pushforward of the JOINT,
  which this row does not return. Gated over the whole record at the call site AND re-checked
  in the emitter, because the per-field screen only sees fields reached before the repeat.
- **Mixed shared and unshared fields.** Every field must integrate the same latent. A
  record where two fields share `z` and a third integrates `w` would need the product of
  this row with `w`'s own — correct in principle, and outside the decided scope.

`iid` over the same model still emits the PRODUCT: it redraws its reified sub-DAG afresh
per copy and never shares ancestors (§06 "iid" entry). `joint` no longer does — §06 "Reified
components share their ancestry" makes `joint(a = lawof(y1), b = lawof(y2))` equivalent to
`lawof(record(a = y1, b = y2))`, so a keyword `joint` over two or more reified components
now reaches this SAME law (`crates/determinizer/src/density.rs`, `lower_keyword_joint`'s
record-law dispatch). The positional spelling reaches the cat-law counterpart the same way.

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
  the rest: it is the only place where every per-row answer is right and the assembled
  result is still wrong, so the row cannot catch it. `conjugate_marginal_measure` therefore
  returns the latent it integrated (`ImplicitMarginal::latent`), and the record path
  compares them across fields. Two fields over DIFFERENT latents is a genuine product and
  lowers.

  Where the shape is the shared-latent record law above — every field a bare
  `Normal(mu = z, …)` draw over one ancestor-free `Normal` prior — the record path now emits
  that law's closed form instead of refusing. Any other shared-latent shape still refuses;
  the list is in that section's *What keeps refusing*.

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
   denies. The record path compares each field's `ImplicitMarginal::latent` and refuses on a
   repeat, so a new row inherits that — but a new *caller* that assembles a product does
   not. The one exception is the shared-latent record law, which answers the repeat with the
   joint closed form; it recognises its own Normal-on-Normal-mean shape only, so a new row
   inherits the refusal, never that law.
