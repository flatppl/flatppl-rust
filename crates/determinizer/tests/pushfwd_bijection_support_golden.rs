//! Threading the base measure's refined SUPPORT (a `ValueSet`) into the pushfwd
//! change-of-variables positivity guard (§06 case 1; determiniser lowering-gap
//! Task 5). `log` / `pow` bijections are defined only where the base is a.e.
//! positive. The guard used to read the COARSE structural type of the base
//! variate (`scalar real` → natural extent `reals`), so it conservatively
//! REFUSED every scalar-real base — even a genuinely positive-support one
//! (`Gamma`, `Exponential`, a positive `interval`). It now reads the base's
//! inferred support, so a positive-support base lowers while a real-support base
//! (and a discrete atom-at-0 / unconstrained one) still refuses.
//!
//! Structural only (flatppl-rust is not a density engine): assert the emitted
//! change-of-variables FlatPIR (an `Ok` whose outer op is `sub(logdensityof(M,
//! f_inv(v)), logvol(f_inv(v)))`) vs a clean refuse.
use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}
fn pir(src: &str) -> String {
    flatppl_flatpir::write(&determinize(&parse_infer(src)).expect("must lower"))
}

#[test]
fn pushfwd_log_over_positive_support_lowers() {
    // `Exponential` has POSITIVE (a.e.) support. Its inferred support is
    // `nonnegreals` (`Exponential` ≡ `Gamma(shape = 1)`); the base is continuous,
    // so the sole boundary point 0 carries no probability mass and `exp` maps
    // −∞ ↦ 0. `log` is thus defined a.e. on the support and the pushforward keeps
    // full mass — `pushfwd(log, Exponential)` lowers. The old guard read the
    // coarse structural type (`scalar real` → `reals`) and refused; threading the
    // base's refined support fixes that.
    let p = pir("d = pushfwd(fn(log(_)), Exponential(1.0))\nlp = logdensityof(d, 0.5)");
    // Change-of-variables: sub(logdensityof(M, exp(v)), logvol(exp(v))). The
    // inner density recurses into the base (`builtin_logdensityof`); the inverse
    // of `log` is `exp`.
    assert!(
        p.contains("builtin_logdensityof") && p.contains("(sub ") && p.contains("(exp "),
        "expected change-of-variables sub(logdensityof(...), ...) with exp inverse, got:\n{p}"
    );
}

#[test]
fn pushfwd_log_over_gamma_lowers_like_exponential() {
    // `Gamma` and `Exponential` both have inferred support `nonnegreals`
    // (`Exponential` ≡ `Gamma(shape = 1)`), so the two MUST lower alike: both
    // are continuous a.e.-positive bases. This pins that the guard accepts the
    // continuous-nonneg `nonnegreals` case regardless of which distribution
    // produced it.
    let p =
        pir("d = pushfwd(fn(log(_)), Gamma(shape = 2.0, rate = 1.0))\nlp = logdensityof(d, 0.5)");
    assert!(
        p.contains("builtin_logdensityof") && p.contains("(sub ") && p.contains("(exp "),
        "Gamma (nonnegreals) must lower like Exponential (nonnegreals):\n{p}"
    );
}

#[test]
fn pushfwd_log_over_positive_interval_lowers() {
    // A strictly-positive `interval(2, 5)` base support (lo = 2 > 0): `log` is
    // defined on the whole support, so the pushforward lowers. Exercises the
    // `Interval` arm of the support guard.
    let p = pir("d = pushfwd(fn(log(_)), Uniform(interval(2.0, 5.0)))\nlp = logdensityof(d, 3.0)");
    assert!(
        p.contains("builtin_logdensityof") && p.contains("(sub ") && p.contains("(exp "),
        "positive-interval base must lower:\n{p}"
    );
}

#[test]
fn pushfwd_log_over_real_support_still_refuses() {
    // `Normal`'s support is all of ℝ (`reals`): `log x` is undefined on x ≤ 0,
    // which carries HALF the probability mass — lowering would synthesize a
    // silently SUB-probability measure (integrates to ~0.5, not 1). This MUST
    // refuse. Widening the guard to accept `reals` would (wrongly) make it lower,
    // so this is the guard's core safety property.
    let e = determinize(&parse_infer(
        "d = pushfwd(fn(log(_)), Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    ))
    .expect_err("pushfwd(log, Normal) over a real-support base must refuse");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}

#[test]
fn pushfwd_pow_over_positive_support_lowers() {
    // `pow` with a literal exponent over a positive-support base: f_inv =
    // pow(_, 1/k), logvol = log|k| + (k−1)·log x — defined on a positive support.
    // `Gamma` (`nonnegreals`) now lowers (was conservatively refused by the
    // coarse structural type).
    let p = pir(
        "d = pushfwd(fn(pow(_, 2.0)), Gamma(shape = 2.0, rate = 1.0))\nlp = logdensityof(d, 0.5)",
    );
    assert!(
        p.contains("builtin_logdensityof") && p.contains("(sub ") && p.contains("(pow "),
        "pow over a positive-support base must lower:\n{p}"
    );
}

#[test]
fn pushfwd_log_over_discrete_atom_at_zero_refuses() {
    // `Poisson`'s support is `nonnegintegers` — a DISCRETE base with a
    // positive-mass atom at 0, where `log 0 = −∞`. Unlike a CONTINUOUS
    // `nonnegreals` base (where 0 is measure-zero), this must REFUSE: the guard is
    // not a naive `subset_of(nonnegreals)` (which would admit it) — it excludes
    // discrete atom-at-0 supports. Conservative refuse-don't-mislower.
    let e = determinize(&parse_infer(
        "d = pushfwd(fn(log(_)), Poisson(rate = 1.0))\nlp = logdensityof(d, 0.5)",
    ))
    .expect_err("a discrete base with a positive-mass atom at 0 must refuse");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}

#[test]
fn pushfwd_log_over_discrete_positive_integer_support_refuses() {
    // `Categorical`'s support is `posintegers` — strictly positive but DISCRETE
    // (1-indexed atoms). Regression guard: the guard's accept arm once listed
    // `PosIntegers` alongside the continuous strictly-positive sets, so this
    // silently LOWERED with a spurious change-of-variables Jacobian term (the
    // emitted density at atom `y = log k` picked up a bogus `+log k`, right only
    // at k = 1). A discrete measure has no Jacobian: `log`/`pow` pushforward of
    // it must refuse, not synthesize a density. Distinct from the
    // `nonnegintegers` (`Poisson`) case above, which already refused via the
    // fallback arm and did not exercise the `PosIntegers` leak.
    let e = determinize(&parse_infer(
        "d = pushfwd(fn(log(_)), Categorical(p = [0.2, 0.3, 0.5]))\nlp = logdensityof(d, 0.5)",
    ))
    .expect_err("a discrete positive-integer support must refuse (no Jacobian)");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}

#[test]
fn pushfwd_log_over_unconstrained_support_refuses() {
    // `Uniform(anything)` has support `anything` — not PROVABLY positive. The
    // fallback conservatism: a support that is not proven ⊆ the positive region
    // (here `anything`; likewise a `None`/`%unknown` support that inference did
    // not track finely) must refuse, NOT default to positive.
    let e = determinize(&parse_infer(
        "d = pushfwd(fn(log(_)), Uniform(anything))\nlp = logdensityof(d, 3.0)",
    ))
    .expect_err("an unconstrained (not-provably-positive) support must refuse");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}
