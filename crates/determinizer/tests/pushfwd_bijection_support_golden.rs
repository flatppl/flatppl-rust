//! Threading the base measure's refined SUPPORT (a `ValueSet`) into the pushfwd
//! change-of-variables domain guard (§06 case 1). A domain-restricted forward is admitted
//! only where the base's support provably lies inside its domain. Reading the COARSE
//! structural type instead (`scalar real` → `reals`) refused every scalar-real base,
//! including positive-support ones (`Gamma`, `Exponential`, a positive `interval`). An
//! out-of-domain base (real support under `log`, an atom where the forward is ±inf) and an
//! unconstrained one still refuse.
//!
//! Structural only (flatppl-rust is not a density engine): assert the emitted
//! change-of-variables FlatPIR (an `Ok` whose outer op is `sub(logdensityof(M,
//! f_inv(v)), logvol(f_inv(v)))` over a continuous base) vs a clean refuse.
use flatppl_determinizer::determinize;

mod common;
use common::{call_arg, pir_head};

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}
fn pir(src: &str) -> String {
    flatppl_flatpir::write(&determinize(&parse_infer(src)).expect("must lower"))
}

/// The density inside a gate — the `ifelse`'s taken arm. Over a discrete base the
/// lattice gate's own condition subtracts, so a bare `(sub` search over the emission no
/// longer isolates a volume term.
fn gated_density(out: &str) -> String {
    call_arg(out, "ifelse", 1)
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
    assert_eq!(e.construct, "log", "got: {e:?}");
    assert!(e.reason.contains("positive reals"), "got: {e:?}");
}

#[test]
fn pushfwd_pow_over_positive_support_lowers() {
    // `pow` with a literal exponent over a continuous base inside its §06 domain
    // `nonnegreals`: f_inv = pow(_, 1/k), logvol = log|k| + (k−1)·log x. `Gamma`
    // (`nonnegreals`) now lowers (was conservatively refused by the coarse
    // structural type).
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
    // `Poisson`'s support is `nonnegintegers`, which is NOT inside `log`'s domain
    // `posreals`: 0 is outside it and here carries positive mass (`log 0 = −∞`, so
    // that atom maps nowhere and its mass is lost). Under a CONTINUOUS
    // `nonnegreals` base the same 0 is a measure-zero boundary and lowers, so the
    // guard cannot be a plain `subset_of` on the closure — it reads the endpoint
    // by measure class.
    let e = determinize(&parse_infer(
        "d = pushfwd(fn(log(_)), Poisson(rate = 1.0))\nlp = logdensityof(d, 0.5)",
    ))
    .expect_err("a discrete base with a positive-mass atom at 0 must refuse");
    assert_eq!(e.construct, "log", "got: {e:?}");
    assert!(e.reason.contains("positive reals"), "got: {e:?}");
}

#[test]
fn pushfwd_log_over_discrete_positive_integer_support_lowers_without_a_jacobian() {
    // `Categorical`'s support is `posintegers` — DISCRETE (1-indexed atoms) and
    // strictly inside `log`'s domain `posreals`, where `log` is injective. So the
    // pushforward is well-defined and §06 case 1 requires the density: `pmf(k)` at
    // the atom `y = log k`, i.e. the base scored at `exp(y)`.
    //
    // Regression guard: the accept arm once listed `PosIntegers` alongside the continuous
    // sets while the volume element was applied unconditionally, so the density at atom
    // `y = log k` picked up a bogus `+log k` (right only at k = 1). The volume element is
    // dropped over a counting reference (§06 "Density convention"), so this asserts the
    // density is emitted AND that no change-of-variables term rides along.
    let p =
        pir("d = pushfwd(fn(log(_)), Categorical(p = [0.2, 0.3, 0.5]))\nlp = logdensityof(d, 0.5)");
    assert!(
        p.contains("builtin_logdensityof Categorical") && p.contains("(exp 0.5)"),
        "the base pmf is scored at the preimage exp(y):\n{p}"
    );
    assert_eq!(
        pir_head(&gated_density(&p)),
        "builtin_logdensityof",
        "a discrete base carries no volume element — no `+log k`:\n{p}"
    );
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
    assert_eq!(e.construct, "log", "got: {e:?}");
    assert!(e.reason.contains("positive reals"), "got: {e:?}");
}

#[test]
fn pushfwd_sqrt_over_discrete_nonnegative_support_lowers() {
    // §06 case 1 puts `sqrt` (and `pow`) on `nonnegreals`, not `posreals`:
    // `Poisson`'s `nonnegintegers` lies inside it and `sqrt` is injective there,
    // so the pushforward is well-defined and must lower. It previously refused
    // with "requires M's support to lie within the positive reals" — an
    // over-refusal, `sqrt`'s domain having been modelled as `log`'s.
    //
    // The inverse is `pow(y, 1/k)` at k = ½, i.e. `y²`, read at the image gate's
    // sanitised point (so it does not const-fold to 0.25); no volume element over a
    // counting reference.
    let p = pir("d = pushfwd(sqrt, Poisson(rate = 3.0))\nlp = logdensityof(d, 0.5)");
    assert!(
        p.contains("builtin_logdensityof Poisson") && p.contains("(pow ") && p.contains(" 2.0)"),
        "the base pmf is scored at the preimage y²:\n{p}"
    );
    assert_eq!(
        pir_head(&gated_density(&p)),
        "builtin_logdensityof",
        "no volume element over an atom:\n{p}"
    );
}

#[test]
fn pushfwd_pow_over_discrete_nonnegative_support_lowers() {
    // The same `nonnegreals` domain reached through `pow`'s own derivation rather
    // than the registry's `sqrt` entry (§06: "`pow` with literal exponent (of
    // which `sqrt` = `pow(_, 1/2)` is a case)"), so both spellings are pinned.
    let p = pir("d = pushfwd(fn(pow(_, 2.0)), Poisson(rate = 3.0))\nlp = logdensityof(d, 0.5)");
    assert!(
        p.contains("builtin_logdensityof Poisson") && p.contains("(pow ") && p.contains(" 0.5)"),
        "the base pmf is scored at the preimage y^(1/2):\n{p}"
    );
    assert_eq!(
        pir_head(&gated_density(&p)),
        "builtin_logdensityof",
        "no volume element over an atom:\n{p}"
    );
}

#[test]
fn pushfwd_sqrt_over_continuous_nonnegative_support_lowers_with_its_volume_term() {
    // A CONTINUOUS base on `nonnegreals` — `interval(0, 5)`, whose lower endpoint
    // is `sqrt`'s domain boundary. It lies inside `nonnegreals`, so it lowers, and
    // being Lebesgue it keeps its volume term `log|k| + (k−1)·log x` at k = ½. The
    // regression half of the domain widening: it must not stop applying.
    let p = pir("d = pushfwd(sqrt, Uniform(interval(0.0, 5.0)))\nlp = logdensityof(d, 1.0)");
    assert!(
        p.contains("builtin_logdensityof Uniform") && p.contains("(sub "),
        "a continuous base keeps the change of variables:\n{p}"
    );
    assert!(
        p.contains("(abs 0.5)") && p.contains("-0.5"),
        "the `pow` log-volume log|½| + (½−1)·log x is emitted:\n{p}"
    );
}

#[test]
fn pushfwd_log_domain_is_not_widened_to_nonnegative() {
    // §06 keeps `log`/`log10` on `posreals`. Widening `sqrt`'s domain must not
    // reach them: a base whose support includes 0 as an ATOM (`nonnegintegers`) or
    // negatives (`reals`, `integers`) still refuses, since `log` maps neither.
    for src in [
        "d = pushfwd(log, Poisson(rate = 1.0))\nlp = logdensityof(d, 0.5)",
        "d = pushfwd(log, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
        "d = pushfwd(log10, Poisson(rate = 1.0))\nlp = logdensityof(d, 0.5)",
        "d = pushfwd(log10, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    ] {
        let e = determinize(&parse_infer(src)).expect_err("log's domain stays posreals");
        assert!(
            e.reason.contains("the positive reals"),
            "must refuse against the positive reals, got: {}",
            e.reason
        );
    }
}

#[test]
fn pushfwd_logit_over_a_boolean_base_refuses() {
    // `booleans` is exactly {0, 1} — both atoms, and both the points where `logit`
    // is ±inf, so no discrete support lies inside `interval(0, 1)`. Admitting
    // discrete supports for the `nonnegreals` domain must not leak here.
    let e = determinize(&parse_infer(
        "d = pushfwd(fn(logit(_)), Bernoulli(p = 0.3))\nlp = logdensityof(d, 0.5)",
    ))
    .expect_err("both of `booleans`' atoms are outside logit's domain");
    assert!(e.reason.contains("(0, 1)"), "got: {}", e.reason);
}
