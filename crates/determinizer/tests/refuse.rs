use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

// kchain(M, K) with a CONTINUOUS latent (Normal) is a non-enumerable marginal —
// the integral ∫ densityof(K(a), v) dM(a) has no closed form in this MVP. The
// determiniser must REFUSE (naming `kchain` + "non-enumerable"), never emit a
// Monte-Carlo / −logN approximation. (The conjugate closed-form table is a
// later follow-on.)
#[test]
fn kchain_continuous_latent_refuses() {
    let src = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
k = kernelof(record(y = draw(Normal(mu = z, sigma = 1.0))), z = z)
pp = kchain(lawof(record(z = z)), k)
lp = logdensityof(pp, record(y = 0.5))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("continuous-latent kchain must refuse, not lower");
    assert!(
        err.construct.contains("kchain"),
        "refusal names kchain: {err:?}"
    );
    assert!(
        err.reason.contains("non-enumerable"),
        "refusal explains the non-enumerable marginal: {err:?}"
    );
}

// `rand(rng, M)` is the sample-side slice (spec §07): it threads an RNG through
// the measure algebra and returns a (value, new_rng) tuple. The determiniser
// lowering is density-only for this MVP; sampling is a later slice. The refusal
// must name `rand` and make clear that sampling is deferred, not that there is
// a generic missing rule.
#[test]
fn rand_refuses_with_sampling_deferred_message() {
    let src = "\
s = rnginit(0)
r = rand(s, lawof(draw(Normal(mu = 0.0, sigma = 1.0))))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("rand must refuse in the density-only MVP");
    assert!(
        err.construct.contains("rand"),
        "refusal names rand: {err:?}"
    );
    assert!(
        err.reason.contains("sampling") || err.reason.contains("rand"),
        "refusal mentions sampling/rand as the deferred construct: {err:?}"
    );
    // Must not be the generic fallback — the message must reference the later-slice deferral.
    assert!(
        err.reason.contains("later")
            || err.reason.contains("slice")
            || err.reason.contains("deferred"),
        "refusal explains this is a later-slice / deferred construct: {err:?}"
    );
}

// `markovchain` is an unsupported measure-algebra combinator in this MVP — it
// requires a Markov kernel and stationary-distribution reasoning that goes well
// beyond density disintegration. The determiniser must refuse naming the
// construct, never emit a partial lowering.
#[test]
fn unsupported_algebra_op_markovchain_refuses() {
    // markovchain(M, K) — we use the same kernel setup as the kchain test;
    // markovchain is a different op (stationary distribution of the chain).
    let src = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
k = kernelof(record(z = draw(Normal(mu = z, sigma = 0.1))), z = z)
mc = markovchain(lawof(record(z = z)), k)
lp = logdensityof(mc, record(z = 1.0))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("markovchain must refuse, not lower");
    assert!(
        err.construct.contains("markovchain"),
        "refusal names markovchain: {err:?}"
    );
    assert!(
        !err.reason.is_empty(),
        "refusal has a non-empty reason: {err:?}"
    );
}
