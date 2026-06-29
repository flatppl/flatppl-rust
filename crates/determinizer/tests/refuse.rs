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

// A `weighted(w, M)` whose weight `w` is VARIATE-DEPENDENT (a `functionof(…)`
// reification — the shape the HS3 converter emits, e.g.
// `weighted(functionof(polynomial(coeffs)), Lebesgue)`) must be APPLIED at the
// variate: spec §06 gives `log densityof(weighted(w, M), x) = log w(x) + …`. This
// MVP does not yet apply the weight at the variate, so it must REFUSE rather than
// emit `log(w)` of a function OBJECT — a silent mislowering that even passes
// `is_flatpdl` (the weight is Function-typed, not Measure-typed). The refusal must
// name `weighted` and explain the variate-dependent (function) weight is deferred.
#[test]
fn weighted_with_function_weight_refuses() {
    let src = "\
fw = functionof(exp(_x_), x = _x_)
m = weighted(fw, Lebesgue(reals))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("variate-dependent weighted must refuse, not mislower");
    assert!(
        err.construct.contains("weighted"),
        "refusal names weighted: {err:?}"
    );
    assert!(
        (err.reason.contains("variate-dependent") || err.reason.contains("function"))
            && err.reason.contains("weight"),
        "refusal explains the variate-dependent function weight: {err:?}"
    );
}

// Same hole on the log scale: a `logweighted(ℓ, M)` whose log-weight `ℓ` is a
// `functionof(…)` (e.g. HS3's `logweighted(functionof(logdensityof(g2, _x_)), g1)`)
// must be applied at the variate (`ℓ(x) + …`), not summed AS-IS with the density.
// The determiniser must REFUSE, naming `logweighted`.
#[test]
fn logweighted_with_function_logweight_refuses() {
    let src = "\
lw = functionof(neg(_x_), x = _x_)
m = logweighted(lw, Normal(mu = 0.0, sigma = 1.0))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("variate-dependent logweighted must refuse, not mislower");
    assert!(
        err.construct.contains("logweighted"),
        "refusal names logweighted: {err:?}"
    );
    assert!(
        (err.reason.contains("variate-dependent") || err.reason.contains("function"))
            && err.reason.contains("weight"),
        "refusal explains the variate-dependent function weight: {err:?}"
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
        err.reason.contains("not implemented") && err.reason.contains("deferred"),
        "refusal reason explains markovchain density lowering is deferred: {err:?}"
    );
}
