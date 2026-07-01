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

// Note: a variate-dependent (function) `weighted`/`logweighted` weight is NO
// LONGER refused — per §06 "Density of composed measures" the weight may be a function
// of the variate, so it
// is APPLIED at the variate (`log w(v)` / `ℓ(v)`). The structural apply-path tests
// live in `tests/numeric.rs` (`weighted_function_weight_oracle`,
// `logweighted_function_weight_oracle`).

// Keyword `joint(name = M, …)` (named components, record variate) shares the
// `joint` op name with the positional form but is deliberately out of scope:
// its components live in `named`, not `args`, so it must not fall through to
// the positional arg-count guard (which would misreport it as an under-sized
// positional `joint`). The determiniser must refuse with a distinct message
// naming keyword joint specifically.
#[test]
fn keyword_joint_refuses_with_distinct_message() {
    let src = "\
a = Normal(mu = 0.0, sigma = 1.0)
b = Normal(mu = 1.0, sigma = 2.0)
d = joint(x = a, y = b)
lp = logdensityof(d, record(x = 0.5, y = 0.5))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("keyword joint must refuse, not lower");
    assert!(
        err.reason.contains("keyword joint"),
        "refusal names keyword joint distinctly: {err:?}"
    );
}

// A positional `joint` whose component is NON-SCALAR (here `iid(Normal, 2)`,
// measure domain array[2]) cannot use the `get0(v, i)` one-slot-per-component
// destructuring: it needs `cat`-slice routing, which is not built. The
// determiniser must REFUSE up front — the OLD code silently mislowered
// (destructured positionally as if scalar, dropping the extra slots), because
// the downstream `build_density_term` domain check compares against
// `get0(v, i)` (which infers to `%deferred`/`%unknown`) and so is skipped. The
// gate here reads each component's OWN measure domain kind, which IS known
// (refuse up front on a non-scalar joint component's measure domain).
#[test]
fn joint_nonscalar_component_refuses() {
    let src = "\
d = joint(iid(Normal(mu = 0.0, sigma = 1.0), 2), Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(d, [0.5, -0.3, 1.2])";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("non-scalar joint component must refuse, not mislower");
    assert!(
        err.reason.contains("non-scalar"),
        "refusal explains the non-scalar component: {err:?}"
    );
    assert!(
        err.reason.contains("cat-slice"),
        "refusal points at the missing cat-slice routing: {err:?}"
    );
}

// `normalize(truncate(base, interval(lo, hi)))` uses the closed-form
// Z = touniform(base, hi) − touniform(base, lo) = CDF(hi) − CDF(lo). That
// identity holds ONLY when `base` is a normalized univariate continuous
// probability measure — `builtin_touniform` is the CDF only for univariate
// continuous kernels (§07 "measure-eval-prims"), and the transport is defined
// only for continuous built-in kernels (§07 "measure-eval-prims"). For an
// UNNORMALIZED base
// (here `Lebesgue(reals)`, whose true
// Z = hi − lo and for which `touniform` is undefined) the CDF path silently
// mislowers, so the determiniser must NOT take it — it falls through to the
// refuse (no closed-form Z for an unnormalized base is built; refuse
// normalize(truncate(<unnormalized base>, …)) rather than use the CDF-Z path).
#[test]
fn normalize_truncate_unnormalized_base_refuses() {
    let src = "\
d = normalize(truncate(Lebesgue(reals), interval(-1.0, 1.0)))
lp = logdensityof(d, 0.5)";
    let m = parse_infer(src);
    let err = determinize(&m)
        .expect_err("normalize(truncate(<unnormalized base>, …)) must refuse, not use the CDF-Z");
    // Must NOT have emitted the touniform CDF-Z path: it refuses instead.
    assert!(
        err.construct.contains("normalize") || err.reason.contains("unnormalized"),
        "refusal is about the unnormalized-base normalize, not the CDF path: {err:?}"
    );
}

// `normalize(truncate(Binomial, interval(lo, hi)))` — the base is NORMALIZED
// (a probability measure) but DISCRETE (`domain = Scalar(Integer)`). The CDF-Z
// path `Z = touniform(base, hi) − touniform(base, lo)` is NOT valid here:
// `builtin_touniform` is the CDF `F` only for a univariate CONTINUOUS kernel
// (§07 "measure-eval-prims"), and use of the transport on a non-continuous kernel
// is an undefined transport / static error — plus `Z = F(hi) − F(lo)` is a
// univariate-continuous identity regardless. The OLD gate keyed only on
// `base.mass == Normalized`, so it WRONGLY admitted this discrete base and
// emitted `builtin_touniform(Binomial, …)` — an undefined transport that still
// passes `is_flatpdl` (a silent mislowering). The tightened gate additionally
// requires `domain = Scalar(Real)`, so a normalized discrete base now refuses.
// This refuse is DISTINCT from the unnormalized-base one (a discrete-truncation
// closed-form Z — e.g. a CMF / finite-support sum — is a legitimate future
// follow-on, not an invalid model), so it names the univariate-continuous
// restriction rather than "unnormalized".
#[test]
fn normalize_truncate_discrete_base_refuses() {
    let src = "\
d = normalize(truncate(Binomial(n = 10, p = 0.5), interval(2, 8)))
lp = logdensityof(d, 5)";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err(
        "normalize(truncate(<normalized discrete base>, …)) must refuse (touniform is CDF only \
         for univariate continuous), not emit an undefined builtin_touniform transport",
    );
    assert!(
        err.construct.contains("normalize"),
        "refusal names normalize: {err:?}"
    );
    // Distinct from the unnormalized-base refuse: it names the univariate-continuous
    // restriction (touniform = CDF only there), not "unnormalized".
    assert!(
        err.reason.contains("univariate continuous") && err.reason.contains("touniform"),
        "refusal explains the univariate-continuous-only touniform restriction: {err:?}"
    );
    assert!(
        !err.reason.contains("unnormalized"),
        "discrete base is normalized — refusal must NOT claim it is unnormalized: {err:?}"
    );
}

// `normalize(truncate(Normal, posreals))` — the truncation set is a NAMED set
// (`posreals`), not a literal `interval(lo, hi)` call. The CDF-Z path can only
// read off `lo`/`hi` from a literal `interval(...)` node, so this shape falls
// out of `truncate_shape` with `non_interval_truncation_set = true`. Before this
// fix that fell through to the generic "normalize of an unnormalized measure"
// refuse, which is misleading here — the base (`Normal`) IS normalized; the
// actual gap is that closed-form Z is only wired up for a literal interval
// bound. This pins the DISTINCT message naming the interval restriction.
#[test]
fn normalize_truncate_non_interval_set_refuses_with_distinct_message() {
    let src = "\
d = normalize(truncate(Normal(mu = 0.0, sigma = 1.0), posreals))
lp = logdensityof(d, 0.5)";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err(
        "normalize(truncate(<normalized base>, <named set>)) must refuse — closed-form Z is only \
         wired up for a literal interval(lo, hi) bound",
    );
    assert!(
        err.construct.contains("normalize"),
        "refusal names normalize: {err:?}"
    );
    assert!(
        err.reason.contains("interval(lo, hi)"),
        "refusal names the interval-only restriction: {err:?}"
    );
    assert!(
        !err.reason.contains("unnormalized"),
        "base is normalized — refusal must NOT claim it is unnormalized: {err:?}"
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

// `iid(M, [2, 3])` — a MULTI-AXIS (vector) `size`. §06 "Independent composition"
// admits `size` as "an integer (1-D length) or a vector of positive integers
// (multi-axis shape)", so this is a VALID model. But the determiniser's O(N)
// static unroll (`literal_usize` + `lower_iid`) reads only a literal SCALAR
// integer `N`; the vectorized broadcast+reduce over a multi-axis shape is the
// noted scale path, not built. So a vector `size` is a CONSERVATIVE refuse
// (refuse-don't-mislower), not a bug — this pins that behavior.
#[test]
fn iid_multi_axis_size_refuses() {
    let src = "\
d = iid(Normal(mu = 0.0, sigma = 1.0), [2, 3])
lp = logdensityof(d, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])";
    let m = parse_infer(src);
    determinize(&m).expect_err(
        "iid with a multi-axis / vector size must refuse (only a literal scalar N is unrolled)",
    );
}

// A θ parameter captured as a `functionof` / `kernelof`
// reification INPUT (a `(coeff, %ref self coeff)` boundary entry) cannot be
// inlined by the per-query θ substitution — `substitute_refs_by_name` walks
// `map_tree` / `children()`, which EXCLUDES a `Call`'s `Inputs` bucket (core
// `node.rs` `for_each_child`). Here the density subtree references its weight by
// name (`(%call (%ref self w) obs)`), and `w = functionof(mul(coeff, _x_), …,
// coeff = coeff)` closes over the θ param `coeff` as a boundary input. Left
// un-inlined, `coeff` would stay a dangling `(%ref self coeff)` inside the
// reification, so the density would score as a function of the FREE `coeff`
// instead of at θ = 2.0 — a silent mislowering that still passes `is_flatpdl`. A
// prior fix guarded this with a `debug_assert!`, which is STRIPPED in release, so
// in release the mislowering shipped. The determiniser must HARD REFUSE in every
// build profile (this test runs in debug and encodes the refuse, not the assert).
#[test]
fn theta_captured_in_reification_input_refuses() {
    let src = "\
coeff = elementof(reals)
w = functionof(mul(coeff, _x_), x = _x_, coeff = coeff)
weight_base = weighted(w, Normal(mu = 0.0, sigma = 1.0))
obs = likelihoodof(weight_base, 0.5)
lp = logdensityof(obs, record(coeff = 2.0))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err(
        "a θ param captured inside a functionof/kernelof reification input must refuse, \
         not silently score at the free param",
    );
    assert!(
        err.reason.contains("reification input") && err.reason.contains("cannot be"),
        "refusal explains the θ-in-reification-input cannot be inlined: {err:?}"
    );
    assert!(
        err.reason.contains("refuse rather than mislower"),
        "refusal makes the refuse-don't-mislower stance explicit: {err:?}"
    );
}

// Composed truncation base: `normalize(truncate(pushfwd(bij, Normal), S))`
// — a truncated log-normal-shaped base. The CDF-Z path emits
// `builtin_touniform(<head>, …)` for ANY builtin head, but `touniform` is the CDF
// only for a LEAF built-in distribution kernel (`Normal`, …), NOT a measure
// combinator (`pushfwd`, …): `builtin_touniform(pushfwd, …)` is an UNDEFINED
// transport (a static error, §07 "Measure kernel evaluation primitives"). The
// determiniser must REFUSE. (With current inference the composed base surfaces
// as `domain = %any`, so this shape happens to refuse via the
// discrete/multivariate arm; the leaf-constructor HEAD gate — which does not rely
// on downstream re-inference — is unit-tested directly in
// `density.rs::tests::normalize_truncate_composed_head_refuses_leaf_constructor_message`.
// This black-box test pins that a composed/pushfwd base is refused, not lowered.)
#[test]
fn normalize_truncate_pushfwd_base_refuses() {
    let src = "\
b = bijection(x -> add(x, 1.0), y -> sub(y, 1.0), z -> 0.0)
comp_base = pushfwd(b, Normal(mu = 0.0, sigma = 1.0))
d = normalize(truncate(comp_base, interval(1.0, 3.0)))
lp = logdensityof(d, 2.0)";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err(
        "normalize(truncate(<composed/pushfwd base>, …)) must refuse — touniform is undefined \
         for a measure-combinator head, not just discrete/multivariate leaf kernels",
    );
    assert!(
        err.construct.contains("normalize"),
        "refusal names normalize: {err:?}"
    );
}
