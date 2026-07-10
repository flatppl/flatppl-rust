use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

// kchain(M, K) with a CONTINUOUS latent that is NOT a recognised conjugate pair
// stays a non-enumerable marginal — the integral ∫ densityof(K(a), v) dM(a) has
// no closed form the determiniser handles. It must REFUSE (naming `kchain` +
// "non-enumerable"), never emit a Monte-Carlo / −logN approximation.
//
// Here the latent feeds the likelihood's `sigma` (a Normal prior on a standard
// deviation), which is NOT the Normal–Normal mean conjugacy the conjugate table
// recognises — so it must still refuse. (The mean-conjugate case that DOES lower
// in closed form is covered in `tests/conjugate_golden.rs`.)
#[test]
fn kchain_non_conjugate_continuous_latent_refuses() {
    let src = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))
k = kernelof(record(y = draw(Normal(mu = 1.0, sigma = z))), z = z)
pp = kchain(lawof(record(z = z)), k)
lp = logdensityof(pp, record(y = 0.5))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("non-conjugate continuous-latent kchain must refuse");
    assert!(
        err.construct.contains("kchain"),
        "refusal names kchain: {err:?}"
    );
    assert!(
        err.reason.contains("non-enumerable"),
        "refusal explains the non-enumerable marginal: {err:?}"
    );
}

// Note: `rand` no longer refuses unconditionally — a single-draw
// `rand(rng, lawof(record(x = draw(M))))` now lowers to `builtin_sample`
// (`tests/sample_golden.rs::single_draw_samples_via_builtin_sample`). The
// construct-specific sample-side refuse tests (the intractable/deferred
// measure set, malformed `rand`/`draw` arity, etc.) live in the "sample path"
// section at the end of this file.

// Note: a variate-dependent (function) `weighted`/`logweighted` weight is NO
// LONGER refused — per §06 "Density of composed measures" the weight may be a function
// of the variate, so it
// is APPLIED at the variate (`log w(v)` / `ℓ(v)`). The structural apply-path tests
// live in `tests/numeric.rs` (`weighted_function_weight_oracle`,
// `logweighted_function_weight_oracle`).

// Note: keyword `joint(name = M, …)` (named components, record variate) is NO
// LONGER refused — it now lowers to `Σᵢ logdensityof(Mᵢ, v.nameᵢ)`, matching
// the positional form's independent-product rule but scored through the
// value record's fields rather than a `get0`-sliced `cat` vector (§04
// example, §06 "joint and iid"). The lowering-path tests live in
// `tests/density_golden.rs` (`keyword_joint_lowers_to_sum_of_field_densities`,
// `keyword_joint_missing_value_field_refuses`,
// `mixed_positional_keyword_joint_refuses`).

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

// A positional `joint` component whose measure-domain kind is UNKNOWN /
// `%deferred` — not confirmed non-scalar, just never resolved — must ALSO
// refuse (fail-closed), not lower via `get0(v, i)` on the strength of "no
// confirmed mismatch". `flatppl_infer::infer` is best-effort: an unrecognized
// builtin distribution name is left `%deferred` (a diagnostic is emitted, but
// inference does not hard-error), so `b`'s inferred type here is
// `Type::Deferred`, and `component_kind` in `lower_joint` is `None` for it.
// Before this fail-closed tightening, only a CONFIRMED non-scalar domain
// refused; an unresolved domain fell through unchecked and was lowered via
// `get0(v, i)` regardless of `b`'s true (unknown) arity — a mislowering hazard
// if `b` ever turned out to be non-scalar. Per refuse-don't-mislower, "unknown"
// must refuse exactly like "confirmed non-scalar".
#[test]
fn joint_deferred_domain_component_refuses() {
    let src = "\
b = SomeUndefinedDist(mu = 0.0, sigma = 1.0)
d = joint(b, Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(d, [0.5, -0.3])";
    let m = parse_infer(src);
    let err = determinize(&m)
        .expect_err("a joint component with an unresolved/deferred domain must refuse");
    assert!(
        err.reason.contains("not confirmed scalar"),
        "refusal explains the domain kind is not confirmed scalar: {err:?}"
    );
    assert!(
        err.reason.contains("unknown") || err.reason.contains("deferred"),
        "refusal names the unknown/deferred case, distinct from a confirmed-non-scalar one: {err:?}"
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

// A GENUINELY dynamic iid size must still refuse (refuse-don't-mislower). The
// determiniser now resolves an iid's repeat count from its const-evaluated
// static domain shape (so a `lengthof(fixed_array)` size lowers), but an
// `external(posintegers)` count is runtime-determined — const-eval yields a
// `%dynamic` dim, not a static one — so the O(N) static unroll has no `N` and
// must refuse rather than guess a size.
#[test]
fn iid_dynamic_size_refuses() {
    let src = "\
n = external(posintegers)
d = iid(Normal(mu = 0.0, sigma = 1.0), n)
lp = logdensityof(d, [0.5, -0.3, 1.2])";
    let m = parse_infer(src);
    determinize(&m).expect_err("iid with a genuinely dynamic (external) size must refuse");
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

// `logdensityof(M)` with the wrong arity. §06's density query is binary —
// `logdensityof(measure, variate)` — so a call with any argument count other than
// 2 (here a single-argument call, no variate) is not a well-formed density query.
// `flatppl_infer::infer` is best-effort and does not reject the arity, so the
// determiniser's own 2-arg guard at the `logdensityof` entry must REFUSE (naming
// `logdensityof`) rather than index a missing `args[1]`.
#[test]
fn logdensityof_wrong_arity_refuses() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(record(a = a)))";
    let m = parse_infer(src);
    let err = determinize(&m)
        .expect_err("a non-binary logdensityof is not a well-formed density query — refuse");
    assert!(
        err.construct.contains("logdensityof"),
        "refusal names logdensityof: {err:?}"
    );
    assert!(
        err.reason.contains("2 args"),
        "refusal explains the binary-query arity: {err:?}"
    );
}

// `likelihoodof(K)` with the wrong arity. §06 constructs a likelihood as
// `likelihoodof(kernel, obs)` — binary. A call with any other arg count (here a
// single argument, no observation) is malformed. When such a likelihood reaches
// the likelihood-query entry (`logdensityof(L, θ)`), the determiniser's 2-arg
// guard inside `lower_likelihood_query` must REFUSE naming `likelihoodof`, rather
// than index a missing `obs` arg.
#[test]
fn likelihoodof_wrong_arity_refuses() {
    let src = "\
mu = elementof(reals)
k = kernelof(record(y = draw(Normal(mu = mu, sigma = 1.0))), mu = mu)
L = likelihoodof(k)
lp = logdensityof(L, record(mu = 2.0))";
    let m = parse_infer(src);
    let err = determinize(&m)
        .expect_err("a non-binary likelihoodof is not a well-formed likelihood — refuse");
    assert!(
        err.construct.contains("likelihoodof"),
        "refusal names likelihoodof: {err:?}"
    );
    assert!(
        err.reason.contains("2 args"),
        "refusal explains the (kernel, obs) arity: {err:?}"
    );
}

// A likelihood query whose θ record names a parameter with NO module binding.
// `logdensityof(likelihoodof(K, obs), θ)` inlines each θ field into the density
// subtree by matching `(%ref self <name>)` against the emitted kernel's free
// params. A θ field naming something that is not a declared param (here
// `nonexistent`) has no such ref to bind and no corresponding param decl — it is
// a mislowering hazard (a θ point that silently scores nothing), not a valid
// parameter point. `theta_field_map` must REFUSE such a θ up front rather than
// build a density that ignores the stray field.
#[test]
fn likelihoodof_query_theta_names_unbound_param_refuses() {
    let src = "\
mu = elementof(reals)
k = kernelof(record(y = draw(Normal(mu = mu, sigma = 1.0))), mu = mu)
L = likelihoodof(k, record(y = 0.5))
lp = logdensityof(L, record(nonexistent = 2.0))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err(
        "a θ field naming a parameter with no module binding is not a valid point — refuse",
    );
    assert!(
        err.reason.contains("no module binding"),
        "refusal explains the θ field has no corresponding param binding: {err:?}"
    );
}

// A likelihood query whose θ argument is NOT a record. `logdensityof(L, θ)` reads
// θ as the parameter point, which §06 models as a field-keyed record (one field
// per free param). A scalar θ (here `2.0`) has no field structure to inline, so
// `theta_field_map` must REFUSE naming the record requirement rather than treat
// the scalar as the variate (the measure-query path) — the likelihood-query entry
// has already committed to the θ interpretation.
#[test]
fn likelihoodof_query_theta_not_a_record_refuses() {
    let src = "\
mu = elementof(reals)
k = kernelof(record(y = draw(Normal(mu = mu, sigma = 1.0))), mu = mu)
L = likelihoodof(k, record(y = 0.5))
lp = logdensityof(L, 2.0)";
    let m = parse_infer(src);
    let err =
        determinize(&m).expect_err("a non-record θ in a likelihood query is malformed — refuse");
    assert!(
        err.reason.contains("θ must be a record"),
        "refusal explains θ must be a field-keyed record: {err:?}"
    );
}

// A MEASURE-side `record` built with POSITIONAL args instead of `%field name =`
// entries. The independent-product density rule pairs each measure component with
// the matching field of the value record by NAME (§06 "Density of composed
// measures"). A positional `record(draw(..))` carries no field names, so there is
// no key to pair against the value record — it is not a field-keyed product. The
// determiniser must REFUSE rather than mis-pair positionally.
#[test]
fn measure_record_with_positional_args_refuses() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(record(a)), record(a = 0.5))";
    let m = parse_infer(src);
    let err = determinize(&m)
        .expect_err("a positional measure record is not a field-keyed product — refuse");
    assert!(
        err.reason.contains("field-keyed product"),
        "refusal explains the record is not field-keyed: {err:?}"
    );
}

// A VALUE-side `record` built with POSITIONAL args. The variate of an independent
// product is matched to the measure components BY FIELD NAME, so a positional
// value record (here `record(0.5)`, no field names) cannot be looked up per
// component. This is distinct from the measure-side positional refuse above — it
// fires on the value record after the measure record has already matched. The
// determiniser must REFUSE.
#[test]
fn value_record_with_positional_args_refuses() {
    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(record(a = a)), record(0.5))";
    let m = parse_infer(src);
    let err = determinize(&m)
        .expect_err("a positional value record has no field keys to match against — refuse");
    assert!(
        err.reason.contains("positional args"),
        "refusal explains the value record carries positional args: {err:?}"
    );
}

// `joint_likelihood` combines a POSITIONAL list of likelihoods (§06 "Combining
// likelihoods": `log L(θ) = Σᵢ log Lᵢ(θ)`). A KEYWORD form (`joint_likelihood(a
// = L1, b = L2)`, named components) has no §06 meaning — mirroring how a keyword
// `joint(name = M, …)` is refused — so it must refuse, not silently drop the
// named components or guess a combination order.
#[test]
fn joint_likelihood_keyword_form_refuses() {
    let src = "\
mu = elementof(reals)
g1 = Normal(mu = mu, sigma = 1.0)
g2 = Normal(mu = mu, sigma = 1.0)
L1 = likelihoodof(iid(g1, 1), [1.0])
L2 = likelihoodof(iid(g2, 1), [2.0])
L = joint_likelihood(a = L1, b = L2)
lp = logdensityof(L, record(mu = 0.0))";
    let m = parse_infer(src);
    let err = determinize(&m)
        .expect_err("keyword joint_likelihood (named components) has no §06 form — refuse");
    assert!(
        err.construct.contains("joint_likelihood"),
        "refusal names joint_likelihood: {err:?}"
    );
    assert!(
        err.reason.contains("keyword joint_likelihood") && err.reason.contains("§06 form"),
        "refusal explains the keyword form is not a §06 form: {err:?}"
    );
}

// `joint_likelihood()` with NO components is not a well-formed §06 combination
// (there is nothing to sum). The determiniser must refuse rather than fold an
// empty term list into a degenerate `0.0`-density.
#[test]
fn joint_likelihood_empty_refuses() {
    let src = "\
mu = elementof(reals)
L = joint_likelihood()
lp = logdensityof(L, record(mu = 0.0))";
    let m = parse_infer(src);
    let err = determinize(&m)
        .expect_err("joint_likelihood with no components has nothing to sum — refuse");
    assert!(
        err.construct.contains("joint_likelihood"),
        "refusal names joint_likelihood: {err:?}"
    );
    assert!(
        err.reason.contains("at least one component"),
        "refusal explains a component is required: {err:?}"
    );
}

// Every `joint_likelihood` component must itself be a likelihood (typically
// `likelihoodof(K, obs)`, a name bound to one, or a nested `joint_likelihood`).
// A component that resolves to a BARE measure (`Normal(…)`, not wrapped in
// `likelihoodof`) cannot be scored as a per-component density at the shared θ,
// so the recursion through the per-likelihood lowering must refuse
// (refuse-don't-mislower) rather than treat the measure as a likelihood.
#[test]
fn joint_likelihood_non_likelihood_component_refuses() {
    let src = "\
mu = elementof(reals)
g1 = Normal(mu = mu, sigma = 1.0)
L1 = likelihoodof(iid(g1, 1), [1.0])
bare = Normal(mu = mu, sigma = 1.0)
L = joint_likelihood(L1, bare)
lp = logdensityof(L, record(mu = 0.0))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err(
        "a joint_likelihood component that is a bare measure, not a likelihood, must refuse",
    );
    assert!(
        err.reason.contains("expected likelihoodof"),
        "refusal explains the component is not a likelihood: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Sample path: the intractable / deferred set (spec §07 `rand` tractable set)
// ---------------------------------------------------------------------------
//
// `sample.rs`'s `lower_measure_sample` dispatcher used to fall through every
// unhandled measure-algebra op to one generic "unsupported measure construct"
// message. These tests pin EXPLICIT, distinct refusals instead, so a refusal
// tells the author WHY (outside rand's tractable set vs. simply not built
// yet), not just THAT.
//
// `weighted`/`logweighted`/`bayesupdate`, `truncate`, and the deferred
// combinators (`jointchain`/`kchain`/`superpose`/`pushfwd`) all reach their
// refuse arm via `draw(<op>(...))` — the common surface shape — rather than
// via `lower_measure_sample`'s own dispatch match (which only sees these ops
// when they are NOT wrapped in `draw`: `lawof`'s direct argument, or an
// un-drawn measure sitting in a record field). `sample.rs::lower_draw`
// classifies its inner measure the same way for exactly this reason — see its
// doc comment.

// `d = draw(weighted(w, Normal(...)))`: a reweighted measure has no direct
// sampling recipe (realizing its law needs a change-of-measure algorithm —
// rejection/importance sampling, MCMC — out of scope for this MVP's exact
// sample lowering). `logweighted`/`bayesupdate` share the same arm/message
// (not separately tested — the match arm covers all three identically).
#[test]
fn sample_weighted_refuses() {
    let src = "\
s = rnginit(0)
w = functionof(mul(2.0, _x_), x = _x_)
m = weighted(w, Normal(mu = 0.0, sigma = 1.0))
d = draw(m)
draws = rand(s, lawof(record(d = d)))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("sampling a weighted measure is intractable — refuse");
    assert!(
        err.reason.contains("weighted") || err.reason.contains("intractable"),
        "refusal explains weighted is outside rand's tractable set: {err:?}"
    );
}

// `lawof(c)` where `c` is a plain constant, not a draw: `lawof`'s OWN phase is
// deterministic (spec §04 "Phase of the reified law" — it absorbs the
// argument's stochasticity rather than propagating it), so the phase that must
// be checked is the ARGUMENT's. `c`'s phase is `Fixed`, not `Stochastic`: there
// is no generative `draw` subgraph for `rand` to re-run, so `rand` must refuse
// rather than echo the constant back out as a "sample".
#[test]
fn sample_lawof_of_nonstochastic_refuses() {
    let src = "\
s = rnginit(0)
c = 3.0
draws = rand(s, lawof(c))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("lawof of a non-stochastic value must refuse");
    assert!(
        err.reason.contains("stochastic"),
        "refusal names the non-stochastic arg: {err:?}"
    );
}

// `d = draw(truncate(MvNormal(...), S))`: the base is CONFIRMED multivariate
// (`truncate(base, S)` infers `base`'s own domain — an `array` here, from
// MvNormal's `VectorFromParam` domain). There is no general sampling recipe
// for an arbitrary multivariate truncated region (rejection sampling is not
// exact/closed-form in general), so this is intractable — a DIFFERENT
// classification from the univariate case (pinned separately below), which is
// plausible future work, just not yet built.
#[test]
fn sample_truncate_multivariate_refuses() {
    let src = "\
s = rnginit(0)
d = draw(truncate(MvNormal(mu = [0.0, 0.0], cov = eye(2)), interval(-1.0, 1.0)))
draws = rand(s, lawof(record(d = d)))";
    let m = parse_infer(src);
    let err = determinize(&m)
        .expect_err("sampling a multivariate truncated measure is intractable — refuse");
    assert!(
        err.reason.contains("multivariate") && err.reason.contains("intractable"),
        "refusal explains the multivariate truncation is outside rand's tractable set: {err:?}"
    );
}

// `d = draw(truncate(Normal(...), S))`: the base is confirmed UNIVARIATE
// (`domain = Scalar(Real)`), so this is NOT the intractable multivariate case
// above — it falls into the deferred-combinator bucket (a closed-form or
// rejection-sampling recipe is plausible future work, simply not built in this
// vertical). Pins the OTHER branch of the truncate split, distinct from the
// multivariate one.
#[test]
fn sample_truncate_univariate_deferred_refuses() {
    let src = "\
s = rnginit(0)
d = draw(truncate(Normal(mu = 0.0, sigma = 1.0), interval(-1.0, 1.0)))
draws = rand(s, lawof(record(d = d)))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("univariate truncate sampling is not yet built — refuse");
    assert!(
        err.reason.contains("deferred"),
        "refusal explains the univariate truncate is deferred, not intractable: {err:?}"
    );
    assert!(
        !err.reason.contains("multivariate"),
        "univariate truncate must not be classified multivariate: {err:?}"
    );
}

// `d = draw(superpose(Normal(...), Normal(...)))`: `superpose` (measure
// addition) is not conceptually intractable — a later vertical could thread
// the rng through it — it is simply not built in this one (direct draws +
// record-of-draws + shared ancestors). `jointchain`/`kchain`/`pushfwd` share
// the identical match arm and message (not separately tested).
#[test]
fn sample_deferred_combinator_refuses() {
    let src = "\
s = rnginit(0)
d = draw(superpose(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 1.0, sigma = 1.0)))
draws = rand(s, lawof(record(d = d)))";
    let m = parse_infer(src);
    let err =
        determinize(&m).expect_err("sample lowering for superpose is deferred — refuse, not lower");
    assert!(
        err.reason.contains("deferred"),
        "refusal explains the combinator's sample lowering is deferred: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Sample path: malformed-shape guards in `lower_rand`/`lower_draw` (Task 1)
// ---------------------------------------------------------------------------
//
// These guards shipped with Task 1 (`sample.rs::lower_rand`/`lower_draw`) but
// had no dedicated test — added here per the same TDD discipline as the rest
// of this file.

// `rand` takes exactly 2 args (rng, measure); a 1-arg call is malformed.
#[test]
fn sample_rand_wrong_arity_refuses() {
    let src = "\
s = rnginit(0)
draws = rand(s)";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("rand with the wrong arity must refuse");
    assert!(
        err.reason.contains("rand expects 2 args"),
        "refusal explains the (rng, measure) arity: {err:?}"
    );
}

// `rand(rng, M)` where `M` is not `lawof(...)` at all — `rand` samples the LAW
// of a stochastic subgraph, so its second argument must be a `lawof(...)`.
#[test]
fn sample_rand_measure_not_lawof_refuses() {
    let src = "\
s = rnginit(0)
draws = rand(s, Normal(mu = 0.0, sigma = 1.0))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("rand's measure must be lawof(...) — refuse");
    assert!(
        err.reason.contains("lawof"),
        "refusal names the lawof requirement: {err:?}"
    );
}

// `draw` takes exactly 1 arg (the measure); a 2-arg call is malformed.
#[test]
fn sample_draw_wrong_arity_refuses() {
    let src = "\
s = rnginit(0)
x = draw(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 1.0, sigma = 1.0))
draws = rand(s, lawof(record(x = x)))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err("draw with the wrong arity must refuse");
    assert!(
        err.reason.contains("draw expects 1 arg"),
        "refusal explains the 1-arg draw shape: {err:?}"
    );
}

// `draw(M)` where `M` is a constructor call with MORE positional args than the
// constructor has parameters (`Normal(0.0, 1.0, 2.0)` — 3 args, 2 params) cannot
// bind by position, so `split_kernel_constructor` returns `None` and the sample
// leaf refuses rather than dropping the extra arg. (A well-formed positional
// constructor DOES lower — see
// `sample_golden::sample_draw_positional_constructor_lowers_same_as_keyword`.)
#[test]
fn sample_draw_over_arity_positional_constructor_refuses() {
    let src = "\
s = rnginit(0)
x = draw(Normal(0.0, 1.0, 2.0))
draws = rand(s, lawof(record(x = x)))";
    let m = parse_infer(src);
    let err = determinize(&m)
        .expect_err("more positional args than parameters is not a valid sample leaf — refuse");
    assert!(
        err.reason.contains("built-in kernel constructor"),
        "refusal explains the constructor-shape requirement: {err:?}"
    );
}

// `draw(M)` where `M` binds a parameter BOTH positionally and by keyword
// (`Normal(0.0, mu = 1.0)` — `mu` bound twice) is a §04 double-bind (static
// error); `split_kernel_constructor` refuses rather than emit a record with
// duplicate `mu` fields.
#[test]
fn sample_draw_double_bound_constructor_refuses() {
    let src = "\
s = rnginit(0)
x = draw(Normal(0.0, mu = 1.0))
draws = rand(s, lawof(record(x = x)))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err(
        "a parameter bound positionally and by keyword is not a valid sample leaf — refuse",
    );
    assert!(
        err.reason.contains("built-in kernel constructor"),
        "refusal explains the constructor-shape requirement: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Sample path: an inline-hierarchical shared latent (untested boundary, T3 M6)
// ---------------------------------------------------------------------------
//
// `record(c = draw(Normal(mu = mu, sigma = 1.0)))` where `mu = draw(Normal(0,
// 10))` is a SEPARATE latent binding, referenced only from INSIDE the record
// field's INLINE draw (not via a `(%ref self mu)` field of its own, and not
// via a named binding consuming `mu` — c's own draw is written inline). Task
// 3's shared-latent detection (`field_draw_binding`/
// `requires_shared_binding_rewrite`) only recognizes a field whose OWN value
// is `(%ref self <draw-binding>)`; an inline `draw(...)` field value returns
// `None` from `field_draw_binding`, so this shape is NOT detected as
// hierarchical and takes the independent-draws fold instead. That fold DOES
// correctly sample `c`'s inline draw (its kernel input keeps its
// `(%ref self mu)`, unresolved) — but `mu`'s OWN `draw`-binding is never
// visited (nothing in the record's own field-list references `mu` directly),
// so it survives the `rand` lowering as a bare, un-sampled, Stochastic-phase
// `draw(...)` binding. The determiniser's own residual-measure-layer sweep
// (`driver.rs`'s next scan iteration) then finds it and refuses with the
// generic "no determinization rule" message — SAFE (no silent mislowering:
// `mu` is never smuggled through as a `%deferred`/un-sampled ref), but
// previously untested. This pins that the boundary refuses cleanly rather than
// silently drops `mu`'s stochasticity.
#[test]
fn sample_inline_hierarchical_shared_latent_refuses() {
    let src = "\
s = rnginit(0)
mu = draw(Normal(mu = 0.0, sigma = 10.0))
draws = rand(s, lawof(record(c = draw(Normal(mu = mu, sigma = 1.0)))))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err(
        "an inline-hierarchical shared latent (mu referenced only from inside c's inline draw) \
         must refuse cleanly, not silently drop mu's stochasticity",
    );
    assert!(
        err.construct.contains("draw"),
        "refusal names the un-sampled mu draw binding: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Sample path: destructured / rng-threaded `rand` lowers to a tuple
// ---------------------------------------------------------------------------
//
// `lower_rand` used to implement ONLY the value-terminal convention, refusing
// a destructured result rather than risk a silent mislowering (see git
// history for the prior `destructured_rand_refuses`/
// `get0_tuple_projection_of_rand_refuses` shape of this guard). It now builds
// the full spec §07 `tuple(value, new_rstate)` when the result is
// destructured (`rand_result_is_destructured`, `crates/determinizer/src/
// sample.rs`), so the parser's `v, s2 = rand(...)` multi-LHS sugar
// (`lower_decomposition`, `crates/syntax/src/parser.rs` — desugars to `__0x1 =
// rand(...); v = get(__0x1, 1); s2 = get(__0x1, 2)`, 1-based integer-literal
// `get` projections off the synthetic tmp binding) resolves against a real
// tuple instead of indexing an erased one. These goldens (the direct
// decomposition, the realistic "thread the rng across two draws" shape the
// spec's own §07 example uses, and the 0-based `get0` spelling) are pinned
// here since they are the exact shapes the former refusal guarded; full
// tuple-lowering coverage lives in `tests/sample_golden.rs`.
#[test]
fn destructured_rand_in_refuse_suite_lowers_to_tuple() {
    let src = "\
s = rnginit(0)
x = draw(Normal(mu = 0.0, sigma = 1.0))
v, s2 = rand(s, lawof(record(x = x)))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("a destructured rand must lower to a tuple");
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("(tuple "),
        "expected tuple(value, advanced_rng):\n{pir}"
    );
    assert!(
        pir.contains("builtin_sample"),
        "expected a builtin_sample under the tuple:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL:\n{pir}"
    );
}

// The realistic "thread the rng across two draws" shape the spec's own §07
// example uses: `s2` (the first `rand`'s advanced rngstate) is destructured
// out and threaded into a second `rand`. Both draws are individually within
// `rand`'s tractable set (single-draw records), and now the destructuring of
// the first `rand`'s result lowers too, so the whole chain lowers.
#[test]
fn destructured_rand_rng_threaded_into_second_rand_lowers() {
    let src = "\
s = rnginit(0)
x = draw(Normal(mu = 0.0, sigma = 1.0))
v, s2 = rand(s, lawof(record(x = x)))
y = draw(Normal(mu = 1.0, sigma = 1.0))
w = rand(s2, lawof(record(y = y)))";
    let m = parse_infer(src);
    let out = determinize(&m)
        .expect("threading a destructured rand's rngstate into a second rand must lower");
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_sample").count(),
        3,
        "two logical samples; the first's shared (value, rng) tuple is re-expanded once more \
         where its rng feeds the second sample (no CSE in the writer):\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL:\n{pir}"
    );
}

// `get0(draws, 0)` / `get0(draws, 1)` — the 0-based spelling of the same
// tuple-projection shape a user could write directly (without the `v, s2 =`
// decomposition sugar, which always emits 1-based `get`). Same dispatch,
// same tuple.
#[test]
fn get0_tuple_projection_of_rand_lowers() {
    let src = "\
s = rnginit(0)
x = draw(Normal(mu = 0.0, sigma = 1.0))
draws = rand(s, lawof(record(x = x)))
v = get0(draws, 0)";
    let m = parse_infer(src);
    let out = determinize(&m)
        .expect("get0(draws, 0) is a tuple-slot projection and must lower against a real tuple");
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("(tuple "),
        "expected tuple(value, advanced_rng):\n{pir}"
    );
    assert!(
        pir.contains("builtin_sample"),
        "expected a builtin_sample under the tuple:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL:\n{pir}"
    );
}

// A record-field STRING selector on a value-terminal rand's result
// (`get(draws, "mu")` / `draws.mu`) is NOT a tuple projection — `get_type`
// keys record-field access on `Node::Lit(Scalar::Str(_))`, never
// `Scalar::Int` — so it must NOT trip the destructuring guard. This is the
// "record fields accessed by string selector" shape the fix must not
// over-refuse (the value-terminal convention must keep lowering).
#[test]
fn value_terminal_rand_field_access_by_string_still_lowers() {
    let src = "\
s = rnginit(0)
mu = draw(Normal(mu = 0.0, sigma = 1.0))
draws = rand(s, lawof(record(mu = mu)))
out = draws.mu";
    let m = parse_infer(src);
    let out = determinize(&m)
        .expect("a string-selector field access on a value-terminal rand must still lower");
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("builtin_sample"),
        "the single draw still lowers to builtin_sample:\n{pir}"
    );
    assert!(
        !pir.contains("(draw ") && !pir.contains("(lawof ") && !pir.contains("(rand "),
        "measure/sample-surface layer eliminated:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL:\n{pir}"
    );
}
