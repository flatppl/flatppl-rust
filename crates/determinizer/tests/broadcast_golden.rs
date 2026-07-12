//! Axis-native density lowering of a broadcast-kernel measure
//! `logdensityof(broadcast(K, args…), obs)` (spec §04 broadcasting, §06 "Density
//! of composed measures", §07 reductions).
//!
//! The measure `broadcast(K, arg0, …)` — what `K.(arg0, …)` desugars to — is an
//! array-of-kernels whose density at an observed array is the sum over cells of
//! the per-cell kernel log-density. The determiniser lowers it to a single
//! axis-level expression, NOT an unroll:
//! ```text
//! sum( broadcast(builtin_logdensityof, K, broadcast(record, pᵢ = argᵢ, …), obs) )
//! ```
//! identical for any length, static or dynamic — no `get0` / `iid`-style element
//! enumeration. `broadcast` / `record` / `sum` are §04/§07 ops that survive into
//! FlatPDL, so `is_flatpdl` accepts the result; only the `logdensityof`
//! measure-query is eliminated.

use flatppl_determinizer::{determinize, is_flatpdl};

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn determinize_src(src: &str) -> flatppl_core::Module {
    determinize(&parse_infer(src)).expect("must lower, not refuse")
}

// A broadcast of a single-parameter kernel — `Poisson.(rates)` — scored at an
// observed integer array lowers to `sum(broadcast(builtin_logdensityof, Poisson,
// broadcast(record, rate = rates), obs))`: one outer broadcast that zips the
// per-cell kernel input with `obs`, one inner broadcast that builds the per-cell
// `record(rate = rates[cell])`, and one `sum` reduction. No unroll, no `lawof`.
#[test]
fn broadcast_poisson_lowers_to_sum_over_broadcast() {
    let src = "\
rates = [1.0, 2.0, 3.0]
lp = logdensityof(lawof(broadcast(Poisson, rates)), [0, 1, 2])";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);

    // Exact nesting: sum over the outer broadcast; the per-cell kernel input is
    // the inner record-broadcast keyed by the constructor's param name (`rate`).
    assert!(pir.contains("(sum "), "sum reduction present:\n{pir}");
    assert!(
        pir.contains("(broadcast builtin_logdensityof Poisson"),
        "outer broadcast zips builtin_logdensityof over the Poisson tag:\n{pir}"
    );
    assert!(
        pir.contains("(broadcast record (%kwarg rate"),
        "inner broadcast builds per-cell record(rate = …):\n{pir}"
    );
    // Exactly the axis-native shape: one `sum`, one `builtin_logdensityof`, two
    // `broadcast`s (outer density + inner record). Anything more would be an
    // unroll or a stray term.
    assert_eq!(pir.matches("(sum ").count(), 1, "one sum:\n{pir}");
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "one density term (not one-per-element):\n{pir}"
    );
    assert_eq!(
        pir.matches("broadcast").count(),
        2,
        "two broadcasts:\n{pir}"
    );

    // No element unroll (the `lower_iid` path) and no surviving measure layer.
    assert!(
        !pir.contains("get0") && !pir.contains("(iid "),
        "must be axis-native, not unrolled:\n{pir}"
    );
    assert!(
        !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl:\n{pir}");
}

// The lowered form is an AXIS expression, so it is byte-for-byte the same shape
// whether the length is a static 3 or a genuinely dynamic (unknown-length)
// array — there is no static-size unroll. `elementof(cartpow(posreals))` (no
// size) declares a dynamic-length parameter array.
#[test]
fn broadcast_kernel_density_is_axis_native_for_dynamic_length() {
    // Dynamic length: the array size `n` is a parameterized-phase scalar (not a
    // literal), so `cartpow(_, n)` carries a `%dynamic` dimension (spec §11) —
    // the size is a valid, statically-unresolved size expression, NOT an omitted
    // arg (`cartpow(S)` with no size is ill-formed: spec §03 requires the size).
    let src = "\
n = elementof(posintegers)
rates = elementof(cartpow(posreals, n))
obs = elementof(cartpow(nonnegintegers, n))
lp = logdensityof(lawof(broadcast(Poisson, rates)), obs)";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);

    // Identical structural shape to the static-length case: one sum, one density
    // term, two broadcasts, no per-element unroll.
    assert_eq!(pir.matches("(sum ").count(), 1, "one sum:\n{pir}");
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "one density term regardless of length:\n{pir}"
    );
    assert_eq!(
        pir.matches("broadcast").count(),
        2,
        "two broadcasts:\n{pir}"
    );
    assert!(
        !pir.contains("get0") && !pir.contains("(iid "),
        "dynamic length must not unroll:\n{pir}"
    );
    assert!(
        pir.contains("(broadcast builtin_logdensityof Poisson")
            && pir.contains("(broadcast record (%kwarg rate"),
        "same emission as the static case:\n{pir}"
    );
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl:\n{pir}");
}

// A MODULE-QUALIFIED distribution head — `hepphys.ContinuedPoisson.(rates)` — is
// resolved to its bare member name (`ContinuedPoisson`) exactly like a base
// built-in constructor: the broadcast head is a `(%ref hepphys ContinuedPoisson)`
// module-member ref, not a bare `Const`, but its member name is a known §09
// distribution (`distribution_param_names` → `["rate"]`). The emitted kernel is
// the BARE `Const(ContinuedPoisson)` tag — the same form the JS REGISTRY keys and
// the same Poisson-shaped emission the broadcast arm already produces. This is the
// histfactory `functionof(hepphys.ContinuedPoisson.(mcstat .* mcstat_tau))` shape.
#[test]
fn broadcast_module_qualified_head_resolves_to_bare_member_kernel() {
    let src = "\
hepphys = standard_module(\"particle-physics\", \"0.1\")
rates = [2.0, 3.0]
obs = [1.0, 2.0]
lp = logdensityof(lawof(broadcast(hepphys.ContinuedPoisson, rates)), obs)";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);

    // Same axis-native emission as the bare-Const broadcast-kernel, but keyed by
    // the resolved member name: one sum, the outer density broadcast over the BARE
    // `ContinuedPoisson` tag, the inner per-cell record keyed by the member's param
    // name (`rate`).
    assert!(pir.contains("(sum "), "sum reduction present:\n{pir}");
    assert!(
        pir.contains("(broadcast builtin_logdensityof ContinuedPoisson"),
        "outer broadcast zips builtin_logdensityof over the bare ContinuedPoisson tag:\n{pir}"
    );
    assert!(
        pir.contains("(broadcast record (%kwarg rate"),
        "inner broadcast builds per-cell record(rate = …):\n{pir}"
    );
    assert_eq!(pir.matches("(sum ").count(), 1, "one sum:\n{pir}");
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "one density term (not one-per-element):\n{pir}"
    );
    assert_eq!(
        pir.matches("broadcast").count(),
        2,
        "two broadcasts:\n{pir}"
    );

    // The kernel tag is emitted BARE — the module-qualified ref does not survive as
    // the kernel head (the JS registry keys `ContinuedPoisson` bare).
    assert!(
        !pir.contains("(%ref hepphys ContinuedPoisson)"),
        "module-qualified ref must not survive as the kernel tag:\n{pir}"
    );
    assert!(
        !pir.contains("get0") && !pir.contains("(iid "),
        "must be axis-native, not unrolled:\n{pir}"
    );
    assert!(
        !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl:\n{pir}");
}

// Refuse-don't-mislower survives the module-member generalization: a module
// *function* member as a broadcast head (`hepphys.interp_poly6_exp`) resolves to a
// name, but that name is NOT a distribution (`distribution_param_names` → `None`),
// so it must still refuse — the module namespace does not by itself make a head a
// kernel. Only a module member that resolves to a known §09 distribution lowers.
#[test]
fn broadcast_module_function_head_still_refuses() {
    let src = "\
hepphys = standard_module(\"particle-physics\", \"0.1\")
lo = [0.9, 0.8]
nom = [1.0, 1.0]
hi = [1.1, 1.2]
alpha = elementof(reals)
lp = logdensityof(lawof(broadcast(hepphys.interp_poly6_exp, lo, nom, hi, alpha)), [1.0, 1.0])";
    let err = determinize(&parse_infer(src)).expect_err(
        "a module FUNCTION member used as a broadcast head is not a distribution constructor \
         and must refuse, not mislower",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("not a known distribution constructor"),
        "refusal should name that the module member is not a known distribution constructor: {msg}"
    );
}

// A multi-parameter kernel — `Normal.(mus, sigmas)` — binds its POSITIONAL
// data-args to the constructor's ordered parameter names (`mu`, then `sigma`)
// when building the per-cell record.
#[test]
fn broadcast_normal_binds_positional_args_to_param_names_in_order() {
    let src = "\
mus = [0.0, 1.0, 2.0]
sigmas = [1.0, 2.0, 3.0]
lp = logdensityof(lawof(broadcast(Normal, mus, sigmas)), [0.5, 0.5, 0.5])";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);

    assert!(
        pir.contains("(broadcast builtin_logdensityof Normal"),
        "outer broadcast over the Normal tag:\n{pir}"
    );
    // Positional `mus, sigmas` bind to `mu, sigma` in the constructor's order.
    assert!(
        pir.contains("(broadcast record (%kwarg mu ") && pir.contains("(%kwarg sigma "),
        "per-cell record keyed by mu and sigma:\n{pir}"
    );
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "one density term:\n{pir}"
    );
    assert!(!pir.contains("get0"), "axis-native, not unrolled:\n{pir}");
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl:\n{pir}");
}

// A data-arg passed by KEYWORD (`broadcast(Normal, sigma = …, mu = …)`) keeps its
// given name — the record field names follow the call's keywords, not the
// positional slot order.
#[test]
fn broadcast_keyword_data_args_honor_given_names() {
    let src = "\
mus = [0.0, 1.0]
sigmas = [1.0, 2.0]
lp = logdensityof(lawof(broadcast(Normal, sigma = sigmas, mu = mus)), [0.5, 0.5])";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("(%kwarg sigma ") && pir.contains("(%kwarg mu "),
        "keyword field names honored in the per-cell record:\n{pir}"
    );
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl:\n{pir}");
}

// Refuse-don't-mislower: a value-broadcast (head is a deterministic op such as
// `add`, not a distribution constructor) scored as a measure must REFUSE, never
// be treated as a kernel-broadcast.
#[test]
fn value_broadcast_used_as_measure_refuses() {
    let src = "\
a = [1.0, 2.0]
b = [3.0, 4.0]
lp = logdensityof(lawof(broadcast(add, a, b)), [5.0, 5.0])";
    let err = determinize(&parse_infer(src))
        .expect_err("a value-broadcast used as a measure must refuse, not mislower");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("distribution constructor"),
        "refusal should name that the head is not a distribution constructor: {msg}"
    );
}

// histfactory scores a broadcast-kernel through a REIFIED measure: the model is a
// `functionof` whose BODY is `broadcast(K, params)`, applied as a likelihood at a
// θ point. `functionof(broadcast(Poisson, [lam, lam, lam]), lam = lam)` is a
// kernel with input `lam`; `likelihoodof(k, obs)` then `logdensityof(L, θ)` scores
// it at θ. The determiniser must UNWRAP the reification (its body is `args[0]`; the
// `(lam, %ref self lam)` boundary is a self-ref that the per-query θ-inliner reaches
// through the body's `(%ref self lam)`) so the body reaches the broadcast-kernel
// arm — yielding the same axis-native `sum(broadcast(builtin_logdensityof, …))`, now
// with θ inlined: each `(%ref self lam)` in the per-cell rate becomes 1.5.
#[test]
fn functionof_broadcast_kernel_scores_via_likelihood() {
    let src = "\
lam = elementof(posreals)
k = functionof(broadcast(Poisson, [lam, lam, lam]), lam = lam)
L = likelihoodof(k, [1, 2, 3])
lp = logdensityof(L, record(lam = 1.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);

    // Same axis-native emission as the bare broadcast-kernel: one sum, one density
    // term, the outer density broadcast over the Poisson tag, the inner per-cell
    // record-broadcast keyed by the constructor's param name (`rate`).
    assert!(pir.contains("(sum "), "sum reduction present:\n{pir}");
    assert!(
        pir.contains("(broadcast builtin_logdensityof Poisson"),
        "outer broadcast zips builtin_logdensityof over the Poisson tag:\n{pir}"
    );
    assert!(
        pir.contains("(broadcast record (%kwarg rate"),
        "inner broadcast builds per-cell record(rate = …):\n{pir}"
    );
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "one density term (not one-per-element):\n{pir}"
    );

    // The reification is UNWRAPPED — no `functionof` and no measure layer survive.
    assert!(
        !pir.contains("(functionof ") && !pir.contains("(kernelof "),
        "reification unwrapped:\n{pir}"
    );
    assert!(
        !pir.contains("(logdensityof ")
            && !pir.contains("(likelihoodof ")
            && !pir.contains("lawof"),
        "measure/likelihood layer gone:\n{pir}"
    );

    // θ = record(lam = 1.5) is inlined through the self-ref boundary: each
    // `(%ref self lam)` in the per-cell rate vector becomes 1.5, and no dangling
    // `(%ref self lam)` survives in the scored density.
    assert!(
        pir.contains("(vector 1.5 1.5 1.5)"),
        "θ value 1.5 inlined into the per-cell rate vector:\n{pir}"
    );
    assert!(
        !pir.contains("(%ref self lam)"),
        "no dangling free-param ref left in the density:\n{pir}"
    );

    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl:\n{pir}");
}

// The unwrap generalizes beyond broadcast: a SCALAR reified measure
// `functionof(Normal(mu = mu, sigma = 1.0), mu = mu)` scored via a likelihood
// lowers exactly like the bare-kernel likelihood query — the body reaches
// `build_density_term`, and θ = record(mu = 2.0) inlines into the `mu` field.
#[test]
fn functionof_scalar_measure_scores_via_likelihood() {
    let src = "\
mu = elementof(reals)
k = functionof(Normal(mu = mu, sigma = 1.0), mu = mu)
L = likelihoodof(k, 0.5)
lp = logdensityof(L, record(mu = 2.0))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("builtin_logdensityof"),
        "kernel density present:\n{pir}"
    );
    assert!(
        pir.contains("(%field mu 2.0)"),
        "θ value 2.0 inlined into the mu field via the unwrapped reification:\n{pir}"
    );
    assert!(
        !pir.contains("(functionof ") && !pir.contains("(kernelof "),
        "reification unwrapped:\n{pir}"
    );
    assert!(
        !pir.contains("(likelihoodof ") && !pir.contains("lawof"),
        "measure/likelihood layer gone:\n{pir}"
    );
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl:\n{pir}");
}

// Refuse-don't-mislower survives the reified-measure unwrap. Unwrapping the
// outer `functionof` measure does NOT bypass the θ-capture guard: here the
// unwrapped body `weighted(w, Normal(…))` scores through `w`, and
// `w = functionof(mul(coeff, _x_), …, coeff = coeff)` closes over the θ param
// `coeff` as a reification INPUT — which the per-query θ-inliner cannot reach
// (`substitute_refs_by_name` walks `children()`, excluding a `Call`'s `Inputs`).
// The `subtree_has_theta_capturing_input` guard follows the density's
// `(%ref self w)` into `w`'s RHS, sees the `(coeff, %ref self coeff)` boundary
// entry, and HARD REFUSES rather than score at the free `coeff`.
#[test]
fn functionof_measure_wrapping_theta_capturing_weight_still_refuses() {
    let src = "\
coeff = elementof(reals)
w = functionof(mul(coeff, _x_), x = _x_, coeff = coeff)
k = functionof(weighted(w, Normal(mu = 0.0, sigma = 1.0)), coeff = coeff)
L = likelihoodof(k, 0.5)
lp = logdensityof(L, record(coeff = 2.0))";
    let err = determinize(&parse_infer(src)).expect_err(
        "a θ param captured inside a reification input must refuse even when reached \
         through an unwrapped functionof-as-measure, not silently score at the free param",
    );
    assert!(
        err.reason.contains("reification input")
            && err.reason.contains("refuse rather than mislower"),
        "refusal is the θ-in-reification-input guard: {err:?}"
    );
}
