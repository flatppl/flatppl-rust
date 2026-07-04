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
    let src = "\
rates = elementof(cartpow(posreals))
obs = elementof(cartpow(nonnegintegers))
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
