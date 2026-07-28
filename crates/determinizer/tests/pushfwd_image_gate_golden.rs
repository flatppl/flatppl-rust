//! A query point OUTSIDE the forward map's image gates to −∞. §06 defines the
//! pushforward as `(f_*M)(Y) = M(f⁻¹(Y))`, so at a `y` outside `f`'s image the
//! preimage is empty, the measure is 0 and the log-density −∞. Ungated, `f⁻¹` is
//! read where it has no preimage and the query returns a FINITE number —
//! `pushfwd(exp, Normal(0,1))` at `y = -0.5` scored the base at `log(-0.5)`.
//!
//! The emitted form is `truncate`'s: `ifelse(in(y, S), <change of variables>,
//! neg(inf))` (one builder, `density::gate_outside_set`). It is a gate rather than
//! a refusal because §06 fixes the value there — refusing would deny a query whose
//! answer the spec gives.
//!
//! Structural only (flatppl-rust is not a density engine): assert the emitted
//! FlatPDL. Each gate is asserted alongside the change of variables it wraps, and
//! an ONTO map is asserted to carry no gate — so neither a gate that swallowed the
//! density nor one emitted unconditionally passes here.
use flatppl_determinizer::{determinize, is_flatpdl};

mod common;
use common::pir_binding;

fn determinize_src(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    determinize(&m).expect("must lower, not refuse")
}

/// The lowered `lp` binding's FlatPIR text, conformance-checked on the way through.
/// Scoped to the binding so nothing emitted elsewhere can satisfy — or defeat — an
/// assertion about the density term.
fn lp(src: &str) -> String {
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl failed:\n{pir}");
    pir_binding(&pir, "lp")
}

#[test]
fn pushfwd_exp_gates_the_query_point_to_the_positive_reals() {
    // `exp`'s image is (0, ∞) = §03's `posreals`. The gate is emitted for every
    // query point, in or out of the image — it is a statically emitted `ifelse`,
    // not a compile-time decision on the literal — so both cases carry it, and the
    // change of variables inside is unchanged either way.
    for point in ["-0.5", "0.5"] {
        let src = format!(
            "d = pushfwd(exp, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, {point})"
        );
        let out = lp(&src);
        assert!(
            out.contains("(in ") && out.contains("posreals") && out.contains("(neg inf)"),
            "`{point}`: expected `ifelse(in(y, posreals), …, neg(inf))`:\n{out}"
        );
        // The regression half: the §06 change of variables is untouched by the gate
        // — `logdensityof(Normal, log y) − log y`, i.e. two `log y` occurrences.
        assert!(
            out.contains("(sub ") && out.contains("builtin_logdensityof Normal"),
            "`{point}`: the change of variables must survive the gate:\n{out}"
        );
        assert_eq!(
            out.matches(&format!("(log {point})")).count(),
            2,
            "`{point}`: preimage AND volume term, `logdensityof(M, log y) − log y`:\n{out}"
        );
    }
}

#[test]
fn image_gate_is_byte_identical_across_the_two_spellings() {
    // §06 nowhere distinguishes `pushfwd(g, M)` from `pushfwd(x -> g(x), M)`, and
    // `bijection`'s result "is a function that is semantically `f`" — so the gate
    // must come off `f` alone, never off which spelling supplied the inverse. The
    // annotation records an inverse and a log-volume, never an image, so a gate
    // read from the annotation instead of from `f` would drop here.
    let base = "Normal(mu = 0.0, sigma = 1.0)";
    let bare = lp(&format!(
        "d = pushfwd(exp, {base})\nlp = logdensityof(d, -0.5)"
    ));
    let lambda = lp(&format!(
        "d = pushfwd(x -> exp(x), {base})\nlp = logdensityof(d, -0.5)"
    ));
    let explicit = lp(&format!(
        "d = pushfwd(bijection(exp, log, x -> x), {base})\nlp = logdensityof(d, -0.5)"
    ));
    assert_eq!(bare, lambda, "bare vs lambda spelling");
    assert_eq!(bare, explicit, "bare vs explicit bijection");
}

#[test]
fn onto_forward_maps_emit_no_gate() {
    // A map whose image is all of ℝ has nothing to gate: every `y` has a preimage.
    // Without this the gate could be emitted unconditionally and the tests above
    // would still pass, while every affine pushforward carried a vacuous `ifelse`.
    for (map, base) in [
        ("neg", "Normal(mu = 0.0, sigma = 1.0)"),
        ("log", "Gamma(shape = 2.0, rate = 1.0)"),
        ("sinh", "Normal(mu = 0.0, sigma = 1.0)"),
        ("x -> 2.0 * x + 1.0", "Normal(mu = 0.0, sigma = 1.0)"),
    ] {
        let out = lp(&format!(
            "d = pushfwd({map}, {base})\nlp = logdensityof(d, 0.5)"
        ));
        assert!(
            !out.contains("(in ") && !out.contains("(neg inf)"),
            "`{map}` is onto — no image gate:\n{out}"
        );
    }
}

#[test]
fn each_registry_image_gates_on_its_own_set() {
    // The image column is per-entry, so pin the set each one gates on rather than
    // only that SOME gate appears: an image copied from the wrong entry would
    // otherwise pass. `sqrt`'s and `pow`'s is §06's `nonnegreals`, `invlogit`'s and
    // `invprobit`'s the unit interval, `expm1`'s (−1, ∞), `tanh`'s (−1, 1),
    // `atan`'s (−π/2, π/2).
    for (map, base, expected) in [
        ("exp", "Normal(mu = 0.0, sigma = 1.0)", "posreals"),
        ("sqrt", "Gamma(shape = 2.0, rate = 1.0)", "nonnegreals"),
        (
            "x -> pow(x, 3.0)",
            "Gamma(shape = 2.0, rate = 1.0)",
            "nonnegreals",
        ),
        ("invlogit", "Normal(mu = 0.0, sigma = 1.0)", "unitinterval"),
        ("invprobit", "Normal(mu = 0.0, sigma = 1.0)", "unitinterval"),
        (
            "expm1",
            "Normal(mu = 0.0, sigma = 1.0)",
            "(interval -1.0 inf)",
        ),
        (
            "tanh",
            "Normal(mu = 0.0, sigma = 1.0)",
            "(interval -1.0 1.0)",
        ),
        ("atan", "Normal(mu = 0.0, sigma = 1.0)", "(divide pi 2.0)"),
    ] {
        let out = lp(&format!(
            "d = pushfwd({map}, {base})\nlp = logdensityof(d, 0.5)"
        ));
        assert!(
            out.contains(expected) && out.contains("(neg inf)"),
            "`{map}` must gate on `{expected}`:\n{out}"
        );
    }
}

#[test]
fn composition_gates_on_the_outermost_ops_image() {
    // The gate for `gₙ∘…∘g₁` is `image(gₙ)`, a SUPERSET of the composition's own
    // image — exact when the inner ops are affine (onto), and never a false −∞
    // otherwise. `x -> exp(2·x)` has image (0, ∞) exactly; the affine-outermost
    // `x -> 2·exp(x) + 1` has image (1, ∞) and is not gated at all, pending the
    // forward interval propagation the chain domain guard also waits on.
    let inner_affine = lp(
        "d = pushfwd(x -> exp(2.0 * x), Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    );
    assert!(
        inner_affine.contains("posreals") && inner_affine.contains("(neg inf)"),
        "an affine op INSIDE `exp` keeps `exp`'s image:\n{inner_affine}"
    );
    let outer_affine = lp(
        "d = pushfwd(x -> 2.0 * exp(x) + 1.0, Normal(mu = 0.0, sigma = 1.0))\n\
         lp = logdensityof(d, 0.5)",
    );
    assert!(
        !outer_affine.contains("(in "),
        "an affine OUTERMOST op reports no image here:\n{outer_affine}"
    );
}

#[test]
fn elementwise_image_gate_is_the_cartesian_power() {
    // Over a vector variate an elementwise map carries its per-cell image in every
    // cell, so the image is `cartpow(image(g), n)` (§03: the set of n-element arrays
    // over that set). Both spellings of the same map gate on the same set.
    let base = "iid(Normal(mu = 0.0, sigma = 1.0), 3)";
    for map in ["exp", "fn(broadcast(exp, _))"] {
        let out = lp(&format!(
            "d = pushfwd({map}, {base})\nlp = logdensityof(d, [0.5, 0.6, 0.7])"
        ));
        assert!(
            out.contains("(cartpow posreals 3)") && out.contains("(neg inf)"),
            "`{map}`: expected a `cartpow(posreals, 3)` gate:\n{out}"
        );
    }
}

#[test]
fn matrix_affine_emits_no_gate() {
    // `mu + L * x` with square `L` is onto ℝⁿ — nothing to gate. Pins that the
    // vector arm reads the elementwise shape rather than gating every vector map.
    let out = lp("mu = [0.0, 0.0]\n\
         L = [[1.0, 0.0], [0.0, 1.0]]\n\
         d = pushfwd(x -> mu + L * x, iid(Normal(0.0, 1.0), 2))\n\
         lp = logdensityof(d, [0.5, 0.5])");
    assert!(
        !out.contains("(in ") && !out.contains("(neg inf)"),
        "a matrix-affine map is onto — no image gate:\n{out}"
    );
}

#[test]
fn discrete_base_is_gated_too() {
    // The image is a property of the forward map, not of the reference measure: a
    // counting-reference pushforward carries no volume element (§06 "Density
    // convention") but still has no mass outside the image.
    let out = lp("d = pushfwd(exp, Poisson(rate = 3.0))\nlp = logdensityof(d, 0.5)");
    assert!(
        out.contains("posreals") && out.contains("(neg inf)"),
        "a discrete base is gated on the image as well:\n{out}"
    );
    assert!(
        !out.contains("(sub "),
        "and still carries no volume element:\n{out}"
    );
}
