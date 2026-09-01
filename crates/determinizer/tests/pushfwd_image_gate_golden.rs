//! A query point OUTSIDE the forward map's image gates to −∞. §06 defines the
//! pushforward as `(f_*M)(Y) = M(f⁻¹(Y))`, so at a `y` outside `f`'s image the
//! preimage is empty, the measure is 0 and the log-density −∞. Ungated, `f⁻¹` is
//! read where it has no preimage and the query returns a FINITE number —
//! `pushfwd(exp, Normal(0,1))` at `y = -0.5` scored the base at `log(-0.5)`.
//!
//! The emitted form is `truncate`'s (`density::gate_point` and `density::gate_density`):
//! the change of variables reads `ifelse(cond, y, witness)` and the result is
//! `ifelse(cond, <change of variables>, neg(inf))`. A gate rather than a refusal because
//! §06 fixes the value there. The sanitised input is not cosmetic: `ifelse` lowers to
//! `stablehlo.select`, which evaluates both operands, and the untaken arm's zero cotangent
//! times a NaN or infinite derivative is NaN.
//!
//! Structural only (flatppl-rust is not a density engine). Each gate is asserted alongside
//! the change of variables it wraps, and an ONTO map is asserted to carry NO gate — so
//! neither a gate that swallowed the density nor one emitted unconditionally passes.
use flatppl_determinizer::{determinize, is_flatpdl};

mod common;
use common::{call_arg, pir_binding, pir_head};

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
            out.matches("(log ").count(),
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
    // A map that carries its base's support ONTO ℝ has nothing to gate: every `y` has
    // a preimage. Without this the gate could be emitted unconditionally and the tests
    // above would still pass, while every affine pushforward carried a vacuous
    // `ifelse`. `log` is onto from `Gamma`'s `[0, ∞)` because `log 0 = −∞`; `neg` is
    // onto only because the base is `reals` — over a positive base it is NOT, and
    // gates (see `a_reflection_over_a_bounded_support_gains_a_gate`).
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
            !out.contains("(neg inf)"),
            "`{map}` is onto — no image gate:\n{out}"
        );
        for gate in ["(in ", "(gt ", "(ge ", "(lt ", "(le "] {
            assert!(
                !out.contains(gate),
                "`{map}` is onto — no `{gate}` condition:\n{out}"
            );
        }
    }
}

#[test]
fn a_reflection_over_a_bounded_support_gains_a_gate() {
    // `neg` carries `Gamma`'s `[0, ∞)` (§08 `nonnegreals`) to `(−∞, 0]`, so it is onto
    // only over an unbounded base. Ungated, `pushfwd(neg, Gamma)` at `y = 0.5` read
    // the base density at `−0.5` and returned a finite number where §06 gives −∞. The
    // endpoint stays CLOSED: 0 is in the support, so `−0` is in the image.
    let out = lp("d = pushfwd(neg, Gamma(shape = 2.0, rate = 1.0))\nlp = logdensityof(d, 0.5)");
    assert!(
        out.contains("(le 0.5 0.0)") && out.contains("(neg inf)"),
        "a reflected positive support gates at 0:\n{out}"
    );
}

#[test]
fn each_registry_image_gates_on_its_own_set() {
    // The endpoint map is per-entry, so pin the set each one gates on rather than
    // only that SOME gate appears: a map copied from the wrong entry would otherwise
    // pass. Over a base whose support is the whole of the map's domain the image is
    // the map's RANGE, which is what the deleted static image column held: `exp`'s
    // is §03's `posreals` = (0, +∞] and `sqrt`'s over a `nonnegreals` base is
    // `nonnegreals` = [0, +∞], both EXACT as §03 sets. The rest have an OPEN endpoint
    // that no §03 set spells (`interval(lo, hi)` "denotes the closed interval";
    // `unitinterval` is [0, 1]), so they gate by comparison: `invlogit`'s and
    // `invprobit`'s (0, 1), `expm1`'s (−1, ∞), `tanh`'s (−1, 1), `atan`'s (−π/2, π/2).
    // Pinned as the whole CONDITION where its shape is flat: `posreals` and
    // `nonnegreals` also appear in the arm's own type annotations, so a bare
    // set-name substring proves nothing about the gate.
    for (map, base, expected) in [
        (
            "exp",
            "Normal(mu = 0.0, sigma = 1.0)",
            vec!["(in 0.5 posreals)"],
        ),
        (
            "sqrt",
            "Gamma(shape = 2.0, rate = 1.0)",
            vec!["(in 0.5 nonnegreals)"],
        ),
        (
            "x -> pow(x, 3.0)",
            "Gamma(shape = 2.0, rate = 1.0)",
            vec!["(in 0.5 nonnegreals)"],
        ),
        (
            "invlogit",
            "Normal(mu = 0.0, sigma = 1.0)",
            vec!["(land ", "(gt 0.5 0.0)", "(lt 0.5 1.0)"],
        ),
        (
            "invprobit",
            "Normal(mu = 0.0, sigma = 1.0)",
            vec!["(land ", "(gt 0.5 0.0)", "(lt 0.5 1.0)"],
        ),
        (
            "expm1",
            "Normal(mu = 0.0, sigma = 1.0)",
            vec!["(gt 0.5 -1.0)"],
        ),
        (
            "tanh",
            "Normal(mu = 0.0, sigma = 1.0)",
            vec!["(land ", "(gt 0.5 -1.0)", "(lt 0.5 1.0)"],
        ),
        (
            "atan",
            "Normal(mu = 0.0, sigma = 1.0)",
            vec!["(land ", "(divide pi 2.0)", "(neg "],
        ),
    ] {
        let out = lp(&format!(
            "d = pushfwd({map}, {base})\nlp = logdensityof(d, 0.5)"
        ));
        for want in &expected {
            assert!(
                out.contains(want) && out.contains("(neg inf)"),
                "`{map}` must gate on its own image — expected `{want}`:\n{out}"
            );
        }
    }
    // And an open image is NOT gated with `in(y, S)`: every §03 set that could stand in
    // is closed, and a closed superset admits the endpoint, where the base density and
    // the volume term are both −∞ and the emitted `sub` is NaN where §06 gives −∞.
    for map in ["invlogit", "invprobit", "expm1", "tanh", "atan"] {
        let out = lp(&format!(
            "d = pushfwd({map}, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)"
        ));
        assert!(
            !out.contains("(in "),
            "`{map}`'s image is open — no closed §03 set may spell it:\n{out}"
        );
    }
}

#[test]
fn composition_gates_on_the_propagated_image() {
    // The support is propagated through the chain INNERMOST-first, so the gate is the
    // composition's OWN image, not the outermost op's superset. `x -> exp(2·x)` over a
    // real base has image (0, ∞) — `2·x` is onto ℝ, so `exp`'s range. The
    // affine-OUTERMOST `x -> 2·exp(x) + 1` has image (1, ∞) and is now gated on it;
    // reading the outermost op alone reported no image at all here.
    let inner_affine = lp(
        "d = pushfwd(x -> exp(2.0 * x), Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    );
    assert!(
        inner_affine.contains("(in 0.5 posreals)") && inner_affine.contains("(neg inf)"),
        "an affine op INSIDE `exp` keeps `exp`'s image:\n{inner_affine}"
    );
    let outer_affine = lp(
        "d = pushfwd(x -> 2.0 * exp(x) + 1.0, Normal(mu = 0.0, sigma = 1.0))\n\
         lp = logdensityof(d, 0.5)",
    );
    assert!(
        outer_affine.contains("(gt 0.5 1.0)") && outer_affine.contains("(neg inf)"),
        "an affine OUTERMOST op gates on the propagated (1, ∞):\n{outer_affine}"
    );
    // An affine SHIFT under `exp` moves the endpoint off `exp`'s own range: `exp(x−5)`
    // over `[0, ∞)` has image `[e⁻⁵, ∞)`, closed because 0 is IN `Gamma`'s support
    // (§08 `nonnegreals`). Reading `exp`'s range instead would gate on (0, ∞).
    let shifted = lp(
        "d = pushfwd(x -> exp(x - 5.0), Gamma(shape = 2.0, rate = 1.0))\n\
         lp = logdensityof(d, 0.5)",
    );
    assert!(
        shifted.contains("(ge 0.5 0.006737946999085467)"),
        "the shift moves the image endpoint to e⁻⁵:\n{shifted}"
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
    // A counting-reference pushforward carries no volume element (§06 "Density
    // convention") but still has no mass outside the image. The support is
    // `nonnegintegers`, whose convex HULL is `[0, ∞)`, so `exp`'s image of it is
    // `[1, ∞)` — the hull's image is a superset of the atoms' image, and the lattice
    // round trip is what cuts it back to `{eᵏ}`. Asserted as the CONDITION, not as a
    // substring: `posreals` also appears in the witness's own type annotation.
    let out = lp("d = pushfwd(exp, Poisson(rate = 3.0))\nlp = logdensityof(d, 0.5)");
    assert!(
        out.contains("(ge 0.5 1.0)") && out.contains("(neg inf)"),
        "a discrete base is gated on the image as well:\n{out}"
    );
    assert_eq!(
        pir_head(&call_arg(&out, "ifelse", 1)),
        "builtin_logdensityof",
        "and still carries no volume element:\n{out}"
    );
}

#[test]
fn the_gated_arm_reads_a_sanitised_point_not_the_query_point() {
    // `ifelse` lowers to `stablehlo.select`, which evaluates both operands (§07: no
    // short-circuit guarantee). The VALUE is safe — select returns one arm. The
    // GRADIENT is not: reverse mode sends a ZERO cotangent to the untaken arm, and
    // `0 · ±inf` and `0 · NaN` are NaN, which propagates into every parameter that arm
    // reaches. Measured with jax 0.4.36: `where(y >= 0, sqrt(y), -inf)` at `y = -0.5`
    // has value −inf and gradient NaN; with the input sanitised, gradient 0.
    //
    // So the change of variables must read `ifelse(cond, y, <witness>)`, never `y`.
    // The witness is the forward op at a point in the base's support — `exp(1.0)` for
    // `pushfwd(exp, Normal)` — whose preimage is that support point, so no dangerous op
    // in the arm sees the excluded value.
    // §11 "Literal values": the surface `-0.5` query point lowers to `(neg
    // 0.5)`, never a signed atom, so the printed gate condition and its
    // matching then-arm both carry that canonical call (three gates in all:
    // the top-level refusal-to-mislower guard, plus one for the preimage and
    // one for the volume term).
    let out = lp("d = pushfwd(exp, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, -0.5)");
    assert_eq!(
        out.matches("(in (%meta ((%scalar real) %fixed reals) (neg 0.5)) posreals)")
            .count(),
        3,
        "every gate condition reads the sanitised negative query point:\n{out}"
    );
    assert_eq!(
        out.matches(
            "(ifelse (%meta ((%scalar boolean) %fixed booleans) \
             (in (%meta ((%scalar real) %fixed reals) (neg 0.5)) posreals)) \
             (%meta ((%scalar real) %fixed reals) (neg 0.5)) "
        )
        .count(),
        2,
        "the preimage and the volume term must both read the raw point in their then-arm:\n{out}"
    );
    assert!(
        !out.contains("-0.5"),
        "no signed atom may reach the printed FlatPIR:\n{out}"
    );
    assert!(
        out.contains("(exp 1.0)"),
        "the witness is the forward op at a support point:\n{out}"
    );
    // Both readings of the point — the preimage and the volume term — take the
    // sanitised one; the raw literal survives only in the gate condition itself.
    assert_eq!(
        out.matches("(exp 1.0)").count(),
        2,
        "preimage AND volume term read the sanitised point:\n{out}"
    );
    // The gate CONDITION still tests the real query point: sanitising it there would
    // make the gate vacuously true.
    assert!(
        out.contains("(in (%meta ((%scalar real) %fixed reals) (neg 0.5)) posreals)"),
        "the condition tests the query point itself:\n{out}"
    );
}

#[test]
fn a_truncation_gate_sanitises_its_point_too() {
    // The same rule on `truncate`, whose gated arm is the base density at an
    // out-of-support point — the dangerous op there is `builtin_logdensityof` itself.
    // One builder, so the two cannot diverge.
    let out = lp(
        "d = truncate(Normal(mu = 0.0, sigma = 1.0), interval(0.0, 1.0))\n\
                  lp = logdensityof(d, 0.5)",
    );
    assert!(
        out.contains("(builtin_logdensityof Normal") && out.contains("(neg inf)"),
        "the truncation still emits its gate:\n{out}"
    );
    assert!(
        out.contains("(ifelse (%meta ((%scalar boolean) %fixed booleans) (in 0.5 "),
        "the base density reads the sanitised point:\n{out}"
    );
    // `Normal`'s support is `reals`, whose witness is 1.
    assert!(
        out.contains(") 0.5 1.0)"),
        "the witness is a point in the base's support:\n{out}"
    );
}

#[test]
fn an_unproven_support_leaves_the_gate_unsanitised() {
    // The regression half: no witness is derivable from a support that names no point,
    // and the gate then emits exactly as it did before — a `None` witness must not
    // become a guessed constant. `lawof(y)`'s support is not tracked here.
    let out = lp("y = draw(Normal(mu = 0.0, sigma = 1.0))\n\
                  d = pushfwd(sqrt, truncate(lawof(y), nonnegreals))\n\
                  lp = logdensityof(d, 0.5)");
    assert!(
        out.contains("nonnegreals") && out.contains("(neg inf)"),
        "the gate is still emitted:\n{out}"
    );
}

#[test]
fn the_image_endpoint_follows_the_bases_support() {
    // The defect this replaced: `sqrt`'s image was the static closed `nonnegreals`
    // whatever the base was. §06 `(f_*M)(Y) = M(f⁻¹(Y))` makes the image `f`'s image
    // of the SUPPORT, so the endpoint's openness is the support's. §08 supports:
    // `InverseGamma` `posreals` (0 excluded), `Gamma` `nonnegreals` (0 included),
    // `Beta` `unitinterval`. A closed superset is not harmless at an excluded
    // endpoint: `y = 0` took the gate and the change of variables differentiated to
    // +inf there (measured, Enzyme-JAX).
    for (base, expected) in [
        (
            "InverseGamma(shape = 5.0, scale = 5.0)",
            "(in 0.5 posreals)",
        ),
        ("Gamma(shape = 2.0, rate = 1.0)", "(in 0.5 nonnegreals)"),
        ("Beta(alpha = 2.0, beta = 2.0)", "(in 0.5 (%meta"),
    ] {
        let out = lp(&format!(
            "d = pushfwd(sqrt, {base})\nlp = logdensityof(d, 0.5)"
        ));
        assert!(
            out.contains(expected) && out.contains("(neg inf)"),
            "`{base}`: expected `{expected}`:\n{out}"
        );
    }
    // The same contrast under ONE forward, so nothing but the support's openness can
    // explain the difference: `exp` carries `LogNormal`'s `posreals` to the OPEN
    // `(1, ∞)` and `Gamma`'s `nonnegreals` to the CLOSED `[1, ∞)`. Both endpoints are
    // finite, which no §03 set spells, so both gate by comparison — `gt` against `ge`
    // is the whole difference.
    for (base, expected) in [
        ("LogNormal(mu = 0.0, sigma = 1.0)", "(gt 0.5 1.0)"),
        ("Gamma(shape = 2.0, rate = 1.0)", "(ge 0.5 1.0)"),
    ] {
        let out = lp(&format!(
            "d = pushfwd(exp, {base})\nlp = logdensityof(d, 0.5)"
        ));
        assert!(
            out.contains(expected) && out.contains("(neg inf)"),
            "`exp` over `{base}`: expected `{expected}`:\n{out}"
        );
    }
    // `Beta`'s `[0, 1]` maps to `[0, 1]` — bounded and closed at both ends, which §03
    // spells `interval(lo, hi)` ("denotes the closed interval"). Not `unitinterval`,
    // the same set: the StableHLO `in` lowering does not take that name.
    let beta = lp("d = pushfwd(sqrt, Beta(alpha = 2.0, beta = 2.0))\nlp = logdensityof(d, 0.5)");
    assert!(
        beta.contains("(interval 0.0 1.0)") && !beta.contains("unitinterval"),
        "a bounded closed image is an `interval`:\n{beta}"
    );
    // Both spellings §06 declares equivalent read the same support.
    let bare = lp(
        "d = pushfwd(sqrt, InverseGamma(shape = 5.0, scale = 5.0))\n\
         lp = logdensityof(d, 0.5)",
    );
    let lambda = lp(
        "d = pushfwd(x -> sqrt(x), InverseGamma(shape = 5.0, scale = 5.0))\n\
         lp = logdensityof(d, 0.5)",
    );
    assert_eq!(bare, lambda, "bare vs lambda spelling over an open support");
}

#[test]
fn a_truncated_base_maps_both_endpoints() {
    // `truncate(M, S)`'s support lies inside `S` (§06 "Support restriction"), so both
    // endpoints are finite and `exp` carries `[0, 5]` to `[e⁰, e⁵]`. The static image
    // gated this on (0, ∞), which admits every point below 1.
    let out = lp(
        "d = pushfwd(exp, truncate(Normal(mu = 0.0, sigma = 1.0), interval(0.0, 5.0)))\n\
         lp = logdensityof(d, 0.5)",
    );
    assert!(
        out.contains("(interval 1.0 148.4131591025766)"),
        "the image is the mapped truncation interval:\n{out}"
    );
}

#[test]
fn a_support_disjoint_from_the_forwards_domain_emits_no_image_gate() {
    // A support intersected with the forward's §06 domain can be EMPTY, and then there
    // is no image to gate on (`Extent::nonempty`). Reachable only through the explicit
    // `bijection` spelling, which skips the domain check a synthesised forward refuses
    // on first.
    //
    // The LOAD-BEARING case is a map defined OUTSIDE its §06 domain. `pow`'s domain is
    // `nonnegreals`, but `x³` is perfectly finite on negatives, so over
    // `interval(-5, -1)` the empty intersection `(Open(0), Closed(-1))` maps to two
    // FINITE endpoints and survives as an inverted extent. Ungarded it emits
    // `0.5 in interval(0.0, -1.0)` — endpoints the wrong way round, which the StableHLO
    // `in` lowering reads as the COMPLEMENT of the intended set (its closed-interval
    // identity is `(v − lo)·(hi − v) >= 0`, non-negative BETWEEN an inverted pair).
    let odd_pow = lp(
        "d = pushfwd(bijection(x -> pow(x, 3.0), x -> pow(x, 0.3333333333333333), \
         x -> add(log(3.0), mul(2.0, log(x)))), \
         truncate(Normal(mu = 0.0, sigma = 1.0), interval(-5.0, -1.0)))\n\
         lp = logdensityof(d, 0.5)",
    );
    assert!(
        !odd_pow.contains("(interval 0.0 -1.0)"),
        "an empty extent must not be spelled as a backwards interval:\n{odd_pow}"
    );
    // The change of variables is left UNGATED: the emission IS §06's subtraction,
    // `logdensityof(M, f_inv(v)) - logvol(f_inv(v))`. Without this the assertion above
    // would also pass on an emission that gated on some other wrong set.
    assert_eq!(
        pir_head(&call_arg(&odd_pow, "%bind", 1)),
        "sub",
        "an empty image must leave the change of variables ungated:\n{odd_pow}"
    );
    // A map that is UNDEFINED outside its domain reaches the same check but cannot show
    // it: `log`'s out-of-domain endpoint maps to NaN and collapses to `Unbounded`, so
    // this emitted no gate before the extent walk existed either. Kept because it is the
    // shape the guard's doc names, and it must stay ungated.
    let out_of_domain = lp("d = pushfwd(bijection(log, exp, x -> neg(log(x))), \
         truncate(Normal(mu = 0.0, sigma = 1.0), interval(-3.0, -1.0)))\n\
         lp = logdensityof(d, 0.5)");
    assert_eq!(
        pir_head(&call_arg(&out_of_domain, "%bind", 1)),
        "sub",
        "an out-of-domain support must leave the change of variables ungated:\n{out_of_domain}"
    );
    // Neither tests the QUERY point. The base truncation's own gate survives in both,
    // and it reads the PREIMAGE (`exp(0.5)`, `0.5 ^ (1/3)`), never `0.5`.
    for out in [&odd_pow, &out_of_domain] {
        assert!(
            !out.contains("(in 0.5 ") && !out.contains("(gt 0.5 ") && !out.contains("(ge 0.5 "),
            "no image gate may test the query point:\n{out}"
        );
    }
}

#[test]
fn an_unproven_support_gates_on_the_maps_range() {
    // The fallback, and the no-regression half: a base whose support the pass cannot
    // read — `Uniform`'s support is its set ARGUMENT, unproven when the bounds are
    // model inputs — gates on the forward's own RANGE, which is exactly the static
    // image this walk replaced (`exp` → §03 `posreals`). A domain-RESTRICTED forward
    // never reaches here: §06 case 1 refuses it on an unproven support first.
    let unproven = lp("a = elementof(reals)\nb = elementof(reals)\n\
         d = pushfwd(exp, Uniform(interval(a, b)))\nlp = logdensityof(d, 0.5)\n\
         inputs = (a, b)\noutputs = (lp)");
    assert!(
        unproven.contains("(in 0.5 posreals)") && unproven.contains("(neg inf)"),
        "an unproven support gates on `exp`'s range:\n{unproven}"
    );
    // And a PROVEN structural support is read: the same constructor over `nonnegreals`
    // gates on the mapped endpoint instead.
    let proven = lp("d = pushfwd(exp, Uniform(nonnegreals))\nlp = logdensityof(d, 0.5)");
    assert!(
        proven.contains("(ge 0.5 1.0)"),
        "a structural support still maps:\n{proven}"
    );
}
