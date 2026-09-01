//! A pushforward over a DISCRETE base carries no volume element. §06 "Density
//! convention": "All density formulas in this section are with respect to a
//! reference measure implied by the constituent distribution types: Lebesgue for
//! continuous variates, counting measure for discrete variates." A
//! counting-measure density has no volume element, and §06's pushforward
//! `(f_*M)(Y) = M(f⁻¹(Y))` at a singleton gives the atom's mass unchanged — so
//! the density of `pushfwd(f, M)` over a discrete `M` is the base pmf at
//! `f⁻¹(y)` SNAPPED to the lattice, with no `logvol` subtracted.
//!
//! Subtracting one rescales every atom: for `pushfwd(exp, Poisson(3))` the
//! emitted density at atom `y = e^k` would be `pmf(k)·e^{-k}`, totalling
//! `e^{-3}·e^{3/e} = 0.15011378939830683` instead of 1; for the affine and
//! `locscale` forms with `scale = 2` it would total exactly ½.
//!
//! Structural only (flatppl-rust is not a density engine): assert the emitted
//! FlatPDL. Each discrete case is paired with the same map over a CONTINUOUS base,
//! which MUST still carry its volume term — so a change that dropped the volume
//! element everywhere fails here rather than passing vacuously.
use flatppl_determinizer::{determinize, is_flatpdl};

mod common;
use common::{call_arg, pir_binding, pir_head};

/// The density inside a gate — the `ifelse`'s taken arm. The whole emission cannot
/// answer "is there a volume term": the lattice gate's own condition subtracts, so a
/// bare `(sub` search over the emission no longer isolates one.
fn gated_density(out: &str) -> String {
    call_arg(out, "ifelse", 1)
}

fn determinize_src(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    determinize(&m).expect("must lower, not refuse")
}

/// The lowered `lp` binding's FlatPIR text, checked for conformance on the way
/// through. Scoped to the binding (see [`pir_binding`]) so nothing emitted
/// elsewhere can satisfy — or defeat — an assertion about the density term.
fn lp(src: &str) -> String {
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl failed:\n{pir}");
    pir_binding(&pir, "lp")
}

#[test]
fn pushfwd_exp_over_discrete_base_emits_no_volume_term() {
    // `exp`'s forward log-volume is the identity, so the volume term is `logvol`
    // applied at the preimage — which would make the gated density a `sub` of the pmf.
    // Over `Poisson` (variate `%scalar integer`) the gated density IS the pmf.
    let discrete = lp("d = pushfwd(exp, Poisson(rate = 3.0))\nlp = logdensityof(d, 0.5)");
    assert!(
        discrete.contains("builtin_logdensityof Poisson"),
        "the base pmf is still scored:\n{discrete}"
    );
    assert_eq!(
        pir_head(&gated_density(&discrete)),
        "builtin_logdensityof",
        "the gated density IS the pmf — a volume term would make it a `sub`:\n{discrete}"
    );

    // Same map over a CONTINUOUS base (the canonical LogNormal): the volume term
    // is required. Without this the assertions above would pass vacuously.
    let continuous =
        lp("d = pushfwd(exp, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)");
    assert!(
        continuous.contains("(sub "),
        "a Lebesgue reference still subtracts the volume term:\n{continuous}"
    );
    assert_eq!(
        continuous.matches("(log ").count(),
        2,
        "preimage AND volume term, `logdensityof(M, log y) − log y`:\n{continuous}"
    );
}

#[test]
fn pushfwd_affine_over_discrete_base_emits_no_volume_term() {
    // `x -> 2·x + 1` has the constant forward log-volume `log|2|`, emitted as
    // `(log (abs 2.0))`. The preimage `(0.5 − 1)/2` const-folds to -0.25, so the
    // discrete emission is the bare `builtin_logdensityof` at that term. §11
    // "Literal values" forbids a signed atom, so the folded term is the
    // canonical `(neg 0.25)` call, not a bare `-0.25` literal.
    let discrete =
        lp("d = pushfwd(x -> 2.0 * x + 1.0, Poisson(rate = 3.0))\nlp = logdensityof(d, 0.5)");
    assert!(
        discrete.contains("builtin_logdensityof Poisson") && discrete.contains("(neg 0.25)"),
        "the base pmf is scored at the preimage (0.5 − 1)/2, folded to (neg 0.25):\n{discrete}"
    );
    assert!(
        !discrete.contains("(abs 2.0)"),
        "no `log|scale|` volume term over a counting reference:\n{discrete}"
    );
    assert_eq!(
        pir_head(&gated_density(&discrete)),
        "builtin_logdensityof",
        "the gated density IS the pmf:\n{discrete}"
    );

    let continuous = lp(
        "d = pushfwd(x -> 2.0 * x + 1.0, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    );
    assert!(
        continuous.contains("(sub ") && continuous.contains("(abs 2.0)"),
        "a Lebesgue reference still subtracts log|2|:\n{continuous}"
    );
}

#[test]
fn locscale_over_discrete_base_emits_no_volume_term() {
    // §06 gives `locscale(m, shift, scale)` as `pushfwd(x -> scale·x + shift, m)`,
    // so it takes the same split: an affine map over a discrete base relabels the
    // atoms and preserves their mass.
    let discrete = lp("d = locscale(Poisson(rate = 3.0), 1.0, 2.0)\nlp = logdensityof(d, 0.5)");
    // §11 "Literal values" forbids a signed atom: the folded preimage is the
    // canonical `(neg 0.25)` call, not a bare `-0.25` literal.
    assert!(
        discrete.contains("builtin_logdensityof Poisson") && discrete.contains("(neg 0.25)"),
        "the base pmf is scored at the preimage:\n{discrete}"
    );
    assert!(
        !discrete.contains("(abs 2.0)"),
        "no `log|scale|` volume term over a counting reference:\n{discrete}"
    );
    assert_eq!(
        pir_head(&gated_density(&discrete)),
        "builtin_logdensityof",
        "the gated density IS the pmf:\n{discrete}"
    );

    let continuous =
        lp("d = locscale(Normal(mu = 0.0, sigma = 1.0), 1.0, 2.0)\nlp = logdensityof(d, 0.5)");
    assert!(
        continuous.contains("(sub ") && continuous.contains("(abs 2.0)"),
        "a Lebesgue reference still subtracts log|2|:\n{continuous}"
    );
}

#[test]
fn explicit_bijection_over_discrete_base_emits_no_volume_term() {
    // §06's `bijection(f, f_inv, logvolume)` annotation is user-asserted, but the
    // reference measure is not: over a counting reference there is no volume
    // element for the annotation to supply, so the term is dropped here too.
    let discrete = lp(
        "d = pushfwd(bijection(exp, log, x -> x), Poisson(rate = 3.0))\nlp = logdensityof(d, 0.5)",
    );
    assert!(
        discrete.contains("builtin_logdensityof Poisson"),
        "the base pmf is still scored:\n{discrete}"
    );
    assert_eq!(
        pir_head(&gated_density(&discrete)),
        "builtin_logdensityof",
        "an asserted logvol is still not a counting-measure volume element:\n{discrete}"
    );

    let continuous = lp(
        "d = pushfwd(bijection(exp, log, x -> x), Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    );
    assert!(
        continuous.contains("(sub "),
        "the asserted logvol still applies over a Lebesgue reference:\n{continuous}"
    );
}

#[test]
fn pushfwd_over_boolean_variate_base_emits_no_volume_term() {
    // `Bernoulli`'s variate is `%scalar integer` on support `booleans` — discrete
    // by the same catalogue slot as `Poisson`, so it takes the same treatment.
    // Pins that the split reads the variate type rather than enumerating supports.
    let discrete = lp("d = pushfwd(exp, Bernoulli(p = 0.3))\nlp = logdensityof(d, 0.5)");
    assert!(
        discrete.contains("builtin_logdensityof Bernoulli"),
        "the base pmf is still scored:\n{discrete}"
    );
    assert_eq!(
        pir_head(&gated_density(&discrete)),
        "builtin_logdensityof",
        "no volume term over a boolean-support discrete base:\n{discrete}"
    );
}

#[test]
fn pushfwd_over_discrete_vector_variate_base_emits_no_volume_term() {
    // `Multinomial`'s variate is a vector of integers (`(%array 1 (n) (%scalar
    // integer))`), so the reference is counting on ℤⁿ (§06 "Reference measure for
    // product measures": the product of the per-component references) — again no
    // volume element.
    let discrete = lp(
        "d = pushfwd(exp, Multinomial(n = 5, p = [0.2, 0.8]))\nlp = logdensityof(d, [1.0, 4.0])",
    );
    assert!(
        discrete.contains("builtin_logdensityof Multinomial"),
        "the base pmf is still scored:\n{discrete}"
    );
    assert_eq!(
        pir_head(&gated_density(&discrete)),
        "builtin_logdensityof",
        "no volume term over a discrete vector variate:\n{discrete}"
    );
}

#[test]
fn pushfwd_volume_preserving_map_over_discrete_base_is_unaffected() {
    // `neg` is volume-preserving (`logvol = 0`), so it is the control that
    // isolates the change: it emitted `sub(pmf, 0.0)` before and the bare `pmf`
    // now, and the density is the same either way.
    let discrete = lp("d = pushfwd(neg, Poisson(rate = 3.0))\nlp = logdensityof(d, 0.5)");
    assert!(
        discrete.contains("builtin_logdensityof Poisson"),
        "the base pmf is still scored:\n{discrete}"
    );
    assert_eq!(
        pir_head(&gated_density(&discrete)),
        "builtin_logdensityof",
        "a volume-preserving map over a discrete base scores the bare pmf:\n{discrete}"
    );
}

#[test]
fn pushfwd_over_a_base_with_an_unproven_variate_refuses() {
    // `pushfwd`'s own result type is the CODOMAIN of its forward map, which this
    // pass does not track (`(%measure (%domain %any))`), so a pushforward OF a
    // pushforward proves neither reference measure. Fail closed: guessing
    // continuous here is what rescales a discrete measure's atoms (the base could
    // be `pushfwd(neg, Poisson(3))`, whose atoms would pick up `exp`'s Jacobian).
    let mut m = flatppl_syntax::parse(
        "d = pushfwd(exp, pushfwd(neg, Poisson(rate = 3.0)))\nlp = logdensityof(d, 0.5)",
    )
    .unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let e = determinize(&m).expect_err("an unproven reference measure must refuse");
    assert!(
        e.reason.contains("reference measure"),
        "must refuse on the reference measure, got: {}",
        e.reason
    );
}

#[test]
fn a_discrete_pushforward_scores_the_snapped_atom() {
    // Scored AT A TRUE ATOM: `pushfwd(sqrt, Poisson(3))` at `√2 =
    // 1.4142135623730951`. The preimage `y²` is `2.0000000000000004` in floating
    // point, where a pmf is 0 — the atom was missed entirely and the density read as
    // −∞ instead of `logpmf(2, 3.0) = -1.4959226032237258`. So the pmf must be scored
    // at the preimage SNAPPED to the lattice, never at the raw preimage.
    //
    // The `exp` spelling agreed with the truth only by float luck (`log(e²)` is exactly
    // `2.0`), so this is not `sqrt`-specific and must not be fixed per operator.
    let out =
        lp("d = pushfwd(sqrt, Poisson(rate = 3.0))\nlp = logdensityof(d, 1.4142135623730951)");
    let arm = gated_density(&out);
    assert!(
        arm.contains("(round "),
        "the pmf is scored at the SNAPPED preimage:\n{out}"
    );
    assert_eq!(
        pir_head(&arm),
        "builtin_logdensityof",
        "and still with no volume element:\n{out}"
    );
}

#[test]
fn an_off_lattice_query_still_gates_to_minus_infinity() {
    // Snapping alone would score the NEAREST atom at a point the pushforward gives no
    // mass — `pushfwd(sqrt, Poisson(3))` at `y = 1.5` has preimage 2.25, and
    // `round(2.25) = 2` is an atom of the base but `√2 ≠ 1.5`. The pushforward of a
    // counting measure through an injective `f` has atoms only at `{f(k)}`, so the gate
    // is that membership test: `y` is EXACTLY the forward image of the snapped preimage.
    //
    // Round-TRIP through the forward rather than a tolerance on `|x − round(x)|`: an
    // atom's image is produced by evaluating `f` at an integer, so `f(round(f⁻¹(y)))`
    // reproduces it bit for bit, while the inverse leg alone need not (the `√2` case).
    let out = lp("d = pushfwd(sqrt, Poisson(rate = 3.0))\nlp = logdensityof(d, 1.5)");
    assert!(
        out.contains("(iszero ") && out.contains("(neg inf)"),
        "an off-lattice query gates to −∞:\n{out}"
    );
    // §07's `iszero` is the exact-zero test that admits reals; `equal` is restricted to
    // discrete domains, so it cannot spell this.
    assert!(
        !out.contains("(equal "),
        "the lattice test is `iszero`, not `equal`:\n{out}"
    );
}

#[test]
fn a_continuous_base_is_neither_snapped_nor_lattice_gated() {
    // The regression half: the snap and its gate are keyed on the COUNTING reference.
    // A Lebesgue base has no atoms to miss, and snapping one would quantise a density.
    for (map, base) in [
        // `sqrt` needs a non-negative support (§06 case 1), hence `Gamma` here.
        ("sqrt", "Gamma(shape = 2.0, rate = 1.0)"),
        ("exp", "Normal(mu = 0.0, sigma = 1.0)"),
        ("neg", "Normal(mu = 0.0, sigma = 1.0)"),
    ] {
        let out = lp(&format!(
            "d = pushfwd({map}, {base})\nlp = logdensityof(d, 1.5)"
        ));
        assert!(
            !out.contains("(round ") && !out.contains("(iszero "),
            "`{map}` over a continuous base must not be snapped:\n{out}"
        );
    }
}

#[test]
fn locscale_over_a_discrete_base_is_snapped_and_gated_too() {
    // §06 gives `locscale(m, shift, scale)` as `pushfwd(x -> scale·x + shift, m)`, so the
    // relabelled atoms take the same treatment: `(y − shift)/scale` need not land on an
    // integer. `locscale` lowers through its own tail, which is why it needs asserting
    // separately — it carried the defect after the `pushfwd` path was fixed.
    let out = lp("d = locscale(Poisson(rate = 3.0), 1.0, 2.0)\nlp = logdensityof(d, 0.5)");
    assert!(
        out.contains("(round ") && out.contains("(iszero ") && out.contains("(neg inf)"),
        "the location-scale preimage is snapped and gated:\n{out}"
    );
    let continuous = lp("d = locscale(Normal(mu = 0.0, sigma = 1.0), 1.0, 2.0)\n\
                         lp = logdensityof(d, 0.5)");
    assert!(
        !continuous.contains("(round "),
        "and a continuous base is not:\n{continuous}"
    );
}
