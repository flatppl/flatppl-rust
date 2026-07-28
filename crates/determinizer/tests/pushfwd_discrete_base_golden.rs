//! A pushforward over a DISCRETE base carries no volume element. §06 "Density
//! convention": "All density formulas in this section are with respect to a
//! reference measure implied by the constituent distribution types: Lebesgue for
//! continuous variates, counting measure for discrete variates." A
//! counting-measure density has no volume element, and §06's pushforward
//! `(f_*M)(Y) = M(f⁻¹(Y))` at a singleton gives the atom's mass unchanged — so
//! the density of `pushfwd(f, M)` over a discrete `M` is the base pmf at
//! `f⁻¹(y)`, with no `logvol` subtracted.
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

/// The `(%bind <name> …)` form for `name`, delimited by its own matching paren.
fn pir_binding(pir: &str, name: &str) -> String {
    let open = format!("(%bind {name} ");
    let start = pir
        .find(&open)
        .unwrap_or_else(|| panic!("no `{name}` binding in:\n{pir}"));
    let mut depth = 0usize;
    for (i, ch) in pir[start..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return pir[start..start + i + 1].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unterminated `{name}` binding in:\n{pir}")
}

#[test]
fn pushfwd_exp_over_discrete_base_emits_no_volume_term() {
    // `exp`'s forward log-volume is the identity, so the volume term is `logvol`
    // applied at the preimage `log(0.5)` — a SECOND `(log 0.5)` beside the one
    // inside `builtin_logdensityof`. Over `Poisson` (variate `%scalar integer`)
    // neither the `sub` nor that second occurrence may appear.
    let discrete = lp("d = pushfwd(exp, Poisson(rate = 3.0))\nlp = logdensityof(d, 0.5)");
    assert!(
        discrete.contains("builtin_logdensityof Poisson"),
        "the base pmf is still scored:\n{discrete}"
    );
    assert!(
        !discrete.contains("(sub "),
        "no change-of-variables subtraction over a counting reference:\n{discrete}"
    );
    assert_eq!(
        discrete.matches("(log 0.5)").count(),
        1,
        "exactly one `log(0.5)` — the preimage; the volume term would add a second:\n{discrete}"
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
        continuous.matches("(log 0.5)").count(),
        2,
        "preimage AND volume term, `logdensityof(M, log y) − log y`:\n{continuous}"
    );
}

#[test]
fn pushfwd_affine_over_discrete_base_emits_no_volume_term() {
    // `x -> 2·x + 1` has the constant forward log-volume `log|2|`, emitted as
    // `(log (abs 2.0))`. The preimage `(0.5 − 1)/2` const-folds to `-0.25`, so the
    // discrete emission is the bare `builtin_logdensityof` at that literal.
    let discrete =
        lp("d = pushfwd(x -> 2.0 * x + 1.0, Poisson(rate = 3.0))\nlp = logdensityof(d, 0.5)");
    assert!(
        discrete.contains("builtin_logdensityof Poisson") && discrete.contains(" -0.25)"),
        "the base pmf is scored at the preimage (0.5 − 1)/2:\n{discrete}"
    );
    assert!(
        !discrete.contains("(sub ") && !discrete.contains("(abs 2.0)"),
        "no `log|scale|` volume term over a counting reference:\n{discrete}"
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
    assert!(
        discrete.contains("builtin_logdensityof Poisson") && discrete.contains(" -0.25)"),
        "the base pmf is scored at the preimage:\n{discrete}"
    );
    assert!(
        !discrete.contains("(sub ") && !discrete.contains("(abs 2.0)"),
        "no `log|scale|` volume term over a counting reference:\n{discrete}"
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
    assert!(
        !discrete.contains("(sub "),
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
        discrete.contains("builtin_logdensityof Bernoulli") && !discrete.contains("(sub "),
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
        discrete.contains("builtin_logdensityof Multinomial") && !discrete.contains("(sub "),
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
        discrete.contains("builtin_logdensityof Poisson") && !discrete.contains("(sub "),
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
