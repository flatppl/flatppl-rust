//! A pushforward scored at a variate of the WRONG KIND refuses. A scalar law
//! scored at a record emitted ill-typed FlatPDL that `is_flatpdl` (phase/type-based)
//! passes: `pushfwd(exp, Normal(0,1))` at `record(a = 1.0)` lowered to
//! `sub(builtin_logdensityof(Normal, …, log(record(a = 1.0))), log(record(a = 1.0)))`
//! — `log` fed a record.
//!
//! Unlike a query outside the forward's image (§06 fixes that value at −∞), there is
//! no correct value to emit here: every forward map reaching the change of variables
//! preserves the variate's kind (§06 case 2's projection, the one kind-changing form,
//! is dispatched earlier), so the query is ill-formed. Refuse.
//!
//! Two guards, at different reaches: the query point's kind against the base's
//! variate kind, checked on the ORIGINAL typed `v`; and, where a type is unknown so
//! that check cannot fire, a bare built-in operator applied to a record — §04
//! auto-splatting whose correspondence the determiniser cannot verify (the
//! sample-side rule in `sample::lower_pushfwd_sample`).
use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn refusal(src: &str) -> String {
    let e = determinize(&parse_infer(src)).expect_err("must refuse, not lower");
    e.reason
}

#[test]
fn scalar_law_scored_at_a_record_refuses() {
    // The live case: byte-identical output on both spellings before this guard.
    for map in ["exp", "x -> exp(x)", "bijection(exp, log, x -> x)"] {
        let reason = refusal(&format!(
            "b = pushfwd({map}, Normal(mu = 0.0, sigma = 1.0))\n\
             lp = logdensityof(b, record(a = 1.0))"
        ));
        assert!(
            reason.contains("query point") || reason.contains("record"),
            "`{map}` at a record must refuse on the variate mismatch, got: {reason}"
        );
    }
}

#[test]
fn scalar_law_scored_at_a_vector_refuses() {
    // Same defect, other kind: a scalar-domain base scored at a vector would emit
    // `log([…])` and subtract a vector from a scalar.
    let reason = refusal(
        "b = pushfwd(exp, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(b, [1.0, 2.0])",
    );
    assert!(
        reason.contains("query point"),
        "a vector query point must refuse on the variate mismatch, got: {reason}"
    );
}

#[test]
fn vector_law_scored_at_a_scalar_refuses() {
    // And the mismatch in the other direction.
    let reason = refusal(
        "b = pushfwd(fn(broadcast(exp, _)), iid(Normal(mu = 0.0, sigma = 1.0), 3))\n\
         lp = logdensityof(b, 0.5)",
    );
    assert!(
        reason.contains("query point"),
        "a scalar query point against a vector base must refuse, got: {reason}"
    );
}

#[test]
fn matching_variate_kinds_still_lower() {
    // The regression half: the guard fires on a DEFINITE kind mismatch only.
    for (src, expect) in [
        (
            "b = pushfwd(exp, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(b, 0.5)",
            "builtin_logdensityof Normal",
        ),
        (
            "b = pushfwd(fn(broadcast(exp, _)), iid(Normal(mu = 0.0, sigma = 1.0), 3))\n\
             lp = logdensityof(b, [0.5, 0.6, 0.7])",
            "builtin_logdensityof Normal",
        ),
        (
            "j = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Normal(mu = 0.0, sigma = 1.0))\n\
             b = pushfwd(fn(get(_, [\"a\"])), j)\n\
             lp = logdensityof(b, record(a = 0.5))",
            "builtin_logdensityof Normal",
        ),
    ] {
        let out = determinize(&parse_infer(src)).expect("must lower, not refuse");
        let pir = flatppl_flatpir::write(&out);
        assert!(pir.contains(expect), "expected `{expect}` in:\n{pir}");
    }
}

#[test]
fn bare_operator_at_a_record_refuses_when_the_kind_check_cannot_fire() {
    // `pushfwd`'s own result type is its forward map's CODOMAIN, which this pass does
    // not track, so a pushforward OF a pushforward has an unproven variate: the kind
    // check reads `None` and passes. The bare-operator arm is what refuses here, and
    // it must do so BEFORE the reference-measure refusal that would otherwise report
    // this program — the inverse is applied first, so the record is what breaks first.
    let reason = refusal(
        "b = pushfwd(exp, pushfwd(neg, Poisson(rate = 3.0)))\n\
         lp = logdensityof(b, record(a = 1.0))",
    );
    assert!(
        reason.contains("auto-splatting") && reason.contains("argument names"),
        "expected the bare-operator record refusal, got: {reason}"
    );
}
