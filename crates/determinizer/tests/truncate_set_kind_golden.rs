//! `truncate(M, S)` refuses when `S`'s space is provably not `M`'s variate's.
//!
//! **The −∞ this replaces was the CORRECT number, so do not "restore" it.** §06
//! "Support restriction" gives `truncate(M, S)` as ν(A) = M(A ∩ S). With a
//! `record(x: real)` variate and a scalar `interval`, A ∩ S is empty, ν is the zero
//! measure, and −∞ is its density at every point. The lowering emitted
//! `ifelse(record(x = …) in interval(lo, hi), …, -inf)` — a gate false everywhere,
//! silently, forever.
//!
//! The change diagnoses an ill-typed restriction, not a wrong value. Nothing licenses
//! the other repair: §03 "Sets" spells no record-valued `interval`, and §04 "Calling
//! conventions"' auto-splat does not reach a record "given alongside other arguments"
//! — `truncate(M, S)` has two — so the gate must NOT project the record's field.
use flatppl_determinizer::determinize;

mod common;
use common::pir_binding;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn refusal(src: &str) -> String {
    let e = determinize(&parse_infer(src)).expect_err("must refuse, not lower");
    e.reason
}

/// The lowered `lp` binding's FlatPIR text.
fn lp(src: &str) -> String {
    let out = determinize(&parse_infer(src)).expect("must lower, not refuse");
    pir_binding(&flatppl_flatpir::write(&out), "lp")
}

#[test]
fn record_variate_truncated_by_a_scalar_set_refuses() {
    // The live case, in both set spellings: a set-constructor call and a §03 named
    // set. Both lowered to an always-false gate before this guard.
    for set in ["interval(0.0, inf)", "posreals"] {
        let reason = refusal(&format!(
            "mb = Normal(mu = 0.0, sigma = 1.0)\n\
             x = draw(mb)\n\
             m = truncate(lawof(record(x = x)), {set})\n\
             lp = logdensityof(m, record(x = 0.5))"
        ));
        assert!(
            reason.contains("truncate's set")
                && reason.contains("set of scalars")
                && reason.contains("a record"),
            "`{set}` must refuse naming both spaces, got: {reason}"
        );
    }
}

#[test]
fn a_preset_set_binding_refuses_through_the_ref() {
    // The one level of ref indirection `truncation_set_kind` resolves: the set argument
    // is a `%ref` to a preset binding, not the constructor call.
    let reason = refusal(
        "S = interval(0.0, inf)\n\
         mb = Normal(mu = 0.0, sigma = 1.0)\n\
         x = draw(mb)\n\
         m = truncate(lawof(record(x = x)), S)\n\
         lp = logdensityof(m, record(x = 0.5))",
    );
    assert!(
        reason.contains("set of scalars") && reason.contains("a record"),
        "a preset set binding must refuse through the ref, got: {reason}"
    );
}

#[test]
fn scalar_variate_truncated_by_a_vector_set_refuses() {
    // The mismatch in the other direction, and the other kind.
    let reason = refusal(
        "m = truncate(Normal(mu = 0.0, sigma = 1.0), cartpow(interval(0.0, inf), 3))\n\
         lp = logdensityof(m, 0.5)",
    );
    assert!(
        reason.contains("set of vectors") && reason.contains("a scalar"),
        "a vector set over a scalar variate must refuse, got: {reason}"
    );
}

#[test]
fn matching_spaces_still_lower() {
    // The regression half. `lower_truncate` is on the shared gate path, so a
    // too-broad check reddens the whole `truncate` family — pin each kind that has a
    // §03 set spelling of its own.
    for (label, src) in [
        (
            "scalar variate, scalar interval",
            "m = truncate(Normal(mu = 0.0, sigma = 1.0), interval(0.0, inf))\n\
             lp = logdensityof(m, 0.5)",
        ),
        (
            "scalar variate, named set",
            "m = truncate(Normal(mu = 0.0, sigma = 1.0), posreals)\n\
             lp = logdensityof(m, 0.5)",
        ),
        (
            // §03: the keyword form of `cartprod` "produces a set of records".
            "record variate, record set",
            "mb = Normal(mu = 0.0, sigma = 1.0)\n\
             x = draw(mb)\n\
             m = truncate(lawof(record(x = x)), cartprod(x = interval(0.0, inf)))\n\
             lp = logdensityof(m, record(x = 0.5))",
        ),
        (
            "vector variate, cartesian power",
            "m = truncate(iid(Normal(mu = 0.0, sigma = 1.0), 3), cartpow(interval(0.0, inf), 3))\n\
             lp = logdensityof(m, [0.5, 0.6, 0.7])",
        ),
    ] {
        let out = lp(src);
        assert!(
            out.contains("builtin_logdensityof") && out.contains("(in "),
            "{label} must keep its gated lowering:\n{out}"
        );
    }
}

#[test]
fn an_unprovable_set_kind_still_lowers() {
    // Refuse on proof, not on absence of evidence. Each set below fails to prove a
    // kind, so none of them refuses — against variates a PROVEN kind would refuse.
    let record_variate = |set: &str| {
        format!(
            "lo = 0.0\n\
             mb = Normal(mu = 0.0, sigma = 1.0)\n\
             x = draw(mb)\n\
             m = truncate(lawof(record(x = x)), {set})\n\
             lp = logdensityof(m, record(x = 0.5))"
        )
    };
    for (label, src) in [
        // Non-literal bounds: inference reads no value-set off the `interval` at all.
        ("non-literal interval", record_variate("interval(lo, inf)")),
        // §03: `anything` "signals that no specific type constraint is imposed".
        ("anything", record_variate("anything")),
        // §03: `rngstates` members are "algorithm-dependent opaque values".
        ("rngstates", record_variate("rngstates")),
        (
            // §03 makes a power over a record set the set of tables, but
            // `ValueSet::natural_of` gives an array-OF-records type the identical
            // value-set — so it proves neither Table nor Vector, and a scalar variate
            // (which both would refuse) must still lower.
            "cartpow over a record set",
            "m = truncate(Normal(mu = 0.0, sigma = 1.0), \
               cartpow(cartprod(a = reals, b = posreals), 2))\n\
             lp = logdensityof(m, 0.5)"
                .to_string(),
        ),
    ] {
        let out = lp(&src);
        assert!(
            out.contains("builtin_logdensityof"),
            "{label} proves no mismatch and must lower:\n{out}"
        );
    }
}
