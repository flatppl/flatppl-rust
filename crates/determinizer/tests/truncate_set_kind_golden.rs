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
            reason.contains("truncation set's space")
                && reason.contains("set of scalars")
                && reason.contains("a record"),
            "`{set}` must refuse naming both spaces, got: {reason}"
        );
    }
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
    // Refuse on proof, not on absence of evidence. `interval(lo, hi)` over non-literal
    // bounds reads as no value-set at all, and `anything` "signals that no specific
    // type constraint is imposed" (§03) — neither proves a mismatch, so both lower.
    for set in ["interval(lo, inf)", "anything"] {
        let out = lp(&format!(
            "lo = 0.0\n\
             mb = Normal(mu = 0.0, sigma = 1.0)\n\
             x = draw(mb)\n\
             m = truncate(lawof(record(x = x)), {set})\n\
             lp = logdensityof(m, record(x = 0.5))"
        ));
        assert!(
            out.contains("builtin_logdensityof"),
            "`{set}` proves no mismatch and must lower:\n{out}"
        );
    }
}
