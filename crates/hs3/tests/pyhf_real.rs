use flatppl_syntax::{Syntax, parse, print_with};

const FIXTURE_2BIN: &str = include_str!("fixtures/2bin_1channel.json");
const FIXTURE_MULTICHAN_OLD: &str = include_str!("fixtures/multichan_old.json");

/// The canonical pyhf 2-bin 1-channel workspace (new format) must convert fully.
#[test]
fn two_bin_one_channel_converts() {
    let m = flatppl_hs3::read(FIXTURE_2BIN).expect("2bin_1channel.json must parse and convert");
    let text = print_with(&m, Syntax::Minimal);
    assert!(
        text.contains("broadcast(Poisson"),
        "missing Poisson obs model, got:\n{text}"
    );
    assert!(
        text.contains("ContinuedPoisson"),
        "missing shapesys aux term, got:\n{text}"
    );
    assert!(
        text.contains("joint_likelihood"),
        "missing joint_likelihood, got:\n{text}"
    );
    // observed data [50.0, 60.0] must appear as the exact in-order vector, bound
    // as the channel's observed counts and fed to its observation term (pinned
    // bracketed form catches a reordered observation).
    assert!(
        text.contains("singlechannel_observed = [50.0, 60.0]")
            && text.contains(
                "singlechannel_likelihood = \
                 likelihoodof(singlechannel_model, singlechannel_observed)"
            ),
        "observed data [50.0, 60.0] not on main term, got:\n{text}"
    );
    // no lambda/fn — must be point-free
    assert!(!text.contains("fn("), "must be point-free, got:\n{text}");
}

/// The old-format multi-channel workspace with normsys, lumi, staterror must
/// convert end-to-end (no Err).
#[test]
fn multichan_old_converts() {
    let m = flatppl_hs3::read(FIXTURE_MULTICHAN_OLD)
        .expect("multichan_old.json must convert end-to-end");
    let text = print_with(&m, Syntax::Minimal);

    // Must have the joint_likelihood binding
    assert!(
        text.contains("joint_likelihood"),
        "missing joint_likelihood, got:\n{text}"
    );

    // Normal appears for normsys aux and lumi aux and staterror aux
    assert!(
        text.contains("Normal"),
        "missing Normal (normsys/lumi/staterror aux), got:\n{text}"
    );

    // hepphys.interp from normsys
    assert!(
        text.contains("hepphys.interp"),
        "missing hepphys interp fn (normsys factor), got:\n{text}"
    );

    // broadcast(Poisson from obs model
    assert!(
        text.contains("broadcast(Poisson"),
        "missing Poisson obs model, got:\n{text}"
    );

    // point-free: no lambda/fn
    assert!(!text.contains("fn("), "must be point-free, got:\n{text}");

    // staterror aux: ROOT-default Poisson (Barlow–Beeston) constraint on the
    // per-bin staterror scales, emitted as a ContinuedPoisson on
    // staterror_channel1. Numerical conformance vs ROOT/pyhf lives in the
    // flatppl-js cross-engine suite.
    assert!(
        text.contains("hepphys.ContinuedPoisson") && text.contains("staterror_channel1"),
        "expected a ContinuedPoisson staterror constraint on staterror_channel1, got:\n{text}"
    );

    // `call(hepphys...)` must never appear — that used a non-existent builtin
    assert!(
        !text.contains("call(hepphys"),
        "must NOT emit `call(hepphys...)` (invalid FlatPPL builtin), got:\n{text}"
    );

    // Module-member call syntax must be used instead
    assert!(
        text.contains("hepphys.interp_poly6_exp("),
        "must emit `hepphys.interp_poly6_exp(...)` (module-member call), got:\n{text}"
    );

    // Round-trip: emitted FlatPPL must re-parse without error
    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "emitted FlatPPL must re-parse cleanly: {:?}\n\nEmitted text:\n{text}",
        parsed.err()
    );
}
