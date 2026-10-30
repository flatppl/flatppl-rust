//! A kernel broadcast or a mixture family scores `K` at the per-cell record of its
//! parameters. §04 "Broadcasting", "Disallowed inputs" allows no record as an operand of
//! `broadcast`, so the lowering never hands one to the outer broadcast: with an array
//! argument the per-cell records are an array, and with scalar arguments the per-cell callee
//! takes the parameters as operands and builds the record itself. The emitted FlatPDL
//! re-infers clean.
//!
//! Closed-form values, scored through flatppl-js on the determinized module (the
//! oracle first, then the scored value):
//! - scalar kernel broadcast over data, `mu = 0.7`: -5.327877066409345 (-5.327877066409345);
//! - scalar mixture family at x = 1: -1.643335713764618 (-1.6433357137646183);
//! - array and scalar arguments, at (0.5, 0.25): -3.325733927529236 (-3.325733927529236).
use flatppl_determinizer::{determinize, is_flatpdl};
use flatppl_infer::Severity;

fn lower(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let out = determinize(&m).unwrap_or_else(|e| panic!("must lower: {e:?}\n{src}"));
    is_flatpdl(&out).unwrap();
    // The output feeds the next inference pass: no operand may be refused.
    let mut again = flatppl_syntax::parse(&flatppl_syntax::print(&out)).unwrap();
    let errors: Vec<_> = flatppl_infer::infer(&mut again)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{errors:?}\n{src}");
    flatppl_flatpir::write(&out)
}

/// The per-cell callee: its record is built inside, from its own inputs.
const CALLEE: &str = "(builtin_logdensityof Normal";

#[test]
fn scalar_arguments_reach_the_per_cell_callee_as_operands() {
    let pir = lower(
        "mu = elementof(reals)\nk = functionof(Normal.(mu, 1.0))\n\
         lik = likelihoodof(k, [2.0, 3.0])\nlp = logdensityof(lik, record(mu = 0.7))",
    );
    assert!(pir.contains(CALLEE), "{pir}");
    assert!(
        pir.contains("(record (%field mu (%ref %local _density_point_"),
        "the callee builds the record from its inputs:\n{pir}"
    );
    assert!(
        !pir.contains("(broadcast record"),
        "no record operand:\n{pir}"
    );
    // The parameter values ride as operands next to the observations.
    assert!(
        pir.contains("0.7") && pir.contains("(vector 2.0 3.0)"),
        "{pir}"
    );
}

#[test]
fn a_scalar_mixture_family_scores_each_component_at_its_own_weight() {
    let pir = lower(
        "w = [0.3, 0.7]\nmix = ksuperpose(Normal, w, mu = 0.5, sigma = 2.0)\n\
         lp = logdensityof(mix, 1.0)",
    );
    assert!(pir.contains("(logsumexp "), "{pir}");
    assert!(pir.contains(CALLEE), "{pir}");
    assert!(!pir.contains("(broadcast record"), "{pir}");
}

#[test]
fn an_array_argument_keeps_the_array_of_per_cell_records() {
    let pir = lower("d = Normal.([0.0, 1.0], 2.0)\nlp = logdensityof(d, [0.5, 0.25])");
    assert!(
        pir.contains("(broadcast record (%kwarg mu"),
        "the record is built per cell, an array of records:\n{pir}"
    );
    assert!(pir.contains("builtin_logdensityof"), "{pir}");
}
