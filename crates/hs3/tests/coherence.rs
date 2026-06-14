//! Table-coherence guard (black-box mirror of the in-lib `MOD_SPECS`
//! `unreachable!` guard in histfactory.rs).
//!
//! `emit_distribution` dispatches on the HS3 distribution `type` string; a kind
//! that is dropped from the dispatch, or whose tabulated `dist_spec` row carries
//! a typo'd `type`, would fall through to `Error::UnknownDistType`. This test
//! drives every distribution kind the converter claims to recognize through the
//! public `read_hs3` entry point with a barebones `{name, type}` document and
//! asserts the failure (if any) is NEVER `UnknownDistType` — i.e. the dispatch
//! arm exists and the kind string is spelled consistently.
//!
//! The match on `kind` happens before any field validation, so a barebones
//! document reaches the arm regardless of which required fields it omits: a
//! recognized kind yields either Ok or a field-level `Unsupported`, never
//! `UnknownDistType`. (Kept in `tests/` because the SPECS table and
//! `emit_distribution` are `pub(crate)`; this asserts the same invariant from
//! outside.)
use flatppl_hs3::Error;

/// Every distribution `type` string that `emit_distribution` has a dispatch arm
/// for (crates/hs3/src/distribution.rs). If an arm is added/removed/renamed,
/// update this list — a drift between this list and the dispatch is exactly what
/// the test is here to surface.
const RECOGNIZED_DIST_KINDS: &[&str] = &[
    "gaussian_dist",
    "normal_dist",
    "poisson_dist",
    "exponential_dist",
    "lognormal_dist",
    "uniform_dist",
    "product_dist",
    "generalized_normal_dist",
    "multivariate_normal_dist",
    "crystalball_dist",
    "argus_dist",
    "mixture_dist",
    "generic_dist",
    "density_function_dist",
    "log_density_function_dist",
    "rate_extended_dist",
    "rate_density_dist",
    "bincounts_extended_dist",
    "bincounts_density_dist",
    "polynomial_dist",
    "barlow_beeston_lite_poisson_constraint_dist",
    // Tabulated-but-deferred: these have an explicit dispatch arm that returns
    // Error::Unsupported (NOT UnknownDistType), so they belong in the guard.
    "relativistic_breit_wigner_dist",
    "histfactory_dist",
];

#[test]
fn every_recognized_dist_kind_has_a_dispatch_arm() {
    for kind in RECOGNIZED_DIST_KINDS {
        // Barebones single-distribution document. The dispatch matches on `type`
        // before validating fields, so this reaches the arm for every kind.
        let json = format!(
            r#"{{"distributions":[{{"name":"d","type":"{kind}"}}],"parameter_points":[]}}"#
        );
        match flatppl_hs3::read_hs3(&json) {
            Ok(_) => {} // converted fine (kind is handled)
            Err(Error::UnknownDistType(t)) => panic!(
                "`{kind}` is listed as recognized but `emit_distribution` has no dispatch arm \
                 for it (got UnknownDistType({t})). The dispatch table and this guard have \
                 drifted — add the arm or remove the kind."
            ),
            Err(_) => {} // any other error (missing field, deferred, …) is fine
        }
    }
}

#[test]
fn analyses_block_is_detected() {
    let with = r#"{"distributions":[],"analyses":[{"name":"a"}]}"#;
    let without = r#"{"distributions":[]}"#;
    assert!(flatppl_hs3::document_has_analyses(with));
    assert!(!flatppl_hs3::document_has_analyses(without));
    // An empty `analyses` array dropped nothing → no note.
    assert!(!flatppl_hs3::document_has_analyses(
        r#"{"distributions":[],"analyses":[]}"#
    ));
}

/// Conversely, a genuinely unknown `type` MUST yield `UnknownDistType` — this
/// pins the negative side so the guard above can't be trivially satisfied by a
/// converter that swallows everything.
#[test]
fn truly_unknown_dist_kind_is_unknown_dist_type() {
    let json = r#"{"distributions":[{"name":"d","type":"definitely_not_a_real_dist"}],"parameter_points":[]}"#;
    match flatppl_hs3::read_hs3(json) {
        Err(Error::UnknownDistType(t)) => assert_eq!(t, "definitely_not_a_real_dist"),
        other => panic!("expected UnknownDistType, got: {other:?}"),
    }
}

/// The checked and unchecked import paths must produce the same module — the
/// round-trip self-check (on by default) only validates, it must not alter the
/// result. (Task 5: `read_*_unchecked` variants for latency-sensitive callers.)
#[test]
fn checked_and_unchecked_agree() {
    let json = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/paper_gaussian.json"
    ))
    .unwrap();
    let checked = flatppl_hs3::read_hs3(&json).unwrap();
    let unchecked = flatppl_hs3::read_hs3_unchecked(&json).unwrap();
    let render = |m| flatppl_syntax::print_with(m, flatppl_syntax::Syntax::Minimal);
    assert_eq!(render(&checked), render(&unchecked));
}

/// A same-observable product of two *discrete* factors shares the counting
/// measure, so it is a valid pointwise density (pmf) product — it must convert,
/// not be rejected by the mixed-measure guard.
#[test]
fn same_variate_product_all_discrete_is_allowed() {
    let json = r#"{"distributions":[
        {"name":"prod","type":"product_dist","factors":["p1","p2"]},
        {"name":"p1","type":"poisson_dist","mean":"l1","x":"n"},
        {"name":"p2","type":"poisson_dist","mean":"l2","x":"n"}
    ],"parameter_points":[{"name":"nom","entries":[
        {"name":"l1","value":3.0},{"name":"l2","value":4.0}
    ]}]}"#;
    let m = flatppl_hs3::read_hs3(json).expect("all-discrete shared product must convert");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    assert!(
        text.contains("logweighted"),
        "expected density-product form, got:\n{text}"
    );
}
