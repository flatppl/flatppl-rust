//! RooFit named sub-ranges must not conflict with the default domain.
//!
//! When a document has a domain named `default_domain` AND additional named
//! domains (e.g. `signal`), the named domains are RooFit fit/integration
//! ranges — they are sub-ranges of the observable support, not redefinitions.
//! The converter must take `default_domain` as the variate support and ignore
//! the others for support resolution.

// Fixture uses `uniform_dist` because its arm actively lowers the domain
// interval into `Uniform(interval(lo, hi))`, making the support selection
// directly observable in the emitted text. `gaussian_dist` ignores the domain
// entirely, so the same test with gaussian would be vacuous.
const NAMED_RANGE_JSON: &str = r#"{
  "distributions": [
    {"name": "u", "type": "uniform_dist", "x": "x"}
  ],
  "domains": [
    {"name": "default_domain", "type": "product_domain",
     "axes": [{"name": "x", "min": -10.0, "max": 10.0}]},
    {"name": "signal", "type": "product_domain",
     "axes": [{"name": "x", "min": -3.0, "max": 3.0}]}
  ]
}"#;

#[test]
fn named_range_does_not_conflict() {
    // Verify `read_hs3` succeeds (no conflict error) and round-trips via parse.
    let m = flatppl_hs3::read_hs3(NAMED_RANGE_JSON).expect("named range must not error");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    flatppl_syntax::parse(&text).expect("re-parse");

    // The Uniform support must come from `default_domain` (x ∈ [−10, 10]),
    // not from the named sub-range `signal` (x ∈ [−3, 3]).
    // This distinguishes support selection from mere sub-range presence.
    assert!(
        text.contains("Uniform(interval(-10.0"),
        "expected default domain support Uniform(interval(-10.0, …)), got: {text}"
    );
    assert!(
        !text.contains("Uniform(interval(-3"),
        "support must not be the named sub-range -3, got: {text}"
    );
}
