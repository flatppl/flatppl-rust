//! RooFit named sub-ranges must not conflict with the default domain.
//!
//! When a document has a domain named `default_domain` AND additional named
//! domains (e.g. `signal`), the named domains are RooFit fit/integration
//! ranges — they are sub-ranges of the observable support, not redefinitions.
//! The converter must take `default_domain` as the variate support and ignore
//! the others for support resolution.

const NAMED_RANGE_JSON: &str = r#"{
  "distributions": [
    {"name": "g", "type": "gaussian_dist", "mean": "mu", "sigma": "s", "x": "x"}
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
    let m = flatppl_hs3::read_hs3(NAMED_RANGE_JSON).expect("named range must not error");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    // default domain (−10,10) is the variate support, not the named (−3,3)
    assert!(
        text.contains("-10") && text.contains("10"),
        "default domain expected: {text}"
    );
    flatppl_syntax::parse(&text).expect("re-parse");
}
