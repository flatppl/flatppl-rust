use flatppl_syntax::{Syntax, parse, print_with};

const POLY_FN_JSON: &str = r#"{
  "distributions": [
    {"name": "model", "type": "gaussian_dist", "mean": "fy", "sigma": "sx", "x": "x"}
  ],
  "functions": [
    {"name": "fy", "type": "polynomial", "coefficients": ["a0", "a1"], "x": "y"}
  ],
  "domains": [{"name": "domain", "type": "product_domain", "axes": [
    {"name": "x", "min": -5.0, "max": 5.0},
    {"name": "y", "min": -5.0, "max": 5.0}
  ]}],
  "data": [{"name": "observations", "type": "unbinned",
    "axes": [{"name": "x"}, {"name": "y"}], "entries": [[0.0, 0.0]]}]
}"#;

#[test]
fn polynomial_function_converts() {
    let m = flatppl_hs3::read_hs3(POLY_FN_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== polynomial fn ===\n{text}\n=== end ===");
    assert!(
        text.contains("polynomial("),
        "missing polynomial call: {text}"
    );
    assert!(
        text.contains("functionof"),
        "missing functionof wrapper: {text}"
    );
    assert!(text.contains("(a0, a1, sx)"), "missing coeffs: {text}");
    // round-trips
    parse(&text).expect("emitted FlatPPL must re-parse");
}
