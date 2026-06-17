//! Bug 4: in a native HS3 `histfactory_dist`, a `shapesys` modifier's per-bin
//! `vals` are RELATIVE uncertainties (RooFit / HS3 convention) — not the
//! absolute uncertainties pyhf uses. The per-bin Poisson-constraint strength is
//! τ = 1/vals², which the shared channel assembler produces from
//! σ_abs = vals × nominal via its τ = (nominal / σ)² rule.
//!
//! Background nominal = [50, 52], shapesys vals = [3, 7] (relative) ⇒
//! σ_abs = [150, 364] ⇒ τ = (50/150)², (52/364)² = 1/9, 1/49 — matching
//! ROOT/RooFit on the same document. (The pyhf path keeps absolute vals.)

const HL: &str = r#"{
  "distributions": [
    { "name": "ch", "type": "histfactory_dist",
      "axes": [{ "name": "obs_x", "edges": [0.0, 1.0, 2.0] }],
      "samples": [
        { "name": "signal", "data": { "contents": [12.0, 11.0] },
          "modifiers": [{ "parameter": "mu", "type": "normfactor" }] },
        { "name": "background", "data": { "contents": [50.0, 52.0] },
          "modifiers": [{ "parameter": "uncorr", "type": "shapesys",
                          "constraint": "Poisson", "data": { "vals": [3.0, 7.0] } }] }
      ] }
  ],
  "data": [
    { "name": "observed", "type": "binned", "contents": [51.0, 48.0],
      "axes": [{ "name": "obs_x", "edges": [0.0, 1.0, 2.0] }] }
  ],
  "likelihoods": [
    { "name": "main", "distributions": ["ch"], "data": ["observed"] }
  ]
}"#;

#[test]
fn shapesys_vals_treated_as_relative() {
    let m = flatppl_hs3::read_hs3(HL).unwrap();
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    // σ_abs = vals × nominal = [3·50, 7·52] = [150, 364].
    assert!(
        text.contains("[150.0, 364.0]"),
        "HS3 shapesys vals should be scaled by nominal (relative convention), got:\n{text}"
    );
    // The raw relative vals must NOT appear as the σ array (that would be the
    // pyhf absolute convention, giving the wrong τ for an HS3 document).
    assert!(
        !text.contains("./ [3.0, 7.0]"),
        "shapesys σ still uses raw vals [3, 7] (absolute), got:\n{text}"
    );
}
