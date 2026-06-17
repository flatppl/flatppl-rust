//! Bug 1: a histfactory_dist `shapesys` modifier may carry its per-bin
//! uncertainties as a `{ "vals": [...] }` object (the RooFit/HS3 high-level
//! form) rather than the bare `[...]` array (the pyhf form). Both are valid
//! HS3; the importer must accept either. Regression for the hl-style example
//! that previously hard-failed with
//!   `shapesys ... data: expected a JSON array of numbers`.

// hl-style native HS3: shapesys `data` is `{ "vals": [3.0, 7.0] }`.
const HL_VALS_OBJECT: &str = r#"{
  "distributions": [
    { "name": "singlechannel_model", "type": "histfactory_dist",
      "axes": [{ "name": "obs_x", "edges": [0.0, 1.0, 2.0] }],
      "samples": [
        { "name": "signal", "data": { "contents": [12.0, 11.0] },
          "modifiers": [{ "parameter": "mu", "type": "normfactor" }] },
        { "name": "background", "data": { "contents": [50.0, 52.0] },
          "modifiers": [{ "parameter": "uncorr_bkguncrt", "type": "shapesys",
                          "constraint": "Poisson", "data": { "vals": [3.0, 7.0] } }] }
      ] }
  ],
  "data": [
    { "name": "observed", "type": "binned", "contents": [51.0, 48.0],
      "axes": [{ "name": "obs_x", "edges": [0.0, 1.0, 2.0] }] }
  ],
  "likelihoods": [
    { "name": "main", "distributions": ["singlechannel_model"], "data": ["observed"] }
  ]
}"#;

#[test]
fn shapesys_data_vals_object_is_accepted() {
    let m = flatppl_hs3::read_hs3(HL_VALS_OBJECT)
        .expect("hl-style shapesys `{ \"vals\": [...] }` should convert");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    // shapesys with a Poisson constraint lowers to a ContinuedPoisson aux term,
    // exactly as the bare-array (pyhf) form does.
    assert!(text.contains("ContinuedPoisson"), "got:\n{text}");
}
