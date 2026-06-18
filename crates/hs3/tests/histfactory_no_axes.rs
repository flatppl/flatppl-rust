//! A `histfactory_dist` is accepted without an `axes` key. The importer derives
//! the bin count from sample `contents`, never from axis metadata, so `axes` is
//! parsed for schema fidelity only. Requiring it would needlessly hard-fail any
//! HS3 producer that omits it.
//!
//! (Audit 2026-06 of ROOT's HS3 writer: it emits `axes` unconditionally, so this
//! is defensive robustness against other producers, not a ROOT-omission
//! workaround. The one ROOT omission that did hard-fail us — unbounded domain
//! axis `min`/`max` — is covered separately.)

const NO_AXES: &str = r#"{
  "distributions": [
    { "name": "singlechannel_model", "type": "histfactory_dist",
      "samples": [
        { "name": "signal", "data": { "contents": [12.0, 11.0] },
          "modifiers": [{ "parameter": "mu", "type": "normfactor" }] },
        { "name": "background", "data": { "contents": [50.0, 52.0] },
          "modifiers": [] }
      ] }
  ],
  "data": [
    { "name": "observed", "type": "binned", "contents": [51.0, 48.0] }
  ],
  "likelihoods": [
    { "name": "main", "distributions": ["singlechannel_model"], "data": ["observed"] }
  ]
}"#;

#[test]
fn histfactory_dist_without_axes_is_accepted() {
    let m = flatppl_hs3::read_hs3(NO_AXES)
        .expect("a histfactory_dist without `axes` should still convert");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    // Two-bin Poisson observation model, assembled from sample contents.
    assert!(
        text.contains("Poisson"),
        "expected a Poisson obs model, got:\n{text}"
    );
    assert!(
        text.contains("flatppl_compat = \"0.1\""),
        "generated module must stamp flatppl_compat, got:\n{text}"
    );
}
