//! Bug 3b: the measurement's parameter-of-interest (`config.poi`) is emitted as
//! a FlatPPL record rather than silently dropped.
//!
//! (Bug 3a — normfactor bounds — was intentionally NOT changed: spec §12:206
//! keeps a normfactor's support as `reals`; pyhf's `[0, 10]` is a fit domain,
//! not measure support. See `docs/hs3_pyhf_converter_bugs.md`.)

use flatppl_syntax::{Syntax, print_with};

const PYHF: &str = r#"{
  "channels": [
    { "name": "singlechannel",
      "samples": [
        { "name": "signal", "data": [12.0, 11.0],
          "modifiers": [{ "name": "mu", "type": "normfactor", "data": null }] },
        { "name": "background", "data": [50.0, 52.0],
          "modifiers": [{ "name": "uncorr_bkguncrt", "type": "shapesys", "data": [3.0, 7.0] }] }
      ] }
  ],
  "observations": [{ "name": "singlechannel", "data": [51.0, 48.0] }],
  "measurements": [{ "name": "Measurement", "config": { "poi": "mu" } }]
}"#;

#[test]
fn poi_emitted_as_record() {
    let m = flatppl_hs3::read(PYHF).unwrap();
    let text = print_with(&m, Syntax::Minimal);
    assert!(
        text.contains("record(poi = mu)"),
        "POI `mu` should be emitted as a record, got:\n{text}"
    );
}

#[test]
fn measurement_labels_preserve_parameter_and_likelihood_bindings() {
    let mut source: serde_json::Value = serde_json::from_str(PYHF).unwrap();
    source["measurements"] = serde_json::json!([
        {"name": "cross-section", "config": {"poi": "mu"}},
        {"name": "cross_section", "config": {"poi": "mu"}},
        {"name": "mu", "config": {"poi": "mu"}},
        {"name": "likelihood", "config": {"poi": "mu"}},
        {"name": "inputs", "config": {"poi": "mu"}},
        {"name": "_x_", "config": {"poi": "mu"}},
        {"name": "_", "config": {"poi": "mu"}},
        {"name": "pi", "config": {"poi": "mu"}}
    ]);
    let module = flatppl_hs3::read(&source.to_string()).unwrap();
    let text = print_with(&module, Syntax::Minimal);
    let metadata: Vec<_> = text
        .lines()
        .filter(|line| line.contains(" = record(poi = mu)"))
        .collect();
    assert_eq!(
        metadata,
        [
            "cross_section = record(poi = mu)",
            "cross_section_2 = record(poi = mu)",
            "mu_2 = record(poi = mu)",
            "likelihood_2 = record(poi = mu)",
            "inputs_2 = record(poi = mu)",
            "_x__2 = record(poi = mu)",
            "__2 = record(poi = mu)",
            "pi_2 = record(poi = mu)",
        ]
    );
    assert!(text.contains("mu = elementof(reals)"));
    assert!(text.contains("likelihood = joint_likelihood(singlechannel_likelihood, uncorr_bkguncrt_constraint_likelihood)"));
    assert!(text.contains("pyhf measurement \"cross-section\""));
}
