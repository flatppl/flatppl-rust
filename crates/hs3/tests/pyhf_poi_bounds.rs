//! Bug 3b: the measurement's parameter-of-interest (`config.poi`) is emitted in
//! the measurement's settings record rather than silently dropped.
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
        text.contains(
            "pyhf_measurement_Measurement = record(name = \"Measurement\", selected = true, poi = \"mu\","
        ),
        "POI `mu` should be emitted in the settings record, got:\n{text}"
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
        .filter(|line| line.starts_with("pyhf_measurement"))
        .filter_map(|line| line.split_once(", selected").map(|(head, _)| head))
        .collect();
    assert_eq!(
        metadata,
        [
            "pyhf_measurement_cross_section = record(name = \"cross-section\"",
            "pyhf_measurement_cross_section_2 = record(name = \"cross_section\"",
            "pyhf_measurement_mu = record(name = \"mu\"",
            "pyhf_measurement_likelihood = record(name = \"likelihood\"",
            "pyhf_measurement_inputs = record(name = \"inputs\"",
            "pyhf_measurement__x_ = record(name = \"_x_\"",
            "pyhf_measurement__ = record(name = \"_\"",
            "pyhf_measurement_pi = record(name = \"pi\"",
        ]
    );
    assert!(text.contains("mu = elementof(reals)"));
    assert!(text.contains("likelihood = joint_likelihood(singlechannel_likelihood, uncorr_bkguncrt_constraint_likelihood)"));
    assert!(text.contains("pyhf measurement \"cross-section\""));
}
