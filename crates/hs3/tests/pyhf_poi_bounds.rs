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
