//! A native HS3 modifier's per-bin `parameters` array may hold non-ASCII names.
//!
//! `derive_vector_param_name` collapses such an array to one vector-binding
//! name by taking the longest common prefix. It counted that prefix in BYTES
//! and then sliced the name with it, which panics whenever the count lands
//! inside a character — `parameters: ["é0", "ê1"]` share only the lead byte
//! `C3`, so the importer aborted with
//! `end byte index 1 is not a char boundary`. The prefix is now taken on
//! characters, so the index is always valid.

/// The audit repro, reduced: a `staterror` modifier with no `parameter` and no
/// `name`, so its per-bin `parameters` array is what names the binding.
const UTF8_PER_BIN: &str = r#"{
  "distributions": [
    { "name": "singlechannel_model", "type": "histfactory_dist",
      "axes": [{ "name": "obs_x", "edges": [0.0, 1.0, 2.0] }],
      "samples": [
        { "name": "signal", "data": { "contents": [12.0, 11.0] },
          "modifiers": [{ "parameter": "mu", "type": "normfactor" }] },
        { "name": "background", "data": { "contents": [50.0, 52.0], "errors": [5.0, 6.0] },
          "modifiers": [{ "type": "staterror", "parameters": ["é0", "ê1"] }] }
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
fn a_non_ascii_per_bin_parameters_array_reaches_a_verdict() {
    // The importer must return a `Result`, never abort the process. `é` is not
    // a FlatPPL identifier character, so the emitted name is rejected by the
    // round-trip self-check — an honest error. Accepting such a name is a
    // separate finding (emitted-name validation at `Builder::bind`), not this
    // slicing site.
    let outcome = std::panic::catch_unwind(|| flatppl_hs3::read_hs3(UTF8_PER_BIN));
    let result = outcome.expect("the importer must not panic on a non-ASCII per-bin name");
    match result {
        Ok(m) => {
            let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
            assert!(
                !text.is_empty(),
                "an accepted conversion must emit a module"
            );
        }
        Err(e) => {
            // A refusal is fine and is the current behaviour: `é` is not a
            // FlatPPL identifier character, so the emitted name is rejected.
            // The message is NOT asserted — whether and how a non-ASCII emitted
            // name is refused belongs to the emitted-name validation at
            // `Builder::bind`, which is a separate finding landing on its own
            // branch. This test's contract is only that the importer returns
            // instead of aborting.
            let msg = format!("{e:?}");
            assert!(!msg.is_empty(), "a refusal must carry a reason");
        }
    }
}
