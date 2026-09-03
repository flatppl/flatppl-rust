//! Several histosys modifiers on one sample, and histosys sharing a sample with
//! shapesys. Both cases have to read the sample's ORIGINAL nominal.

const TWO_HISTOSYS: &str = r#"{
  "channels": [
    { "name": "ch1",
      "samples": [
        { "name": "sig", "data": [20.0, 10.0],
          "modifiers": [{ "name": "mu", "type": "normfactor", "data": null }] },
        { "name": "bkg", "data": [50.0, 60.0],
          "modifiers": [
            { "name": "a1", "type": "histosys",
              "data": { "hi_data": [55.0, 63.0], "lo_data": [45.0, 58.0] } },
            { "name": "a2", "type": "histosys",
              "data": { "hi_data": [52.0, 66.0], "lo_data": [48.0, 54.0] } }
          ] }
      ] }
  ],
  "observations": [{ "name": "ch1", "data": [72.0, 68.0] }],
  "measurements": [{ "name": "m", "config": { "poi": "mu" } }]
}"#;

#[test]
fn two_histosys_shifts_add_against_the_original_nominal() {
    let m = flatppl_hs3::read(TWO_HISTOSYS).unwrap();
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    // Each modifier interpolates the SAME nominal binding. Nesting the second
    // interpolation around the first one's output uses a wrong nominal and, at a
    // knot, discards the first shift entirely (pyhf sums the additive deltas).
    assert_eq!(
        text.matches("interp_poly6_lin").count(),
        2,
        "expected two interpolations, got:\n{text}"
    );
    for side in ["[45.0, 58.0]", "[48.0, 54.0]"] {
        assert!(
            text.contains(&format!("interp_poly6_lin({side}, ch1_bkg_nominal,")),
            "{side} must interpolate ch1_bkg_nominal directly, got:\n{text}"
        );
    }
    assert!(
        text.contains(
            "ch1_bkg_expected = broadcast(add, broadcast(add, ch1_bkg_nominal, \
                       ch1_bkg_a1_shift), ch1_bkg_a2_shift)"
        ),
        "expected nominal + a1 shift + a2 shift, got:\n{text}"
    );
    for param in ["a1", "a2"] {
        assert!(
            text.contains(&format!(
                "ch1_bkg_{param}_shift = broadcast(sub, hepphys.interp_poly6_lin"
            )) && text.contains(&format!("{param}), ch1_bkg_nominal)")),
            "{param} shift must be interp - nominal, got:\n{text}"
        );
    }
}

const HISTOSYS_AND_SHAPESYS: &str = r#"{
  "channels": [
    { "name": "ch1",
      "samples": [
        { "name": "bkg", "data": [50.0, 60.0],
          "modifiers": [
            { "name": "a1", "type": "histosys",
              "data": { "hi_data": [55.0, 63.0], "lo_data": [45.0, 58.0] } },
            { "name": "g", "type": "shapesys", "data": [5.0, 6.0] },
            { "name": "mu", "type": "normfactor", "data": null }
          ] }
      ] }
  ],
  "observations": [{ "name": "ch1", "data": [52.0, 58.0] }],
  "measurements": [{ "name": "m", "config": { "poi": "mu" } }]
}"#;

#[test]
fn shapesys_tau_uses_the_unshifted_nominal() {
    let m = flatppl_hs3::read(HISTOSYS_AND_SHAPESYS).unwrap();
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    // tau is the constraint's OBSERVED aux data. Built from the post-histosys
    // nominal it would depend on the histosys alpha, which pyhf's tau does not.
    assert!(
        text.contains("g_tau = broadcast(pow, broadcast(divide, ch1_bkg_nominal, g_sigma), 2)"),
        "tau must divide the unshifted nominal, got:\n{text}"
    );
}
