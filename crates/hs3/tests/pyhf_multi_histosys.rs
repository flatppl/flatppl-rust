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

/// One `staterror` name across two channels.
///
/// pyhf gives such a name ONE paramset spanning every channel that carries it,
/// each channel masking its own slice, and the components are independent. For
/// the workspace below that is four components with sigmas
/// [0.5, 0.3, 1/30, 0.05] and auxdata [1, 1, 1, 1]; pyhf's logpdf is
/// -5.269013574690845 at init and -32.9547729941094 at [1.1, 0.9, 1.2, 0.8],
/// which the flatppl-js engine reproduces to 2e-14 on this emission.
const SPANNING_STATERROR: &str = r#"{
  "channels": [
    { "name": "chA",
      "samples": [{ "name": "s", "data": [10.0, 20.0],
        "modifiers": [{ "name": "mcstat", "type": "staterror", "data": [5.0, 6.0] }] }] },
    { "name": "chB",
      "samples": [{ "name": "s", "data": [30.0, 40.0],
        "modifiers": [{ "name": "mcstat", "type": "staterror", "data": [1.0, 2.0] }] }] }
  ],
  "observations": [{ "name": "chA", "data": [10.0, 20.0] },
                   { "name": "chB", "data": [30.0, 40.0] }],
  "measurements": [{ "name": "m", "config": { "poi": "", "parameters": [] } }]
}"#;

#[test]
fn a_staterror_name_spans_the_channels_that_carry_it() {
    let m = flatppl_hs3::read(SPANNING_STATERROR).unwrap();
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    // ONE parameter over the union of both channels' bins, not one per channel.
    assert!(
        text.contains("mcstat = elementof(cartpow(posreals, 4))"),
        "expected a 4-component parameter, got:\n{text}"
    );
    assert_eq!(
        text.matches("mcstat = elementof").count(),
        1,
        "one declaration only, got:\n{text}"
    );
    // Each channel multiplies in its own disjoint slice. `get` is 1-based.
    assert!(
        text.contains(
            "chA_s_expected = broadcast(mul, chA_s_nominal, \
                       [get(mcstat, 1), get(mcstat, 2)])"
        ) && text.contains(
            "chB_s_expected = broadcast(mul, chB_s_nominal, \
                              [get(mcstat, 3), get(mcstat, 4)])"
        ),
        "expected per-channel slices, got:\n{text}"
    );
    // One constraint over the whole vector, with pyhf's per-channel sigmas.
    assert!(
        text.contains("mcstat_delta = [0.5, 0.3, 0.03333333333333333, 0.05]")
            && text.contains("likelihoodof(mcstat_constraint, [1.0, 1.0, 1.0, 1.0])"),
        "expected one spanning constraint, got:\n{text}"
    );
    assert_eq!(
        text.matches("mcstat_constraint_likelihood =").count(),
        1,
        "one constraint term only, got:\n{text}"
    );
}

/// A staterror name confined to ONE channel must emit exactly what it did
/// before spanning existed: a plain reference, not a one-element slice.
#[test]
fn a_single_channel_staterror_is_not_sliced() {
    let m = flatppl_hs3::read(
        r#"{
  "channels": [
    { "name": "c",
      "samples": [{ "name": "b", "data": [50.0, 60.0],
        "modifiers": [{ "name": "mu", "type": "normfactor", "data": null },
                      { "name": "st", "type": "staterror", "data": [5.0, 6.0] }] }] }
  ],
  "observations": [{ "name": "c", "data": [52.0, 58.0] }],
  "measurements": [{ "name": "m", "config": { "poi": "mu", "parameters": [] } }]
}"#,
    )
    .unwrap();
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    assert!(
        text.contains("st = elementof(cartpow(posreals, 2))") && !text.contains("get(st,"),
        "a single-channel staterror must not be sliced, got:\n{text}"
    );
}
