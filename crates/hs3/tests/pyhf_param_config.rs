//! A measurement config's `parameters` block can replace what a modifier
//! derived for its auxiliary measurement: `auxdata` is the value the constraint
//! is observed at, `sigmas` the Gaussian width, `factors` the Poisson rate
//! factor. An override of a field the paramset does not define, or of the wrong
//! length, is refused.
//!
//! Also here: a bin whose nominal or uncertainty is not positive. The derived
//! rate factor and Gaussian width are infinite or zero there, so pyhf fixes the
//! bin's parameter and substitutes 1.0. Without that substitution the density is
//! NaN or infinite.

fn convert(src: &str) -> String {
    let m = flatppl_hs3::read(src).unwrap();
    flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal)
}

fn ws(modifier: &str, params: &str) -> String {
    format!(
        r#"{{
  "channels": [
    {{ "name": "c",
      "samples": [
        {{ "name": "sig", "data": [20.0, 10.0],
          "modifiers": [{{ "name": "mu", "type": "normfactor", "data": null }}] }},
        {{ "name": "bkg", "data": [50.0, 60.0], "modifiers": [{modifier}] }}
      ] }}
  ],
  "observations": [{{ "name": "c", "data": [72.0, 68.0] }}],
  "measurements": [{{ "name": "m", "config": {{ "poi": "mu", "parameters": [{params}] }} }}]
}}"#
    )
}

const NORMSYS: &str = r#"{ "name": "t", "type": "normsys", "data": { "lo": 0.9, "hi": 1.15 } }"#;
const HISTOSYS: &str = r#"{ "name": "t", "type": "histosys",
    "data": { "lo_data": [45.0, 58.0], "hi_data": [55.0, 63.0] } }"#;
const SHAPESYS: &str = r#"{ "name": "g", "type": "shapesys", "data": [5.0, 6.0] }"#;
const STATERROR: &str = r#"{ "name": "st", "type": "staterror", "data": [5.0, 6.0] }"#;

#[test]
fn normsys_auxdata_moves_the_observed_point() {
    let text = convert(&ws(NORMSYS, r#"{ "name": "t", "auxdata": [0.5] }"#));
    assert!(
        text.contains("t_constraint_likelihood = likelihoodof(t_constraint, 0.5)"),
        "expected the constraint observed at 0.5, got:\n{text}"
    );
}

#[test]
fn staterror_sigmas_replace_the_derived_width() {
    let text = convert(&ws(STATERROR, r#"{ "name": "st", "sigmas": [0.2, 0.3] }"#));
    assert!(
        text.contains("st_delta = [0.2, 0.3]"),
        "expected the configured sigmas, got:\n{text}"
    );
}

#[test]
fn shapesys_factors_replace_the_derived_rate_factor() {
    let text = convert(&ws(
        SHAPESYS,
        r#"{ "name": "g", "factors": [50.0, 80.0], "auxdata": [50.0, 80.0] }"#,
    ));
    assert!(
        text.contains("g_tau = [50.0, 80.0]")
            && text.contains("likelihoodof(g_constraint, [50.0, 80.0])"),
        "expected the configured factors and auxdata, got:\n{text}"
    );
}

#[test]
fn a_field_the_paramset_does_not_use_is_refused() {
    // pyhf: InvalidModel. A normsys / histosys paramset has no sigmas.
    for modifier in [NORMSYS, HISTOSYS] {
        let err = flatppl_hs3::read(&ws(modifier, r#"{ "name": "t", "sigmas": [2.0] }"#))
            .expect_err("a sigmas override on a normsys/histosys must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("does not use `sigmas`"),
            "message must name the field, got: {msg}"
        );
    }
    let err = flatppl_hs3::read(&ws(STATERROR, r#"{ "name": "st", "factors": [1.0, 1.0] }"#))
        .expect_err("a factors override on a staterror must be refused");
    assert!(err.to_string().contains("does not use `factors`"), "{err}");
}

#[test]
fn a_wrong_length_override_is_refused() {
    // pyhf: InvalidModel ("Incorrect number of values").
    let err = flatppl_hs3::read(&ws(STATERROR, r#"{ "name": "st", "sigmas": [0.2] }"#))
        .expect_err("one sigma for a two-bin staterror must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("sets 1 `sigmas` value(s)") && msg.contains("which has 2"),
        "message must give both counts, got: {msg}"
    );
}

#[test]
fn an_entry_for_an_undeclared_parameter_is_ignored() {
    // pyhf walks the paramsets its modifiers required, so a config entry for a
    // name no modifier declares never reaches a paramset.
    let text = convert(&ws(NORMSYS, r#"{ "name": "nosuch", "auxdata": [3.0] }"#));
    assert!(
        text.contains("t_constraint_likelihood = likelihoodof(t_constraint, 0.0)"),
        "the stray entry must not move anything, got:\n{text}"
    );
}

#[test]
fn shapesys_degenerate_bin_gets_a_unit_rate_factor() {
    // Zero uncertainty in bin 1: (nominal/sigma)^2 is infinite there, so the
    // whole vector is pinned to the values pyhf uses.
    let text = convert(&ws(
        r#"{ "name": "g", "type": "shapesys", "data": [5.0, 0.0] }"#,
        "",
    ));
    assert!(
        text.contains("g_tau = [100.0, 1.0]"),
        "expected tau [100, 1], got:\n{text}"
    );
    // Zero nominal in bin 1 is the same defect from the other side.
    let text = convert(
        r#"{
  "channels": [
    { "name": "c",
      "samples": [
        { "name": "bkg", "data": [50.0, 0.0],
          "modifiers": [{ "name": "mu", "type": "normfactor", "data": null },
                        { "name": "g", "type": "shapesys", "data": [5.0, 6.0] }] }
      ] }
  ],
  "observations": [{ "name": "c", "data": [52.0, 1.0] }],
  "measurements": [{ "name": "m", "config": { "poi": "mu", "parameters": [] } }]
}"#,
    );
    assert!(
        text.contains("g_tau = [100.0, 1.0]"),
        "expected tau [100, 1], got:\n{text}"
    );
}

#[test]
fn staterror_degenerate_bin_gets_a_unit_width() {
    // Zero error in bin 1 would give Normal(gamma, 0), an infinite density.
    let text = convert(&ws(
        r#"{ "name": "st", "type": "staterror", "data": [5.0, 0.0] }"#,
        "",
    ));
    assert!(
        text.contains("st_delta = [0.1, 1.0]"),
        "expected delta [0.1, 1.0], got:\n{text}"
    );
}

#[test]
fn the_first_measurement_wins_a_disagreement() {
    // A module has one `likelihood`, so it carries one measurement's
    // configuration. pyhf's own default is index 0: with no `measurement_name`,
    // `Workspace.get_measurement` takes the first and logs "multiple
    // measurements defined. Taking the first measurement." Taking the LAST
    // silently disagreed with that default.
    let text = convert(
        r#"{
  "channels": [
    { "name": "c",
      "samples": [
        { "name": "sig", "data": [20.0, 10.0],
          "modifiers": [{ "name": "mu", "type": "normfactor", "data": null }] },
        { "name": "bkg", "data": [50.0, 60.0],
          "modifiers": [{ "name": "t", "type": "normsys",
                          "data": { "lo": 0.9, "hi": 1.15 } }] }
      ] }
  ],
  "observations": [{ "name": "c", "data": [72.0, 68.0] }],
  "measurements": [
    { "name": "meas_a",
      "config": { "poi": "mu", "parameters": [{ "name": "t", "auxdata": [0.5] }] } },
    { "name": "meas_b",
      "config": { "poi": "mu", "parameters": [{ "name": "t", "auxdata": [-0.75] }] } }
  ]
}"#,
    );
    assert!(
        text.contains("t_constraint_likelihood = likelihoodof(t_constraint, 0.5)"),
        "expected the first measurement's auxdata, got:\n{text}"
    );
    // Both measurements still declare their parameter of interest.
    assert!(
        text.contains("meas_a = record(poi = mu)") && text.contains("meas_b = record(poi = mu)"),
        "got:\n{text}"
    );
}

#[test]
fn a_spanning_staterror_override_covers_every_channels_bins() {
    // A staterror name shared across channels is ONE parameter over the union
    // of their bins, so a measurement's `sigmas` override carries that many
    // values. Sizing the check by one channel's bin count refused a workspace
    // pyhf accepts (it reports n_parameters 4 and takes all four sigmas).
    let text = convert(
        r#"{
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
  "measurements": [{ "name": "m", "config": { "poi": "", "parameters": [
    { "name": "mcstat", "sigmas": [0.4, 0.3, 0.2, 0.1] }] } }]
}"#,
    );
    assert!(
        text.contains("mcstat_delta = [0.4, 0.3, 0.2, 0.1]"),
        "expected all four configured sigmas, got:\n{text}"
    );
    // A wrong length is still refused, now against the spanning count.
    let err = flatppl_hs3::read(
        r#"{
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
  "measurements": [{ "name": "m", "config": { "poi": "", "parameters": [
    { "name": "mcstat", "sigmas": [0.4, 0.3] }] } }]
}"#,
    )
    .expect_err("two sigmas for a four-component spanning staterror must be refused");
    assert!(
        err.to_string().contains("which has 4"),
        "message must give the spanning count, got: {err}"
    );
}
