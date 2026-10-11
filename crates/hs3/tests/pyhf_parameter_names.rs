use flatppl_syntax::{Syntax, print_with};
use serde_json::{Value, json};

fn workspace(names: [&str; 3]) -> Value {
    let [shared, other, valid] = names;
    json!({
        "channels": [{"name": "c", "samples": [
            {"name": "s", "data": [10.0], "modifiers": [
                {"name": shared, "type": "normsys", "data": {"lo": 0.9, "hi": 1.2}},
                {"name": other, "type": "normfactor", "data": null}]},
            {"name": "b", "data": [20.0], "modifiers": [
                {"name": shared, "type": "normsys", "data": {"lo": 0.8, "hi": 1.1}},
                {"name": valid, "type": "normfactor", "data": null}]}]}],
        "observations": [{"name": "c", "data": [32.0]}],
        "measurements": [{"name": "measurement", "config": {
            "poi": other,
            "parameters": [{"name": shared, "auxdata": [0.25]},
                           {"name": "rw_oneside_2", "auxdata": [7.0]}]}}],
        "version": "1.0.0"
    })
}

fn convert(value: &Value) -> String {
    print_with(
        &flatppl_hs3::read_pyhf(&value.to_string()).unwrap(),
        Syntax::Minimal,
    )
}

/// The settings record differs between spellings only in its `pyhf` names.
fn without_settings(text: &str) -> String {
    text.split("% Original pyhf")
        .next()
        .unwrap()
        .lines()
        .filter(|line| !line.starts_with("pyhf_measurement"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn parameter_names_preserve_shared_effects_config_and_poi() {
    let source = workspace(["rw-oneside", "rw.oneside", "rw_oneside"]);
    let canonical = workspace(["rw_oneside_3", "rw_oneside_4", "rw_oneside"]);
    let text = convert(&source);
    let expected = convert(&canonical);
    assert_eq!(without_settings(&text), without_settings(&expected));
    assert!(text.trim_end().ends_with(
        "% Original pyhf parameter names for renamed bindings.\n\
         pyhf_parameter_names = record(rw_oneside_3 = \"rw-oneside\", rw_oneside_4 = \"rw.oneside\")"
    ));
    assert!(text.contains(
        "poi = \"rw_oneside_4\", parameters = record(rw_oneside = record(pyhf = \"rw_oneside\","
    ));
    assert!(text.contains("rw_oneside_4 = record(pyhf = \"rw.oneside\","));
    assert!(text.contains("rw_oneside_3 = record(pyhf = \"rw-oneside\", type = \"normsys\", init = 0.0, lo = -5.0, hi = 5.0, fixed = false, constraint = \"normal\", width = 1.0, auxdata = 0.25)"));

    // A model-only document carries the same overrides without a measurement.
    let model = |workspace: &Value| {
        json!({
            "channels": workspace["channels"],
            "parameters": workspace["measurements"][0]["config"]["parameters"]
        })
    };
    let model_text = convert(&model(&source));
    let model_expected = convert(&model(&canonical));
    assert_eq!(
        without_settings(&model_text),
        without_settings(&model_expected)
    );
    assert!(model_text.contains("rw_oneside_3 = record(pyhf = \"rw-oneside\", type = \"normsys\", init = 0.0, lo = -5.0, hi = 5.0, fixed = false, constraint = \"normal\", width = 1.0, auxdata = 0.25)"));
}

#[test]
fn metadata_name_does_not_shadow_a_parameter() {
    let source = workspace(["pi", "2 rate", "pyhf_parameter_names"]);
    let canonical = workspace(["pi_2", "_2_rate", "pyhf_parameter_names"]);
    let text = convert(&source);
    let expected = convert(&canonical);
    assert_eq!(without_settings(&text), without_settings(&expected));
    assert!(text.contains("pyhf_parameter_names_2 = record(_2_rate = \"2 rate\", pi_2 = \"pi\")"));
}

#[test]
fn helper_alias_does_not_capture_parameters() {
    let text = convert(&workspace(["pyhf_helpers", "pyhf_helpers_2", "mu"]));
    assert!(text.contains("pyhf_helpers_3 = standard_module(\"pyhf_helpers\", \"0.1\")"));
    assert!(text.contains("pyhf_helpers = elementof(reals)"));
    assert!(text.contains("pyhf_helpers_2 = elementof(reals)"));
    assert!(text.contains("broadcast(pyhf_helpers_3.normsys_factor,"));
    assert!(text.contains("[pyhf_helpers, pyhf_helpers]"));
    assert!(text.contains(
        "pyhf_measurement_measurement = record(name = \"measurement\", selected = true, poi = \"pyhf_helpers_2\","
    ));
}

#[test]
fn empty_poi_stays_unset_when_an_empty_parameter_name_is_mapped() {
    let mut source = workspace(["", "_x_", "µ_rate"]);
    let mut canonical = workspace(["__2", "_x__2", "__rate"]);
    source["measurements"][0]["config"]["poi"] = json!("");
    canonical["measurements"][0]["config"]["poi"] = json!("");
    let text = convert(&source);
    assert_eq!(
        without_settings(&text),
        without_settings(&convert(&canonical))
    );
    assert!(text.contains(
        "pyhf_measurement_measurement = record(name = \"measurement\", selected = true, parameters ="
    ));
}
