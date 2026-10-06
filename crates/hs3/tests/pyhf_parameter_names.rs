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

#[test]
fn parameter_names_preserve_shared_effects_config_and_poi() {
    let source = workspace(["rw-oneside", "rw.oneside", "rw_oneside"]);
    let canonical = workspace(["rw_oneside_3", "rw_oneside_4", "rw_oneside"]);
    let text = convert(&source);
    let expected = convert(&canonical);
    assert_eq!(
        text.trim_end(),
        format!(
            "{}\n% Original pyhf parameter names for renamed bindings.\n\
             pyhf_parameter_names = record(rw_oneside_3 = \"rw-oneside\", rw_oneside_4 = \"rw.oneside\")",
            expected.trim_end()
        )
    );

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
        model_text
            .split("% Original pyhf")
            .next()
            .unwrap()
            .trim_end(),
        model_expected.trim_end()
    );
}

#[test]
fn metadata_name_does_not_shadow_a_parameter() {
    let source = workspace(["pi", "2 rate", "pyhf_parameter_names"]);
    let canonical = workspace(["pi_2", "_2_rate", "pyhf_parameter_names"]);
    let text = convert(&source);
    let expected = convert(&canonical);
    assert_eq!(
        text.split("% Original pyhf").next().unwrap().trim_end(),
        expected.trim_end()
    );
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
    assert!(text.contains("measurement = record(poi = pyhf_helpers_2)"));
}

#[test]
fn empty_poi_stays_unset_when_an_empty_parameter_name_is_mapped() {
    let mut source = workspace(["", "_x_", "µ_rate"]);
    let mut canonical = workspace(["__2", "_x__2", "__rate"]);
    source["measurements"][0]["config"]["poi"] = json!("");
    canonical["measurements"][0]["config"]["poi"] = json!("");
    let text = convert(&source);
    assert_eq!(
        text.split("% Original pyhf").next().unwrap().trim_end(),
        convert(&canonical).trim_end()
    );
}
