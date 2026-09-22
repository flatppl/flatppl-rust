//! Keep independent normalization effects on one tensor interpolation axis.

#[test]
fn normsys_interpolation_reuses_factors_across_samples_and_channels() {
    let mut doc: serde_json::Value = serde_json::from_str(
        r#"{
          "channels": [{"name": "c", "samples": [
            {"name": "signal", "data": [10.0, 20.0], "modifiers": [
              {"name": "mu", "type": "normfactor", "data": null},
              {"name": "shared", "type": "normsys", "data": {"lo": 0.8, "hi": 1.3}}
            ]},
            {"name": "background", "data": [30.0, 40.0], "modifiers": [
              {"name": "shared", "type": "normsys", "data": {"lo": 0.9, "hi": 1.1}},
              {"name": "other", "type": "normsys", "data": {"lo": 0.8, "hi": 1.3}}
            ]}
          ]}],
          "observations": [{"name": "c", "data": [42.0, 58.0]}],
          "measurements": [{"name": "m", "config": {"poi": "mu"}}]
        }"#,
    )
    .unwrap();
    let mut repeated = doc["channels"][0]["samples"][0].clone();
    repeated["name"] = "other_signal".into();
    doc["channels"][0]["samples"]
        .as_array_mut()
        .unwrap()
        .push(repeated);
    let mut distinct = doc["channels"][0]["samples"][0].clone();
    distinct["name"] = "other_response".into();
    distinct["modifiers"][1]["data"]["hi"] = 1.1.into();
    doc["channels"][0]["samples"]
        .as_array_mut()
        .unwrap()
        .push(distinct);
    let mut channel = doc["channels"][0].clone();
    channel["name"] = "d".into();
    doc["channels"].as_array_mut().unwrap().push(channel);
    doc["observations"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"name": "d", "data": [51.0, 79.0]}));
    let module = flatppl_hs3::read(&doc.to_string()).unwrap();
    let text = flatppl_syntax::print_with(&module, flatppl_syntax::Syntax::Minimal);
    // Scalar calls clone the degree-six interpolation body for every effect.
    // The published nuisance identities and per-sample responses remain distinct.
    assert_eq!(
        text.matches("hepphys.interp_poly6_exp").count(),
        1,
        "{text}"
    );
    // The four unique factors differ in parameter, low, or high. Only exact
    // repeats share a lane; three distinct factors have a low endpoint of 0.8.
    assert_eq!(text.matches("0.8").count(), 3, "{text}");
}

#[test]
fn singleton_factors_keep_native_interpolation_kinds() {
    let channel = |name, interpolation| {
        serde_json::json!({
            "name": name, "type": "histfactory_dist",
            "samples": [{"name": "s", "data": {"contents": [10.0]},
                "modifiers": [{"parameter": "alpha", "type": "normsys",
                    "interpolation": interpolation, "data": {"lo": 0.8, "hi": 1.3}}]}]
        })
    };
    let doc = serde_json::json!({
        "distributions": [channel("a", "lin"), channel("b", "log"), channel("c", "lin")],
        "data": [{"name": "observed", "type": "binned", "contents": [12.0]}],
        "likelihoods": [{"name": "main", "distributions": ["a", "b", "c"],
            "data": ["observed", "observed", "observed"]}]
    });
    let module = flatppl_hs3::read_hs3(&doc.to_string()).unwrap();
    let text = flatppl_syntax::print_with(&module, flatppl_syntax::Syntax::Minimal);
    assert_eq!(
        text.matches("hepphys.interp_pwlin(0.8, 1.0, 1.3, alpha)")
            .count(),
        2,
        "{text}"
    );
    assert_eq!(
        text.matches("hepphys.interp_pwexp(0.8, 1.0, 1.3, alpha)")
            .count(),
        1,
        "{text}"
    );
    assert_eq!(
        text.matches("alpha_constraint_likelihood =").count(),
        1,
        "{text}"
    );
}
