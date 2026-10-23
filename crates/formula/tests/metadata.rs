use flatppl_core::{CallHead, Module, Node, NodeId, Scalar};
use flatppl_formula::{Column, ColumnValues, ParameterRole, PriorSampling, build, read};
use serde_json::{Value, json};

fn metadata(module: &Module, name: &str) -> Value {
    let reparsed = flatppl_syntax::parse(&flatppl_syntax::print(module)).unwrap();
    let (_, binding) = reparsed
        .public_bindings()
        .find(|(_, binding)| reparsed.resolve(binding.name) == name)
        .unwrap();
    literal(&reparsed, binding.rhs)
}

fn literal(module: &Module, node: NodeId) -> Value {
    match module.node(node) {
        Node::Lit(Scalar::Bool(value)) => json!(value),
        Node::Lit(Scalar::Int(value)) => json!(value),
        Node::Lit(Scalar::Real(value)) => json!(value),
        Node::Lit(Scalar::Str(value)) => json!(value),
        Node::Call(call) => {
            let CallHead::Builtin(head) = call.head else {
                panic!("metadata must contain only literal values");
            };
            match module.resolve(head) {
                "record" => Value::Object(
                    call.named
                        .iter()
                        .map(|field| {
                            (
                                module.resolve(field.name).into(),
                                literal(module, field.value),
                            )
                        })
                        .collect(),
                ),
                "neg" => {
                    let value = literal(module, call.args[0]);
                    value
                        .as_i64()
                        .map_or_else(|| json!(-value.as_f64().unwrap()), |value| json!(-value))
                }
                operation => panic!("nonliteral metadata operation: {operation}"),
            }
        }
        other => panic!("nonliteral metadata: {other:?}"),
    }
}

#[test]
fn exported_coordinates_distinguish_fixed_sampled_and_derived_values() {
    let mut spec = read(include_str!("fixtures/gaussian.formula.json")).unwrap();
    spec.formula = "y ~ x + z + (1 | g)".into();
    spec.data.as_mut().unwrap().columns.push(Column {
        name: "z".into(),
        values: ColumnValues::Real64(vec![2.0, 3.0, 5.0]),
        valid: None,
    });
    spec.priors = serde_json::from_value(json!([
        {"class":"b", "prior":"normal(0, 2)"},
        {"class":"b", "coef":"z", "prior":"constant(0.5)"},
        {"class":"sd", "prior":"constant(0.7)"},
        {"class":"sigma", "prior":"constant(1)"}
    ]))
    .unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    let names = metadata(&converted.module, "formula_parameter_names");
    assert_eq!(
        names["b_1"]["scalar"],
        json!({
            "name":"b_x", "binding":"b_1", "role":"sampled", "derived":false
        })
    );
    assert_eq!(
        names["b"]["_2"],
        json!({
            "name":"b_z", "binding":"b", "index":2, "role":"fixed", "derived":true
        })
    );
    assert_eq!(names["b_Intercept"]["scalar"]["role"], "derived");
    assert_eq!(names["sd_g"]["scalar"]["role"], "fixed");
    assert_eq!(names["sigma"]["scalar"]["role"], "fixed");
    assert_eq!(names["z_g"]["_1"]["name"], "z_1[1,1]");
    assert_eq!(names["r_g"]["_1"]["name"], "r_g[10,Intercept]");
    for parameter in &converted.info.parameters {
        let coordinates = names[&parameter.binding].as_object().unwrap();
        assert!(coordinates.values().any(|value| value == &json!(parameter)));
        assert_eq!(parameter.derived, parameter.role != ParameterRole::Sampled);
    }
    let model = metadata(&converted.module, "formula_model");
    assert_eq!(model["parameters"], names);
    assert_eq!(
        model["prior_factors"],
        json!({
            "b_1":{"mass":"normalized"},
            "temp_Intercept":{"mass":"normalized"},
            "z_g":{"mass":"normalized"}
        })
    );
    assert_eq!(model["prior_sampling"], "not_checked");
}

#[test]
fn reference_prior_metadata_tracks_mass_and_inference_coordinates() {
    let mut spec = read(include_str!("fixtures/gaussian.formula.json")).unwrap();
    spec.formula = "y ~ 0 + x".into();
    for (prior, mass, sampling) in [
        (
            "Lebesgue(reals)",
            "locallyfinite",
            PriorSampling::Unavailable,
        ),
        (
            "Lebesgue(interval(0, 2))",
            "finite",
            PriorSampling::Unavailable,
        ),
        (
            "normalize(Lebesgue(unitinterval))",
            "normalized",
            PriorSampling::NotChecked,
        ),
    ] {
        spec.priors = serde_json::from_value(json!([
            {"class":"b", "prior":prior},
            {"class":"sigma", "prior":"constant(1)"}
        ]))
        .unwrap();
        let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
        let factor = &converted.info.prior_factors["b"];
        assert_eq!(factor.mass, mass, "{prior}");
        assert_eq!(converted.info.prior_sampling, sampling, "{prior}");
        assert_eq!(converted.info.parameters[0].role, ParameterRole::Sampled);
        let exported = metadata(&converted.module, "formula_model");
        assert_eq!(exported["prior_factors"]["b"], json!(factor));
        assert_eq!(exported["prior_sampling"], json!(sampling));
    }
}

#[test]
fn a_dependent_prior_chain_reads_the_preceding_parameters_from_the_record() {
    // Each later kernel receives the record of the preceding variates by the ordinary
    // calling convention (spec §06 "Keyword form"), so a kernel of one preceding
    // parameter takes the record whole and reads its field.
    let mut spec = read(include_str!("fixtures/gaussian.formula.json")).unwrap();
    spec.formula = "y ~ 0 + x".into();
    // The masses pin what the positional chain reported before the keyword form: a
    // symbolic interval bound has an unknown mass, and a proper factor reads its own law.
    for (priors, chain, masses) in [
        (
            json!([
                {"class":"b", "prior":"Lebesgue(interval(0, sigma))"},
                {"class":"sigma", "prior":"exponential(1)"}
            ]),
            "iid(Lebesgue(interval(0, _point_",
            ("unknown", "normalized"),
        ),
        (
            json!([
                {"class":"b", "prior":"Lebesgue(interval(sigma, sigma + 2))"},
                {"class":"sigma", "prior":"exponential(1)"}
            ]),
            "iid(Lebesgue(interval(_point_",
            ("unknown", "normalized"),
        ),
        (
            json!([
                {"class":"b", "prior":"normal(0, sigma)"},
                {"class":"sigma", "prior":"Lebesgue(posreals)"}
            ]),
            "Normal(0, _point_",
            ("normalized", "locallyfinite"),
        ),
    ] {
        spec.priors = serde_json::from_value(priors).unwrap();
        let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
        let text = flatppl_syntax::print(&converted.module);
        assert!(text.contains("jointchain("), "{text}");
        assert!(text.contains(chain), "{chain}\n{text}");
        let factors = &converted.info.prior_factors;
        assert_eq!(factors["b"].mass, masses.0, "{text}");
        assert_eq!(factors["sigma"].mass, masses.1, "{text}");
    }
}

#[test]
fn exported_levels_and_names_keep_sanitized_collisions_distinct() {
    let spec = read(&json!({
        "formula":"y ~ f.a + f_a + (1 | g.a) + (1 | g_a)",
        "dpars":{"sigma":{"formula":"sigma ~ f_a"}},
        "data":{"nrows":3,"columns":[
            {"name":"y","values":{"kind":"real64","values":[2,4,8]}},
            {"name":"f.a","values":{"kind":"factor","values":{"levels":["B","A"],"codes":[0,1,0]}}},
            {"name":"f_a","values":{"kind":"factor","values":{"levels":["C","D"],"codes":[1,0,1]}}},
            {"name":"g.a","values":{"kind":"int","values":[20,i64::MIN,20]}},
            {"name":"g_a","values":{"kind":"int","values":[7,7,7]}}
        ]}
    }).to_string()).unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    let model = metadata(&converted.module, "formula_model");
    assert_eq!(
        model["factor_levels"],
        json!({
            "f_a":{"name":"f.a","levels":{"length":2,"_1":"B","_2":"A"},"ordered":false},
            "f_a_2":{"name":"f_a","levels":{"length":2,"_1":"C","_2":"D"},"ordered":false}
        })
    );
    assert_eq!(model["group_levels"]["g_a"]["name"], "g.a");
    assert_eq!(model["group_levels"]["g_a_2"]["name"], "g_a");
    assert_eq!(
        model["group_levels"]["g_a"]["levels"],
        json!({"length":2,"_1":{"kind":"integer","value":i64::MIN.to_string()}, "_2":{"kind":"integer","value":"20"}})
    );
    assert_eq!(
        model["group_levels"]["g_a_2"]["levels"],
        json!({"length":1,"_1":{"kind":"integer","value":"7"}})
    );
    assert_eq!(
        model["dpars"]["sigma"]["design_columns"],
        json!({"length":1,"_1":"f_aD"})
    );
    assert_eq!(
        model["dpars"]["sigma"]["factor_levels"]["f_a"]["levels"],
        json!({"length":2,"_1":"C","_2":"D"})
    );
    let names = metadata(&converted.module, "formula_parameter_names");
    let group_names: Vec<_> = converted
        .info
        .parameters
        .iter()
        .filter(|parameter| parameter.name.starts_with("sd_"))
        .map(|parameter| (&parameter.name, &parameter.binding))
        .collect();
    assert_eq!(group_names.len(), 2);
    assert_ne!(group_names[0].1, group_names[1].1);
    for (name, binding) in group_names {
        assert_eq!(names[binding]["scalar"]["name"], json!(name));
    }
}
