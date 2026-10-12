use flatppl_formula::{
    DataMode, DataPlacement, DataValue, GroupLevel, build, parse_formula, prepare_data, read,
};

const SPEC: &str = r#"{
  "formula": "y ~ x + (1 | g)",
  "data": {
    "nrows": 4,
    "columns": [
      {"name": "x", "values": {"kind": "real32", "values": [1, 2, 3, 9]}},
      {"name": "y", "values": {"kind": "real64", "values": [2, 4, 8, 100]}, "valid": [true, true, true, false]},
      {"name": "g", "values": {"kind": "int", "values": [20, 10, 20, 30]}},
      {"name": "unused", "values": {"kind": "int", "values": [1, 2, 3, 4]}, "valid": [false, false, false, false]}
    ]
  }
}"#;

#[test]
fn gaussian_model_frame_preserves_brms_coordinates() {
    let spec = read(SPEC).unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    // R model.frame drops only the last response row. The numeric grouping
    // labels sort to 10,20. X=[1,2,3], Xc=[-1,0,1], median(y)=4,
    // mad(y)=1.4826*2=2.9652, hence brms's rounded prior scale is 3.
    assert_eq!(converted.data.rows, [0, 1, 2]);
    assert_eq!(converted.recipe.means, [2.0]);
    assert_eq!(converted.recipe.priors.intercept_location, 4.0);
    assert_eq!(converted.recipe.priors.scale, 3.0);
    assert_eq!(
        converted.recipe.groups[0].levels,
        [GroupLevel::Integer(10), GroupLevel::Integer(20)]
    );
    assert_eq!(
        converted.data.constants["index_g"],
        DataValue::Integers(vec![2, 1, 2])
    );
    assert_eq!(
        converted.data.constants["x_1"],
        DataValue::Real32(vec![1.0, 2.0, 3.0])
    );
    assert!(converted.info.has_improper_prior);
    let formula = parse_formula("y ~ x + x + discarded - discarded + (1 | g)").unwrap();
    assert_eq!(formula, converted.recipe.formula);
    let no_intercept = parse_formula("y ~ x - 1").unwrap();
    assert_eq!(no_intercept, parse_formula("y ~ 0 + x").unwrap());
    assert!(!no_intercept.intercept);
    assert_eq!(no_intercept, parse_formula("y ~ x + -1").unwrap());
    assert!(parse_formula("y ~ x - -1").unwrap().intercept);
    assert_eq!(
        parse_formula("y ~ x + (1 | g) - (1 || g)").unwrap(),
        formula
    );
    assert!(parse_formula("y ~ x + (1 | g) + (1 || g)").is_err());
}

#[test]
fn prediction_reuses_training_recipe_without_response() {
    let spec = read(SPEC).unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    let mut table = spec.data.unwrap();
    table.columns.retain(|c| c.name != "y");
    table.columns[0].values = flatppl_formula::ColumnValues::Real32(vec![5.0, 7.0, 9.0, 11.0]);
    table.columns[1].values = flatppl_formula::ColumnValues::Int(vec![20, 10, 20, 10]);
    let prepared = prepare_data(&converted.recipe, &table, DataMode::Prediction).unwrap();
    assert_eq!(prepared.rows, [0, 1, 2, 3]);
    assert_eq!(
        prepared.constants["x_1"],
        DataValue::Real32(vec![5.0, 7.0, 9.0, 11.0])
    );
    assert_eq!(
        prepared.constants["index_g"],
        DataValue::Integers(vec![2, 1, 2, 1])
    );
    assert!(!prepared.constants.contains_key("y_data"));
    assert_eq!(converted.recipe.means, [2.0]);
    let saved = serde_json::to_string(&converted.recipe).unwrap();
    assert_eq!(
        prepare_data(
            &serde_json::from_str(&saved).unwrap(),
            &table,
            DataMode::Prediction
        )
        .unwrap(),
        prepared
    );
}

#[test]
fn explicit_normal_priors_and_inline_no_intercept_model() {
    let mut spec = read(SPEC).unwrap();
    spec.formula = "y ~ 0 + x".into();
    spec.options.data_placement = DataPlacement::Inline;
    spec.priors = serde_json::from_value(serde_json::json!([
        {"class": "b", "prior": "normal(-1, 2)"},
        {"class": "sigma", "prior": "Normal(0, 3)"}
    ]))
    .unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert!(!converted.info.has_improper_prior);
    assert!(!converted.recipe.center);
    assert_eq!(
        converted
            .info
            .parameters
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>(),
        ["b_x", "sigma"]
    );
}

#[test]
fn factor_group_order_and_specific_priors_survive_name_collisions() {
    let mut spec = read(SPEC).unwrap();
    spec.formula = "y ~ x + (1 | g_a) + (1 | g.a)".into();
    spec.priors = serde_json::from_value(serde_json::json!([
        {"class": "sd", "group": "g.a", "prior": "Normal(0, 4)"},
        {"class": "sd", "prior": "Normal(0, 2)"}
    ]))
    .unwrap();
    let table = spec.data.as_mut().unwrap();
    table.columns[2].name = "g_a".into();
    table.columns.push(flatppl_formula::Column {
        name: "g.a".into(),
        values: flatppl_formula::ColumnValues::Factor {
            levels: vec!["B".into(), "A".into(), "unused".into()],
            codes: vec![1, 0, 1, 2],
            ordered: false,
        },
        valid: None,
    });
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    // brms orders grouping terms by group name, not formula occurrence order.
    // Under the pinned C collation, g.a precedes g_a.
    assert_eq!(
        converted
            .recipe
            .formula
            .groups
            .iter()
            .map(|g| g.name())
            .collect::<Vec<_>>(),
        ["g.a", "g_a"]
    );
    let first_latent = converted
        .info
        .parameters
        .iter()
        .find(|p| p.name == "z_1[1,1]")
        .unwrap();
    assert_eq!(first_latent.binding, converted.recipe.groups[0].z);
    assert_eq!(
        converted.recipe.groups[0].levels,
        [GroupLevel::Text("B".into()), GroupLevel::Text("A".into())]
    );
    assert_eq!(
        converted.data.constants[&converted.recipe.groups[0].index],
        DataValue::Integers(vec![2, 1, 2])
    );
    assert_ne!(converted.recipe.groups[0].sd, converted.recipe.groups[1].sd);
    assert_ne!(
        converted.recipe.groups[0].index,
        converted.recipe.groups[1].index
    );
}
