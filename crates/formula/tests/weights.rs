use flatppl_formula::{
    ColumnValues, DataMode, DataValue, Trials, build, parse_formula, prepare_data, read,
};

#[test]
fn response_weights_compose_with_trials_without_changing_terms() {
    let formula = parse_formula("y | trials(n) + weights(w) ~ x * f").unwrap();
    assert_eq!(formula.trials, Some(Trials::Column("n".into())));
    assert_eq!(formula.weights.as_deref(), Some("w"));
    assert_eq!(
        formula,
        parse_formula("y | weights(w) + trials(n) + weights(w) ~ x + f + x:f").unwrap()
    );
    assert_eq!(
        parse_formula("y | trials(+2) + weights(w) ~ x")
            .unwrap()
            .trials,
        Some(Trials::Constant(2))
    );
    assert!(parse_formula("y | weights(w) + weights(v) ~ x").is_err());
    assert!(parse_formula("y | trials(n) + trials(m) + weights(w) ~ x").is_err());
}

#[test]
fn weights_follow_model_rows_and_are_not_prediction_inputs() {
    let spec = read(
        &serde_json::json!({
            "formula": "y | weights(w) ~ x",
            "data": {"nrows": 6, "columns": [
                {"name": "x", "valid": [true, true, true, false, true, true],
                 "values": {"kind": "real32", "values": [1, 2, 3, 4, 5, 6]}},
                {"name": "y", "valid": [true, true, true, true, false, true],
                 "values": {"kind": "real64", "values": [2, 4, 6, 8, 10, 12]}},
                {"name": "w", "valid": [true, true, true, true, true, false],
                 "values": {"kind": "real32", "values": [0, 0.5, 2, -1, 0, 3]}},
                {"name": "unused", "valid": [false, false, false, false, false, false],
                 "values": {"kind": "int", "values": [0, 0, 0, 0, 0, 0]}}
            ]}
        })
        .to_string(),
    )
    .unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(converted.data.rows, [0, 1, 2]);
    assert_eq!(converted.recipe.predictors["mu"].means, [2.0]);
    assert_eq!(
        converted.data.constants["weights_data"],
        DataValue::Real32(vec![0.0, 0.5, 2.0])
    );
    assert_eq!(
        converted.data.constants["y_data"],
        DataValue::Real64(vec![2.0, 4.0, 6.0])
    );

    let saved = serde_json::to_string(&converted.recipe).unwrap();
    let recipe = serde_json::from_str(&saved).unwrap();
    let mut table = spec.data.unwrap();
    assert_eq!(
        prepare_data(&recipe, &table, DataMode::Training).unwrap(),
        converted.data
    );
    table.columns[2].values = ColumnValues::Real32(vec![0.0; 6]);
    assert_eq!(
        prepare_data(&recipe, &table, DataMode::Training)
            .unwrap()
            .constants["weights_data"],
        DataValue::Real32(vec![0.0; 3])
    );
    table.columns[2].values = ColumnValues::Real32(vec![-1.0; 6]);
    assert!(prepare_data(&recipe, &table, DataMode::Training).is_err());
    table.columns[2].values = ColumnValues::Real32(vec![f32::INFINITY; 6]);
    assert!(prepare_data(&recipe, &table, DataMode::Training).is_err());
    table.columns[2].values = ColumnValues::Bool(vec![true; 6]);
    assert!(prepare_data(&recipe, &table, DataMode::Training).is_err());

    table
        .columns
        .retain(|column| column.name != "y" && column.name != "w");
    let prediction = prepare_data(&recipe, &table, DataMode::Prediction).unwrap();
    assert_eq!(prediction.rows, [0, 1, 2, 4, 5]);
    assert_eq!(
        prediction.constants["x_1"],
        DataValue::Real32(vec![1.0, 2.0, 3.0, 5.0, 6.0])
    );
    assert!(!prediction.constants.contains_key("weights_data"));
}
