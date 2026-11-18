use flatppl_formula::{
    DataMode, DataValue, Recipe, build, prepare_data, prepare_forecast, read, read_json_table,
};

#[test]
fn forecast_design_uses_only_needed_anchors_and_retains_future_rows() {
    let spec = read(r#"{"formula":"y | ar(time=t, group=g) + weights(w) ~ x"}"#).unwrap();
    let training =
        read_json_table(r#"{"y":[1,2,3],"x":[0,1,2],"w":[1,1,1],"g":["a","a","b"],"t":[0,1,0]}"#)
            .unwrap();
    let history =
        read_json_table(r#"{"y":[3,2,8],"x":[0.5,0.0,1.0],"g":["a","a","unused"],"t":[2,1,20]}"#)
            .unwrap();
    let future = read_json_table(r#"{"x":[7,2,4],"g":["a","a","b"],"t":[7,4,1]}"#).unwrap();
    let fit = build(&spec, &training).unwrap();
    let forecast = prepare_forecast(&fit.recipe, &future, &history, false).unwrap();
    let data = forecast.data;
    assert_eq!(data.rows, [1, 0, 2]);
    let constants = &data.constants;
    assert_eq!(constants["N"], DataValue::Integer(3));
    assert_eq!(constants["ar_design_N"], DataValue::Integer(4));
    assert_eq!(constants["ar_gap_data"], DataValue::Integers(vec![2, 3, 0]));
    assert_eq!(
        constants["ar_previous_data"],
        DataValue::Integers(vec![4, 1, 3])
    );
    assert_eq!(
        constants["ar_anchor_data"],
        DataValue::Integers(vec![4, 4, 3])
    );
    assert_eq!(
        constants["ar_elapsed_data"],
        DataValue::Integers(vec![2, 5, 0])
    );
    assert_eq!(
        constants["ar_observed_data"],
        DataValue::Real64(vec![3.0, 3.0, 0.0])
    );
    assert_eq!(
        constants["x_1"],
        DataValue::Real64(vec![2.0, 7.0, 4.0, 0.5])
    );
    assert!(!constants.contains_key("y_data"));
    assert!(!constants.contains_key("weights_data"));
    assert!(data.columns_json().unwrap().get("ar_design_N").is_none());
}

#[test]
fn grouped_ar_sorts_rows_and_keeps_integer_gaps_in_training_and_prediction() {
    let spec = read(r#"{"formula":"y | ar(time=t, group=g) ~ x"}"#).unwrap();
    let data = read_json_table(
        r#"{"y":[3.0,1.0,2.0,5.0,4.0], "t":[6,1,4,3,1],
            "g":["b","a","a","b","b"], "x":[0,1,4,9,16]}"#,
    )
    .unwrap();
    let model = build(&spec, &data).unwrap();
    assert_eq!(model.data.rows, [1, 2, 4, 3, 0]);
    assert_eq!(
        model.data.constants["ar_gap_data"],
        DataValue::Integers(vec![0, 3, 0, 2, 3])
    );
    assert_eq!(
        model.data.constants["ar_previous_data"],
        DataValue::Integers(vec![1, 1, 3, 3, 4])
    );
    assert!(model.info.parameters.iter().any(|p| p.name == "ar"));

    let saved = serde_json::to_string(&model.recipe).unwrap();
    let recipe: Recipe = serde_json::from_str(&saved).unwrap();
    let rows = read_json_table(r#"{"g":["z","a","z"],"t":[12,12,10],"x":[1,2,3]}"#).unwrap();
    let predicted = prepare_data(&recipe, &rows, DataMode::Prediction).unwrap();
    assert_eq!(predicted.rows, [1, 2, 0]);
    assert_eq!(
        predicted.constants["ar_gap_data"],
        DataValue::Integers(vec![0, 0, 2])
    );
    assert!(!predicted.constants.contains_key("y_data"));
}

#[test]
fn grouped_ar_rejects_invalid_time_or_model() {
    let spec = read(r#"{"formula":"y | ar(time=t, group=g) ~ 1"}"#).unwrap();
    for (times, message) in [
        ("[1, 1]", "unique"),
        ("[1, 0.5]", "integer-valued"),
        ("[1, null]", "cannot be missing"),
    ] {
        let table =
            read_json_table(&format!(r#"{{"y":[1,2],"t":{times},"g":["a","a"]}}"#)).unwrap();
        assert!(build(&spec, &table).err().unwrap().0.contains(message));
    }
    let table = read_json_table(r#"{"y":[1,2],"t":[0,1],"g":["a","a"]}"#).unwrap();
    let spec = read(
        r#"{"formula":"y | ar(time=t, group=g) ~ 1",
            "priors":[{"class":"ar","prior":"constant(1)"}]}"#,
    )
    .unwrap();
    assert!(
        build(&spec, &table)
            .err()
            .unwrap()
            .0
            .contains("strictly between")
    );
}
