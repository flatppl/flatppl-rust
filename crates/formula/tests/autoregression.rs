use flatppl_formula::{DataMode, DataValue, Recipe, build, prepare_data, read, read_json_table};

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
