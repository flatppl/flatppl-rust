use flatppl_formula::{DataMode, DataValue, Recipe, build, prepare_data, read, read_json_table};

#[test]
fn censoring_reads_upper_only_on_interval_rows_and_prediction_keeps_truncation() {
    let spec =
        read(r#"{"formula":"y | trunc(lb=lo, ub=hi) + cens(c, y2) + weights(w) ~ x"}"#).unwrap();
    let data = read_json_table(
        r#"{"y":[0.0,-0.5,0.5,-0.1], "c":["none","left","right","interval"],
            "y2":[null,null,null,0.2], "x":[1,2,3,4], "w":[1,2,3,0],
            "lo":[-1,-1,-2,-1], "hi":[1,1,1,1]}"#,
    )
    .unwrap();
    let converted = build(&spec, &data).unwrap();
    assert_eq!(converted.data.rows, [0, 1, 2, 3]);
    assert_eq!(
        converted.data.constants["cens_data"],
        DataValue::Integers(vec![0, -1, 1, 2])
    );
    let saved = serde_json::to_string(&converted.recipe).unwrap();
    let recipe: Recipe = serde_json::from_str(&saved).unwrap();
    let new_rows = read_json_table(r#"{"x":[6,5], "lo":[-0.5,-0.3], "hi":[0.8,0.9]}"#).unwrap();
    let prediction = prepare_data(&recipe, &new_rows, DataMode::Prediction).unwrap();
    assert_eq!(prediction.rows, [0, 1]);
    assert_eq!(
        prediction.constants["trunc_lower_data"],
        DataValue::Real64(vec![-0.5, -0.3])
    );
    assert_eq!(
        prediction.constants["trunc_upper_data"],
        DataValue::Real64(vec![0.8, 0.9])
    );
    for name in ["y_data", "weights_data", "cens_data", "cens_upper_data"] {
        assert!(!prediction.constants.contains_key(name));
    }
}

#[test]
fn censoring_keeps_events_that_overlap_a_one_sided_truncation_range() {
    let spec = read(r#"{"formula":"y | cens(c, y2) + trunc(ub=1) ~ 1"}"#).unwrap();
    let table = read_json_table(r#"{"y":[-1,0,2], "y2":[null,2,null], "c":[0,2,-1]}"#).unwrap();
    let model = build(&spec, &table).unwrap();
    assert_eq!(model.data.rows, [0, 1, 2]);
}

#[test]
fn survival_censoring_preserves_exact_and_censored_events() {
    let data = read_json_table(
        r#"{"y":[0.1,0.5,0.0,10.0], "c":["none","left","right","interval"],
            "hi":[null,null,null,10.01]}"#,
    )
    .unwrap();
    for family in ["exponential", "weibull"] {
        let spec = read(&format!(
            r#"{{"formula":"y | cens(c, hi) ~ 1","family":"{family}"}}"#
        ))
        .unwrap();
        let model = build(&spec, &data).unwrap();
        assert_eq!(model.data.rows, [0, 1, 2, 3]);
        assert_eq!(
            model.data.constants["cens_data"],
            DataValue::Integers(vec![0, -1, 1, 2])
        );
        let prediction = read_json_table(r#"{"unused":[0,0]}"#).unwrap();
        let prepared = prepare_data(&model.recipe, &prediction, DataMode::Prediction).unwrap();
        assert_eq!(prepared.rows, [0, 1]);
        assert!(!prepared.constants.contains_key("cens_data"));
    }
}
