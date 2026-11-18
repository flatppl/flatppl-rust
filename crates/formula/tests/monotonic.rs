use flatppl_formula::{
    ColumnValues, DataMode, DataValue, ParameterRole, Recipe, Spec, build, prepare_data,
    read_json_table,
};
use serde_json::json;

#[test]
fn monotonic_integer_gaps_survive_recipe_round_trip() {
    let spec: Spec = serde_json::from_value(json!({
        "formula": "y ~ mo(dose) + (1 | site)",
        "priors": [
            {"class": "b", "coef": "modose", "prior": "normal(0, 1)"},
            {"class": "simo", "coef": "modose1", "prior": "dirichlet(c(2, 1, 3))"}
        ]
    }))
    .unwrap();
    let training = read_json_table(
        r#"{"dose":[9007199254740992,null,9007199254740995],"site":["a","a","b"],"y":[0,1,2]}"#,
    )
    .unwrap();
    let model = build(&spec, &training).unwrap();
    assert_eq!(model.data.rows, [0, 2]);
    assert_eq!(
        model.data.constants["mo_1"],
        DataValue::Integers(vec![1, 4])
    );
    assert_eq!(
        model
            .info
            .parameters
            .iter()
            .filter(|p| p.binding == "simo_1" && p.role == ParameterRole::Sampled)
            .map(|p| (p.name.as_str(), p.index))
            .collect::<Vec<_>>(),
        [
            ("simo_modose1[1]", Some(1)),
            ("simo_modose1[2]", Some(2)),
            ("simo_modose1[3]", Some(3)),
        ]
    );
    let recipe: Recipe =
        serde_json::from_value(serde_json::to_value(&model.recipe).unwrap()).unwrap();
    let held_out =
        read_json_table(r#"{"dose":[9007199254740994,9007199254740993],"site":["b","a"]}"#)
            .unwrap();
    let prediction = prepare_data(&recipe, &held_out, DataMode::Prediction).unwrap();
    assert_eq!(
        prediction.constants["mo_1"],
        DataValue::Integers(vec![3, 2])
    );
    assert!(!prediction.constants.contains_key("y_data"));
}

#[test]
fn monotonic_factors_reuse_declared_training_order() {
    let spec = serde_json::from_value(json!({"formula": "y ~ mo(dose)"})).unwrap();
    let mut training = read_json_table(r#"{"dose":["z","b","a"],"y":[0,1,2]}"#).unwrap();
    training
        .columns
        .iter_mut()
        .find(|column| column.name == "dose")
        .unwrap()
        .values = ColumnValues::Factor {
        levels: vec!["z".into(), "unused".into(), "a".into(), "b".into()],
        codes: vec![0, 3, 2],
        ordered: true,
    };
    let model = build(&spec, &training).unwrap();
    assert_eq!(model.info.factor_levels["dose"].levels, ["z", "a", "b"]);
    let held_out = read_json_table(r#"{"dose":["b","a","z"]}"#).unwrap();
    let prediction = prepare_data(&model.recipe, &held_out, DataMode::Prediction).unwrap();
    assert_eq!(
        prediction.constants["mo_1"],
        DataValue::Integers(vec![3, 2, 1])
    );
}

#[test]
fn binary_monotonic_predictors_have_no_sampled_simplex() {
    let spec = serde_json::from_value(json!({"formula": "y ~ mo(dose)"})).unwrap();
    let training = read_json_table(r#"{"dose":[-2.0,-1.0],"y":[0,1]}"#).unwrap();
    let model = build(&spec, &training).unwrap();
    let simplex = model
        .info
        .parameters
        .iter()
        .find(|p| p.binding == "simo_1")
        .unwrap();
    assert_eq!(simplex.role, ParameterRole::Fixed);
}

#[test]
fn monotonic_step_limit_rejects_sparse_scores_before_building_the_model() {
    let spec = serde_json::from_value(json!({"formula": "y ~ mo(dose)"})).unwrap();
    let training = read_json_table(r#"{"dose":[0,4097],"y":[0,1]}"#).unwrap();
    let error = build(&spec, &training)
        .err()
        .expect("oversized integer range");
    assert!(error.0.contains("limit of 4096 steps"), "{error}");
}
