use flatppl_formula::{ColumnValues, DataMode, DataValue, build, prepare_data, read};
use serde_json::json;

#[test]
fn ordinal_training_and_prediction_preserve_order_and_centering() {
    let spec = read(include_str!("fixtures/cumulative.formula.json")).unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(
        converted.data.constants["y_data"],
        DataValue::Integers(vec![2, 1, 3, 2])
    );
    assert_eq!(converted.recipe.predictors["mu"].means, [3.0]);
    assert_eq!(converted.info.score_offset, 2.0_f64.ln());
    let threshold = converted
        .info
        .parameters
        .iter()
        .find(|p| p.name == "Intercept[2]")
        .unwrap();
    assert_eq!((&*threshold.binding, threshold.index), ("cuts", Some(2)));
    let saved = serde_json::to_string(&converted.recipe).unwrap();
    let recipe = serde_json::from_str(&saved).unwrap();
    let mut table = spec.data.unwrap();
    table.columns[0].values = ColumnValues::Factor {
        levels: vec!["high".into(), "low".into(), "medium".into()],
        codes: vec![2, 1, 0, 2],
        ordered: true,
    };
    assert_eq!(
        prepare_data(&recipe, &table, DataMode::Training).unwrap(),
        converted.data
    );
    table.columns.retain(|column| column.name != "y");
    table.columns[0].values = ColumnValues::Real64(vec![7.0, 8.0, 9.0, 10.0]);
    assert_eq!(
        prepare_data(&recipe, &table, DataMode::Prediction)
            .unwrap()
            .constants["x_1"],
        DataValue::Real64(vec![7.0, 8.0, 9.0, 10.0])
    );
    assert_eq!(converted.recipe.predictors["mu"].means, [3.0]);
}

#[test]
fn ordinal_numeric_gaps_and_unused_factor_levels_match_brms() {
    let mut spec = read(include_str!("fixtures/cumulative.formula.json")).unwrap();
    spec.data.as_mut().unwrap().columns[0].values = ColumnValues::Int(vec![1, 3, 1, 3]);
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(converted.recipe.response_levels.as_ref().unwrap().len(), 3);
    assert_eq!(
        converted.data.constants["y_data"],
        DataValue::Integers(vec![1, 3, 1, 3])
    );
    spec.data.as_mut().unwrap().columns[0].values = ColumnValues::Factor {
        levels: vec!["low".into(), "medium".into(), "high".into(), "last".into()],
        codes: vec![2, 0, 2, 0],
        ordered: true,
    };
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(converted.recipe.response_levels.as_ref().unwrap().len(), 2);
    assert_eq!(
        converted.data.constants["y_data"],
        DataValue::Integers(vec![2, 1, 2, 1])
    );
    assert_eq!(converted.info.score_offset, 0.0);
}

#[test]
fn threshold_selectors_share_alias_resolved_laws_and_refuse_unequal_priors() {
    let mut spec = read(include_str!("fixtures/cumulative.formula.json")).unwrap();
    let entry = json!({"class": "Intercept", "coef": "2", "prior": "Normal(0.0, 2.0)"});
    spec.priors.push(serde_json::from_value(entry).unwrap());
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert!(!converted.info.has_improper_prior);
    assert_eq!(converted.info.score_offset, 2.0_f64.ln());
    spec.priors.last_mut().unwrap().prior = "Normal(1, 2)".into();
    assert!(build(&spec, spec.data.as_ref().unwrap()).is_err());
}
