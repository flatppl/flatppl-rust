use flatppl_formula::{DataMode, DataValue, Recipe, Spec, build, prepare_data, read_json_table};
use serde_json::json;

fn growth_spec() -> Spec {
    serde_json::from_value(json!({
        "formula": "y ~ a * exp(b * x)",
        "nlpars": {"a": "a ~ 1", "b": "b ~ 1"},
        "priors": [
            {"class": "b", "nlpar": "a", "prior": "normal(1, 2)"},
            {"class": "b", "nlpar": "b", "prior": "normal(0, 2)"}
        ]
    }))
    .unwrap()
}

#[test]
fn nonlinear_recipe_prepares_expression_columns_on_new_rows() {
    let data = read_json_table(r#"{"x":[0,1,null,2],"y":[2.2,3.8,0,8.5]}"#).unwrap();
    let model = build(&growth_spec(), &data).unwrap();
    assert_eq!(model.data.rows, [0, 1, 3]);
    assert_eq!(model.info.nlpars["a"].formula, "a ~ 1");
    assert!(model.info.dpars.is_empty());
    assert!(!model.recipe.predictors["a"].center);
    assert_eq!(
        model.recipe.predictors["a"].priors.intercept.prior,
        "normal(1, 2)"
    );
    let json = serde_json::to_string(&model.recipe).unwrap();
    let recipe: Recipe = serde_json::from_str(&json).unwrap();
    let new_data = read_json_table(r#"{"x":[3,-1]}"#).unwrap();
    let prepared = prepare_data(&recipe, &new_data, DataMode::Prediction).unwrap();
    assert_eq!(
        prepared.constants["nl_data_1"],
        DataValue::Real64(vec![3.0, -1.0])
    );
    assert!(!prepared.constants.contains_key("y_data"));
}

#[test]
fn nonlinear_arithmetic_accepts_real_division() {
    let mut spec = growth_spec();
    spec.formula = "y ~ -log(a^2 + x) / exp(b)".into();
    let data = read_json_table(r#"{"x":[0.5,1.5,3],"y":[-0.3,-0.6,-0.8]}"#).unwrap();
    build(&spec, &data).unwrap();
}

#[test]
fn nonlinear_binomial_prepares_new_trials_and_predictor_values() {
    let mut spec = growth_spec();
    spec.family = "binomial".into();
    spec.formula = "y | trials(n) ~ a + b * x^2".into();
    let data = read_json_table(r#"{"x":[0.5,1,2],"y":[0,2,3],"n":[2,4,5]}"#).unwrap();
    let model = build(&spec, &data).unwrap();
    let new_data = read_json_table(r#"{"x":[1.5,0],"n":[6,2]}"#).unwrap();
    let fitted = prepare_data(&model.recipe, &new_data, DataMode::Prediction).unwrap();
    assert_eq!(
        fitted.constants["trials_data"],
        DataValue::Integers(vec![6, 2])
    );
    assert_eq!(
        fitted.constants["nl_data_1"],
        DataValue::Real64(vec![1.5, 0.0])
    );
}

#[test]
fn nonlinear_beta_also_prepares_a_precision_predictor() {
    let mut spec = growth_spec();
    spec.family = "beta".into();
    spec.dpars.insert(
        "phi".into(),
        serde_json::from_value(json!({"formula": "phi ~ z", "link": "log"})).unwrap(),
    );
    let data = read_json_table(r#"{"x":[0.5,1,2],"z":[1,2,3],"y":[0.2,0.4,0.8]}"#).unwrap();
    let model = build(&spec, &data).unwrap();
    let held = read_json_table(r#"{"x":[3,2],"z":[1,4]}"#).unwrap();
    let fitted = prepare_data(&model.recipe, &held, DataMode::Prediction).unwrap();
    assert_eq!(
        fitted.constants["nl_data_1"],
        DataValue::Real64(vec![3.0, 2.0])
    );
    assert_eq!(fitted.constants["x_1_phi"], DataValue::Integers(vec![1, 4]));
}

#[test]
fn nonlinear_cumulative_retains_thresholds_and_prediction_data() {
    let spec = serde_json::from_value(json!({
        "family": "cumulative",
        "formula": "y | weights(w) ~ exp(a) * x^2",
        "nlpars": {"a": "a ~ 1"},
        "priors": [
            {"class": "Intercept", "prior": "normal(0, 2)"},
            {"class": "b", "nlpar": "a", "prior": "normal(0, 1)"}
        ]
    }))
    .unwrap();
    let training = read_json_table(r#"{"x":[0.5,1,2],"y":[1,2,3],"w":[2,1,3]}"#).unwrap();
    let model = build(&spec, &training).unwrap();
    assert_eq!(
        model
            .info
            .parameters
            .iter()
            .filter(|p| p.binding == "cuts")
            .map(|p| p.index)
            .collect::<Vec<_>>(),
        [Some(1), Some(2)]
    );
    let recipe: Recipe =
        serde_json::from_value(serde_json::to_value(&model.recipe).unwrap()).unwrap();
    let held = read_json_table(r#"{"x":[1.5,-0.1]}"#).unwrap();
    let prediction = prepare_data(&recipe, &held, DataMode::Prediction).unwrap();
    assert_eq!(
        prediction.constants["nl_data_1"],
        DataValue::Real64(vec![1.5, -0.1])
    );
}

#[test]
fn nonlinear_coefficients_need_explicit_scoped_priors() {
    let data = read_json_table(r#"{"x":[0,1,2],"y":[2.2,3.8,8.5]}"#).unwrap();
    let mut spec = growth_spec();
    spec.priors.pop();
    let error = build(&spec, &data).err().unwrap();
    assert!(error.0.contains("requires an explicit prior"), "{error}");
}

#[test]
fn nonlinear_parameter_names_do_not_shadow_generated_intercepts() {
    let mut spec = growth_spec();
    spec.formula = "y ~ a + Intercept_a".into();
    spec.nlpars.remove("b");
    spec.nlpars
        .insert("Intercept_a".into(), "Intercept_a ~ 0 + x".into());
    spec.priors[1].nlpar = Some("Intercept_a".into());
    let data = read_json_table(r#"{"x":[0,1,2],"y":[2.2,3.8,8.5]}"#).unwrap();
    let model = build(&spec, &data).unwrap();
    assert!(
        model
            .info
            .parameters
            .iter()
            .any(|p| p.binding == "b_Intercept_nl_a")
    );
    assert!(
        model
            .info
            .parameters
            .iter()
            .any(|p| p.binding == "b_nl_Intercept_a")
    );
}

#[test]
fn nonlinear_indicator_predictors_keep_training_level_order() {
    let spec: Spec = serde_json::from_value(json!({
        "formula": "y ~ exp(ld) * (theta - difficulty)",
        "family": "bernoulli",
        "nlpars": {
            "theta": "theta ~ 0 + person",
            "difficulty": "difficulty ~ 0 + item",
            "ld": "ld ~ 0 + item"
        },
        "priors": [
            {"class": "b", "nlpar": "theta", "prior": "normal(0, 1)"},
            {"class": "b", "nlpar": "difficulty", "prior": "normal(0, 1)"},
            {"class": "b", "nlpar": "ld", "prior": "normal(0, 0.5)"}
        ]
    }))
    .unwrap();
    let data = read_json_table(
        r#"{"person":["p1","p1","p2","p2"],"item":["i1","i2","i1","i2"],"y":[1,0,0,1]}"#,
    )
    .unwrap();
    let model = build(&spec, &data).unwrap();
    assert_eq!(
        model.info.nlpars["theta"].design_columns,
        ["personp1", "personp2"]
    );
    let data = read_json_table(r#"{"person":["p2","p1"],"item":["i2","i1"]}"#).unwrap();
    let fitted = prepare_data(&model.recipe, &data, DataMode::Prediction).unwrap();
    assert_eq!(
        fitted.constants["x_1_nl_theta"],
        DataValue::Integers(vec![2, 1])
    );
    assert_eq!(
        fitted.constants["x_1_nl_ld"],
        DataValue::Integers(vec![2, 1])
    );
}
