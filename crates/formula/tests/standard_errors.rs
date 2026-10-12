use flatppl_formula::{ColumnValues, DataMode, DataValue, Recipe, build, prepare_data, read};
use serde_json::json;

fn spec() -> flatppl_formula::Spec {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/standard-errors.json")).unwrap();
    read(&oracle["spec"].to_string()).unwrap()
}

#[test]
fn known_scale_expressions_survive_recipe_and_response_free_prediction() {
    let mut spec = spec();
    spec.formula = "y | se(log(s)) ~ x".into();
    spec.data.as_mut().unwrap().columns[2].values = ColumnValues::Real32(vec![2.0, -1.0, 4.0]);
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(converted.data.rows, [0, 2]);
    assert_eq!(converted.recipe.mu.means, [1.5]);
    assert_eq!(
        converted.data.constants["se_data"],
        DataValue::Real32(vec![2.0f32.ln(), 4.0f32.ln()])
    );
    let saved = serde_json::to_string(&converted.recipe).unwrap();
    let recipe: Recipe = serde_json::from_str(&saved).unwrap();
    let mut table = spec.data.take().unwrap();
    table.columns.retain(|column| column.name != "y");
    table.columns[1].values = ColumnValues::Real32(vec![4.0, -1.0, 8.0]);
    let prediction = prepare_data(&recipe, &table, DataMode::Prediction).unwrap();
    assert_eq!(prediction.rows, [0, 2]);
    assert_eq!(
        prediction.constants["se_data"],
        DataValue::Real32(vec![4.0f32.ln(), 8.0f32.ln()])
    );
    assert_eq!(serde_json::to_string(&recipe).unwrap(), saved);

    table.columns.retain(|column| column.name != "s");
    assert!(prepare_data(&recipe, &table, DataMode::Prediction).is_err());
    spec.formula = "y | se(0.5) ~ x".into();
    let training = serde_json::from_value(json!({"nrows": 3, "columns": [
        {"name": "y", "values": {"kind": "real64", "values": [1, 2, 4]}},
        {"name": "x", "values": {"kind": "real64", "values": [0, 1, 3]}}
    ]}))
    .unwrap();
    let converted = build(&spec, &training).unwrap();
    assert_eq!(
        prepare_data(&converted.recipe, &table, DataMode::Prediction)
            .unwrap()
            .constants["se_data"],
        DataValue::Real64(vec![0.5; 3])
    );
}

#[test]
fn residual_scale_flag_controls_coordinates_priors_and_zero_known_scale() {
    let mut spec = spec();
    let coordinates = |result: &flatppl_formula::Conversion| {
        result
            .info
            .parameters
            .iter()
            .filter(|p| !p.derived)
            .map(|p| p.name.clone())
            .collect::<Vec<_>>()
    };
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(coordinates(&converted), ["Intercept", "b_x"]);
    let sigma = serde_json::from_value(json!({"class": "sigma", "prior": "normal(0, 1)"})).unwrap();
    spec.priors.push(sigma);
    assert!(build(&spec, spec.data.as_ref().unwrap()).is_err());

    spec.formula = "y | se(s, sigma = TRUE) ~ x".into();
    spec.family = "student".into();
    spec.priors
        .push(serde_json::from_value(json!({"class": "nu", "prior": "constant(5)"})).unwrap());
    spec.data.as_mut().unwrap().columns[2].values = ColumnValues::Real32(vec![0.0, 0.6, 0.9]);
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(coordinates(&converted), ["Intercept", "b_x", "sigma"]);
    spec.formula = "y | se(s, TRUE) ~ x".into();
    let positional = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(positional.recipe.mu.formula, converted.recipe.mu.formula);
    spec.priors.retain(|prior| prior.class != "sigma");
    spec.formula = "y | se(s, FALSE) ~ x".into();
    assert!(build(&spec, spec.data.as_ref().unwrap()).is_err());
    spec.formula = "y | se(-s, TRUE) ~ x".into();
    assert!(build(&spec, spec.data.as_ref().unwrap()).is_err());
    spec.formula = "y | se(s, TRUE) ~ x".into();
    spec.priors.retain(|prior| prior.class != "nu");
    spec.family = "poisson".into();
    assert!(build(&spec, spec.data.as_ref().unwrap()).is_err());
}
