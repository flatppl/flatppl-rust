use flatppl_formula::{DataMode, DataValue, Recipe, build, prepare_data, read};

const INPUT: &str = r#"{
  "formula": "y ~ I(x^2) + log(z) + offset(log(z))",
  "data": {"nrows":4,"columns":[
    {"name":"y","values":{"kind":"real64","values":[2,4,8,7]}},
    {"name":"x","values":{"kind":"real32","values":[1,2,3,4]}},
    {"name":"z","values":{"kind":"real64","values":[2,3,5,7]}}
  ]}
}"#;

#[test]
fn numeric_expression_matrix_and_prediction_match_r() {
    let spec = read(INPUT).unwrap();
    let result = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    // R 4.6.1 model.matrix / model.offset; regenerate with expressions.R.
    let logs = [
        std::f64::consts::LN_2,
        1.0986122886681098,
        1.6094379124341003,
        1.9459101490553132,
    ];
    assert_eq!(result.info.design_columns, ["I(x^2)", "log(z)"]);
    assert_eq!(
        result.data.constants["x_1"],
        DataValue::Real32(vec![1.0, 4.0, 9.0, 16.0])
    );
    assert_eq!(
        result.data.constants["x_2"],
        DataValue::Real64(logs.to_vec())
    );
    assert_eq!(
        result.data.constants["offset"],
        DataValue::Real64(logs.to_vec())
    );
    let mean_log = logs.iter().sum::<f64>() / 4.0;
    assert_eq!(result.recipe.predictors["mu"].means, [7.5, mean_log]);
    assert_eq!(
        result.recipe.predictors["mu"].priors.intercept_location,
        5.5 - mean_log
    );
    assert_eq!(result.info.parameters[2].name, "b_IxE2");
    assert_eq!(result.info.parameters[3].name, "b_logz");

    let mut table = spec.data.unwrap();
    table.columns.retain(|c| c.name != "y");
    table.columns[0].values = flatppl_formula::ColumnValues::Real32(vec![4.0, 3.0, 2.0, 1.0]);
    let recipe: Recipe =
        serde_json::from_str(&serde_json::to_string(&result.recipe).unwrap()).unwrap();
    let prediction = prepare_data(&recipe, &table, DataMode::Prediction).unwrap();
    assert_eq!(
        prediction.constants["x_1"],
        DataValue::Real32(vec![16.0, 9.0, 4.0, 1.0])
    );
    assert_eq!(
        prediction.constants["offset"],
        result.data.constants["offset"]
    );
    assert_eq!(
        recipe.predictors["mu"].means,
        result.recipe.predictors["mu"].means
    );
}

#[test]
fn offset_subtraction_and_expression_missing_rows_follow_r() {
    let mut spec = read(INPUT).unwrap();
    spec.formula = "y ~ log(z) + offset(x) - offset(x)".into();
    spec.data.as_mut().unwrap().columns[2].values =
        flatppl_formula::ColumnValues::Real64(vec![2.0, -1.0, 5.0, 7.0]);
    let result = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(result.data.rows, [0, 2, 3]);
    assert_eq!(
        result.data.constants["offset"],
        DataValue::Real32(vec![1.0, 3.0, 4.0])
    );
    assert_eq!(
        result.recipe.predictors["mu"].priors.intercept_location,
        7.0 - 8.0 / 3.0
    );
}

#[test]
fn unary_expression_names_preserve_r_coefficient_identity() {
    let mut spec = read(INPUT).unwrap();
    spec.formula = "y ~ I(+x) + I(-x)".into();
    let result = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(result.info.design_columns, ["I(+x)", "I(-x)"]);
    assert_eq!(result.info.parameters[2].name, "b_IPx");
    assert_eq!(result.info.parameters[3].name, "b_IMx");
    assert_eq!(
        result.data.constants["x_1"],
        DataValue::Real32(vec![1.0, 2.0, 3.0, 4.0])
    );
    assert_eq!(
        result.data.constants["x_2"],
        DataValue::Real32(vec![-1.0, -2.0, -3.0, -4.0])
    );
}
