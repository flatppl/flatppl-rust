use flatppl_formula::{
    Column, ColumnValues, DataMode, DataValue, Recipe, Spec, build, parse_formula, prepare_data,
    read,
};
use serde_json::json;

fn spec(formula: &str) -> Spec {
    read(&json!({
        "formula": formula,
        "data": {"nrows": 4, "columns": [
            {"name":"y", "values":{"kind":"real64", "values":[2,4,8,7]}},
            {"name":"x", "values":{"kind":"real32", "values":[1,2,3,4]}},
            {"name":"f", "values":{"kind":"factor", "values":{"levels":["b","a"], "codes":[0,1,0,1]}}}
        ]}
    }).to_string()).unwrap()
}

#[test]
fn explicit_intercept_is_an_uncentred_coefficient_in_training_and_prediction() {
    let mut spec = spec("y ~ 0 + Intercept + x");
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    let mu = &converted.recipe.predictors["mu"];
    assert!(!mu.formula.intercept);
    assert!(!mu.center);
    assert_eq!(mu.means, [1.0, 2.5]);
    assert!(mu.priors.b.iter().all(|p| p.prior == "Lebesgue(reals)"));
    assert!(converted.info.has_improper_prior);
    assert_eq!(
        converted.data.constants["x_1"],
        DataValue::Real64(vec![1.0; 4])
    );
    assert_eq!(
        converted.data.constants["x_2"],
        DataValue::Real32(vec![1.0, 2.0, 3.0, 4.0])
    );
    let parameters: Vec<_> = converted
        .info
        .parameters
        .iter()
        .map(|p| (p.name.as_str(), p.derived))
        .collect();
    assert_eq!(
        parameters,
        [("b_Intercept", false), ("b_x", false), ("sigma", false)]
    );

    spec.priors = serde_json::from_value(json!([
        {"class":"b", "prior":"normal(0, 2)"},
        {"class":"b", "coef":"Intercept", "prior":"normal(1, 3)"},
        {"class":"sigma", "prior":"exponential(1)"}
    ]))
    .unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(
        converted.recipe.predictors["mu"].priors.b[0].prior,
        "normal(1, 3)"
    );
    assert_eq!(
        converted.recipe.predictors["mu"].priors.b[1].prior,
        "normal(0, 2)"
    );
    assert!(!converted.info.has_improper_prior);
    let saved: Recipe =
        serde_json::from_str(&serde_json::to_string(&converted.recipe).unwrap()).unwrap();
    let mut table = spec.data.unwrap();
    table.nrows = 2;
    table.columns.retain(|column| column.name == "x");
    table.columns[0].values = ColumnValues::Real32(vec![5.0, 8.0]);
    let prediction = prepare_data(&saved, &table, DataMode::Prediction).unwrap();
    assert_eq!(
        prediction.constants["x_1"],
        DataValue::Real64(vec![1.0, 1.0])
    );
    assert_eq!(
        prediction.constants["x_2"],
        DataValue::Real32(vec![5.0, 8.0])
    );
}

#[test]
fn explicit_intercept_preserves_brms_factor_coding_and_expression_names() {
    // Frozen brms 2.23.2 cases are reproduced by fixtures/explicit-intercept.R.
    for (formula, names) in [
        ("y ~ 0 + Intercept + f", vec!["Intercept", "fa"]),
        ("y ~ 0 + I(2*Intercept) + f", vec!["I(2 * Intercept)", "fa"]),
        ("y ~ 0 + Intercept:x + f", vec!["fa", "Intercept:x"]),
        ("y ~ 0 + x + Intercept", vec!["x", "Intercept"]),
    ] {
        let spec = spec(formula);
        let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
        assert_eq!(
            converted.recipe.predictors["mu"]
                .design
                .columns
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            names
        );
        assert!(!converted.recipe.predictors["mu"].center);
        if formula.contains("I(") {
            assert_eq!(
                converted.data.constants["x_1"],
                DataValue::Real64(vec![2.0; 4])
            );
        }
    }
    let spec = spec("y ~ 0 + Intercept + (0 + Intercept + x | f)");
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(
        converted.recipe.predictors["mu"].groups[0].coefficients,
        ["Intercept", "x"]
    );
    let mut spec = spec;
    spec.formula = "y ~ 0 + Intercept + (0 + Intercept + f | x)".into();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(
        converted.recipe.predictors["mu"].groups[0].coefficients,
        ["Intercept", "fb", "fa"]
    );
    spec.formula = "y ~ x".into();
    spec.dpars =
        serde_json::from_value(json!({"sigma":{"formula":"sigma ~ 0 + Intercept"}})).unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert!(!converted.recipe.predictors["sigma"].center);
    assert_eq!(
        converted.recipe.predictors["sigma"].priors.b[0].prior,
        "Lebesgue(reals)"
    );
}

#[test]
fn reserved_intercept_does_not_accept_observed_values_or_an_implicit_intercept() {
    for formula in [
        "y ~ Intercept + x",
        "y ~ 0 + x + (0 + Intercept | f)",
        "y ~ 0 + x + offset(Intercept)",
    ] {
        assert!(parse_formula(formula).is_err(), "{formula}");
    }
    let mut spec = spec("y ~ 0 + Intercept + x");
    spec.data.as_mut().unwrap().columns.push(Column {
        name: "Intercept".into(),
        values: ColumnValues::Int(vec![1; 4]),
        valid: None,
    });
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    spec.data
        .as_mut()
        .unwrap()
        .columns
        .last_mut()
        .unwrap()
        .values = ColumnValues::Int(vec![1, 2, 1, 1]);
    assert!(
        build(&spec, spec.data.as_ref().unwrap())
            .err()
            .unwrap()
            .0
            .contains("only ones")
    );
    assert!(
        prepare_data(
            &converted.recipe,
            spec.data.as_ref().unwrap(),
            DataMode::Prediction
        )
        .unwrap_err()
        .0
        .contains("only ones")
    );
    let column = spec.data.as_mut().unwrap().columns.last_mut().unwrap();
    column.values = ColumnValues::Int(vec![1; 4]);
    column.valid = Some(vec![true, false, true, true]);
    assert!(
        build(&spec, spec.data.as_ref().unwrap())
            .err()
            .unwrap()
            .0
            .contains("only ones")
    );
    spec.priors =
        serde_json::from_value(json!([{"class":"Intercept", "prior":"normal(0,1)"}])).unwrap();
    spec.data.as_mut().unwrap().columns.pop();
    assert!(build(&spec, spec.data.as_ref().unwrap()).is_err());
}
