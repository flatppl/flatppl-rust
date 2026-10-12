use flatppl_formula::{
    ColumnValues, DataMode, DataValue, Family, Link, Trials, build, prepare_data, read,
};
use serde_json::json;

#[test]
fn response_support_matches_brms() {
    for (family, valid, invalid) in [
        ("lognormal", vec![0.1, 1.0, 2.0], vec![0.0, -0.1]),
        ("geometric", vec![0.0, 1.0, 2.0], vec![-1.0, 0.5]),
        (
            "zero_inflated_poisson",
            vec![0.0, 1.0, 2.0],
            vec![-1.0, 0.5],
        ),
        (
            "zero_inflated_negbinomial",
            vec![0.0, 1.0, 2.0],
            vec![-1.0, 0.5],
        ),
        ("weibull", vec![0.1, 1.0, 2.0], vec![0.0, -0.1]),
        ("beta", vec![0.1, 0.5, 0.9], vec![0.0, 1.0]),
        (
            "von_mises",
            vec![-std::f64::consts::PI, 0.0, std::f64::consts::PI],
            vec![-3.2, 3.2],
        ),
    ] {
        let mut spec = read(
            &json!({
                "formula": "y ~ 1", "family": family,
                "data": {"nrows": 3, "columns": [
                    {"name": "y", "values": {"kind": "real64", "values": valid}}
                ]}
            })
            .to_string(),
        )
        .unwrap();
        build(&spec, spec.data.as_ref().unwrap()).unwrap();
        for value in invalid {
            spec.data.as_mut().unwrap().columns[0].values = ColumnValues::Real64(vec![value; 3]);
            assert!(
                build(&spec, spec.data.as_ref().unwrap()).is_err(),
                "{family}: {value}"
            );
        }
    }
}

#[test]
fn mixture_probability_constraints_apply_to_constant_overrides() {
    for (value, valid) in [
        (0.0, true),
        (0.4, true),
        (1.0, true),
        (-0.1, false),
        (1.1, false),
    ] {
        let spec = read(
            &json!({
                "formula": "y ~ x", "family": "zero_inflated_poisson",
                "priors": [{"class": "zi", "prior": format!("constant({value})")}],
                "data": {"nrows": 4, "columns": [
                    {"name": "x", "values": {"kind": "int", "values": [1, 2, 3, 4]}},
                    {"name": "y", "values": {"kind": "int", "values": [0, 2, 4, 0]}}
                ]}
            })
            .to_string(),
        )
        .unwrap();
        let result = build(&spec, spec.data.as_ref().unwrap());
        assert_eq!(result.is_ok(), valid, "{value}: {:?}", result.err());
    }
}

#[test]
fn link_scaled_default_priors_match_brms() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/family-priors.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let spec = read(
            &json!({
                "formula": case.get("formula").unwrap_or(&fixture["formula"]), "family": case["family"], "link": case["link"],
                "data": {"nrows": 4, "columns": [
                    {"name": "y", "values": case["response"]},
                    {"name": "x", "values": {"kind": "int", "values": [1, 2, 3, 4]}},
                    {"name": "g", "values": {"kind": "int", "values": [1, 2, 1, 2]}}
                ]}
            })
            .to_string(),
        )
        .unwrap();
        let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
        let priors = &converted.recipe.priors;
        assert_eq!(
            priors.intercept_location,
            case["intercept_location"].as_f64().unwrap()
        );
        assert_eq!(priors.scale, case["scale"].as_f64().unwrap());
        if let Some(scale) = case["sigma_scale"].as_f64() {
            assert_eq!(priors.sigma_scale, scale);
        }
        if let Some(prior) = case["intercept_prior"].as_str() {
            assert_eq!(priors.intercept.prior, prior);
        }
    }
}

#[test]
fn bernoulli_response_levels_survive_new_data_and_recipe_roundtrip() {
    let spec = read(&json!({
        "formula": "y ~ x", "family": "bernoulli",
        "data": {"nrows": 4, "columns": [
            {"name": "x", "values": {"kind": "real32", "values": [1, 2, 3, 4]}},
            {"name": "y", "valid": [true, true, true, false], "values": {
                "kind": "factor", "values": {"levels": ["no", "yes", "unused"], "codes": [1, 0, 1, 2]}
            }}
        ]}
    }).to_string()).unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(converted.recipe.family, Family::Bernoulli);
    assert_eq!(converted.recipe.link, Link::Logit);
    assert_eq!(converted.data.rows, [0, 1, 2]);
    assert_eq!(
        converted.data.constants["y_data"],
        DataValue::Integers(vec![1, 0, 1])
    );
    assert_eq!(converted.recipe.priors.intercept_location, 0.0);
    assert_eq!(converted.recipe.priors.scale, 2.5);
    let saved = serde_json::to_string(&converted.recipe).unwrap();
    let recipe = serde_json::from_str(&saved).unwrap();
    let mut table = spec.data.unwrap();
    table.columns[1].values = ColumnValues::Factor {
        levels: vec!["yes".into(), "no".into()],
        codes: vec![0, 1, 0, 1],
        ordered: false,
    };
    assert_eq!(
        prepare_data(&recipe, &table, DataMode::Training).unwrap(),
        converted.data
    );
    table.columns.retain(|column| column.name != "y");
    let prediction = prepare_data(&recipe, &table, DataMode::Prediction).unwrap();
    assert_eq!(prediction.rows, [0, 1, 2, 3]);
    assert_eq!(
        prediction.constants["x_1"],
        DataValue::Real32(vec![1.0, 2.0, 3.0, 4.0])
    );
}

#[test]
fn bernoulli_numeric_event_coding_matches_brms() {
    for (values, expected) in [
        (vec![-3, 7, -3], vec![0, 1, 0]),
        (vec![0, 0, 0], vec![0, 0, 0]),
        (vec![5, 5, 5], vec![1, 1, 1]),
    ] {
        let spec = read(&json!({
            "formula": "y ~ 1", "family": "bernoulli", "link": "identity",
            "data": {"nrows": 3, "columns": [{"name": "y", "values": {"kind": "int", "values": values}}]}
        }).to_string()).unwrap();
        let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
        assert_eq!(
            converted.data.constants["y_data"],
            DataValue::Integers(expected)
        );
    }
    let spec = read(&json!({
        "formula": "y ~ 1", "family": "bernoulli",
        "data": {"nrows": 2, "columns": [{"name": "y", "values": {"kind": "int", "values": [-2, 7]}}]}
    }).to_string()).unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    let mut table = spec.data.as_ref().unwrap().clone();
    table.columns[0].values = ColumnValues::Int(vec![7, 7]);
    let prepared = prepare_data(&converted.recipe, &table, DataMode::Training).unwrap();
    assert_eq!(
        prepared.constants["y_data"],
        DataValue::Integers(vec![1, 1])
    );
    table.columns[0].values = ColumnValues::Bool(vec![true, true]);
    let converted = build(&spec, &table).unwrap();
    assert_eq!(
        converted.data.constants["y_data"],
        DataValue::Integers(vec![0, 0])
    );
}

#[test]
fn binomial_trials_follow_selected_rows_and_remain_prediction_inputs() {
    for family in ["binomial", "zero_inflated_binomial"] {
        let mut spec = read(&json!({
            "formula": "y | trials(n) ~ x", "family": family,
            "data": {"nrows": 4, "columns": [
                {"name": "x", "values": {"kind": "real32", "values": [1, 2, 3, 4]}},
                {"name": "y", "values": {"kind": "real64", "values": [2, 1, 0, 9]}},
                {"name": "n", "valid": [true, true, true, false], "values": {"kind": "int", "values": [4, 1, 0, 0]}}
            ]}
        }).to_string()).unwrap();
        let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
        let mut invalid = spec.data.as_ref().unwrap().clone();
        invalid.columns[1].values = ColumnValues::Real64(vec![5.0, 1.0, 0.0, 9.0]);
        assert!(build(&spec, &invalid).is_err());
        assert_eq!(
            converted.recipe.formula.trials,
            Some(Trials::Column("n".into()))
        );
        assert_eq!(converted.data.rows, [0, 1, 2]);
        assert_eq!(
            converted.data.constants["y_data"],
            DataValue::Integers(vec![2, 1, 0])
        );
        assert_eq!(
            converted.data.constants["trials_data"],
            DataValue::Integers(vec![4, 1, 0])
        );
        let mut table = spec.data.take().unwrap();
        table.columns.retain(|column| column.name != "y");
        let prediction = prepare_data(&converted.recipe, &table, DataMode::Prediction).unwrap();
        assert_eq!(
            prediction.constants["trials_data"],
            converted.data.constants["trials_data"]
        );
        table.columns[1].values = ColumnValues::Real64(vec![4.0, 1.5, 0.0, 0.0]);
        assert!(prepare_data(&converted.recipe, &table, DataMode::Prediction).is_err());

        spec.formula = "y | trials(10) ~ 1".into();
        let table = serde_json::from_value(json!({"nrows": 3, "columns": [
            {"name": "y", "values": {"kind": "int", "values": [0, 3, 10]}}
        ]}))
        .unwrap();
        let converted = build(&spec, &table).unwrap();
        assert_eq!(converted.recipe.formula.trials, Some(Trials::Constant(10)));
        assert_eq!(
            converted.data.constants["y_data"],
            DataValue::Integers(vec![0, 3, 10])
        );
    }
}
