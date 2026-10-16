use flatppl_formula::{ColumnValues, DataMode, DataValue, Recipe, build, prepare_data, read};
use serde_json::json;

#[test]
fn categorical_predictors_and_saved_response_coding_follow_brms() {
    let spec = read(include_str!("fixtures/categorical.formula.json")).unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/categorical-oracle.json")).unwrap();
    assert_eq!(converted.recipe.category_predictors, ["mublue", "mugreen"]);
    assert_eq!(
        converted.data.constants["y_data"],
        DataValue::Integers(serde_json::from_value(oracle["Y"].clone()).unwrap())
    );
    assert_eq!(converted.recipe.predictors.len(), 2);
    for name in &converted.recipe.category_predictors {
        let predictor = &converted.recipe.predictors[name];
        let mean = oracle["X"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row[1].as_f64().unwrap())
            .sum::<f64>()
            / 6.0;
        assert_eq!(predictor.means, [mean]);
        assert_eq!(predictor.priors.intercept.prior, "student_t(3, 0, 2.5)");
        assert_eq!(predictor.priors.b[0].prior, "Lebesgue(reals)");
    }
    assert_eq!(
        converted
            .info
            .parameters
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>(),
        [
            "Intercept_mublue",
            "b_mublue_Intercept",
            "b_mublue_x",
            "Intercept_mugreen",
            "b_mugreen_Intercept",
            "b_mugreen_x"
        ]
    );
    assert!(converted.info.has_improper_prior);
    let recipe: Recipe =
        serde_json::from_str(&serde_json::to_string(&converted.recipe).unwrap()).unwrap();
    let mut table = spec.data.unwrap();
    table.nrows = 3;
    table.columns.truncate(2);
    table.columns[0].values = ColumnValues::Factor {
        levels: vec!["green".into(), "blue".into(), "red".into(), "unused".into()],
        codes: vec![0, 1, 2],
        ordered: false,
    };
    table.columns[1].values = ColumnValues::Real64(vec![-1.0, 2.0, 8.0]);
    assert_eq!(
        prepare_data(&recipe, &table, DataMode::Training)
            .unwrap()
            .constants["y_data"],
        DataValue::Integers(serde_json::from_value(oracle["prediction_Y"].clone()).unwrap())
    );
    table.columns.remove(0);
    let prediction = prepare_data(&recipe, &table, DataMode::Prediction).unwrap();
    for name in &recipe.category_predictors {
        assert_eq!(
            prediction.constants[&format!("x_1_{name}")],
            DataValue::Real64(vec![-1.0, 2.0, 8.0])
        );
        assert_eq!(recipe.predictors[name].means, [3.5]);
    }
}

#[test]
fn category_overrides_select_their_own_rows_designs_and_priors() {
    let mut spec = read(include_str!("fixtures/categorical.formula.json")).unwrap();
    spec.formula = "y | weights(w) ~ x".into();
    spec.dpars = serde_json::from_value(json!({
        "mublue":{"formula":"mublue ~ 0 + z"},
        "mugreen":{"formula":"mugreen ~ z + offset(z)"}
    }))
    .unwrap();
    spec.priors = serde_json::from_value(json!([
        {"class":"b","dpar":"mublue","prior":"normal(0,1.1)"},
        {"class":"b","dpar":"mugreen","prior":"normal(0,0.7)"},
        {"class":"Intercept","dpar":"mugreen","prior":"normal(0.2,1.3)"}
    ]))
    .unwrap();
    let table = spec.data.as_mut().unwrap();
    table.columns[1].valid = Some(vec![false, true, true, true, true, true]);
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(converted.data.rows, [0, 1, 2, 3, 4, 5]);
    assert!(!converted.recipe.predictors["mublue"].center);
    assert_eq!(converted.recipe.predictors["mugreen"].means, [2.0 / 3.0]);
    assert_eq!(
        converted.recipe.predictors["mublue"].priors.b[0].prior,
        "normal(0,1.1)"
    );
    assert_eq!(
        converted.recipe.predictors["mugreen"]
            .priors
            .intercept
            .prior,
        "normal(0.2,1.3)"
    );
    assert!(!converted.info.has_improper_prior);
    assert_eq!(
        converted.data.constants["offset_mugreen"],
        DataValue::Real64(vec![0., 1., 0., 2., 1., 0.])
    );
    let mut prediction = spec.data.unwrap();
    prediction.columns.retain(|c| c.name == "z");
    assert_eq!(
        prepare_data(&converted.recipe, &prediction, DataMode::Prediction)
            .unwrap()
            .rows,
        converted.data.rows
    );
}

#[test]
fn category_names_preserve_numeric_levels_and_detect_ambiguous_coding() {
    let mut spec = read(include_str!("fixtures/categorical.formula.json")).unwrap();
    spec.data.as_mut().unwrap().columns[0].values = ColumnValues::Int(vec![-2, 7, 3, -2, 7, 3]);
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(converted.recipe.category_predictors, ["mu3", "mu7"]);
    assert_eq!(
        converted.data.constants["y_data"],
        DataValue::Integers(vec![1, 3, 2, 1, 3, 2])
    );
    spec.data.as_mut().unwrap().columns[0].values = ColumnValues::Real64(vec![
        0.0,
        100000.0,
        1.234567890123456,
        0.0,
        100000.0,
        1.234567890123456,
    ]);
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(
        converted.recipe.category_predictors,
        ["mu123456789012346", "mu1e05"]
    );
    spec.data.as_mut().unwrap().columns[0].values = ColumnValues::Real64(vec![
        0.0,
        6.508_174_408_227_206e-15,
        3161646742373.705,
        0.0,
        6.508_174_408_227_206e-15,
        3161646742373.705,
    ]);
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(
        converted.recipe.category_predictors,
        ["mu650817440822721e15", "mu31616467423737"]
    );
    spec.data.as_mut().unwrap().columns[0].values = ColumnValues::Factor {
        levels: vec!["red".into(), "x.y".into(), "7 weird".into()],
        codes: vec![0, 1, 2, 0, 2, 1],
        ordered: false,
    };
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(converted.recipe.category_predictors, ["muxy", "mu7weird"]);
    assert_eq!(
        converted
            .info
            .parameters
            .iter()
            .filter(|p| !p.derived)
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>(),
        [
            "Intercept_muxy",
            "b_muxy_x",
            "Intercept_mu7weird",
            "b_mu7weird_x"
        ]
    );
    if let ColumnValues::Factor { levels, .. } = &mut spec.data.as_mut().unwrap().columns[0].values
    {
        levels[1] = "a b".into();
        levels[2] = "a.b".into();
    }
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(converted.recipe.category_predictors, ["muab1", "muab"]);
    if let ColumnValues::Factor { levels, .. } = &mut spec.data.as_mut().unwrap().columns[0].values
    {
        levels[1] = "a_b".into();
        levels[2] = "ab".into();
    }
    assert!(build(&spec, spec.data.as_ref().unwrap()).is_err());
    spec.data.as_mut().unwrap().columns[0].values = ColumnValues::Int(vec![-2, 7, 3, -2, 7, 3]);
    spec.data.as_mut().unwrap().columns[1].valid = Some(vec![false, true, true, false, true, true]);
    assert!(build(&spec, spec.data.as_ref().unwrap()).is_err());
}

#[test]
fn category_correlations_use_global_and_group_selectors() {
    let mut spec = read(include_str!("fixtures/categorical.formula.json")).unwrap();
    spec.formula = "y ~ x + (1 + x | z)".into();
    spec.priors = serde_json::from_value(json!([
        {"class":"cor","prior":"lkj(2)"},
        {"class":"L","group":"z","prior":"lkj(4)"},
        {"class":"sd","dpar":"mublue","prior":"normal(0,1)"}
    ]))
    .unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    for name in &converted.recipe.category_predictors {
        assert_eq!(
            converted.recipe.predictors[name].priors.cor[0]
                .as_ref()
                .unwrap()
                .prior,
            "lkj(4)"
        );
    }
    assert_eq!(
        converted.recipe.predictors["mublue"].priors.sd[0][0].prior,
        "normal(0,1)"
    );
    assert_eq!(
        converted.recipe.predictors["mugreen"].priors.sd[0][0].prior,
        "student_t(3, 0, 2.5)"
    );
    spec.priors.truncate(2);
    spec.dpars = serde_json::from_value(json!({"mublue":{"formula":"mublue ~ x"}})).unwrap();
    assert!(build(&spec, spec.data.as_ref().unwrap()).is_ok());
    for prior in [
        json!({"class":"cor","dpar":"mugreen","prior":"lkj(3)"}),
        json!({"class":"b","prior":"normal(0,2)"}),
        json!({"class":"sd","prior":"normal(0,2)"}),
    ] {
        spec.priors = vec![serde_json::from_value(prior).unwrap()];
        assert!(build(&spec, spec.data.as_ref().unwrap()).is_err());
    }
}
