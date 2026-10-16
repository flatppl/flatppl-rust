use flatppl_formula::{ColumnValues, DataMode, DataValue, Recipe, build, prepare_data, read};
use serde_json::{Value, json};

#[test]
fn distributional_coordinates_and_priors_follow_brms() {
    let mut spec = read(include_str!("fixtures/distributional.formula.json")).unwrap();
    spec.priors = serde_json::from_value(json!([
        {"class": "b", "prior": "normal(0, 2)"},
        {"class": "Intercept", "dpar": "sigma", "prior": "normal(0.2, 0.8)"},
        {"class": "b", "dpar": "sigma", "prior": "normal(0, 0.7)"}
    ]))
    .unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    let mu = &converted.recipe.predictors["mu"];
    let sigma = &converted.recipe.predictors["sigma"];
    assert_eq!(mu.means, [2.5]);
    assert_eq!(sigma.means, [0.75]);
    let oracle: Value =
        serde_json::from_str(include_str!("fixtures/distributional-gaussian-oracle.json")).unwrap();
    for (predictor, field) in [(mu, "X"), (sigma, "X_sigma")] {
        let matrix = oracle["gaussian_custom"][field].as_array().unwrap();
        let mean = matrix
            .iter()
            .map(|row| row[1].as_f64().unwrap())
            .sum::<f64>()
            / matrix.len() as f64;
        assert_eq!(predictor.means, [mean]);
    }
    assert_eq!(mu.priors.intercept.prior, "student_t(3, 5.5, 3)");
    assert_eq!(mu.priors.b[0].prior, "normal(0, 2)");
    assert_eq!(sigma.priors.intercept.prior, "normal(0.2, 0.8)");
    assert_eq!(sigma.priors.b[0].prior, "normal(0, 0.7)");
    assert!(converted.recipe.auxiliary_priors.is_empty());
    let coordinates: Vec<_> = converted
        .info
        .parameters
        .iter()
        .map(|p| (p.name.as_str(), p.binding.as_str(), p.index, p.derived))
        .collect();
    assert_eq!(
        coordinates,
        [
            ("Intercept", "temp_Intercept", None, false),
            ("b_Intercept", "b_Intercept", None, true),
            ("b_x", "b", Some(1), false),
            ("Intercept_sigma", "temp_Intercept_sigma", None, false),
            ("b_sigma_Intercept", "b_Intercept_sigma", None, true),
            ("b_sigma_g", "b_sigma", Some(1), false),
        ]
    );
    assert!(!converted.info.has_improper_prior);
    // A class-wide prior with no dpar affects only the mean, exactly as brms.
    spec.priors.retain(|p| p.dpar.is_none());
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(
        converted.recipe.predictors["sigma"].priors.b[0].prior,
        "Lebesgue(reals)"
    );
    assert!(converted.info.has_improper_prior);
    spec.priors
        .push(serde_json::from_value(json!({"class":"sigma","prior":"normal(0,1)"})).unwrap());
    assert!(build(&spec, spec.data.as_ref().unwrap()).is_err());
}

#[test]
fn missing_rows_and_prediction_share_each_fitted_recipe() {
    let spec = read(&json!({
        "formula":"y ~ x", "dpars":{"sigma":{"formula":"sigma ~ g"}},
        "data":{"nrows":6,"columns":[
            {"name":"y","values":{"kind":"real64","values":[2,4,8,7,9,5]},"valid":[true,true,true,false,true,true]},
            {"name":"x","values":{"kind":"real32","values":[1,2,3,4,5,6]},"valid":[true,true,true,true,true,false]},
            {"name":"g","values":{"kind":"factor","values":{"levels":["B","A","unused"],"codes":[0,1,0,1,0,1]}},"valid":[true,true,false,true,true,true]}
        ]}
    }).to_string()).unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    assert_eq!(converted.data.rows, [0, 1, 4]);
    let oracle: Value =
        serde_json::from_str(include_str!("fixtures/distributional-missing.json")).unwrap();
    assert_eq!(
        converted.data.constants["y_data"],
        DataValue::Real64(serde_json::from_value(oracle["Y"].clone()).unwrap())
    );
    assert_eq!(converted.recipe.predictors["mu"].means, [8.0 / 3.0]);
    assert_eq!(converted.recipe.predictors["sigma"].means, [1.0 / 3.0]);
    let saved = serde_json::to_string(&converted.recipe).unwrap();
    let restored: Recipe = serde_json::from_str(&saved).unwrap();
    let mut new_data = spec.data.unwrap();
    new_data.nrows = 2;
    new_data.columns.remove(0);
    for column in &mut new_data.columns {
        column.valid = None;
    }
    new_data.columns[0].values = ColumnValues::Real32(vec![7.0, 9.0]);
    new_data.columns[1].values = ColumnValues::Factor {
        levels: vec!["A".into(), "B".into()],
        codes: vec![0, 1],
        ordered: false,
    };
    let prediction = prepare_data(&restored, &new_data, DataMode::Prediction).unwrap();
    assert_eq!(
        prediction.constants["x_1"],
        DataValue::Real32(vec![7.0, 9.0])
    );
    assert_eq!(
        prediction.constants["x_1_sigma"],
        DataValue::Integers(vec![2, 1])
    );
    assert!(!prediction.constants.contains_key("y_data"));
    assert_eq!(serde_json::to_string(&restored).unwrap(), saved);
    assert_eq!(converted.data.rows, [0, 1, 4]);
}

#[test]
fn auxiliary_predictor_defaults_match_pinned_brms() {
    let cases: Value =
        serde_json::from_str(include_str!("fixtures/distributional-defaults.json")).unwrap();
    for (name, case) in cases.as_object().unwrap() {
        let dpar = case["dpar"].as_str().unwrap();
        let spec = read(
            &json!({
                "formula":"y ~ x", "family":case["family"],
                "dpars":{dpar:{"formula":format!("{dpar} ~ g")}},
                "data":{"nrows":4,"columns":[
                    {"name":"y","values":{"kind":"real64","values":case["y"]}},
                    {"name":"x","values":{"kind":"int","values":[1,2,3,4]}},
                    {"name":"g","values":{"kind":"real64","values":[0,1,0,2]}}
                ]}
            })
            .to_string(),
        )
        .unwrap();
        let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
        let predictor = &converted.recipe.predictors[dpar];
        assert_eq!(
            predictor.priors.intercept.prior,
            case["prior"].as_str().unwrap(),
            "{name}"
        );
        assert_eq!(
            predictor.link.name(),
            case["link"].as_str().unwrap(),
            "{name}"
        );
        assert!(!converted.recipe.auxiliary_priors.contains_key(dpar));
    }
}

#[test]
fn group_predictors_keep_dpar_selectors_and_brms_coordinate_names() {
    let mut spec = read(include_str!("fixtures/distributional.formula.json")).unwrap();
    spec.formula = "y ~ x + (1 | h)".into();
    spec.dpars.get_mut("sigma").unwrap().formula = "sigma ~ g + (1 + g || h)".into();
    spec.data.as_mut().unwrap().columns.push(
        serde_json::from_value(json!({
            "name":"h", "values":{"kind":"factor","values":{
                "levels":["B","A"],"codes":[0,1,0,1]
            }}
        }))
        .unwrap(),
    );
    spec.priors = serde_json::from_value(json!([
        {"class":"sd","prior":"normal(0, 2)"},
        {"class":"sd","group":"h","dpar":"sigma","prior":"normal(0, 1)"},
        {"class":"sd","coef":"g","group":"h","dpar":"sigma","prior":"normal(0, 0.5)"}
    ]))
    .unwrap();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    let mu = &converted.recipe.predictors["mu"];
    let sigma = &converted.recipe.predictors["sigma"];
    assert_eq!(mu.priors.sd[0][0].prior, "normal(0, 2)");
    assert_eq!(sigma.priors.sd[0][0].prior, "normal(0, 1)");
    assert_eq!(sigma.priors.sd[0][1].prior, "normal(0, 0.5)");
    let oracle: Value =
        serde_json::from_str(include_str!("fixtures/distributional-groups.json")).unwrap();
    for (predictor, field) in [(mu, "J_mu"), (sigma, "J_sigma")] {
        assert_eq!(
            converted.data.constants[&predictor.groups[0].index],
            DataValue::Integers(serde_json::from_value(oracle[field].clone()).unwrap())
        );
    }
    let parameter = |name: &str| {
        converted
            .info
            .parameters
            .iter()
            .find(|p| p.name == name)
            .unwrap()
    };
    assert_eq!(parameter("z_1[1,2]").binding, mu.groups[0].z);
    assert_eq!(parameter("z_1[1,2]").index, Some(2));
    assert_eq!(parameter("z_2[2,1]").binding, sigma.groups[0].z);
    assert_eq!(parameter("z_2[2,1]").indices, Some(vec![2, 1]));
    assert_eq!(parameter("sd_h__sigma_g").binding, "sd_h_sigma_2");
    assert!(parameter("r_h[B,sigma_g]").derived);
    assert!(sigma.groups[0].correlation.is_none());

    spec.priors.clear();
    let converted = build(&spec, spec.data.as_ref().unwrap()).unwrap();
    let sigma = &converted.recipe.predictors["sigma"];
    assert_eq!(sigma.priors.intercept.prior, "student_t(3, 0, 2.5)");
    // brms uses the response-family scale for all group sds, including sigma's.
    assert_eq!(sigma.priors.sd[0][0].prior, "student_t(3, 0, 3)");
    assert_eq!(sigma.priors.sd[0][1].prior, "student_t(3, 0, 3)");
}
