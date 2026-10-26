use flatppl_host::{
    Context, EmitOptions,
    flatppl_formula::{self, DataMode},
};

#[test]
fn categorical_prediction_reuses_each_category_predictor() -> Result<(), Box<dyn std::error::Error>>
{
    let spec = flatppl_formula::read(include_str!(
        "../../formula/tests/fixtures/categorical.formula.json"
    ))?;
    let mut context = Context::default();
    let model = context.import_formula(&spec, spec.data.as_ref().unwrap(), None, None)?;
    context.register("categorical.flatppl", &model.module)?;
    let parameters = "temp_Intercept_mublue=p[1], b_mublue=[p[2]], temp_Intercept_mugreen=p[3], b_mugreen=[p[4]]";
    let score = context.parse(
        &format!(
            r#"
        m = load_module("categorical.flatppl")
        p = elementof(cartpow(reals, 4))
        inputs = p
        outputs = record(posterior = logdensityof(m.Pi_post, record({parameters})),
            pointwise = m.log_lik({parameters}))
    "#
        ),
        None,
        None,
    )?;
    let training = score.compile(&EmitOptions::default())?;
    let prediction = flatppl_formula::read_json_table(r#"{"x":[-1,2,8]}"#)?;
    let prediction = model.prepare(&mut context, &prediction, DataMode::Prediction)?;
    context.register("prediction.flatppl", &prediction)?;
    let query = context.parse(
        &format!(
            r#"
        m = load_module("prediction.flatppl")
        p = elementof(cartpow(reals, 4))
        inputs = p
        outputs = logdensityof(m.K_model({parameters}), record(y=[3,2,1]))
    "#
        ),
        None,
        None,
    )?;
    query.compile(&EmitOptions::default())?;
    assert_eq!(
        score.compile(&EmitOptions::default())?.stablehlo,
        training.stablehlo
    );
    Ok(())
}

#[test]
fn coefficient_priors_can_reference_later_family_auxiliaries()
-> Result<(), Box<dyn std::error::Error>> {
    let spec = flatppl_formula::read(
        &serde_json::json!({
            "formula": "y ~ 0 + x", "family": "beta",
            "priors": [
                {"class": "b", "prior": "normal(0, phi)"},
                {"class": "phi", "prior": "constant(2.5)"}
            ],
            "data": {"nrows": 2, "columns": [
                {"name": "y", "values": {"kind": "real64", "values": [0.2, 0.6]}},
                {"name": "x", "values": {"kind": "real64", "values": [1, 2]}}
            ]}
        })
        .to_string(),
    )?;
    let mut context = Context::default();
    let model = context.import_formula(&spec, spec.data.as_ref().unwrap(), None, None)?;
    context.register("model.flatppl", &model.module)?;
    let query = context.parse(
        "m = load_module(\"model.flatppl\")\np = elementof(cartpow(reals, 1))\n\
                 inputs = p\noutputs = logdensityof(m.Pi_post, record(b = p))",
        None,
        None,
    )?;
    query.compile(&EmitOptions::default())?;
    Ok(())
}

#[test]
fn formula_training_and_prediction_compile_from_distinct_snapshots()
-> Result<(), Box<dyn std::error::Error>> {
    let mut spec = flatppl_formula::read(include_str!(
        "../../formula/tests/fixtures/gaussian.formula.json"
    ))?;
    compile_from_distinct_snapshots(&spec)?;
    spec.priors.clear();
    compile_from_distinct_snapshots(&spec)
}

#[test]
fn distributional_prediction_reuses_both_predictors_without_response()
-> Result<(), Box<dyn std::error::Error>> {
    let oracle: serde_json::Value = serde_json::from_str(include_str!(
        "../../formula/tests/fixtures/distributional-prediction.json"
    ))?;
    let spec = flatppl_formula::read(include_str!(
        "../../formula/tests/fixtures/distributional.formula.json"
    ))?;
    let mut context = Context::default();
    let model = context.import_formula(&spec, spec.data.as_ref().unwrap(), None, None)?;
    context.register("training.flatppl", &model.module)?;
    let parameters = "p = elementof(cartprod(temp_Intercept = reals, b = cartpow(reals, 1), temp_Intercept_sigma = reals, b_sigma = cartpow(reals, 1)))";
    let score = context.parse(
        &format!(
            r#"
        m = load_module("training.flatppl")
        {parameters}
        inputs = p
        outputs = record(
            posterior = logdensityof(m.Pi_post, p),
            pointwise = m.log_lik(temp_Intercept = p.temp_Intercept, b = p.b,
                temp_Intercept_sigma = p.temp_Intercept_sigma, b_sigma = p.b_sigma))
    "#
        ),
        None,
        None,
    )?;
    let training = score.compile(&EmitOptions::default())?;
    let data = serde_json::from_value(serde_json::json!({"nrows": 2, "columns": [
        {"name": "x", "values": {"kind": "real32", "values": oracle["data"]["x"]}},
        {"name": "g", "values": {"kind": "real64", "values": oracle["data"]["g"]}}
    ]}))?;
    let prediction = model.prepare(&mut context, &data, DataMode::Prediction)?;
    context.register("prediction.flatppl", &prediction)?;
    let forward = context.parse(
        &format!(
            r#"
        m = load_module("prediction.flatppl")
        {parameters}
        inputs = p
        outputs = logdensityof(m.K_model(p), record(y = [1.0, 2.0]))
    "#
        ),
        None,
        None,
    )?;
    forward.compile(&EmitOptions::default())?;
    assert_eq!(
        score.compile(&EmitOptions::default())?.stablehlo,
        training.stablehlo
    );
    Ok(())
}

fn compile_from_distinct_snapshots(
    spec: &flatppl_formula::Spec,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let model = context.import_formula(spec, spec.data.as_ref().unwrap(), None, None)?;
    context.register("training.flatppl", &model.module)?;
    let score = context.parse(r#"
        m = load_module("training.flatppl")
        p = elementof(cartprod(temp_Intercept = reals, b = cartpow(reals, 1), sd_g = posreals, z_g = cartpow(reals, 2), sigma = posreals))
        inputs = p
        outputs = logdensityof(m.Pi_post, p)
    "#, None, None)?;
    let export = score.compile(&EmitOptions::default())?;
    assert_eq!(export.inputs.len(), 1);
    let mut data = spec.data.as_ref().unwrap().clone();
    data.columns.retain(|c| c.name != "y");
    let prediction = model.prepare(&mut context, &data, DataMode::Prediction)?;
    context.register("prediction.flatppl", &prediction)?;
    let forward = context.parse(r#"
        m = load_module("prediction.flatppl")
        p = elementof(cartprod(temp_Intercept = reals, b = cartpow(reals, 1), sd_g = posreals, z_g = cartpow(reals, 2), sigma = posreals))
        inputs = p
        outputs = logdensityof(m.K_model(p), record(y = [2.0, 4.0, 8.0]))
    "#, None, None)?;
    forward.compile(&EmitOptions::default())?;
    // Reusing the unbound template must not consume or mutate the training data.
    assert_eq!(
        score.compile(&EmitOptions::default())?.stablehlo,
        export.stablehlo
    );
    Ok(())
}

#[test]
fn factor_prediction_reuses_training_contrasts_across_level_orders()
-> Result<(), Box<dyn std::error::Error>> {
    let oracle: serde_json::Value = serde_json::from_str(include_str!(
        "../../formula/tests/fixtures/factor-matrix.json"
    ))?;
    let case = oracle["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "prediction_interaction")
        .unwrap();
    let spec: flatppl_formula::Spec = serde_json::from_value(serde_json::json!({
        "formula": case["formula"],
        "options": {"contrasts": case["contrasts"]},
        "data": oracle["tables"][case["table"].as_str().unwrap()],
    }))?;
    let mut context = Context::default();
    let model = context.import_formula(&spec, spec.data.as_ref().unwrap(), None, None)?;
    let mut data: flatppl_formula::Table =
        serde_json::from_value(oracle["tables"]["prediction"].clone())?;
    data.columns.retain(|column| column.name != "y");
    let first = model.prepare(&mut context, &data, DataMode::Prediction)?;
    context.register("first.flatppl", &first)?;
    for column in &mut data.columns {
        if let flatppl_formula::ColumnValues::Factor { levels, codes, .. } = &mut column.values {
            levels.reverse();
            for code in codes {
                *code = levels.len() as u32 - 1 - *code;
            }
        }
    }
    let second = model.prepare(&mut context, &data, DataMode::Prediction)?;
    context.register("second.flatppl", &second)?;
    let coefficients = case["expected"]["x"]["column_names"]
        .as_array()
        .unwrap()
        .len()
        - 1;
    let query = |name: &str| {
        format!(
            r#"
        m = load_module("{name}.flatppl")
        p = elementof(cartprod(temp_Intercept = reals, b = cartpow(reals, {coefficients}), sigma = posreals))
        inputs = p
        outputs = logdensityof(m.K_model(p), record(y = [0.0, 0.0, 0.0]))
    "#
        )
    };
    let first = context
        .parse(&query("first"), None, None)?
        .compile(&EmitOptions::default())?;
    let second = context
        .parse(&query("second"), None, None)?
        .compile(&EmitOptions::default())?;
    assert_eq!(first.stablehlo, second.stablehlo);
    Ok(())
}

#[test]
fn arrow_import_and_prediction_keep_context_and_training_snapshot()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let model = context.import_formula_json(
        r#"{"formula":"y ~ x + f"}"#,
        Some(include_bytes!(
            "../../formula/tests/fixtures/arrow/training.arrow"
        )),
        Some("training"),
        None,
    )?;
    assert_eq!(model.info.training_rows, vec![0, 1, 3]);
    context.register("training.flatppl", &model.module)?;
    let score = context.parse(
        r#"
        m = load_module("training.flatppl")
        p = elementof(cartprod(temp_Intercept = reals, b = cartpow(reals, 2), sigma = posreals))
        inputs = p
        outputs = logdensityof(m.Pi_post, p)
    "#,
        None,
        None,
    )?;
    let training = score.compile(&EmitOptions::default())?;
    let bytes = include_bytes!("../../formula/tests/fixtures/arrow/prediction.arrows");
    let (prediction, rows) = model.prepare_arrow(&mut context, bytes, DataMode::Prediction)?;
    assert_eq!(rows, vec![0, 2]);
    context.register("prediction.flatppl", &prediction)?;
    let forward = context.parse(
        r#"
        m = load_module("prediction.flatppl")
        p = elementof(cartprod(temp_Intercept = reals, b = cartpow(reals, 2), sigma = posreals))
        inputs = p
        outputs = logdensityof(m.K_model(p), record(y = [2.0, 6.0]))
    "#,
        None,
        None,
    )?;
    forward.compile(&EmitOptions::default())?;
    assert_eq!(
        score.compile(&EmitOptions::default())?.stablehlo,
        training.stablehlo
    );
    assert!(
        model
            .prepare_arrow(&mut Context::default(), bytes, DataMode::Prediction)
            .is_err()
    );
    Ok(())
}

#[test]
fn scalar_families_compile_default_posteriors_and_fresh_predictions()
-> Result<(), Box<dyn std::error::Error>> {
    for family in [
        "gaussian",
        "student",
        "lognormal",
        "hurdle_lognormal",
        "bernoulli",
        "binomial",
        "zero_inflated_binomial",
        "poisson",
        "hurdle_poisson",
        "hurdle_negbinomial",
        "zero_inflated_poisson",
        "zero_inflated_negbinomial",
        "negbinomial",
        "geometric",
        "Gamma",
        "hurdle_gamma",
        "exponential",
        "weibull",
        "beta",
        "zero_inflated_beta",
        "von_mises",
    ] {
        let formula = if matches!(family, "binomial" | "zero_inflated_binomial") {
            "y | trials(n) ~ x"
        } else {
            "y ~ x"
        };
        let y = if family == "bernoulli" {
            vec![0.0, 1.0, 1.0, 0.0]
        } else if family == "beta" {
            vec![0.1, 0.4, 0.8, 0.7]
        } else if family == "zero_inflated_beta" {
            vec![0.0, 0.2, 0.8, 0.0]
        } else if family == "hurdle_gamma" {
            vec![0.0, 0.8, 2.3, 0.0]
        } else if family == "hurdle_lognormal" {
            vec![0.0, (-1.0_f64).exp(), 2.0_f64.exp(), 0.0]
        } else if family == "von_mises" {
            vec![-2.0, -0.4, 0.8, 2.0]
        } else if family.starts_with("zero_inflated_") || family.starts_with("hurdle_") {
            vec![0.0, 2.0, 4.0, 0.0]
        } else {
            vec![2.0, 4.0, 8.0, 7.0]
        };
        let spec = flatppl_formula::read(
            &serde_json::json!({
                "formula": formula, "family": family,
                "data": {"nrows": 4, "columns": [
                    {"name": "y", "values": {"kind": "real64", "values": y}},
                    {"name": "x", "values": {"kind": "real32", "values": [1, 2, 3, 4]}},
                    {"name": "n", "values": {"kind": "int", "values": [3, 5, 8, 9]}}
                ]}
            })
            .to_string(),
        )?;
        let mut context = Context::default();
        let model = context.import_formula(&spec, spec.data.as_ref().unwrap(), None, None)?;
        context.register("training.flatppl", &model.module)?;
        let auxiliaries = match family {
            "gaussian" | "lognormal" => ", sigma = posreals",
            "student" => ", sigma = posreals, nu = interval(1, inf)",
            "Gamma" | "negbinomial" | "weibull" => ", shape = posreals",
            "beta" => ", phi = posreals",
            "von_mises" => ", kappa = posreals",
            "zero_inflated_poisson" | "zero_inflated_binomial" => ", zi = unitinterval",
            "zero_inflated_negbinomial" => ", shape = posreals, zi = unitinterval",
            "zero_inflated_beta" => ", phi = posreals, zi = unitinterval",
            "hurdle_poisson" => ", hu = unitinterval",
            "hurdle_negbinomial" | "hurdle_gamma" => ", shape = posreals, hu = unitinterval",
            "hurdle_lognormal" => ", sigma = posreals, hu = unitinterval",
            _ => "",
        };
        let parameter = format!(
            "p = elementof(cartprod(temp_Intercept = reals, b = cartpow(reals, 1){auxiliaries}))"
        );
        let score = context.parse(
            &format!(
                r#"
            m = load_module("training.flatppl")
            {parameter}
            inputs = p
            outputs = logdensityof(m.Pi_post, p)
        "#
            ),
            None,
            None,
        )?;
        let training = score.compile(&EmitOptions::default())?;
        let data = serde_json::from_value(serde_json::json!({"nrows": 2, "columns": [
            {"name": "x", "values": {"kind": "real32", "values": [2, 5]}},
            {"name": "n", "values": {"kind": "int", "values": [3, 2]}}
        ]}))?;
        let prediction = model.prepare(&mut context, &data, DataMode::Prediction)?;
        context.register("prediction.flatppl", &prediction)?;
        let observation = if family == "beta" {
            "[0.2, 0.8]"
        } else if family == "zero_inflated_beta" {
            "[0.0, 0.8]"
        } else if family.starts_with("zero_inflated_") || family.starts_with("hurdle_") {
            "[0, 2]"
        } else {
            "[1, 1]"
        };
        let forward = context.parse(
            &format!(
                r#"
            m = load_module("prediction.flatppl")
            {parameter}
            inputs = p
            outputs = logdensityof(m.K_model(p), record(y = {observation}))
        "#
            ),
            None,
            None,
        )?;
        forward.compile(&EmitOptions::default())?;
        assert_eq!(
            training.stablehlo,
            score.compile(&EmitOptions::default())?.stablehlo,
            "{family}"
        );
    }
    Ok(())
}

#[test]
fn formula_inverse_links_compile_in_likelihood_queries() -> Result<(), Box<dyn std::error::Error>> {
    for link in [
        "identity",
        "log",
        "inverse",
        "sqrt",
        "softplus",
        "squareplus",
        "logit",
        "probit",
        "probit_approx",
        "cloglog",
        "cauchit",
        "softit",
        "tan_half",
    ] {
        let family = if link == "sqrt" {
            "poisson"
        } else if link == "tan_half" {
            "von_mises"
        } else {
            "gaussian"
        };
        let spec = flatppl_formula::read(&serde_json::json!({
            "formula": "y ~ 1", "family": family, "link": link,
            "data": {"nrows": 1, "columns": [{"name": "y", "values": {"kind": "int", "values": [1]}}]}
        }).to_string())?;
        let mut context = Context::default();
        let model = context.import_formula(&spec, spec.data.as_ref().unwrap(), None, None)?;
        context.register("model.flatppl", &model.module)?;
        let parameters = if family == "gaussian" {
            "record(temp_Intercept = eta, sigma = 1)"
        } else if family == "von_mises" {
            "record(temp_Intercept = eta, kappa = 1)"
        } else {
            "temp_Intercept = eta"
        };
        let query = context.parse(
            &format!(
                r#"
            m = load_module("model.flatppl")
            eta = elementof(reals)
            inputs = eta
            outputs = logdensityof(m.K_model({parameters}), record(y = [0]))
        "#
            ),
            None,
            None,
        )?;
        query.compile(&EmitOptions::default())?;
    }
    Ok(())
}

#[test]
fn weighted_training_retains_unweighted_diagnostics_and_prediction()
-> Result<(), Box<dyn std::error::Error>> {
    let spec = flatppl_formula::read(
        &serde_json::json!({
            "formula": "y | weights(w) ~ x",
            "priors": [
                {"class": "b", "prior": "normal(0, 2)"},
                {"class": "Intercept", "prior": "normal(0, 3)"},
                {"class": "sigma", "prior": "exponential(1.2)"}
            ],
            "data": {"nrows": 4, "columns": [
                {"name": "y", "values": {"kind": "int", "values": [1, 2, 4, 7]}},
                {"name": "x", "values": {"kind": "real32", "values": [-1, 0, 1, 2]}},
                {"name": "w", "values": {"kind": "real32", "values": [0, 0.5, 2, 3]}}
            ]}
        })
        .to_string(),
    )?;
    let mut context = Context::default();
    let model = context.import_formula(&spec, spec.data.as_ref().unwrap(), None, None)?;
    context.register("training.flatppl", &model.module)?;
    let score = context.parse(
        r#"
        m = load_module("training.flatppl")
        p = elementof(cartprod(temp_Intercept = reals, b = cartpow(reals, 1), sigma = posreals))
        inputs = p
        outputs = record(
            posterior = logdensityof(m.Pi_post, p),
            pointwise = m.log_lik(temp_Intercept = p.temp_Intercept, b = p.b, sigma = p.sigma),
            weights = m.weights_data,
            forward = logdensityof(m.K_model(p), record(y = m.y_data)))
    "#,
        None,
        None,
    )?;
    let training = score.compile(&EmitOptions::default())?;
    let data = serde_json::from_value(serde_json::json!({"nrows": 2, "columns": [
        {"name": "x", "values": {"kind": "real32", "values": [2, 5]}}
    ]}))?;
    let prediction = model.prepare(&mut context, &data, DataMode::Prediction)?;
    context.register("prediction.flatppl", &prediction)?;
    let forward = context.parse(
        r#"
        m = load_module("prediction.flatppl")
        p = elementof(cartprod(temp_Intercept = reals, b = cartpow(reals, 1), sigma = posreals))
        inputs = p
        outputs = logdensityof(m.K_model(p), record(y = [1, 1]))
    "#,
        None,
        None,
    )?;
    forward.compile(&EmitOptions::default())?;
    assert_eq!(
        score.compile(&EmitOptions::default())?.stablehlo,
        training.stablehlo
    );
    Ok(())
}
