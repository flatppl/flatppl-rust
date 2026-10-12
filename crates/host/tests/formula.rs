use flatppl_host::{
    Context, EmitOptions,
    flatppl_formula::{self, DataMode},
};

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
        "bernoulli",
        "binomial",
        "poisson",
        "negbinomial",
        "geometric",
        "Gamma",
        "exponential",
    ] {
        let formula = if family == "binomial" {
            "y | trials(n) ~ x"
        } else {
            "y ~ x"
        };
        let y = if family == "bernoulli" {
            vec![0, 1, 1, 0]
        } else {
            vec![2, 4, 8, 7]
        };
        let spec = flatppl_formula::read(
            &serde_json::json!({
                "formula": formula, "family": family,
                "data": {"nrows": 4, "columns": [
                    {"name": "y", "values": {"kind": "int", "values": y}},
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
            "Gamma" | "negbinomial" => ", shape = posreals",
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
        let forward = context.parse(
            &format!(
                r#"
            m = load_module("prediction.flatppl")
            {parameter}
            inputs = p
            outputs = logdensityof(m.K_model(p), record(y = [1, 1]))
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
    ] {
        let family = if link == "sqrt" {
            "poisson"
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
        } else {
            "record(temp_Intercept = eta)"
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
