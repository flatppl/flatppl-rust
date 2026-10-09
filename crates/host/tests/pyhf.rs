use flatppl_host::{Context, EmitOptions};

#[test]
fn pyhf_observation_batches_preserve_input_order_and_channel_scope()
-> Result<(), Box<dyn std::error::Error>> {
    let counts = [3.5, 4.5, 5.0, 6.0, 7.5];
    let parameters = ["z", "a", "y", "b", "x"];
    let channels: Vec<_> = counts
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let mut modifiers = vec![serde_json::json!({
                "name": parameters[i], "type": "normfactor", "data": null
            })];
            for name in match i {
                0 => &["alpha", "beta"][..],
                2 => &["alpha"][..],
                _ => &[],
            } {
                modifiers.push(serde_json::json!({
                    "name": name, "type": "normsys", "data": {"lo": 0.9, "hi": 1.1}
                }));
            }
            serde_json::json!({
                "name": format!("ch{i}"),
                "samples": [{"name": "s", "data": [i as f64 + 3.0], "modifiers": modifiers}]
            })
        })
        .collect();
    let observations: Vec<_> = counts
        .iter()
        .enumerate()
        .map(|(i, count)| serde_json::json!({"name": format!("ch{i}"), "data": [count]}))
        .collect();
    let source = serde_json::json!({
        "channels": channels,
        "observations": observations,
        "measurements": [{"name": "m", "config": {"poi": "z"}}],
        "version": "1.0.0"
    })
    .to_string();
    let mut context = Context::default();
    let model = context.import_pyhf(&source, None, None)?;
    let likelihood = model
        .bindings()
        .into_iter()
        .find(|b| b.name == "likelihood")
        .unwrap();
    assert_eq!(
        likelihood.value_type.as_deref(),
        Some("likelihood(alpha, beta, z, a, y, b, x) over real[7]")
    );
    context.register("model.flatppl", &model)?;
    let query = context.parse(
        r#"
        m = load_module("model.flatppl")
        alpha = elementof(reals)
        inputs = alpha
        channel = logdensityof(m.ch0_likelihood, record(alpha = alpha, beta = 0.0, z = 1.0))
        full = logdensityof(m.likelihood, record(alpha = alpha, beta = 0.0, z = 1.0, a = 1.0, y = 1.0, b = 1.0, x = 1.0))
        outputs = [channel, full]
        "#,
        None,
        None,
    )?;
    let export = query.compile(&EmitOptions::default())?;
    assert_eq!(export.inputs.len(), 1);
    Ok(())
}

#[test]
fn pyhf_import_composes_with_registered_queries() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let model = context.import_pyhf(
        include_str!("../../hs3/tests/fixtures/2bin_1channel.json"),
        Some("workspace.json"),
        None,
    )?;
    context.register("model.flatppl", &model)?;
    let query = context.parse(
        r#"
        m = load_module("model.flatppl")
        p = elementof(cartprod(mu = reals, uncorr_bkguncrt = cartpow(posreals, 2)))
        inputs = p
        outputs = logdensityof(m.likelihood, p)
        "#,
        None,
        None,
    )?;
    drop(context);
    let exported = query.compile(&EmitOptions::default())?;
    let schema = serde_json::to_value(exported)?;
    assert_eq!(schema["inputs"][0]["value"]["fields"][0]["name"], "mu");
    assert_eq!(
        schema["inputs"][0]["value"]["fields"][1]["value"]["shape"],
        serde_json::json!([2])
    );
    assert_eq!(schema["output"]["dtype"], "float32");
    assert_eq!(schema["output"]["shape"], serde_json::json!([]));
    Ok(())
}
