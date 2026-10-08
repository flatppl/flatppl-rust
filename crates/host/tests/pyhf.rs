use flatppl_host::{Context, EmitOptions};

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
