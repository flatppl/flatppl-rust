use flatppl_host::{Context, EmitOptions, IntegrationOptions};

#[test]
fn weighted_normalizer_keeps_live_support_breakpoints() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        r#"
        scale = 2.0
        point = 3.0
        hi = 10.0
        inputs = (scale, point, hi)
        outputs = logdensityof(normalize(truncate(
            weighted(x -> 1.0/x, Pareto(3.0, scale)), interval(0.0, hi))), point)
        "#,
        None,
        None,
    )?;
    query.compile(&EmitOptions {
        integration: Some(IntegrationOptions::default()),
        ..EmitOptions::default()
    })?;
    Ok(())
}
