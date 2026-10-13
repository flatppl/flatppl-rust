use flatppl_host::{BatchSpec, Context, EmitOptions};

#[test]
fn imported_domains_preserve_query_inputs_through_batching()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let model = context.parse(
        "theta ~ Beta(2, 3)\np = lawof(record(theta = theta))\nD = cartprod(theta = unitinterval)",
        None,
        None,
    )?;
    context.register("model.flatppl", &model)?;
    let query = |domain: &str| {
        format!(
            "m = load_module(\"model.flatppl\")\npoint = elementof({domain})\n\
             inputs = point\noutputs = logdensityof(m.p, point)"
        )
    };
    let inline = context.parse(&query("cartprod(theta = unitinterval)"), None, None)?;
    let imported = context.parse(&query("m.D"), None, None)?;
    let options = EmitOptions::default();
    let batch = BatchSpec {
        shape: vec![3],
        input_axes: vec![vec![Some(0)]],
    };
    assert_eq!(
        serde_json::to_value(imported.compile(&options)?)?,
        serde_json::to_value(inline.compile(&options)?)?
    );
    assert_eq!(
        serde_json::to_value(imported.compile_batched(&options, &batch)?)?,
        serde_json::to_value(inline.compile_batched(&options, &batch)?)?
    );
    Ok(())
}
