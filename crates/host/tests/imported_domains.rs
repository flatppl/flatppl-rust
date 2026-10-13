use flatppl_host::{BatchSpec, Context, EmitOptions};

#[test]
fn imported_domains_preserve_query_inputs_through_transforms_and_batching()
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
    let inline = context.transform(&inline, true)?;
    let imported = context.transform(&imported, true)?;
    assert_eq!(
        serde_json::to_value(imported.unconstrained.compile(&options)?)?,
        serde_json::to_value(inline.unconstrained.compile(&options)?)?
    );
    assert_eq!(
        serde_json::to_value(imported.unconstrained.compile_batched(&options, &batch)?)?,
        serde_json::to_value(inline.unconstrained.compile_batched(&options, &batch)?)?
    );
    Ok(())
}

#[test]
fn reexported_domains_preserve_matrix_and_nested_vector_operations()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let sets = context.parse(
        "matrix = cartpow(reals, [2, 3])\nnested = cartpow(cartpow(reals, 3), 2)\nscalar = posreals",
        None,
        None,
    )?;
    context.register("sets.flatppl", &sets)?;
    let aliases = context.parse(
        "s = load_module(\"sets.flatppl\")\nmatrix = s.matrix\nnested = s.nested\nscalar = s.scalar",
        None,
        None,
    )?;
    context.register("aliases.flatppl", &aliases)?;
    for (name, set, body) in [
        ("matrix", "cartpow(reals, [2, 3])", "point * [1, 2, 3]"),
        ("nested", "cartpow(cartpow(reals, 3), 2)", "sum.(point)"),
        ("scalar", "posreals", "log(point)"),
    ] {
        let query = |domain: &str| {
            format!(
                "m = load_module(\"aliases.flatppl\")\npoint = elementof({domain})\n\
                 inputs = point\noutputs = {body}"
            )
        };
        let inline = context.parse(&query(set), None, None)?;
        let imported = context.parse(&query(&format!("m.{name}")), None, None)?;
        let point_type = |module: &flatppl_host::LoadedModule| {
            module
                .bindings()
                .into_iter()
                .find(|b| b.name == "point")
                .unwrap()
                .value_type
        };
        assert_eq!(point_type(&imported), point_type(&inline));
        assert_eq!(
            serde_json::to_value(imported.compile(&EmitOptions::default())?)?,
            serde_json::to_value(inline.compile(&EmitOptions::default())?)?
        );
        if name != "nested" {
            let inline = context.transform(&inline, false)?;
            let imported = context.transform(&imported, false)?;
            assert_eq!(
                serde_json::to_value(imported.unconstrained.compile(&EmitOptions::default())?)?,
                serde_json::to_value(inline.unconstrained.compile(&EmitOptions::default())?)?
            );
        }
    }
    Ok(())
}
