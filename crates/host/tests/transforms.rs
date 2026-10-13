use flatppl_host::{Context, EmitOptions};
use serde_json::json;

#[test]
fn coordinate_queries_retain_imports_and_change_only_free_dimensions()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let model = context.parse("scale = 2.0", None, None)?;
    context.register("model.flatppl", &model)?;
    let query = context.parse(
        r#"
        model = load_module("model.flatppl")
        p = elementof(cartprod(cuts = ordered(3), effects = zerosum(3),
                              weights = stdsimplex(4)))
        inputs = p
        outputs = -model.scale * (sum(p.cuts .* p.cuts) + sum(p.effects .* p.effects)
                                  + sum(p.weights .* p.weights))
        "#,
        None,
        None,
    )?;
    let options = EmitOptions::default();
    let original = serde_json::to_value(query.compile(&options)?)?;
    let transformed = context.transform(&query, true)?;
    drop(context);
    let free = serde_json::to_value(transformed.unconstrained.compile(&options)?)?;
    let forward = serde_json::to_value(transformed.constrain.compile(&options)?)?;
    let inverse = serde_json::to_value(transformed.unconstrain.compile(&options)?)?;
    assert_eq!(original["output"], free["output"]);
    let shapes = |schema: &serde_json::Value| {
        schema["fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field["value"]["shape"].clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        shapes(&free["inputs"][0]["value"]),
        [json!([3]), json!([2]), json!([3])]
    );
    assert_eq!(
        shapes(&forward["output"]),
        [json!([3]), json!([3]), json!([4])]
    );
    assert_eq!(forward["output"], inverse["inputs"][0]["value"]);
    assert_eq!(inverse["output"], free["inputs"][0]["value"]);
    assert_eq!(serde_json::to_value(query.compile(&options)?)?, original);
    Ok(())
}

#[test]
fn array_nesting_is_not_silently_replaced_with_matrix_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let nested = context.parse(
        "p = elementof(cartpow(cartpow(reals, 3), 2))\ninputs = p\noutputs = p",
        None,
        None,
    )?;
    let matrix = context.parse(
        "p = elementof(cartpow(reals, [2, 3]))\ninputs = p\noutputs = p",
        None,
        None,
    )?;
    // Both use tensor<2x3x...>, but reductions and broadcasts differ by type.
    nested.compile(&EmitOptions::default())?;
    assert!(context.transform(&nested, false).is_err());
    context
        .transform(&matrix, false)?
        .constrain
        .compile(&EmitOptions::default())?;
    Ok(())
}
