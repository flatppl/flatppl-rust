use flatppl_host::{Context, EmitOptions};
use serde_json::json;

#[test]
fn registered_module_exports_a_typed_query_abi() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let model = context.parse("distribution = Normal(3.0, 1.0)", None, None)?;
    context.register("model.flatppl", &model)?;
    let query = context.parse(
        r#"
        m = load_module("model.flatppl")
        distribution = m.distribution
        p = elementof(cartprod(z = reals, a = cartpow(reals, 2)))
        inputs = p
        result = record(total = p.z + logdensityof(iid(distribution, 2), p.a), original = p.a)
        outputs = result
        "#,
        None,
        None,
    )?;
    // The loaded module owns its source and resolved imports.
    drop(context);
    drop(model);
    let exported = query.compile(&EmitOptions::default())?;
    let schema = serde_json::to_value(&exported)?;
    assert_eq!(
        schema["inputs"],
        json!([{
            "name": "p",
            "value": {
                "kind": "record",
                "fields": [
                    {"name": "z", "value": {
                        "kind": "tensor", "index": 0, "dtype": "float32",
                        "shape": [], "value_type": "real"
                    }},
                    {"name": "a", "value": {
                        "kind": "tensor", "index": 1, "dtype": "float32",
                        "shape": [2], "value_type": "real[2]"
                    }}
                ]
            }
        }])
    );
    assert_eq!(
        schema["output"],
        json!({
            "kind": "record",
            "fields": [
                {"name": "total", "value": {
                    "kind": "tensor", "index": 0, "dtype": "float32",
                    "shape": [], "value_type": "real"
                }},
                {"name": "original", "value": {
                    "kind": "tensor", "index": 1, "dtype": "float32",
                    "shape": [2], "value_type": "real[2]"
                }}
            ]
        })
    );
    assert_eq!(exported.output_names, [Some("result".into())]);
    assert!(!exported.multiple_outputs);
    let signature = exported.stablehlo.lines().nth(1).unwrap().trim();
    assert_eq!(
        signature,
        "func.func @main(%arg0: tensor<f32>, %arg1: tensor<2xf32>) -> (tensor<f32>, tensor<2xf32>) {"
    );
    Ok(())
}
