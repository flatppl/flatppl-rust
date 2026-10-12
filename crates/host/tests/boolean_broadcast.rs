use flatppl_host::{BatchSpec, Context, EmitOptions};
use serde_json::json;

#[test]
fn boolean_broadcasts_expand_singleton_axes_and_host_batches()
-> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        r#"
        a = external(cartpow(booleans, 3))
        b = external(cartpow(booleans, 2))
        x = addaxes(a, 0, 1)
        y = addaxes(b, 1, 0)
        inputs = (a, b)
        outputs = (land.(x, y), lor.(x, y), lxor.(x, y))
        "#,
        None,
        None,
    )?;
    let exported = query.compile_batched(
        &EmitOptions::default(),
        &BatchSpec {
            shape: vec![4],
            input_axes: vec![vec![Some(0)], vec![None]],
        },
    )?;
    let schema = serde_json::to_value(&exported)?;
    for output in schema["output"]["items"].as_array().unwrap() {
        assert_eq!(output["shape"], json!([4, 3, 2]));
        assert_eq!(output["dtype"], "bool");
    }
    Ok(())
}
