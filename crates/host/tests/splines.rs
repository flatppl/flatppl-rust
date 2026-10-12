use flatppl_host::{BatchSpec, Context, EmitOptions};
use serde_json::json;

#[test]
fn thinplate_arrays_export_shapes_and_tensorized_batches() -> Result<(), Box<dyn std::error::Error>>
{
    let mut context = Context::default();
    let query = context.parse(
        r#"
        sp = standard_module("splines", "0.1")
        x = external(reals)
        centers = external(cartpow(reals, 3))
        inputs = (x, centers)
        outputs = record(basis = sp.thinplate_basis(centers = centers, x = x), penalty = sp.thinplate_penalty(centers))
        "#,
        None,
        None,
    )?;
    let exported = query.compile(&EmitOptions::default())?;
    let schema = serde_json::to_value(&exported)?;
    assert_eq!(schema["output"]["fields"][0]["value"]["shape"], json!([5]));
    assert_eq!(
        schema["output"]["fields"][1]["value"]["shape"],
        json!([5, 5])
    );
    query.compile_batched(
        &EmitOptions::default(),
        &BatchSpec {
            shape: vec![4],
            input_axes: vec![vec![Some(0)], vec![None]],
        },
    )?;
    Ok(())
}
