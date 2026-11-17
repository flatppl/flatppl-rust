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

#[test]
fn arbitrary_knot_splines_keep_shapes_defaults_and_runtime_domains()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let query = context.parse(
        r#"
        sp = standard_module("splines", "0.1")
        x = external(reals)
        knots = external(cartpow(reals, 8))
        values = external(cartpow(reals, 8))
        inputs = (x, knots, values)
        outputs = record(
            linear = sp.interp_linear(x, knots, values),
            cubic = sp.interp_cubic(x, knots, values, "linear"),
            pchip = sp.interp_pchip(values = values, knots = knots, x = x, extrapolation = "error"),
            basis = sp.bspline_basis(x, knots, 2),
            derivative = sp.bspline_derivative(x, knots, 2),
            integral = sp.bspline_integral(0.0, x, knots, 2),
            penalty = sp.bspline_penalty(knots, 2))
        "#,
        None,
        None,
    )?;
    let exported = query.compile(&EmitOptions::default())?;
    let schema = serde_json::to_value(&exported)?;
    let fields = &schema["output"]["fields"];
    assert_eq!(fields[3]["value"]["shape"], json!([5]));
    assert_eq!(fields[6]["value"]["shape"], json!([5, 5]));
    assert_eq!(schema["runtime_status"]["codes"], json!([12]));
    query.compile_batched(
        &EmitOptions::default(),
        &BatchSpec {
            shape: vec![3],
            input_axes: vec![vec![Some(0)], vec![None], vec![None]],
        },
    )?;
    Ok(())
}

#[test]
fn spline_broadcast_callbacks_and_zero_degree_keep_cell_dimensions()
-> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::default();
    let query = context.parse(
        r#"
        sp = standard_module("splines", "0.1")
        xs = external(cartpow(reals, 3))
        knots = external(cartpow(reals, 4))
        values = external(cartpow(reals, 4))
        z = elementof(reals)
        f = functionof(sp.interp_cubic(z, knots, values), z=z)
        inputs = (xs, knots, values)
        outputs = record(interp = broadcast(f, xs), basis = broadcast(sp.bspline_basis, xs, [knots], 0),
            derivative = sp.bspline_derivative(xs[1], knots, 0, 4))
    "#,
        None,
        None,
    )?;
    let exported = query.compile(&EmitOptions::default())?;
    let schema = serde_json::to_value(&exported)?;
    assert_eq!(
        schema["output"]["fields"][1]["value"]["shape"],
        json!([3, 3])
    );
    Ok(())
}
