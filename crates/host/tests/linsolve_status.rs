use flatppl_host::{BatchSpec, Context, EmitOptions};

#[test]
fn linear_solve_status_is_separate_per_host_call() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse("A = elementof(cartpow(reals, [2,2]))\nb = elementof(cartpow(reals, [2]))\ninputs = (A,b)\noutputs = linsolve(A,b)\n", None, None)?;
    let options = EmitOptions::default();
    let plain = query.compile(&options)?;
    let status = plain.runtime_status.as_ref().expect("runtime status");
    assert_eq!(status.index, 1);
    assert_eq!(status.dtype, "int32");
    assert!(status.shape.is_empty());
    assert_eq!(status.codes, [4]);
    let batched = query.compile_batched(
        &options,
        &BatchSpec {
            shape: vec![3],
            input_axes: vec![vec![Some(0)], vec![Some(0)]],
        },
    )?;
    assert_eq!(batched.runtime_status.as_ref().unwrap().shape, [3]);
    Ok(())
}

#[test]
fn inverse_solve_boundary_keeps_the_matrix_only_output() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse(
        "A = elementof(cartpow(reals, [8,8]))\ninputs = A\noutputs = inv(A)\n",
        None,
        None,
    )?;
    let export = query.compile(&EmitOptions::default())?;
    let output = serde_json::to_value(&export.output)?;
    assert_eq!(output["index"], 0);
    assert_eq!(output["shape"], serde_json::json!([8, 8]));
    assert!(export.runtime_status.is_none());
    Ok(())
}
