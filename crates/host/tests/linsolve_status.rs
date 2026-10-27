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
