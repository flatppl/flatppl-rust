use flatppl_host::{BatchSpec, Context, EmitOptions};

#[test]
fn inverse_uniform_reports_status_for_each_host_call() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse("u = elementof(reals)\ninputs = u\noutputs = builtin_fromuniform(Exponential, record(rate = 2), u)\n", None, None)?;
    let options = EmitOptions::default();
    let ordinary = query.compile(&options)?;
    let status = ordinary.runtime_status.as_ref().expect("runtime status");
    assert_eq!(status.index, 1);
    assert_eq!(status.dtype, "int32");
    assert!(status.shape.is_empty());
    assert!(ordinary.stablehlo.contains("dense<5>"));
    let mapped = query.compile_batched(
        &options,
        &BatchSpec {
            shape: vec![3],
            input_axes: vec![vec![Some(0)]],
        },
    )?;
    assert_eq!(mapped.runtime_status.as_ref().unwrap().shape, [3]);
    Ok(())
}
