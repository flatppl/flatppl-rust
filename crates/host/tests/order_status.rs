use flatppl_host::{BatchSpec, Context, EmitOptions};

#[test]
fn quantile_status_is_separate_per_host_call() -> Result<(), Box<dyn std::error::Error>> {
    let query = Context::default().parse("v = elementof(cartpow(reals, [4]))\np = elementof(reals)\ninputs = (v,p)\noutputs = quantile(v,p)\n", None, None)?;
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
            input_axes: vec![vec![None], vec![Some(0)]],
        },
    )?;
    assert_eq!(batched.runtime_status.as_ref().unwrap().shape, [3]);
    for output in [
        "quantile(v,0.25)",
        "quantile(v,0)",
        "quantile(v,1)",
        "quantile(v,0.0)",
        "quantile(v,1.0)",
        "median(v)",
    ] {
        let source =
            format!("v = elementof(cartpow(reals, [4]))\ninputs = v\noutputs = {output}\n");
        let query = Context::default().parse(&source, None, None)?;
        assert!(
            query.compile(&options)?.runtime_status.is_none(),
            "{output} unexpectedly adds runtime status"
        );
    }
    for output in ["quantile.(v,0.25)", "median.(v)"] {
        let source = format!(
            "v = elementof(cartpow(cartpow(reals, [4]), [2]))\ninputs = v\noutputs = {output}\n"
        );
        let query = Context::default().parse(&source, None, None)?;
        assert!(
            query.compile(&options)?.runtime_status.is_none(),
            "{output}"
        );
        let batch = BatchSpec {
            shape: vec![3],
            input_axes: vec![vec![Some(0)]],
        };
        assert!(
            query
                .compile_batched(&options, &batch)?
                .runtime_status
                .is_none(),
            "mapped {output}"
        );
    }
    Ok(())
}
