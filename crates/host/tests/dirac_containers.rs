use flatppl_host::{BatchSpec, Context, EmitOptions};

#[test]
fn dirac_event_reduction_preserves_host_batches() -> Result<(), Box<dyn std::error::Error>> {
    for (domain, axes) in [
        ("cartpow(reals, 2)", vec![vec![None], vec![Some(0)]]),
        (
            "cartprod(a = reals, b = cartpow(reals, 2))",
            vec![vec![None], vec![None], vec![Some(0)], vec![Some(0)]],
        ),
    ] {
        let query = Context::default().parse(
            &format!(
                "atom = external({domain})\npoint = external({domain})\n\
                 inputs = (atom, point)\noutputs = logdensityof(Dirac(atom), point)"
            ),
            None,
            None,
        )?;
        let output = query.compile_batched(
            &EmitOptions::default(),
            &BatchSpec {
                // Deliberately equal to the vector event extent.
                shape: vec![2],
                input_axes: axes,
            },
        )?;
        let ir = &output.stablehlo;
        assert!(ir.contains("-> tensor<2xf32> {"), "{ir}");
        assert!(
            ir.contains("applies stablehlo.and across dimensions = [1]"),
            "{ir}"
        );
        assert!(!ir.contains("stablehlo.while"), "{ir}");
    }
    Ok(())
}
