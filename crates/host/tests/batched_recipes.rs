//! Host batches replay typed recipes instead of looping a scalar cell per lane.

use flatppl_host::{BatchSpec, Context, EmitOptions, Target};

fn batched(source: &str, options: &EmitOptions) -> Result<String, Box<dyn std::error::Error>> {
    let query = Context::default().parse(source, None, None)?;
    let leaves = query.compile(options)?.inputs.len();
    let exported = query.compile_batched(
        options,
        &BatchSpec {
            shape: vec![8],
            input_axes: vec![vec![Some(0)]; leaves],
        },
    )?;
    Ok(exported.stablehlo)
}

fn assert_tensorized(module: &str, op: &str) {
    assert!(module.contains(op), "{op} missing:\n{module}");
    assert!(!module.contains("_cell("), "lane loop:\n{module}");
    assert!(!module.contains("stablehlo.while"), "lane loop:\n{module}");
}

#[test]
fn poisson_exponent_split_bitcasts_every_lane() -> Result<(), Box<dyn std::error::Error>> {
    let module = batched(
        "rate = elementof(posreals)\ninputs = (rate)\noutputs = logdensityof(Poisson(rate), 100)\n",
        &EmitOptions::default(),
    )?;
    assert_tensorized(&module, "stablehlo.bitcast_convert");
    Ok(())
}

#[test]
fn diagonal_iotas_are_lane_invariant() -> Result<(), Box<dyn std::error::Error>> {
    // The Enzyme-restricted GPU profile unrolls the 2x2 Cholesky factor.
    let options = EmitOptions {
        restrict_enzyme_compatible: true,
        target: Target::Gpu,
        ..EmitOptions::default()
    };
    let module = batched(
        "mu = elementof(cartpow(reals, 2))\ncov = elementof(cartpow(reals, [2, 2]))\na = draw(MvNormal(mu = mu, cov = cov))\ninputs = (mu, cov)\noutputs = logdensityof(lawof(record(a = a)), record(a = [0.2, 0.1]))\n",
        &options,
    )?;
    assert_tensorized(&module, "stablehlo.iota");
    Ok(())
}

#[test]
fn metric_contractions_map_every_lane() -> Result<(), Box<dyn std::error::Error>> {
    let module = batched(
        "m = elementof(cartpow(reals, 4))\nv = elementof(cartpow(reals, 2))\ng = rowstack([[m[1], m[2]], [m[3], m[4]]])\ng: norm[] := v[.i^]*v[.i_]\ninputs = (m, v)\noutputs = norm\n",
        &EmitOptions::default(),
    )?;
    assert_tensorized(&module, "stablehlo.dot_general");
    Ok(())
}

#[test]
fn scalar_predicates_select_every_lane() -> Result<(), Box<dyn std::error::Error>> {
    let module = batched(
        "b = elementof(cartpow(reals, 2))\nA = rowstack([[3.0, 1.0], [2.0, 2.0]])\nx = linsolve(A, b)\ninputs = (b)\noutputs = x[1] + 3.0 * x[2]\n",
        &EmitOptions::default(),
    )?;
    assert_tensorized(&module, "stablehlo.select");
    Ok(())
}

#[test]
fn native_factorizations_batch_every_lane() -> Result<(), Box<dyn std::error::Error>> {
    let source = "mu = elementof(cartpow(reals, 3))\ncov = elementof(cartpow(reals, [3, 3]))\na = draw(MvNormal(mu = mu, cov = cov))\ninputs = (mu, cov)\noutputs = logdensityof(lawof(record(a = a)), record(a = [0.2, 0.1, -0.3]))\n";
    // The Enzyme-restricted GPU profile unrolls matrices up to 8x8.
    for (target, restrict_enzyme_compatible) in [(Target::Cpu, true), (Target::Gpu, false)] {
        let options = EmitOptions {
            target,
            restrict_enzyme_compatible,
            ..EmitOptions::default()
        };
        let module = batched(source, &options)?;
        assert_tensorized(&module, "stablehlo.cholesky");
        assert_tensorized(&module, "\"stablehlo.triangular_solve\"");
    }
    Ok(())
}
