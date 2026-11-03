//! §07 `bincounts`: "Each edge vector must be strictly increasing". Written and
//! constant edges are checked statically, computed edges on the device (status 8).

use flatppl_host::{Context, EmitOptions, Export};

fn compile(source: &str) -> Result<Export, Box<dyn std::error::Error>> {
    Ok(Context::default()
        .parse(source, None, None)?
        .compile(&EmitOptions::default())?)
}

#[test]
fn input_edges_get_a_runtime_status() -> Result<(), Box<dyn std::error::Error>> {
    let compiled = compile(
        "e = elementof(cartpow(reals, [3]))\nx = elementof(cartpow(reals, [4]))\n\
         inputs = (e, x)\noutputs = sum(bincounts(e, x))\n",
    )?;
    assert!(
        compiled.runtime_status.is_some(),
        "no status for input edges"
    );
    assert!(
        compiled.stablehlo.contains("dense<8>"),
        "{}",
        compiled.stablehlo
    );
    Ok(())
}

#[test]
fn increasing_fixed_edges_need_no_status() -> Result<(), Box<dyn std::error::Error>> {
    for edges in [
        "[0.0, 1.0, inf]",
        "linspace(0.0, 1.0, 3)",
        "extlinspace(0.0, 1.0, 3)",
    ] {
        let compiled = compile(&format!(
            "x = elementof(cartpow(reals, [4]))\ninputs = x\noutputs = sum(bincounts({edges}, x))\n"
        ))?;
        assert!(compiled.runtime_status.is_none(), "{edges}");
    }
    Ok(())
}

#[test]
fn non_increasing_fixed_edges_are_refused() {
    let error = compile(
        "x = elementof(cartpow(reals, [4]))\ninputs = x\noutputs = sum(bincounts([0.0, 2.0, 1.0], x))\n",
    )
    .err()
    .expect("decreasing edges must not compile");
    assert!(error.to_string().contains("strictly increasing"), "{error}");
}
