use flatppl_host::{Context, EmitOptions};

/// The host lists every status code a program can report, so an adapter can
/// handle a parameter outside its value set (status 9) apart from other checks.
#[test]
fn a_program_lists_the_codes_its_status_can_report() -> Result<(), Box<dyn std::error::Error>> {
    let source = "sigma = elementof(reals)\nk = elementof(integers)\nw = elementof(cartpow(reals, 3))\n\
                  inputs = (sigma, k, w)\noutputs = logdensityof(Normal(0.0, sigma), get(w, k))\n";
    let query = Context::default().parse(source, None, None)?;
    let status = query
        .compile(&EmitOptions::default())?
        .runtime_status
        .expect("runtime status");
    assert_eq!(status.codes, [1, 9]);
    let source = "sigma = elementof(reals)\nx = elementof(reals)\n\
                  inputs = (sigma, x)\noutputs = logdensityof(Normal(0.0, sigma), x)\n";
    let query = Context::default().parse(source, None, None)?;
    let status = query
        .compile(&EmitOptions::default())?
        .runtime_status
        .expect("runtime status");
    assert_eq!(status.codes, [9]);
    Ok(())
}
