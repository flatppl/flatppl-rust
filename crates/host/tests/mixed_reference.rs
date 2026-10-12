use flatppl_host::{Context, EmitOptions};

#[test]
fn mixed_density_requires_fixed_atoms_after_callback_substitution()
-> Result<(), Box<dyn std::error::Error>> {
    let prefix = "alpha = external(posreals)\nbeta = external(posreals)\n\
        x = external(reals)\natom = 0.0\n\
        K(a, b, z) = superpose(weighted(0.3, Dirac(z)), weighted(0.7, Beta(a, b)))\n\
        outputs = logdensityof(K(alpha, beta, atom), x)\n";
    let mut context = Context::default();
    let fixed = context.parse(&format!("{prefix}inputs = (alpha, beta, x)"), None, None)?;
    fixed.compile(&EmitOptions::default())?;
    let moving = context.parse(
        &format!("{prefix}inputs = (alpha, beta, x, atom)"),
        None,
        None,
    )?;
    assert!(moving.compile(&EmitOptions::default()).is_err());
    let named = context.parse(
        "alpha = external(posreals)\nbeta = external(posreals)\n\
        x = external(reals)\nz = elementof(reals)\n\
        M = superpose(weighted(0.3, Dirac(z)), weighted(0.7, Beta(alpha, beta)))\n\
        K = functionof(M, z = z)\n\
        inputs = (alpha, beta, x)\noutputs = logdensityof(K(0.0), x)",
        None,
        None,
    )?;
    named.compile(&EmitOptions::default())?;
    Ok(())
}
