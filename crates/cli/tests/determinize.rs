use std::process::Command;

fn flatppl() -> Command {
    Command::new(env!("CARGO_BIN_EXE_flatppl"))
}

#[test]
fn determinize_lowers_a_gaussian_to_stdout() {
    let dir = std::env::temp_dir().join(format!("flatppl-det-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("g.flatppl");
    std::fs::write(
        &input,
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(lawof(record(a = a)), record(a = 0.5))\n",
    )
    .unwrap();
    let out = flatppl().arg("determinize").arg(&input).output().unwrap();
    assert!(
        out.status.success(),
        "exit 0; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("builtin_logdensityof"),
        "emitted FlatPDL:\n{stdout}"
    );
}

/// `determinize` resolves a `load_module` cross-module measure ref: the CLI
/// must assemble the `ModuleBundle` (same cache-only resolver as `infer`) and
/// pass it to `determinize_with`, or this refuses on an unresolved module ref
/// instead of lowering. Mirrors `crates/determinizer/tests/crossmodule_golden.rs`'s
/// `cross_module_likelihood_lowers`, but through the real CLI binary and
/// filesystem-resolved `load_module` (not an in-process `ModuleBundle`).
#[test]
fn determinize_resolves_cross_module_load_module() {
    let dir = std::env::temp_dir().join(format!("flatppl-det-cli-xmod-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("helpers.flatppl"),
        "flatppl_compat = \"0.1\"\n\
         center = elementof(reals)\n\
         obs_kernel = functionof(Normal(mu = center, sigma = 1.0), center = center)\n",
    )
    .unwrap();
    let input = dir.join("model.flatppl");
    std::fs::write(
        &input,
        "flatppl_compat = \"0.1\"\n\
         a = elementof(reals)\n\
         helpers = load_module(\"helpers.flatppl\", center = a)\n\
         input_data = 2.5\n\
         L = likelihoodof(helpers.obs_kernel, input_data)\n\
         lp = logdensityof(L, record(a = 0.0))\n",
    )
    .unwrap();
    let out = flatppl().arg("determinize").arg(&input).output().unwrap();
    assert!(
        out.status.success(),
        "exit 0; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("builtin_logdensityof"),
        "emitted FlatPDL did not resolve the cross-module kernel:\n{stdout}"
    );
}

#[test]
fn determinize_refuses_with_exit_3() {
    let dir = std::env::temp_dir().join(format!("flatppl-det-cli-refuse-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("r.flatppl");
    // A continuous-latent kchain marginal where the latent feeds the likelihood
    // SCALE (non-conjugate — no closed-form marginal) → refuse. (A latent feeding
    // the Normal mean is the Normal–Normal conjugate case, which now lowers.)
    std::fs::write(
        &input,
        "z = draw(Normal(mu = 0.0, sigma = 1.0))\nk = kernelof(record(y = draw(Normal(mu = 1.0, sigma = z))), z = z)\npp = kchain(lawof(record(z = z)), k)\nlp = logdensityof(pp, record(y = 0.5))\n",
    )
    .unwrap();
    let out = flatppl().arg("determinize").arg(&input).output().unwrap();
    assert_eq!(
        out.status.code(),
        Some(3),
        "refuse must exit 3; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("determinize: refuse"),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
