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

#[test]
fn determinize_refuses_with_exit_3() {
    let dir = std::env::temp_dir().join(format!("flatppl-det-cli-refuse-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("r.flatppl");
    // A continuous-latent kchain marginal is intractable → refuse.
    std::fs::write(
        &input,
        "z = draw(Normal(mu = 0.0, sigma = 1.0))\nk = kernelof(record(y = draw(Normal(mu = z, sigma = 1.0))), z = z)\npp = kchain(lawof(record(z = z)), k)\nlp = logdensityof(pp, record(y = 0.5))\n",
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
