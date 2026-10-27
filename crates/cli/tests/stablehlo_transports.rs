use std::{fs, process::Command};

mod common;
use common::Scratch;

#[test]
fn closed_form_transports_emit_and_unsupported_refuse() {
    let dir = Scratch::new("transports");
    let input = dir.path("model.flatppl");
    for call in [
        "builtin_tonormal(Normal, record(mu = 2, sigma = 3), x)",
        "builtin_fromnormal(Normal, record(mu = 2, sigma = 3), x)",
        "builtin_fromuniform(Cauchy, record(location = 2, scale = 3), x)",
        "builtin_fromuniform(Exponential, record(rate = 2), x)",
        "builtin_fromuniform(Logistic, record(mu = 2, s = 3), x)",
    ] {
        fs::write(
            &input,
            format!("x = elementof(reals)\ninputs = x\noutputs = {call}\n"),
        )
        .unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
            .arg("stablehlo")
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{call}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("tensor<f32>"));
    }
    for call in [
        "builtin_fromuniform(Normal, record(mu = 0, sigma = 1), 0.5)",
        "builtin_fromnormal(Bernoulli, record(p = 0.5), 0.0)",
    ] {
        fs::write(&input, format!("outputs = {call}\n")).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
            .arg("stablehlo")
            .arg(&input)
            .output()
            .unwrap();
        assert_eq!(
            result.status.code(),
            Some(3),
            "{call}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stderr).contains("unavailable or undefined"));
    }
}

#[test]
fn inverse_uniform_rejects_invalid_literal() {
    let dir = Scratch::new("transport-domain");
    let input = dir.path("model.flatppl");
    fs::write(
        &input,
        "outputs = builtin_fromuniform(Exponential, record(rate = 2), 2)\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("stablehlo")
        .arg(&input)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&result.stderr).contains("uniform variate must be in [0, 1]"));
}
