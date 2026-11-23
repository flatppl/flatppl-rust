#![cfg(feature = "stablehlo")]

mod common;

use std::{fs, process::Command};

fn emit(label: &str, source: &str, dtype: &str) -> String {
    let scratch = common::Scratch::new(label);
    let file = scratch.path("query.flatppl");
    fs::write(&file, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args(["stablehlo", "--dtype", dtype])
        .arg(file)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn closed_gaussian_transports_and_scores_emit_the_analytic_results() {
    let original = "\
x = external(reals)
inputs = x
outputs = (
    builtin_tonormal(Normal, record(mu = 0.0, sigma = 1.0), x),
    builtin_tonormal(Normal, record(mu = 0.5, sigma = 2.0), 3.5),
    builtin_fromnormal(Normal, record(mu = 0.5, sigma = 2.0), -1.5),
    logdensityof(Normal(0.0, 1.0), 2.0),
    logdensityof(Normal(0.5, 2.0), inf),
    logdensityof(Normal(0.5, 2.0), -inf),
)
";
    for dtype in ["f32", "f64"] {
        let hlo = emit("gaussian-closed", original, dtype);
        for head in ["divide", "multiply", "log", "subtract"] {
            assert!(!hlo.contains(&format!("stablehlo.{head}")), "{hlo}");
        }
        assert!(hlo.contains("return %arg0,"), "{hlo}");
        assert!(hlo.contains("dense<1.5>"), "{hlo}");
    }
}

#[test]
fn gaussian_transports_keep_signed_zero_and_runtime_scale_checks() {
    let output = emit(
        "gaussian-fallback",
        "\
x = external(reals)
mu = 0.0
sigma = 1.0
inputs = (x, mu, sigma)
outputs = (
    builtin_fromnormal(Normal, record(mu = 0.0, sigma = 1.0), x),
    builtin_tonormal(Normal, record(mu = mu, sigma = 1.0), x),
    builtin_tonormal(Normal, record(mu = 0.0, sigma = sigma), x),
    builtin_fromnormal(Normal, record(mu = 0.0, sigma = 1e-22), 1e-22),
)
",
        "f32",
    );
    assert!(output.contains("flatppl.runtime_status = 4"), "{output}");
    assert_eq!(output.matches("stablehlo.divide").count(), 2, "{output}");
    assert_eq!(output.matches("stablehlo.multiply").count(), 1, "{output}");
    assert!(output.contains("stablehlo.add"), "{output}");
}

#[test]
fn affine_transports_keep_only_the_nonidentity_arithmetic() {
    let abi = "\
a = external(reals)
b = external(reals)
c = external(reals)
d = external(reals)
left = external(reals)
right = external(reals)
inputs = (a, b, c, d, left, right)
";
    let native = "\
scaled = builtin_tonormal(Normal, record(mu = 0.0, sigma = 2.0), a)
translated = builtin_fromnormal(Normal, record(mu = 1.25, sigma = 1.0), b)
centered = builtin_tonormal(Normal, record(mu = 3.5, sigma = 1.0), c)
shifted = builtin_fromnormal(Normal, record(mu = 0.0, sigma = 1.0), d)
outputs = (left * scaled + right * scaled, translated, centered, shifted)
";
    let reference = "\
scaled = a / 2.0
outputs = (left * scaled + right * scaled, 1.25 + b, c - 3.5, 0.0 + d)
";
    for dtype in ["f32", "f64"] {
        assert_eq!(
            emit("gaussian-affine", &format!("{abi}{native}"), dtype),
            emit("gaussian-affine", &format!("{abi}{reference}"), dtype)
        );
    }
    let shared = emit(
        "gaussian-affine",
        &format!(
            "{}outputs = (builtin_tonormal(Normal, record(mu = 0.0, sigma = 2.0), a), a / 2.0)\n",
            abi
        ),
        "f64",
    );
    assert_eq!(shared.matches("stablehlo.divide").count(), 2, "{shared}");
}

#[test]
fn unequally_weighted_antithetic_gaussian_scores_keep_both_sites() {
    let output = emit(
        "gaussian-antithetic",
        "\
x = external(reals)
sigma = external(reals)
a = external(reals)
b = external(reals)
law = Normal(0.0, sigma)
inputs = (x, sigma, a, b)
outputs = a * logdensityof(law, x) + b * logdensityof(law, -x)
",
        "f64",
    );
    assert_eq!(output.matches("stablehlo.divide").count(), 2, "{output}");
}
