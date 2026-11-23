#![cfg(all(feature = "autodiff", feature = "stablehlo"))]

use std::process::Command;

mod common;

#[test]
fn named_rectangle_derivative_reloads_with_a_live_endpoint() {
    let model = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/named_rectangle/model.flatppl"
    );
    let scratch = common::Scratch::new("autodiff-named-rectangle");
    let source = scratch.path("derivative.flatppl");
    let binary = env!("CARGO_BIN_EXE_flatppl");
    let ad = Command::new(binary)
        .args(["autodiff", model, "--numerical-integrals"])
        .args(["--wrt", "a", "--wrt", "hi", "--wrt", "xx", "--wrt", "yy"])
        .arg("-o")
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        ad.status.success(),
        "{}",
        String::from_utf8_lossy(&ad.stderr)
    );
    let lower = Command::new(binary)
        .arg("stablehlo")
        .arg(source)
        .args(["--numerical-integrals", "--dtype", "f64"])
        .output()
        .unwrap();
    assert!(
        lower.status.success(),
        "{}",
        String::from_utf8_lossy(&lower.stderr)
    );
    assert!(
        String::from_utf8(lower.stdout)
            .unwrap()
            .contains("stablehlo.while")
    );
}
