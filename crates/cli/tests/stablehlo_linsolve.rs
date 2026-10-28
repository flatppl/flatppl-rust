use std::{fs, process::Command};

mod common;
use common::Scratch;

#[test]
fn linsolve_types_fractional_result_and_checks_shapes() {
    let dir = Scratch::new("linsolve-types");
    let input = dir.path("model.flatppl");
    let output = dir.path("out.flatpir");
    fs::write(
        &input,
        "A = rowstack([[2, 0], [0, 2]])\nb = [1, 3]\nx = linsolve(A, b)\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args(["infer", "--level=valueset"])
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let out = fs::read_to_string(&output).unwrap();
    let line = out.lines().find(|line| line.contains("(%bind x ")).unwrap();
    assert!(
        line.contains("(%array 1 (2) (%scalar real))") && line.contains("(cartpow reals 2)"),
        "{line}"
    );
    fs::write(
        &input,
        "A = rowstack([[1, 0], [0, 1]])\nb = [1, 2, 3]\nx = linsolve(A, b)\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("infer")
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&result.stderr).contains("matching vector"));
}

#[test]
fn batched_linsolve_emits_vector_cells() {
    let dir = Scratch::new("batched-linsolve");
    let input = dir.path("model.flatppl");
    fs::write(&input, "A = elementof(cartpow(cartpow(reals, [2,2]), [2]))\nb = elementof(cartpow(cartpow(reals, [2]), [2]))\ninputs = (A, b)\noutputs = linsolve.(A, b)\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("stablehlo")
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("tensor<2x2xf32>"));
    fs::write(&input, "A = elementof(cartpow(cartpow(reals, [2,2]), [2]))\nb = elementof(cartpow(cartpow(reals, [2]), [2]))\ninputs = (A, b)\noutputs = broadcast(linsolve, b = b, A = A)\n").unwrap();
    let named = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("stablehlo")
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        named.status.success(),
        "{}",
        String::from_utf8_lossy(&named.stderr)
    );
    assert_eq!(result.stdout, named.stdout);
}
