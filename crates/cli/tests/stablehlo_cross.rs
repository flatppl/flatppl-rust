use std::{fs, process::Command};

mod common;
use common::Scratch;

#[test]
fn cross_preserves_promoted_element_type_and_requires_three_elements() {
    let dir = Scratch::new("cross-domain");
    let input = dir.path("model.flatppl");
    let output = dir.path("out.flatpir");
    fs::write(
        &input,
        "a = [1, 2, 3]\nb = [1.0, 0.5, 0.0]\nx = cross(a, b)\n",
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
    assert!(line.contains("(%array 1 (3) (%scalar real))"), "{line}");
    assert!(line.contains("(cartpow reals 3)"), "{line}");
    fs::write(&input, "x = cross([1, 2], [1, 2, 3])\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("infer")
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&result.stderr).contains("length 3"));
}

#[test]
fn broadcast_cross_emits_vector_cells() {
    let dir = Scratch::new("broadcast-cross");
    let input = dir.path("model.flatppl");
    fs::write(&input, "a = elementof(cartpow(cartpow(reals, [3]), [2]))\nb = elementof(cartpow(cartpow(reals, [3]), [2]))\ninputs = (a, b)\noutputs = cross.(a, b)\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("stablehlo")
        .arg(input)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("tensor<2x3xf32>"));
}
