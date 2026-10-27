use std::{fs, process::Command};

mod common;
use common::Scratch;

#[test]
fn qr_record_has_thin_real_shapes_and_rejects_wide_matrices() {
    let dir = Scratch::new("qr-types");
    let input = dir.path("model.flatppl");
    let output = dir.path("out.flatpir");
    fs::write(
        &input,
        "A = rowstack([[1, 0], [0, 1], [1, 1]])\nx = qr(A)\n",
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
        line.contains("(Q (%array 2 (3 2) (%scalar real)))")
            && line.contains("(R (%array 2 (2 2) (%scalar real)))"),
        "{line}"
    );
    fs::write(&input, "A = rowstack([[1, 0, 2], [0, 1, 3]])\nx = qr(A)\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("infer")
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&result.stderr).contains("at least as many rows"));
}

#[test]
fn qr_record_fields_can_be_projected() {
    let dir = Scratch::new("qr-fields");
    let input = dir.path("model.flatppl");
    fs::write(&input, "A = elementof(cartpow(reals, [3,2]))\nfactor = qr(A)\ninputs = A\noutputs = (factor.Q, factor.R)\n").unwrap();
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
    let out = String::from_utf8_lossy(&result.stdout);
    assert!(
        out.contains("tensor<3x2xf32>") && out.contains("tensor<2x2xf32>"),
        "{out}"
    );
    assert_eq!(out.matches("= stablehlo.while(").count(), 1, "{out}");
}
