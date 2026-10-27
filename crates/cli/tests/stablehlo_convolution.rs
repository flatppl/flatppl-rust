use std::{fs, process::Command};

mod common;
use common::Scratch;

#[test]
fn convolution_types_and_rejects_long_kernel() {
    let dir = Scratch::new("convolution-types");
    let input = dir.path("model.flatppl");
    let output = dir.path("out.flatpir");
    fs::write(
        &input,
        "v = [1, 2, 3, 4]\nk = [1.0, 0.5, -1.0]\nx = conv(v, k)\n",
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
    fs::write(&input, "x = crosscorr([1, 2], [1, 2, 3])\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("infer")
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("kernel length at most signal length")
    );
}

#[test]
fn broadcast_convolution_accepts_vector_cells_and_named_arguments() {
    let dir = Scratch::new("convolution-broadcast");
    let input = dir.path("model.flatppl");
    for head in ["conv", "crosscorr"] {
        let prefix = "v = elementof(cartpow(cartpow(reals, [4]), [2]))\nk = elementof(cartpow(cartpow(reals, [3]), [2]))\ninputs = (v, k)\n";
        fs::write(&input, format!("{prefix}outputs = {head}.(v, k)\n")).unwrap();
        let positional = Command::new(env!("CARGO_BIN_EXE_flatppl"))
            .arg("stablehlo")
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            positional.status.success(),
            "{}",
            String::from_utf8_lossy(&positional.stderr)
        );
        assert!(String::from_utf8_lossy(&positional.stdout).contains("tensor<2x2xf32>"));
        fs::write(
            &input,
            format!("{prefix}outputs = broadcast({head}, kernel = k, v = v)\n"),
        )
        .unwrap();
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
        assert_eq!(positional.stdout, named.stdout);
    }
    fs::write(&input, "v = [1, 2, 3]\nk = [1, 2, 3]\nx = conv.(v, k)\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("infer")
        .arg(&input)
        .arg(dir.path("bad.flatpir"))
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
}
