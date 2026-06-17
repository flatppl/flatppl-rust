//! Integration tests for `flatppl convert --from hs3` and `--from pyhf`.

use std::path::Path;
use std::process::Command;

#[test]
fn convert_from_hs3_minimal() {
    let dir = std::env::temp_dir();
    let inp = dir.join("hs3_min_cli.json");
    let out = dir.join("hs3_min_cli.flatppl");
    std::fs::write(&inp, r#"{"distributions":[{"name":"mass","type":"gaussian_dist","mean":"mu","sigma":"s","x":"m_obs"}],"parameter_points":[{"name":"nom","entries":[{"name":"mu","value":5.28},{"name":"s","value":0.003}]}]}"#).unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args([
            "convert",
            "--from",
            "hs3",
            inp.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success(), "flatppl convert --from hs3 failed");
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(
        text.contains("Normal") && text.contains("record"),
        "got:\n{text}"
    );
    assert!(text.contains("relabel"), "got:\n{text}");
}

/// Path to the committed HS3 fixture directory (relative to CARGO_MANIFEST_DIR,
/// which for the CLI crate is `crates/cli`).
fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/hs3/tests/fixtures")
        .join(name)
}

#[test]
fn convert_from_pyhf_fixture() {
    let inp = fixture("2bin_1channel.json");
    let out = std::env::temp_dir().join("pyhf_2bin_cli.flatppl");
    let status = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args([
            "convert",
            "--from",
            "pyhf",
            inp.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success(), "flatppl convert --from pyhf failed");
    let text = std::fs::read_to_string(&out).unwrap();
    // The 2-bin/1-channel pyhf workspace must assemble into the point-free
    // histfactory likelihood: a Poisson observation model, a shapesys aux term,
    // and the joint_likelihood binding tying them together. Observed data
    // [50.0, 60.0] must appear literally.
    assert!(
        text.contains("obs_model_singlechannel"),
        "missing assembled obs_model binding, got:\n{text}"
    );
    assert!(
        text.contains("Poisson"),
        "missing Poisson observation model, got:\n{text}"
    );
    assert!(
        text.contains("ContinuedPoisson"),
        "missing shapesys ContinuedPoisson aux term, got:\n{text}"
    );
    assert!(
        text.contains("joint_likelihood("),
        "missing joint_likelihood binding, got:\n{text}"
    );
    assert!(
        text.contains("[50.0, 60.0]"),
        "missing observed data vector [50.0, 60.0], got:\n{text}"
    );
}

#[test]
fn convert_from_hs3_fixture() {
    let inp = fixture("paper_gaussian.json");
    let out = std::env::temp_dir().join("hs3_paper_gaussian_cli.flatppl");
    let status = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args([
            "convert",
            "--from",
            "hs3",
            inp.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(
        status.success(),
        "flatppl convert --from hs3 (paper_gaussian) failed"
    );
    let text = std::fs::read_to_string(&out).unwrap();
    // HS3 paper § A.1: a single gaussian_dist relabeled onto the observed
    // variate, a free mean param, a const-fixed sigma, the unbinned observation
    // value 1.27, and the likelihoodof wiring.
    assert!(
        text.contains("Normal(") && text.contains("relabel"),
        "missing relabeled Normal, got:\n{text}"
    );
    assert!(
        text.contains("mu = elementof(reals)"),
        "missing free mean parameter declaration, got:\n{text}"
    );
    assert!(
        text.contains("fixed(1.0)"),
        "missing const-fixed sigma, got:\n{text}"
    );
    assert!(
        text.contains("1.27"),
        "missing observed value 1.27, got:\n{text}"
    );
    assert!(
        text.contains("likelihoodof("),
        "missing likelihoodof wiring, got:\n{text}"
    );
}
