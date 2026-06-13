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
    assert!(!text.trim().is_empty(), "output should be non-empty");
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
    assert!(!text.trim().is_empty(), "output should be non-empty");
}
