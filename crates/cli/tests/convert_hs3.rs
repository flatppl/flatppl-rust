//! Integration tests for `flatppl convert --from hs3` and `--from pyhf`.
//!
//! `hs3` is an opt-in CLI feature, so these only compile/run with
//! `--features hs3` (CI exercises them via `--all-features`); the default
//! `cargo test` build skips the whole file.
#![cfg(feature = "hs3")]

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

/// `.hs3.json` / `.pyhf.json` names are auto-detected without `--from`
/// (mirroring the `.flatpir.json` convention; an explicit `--from` overrides).
/// The discriminator is the emitted content (HS3 → a `Normal`, pyhf → an
/// assembled `joint_likelihood`); both stamp the leading `flatppl_compat`.
#[test]
fn auto_detects_hs3_and_pyhf_by_extension() {
    let dir = std::env::temp_dir();

    // `*.hs3.json` → HS3 importer, no `--from`.
    let hs3_in = dir.join("auto_detect.hs3.json");
    std::fs::write(&hs3_in, r#"{"distributions":[{"name":"mass","type":"gaussian_dist","mean":"mu","sigma":"s","x":"m_obs"}],"parameter_points":[{"name":"nom","entries":[{"name":"mu","value":5.28},{"name":"s","value":0.003}]}]}"#).unwrap();
    let hs3_out = dir.join("auto_detect_hs3.flatppl");
    let status = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args(["convert", hs3_in.to_str().unwrap(), hs3_out.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(status.success(), "auto-detected .hs3.json convert failed");
    let text = std::fs::read_to_string(&hs3_out).unwrap();
    assert!(text.contains("Normal"), "HS3 path must emit a Normal, got:\n{text}");
    assert!(
        text.contains("flatppl_compat = \"0.1\""),
        "generated module must stamp flatppl_compat, got:\n{text}"
    );

    // `*.pyhf.json` → pyhf importer, no `--from`.
    let pyhf_in = dir.join("auto_detect.pyhf.json");
    std::fs::copy(fixture("2bin_1channel.json"), &pyhf_in).unwrap();
    let pyhf_out = dir.join("auto_detect_pyhf.flatppl");
    let status = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args(["convert", pyhf_in.to_str().unwrap(), pyhf_out.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(status.success(), "auto-detected .pyhf.json convert failed");
    let text = std::fs::read_to_string(&pyhf_out).unwrap();
    assert!(
        text.contains("joint_likelihood("),
        "pyhf path must assemble a joint_likelihood, got:\n{text}"
    );
    assert!(
        text.contains("flatppl_compat = \"0.1\""),
        "generated module must stamp flatppl_compat, got:\n{text}"
    );
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

/// HS3/pyhf conversions carry a minimal "do not edit" banner by default (a single
/// FlatPPL `#` line comment) and stamp `flatppl_compat = "0.1"` as the leading
/// binding. The banner leaks no personal/system information (no timestamp, user,
/// host, platform, or command line). `--no-header` drops the banner but keeps the
/// `flatppl_compat` binding — it is part of the model, not the comment.
#[test]
fn hs3_convert_emits_banner_and_compat() {
    let dir = std::env::temp_dir();
    let inp = dir.join("hs3_prov_cli.json");
    let out = dir.join("hs3_prov_cli.flatppl");
    std::fs::write(
        &inp,
        r#"{"distributions":[{"name":"mass","type":"gaussian_dist","mean":"mu","sigma":"s","x":"m_obs"}],"parameter_points":[{"name":"nom","entries":[{"name":"mu","value":1.0},{"name":"s","value":1.0}]}]}"#,
    )
    .unwrap();

    // Default: minimal banner, then the leading flatppl_compat binding.
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
    assert!(status.success());
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(
        text.starts_with("# AUTOMATICALLY GENERATED - do not edit\n"),
        "expected a minimal leading FlatPPL banner, got:\n{text}"
    );
    assert!(
        text.contains("flatppl_compat = \"0.1\""),
        "generated module must stamp the leading flatppl_compat binding, got:\n{text}"
    );
    // No pseudo-provenance / personal information of any kind.
    for leaked in [
        "generator:",
        "from:",
        "by:",
        "platform:",
        "command:",
        "generated:",
    ] {
        assert!(!text.contains(leaked), "banner must not leak `{leaked}`, got:\n{text}");
    }

    // --no-header: banner gone, but flatppl_compat (a binding) leads the output.
    let status = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args([
            "convert",
            "--from",
            "hs3",
            "--no-header",
            inp.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(
        !text.contains("AUTOMATICALLY GENERATED"),
        "--no-header must omit the banner, got:\n{text}"
    );
    assert!(
        text.starts_with("flatppl_compat = \"0.1\""),
        "flatppl_compat must persist and lead under --no-header, got:\n{text}"
    );
}
