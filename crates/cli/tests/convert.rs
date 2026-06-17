//! End-to-end tests for `flatppl convert`, exercising the built binary.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

mod common;
use common::Scratch;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_flatppl"))
}

/// Run `flatppl convert --no-header <input> <output>`, asserting success.
fn convert_nh(input: &std::path::Path, output: &std::path::Path) {
    let status = bin()
        .args(["convert", "--no-header"])
        .arg(input)
        .arg(output)
        .status()
        .unwrap();
    assert!(
        status.success(),
        "convert {} -> {} failed",
        input.display(),
        output.display()
    );
}

#[test]
fn converts_flatppl_to_flatpir_and_back() {
    let dir = Scratch::new("roundtrip");
    let src = dir.path("model.flatppl");
    let pir = dir.path("model.flatpir");
    let back = dir.path("back.flatppl");
    fs::write(
        &src,
        "mu = elementof(reals)\nx ~ Normal(mu = mu, sigma = 1.0)\n",
    )
    .unwrap();

    let status = bin().arg("convert").arg(&src).arg(&pir).status().unwrap();
    assert!(status.success());
    let pir_text = fs::read_to_string(&pir).unwrap();
    assert!(
        pir_text.contains("(%bind x (draw (Normal"),
        "got:\n{pir_text}"
    );
    assert!(pir_text.ends_with('\n'));

    let status = bin().arg("convert").arg(&pir).arg(&back).status().unwrap();
    assert!(status.success());
    let back_text = fs::read_to_string(&back).unwrap();
    assert!(
        back_text.contains("x ~ Normal(mu = mu, sigma = 1.0)"),
        "got:\n{back_text}"
    );
}

/// Same-format conversion canonicalizes (one stmt per line, sugar re-applied
/// by default; `--syntax minimal` emits the lowered call form instead).
#[test]
fn same_format_canonicalizes() {
    let dir = Scratch::new("canon");
    let src = dir.path("messy.flatppl");
    let out = dir.path("canonical.flatppl");
    fs::write(&src, "x = add(1, 2); y ~ Normal(0, 1)\n").unwrap();

    // --no-header keeps canonicalization byte-exact (the provenance header is
    // covered separately in provenance_header_*).
    let status = bin()
        .args(["convert", "--no-header"])
        .arg(&src)
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(
        fs::read_to_string(&out).unwrap(),
        "x = 1 + 2\ny ~ Normal(0, 1)\n"
    );

    let status = bin()
        .args(["convert", "--no-header", "--syntax", "minimal"])
        .arg(&src)
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(
        fs::read_to_string(&out).unwrap(),
        "x = add(1, 2)\ny ~ Normal(0, 1)\n"
    );
}

/// Generated files carry a minimal "do not edit" banner by default: a single
/// leading comment line (`;` for FlatPIR). It records nothing else — no
/// timestamp, user, host, platform, or command line — so no personal or system
/// information leaks (see `provenance.rs`).
#[test]
fn provenance_header_present_by_default() {
    let dir = Scratch::new("prov");
    let src = dir.path("m.flatppl");
    let out = dir.path("m.flatpir");
    fs::write(&src, "x ~ Normal(0, 1)\n").unwrap();

    let status = bin().arg("convert").arg(&src).arg(&out).status().unwrap();
    assert!(status.success());
    let text = fs::read_to_string(&out).unwrap();
    assert!(
        text.starts_with("; AUTOMATICALLY GENERATED - do not edit\n"),
        "expected a leading FlatPIR banner, got:\n{text}"
    );
    // No pseudo-provenance / personal fields leak into the banner.
    for leaked in [
        "generator:",
        "from:",
        "by:",
        "platform:",
        "command:",
        "generated:",
    ] {
        assert!(!text.contains(leaked), "banner must not leak `{leaked}` in:\n{text}");
    }
    // The model still follows the banner.
    assert!(
        text.contains("(%module"),
        "model body missing, got:\n{text}"
    );
}

/// `--no-header` suppresses the provenance block entirely.
#[test]
fn no_header_suppresses_provenance() {
    let dir = Scratch::new("noprov");
    let src = dir.path("m.flatppl");
    let out = dir.path("m.flatpir");
    fs::write(&src, "x ~ Normal(0, 1)\n").unwrap();

    let status = bin()
        .args(["convert", "--no-header"])
        .arg(&src)
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let text = fs::read_to_string(&out).unwrap();
    assert!(
        !text.contains("AUTOMATICALLY GENERATED"),
        "--no-header must omit the provenance block, got:\n{text}"
    );
}

#[test]
fn rejects_unknown_extension() {
    let out = bin()
        .args(["convert", "model.txt", "model.flatpir"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("extension"));
}

#[test]
fn reports_parse_errors_with_the_file_name() {
    let dir = Scratch::new("err");
    let src = dir.path("bad.flatppl");
    fs::write(&src, "x = \n").unwrap();

    let out = bin()
        .arg("convert")
        .arg(&src)
        .arg(dir.path("out.flatpir"))
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("bad.flatppl"), "got:\n{stderr}");
}

/// Parse errors render as a source-annotated report: file:line:col header,
/// the offending source line, and a span marker — not just a bare message.
#[test]
fn renders_span_diagnostics() {
    let dir = Scratch::new("diag");
    let src = dir.path("bad.flatppl");
    fs::write(&src, "x = 1\nself = elementof(reals)\n").unwrap();

    let out = bin()
        .arg("convert")
        .arg(&src)
        .arg(dir.path("out.flatpir"))
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("`self` is a reserved module name"),
        "got:\n{stderr}"
    );
    assert!(stderr.contains("bad.flatppl:2:1"), "got:\n{stderr}");
    // The offending source line is quoted in the snippet.
    assert!(stderr.contains("self = elementof(reals)"), "got:\n{stderr}");
}

/// `flatppl infer` annotates a module with `%meta` and reports honest gaps
/// as notes on stderr.
#[test]
fn infer_emits_annotated_flatpir() {
    let dir = Scratch::new("infer");
    let src = dir.path("m.flatppl");
    let out_path = dir.path("m.flatpir");
    fs::write(
        &src,
        "a = elementof(reals)\nb ~ Normal(a, 1.0)\nc = mystery(b)\n",
    )
    .unwrap();

    let out = bin()
        .arg("infer")
        .arg(&src)
        .arg(&out_path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("note: no type rule for `mystery`"),
        "got:\n{stderr}"
    );

    let written = fs::read_to_string(&out_path).unwrap();
    assert!(written.contains("(%meta ((%scalar real) %parameterized reals) (elementof reals))"));
    assert!(written.contains("(%meta ((%scalar real) %stochastic reals) (draw "));
    assert!(
        written.contains("(%meta (%deferred %stochastic %unknown) (mystery "),
        "got:\n{written}"
    );
}

/// `flatppl infer` refuses a FlatPPL output path (annotations need FlatPIR).
#[test]
fn infer_rejects_flatppl_output() {
    let dir = Scratch::new("infer-out");
    let src = dir.path("m.flatppl");
    fs::write(&src, "x = 1\n").unwrap();
    let out = bin()
        .arg("infer")
        .arg(&src)
        .arg(dir.path("m.flatppl"))
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains(".flatpir"));
}

/// `--level=valueset` fills the value-set slot but leaves masses %deferred
/// (the level hierarchy is observable on the wire).
#[test]
fn infer_level_valueset_leaves_mass_deferred() {
    let dir = Scratch::new("level");
    let src = dir.path("m.flatppl");
    let out_path = dir.path("m.flatpir");
    fs::write(&src, "m = Normal(0.0, 1.0)\n").unwrap();

    let out = bin()
        .arg("infer")
        .arg("--level=valueset")
        .arg(&src)
        .arg(&out_path)
        .output()
        .unwrap();
    assert!(out.status.success());
    let written = fs::read_to_string(&out_path).unwrap();
    assert!(
        written.contains("(%measure (%domain (%scalar real)) (%mass %deferred)) %fixed reals)"),
        "got:\n{written}"
    );
}

/// `flatppl completions <shell>` emits a non-empty completion script naming the
/// driver and its subcommands, for every shell clap_complete supports.
#[test]
fn completions_generate_for_each_shell() {
    for shell in ["bash", "zsh", "fish", "powershell", "elvish"] {
        let out = bin().args(["completions", shell]).output().unwrap();
        assert!(out.status.success(), "completions {shell} failed");
        let script = String::from_utf8_lossy(&out.stdout);
        assert!(
            !script.trim().is_empty(),
            "{shell}: empty completion script"
        );
        assert!(
            script.contains("flatppl")
                && script.contains("convert")
                && script.contains("completions"),
            "{shell}: completion script missing driver/subcommands:\n{script}"
        );
    }
}

/// `convert` round-trips through the FlatPIR JSON encoding (`.flatpir.json`):
/// FlatPPL → JSON → FlatPIR. The JSON output is valid JSON with NO provenance
/// header (JSON has no comment syntax), and the value round-trips back.
#[test]
fn converts_through_flatpir_json() {
    let dir = Scratch::new("json");
    let src = dir.path("model.flatppl");
    let js = dir.path("model.flatpir.json");
    let pir = dir.path("back.flatpir");
    fs::write(
        &src,
        "mu = elementof(reals)\nx ~ Normal(mu = mu, sigma = 1.0)\n",
    )
    .unwrap();

    // FlatPPL → .flatpir.json
    let status = bin().arg("convert").arg(&src).arg(&js).status().unwrap();
    assert!(status.success());
    let json = fs::read_to_string(&js).unwrap();
    // Valid JSON, no comment/provenance header prepended.
    assert!(json.trim_start().starts_with('{'), "got:\n{json}");
    assert!(
        !json.contains("generated by"),
        "JSON must carry no header:\n{json}"
    );
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("output is valid JSON");
    assert!(parsed.get("%module").is_some(), "got:\n{json}");

    // .flatpir.json → FlatPIR
    let status = bin().arg("convert").arg(&js).arg(&pir).status().unwrap();
    assert!(status.success());
    let pir_text = fs::read_to_string(&pir).unwrap();
    assert!(
        pir_text.contains("(%bind x (draw (Normal"),
        "got:\n{pir_text}"
    );
}

/// A bare `.json` (not `.flatpir.json`) is not a recognized format.
#[test]
fn rejects_bare_json_extension() {
    let out = bin()
        .args(["convert", "model.flatpir", "model.json"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("extension"));
}

/// Integrity of the FlatPIR↔JSON round-trip over real models: take every
/// `.flatppl` fixture, lower it to FlatPIR, encode that to `.flatpir.json`, and
/// decode back to FlatPIR — the FlatPIR→JSON→FlatPIR leg must be byte-identical
/// (the JSON encoding loses nothing). All conversions use `--no-header` so the
/// canonical text is directly comparable.
#[test]
fn flatppl_through_flatpir_json_is_lossless() {
    let fixtures: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "..",
        "..",
        "fixtures",
        "flatppl",
    ]
    .iter()
    .collect();
    let dir = Scratch::new("pipeline");
    let mut count = 0;
    for entry in fs::read_dir(&fixtures).expect("read fixtures/flatppl") {
        let src = entry.unwrap().path();
        if src.extension().and_then(|e| e.to_str()) != Some("flatppl") {
            continue;
        }
        let stem = src.file_stem().unwrap().to_str().unwrap();
        let pir = dir.path(&format!("{stem}.flatpir"));
        let json = dir.path(&format!("{stem}.flatpir.json"));
        let back = dir.path(&format!("{stem}.back.flatpir"));

        convert_nh(&src, &pir); // FlatPPL  → FlatPIR
        convert_nh(&pir, &json); // FlatPIR  → JSON
        convert_nh(&json, &back); // JSON     → FlatPIR

        // The FlatPIR → JSON → FlatPIR leg must round-trip exactly.
        assert_eq!(
            fs::read_to_string(&pir).unwrap(),
            fs::read_to_string(&back).unwrap(),
            "FlatPIR→JSON→FlatPIR not lossless for `{stem}`"
        );
        count += 1;
    }
    assert!(
        count >= 8,
        "expected at least 8 .flatppl fixtures, found {count}"
    );
}
