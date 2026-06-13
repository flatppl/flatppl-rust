//! End-to-end tests for `flatppl convert`, exercising the built binary.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_flatppl"))
}

/// A scratch dir unique to this test process, cleaned up on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("flatppl-cli-{label}-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).ok();
    }
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

/// Generated files carry a provenance header by default: a leading comment
/// block (`;` for FlatPIR) recording generator, source format + file, and platform.
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
        text.starts_with("; AUTOMATICALLY GENERATED"),
        "expected a leading FlatPIR provenance comment, got:\n{text}"
    );
    for field in [
        "generator:  flatppl",
        "from:       FlatPPL file `m.flatppl`",
        "platform:",
    ] {
        assert!(text.contains(field), "missing `{field}` in:\n{text}");
    }
    // The model still follows the header.
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
    assert!(written.contains("(elementof (%meta (%scalar real) %parameterized reals) reals)"));
    assert!(written.contains("(draw (%meta (%scalar real) %stochastic reals)"));
    assert!(
        written.contains("(mystery (%meta %deferred %stochastic %unknown)"),
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
