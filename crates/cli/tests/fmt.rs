//! End-to-end tests for the standalone `flatppl-fmt fmt`.

use std::fs;
use std::process::Command;

mod common;
use common::Scratch;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_flatppl-fmt"))
}

#[test]
fn fmt_rewrites_in_place_and_is_idempotent() {
    let dir = Scratch::new("inplace");
    let f = dir.path("m.flatppl");
    fs::write(&f, "x ~ Normal(mu=0.0,sigma=1.0)\n").unwrap();

    assert!(bin().arg("fmt").arg(&f).status().unwrap().success());
    let once = fs::read_to_string(&f).unwrap();
    assert!(once.contains("mu = 0.0"), "got:\n{once}");
    assert!(once.ends_with('\n'));

    assert!(bin().arg("fmt").arg(&f).status().unwrap().success());
    assert_eq!(fs::read_to_string(&f).unwrap(), once);
}

#[test]
fn fmt_check_succeeds_on_canonical_and_fails_on_dirty() {
    let dir = Scratch::new("check");
    let f = dir.path("m.flatppl");
    fs::write(&f, "x ~ Normal(mu=0.0,sigma=1.0)\n").unwrap();

    assert_eq!(
        bin()
            .arg("fmt")
            .arg("--check")
            .arg(&f)
            .status()
            .unwrap()
            .code(),
        Some(1)
    );
    assert!(bin().arg("fmt").arg(&f).status().unwrap().success());
    assert!(
        bin()
            .arg("fmt")
            .arg("--check")
            .arg(&f)
            .status()
            .unwrap()
            .success()
    );
}

#[test]
fn fmt_stdin_to_stdout() {
    use std::io::Write;
    let mut child = bin()
        .arg("fmt")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"x ~ Normal(mu=0.0,sigma=1.0)\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("mu = 0.0"));
}

#[test]
fn fmt_check_stdin_fails_on_dirty_and_passes_on_canonical() {
    use std::io::Write;

    // Non-canonical stdin: `--check` must exit non-zero.
    let mut child = bin()
        .arg("fmt")
        .arg("--check")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"x ~ Normal(mu=0.0,sigma=1.0)\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(1));

    // Canonical stdin: `--check` must succeed.
    let mut child = bin()
        .arg("fmt")
        .arg("--check")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"x ~ Normal(mu = 0.0, sigma = 1.0)\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
}

#[test]
fn fmt_rejects_flatpir_input() {
    let dir = Scratch::new("flatpir");
    let f = dir.path("m.flatpir");
    fs::write(&f, "(%module)\n").unwrap();
    let out = bin().arg("fmt").arg(&f).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("only formats FlatPPL"));
}

#[test]
fn fmt_unknown_extension_errors() {
    let dir = Scratch::new("ext");
    let f = dir.path("m.txt");
    fs::write(&f, "x = 1.0\n").unwrap();
    let out = bin().arg("fmt").arg(&f).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn fmt_missing_file_errors() {
    let dir = Scratch::new("missing");
    let out = bin()
        .arg("fmt")
        .arg(dir.path("nope.flatppl"))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("reading"));
}

#[test]
fn fmt_invalid_syntax_reports_diagnostic() {
    let dir = Scratch::new("badsyntax");
    let f = dir.path("m.flatppl");
    fs::write(&f, "x\n").unwrap(); // bare name — parse error
    let out = bin().arg("fmt").arg(&f).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(!out.stderr.is_empty());
}

#[test]
fn fmt_stdin_invalid_syntax_errors() {
    use std::io::Write;
    let mut child = bin()
        .arg("fmt")
        .stdin(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"x\n").unwrap();
    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn fmt_check_stdin_dirty_fails() {
    use std::io::Write;
    let mut child = bin()
        .arg("fmt")
        .arg("--check")
        .stdin(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"x ~ Normal(mu=0.0,sigma=1.0)\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn fmt_check_reports_multiple_dirty_files() {
    let dir = Scratch::new("multidirty");
    let a = dir.path("a.flatppl");
    let b = dir.path("b.flatppl");
    fs::write(&a, "x ~ Normal(mu=0.0,sigma=1.0)\n").unwrap();
    fs::write(&b, "y ~ Normal(mu=0.0,sigma=1.0)\n").unwrap();
    let out = bin()
        .arg("fmt")
        .arg("--check")
        .arg(&a)
        .arg(&b)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("a.flatppl") && stderr.contains("b.flatppl"));
}

// The full `flatppl` driver also exposes fmt/lint (behind the default fmtlint
// feature) — exercise those dispatch arms on the main binary.
#[test]
fn full_flatppl_binary_exposes_fmt_and_lint() {
    let dir = Scratch::new("fullbin");
    let f = dir.path("m.flatppl");
    fs::write(&f, "mu = 0.0\nx ~ Normal(mu = mu, sigma = 1.0)\n").unwrap();
    let flatppl = || Command::new(env!("CARGO_BIN_EXE_flatppl"));
    assert!(
        flatppl()
            .arg("fmt")
            .arg("--check")
            .arg(&f)
            .status()
            .unwrap()
            .success()
    );
    assert!(flatppl().arg("lint").arg(&f).status().unwrap().success());
}

#[test]
fn fmt_no_extension_errors() {
    let dir = Scratch::new("noext");
    let f = dir.path("noextension");
    fs::write(&f, "x = 1.0\n").unwrap();
    let out = bin().arg("fmt").arg(&f).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("no file extension"));
}
