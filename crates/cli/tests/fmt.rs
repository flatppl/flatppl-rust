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
