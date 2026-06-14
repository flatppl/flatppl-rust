//! End-to-end tests for the standalone `flatppl-fmt lint`.

use std::fs;
use std::process::Command;

mod common;
use common::Scratch;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_flatppl-fmt"))
}

#[test]
fn clean_file_lints_clean() {
    let dir = Scratch::new("clean");
    let f = dir.path("m.flatppl");
    fs::write(&f, "mu = 0.0\nx ~ Normal(mu = mu, sigma = 1.0)\n").unwrap();
    let out = bin().arg("lint").arg(&f).output().unwrap();
    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn warnings_alone_do_not_fail_but_print() {
    let dir = Scratch::new("warn");
    let f = dir.path("m.flatppl");
    fs::write(&f, "_helper = 1.0\nx ~ Normal(mu = 0.0, sigma = 1.0)\n").unwrap();
    let out = bin().arg("lint").arg(&f).output().unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("unused-binding"));
}

#[test]
fn deny_warnings_makes_warnings_fail() {
    let dir = Scratch::new("denywarn");
    let f = dir.path("m.flatppl");
    fs::write(&f, "_helper = 1.0\nx ~ Normal(mu = 0.0, sigma = 1.0)\n").unwrap();
    let out = bin()
        .arg("lint")
        .arg("--deny-warnings")
        .arg(&f)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn allow_flag_silences_a_rule() {
    let dir = Scratch::new("allow");
    let f = dir.path("m.flatppl");
    fs::write(&f, "_helper = 1.0\nx ~ Normal(mu = 0.0, sigma = 1.0)\n").unwrap();
    let out = bin()
        .arg("lint")
        .arg("--allow")
        .arg("unused-binding")
        .arg("--deny-warnings")
        .arg(&f)
        .output()
        .unwrap();
    assert!(out.status.success());
}

#[test]
fn inline_directive_suppresses_a_rule_file_wide() {
    let dir = Scratch::new("suppress");
    let f = dir.path("m.flatppl");
    // Note: this content is canonical (spaces around = and after ,), so not-canonical will not fire.
    // The inline directive suppresses unused-binding, allowing --deny-warnings to pass.
    fs::write(
        &f,
        "% flatppl-lint: allow unused-binding\n_helper = 1.0\nx ~ Normal(mu = 0.0, sigma = 1.0)\n",
    )
    .unwrap();
    let out = bin()
        .arg("lint")
        .arg("--deny-warnings")
        .arg(&f)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn not_canonical_is_reported() {
    let dir = Scratch::new("noncanon");
    let f = dir.path("m.flatppl");
    fs::write(&f, "x ~ Normal(mu=0.0,sigma=1.0)\n").unwrap();
    let out = bin().arg("lint").arg(&f).output().unwrap();
    assert!(String::from_utf8_lossy(&out.stderr).contains("not-canonical"));
}

#[test]
fn unknown_rule_name_errors() {
    let dir = Scratch::new("badrule");
    let f = dir.path("m.flatppl");
    fs::write(&f, "mu = 0.0\n").unwrap();
    let out = bin()
        .arg("lint")
        .arg("--deny")
        .arg("no-such-rule")
        .arg(&f)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn lints_a_repo_fixture_without_crashing() {
    // A real corpus model should lint to a clean exit (warnings allowed; only
    // deny-level issues — unresolved names, cycles — fail).
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/flatppl/eight-schools.flatppl"
    );
    let out = bin().arg("lint").arg(path).output().unwrap();
    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
