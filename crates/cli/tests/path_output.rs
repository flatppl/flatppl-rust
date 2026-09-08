//! File paths are terminal data: their bytes must not control diagnostic layout.

use std::fs;
use std::process::Command;

mod common;
use common::Scratch;

const ESCAPED_TAIL: &str = "missing\\n\\u{1b}\\u{202e}\\u{200b}café.flatppl";

#[test]
fn missing_dependency_path_cannot_forge_a_diagnostic_line() {
    let dir = Scratch::new("safe-dependency-path");
    let input = dir.path("model.flatppl");
    fs::write(
        &input,
        "m = load_module(\"missing\\n\u{1b}\u{202e}\u{200b}café.flatppl\")\n",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("prepare")
        .arg(&input)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains(ESCAPED_TAIL), "stderr:\n{stderr}");
    assert!(!stderr.contains("missing\n"), "stderr:\n{stderr}");
}

#[cfg(unix)]
#[test]
fn diagnostic_input_path_cannot_forge_a_diagnostic_line() {
    let dir = Scratch::new("safe-diagnostic-path");
    let input = dir.path("missing\n\u{1b}\u{202e}\u{200b}café.flatppl");
    fs::write(&input, "x =\n").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("fmt")
        .arg("--check")
        .arg(&input)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains(ESCAPED_TAIL), "stderr:\n{stderr}");
    assert!(!stderr.contains("missing\n"), "stderr:\n{stderr}");
}

#[cfg(unix)]
#[test]
fn unsupported_extension_cannot_forge_a_diagnostic_line() {
    let dir = Scratch::new("safe-extension");
    let input = dir.path("model.bad\n\u{1b}\u{202e}\u{200b}café");

    let output = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("fmt")
        .arg("--check")
        .arg(&input)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains(".bad\\n\\u{1b}\\u{202e}\\u{200b}café"),
        "stderr:\n{stderr}"
    );
    assert!(!stderr.contains(".bad\n"), "stderr:\n{stderr}");
}
