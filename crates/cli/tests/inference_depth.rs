//! Reference depth is independent of parsed expression depth.

use std::fs;
use std::process::Command;

use flatppl_core::depth::DEFAULT_MAX_DEPTH;

mod common;
use common::Scratch;

#[test]
fn deep_forward_references_return_a_resource_error() {
    let dir = Scratch::new("infer-reference-depth");
    for count in [DEFAULT_MAX_DEPTH, DEFAULT_MAX_DEPTH + 1, 4001] {
        let mut source = String::new();
        for i in 0..count - 1 {
            source.push_str(&format!("x{i} = x{}\n", i + 1));
        }
        source.push_str(&format!("x{} = 1\n", count - 1));
        let input = dir.path(&format!("chain-{count}.flatppl"));
        fs::write(&input, source).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_flatppl"))
            .arg("infer")
            .arg(input)
            .arg(dir.path(&format!("chain-{count}.flatpir")))
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        if count == DEFAULT_MAX_DEPTH {
            assert!(output.status.success(), "{stderr}");
        } else {
            assert_eq!(output.status.code(), Some(1), "{stderr}");
            assert!(stderr.contains("inference graph"), "{stderr}");
            assert!(stderr.contains("resource guard"), "{stderr}");
        }
    }
}

#[test]
fn wide_modules_do_not_consume_the_reference_depth_budget() {
    let dir = Scratch::new("infer-reference-width");
    let source: String = (0..4001).map(|i| format!("x{i} = 1\n")).collect();
    let input = dir.path("wide.flatppl");
    fs::write(&input, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("infer")
        .arg(input)
        .arg(dir.path("wide.flatpir"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
