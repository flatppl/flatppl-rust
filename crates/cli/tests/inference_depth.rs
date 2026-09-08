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

/// Inference annotations nest as deep as the value they describe, so a model
/// well inside every source limit can render FlatPIR past the reader's own
/// depth guard. `infer` used to exit 0 and write that file. It must now either
/// succeed with output the toolchain reads back, or refuse by name.
#[test]
fn inferred_flatpir_either_reads_back_or_is_refused_by_name() {
    let dir = Scratch::new("infer-flatpir-roundtrip");
    // A rank-64 array (the `addaxes` structural-rank cap) under enough record
    // wrappers to cross the reader's guard, and few enough to stay inside it.
    let mut crossed = (false, false);
    for levels in [10usize, 24] {
        let mut inner = String::from("addaxes(v, 0, 63)");
        for _ in 0..levels {
            inner = format!("record(f = {inner})");
        }
        let input = dir.path(&format!("nested-{levels}.flatppl"));
        let emitted = dir.path(&format!("nested-{levels}.flatpir"));
        fs::write(&input, format!("v = [1.0, 2.0, 3.0]\nw = {inner}\n")).unwrap();
        let inferred = Command::new(env!("CARGO_BIN_EXE_flatppl"))
            .arg("infer")
            .arg(&input)
            .arg(&emitted)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&inferred.stderr);
        if inferred.status.success() {
            crossed.0 = true;
            let reread = Command::new(env!("CARGO_BIN_EXE_flatppl"))
                .arg("convert")
                .arg(&emitted)
                .arg(dir.path(&format!("nested-{levels}.roundtrip.flatppl")))
                .output()
                .unwrap();
            assert!(
                reread.status.success(),
                "level {levels}: infer exited 0 but its output does not read back: {}",
                String::from_utf8_lossy(&reread.stderr)
            );
        } else {
            crossed.1 = true;
            assert_eq!(inferred.status.code(), Some(1), "{stderr}");
            assert!(stderr.contains("cannot be read back"), "{stderr}");
            assert!(
                !emitted.exists(),
                "level {levels}: a refused render must leave no file"
            );
        }
    }
    assert_eq!(crossed, (true, true), "the sweep must cross the boundary");
}
