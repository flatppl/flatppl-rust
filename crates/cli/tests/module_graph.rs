//! Two importer-local `common.flatppl` files are two distinct dependencies.
//!
//! Spec §04 "Path resolution": "Relative file paths in `load_module(...)` are
//! resolved relative to the directory of the FlatPPL file containing that
//! `load_module(...)` call". `a/model.flatppl` and `b/model.flatppl` may each
//! load their own `common.flatppl`, so the graph below is valid and must not be
//! refused. The bundle keys each dependency by resolved location, not by the
//! directive string, so the two entries coexist.

use std::fs;
use std::process::Command;

mod common;
use common::Scratch;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_flatppl"))
}

/// The `a`/`b` graph: a top model importing two intermediates that each load
/// their own `common.flatppl` under that one spelling.
fn write_graph(dir: &Scratch) -> std::path::PathBuf {
    for side in ["a", "b"] {
        fs::create_dir_all(dir.path(side)).unwrap();
        fs::write(
            dir.path(&format!("{side}/model.flatppl")),
            "c = load_module(\"common.flatppl\")\nx = c.val\n",
        )
        .unwrap();
    }
    fs::write(dir.path("a/common.flatppl"), "val = 1.5\n").unwrap();
    fs::write(dir.path("b/common.flatppl"), "val = true\n").unwrap();
    let top = dir.path("top.flatppl");
    fs::write(
        &top,
        "ma = load_module(\"a/model.flatppl\")\n\
         mb = load_module(\"b/model.flatppl\")\n\
         ya = ma.x\n\
         yb = mb.x\n",
    )
    .unwrap();
    top
}

#[test]
fn two_importer_local_commons_are_not_a_conflict() {
    let dir = Scratch::new("modgraph");
    let top = write_graph(&dir);
    let out = bin()
        .arg("infer")
        .arg(&top)
        .arg(dir.path("out.flatpir"))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "the graph is valid per §04; stderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("two different files"),
        "no conflict to report; stderr:\n{stderr}"
    );
    // Both intermediates resolved, so both refs are in the emitted module.
    let emitted = fs::read_to_string(dir.path("out.flatpir")).unwrap();
    assert!(emitted.contains("(%ref ma x)"), "got:\n{emitted}");
    assert!(emitted.contains("(%ref mb x)"), "got:\n{emitted}");
}
