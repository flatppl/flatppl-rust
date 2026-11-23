#![cfg(feature = "optimise")]

mod common;

use std::fs;
use std::process::Command;

#[test]
fn optimises_source_and_flatpir_without_eliminating_the_measure_layer() {
    let scratch = common::Scratch::new("optimise");
    let source = "# A seeded sample\nv = external(cartpow(reals, 3))\n\
                  answer  = (transpose(transpose(v)))  # keep this comment\n\
                  x ~ Normal(0, 1)\ns = rnginit(51)\nsample , next = rand(lawof(x), s)\n";
    let source_path = scratch.path("model.flatppl");
    let ir_path = scratch.path("model.flatpir");
    let output_path = scratch.path("optimised.flatpir");
    fs::write(&source_path, source).unwrap();
    let parsed = flatppl_syntax::parse(source).unwrap();
    fs::write(&ir_path, flatppl_flatpir::try_write(&parsed).unwrap()).unwrap();

    let surface = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("optimise")
        .arg(source_path)
        .output()
        .unwrap();
    assert!(
        surface.status.success(),
        "{}",
        String::from_utf8_lossy(&surface.stderr)
    );
    let surface = String::from_utf8(surface.stdout).unwrap();
    assert_eq!(surface, source.replace("(transpose(transpose(v)))", "v"));

    let ir = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("optimize")
        .arg(ir_path)
        .args(["--emit", "flatpir", "-o"])
        .arg(&output_path)
        .output()
        .unwrap();
    assert!(
        ir.status.success(),
        "{}",
        String::from_utf8_lossy(&ir.stderr)
    );
    let rendered = fs::read_to_string(output_path).unwrap();
    let reparsed = flatppl_flatpir::read(&rendered).unwrap();
    let reparsed_surface = flatppl_syntax::parse(&surface).unwrap();
    assert_eq!(
        flatppl_syntax::print(&reparsed),
        flatppl_syntax::print(&reparsed_surface)
    );
}

#[test]
fn optimises_an_expression_typed_from_a_relative_import() {
    let scratch = common::Scratch::new("optimise-import");
    let source = scratch.path("model.flatppl");
    fs::write(scratch.path("dep.flatppl"), "values = [1.0, 2.0, 3.0]\n").unwrap();
    fs::write(
        &source,
        "d = load_module(\"dep.flatppl\")\nanswer = transpose(transpose(d.values))\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("optimise")
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let rendered = String::from_utf8(result.stdout).unwrap();
    assert!(rendered.contains("answer = d.values"), "{rendered}");
    fs::write(&source, rendered).unwrap();
    let inference = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("infer")
        .arg(source)
        .arg(scratch.path("roundtrip.flatpir"))
        .output()
        .unwrap();
    assert!(
        inference.status.success(),
        "{}",
        String::from_utf8_lossy(&inference.stderr)
    );
}
