#![cfg(all(feature = "optimise", feature = "stablehlo"))]

mod common;

use std::fs;
use std::path::Path;
use std::process::Command;

fn execute(path: &Path, args: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args(args)
        .arg(path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn overlapping_intervals_stage_one_exact_density_and_mass_query() {
    let scratch = common::Scratch::new("measure-algebra");
    let source = scratch.path("nested.flatppl");
    let reference = scratch.path("single.flatppl");
    let prefix = "rate = external(reals)\npoint = external(reals)\n\
                  law = Exponential(rate)\ninputs = (rate, point)\n";
    let restrictions = "first = truncate(law, interval(0.25, 3.0))\n\
                        second = truncate(first, interval(1.0, 5.0))\n\
                        window = truncate(second, interval(0.5, 2.5))\n";
    let observations = "conditioned = normalize(window)\n\
                        score = logdensityof(conditioned, point)\n\
                        mass = totalmass(window)\n\
                        raw = densityof(window, point)\n\
                        outputs = (score, mass, raw)\n";
    fs::write(&source, format!("{prefix}{restrictions}{observations}")).unwrap();
    fs::write(
        &reference,
        format!("{prefix}window = truncate(law, interval(1.0, 2.5))\n{observations}"),
    )
    .unwrap();

    let rendered = execute(&source, &["optimise"]);
    assert!(rendered.contains(restrictions), "{rendered}");
    assert!(rendered.contains("inputs = (rate, point)"), "{rendered}");
    assert!(rendered.contains("conditioned = normalize(window)"));
    assert!(!rendered.contains("score = logdensityof(conditioned, point)"));
    assert!(!rendered.contains("mass = totalmass(window)"), "{rendered}");
    let source_ir = scratch.path("nested.flatpir");
    let module = flatppl_syntax::parse(&fs::read_to_string(&source).unwrap()).unwrap();
    fs::write(&source_ir, flatppl_flatpir::try_write(&module).unwrap()).unwrap();
    let rendered_ir = execute(&source_ir, &["optimise", "--emit", "flatpir"]);
    let restored = flatppl_flatpir::read(&rendered_ir).unwrap();
    assert_eq!(
        flatppl_syntax::print(&restored),
        flatppl_syntax::print(&flatppl_syntax::parse(&rendered).unwrap())
    );

    let options = ["stablehlo", "--dtype", "f64"];
    assert_eq!(execute(&source, &options), execute(&reference, &options));

    let shared = scratch.path("shared-sample.flatppl");
    let draw = "rng = rnginit(42)\nvalue, next = rand(conditioned, rng)\n";
    fs::write(
        &shared,
        format!("{prefix}{restrictions}{observations}{draw}"),
    )
    .unwrap();
    let shared = execute(&shared, &["optimise"]);
    assert!(shared.ends_with(draw), "{shared}");
    assert!(!shared.contains("mass = totalmass(window)"), "{shared}");

    let controls = scratch.path("controls.flatppl");
    let controls_source = "edge = 1.0\nrate = external(reals)\npoint = external(reals)\n\
        inputs = (edge, rate, point)\nlaw = Exponential(rate)\n\
        left = truncate(law, interval(0.25, 1.0))\n\
        touching = totalmass(truncate(left, interval(1.0, 2.0)))\n\
        disjoint = totalmass(truncate(left, interval(2.0, 3.0)))\n\
        promoted = totalmass(truncate(left, interval(edge, 2.5)))\n\
        rng = rnginit(42)\nvalue, next = rand(normalize(left), rng)\n        outputs = (touching, disjoint, promoted)\n";
    fs::write(&controls, controls_source).unwrap();
    assert_eq!(execute(&controls, &["optimise"]), controls_source);
}
