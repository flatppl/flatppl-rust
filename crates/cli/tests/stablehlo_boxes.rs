use std::{fs, process::Command};

mod common;
use common::Scratch;

const DESCRIBE: &str = r#"{"posterior": {"kind": "measure", "queries": ["builtin_logdensityof"], "gradient": true, "vjp": false, "variate": "cartpow(reals, 2)", "normalized": false}}"#;

fn run(dir: &Scratch, verb: &str, boxes: &[&str]) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_flatppl"));
    command.arg(verb).arg(dir.path("model.flatppl"));
    for value in boxes {
        command.args(["--box", value]);
    }
    command.output().unwrap()
}

#[test]
fn box_flag_binds_an_interface_module() {
    let dir = Scratch::new("stablehlo-boxes");
    fs::write(
        dir.path("iface.flatppl"),
        "posterior = external(anything)\n",
    )
    .unwrap();
    fs::write(dir.path("describe.json"), DESCRIBE).unwrap();
    fs::write(
        dir.path("model.flatppl"),
        "bb = load_module(\"iface.flatppl\")\n\
         theta = elementof(cartpow(reals, 2))\n\
         inputs = theta\n\
         outputs = logdensityof(bb.posterior, theta)\n",
    )
    .unwrap();
    let describe = dir.path("describe.json");
    let binding = format!("bb=stan:{}", describe.display());
    let result = run(&dir, "stablehlo", &[&binding]);
    let out = String::from_utf8_lossy(&result.stdout);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        out.contains("stablehlo.custom_call @flatppl.box(%arg0)"),
        "{out}"
    );
    assert!(
        out.contains("box = \"stan\", member = \"posterior\""),
        "{out}"
    );

    let result = run(&dir, "determinize", &[&binding]);
    let out = String::from_utf8_lossy(&result.stdout);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        out.contains("builtin_logdensityof(bb.posterior, theta)"),
        "{out}"
    );

    for (value, message) in [
        ("bb", "expected ALIAS=BOX:DESCRIBE"),
        ("theta=stan:describe.json", "is not a `load_module` binding"),
        ("bb=1stan:describe.json", "is not an identifier"),
    ] {
        let value = value.replace("describe.json", &describe.display().to_string());
        let result = run(&dir, "stablehlo", &[&value]);
        assert_eq!(result.status.code(), Some(2), "{value}");
        let err = String::from_utf8_lossy(&result.stderr);
        assert!(err.contains(message), "{value}: {err}");
    }
}
