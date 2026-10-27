use std::{fs, process::Command};

mod common;
use common::Scratch;

fn emit(input: &std::path::Path, mode: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .arg("stablehlo")
        .arg(input)
        .args(["--mode", mode, "--restrict-enzyme-compatible=false"])
        .output()
        .unwrap()
}

#[test]
fn runtime_seed_is_not_replaced_by_a_key_argument() {
    let dir = Scratch::new("rnginit-parametric");
    let input = dir.path("model.flatppl");
    fs::write(&input, "seed = elementof(nonnegintegers)\ns = rnginit(seed)\nx, next = rand(Normal(0,1), s)\ninputs = seed\noutputs = (x, next)\n").unwrap();
    let result = emit(&input, "sample");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let out = String::from_utf8_lossy(&result.stdout);
    assert!(out.contains("@sample(%arg0: tensor<i32>)"), "{out}");
    assert!(
        out.contains("stablehlo.convert %") && out.contains("stablehlo.rng_bit_generator"),
        "{out}"
    );
    assert!(!out.contains("%key:"), "{out}");
    fs::write(&input, "seed = elementof(nonnegintegers)\ns = rnginit(seed)\nx, next = rand(Normal(0,1), s)\ninputs = s\noutputs = (x, next)\n").unwrap();
    let result = emit(&input, "sample");
    assert_eq!(result.status.code(), Some(3));
    assert!(
        String::from_utf8_lossy(&result.stderr)
            .contains("neither an elementof parameter nor a fixed-phase")
    );
    fs::write(&input, "seed = external(nonnegintegers)\ns = rnginit(seed)\nx, next = rand(Normal(0,1), s)\ninputs = seed\noutputs = (x, next)\n").unwrap();
    let result = emit(&input, "sample");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let out = String::from_utf8_lossy(&result.stdout);
    assert!(
        out.contains("@sample(%arg0: tensor<i32>)") && !out.contains("%key:"),
        "{out}"
    );
    fs::write(
        &input,
        "s = rnginit(23)\nx, next = rand(Normal(0,1), s)\ninputs = s\noutputs = (x, next)\n",
    )
    .unwrap();
    let result = emit(&input, "sample");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("@sample(%key: tensor<2xui64>)"));
    fs::write(&input, "s = rnginit(23)\nalias = s\nx, next = rand(Normal(0,1), alias)\ninputs = s\noutputs = (x, next)\n").unwrap();
    let result = emit(&input, "sample");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("@sample(%key: tensor<2xui64>)"));
    fs::write(&input, "s = rnginit(23)\nalias = s\nx, next = rand(Normal(0,1), s)\ninputs = alias\noutputs = (x, next)\n").unwrap();
    let result = emit(&input, "sample");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("@sample(%key: tensor<2xui64>)"));
    fs::write(&input, "seed = elementof(nonnegintegers)\nother = elementof(reals)\ns = rnginit(seed)\nx, next = rand(Normal(0,1), s)\ninputs = other\noutputs = (x, next)\n").unwrap();
    let result = emit(&input, "sample");
    assert_eq!(result.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&result.stderr).contains("not listed in `inputs`"));
}

#[test]
fn seed_forms_emit_keys_and_rngstate_refuses_explicitly() {
    let dir = Scratch::new("rnginit-forms");
    let input = dir.path("model.flatppl");
    for (decl, expr) in [
        ("elementof(integers)", "rnginit(rngseed = seed)"),
        ("elementof(cartpow(integers, [8]))", "rnginit(seed)"),
    ] {
        fs::write(
            &input,
            format!("seed = {decl}\ninputs = seed\noutputs = {expr}\n"),
        )
        .unwrap();
        let result = emit(&input, "logdensity");
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("tensor<2xui64>"));
    }
    fs::write(
        &input,
        "b = elementof(cartpow(integers, [16]))\ninputs = b\noutputs = rngstate(b)\n",
    )
    .unwrap();
    let result = emit(&input, "logdensity");
    assert_eq!(result.status.code(), Some(3));
    assert!(
        String::from_utf8_lossy(&result.stderr)
            .contains("rngstate byte deserialization is not supported")
    );
}
