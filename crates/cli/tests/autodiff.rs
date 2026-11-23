#![cfg(feature = "autodiff")]

mod common;

use std::process::Command;

#[test]
fn autodiff_emits_flatppl_only_when_requested() {
    let dir = common::Scratch::new("autodiff");
    let model = dir.path("model.flatppl");
    std::fs::write(
        &model,
        "x = elementof(reals)\noffset = 2.0 * 3.0\ninputs = x\noutputs = x * x + offset\n",
    )
    .unwrap();
    let run = |verb| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_flatppl"));
        command.arg(verb).arg(&model);
        if verb == "autodiff" {
            command.args(["--wrt", "x"]);
        }
        let out = command.output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    };

    let derivative = run("autodiff");
    assert!(derivative.contains("offset = 2.0 * 3.0"), "{derivative}");
    assert!(
        derivative.contains("d_x = 1.0 * x + 1.0 * x"),
        "{derivative}"
    );
    assert!(flatppl_syntax::parse(&derivative).is_ok(), "{derivative}");
    let plain = run("determinize");
    assert!(plain.contains("offset = 6.0"), "{plain}");
    assert!(!plain.contains("d_x"), "{plain}");
}

#[cfg(feature = "stablehlo")]
#[test]
fn numerical_ad_accepts_moving_bounds_and_cutoffs_under_explicit_opt_in() {
    let dir = common::Scratch::new("numerical-autodiff");
    let model = dir.path("model.flatppl");
    std::fs::write(
        &model,
        "
a = elementof(reals)
lo = elementof(reals)
hi = elementof(reals)
cut = elementof(reals)
weight(t, p) = exp(p.a * t) * ifelse(t < p.cut, 1.0, 2.0)
measure = normalize(weighted(weight, Lebesgue(interval(lo, hi)), record(a = a, cut = cut)))
inputs = (a, lo, hi, cut)
outputs = logdensityof(measure, 0.45)
",
    )
    .unwrap();
    let run = |options: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_flatppl"))
            .arg("stablehlo")
            .arg(&model)
            .args(options)
            .output()
            .unwrap()
    };
    let disabled = run(&["--ad-wrt", "a"]);
    assert_eq!(disabled.status.code(), Some(3));
    for mode in ["gradient", "jvp", "vjp"] {
        let result = run(&[
            "--numerical-integrals",
            "--dtype",
            "f64",
            "--ad-mode",
            mode,
            "--ad-wrt",
            "a",
            "--ad-wrt",
            "lo",
            "--ad-wrt",
            "hi",
            "--ad-wrt",
            "cut",
        ]);
        assert!(
            result.status.success(),
            "{mode}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let hlo = String::from_utf8(result.stdout).unwrap();
        assert!(hlo.contains("stablehlo.while"));
    }
}

#[test]
fn autodiff_accepts_an_explicit_empty_selection() {
    let dir = common::Scratch::new("autodiff-empty");
    let model = dir.path("model.flatppl");
    std::fs::write(&model, "primal = 2.0\noutputs = primal\n").unwrap();
    let run = |options: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_flatppl"))
            .arg("autodiff")
            .arg(&model)
            .args(options)
            .output()
            .unwrap()
    };
    assert_eq!(run(&[]).status.code(), Some(2));
    assert_eq!(run(&["--empty-wrt", "--wrt", "x"]).status.code(), Some(2));
    let explicit = run(&["--empty-wrt"]);
    assert!(
        explicit.status.success(),
        "{}",
        String::from_utf8_lossy(&explicit.stderr)
    );
    let output = String::from_utf8(explicit.stdout).unwrap();
    assert!(output.contains("outputs = primal"), "{output}");
}

#[cfg(feature = "stablehlo")]
#[test]
fn probability_scan_vjp_emits_a_guarded_reverse_loop_with_the_seed_abi() {
    let model = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/probability_scan/masked.flatppl"
    );
    let output = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args([
            "stablehlo",
            model,
            "--dtype",
            "f64",
            "--ad-mode",
            "vjp",
            "--ad-wrt",
            "q",
            "--ad-wrt",
            "obs",
            "--ad-wrt",
            "high",
            "--ad-wrt",
            "low",
            "--ad-wrt",
            "unused",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let hlo = String::from_utf8(output.stdout).unwrap();
    let signature = hlo.lines().find(|line| line.contains("func.func")).unwrap();
    assert!(
        signature.ends_with(
            "%arg6: tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<65xf64>, tensor<f64>, tensor<f64>, tensor<f64>) {"
        ),
        "{signature}"
    );
    assert!(
        hlo.split("stablehlo.while").skip(1).any(|region| {
            let body = region.split_once("} do {").unwrap().1;
            let body = body.split_once("stablehlo.return").unwrap().0;
            body.contains("stablehlo.compare EQ") && body.contains("stablehlo.select")
        }),
        "{hlo}"
    );
}
