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
    let source = |options: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_flatppl"))
            .arg("autodiff")
            .arg(&model)
            .args(options)
            .output()
            .unwrap()
    };
    assert_eq!(source(&["--wrt", "a"]).status.code(), Some(3));
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
        let printed = source(&[
            "--numerical-integrals",
            "--mode",
            mode,
            "--wrt",
            "a",
            "--wrt",
            "lo",
            "--wrt",
            "hi",
            "--wrt",
            "cut",
        ]);
        assert!(
            printed.status.success(),
            "{mode}: {}",
            String::from_utf8_lossy(&printed.stderr)
        );
        let path = dir.path(&format!("{mode}.flatppl"));
        std::fs::write(&path, printed.stdout).unwrap();
        let compiled = Command::new(env!("CARGO_BIN_EXE_flatppl"))
            .arg("stablehlo")
            .arg(path)
            .args(["--numerical-integrals", "--dtype", "f64"])
            .output()
            .unwrap();
        assert!(
            compiled.status.success(),
            "{mode}: {}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        let hlo = String::from_utf8(compiled.stdout).unwrap();
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

#[test]
fn list_rules_prints_each_key_once_with_guards_expanded() {
    let out = Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args(["autodiff", "--list-rules"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<_> = text.lines().collect();
    let mut sorted = lines.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(lines.len(), sorted.len(), "duplicate keys:\n{text}");
    for key in [
        "Pullback\tsqrt\t1\t0",
        "Pushforward\tsoftmax\t1\t0",
        "Pullback\tvector\t*\t*",
        "Pullback\tkron\t2\t*",
        // The scalar-predictor guard expands to one key per family.
        "Pullback\tNormal GLM log density\t5\t0",
        "Pushforward\tStudentT GLM log density\t6\t2",
    ] {
        assert!(lines.contains(&key), "missing {key:?}:\n{text}");
    }
    // Neither the unguarded fallback nor an unexpanded guard is a key.
    assert!(
        lines
            .iter()
            .all(|line| line.split('\t').nth(1) != Some("*")),
        "{text}"
    );
}

#[test]
fn explain_rules_reports_the_requested_keys_on_stderr() {
    let dir = common::Scratch::new("autodiff-explain-rules");
    let model = dir.path("model.flatppl");
    std::fs::write(
        &model,
        "x = elementof(reals)\nv = elementof(cartpow(reals, 3))\ninputs = (x, v)\n\
         outputs = x * x + sqrt(x) + sum(softmax(v))\n",
    )
    .unwrap();
    let run = |mode: &str| {
        let out = Command::new(env!("CARGO_BIN_EXE_flatppl"))
            .arg("autodiff")
            .arg(&model)
            .args([
                "--wrt",
                "x",
                "--wrt",
                "v",
                "--explain-rules",
                "--mode",
                mode,
            ])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(String::from_utf8(out.stdout).unwrap().contains("d_"));
        String::from_utf8(out.stderr).unwrap()
    };
    let reverse = run("gradient");
    for key in [
        "rule\tPullback\tmul\t2\t0",
        "rule\tPullback\tmul\t2\t1",
        "rule\tPullback\tsqrt\t1\t0",
        "rule\tPullback\tsoftmax\t1\t0",
    ] {
        assert!(
            reverse.lines().any(|line| line == key),
            "missing {key:?}:\n{reverse}"
        );
    }
    let forward = run("jvp");
    assert!(
        forward
            .lines()
            .any(|line| line == "rule\tPushforward\tsoftmax\t1\t0"),
        "{forward}"
    );
    assert!(
        forward
            .lines()
            .any(|line| line == "rule\tPullback\tsqrt\t1\t0"),
        "{forward}"
    );
}
