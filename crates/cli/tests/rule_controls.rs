#![cfg(any(
    feature = "optimise",
    feature = "determinize",
    feature = "stablehlo",
    feature = "stan"
))]

mod common;

use common::Scratch;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn run(args: &[&str], paths: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_flatppl"))
        .args(args)
        .args(paths)
        .output()
        .expect("run flatppl")
}

fn success(output: Output) -> String {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("FlatPPL output is UTF-8")
}

fn model(scratch: &Scratch, name: &str, source: &str) -> PathBuf {
    let path = scratch.path(name);
    fs::write(&path, source).expect("write test model");
    path
}

#[cfg(feature = "optimise")]
#[test]
fn a_didactic_rule_mask_keeps_double_negation_and_simplifies_its_sibling() {
    let source = r#"detected = external(booleans)
reading = external(reals)
confirmed = lnot(lnot(detected))  # Show both negations in the lesson.
reported = identity(reading)
"#;
    let scratch = Scratch::new("didactic-rule-mask");
    let lesson = model(&scratch, "lesson.flatppl", source);
    let masked = success(run(
        &[
            "optimise",
            "--disable-egglog",
            "rule:portable.double-negation",
        ],
        &[&lesson],
    ));
    let expected = source.replace("identity(reading)", "reading");
    assert_eq!(masked, expected);
    let default = success(run(&["optimise"], &[&lesson]));
    assert_eq!(
        default,
        expected.replace("lnot(lnot(detected))", "detected")
    );
    for selection in ["ruleset:portable", "all"] {
        assert_eq!(
            success(run(
                &["optimise", "--disable-egglog", selection],
                &[&lesson]
            )),
            source
        );
    }
    assert_eq!(
        success(run(
            &[
                "optimise",
                "--disable-egglog",
                "rule:portable.double-negation",
                "--disable-egglog",
                "rule:portable.identity",
            ],
            &[&lesson],
        )),
        source
    );

    let parsed = flatppl_syntax::parse(source).expect("parse lesson");
    let input_ir = model(
        &scratch,
        "lesson.flatpir",
        &flatppl_flatpir::try_write(&parsed).expect("write lesson IR"),
    );
    let output_ir = success(run(
        &[
            "optimise",
            "--emit",
            "flatpir",
            "--disable-egglog",
            "rule:portable.double-negation",
        ],
        &[&input_ir],
    ));
    let ir = flatppl_flatpir::read(&output_ir).expect("parse optimized IR");
    let expected = flatppl_syntax::parse(&masked).expect("parse optimized source");
    assert_eq!(flatppl_syntax::print(&ir), flatppl_syntax::print(&expected));

    #[cfg(all(feature = "convert", feature = "mathdoc"))]
    {
        let masked_model = model(&scratch, "masked.flatppl", &masked);
        let default_model = model(&scratch, "default.flatppl", &default);
        let shows_double_negation = |source: &Path, name| {
            let document = scratch.path(name);
            success(run(&["convert", "--no-header"], &[source, &document]));
            fs::read_to_string(document)
                .expect("read the generated lesson")
                .lines()
                .any(|line| line.matches(r"\neg").count() == 2)
        };
        assert!(shows_double_negation(&masked_model, "masked.md"));
        assert!(!shows_double_negation(&default_model, "default.md"));
    }
}

#[cfg(feature = "optimise")]
#[test]
fn disabling_all_rules_also_keeps_cell_reads() {
    let source = r#"reading = external(reals)
_settings = record(scale = 2.0, unused = exp(3.0))
scaled = reading * get(_settings, "scale")
"#;
    let scratch = Scratch::new("demand-prune-mask");
    let lesson = model(&scratch, "lesson.flatppl", source);
    assert_eq!(
        success(run(&["optimise"], &[&lesson])),
        source.replace(r#"get(_settings, "scale")"#, "2.0")
    );
    for selection in ["pass:demand-prune", "all"] {
        assert_eq!(
            success(run(
                &["optimise", "--disable-egglog", selection],
                &[&lesson]
            )),
            source,
            "{selection}"
        );
    }
}

#[test]
fn rule_names_are_discoverable_and_invalid_names_are_rejected() {
    let catalogue = success(run(&["egglog-rules"], &[]));
    for selector in [
        "all",
        "rule:portable.double-negation",
        "ruleset:portable",
        "ruleset:glm",
        "pass:demand-prune",
        "stage:lowered",
        "stage:source",
    ] {
        assert!(
            catalogue.split_whitespace().any(|word| word == selector),
            "missing {selector} in:\n{catalogue}"
        );
    }

    let scratch = Scratch::new("unknown-egglog-rule");
    let input = model(
        &scratch,
        "model.flatppl",
        "x = elementof(reals)\ninputs = x\noutputs = logdensityof(Normal(0, 1), x)\n",
    );
    for verb in [
        #[cfg(feature = "optimise")]
        "optimise",
        #[cfg(feature = "determinize")]
        "determinize",
        #[cfg(feature = "stablehlo")]
        "stablehlo",
        #[cfg(feature = "stan")]
        "stan",
    ] {
        for selector in [
            "rule:not-a-flatppl-rule",
            "ruleset:not-a-flatppl-ruleset",
            "pass:not-a-flatppl-pass",
            "stage:not-a-flatppl-stage",
        ] {
            let output = run(&[verb, "--disable-egglog", selector], &[&input]);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(!output.status.success(), "{verb} accepted {selector}");
            assert!(stderr.contains(selector), "{verb}: {stderr}");
            assert!(output.stdout.is_empty(), "{verb} emitted a result");
        }
    }
}

#[cfg(any(feature = "determinize", feature = "stablehlo"))]
#[test]
fn portable_rule_masks_reach_the_measure_lowering_front_end() {
    let prefix = "rate = external(reals)\ninputs = rate\noutputs = totalmass(";
    let scratch = Scratch::new("measure-lowering-rule-mask");
    let nested = model(
        &scratch,
        "nested.flatppl",
        &format!(
            "{prefix}truncate(truncate(Exponential(rate), interval(0.25, 3.0)), interval(1.0, 5.0)))\n"
        ),
    );
    let intersection = model(
        &scratch,
        "intersection.flatppl",
        &format!("{prefix}truncate(Exponential(rate), interval(1.0, 3.0)))\n"),
    );
    for verb in [
        #[cfg(feature = "determinize")]
        "determinize",
        #[cfg(feature = "stablehlo")]
        "stablehlo",
    ] {
        let lowered = success(run(&[verb], &[&nested]));
        assert_eq!(lowered, success(run(&[verb], &[&intersection])), "{verb}");
        assert_eq!(
            lowered,
            success(run(
                &[verb, "--disable-egglog", "ruleset:scan-demand"],
                &[&nested]
            )),
            "{verb}"
        );
        for selector in ["ruleset:measure-algebra", "all"] {
            let masked = run(&[verb, "--disable-egglog", selector], &[&nested]);
            assert_eq!(masked.status.code(), Some(3), "{verb}: {masked:?}");
            assert!(
                String::from_utf8_lossy(&masked.stderr).contains("totalmass"),
                "{verb}: {masked:?}"
            );
        }
    }
}

#[cfg(feature = "stablehlo")]
#[test]
fn stablehlo_rule_masks_reach_the_cpu_scan_demand_planner() {
    let source = r#"seed = external(cartpow(reals, 2))
xs = external(cartpow(reals, 5))
terminal = scan((s, x) -> s .* x, seed, xs)
step(s, x) = record(wanted = s.wanted + x, hidden = s.hidden * x)
histories = scan(step, record(wanted = 0.0, hidden = 0.5), xs)
inputs = (seed, xs)
outputs = (terminal[5], histories.wanted)
"#;
    let scratch = Scratch::new("scan-demand-rule-mask");
    let query = model(&scratch, "query.flatppl", source);
    let emit = |target, selector| {
        let mut args = vec!["stablehlo", "--target", target, "--dtype", "f64"];
        if let Some(selector) = selector {
            args.extend(["--disable-egglog", selector]);
        }
        success(run(&args, &[&query]))
    };
    let regular = emit("cpu", None);
    let terminal_kept = emit("cpu", Some("rule:scan-demand.terminal-choice"));
    let all_kept = emit("cpu", Some("ruleset:scan-demand"));

    for (ir, history_type) in [
        (&regular, "tensor<1x2xf64>"),
        (&terminal_kept, "tensor<5x2xf64>"),
        (&all_kept, "tensor<5x2xf64>"),
    ] {
        let terminal_scan = ir
            .lines()
            .find(|line| line.contains("stablehlo.while"))
            .expect("terminal scan loop");
        assert!(terminal_scan.ends_with(history_type), "{terminal_scan}");
    }
    assert_eq!(regular.matches("stablehlo.dynamic_update_slice").count(), 2);
    assert_eq!(
        terminal_kept
            .matches("stablehlo.dynamic_update_slice")
            .count(),
        2
    );
    assert_eq!(
        all_kept.matches("stablehlo.dynamic_update_slice").count(),
        3
    );
    let signature = |ir: &str| {
        ir.lines()
            .find(|line| line.contains("func.func"))
            .expect("query signature")
            .to_owned()
    };
    assert_eq!(signature(&regular), signature(&terminal_kept));
    assert_eq!(signature(&regular), signature(&all_kept));

    let gpu = emit("gpu", None);
    assert_eq!(gpu, emit("gpu", Some("ruleset:scan-demand")));
    assert_eq!(regular, emit("cpu", None));
}

#[cfg(feature = "stan")]
#[test]
fn the_stan_profile_can_keep_an_explicit_gram_product() {
    let source = r#"theta = elementof(reals)
a = rowstack([[theta, 2.0], [3.0, 4.0], [5.0, 6.0]])
gram = a * transpose(a)
inputs = theta
outputs = logdensityof(Normal(trace(gram), 1), theta)
"#;
    let scratch = Scratch::new("stan-gram-rule-mask");
    let query = model(&scratch, "query.flatppl", source);
    let regular = success(run(&["stan", "--target", "outputs"], &[&query]));
    let masked = success(run(
        &[
            "stan",
            "--target",
            "outputs",
            "--disable-egglog",
            "ruleset:stan",
        ],
        &[&query],
    ));
    assert!(regular.contains("gram = tcrossprod(a);"), "{regular}");
    assert!(masked.contains("gram = a * a';"), "{masked}");
    assert_eq!(regular.replace("tcrossprod(a)", "a * a'"), masked);
}
