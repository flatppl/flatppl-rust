//! CLI-level coverage for `flatppl stablehlo`'s `inputs`/`outputs` ABI
//! recognition (PR-1, design doc
//! `crates/stablehlo/docs/inputs-outputs-abi.md`):
//! `stablehlo_cmd` roots on the declared `inputs`/`outputs` binding names
//! when present (no deprecation warning), and falls back to the legacy
//! last-public-binding convention — WITH a one-line deprecation warning on
//! stderr — when neither reserved binding exists.

use std::process::Command;

fn flatppl() -> Command {
    Command::new(env!("CARGO_BIN_EXE_flatppl"))
}

fn write_model(name: &str, src: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "flatppl-stablehlo-cli-{name}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("m.flatppl");
    std::fs::write(&input, src).unwrap();
    input
}

#[test]
fn integer_broadcast_accepts_rounded_cells() {
    for expression in [
        "integer.(round.(v))",
        "integer.(floor.(v))",
        "integer.(ceil.(v))",
        "sum(integer.(round.(v)))",
    ] {
        let input = write_model(
            "rounded-cells",
            &format!("v = elementof(cartpow(reals, [3]))\ninputs = v\noutputs = {expression}\n"),
        );
        let output = flatppl().arg("stablehlo").arg(input).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let input = write_model(
        "unrounded-cells",
        "v = elementof(cartpow(reals, [3]))\ninputs = v\noutputs = integer.(abs.(v))\n",
    );
    assert!(
        !flatppl()
            .arg("stablehlo")
            .arg(input)
            .output()
            .unwrap()
            .status
            .success()
    );
}

#[test]
fn syntax_sugar_exports_the_same_ordered_query() {
    let sugar = write_model(
        "syntax-sugar",
        "a = elementof(reals)\nb = elementof(reals)\nf = functionof(a + 2.0 * b, a, b)\ny ~ Normal(a + 2.0 * b, 2.0)\nk = kernelof((y = y), a, b)\nL = likelihoodof(k, (y = 15.0))\nlp = logdensityof(L, (a = a, b = b))\ninputs = (a, b)\noutputs = (f(a, b), lp)",
    );
    let explicit = write_model(
        "syntax-explicit",
        "a = elementof(reals)\nb = elementof(reals)\nf = functionof(a + 2.0 * b, a = a, b = b)\ny ~ Normal(a + 2.0 * b, 2.0)\nk = kernelof(record(y = y), a = a, b = b)\nL = likelihoodof(k, record(y = 15.0))\nlp = logdensityof(L, record(a = a, b = b))\ninputs = (a, b)\noutputs = (f(a, b), lp)",
    );
    let emit = |input| {
        let output = flatppl()
            .args(["stablehlo", "--dtype", "f64"])
            .arg(input)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        output.stdout
    };
    assert_eq!(emit(sugar), emit(explicit));
}

/// Spec §04 "Calling conventions" and §11: a sole tuple splats into positional
/// arguments, each a projection of the original, so a tuple spelling exports the
/// same StableHLO as the positional spelling.
#[test]
fn a_splatted_tuple_exports_the_same_module_as_positional_arguments() {
    let tuple = write_model(
        "tuple-splat",
        "u = elementof(reals)\nv = elementof(posreals)\nf(a, b) = a * 2.0 + b\nt = (u, v)\na = elementof(reals)\nb = elementof(posreals)\nK = functionof(Normal(mu = a, sigma = b), a = a, b = b)\nlp = logdensityof(K(t), 0.3)\ninputs = (u, v)\noutputs = (f(t), add((u, 2.0)), lp)",
    );
    let positional = write_model(
        "tuple-positional",
        "u = elementof(reals)\nv = elementof(posreals)\nf(a, b) = a * 2.0 + b\na = elementof(reals)\nb = elementof(posreals)\nK = functionof(Normal(mu = a, sigma = b), a = a, b = b)\nlp = logdensityof(K(a = u, b = v), 0.3)\ninputs = (u, v)\noutputs = (f(u, v), add(u, 2.0), lp)",
    );
    let emit = |input| {
        let output = flatppl()
            .args(["stablehlo", "--dtype", "f64"])
            .arg(input)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        output.stdout
    };
    assert_eq!(emit(tuple), emit(positional));
}

#[test]
fn fixed_json_in_an_import_matches_explicit_literals() {
    let input = write_model(
        "fixed-json",
        "m = load_module(\"nested/model.flatppl\")\nx = elementof(reals)\ninputs = x\noutputs = m.data.x .* x",
    );
    let dir = input.parent().unwrap();
    std::fs::create_dir_all(dir.join("nested")).unwrap();
    std::fs::write(
        dir.join("nested/model.flatppl"),
        "data = load_data(\"data.json\", cartprod(x = cartpow(reals, 2)))",
    )
    .unwrap();
    std::fs::write(
        dir.join("nested/data.json"),
        r#"{"x":[1.5,3.5,99],"unused":"ignored"}"#,
    )
    .unwrap();
    let expected = write_model(
        "fixed-json-literal",
        "x = elementof(reals)\ninputs = x\noutputs = [1.5, 3.5] .* x",
    );
    let actual = flatppl().arg("stablehlo").arg(&input).output().unwrap();
    assert!(
        actual.status.success(),
        "{}",
        String::from_utf8_lossy(&actual.stderr)
    );
    let literal = flatppl().arg("stablehlo").arg(expected).output().unwrap();
    assert!(
        literal.status.success(),
        "{}",
        String::from_utf8_lossy(&literal.stderr)
    );
    assert_eq!(actual.stdout, literal.stdout);
}

/// A model with no `inputs`/`outputs` bindings is refused (exit 3): the
/// last-public-binding query heuristic has been removed, so the ABI must be
/// declared explicitly — there is no fallback.
#[test]
fn stablehlo_model_without_abi_refuses_with_exit_3() {
    let input = write_model(
        "no-abi",
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         lp = logdensityof(lawof(record(a = a)), record(a = 0.5))\n",
    );
    let out = flatppl().arg("stablehlo").arg(&input).output().unwrap();
    assert_eq!(
        out.status.code(),
        Some(3),
        "expected exit 3 (refuse); stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no inputs/outputs ABI declared"),
        "expected the ABI-required refusal on stderr, got:\n{stderr}"
    );
}

/// The ordered ABI uses f32 by default and f64 when requested.
#[test]
fn stablehlo_abi_model_preserves_precision_and_order() {
    let input = write_model(
        "abi",
        "a = elementof(reals)\n\
         b = elementof(reals)\n\
         dead_helper = a * 2.0\n\
         m = lawof(record(a = draw(Normal(mu = 0.0, sigma = 1.0)), b = draw(Normal(mu = 0.0, sigma = 1.0))))\n\
         q1 = logdensityof(m, record(a = a, b = b))\n\
         q2 = logdensityof(m, record(a = a, b = b))\n\
         inputs = (a, b)\n\
         outputs = (q1, q2)\n",
    );
    for (args, dtype) in [(vec![], "f32"), (vec!["--dtype", "f64"], "f64")] {
        let out = flatppl()
            .arg("stablehlo")
            .arg(&input)
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.contains(&format!(
                "func.func @logdensity(%arg0: tensor<{dtype}>, %arg1: tensor<{dtype}>) -> (tensor<{dtype}>, tensor<{dtype}>)"
            )),
            "expected the ordered ABI signature:\n{stdout}"
        );
    }
}

/// Both lowering profiles keep the ABI. An unknown profile is a usage error.
#[test]
fn stablehlo_target_selects_a_profile_with_the_same_abi() {
    let input = write_model(
        "target",
        "a = elementof(reals)\n\
         m = lawof(record(a = draw(Normal(mu = 0.0, sigma = 1.0))))\n\
         inputs = a\n\
         outputs = logdensityof(m, record(a = a))\n",
    );
    for target in ["cpu", "gpu"] {
        let out = flatppl()
            .args(["stablehlo", "--target", target])
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.contains("func.func @logdensity(%arg0: tensor<f32>) -> tensor<f32>"),
            "{stdout}"
        );
    }
    let out = flatppl()
        .args(["stablehlo", "--target", "tpu"])
        .arg(&input)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
}

/// `elementof(ordered(n))` and `elementof(zerosum(n))` are real n-vector inputs
/// (§03): the emitter shapes them like `stdsimplex(n)` and checks no constraint.
#[test]
fn stablehlo_ordered_and_zerosum_parameters_are_real_vector_inputs() {
    let input = write_model(
        "abi-ordered-zerosum",
        "c = elementof(ordered(3))\n\
         z = elementof(zerosum(4))\n\
         a = draw(Normal(mu = sum(c) + sum(z), sigma = 1.0))\n\
         lp = logdensityof(lawof(record(a = a)), record(a = 0.5))\n\
         inputs = (c, z)\n\
         outputs = lp\n",
    );
    let out = flatppl().arg("stablehlo").arg(&input).output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(
            "func.func @logdensity(%arg0: tensor<3xf32>, %arg1: tensor<4xf32>) -> tensor<f32>"
        ),
        "{stdout}"
    );
}

/// A `load_data` ABI input is shaped from its declared `valueset` end-to-end
/// through the real CLI binary, and THE SOURCE FILE NEED NOT EXIST: nothing in
/// the pipeline opens it (spec §07 `load_data`: "`valueset` fully determines the
/// result's shape"; §13 `sec:determinization-signature`: "function argument
/// (shape from its `valueset`, contents at runtime)"). `data.csv` is
/// deliberately never written — the compile-time row-count read this replaces
/// would have failed here.
#[test]
fn stablehlo_abi_load_data_shape_comes_from_valueset_without_reading_the_source() {
    let input = write_model(
        "abi-load-data-valueset",
        "a = elementof(reals)\n\
         y = load_data(\"data.csv\", cartpow(reals, 4))\n\
         m = lawof(record(a = draw(Normal(mu = 0.0, sigma = 1.0))))\n\
         q1 = logdensityof(m, record(a = a))\n\
         inputs = (a, y)\n\
         outputs = q1\n",
    );
    assert!(
        !input.with_file_name("data.csv").exists(),
        "the test's premise is that the source is absent"
    );
    let out = flatppl().arg("stablehlo").arg(&input).output().unwrap();
    assert!(
        out.status.success(),
        "emission must not depend on the source file; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(
            "func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<4xf32>) -> tensor<f32>"
        ),
        "expected `y` typed `tensor<4xf32>` from `cartpow(reals, 4)`:\n{stdout}"
    );
    assert!(
        !stdout.contains("tensor<?x"),
        "a valueset-shaped load_data arg must not carry a dynamic `?` dim:\n{stdout}"
    );
}

/// One CSV holding several columns: a table-valueset `load_data` input becomes
/// one argument per column in declared order, end-to-end through the CLI. `data`
/// is a single `inputs` entry and contributes `%arg3` (x) and `%arg4` (y).
#[test]
fn stablehlo_abi_load_data_table_destructures_into_column_args() {
    let input = write_model(
        "abi-load-data-table",
        "alpha = elementof(reals)\n\
         beta = elementof(reals)\n\
         sigma = elementof(posreals)\n\
         data = load_data(\"data.csv\", cartpow(cartprod(x = reals, y = reals), 4))\n\
         means = alpha .+ beta .* data.x\n\
         y ~ Normal.(means, sigma)\n\
         k = kernelof(record(y = y), alpha = alpha, beta = beta, sigma = sigma)\n\
         L = likelihoodof(k, record(y = data.y))\n\
         lp = logdensityof(L, record(alpha = alpha, beta = beta, sigma = sigma))\n\
         inputs = (alpha, beta, sigma, data)\n\
         outputs = (lp)\n",
    );
    let out = flatppl().arg("stablehlo").arg(&input).output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(
            "func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>, \
             %arg2: tensor<f32>, %arg3: tensor<4xf32>, %arg4: tensor<4xf32>) -> (tensor<f32>, tensor<i32>)"
        ),
        "expected `data` destructured into two column args:\n{stdout}"
    );
}

/// `b` is reachable from `q1` (root-DCE keeps it — the query needs it) but is
/// not listed in `inputs` (which declares only `a`): the exhaustiveness check
/// (design doc: `inputs` is "authoritative and exhaustive") must refuse this
/// end-to-end through the real CLI binary, at exit 3.
#[test]
fn stablehlo_abi_model_refuses_non_exhaustive_inputs_with_exit_3() {
    let input = write_model(
        "abi-nonexhaustive",
        "a = elementof(reals)\n\
         b = elementof(reals)\n\
         m = lawof(record(a = draw(Normal(mu = 0.0, sigma = 1.0)), b = draw(Normal(mu = 0.0, sigma = 1.0))))\n\
         q1 = logdensityof(m, record(a = a, b = b))\n\
         inputs = a\n\
         outputs = q1\n",
    );
    let out = flatppl().arg("stablehlo").arg(&input).output().unwrap();
    assert_eq!(
        out.status.code(),
        Some(3),
        "expected exit 3 (refuse); stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not listed in `inputs`"),
        "expected the exhaustiveness refusal message, got:\n{stderr}"
    );
}

/// A source-controlled vector length must not allocate or emit billions of
/// component samplers. All components share one rejection loop, so the
/// module size does not grow with the length.
#[test]
fn stablehlo_dirichlet_sample_size_is_independent_of_length() {
    let input = write_model(
        "dirichlet-large-length",
        "alpha = elementof(cartpow(posreals, 4294967295))\n\
         s = rnginit(0)\n\
         x = draw(Dirichlet(alpha = alpha))\n\
         draws = rand(lawof(x), s)\n\
         inputs = (alpha)\n\
         outputs = (draws)\n",
    );
    let out = flatppl()
        .arg("stablehlo")
        .arg(&input)
        .args(["--mode", "sample", "--restrict-enzyme-compatible=false"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let module = String::from_utf8_lossy(&out.stdout);
    assert_eq!(module.matches("stablehlo.while").count(), 1, "{module}");
    assert!(module.lines().count() < 200, "{module}");
}

/// Static shape products must not wrap in moment denominators. Both whole-array
/// and aggregate reductions share the checked element-count path.
#[test]
fn stablehlo_moments_reject_shape_product_overflow() {
    let cases = [
        ("mean", "y = mean(x)"),
        ("var", "y = var(x)"),
        ("std", "y = std(x)"),
        (
            "aggregate-mean",
            "y = aggregate(mean, [.l], x[.i, .j, .k, .l])",
        ),
        (
            "aggregate-var",
            "y = aggregate(var, [.l], x[.i, .j, .k, .l])",
        ),
        (
            "aggregate-std",
            "y = aggregate(std, [.l], x[.i, .j, .k, .l])",
        ),
    ];
    for (name, expr) in cases {
        let input = write_model(
            name,
            &format!(
                "x = elementof(cartpow(reals, [4294967295, 4294967295, 2, 1]))\n\
                 {expr}\n\
                 inputs = (x)\n\
                 outputs = (y)\n"
            ),
        );
        let out = flatppl().arg("stablehlo").arg(&input).output().unwrap();
        assert_eq!(
            out.status.code(),
            Some(3),
            "{name}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("element count exceeds u64"),
            "{name}: {stderr}"
        );
        assert!(out.stdout.is_empty(), "{name}: refusal emitted partial IR");
    }
}

const TERMINAL_SCAN: &str = r#"
xs = external(cartpow(reals, 5))
seed = external(cartpow(reals, 2))
gain = 0.5
step(state, x, a) = tanh.(a .* state .+ x)
history = scan(step, seed, xs, gain)
alias = history
inputs = (seed, xs, gain)
"#;

const RECORD_SCAN: &str = r#"
xs = external(cartpow(reals, 5))
seed = external(reals)
gain = 0.5
step(state, x, a) = record(
    mean = a * state.mean + state.energy + x,
    energy = 0.25 * state.energy + x * x,
)
history = scan(step, record(mean = seed, energy = 0.5), xs, gain)
read_mean(state) = state.mean
alias = history
inputs = (xs, gain, seed)
"#;

const CHECKED_SCAN: &str = r#"
xs = external(cartpow(reals, 5))
seed = external(reals)
step(state, x) = record(
    mean = state.mean + x,
    scored = logdensityof(Normal(0.0, x), 0.25),
)
history = scan(step, record(mean = seed, scored = 0.0), xs)
inputs = (xs, seed)
outputs = history.mean
"#;

fn emit_scan(name: &str, source: &str) -> String {
    let input = write_model(name, source);
    let output = flatppl()
        .args(["stablehlo", "--target", "cpu", "--dtype", "f64"])
        .arg(input)
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
fn terminal_scan_selectors_keep_one_history_row() {
    for length in [1, 2, 3, 5] {
        let prefix =
            TERMINAL_SCAN.replace("cartpow(reals, 5)", &format!("cartpow(reals, {length})"));
        let one = emit_scan(
            &format!("scan-terminal-one-{length}"),
            &format!("{prefix}outputs = alias[{length}]\n"),
        );
        let zero = emit_scan(
            &format!("scan-terminal-zero-{length}"),
            &format!("{prefix}outputs = get0(alias, {})\n", length - 1),
        );
        assert_eq!(one, zero);
        assert!(one.contains(&format!(
            "@logdensity(%arg0: tensor<2xf64>, %arg1: tensor<{length}xf64>, %arg2: tensor<f64>) -> tensor<2xf64>"
        )));
        let scan = one
            .lines()
            .find(|line| line.contains("stablehlo.while"))
            .unwrap();
        assert!(scan.contains("tensor<1x2xf64>"), "{scan}");
        if length > 1 {
            assert!(!scan.contains(&format!("tensor<{length}x2xf64>")), "{scan}");
        }
        assert_eq!(
            one.matches("stablehlo.dynamic_update_slice").count(),
            1,
            "{one}"
        );
        assert!(
            one.lines()
                .any(|line| line.contains("stablehlo.slice") && line.contains("[0:1, 0:2]")),
            "{one}"
        );
    }

    let promoted = TERMINAL_SCAN.replace("inputs = (seed, xs, gain)", "inputs = alias");
    let promoted = emit_scan(
        "scan-terminal-promoted",
        &format!("{promoted}outputs = alias[5]\n"),
    );
    assert!(promoted.contains("@logdensity(%arg0: tensor<5x2xf64>) -> tensor<2xf64>"));
    assert!(!promoted.contains("stablehlo.while"), "{promoted}");
}

#[test]
fn a_projected_scan_history_keeps_other_fields_in_the_recurrence() {
    for (name, projection) in [
        ("scan-field", "alias.mean"),
        ("scan-accessor", "read_mean.(alias)"),
    ] {
        let ir = emit_scan(name, &format!("{RECORD_SCAN}outputs = {projection}\n"));
        assert!(ir.contains(
            "@logdensity(%arg0: tensor<5xf64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<5xf64>"
        ));
        assert_eq!(ir.matches("stablehlo.while").count(), 1);
        assert_eq!(
            ir.matches("stablehlo.dynamic_update_slice").count(),
            1,
            "{ir}"
        );
    }
}

#[test]
fn mixed_scan_output_requests_keep_a_shared_complete_history() {
    for (name, prefix, outputs, types, histories) in [
        (
            "scan-field-first",
            RECORD_SCAN,
            "(alias.mean[2], alias)",
            "(tensor<f64>, tensor<5xf64>, tensor<5xf64>)",
            2,
        ),
        (
            "scan-record-first",
            RECORD_SCAN,
            "(alias, alias.mean[2])",
            "(tensor<5xf64>, tensor<5xf64>, tensor<f64>)",
            2,
        ),
        (
            "scan-terminal-first",
            TERMINAL_SCAN,
            "(alias[5], alias)",
            "(tensor<2xf64>, tensor<5x2xf64>)",
            1,
        ),
        (
            "scan-tensor-first",
            TERMINAL_SCAN,
            "(alias, get0(alias, 4))",
            "(tensor<5x2xf64>, tensor<2xf64>)",
            1,
        ),
    ] {
        let ir = emit_scan(name, &format!("{prefix}outputs = {outputs}\n"));
        assert_eq!(ir.matches("stablehlo.while").count(), 1, "{ir}");
        assert_eq!(
            ir.matches("stablehlo.dynamic_update_slice").count(),
            histories,
            "{ir}"
        );
        assert!(ir.contains(&format!("-> {types} {{")), "{ir}");
        if histories == 1 {
            let scan = ir
                .lines()
                .find(|line| line.contains("stablehlo.while"))
                .unwrap();
            assert!(scan.contains("tensor<5x2xf64>"), "{scan}");
        }
    }
}

#[test]
fn a_promoted_scan_index_keeps_the_history_and_runtime_status() {
    let prefix = TERMINAL_SCAN.replace(
        "inputs = (seed, xs, gain)",
        "index = 4\ninputs = (seed, xs, gain, index)",
    );
    let ir = emit_scan(
        "scan-runtime-index",
        &format!("{prefix}outputs = get0(alias, index)\n"),
    );
    assert!(ir.contains(
        "@logdensity(%arg0: tensor<2xf64>, %arg1: tensor<5xf64>, %arg2: tensor<f64>, %arg3: tensor<i64>) -> (tensor<2xf64>, tensor<i64>)"
    ));
    assert_eq!(ir.matches("stablehlo.while").count(), 1);
    assert_eq!(
        ir.matches("stablehlo.dynamic_update_slice").count(),
        1,
        "{ir}"
    );
    assert!(ir.contains("flatppl.runtime_status"), "{ir}");
}

#[test]
fn a_scan_checks_unreturned_fields_on_every_step() {
    let ir = emit_scan("scan-checked-field", CHECKED_SCAN);
    assert!(ir.contains("flatppl.runtime_status"), "{ir}");
    assert!(ir.contains("-> (tensor<5xf64>, tensor<i64>)"), "{ir}");
    let body = ir.split_once("} do {").expect("scan body").1;
    assert!(body.contains("dense<9>"), "{body}");
}

#[test]
fn f32_contractions_keep_their_abi_across_compensated_and_native_shapes() {
    let emit = |target, dtype, elem, length, batched, wide| {
        let rows = if wide { length } else { 2 };
        let mut a = format!("cartpow({elem}, [{rows}, {length}])");
        let mut b = if wide {
            format!("cartpow({elem}, [{length}, {length}])")
        } else {
            format!("cartpow({elem}, {length})")
        };
        let product = if batched {
            a = format!("cartpow({a}, 3)");
            b = format!("cartpow({b}, 3)");
            "product.(a, b)"
        } else {
            "product(a, b)"
        };
        let input = write_model(
            "contraction-precision",
            &format!(
                "a = elementof({a})\nb = elementof({b})\nproduct(x,y) = x * y\ninputs = (a,b)\noutputs = {product}\n"
            ),
        );
        let result = flatppl()
            .args(["stablehlo", "--target", target, "--dtype", dtype])
            .arg(input)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        String::from_utf8(result.stdout).unwrap()
    };

    for target in ["cpu", "gpu"] {
        for (batched, abi) in [(false, "2"), (true, "3x2")] {
            let ir = emit(target, "f32", "reals", 65, batched, false);
            assert!(ir.contains("stablehlo.is_finite"));
            assert!(ir.contains(&format!("-> tensor<{abi}xf32> {{")), "{ir}");
            assert!(!ir.contains("xf64>") && !ir.contains("tensor<f64>"));
        }
        for (dtype, elem, length, wide) in [
            ("f64", "reals", 65, false),
            ("f32", "reals", 8, false),
            ("f32", "integers", 65, false),
            ("f32", "reals", 65, true),
        ] {
            let ir = emit(target, dtype, elem, length, false, wide);
            assert!(ir.contains("stablehlo.dot_general"), "{ir}");
            assert!(!ir.contains("stablehlo.is_finite"), "{ir}");
        }
    }
}
