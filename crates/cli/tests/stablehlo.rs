//! CLI-level coverage for `flatppl stablehlo`'s `inputs`/`outputs` ABI
//! recognition (PR-1, design doc
//! `docs/superpowers/specs/2026-07-17-inputs-outputs-abi-design.md`):
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

/// A model declaring `inputs`/`outputs`: no deprecation warning, and the
/// emitted `func.func` carries the ordered 2-arg/2-result ABI signature
/// (`inputs = (a, b)` / `outputs = (q1, q2)`).
#[test]
fn stablehlo_abi_model_emits_ordered_signature_with_no_warning() {
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
    let out = flatppl().arg("stablehlo").arg(&input).output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("no inputs/outputs bindings"),
        "an ABI-declaring model must not print the legacy deprecation warning, stderr:\n{stderr}"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(
            "func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>)"
        ),
        "expected the ordered ABI signature:\n{stdout}"
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
             %arg2: tensor<f32>, %arg3: tensor<4xf32>, %arg4: tensor<4xf32>) -> tensor<f32>"
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
