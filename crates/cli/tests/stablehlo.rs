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

/// A model with no `inputs`/`outputs` bindings: `stablehlo_cmd` falls back to
/// the legacy last-public-binding query AND prints a one-line deprecation
/// warning to stderr (design doc "Fallback + migration"; brief step 3).
#[test]
fn stablehlo_legacy_model_emits_deprecation_warning() {
    let input = write_model(
        "legacy",
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         lp = logdensityof(lawof(record(a = a)), record(a = 0.5))\n",
    );
    let out = flatppl().arg("stablehlo").arg(&input).output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no inputs/outputs bindings") && stderr.contains("declare inputs/outputs"),
        "expected a deprecation warning on stderr, got:\n{stderr}"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("func.func @logdensity"),
        "legacy path must still emit:\n{stdout}"
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

/// PR-2: a `load_data(...)` binding listed in `inputs` becomes a shape-pinned
/// tensor argument end-to-end through the real CLI binary — `stablehlo_cmd`
/// reads the resolved data file for its LENGTH only (here `data.csv` has 4 data
/// rows) and types the argument `tensor<4xf32>`, NOT `tensor<?xf32>` and NOT a
/// baked constant (design doc "load_data — shape, not values"). Both the model
/// and its `data.csv` are written to the same dir so the relative source
/// resolves.
#[test]
fn stablehlo_abi_load_data_pins_tensor_arg_from_file_length() {
    let dir = std::env::temp_dir().join(format!(
        "flatppl-stablehlo-cli-load-data-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    // Header row + 4 data rows → a length-4 vector.
    std::fs::write(dir.join("data.csv"), "y\n1.0\n2.0\n3.0\n4.0\n").unwrap();
    let input = dir.join("m.flatppl");
    std::fs::write(
        &input,
        "a = elementof(reals)\n\
         y = load_data(\"data.csv\", reals)\n\
         m = lawof(record(a = draw(Normal(mu = 0.0, sigma = 1.0))))\n\
         q1 = logdensityof(m, record(a = a))\n\
         inputs = (a, y)\n\
         outputs = q1\n",
    )
    .unwrap();
    let out = flatppl().arg("stablehlo").arg(&input).output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(
            "func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<4xf32>) -> tensor<f32>"
        ),
        "expected `y` pinned to `%arg1: tensor<4xf32>` from the file length:\n{stdout}"
    );
    assert!(
        !stdout.contains("tensor<?x"),
        "the pinned load_data arg must not carry a dynamic `?` dim:\n{stdout}"
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
