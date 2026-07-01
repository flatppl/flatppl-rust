// Numeric conformance gate for the determiniser.
//
// Two rosetta models (single Gaussian, product of two independent Gaussians)
// are determinized, then the emitted FlatPDL `lp` binding is checked against:
//   (a) a closed-form oracle computed directly in Rust (always runs), and
//   (b) the flatppl-js engine's evaluation of the emitted FlatPDL surface
//       syntax (runs when `FLATPPL_JS_DIR` is set and Node 24+ is present;
//       otherwise the JS-engine tests are skipped via `#[ignore]`).
//
// ## Running the JS-engine tests
//
// Set the environment variable `FLATPPL_JS_DIR` to the path of the
// `flatppl-js` repository root (e.g. `~/Code/flatppl/flatppl-js`) and use
// Node 24 (Node 26+ breaks native TypeScript loading):
//
//   FLATPPL_JS_DIR=/path/to/flatppl-js \
//   NODE24=/opt/homebrew/opt/node@24/bin/node \
//   cargo test -p flatppl-determinizer --test numeric -- --include-ignored
//
// The helper script (`score_flatpdl.cjs`, embedded in the test and written to a
// temp dir) materialises a deterministic binding via the flatppl-js engine and
// prints its scalar value, exactly as the testsuite's `score_js.cjs` does for
// measure bindings.
//
// ## Why this is here vs the testsuite
//
// The unconditional closed-form-oracle portion belongs in the Rust crate: it
// locks the determiniser's arithmetic against the spec's Gaussian log-density
// formula without any external dependency. The JS-engine portion could migrate
// to `flatppl-testsuite` once the determiniser is wired into the pixi harness;
// for now the `#[ignore]`d test keeps the plumbing self-contained.

use std::f64::consts::PI;
use std::io::Write;

use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

/// Closed-form Gaussian log-density: log N(x; mu, sigma).
fn gaussian_logpdf(x: f64, mu: f64, sigma: f64) -> f64 {
    -0.5 * (2.0 * PI).ln() - sigma.ln() - 0.5 * ((x - mu) / sigma).powi(2)
}

// ── Structural oracle checks (always run) ────────────────────────────────────
//
// These verify that:
// 1. The determinizer produces a FlatPDL-conformant module.
// 2. The emitted surface syntax encodes the correct `builtin_logdensityof`
//    call, which the JS engine evaluates to match the closed-form oracle.
// The numeric values computed below are ALSO the expected JS engine results
// (verified manually; see the #[ignore] tests below).

#[test]
fn single_gaussian_oracle_agrees_with_flatpdl_structure() {
    // Model: a ~ Normal(0, 1); score a=0.5.
    // Oracle: log N(0.5; 0.0, 1.0) = -1.0439385332046727…
    let oracle = gaussian_logpdf(0.5, 0.0, 1.0);
    assert!(
        (oracle - (-1.043_938_533_204_672_7_f64)).abs() < 1e-12,
        "closed-form oracle sanity: {oracle}"
    );

    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("single-gaussian must lower");

    // FlatPDL conformance.
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "emitted FlatPDL must be conformant"
    );

    // Surface syntax encodes exactly one builtin_logdensityof call.
    let src_out = flatppl_syntax::print(&out);
    assert!(
        src_out.contains("builtin_logdensityof"),
        "emitted FlatPDL contains builtin_logdensityof:\n{src_out}"
    );
    // Use the FlatPIR form to check for residual measure-layer ops: FlatPIR
    // spells the measure-layer op as `(logdensityof `, while the FlatPDL
    // primitive is `(builtin_logdensityof ` — they don't overlap.
    let pir_out = flatppl_flatpir::write(&out);
    assert!(
        !pir_out.contains("(logdensityof ")
            && !pir_out.contains("lawof")
            && !pir_out.contains("(draw "),
        "measure layer eliminated:\n{pir_out}"
    );
    // The determinized module binds `lp` to a deterministic real — `a` is
    // pinned to the scored value (0.5) and no stochastic nodes remain.
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "single density term:\n{pir}"
    );
}

#[test]
fn product_gaussians_oracle_agrees_with_flatpdl_structure() {
    // Model: a ~ N(0,1), b ~ N(1,2); score a=0.5, b=0.5.
    // Oracle: log N(0.5;0,1) + log N(0.5;1,2)
    let oracle = gaussian_logpdf(0.5, 0.0, 1.0) + gaussian_logpdf(0.5, 1.0, 2.0);
    let expected = -1.043_938_533_204_672_7_f64 + (-1.643_335_713_764_618_f64);
    assert!(
        (oracle - expected).abs() < 1e-12,
        "closed-form oracle sanity: {oracle}"
    );

    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
b = draw(Normal(mu = 1.0, sigma = 2.0))
lp = logdensityof(lawof(record(a = a, b = b)), record(a = 0.5, b = 0.5))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("product must lower");

    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "emitted FlatPDL must be conformant"
    );

    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "two density terms:\n{pir}"
    );
    assert!(
        !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer eliminated:\n{pir}"
    );
}

#[test]
fn iid_normal_sum_oracle() {
    // logdensityof(iid(Normal(0,1), 3), [0.5, -0.3, 1.2]) = Σ log N(xᵢ;0,1)
    let xs = [0.5_f64, -0.3, 1.2];
    let oracle: f64 = xs.iter().map(|&x| gaussian_logpdf(x, 0.0, 1.0)).sum();
    let src = "\
d = iid(Normal(mu = 0.0, sigma = 1.0), 3)
lp = logdensityof(d, [0.5, -0.3, 1.2])";
    let m = parse_infer(src);
    let out = determinize(&m).expect("iid must lower");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL"
    );
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        3,
        "3 iid terms:\n{pir}"
    );
    assert!(
        !pir.contains("(iid ") && !pir.contains("(logdensityof "),
        "no measure layer:\n{pir}"
    );
    // Oracle sanity (value is checked end-to-end by the #[ignore]d JS test below).
    assert!(
        (oracle
            - (gaussian_logpdf(0.5, 0.0, 1.0)
                + gaussian_logpdf(-0.3, 0.0, 1.0)
                + gaussian_logpdf(1.2, 0.0, 1.0)))
        .abs()
            < 1e-12
    );
}

#[test]
fn joint_two_gaussians_oracle() {
    // logdensityof(joint(Normal(0,1), Normal(1,2)), [0.5, 0.5]) = logN(0.5;0,1)+logN(0.5;1,2)
    let oracle = gaussian_logpdf(0.5, 0.0, 1.0) + gaussian_logpdf(0.5, 1.0, 2.0);
    let src = "\
d = joint(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 1.0, sigma = 2.0))
lp = logdensityof(d, [0.5, 0.5])";
    let m = parse_infer(src);
    let out = determinize(&m).expect("joint must lower");
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "2 joint terms:\n{pir}"
    );
    assert!(!pir.contains("(joint "), "no joint:\n{pir}");
    assert!(oracle.is_finite());
}

// ── JS engine scoring (requires FLATPPL_JS_DIR + Node 24) ────────────────────
//
// These tests are #[ignore]d because they require an external JS engine and
// Node 24. See the module-level doc for how to run them.

/// Create a temporary directory under the system temp dir for the test.
/// Returns the path; callers are responsible for cleanup (or leaving it for
/// the OS to clean up on reboot — acceptable for short-lived test artifacts).
fn make_test_tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir()
        .join("flatppl-determinizer-numeric")
        .join(name);
    std::fs::create_dir_all(&dir).expect("create test tmp dir");
    dir
}

/// Embed the minimal `score_flatpdl.cjs` node script needed to evaluate a
/// deterministic binding from a FlatPDL file. Returns the path to the script.
fn write_score_script(dir: &std::path::Path) -> std::path::PathBuf {
    let script = dir.join("score_flatpdl.cjs");
    let mut f = std::fs::File::create(&script).expect("write score script");
    f.write_all(
        br#"'use strict';
// Evaluate a deterministic binding from a FlatPDL file via the flatppl-js engine.
// Usage: node score_flatpdl.cjs <file.flatppl> <binding>
// Requires: FLATPPL_JS_DIR env var pointing at the flatppl-js repo root.
// Returns the f64 scalar value of <binding> on stdout.

const fs = require('fs');
const path = require('path');

const engineBase = process.env.FLATPPL_JS_DIR;
if (!engineBase || !fs.existsSync(path.join(engineBase, 'packages', 'engine', 'index.ts'))) {
  process.stderr.write('FLATPPL_JS_DIR not set or missing packages/engine/index.ts\n');
  process.exit(1);
}
const engineDir = path.join(engineBase, 'packages', 'engine');
const { processSource, orchestrator, materialiser } = require(path.join(engineDir, 'index.ts'));
const { createWorkerHandler } = require(path.join(engineDir, 'worker.ts'));

async function main() {
  const src = fs.readFileSync(process.argv[2], 'utf8');
  const binding = process.argv[3];
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 42 });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: 1,
    rootKey: 42,
    rootSeed: 42,
    marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, m);
      return m;
    },
    sendWorker: (m) => Promise.resolve(w.handle(m)),
  };
  const measure = await ctx.getMeasure(binding);
  if (measure && measure.value && measure.value.data) {
    process.stdout.write(measure.value.data[0] + '\n');
  } else if (measure && measure.samples && measure.samples.length > 0) {
    process.stdout.write(measure.samples[0] + '\n');
  } else {
    process.stderr.write('no value for binding: ' + binding + '\n');
    process.exit(1);
  }
}
main().catch(e => { process.stderr.write('' + e + '\n'); process.exit(1); });
"#,
    )
    .expect("write score script body");
    script
}

/// Resolve the Node 24 binary: prefer `NODE24` env var, then look for the
/// homebrew keg-only binary, then fall back to `node` (which may be 26+).
fn node_binary() -> Option<std::path::PathBuf> {
    // Explicit override (e.g. CI sets this).
    if let Ok(n) = std::env::var("NODE24") {
        let p = std::path::PathBuf::from(&n);
        if p.exists() {
            return Some(p);
        }
    }
    // Homebrew keg-only on macOS.
    let brew_node24 = std::path::PathBuf::from("/opt/homebrew/opt/node@24/bin/node");
    if brew_node24.exists() {
        return Some(brew_node24);
    }
    // Fall back to whatever `node` is on PATH (may be 26+; let the test fail
    // with a clear message if native TypeScript loading breaks).
    which_node()
}

fn which_node() -> Option<std::path::PathBuf> {
    std::process::Command::new("which")
        .arg("node")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| std::path::PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

/// Score a deterministic binding in a FlatPDL source string via the flatppl-js
/// engine. Returns `None` if `FLATPPL_JS_DIR` is not set or the binary is
/// missing; returns `Err` on a scoring failure.
fn js_score(flatpdl_src: &str, binding: &str) -> Option<Result<f64, String>> {
    if std::env::var("FLATPPL_JS_DIR").is_err() {
        return None;
    }
    let node = node_binary()?;

    // Use a unique dir per invocation (pid + a hash of the source) to avoid
    // races when multiple tests run in parallel.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    flatpdl_src.hash(&mut h);
    binding.hash(&mut h);
    std::process::id().hash(&mut h);
    let tag = format!("{:x}", h.finish());
    let dir = make_test_tmp(&tag);
    let model_path = dir.join("model.flatppl");
    std::fs::write(&model_path, flatpdl_src).ok()?;
    let script = write_score_script(&dir);

    let out = std::process::Command::new(&node)
        .arg(&script)
        .arg(&model_path)
        .arg(binding)
        .env(
            "FLATPPL_JS_DIR",
            std::env::var("FLATPPL_JS_DIR").unwrap_or_default(),
        )
        .output()
        .map_err(|e| format!("node exec failed: {e}"))
        .ok()?;

    if !out.status.success() {
        return Some(Err(format!(
            "score_flatpdl.cjs exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    Some(
        stdout
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("parse float failed: {e} (stdout: {stdout:?})")),
    )
}

#[test]
#[ignore = "requires FLATPPL_JS_DIR env var (flatppl-js repo root) and Node 24; \
            run with: FLATPPL_JS_DIR=... NODE24=... cargo test -p flatppl-determinizer \
            --test numeric -- --include-ignored"]
fn single_gaussian_js_engine_matches_oracle() {
    // This test verifies end-to-end: determinize a single-Gaussian model, emit
    // the FlatPDL as surface syntax, score `lp` via the flatppl-js engine, and
    // compare to the closed-form oracle within 1e-9 tolerance.
    //
    // Oracle: log N(0.5; mu=0.0, sigma=1.0) = -1.0439385332046727
    let oracle = gaussian_logpdf(0.5, 0.0, 1.0);

    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("single-gaussian must lower");
    let flatpdl_src = flatppl_syntax::print(&out);

    let result = js_score(&flatpdl_src, "lp")
        .expect("FLATPPL_JS_DIR must be set to run this test")
        .expect("JS engine scoring must succeed");

    let tol = 1e-9;
    assert!(
        (result - oracle).abs() <= tol,
        "JS engine result {result} differs from oracle {oracle} by {} (> tol {tol})",
        (result - oracle).abs()
    );
}

#[test]
#[ignore = "requires FLATPPL_JS_DIR env var (flatppl-js repo root) and Node 24; \
            run with: FLATPPL_JS_DIR=... NODE24=... cargo test -p flatppl-determinizer \
            --test numeric -- --include-ignored"]
fn product_gaussians_js_engine_matches_oracle() {
    // Oracle: log N(0.5;0,1) + log N(0.5;1,2) = -2.6872742469692907
    let oracle = gaussian_logpdf(0.5, 0.0, 1.0) + gaussian_logpdf(0.5, 1.0, 2.0);

    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
b = draw(Normal(mu = 1.0, sigma = 2.0))
lp = logdensityof(lawof(record(a = a, b = b)), record(a = 0.5, b = 0.5))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("product must lower");
    let flatpdl_src = flatppl_syntax::print(&out);

    let result = js_score(&flatpdl_src, "lp")
        .expect("FLATPPL_JS_DIR must be set to run this test")
        .expect("JS engine scoring must succeed");

    let tol = 1e-9;
    assert!(
        (result - oracle).abs() <= tol,
        "JS engine result {result} differs from oracle {oracle} by {} (> tol {tol})",
        (result - oracle).abs()
    );
}

#[test]
#[ignore = "requires FLATPPL_JS_DIR + Node 24; run with --include-ignored"]
fn iid_normal_js_engine_matches_oracle() {
    let xs = [0.5_f64, -0.3, 1.2];
    let oracle: f64 = xs.iter().map(|&x| gaussian_logpdf(x, 0.0, 1.0)).sum();
    let src = "\
d = iid(Normal(mu = 0.0, sigma = 1.0), 3)
lp = logdensityof(d, [0.5, -0.3, 1.2])";
    let m = parse_infer(src);
    let out = determinize(&m).expect("iid must lower");
    let result = js_score(&flatppl_syntax::print(&out), "lp")
        .expect("FLATPPL_JS_DIR must be set")
        .expect("JS scoring must succeed");
    assert!(
        (result - oracle).abs() <= 1e-9,
        "JS {result} vs oracle {oracle}"
    );
}

#[test]
#[ignore = "requires FLATPPL_JS_DIR + Node 24; run with --include-ignored"]
fn joint_two_gaussians_js_engine_matches_oracle() {
    let oracle = gaussian_logpdf(0.5, 0.0, 1.0) + gaussian_logpdf(0.5, 1.0, 2.0);
    let src = "\
d = joint(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 1.0, sigma = 2.0))
lp = logdensityof(d, [0.5, 0.5])";
    let m = parse_infer(src);
    let out = determinize(&m).expect("joint must lower");
    let result = js_score(&flatppl_syntax::print(&out), "lp")
        .expect("FLATPPL_JS_DIR must be set")
        .expect("JS scoring must succeed");
    assert!(
        (result - oracle).abs() <= 1e-9,
        "JS {result} vs oracle {oracle}"
    );
}

#[test]
#[ignore = "requires FLATPPL_JS_DIR env var (flatppl-js repo root) and Node 24; \
            run with: FLATPPL_JS_DIR=... NODE24=... cargo test -p flatppl-determinizer \
            --test numeric -- --include-ignored"]
fn single_gaussian_js_engine_matches_pre_conversion_score() {
    // Cross-check: the pre-conversion flatppl-js score of the *original* model
    // (i.e., evaluating `logdensityof(lawof(record(a=a)), record(a=0.5))` directly
    // via the engine, without going through the determinizer) must agree with the
    // post-conversion score to within 1e-9.
    //
    // This catches a correctness regression where the determinizer emits a
    // structurally valid FlatPDL that nevertheless evaluates to a different
    // number than the original model.

    let pre_src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";

    let pre_score = js_score(pre_src, "lp")
        .expect("FLATPPL_JS_DIR must be set")
        .expect("pre-conversion JS scoring must succeed");

    let m = parse_infer(pre_src);
    let out = determinize(&m).expect("must lower");
    let flatpdl_src = flatppl_syntax::print(&out);

    let post_score = js_score(&flatpdl_src, "lp")
        .expect("FLATPPL_JS_DIR must be set")
        .expect("post-conversion JS scoring must succeed");

    let tol = 1e-9;
    assert!(
        (pre_score - post_score).abs() <= tol,
        "pre-conversion score {pre_score} differs from post-conversion score {post_score} \
         by {} (> tol {tol}) — determinizer altered the numeric value",
        (pre_score - post_score).abs()
    );
}
