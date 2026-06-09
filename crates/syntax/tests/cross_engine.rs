//! Cross-engine FlatPIR parity harness — the one spec-defined cross-engine
//! check: the same FlatPPL source, lowered independently by `flatppl-js`
//! (reference engine) and `flatppl-rust`, must yield structurally identical
//! FlatPIR.
//!
//! Stimuli are the `.flatppl` fixture corpora of BOTH repos (reuse proven
//! stimuli; never bake one engine's output in as the expected value — both
//! sides are derived live, per run). Structural comparison: the JS engine's
//! S-expression text is normalized through this workspace's FlatPIR
//! `read → write`, so formatting differences can't confound; what's compared
//! is the canonical projection of two in-memory modules.
//!
//! Needs a `flatppl-js` checkout (env `FLATPPL_JS_DIR`, default: sibling of
//! this workspace, mirroring the `GRAMMARS_DIR` CI pattern) and `node`
//! ≥ 22.18 (runs the engine's TypeScript via built-in type-stripping). When
//! either is missing the test SKIPS with a notice rather than failing —
//! plain `cargo test` must stay green on a standalone checkout.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use flatppl_syntax::parse;

const BEGIN: &str = "<<<FLATPPL-PARITY:BEGIN";
const END: &str = "<<<FLATPPL-PARITY:END";

/// Understood, filed divergences pending a fix in one engine: `(label,
/// reason)`. A listed fixture that STOPS diverging fails the test as stale,
/// so this list can only shrink truthfully.
///
/// Every entry below was adjudicated against the spec (2026-06-11) and falls
/// on the JS side — its `toSexpr` export reflects post-analysis engine
/// internals rather than the §04 lowering projection. Filed as
/// "Spec-faithful FlatPIR export" (+ two fixture items) in
/// `flatppl-dev/TODO-flatppl-js.md` §11; the reason names the FIRST
/// divergence class hit (fixtures may hit several — when JS fixes one class,
/// entries either flip to agree (then fail as stale, shrinking this list) or
/// surface the next class).
const KNOWN_DIVERGENCES: &[(&str, &str)] = &[
    (
        "rust:eight-schools",
        "JS folds `neg` of a literal into a negative literal",
    ),
    (
        "rust:einsum-matmul",
        "JS decomposition export: `%mlhs:` temps + 0-based `tuple_get`",
    ),
    (
        "rust:expressions",
        "JS emits `get_field` for field access (spec: `get`)",
    ),
    (
        "rust:minimal",
        "JS eagerly lowers `kernelof` to `functionof(lawof(…))`",
    ),
    (
        "rust:modules",
        "JS `toSexpr` crashes on `load_module` %assign substitutions",
    ),
    (
        "js:bayesian_inference_1",
        "JS eagerly lowers `kernelof` to `functionof(lawof(…))`",
    ),
    (
        "js:bayesian_inference_2",
        "JS eagerly lowers `kernelof` to `functionof(lawof(…))`",
    ),
    (
        "js:bayesian_inference_3",
        "JS splices in symbolically-executed `disintegrate` results",
    ),
    (
        "js:bayesian_inference_4",
        "JS splices in symbolically-executed `restrict` results",
    ),
    (
        "js:beta-binomial-pushfwd",
        "JS eta-expands dotted-op `broadcast(add, …)` heads",
    ),
    (
        "js:beverton-holt",
        "JS eta-expands dotted-op `broadcast(add, …)` heads",
    ),
    (
        "js:disintegrate-complex",
        "JS splices in symbolically-executed `disintegrate` results",
    ),
    (
        "js:disintegrate-unsupported",
        "JS splices in symbolically-executed `disintegrate` results",
    ),
    (
        "js:disintegrate_dual_kernel",
        "JS splices in symbolically-executed `disintegrate` results",
    ),
    (
        "js:eight-schools",
        "JS folds `neg` of a literal into a negative literal",
    ),
    (
        "js:einsum-matmul",
        "JS decomposition export: `%mlhs:` temps + 0-based `tuple_get`",
    ),
    (
        "js:einsum-patterns",
        "JS prints built-in values as `(%ref self sum)` (spec: bare symbol)",
    ),
    (
        "js:flatppl-uncorrelated_background-draws-auxm",
        "JS prints built-in values as `(%ref self add)` (spec: bare symbol)",
    ),
    (
        "js:flatppl-uncorrelated_background-draws-priors",
        "JS prints built-in values as `(%ref self add)` (spec: bare symbol)",
    ),
    (
        "js:flatppl-uncorrelated_background-ma-auxm",
        "JS prints built-in values as `(%ref self add)` (spec: bare symbol)",
    ),
    (
        "js:flatppl-uncorrelated_background-ma-priors",
        "JS prints built-in values as `(%ref self add)` (spec: bare symbol)",
    ),
    (
        "js:hadron-physics-resonance",
        "JS resolves a user alias binding to its target at call sites",
    ),
    (
        "js:hierarchical-repeated-measures",
        "fixture uses off-spec bare-name `kernelof` boundary inputs",
    ),
    (
        "js:horseshoe",
        "JS folds `neg` of a literal into a negative literal",
    ),
    (
        "js:joint-mvnormal-component",
        "fixture uses off-spec bare-name `kernelof` boundary inputs",
    ),
    (
        "js:metricsum-tensor",
        "JS folds `neg` of a literal into a negative literal",
    ),
    (
        "js:minimal",
        "JS eagerly lowers `kernelof` to `functionof(lawof(…))`",
    ),
    (
        "js:normal-mixture",
        "JS folds `neg` of a literal into a negative literal",
    ),
    (
        "js:polyeval-iid-broadcast",
        "JS eta-expands dotted-op `broadcast(mul, …)` heads",
    ),
    (
        "js:polyeval-iid-broadcast-chain",
        "JS eta-expands dotted-op `broadcast(mul, …)` heads",
    ),
    (
        "js:rasch-two-parameter",
        "JS eta-expands dotted-op `broadcast(…)` heads",
    ),
    (
        "js:simple-transport1",
        "JS emits `get_field` for field access (spec: `get`)",
    ),
    (
        "js:simple-transport2",
        "JS eagerly lowers `kernelof` to `functionof(lawof(…))`",
    ),
    (
        "js:vector-obs-mvnormal",
        "JS folds `neg` of a literal into a negative literal",
    ),
    (
        "js:zero-inflated-binomial",
        "JS eagerly lowers `kernelof` to `functionof(lawof(…))`",
    ),
];

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Locate the flatppl-js checkout. An explicit `FLATPPL_JS_DIR` must exist
/// (silently skipping a deliberate override would hide misconfiguration);
/// the sibling default is optional.
fn js_repo() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("FLATPPL_JS_DIR") {
        let path = PathBuf::from(dir);
        assert!(
            path.join("packages/engine").is_dir(),
            "FLATPPL_JS_DIR is set but {} has no packages/engine",
            path.display()
        );
        return Some(path);
    }
    let sibling = manifest_dir().join("../../../flatppl-js");
    sibling
        .join("packages/engine")
        .is_dir()
        .then(|| sibling.canonicalize().unwrap_or(sibling))
}

/// `node` present and new enough for built-in TS type-stripping (≥ 22.18)?
fn usable_node() -> Result<(), String> {
    let out = Command::new("node")
        .arg("--version")
        .output()
        .map_err(|e| format!("node not runnable: {e}"))?;
    let version = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let mut parts = version.trim_start_matches('v').split('.');
    let major: u32 = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    let minor: u32 = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    if major > 22 || (major == 22 && minor >= 18) {
        Ok(())
    } else {
        Err(format!(
            "node {version} lacks TS type-stripping (need ≥ 22.18)"
        ))
    }
}

/// The stimulus corpus: every `.flatppl` fixture from both engines' suites,
/// labelled `rust:<stem>` / `js:<stem>`.
fn corpus(js_repo: &Path) -> Vec<(String, PathBuf)> {
    let dirs = [
        ("rust", manifest_dir().join("../../fixtures/flatppl")),
        ("js", js_repo.join("packages/engine/test/fixtures")),
    ];
    let mut fixtures = Vec::new();
    for (tag, dir) in dirs {
        let mut paths: Vec<PathBuf> = fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
            .map(|entry| entry.unwrap().path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "flatppl"))
            .collect();
        paths.sort();
        for path in paths {
            let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
            fixtures.push((format!("{tag}:{stem}"), path));
        }
    }
    fixtures
}

/// Run the node shim once over the whole corpus; returns per-path
/// `(ok, body)` records (body = FlatPIR text or one-line error).
fn run_js_engine(js_repo: &Path, files: &[PathBuf]) -> Vec<(String, bool, String)> {
    let script = manifest_dir().join("tests/cross_engine/emit-flatpir.cjs");
    let out = Command::new("node")
        .arg(&script)
        .arg(js_repo)
        .args(files)
        .output()
        .expect("spawning node");
    assert!(
        out.status.success(),
        "emit-flatpir.cjs failed (is flatppl-js npm-installed?):\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);

    let mut records = Vec::new();
    let mut lines = stdout.lines();
    while let Some(line) = lines.next() {
        let Some(header) = line.strip_prefix(BEGIN) else {
            continue;
        };
        let (status, path) = header
            .trim()
            .split_once(' ')
            .unwrap_or_else(|| panic!("malformed parity header: {line}"));
        let mut body = Vec::new();
        for body_line in lines.by_ref() {
            if body_line == END {
                break;
            }
            body.push(body_line);
        }
        records.push((path.to_string(), status == "ok", body.join("\n")));
    }
    assert_eq!(records.len(), files.len(), "shim returned a partial corpus");
    records
}

/// Render the first point of divergence between two canonical FlatPIR texts.
fn first_difference(rust: &str, js: &str) -> String {
    let (rust_lines, js_lines): (Vec<_>, Vec<_>) = (rust.lines().collect(), js.lines().collect());
    let n = rust_lines.len().max(js_lines.len());
    let at = (0..n)
        .find(|&i| rust_lines.get(i) != js_lines.get(i))
        .unwrap_or(0);
    let mut report = format!("first difference at canonical FlatPIR line {}:\n", at + 1);
    for i in at.saturating_sub(2)..(at + 3).min(n) {
        report.push_str(&format!(
            "    rust | {}\n    js   | {}\n",
            rust_lines.get(i).unwrap_or(&"<end>"),
            js_lines.get(i).unwrap_or(&"<end>"),
        ));
    }
    report
}

#[test]
fn cross_engine_flatpir_parity() {
    let Some(js_repo) = js_repo() else {
        eprintln!("SKIPPED cross-engine parity: no flatppl-js checkout (set FLATPPL_JS_DIR)");
        return;
    };
    if let Err(reason) = usable_node() {
        // An explicit FLATPPL_JS_DIR (the CI configuration) demands the
        // check actually runs — a quiet skip there would report green
        // without comparing anything.
        assert!(
            std::env::var_os("FLATPPL_JS_DIR").is_none(),
            "FLATPPL_JS_DIR is set but the parity check cannot run: {reason}"
        );
        eprintln!("SKIPPED cross-engine parity: {reason}");
        return;
    }

    let fixtures = corpus(&js_repo);
    let paths: Vec<PathBuf> = fixtures.iter().map(|(_, p)| p.clone()).collect();
    let js_results = run_js_engine(&js_repo, &paths);

    let mut agree = 0u32;
    let mut both_reject: Vec<String> = Vec::new();
    let mut findings: Vec<(String, String)> = Vec::new(); // (label, report)
    for ((label, path), (_, js_ok, js_body)) in fixtures.iter().zip(&js_results) {
        let src = fs::read_to_string(path).unwrap();
        let rust = parse(&src).map(|m| flatppl_flatpir::write(&m));
        let report = match (&rust, js_ok) {
            (Err(e), false) => {
                // Agreement: both engines refuse this source. Reported (not
                // failed) so an asymmetric pair of rejection REASONS stays
                // visible — that can hide a wrong rejection on either side.
                both_reject.push(format!(
                    "  BOTH-REJECT {label}\n    rust: {e}\n    js:   {js_body}"
                ));
                continue;
            }
            (Err(e), true) => format!("rust rejects what js lowers: {e}"),
            (Ok(_), false) => format!("js rejects what rust lowers: {js_body}"),
            (Ok(rust_pir), true) => match flatppl_flatpir::read(js_body) {
                Err(e) => {
                    format!("js-emitted FlatPIR is unreadable here: {e}\n--- js ---\n{js_body}")
                }
                Ok(js_module) => {
                    let js_pir = flatppl_flatpir::write(&js_module);
                    if &js_pir == rust_pir {
                        agree += 1;
                        continue;
                    }
                    first_difference(rust_pir, &js_pir)
                }
            },
        };
        findings.push((label.clone(), report));
    }

    let mut unexpected = Vec::new();
    let mut seen_known = Vec::new();
    for (label, report) in &findings {
        match KNOWN_DIVERGENCES.iter().find(|(l, _)| l == label) {
            Some((_, reason)) => seen_known.push(format!("  KNOWN  {label}: {reason}")),
            None => unexpected.push(format!("DIVERGE {label}\n  {report}")),
        }
    }
    let stale: Vec<&str> = KNOWN_DIVERGENCES
        .iter()
        .filter(|(l, _)| !findings.iter().any(|(fl, _)| fl == l))
        .map(|(l, _)| *l)
        .collect();

    println!(
        "cross-engine FlatPIR parity: {} fixtures — {agree} agree, {} both-reject, \
         {} known divergences, {} unexpected",
        fixtures.len(),
        both_reject.len(),
        seen_known.len(),
        unexpected.len(),
    );
    for line in both_reject.iter().chain(&seen_known) {
        println!("{line}");
    }
    // One combined failure: stale entries and unexpected divergences often
    // arrive together (a fixture RENAME in the flatppl-js corpus is one
    // stale label + one unexpected label), so failing on the first would
    // hide the half that explains it.
    let mut failure = String::new();
    if !stale.is_empty() {
        failure.push_str(&format!(
            "stale KNOWN_DIVERGENCES entries — no longer diverging (fixed \
             upstream, or the fixture was renamed/removed in the flatppl-js \
             corpus); remove or relabel them:\n  {stale:?}\n\n"
        ));
    }
    if !unexpected.is_empty() {
        failure.push_str(&format!(
            "cross-engine FlatPIR divergence:\n\n{}",
            unexpected.join("\n\n")
        ));
    }
    assert!(failure.is_empty(), "{failure}");
}
