//! Corpus sweep: every fixture renders with no row falling back to source
//! text, and every row carries its back-reference hooks.
//!
//! Covers the resilience copies in `tests/fixtures/` and the repo's own
//! `fixtures/flatppl/` tree. `load_module` dependencies are supplied as a host
//! would supply them: a `{resolved path: text}` bundle, each directive joined
//! to its importer's directory.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

fn fixture_dirs() -> Vec<PathBuf> {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo = crate_dir.join("../../fixtures/flatppl");
    vec![
        crate_dir.join("tests/fixtures"),
        repo.clone(),
        repo.join("bayesian_inference"),
        repo.join("load_module"),
        repo.join("queries"),
    ]
}

fn flatppl_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("flatppl"))
        .collect();
    files.sort();
    files
}

/// `load_module("<literal>")` directives of `source`, as spelled.
fn directives(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = source;
    while let Some(pos) = rest.find("load_module(") {
        rest = &rest[pos + "load_module(".len()..];
        let rest_trim = rest.trim_start();
        if let Some(body) = rest_trim.strip_prefix('"')
            && let Some(end) = body.find('"')
        {
            out.push(body[..end].to_string());
        }
    }
    out
}

/// The bundle a host would pass: each directive resolved against the
/// importing file's directory, recursively, keyed by the resolved path.
fn bundle_for(file: &Path, source: &str) -> HashMap<String, String> {
    let mut bundle = HashMap::new();
    let dir = file.parent().unwrap();
    let mut todo: Vec<PathBuf> = directives(source).iter().map(|d| dir.join(d)).collect();
    while let Some(path) = todo.pop() {
        let key = path.to_string_lossy().to_string();
        if bundle.contains_key(&key) {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        for d in directives(&text) {
            todo.push(path.parent().unwrap().join(&d));
        }
        bundle.insert(key, text);
    }
    bundle
}

#[test]
fn every_fixture_renders_without_source_text_fallbacks() {
    let mut rendered = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for dir in fixture_dirs() {
        for file in flatppl_files(&dir) {
            let source = fs::read_to_string(&file).unwrap();
            let bundle = bundle_for(&file, &source);
            let name = file.file_name().unwrap().to_string_lossy().to_string();
            let path = file.to_string_lossy();
            let rendering = match flatppl_mathdoc::render_source(&source, &path, &bundle) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!("{name}: does not render: {e}"));
                    continue;
                }
            };
            rendered += 1;
            for b in &rendering.bindings {
                if b.mathml.contains("flatppl-code") {
                    failures.push(format!("{name}: row `{}` fell back to source text", b.name));
                }
                if !b
                    .mathml
                    .starts_with("<math display=\"block\" data-flatppl-binding=\"")
                {
                    failures.push(format!("{name}: row `{}` has no binding hook", b.name));
                }
                if b.mathml.contains("<mn></mn>") {
                    failures.push(format!("{name}: row `{}` has an empty number", b.name));
                }
            }
            for d in &rendering.diagnostics {
                // Inference errors are reported, not failed: a fixture may
                // deliberately exercise an engine gap. Rendering defects are
                // rows that fell back, checked above.
                eprintln!("{name}: {}: {}", d.binding, d.message);
            }
        }
    }
    assert!(rendered >= 40, "only {rendered} fixtures rendered");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_examples_render_every_binding_once_in_source_order() {
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eight-schools.flatppl");
    let source = fs::read_to_string(&file).unwrap();
    let rendering =
        flatppl_mathdoc::render_source(&source, "eight-schools.flatppl", &HashMap::new())
            .expect("renders");
    assert_eq!(
        rendering.order,
        [
            "y_data",
            "std_errs_data",
            "J",
            "mu",
            "tau",
            "theta",
            "prior",
            "y",
            "forward_kernel",
            "L",
            "posterior"
        ]
    );
    assert!(
        rendering.diagnostics.is_empty(),
        "{:?}",
        rendering.diagnostics
    );
}
