//! Emitted Stan text for each `golden/*.flatppl`, compared with `*.stan` and
//! `*.data.json`. Set `FLATPPL_BLESS=1` to rewrite the expected files.
//!
//! These goldens check structure only. The posteriordb round trip in
//! flatppl-testsuite checks the numbers.

use std::path::Path;

fn emit(source: &str) -> flatppl_stan::StanProgram {
    let mut module = flatppl_syntax::parse(source).expect("parse");
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    flatppl_stan::emit(&module, &flatppl_stan::EmitOptions::default()).expect("emit")
}

#[test]
fn goldens() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let bless = std::env::var_os("FLATPPL_BLESS").is_some();
    let mut failures = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "flatppl"))
        .collect();
    entries.sort();
    assert!(!entries.is_empty());
    for path in entries {
        let program = emit(&std::fs::read_to_string(&path).unwrap());
        let data =
            serde_json::to_string_pretty(&serde_json::Value::Object(program.data)).unwrap() + "\n";
        for (ext, got) in [("stan", &program.code), ("data.json", &data)] {
            let expected = path.with_extension(ext);
            if bless {
                std::fs::write(&expected, got).unwrap();
            } else if std::fs::read_to_string(&expected).ok().as_deref() != Some(got.as_str()) {
                failures.push(format!("{}:\n{got}", expected.display()));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "golden mismatch:\n{}",
        failures.join("\n")
    );
}
