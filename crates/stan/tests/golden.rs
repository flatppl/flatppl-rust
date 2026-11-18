//! Emitted Stan text for each `golden/*.flatppl`, compared with `*.stan` and
//! `*.data.json`. Set `FLATPPL_BLESS=1` to rewrite the expected files. A first
//! line `# emit: derived=transformed-parameters target=NAME` sets options.
//!
//! These goldens check structural translation with the default options (no
//! `--reduce`). The statistics tests cover `--reduce sufficient`. The
//! posteriordb round trip in flatppl-testsuite checks the numbers.

use std::path::Path;

/// Options from a leading `# emit: derived=transformed-parameters` line.
fn options(source: &str) -> flatppl_stan::EmitOptions {
    let mut opts = flatppl_stan::EmitOptions::default();
    if let Some(rest) = source
        .lines()
        .next()
        .and_then(|l| l.strip_prefix("# emit: "))
    {
        for kv in rest.split_whitespace() {
            match kv.split_once('=') {
                Some(("derived", v)) => {
                    opts.derived = flatppl_stan::Placement::parse(v).expect("derived value")
                }
                Some(("target", v)) => opts.target = Some(v.to_string()),
                _ => panic!("unknown emit option `{kv}`"),
            }
        }
    }
    opts
}

fn emit(source: &str) -> flatppl_stan::StanProgram {
    let mut module = flatppl_syntax::parse(source).expect("parse");
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    flatppl_stan::emit(&module, &options(source)).expect("emit")
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

/// A size is named by its source, never by its value: two data sets of
/// different shape give the same program.
#[test]
fn extent_names_follow_their_source() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).unwrap();
    let a = emit(&read("extent_source_a.flatppl"));
    let b = emit(&read("extent_source_b.flatppl"));
    assert_eq!(a.code, b.code);
    assert_ne!(a.data, b.data);
}
