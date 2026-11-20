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
    emit_named("inline source", source)
}

/// [`emit`] naming the golden in a failure, so a refusal says which file.
fn emit_named(name: &str, source: &str) -> flatppl_stan::StanProgram {
    let mut module = flatppl_syntax::parse(source).expect("parse");
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{name}: {diagnostics:?}"
    );
    flatppl_stan::emit(&module, &options(source)).unwrap_or_else(|r| panic!("{name}: emit: {r:?}"))
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
        let program = emit_named(
            &path.display().to_string(),
            &std::fs::read_to_string(&path).unwrap(),
        );
        let data =
            serde_json::to_string_pretty(&serde_json::Value::Object(program.data)).unwrap() + "\n";
        // A golden with a `.manifest.json` beside it checks the manifest too.
        let manifest = serde_json::to_string_pretty(&program.manifest.to_json()).unwrap() + "\n";
        let mut outputs = vec![("stan", &program.code), ("data.json", &data)];
        if path.with_extension("manifest.json").exists() {
            outputs.push(("manifest.json", &manifest));
        }
        for (ext, got) in outputs {
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

#[test]
fn deferred_complex_data_places_component_pairs_at_the_last_axis() {
    let mut program = emit(
        r#"
flatppl_compat = "0.1"
z = load_data("z.json", complexes)
calibration = load_data("matrix.json", cartprod(B = cartpow(complexes, [2, 2])))
fixed = complex(2.5, 3.5)
a ~ Normal(0, 1)
y ~ Normal(a + real(z) + imag(calibration.B[1, 2]) + imag(fixed), 1)
prior = lawof(record(a = a))
L = likelihoodof(kernelof(record(y = y), a), record(y = 1.5))
post = bayesupdate(L, prior)
"#,
    );
    let loaded = flatppl_syntax::parse(
        r#"
flatppl_compat = "0.1"
z = complex(1.25, 0.0)
calibration = record(B = array([complex(1.0, 0.0), complex(2.0, 0.0), complex(3.0, 0.0), complex(4.0, 0.0)], [2, 2], [1, 2]))
"#,
    )
    .expect("parse loaded literals");
    flatppl_stan::complete_data(&mut program, &loaded).expect("serialize loaded complex data");
    assert_eq!(program.data["z"], serde_json::json!([1.25, 0.0]));
    assert_eq!(
        program.data["B"],
        serde_json::json!([[[1.0, 0.0], [2.0, 0.0]], [[3.0, 0.0], [4.0, 0.0]]]),
    );
    assert!(!program.data.contains_key("fixed"));
    assert!(program.code.contains("transformed data {\n  complex fixed"));
}
