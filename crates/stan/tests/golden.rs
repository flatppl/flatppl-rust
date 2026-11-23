//! Emitted Stan text for each `golden/*.flatppl`, compared with `*.stan` and
//! `*.data.json`. Set `FLATPPL_BLESS=1` to rewrite the expected files. A first
//! line `# emit: derived=transformed-parameters target=NAME` sets options;
//! `module=LITERAL` (repeatable) puts `golden/modules/<file name of
//! LITERAL>` in the bundle under the literal a `load_module` spells, so a
//! module the CLI finds by path is the module the test reads.
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
                Some(("module", _)) => {}
                _ => panic!("unknown emit option `{kv}`"),
            }
        }
    }
    opts
}

/// The `module=` literals of a leading `# emit:` line, as a bundle of the
/// files in `tests/golden/modules` keyed by those literals.
fn bundle(source: &str) -> Option<flatppl_infer::ModuleBundle> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/modules");
    let rest = source.lines().next()?.strip_prefix("# emit: ")?;
    let mut bundle = flatppl_infer::ModuleBundle::new();
    let mut any = false;
    for literal in rest
        .split_whitespace()
        .filter_map(|kv| kv.strip_prefix("module="))
    {
        let file = Path::new(literal).file_name().expect("module file name");
        let text = std::fs::read_to_string(dir.join(file)).expect("module file");
        let module = flatppl_syntax::parse(&text).expect("parse module");
        bundle.insert(literal, std::sync::Arc::new(module));
        any = true;
    }
    any.then_some(bundle)
}

fn emit(source: &str) -> flatppl_stan::StanProgram {
    emit_named("inline source", source)
}

/// [`emit`] naming the golden in a failure, so a refusal says which file.
fn emit_named(name: &str, source: &str) -> flatppl_stan::StanProgram {
    let mut module = flatppl_syntax::parse(source).expect("parse");
    let opts = options(source);
    if let Some(bundle) = bundle(source) {
        // As `flatppl stan` infers a model with modules.
        let (diagnostics, bundle) = flatppl_infer::infer_module_with_inputs(
            &mut module,
            bundle,
            flatppl_infer::Level::Shape,
        );
        assert!(
            diagnostics
                .iter()
                .all(|d| d.severity != flatppl_infer::Severity::Error),
            "{name}: {diagnostics:?}"
        );
        return flatppl_stan::emit_with_bundle(&module, &bundle, &opts)
            .unwrap_or_else(|r| panic!("{name}: emit: {r:?}"));
    }
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{name}: {diagnostics:?}"
    );
    flatppl_stan::emit(&module, &opts).unwrap_or_else(|r| panic!("{name}: emit: {r:?}"))
}

#[test]
fn real_matrix_gram_products_use_stan_crossproducts() {
    let source = r#"# emit: target=outputs
flatppl_compat = "0.1"
theta = elementof(reals)
a = rowstack([[theta, 2.0], [3.0, 4.0], [5.0, 6.0]])
b = rowstack([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]])
row_product = a * transpose(a)
column_product = adjoint(a) * a
mixed_product = a * transpose(b)
mu = theta + trace(row_product) + trace(column_product) + trace(mixed_product)
inputs = theta
outputs = logdensityof(Normal(0, 1), theta) + logdensityof(Normal(mu, 1), 0.0)
"#;
    let code = emit(source).code;
    assert!(code.contains("row_product = tcrossprod(a);"), "{code}");
    assert!(code.contains("column_product = crossprod(a);"), "{code}");
    assert!(code.contains("mixed_product = a * b';"), "{code}");
}

#[test]
fn stan_gram_preserves_a_named_random_matrix() {
    let source = r#"
flatppl_compat = "0.1"
L_corr ~ LKJCholesky(2, 1)
b = rowstack([[2.0, 1.0], [4.0, 3.0]])
gram = L_corr * transpose(L_corr)
mixed = L_corr * transpose(b)
y ~ Normal(trace(gram) + trace(mixed), 1)
prior = lawof(record(L_corr = L_corr))
forward = kernelof(record(y = y), L_corr = L_corr)
likelihood = likelihoodof(forward, record(y = 0.0))
posterior = bayesupdate(likelihood, prior)
"#;
    let code = emit(source).code;
    assert!(code.contains("L_corr ~ lkj_corr_cholesky(1);"), "{code}");
    assert!(code.contains("gram = tcrossprod(L_corr);"), "{code}");
    assert!(code.contains("mixed = L_corr * b';"), "{code}");
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
