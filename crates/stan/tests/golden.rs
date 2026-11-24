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
    emit_named_with_rules(name, source, &flatppl_stan::RulePolicy::default())
}

fn emit_named_with_rules(
    name: &str,
    source: &str,
    rules: &flatppl_stan::RulePolicy,
) -> flatppl_stan::StanProgram {
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
        return flatppl_stan::emit_with_bundle_and_rules(&module, &bundle, &opts, rules)
            .unwrap_or_else(|r| panic!("{name}: emit: {r:?}"));
    }
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{name}: {diagnostics:?}"
    );
    flatppl_stan::emit_with_rules(&module, &opts, rules)
        .unwrap_or_else(|r| panic!("{name}: emit: {r:?}"))
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
fn fixed_record_fields_stay_stan_data() {
    let source = r#"# emit: target=outputs
a = elementof(cartpow(reals, 3))
b = elementof(cartpow(reals, 3))
s = elementof(posreals)
d = (g = [3, 1, 2, 1, 3], y = [0.4, 1.0, 1.3, 2.0, 0.3])
inputs = (a, b, s)
outputs = logdensityof(Normal.(a[d.g], s), d.y) + logdensityof(Normal.(b[d.g], s), d.y)
"#;
    let program = emit(source);
    let code = program.code;
    assert!(code.contains("array[N] int g;"), "{code}");
    assert!(code.contains("y ~ normal(a[g], s);"), "{code}");
    assert!(code.contains("y ~ normal(b[g], s);"), "{code}");
    assert!(program.data.contains_key("g"), "{:?}", program.data);
}

#[test]
fn a_written_singleton_matrix_stays_stan_data() {
    let source = r#"# emit: target=outputs
theta = elementof(reals)
x = rowstack([[0.5, 1.5]])
inputs = theta
outputs = logdensityof(Normal(0, 1), theta) + logdensityof(Normal(theta, 1), x[1, 2])
"#;
    let program = emit(source);
    let code = program.code;
    assert!(code.contains("matrix[N, N_x_2] x;"), "{code}");
    assert!(!code.contains("to_matrix"), "{code}");
    assert!(program.data.contains_key("x"), "{:?}", program.data);
}

#[test]
fn an_unnamed_singleton_stack_keeps_its_stan_form() {
    let source = r#"# emit: target=outputs
theta = elementof(reals)
inputs = theta
outputs = logdensityof(Normal(0, 1), theta) + logdensityof(Normal(theta, 1), rowstack([[0.5, 1.5]])[1, 2])
"#;
    let code = emit(source).code;
    assert!(code.contains("to_matrix([0.5, 1.5])[1, 2]"), "{code}");
}

/// A regression on a data design matrix, observed through a kernel.
fn regression(law: &str, observed: &str) -> String {
    format!(
        r#"# emit: target=posterior
flatppl_compat = "0.1"
X = rowstack([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]])
alpha = elementof(reals)
beta = elementof(cartpow(reals, 2))
sigma ~ Exponential(1)
y ~ {law}
prior = joint(Lebesgue(cartprod(alpha = reals, beta = cartpow(reals, 2))), lawof((sigma = sigma)))
forward = kernelof(record(y = y), alpha = alpha, beta = beta, sigma = sigma)
posterior = bayesupdate(likelihoodof(forward, record(y = {observed})), prior)
"#
    )
}

fn emit_glm(source: &str) -> (String, String) {
    let glm_off = flatppl_stan::RulePolicy::from_selectors(["ruleset:glm".parse().unwrap()]);
    let default = emit(source).code;
    let written = emit_named_with_rules("inline source", source, &glm_off).code;
    (default, written)
}

#[test]
fn a_linear_predictor_on_data_uses_the_stan_glm_functions() {
    let reals = "[1.0, 2.0, 3.0]";
    let counts = "[1, 0, 1]";
    for (law, observed, glm, written) in [
        (
            "Normal.(X * beta, sigma)",
            reals,
            "y ~ normal_id_glm(X, 0",
            "y ~ normal(X * beta, sigma);",
        ),
        (
            "Normal.(alpha .+ X * beta, sigma)",
            reals,
            "y ~ normal_id_glm(X, alpha, beta, sigma);",
            "y ~ normal(alpha + X * beta, sigma);",
        ),
        (
            "Bernoulli.(invlogit.(alpha .+ X * beta))",
            counts,
            "y ~ bernoulli_logit_glm(X, alpha, beta);",
            "y ~ bernoulli_logit(alpha + X * beta);",
        ),
        (
            "Poisson.(exp.(X * beta .+ alpha))",
            counts,
            "y ~ poisson_log_glm(X, alpha, beta);",
            "y ~ poisson_log(X * beta + alpha);",
        ),
    ] {
        let source = regression(law, observed);
        let (default, unrewritten) = emit_glm(&source);
        assert!(default.contains(glm), "{law}:\n{default}");
        assert!(unrewritten.contains(written), "{law}:\n{unrewritten}");
    }
}

/// The design matrix is a field of a record, read by its name.
fn record_regression(record: &str) -> String {
    format!(
        r#"# emit: target=posterior
flatppl_compat = "0.1"
beta = elementof(cartpow(reals, 2))
sigma ~ Exponential(1)
d = {record}
y ~ Normal.(d.X * beta, sigma)
prior = joint(Lebesgue(cartprod(beta = cartpow(reals, 2))), lawof((sigma = sigma)))
forward = kernelof(record(y = y), beta = beta, sigma = sigma)
posterior = bayesupdate(likelihoodof(forward, record(y = [1.0, 2.0, 3.0])), prior)
"#
    )
}

#[test]
fn a_design_matrix_field_of_a_data_record_uses_the_stan_glm_function() {
    let source = record_regression("record(X = rowstack([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]]))");
    let (default, unrewritten) = emit_glm(&source);
    assert!(default.contains("y ~ normal_id_glm("), "{default}");
    assert!(!unrewritten.contains("_glm"), "{unrewritten}");
}

#[test]
fn a_design_matrix_field_of_a_record_that_is_not_data_keeps_the_written_law() {
    let source =
        record_regression("record(X = sigma .* rowstack([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]]))");
    let code = emit(&source).code;
    assert!(!code.contains("_glm"), "{code}");
}

#[test]
fn a_regression_law_on_a_parameter_design_is_its_vectorized_statement() {
    for (law, observed, statement) in [
        (
            "reg.NormalGLM(x, 0.0, beta, 1.0)",
            "[1.0, 2.0, 3.0]",
            "y ~ normal(x * beta, 1.0);",
        ),
        (
            "reg.BernoulliLogitGLM(x, alpha, beta)",
            "[1, 0, 1]",
            "y ~ bernoulli_logit(alpha + x * beta);",
        ),
        (
            "reg.PoissonLogGLM(x, alpha, beta)",
            "[1, 0, 2]",
            "y ~ poisson_log(alpha + x * beta);",
        ),
    ] {
        let source = format!(
            r#"# emit: target=L
flatppl_compat = "0.1"
reg = standard_module("regression", "0.1")
x = elementof(cartpow(reals, [3, 2]))
alpha = elementof(reals)
beta = elementof(cartpow(reals, 2))
y ~ {law}
K = kernelof(record(y = y), x = x, alpha = alpha, beta = beta)
L = likelihoodof(K, record(y = {observed}))
"#
        );
        let code = emit(&source).code;
        assert!(code.contains(statement), "{law}:\n{code}");
        assert!(!code.contains("_glm"), "{law}:\n{code}");
    }
}

#[test]
fn a_design_matrix_that_is_not_data_keeps_the_written_law() {
    let source = regression("Normal.((sigma .* X) * beta, sigma)", "[1.0, 2.0, 3.0]");
    let code = emit(&source).code;
    assert!(!code.contains("_glm"), "{code}");
}

fn golden_models() -> Vec<std::path::PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "flatppl"))
        .collect();
    entries.sort();
    assert!(!entries.is_empty());
    entries
}

fn data_text(program: &flatppl_stan::StanProgram) -> String {
    serde_json::to_string_pretty(&serde_json::Value::Object(program.data.clone())).unwrap() + "\n"
}

/// The emitter runs after optimisation and must also translate the model as
/// written. With every rule and pass off, each golden keeps its data and its
/// `~` statements, and its text differs only where a default rule rewrites an
/// expression. A golden whose law a default rule replaces by design prints
/// its written law, and differs from the golden in that one line only.
#[test]
fn goldens_without_optimisation() {
    // The mixture weights are already normalised, so the default drops the
    // zero `- log(sum(weights))` term.
    const REWRITTEN: &[&str] = &["ksuperpose_shared_argument"];
    // A linear predictor on a data matrix uses a Stan GLM function by default.
    const LAW_REPLACED: &[(&str, &str)] =
        &[("user_function_loop", "y ~ normal(alpha + X * beta, sigma);")];
    let rules = flatppl_stan::RulePolicy::all_disabled();
    let tildes = |code: &str| -> Vec<String> {
        code.lines()
            .filter(|line| line.contains(" ~ "))
            .map(|line| line.trim().to_owned())
            .collect()
    };
    let mut failures = Vec::new();
    for path in golden_models() {
        let name = path.display().to_string();
        let program =
            emit_named_with_rules(&name, &std::fs::read_to_string(&path).unwrap(), &rules);
        let expected_data = std::fs::read_to_string(path.with_extension("data.json")).unwrap();
        if data_text(&program) != expected_data {
            failures.push(format!("{name}: data\n{}", data_text(&program)));
        }
        let expected = std::fs::read_to_string(path.with_extension("stan")).unwrap();
        let stem = path.file_stem().unwrap().to_str().unwrap();
        if let Some(&(_, written)) = LAW_REPLACED.iter().find(|(name, _)| *name == stem) {
            let changed: Vec<_> = program
                .code
                .lines()
                .zip(expected.lines())
                .filter(|(unoptimised, golden)| unoptimised != golden)
                .map(|(unoptimised, _)| unoptimised.trim())
                .collect();
            let same_length = program.code.lines().count() == expected.lines().count();
            if !same_length || changed != [written] {
                failures.push(format!(
                    "{name}: expected only `{written}` to differ\n{}",
                    program.code
                ));
            }
        } else if !REWRITTEN.contains(&stem) {
            if program.code != expected {
                failures.push(format!("{name}: text\n{}", program.code));
            }
        } else if program.code == expected {
            failures.push(format!("{name}: no default rule rewrites it now"));
        } else if tildes(&program.code) != tildes(&expected) {
            failures.push(format!("{name}: `~` statements\n{}", program.code));
        }
    }
    assert!(
        failures.is_empty(),
        "unoptimised emission:\n{}",
        failures.join("\n")
    );
}

#[test]
fn goldens() {
    let bless = std::env::var_os("FLATPPL_BLESS").is_some();
    let mut failures = Vec::new();
    for path in golden_models() {
        let program = emit_named(
            &path.display().to_string(),
            &std::fs::read_to_string(&path).unwrap(),
        );
        let data = data_text(&program);
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
