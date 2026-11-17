use flatppl_stan::{EmitOptions, StanProgram};

fn emit(source: &str, enabled: bool) -> StanProgram {
    let mut m = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut m);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    let options = EmitOptions {
        target: Some("lik".into()),
        sufficient_statistics: enabled,
        ..Default::default()
    };
    flatppl_stan::emit(&m, &options).unwrap()
}

const MODEL: &str = r#"flatppl_compat = "0.1"
N = 3
samples = [999999.75, 1000000.0, 1000000.25]
mu = elementof(reals)
sigma = elementof(reals)
y ~ iid(Normal(sigma = sigma, mu = mu), N)
lik = likelihoodof(kernelof(record(y = y), mu = mu, sigma = sigma), record(y = samples))
"#;

#[test]
fn prepares_data_once_and_preserves_the_native_density_and_interface() {
    assert!(EmitOptions::default().sufficient_statistics);
    let on = emit(MODEL, true);
    let off = emit(MODEL, false);
    assert_eq!(on.data, off.data);
    assert_eq!(on.deferred_data, off.deferred_data);
    assert_eq!(on.manifest, off.manifest);
    assert!(off.code.contains("samples ~ normal(mu, sigma);"));
    assert!(!off.code.contains("transformed data"));
    let (setup, model) = on.code.split_once("model {").unwrap();
    assert!(setup.contains("transformed data {"));
    assert!(setup.contains("if (num_elements(samples) == 3"));
    assert!(setup.contains("sum("));
    assert_eq!(model.matches("normal_lpdf(").count(), 2);
    assert!(model.contains("normal_lpdf(samples |"));
    assert!(!model.contains("sum("));
}

#[test]
fn separate_observations_with_different_means_keep_their_own_scores() {
    let on = emit(
        r#"flatppl_compat = "0.1"
mu = elementof(reals)
a = [1.0, 2.0, 3.0]
y ~ iid(Normal(mu, 1.0), 3)
z ~ iid(Normal(2.0 * mu, 0.5), 3)
lik = likelihoodof(kernelof(record(y = y, z = z), mu = mu),
                   record(y = a, z = [0.0, 4.0, 2.0]))
"#,
        true,
    );
    let (setup, model) = on.code.split_once("model {").unwrap();
    assert_eq!(setup.matches("if (num_elements(").count(), 2);
    assert_eq!(model.matches("normal_lpdf(").count(), 4);
    assert!(model.contains("2.0 * mu"));
    assert!(!setup.contains("2.0 * mu"));
}

#[test]
fn summary_names_preserve_data_names_used_by_later_observations() {
    let source = MODEL
        .replace(
            "lik = likelihoodof",
            "samples_center = [1.0, 2.0, 3.0]\nz ~ iid(Normal(mu, sigma), N)\nlik = likelihoodof",
        )
        .replace("record(y = y)", "record(y = y, z = z)")
        .replace(
            "record(y = samples)",
            "record(y = samples, z = samples_center)",
        );
    let on = emit(&source, true);
    let off = emit(&source, false);
    assert_eq!(on.data, off.data);
    assert_eq!(on.manifest, off.manifest);
    assert!(on.code.contains("real samples_center_2 = 0;"));
}

#[test]
fn skips_scoring_inputs_and_unsupported_laws() {
    let promoted = MODEL.replace(
        "mu = mu, sigma = sigma), record",
        "mu = mu, sigma = sigma, obs = samples), record",
    );
    let other_law = MODEL.replace("Normal(sigma = sigma, mu = mu)", "Cauchy(mu, sigma)");
    for source in [promoted, other_law] {
        assert_eq!(emit(&source, true).code, emit(&source, false).code);
    }
}

/// A `logdensityof` target scores its free inputs as a likelihood does: the
/// summaries prepare the data only, and the interface does not change.
#[test]
fn logdensityof_target_uses_the_same_summaries() {
    const TARGET: &str = r#"flatppl_compat = "0.1"
N = 3
samples = [0.25, -1.5, 2.0]
mu = elementof(reals)
sigma = elementof(posreals)
lp = logdensityof(iid(Normal(mu = mu, sigma = sigma), N), samples)
"#;
    let emit = |enabled: bool| {
        let mut m = flatppl_syntax::parse(TARGET).unwrap();
        flatppl_infer::infer(&mut m);
        let options = EmitOptions {
            target: Some("lp".into()),
            sufficient_statistics: enabled,
            ..Default::default()
        };
        flatppl_stan::emit(&m, &options).unwrap()
    };
    let (on, off) = (emit(true), emit(false));
    assert_eq!(on.data, off.data);
    assert_eq!(on.manifest, off.manifest);
    assert!(
        off.code.contains("samples ~ normal(mu, sigma);"),
        "{}",
        off.code
    );
    let (setup, model) = on.code.split_once("model {").unwrap();
    assert!(setup.contains("transformed data {"), "{}", on.code);
    // The summaries are data: they never read a parameter.
    let tdata = setup.split("parameters {").next().unwrap();
    assert!(
        !tdata
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|w| w == "mu" || w == "sigma"),
        "{}",
        on.code
    );
    assert!(
        model.contains("target += normal_lpdf(samples | normal_mu, normal_sigma);"),
        "{}",
        on.code
    );
}
