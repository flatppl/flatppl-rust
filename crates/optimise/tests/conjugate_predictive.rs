use flatppl_core::Module;
use flatppl_optimise::{Profile, optimise};

const COUNTS: &str = r#"flag = external(booleans)
also = external(booleans)
trials = 7
alpha = 2.0
point = ifelse(flag, 7, ifelse(also, 3, 0))
m = kchain(Beta(alpha, 3.0), kernelof(draw(Binomial(trials, _p_)), p = _p_))
state = rnginit([1, 2, 3, 4, 5, 6, 7, 8])
sample, successor = rand(m, state)
inputs = (flag, also)
outputs = (logdensityof(m, point), densityof(m, point), totalmass(m))
"#;

const VARIANCE: &str = r#"point = external(reals)
location = -1.25
rate = 2.0
beta = 12.0
laplace_chain = kchain(Exponential(rate),
    kernelof(draw(Normal(mu = location, sigma = sqrt(_v_))), v = _v_))
student_chain = kchain(InverseGamma(scale = beta, shape = 3.0),
    kernelof(draw(Normal(location, sqrt(_v_))), v = _v_))
state = rnginit([1, 2, 3, 4, 5, 6, 7, 8])
sample, successor = rand(student_chain, state)
inputs = point
outputs = (logdensityof(laplace_chain, point), densityof(student_chain, point),
    logdensityof(student_chain, point), totalmass(laplace_chain), totalmass(student_chain))
"#;

fn rendered(mut module: Module) -> String {
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    let change = optimise(&module, None, Profile::Portable);
    let result = change.as_ref().unwrap_or(&module);
    let ir = flatppl_flatpir::write(result);
    flatppl_flatpir::read(&ir).expect("optimized FlatPIR must parse");
    flatppl_syntax::print(result)
}

fn from_source(source: &str) -> String {
    rendered(flatppl_syntax::parse(source).unwrap())
}

#[test]
fn beta_binomial_scores_keep_only_the_certified_count_alternatives() {
    let module = flatppl_syntax::parse(COUNTS).unwrap();
    let ir = flatppl_flatpir::write(&module);
    for result in [
        rendered(module),
        rendered(flatppl_flatpir::read(&ir).unwrap()),
    ] {
        assert!(result.contains("point == 0"), "{result}");
        assert!(result.contains("point == 3"), "{result}");
        assert!(!result.contains("logdensityof(m, point)"), "{result}");
        assert!(!result.contains("densityof(m, point)"), "{result}");
        assert!(result.contains("m = kchain("), "{result}");
        assert!(result.contains("totalmass(m)"), "{result}");
        assert!(result.contains("rand(m, state)"), "{result}");
        assert!(result.contains("inputs = (flag, also)"), "{result}");
    }
    let literal = from_source(&COUNTS.replace("ifelse(flag, 7, ifelse(also, 3, 0))", "3"));
    assert!(!literal.contains("logdensityof(m, point)"), "{literal}");
    assert!(literal.contains("totalmass(m)"), "{literal}");
}

#[test]
fn beta_binomial_preserves_the_fallback_when_any_outcome_is_unproven() {
    let cases = [
        COUNTS.replace("ifelse(also, 3, 0)", "ifelse(also, 8, 0)"),
        COUNTS.replace("ifelse(also, 3, 0)", "ifelse(also, -1, 0)"),
        COUNTS.replace("trials = 7", "trials = ifelse(flag, 7, 8)"),
        COUNTS.replace("trials = 7", "trials = 4096"),
        COUNTS.replace("alpha = 2.0", "alpha = 1073741824.0"),
        COUNTS.replace("inputs = (flag, also)", "inputs = (flag, also, trials)"),
        COUNTS.replace("inputs = (flag, also)", "inputs = (flag, also, alpha)"),
        COUNTS.replace("inputs = (flag, also)", "inputs = (flag, also, point)"),
        COUNTS.replace("ifelse(flag, 7, ifelse(also, 3, 0))", "ifelse(flag, 3, 3)"),
    ];
    for source in cases {
        let result = from_source(&source);
        assert!(result.contains("logdensityof(m, point)"), "{result}");
        assert!(result.contains("densityof(m, point)"), "{result}");
    }
}

#[test]
fn variance_mixtures_use_laplace_and_the_stable_student_score() {
    let module = flatppl_syntax::parse(VARIANCE).unwrap();
    let ir = flatppl_flatpir::write(&module);
    for result in [
        rendered(module),
        rendered(flatppl_flatpir::read(&ir).unwrap()),
    ] {
        assert!(
            result.contains("logdensityof(Laplace(-1.25, 0.5), point)"),
            "{result}"
        );
        assert!(result.contains("log1p("), "{result}");
        assert!(
            !result.contains("densityof(student_chain, point)"),
            "{result}"
        );
        assert!(result.contains("rand(student_chain, state)"), "{result}");
        assert!(result.contains("totalmass(laplace_chain)"), "{result}");
        assert!(result.contains("totalmass(student_chain)"), "{result}");
    }
}

#[test]
fn variance_mixtures_do_not_absorb_runtime_or_differently_wired_parameters() {
    for (replacement, preserved) in [
        (
            "inputs = (point, location)",
            "logdensityof(laplace_chain, point)",
        ),
        (
            "inputs = (point, rate)",
            "logdensityof(laplace_chain, point)",
        ),
        (
            "inputs = (point, beta)",
            "logdensityof(student_chain, point)",
        ),
    ] {
        let result = from_source(&VARIANCE.replace("inputs = point", replacement));
        assert!(result.contains(preserved), "{result}");
    }
    for sqrt in ["sqrt(_v_ + 0.25)", "_v_"] {
        let result = from_source(&VARIANCE.replace("sqrt(_v_)", sqrt));
        assert!(
            result.contains("logdensityof(laplace_chain, point)"),
            "{result}"
        );
        assert!(
            result.contains("logdensityof(student_chain, point)"),
            "{result}"
        );
    }
    let result = from_source(&VARIANCE.replace("shape = 3.0", "shape = 3.25"));
    assert!(
        result.contains("logdensityof(student_chain, point)"),
        "{result}"
    );
}
