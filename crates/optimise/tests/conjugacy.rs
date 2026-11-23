use flatppl_optimise::{Profile, optimise};

const MODEL: &str = "\
point = external(reals)
prior_sigma = 3.0
m = kchain(Normal(mu = -1.5, sigma = prior_sigma),
           kernelof(draw(Normal(mu = _m_, sigma = 4.0)), m = _m_))
inputs = point
outputs = (logdensityof(m, point), totalmass(m))
";

const UNIT_TRIAL: &str = "\
flag = external(booleans)
point = ifelse(flag, 1, 0)
trials = 1
m = kchain(Beta(alpha = 2.0, beta = 3.0),
           kernelof(draw(Binomial(p = _p_, n = trials)), p = _p_))
state = rnginit([1, 2, 3, 4, 5, 6, 7, 8])
sample, successor = rand(m, state)
inputs = flag
outputs = (logdensityof(m, point), densityof(m, point), totalmass(m))
";

fn rendered(source: &str) -> String {
    rendered_module(flatppl_syntax::parse(source).unwrap())
}

fn rendered_module(mut module: flatppl_core::Module) -> String {
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    let optimised = optimise(&module, None, Profile::Portable);
    flatppl_syntax::print(optimised.as_ref().unwrap_or(&module))
}

#[test]
fn normal_predictive_changes_the_score_conserving_the_measure() {
    let out = rendered(MODEL);
    assert!(
        out.contains("logdensityof(Normal(-1.5, 5.0), point)"),
        "{out}"
    );
    assert!(out.contains("m = kchain("), "{out}");
    assert!(out.contains("totalmass(m)"), "{out}");
}

#[test]
fn normal_predictive_keeps_a_prior_scale_supplied_at_runtime() {
    let out = rendered(&MODEL.replace("inputs = point", "inputs = (point, prior_sigma)"));
    assert!(out.contains("logdensityof(m, point)"), "{out}");
    assert!(out.contains("inputs = (point, prior_sigma)"), "{out}");
}

#[test]
fn unit_trial_predictive_shares_binary_score_proof_across_flatppl_and_flatpir() {
    let module = flatppl_syntax::parse(UNIT_TRIAL).unwrap();
    let ir = flatppl_flatpir::write(&module);
    for output in [
        rendered(UNIT_TRIAL),
        rendered_module(flatppl_flatpir::read(&ir).unwrap()),
    ] {
        assert!(
            output.contains("logdensityof(Bernoulli(0.4), point)"),
            "{output}"
        );
        assert!(
            output.contains(", densityof(Bernoulli(0.4), point)"),
            "{output}"
        );
        assert!(output.contains("Binomial("), "{output}");
        assert!(output.contains("totalmass(m)"), "{output}");
        assert!(output.contains("rand(m, state)"), "{output}");
        assert!(output.contains("inputs = flag"), "{output}");
    }
}

#[test]
fn unit_trial_predictive_requires_a_fixed_trial_count_and_binary_support() {
    let cases = [
        UNIT_TRIAL.replace("trials = 1", "trials = ifelse(flag, 1, 2)"),
        UNIT_TRIAL.replace("inputs = flag", "inputs = (flag, trials)"),
        UNIT_TRIAL.replace("point = ifelse(flag, 1, 0)", "point = ifelse(flag, -1, 0)"),
        UNIT_TRIAL.replace("point = ifelse(flag, 1, 0)", "point = ifelse(flag, 1, 2)"),
        UNIT_TRIAL.replace("inputs = flag", "inputs = (flag, point)"),
        UNIT_TRIAL.replace(
            "alpha = 2.0, beta = 3.0",
            "alpha = 4194304.0, beta = 4194304.0",
        ),
        UNIT_TRIAL.replace(
            "alpha = 2.0, beta = 3.0",
            "alpha = 8.673617379884035e-19, beta = 1.0",
        ),
        UNIT_TRIAL.replace(
            "alpha = 2.0, beta = 3.0",
            "alpha = 1.7014118346046923e38, beta = 1.7014118346046923e38",
        ),
    ];
    for source in cases {
        let output = rendered(&source);
        assert!(
            output.contains("logdensityof(m, point)"),
            "{source}\n{output}"
        );
    }
}
