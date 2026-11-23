use flatppl_optimise::{Profile, optimise};

const MODEL: &str = "\
point = external(reals)
prior_sigma = 3.0
m = kchain(Normal(mu = -1.5, sigma = prior_sigma),
           kernelof(draw(Normal(mu = _m_, sigma = 4.0)), m = _m_))
inputs = point
outputs = (logdensityof(m, point), totalmass(m))
";

fn rendered(source: &str) -> String {
    let mut module = flatppl_syntax::parse(source).unwrap();
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
