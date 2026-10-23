//! Spec §06 `likelihoodof`: the parameter point of a likelihood is a record with one
//! field per input of `K`, bound to those inputs by name for any number of inputs, so
//! a kernel with the single input `theta` is evaluated at `record(theta = 0.3)`, not at
//! `0.3`. The lowering reads each input from its field.
use flatppl_determinizer::determinize;

fn lower(src: &str) -> Result<String, String> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    determinize(&m)
        .map(|out| flatppl_flatpir::write(&out))
        .map_err(|e| format!("{e:?}"))
}

const ONE_INPUT: &str = "\
theta = elementof(reals)
K = functionof(Normal(mu = theta, sigma = 1.0), theta = theta)
L = likelihoodof(K, 0.5)
";

#[test]
fn a_one_input_kernel_reads_its_input_from_the_named_field() {
    let by_name = lower(&format!(
        "{ONE_INPUT}lp = logdensityof(L, record(theta = 0.3))"
    ))
    .unwrap();
    let direct =
        lower("lp = builtin_logdensityof(Normal, record(mu = 0.3, sigma = 1.0), 0.5)").unwrap();
    // The scored density is the kernel at theta = 0.3, observed at 0.5.
    let value = |pir: &str| {
        let at = pir.find("(%bind lp").expect("lp binding");
        pir[at..].to_string()
    };
    assert_eq!(value(&by_name), value(&direct));
}

#[test]
fn a_one_input_kernel_scores_a_posterior_at_the_prior_record() {
    let pir = lower(&format!(
        "{ONE_INPUT}prior = joint(theta = Normal(mu = 0.0, sigma = 1.0))\n\
         post = bayesupdate(L, prior)\nlp = logdensityof(post, record(theta = 0.3))"
    ))
    .unwrap();
    assert_eq!(pir.matches("builtin_logdensityof").count(), 2, "{pir}");
    assert!(pir.contains("(%field mu 0.3)"), "{pir}");
}

#[test]
fn a_field_that_names_no_input_refuses() {
    let err = lower(&format!(
        "{ONE_INPUT}lp = logdensityof(L, record(phi = 0.3))"
    ))
    .unwrap_err();
    assert!(err.contains("missing a kernel input"), "{err}");
}
