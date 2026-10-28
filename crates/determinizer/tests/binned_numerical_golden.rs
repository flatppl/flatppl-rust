//! Explicit numerical bin masses for function-weighted Lebesgue intervals.

use flatppl_determinizer::{LoweringOptions, determinize, determinize_with_options};

fn parse(src: &str) -> flatppl_core::Module {
    let mut module = flatppl_syntax::parse(src).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    module
}

fn numerical(src: &str) -> String {
    let output = determinize_with_options(
        &parse(src),
        &flatppl_infer::ModuleBundle::new(),
        None,
        &LoweringOptions {
            numerical_integrals: true,
            ..Default::default()
        },
    )
    .expect("explicit numerical lowering");
    assert!(
        flatppl_determinizer::is_flatpdl_with_options(
            &output,
            &LoweringOptions {
                numerical_integrals: true,
                ..Default::default()
            }
        )
        .is_ok()
    );
    assert!(flatppl_determinizer::is_flatpdl(&output).is_err());
    flatppl_flatpir::write(&output)
}

#[test]
fn function_weights_require_opt_in_and_clip_each_bin() {
    for weight in [
        "weighted((x,p) -> exp(p*x), Lebesgue(interval(0, 1)), a)",
        "logweighted((x,p) -> p*x, Lebesgue(interval(0, 1)), a)",
    ] {
        let source = format!(
            "a = elementof(reals)\nI = {weight}\nJ = I\nlp = logdensityof(BinnedPoissonProcess([-1,0,0.5,1,2], J), [0,1,2,0])\ninputs = a\noutputs = lp"
        );
        let error = determinize(&parse(&source)).expect_err("numerical path requires opt-in");
        assert!(
            error
                .reason
                .contains("numerical integration requires explicit opt-in"),
            "{error:?}"
        );
        let text = numerical(&source);
        let intrinsic = if weight.starts_with("logweighted") {
            "%log_integral"
        } else {
            "%mass_integral"
        };
        assert_eq!(text.matches(intrinsic).count(), 3, "{text}");
        assert!(text.contains("(%ref self a)"), "live extra: {text}");
        assert!(!text.contains("BinnedPoissonProcess"), "{text}");
    }
}

#[test]
fn dynamic_bounds_and_other_references_still_refuse() {
    for source in [
        "a = elementof(reals)\nI = weighted(x -> exp(x), Lebesgue(interval(0,a)))\nlp = logdensityof(BinnedPoissonProcess([0,1], I), [0])\ninputs = a\noutputs = lp",
        "a = elementof(reals)\nI = weighted(x -> exp(x), Lebesgue(interval(0,1)))\nlp = logdensityof(BinnedPoissonProcess([0,a], I), [0])\ninputs = a\noutputs = lp",
        "I = weighted(x -> exp(x), Lebesgue(reals))\nlp = logdensityof(BinnedPoissonProcess([0,1], I), [0])",
        "I = weighted(x -> exp(x), Normal(0,1))\nlp = logdensityof(BinnedPoissonProcess([0,1], I), [0])",
    ] {
        let result = determinize_with_options(
            &parse(source),
            &flatppl_infer::ModuleBundle::new(),
            None,
            &LoweringOptions {
                numerical_integrals: true,
                ..Default::default()
            },
        );
        assert!(result.is_err(), "unsupported shape: {source}");
    }
}

#[test]
fn disjoint_bins_have_only_the_full_support_integral() {
    let source = "a = elementof(reals)\nI = weighted((x,p) -> exp(p*x), Lebesgue(interval(5,6)), a)\nlp = logdensityof(BinnedPoissonProcess([0,1,2], I), [0,0])\ninputs = a\noutputs = lp";
    let text = numerical(source);
    assert_eq!(text.matches("%mass_integral").count(), 1, "{text}");
    assert!(text.contains("(vector 0.0 0.0)"), "{text}");
}
