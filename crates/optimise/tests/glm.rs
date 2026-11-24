use flatppl_optimise::{Dtype, Profile, Target, optimise};

// The portable stage rewrites the broadcast law for every target that lowers
// from it. The StableHLO stage runs on a lowered query and has no such law.
#[test]
fn the_portable_stage_recognises_a_regression_law() {
    let source = "X = rowstack([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]])
alpha = elementof(reals)
beta = elementof(cartpow(reals, 2))
sigma = elementof(posreals)
law = Normal.(alpha .+ X * beta, sigma)
inputs = (alpha, beta, sigma)
outputs = logdensityof(law, [1.0, 2.0, 3.0])
";
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    let outputs = module.intern("outputs");
    let portable = optimise(&module, Some(&[outputs]), Profile::Portable).expect("rewrite");
    let text = flatppl_syntax::print(&portable);
    assert!(text.contains("NormalGLM(X, alpha, beta, sigma)"), "{text}");
    for profile in [
        Profile::StableHlo {
            target: Target::Cpu,
            dtype: Dtype::F64,
        },
        Profile::StableHlo {
            target: Target::Gpu,
            dtype: Dtype::F32,
        },
    ] {
        if let Some(out) = optimise(&module, Some(&[outputs]), profile) {
            let text = flatppl_syntax::print(&out);
            assert!(!text.contains("GLM"), "{profile:?}:\n{text}");
        }
    }
}
