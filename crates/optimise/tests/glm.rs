use flatppl_optimise::{Dtype, Profile, Target, optimise};

// A profile that seeds no data boundary keeps the broadcast law, so its own
// reductions of the broadcast form still apply.
#[test]
fn profiles_without_a_data_boundary_keep_the_broadcast_law() {
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
    for profile in [
        Profile::Portable,
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
