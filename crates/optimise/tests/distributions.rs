use flatppl_core::graph::expect_builtin_call;
use flatppl_core::{Module, Node, NodeId, Scalar};
use flatppl_optimise::{Profile, optimise};

const BETA_CHAIN: &str =
    "kchain(Beta(alpha = 1.0, beta = 3.0), kernelof(draw(Bernoulli(p = _p_)), p = _p_))";
const GAMMA_CHAIN: &str =
    "kchain(Gamma(shape = 2.0, rate = 3.0), kernelof(draw(Poisson(rate = _r_)), rate = _r_))";
const INPUT: &str = "y = external(integers)\ninputs = y\n";

fn infer(source: &str) -> Module {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diagnostics:?}"
    );
    module
}

fn rhs(module: &Module, name: &str) -> NodeId {
    module
        .bindings()
        .find(|(_, binding)| module.resolve(binding.name) == name)
        .unwrap()
        .1
        .rhs
}

#[test]
fn closed_predictive_queries_reduce_to_their_exact_native_distributions() {
    for (measure, constructor, expected) in [
        (BETA_CHAIN, "Bernoulli", &[0.25][..]),
        (GAMMA_CHAIN, "NegativeBinomial", &[2.0, 3.0][..]),
    ] {
        for head in ["logdensityof", "densityof"] {
            let mut model = infer(&format!("{INPUT}outputs = {head}({measure}, y)\n"));
            let output = model.intern("outputs");
            let original = expect_builtin_call(&model, rhs(&model, "outputs"), head).unwrap();
            let out = optimise(&model, Some(&[output]), Profile::Portable)
                .expect("closed predictive score should reduce");
            let score = expect_builtin_call(&out, rhs(&out, "outputs"), head).unwrap();
            assert_eq!(out.node(score.args[1]), model.node(original.args[1]));
            let native = expect_builtin_call(&out, score.args[0], constructor).unwrap();
            assert_eq!(native.args.len(), expected.len());
            for (&arg, &value) in native.args.iter().zip(expected) {
                assert_eq!(out.node(arg), &Node::Lit(Scalar::Real(value)));
            }
        }
    }
}

#[test]
fn predictive_proofs_keep_weights_named_draws_and_runtime_hyperparameters() {
    for source in [
        format!("{INPUT}outputs = logdensityof(weighted(2.0, {BETA_CHAIN}), y)\n"),
        format!(
            "{INPUT}p ~ Beta(1.0, 3.0)\nz ~ Bernoulli(p)\noutputs = logdensityof(lawof(z), y)\n"
        ),
        format!(
            "y = external(integers)\na = elementof(posreals)\ninputs = (y, a)\noutputs = logdensityof({}, y)\n",
            BETA_CHAIN.replace("alpha = 1.0", "alpha = a")
        ),
    ] {
        let mut model = infer(&source);
        let output = model.intern("outputs");
        if let Some(out) = optimise(&model, Some(&[output]), Profile::Portable) {
            assert_eq!(rhs(&out, "outputs"), rhs(&model, "outputs"), "{source}");
            for (id, original) in model.bindings() {
                let updated = out.binding(id);
                assert_eq!(updated.rhs, original.rhs, "{source}");
            }
        }
    }
}

#[test]
fn reducing_a_score_preserves_a_shared_sampling_measure_and_origin() {
    let mut model = infer(&format!(
        "{INPUT}predictive = {BETA_CHAIN}\nsample ~ predictive\nscore = logdensityof(predictive, y)\noutputs = (score, sample)\n"
    ));
    let output = model.intern("outputs");
    let law = rhs(&model, "predictive");
    let sample = rhs(&model, "sample");
    model.trace_sample_origins(std::iter::once(law));
    model.record_sample_origin(sample, law);

    let out = optimise(&model, Some(&[output]), Profile::Portable)
        .expect("the score can reduce independently of the sample");
    let score = expect_builtin_call(&out, rhs(&out, "score"), "logdensityof").unwrap();
    assert!(expect_builtin_call(&out, score.args[0], "Bernoulli").is_some());
    assert_eq!(rhs(&out, "predictive"), law);
    assert_eq!(rhs(&out, "sample"), sample);
    assert_eq!(out.node(law), model.node(law));
    assert_eq!(out.node(sample), model.node(sample));
    assert_eq!(out.sample_origin(law), Some(law));
    assert_eq!(out.sampled_measure(sample), Some(law));
}
