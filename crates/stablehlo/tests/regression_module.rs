//! §09 vector response laws use one link-scale score and one independent
//! response draw. Numerical oracle checks run through the emitted CPU/GPU ABI.

use flatppl_infer::ModuleBundle;

const MODELS: &[(&str, &str)] = &[
    ("BernoulliLogitGLM(x, alpha, beta)", "[1, 0, 1]"),
    ("BinomialLogitGLM(x, 4, alpha, beta)", "[1, 2, 3]"),
    (
        "CategoricalLogitGLM(x, [0.1, -0.2, 0.3], rowstack([[0.2, -0.5, 0.1], [-0.1, 0.4, 0.3]]))",
        "[1, 2, 3]",
    ),
    ("NormalGLM(x, alpha, beta, 1.3)", "[0.5, -1.0, 2.0]"),
    ("PoissonLogGLM(x, alpha, beta)", "[0, 1, 3]"),
    ("NegativeBinomial2LogGLM(x, alpha, beta, 2.0)", "[0, 1, 3]"),
    ("StudentTGLM(x, 1.0, alpha, beta, 1.3)", "[0.5, -1.0, 2.0]"),
    ("GammaLogGLM(x, alpha, beta, 2.0)", "[0.5, 1.0, 2.0]"),
    ("OrderedLogitGLM(x, beta, [-0.5, 0.8])", "[1, 2, 3]"),
    ("BetaLogitGLM(x, alpha, beta, 4.0)", "[0.2, 0.5, 0.8]"),
];

fn emit(source: &str) -> String {
    emit_with_restrictions(source, false)
}

fn emit_with_restrictions(source: &str, restricted: bool) -> String {
    emit_for(source, restricted, flatppl_stablehlo::Target::Cpu)
}

fn emit_for(source: &str, restricted: bool, target: flatppl_stablehlo::Target) -> String {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let (_, bundle) = flatppl_infer::infer_module_with_inputs(
        &mut module,
        ModuleBundle::new(),
        flatppl_infer::Level::Shape,
    );
    let roots = [module.intern("inputs"), module.intern("outputs")];
    let flat = flatppl_determinizer::determinize_with_options(
        &module,
        &bundle,
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .unwrap();
    flatppl_stablehlo::emit(
        &flat,
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions {
            restrict_enzyme_compatible: restricted,
            target,
            ..Default::default()
        },
    )
    .unwrap()
}

const PREFIX: &str = "reg = standard_module(\"regression\", \"0.1\")\n\
    x = external(cartpow(reals, [3, 2]))\n\
    alpha = elementof(reals)\n\
    beta = elementof(cartpow(reals, 2))\n";

#[test]
fn ten_qualified_laws_emit_one_vector_score_and_independent_draws() {
    for &(ctor, response) in MODELS {
        let score = emit(&format!(
            "{PREFIX}inputs=(x,alpha,beta)\n\
            outputs=logdensityof(reg.{ctor},{response})"
        ));
        assert!(score.contains("stablehlo.dot_general"), "{ctor}");
        assert!(score.contains("-> tensor<f32>"), "{ctor}\n{score}");
        for count in ["", "5"] {
            let law = if count.is_empty() {
                format!("reg.{ctor}")
            } else {
                format!("iid(reg.{ctor}, 5)")
            };
            let draw = emit(&format!(
                "{PREFIX}r=rnginit(41)\ninputs=(x,alpha,beta,r)\n\
                outputs=rand({law},r)"
            ));
            let shape = if count.is_empty() {
                "tensor<3xf32>"
            } else {
                "tensor<5x3xf32>"
            };
            assert!(draw.contains(shape), "{ctor}, fanout={count}\n{draw}");
            assert!(draw.contains("rng_bit_generator"), "{ctor}");
        }
    }
}

#[test]
fn link_scale_density_keeps_logit_tails_before_probability_rounding() {
    let source = "reg=standard_module(\"regression\",\"0.1\")\n\
        x=rowstack([[1.0],[1.0]])\n\
        beta=elementof(cartpow(reals,1))\ninputs=beta\n\
        outputs=logdensityof(reg.BernoulliLogitGLM(x,0.0,beta),[0,1])";
    let mlir = emit(source);
    assert!(mlir.contains("stablehlo.log_plus_one"), "{mlir}");
    assert!(!mlir.contains("chlo.logistic"), "{mlir}");
}

#[test]
fn ordered_score_uses_log_expm1_of_the_cutpoint_gap() {
    let mlir = emit(&format!(
        "{PREFIX}cuts=elementof(ordered(2))\n\
        inputs=(x,beta,cuts)\n\
        outputs=logdensityof(reg.OrderedLogitGLM(x,beta,cuts),[1,2,3])"
    ));
    assert!(mlir.contains("stablehlo.exponential_minus_one"), "{mlir}");
    assert!(mlir.contains("stablehlo.gather"), "{mlir}");
}

#[test]
fn qualified_vector_laws_keep_ordinary_outer_broadcast() {
    for law in [
        "reg.NormalGLM.([x,x], 0.0, [beta,beta], 1.0)",
        "reg.CategoricalLogitGLM.([x,x], [[0.1,-0.2,0.3]], [rowstack([[0.2,-0.5,0.1],[-0.1,0.4,0.3]])])",
        "reg.OrderedLogitGLM.([x,x], [beta,beta], [[-0.5,0.8]])",
    ] {
        let mlir = emit(&format!(
            "{PREFIX}inputs=(x,beta)\nM={law}\noutputs=logdensityof(M,[[1,2,3],[3,2,1]])"
        ));
        assert!(mlir.contains("-> tensor<f32>"), "{law}\n{mlir}");
    }
}

#[test]
fn guarded_student_and_beta_scores_emit_with_default_restrictions() {
    for law in [
        "reg.StudentTGLM(x, nu, alpha, beta, 1.0)",
        "reg.BetaLogitGLM(x, alpha, beta, nu)",
    ] {
        let source = format!(
            "{PREFIX}nu=elementof(posreals)\ninputs=(x,alpha,beta,nu)\noutputs=logdensityof({law},[0.2,0.5,0.8])"
        );
        for target in [
            flatppl_stablehlo::Target::Cpu,
            flatppl_stablehlo::Target::Gpu,
        ] {
            emit_for(&source, true, target);
        }
    }
}

#[test]
fn guarded_scores_keep_outer_scalar_parameter_cells() {
    for law in [
        "reg.StudentTGLM.([x,x], [1.0,1000.0], 0.0, [beta,beta], 1.0)",
        "reg.BetaLogitGLM.([x,x], 0.0, [beta,beta], [6.0,1000.0])",
    ] {
        let source = format!(
            "{PREFIX}inputs=(x,beta)\nM={law}\noutputs=logdensityof(M,[[0.2,0.5,0.8],[0.8,0.5,0.2]])"
        );
        let mlir = emit_with_restrictions(&source, true);
        assert!(mlir.contains("\"stablehlo.case\""), "{law}\n{mlir}");
        // Every case predicate is a scalar i32, even when scalar kernel
        // parameters vary across the ordinary outer collection axis.
        assert!(mlir.contains(") : (tensor<i32>) ->"), "{law}\n{mlir}");
    }
}
