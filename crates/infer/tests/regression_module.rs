use flatppl_core::{Dim, Mass, ScalarType, Type, ValueSet};
use flatppl_infer::{Level, ModuleBundle, Severity, infer_module};

#[test]
fn regression_constructors_keep_matrix_rows_and_coefficient_axes() {
    let cases = [
        ("BernoulliLogitGLM(x, 0.1, b)", ScalarType::Integer),
        ("BinomialLogitGLM(x, 4, 0.1, b)", ScalarType::Integer),
        (
            "CategoricalLogitGLM(x, [0.1, 0.2, -0.1], rowstack([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]))",
            ScalarType::Integer,
        ),
        (
            "NormalGLM(sigma = 1.0, beta = b, alpha = 0.1, x = x)",
            ScalarType::Real,
        ),
        ("PoissonLogGLM(x, 0.1, b)", ScalarType::Integer),
        (
            "NegativeBinomial2LogGLM(x, 0.1, b, 2.0)",
            ScalarType::Integer,
        ),
        ("StudentTGLM(x, 4.0, 0.1, b, 1.0)", ScalarType::Real),
        ("GammaLogGLM(x, 0.1, b, 2.0)", ScalarType::Real),
        ("OrderedLogitGLM(x, b, [-1.0, 1.0])", ScalarType::Integer),
        ("BetaLogitGLM(x, 0.1, b, 4.0)", ScalarType::Real),
    ];
    for (call, elem) in cases {
        let source = format!(
            "r = standard_module(\"regression\", \"0.1\")\nx = rowstack([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])\nb = [0.2, -0.3]\nlaw = r.{call}\ny ~ law\n"
        );
        let mut module = flatppl_syntax::parse(&source).unwrap();
        let diagnostics = infer_module(&mut module, &ModuleBundle::new(), Level::Normalization);
        assert!(
            !diagnostics.iter().any(|d| d.severity == Severity::Error),
            "{call}: {diagnostics:?}"
        );
        let (_, binding) = module
            .bindings()
            .find(|(_, b)| module.resolve(b.name) == "law")
            .unwrap();
        let expected = Type::Measure {
            domain: Box::new(Type::Array {
                shape: Box::new([Dim::Static(3)]),
                elem: Box::new(Type::Scalar(elem)),
            }),
            mass: Mass::Normalized,
        };
        assert_eq!(module.type_of(binding.rhs), Some(&expected), "{call}");
    }
}

#[test]
fn regression_predictor_axis_mismatch_is_a_shape_error() {
    for call in [
        "NormalGLM(x, 0.0, [1.0], 1.0)",
        "CategoricalLogitGLM(x, [0.0, 0.0], rowstack([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))",
    ] {
        let mut module = flatppl_syntax::parse(&format!("r = standard_module(\"regression\", \"0.1\")\nx = rowstack([[1.0, 2.0], [3.0, 4.0]])\nlaw = r.{call}\n")).unwrap();
        let diagnostics = infer_module(&mut module, &ModuleBundle::new(), Level::Shape);
        assert!(
            diagnostics.iter().any(|d| d.severity == Severity::Error),
            "{call}: {diagnostics:?}"
        );
    }
}

#[test]
fn lowered_module_sampler_keeps_the_vector_variate() {
    let source = "x = rowstack([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])\nparameters = record(x = x, alpha = 0.1, beta = [0.2, -0.3], sigma = 1.0)\nsample = builtin_sample(rnginit(0), NormalGLM, parameters)[1]\n";
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = infer_module(&mut module, &ModuleBundle::new(), Level::Shape);
    assert!(
        !diagnostics.iter().any(|d| d.severity == Severity::Error),
        "{diagnostics:?}"
    );
    let (_, binding) = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == "sample")
        .unwrap();
    assert_eq!(
        module.type_of(binding.rhs),
        Some(&Type::Array {
            shape: Box::new([Dim::Static(3)]),
            elem: Box::new(Type::Scalar(ScalarType::Real)),
        })
    );
}

#[test]
fn bounded_regression_supports_keep_trial_and_class_bounds() {
    for (call, lo, hi) in [
        ("BinomialLogitGLM(x, trials, 0.1, [0.2, -0.3])", 0.0, 4.0),
        (
            "CategoricalLogitGLM(x, [0.1, -0.2], rowstack([[0.2, -0.3], [0.3, -0.1]]))",
            1.0,
            2.0,
        ),
        ("OrderedLogitGLM(x, [0.2, -0.3], [-1.0, 1.0])", 1.0, 3.0),
    ] {
        let mut module = flatppl_syntax::parse(&format!("r = standard_module(\"regression\", \"0.1\")\nx = rowstack([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])\ntrials = 4\nlaw = r.{call}\n")).unwrap();
        let diagnostics = infer_module(&mut module, &ModuleBundle::new(), Level::Shape);
        assert!(
            !diagnostics.iter().any(|d| d.severity == Severity::Error),
            "{diagnostics:?}"
        );
        let (_, binding) = module
            .bindings()
            .find(|(_, b)| module.resolve(b.name) == "law")
            .unwrap();
        assert_eq!(
            module.valueset_of(binding.rhs),
            Some(&ValueSet::CartPow(
                Box::new(ValueSet::Interval(lo, hi)),
                Dim::Static(3)
            )),
            "{call}"
        );
    }
}
