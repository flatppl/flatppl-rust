use flatppl_core::{Phase, Type};
use flatppl_infer::{Severity, infer};

#[test]
fn explicit_weights_keep_the_parameter_phase_and_measure_domain() {
    for op in ["weighted", "logweighted"] {
        let source = format!("p ~ Normal(0, 1)\nf(x, p) = exp(p*x)\nM = {op}(f, Normal(0, 1), p)");
        let mut module = flatppl_syntax::parse(&source).unwrap();
        let diagnostics = infer(&mut module);
        assert!(
            !diagnostics.iter().any(|d| d.severity == Severity::Error),
            "{diagnostics:?}"
        );
        let rhs = module
            .bindings()
            .find(|(_, b)| module.resolve(b.name) == "M")
            .unwrap()
            .1
            .rhs;
        assert!(matches!(module.type_of(rhs), Some(Type::Measure { .. })));
        assert_eq!(module.phase_of(rhs), Some(Phase::Stochastic));
    }
}

#[test]
fn explicit_weights_require_a_scalar_two_input_function() {
    for source in [
        "M = weighted(2, Normal(0, 1), 3)",
        "M = logweighted(x -> x, Normal(0, 1), 3)",
        "M = weighted((x, y, z) -> x + y + z, Normal(0, 1), 3)",
        "M = weighted((x, p) -> x * p, iid(Normal(0, 1), 2), 3)",
        "f = lt\nM = weighted(f, Uniform(unitinterval), 0.5)",
        "M = weighted(sin, Uniform(unitinterval), 2)",
        "p ~ Normal(0, 1)\nM = logweighted((x, q) -> p*x + q, Normal(0, 1), 3)",
    ] {
        let mut module = flatppl_syntax::parse(source).unwrap();
        let diagnostics = infer(&mut module);
        assert!(
            diagnostics.iter().any(|d| d.severity == Severity::Error),
            "{source}: {diagnostics:?}"
        );
    }
}
