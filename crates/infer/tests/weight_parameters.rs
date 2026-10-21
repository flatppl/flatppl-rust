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
        "M = weighted((x, y) -> x*y, iid(Normal(0, 1), 2))",
        "M = pushfwd((x, y) -> x+y, Normal(0,1))",
        "M = markovchain((s, x) -> Normal(s+x,1), 0.0, 3)",
        "M = kscan(s -> Normal(s,1), 0.0, [1.0,2.0])",
        "x = scan((a,x) -> a+x, 0.0, [1.0,2.0], 3)",
        "x = reduce(x -> x, [1.0,2.0])",
        "x = filter((x,p) -> x<p, [1.0,2.0])",
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

#[test]
fn callback_parameters_preserve_whole_objects_and_sequence_shapes() {
    for source in [
        "p = elementof(reals)\ny = filter((x, q) -> x < q, [1.0, 2.0], p)",
        "p = elementof(reals)\ny = reduce((a, x, q) -> a + x + q, [1.0, 2.0], p)",
        "p = elementof(reals)\ny = scan((a, x, q) -> a + x + q, 0.0, [1.0, 2.0], p)",
        "p = elementof(reals)\ny = pushfwd((x, q) -> x + q[1], Normal(0, 1), (p, 3))",
        "p = elementof(reals)\ny = weighted((x, q) -> densityof(q, x), Normal(0, 1), Normal(p, 1))",
        "y = weighted(r -> exp(r.x), joint(x=Normal(0, 1)))",
    ] {
        let mut module = flatppl_syntax::parse(source).unwrap();
        let diagnostics = infer(&mut module);
        assert!(
            !diagnostics.iter().any(|d| d.severity == Severity::Error),
            "{source}: {diagnostics:?}"
        );
    }
}
