//! `pushfwd(f, M)` with a built-in function named as the map (§06: engines
//! recognize the bijections by name): the variate's type is the call of `f`
//! at M's domain, the rule `weighted` already applies to a built-in weight.

use flatppl_core::{Dim, ScalarType, Type};
use flatppl_infer::{Severity, infer};

/// The domain of binding `M` and whether inference reported an error.
fn domain(source: &str) -> (Option<Type>, bool) {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = infer(&mut module);
    let error = diagnostics.iter().any(|d| d.severity == Severity::Error);
    let rhs = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == "M")
        .unwrap()
        .1
        .rhs;
    let domain = match module.type_of(rhs) {
        Some(Type::Measure { domain, .. }) => Some(domain.as_ref().clone()),
        _ => None,
    };
    (domain, error)
}

#[test]
fn a_builtin_map_types_the_variate_at_the_base_domain() {
    for source in [
        "M = pushfwd(log, Exponential(1.5))",
        "M = pushfwd(exp, Normal(0, 1))",
        "M = pushfwd(sqrt, InverseGamma(5, 5))",
        "M = pushfwd(neg, Exponential(2))",
    ] {
        let (d, error) = domain(source);
        assert!(!error, "{source}");
        assert_eq!(d, Some(Type::Scalar(ScalarType::Real)), "{source}");
    }
}

#[test]
fn a_builtin_map_of_an_integer_base_keeps_its_integer_type() {
    // `neg` of a count is an integer: the base's reference measure survives.
    let (d, error) = domain("M = pushfwd(neg, Poisson(3.0))");
    assert!(!error);
    assert_eq!(d, Some(Type::Scalar(ScalarType::Integer)));
}

#[test]
fn a_builtin_map_over_a_vector_law_applies_per_cell() {
    // §06 has no sentence for a bare scalar name over an array law; the
    // determinizer reads it elementwise, and inference types it the same way:
    // the variate of `broadcast(f, x)`.
    let real = |n| Type::Array {
        shape: vec![Dim::Static(n)].into(),
        elem: Box::new(Type::Scalar(ScalarType::Real)),
    };
    for (source, want) in [
        ("M = pushfwd(exp, iid(Normal(0, 1), 3))", real(3)),
        (
            "M = pushfwd(exp, MvNormal(mu = [0.0, 0.0], cov = rowstack([[1.0, 0.0], [0.0, 1.0]])))",
            real(2),
        ),
        ("M = pushfwd(exp, Dirichlet([1.0, 2.0]))", real(2)),
    ] {
        let (d, error) = domain(source);
        assert!(!error, "{source}");
        assert_eq!(d, Some(want), "{source}");
    }
    let (d, error) = domain("M = pushfwd(neg, iid(Poisson(3.0), 3))");
    assert!(!error);
    assert_eq!(
        d,
        Some(Type::Array {
            shape: vec![Dim::Static(3)].into(),
            elem: Box::new(Type::Scalar(ScalarType::Integer)),
        })
    );
}

#[test]
fn a_builtin_map_over_a_matrix_law_types_as_its_elementwise_form() {
    // Any array variate types as `exp.(_)` does; whether a matrix lowers is
    // the determinizer's call (it refuses, as for the written form).
    let (d, error) = domain("M = pushfwd(exp, iid(Normal(0, 1), [2, 3]))");
    assert!(!error);
    assert_eq!(
        d,
        Some(Type::Array {
            shape: vec![Dim::Static(2), Dim::Static(3)].into(),
            elem: Box::new(Type::Scalar(ScalarType::Real)),
        })
    );
}

#[test]
fn a_builtin_map_with_an_array_domain_takes_the_whole_variate() {
    // §08: `InverseWishart` is `pushfwd(inv, Wishart(…))` and `LKJ` is
    // `pushfwd(row_gram, LKJCholesky(…))`. A name whose §07 domain is an
    // array is called with the whole variate, never per cell.
    for source in [
        "S = rowstack([[1.0, 0.2], [0.2, 1.0]])\nM = pushfwd(inv, Wishart(4.0, inv(S)))",
        "M = pushfwd(row_gram, LKJCholesky(3, 2.0))",
        "M = pushfwd(cumsum, iid(Normal(0, 1), 3))",
        "M = pushfwd(reverse, iid(Normal(0, 1), 3))",
        "M = pushfwd(softmax, iid(Normal(0, 1), 3))",
        "M = pushfwd(transpose, iid(Normal(0, 1), 3))",
    ] {
        let (d, error) = domain(source);
        assert!(!error, "{source}");
        assert!(
            !matches!(d, None | Some(Type::Any) | Some(Type::Scalar(_))),
            "{source}: {d:?}"
        );
    }
    let (d, error) = domain("M = pushfwd(sum, iid(Normal(0, 1), 3))");
    assert!(!error);
    assert_eq!(d, Some(Type::Scalar(ScalarType::Real)));
}

#[test]
fn a_two_argument_name_over_a_vector_law_is_an_arity_error() {
    // Not a one-argument scalar function, so no per-cell reading: the call
    // check reports the arity, as on main.
    let mut module = flatppl_syntax::parse("M = pushfwd(atan2, iid(Normal(0, 1), 3))").unwrap();
    let errors: Vec<String> = infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect();
    assert!(
        errors
            .iter()
            .any(|m| m.contains("`atan2` takes 2 arguments (spec §07), got 1")),
        "{errors:?}"
    );
}

#[test]
fn a_distribution_name_as_the_map_keeps_main_typing() {
    // A distribution is no function map; the variate stays untyped, without
    // an inference error, as on main.
    for source in [
        "M = pushfwd(Exponential, iid(Normal(0, 1), 3))",
        "M = pushfwd(Exponential, Normal(0, 1))",
    ] {
        let (d, error) = domain(source);
        assert!(!error, "{source}");
        assert_eq!(d, Some(Type::Any), "{source}");
    }
}

#[test]
fn a_builtin_map_over_a_nested_array_or_a_record_is_an_error() {
    // A nested array fails as `exp.(_)` does (a cell is a vector); a record is
    // no array, so the call check applies.
    let nested = "M = pushfwd(exp, iid(iid(Normal(0, 1), 2), 3))";
    let written = "M = pushfwd(fn(exp.(_)), iid(iid(Normal(0, 1), 2), 3))";
    for source in [nested, written, "M = pushfwd(exp, joint(a = Normal(0, 1)))"] {
        let (_, error) = domain(source);
        assert!(error, "{source}");
    }
}

#[test]
fn a_constant_as_the_map_is_an_error() {
    let (_, error) = domain("M = pushfwd(pi, Normal(0, 1))");
    assert!(error);
}

#[test]
fn a_builtin_map_outside_its_domain_is_left_to_the_engine() {
    // Inference types the call; whether the base support lies within the
    // map's domain (§06) is the density engine's check.
    let (d, error) = domain("M = pushfwd(log, Normal(0, 1))");
    assert!(!error);
    assert_eq!(d, Some(Type::Scalar(ScalarType::Real)));
}

#[test]
fn a_builtin_map_of_two_arguments_is_an_error() {
    let (_, error) = domain("M = pushfwd(pow, Gamma(2, 1))");
    assert!(error);
}

#[test]
fn a_user_function_map_keeps_its_type() {
    let (d, error) = domain("f(x) = 2.0 * x + 1.0\nM = pushfwd(f, Normal(0, 1))");
    assert!(!error);
    assert_eq!(d, Some(Type::Scalar(ScalarType::Real)));
}
