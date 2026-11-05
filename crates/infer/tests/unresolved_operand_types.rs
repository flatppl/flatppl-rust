//! An unresolved operand (`%any`, such as a lambda's own input) keeps an
//! arithmetic result unresolved. §07 gives `mul` a scalar-vector and a
//! scalar-matrix domain and `divide` an array-scalar domain, so `2.0 * x` and
//! `x / 2.0` may be arrays. `add`, `sub` and `pow` take real or complex
//! scalars, so `x + 1` may be complex. A lambda body is typed before any
//! argument is known, so none of these is a scalar of a fixed kind.
use flatppl_core::{Node, Type};
use flatppl_infer::{Severity, infer};

/// The inferred type of the body of the one-input callable bound to `name`.
fn body_type(source: &str, name: &str) -> Type {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let errors: Vec<_> = infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect();
    assert!(errors.is_empty(), "{source}: {errors:?}");
    let symbol = module.intern(name);
    let binding = module.binding_by_name(symbol).expect("binding");
    let Node::Call(call) = module.node(module.binding(binding).rhs) else {
        panic!("{name} is not a functionof");
    };
    module.type_of(call.args[0]).cloned().expect("typed body")
}

#[test]
fn a_scalar_times_or_into_an_unresolved_operand_is_unresolved() {
    for body in ["2.0 * x", "x * 2.0", "x / 2.0", "2 * x"] {
        assert_eq!(
            body_type(&format!("f = x -> {body}\n"), "f"),
            Type::Any,
            "{body}"
        );
    }
}

#[test]
fn a_sum_or_power_with_an_unresolved_operand_is_unresolved() {
    // The unresolved operand may be complex, so the result is not an integer.
    for body in ["x + 1", "1 - x", "x ^ 2", "(x - 1) + (x * 2.0) * im"] {
        assert_eq!(
            body_type(&format!("f = x -> {body}\n"), "f"),
            Type::Any,
            "{body}"
        );
    }
}

#[test]
fn a_scaled_lambda_input_plus_a_vector_is_not_a_domain_error() {
    // `2.0 * x + c` over a vector input is §07 `add` of two same-shape arrays.
    let source = "c = [1.0, 0.0, -1.0]\nf = x -> 2.0 * x + c\n";
    let mut module = flatppl_syntax::parse(source).unwrap();
    let errors: Vec<_> = infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
}
