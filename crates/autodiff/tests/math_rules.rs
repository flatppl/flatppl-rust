use flatppl_autodiff::{determinize_query, gradient, jvp, vjp};
use flatppl_core::Module;
use flatppl_infer::{ModuleBundle, Severity};

fn lower(source: &str) -> Module {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{diagnostics:?}"
    );
    determinize_query(&module, &ModuleBundle::new()).unwrap()
}

#[test]
fn determinant_derivatives_use_cofactors() {
    let module = lower(
        "
A = elementof(cartpow(reals, [3, 3]))
inputs = A
outputs = det(A)
",
    );
    for differentiate in [gradient, vjp, jvp] {
        let differentiated = differentiate(&module, &["A"]).unwrap();
        let text = flatppl_syntax::print(&differentiated);
        assert!(text.contains("rowstack("), "{text}");
        assert!(!text.contains("inv("), "{text}");
    }
}

#[test]
fn solve_derivatives_keep_matrix_and_vector_shapes() {
    let module = lower(
        "
A = elementof(cartpow(reals, [2, 2]))
b = elementof(cartpow(reals, 2))
inputs = (A, b)
outputs = linsolve(A, b)
",
    );
    vjp(&module, &["A", "b"]).unwrap();
    jvp(&module, &["A", "b"]).unwrap();
}

#[test]
fn cholesky_jvp_and_vjp_use_symmetric_matrix_shapes() {
    let module = lower(
        "
A = elementof(cartpow(reals, [2, 2]))
inputs = A
outputs = lower_cholesky(A)
",
    );
    vjp(&module, &["A"]).unwrap();
    jvp(&module, &["A"]).unwrap();
}

#[test]
fn real_queries_can_differentiate_complex_intermediates() {
    let module = lower(
        "
x = elementof(reals)
z = complex(x, 1.0)
score = ifelse(x < 3.0, abs2(z * cis(x)), 0.0)
inputs = x
outputs = score
",
    );
    for differentiate in [gradient, vjp, jvp] {
        differentiate(&module, &["x"]).unwrap();
    }
}

#[test]
fn polynomial_differentiates_coefficients_and_argument() {
    let module = lower(
        "
coefficients = elementof(cartpow(reals, 3))
x = elementof(reals)
inputs = (coefficients, x)
outputs = polynomial(coefficients, x)
",
    );
    for differentiate in [gradient, vjp, jvp] {
        differentiate(&module, &["coefficients", "x"]).unwrap();
    }
}

#[test]
fn indexed_sum_products_keep_bound_axes_in_both_modes() {
    let module = lower(
        "
A = elementof(cartpow(reals, [2, 3]))
b = elementof(cartpow(reals, 3))
inputs = (A, b)
outputs = aggregate(sum, [.i], b[.j] * A[.i, .j])
",
    );
    for differentiate in [vjp, jvp] {
        let differentiated = differentiate(&module, &["A", "b"]).unwrap();
        let text = flatppl_syntax::print(&differentiated);
        assert!(text.contains("b[.j] * A[.i, .j]"), "{text}");
    }
}

#[test]
fn indexed_sum_product_can_reduce_an_axis_only_one_operand_has() {
    let module = lower(
        "
A = elementof(cartpow(reals, 2))
b = elementof(cartpow(reals, 3))
inputs = (A, b)
outputs = aggregate(sum, [], A[.i] * b[.j])
",
    );
    for differentiate in [gradient, vjp, jvp] {
        differentiate(&module, &["A", "b"]).unwrap();
    }
}

#[test]
fn masked_matrix_product_uses_a_scoped_contraction() {
    let module = lower(
        "
A = elementof(cartpow(reals, [2, 2]))
b = elementof(cartpow(reals, 2))
product = A * b
inputs = (A, b)
outputs = ifelse.([true, false], product, 0.0)
",
    );
    let derivative = vjp(&module, &["A", "b"]).unwrap();
    let text = flatppl_syntax::print(&derivative);
    assert!(text.contains(".adinner"), "{text}");
}
