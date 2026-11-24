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
fn real_rounding_and_conjugation_emit_differentiable_real_queries() {
    for (input, output) in [
        ("reals", "conj(x)"),
        ("reals", "real(floor(x))"),
        ("reals", "real(ceil(x))"),
        ("reals", "real(round(x))"),
        ("cartpow(reals, 2)", "sum(real.(round.(x)))"),
    ] {
        let module = lower(&format!(
            "x = elementof({input})\ninputs = x\noutputs = {output}\n"
        ));
        for differentiate in [gradient, jvp, vjp] {
            let differentiated = differentiate(&module, &["x"]).unwrap();
            lower(&flatppl_syntax::print(&differentiated));
        }
    }
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
fn matrix_exponential_derivatives_roundtrip_standard_module_calls() {
    for (mode, output) in
        [gradient, jvp, vjp]
            .into_iter()
            .zip(["sum(e.matexp(A))", "e.matexp(A = A)", "e.matexp(A)"])
    {
        let module = lower(&format!(
            "e = standard_module(\"ext-linear-algebra\", \"0.1\")\n\
             A = elementof(cartpow(reals, [2, 2]))\n\
             inputs = A\noutputs = {output}"
        ));
        let mut derivative = mode(&module, &["A"]).unwrap();
        flatppl_autodiff::prepare_source(&mut derivative);
        lower(&flatppl_syntax::print(&derivative));
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

#[test]
fn indexed_products_keep_nested_axes_in_both_modes() {
    let module = lower(
        "
A = elementof(cartpow(reals, [2, 3, 2]))
w = elementof(cartpow(reals, [2, 2]))
lanes = aggregate(prod, [.i, .j], A[.i, .k, .j] / 1.0)
inputs = (A, w)
outputs = aggregate(sum, [], lanes[.i, .j] * w[.i, .j])
",
    );
    for differentiate in [gradient, vjp, jvp] {
        let result = differentiate(&module, &["A", "w"]).unwrap();
        lower(&flatppl_syntax::print(&result));
    }
}

#[test]
fn bernoulli_logit_pullbacks_skip_rounded_probabilities() {
    for source in [
        "
eta = elementof(reals)
p = invlogit(eta)
observed = elementof(integers)
inputs = (eta, observed)
outputs = logdensityof(Bernoulli(p), observed)
",
        "
eta = elementof(cartpow(reals, 2))
p = invlogit.(eta)
observed = elementof(cartpow(integers, 2))
inputs = (eta, observed)
outputs = sum(builtin_logdensityof.(Bernoulli, record.(p = p), observed))
",
    ] {
        let model = lower(source);
        for differentiate in [gradient, jvp, vjp] {
            let derivative = differentiate(&model, &["eta"]).unwrap();
            let text = flatppl_syntax::print(&derivative);
            assert!(!text.contains(" / "), "{text}");
        }
    }

    let model = lower(
        "
eta = elementof(reals)
p = invlogit(eta)
inputs = (eta, p)
outputs = logdensityof(Bernoulli(p), 0)
",
    );
    let derivative = gradient(&model, &["eta", "p"]).unwrap();
    let text = flatppl_syntax::print(&derivative);
    assert!(text.contains(" / "), "{text}");
}
