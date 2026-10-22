//! A sole positional tuple splats into positional arguments for a callable with
//! ordered inputs that does not accept a lone positional argument (spec §04
//! "Calling conventions"), and §11 resolves it before normalization: each element
//! becomes a positional argument, a projection of the tuple unless the tuple is
//! written out.

use flatppl_determinizer::determinize;
use flatppl_syntax::{Syntax, print_with};

fn determinize_text(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = flatppl_infer::infer(&mut m);
    assert!(
        diags
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "fixture must infer: {diags:?}"
    );
    print_with(
        &determinize(&m).expect("must lower, not refuse"),
        Syntax::Minimal,
    )
}

#[test]
fn a_written_tuple_inlines_its_elements_into_a_user_function() {
    let text = determinize_text(
        "f(a, b) = a * 2.0 + b
y = f((1.5, 0.5))",
    );
    assert!(text.contains("y = 3.5"), "{text}");
}

#[test]
fn a_tuple_value_splats_through_projections() {
    let text = determinize_text(
        "u = elementof(reals)
v = elementof(posreals)
f(a, b) = a * 2.0 + b
t = (u, v)
y = f(t)
inputs = (u, v)
outputs = y",
    );
    assert!(text.contains("y = add(mul(u, 2.0), v)"), "{text}");
}

#[test]
fn a_tuple_binds_a_kernel_in_input_order() {
    // Closed form: `Normal(mu = a, sigma = b)` at mu = 0.5, sigma = 2.0.
    let text = determinize_text(
        "a = elementof(reals)
b = elementof(posreals)
K = functionof(Normal(mu = a, sigma = b), a = a, b = b)
lp = logdensityof(K((0.5, 2.0)), 0.3)",
    );
    assert!(
        text.contains("builtin_logdensityof(Normal, record(mu = 0.5, sigma = 2.0), 0.3)"),
        "{text}"
    );
}

#[test]
fn a_tuple_splats_into_a_builtin_with_ordered_inputs() {
    let text = determinize_text(
        "x = elementof(reals)
y = add((x, 2.0))
inputs = x
outputs = y",
    );
    assert!(text.contains("y = add(x, 2.0)"), "{text}");
}
