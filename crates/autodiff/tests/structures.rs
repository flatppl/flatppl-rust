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
fn slices_stacks_and_accumulates_repeated_matrix_gathers() {
    let module = lower(
        "
x = elementof(cartpow(reals, 3))
y = elementof(cartpow(reals, 3))
M = rowstack([x, y])
N = colstack([x, y])
z = M[:, [2, 2]] .+ N[[3, 1], :][:, :]
inputs = (x, y)
outputs = sum(z .* z) + get0(M, 1, 2)
",
    );
    for transformed in [gradient, jvp, vjp] {
        let derivative = transformed(&module, &["x", "y"]).unwrap();
        let printed = flatppl_syntax::print(&derivative);
        let reparsed = lower(&printed);
        flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
    }
}

#[test]
fn only_a_projected_constructor_field_is_active() {
    let module = lower(
        "
x = elementof(reals)
p = record(y = record(value = x * x), ignored = floor(x))
u = tuple(p, floor(x))
inputs = x
outputs = get(u, 1).y.value
",
    );
    let differentiated = gradient(&module, &["x"]).unwrap();
    let text = flatppl_syntax::print(&differentiated);
    assert!(text.contains("outputs = "), "{text}");
    assert!(text.contains("d_x = "), "{text}");
    assert_eq!(text.matches("floor(x)").count(), 2, "{text}");
    lower(&text);
}

#[test]
fn nested_real_vectors_keep_their_shape_in_gradient_and_tangent_seeds() {
    let module = lower(
        "
x = elementof(cartpow(cartpow(reals, 3), 2))
unused = elementof(cartpow(cartpow(reals, 2), 3))
M = rowstack(x)
N = colstack(x)
y = M[2, :] .+ N[:, 1] .+ x[2]
inputs = (x, unused)
outputs = sum(y .* y)
",
    );
    for transformed in [gradient, jvp, vjp] {
        let derivative = transformed(&module, &["x", "unused"]).unwrap();
        let printed = flatppl_syntax::print(&derivative);
        let reparsed = lower(&printed);
        flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
    }
}

#[test]
fn real_record_inputs_keep_their_nested_field_shapes() {
    let module = lower(
        "
p = elementof(cartprod(a = reals, b = cartpow(reals, 3), c = cartprod(d = reals, e = reals)))
inputs = p
outputs = 2.0 * p.a + sum(p.b .* p.b) + p.c.d * p.a
",
    );
    for transformed in [gradient, jvp, vjp] {
        let derivative = transformed(&module, &["p"]).unwrap();
        let printed = flatppl_syntax::print(&derivative);
        let reparsed = lower(&printed);
        flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
    }
}

#[test]
fn directional_products_preserve_record_outputs_and_nested_vectors() {
    let module = lower(
        "
x = elementof(cartpow(reals, 3))
y = elementof(reals)
result = record(loss = y * sum(x), extras = record(rows = [x, x .* y], idle = 3.0))
inputs = (x, y)
outputs = (result, x)
",
    );
    for transformed in [jvp, vjp] {
        let derivative = transformed(&module, &["x", "y"]).unwrap();
        let printed = flatppl_syntax::print(&derivative);
        let reparsed = lower(&printed);
        flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
    }
}

#[test]
fn concatenating_differently_sized_vectors_splits_cotangents_at_the_original_boundaries() {
    let module = lower(
        "
x = elementof(cartpow(reals, 2))
z = elementof(cartpow(reals, 3))
joined = cat(x, [4, 5, 6, 7], z)
weighted = joined .* [1, 2, 3, 4, 5, 6, 7, 8, 9]
inputs = (x, z)
outputs = sum(weighted)
",
    );
    for transformed in [gradient, jvp, vjp] {
        let derivative = transformed(&module, &["x", "z"]).unwrap();
        let printed = flatppl_syntax::print(&derivative);
        let reparsed = lower(&printed);
        flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
    }
}

#[test]
fn directional_products_transport_gathered_nested_vector_cells() {
    let module = lower(
        "
x = elementof(cartpow(cartpow(reals, 3), 2))
y = elementof(cartpow(cartpow(reals, 3), 1))
inputs = (x, y)
outputs = (get0(cat(x, y), [2, 0, 2]), get(y, only), get0(y, all))
",
    );
    for transformed in [jvp, vjp] {
        let derivative = transformed(&module, &["x", "y"]).unwrap();
        lower(&flatppl_syntax::print(&derivative));
    }
}
