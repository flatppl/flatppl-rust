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
fn reverse_quadrature_preserves_nested_record_captures() {
    use flatppl_autodiff::{
        NUMERICAL_SOURCE_OPTIONS, determinize_query_with_options, gradient_with_options,
        prepare_source_with_options, vjp_with_options,
    };

    let source = "
p = elementof(cartprod(hi = nonnegreals, block = cartprod(a = reals, v = cartpow(reals, 2), m = cartpow(reals, [2, 2]))))
present = elementof(booleans)
weight(x, p) = 1.0 + (p.block.a + p.block.v[2] * x + p.block.m[2, 1] * x^2)^2
score = log(totalmass(weighted(weight, Lebesgue(interval(0.0, p.hi)), p)))
inputs = (p, present)
outputs = ifelse(present, score, p.block.a)
";
    let lower = |source: &str| {
        let mut module = flatppl_syntax::parse(source).unwrap();
        let diagnostics = flatppl_infer::infer(&mut module);
        assert!(
            diagnostics.iter().all(|d| d.severity != Severity::Error),
            "{diagnostics:?}"
        );
        determinize_query_with_options(&module, &ModuleBundle::new(), &NUMERICAL_SOURCE_OPTIONS)
            .unwrap()
    };
    let module = lower(source);
    for transform in [gradient_with_options, vjp_with_options] {
        let mut result = transform(&module, &["p"], &NUMERICAL_SOURCE_OPTIONS).unwrap();
        flatppl_determinizer::is_flatpdl_with_options(&result, &NUMERICAL_SOURCE_OPTIONS).unwrap();
        prepare_source_with_options(&mut result, &NUMERICAL_SOURCE_OPTIONS).unwrap();
        let printed = flatppl_syntax::print(&result);
        let reparsed = lower(&printed);
        flatppl_determinizer::is_flatpdl_with_options(&reparsed, &NUMERICAL_SOURCE_OPTIONS)
            .unwrap();
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

#[test]
fn real_table_derivatives_reparse_with_named_columns_and_row_projections() {
    let module = lower(
        "
t = elementof(cartpow(cartprod(b = reals, a = reals), 3))
i = elementof(integers)
r = t[i]
inputs = (t, i)
outputs = (table(z = t.b, a = t.a[[2, 2, 1]]), r)
",
    );
    for transformed in [jvp, vjp] {
        let derivative = transformed(&module, &["t"]).unwrap();
        let reparsed = lower(&flatppl_syntax::print(&derivative));
        flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
    }
}

#[test]
fn record_products_only_add_overlapping_field_contributions() {
    let input = lower(
        "
p = elementof(cartprod(z = reals, a = reals, m = reals, b = reals))
inputs = p
z = p.z
a = p.a
shared = z + a
outputs = (shared + z, shared + shared, p.m + p.b)
",
    );
    let derivative = vjp(&input, &["p"]).unwrap();
    let printed = flatppl_syntax::print(&derivative);
    // Four primal additions, two seed additions at shared and one at z.
    assert_eq!(printed.matches(" + ").count(), 7, "{printed}");
    assert!(!printed.contains("0.0"), "{printed}");
    lower(&printed);
}

#[test]
fn constructor_tangent_builds_each_nested_field_once() {
    let input = lower(
        "
x = elementof(reals)
y = elementof(reals)
v = elementof(cartpow(reals, 2))
inputs = (x, y, v)
outputs = record(z = x, inner = record(y = y, unused = v), a = y, b = x)
",
    );
    let derivative = jvp(&input, &["x", "y"]).unwrap();
    let printed = flatppl_syntax::print(&derivative);
    assert!(!printed.contains(" + "), "{printed}");
    // Only the inactive vector field needs a zero.
    assert_eq!(printed.matches("fill(").count(), 1, "{printed}");
    lower(&printed);
}

#[test]
fn table_matrix_cell_derivatives_preserve_nested_array_axes() {
    let module = lower(
        "
t = elementof(cartpow(cartprod(block = cartpow(reals, [2, 2]), weight = reals), 2))
inputs = t
outputs = (t[2], t.block[[2, 2, 1]])
",
    );
    for transformed in [jvp, vjp] {
        let derivative = transformed(&module, &["t"]).unwrap();
        let reparsed = lower(&flatppl_syntax::print(&derivative));
        flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
    }
}

#[test]
fn mixed_records_reparse_with_passive_fields_only_in_the_primal() {
    let module = lower(
        "
p = elementof(cartprod(tag = integers, x = reals, meta = cartprod(flag = booleans), unused = cartpow(reals, 2)))
z = ifelse(p.meta.flag, p.x * p.x, p.x)
q = record(label = p.tag, result = z, nested = record(flag = p.meta.flag, x = p.x))
inputs = p
outputs = (q, p.tag, z)
",
    );
    for transform in [jvp, vjp] {
        let mut derivative = transform(&module, &["p"]).unwrap();
        flatppl_autodiff::prepare_source(&mut derivative);
        lower(&flatppl_syntax::print(&derivative));
    }
}
