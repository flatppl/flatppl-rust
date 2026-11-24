use flatppl_autodiff::{determinize_query, jvp, vjp};
use flatppl_core::Module;
use flatppl_infer::{ModuleBundle, Severity};

fn lower(text: &str) -> Module {
    let mut module = flatppl_syntax::parse(text).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(
        diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{diagnostics:?}"
    );
    determinize_query(&module, &ModuleBundle::new()).unwrap()
}

#[test]
fn mixed_table_products_keep_primal_labels_and_project_numeric_seeds() {
    let model = lower(
        "
t = elementof(cartpow(cartprod(v = reals, label = integers, nested = cartprod(keep = booleans, z = reals)), 3))
i = elementof(integers)
r = get0(t, i)
out = table(v = 2.0 .* t.v, label = t.label, nested = t.nested)
inputs = (t, i)
outputs = (out, r, t.v[[3, 1, 3]])
",
    );
    for product in [jvp, vjp] {
        let derivative = product(&model, &["t"]).unwrap();
        let source = flatppl_syntax::print(&derivative);
        let reparsed = lower(&source);
        flatppl_determinizer::is_flatpdl(&reparsed).unwrap();
    }
}

#[test]
fn fully_discrete_table_is_not_an_active_input() {
    let model = lower(
        "
t = elementof(cartpow(cartprod(label = integers, keep = booleans), 2))
inputs = t
outputs = 1.0
",
    );
    for product in [jvp, vjp] {
        let error = product(&model, &["t"]).unwrap_err();
        assert!(error.to_string().contains("differentiable real"));
    }
}
