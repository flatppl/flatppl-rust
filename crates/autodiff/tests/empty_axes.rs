use flatppl_autodiff::{determinize_query, gradient, jvp, vjp};
use flatppl_core::{Dim, Module, Type};
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
fn computed_empty_axes_keep_portable_annotations() {
    let source = r#"
no_indices = filter(i -> false, [1, 2])
n = lengthof(no_indices)
sample = elementof(cartpow(reals, [n, 3]))
inputs = sample
outputs = sample
"#;
    let module = lower(source);
    let output = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == "outputs")
        .unwrap()
        .0;
    let output = module.binding(output).rhs;
    assert!(
        matches!(module.type_of(output), Some(Type::Array { shape, .. })
        if shape.as_ref() == [Dim::Dynamic, Dim::Static(3)])
    );
    let specialized = flatppl_infer::specialize_empty_axes(&module).unwrap();
    assert!(
        matches!(specialized.type_of(output), Some(Type::Array { shape, .. })
        if shape.as_ref() == [Dim::Static(0), Dim::Static(3)])
    );
    lower(&flatppl_syntax::print(&jvp(&module, &["sample"]).unwrap()));
    lower(&flatppl_syntax::print(&vjp(&module, &["sample"]).unwrap()));
}

#[test]
fn pyhf_empty_model_axes_emit_reparsable_derivatives() {
    let source = r#"
pyhf = standard_module("pyhf-helpers", "0.1")
no_indices = filter(i -> false, [1, 2])
n = lengthof(no_indices)
nominal = elementof(cartpow(reals, [2, 3]))
shifts = elementof(cartpow(reals, [2, 2, 3]))
factors = elementof(cartpow(reals, [2, n, 3]))
inputs = (nominal, shifts, factors)
regular = pyhf.sample_yields(nominal, shifts[:, no_indices, :], factors)
empty = regular[no_indices, :]
counts = pyhf.expected_counts(empty)
"#;
    for output in ["regular", "empty", "counts"] {
        for (transform, output) in [
            (
                gradient as fn(&Module, &[&str]) -> _,
                format!("sum({output})"),
            ),
            (jvp, output.into()),
            (vjp, output.into()),
        ] {
            let module = lower(&format!("{source}\noutputs = {output}\n"));
            let derivative = transform(&module, &["nominal", "shifts", "factors"]).unwrap();
            lower(&flatppl_syntax::print(&derivative));
        }
    }
}
