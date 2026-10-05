use flatppl_core::{Dim, Module, ScalarType, Type};
use flatppl_infer::{Diagnostic, Level, ModuleBundle, Severity, infer_module};

fn infer(body: &str) -> (Module, Vec<Diagnostic>) {
    let source = format!("p = standard_module(\"pyhf_helpers\", \"0.1\")\n{body}");
    let mut module = flatppl_syntax::parse(&source).unwrap();
    let diagnostics = infer_module(&mut module, &ModuleBundle::new(), Level::Shape);
    (module, diagnostics)
}

fn ty(module: &Module, name: &str) -> Type {
    let (_, binding) = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == name)
        .unwrap();
    module.type_of(binding.rhs).unwrap().clone()
}

fn real_array(shape: &[u32]) -> Type {
    Type::Array {
        shape: shape.iter().map(|&n| Dim::Static(n)).collect(),
        elem: Box::new(Type::Scalar(ScalarType::Real)),
    }
}

fn no_errors(diagnostics: &[Diagnostic]) {
    assert!(
        !diagnostics.iter().any(|d| d.severity == Severity::Error),
        "{diagnostics:?}"
    );
}

#[test]
fn scalar_helper_requires_explicit_broadcast_for_arrays() {
    let (module, diagnostics) = infer(
        r#"
x = elementof(cartpow(reals, 3))
out = p.normsys_factor(0.9, 1.1, x)
"#,
    );
    assert!(diagnostics.iter().any(|d| d.severity == Severity::Error));
    assert!(matches!(ty(&module, "out"), Type::Failed(_)));
    let (module, diagnostics) = infer(
        r#"
x = elementof(cartpow(reals, 3))
out = p.normsys_factor.(0.9, hi = 1.1, alpha = x)
normsys_factor(a, b, x) = x
local = normsys_factor(0.9, 1.1, x)
"#,
    );
    no_errors(&diagnostics);
    assert_eq!(ty(&module, "out"), real_array(&[3]));
    assert_eq!(ty(&module, "local"), real_array(&[3]));
}
