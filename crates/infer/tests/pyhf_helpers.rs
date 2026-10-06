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
fn collection_call_forms_preserve_axes_and_promote_integer_cells() {
    let (module, diagnostics) = infer(
        r#"
n = elementof(cartpow(integers, [2, 3]))
s = elementof(cartpow(integers, [2, 4, 3]))
f = elementof(cartpow(integers, [2, 5, 3]))
a = p.sample_yields(n, factors = f, shifts = s)
b = p.sample_yields(record(factors = f, nominal = n, shifts = s))
r = external(cartprod(factors = cartpow(reals, [2, 5, 3]), shifts = cartpow(reals, [2, 4, 3]), nominal = cartpow(reals, [2, 3])))
c = p.sample_yields(r)
zero = get([0], 1)
empty = s[:, fill(1, zero), :]
empty_shifts = p.sample_yields(n, empty, f)
counts = p.expected_counts(samples = a)
alias = p.expected_counts
again = alias(record(samples = n))
"#,
    );
    no_errors(&diagnostics);
    for name in ["a", "b", "c", "empty_shifts"] {
        assert_eq!(ty(&module, name), real_array(&[2, 3]));
    }
    for name in ["counts", "again"] {
        assert_eq!(ty(&module, name), real_array(&[3]));
    }
}

#[test]
fn outer_broadcast_checks_cells_and_preserves_its_batch() {
    let (module, diagnostics) = infer(
        r#"
n = elementof(cartpow(cartpow(reals, [2, 3]), 7))
s = elementof(cartpow(cartpow(reals, [2, 4, 3]), 7))
f = elementof(cartpow(cartpow(reals, [2, 5, 3]), 1))
a = p.sample_yields.(n, factors = f, shifts = s)
b = p.expected_counts.(samples = n)
t = table(lo = [0.8, 0.9], hi = [1.2, 1.1], alpha = [0.0, 0.5])
c = p.normsys_factor.(t)
"#,
    );
    no_errors(&diagnostics);
    assert_eq!(
        ty(&module, "a"),
        Type::Array {
            shape: vec![Dim::Static(7)].into(),
            elem: Box::new(real_array(&[2, 3])),
        }
    );
    assert_eq!(
        ty(&module, "b"),
        Type::Array {
            shape: vec![Dim::Static(7)].into(),
            elem: Box::new(real_array(&[3])),
        }
    );
    assert_eq!(ty(&module, "c"), real_array(&[2]));
}

#[test]
fn invalid_model_axes_do_not_become_broadcast_replication() {
    for (nominal, shifts, factors, call) in [
        (
            "[1, 3]",
            "[2, 4, 3]",
            "[2, 5, 3]",
            "p.sample_yields(n, s, f)",
        ),
        (
            "[2, 3]",
            "[2, 4, 1]",
            "[2, 5, 3]",
            "p.sample_yields(n, shifts = s, factors = f)",
        ),
        (
            "[2, 3]",
            "[3, 4, 3]",
            "[2, 5, 3]",
            "p.sample_yields(record(nominal = n, shifts = s, factors = f))",
        ),
        ("[2, 3]", "[2, 3]", "[2, 5, 3]", "p.sample_yields(n, s, f)"),
        ("[2, 3]", "[2, 4, 3]", "[2, 5, 3]", "p.expected_counts(s)"),
    ] {
        let (module, diagnostics) = infer(&format!(
            "n = elementof(cartpow(reals, {nominal}))\ns = elementof(cartpow(reals, {shifts}))\nf = elementof(cartpow(reals, {factors}))\nout = {call}"
        ));
        assert!(
            diagnostics.iter().any(|d| d.severity == Severity::Error),
            "{call}: {diagnostics:?}"
        );
        assert!(matches!(ty(&module, "out"), Type::Failed(_)), "{call}");
    }
    let (module, diagnostics) = infer(
        r#"
n = elementof(cartpow(cartpow(reals, [2, 3]), 7))
s = elementof(cartpow(cartpow(reals, [1, 4, 3]), 7))
f = elementof(cartpow(cartpow(reals, [2, 5, 3]), 7))
out = p.sample_yields.(n, s, f)
"#,
    );
    assert!(diagnostics.iter().any(|d| d.severity == Severity::Error));
    assert!(matches!(ty(&module, "out"), Type::Failed(_)));
}

#[test]
fn unknown_extents_are_not_assumed_equal_or_rejected() {
    let (module, diagnostics) = infer(
        r#"
size = external(posintegers)
n = elementof(cartpow(reals, [size, 3]))
s = elementof(cartpow(reals, [2, 4, 3]))
f = elementof(cartpow(reals, [2, 5, 3]))
out = p.sample_yields(n, s, f)
"#,
    );
    no_errors(&diagnostics);
    assert_eq!(
        ty(&module, "out"),
        Type::Array {
            shape: vec![Dim::Dynamic, Dim::Static(3)].into(),
            elem: Box::new(Type::Scalar(ScalarType::Real)),
        }
    );
    let (_, diagnostics) = infer(
        r#"
size = external(posintegers)
n = elementof(cartpow(reals, [size, 3]))
s = elementof(cartpow(reals, [2, 4, 3]))
f = elementof(cartpow(reals, [3, 5, 3]))
out = p.sample_yields(n, s, f)
"#,
    );
    assert!(diagnostics.iter().any(|d| d.severity == Severity::Error));
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
