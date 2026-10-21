//! A binding or argument named `im`, `pi` or `inf` shadows the built-in constant
//! (spec §04 "Name resolution"), so inference types the binding's value.

use flatppl_infer::{Severity, infer};

/// The trimmed FlatPIR line of the binding `name`, with its `%meta` annotation.
fn bind_line(src: &str, name: &str) -> String {
    let mut module = flatppl_syntax::parse(src).unwrap();
    let diags = infer(&mut module);
    assert!(
        diags.iter().all(|d| d.severity != Severity::Error),
        "unexpected diagnostics: {diags:?}"
    );
    let out = flatppl_flatpir::write(&module);
    let prefix = format!("(%bind {name} ");
    out.lines()
        .map(str::trim)
        .find(|l| l.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no binding `{name}` in:\n{out}"))
        .to_string()
}

// Plain aliases (`a = pi`) carry no annotation, so each case wraps the name in
// arithmetic whose result type depends on the name's type.

#[test]
fn shadowed_constant_takes_the_binding_type() {
    // The built-in `pi` is real and `im` complex; the bindings are integer and real.
    let src = "pi = 3\nim = 2.0\na = pi + 1\nb = base.pi + 1\nc = im * 2.0\nd = base.im * 2.0";
    assert!(bind_line(src, "a").contains("(%scalar integer)"));
    assert!(bind_line(src, "b").contains("(%scalar real)"));
    assert!(bind_line(src, "c").contains("(%scalar real)"));
    assert!(bind_line(src, "d").contains("(%scalar complex)"));
}

#[test]
fn unshadowed_constants_keep_their_builtin_type() {
    let src = "a = pi + 1\nb = inf + 1\nc = im * 2.0";
    assert!(bind_line(src, "a").contains("(%scalar real)"));
    assert!(bind_line(src, "b").contains("(%scalar real)"));
    assert!(bind_line(src, "c").contains("(%scalar complex)"));
}

#[test]
fn argument_named_like_a_constant_shadows_in_the_body() {
    let src = "f = im -> im + 1\nr = f(2)\ns = im * 2.0";
    assert!(bind_line(src, "r").contains("(%scalar integer)"));
    assert!(bind_line(src, "s").contains("(%scalar complex)"));
}
