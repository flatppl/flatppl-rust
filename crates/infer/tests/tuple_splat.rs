//! Spec §04 "Calling conventions": a sole positional tuple splats into positional
//! arguments for a callable with ordered inputs that does not accept a lone
//! positional argument. A callable that does receives the tuple whole. Tuples with
//! fewer elements than the required inputs, or more than the inputs, are a static
//! error, and splatting is shallow.

use flatppl_infer::{Severity, infer};

fn module(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = infer(&mut m);
    m
}

fn errors(src: &str) -> Vec<String> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

/// The FlatPIR `%meta` line of the binding `name`.
fn bind_line(src: &str, name: &str) -> String {
    let out = flatppl_flatpir::write(&module(src));
    let prefix = format!("(%bind {name} ");
    out.lines()
        .map(str::trim)
        .find(|l| l.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no binding `{name}` in:\n{out}"))
        .to_string()
}

const CALLEES: &str = "\
f(a, b) = a + b
g(a, b, c) = a + b + c
h(a) = a
";

#[test]
fn a_tuple_splats_into_positional_arguments() {
    let src = format!("{CALLEES}t = (1.0, 2.0)\nliteral = f((1.0, 2.0))\nnamed = f(t)\n");
    assert_eq!(errors(&src), Vec::<String>::new());
    // The splatted arguments type the body: a real sum, not the tuple.
    assert!(bind_line(&src, "literal").contains("(%scalar real)"));
    assert!(bind_line(&src, "named").contains("(%scalar real)"));
}

#[test]
fn a_builtin_with_ordered_inputs_splats_a_tuple() {
    let src = "x = add((1.0, 2.0))\ny = pow((2, 3))\n";
    assert_eq!(errors(src), Vec::<String>::new());
    assert!(bind_line(src, "x").contains("(%scalar real)"));
}

#[test]
fn a_tuple_with_too_few_elements_is_refused() {
    let found = errors(&format!("{CALLEES}r = g((1.0, 2.0))\n"));
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("`g` declares 3 parameters, got 2 arguments"),
        "{found:?}"
    );
    assert!(found[0].contains("positional tuple splatted"), "{found:?}");
}

#[test]
fn a_tuple_with_too_many_elements_is_refused() {
    let found = errors(&format!("{CALLEES}r = f((1.0, 2.0, 3.0))\n"));
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("`f` declares 2 parameters, got 3 arguments"),
        "{found:?}"
    );
    let found = errors("x = add((1.0, 2.0, 3.0))\n");
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].contains("`add` takes 2 arguments"), "{found:?}");
}

#[test]
fn a_callable_accepting_a_lone_argument_receives_the_tuple_whole() {
    // `h(a)` takes one positional input, so the interface decides before any
    // domain: the tuple is `a`, and the call types as that tuple.
    let src = format!("{CALLEES}r = h((1.0, 2.0))\n");
    assert_eq!(errors(&src), Vec::<String>::new());
    assert!(bind_line(&src, "r").contains("(%tuple (%scalar real) (%scalar real))"));
}

#[test]
fn keyword_only_inputs_do_not_splat_a_tuple() {
    let found = errors(
        "x = elementof(reals)\ny = elementof(reals)\nk = functionof(x + y)\nr = k((1.0, 2.0))\n",
    );
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("`k` has keyword-only inputs"),
        "{found:?}"
    );
}

#[test]
fn splatting_is_shallow() {
    // The outer tuple splats; its first element, a tuple, binds to `a` whole.
    let src = format!("{CALLEES}r = h(((1.0, 2.0), 3.0))\ns = f(((1.0, 2.0), 3.0))\n");
    let found = errors(&src);
    // `h` takes the outer tuple whole, and `f` binds `a = (1.0, 2.0)` without splatting it.
    assert!(
        found.iter().all(|m| !m.contains("declares")),
        "no arity error from a deeper splat: {found:?}"
    );
}

#[test]
fn a_tuple_point_is_checked_against_an_ordered_likelihood() {
    let model = "\
a = elementof(reals)
b = elementof(reals)
K = functionof(Normal(mu = a, sigma = b), a = a, b = b)
L = likelihoodof(K, 0.5)
";
    assert_eq!(
        errors(&format!("{model}lp = logdensityof(L, (0.1, 2.0))\n")),
        Vec::<String>::new()
    );
    let found = errors(&format!("{model}lp = logdensityof(L, (0.1, 2.0, 3.0))\n"));
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].contains("a tuple of 3 elements"), "{found:?}");
}
