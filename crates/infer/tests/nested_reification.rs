//! Spec §04 "Lambda notation" rewrites every free occurrence of a lambda argument to
//! its placeholder, inside nested reifications too. A nested `functionof` or
//! `kernelof` that lists the placeholder as a boundary input binds it, so
//! `g(mu) = functionof(Normal(mu, 1.0), mu = mu)` is legal. A nested reification
//! that does not bind it leaves the placeholder free there, which §04 "Placeholders
//! and holes" refuses: closures stay refused.

use flatppl_infer::{Severity, infer};

fn errors(src: &str) -> Vec<String> {
    let mut module = flatppl_syntax::parse(src).unwrap();
    infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

#[test]
fn a_nested_reification_may_bind_an_enclosing_argument() {
    for src in [
        "g(mu) = functionof(Normal(mu, 1.0), mu = mu)\n",
        "f(x) = functionof(2.0 * x, y = x)\n",
        "g(mu) = kernelof(draw(Normal(mu, 1.0)), mu = mu)\n",
        "h = (a, b) -> functionof(a * b, p = a, q = b)\n",
    ] {
        assert_eq!(errors(src), Vec::<String>::new(), "{src}");
    }
}

#[test]
fn the_nested_reification_binds_the_placeholder_it_reads() {
    // `f(x) = functionof(2.0 * x, y = x)` is `functionof(functionof(2.0 * _x_, y = _x_),
    // x = _x_)`: the inner function maps `y` to `2.0 * y`.
    let module = flatppl_syntax::parse("f(x) = functionof(2.0 * x, y = x)\n").unwrap();
    let printed = flatppl_syntax::print_with(&module, flatppl_syntax::Syntax::Minimal);
    assert!(
        printed.contains("functionof(mul(2.0, _x_), y = _x_)"),
        "got:\n{printed}"
    );
}

#[test]
fn a_closure_over_an_enclosing_argument_stays_refused() {
    for src in [
        "h = x -> functionof(x + 1.0, z = _z_)\n",
        "v = [1.0, 2.0]\nk = x -> broadcast(y -> x * y, v)\n",
        "m = x -> fn(_ + x)\n",
        "a = elementof(reals)\nf = x -> functionof(a + x, p = a)\n",
    ] {
        let found = errors(src);
        assert!(
            found
                .iter()
                .any(|e| e.contains("placeholder `_x_`")
                    && e.contains("no boundary input declares it")),
            "{src}: {found:?}"
        );
    }
}
