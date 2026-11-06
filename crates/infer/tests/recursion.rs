//! Spec §04: a module is a directed acyclic graph, so no function or kernel may
//! reach its own binding. Inference reports the cycle once and returns.

use flatppl_infer::{Severity, infer};

fn errors(src: &str) -> Vec<String> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

#[test]
fn a_called_self_recursive_function_is_refused() {
    let errors = errors("f(x) = f(x)\ntheta ~ Normal(0, 1)\ny ~ Normal(f(theta), 0.5)\n");
    assert!(
        errors
            .iter()
            .any(|e| e.contains("`f` calls itself (f → f)")),
        "{errors:?}"
    );
}

#[test]
fn mutual_recursion_is_one_cycle() {
    let errors = errors("f(x) = g(x) + 1\ng(x) = f(x) * 2\nz = f(1.0)\n");
    let cycles: Vec<_> = errors
        .iter()
        .filter(|e| e.contains("calls itself"))
        .collect();
    assert_eq!(cycles.len(), 1, "{errors:?}");
    assert!(cycles[0].contains("(f → g → f)"), "{errors:?}");
}

#[test]
fn a_function_passed_through_a_higher_order_one_twice_is_not_recursion() {
    let errors = errors(
        "apply(h, x) = h(x)\nid(y) = y\ng(y) = apply(id, y)\nf(x) = apply(g, x)\nz = f(1.0)\n",
    );
    assert!(
        errors.iter().all(|e| !e.contains("calls itself")),
        "{errors:?}"
    );
}
