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
fn membership_requires_a_compatible_element_type() {
    for expr in [
        "in(2.0, integers)",
        "in(1, booleans)",
        "in([1.0, 2.0], cartpow(integers, 2))",
    ] {
        assert!(!errors(&format!("x = {expr}\n")).is_empty(), "{expr}");
    }
    for expr in [
        "in(2, integers)",
        "in(true, integers)",
        "in(2, reals)",
        "in(integer(2.0), integers)",
        "in([1, 2], cartpow(reals, 2))",
    ] {
        assert!(errors(&format!("x = {expr}\n")).is_empty(), "{expr}");
    }
}
