//! Spec §04 "Explicit callback parameters": `weighted(f, M)`, `logweighted(f, M)`
//! and `pushfwd(f, M)` apply `f(x)` to the whole variate `x`. That application is
//! an ordinary call, so §04 "Calling conventions" decides how `x` binds: a callable
//! with two or more required inputs receives a record variate's fields as keyword
//! arguments, and a callable that accepts a lone positional argument receives the
//! record whole. Field names that do not match the inputs are a static error.
use flatppl_core::{ScalarType, Type};
use flatppl_infer::{Severity, infer};

const BOX: &str = "x = elementof(interval(0.0, 1.0))\n\
                   y = elementof(interval(0.0, 2.0))\n\
                   S = cartprod(x = interval(0.0, 1.0), y = interval(0.0, 2.0))\n\
                   B = Lebesgue(support = S)\n";
const NAMED: &str = "xy = x * y\nf = functionof(xy, x, y)\ns = x + y\ng = functionof(s, x, y)\n";

fn errors(source: &str) -> Vec<String> {
    let mut module = flatppl_syntax::parse(source).unwrap();
    infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

fn type_of(source: &str, name: &str) -> Type {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let errors: Vec<_> = infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{source}: {errors:?}");
    let rhs = module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == name)
        .unwrap()
        .1
        .rhs;
    module.type_of(rhs).cloned().unwrap()
}

#[test]
fn a_record_variate_splats_into_a_several_input_callback() {
    for tail in [
        format!("{NAMED}M = weighted(f, B)\nQ = pushfwd(g, normalize(M))\n"),
        format!("{NAMED}M = logweighted(f, B)\n"),
        "M = weighted((x, y) -> x * y, B)\nQ = pushfwd((x, y) -> x + y, normalize(M))\n".into(),
    ] {
        let source = format!("{BOX}{tail}");
        assert_eq!(errors(&source), Vec::<String>::new(), "{source}");
    }
}

#[test]
fn the_splatted_pushforward_has_the_callback_codomain() {
    let source = format!("{BOX}Q = pushfwd((x, y) -> x + y, normalize(B))\n");
    match type_of(&source, "Q") {
        Type::Measure { domain, .. } => {
            assert_eq!(*domain, Type::Scalar(ScalarType::Real), "{source}")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_one_input_callback_still_receives_the_record_whole() {
    let source = format!("{BOX}M = weighted(r -> r.x * r.y, B)\n");
    assert_eq!(errors(&source), Vec::<String>::new());
}

#[test]
fn a_field_the_callback_does_not_declare_is_a_static_error() {
    let errs = errors(&format!("{BOX}M = weighted((x, z) -> x * z, B)\n"));
    assert!(
        errs.iter().any(|m| m.contains("no parameter `y`")),
        "{errs:?}"
    );
}

#[test]
fn a_scalar_variate_is_still_refused_for_a_several_input_callback() {
    for op in ["weighted", "logweighted", "pushfwd"] {
        let errs = errors(&format!(
            "M = Uniform(interval(0.0, 1.0))\nm = {op}((x, y) -> x * y, M)\n"
        ));
        assert!(
            errs.iter()
                .any(|m| m.contains("declares 2 parameters, got 1 arguments")),
            "{op}: {errs:?}"
        );
    }
}

#[test]
fn a_final_parameter_keeps_the_variate_whole() {
    // With `p`, the application is `f(x, p)`: the record binds `x` whole and `p`
    // the next input, so a third input stays unbound.
    let errs = errors(&format!(
        "{BOX}M = weighted((x, y, p) -> x * y * p, B, 2.0)\n"
    ));
    assert!(
        errs.iter().any(|m| m.contains("input `p` is left unbound")),
        "{errs:?}"
    );
    let source = format!("{BOX}M = weighted((r, p) -> r.x * r.y * p, B, 2.0)\n");
    assert_eq!(errors(&source), Vec::<String>::new());
}
