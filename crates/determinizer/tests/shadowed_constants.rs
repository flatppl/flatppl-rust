//! A module binding or argument named `im`, `pi` or `inf` shadows the
//! built-in constant (spec §04 "Name resolution"). Determinizing must fold and
//! lower the binding's value, never the built-in.

use flatppl_determinizer::determinize;
use flatppl_syntax::{Syntax, print_with};

fn determinize_text(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let out = determinize(&m).expect("must lower, not refuse");
    print_with(&out, Syntax::Minimal)
}

#[test]
fn shadowed_pi_folds_to_the_binding() {
    let text = determinize_text(
        "pi = 3.0
p = elementof(reals)
bound = 2.0 * pi
builtin = 2.0 * base.pi
live = p * pi
inputs = p
outputs = (bound, builtin, live)",
    );
    assert!(text.contains("bound = 6.0"), "{text}");
    assert!(
        text.contains("builtin = mul(2.0, base.pi)"),
        "`base.pi` is still the constant:\n{text}"
    );
    assert!(text.contains("live = mul(p, 3.0)"), "{text}");
}

#[test]
fn argument_named_inf_shadows_inside_the_body_only() {
    let text = determinize_text(
        "f = inf -> inf + 1.0
p = elementof(reals)
folded = f(2.0)
outside = p * inf
inputs = p
outputs = (folded, outside)",
    );
    assert!(text.contains("folded = 3.0"), "{text}");
    assert!(text.contains("outside = mul(p, inf)"), "{text}");
}
