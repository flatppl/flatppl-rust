//! Variadic minimum arity decides whether a sole aggregate supplies named inputs.
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
fn unary_variadic_calls_keep_whole_records() {
    assert!(errors("x=cat((a=1,b=2))").is_empty());
    assert!(errors("f(p)=cat(p)\nx=f([1,2])").is_empty());
}
#[test]
fn access_requires_a_selector_instead_of_spreading_unnamed_inputs() {
    for f in ["get", "get0"] {
        assert!(!errors(&format!("x={f}((a=1,b=2))")).is_empty());
    }
    assert!(errors("x=get((a=1,b=2),\"a\")").is_empty());
}
#[test]
fn explicit_names_are_checked_after_spreading() {
    assert!(errors("x=atan2((x=2.0,y=1.0))").is_empty());
    assert!(!errors("x=atan2((wrong=2.0,y=1.0))").is_empty());
    assert!(errors("x=checked((condition=true,value=2))").is_empty());
}
#[test]
fn sample_distinguished_inputs_bind_by_name_and_keep_kernel_types() {
    let prefix = "s=rnginit([1,2,3])\np=(mu=0.0,sigma=1.0)\n";
    assert!(
        errors(&format!(
            "{prefix}x=builtin_sample((rngstate=s,kernel=Normal,kernel_input=p))"
        ))
        .is_empty()
    );
    assert!(
        errors(&format!(
            "{prefix}x=builtin_sample((rngstate=s,kernel=Normal,kernel_input=p,n=4))"
        ))
        .iter()
        .any(|message| message.contains("no parameter `n`"))
    );
    for source in [
        "x=builtin_sample(record(rngstate=1,kernel=2,kernel_input=3))",
        "xs=elementof(cartpow(reals,4))\nx=builtin_sample(table(rngstate=xs,kernel=xs,kernel_input=xs))",
    ] {
        assert!(
            errors(source)
                .iter()
                .any(|message| message.contains("must be a distribution kernel")),
            "{source}"
        );
    }
}
#[test]
fn a_vector_still_cannot_contain_records() {
    assert!(!errors("x=vector((a=1,b=2))").is_empty());
}
