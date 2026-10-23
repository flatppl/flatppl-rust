//! Spec §06 "Joint measures": `joint(K1, K2, ...)` of kernels takes the union of the
//! components' inputs by name and is keyword-only, so as a `jointchain` step it is fed the
//! record of the preceding variates, which splats. Its variate is the joint of the
//! components' variates.
use flatppl_core::Type;
use flatppl_infer::{Severity, infer};

fn infer_src(src: &str) -> (flatppl_core::Module, Vec<String>) {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let errors = infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect();
    (m, errors)
}

fn fields(src: &str, name: &str) -> Vec<String> {
    let (m, errors) = infer_src(src);
    assert!(errors.is_empty(), "{errors:?}");
    let rhs = m
        .bindings()
        .find(|(_, b)| m.resolve(b.name) == name)
        .unwrap()
        .1
        .rhs;
    let Some(Type::Measure { domain, .. }) = m.type_of(rhs) else {
        panic!("{name} is not a measure");
    };
    let Type::Record(fields) = domain.as_ref() else {
        panic!("{name} is not over records: {domain:?}");
    };
    fields
        .iter()
        .map(|(n, _)| m.resolve(*n).to_string())
        .collect()
}

const KERNELS: &str = "\
x = draw(Normal(mu = 0.0, sigma = 1.0))
a = elementof(reals)
K1 = kernelof(draw(Normal(mu = a, sigma = 1.0)))
K2 = kernelof(draw(Normal(mu = 2.0 * a, sigma = 0.5)))
";

#[test]
fn the_chain_variate_includes_the_joint_kernels_components() {
    let src =
        format!("{KERNELS}J = joint(p = K1, q = K2)\njc = jointchain(lawof(record(a = x)), J)\n");
    assert_eq!(fields(&src, "jc"), ["a", "p", "q"]);
    let src =
        format!("{KERNELS}J = joint(p = K1, q = K2)\njc = jointchain(a = lawof(x), jj = J)\n");
    assert_eq!(fields(&src, "jc"), ["a", "jj"]);
}

#[test]
fn a_joint_kernel_step_takes_the_record_by_its_inputs() {
    // A scalar variate is not a record to splat.
    let src = format!("{KERNELS}J = joint(p = K1, q = K2)\njc = jointchain(lawof(x), J)\n");
    let (_, errors) = infer_src(&src);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("needs a record to splat"), "{errors:?}");
    // A field that is no input of the joint, and an input that has no field, are errors.
    let src =
        format!("{KERNELS}J = joint(p = K1, q = K2)\njc = jointchain(lawof(record(c = x)), J)\n");
    let (_, errors) = infer_src(&src);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("no input for field `c`") && errors[0].contains("lacks field `a`"));
}
