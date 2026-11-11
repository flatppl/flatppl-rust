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

fn mass_of(m: &flatppl_core::Module, name: &str) -> flatppl_core::Mass {
    let rhs = m
        .bindings()
        .find(|(_, b)| m.resolve(b.name) == name)
        .unwrap()
        .1
        .rhs;
    match m.type_of(rhs) {
        Some(Type::Measure { mass, .. } | Type::Kernel { mass, .. }) => *mass,
        other => panic!("{name} is not a measure or kernel: {other:?}"),
    }
}

/// §04, §06: a broadcast of probability measures has mass 1 whatever its length. A step
/// kernel broadcasting over an array field of the preceding variate cannot know that
/// length while the field is a placeholder, but its mass is still 1, so `rand` accepts
/// the chain.
#[test]
fn a_step_kernel_broadcasting_over_an_array_field_is_normalized() {
    for body in ["Normal.(0.0, r.v)", "Normal.(0.0, sqrt.(r.v))"] {
        let src = format!(
            "k = r -> joint(y = {body})\n\
             prior = jointchain(joint(v = iid(Gamma(2.0, 1.0), 2)), k)\n\
             seed = rnginit([0])\n\
             d = rand(prior, seed)\n"
        );
        let (m, errors) = infer_src(&src);
        assert!(errors.is_empty(), "{body}: {errors:?}");
        assert_eq!(mass_of(&m, "k"), flatppl_core::Mass::Normalized, "{body}");
        assert_eq!(
            mass_of(&m, "prior"),
            flatppl_core::Mass::Normalized,
            "{body}"
        );
    }
}

fn domain_of(m: &flatppl_core::Module, name: &str) -> Type {
    let rhs = m
        .bindings()
        .find(|(_, b)| m.resolve(b.name) == name)
        .unwrap()
        .1
        .rhs;
    match m.type_of(rhs) {
        Some(Type::Measure { domain, .. }) => domain.as_ref().clone(),
        other => panic!("{name} is not a measure: {other:?}"),
    }
}

fn real_vector(n: u32) -> Type {
    Type::Array {
        shape: Box::new([flatppl_core::Dim::Static(n)]),
        elem: Box::new(Type::Scalar(flatppl_core::ScalarType::Real)),
    }
}

fn record(m: &mut flatppl_core::Module, fields: &[(&str, Type)]) -> Type {
    Type::Record(
        fields
            .iter()
            .map(|(name, ty)| (m.intern(name), ty.clone()))
            .collect(),
    )
}

/// §06: a step kernel's inputs bind to the variates to its left, so its variate is its
/// body typed at them. Unapplied, `Normal.(0.0, sqrt.(r.v))` has no shape; applied to
/// `record(v = real[2])` it is `real[2]`.
#[test]
fn a_step_broadcasting_over_a_field_types_at_the_preceding_variates() {
    let (mut m, errors) = infer_src(
        "prior = jointchain(joint(v = iid(Gamma(2.0, 1.0), 2)), r -> joint(y = Normal.(0.0, sqrt.(r.v))))\n",
    );
    assert!(errors.is_empty(), "{errors:?}");
    let want = record(&mut m, &[("v", real_vector(2)), ("y", real_vector(2))]);
    assert_eq!(domain_of(&m, "prior"), want);
}

/// Each later step is typed at every variate to its left, through the keyword form too.
#[test]
fn every_step_of_a_longer_chain_types_at_its_prior() {
    let (mut m, errors) = infer_src(
        "prior = jointchain(joint(v = iid(Gamma(2.0, 1.0), 3)), \
         r -> joint(y = Normal.(0.0, sqrt.(r.v))), \
         r -> joint(w = Normal.(r.y, 1.0)))\n\
         named = jointchain(v = iid(Gamma(2.0, 1.0), 3), y = r -> Normal.(0.0, sqrt.(r.v)), \
         w = r -> Normal.(r.y, 1.0))\n",
    );
    assert!(errors.is_empty(), "{errors:?}");
    let want = record(
        &mut m,
        &[
            ("v", real_vector(3)),
            ("y", real_vector(3)),
            ("w", real_vector(3)),
        ],
    );
    assert_eq!(domain_of(&m, "prior"), want);
    assert_eq!(domain_of(&m, "named"), want);
}
