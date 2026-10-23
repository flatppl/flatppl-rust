//! Spec §06 keyword forms of `joint` and `jointchain`: "Each name holds its component's
//! variate as it is, whether scalar, array or record, so `joint(obs = iid(Normal(0, 1), 3))`
//! is a measure over records `record(obs = v)` with a vector `v`. A record-valued component
//! becomes a nested record under its name."
use flatppl_core::{Dim, Symbol, Type};
use flatppl_infer::{Severity, infer};

fn domain_of(source: &str, name: &str) -> (flatppl_core::Module, Type) {
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
    let Some(Type::Measure { domain, .. }) = module.type_of(rhs).cloned() else {
        panic!("{name} is not a measure");
    };
    (module, *domain)
}

fn field<'a>(module: &flatppl_core::Module, fields: &'a [(Symbol, Type)], name: &str) -> &'a Type {
    &fields
        .iter()
        .find(|(n, _)| module.resolve(*n) == name)
        .unwrap_or_else(|| panic!("no field {name}"))
        .1
}

#[test]
fn keyword_joint_holds_an_array_component_whole() {
    let (module, domain) = domain_of("m = joint(obs = iid(Normal(0, 1), 3))\n", "m");
    let Type::Record(fields) = domain else {
        panic!("{domain:?}");
    };
    assert!(matches!(
        field(&module, &fields, "obs"),
        Type::Array { shape, .. } if shape.as_ref() == [Dim::Static(3)]
    ));
}

#[test]
fn keyword_jointchain_holds_array_and_record_components_whole() {
    let (module, domain) = domain_of(
        "a = draw(Normal(0, 1))\nb = draw(Normal(1, 2))\nM = lawof(record(a = a, b = b))\n\
         theta = elementof(cartprod(a = reals, b = reals))\n\
         K = functionof(Normal(mu = theta.a + theta.b, sigma = 1.0), theta = theta)\n\
         j = jointchain(theta = M, o = K)\n",
        "j",
    );
    let Type::Record(fields) = domain else {
        panic!("{domain:?}");
    };
    // The record-valued component nests under its name, and `K` takes it by that name.
    assert!(matches!(field(&module, &fields, "theta"), Type::Record(inner) if inner.len() == 2));
    assert!(matches!(field(&module, &fields, "o"), Type::Scalar(_)));

    let (module, domain) = domain_of(
        "v = draw(iid(Normal(0, 1), 3))\nM = lawof(v)\nw = elementof(cartpow(reals, 3))\n\
         K = functionof(Normal(mu = sum(w), sigma = 1.0), w = w)\nj = jointchain(w = M, o = K)\n",
        "j",
    );
    let Type::Record(fields) = domain else {
        panic!("{domain:?}");
    };
    assert!(matches!(
        field(&module, &fields, "w"),
        Type::Array { shape, .. } if shape.as_ref() == [Dim::Static(3)]
    ));
}

#[test]
fn a_kernel_reads_a_record_component_by_its_label_not_by_its_fields() {
    // `K` has the inputs `mu` and `sigma`, but the record `prior` is nested under `theta`,
    // so no input names a preceding label: a static error.
    let mut module = flatppl_syntax::parse(
        "mu = elementof(reals)\nsigma = elementof(posreals)\n\
         prior = joint(mu = Normal(0.0, 1.0), sigma = Exponential(1.0))\n\
         K = functionof(Normal(mu = mu, sigma = sigma), mu = mu, sigma = sigma)\n\
         jc = jointchain(theta = prior, y = K)\n",
    )
    .unwrap();
    let found: Vec<String> = infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect();
    assert!(
        found
            .iter()
            .any(|m| m.contains("names no preceding component")),
        "{found:?}"
    );
}
