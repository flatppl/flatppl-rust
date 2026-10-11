use flatppl_core::{Dim, Mass, Module, ScalarType, Type, ValueSet};

fn infer(source: &str) -> Module {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    module
}

#[test]
fn rebase_is_normalized_and_keeps_the_base_support() {
    let mut module = infer("q = rebase(iid(Normal(0, 1), 3), Lebesgue(zerosum(3)))\nx ~ q");
    let q = module.intern("q");
    let q = module.binding(module.binding_by_name(q).unwrap()).rhs;
    assert_eq!(
        module.type_of(q),
        Some(&Type::Measure {
            domain: Box::new(Type::Array {
                shape: vec![Dim::Static(3)].into(),
                elem: Box::new(Type::Scalar(ScalarType::Real)),
            }),
            mass: Mass::Normalized,
        })
    );
    assert_eq!(
        module.valueset_of(q),
        Some(&ValueSet::ZeroSum(Dim::Static(3)))
    );
}

#[test]
fn rebase_kernel_uses_the_first_nonnullary_input_order() {
    let source = "a = elementof(reals)\nb = elementof(posreals)\n\
                  left = functionof(Normal(a, b), a = a, b = b)\n\
                  right = functionof(Normal(a, b), b = b, a = a)\n";
    for (expression, order) in [
        ("rebase(left, right)", ["a", "b"]),
        ("rebase(Normal(0, 1), right)", ["b", "a"]),
    ] {
        let mut module = infer(&format!("{source}q = {expression}\nx ~ q(a = 0, b = 1)"));
        let q = module.intern("q");
        let q = module.binding(module.binding_by_name(q).unwrap()).rhs;
        let inputs = order.map(|name| module.intern(name));
        assert_eq!(
            module.type_of(q),
            Some(&Type::Kernel {
                inputs: inputs.into(),
                mass: Mass::Normalized,
            })
        );
    }
}

#[test]
fn rebase_rejects_wrong_operands_shapes_and_interfaces() {
    for source in [
        "q = rebase(Normal(0, 1))",
        "q = rebase(M = Normal(0, 1), B = Normal(0, 1))",
        "q = rebase(1, Normal(0, 1))",
        "q = rebase(iid(Normal(0, 1), 2), Lebesgue(zerosum(3)))",
        "a = elementof(reals)\nb = elementof(reals)\nq = rebase(functionof(Normal(a, 1), a = a), functionof(Normal(b, 1), b = b))",
        "a = elementof(reals)\nb = elementof(cartpow(reals, 2))\nq = rebase(functionof(Normal(a, 1), x = a), functionof(Normal(sum(b), 1), x = b))",
    ] {
        let mut module = flatppl_syntax::parse(source).unwrap();
        let diagnostics = flatppl_infer::infer(&mut module);
        assert!(!diagnostics.is_empty(), "must reject {source}");
    }
}
