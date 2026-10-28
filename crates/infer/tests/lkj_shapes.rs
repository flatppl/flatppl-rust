use flatppl_core::{Dim, Mass, ScalarType, Type};

fn inferred(source: &str, name: &str) -> Type {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut module);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let (_, binding) = module
        .bindings()
        .find(|(_, binding)| module.resolve(binding.name) == name)
        .unwrap();
    module.type_of(binding.rhs).unwrap().clone()
}

fn matrix(dimension: Dim) -> Type {
    Type::Array {
        shape: Box::new([dimension, dimension]),
        elem: Box::new(Type::Scalar(ScalarType::Real)),
    }
}

#[test]
fn known_lkj_dimensions_reach_constructors_and_sampling_primitives() {
    for constructor in ["LKJ", "LKJCholesky"] {
        for law in [
            format!("{constructor}(3, 1.5)"),
            format!("{constructor}(eta = 1.5, n = n)"),
        ] {
            let source = format!("start = 2\nn = start + 1\nmeasure = {law}");
            assert_eq!(
                inferred(&source, "measure"),
                Type::Measure {
                    domain: Box::new(matrix(Dim::Static(3))),
                    mass: Mass::Normalized,
                },
                "{law}"
            );
        }
        let source = format!(
            "start = 2\nn = start + 1\nparameters = record(n = n, eta = 1.5)\n\
             result = builtin_sample(rnginit(0), {constructor}, parameters)"
        );
        assert_eq!(
            inferred(&source, "result"),
            Type::Tuple(Box::new([matrix(Dim::Static(3)), Type::RngState]))
        );
    }
}

#[test]
fn unresolved_lkj_dimensions_remain_dynamic() {
    for constructor in ["LKJ", "LKJCholesky"] {
        let source = format!(
            "n = external(posintegers)\nmeasure = {constructor}(n, 1.5)\n\
             result = builtin_sample(rnginit(0), {constructor}, record(n = n, eta = 1.5))"
        );
        assert_eq!(
            inferred(&source, "measure"),
            Type::Measure {
                domain: Box::new(matrix(Dim::Dynamic)),
                mass: Mass::Normalized,
            }
        );
        assert_eq!(
            inferred(&source, "result"),
            Type::Tuple(Box::new([matrix(Dim::Dynamic), Type::RngState]))
        );
    }
}
