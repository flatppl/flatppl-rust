use flatppl_core::{Dim, Module, ScalarType, Type};
use flatppl_infer::{Severity, infer};

fn infer_ok(source: &str) -> Module {
    let mut module = flatppl_syntax::parse(source).unwrap();
    let diagnostics = infer(&mut module);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    module
}

fn binding_type(module: &mut Module, name: &str) -> Type {
    let name = module.intern(name);
    let rhs = module.binding(module.binding_by_name(name).unwrap()).rhs;
    module.type_of(rhs).unwrap().clone()
}

fn vector(kind: ScalarType) -> Type {
    Type::Array {
        shape: Box::new([Dim::Static(2)]),
        elem: Box::new(Type::Scalar(kind)),
    }
}

#[test]
fn array_scaling_promotes_complex_elements_in_both_operand_orders() {
    let mut module = infer_ok(
        "z = elementof(complexes)\nv = elementof(cartpow(reals, 2))\n\
         left = z * v\nright = v * z\n",
    );
    for name in ["left", "right"] {
        assert_eq!(binding_type(&mut module, name), vector(ScalarType::Complex));
    }
}

#[test]
fn dotted_scalar_functions_keep_their_complex_domain_and_codomain() {
    let mut module = infer_ok(
        "z = elementof(cartpow(complexes, 2))\n\
         magnitude = abs.(z)\nconjugate = conj.(z)\nangle = asinh.(z)\n",
    );
    assert_eq!(
        binding_type(&mut module, "magnitude"),
        vector(ScalarType::Real)
    );
    for name in ["conjugate", "angle"] {
        assert_eq!(binding_type(&mut module, name), vector(ScalarType::Complex));
    }
}

#[test]
fn scaling_by_a_real_norm_preserves_complex_vector_cells() {
    let mut module = infer_ok(
        "z = elementof(cartpow(complexes, 2))\n\
         one_norm = l1unit(z)\ntwo_norms = l2unit.([z, z])\n",
    );
    assert_eq!(
        binding_type(&mut module, "one_norm"),
        vector(ScalarType::Complex)
    );
    assert_eq!(
        binding_type(&mut module, "two_norms"),
        Type::Array {
            shape: Box::new([Dim::Static(2)]),
            elem: Box::new(vector(ScalarType::Complex)),
        },
    );
}

#[test]
fn polynomial_and_quadratic_form_widen_from_complex_second_argument() {
    let mut module = infer_ok(
        "z = elementof(complexes)\np = polynomial([1.0, 2.0], z)\n\
         A = rowstack([[0.0, 1.0], [0.0, 0.0]])\nq = quadform(A, [1.0, z])\n",
    );
    for name in ["p", "q"] {
        assert_eq!(
            binding_type(&mut module, name),
            Type::Scalar(ScalarType::Complex)
        );
    }
}

#[test]
fn complex_row_gram_and_outer_product_keep_static_square_shapes() {
    let mut module = infer_ok(
        "A = elementof(cartpow(complexes, [2, 3]))\nv = elementof(cartpow(complexes, 2))\n\
         gram = row_gram(A)\nouter = self_outer(v)\n",
    );
    let square = Type::Array {
        shape: Box::new([Dim::Static(2), Dim::Static(2)]),
        elem: Box::new(Type::Scalar(ScalarType::Complex)),
    };
    for name in ["gram", "outer"] {
        assert_eq!(binding_type(&mut module, name), square);
    }
}

#[test]
fn complex_diagonal_length_accounts_for_rectangular_shape_and_offset() {
    let mut module = infer_ok(
        "A = elementof(cartpow(complexes, [3, 2]))\nk = elementof(integers)\n\
         main = diag(A)\nupper = diag(A, 1)\nlower = diag(A = A, k = -2)\n\
         empty = diag(A, 2)\nunknown = diag(A, k)\n",
    );
    for (name, len) in [
        ("main", Dim::Static(2)),
        ("upper", Dim::Static(1)),
        ("lower", Dim::Static(1)),
        ("empty", Dim::Static(0)),
        ("unknown", Dim::Dynamic),
    ] {
        assert_eq!(
            binding_type(&mut module, name),
            Type::Array {
                shape: Box::new([len]),
                elem: Box::new(Type::Scalar(ScalarType::Complex)),
            }
        );
    }
}

#[test]
fn complex_values_do_not_enter_real_only_math_or_measures() {
    for source in [
        "z = elementof(complexes)\ny = floor(z)",
        "z = elementof(cartpow(complexes, 2))\ny = log2.(z)",
        "z = elementof(cartpow(complexes, 2))\ny = maximum(z)",
        "z = elementof(cartpow(complexes, 2))\ny = aggregate(var, [], z[.i])",
        "z = elementof(complexes)\ny = Normal(0.0, z)",
        "z = elementof(complexes)\ny = weighted(z, Normal(0.0, 1.0))",
    ] {
        let mut module = flatppl_syntax::parse(source).unwrap();
        let diagnostics = infer(&mut module);
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.severity == Severity::Error),
            "{source}: {diagnostics:?}"
        );
    }
}
