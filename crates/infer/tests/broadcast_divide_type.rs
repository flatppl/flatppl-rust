//! §07 `divide` is `a / b` over the reals: two integers divide to a real. The
//! elementwise form `a ./ b` must type each cell the same way, so an integer
//! array divided by an integer array is an array of reals, as `k ./ m` on two
//! integer scalars already was.

use flatppl_core::{ScalarType, Type};
use flatppl_infer::infer;

fn type_of(src: &str, name: &str) -> Type {
    let mut m = flatppl_syntax::parse(src).unwrap();
    assert!(infer(&mut m).is_empty());
    let sym = m.intern(name);
    let rhs = m.binding(m.binding_by_name(sym).unwrap()).rhs;
    m.type_of(rhs).unwrap().clone()
}

fn elem(ty: &Type) -> &Type {
    match ty {
        Type::Array { elem, .. } => elem,
        other => other,
    }
}

#[test]
fn integer_cells_divide_to_reals() {
    let src = "a = [3, 5]\nb = [10, 8]\nk = 2\nr = a ./ b\ns = a ./ k\nt = k ./ b\nq = k ./ k\n";
    for name in ["r", "s", "t", "q"] {
        let ty = type_of(src, name);
        assert_eq!(
            elem(&ty),
            &Type::Scalar(ScalarType::Real),
            "`{name}` is {ty:?}"
        );
    }
}

#[test]
fn integer_cells_keep_integer_sums_and_products() {
    let src = "a = [3, 5]\nb = [10, 8]\nr = a .+ b\ns = a .* b\n";
    for name in ["r", "s"] {
        let ty = type_of(src, name);
        assert_eq!(
            elem(&ty),
            &Type::Scalar(ScalarType::Integer),
            "`{name}` is {ty:?}"
        );
    }
}
