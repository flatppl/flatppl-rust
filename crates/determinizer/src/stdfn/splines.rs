//! Portable thin-plate basis and penalty, using tensor operations only.

use super::*;

pub(super) fn lower_function(m: &mut Module, args: &[NodeId], member: &str) -> Option<NodeId> {
    let centers = resolve_ref_one(m, *args.last()?).0;
    if matches!(m.type_of(centers), Some(Type::Array { shape, .. })
        if shape.as_ref() == [flatppl_core::Dim::Static(0)])
    {
        return None;
    }
    match member {
        "thinplate_basis" => {
            let [x, centers] = arity::<2>(args)?;
            let radial = radial(m, x, centers);
            let one = lit(m, 1.0);
            let affine = build_call(m, "vector", &[one, x]);
            Some(build_call(m, "cat", &[radial, affine]))
        }
        "thinplate_penalty" => {
            let [centers] = arity::<1>(args)?;
            let count = build_call(m, "lengthof", &[centers]);
            let zero = lit(m, 0.0);
            let padding = build_call(m, "vector", &[zero, zero]);
            let points = build_call(m, "cat", &[centers, padding]);
            let yes = m.alloc(Node::Lit(Scalar::Bool(true)));
            let no = m.alloc(Node::Lit(Scalar::Bool(false)));
            let active = build_call(m, "fill", &[yes, count]);
            let padding = build_call(m, "vector", &[no, no]);
            let active = build_call(m, "cat", &[active, padding]);
            let rows = axis(m, points, false);
            let cols = axis(m, points, true);
            let kernel = radial(m, rows, cols);
            let rows = axis(m, active, false);
            let cols = axis(m, active, true);
            let active = broadcast(m, "land", &[rows, cols]);
            Some(broadcast(m, "ifelse", &[active, kernel, zero]))
        }
        _ => None,
    }
}

fn radial(m: &mut Module, x: NodeId, y: NodeId) -> NodeId {
    let difference = broadcast(m, "sub", &[x, y]);
    let distance = broadcast(m, "abs", &[difference]);
    let square = broadcast(m, "mul", &[distance, distance]);
    let cube = broadcast(m, "mul", &[square, distance]);
    let twelve = lit(m, 12.0);
    broadcast(m, "divide", &[cube, twelve])
}

fn axis(m: &mut Module, value: NodeId, leading: bool) -> NodeId {
    let zero = m.alloc(Node::Lit(Scalar::Int(0)));
    let one = m.alloc(Node::Lit(Scalar::Int(1)));
    let (before, after) = if leading { (one, zero) } else { (zero, one) };
    build_call(m, "addaxes", &[value, before, after])
}

fn broadcast(m: &mut Module, name: &str, args: &[NodeId]) -> NodeId {
    let name = m.intern(name);
    let head = m.alloc(Node::Const(name));
    let mut args = args.to_vec();
    args.insert(0, head);
    build_call(m, "broadcast", &args)
}
