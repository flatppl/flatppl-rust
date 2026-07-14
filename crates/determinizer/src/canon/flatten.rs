//! Pass 3: resolve statically-known structural projections — `get(record, k)`
//! and `get0(vector, i)` over literal constructors (Buffy #263).
//!
//! Density lowering routinely builds a `get`/`get0` accessor onto a container
//! it cannot read field-wise from the node at build time (§04 auto-splat onto
//! an opaque multi-output call; `joint`'s positional `get0(v, i)` onto the
//! score variate). Once Pass 2 (`inline_user_calls`) beta-reduces the opaque
//! call away, or when the container was always a literal `vector`/`record`
//! constructor (a literal-variate score), the accessor's container becomes
//! statically known — the projection can be resolved to the element directly
//! rather than left for a consuming engine to evaluate. Dynamic containers
//! (a `Ref`, a still-opaque call, a computed index) are left untouched:
//! refuse-free, matching every other canon pass.

use flatppl_core::{CallHead, Module, NamedKind, Node, NodeId, Scalar};

use crate::driver::map_tree;

pub(crate) fn flatten_structural(m: &mut Module) -> bool {
    let mut changed = false;
    let pairs: Vec<(flatppl_core::BindingId, NodeId)> =
        m.bindings().map(|(bid, b)| (bid, b.rhs)).collect();
    for (bid, root) in pairs {
        let new = map_tree(m, root, &mut project);
        if new != root {
            m.set_binding_rhs(bid, new);
            changed = true;
        }
    }
    changed
}

/// `get(record(%field k v ...), "k") -> v`; `get0(vector(e0, e1, ...), i) -> ei`
/// for a literal `i`. Returns an EXISTING child NodeId (no alloc needed), so the
/// `&Module` closure signature suffices. Dynamic index/field → None.
fn project(m: &Module, id: NodeId) -> Option<NodeId> {
    let Node::Call(c) = m.node(id) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    match m.resolve(sym) {
        "get" if c.args.len() == 2 => {
            let (container, key) = (c.args[0], c.args[1]);
            let Node::Lit(Scalar::Str(field)) = m.node(key) else {
                return None;
            };
            let rec = expect_builtin(m, container, "record")?;
            rec.named
                .iter()
                .find(|na| na.kind == NamedKind::Field && m.resolve(na.name) == &**field)
                .map(|na| na.value)
        }
        "get0" if c.args.len() == 2 => {
            let (container, idx) = (c.args[0], c.args[1]);
            let Node::Lit(Scalar::Int(i)) = m.node(idx) else {
                return None;
            };
            let vec = expect_builtin(m, container, "vector")?;
            usize::try_from(*i)
                .ok()
                .and_then(|i| vec.args.get(i))
                .copied()
        }
        _ => None,
    }
}

fn expect_builtin<'a>(m: &'a Module, id: NodeId, name: &str) -> Option<&'a flatppl_core::Call> {
    let Node::Call(c) = m.node(id) else {
        return None;
    };
    let CallHead::Builtin(s) = c.head else {
        return None;
    };
    (m.resolve(s) == name).then_some(c)
}
