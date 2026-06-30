use crate::refuse::{NonConformKind, NonConformance};
use flatppl_core::{CallHead, Module, Node, NodeId, Phase, Type};

/// FlatPDL conformance over `flatppl-infer` output: no `Measure`/`Likelihood`-typed node;
/// `Kernel` type only as an argument of a `builtin_*` primitive (the constructor-tag arg —
/// its position varies: arg 0 for `builtin_logdensityof` and the transports, arg 1 for
/// `builtin_sample`); no `Stochastic` phase. Pure read of the inferred side-tables — run
/// `infer` first.
pub fn is_flatpdl(m: &Module) -> Result<(), Vec<NonConformance>> {
    let mut bad = Vec::new();
    for (_bid, binding) in m.bindings() {
        visit(m, binding.rhs, None, &mut bad);
    }
    if bad.is_empty() { Ok(()) } else { Err(bad) }
}

// `parent_builtin`: the interned name of the enclosing builtin call head, so a
// `Kernel`-typed node is allowed iff it sits inside a `builtin_*` call. The kernel arg's
// position varies by primitive, so the check is by-enclosing-call, not by-index; non-kernel
// args are never `Kernel`-typed, so this admits no stray kernel.

fn visit(m: &Module, id: NodeId, parent_builtin: Option<&str>, bad: &mut Vec<NonConformance>) {
    if matches!(m.phase_of(id), Some(Phase::Stochastic)) {
        bad.push(NonConformance {
            node: id,
            kind: NonConformKind::StochasticPhase,
            reason: "stochastic-phase node (a `draw` survives)".into(),
        });
    }
    match m.type_of(id) {
        Some(Type::Measure { .. }) => bad.push(NonConformance {
            node: id,
            kind: NonConformKind::MeasureTyped,
            reason: "measure-typed node".into(),
        }),
        Some(Type::Likelihood { .. }) => bad.push(NonConformance {
            node: id,
            kind: NonConformKind::LikelihoodTyped,
            reason: "likelihood-typed node".into(),
        }),
        Some(Type::Kernel { .. }) if !parent_builtin.is_some_and(|h| h.starts_with("builtin_")) => {
            bad.push(NonConformance {
                node: id,
                kind: NonConformKind::KernelNotBuiltinArg,
                reason: "kernel outside a builtin_* argument".into(),
            })
        }
        _ => {}
    }

    // Collect children and determine the builtin head symbol before recursing,
    // keeping the `m.node(id)` borrow scoped so it doesn't conflict with
    // `m.resolve(sym)` in the recursive call.
    let (children, head_sym) = {
        let node = m.node(id);
        let sym = match node {
            Node::Call(c) => match c.head {
                CallHead::Builtin(op) => Some(op),
                _ => None,
            },
            _ => None,
        };
        (node.children(), sym)
    };
    let this_builtin: Option<&str> = head_sym.map(|op| m.resolve(op));

    for child in children {
        visit(m, child, this_builtin, bad);
    }
}
