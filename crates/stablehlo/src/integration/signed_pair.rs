//! Pair complementary positive cubatures without changing surface measures.

use flatppl_core::graph::expect_builtin_call;

use super::*;

struct Half<'m> {
    integral: &'m [NodeId; 7],
    score: NodeId,
    log_weight: Option<(NodeId, NodeId)>,
}

impl Emitter<'_> {
    pub(crate) fn lower_signed_rectangle_difference(
        &mut self,
        id: NodeId,
        left: NodeId,
        right: NodeId,
    ) -> Option<Result<Value, EmitError>> {
        if self.integration.is_none() || self.integrating {
            return None;
        }
        let (left, right) = (half(self, left)?, half(self, right)?);
        let [right_score] = *call(self, right.score, "neg")? else {
            return None;
        };
        let [_, lx, lx0, lx1, ly, ly0, ly1] = *left.integral;
        let [_, rx, rx0, rx1, ry, ry0, ry1] = *right.integral;
        let same_bounds = same(
            self,
            (&[], &[]),
            [(lx0, rx0), (lx1, rx1), (ly0, ry0), (ly1, ry1)],
        );
        let points = (&[lx, ly][..], &[rx, ry][..]);
        if !same_bounds || !same(self, points, [(left.score, right_score)]) {
            return None;
        }
        match (left.log_weight, right.log_weight) {
            (None, None) => {}
            (Some((weight, normalizer)), Some((other_weight, other_normalizer)))
                if same(self, points, [(weight, other_weight)])
                    && same(self, (&[], &[]), [(normalizer, other_normalizer)]) => {}
            _ => return None,
        }
        Some(self.lower_integral(
            id,
            left.integral,
            IntegralKind::Signed {
                score: left.score,
                log_weight: left.log_weight,
            },
        ))
    }
}

fn half<'m>(emitter: &Emitter<'m>, node: NodeId) -> Option<Half<'m>> {
    let (integral, part, log_weight) = if let Some([log]) = call(emitter, node, "exp") {
        let integral: &[NodeId; 7] = call(emitter, *log, flatppl_core::LOG_INTEGRAL)?
            .try_into()
            .ok()?;
        let [body, x, _, _, y, _, _] = *integral;
        let [masked, normalizer] = *call(emitter, body, "sub")? else {
            return None;
        };
        let [outside, zero, weighted] = *call(emitter, masked, "ifelse")? else {
            return None;
        };
        let [weight, part] = *call(emitter, weighted, "add")? else {
            return None;
        };
        let [outside_weight] = *call(emitter, outside, "isinf")? else {
            return None;
        };
        let [part] = *call(emitter, part, "log")? else {
            return None;
        };
        if !minus_infinity(emitter, zero)
            || !same(emitter, (&[x, y], &[x, y]), [(weight, outside_weight)])
        {
            return None;
        }
        (integral, part, Some((weight, normalizer)))
    } else {
        let integral: &[NodeId; 7] = call(emitter, node, flatppl_core::MASS_INTEGRAL)?
            .try_into()
            .ok()?;
        let [body, ..] = *integral;
        (integral, body, None)
    };
    let [score, zero] = *call(emitter, part, "max")? else {
        return None;
    };
    let zero = matches!(
        emitter.node(alias(emitter, zero)),
        Node::Lit(Scalar::Real(x)) if x.to_bits() == 0
    );
    zero.then_some(Half {
        integral,
        score,
        log_weight,
    })
}

fn minus_infinity(emitter: &Emitter<'_>, node: NodeId) -> bool {
    let node = alias(emitter, node);
    if matches!(emitter.node(node), Node::Lit(Scalar::Real(x)) if *x == f64::NEG_INFINITY) {
        return true;
    }
    let Some([argument]) = call(emitter, node, "neg") else {
        return false;
    };
    match emitter.node(alias(emitter, *argument)) {
        Node::Const(name) => emitter.resolve(*name) == "inf",
        Node::Lit(Scalar::Real(x)) => *x == f64::INFINITY,
        _ => false,
    }
}

fn call<'m>(emitter: &Emitter<'m>, node: NodeId, name: &str) -> Option<&'m [NodeId]> {
    let node = alias(emitter, node);
    if emitter.memo.contains_key(&node) {
        return None;
    }
    let call = expect_builtin_call(emitter.m, node, name)?;
    (call.named.is_empty() && call.inputs.is_none()).then_some(&call.args)
}

// Promoted query or callback inputs can have authored right-hand sides.
// The emitter has already bound their runtime values in the active scope.
fn alias(emitter: &Emitter<'_>, mut node: NodeId) -> NodeId {
    for _ in 0..emitter.m.binding_count() {
        let next = emitter.resolve_ref_one(node);
        if next == node || emitter.memo.contains_key(&node) || emitter.memo.contains_key(&next) {
            break;
        }
        node = next;
    }
    node
}

fn same(
    emitter: &Emitter<'_>,
    (left_axes, right_axes): (&[NodeId], &[NodeId]),
    roots: impl IntoIterator<Item = (NodeId, NodeId)>,
) -> bool {
    let mut pending = roots.into_iter().collect::<Vec<_>>();
    let mut seen = HashSet::new();
    while let Some((left, right)) = pending.pop() {
        let (left, right) = (alias(emitter, left), alias(emitter, right));
        let left_axis = left_axes.iter().position(|&point| point == left);
        let right_axis = right_axes.iter().position(|&point| point == right);
        if left_axis != right_axis {
            return false;
        }
        if left_axis.is_some() || left == right || !seen.insert((left, right)) {
            continue;
        }
        if let (Some(a), Some(b)) = (emitter.memo.get(&left), emitter.memo.get(&right)) {
            if a.ssa == b.ssa && a.ty == b.ty && a.elem == b.elem {
                continue;
            }
            return false;
        }
        if emitter.memo.contains_key(&left) || emitter.memo.contains_key(&right) {
            return false;
        }
        let (a, b) = (emitter.node(left), emitter.node(right));
        match (a, b) {
            (Node::Lit(Scalar::Real(a)), Node::Lit(Scalar::Real(b)))
                if a.to_bits() == b.to_bits() => {}
            (Node::Lit(Scalar::Real(_)), Node::Lit(Scalar::Real(_))) => return false,
            (Node::Call(a), Node::Call(b))
                if a.inputs.is_none()
                    && b.inputs.is_none()
                    && a.args.len() == b.args.len()
                    && a.named.len() == b.named.len() =>
            {
                match (a.head, b.head) {
                    (CallHead::Builtin(a), CallHead::Builtin(b)) if a == b => {}
                    (CallHead::User(a), CallHead::User(b)) => pending.push((a, b)),
                    _ => return false,
                }
                pending.extend(a.args.iter().copied().zip(b.args.iter().copied()));
                for (a, b) in a.named.iter().zip(b.named.iter()) {
                    if a.kind != b.kind || a.name != b.name {
                        return false;
                    }
                    pending.push((a.value, b.value));
                }
            }
            (Node::Call(_), _) | (_, Node::Call(_)) => return false,
            (a, b) if a == b => {}
            _ => return false,
        }
    }
    true
}
