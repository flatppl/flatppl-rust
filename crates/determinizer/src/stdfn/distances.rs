//! §09 distances expressed through broadcasts, reductions, and scalar math.

use super::*;

pub(super) fn lower_function(m: &mut Module, args: &[NodeId], member: &str) -> Option<NodeId> {
    match (member, args) {
        ("pairwise_distance", [distance, x]) => Some(cross(m, *distance, *x, *x)),
        ("cross_distance", [distance, x, y]) => Some(cross(m, *distance, *x, *y)),
        ("minkowski", [u, v, p]) => {
            let delta = broadcast(m, "sub", &[*u, *v]);
            let delta = broadcast(m, "abs", &[delta]);
            let scale = build_call(m, "maximum", &[delta]);
            let zero = lit(m, 0.0);
            let one = lit(m, 1.0);
            let positive = build_call(m, "gt", &[scale, zero]);
            let safe = build_call(m, "ifelse", &[positive, scale, one]);
            let normalized = broadcast(m, "divide", &[delta, safe]);
            let powers = broadcast(m, "pow", &[normalized, *p]);
            let total = build_call(m, "sum", &[powers]);
            let inverse = divide(m, one, *p);
            let norm = build_call(m, "pow", &[total, inverse]);
            Some(mul(m, safe, norm))
        }
        ("jensenshannon", [u, v]) => Some(jensen_shannon(m, *u, *v)),
        _ => None,
    }
}

/// The symmetric component formula avoids cancellation between the two KL
/// terms. For nearby vectors, its even series retains the small distance.
fn jensen_shannon(m: &mut Module, u: NodeId, v: NodeId) -> NodeId {
    let zero = lit(m, 0.0);
    let one = lit(m, 1.0);
    let two = lit(m, 2.0);
    let sum = broadcast(m, "add", &[u, v]);
    let positive = broadcast(m, "gt", &[sum, zero]);
    let safe_sum = broadcast(m, "ifelse", &[positive, sum, one]);
    let difference = broadcast(m, "sub", &[u, v]);
    let d = broadcast(m, "divide", &[difference, safe_sum]);
    let d2 = broadcast(m, "mul", &[d, d]);
    let mut polynomial = lit(m, 1.0 / (8.0 * 15.0));
    for k in (1..8).rev() {
        let coefficient = lit(m, 1.0 / (k as f64 * (2 * k - 1) as f64));
        polynomial = broadcast(m, "mul", &[polynomial, d2]);
        polynomial = broadcast(m, "add", &[coefficient, polynomial]);
    }
    let near = broadcast(m, "mul", &[d2, polynomial]);
    let plus = broadcast(m, "add", &[one, d]);
    let minus = broadcast(m, "sub", &[one, d]);
    let pos = broadcast(m, "gt", &[plus, zero]);
    let neg = broadcast(m, "gt", &[minus, zero]);
    let dp = broadcast(m, "ifelse", &[pos, d, zero]);
    let dn = broadcast(m, "neg", &[d]);
    let dn = broadcast(m, "ifelse", &[neg, dn, zero]);
    let lp = broadcast(m, "log1p", &[dp]);
    let ln = broadcast(m, "log1p", &[dn]);
    let left = broadcast(m, "mul", &[plus, lp]);
    let right = broadcast(m, "mul", &[minus, ln]);
    let far = broadcast(m, "add", &[left, right]);
    let magnitude = broadcast(m, "abs", &[d]);
    let threshold = lit(m, 0.1);
    let close = broadcast(m, "le", &[magnitude, threshold]);
    let value = broadcast(m, "ifelse", &[close, near, far]);
    let four = mul(m, two, two);
    let weighted = broadcast(m, "mul", &[sum, value]);
    let weighted = broadcast(m, "divide", &[weighted, four]);
    let total = build_call(m, "sum", &[weighted]);
    build_call(m, "sqrt", &[total])
}

fn cross(m: &mut Module, distance: NodeId, x: NodeId, y: NodeId) -> NodeId {
    let zero = m.alloc(Node::Lit(Scalar::Int(0)));
    let one = m.alloc(Node::Lit(Scalar::Int(1)));
    let rows = build_call(m, "addaxes", &[x, zero, one]);
    let columns = build_call(m, "addaxes", &[y, one, zero]);
    build_call(m, "broadcast", &[distance, rows, columns])
}

fn broadcast(m: &mut Module, name: &str, args: &[NodeId]) -> NodeId {
    let name = m.intern(name);
    let head = m.alloc(Node::Const(name));
    let mut args = args.to_vec();
    args.insert(0, head);
    build_call(m, "broadcast", &args)
}
