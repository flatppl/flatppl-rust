//! Three-term recurrence for the §09 orthogonal polynomials.

use flatppl_core::ScalarType;
use flatppl_core::graph::build_record;

use super::*;

// The finite runtime envelope matches the portable literal profile.
const MAX_DEGREE: i64 = 128;
const UNROLL_DEGREE: i64 = 8;

pub(super) fn lower(m: &mut Module, name: &str, degree: NodeId, x: NodeId) -> Option<NodeId> {
    let n = match m.node(resolve_ref(m, degree).0) {
        Node::Lit(Scalar::Int(n)) if (0..=MAX_DEGREE).contains(n) => Some(*n),
        Node::Lit(_) => return None,
        _ if m.type_of(degree) == Some(&Type::Scalar(ScalarType::Integer))
            && m.type_of(x) == Some(&Type::Scalar(ScalarType::Real)) =>
        {
            None
        }
        _ => return None,
    };
    let zero_degree = lit(m, 1.0);
    if n == Some(0) {
        return Some(zero_degree);
    }
    let first_degree = match name {
        "legendre" | "chebyshev" => x,
        "hermite" => scale(m, x, 2.0),
        "laguerre" => sub(m, zero_degree, x),
        _ => return None,
    };
    if let Some(n) = n.filter(|&n| n <= UNROLL_DEGREE) {
        let mut prev = zero_degree;
        let mut curr = first_degree;
        for k in 1..n {
            let k = lit(m, k as f64);
            let next = step(m, name, k, x, prev, curr);
            (prev, curr) = (curr, next);
        }
        return Some(curr);
    }

    // Keep the recurrence in a bounded scan so deployment source and AD remain
    // linear in the number of steps, including a runtime degree near the cap.
    let prev_name = m.intern("previous");
    let curr_name = m.intern("current");
    let start = build_record(m, &[(prev_name, zero_degree), (curr_name, first_degree)]);
    let carry = local(m, "_poly_carry_");
    let k = local(m, "_poly_index_");
    let params = local(m, "_poly_params_");
    let prev = field(m, carry, prev_name);
    let curr = field(m, carry, curr_name);
    let next = step(m, name, k, params, prev, curr);
    let body = build_record(m, &[(prev_name, curr), (curr_name, next)]);
    let inputs = [carry, k, params].map(|node| {
        let Node::Ref(reference) = m.node(node) else {
            unreachable!()
        };
        (reference.name, *reference)
    });
    let function = m.intern("functionof");
    let callback = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(function),
        args: Box::new([body]),
        named: Box::new([]),
        inputs: Some(Inputs::Spec(Box::new(inputs))),
    }));
    let count = n.unwrap_or(MAX_DEGREE) - 1;
    let indices: Vec<_> = (1..=count).map(|k| lit(m, k as f64)).collect();
    let sequence = build_call(m, "vector", &indices);
    let scan = build_call(m, "scan", &[callback, start, sequence, x]);
    let column = field(m, scan, curr_name);
    if n.is_some() {
        let last = integer(m, count);
        return Some(build_call(m, "get", &[column, last]));
    }

    let zero = integer(m, 0);
    let one = integer(m, 1);
    let two = integer(m, 2);
    let max = integer(m, MAX_DEGREE);
    let in_domain = build_call(m, "ge", &[degree, zero]);
    let bounded = build_call(m, "le", &[degree, max]);
    let valid = build_call(m, "land", &[in_domain, bounded]);
    let capped = build_call(m, "min", &[degree, max]);
    let safe = build_call(m, "max", &[capped, two]);
    let index = build_call(m, "sub", &[safe, one]);
    let result = build_call(m, "get", &[column, index]);
    let positive = build_call(m, "gt", &[degree, zero]);
    let initial = build_call(m, "ifelse", &[positive, first_degree, zero_degree]);
    let recurrence = build_call(m, "ge", &[degree, two]);
    let value = build_call(m, "ifelse", &[recurrence, result, initial]);
    let nan = lit(m, f64::NAN);
    Some(build_call(m, "ifelse", &[valid, value, nan]))
}

fn step(m: &mut Module, name: &str, k: NodeId, x: NodeId, prev: NodeId, curr: NodeId) -> NodeId {
    let one = lit(m, 1.0);
    let twice_k = scale(m, k, 2.0);
    let lead = add(m, twice_k, one);
    let (coefficient, trailing) = match name {
        "legendre" => (mul(m, lead, x), k),
        "laguerre" => (sub(m, lead, x), k),
        "hermite" => (scale(m, x, 2.0), twice_k),
        "chebyshev" => (scale(m, x, 2.0), one),
        _ => unreachable!(),
    };
    let current = mul(m, coefficient, curr);
    let previous = mul(m, trailing, prev);
    let next = sub(m, current, previous);
    if matches!(name, "legendre" | "laguerre") {
        let denominator = add(m, k, one);
        divide(m, next, denominator)
    } else {
        next
    }
}

fn local(m: &mut Module, name: &str) -> NodeId {
    let name = m.intern(name);
    m.alloc(Node::Ref(Ref {
        ns: RefNs::Local,
        name,
    }))
}

fn field(m: &mut Module, record: NodeId, name: flatppl_core::Symbol) -> NodeId {
    let key = m.alloc(Node::Lit(Scalar::Str(m.resolve(name).into())));
    build_call(m, "get", &[record, key])
}

fn integer(m: &mut Module, value: i64) -> NodeId {
    m.alloc(Node::Lit(Scalar::Int(value)))
}
