//! Read the original Gaussian constructor or kernel fields without changing the IR.

use flatppl_core::{Call, CallHead, Module, NamedKind, Node, NodeId, Scalar, ScalarType, Type};

use crate::Profile;
use crate::engine::{Fact, FactArg};

pub(super) struct Witness {
    pub(super) nodes: Vec<NodeId>,
    kind: Kind,
}

enum Kind {
    Law(&'static str),
    NormalOperation {
        head: &'static str,
        fixed_cdf: Option<[i64; 3]>,
    },
    MvNormalScore,
}

pub(super) fn witness(m: &Module, id: NodeId) -> Option<Witness> {
    let (call, name) = builtin(m, id)?;
    if matches!(m.type_of(id), Some(Type::Measure { .. })) {
        let (relation, fields) = match name {
            "Normal" => ("gaussian-normal", ["mu", "sigma"]),
            "MvNormal" => ("gaussian-mvnormal", ["mu", "cov"]),
            _ => return None,
        };
        return Some(Witness {
            nodes: parameters(m, call, fields, NamedKind::Kwarg)?.into(),
            kind: Kind::Law(relation),
        });
    }
    let head = match name {
        "builtin_logdensityof" => "builtin_logdensityof",
        "builtin_touniform" => "builtin_touniform",
        "builtin_tonormal" => "builtin_tonormal",
        "builtin_fromnormal" => "builtin_fromnormal",
        _ => return None,
    };
    let [kernel, input, point] = *call.args else {
        return None;
    };
    if !call.named.is_empty() || m.type_of(id) != Some(&Type::Scalar(ScalarType::Real)) {
        return None;
    }
    let Node::Const(kernel) = m.node(kernel) else {
        return None;
    };
    let (record, "record") = builtin(m, input)? else {
        return None;
    };
    if !record.args.is_empty() {
        return None;
    }
    match m.resolve(*kernel) {
        "Normal" => {
            let [location, scale] = parameters(m, record, ["mu", "sigma"], NamedKind::Field)?;
            let fixed_cdf = (head == "builtin_touniform")
                .then(|| fixed_cdf_bits(m, location, scale, point))
                .flatten();
            Some(Witness {
                nodes: vec![location, scale, point],
                kind: Kind::NormalOperation { head, fixed_cdf },
            })
        }
        "MvNormal" if head == "builtin_logdensityof" => {
            let [location, covariance] = parameters(m, record, ["mu", "cov"], NamedKind::Field)?;
            Some(Witness {
                nodes: vec![location, covariance, point],
                kind: Kind::MvNormalScore,
            })
        }
        _ => None,
    }
}

pub(super) fn seed(
    witness: &Witness,
    term: usize,
    args: &[usize],
    facts: &mut Vec<Fact>,
    profile: Profile,
) {
    let mut values = vec![FactArg::Expr(term)];
    let name = match witness.kind {
        Kind::Law(name) => name,
        Kind::NormalOperation { head, fixed_cdf } => {
            values.push(FactArg::Str(head));
            if let (Profile::StableHlo { .. }, Some(bits)) = (profile, fixed_cdf) {
                let mut raw = vec![FactArg::Expr(term)];
                raw.extend(bits.into_iter().map(FactArg::Int));
                facts.push(Fact {
                    name: "gaussian-fixed-cdf",
                    args: raw,
                });
            }
            "gaussian-native-normal"
        }
        Kind::MvNormalScore => "gaussian-native-mvnormal",
    };
    values.extend(args.iter().copied().map(FactArg::Expr));
    facts.push(Fact { name, args: values });
}

fn builtin(m: &Module, id: NodeId) -> Option<(&Call, &str)> {
    let Node::Call(call) = m.node(id) else {
        return None;
    };
    let CallHead::Builtin(head) = call.head else {
        return None;
    };
    call.inputs.is_none().then_some((call, m.resolve(head)))
}

fn parameters(m: &Module, call: &Call, names: [&str; 2], kind: NamedKind) -> Option<[NodeId; 2]> {
    if call.args.len() + call.named.len() != names.len() {
        return None;
    }
    let mut args = [None, None];
    for (slot, &id) in args.iter_mut().zip(call.args.iter()) {
        *slot = Some(id);
    }
    for arg in &call.named {
        if arg.kind != kind {
            return None;
        }
        let slot = names.iter().position(|&name| name == m.resolve(arg.name))?;
        if args[slot].replace(arg.value).is_some() {
            return None;
        }
    }
    Some([args[0]?, args[1]?])
}

// Inspect the original fields. An equivalent literal in the e-graph alone
// does not establish that the emitter skips parameter or argument evaluation.
fn fixed_cdf_bits(m: &Module, location: NodeId, scale: NodeId, point: NodeId) -> Option<[i64; 3]> {
    let bits = |id| literal(m, id).map(|value| value.to_bits() as i64);
    Some([
        bits(location)?,
        direct_literal(m, scale)?.to_bits() as i64,
        bits(point)?,
    ])
}

fn literal(m: &Module, id: NodeId) -> Option<f64> {
    if let Some(value) = direct_literal(m, id) {
        return Some(value);
    }
    let (call, "neg") = builtin(m, id)? else {
        return None;
    };
    let [arg] = *call.args else {
        return None;
    };
    if !call.named.is_empty() {
        return None;
    }
    let value = direct_literal(m, arg)?;
    // Integer negation in the f32 target cannot represent -i32::MIN.
    if matches!(m.node(arg), Node::Lit(Scalar::Int(n)) if *n == i64::from(i32::MIN)) {
        return None;
    }
    Some(-value)
}

fn direct_literal(m: &Module, id: NodeId) -> Option<f64> {
    match m.node(id) {
        Node::Lit(Scalar::Real(value)) => Some(*value),
        Node::Lit(Scalar::Int(value)) => i32::try_from(*value).ok().map(f64::from),
        Node::Const(name) if m.resolve(*name) == "inf" => Some(f64::INFINITY),
        _ => None,
    }
}
