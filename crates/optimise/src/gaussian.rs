//! Read the original Gaussian constructor or kernel fields without changing the IR.

use flatppl_core::{
    Call, CallHead, Idx, Module, NamedKind, Node, NodeId, Scalar, ScalarType, Type,
};

use crate::engine::{Fact, FactArg, Term};
use crate::{Dtype, Profile};

pub(super) struct Witness {
    pub(super) nodes: Vec<NodeId>,
    kind: Kind,
}

enum Kind {
    Law(&'static str),
    NormalOperation {
        head: &'static str,
        fixed: Option<[i64; 3]>,
        affine: Option<Affine>,
        site: NodeId,
    },
    MvNormalScore,
}

enum Affine {
    Identity,
    Partial { location: i64, scale: i64 },
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
            let affine = (m.type_of(point) == Some(&Type::Scalar(ScalarType::Real)))
                .then(|| affine_transport(m, head, location, scale))
                .flatten();
            Some(Witness {
                nodes: vec![location, scale, point],
                kind: Kind::NormalOperation {
                    head,
                    fixed: fixed_input_bits(m, location, scale, point),
                    affine,
                    site: id,
                },
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
        Kind::NormalOperation {
            head,
            fixed,
            ref affine,
            site,
        } => {
            values.push(FactArg::Str(head));
            if let Profile::StableHlo { dtype, .. } = profile {
                match *affine {
                    Some(Affine::Identity) => facts.push(Fact {
                        name: "gaussian-unit-transport-source",
                        args: vec![FactArg::Expr(term), FactArg::Int(site.index() as i64)],
                    }),
                    Some(Affine::Partial { location, scale }) => facts.push(Fact {
                        name: "gaussian-affine-source",
                        args: vec![
                            FactArg::Expr(term),
                            FactArg::Int(site.index() as i64),
                            FactArg::Expr(args[2]),
                            FactArg::Int(location),
                            FactArg::Int(scale),
                        ],
                    }),
                    None => {}
                }
                if let Some(bits) = fixed {
                    let mut raw = vec![FactArg::Expr(term)];
                    raw.extend(bits.into_iter().map(FactArg::Int));
                    if head == "builtin_touniform" {
                        facts.push(Fact {
                            name: "gaussian-fixed-cdf",
                            args: raw,
                        });
                    } else if let Some(value) = fixed_native_result(head, dtype, bits) {
                        raw.push(FactArg::Int(value.to_bits() as i64));
                        facts.push(Fact {
                            name: "gaussian-native-value",
                            args: raw,
                        });
                    }
                }
            }
            "gaussian-native-normal"
        }
        Kind::MvNormalScore => "gaussian-native-mvnormal",
    };
    values.extend(args.iter().copied().map(FactArg::Expr));
    facts.push(Fact { name, args: values });
}

pub(super) fn has_candidate(fact: &Fact, terms: &[Term]) -> bool {
    matches!(
        fact.name,
        "gaussian-unit-transport-source" | "gaussian-affine-exclusive" | "gaussian-native-value"
    ) && matches!(fact.args.first(), Some(FactArg::Expr(id)) if matches!(terms[*id], Term::Leaf(_)))
}

pub(super) fn partial_transports(facts: &[Fact]) -> Vec<(usize, usize)> {
    facts
        .iter()
        .filter_map(|fact| match (fact.name, fact.args.as_slice()) {
            ("gaussian-affine-source", [FactArg::Expr(site), _, FactArg::Expr(point), ..]) => {
                Some((*site, *point))
            }
            _ => None,
        })
        .collect()
}

pub(super) fn seed_exclusive_transport(facts: &mut Vec<Fact>, site: usize) {
    facts.push(Fact {
        name: "gaussian-affine-exclusive",
        args: vec![FactArg::Expr(site)],
    });
}

fn affine_transport(m: &Module, head: &str, location: NodeId, scale: NodeId) -> Option<Affine> {
    let origin = match head {
        "builtin_tonormal" => 0.0f64,
        "builtin_fromnormal" => -0.0f64,
        _ => return None,
    };
    let location = literal(m, location)?;
    let scale = direct_literal(m, scale)?;
    if !location.is_finite()
        || !(f32::MIN_POSITIVE as f64..=f32::MAX as f64).contains(&scale)
        || location.abs() > f32::MAX as f64
    {
        return None;
    }
    let neutral_location = location.to_bits() == origin.to_bits();
    match (neutral_location, scale == 1.0) {
        (true, true) => Some(Affine::Identity),
        (false, false) => None,
        _ => Some(Affine::Partial {
            location: location.to_bits() as i64,
            scale: scale.to_bits() as i64,
        }),
    }
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
fn fixed_input_bits(
    m: &Module,
    location: NodeId,
    scale: NodeId,
    point: NodeId,
) -> Option<[i64; 3]> {
    let bits = |id| literal(m, id).map(|value| value.to_bits() as i64);
    Some([
        bits(location)?,
        direct_literal(m, scale)?.to_bits() as i64,
        bits(point)?,
    ])
}

fn fixed_native_result(head: &str, dtype: Dtype, bits: [i64; 3]) -> Option<f64> {
    let [mu, sigma, x] = bits.map(|bits| f64::from_bits(bits as u64));
    if !mu.is_finite() || !sigma.is_normal() || sigma <= 0.0 {
        return None;
    }
    if head == "builtin_logdensityof" && x.is_infinite() {
        return Some(f64::NEG_INFINITY);
    }
    let normalizer = -0.5 * (2.0 * std::f64::consts::PI).ln();
    // Evaluate each operation in the target precision. Subnormal operands
    // and intermediates stay with the emitter because its targets can flush them.
    macro_rules! evaluate {
        ($float:ty, $mu:expr, $sigma:expr, $x:expr, $normalizer:expr) => {{
            let ordinary = |x: $float| (x == 0.0 || x.is_normal()).then_some(x);
            let mu = ordinary($mu)?;
            let sigma = ordinary($sigma)?;
            let x = ordinary($x)?;
            let value = match head {
                "builtin_tonormal" => ordinary(x - mu)? / sigma,
                "builtin_fromnormal" => {
                    let output = mu + ordinary(sigma * x)?;
                    if sigma.mul_add(x, mu).to_bits() != output.to_bits() {
                        return None;
                    }
                    output
                }
                // The native score uses log(sigma). Unit scale avoids making
                // a host libm result stand in for a target's implementation.
                "builtin_logdensityof" if sigma == 1.0 => {
                    let z = ordinary(x - mu)?;
                    let square = ordinary(z * z)?;
                    $normalizer + ordinary(-0.5 * square)?
                }
                _ => return None,
            };
            ordinary(value)
        }};
    }
    match dtype {
        Dtype::F32 => {
            evaluate!(f32, mu as f32, sigma as f32, x as f32, normalizer as f32).map(f64::from)
        }
        Dtype::F64 => evaluate!(f64, mu, sigma, x, normalizer),
    }
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
    if matches!(m.node(arg), Node::Lit(Scalar::Int(0))) {
        return Some(0.0);
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
