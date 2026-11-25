//! Heads of a lowered (determinized) module that the lowered stage projects.
//!
//! Lowering writes a likelihood as `builtin_logdensityof(F, record(k = v, …), y)`,
//! one cell of it as the same call under `broadcast`, and a law parameter as a
//! `broadcast` of a link such as `invlogit`. The source stage keeps these calls
//! opaque. On the lowered stage each projects to a named head, so rules can
//! match the family, the fields and the link, and each head rebuilds the call
//! it came from.

use flatppl_core::graph::build_call;
use flatppl_core::{Call, CallHead, Idx, Module, NamedArg, NamedKind, Node, NodeId};

use crate::RulePolicy;

/// Element-wise operations whose broadcast the lowered stage names.
const LINKS: &[&str] = &["invlogit", "exp", "softmax"];

/// Rulesets of the lowered stage and the density families each one reads.
/// A run starts only when a loaded ruleset has a term to match.
const RULESETS: &[(&str, &[&str])] = &[];

fn builtin<'a>(m: &'a Module, c: &Call) -> Option<&'a str> {
    match c.head {
        CallHead::Builtin(s) => Some(m.resolve(s)),
        _ => None,
    }
}

fn constant(m: &Module, id: NodeId) -> Option<&str> {
    match m.node(id) {
        Node::Const(s) => Some(m.resolve(*s)),
        _ => None,
    }
}

/// `record.<kind><name>…` for 1 to 4 named entries and no positional ones.
fn record_head(m: &Module, c: &Call, prefix: &str) -> Option<(String, Vec<NodeId>)> {
    if !(1..=4).contains(&c.named.len()) {
        return None;
    }
    let mut head = String::from(prefix);
    for field in c.named.iter() {
        let kind = match field.kind {
            NamedKind::Field => 'f',
            NamedKind::Kwarg => 'k',
            NamedKind::Assign => return None,
        };
        head.push('.');
        head.push(kind);
        head.push_str(m.resolve(field.name));
    }
    Some((head, c.named.iter().map(|field| field.value).collect()))
}

/// The lowered head of a call and its operands, in rebuild order.
pub(super) fn head(m: &Module, c: &Call) -> Option<(String, Vec<NodeId>)> {
    if c.inputs.is_some() {
        return None;
    }
    match (builtin(m, c)?, c.args.as_ref()) {
        ("record", []) => record_head(m, c, "record"),
        ("builtin_logdensityof", &[family, params, observed]) if c.named.is_empty() => Some((
            format!("density.{}", constant(m, family)?),
            vec![params, observed],
        )),
        ("broadcast", [op, rest @ ..]) => match (constant(m, *op)?, rest) {
            ("record", []) => record_head(m, c, "broadcast.record"),
            ("builtin_logdensityof", &[family, params, observed]) if c.named.is_empty() => Some((
                format!("broadcast.density.{}", constant(m, family)?),
                vec![params, observed],
            )),
            (link, args) if LINKS.contains(&link) && c.named.is_empty() && args.len() == 1 => {
                Some((format!("broadcast.{link}"), args.to_vec()))
            }
            _ => None,
        },
        _ => None,
    }
}

fn record_call(m: &mut Module, names: &str, values: &[NodeId], op: Option<NodeId>) -> NodeId {
    let named: Vec<NamedArg> = names
        .split('.')
        .zip(values)
        .map(|(entry, &value)| {
            let (kind, name) = entry.split_at(1);
            NamedArg {
                kind: if kind == "f" {
                    NamedKind::Field
                } else {
                    NamedKind::Kwarg
                },
                name: m.intern(name),
                value,
            }
        })
        .collect();
    let (head, args) = match op {
        Some(op) => ("broadcast", vec![op]),
        None => ("record", vec![]),
    };
    let head = m.intern(head);
    m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: args.into(),
        named: named.into(),
        inputs: None,
    }))
}

fn constant_node(m: &mut Module, name: &str) -> NodeId {
    let name = m.intern(name);
    m.alloc(Node::Const(name))
}

/// The call a lowered head came from, or `None` for any other head.
pub(super) fn rebuild(m: &mut Module, head: &str, args: &[NodeId]) -> Option<NodeId> {
    if let Some(names) = head.strip_prefix("broadcast.record.") {
        let op = constant_node(m, "record");
        return Some(record_call(m, names, args, Some(op)));
    }
    if let Some(names) = head.strip_prefix("record.") {
        return Some(record_call(m, names, args, None));
    }
    if let Some(family) = head.strip_prefix("broadcast.density.") {
        let op = constant_node(m, "builtin_logdensityof");
        let family = constant_node(m, family);
        return Some(build_call(m, "broadcast", &[op, family, args[0], args[1]]));
    }
    if let Some(family) = head.strip_prefix("density.") {
        let family = constant_node(m, family);
        return Some(build_call(
            m,
            "builtin_logdensityof",
            &[family, args[0], args[1]],
        ));
    }
    None
}

/// Whether a loaded lowered ruleset can match a density in the module.
pub(super) fn trigger(m: &Module, rules: &RulePolicy) -> bool {
    let families: Vec<&str> = RULESETS
        .iter()
        .filter(|(ruleset, _)| rules.enables_ruleset(ruleset))
        .flat_map(|(_, families)| families.iter().copied())
        .collect();
    !families.is_empty()
        && (0..m.node_count()).any(|i| {
            let Node::Call(c) = m.node(NodeId::from_usize(i)) else {
                return false;
            };
            match (builtin(m, c), c.args.as_ref()) {
                (Some("builtin_logdensityof"), [family, ..]) => {
                    constant(m, *family).is_some_and(|f| families.contains(&f))
                }
                (Some("broadcast"), [op, family, ..]) => {
                    constant(m, *op) == Some("builtin_logdensityof")
                        && constant(m, *family).is_some_and(|f| families.contains(&f))
                }
                _ => false,
            }
        })
}
