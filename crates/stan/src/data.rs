//! Stan JSON data from literal FlatPPL values.

use crate::refuse::{Refusal, Result};
use flatppl_core::{CallHead, Module, NamedKind, Node, NodeId, RefNs, Scalar};
use serde_json::{Map, Number, Value};

/// A data variable whose value lives in a `load_data` source: the CLI loads
/// the source and [`crate::complete_data`] fills the value in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataSource {
    /// The Stan data variable.
    pub stan: String,
    /// The FlatPPL binding holding the loaded value.
    pub binding: String,
    /// Field path inside that value.
    pub path: Vec<String>,
}

fn head(m: &Module, id: NodeId) -> Option<&str> {
    match m.node(id) {
        Node::Call(c) => match c.head {
            CallHead::Builtin(h) => Some(m.resolve(h)),
            CallHead::User(_) => None,
        },
        _ => None,
    }
}

fn resolve(m: &Module, mut id: NodeId) -> NodeId {
    for _ in 0..64 {
        match m.node(id) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => match m.binding_by_name(r.name) {
                Some(b) => id = m.binding(b).rhs,
                None => return id,
            },
            _ => return id,
        }
    }
    id
}

fn real(x: f64) -> Value {
    if x.is_finite() {
        Number::from_f64(x)
            .map(Value::Number)
            .unwrap_or(Value::Null)
    } else if x.is_nan() {
        Value::String("NaN".into())
    } else if x > 0.0 {
        Value::String("Infinity".into())
    } else {
        Value::String("-Infinity".into())
    }
}

/// The JSON value of a literal scalar, array or `array(...)` constructor;
/// `None` for anything computed.
pub(crate) fn const_json(m: &Module, id: NodeId) -> Option<Value> {
    let id = resolve(m, id);
    match m.node(id) {
        Node::Lit(Scalar::Int(n)) => Some(Value::from(*n)),
        Node::Lit(Scalar::Real(x)) => Some(real(*x)),
        Node::Lit(Scalar::Bool(b)) => Some(Value::from(i64::from(*b))),
        Node::Call(c) => match head(m, id)? {
            "neg" if c.args.len() == 1 => match const_json(m, c.args[0])? {
                Value::Number(n) => n
                    .as_i64()
                    .map(|i| Value::from(-i))
                    .or_else(|| n.as_f64().map(|x| real(-x))),
                _ => None,
            },
            "vector" => c
                .args
                .iter()
                .map(|&a| const_json(m, a))
                .collect::<Option<Vec<_>>>()
                .map(Value::Array),
            // Stan JSON nests a matrix row by row.
            "rowstack" | "colstack" if c.args.len() == 1 => {
                let Value::Array(lines) = const_json(m, c.args[0])? else {
                    return None;
                };
                let lines: Vec<Vec<Value>> = lines
                    .into_iter()
                    .map(|l| match l {
                        Value::Array(v) if v.iter().all(|x| x.is_number()) => Some(v),
                        _ => None,
                    })
                    .collect::<Option<_>>()?;
                let n = lines.first()?.len();
                if lines.iter().any(|l| l.len() != n) {
                    return None;
                }
                if head(m, id)? == "rowstack" {
                    return Some(Value::Array(lines.into_iter().map(Value::Array).collect()));
                }
                Some(Value::Array(
                    (0..n)
                        .map(|j| Value::Array(lines.iter().map(|l| l[j].clone()).collect()))
                        .collect(),
                ))
            }
            "array" if c.args.len() == 3 => {
                let flat = match const_json(m, c.args[0])? {
                    Value::Array(v) => v,
                    _ => return None,
                };
                let ints = |v: Value| -> Option<Vec<usize>> {
                    match v {
                        Value::Array(v) => {
                            v.iter().map(|x| x.as_u64().map(|n| n as usize)).collect()
                        }
                        _ => None,
                    }
                };
                let size = ints(const_json(m, c.args[1])?)?;
                let order = ints(const_json(m, c.args[2])?)?;
                reshape(&flat, &size, &order)
            }
            _ => None,
        },
        _ => None,
    }
}

/// Row-major nesting of a flat vector; `order` lists axes slowest first.
fn reshape(flat: &[Value], size: &[usize], order: &[usize]) -> Option<Value> {
    if size.iter().product::<usize>() != flat.len() || order.len() != size.len() {
        return None;
    }
    // Stride of each axis in the flat data.
    let mut stride = vec![0usize; size.len()];
    let mut s = 1;
    for &axis in order.iter().rev() {
        let a = axis.checked_sub(1)?;
        *stride.get_mut(a)? = s;
        s *= size[a];
    }
    fn build(flat: &[Value], size: &[usize], stride: &[usize], axis: usize, base: usize) -> Value {
        if axis == size.len() {
            return flat[base].clone();
        }
        Value::Array(
            (0..size[axis])
                .map(|i| build(flat, size, stride, axis + 1, base + i * stride[axis]))
                .collect(),
        )
    }
    Some(build(flat, size, &stride, 0, 0))
}

/// `true` when every field of a literal record is a literal.
pub(crate) fn const_json_record(m: &Module, id: NodeId) -> bool {
    match m.node(id) {
        Node::Call(c) => c
            .named
            .iter()
            .all(|na| na.kind == NamedKind::Field && const_json(m, na.value).is_some()),
        _ => false,
    }
}

/// The JSON value of one field of a literal record or table.
pub(crate) fn field_json(m: &Module, record: NodeId, field: &str) -> Option<Value> {
    let record = resolve(m, record);
    let Node::Call(c) = m.node(record) else {
        return None;
    };
    if !matches!(head(m, record), Some("record" | "table")) {
        return None;
    }
    let na = c.named.iter().find(|na| m.resolve(na.name) == field)?;
    const_json(m, na.value)
}

/// Read every deferred data value from `loaded`, a module whose `load_data`
/// calls the host has replaced by literals.
pub(crate) fn complete(
    data: &mut Map<String, Value>,
    deferred: &[DataSource],
    loaded: &Module,
) -> Result<()> {
    for d in deferred {
        let sym = loaded
            .bindings()
            .find(|(_, b)| loaded.resolve(b.name) == d.binding)
            .map(|(_, b)| b.rhs)
            .ok_or_else(|| Refusal::not_yet(None, format!("the data binding `{}`", d.binding)))?;
        let value = match d.path.as_slice() {
            [] => const_json(loaded, sym),
            [field] => field_json(loaded, sym, field),
            _ => None,
        }
        .ok_or_else(|| {
            Refusal::not_yet(
                None,
                format!(
                    "the unloaded data `{}`",
                    std::iter::once(&d.binding)
                        .chain(&d.path)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(".")
                ),
            )
        })?;
        data.insert(d.stan.clone(), value);
    }
    Ok(())
}
