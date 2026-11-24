//! Values of fixed data expressions, for finiteness proofs.
//!
//! The host bakes loaded data into literal nodes before it prepares fixed
//! summaries. A data-only expression over those literals (a column, its
//! logarithm, a product of columns, a centred or z-scored column) then has
//! values in hand. A reduction that needs every entry finite can prove it from
//! those values where the declared value sets (`reals`, `posreals`) cannot.
//!
//! Evaluation follows single precision, the narrower of the two the emitted
//! programs use: each loaded value and each intermediate is rounded to `f32`
//! and must stay within half of `f32::MAX`. A logarithm or a divisor needs an
//! argument of at least twice `f32::MIN_POSITIVE`, a square root a
//! nonnegative one, and an exponential an argument whose result stays in
//! range. With that headroom a last-place difference in the device's
//! arithmetic cannot turn a proven value into an infinity or a NaN.
//!
//! A query input is never read: its value is the caller's, not the literal the
//! source writes.

use flatppl_core::{CallHead, Module, Node, NodeId, RefNs, Scalar, Symbol};
use std::collections::HashSet;

/// A fixed value in C order. A scalar has an empty shape.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedValues {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl FixedValues {
    fn scalar(x: f64) -> Self {
        Self {
            shape: vec![],
            data: vec![x],
        }
    }

    fn vector(data: Vec<f64>) -> Self {
        Self {
            shape: vec![data.len()],
            data,
        }
    }
}

/// Half of the largest `f32`: a later doubling stays finite in either precision.
const LIMIT: f64 = (f32::MAX / 2.0) as f64;
/// The smallest magnitude of a divisor or of a logarithm's argument.
const TINY: f64 = 2.0 * f32::MIN_POSITIVE as f64;
/// Bounds the recursion through bindings and nested expressions.
const DEPTH: usize = 256;

/// The values of the data-only expression `id` over literal leaves, every one
/// proven finite as the module docs state. `None` when a leaf is a query input
/// in `boundary` or not a literal, when an operation is outside the evaluator,
/// or when a value is not proven.
pub fn fixed_values(m: &Module, id: NodeId, boundary: &HashSet<Symbol>) -> Option<FixedValues> {
    eval(m, id, boundary, 0)
}

/// The smallest and largest value of [`fixed_values`], when it proves them.
pub fn proven_finite_bounds(
    m: &Module,
    id: NodeId,
    boundary: &HashSet<Symbol>,
) -> Option<(f64, f64)> {
    let values = fixed_values(m, id, boundary)?;
    values
        .data
        .iter()
        .try_fold(None, |bounds: Option<(f64, f64)>, &x| {
            Some(Some(match bounds {
                None => (x, x),
                Some((lo, hi)) => (lo.min(x), hi.max(x)),
            }))
        })?
}

/// `x` rounded to single precision, when it lies within [`LIMIT`].
fn checked(x: f64) -> Option<f64> {
    let rounded = f64::from(x as f32);
    (rounded.is_finite() && rounded.abs() <= LIMIT).then_some(rounded)
}

/// Follow bindings that are not query inputs, and fields of literal records.
fn follow(m: &Module, mut id: NodeId, boundary: &HashSet<Symbol>) -> Option<NodeId> {
    for _ in 0..DEPTH {
        match m.node(id) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                if boundary.contains(&r.name) {
                    return None;
                }
                id = m.binding(m.binding_by_name(r.name)?).rhs;
            }
            Node::Call(c) if builtin(m, c.head) == Some("get") => {
                let [record, key] = c.args[..] else {
                    return Some(id);
                };
                let Node::Lit(Scalar::Str(field)) = m.node(key) else {
                    return Some(id);
                };
                let record = follow(m, record, boundary)?;
                let Node::Call(r) = m.node(record) else {
                    return None;
                };
                if !matches!(builtin(m, r.head), Some("record" | "table")) {
                    return None;
                }
                id = r
                    .named
                    .iter()
                    .find(|arg| m.resolve(arg.name) == field.as_ref())?
                    .value;
            }
            _ => return Some(id),
        }
    }
    None
}

fn builtin(m: &Module, head: CallHead) -> Option<&str> {
    match head {
        CallHead::Builtin(name) => Some(m.resolve(name)),
        _ => None,
    }
}

fn eval(m: &Module, id: NodeId, boundary: &HashSet<Symbol>, depth: usize) -> Option<FixedValues> {
    if depth > DEPTH {
        return None;
    }
    let id = follow(m, id, boundary)?;
    let c = match m.node(id) {
        Node::Lit(Scalar::Real(x)) => return checked(*x).map(FixedValues::scalar),
        Node::Lit(Scalar::Int(x)) => return checked(*x as f64).map(FixedValues::scalar),
        Node::Call(c) => c,
        _ => return None,
    };
    let mut name = builtin(m, c.head)?;
    let mut args = &c.args[..];
    // `f.(x, …)` applies `f` cell by cell. Without the dot, §07 gives arrays
    // only to `neg` and to `add`/`sub` of one shape, gives `mul` of two arrays
    // the matrix product, and keeps the other arms scalar.
    let mut cellwise_call = false;
    if name == "broadcast" {
        let (head, rest) = args.split_first()?;
        let Node::Const(head) = m.node(*head) else {
            return None;
        };
        name = m.resolve(*head);
        args = rest;
        cellwise_call = true;
    }
    if !c.named.is_empty() {
        return None;
    }
    let sub = |x: NodeId| eval(m, x, boundary, depth + 1);
    match (name, args) {
        ("vector", items) => items
            .iter()
            .map(|&x| sub(x).filter(|v| v.shape.is_empty()).map(|v| v.data[0]))
            .collect::<Option<Vec<_>>>()
            .map(FixedValues::vector),
        ("array", [flat, sizes, order]) => {
            let flat = sub(*flat)?;
            let sizes = integers(m, follow(m, *sizes, boundary)?)?;
            let order = integers(m, follow(m, *order, boundary)?)?;
            if flat.shape.len() != 1
                || !order.iter().copied().eq(1..=sizes.len())
                || sizes.iter().product::<usize>() != flat.data.len()
            {
                return None;
            }
            Some(FixedValues {
                shape: sizes,
                data: flat.data,
            })
        }
        ("rowstack" | "colstack" | "cat", parts) => {
            let parts = stacked(m, parts, boundary, depth)?;
            let n = parts.first()?.data.len();
            if parts.iter().any(|p| p.shape.len() != 1) {
                return None;
            }
            if name == "cat" {
                return Some(FixedValues::vector(
                    parts.into_iter().flat_map(|p| p.data).collect(),
                ));
            }
            if parts.iter().any(|p| p.data.len() != n) {
                return None;
            }
            let k = parts.len();
            let data = if name == "rowstack" {
                parts.into_iter().flat_map(|p| p.data).collect()
            } else {
                (0..n)
                    .flat_map(|i| parts.iter().map(move |p| p.data[i]))
                    .collect()
            };
            let shape = if name == "rowstack" {
                vec![k, n]
            } else {
                vec![n, k]
            };
            Some(FixedValues { shape, data })
        }
        ("fill", [value, count]) => {
            let value = sub(*value).filter(|v| v.shape.is_empty())?.data[0];
            Some(FixedValues::vector(vec![
                value;
                count_of(m, *count, boundary)?
            ]))
        }
        ("ones" | "zeros", [count]) => {
            Some(FixedValues::vector(vec![
                if name == "ones" {
                    1.0
                } else {
                    0.0
                };
                count_of(m, *count, boundary)?
            ]))
        }
        ("real", [x]) => sub(*x).filter(|x| cellwise_call || x.shape.is_empty()),
        ("neg" | "abs" | "log" | "exp" | "sqrt", [x]) => {
            let x = sub(*x)?;
            if !cellwise_call && name != "neg" && !x.shape.is_empty() {
                return None;
            }
            let f = |a: f64| match name {
                "neg" => checked(-a),
                "abs" => checked(a.abs()),
                "log" => (a >= TINY).then(|| checked(a.ln())).flatten(),
                "exp" => (a <= LIMIT.ln()).then(|| checked(a.exp())).flatten(),
                _ => (a >= 0.0).then(|| checked(a.sqrt())).flatten(),
            };
            let data = x.data.iter().map(|&a| f(a)).collect::<Option<Vec<_>>>()?;
            Some(FixedValues {
                shape: x.shape,
                data,
            })
        }
        ("add" | "sub" | "mul" | "divide", [a, b]) => {
            let (a, b) = (sub(*a)?, sub(*b)?);
            let scalar = |v: &FixedValues| v.shape.is_empty();
            let operands_in_domain = cellwise_call
                || match name {
                    "add" | "sub" => a.shape == b.shape,
                    "mul" => scalar(&a) || scalar(&b),
                    _ => scalar(&b),
                };
            if !operands_in_domain {
                return None;
            }
            let f = |x: f64, y: f64| match name {
                "add" => checked(x + y),
                "sub" => checked(x - y),
                "mul" => checked(x * y),
                _ => (y.abs() >= TINY).then(|| checked(x / y)).flatten(),
            };
            cellwise(a, b, f)
        }
        ("pow", [base, exponent]) => {
            let base = sub(*base).filter(|b| cellwise_call || b.shape.is_empty())?;
            // A small integer exponent is repeated multiplication. Others are
            // outside the evaluator.
            let Node::Lit(Scalar::Int(k)) = m.node(follow(m, *exponent, boundary)?) else {
                return None;
            };
            if !(0..=8).contains(k) {
                return None;
            }
            let data = base
                .data
                .iter()
                .map(|&a| (0..*k).try_fold(1.0, |acc, _| checked(acc * a)))
                .collect::<Option<Vec<_>>>()?;
            Some(FixedValues {
                shape: base.shape,
                data,
            })
        }
        ("sum" | "mean" | "var" | "std" | "maximum" | "minimum", [x]) => {
            let x = sub(*x)?;
            reduce(name, &x.data).map(FixedValues::scalar)
        }
        _ => None,
    }
}

/// The operands of a stacking call: its arguments, or the items of a single
/// `vector` argument that holds arrays.
fn stacked(
    m: &Module,
    parts: &[NodeId],
    boundary: &HashSet<Symbol>,
    depth: usize,
) -> Option<Vec<FixedValues>> {
    if let [only] = parts
        && let Node::Call(c) = m.node(follow(m, *only, boundary)?)
        && builtin(m, c.head) == Some("vector")
    {
        let items = c
            .args
            .iter()
            .map(|&x| eval(m, x, boundary, depth + 1))
            .collect::<Option<Vec<_>>>()?;
        if items.iter().all(|v| !v.shape.is_empty()) {
            return Some(items);
        }
    }
    parts
        .iter()
        .map(|&x| eval(m, x, boundary, depth + 1))
        .collect()
}

/// Apply `f` cell by cell, a scalar operand standing for every cell.
fn cellwise(
    a: FixedValues,
    b: FixedValues,
    f: impl Fn(f64, f64) -> Option<f64>,
) -> Option<FixedValues> {
    let shape = match (a.shape.is_empty(), b.shape.is_empty()) {
        (true, _) => b.shape.clone(),
        (_, true) => a.shape.clone(),
        _ if a.shape == b.shape => a.shape.clone(),
        _ => return None,
    };
    let n = a.data.len().max(b.data.len());
    let at = |v: &FixedValues, i: usize| {
        if v.data.len() == 1 {
            v.data[0]
        } else {
            v.data[i]
        }
    };
    let data = (0..n)
        .map(|i| f(at(&a, i), at(&b, i)))
        .collect::<Option<Vec<_>>>()?;
    Some(FixedValues { shape, data })
}

/// §07 reductions over every cell. A sum is bounded by its magnitudes, so any
/// summation order stays finite.
fn reduce(name: &str, xs: &[f64]) -> Option<f64> {
    let n = xs.len();
    if n == 0 {
        return None;
    }
    let total = || {
        xs.iter()
            .try_fold(0.0, |acc: f64, &x| checked(acc + x.abs()))?;
        Some(xs.iter().sum::<f64>())
    };
    match name {
        "sum" => checked(total()?),
        "mean" => checked(total()? / n as f64),
        "maximum" => checked(xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        "minimum" => checked(xs.iter().copied().fold(f64::INFINITY, f64::min)),
        "var" | "std" => {
            if n < 2 {
                return None;
            }
            let mean = checked(total()? / n as f64)?;
            // Bound every square before the sum, as for `total`.
            xs.iter().try_fold(0.0, |acc: f64, &x| {
                let spread = checked(x.abs() + mean.abs())?;
                checked(acc + checked(spread * spread)?)
            })?;
            let var =
                checked(xs.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64)?;
            if name == "var" {
                Some(var)
            } else {
                checked(var.sqrt())
            }
        }
        _ => None,
    }
}

/// The extents written as a literal integer vector.
fn integers(m: &Module, id: NodeId) -> Option<Vec<usize>> {
    let Node::Call(c) = m.node(id) else {
        return None;
    };
    if builtin(m, c.head) != Some("vector") {
        return None;
    }
    c.args
        .iter()
        .map(|&x| match m.node(x) {
            Node::Lit(Scalar::Int(k)) => usize::try_from(*k).ok(),
            _ => None,
        })
        .collect()
}

/// A count written as a literal integer, through bindings that are not inputs.
fn count_of(m: &Module, id: NodeId, boundary: &HashSet<Symbol>) -> Option<usize> {
    match m.node(follow(m, id, boundary)?) {
        Node::Lit(Scalar::Int(k)) => usize::try_from(*k).ok(),
        _ => None,
    }
}
