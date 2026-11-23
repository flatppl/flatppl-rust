//! The **value domain** of the inference trace (engine-concepts §17): a
//! demand-driven const-evaluator for fixed-phase expressions at shape positions.
//! The spec lets shapes depend on fixed-phase values — `iid(M, lengthof(data))`,
//! `zeros(sizeof(M))`, `cartpow(reals, N)` — so pure-structural inference is
//! incomplete (§17.1). This resolves exactly those, "resolve, don't rewrite":
//! it READS the graph and populates dims; it never mutates the IR.
//!
//! **This is the seed of the future `flatppl-interpreter` (Phase 3), not a
//! parallel walker.** The split is deliberate (see ARCHITECTURE "The value
//! domain: const-eval and the interpreter"):
//!  - the **pure value-op core** ([`FixedValue`] + `eval_*`) carries no inference
//!    state and is the piece the interpreter will lift and reuse verbatim;
//!  - the **driver** ([`const_eval`]) walks the graph via the `Inferencer`;
//!  - the **shape observers** (`lengthof`/`sizeof`) read the inferred TYPE, not
//!    the value (the §17.1 laziness short-circuit — evaluating a shape must not
//!    force a deferred value). These are inference-specific: the interpreter,
//!    holding real values, reads the value instead. So only the pure core is
//!    shared; observers stay per-domain.
//!
//! **The op-gap / `%dynamic` distinction (§17.1 "the fixed-value boundary").**
//! [`ConstEval`] is three-valued: `Val` resolved; `Dynamic` genuinely not
//! statically knowable (a non-fixed ancestor, an `external`/`load_data` runtime
//! value, or a shape observer over a `%dynamic` dim) → legitimately `%dynamic`;
//! `Gap` every input resolved but the evaluator cannot fold this op → a LOUD
//! diagnostic at the demand site, never a silent `%dynamic`. `Dynamic` dominates
//! `Gap`: a value that is dynamic anyway is not worth nagging about.

use std::collections::HashSet;

use flatppl_core::query::promoted_input_names;
use flatppl_core::{
    Call, CallHead, Dim, Node, NodeId, Phase, RefNs, Scalar, ScalarType, Symbol, Type,
};

use crate::trace::Inferencer;
use crate::{Diagnostic, Level};

// ---------------------------------------------------------------------------
// Pure value-op core — no `Inferencer`, no graph. The liftable seed: the
// interpreter reuses these verbatim (widening `FixedValue` to its richer
// batched value; adding real/bool/complex arithmetic as it needs them).
// ---------------------------------------------------------------------------

/// A statically resolved fixed-phase value. Deliberately minimal — the shape
/// domain needs integers and integer vectors (shapes are 1-D int vectors). The
/// interpreter's richer value (reals, bools, batched tensors — engine-concepts
/// §2.1) is a superset added when the core is lifted; there is no reason to
/// carry it here, where a shape can only ever be an integer.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FixedValue {
    Int(i64),
    /// A real element of fixed filter data. A real shape position stays
    /// `%dynamic`.
    Real(f64),
    Vec(Vec<FixedValue>),
}

/// Integer arithmetic (`add`/`sub`/`mul`). `None` on overflow or a non-integer
/// operand — the driver maps that to `Dynamic` (an implemented op that cannot
/// represent its result is dynamic, *not* an op-gap).
fn eval_arith(op: &str, a: &FixedValue, b: &FixedValue) -> Option<FixedValue> {
    let (FixedValue::Int(x), FixedValue::Int(y)) = (a, b) else {
        return None;
    };
    Some(FixedValue::Int(match op {
        "add" => x.checked_add(*y)?,
        "sub" => x.checked_sub(*y)?,
        "mul" => x.checked_mul(*y)?,
        "div" => floor_div(*x, *y)?,
        "mod" => floor_mod(*x, *y)?,
        _ => return None,
    }))
}

/// Spec §07 integer floor division `⌊a/b⌋` — distinct from Rust's truncating `/`
/// when the signs differ. `None` on division by zero or `i64::MIN / -1`.
fn floor_div(a: i64, b: i64) -> Option<i64> {
    let q = a.checked_div(b)?; // b == 0 or MIN/-1 → None
    let r = a % b;
    Some(if r != 0 && (r < 0) != (b < 0) {
        q - 1
    } else {
        q
    })
}

/// Spec §07 modulo `a − b·⌊a/b⌋` (sign of `b`; `None` on `b == 0` / overflow).
fn floor_mod(a: i64, b: i64) -> Option<i64> {
    let q = floor_div(a, b)?;
    a.checked_sub(b.checked_mul(q)?)
}

/// `prod`/`sum` over a vector of integers. `None` if an element is not a scalar
/// integer or the fold overflows → `Dynamic`.
fn eval_reduce(op: &str, elems: &[FixedValue]) -> Option<FixedValue> {
    let mut acc: i64 = i64::from(op == "prod");
    for e in elems {
        let FixedValue::Int(n) = e else {
            return None;
        };
        acc = match op {
            "prod" => acc.checked_mul(*n)?,
            "sum" => acc.checked_add(*n)?,
            _ => return None,
        };
    }
    Some(FixedValue::Int(acc))
}

// ---------------------------------------------------------------------------
// Driver — the demand-driven graph walk.
// ---------------------------------------------------------------------------

/// Three-valued const-eval result (see module docs): resolved, legitimately
/// `%dynamic`, or an op-gap carrying the unfoldable op's name.
pub(crate) enum ConstEval {
    Val(FixedValue),
    Dynamic,
    /// Every input resolved, but the evaluator cannot fold this op — a loud
    /// diagnostic at the demand site, never a silent `%dynamic`.
    Gap(String),
}

/// Const-evaluate `node`'s fixed value. Recursion is depth-bounded (ref/self
/// cycles); a non-fixed ancestor yields `Dynamic` (the value varies).
fn const_eval(inf: &mut Inferencer<'_, '_>, node: NodeId, depth: u32) -> ConstEval {
    // A substituted boundary replaces its original computation (§04). Seeds
    // carry types and sets, not concrete values, even when their phase is fixed.
    if inf.is_substituted(node) || depth > 64 {
        return ConstEval::Dynamic;
    }
    match inf.module.node(node).clone() {
        Node::Lit(Scalar::Int(n)) => ConstEval::Val(FixedValue::Int(n)),
        // Reals/bools/strings are not shape values → `Dynamic` (conservative;
        // the richer interpreter value represents them, this one need not).
        Node::Ref(r) if r.ns == RefNs::SelfMod => {
            let Some(binding) = inf.module.binding_by_name(r.name) else {
                return ConstEval::Dynamic;
            };
            let rhs = inf.module.binding(binding).rhs;
            // A non-fixed binding's value varies between evaluations — the
            // §17.1 "legitimately %dynamic" case, never an op-gap.
            if inf.node_annotation(rhs).1 == Phase::Fixed {
                const_eval(inf, rhs, depth + 1)
            } else {
                ConstEval::Dynamic
            }
        }
        Node::Call(c) => eval_call(inf, node, &c, depth),
        _ => ConstEval::Dynamic,
    }
}

fn eval_call(inf: &mut Inferencer<'_, '_>, node: NodeId, c: &Call, depth: u32) -> ConstEval {
    // A user-callable application is a widening the interpreter will handle
    // (inline the body); the shape resolver does not, so it is `Dynamic`, NOT a
    // gap — an unfoldable user call must not make a valid model ill-formed.
    let CallHead::Builtin(op) = c.head else {
        return ConstEval::Dynamic;
    };
    match inf.module.resolve(op).to_string().as_str() {
        // Shape observers: read the inferred TYPE, never recurse into the value
        // (§17.1 laziness short-circuit). Inference-specific — see module docs.
        "lengthof" => length_observer(inf, c, depth),
        "sizeof" => sizeof_observer(inf, c),

        // Pure value ops (the liftable core).
        op @ ("add" | "sub" | "mul" | "div" | "mod") => binary(inf, c, depth, op),
        "neg" => unary_neg(inf, c, depth),
        op @ ("prod" | "sum") => reduce(inf, c, depth, op),
        "get" => index(inf, c, depth, 1),
        "get0" => index(inf, c, depth, 0),
        "vector" => match eval_all(inf, &c.args, depth) {
            Ok(vs) => ConstEval::Val(FixedValue::Vec(vs)),
            Err(e) => e,
        },
        "cat" => cat_value(inf, c, depth),
        "filter" => filter_value(inf, c, depth),

        // Runtime-determined value sources: known `%dynamic`, never a gap
        // (`external`/`load_data` are compile-time-unknown; the
        // parametric/stochastic sources are already non-fixed). The same set is
        // what [`crate::ops::is_opaque_value_source`] names, so it is enumerated
        // there once and read here.
        name if crate::ops::is_opaque_value_source(name) => ConstEval::Dynamic,

        // Any other head: an op-gap only if it is a fixed, value-typed op whose
        // every input resolved; otherwise `%dynamic` (see `gap_or_dynamic`).
        name => gap_or_dynamic(inf, node, c, name, depth),
    }
}

/// Distinguish an op-gap from a legitimate `%dynamic` for an unimplemented head.
/// A gap requires a fixed, value-typed node whose every input is itself
/// resolvable — a genuinely-dynamic input (e.g. `external`) dominates → dynamic.
fn gap_or_dynamic(
    inf: &mut Inferencer<'_, '_>,
    node: NodeId,
    c: &Call,
    name: &str,
    depth: u32,
) -> ConstEval {
    let (ty, phase) = inf.node_annotation(node);
    // Non-fixed ⇒ value varies; non-value (measure/kernel/function/…) ⇒ no fixed
    // value to compute. Either way `%dynamic`, not a gap.
    if phase != Phase::Fixed || !is_value_type(&ty) {
        return ConstEval::Dynamic;
    }
    match eval_all(inf, &c.args, depth) {
        // All inputs known, but we cannot fold this op → the op-gap.
        Ok(_) => ConstEval::Gap(name.to_string()),
        // A dynamic input dominates; a propagated inner gap is reported as-is.
        Err(e) => e,
    }
}

/// Evaluate a slice of nodes. `Dynamic` short-circuits and dominates any `Gap`
/// (a value dynamic anyway is not worth an op-gap error); a `Gap` with no
/// `Dynamic` propagates.
fn eval_all(
    inf: &mut Inferencer<'_, '_>,
    nodes: &[NodeId],
    depth: u32,
) -> Result<Vec<FixedValue>, ConstEval> {
    let mut out = Vec::with_capacity(nodes.len());
    let mut gap: Option<ConstEval> = None;
    for &n in nodes {
        match const_eval(inf, n, depth + 1) {
            ConstEval::Val(v) => out.push(v),
            ConstEval::Dynamic => return Err(ConstEval::Dynamic),
            ConstEval::Gap(g) => {
                gap.get_or_insert(ConstEval::Gap(g));
            }
        }
    }
    match gap {
        Some(g) => Err(g),
        None => Ok(out),
    }
}

fn binary(inf: &mut Inferencer<'_, '_>, c: &Call, depth: u32, op: &str) -> ConstEval {
    let (Some(&a), Some(&b)) = (c.args.first(), c.args.get(1)) else {
        return ConstEval::Dynamic;
    };
    match (const_eval(inf, a, depth + 1), const_eval(inf, b, depth + 1)) {
        // Overflow / non-integer → `Dynamic` (implemented-but-unrepresentable),
        // never a gap.
        (ConstEval::Val(x), ConstEval::Val(y)) => {
            eval_arith(op, &x, &y).map_or(ConstEval::Dynamic, ConstEval::Val)
        }
        // `Dynamic` dominates `Gap`.
        (ConstEval::Dynamic, _) | (_, ConstEval::Dynamic) => ConstEval::Dynamic,
        (ConstEval::Gap(g), _) | (_, ConstEval::Gap(g)) => ConstEval::Gap(g),
    }
}

fn unary_neg(inf: &mut Inferencer<'_, '_>, c: &Call, depth: u32) -> ConstEval {
    let Some(&a) = c.args.first() else {
        return ConstEval::Dynamic;
    };
    match const_eval(inf, a, depth + 1) {
        ConstEval::Val(FixedValue::Int(n)) => n
            .checked_neg()
            .map_or(ConstEval::Dynamic, |v| ConstEval::Val(FixedValue::Int(v))),
        ConstEval::Val(_) => ConstEval::Dynamic,
        other => other,
    }
}

fn reduce(inf: &mut Inferencer<'_, '_>, c: &Call, depth: u32, op: &str) -> ConstEval {
    let Some(&a) = c.args.first() else {
        return ConstEval::Dynamic;
    };
    match const_eval(inf, a, depth + 1) {
        ConstEval::Val(FixedValue::Vec(elems)) => {
            eval_reduce(op, &elems).map_or(ConstEval::Dynamic, ConstEval::Val)
        }
        // `prod`/`sum` of a non-vector is a type error handled elsewhere.
        ConstEval::Val(_) => ConstEval::Dynamic,
        other => other,
    }
}

/// A single integer index into a fixed vector — `get(v, i)` / `get0(v, i)`, the
/// shape case `sizeof(A)[2]`. `base` is 1 for `get`, 0 for `get0`. Only this
/// single-scalar-into-vector form folds; record / multi-index / subset `get`
/// shapes stay `Dynamic`.
fn index(inf: &mut Inferencer<'_, '_>, c: &Call, depth: u32, base: i64) -> ConstEval {
    // Exactly one selector (a second would be multi-dim array indexing).
    let (Some(&container), Some(&sel), None) = (c.args.first(), c.args.get(1), c.args.get(2))
    else {
        return ConstEval::Dynamic;
    };
    match (
        const_eval(inf, container, depth + 1),
        const_eval(inf, sel, depth + 1),
    ) {
        // An index whose offset from `base` overflows `i64` is out of range, like
        // any other index outside the vector, and does not fold.
        (ConstEval::Val(FixedValue::Vec(elems)), ConstEval::Val(FixedValue::Int(i))) => i
            .checked_sub(base)
            .and_then(|k| usize::try_from(k).ok())
            .and_then(|k| elems.get(k).cloned())
            .map_or(ConstEval::Dynamic, ConstEval::Val),
        // `Dynamic` dominates `Gap` (match order matters).
        (ConstEval::Dynamic, _) | (_, ConstEval::Dynamic) => ConstEval::Dynamic,
        (ConstEval::Gap(g), _) | (_, ConstEval::Gap(g)) => ConstEval::Gap(g),
        // A non-vector container or non-integer selector: a type error elsewhere.
        _ => ConstEval::Dynamic,
    }
}

/// Value-level `cat` (spec §07): all scalars → a vector of them, all vectors →
/// their concatenation. A scalar/vector mix is not permitted (§07) and stays
/// `Dynamic` (a type error handled elsewhere).
fn cat_value(inf: &mut Inferencer<'_, '_>, c: &Call, depth: u32) -> ConstEval {
    let parts = match eval_all(inf, &c.args, depth) {
        Ok(p) => p,
        Err(e) => return e,
    };
    if parts.iter().all(|p| matches!(p, FixedValue::Int(_))) {
        ConstEval::Val(FixedValue::Vec(parts))
    } else if parts.iter().all(|p| matches!(p, FixedValue::Vec(_))) {
        let concatenated = parts
            .into_iter()
            .flat_map(|p| match p {
                FixedValue::Vec(v) => v,
                FixedValue::Int(_) | FixedValue::Real(_) => Vec::new(), // unreachable (guarded above)
            })
            .collect();
        ConstEval::Val(FixedValue::Vec(concatenated))
    } else {
        ConstEval::Dynamic
    }
}

/// `lengthof(x)` — the single dim of a rank-1 array / transposed vector, or a
/// table's row count. Reads the inferred TYPE (never the value), with one
/// exception: a `filter` over fixed data with a fixed mask has a fixed length
/// (spec §07 `filter`) that its type cannot carry, so it is folded.
fn length_observer(inf: &mut Inferencer<'_, '_>, c: &Call, depth: u32) -> ConstEval {
    let Some(&arg) = c.args.first() else {
        return ConstEval::Dynamic;
    };
    if let Some(filter) = fixed_filter_call(inf, arg, depth) {
        return match filter_value(inf, &filter, depth + 1) {
            ConstEval::Val(FixedValue::Vec(kept)) => i64::try_from(kept.len())
                .map_or(ConstEval::Dynamic, |n| ConstEval::Val(FixedValue::Int(n))),
            _ => ConstEval::Dynamic,
        };
    }
    match inf.node_annotation(arg).0 {
        Type::Array { shape, .. } if shape.len() == 1 => dim_to_ce(shape[0]),
        Type::TVector { len, .. } => dim_to_ce(len),
        Type::Table { nrows, .. } => dim_to_ce(nrows),
        _ => ConstEval::Dynamic,
    }
}

/// `sizeof(x)` — the fixed vector of an array's dimensions (reads the inferred
/// TYPE). Any `%dynamic` axis makes the whole size dynamic (the value can only
/// be an all-static integer vector).
fn sizeof_observer(inf: &mut Inferencer<'_, '_>, c: &Call) -> ConstEval {
    let Some(&arg) = c.args.first() else {
        return ConstEval::Dynamic;
    };
    match inf.node_annotation(arg).0 {
        Type::Array { shape, .. } => {
            let mut dims = Vec::with_capacity(shape.len());
            for d in shape.iter() {
                match d {
                    Dim::Static(n) => dims.push(FixedValue::Int(i64::from(*n))),
                    Dim::Dynamic => return ConstEval::Dynamic,
                }
            }
            ConstEval::Val(FixedValue::Vec(dims))
        }
        Type::TVector {
            len: Dim::Static(n),
            ..
        } => ConstEval::Val(FixedValue::Vec(vec![FixedValue::Int(i64::from(n))])),
        _ => ConstEval::Dynamic,
    }
}

/// A node's fixed integer value, or `None` when it does not fold to one.
pub(crate) fn fixed_int(inf: &mut Inferencer<'_, '_>, node: NodeId) -> Option<i64> {
    match const_eval(inf, node, 0) {
        ConstEval::Val(FixedValue::Int(n)) => Some(n),
        _ => None,
    }
}

fn dim_to_ce(d: Dim) -> ConstEval {
    match d {
        Dim::Static(n) => ConstEval::Val(FixedValue::Int(i64::from(n))),
        Dim::Dynamic => ConstEval::Dynamic,
    }
}

// ---------------------------------------------------------------------------
// Shape inference can count a fixed filter over literal values or a small
// index range. A runtime source or an unsupported predicate stays dynamic.
const MAX_FIXED_FILTER_INDICES: u32 = 65_536;

struct Filter<'a, 'm, 's> {
    inf: &'a Inferencer<'m, 's>,
    promoted: HashSet<Symbol>,
}

impl<'a, 'm, 's> Filter<'a, 'm, 's> {
    fn new(inf: &'a Inferencer<'m, 's>) -> Self {
        let promoted = promoted_input_names(inf.module);
        Self { inf, promoted }
    }

    fn fold_kept<T>(
        &self,
        pred: NodeId,
        data: NodeId,
        initial: T,
        mut keep: impl FnMut(T, FixedValue) -> T,
    ) -> Option<T> {
        let (param, body) = self.single_input_function(pred)?;
        self.vector(data, 0)?
            .into_iter()
            .try_fold(initial, |state, value| {
                if self.predicate(body, (param, &value), 0)? {
                    Some(keep(state, value))
                } else {
                    Some(state)
                }
            })
    }
}

pub(crate) fn fixed_filter_count(
    inf: &mut Inferencer<'_, '_>,
    pred: Option<NodeId>,
    data: Option<NodeId>,
) -> Option<usize> {
    if inf.level < Level::Shape {
        return None;
    }
    Filter::new(inf).fold_kept(pred?, data?, 0, |count, _| count + 1)
}

/// The `filter` call behind `node`, through refs to fixed bindings.
fn fixed_filter_call(inf: &mut Inferencer<'_, '_>, node: NodeId, depth: u32) -> Option<Call> {
    if depth > 64 {
        return None;
    }
    match inf.module.node(node).clone() {
        Node::Ref(r) if r.ns == RefNs::SelfMod => {
            let rhs = inf.module.binding(inf.module.binding_by_name(r.name)?).rhs;
            (inf.node_annotation(rhs).1 == Phase::Fixed).then_some(())?;
            fixed_filter_call(inf, rhs, depth + 1)
        }
        Node::Call(c) if matches!(c.head, CallHead::Builtin(op) if inf.module.resolve(op) == "filter") => {
            Some(c)
        }
        _ => None,
    }
}

fn filter_value(inf: &mut Inferencer<'_, '_>, c: &Call, depth: u32) -> ConstEval {
    if depth > 64 || !c.named.is_empty() {
        return ConstEval::Dynamic;
    }
    let [pred, data] = *c.args else {
        return ConstEval::Dynamic;
    };
    Filter::new(inf)
        .fold_kept(pred, data, Vec::new(), |mut kept, value| {
            kept.push(value);
            kept
        })
        .map_or(ConstEval::Dynamic, |kept| {
            ConstEval::Val(FixedValue::Vec(kept))
        })
}

impl Filter<'_, '_, '_> {
    // Resolve literal containers and record projections without observing the
    // authored value of a promoted input.
    fn container(&self, node: NodeId, depth: u32) -> Option<NodeId> {
        if depth > 64 || self.inf.is_substituted(node) {
            return None;
        }
        let module = &self.inf.module;
        match module.node(node) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => {
                if self.promoted.contains(&r.name) {
                    return None;
                }
                let rhs = module.binding(module.binding_by_name(r.name)?).rhs;
                (self.inf.node_annotation(rhs).1 == Phase::Fixed).then_some(())?;
                self.container(rhs, depth + 1)
            }
            Node::Call(c) if is_builtin(self.inf, c, "get") && c.named.is_empty() => {
                let [record, field] = *c.args else {
                    return Some(node);
                };
                let Node::Lit(Scalar::Str(field)) = module.node(field) else {
                    return Some(node);
                };
                let record = self.container(record, depth + 1)?;
                let Node::Call(record) = module.node(record) else {
                    return None;
                };
                if !is_builtin(self.inf, record, "record") || !record.args.is_empty() {
                    return None;
                }
                let value = record
                    .named
                    .iter()
                    .find(|a| module.resolve(a.name) == &**field)?
                    .value;
                self.container(value, depth + 1)
            }
            _ => Some(node),
        }
    }

    fn single_input_function(&self, node: NodeId) -> Option<(Symbol, NodeId)> {
        let node = self.container(node, 0)?;
        let Node::Call(c) = self.inf.module.node(node) else {
            return None;
        };
        if !is_builtin(self.inf, c, "functionof") {
            return None;
        }
        let Some(flatppl_core::Inputs::Spec(inputs)) = &c.inputs else {
            return None;
        };
        let [(_, r)] = inputs.as_ref() else {
            return None;
        };
        (r.ns == RefNs::Local).then_some((r.name, *c.args.first()?))
    }

    fn vector(&self, node: NodeId, depth: u32) -> Option<Vec<FixedValue>> {
        let node = self.container(node, depth)?;
        let Node::Call(c) = self.inf.module.node(node) else {
            return None;
        };
        if !c.named.is_empty() {
            return None;
        }
        if is_builtin(self.inf, c, "vector") {
            let real = matches!(
                self.inf.node_annotation(node).0,
                Type::Array { elem, .. } if *elem == Type::Scalar(ScalarType::Real)
            );
            return c
                .args
                .iter()
                .map(|&a| {
                    let value = self.number(a, None, depth + 1)?;
                    if real {
                        Some(FixedValue::Real(exact_real(&value)?))
                    } else {
                        Some(value)
                    }
                })
                .collect();
        }
        let [array] = *c.args else {
            return None;
        };
        let first = if is_builtin(self.inf, c, "indicesof") {
            1i64
        } else if is_builtin(self.inf, c, "indicesof0") {
            0
        } else {
            return None;
        };
        let Type::Array { shape, .. } = self.inf.node_annotation(array).0 else {
            return None;
        };
        let [Dim::Static(len)] = *shape else {
            return None;
        };
        if len > MAX_FIXED_FILTER_INDICES {
            return None;
        }
        Some(
            (first..first + i64::from(len))
                .map(FixedValue::Int)
                .collect(),
        )
    }

    fn number(
        &self,
        node: NodeId,
        param: Option<(Symbol, &FixedValue)>,
        depth: u32,
    ) -> Option<FixedValue> {
        let node = self.container(node, depth)?;
        match self.inf.module.node(node) {
            Node::Lit(Scalar::Int(n)) => exact_filter_int(*n),
            Node::Lit(Scalar::Real(x)) => Some(FixedValue::Real(*x)),
            Node::Ref(r) if r.ns == RefNs::Local => {
                let (name, value) = param?;
                (r.name == name).then(|| value.clone())
            }
            Node::Call(c) if c.named.is_empty() => {
                let CallHead::Builtin(head) = c.head else {
                    return None;
                };
                match self.inf.module.resolve(head) {
                    "neg" => {
                        let [arg] = *c.args else { return None };
                        match self.number(arg, param, depth + 1)? {
                            FixedValue::Int(x) => exact_filter_int(x.checked_neg()?),
                            FixedValue::Real(x) => Some(FixedValue::Real(-x)),
                            FixedValue::Vec(_) => None,
                        }
                    }
                    op @ ("add" | "sub" | "mul" | "div" | "mod") => {
                        let [a, b] = *c.args else { return None };
                        let a = self.number(a, param, depth + 1)?;
                        let b = self.number(b, param, depth + 1)?;
                        let FixedValue::Int(value) = eval_arith(op, &a, &b)? else {
                            return None;
                        };
                        exact_filter_int(value)
                    }
                    op @ ("get" | "get0") => {
                        let (&data, selectors) = c.args.split_first()?;
                        scalar_number(self.indexed_cell(data, selectors, op, param, depth)?)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn indexed_cell(
        &self,
        data: NodeId,
        selectors: &[NodeId],
        op: &str,
        param: Option<(Symbol, &FixedValue)>,
        depth: u32,
    ) -> Option<Scalar> {
        let base = i64::from(op == "get");
        let coordinates = selectors
            .iter()
            .map(|&selector| {
                let FixedValue::Int(index) = self.number(selector, param, depth + 1)? else {
                    return None;
                };
                usize::try_from(index.checked_sub(base)?).ok()
            })
            .collect::<Option<Vec<_>>>()?;
        self.cell(data, &coordinates, depth + 1)
    }

    // Read a single fixed cell to determine the length of a filtered sequence.
    // Scalar broadcasts and singleton axes do not require materializing a mask.
    fn cell(&self, node: NodeId, coordinates: &[usize], depth: u32) -> Option<Scalar> {
        let node = self.container(node, depth)?;
        if coordinates.is_empty() {
            if let Node::Lit(value) = self.inf.module.node(node) {
                return Some(value.clone());
            }
            return match self.number(node, None, depth + 1)? {
                FixedValue::Int(value) => Some(Scalar::Int(value)),
                FixedValue::Real(value) => Some(Scalar::Real(value)),
                FixedValue::Vec(_) => None,
            };
        }
        let Type::Array { shape, elem } = self.inf.node_annotation(node).0 else {
            return None;
        };
        if coordinates.len() != shape.len()
            || coordinates
                .iter()
                .zip(shape.iter())
                .any(|(&index, dim)| !matches!(dim, Dim::Static(size) if index < *size as usize))
        {
            return None;
        }
        let Node::Call(call) = self.inf.module.node(node) else {
            return None;
        };
        let CallHead::Builtin(head) = call.head else {
            return None;
        };
        if !call.named.is_empty() {
            return None;
        }
        let value = match (self.inf.module.resolve(head), call.args.as_ref()) {
            ("vector", values) => {
                let [index] = *coordinates else { return None };
                self.cell(*values.get(index)?, &[], depth + 1)
            }
            ("array", [values, sizes, order]) => {
                let sizes = self.literal_sizes(*sizes, depth + 1)?;
                let order = self.literal_sizes(*order, depth + 1)?;
                if sizes.len() != coordinates.len() || sizes.len() != order.len() {
                    return None;
                }
                let mut used = HashSet::new();
                let offset = order.into_iter().try_fold(0usize, |offset, axis| {
                    let axis = axis.checked_sub(1)?;
                    let size = *sizes.get(axis)?;
                    let coordinate = coordinates[axis];
                    if !used.insert(axis) || coordinate >= size {
                        return None;
                    }
                    offset.checked_mul(size)?.checked_add(coordinate)
                })?;
                self.cell(*values, &[offset], depth + 1)
            }
            (op @ ("indicesof" | "indicesof0"), [_]) => {
                let [index] = *coordinates else { return None };
                let base = i64::from(op == "indicesof");
                Some(Scalar::Int(i64::try_from(index).ok()?.checked_add(base)?))
            }
            ("addaxes", [value, leading, trailing]) => {
                let count = |node| match self.number(node, None, depth + 1)? {
                    FixedValue::Int(value) => usize::try_from(value).ok(),
                    _ => None,
                };
                let leading = count(*leading)?;
                let end = coordinates.len().checked_sub(count(*trailing)?)?;
                self.cell(*value, coordinates.get(leading..end)?, depth + 1)
            }
            ("broadcast", [op, values @ ..]) => {
                let Node::Const(op) = self.inf.module.node(*op) else {
                    return None;
                };
                let values = values
                    .iter()
                    .map(|&value| self.broadcast_cell(value, coordinates, depth + 1))
                    .collect::<Option<Vec<_>>>()?;
                match (self.inf.module.resolve(*op), values.as_slice()) {
                    ("ifelse", [Scalar::Bool(condition), yes, no]) => {
                        Some(if *condition { yes } else { no }.clone())
                    }
                    ("land", [Scalar::Bool(lhs), Scalar::Bool(rhs)]) => {
                        Some(Scalar::Bool(*lhs && *rhs))
                    }
                    ("lor", [Scalar::Bool(lhs), Scalar::Bool(rhs)]) => {
                        Some(Scalar::Bool(*lhs || *rhs))
                    }
                    ("lnot", [Scalar::Bool(value)]) => Some(Scalar::Bool(!value)),
                    (op, [lhs, rhs]) => fixed_comparison(
                        op,
                        &scalar_number(lhs.clone())?,
                        &scalar_number(rhs.clone())?,
                    )
                    .map(Scalar::Bool),
                    _ => None,
                }
            }
            _ => None,
        }?;
        // Array construction and ifelse join their scalar types before a cell
        // is read. Mirror numeric promotion, including its rounding.
        match (elem.as_ref(), value) {
            (Type::Scalar(ScalarType::Real), Scalar::Int(value)) => {
                Some(Scalar::Real(value as f64))
            }
            (Type::Scalar(ScalarType::Real), Scalar::Bool(value)) => {
                Some(Scalar::Real(f64::from(u8::from(value))))
            }
            (Type::Scalar(ScalarType::Integer), Scalar::Bool(value)) => {
                Some(Scalar::Int(i64::from(value)))
            }
            (_, value) => Some(value),
        }
    }

    fn broadcast_cell(&self, node: NodeId, coordinates: &[usize], depth: u32) -> Option<Scalar> {
        match self.inf.node_annotation(node).0 {
            Type::Scalar(_) => self.cell(node, &[], depth + 1),
            Type::Array { shape, .. } if shape.len() == coordinates.len() => {
                let indices = shape.iter().zip(coordinates).map(|(dim, &index)| {
                    if *dim == Dim::Static(1) { 0 } else { index }
                });
                self.cell(node, &indices.collect::<Vec<_>>(), depth + 1)
            }
            _ => None,
        }
    }

    fn literal_sizes(&self, node: NodeId, depth: u32) -> Option<Vec<usize>> {
        let node = self.container(node, depth)?;
        let Node::Call(c) = self.inf.module.node(node) else {
            return None;
        };
        if !is_builtin(self.inf, c, "vector") || !c.named.is_empty() {
            return None;
        }
        c.args
            .iter()
            .map(|&id| {
                let Node::Lit(Scalar::Int(n)) = self.inf.module.node(id) else {
                    return None;
                };
                usize::try_from(*n).ok()
            })
            .collect()
    }

    fn interval(&self, node: NodeId, depth: u32) -> Option<(f64, f64)> {
        let node = self.container(node, depth)?;
        let Node::Call(c) = self.inf.module.node(node) else {
            return None;
        };
        if !is_builtin(self.inf, c, "interval") || !c.named.is_empty() {
            return None;
        }
        let [lo, hi] = *c.args else {
            return None;
        };
        Some((
            exact_real(&self.number(lo, None, depth + 1)?)?,
            exact_real(&self.number(hi, None, depth + 1)?)?,
        ))
    }

    fn predicate(&self, body: NodeId, param: (Symbol, &FixedValue), depth: u32) -> Option<bool> {
        if depth > 64 || self.inf.is_substituted(body) {
            return None;
        }
        let c = match self.inf.module.node(body) {
            Node::Lit(Scalar::Bool(value)) => return Some(*value),
            Node::Call(c) => c,
            _ => return None,
        };
        let CallHead::Builtin(op) = c.head else {
            return None;
        };
        if !c.named.is_empty() {
            return None;
        }
        let op = self.inf.module.resolve(op);
        match (op, c.args.as_ref()) {
            (op @ ("get" | "get0"), [data, selectors @ ..]) => {
                let Scalar::Bool(value) =
                    self.indexed_cell(*data, selectors, op, Some(param), depth)?
                else {
                    return None;
                };
                Some(value)
            }
            ("in", [a, set]) => {
                let x = exact_real(&self.number(*a, Some(param), depth + 1)?)?;
                let (lo, hi) = self.interval(*set, depth + 1)?;
                Some(lo <= x && x <= hi)
            }
            (cmp @ ("lt" | "le" | "gt" | "ge" | "equal" | "unequal"), [a, b]) => {
                let lhs = self.number(*a, Some(param), depth + 1)?;
                let rhs = self.number(*b, Some(param), depth + 1)?;
                fixed_comparison(cmp, &lhs, &rhs)
            }
            ("land", [a, b]) => Some(
                self.predicate(*a, param, depth + 1)? && self.predicate(*b, param, depth + 1)?,
            ),
            ("lor", [a, b]) => Some(
                self.predicate(*a, param, depth + 1)? || self.predicate(*b, param, depth + 1)?,
            ),
            ("lnot", [a]) => Some(!self.predicate(*a, param, depth + 1)?),
            _ => None,
        }
    }
}

fn scalar_number(value: Scalar) -> Option<FixedValue> {
    match value {
        Scalar::Int(value) => exact_filter_int(value),
        Scalar::Real(value) => Some(FixedValue::Real(value)),
        _ => None,
    }
}

fn fixed_comparison(op: &str, lhs: &FixedValue, rhs: &FixedValue) -> Option<bool> {
    let order = match (lhs, rhs) {
        (FixedValue::Int(a), FixedValue::Int(b)) => Some(a.cmp(b)),
        _ if matches!(op, "equal" | "unequal") => return None,
        _ => exact_real(lhs)?.partial_cmp(&exact_real(rhs)?),
    };
    Some(match op {
        "lt" => order.is_some_and(|o| o.is_lt()),
        "le" => order.is_some_and(|o| o.is_le()),
        "gt" => order.is_some_and(|o| o.is_gt()),
        "ge" => order.is_some_and(|o| o.is_ge()),
        "equal" => order.is_some_and(|o| o.is_eq()),
        "unequal" => !order.is_some_and(|o| o.is_eq()),
        _ => return None,
    })
}

fn exact_filter_int(value: i64) -> Option<FixedValue> {
    i32::try_from(value).ok().map(|_| FixedValue::Int(value))
}

// Shape inference runs before a target precision is selected. A real
// comparison only proves a mask element if its operands survive conversion
// to f32 exactly. Decline subnormals as targets may flush them to zero.
fn exact_real(value: &FixedValue) -> Option<f64> {
    let real = match value {
        FixedValue::Int(x) => *x as f64,
        FixedValue::Real(x) => *x,
        FixedValue::Vec(_) => return None,
    };
    let narrow = real as f32;
    (f64::from(narrow) == real && (narrow.is_normal() || narrow == 0.0)).then_some(real)
}

fn is_builtin(inf: &Inferencer<'_, '_>, c: &Call, name: &str) -> bool {
    matches!(c.head, CallHead::Builtin(op) if inf.module.resolve(op) == name)
}

/// The predicate body's truth at `param = x`, or `None` outside the narrow
/// shapes this fold reads.
/// Value types can carry a fixed value; measures/kernels/functions/etc. cannot.
fn is_value_type(t: &Type) -> bool {
    matches!(
        t,
        Type::Scalar(_)
            | Type::Array { .. }
            | Type::TVector { .. }
            | Type::Record(_)
            | Type::Table { .. }
    )
}

// ---------------------------------------------------------------------------
// Shape-facing entry points — the demand sites (`iid`/`cartpow`/`zeros`/…).
// These emit the op-gap diagnostic; the pure walk above never does.
// ---------------------------------------------------------------------------

/// A single shape dim from a size expression. Literal integers resolve at every
/// level; anything else needs `Level::Shape` (§17.3 — dims stay `%dynamic`
/// below it). An op-gap here is a loud diagnostic.
///
/// A size of 0 becomes `%dynamic`, never `Dim::Static(0)` — see [`size_dim`].
pub(crate) fn resolve_dim(inf: &mut Inferencer<'_, '_>, node: NodeId) -> Dim {
    resolve_size_axis(inf, node).0
}

/// One size axis: the dim as typed, plus whether it RESOLVED to 0 (which types
/// as `%dynamic`, so the dim alone cannot say).
fn resolve_size_axis(inf: &mut Inferencer<'_, '_>, node: NodeId) -> (Dim, bool) {
    if let Node::Lit(Scalar::Int(n)) = inf.module.node(node) {
        let n = *n;
        return (size_dim(inf, node, n), n == 0);
    }
    if inf.level >= Level::Shape {
        match const_eval(inf, node, 0) {
            ConstEval::Val(FixedValue::Int(n)) => return (size_dim(inf, node, n), n == 0),
            ConstEval::Gap(op) => emit_gap(inf, node, &op),
            ConstEval::Val(_) | ConstEval::Dynamic => {}
        }
    }
    (Dim::Dynamic, false)
}

/// A COUNT, not a size: `addaxes`'s `n_leading`/`n_trailing` are non-negative
/// (spec §07 `addaxes`), so 0 is a legal resolved value and must stay
/// `Dim::Static(0)` — the size flooring of [`size_dim`] would turn
/// `addaxes(A, 0, 2)` into an unresolvable shape.
pub(crate) fn resolve_count(inf: &mut Inferencer<'_, '_>, node: NodeId) -> Dim {
    if let Node::Lit(Scalar::Int(n)) = inf.module.node(node) {
        return static_dim(*n);
    }
    if inf.level >= Level::Shape {
        match const_eval(inf, node, 0) {
            ConstEval::Val(FixedValue::Int(n)) => return static_dim(n),
            ConstEval::Gap(op) => emit_gap(inf, node, &op),
            ConstEval::Val(_) | ConstEval::Dynamic => {}
        }
    }
    Dim::Dynamic
}

/// The dims of a `size` argument (spec §07 `zeros`/`fill`/`array`/… : a scalar
/// size → rank-1, a vector size → one dim per element). A syntactic `vector`
/// literal contributes its arity even when element values are dynamic; a
/// non-literal size is const-evaluated whole (so `zeros(sizeof(M))` recovers
/// M's rank, not a rank-1 `%dynamic`).
pub(crate) fn count_dims(inf: &mut Inferencer<'_, '_>, node: NodeId) -> Box<[Dim]> {
    count_dims_with_zero(inf, node).0
}

/// [`count_dims`] plus whether any axis RESOLVED to 0. The flag is separate
/// because a resolved 0 is typed `%dynamic` (see [`size_dim`]), which makes it
/// indistinguishable from a size that never resolved. Spec §06 `iid` over a
/// record-valued measure is the one rule that must tell the two apart: a table
/// has no zero-row form.
pub(crate) fn count_dims_with_zero(
    inf: &mut Inferencer<'_, '_>,
    node: NodeId,
) -> (Box<[Dim]>, bool) {
    if let Node::Call(c) = inf.module.node(node).clone()
        && matches!(c.head, CallHead::Builtin(op) if inf.module.resolve(op) == "vector")
    {
        let mut zero = false;
        let dims = c
            .args
            .iter()
            .map(|&a| {
                let (d, z) = resolve_size_axis(inf, a);
                zero |= z;
                d
            })
            .collect();
        return (dims, zero);
    }
    if inf.level >= Level::Shape {
        match const_eval(inf, node, 0) {
            ConstEval::Val(FixedValue::Vec(elems)) => {
                let zero = elems.contains(&FixedValue::Int(0));
                return (
                    elems.iter().map(|e| fixed_to_dim(inf, node, e)).collect(),
                    zero,
                );
            }
            ConstEval::Val(FixedValue::Int(n)) => {
                return (Box::new([size_dim(inf, node, n)]), n == 0);
            }
            ConstEval::Gap(op) => emit_gap(inf, node, &op),
            ConstEval::Val(FixedValue::Real(_)) | ConstEval::Dynamic => {}
        }
    }
    (Box::new([Dim::Dynamic]), false)
}

fn fixed_to_dim(inf: &mut Inferencer<'_, '_>, node: NodeId, v: &FixedValue) -> Dim {
    match v {
        FixedValue::Int(n) => size_dim(inf, node, *n),
        FixedValue::Real(_) | FixedValue::Vec(_) => Dim::Dynamic,
    }
}

/// A dim from an `i64`; out-of-range (negative / > u32) falls back to `%dynamic`
/// rather than panicking (overflowed shape arithmetic).
pub(crate) fn static_dim(n: i64) -> Dim {
    u32::try_from(n).map(Dim::Static).unwrap_or(Dim::Dynamic)
}

/// A dim from a resolved SIZE, per `flatppl-dev/empty-arrays-ruling.md`
/// (2026-08-20, shape C): an extent of 0 is `%dynamic`, never `Dim::Static(0)`,
/// so §11's "a positive integer dimension size, or `%dynamic`" holds for every
/// type this crate produces. A written 0 is refused earlier, at the call
/// ([`crate::ops::written_size_check`]); a 0 reaching here is DERIVED, and
/// legality must not depend on how much of it the const-eval table can fold.
///
/// A NEGATIVE extent is no size at any phase, so it is a diagnostic rather
/// than the silent `%dynamic` it used to become.
fn size_dim(inf: &mut Inferencer<'_, '_>, node: NodeId, n: i64) -> Dim {
    if n < 0 {
        inf.diags.push(Diagnostic::error_at(
            node,
            format!(
                "this size resolves to {n}, but an array extent is a positive \
                 integer (spec §03 \"Cartesian power\", §07 \"Array constructors\")"
            ),
        ));
        return Dim::Dynamic;
    }
    if n == 0 {
        return Dim::Dynamic;
    }
    static_dim(n)
}

fn emit_gap(inf: &mut Inferencer<'_, '_>, node: NodeId, op: &str) {
    inf.diags.push(Diagnostic::error_at(
        node,
        format!(
            "could not evaluate the fixed value of `{op}`, needed for shape inference \
             (spec §17.1): the const-eval table does not fold this op"
        ),
    ));
}
