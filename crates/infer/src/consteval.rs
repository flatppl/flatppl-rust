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

use flatppl_core::{Call, CallHead, Dim, Node, NodeId, Phase, RefNs, Scalar, Type};

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
    if depth > 64 {
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
            if inf.infer_node(rhs).1 == Phase::Fixed {
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
        "lengthof" | "length" => length_observer(inf, c),
        "sizeof" => sizeof_observer(inf, c),

        // Pure value ops (the liftable core).
        op @ ("add" | "sub" | "mul" | "div" | "mod") => binary(inf, c, depth, op),
        "neg" => unary_neg(inf, c, depth),
        op @ ("prod" | "sum") => reduce(inf, c, depth, op),
        "get" => index(inf, c, depth, 1),
        "get0" => index(inf, c, depth, 0),

        // Fixed-phase but runtime-determined value sources: known `%dynamic`,
        // never a gap (`external`/`load_data` are compile-time-unknown; the
        // parameterized/stochastic sources are already non-fixed).
        "external" | "load_data" | "elementof" | "draw" | "rand" | "rnginit" | "rngstate" => {
            ConstEval::Dynamic
        }

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
    let (ty, phase) = inf.infer_node(node);
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
        (ConstEval::Val(FixedValue::Vec(elems)), ConstEval::Val(FixedValue::Int(i))) => {
            usize::try_from(i - base)
                .ok()
                .and_then(|k| elems.get(k).cloned())
                .map_or(ConstEval::Dynamic, ConstEval::Val)
        }
        // `Dynamic` dominates `Gap` (match order matters).
        (ConstEval::Dynamic, _) | (_, ConstEval::Dynamic) => ConstEval::Dynamic,
        (ConstEval::Gap(g), _) | (_, ConstEval::Gap(g)) => ConstEval::Gap(g),
        // A non-vector container or non-integer selector: a type error elsewhere.
        _ => ConstEval::Dynamic,
    }
}

/// `lengthof(x)` — the single dim of a rank-1 array / transposed vector, or a
/// table's row count. Reads the inferred TYPE (never the value).
fn length_observer(inf: &mut Inferencer<'_, '_>, c: &Call) -> ConstEval {
    let Some(&arg) = c.args.first() else {
        return ConstEval::Dynamic;
    };
    match inf.infer_node(arg).0 {
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
    match inf.infer_node(arg).0 {
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

fn dim_to_ce(d: Dim) -> ConstEval {
    match d {
        Dim::Static(n) => ConstEval::Val(FixedValue::Int(i64::from(n))),
        Dim::Dynamic => ConstEval::Dynamic,
    }
}

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
pub(crate) fn resolve_dim(inf: &mut Inferencer<'_, '_>, node: NodeId) -> Dim {
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
    if let Node::Call(c) = inf.module.node(node).clone()
        && matches!(c.head, CallHead::Builtin(op) if inf.module.resolve(op) == "vector")
    {
        return c.args.iter().map(|&a| resolve_dim(inf, a)).collect();
    }
    if inf.level >= Level::Shape {
        match const_eval(inf, node, 0) {
            ConstEval::Val(FixedValue::Vec(elems)) => {
                return elems.iter().map(fixed_to_dim).collect();
            }
            ConstEval::Val(FixedValue::Int(n)) => return Box::new([static_dim(n)]),
            ConstEval::Gap(op) => emit_gap(inf, node, &op),
            ConstEval::Dynamic => {}
        }
    }
    Box::new([Dim::Dynamic])
}

/// Demand-driven fixed integer, for callers that want an `Option<i64>` and do
/// their own fallback (`addaxes` axis counts). No diagnostic — the shape-dim
/// entry points ([`resolve_dim`] / [`count_dims`]) own the op-gap report.
pub(crate) fn resolve_fixed_int(inf: &mut Inferencer<'_, '_>, node: NodeId) -> Option<i64> {
    match const_eval(inf, node, 0) {
        ConstEval::Val(FixedValue::Int(n)) => Some(n),
        _ => None,
    }
}

fn fixed_to_dim(v: &FixedValue) -> Dim {
    match v {
        FixedValue::Int(n) => static_dim(*n),
        FixedValue::Vec(_) => Dim::Dynamic,
    }
}

/// A dim from an `i64`; out-of-range (negative / > u32) falls back to `%dynamic`
/// rather than panicking (overflowed shape arithmetic).
pub(crate) fn static_dim(n: i64) -> Dim {
    u32::try_from(n).map(Dim::Static).unwrap_or(Dim::Dynamic)
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
