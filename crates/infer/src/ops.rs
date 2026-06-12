//! The per-op type/phase rule catalogue (engine-concepts §18: one source of
//! truth per op — this table is what later passes share).
//!
//! Coverage is incremental and honest: ops without a rule yield `%deferred`
//! plus a once-per-op note (see crate docs). Rules mirror the spec tables —
//! §07 functions (domains/results), §08 distributions (variate domains),
//! §06 measure combinators, §04 reified callables.

use flatppl_core::{Call, Dim, Inputs, Node, NodeId, Phase, Scalar, ScalarType, Symbol, Type};

use crate::Level;
use crate::trace::Inferencer;

/// `(node, type, phase)` of an inferred positional argument.
type ArgInfo = (NodeId, Type, Phase);
/// `(name, node, type, phase)` of an inferred named argument.
type NamedInfo = (Symbol, NodeId, Type, Phase);

pub(crate) fn literal_type(s: &Scalar) -> Type {
    match s {
        Scalar::Int(_) => Type::Scalar(ScalarType::Integer),
        Scalar::Real(_) => Type::Scalar(ScalarType::Real),
        Scalar::Bool(_) => Type::Scalar(ScalarType::Boolean),
        // Strings have no FlatPIR value type (paths / field names — metadata);
        // `Any` keeps them neutral in joins, and literals never emit `%meta`.
        Scalar::Str(_) => Type::Any,
    }
}

/// Built-in constants in value position. Sets and ops-as-values have no
/// first-class `Type` (they are resolved structurally where consumed:
/// `elementof` reads its set argument's *node*, `broadcast` its head).
pub(crate) fn const_type(name: &str) -> Type {
    match name {
        "pi" | "inf" => Type::Scalar(ScalarType::Real),
        "im" => Type::Scalar(ScalarType::Complex),
        _ => Type::Any,
    }
}

/// Dispatch a call to its op rule. `joined` is the §04 ancestor-rule phase
/// join over all inputs; rules override it only where the op itself
/// introduces a phase (`elementof`, `draw`, reification closure, loaders).
pub(crate) fn call_rule(
    inf: &mut Inferencer<'_>,
    id: NodeId,
    call: &Call,
    callee: Option<(NodeId, Type)>,
    args: &[ArgInfo],
    named: &[NamedInfo],
    joined: Phase,
) -> (Type, Phase) {
    // User-defined callable application: the result looks through the callee
    // to the reified body (spec §11 reified callables).
    if let Some((callee_node, callee_ty)) = callee {
        let ty = user_call_type(inf, callee_node, &callee_ty);
        return (ty, joined);
    }

    let flatppl_core::CallHead::Builtin(op) = call.head else {
        unreachable!("user calls handled above");
    };
    let name = inf.module.resolve(op).to_string();

    // Reified callables (`functionof` / `kernelof`) — typed by their boundary
    // + body, and always *fixed* (a reification closes over its ancestry).
    if call.inputs.is_some() {
        return (reification_type(inf, id, call, &name, args), Phase::Fixed);
    }

    let ty = match name.as_str() {
        // ---- arithmetic (spec §07) ----
        "add" | "sub" => elementwise2(&args.first(), &args.get(1)),
        "mul" => mul_type(args),
        "divide" => Type::Scalar(ScalarType::Real),
        "pow" => promote2(arg_ty(args, 0), arg_ty(args, 1)),
        "neg" => args.first().map_or(Type::Deferred, |(_, t, _)| t.clone()),
        "min" | "max" | "atan2" => promote2(arg_ty(args, 0), arg_ty(args, 1)),
        "div" | "mod" => Type::Scalar(ScalarType::Integer),
        "abs" | "abs2" => real_or_complex(arg_ty(args, 0)),
        "exp" | "log" | "log2" | "log10" | "sqrt" | "sin" | "cos" | "tan" | "asin" | "acos"
        | "atan" | "sinh" | "cosh" | "tanh" | "asinh" | "acosh" | "atanh" | "log1p" | "expm1"
        | "gamma" | "loggamma" | "logit" | "invlogit" | "probit" | "invprobit" => {
            real_or_complex(arg_ty(args, 0))
        }
        "floor" | "ceil" | "round" | "integer" => Type::Scalar(ScalarType::Integer),
        "cis" | "complex" => Type::Scalar(ScalarType::Complex),
        "conj" => real_or_complex(arg_ty(args, 0)),

        // ---- comparisons / logic (spec §07) ----
        "equal" | "unequal" | "lt" | "le" | "gt" | "ge" | "in" | "land" | "lor" | "lnot"
        | "isfinite" | "isinf" | "isnan" | "iszero" => Type::Scalar(ScalarType::Boolean),

        // ---- containers (spec §03) ----
        "vector" => vector_type(args),
        "tuple" => Type::Tuple(args.iter().map(|(_, t, _)| t.clone()).collect()),
        "record" => Type::Record(named.iter().map(|(n, _, t, _)| (*n, t.clone())).collect()),
        "rowstack" => rowstack_type(arg_ty(args, 0)),
        "get" => get_type(inf, args, /*base=*/ 1),
        "get0" => get_type(inf, args, /*base=*/ 0),
        "lengthof" | "length" => Type::Scalar(ScalarType::Integer),
        "indicesof" | "indicesof0" => Type::Array {
            shape: Box::new([Dim::Dynamic]),
            elem: Box::new(Type::Scalar(ScalarType::Integer)),
        },
        "sum" | "prod" => reduce_type(arg_ty(args, 0)),
        "mean" => Type::Scalar(ScalarType::Real),

        // ---- value-preserving assertion (spec §07) ----
        "checked" => args.first().map_or(Type::Deferred, |(_, t, _)| t.clone()),

        // ---- parameters / inputs (spec §04) ----
        "elementof" | "external" => set_element_type(inf, args.first().map(|a| a.0)),

        // ---- measure algebra (spec §06) ----
        "lawof" => Type::Measure {
            domain: Box::new(args.first().map_or(Type::Any, |(_, t, _)| t.clone())),
        },
        "draw" => measure_domain(arg_ty(args, 0)),
        "iid" => iid_type(inf, args),
        "truncate" | "weight" | "normalize" => {
            args.first().map_or(Type::Deferred, |(_, t, _)| t.clone())
        }
        // `bayesupdate(L, prior)` (spec §06): the unnormalized posterior is a
        // measure over the prior's domain — pick the measure-typed argument.
        "bayesupdate" => args
            .iter()
            .map(|(_, t, _)| t)
            .find(|t| matches!(t, Type::Measure { .. }))
            .cloned()
            .unwrap_or(Type::Deferred),
        "joint" => joint_type(named),
        "likelihoodof" => likelihood_type(inf, args),
        "logdensityof" | "densityof" => Type::Scalar(ScalarType::Real),

        // ---- explicit RNG (spec §07) ----
        "rnginit" => Type::RngState,
        "rand" => match measure_domain(arg_ty(args, 1)) {
            Type::Deferred => Type::Deferred,
            domain => Type::Tuple(Box::new([domain, Type::RngState])),
        },

        // ---- multi-file (deferred — see TODO) ----
        "load_module" | "standard_module" => Type::Module,

        // ---- set constructors (spec §03) — set objects have no first-class
        // type; consumers (`elementof`, `truncate`, …) read them structurally.
        "interval" | "cartprod" | "cartpow" => Type::Any,

        // ---- broadcasting (spec §04) ----
        "broadcast" => broadcast_type(inf, args),

        // ---- distributions (spec §08) ----
        _ => match distribution_domain(inf, &name, args, named) {
            Some(domain) => Type::Measure {
                domain: Box::new(domain),
            },
            None => {
                inf.note_gap(op);
                Type::Deferred
            }
        },
    };

    let phase = match name.as_str() {
        "elementof" => Phase::Parameterized,
        "external" | "load_data" | "load_module" | "standard_module" => Phase::Fixed,
        "draw" => Phase::Stochastic,
        _ => joined,
    };
    (ty, phase)
}

fn arg_ty(args: &[ArgInfo], i: usize) -> Option<&Type> {
    args.get(i).map(|(_, t, _)| t)
}

/// Numeric promotion: integer ⊔ integer = integer, real dominates integer,
/// complex dominates real; `Any` (placeholders) is absorbed.
fn promote2(a: Option<&Type>, b: Option<&Type>) -> Type {
    use ScalarType::*;
    let rank = |t: Option<&Type>| match t {
        Some(Type::Scalar(Integer)) | Some(Type::Scalar(Boolean)) => Some(0),
        Some(Type::Scalar(Real)) => Some(1),
        Some(Type::Scalar(Complex)) => Some(2),
        Some(Type::Any) => Some(-1),
        _ => None,
    };
    match (rank(a), rank(b)) {
        (Some(x), Some(y)) => match x.max(y) {
            -1 => Type::Any, // both unconstrained placeholders
            0 => Type::Scalar(Integer),
            1 => Type::Scalar(Real),
            _ => Type::Scalar(Complex),
        },
        _ => Type::Deferred,
    }
}

/// `reals, complexes` unary domain: complex in, complex out; else real.
fn real_or_complex(a: Option<&Type>) -> Type {
    match a {
        Some(Type::Scalar(ScalarType::Complex)) => Type::Scalar(ScalarType::Complex),
        _ => Type::Scalar(ScalarType::Real),
    }
}

/// `add`/`sub`: scalars promote; same-shape arrays go elementwise.
fn elementwise2(a: &Option<&ArgInfo>, b: &Option<&ArgInfo>) -> Type {
    let (a, b) = match (a, b) {
        (Some(a), Some(b)) => (&a.1, &b.1),
        _ => return Type::Deferred,
    };
    match (a, b) {
        (
            Type::Array {
                shape: sa,
                elem: ea,
            },
            Type::Array {
                shape: sb,
                elem: eb,
            },
        ) if sa == sb => Type::Array {
            shape: sa.clone(),
            elem: Box::new(promote2(Some(ea), Some(eb))),
        },
        _ => promote2(Some(a), Some(b)),
    }
}

/// `mul`: scalar·scalar and scalar·array for now; the matrix/vector forms
/// arrive with the shape work.
fn mul_type(args: &[ArgInfo]) -> Type {
    let (a, b) = match (arg_ty(args, 0), arg_ty(args, 1)) {
        (Some(a), Some(b)) => (a, b),
        _ => return Type::Deferred,
    };
    let scalarish = |t: &Type| matches!(t, Type::Scalar(_) | Type::Any);
    match (a, b) {
        _ if scalarish(a) && scalarish(b) => promote2(Some(a), Some(b)),
        (Type::Array { .. }, s) if scalarish(s) => a.clone(),
        (s, Type::Array { .. }) if scalarish(s) => b.clone(),
        _ => Type::Deferred,
    }
}

/// `vector(e1, …, en)` — a static-length array of the unified element type.
fn vector_type(args: &[ArgInfo]) -> Type {
    let mut elem: Option<Type> = None;
    for (_, t, _) in args {
        elem = Some(match elem {
            None => t.clone(),
            Some(prev) if &prev == t => prev,
            Some(prev) => match promote2(Some(&prev), Some(t)) {
                Type::Deferred => Type::Any, // heterogeneous non-numeric
                p => p,
            },
        });
    }
    Type::Array {
        shape: Box::new([Dim::Static(args.len() as u32)]),
        elem: Box::new(elem.unwrap_or(Type::Any)),
    }
}

/// `rowstack([rows…])`: an array of equal-length vectors becomes a matrix.
fn rowstack_type(a: Option<&Type>) -> Type {
    match a {
        Some(Type::Array { shape, elem }) if shape.len() == 1 => match elem.as_ref() {
            Type::Array {
                shape: inner,
                elem: cell,
            } if inner.len() == 1 => Type::Array {
                shape: Box::new([shape[0], inner[0]]),
                elem: cell.clone(),
            },
            _ => Type::Deferred,
        },
        _ => Type::Deferred,
    }
}

/// `sum`/`prod` over an array reduce to the element type.
fn reduce_type(a: Option<&Type>) -> Type {
    match a {
        Some(Type::Array { elem, .. }) => elem.as_ref().clone(),
        Some(Type::Any) => Type::Any,
        _ => Type::Deferred,
    }
}

/// `get` with static selectors: integer indices consume array axes / pick
/// tuple components; string keys pick record fields. Anything dynamic or
/// sliced (`all` / `only` / axes) is deferred until the shape work.
fn get_type(inf: &mut Inferencer<'_>, args: &[ArgInfo], base: i64) -> Type {
    let Some((_, container, _)) = args.first() else {
        return Type::Deferred;
    };
    let mut current = container.clone();
    for (node, _, _) in &args[1..] {
        let selector = inf.module.node(*node).clone();
        current = match (&current, &selector) {
            (Type::Tuple(comps), Node::Lit(Scalar::Int(k))) => {
                match usize::try_from(k - base).ok().and_then(|i| comps.get(i)) {
                    Some(t) => t.clone(),
                    None => return Type::Failed("tuple index out of range".into()),
                }
            }
            (Type::Array { shape, elem }, Node::Lit(Scalar::Int(_))) => {
                if shape.len() == 1 {
                    elem.as_ref().clone()
                } else {
                    Type::Array {
                        shape: shape[1..].into(),
                        elem: elem.clone(),
                    }
                }
            }
            (Type::TVector { elem, .. }, Node::Lit(Scalar::Int(_))) => elem.as_ref().clone(),
            (Type::Record(fields), Node::Lit(Scalar::Str(s))) => {
                let sym = fields.iter().find(|(n, _)| inf.module.resolve(*n) == &**s);
                match sym {
                    Some((_, t)) => t.clone(),
                    None => return Type::Failed(format!("record has no field `{s}`").into()),
                }
            }
            (Type::Any | Type::Deferred, _) => return current.clone(),
            _ => return Type::Deferred,
        };
    }
    current
}

/// The element type of a set expression (`elementof` / `external` argument),
/// read structurally — sets are not first-class in the type grammar.
fn set_element_type(inf: &mut Inferencer<'_>, node: Option<NodeId>) -> Type {
    let Some(node) = node else {
        return Type::Deferred;
    };
    let module = &*inf.module;
    match module.node(node) {
        Node::Const(sym) => match module.resolve(*sym) {
            "reals" | "posreals" | "nonnegreals" | "unitinterval" => Type::Scalar(ScalarType::Real),
            "integers" | "posintegers" | "nonnegintegers" => Type::Scalar(ScalarType::Integer),
            "booleans" => Type::Scalar(ScalarType::Boolean),
            "complexes" => Type::Scalar(ScalarType::Complex),
            "rngstates" => Type::RngState,
            "anything" => Type::Any,
            _ => Type::Deferred,
        },
        Node::Call(c) => match c.head {
            flatppl_core::CallHead::Builtin(op) => match module.resolve(op).to_string().as_str() {
                "interval" => Type::Scalar(ScalarType::Real),
                "cartpow" => {
                    let (set_arg, len_arg) = (c.args.first().copied(), c.args.get(1).copied());
                    let elem = set_element_type(inf, set_arg);
                    let dim = len_arg.map_or(Dim::Dynamic, |n| resolve_dim(inf, n));
                    Type::Array {
                        shape: Box::new([dim]),
                        elem: Box::new(elem),
                    }
                }
                _ => Type::Deferred,
            },
            _ => Type::Deferred,
        },
        _ => Type::Deferred,
    }
}

/// The domain of a measure type, for `draw` / `rand`.
fn measure_domain(m: Option<&Type>) -> Type {
    match m {
        Some(Type::Measure { domain }) => domain.as_ref().clone(),
        _ => Type::Deferred,
    }
}

/// `iid(M, n)`: n iid draws bundle into an array over M's domain. A literal
/// count (or literal count vector) gives static dims; anything computed is
/// dynamic until fixed-value const-eval lands (engine-concepts §17.1).
fn iid_type(inf: &mut Inferencer<'_>, args: &[ArgInfo]) -> Type {
    let domain = match arg_ty(args, 0) {
        Some(Type::Measure { domain }) => domain.as_ref().clone(),
        _ => return Type::Deferred,
    };
    let Some((count_node, _, _)) = args.get(1) else {
        return Type::Deferred;
    };
    Type::Measure {
        domain: Box::new(Type::Array {
            shape: count_dims(inf, *count_node),
            elem: Box::new(domain),
        }),
    }
}

/// The dims of an `iid` count argument: a vector literal contributes one dim
/// per element, anything else a single dim (see [`resolve_dim`]).
fn count_dims(inf: &mut Inferencer<'_>, node: NodeId) -> Box<[Dim]> {
    if let Node::Call(c) = inf.module.node(node)
        && matches!(c.head, flatppl_core::CallHead::Builtin(op)
            if inf.module.resolve(op) == "vector")
    {
        let elements: Vec<NodeId> = c.args.to_vec();
        return elements.iter().map(|&a| resolve_dim(inf, a)).collect();
    }
    Box::new([resolve_dim(inf, node)])
}

/// A single shape dim: literal integers are static at every level; at
/// `Level::Shape` the demand-driven fixed-integer resolver kicks in.
fn resolve_dim(inf: &mut Inferencer<'_>, node: NodeId) -> Dim {
    if let Node::Lit(Scalar::Int(n)) = inf.module.node(node) {
        return static_dim(*n);
    }
    if inf.level >= Level::Shape {
        if let Some(n) = resolve_fixed_int(inf, node, 0) {
            return static_dim(n);
        }
    }
    Dim::Dynamic
}

/// Demand-driven const-eval of a fixed-phase integer expression at a shape
/// position (engine-concepts §17.1, first slice: integers only). "Resolve,
/// don't rewrite" — the IR is read, never modified. Shape observers
/// (`lengthof`) short-circuit off the inferred type instead of recursing
/// into the value, so deferred-by-design computation stays deferred.
/// `None` means not statically resolvable — a non-fixed ancestor, or a
/// value op outside this resolver's reach (legitimately `%dynamic` for now;
/// the op-gap-vs-dynamic distinction hardens with the full §17.1 slice).
fn resolve_fixed_int(inf: &mut Inferencer<'_>, node: NodeId, depth: u32) -> Option<i64> {
    if depth > 64 {
        return None;
    }
    match inf.module.node(node).clone() {
        Node::Lit(Scalar::Int(n)) => Some(n),
        Node::Ref(r) if r.ns == flatppl_core::RefNs::SelfMod => {
            let binding = inf.module.binding_by_name(r.name)?;
            let rhs = inf.module.binding(binding).rhs;
            let (_, phase) = inf.infer_node(rhs);
            if phase == Phase::Fixed {
                resolve_fixed_int(inf, rhs, depth + 1)
            } else {
                None
            }
        }
        Node::Call(c) => {
            let flatppl_core::CallHead::Builtin(op) = c.head else {
                return None;
            };
            let name = inf.module.resolve(op).to_string();
            match name.as_str() {
                // Shape observer: read the inferred dim, never the value.
                "lengthof" | "length" => {
                    let arg = *c.args.first()?;
                    match inf.infer_node(arg).0 {
                        Type::Array { shape, .. } if shape.len() == 1 => match shape[0] {
                            Dim::Static(n) => Some(i64::from(n)),
                            Dim::Dynamic => None,
                        },
                        Type::TVector {
                            len: Dim::Static(n),
                            ..
                        } => Some(i64::from(n)),
                        _ => None,
                    }
                }
                "add" | "sub" | "mul" => {
                    let a = resolve_fixed_int(inf, *c.args.first()?, depth + 1)?;
                    let b = resolve_fixed_int(inf, *c.args.get(1)?, depth + 1)?;
                    match name.as_str() {
                        "add" => a.checked_add(b),
                        "sub" => a.checked_sub(b),
                        _ => a.checked_mul(b),
                    }
                }
                "neg" => resolve_fixed_int(inf, *c.args.first()?, depth + 1).map(|n| -n),
                _ => None,
            }
        }
        _ => None,
    }
}

fn static_dim(n: i64) -> Dim {
    u32::try_from(n).map(Dim::Static).unwrap_or(Dim::Dynamic)
}

/// `joint(a = M1, b = M2, …)` — a measure over the record of the components'
/// domains (the positional form is deferred with the shape work).
fn joint_type(named: &[NamedInfo]) -> Type {
    let mut fields = Vec::with_capacity(named.len());
    for (name, _, t, _) in named {
        match t {
            Type::Measure { domain } => fields.push((*name, domain.as_ref().clone())),
            _ => return Type::Deferred,
        }
    }
    Type::Measure {
        domain: Box::new(Type::Record(fields.into())),
    }
}

/// `functionof` / `kernelof` (spec §04 reification, §11 reified callables).
/// A `functionof` whose body is a measure *is* a kernel.
fn reification_type(
    inf: &mut Inferencer<'_>,
    id: NodeId,
    call: &Call,
    name: &str,
    args: &[ArgInfo],
) -> Type {
    let inputs: Box<[Symbol]> = match call.inputs.as_ref() {
        Some(Inputs::Spec(entries)) => entries.iter().map(|(n, _)| *n).collect(),
        Some(Inputs::Auto) => match inf.module.auto_inputs_of(id) {
            Some(entries) => entries.iter().map(|(n, _)| *n).collect(),
            None => {
                // The §04 auto-trace (parametric-leaf discovery) is future
                // work; a boundary-less reification stays %deferred.
                inf.note_once_str(
                    "boundary-less reifications (`%autoinputs`) are not traced yet \
                     — their types are left %deferred",
                );
                return Type::Deferred;
            }
        },
        None => unreachable!("reification_type called only when inputs are present"),
    };
    let body_ty = args.first().map(|(_, t, _)| t);
    match (name, body_ty) {
        ("kernelof", _) => Type::Kernel { inputs },
        ("functionof", Some(Type::Measure { .. })) => Type::Kernel { inputs },
        ("functionof", _) => Type::Function { inputs },
        _ => Type::Deferred,
    }
}

/// `likelihoodof(K, obs)` — inputs ride over from the kernel; the obstype is
/// the kernel's measure domain, recovered by looking through to the reified
/// body (spec §11 `%likelihood`).
fn likelihood_type(inf: &mut Inferencer<'_>, args: &[ArgInfo]) -> Type {
    let Some((k_node, Type::Kernel { inputs }, _)) = args.first() else {
        return Type::Deferred;
    };
    let inputs = inputs.clone();
    // A `functionof`-of-measure body exposes its domain; a `kernelof` body is
    // the random *value* itself, so its type is the observation domain.
    match reified_result_type(inf, *k_node) {
        Some(Type::Measure { domain }) => Type::Likelihood {
            inputs,
            obstype: domain,
        },
        Some(Type::Deferred) | None => Type::Deferred,
        Some(value_ty) => Type::Likelihood {
            inputs,
            obstype: Box::new(value_ty),
        },
    }
}

/// Calling a user-defined callable: a function returns its body's type, a
/// kernel returns the *measure* its body denotes (`kernelof` reifies the law
/// of a value-typed body).
fn user_call_type(inf: &mut Inferencer<'_>, callee: NodeId, callee_ty: &Type) -> Type {
    match callee_ty {
        Type::Function { .. } => reified_result_type(inf, callee).unwrap_or(Type::Deferred),
        Type::Kernel { .. } => match reified_result_type(inf, callee) {
            Some(m @ Type::Measure { .. }) => m,
            Some(value_ty) => Type::Measure {
                domain: Box::new(value_ty),
            },
            None => Type::Deferred,
        },
        _ => Type::Deferred,
    }
}

/// Look through a callable expression (a `%ref` to a binding, or an inline
/// reification) to its reified body's inferred type.
fn reified_result_type(inf: &mut Inferencer<'_>, mut node: NodeId) -> Option<Type> {
    // Deref self-module refs to the bound RHS (already inferred — typing the
    // callee forced the binding).
    loop {
        match inf.module.node(node) {
            Node::Ref(r) if r.ns == flatppl_core::RefNs::SelfMod => {
                let binding = inf.module.binding_by_name(r.name)?;
                node = inf.module.binding(binding).rhs;
            }
            Node::Call(c) if c.inputs.is_some() => {
                let body = *c.args.first()?;
                return inf.lookup_type(body).cloned();
            }
            _ => return None,
        }
    }
}

/// `broadcast(f, arrays…)` with a deterministic scalar built-in head: maps
/// elementwise over same-shape arrays (scalars ride along). Kernel/measure
/// heads and shape-mismatch handling arrive with the shape work.
fn broadcast_type(inf: &mut Inferencer<'_>, args: &[ArgInfo]) -> Type {
    let Some((head_node, _, _)) = args.first() else {
        return Type::Deferred;
    };
    let Node::Const(op) = inf.module.node(*head_node) else {
        return Type::Deferred;
    };
    let op_name = inf.module.resolve(*op).to_string();

    // Common shape: all array inputs agree.
    let mut shape: Option<Box<[Dim]>> = None;
    let mut elems: Vec<Type> = Vec::new();
    for (_, t, _) in &args[1..] {
        match t {
            Type::Array { shape: s, elem } => {
                match &shape {
                    None => shape = Some(s.clone()),
                    Some(prev) if prev == s => {}
                    Some(_) => return Type::Deferred,
                }
                elems.push(elem.as_ref().clone());
            }
            other => elems.push(other.clone()),
        }
    }
    let Some(shape) = shape else {
        return Type::Deferred; // no array input — scalar broadcast is a no-op shape-wise
    };

    let cell = match (op_name.as_str(), elems.as_slice()) {
        ("add" | "sub" | "mul" | "divide" | "pow" | "min" | "max", [a, b]) => {
            promote2(Some(a), Some(b))
        }
        ("neg", [a]) => a.clone(),
        (
            "exp" | "log" | "sqrt" | "invlogit" | "logit" | "log1p" | "expm1" | "abs" | "sin"
            | "cos" | "tan" | "tanh",
            [a],
        ) => real_or_complex(Some(a)),
        _ => return Type::Deferred,
    };
    Type::Array {
        shape,
        elem: Box::new(cell),
    }
}

/// The variate domain of a spec-§08 distribution constructor, or `None` when
/// the name is not a known distribution.
fn distribution_domain(
    inf: &mut Inferencer<'_>,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    use ScalarType::*;
    let scalar = |s: ScalarType| Some(Type::Scalar(s));
    let dynmat = || {
        Some(Type::Array {
            shape: Box::new([Dim::Dynamic, Dim::Dynamic]),
            elem: Box::new(Type::Scalar(Real)),
        })
    };
    match name {
        // Continuous univariate — domain ⊆ reals.
        "Normal" | "GeneralizedNormal" | "Cauchy" | "StudentT" | "Logistic" | "LogNormal"
        | "Exponential" | "Gamma" | "Weibull" | "InverseGamma" | "Beta" | "ChiSquared"
        | "VonMises" | "Laplace" => scalar(Real),
        // `Uniform(support)` — over its support set; scalar-real for now
        // (set-valued supports refine with the shape work).
        "Uniform" => scalar(Real),
        // Discrete univariate.
        "Bernoulli" => scalar(Boolean),
        "Categorical" | "Categorical0" | "Binomial" | "Geometric" | "NegativeBinomial"
        | "NegativeBinomial2" | "Poisson" => scalar(Integer),
        // Multivariate — at Level::Shape the dim comes from the length
        // parameter's inferred type (`mu` / `alpha` / `p`).
        "MvNormal" | "Dirichlet" => {
            let dim = param_dim(
                inf,
                args,
                named,
                if name == "MvNormal" { "mu" } else { "alpha" },
            );
            Some(Type::Array {
                shape: Box::new([dim]),
                elem: Box::new(Type::Scalar(Real)),
            })
        }
        "Multinomial" => {
            let dim = param_dim(inf, args, named, "p");
            Some(Type::Array {
                shape: Box::new([dim]),
                elem: Box::new(Type::Scalar(Integer)),
            })
        }
        "Wishart" | "InverseWishart" | "LKJ" | "LKJCholesky" => dynmat(),
        _ => None,
    }
}

/// The static dim of a distribution's length-defining parameter (`mu`,
/// `alpha`, `p`): its inferred type's single array dim, at `Level::Shape`.
fn param_dim(inf: &Inferencer<'_>, args: &[ArgInfo], named: &[NamedInfo], kwarg: &str) -> Dim {
    if inf.level < Level::Shape {
        return Dim::Dynamic;
    }
    let ty = named
        .iter()
        .find(|(n, _, _, _)| inf.module.resolve(*n) == kwarg)
        .map(|(_, _, t, _)| t)
        .or_else(|| args.first().map(|(_, t, _)| t));
    match ty {
        Some(Type::Array { shape, .. }) if shape.len() == 1 => shape[0],
        _ => Dim::Dynamic,
    }
}
