//! The per-op type/phase rule catalogue (engine-concepts §18: one source of
//! truth per op — this table is what later passes share).
//!
//! Coverage is incremental and honest: ops without a rule yield `%deferred`
//! plus a once-per-op note (see crate docs). Rules mirror the spec tables —
//! §07 functions (domains/results), §08 distributions (variate domains),
//! §06 measure combinators, §04 reified callables.

use flatppl_core::{
    Call, CallHead, Dim, Inputs, Mass, Node, NodeId, Phase, Scalar, ScalarType, Symbol, Type,
    ValueSet,
};

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
        "l1norm" | "l2norm" | "logsumexp" => Type::Scalar(ScalarType::Real),
        // Vector normalizations map a vector to a same-shape real vector.
        "softmax" | "logsoftmax" | "l1unit" | "l2unit" => match arg_ty(args, 0) {
            Some(Type::Array { shape, .. }) if shape.len() == 1 => Type::Array {
                shape: shape.clone(),
                elem: Box::new(Type::Scalar(ScalarType::Real)),
            },
            _ => Type::Deferred,
        },
        "mean" => Type::Scalar(ScalarType::Real),

        // ---- value-preserving assertion (spec §07) ----
        "checked" => args.first().map_or(Type::Deferred, |(_, t, _)| t.clone()),

        // ---- parameters / inputs (spec §04) ----
        "elementof" | "external" => set_element_type(inf, args.first().map(|a| a.0)),

        // ---- measure algebra (spec §06) ----
        "lawof" => Type::Measure {
            domain: Box::new(args.first().map_or(Type::Any, |(_, t, _)| t.clone())),
            mass: Mass::Deferred,
        },
        "draw" => measure_domain(arg_ty(args, 0)),
        "iid" => iid_type(inf, args),
        // Measure-transforming ops keep the domain but get a FRESH mass slot
        // — their total mass differs from the base's and is computed by the
        // normalization-level rules (inheriting it via the type clone would
        // smuggle the base's class through `fill_mass`).
        "truncate" | "normalize" => fresh_measure(arg_ty(args, 0)),
        // `weighted(weight, base)` / `logweighted(logweight, base)` (spec
        // §06): the measure is the SECOND argument.
        "weighted" | "logweighted" => fresh_measure(arg_ty(args, 1)),
        // Reference measures (spec §06): measures over their support set.
        "Lebesgue" | "Counting" => Type::Measure {
            domain: Box::new(set_element_type(inf, args.first().map(|a| a.0))),
            mass: Mass::Deferred,
        },
        // `bayesupdate(L, prior)` (spec §06): the unnormalized posterior is a
        // measure over the prior's domain — pick the measure-typed argument,
        // with a fresh mass slot (the posterior's mass is the evidence).
        "bayesupdate" => fresh_measure(
            args.iter()
                .map(|(_, t, _)| t)
                .find(|t| matches!(t, Type::Measure { .. })),
        ),
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
        "broadcast" => broadcast_type(inf, args, named),

        // ---- distributions (spec §08) ----
        _ => match distribution_domain(inf, &name, args, named) {
            Some(domain) => Type::Measure {
                domain: Box::new(domain),
                mass: Mass::Deferred,
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

/// Clone a measure type with its mass reset to `Deferred` (to be filled by
/// the normalization-level rule for the op at hand); non-measures clone
/// as-is, absent arguments defer.
fn fresh_measure(t: Option<&Type>) -> Type {
    match t {
        Some(Type::Measure { domain, .. }) => Type::Measure {
            domain: domain.clone(),
            mass: Mass::Deferred,
        },
        Some(other) => other.clone(),
        None => Type::Deferred,
    }
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
        Some(Type::Measure { domain, .. }) => domain.as_ref().clone(),
        _ => Type::Deferred,
    }
}

/// `iid(M, n)`: n iid draws bundle into an array over M's domain. A literal
/// count (or literal count vector) gives static dims; anything computed is
/// dynamic until fixed-value const-eval lands (engine-concepts §17.1).
fn iid_type(inf: &mut Inferencer<'_>, args: &[ArgInfo]) -> Type {
    let domain = match arg_ty(args, 0) {
        Some(Type::Measure { domain, .. }) => domain.as_ref().clone(),
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
        mass: Mass::Deferred,
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
            Type::Measure { domain, .. } => fields.push((*name, domain.as_ref().clone())),
            _ => return Type::Deferred,
        }
    }
    Type::Measure {
        domain: Box::new(Type::Record(fields.into())),
        mass: Mass::Deferred,
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
        // `kernelof` reifies the LAW of a value-typed body — a probability
        // measure per input, i.e. a Markov kernel.
        ("kernelof", _) => Type::Kernel {
            inputs,
            mass: Mass::Normalized,
        },
        ("functionof", Some(Type::Measure { mass, .. })) => Type::Kernel {
            inputs,
            mass: *mass,
        },
        ("functionof", _) => Type::Function { inputs },
        _ => Type::Deferred,
    }
}

/// `likelihoodof(K, obs)` — inputs ride over from the kernel; the obstype is
/// the kernel's measure domain, recovered by looking through to the reified
/// body (spec §11 `%likelihood`).
fn likelihood_type(inf: &mut Inferencer<'_>, args: &[ArgInfo]) -> Type {
    let Some((k_node, Type::Kernel { inputs, .. }, _)) = args.first() else {
        return Type::Deferred;
    };
    let inputs = inputs.clone();
    // A `functionof`-of-measure body exposes its domain; a `kernelof` body is
    // the random *value* itself, so its type is the observation domain.
    match reified_result_type(inf, *k_node) {
        Some(Type::Measure { domain, .. }) => Type::Likelihood {
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
        Type::Kernel { mass, .. } => match reified_result_type(inf, callee) {
            Some(Type::Measure { domain, .. }) => Type::Measure {
                domain,
                mass: *mass,
            },
            Some(value_ty) => Type::Measure {
                domain: Box::new(value_ty),
                mass: *mass,
            },
            None => Type::Deferred,
        },
        _ => Type::Deferred,
    }
}

/// Look through a callable expression (a `%ref` to a binding, or an inline
/// reification) to its reified body node.
fn reified_body(inf: &Inferencer<'_>, mut node: NodeId) -> Option<NodeId> {
    // Deref self-module refs to the bound RHS (already inferred — typing the
    // callee forced the binding).
    loop {
        match inf.module.node(node) {
            Node::Ref(r) if r.ns == flatppl_core::RefNs::SelfMod => {
                let binding = inf.module.binding_by_name(r.name)?;
                node = inf.module.binding(binding).rhs;
            }
            Node::Call(c) if c.inputs.is_some() => return c.args.first().copied(),
            _ => return None,
        }
    }
}

/// The inferred type of a callable's reified body.
fn reified_result_type(inf: &mut Inferencer<'_>, node: NodeId) -> Option<Type> {
    let body = reified_body(inf, node)?;
    inf.lookup_type(body).cloned()
}

/// `broadcast(f_or_K, args…)` (spec §04 broadcasting): a deterministic head
/// maps elementwise over same-shape arrays (scalars ride along) into an
/// array; a kernel / distribution-constructor head yields a **measure over
/// the array** of per-cell variates — that is why `draw` of a broadcast
/// distribution produces the observation array.
fn broadcast_type(inf: &mut Inferencer<'_>, args: &[ArgInfo], named: &[NamedInfo]) -> Type {
    let Some((head_node, head_ty, _)) = args.first() else {
        return Type::Deferred;
    };
    let (head_node, head_ty) = (*head_node, head_ty.clone());

    // Common shape over every data input — positional and keyword alike;
    // mismatching array shapes are deferred until real shape-broadcasting.
    let mut shape: Option<Box<[Dim]>> = None;
    let mut elems: Vec<Type> = Vec::new();
    let data_types = args[1..]
        .iter()
        .map(|(_, t, _)| t)
        .chain(named.iter().map(|(_, _, t, _)| t));
    for t in data_types {
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

    // User-callable head (`broadcast(group_kernel, mu = mu_g)`): the cell
    // comes from the reified body, exactly as in a direct call.
    match &head_ty {
        Type::Function { .. } => {
            let cell = reified_result_type(inf, head_node).unwrap_or(Type::Deferred);
            return Type::Array {
                shape,
                elem: Box::new(cell),
            };
        }
        Type::Kernel { mass, .. } => {
            let mass = *mass;
            let cell = match reified_result_type(inf, head_node) {
                Some(Type::Measure { domain, .. }) => *domain,
                Some(value_ty) => value_ty,
                None => return Type::Deferred,
            };
            return Type::Measure {
                domain: Box::new(Type::Array {
                    shape,
                    elem: Box::new(cell),
                }),
                mass: broadcast_mass(mass),
            };
        }
        _ => {}
    }

    // Built-in head: a distribution constructor broadcasts into a measure
    // over the array; a deterministic scalar op maps elementwise.
    let Node::Const(op) = inf.module.node(head_node) else {
        return Type::Deferred;
    };
    let op_name = inf.module.resolve(*op).to_string();
    if let Some(cell_domain) = distribution_domain(inf, &op_name, &[], &[]) {
        return Type::Measure {
            domain: Box::new(Type::Array {
                shape,
                elem: Box::new(cell_domain),
            }),
            // Independent product of per-cell distributions.
            mass: Mass::Normalized,
        };
    }

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

// =====================================================================
// Value sets (Level::Valueset) — the third `%meta` slot
// =====================================================================

pub(crate) fn literal_valueset(s: &Scalar) -> ValueSet {
    match s {
        Scalar::Int(n) if *n > 0 => ValueSet::PosIntegers,
        Scalar::Int(n) if *n == 0 => ValueSet::NonNegIntegers,
        Scalar::Int(_) => ValueSet::Integers,
        // A real literal is its own singleton interval.
        Scalar::Real(r) => ValueSet::Interval(*r, *r),
        Scalar::Bool(_) => ValueSet::Booleans,
        Scalar::Str(_) => ValueSet::Unknown,
    }
}

pub(crate) fn const_valueset(name: &str) -> ValueSet {
    match name {
        "pi" | "inf" => ValueSet::PosReals,
        "im" => ValueSet::Complexes,
        _ => ValueSet::Unknown,
    }
}

/// The value set of a call node: a measure node's support, a value node's
/// strongest known containing set. Conservative — `Unknown` is always sound.
pub(crate) fn call_valueset(
    inf: &mut Inferencer<'_>,
    call: &Call,
    callee: Option<&(NodeId, Type)>,
    args: &[ArgInfo],
    named: &[NamedInfo],
    ty: &Type,
) -> ValueSet {
    // User-callable application: the reified body's set rides over (for a
    // kernel call, the body set IS the output measure's support).
    if let Some((callee_node, _)) = callee {
        return match reified_body(inf, *callee_node) {
            Some(body) => inf.lookup_valueset(body),
            None => ValueSet::Unknown,
        };
    }
    let CallHead::Builtin(op) = call.head else {
        return ValueSet::Unknown;
    };
    // Reifications are callables, not values.
    if call.inputs.is_some() {
        return ValueSet::Unknown;
    }
    let name = inf.module.resolve(op).to_string();
    match name.as_str() {
        // Parameters / loaded sets.
        "elementof" | "external" => set_expr_valueset(inf, args.first().map(|a| a.0)),
        // Measure supports (the measure node's value set IS its support).
        "Lebesgue" | "Counting" => set_expr_valueset(inf, args.first().map(|a| a.0)),
        "lawof" => args
            .first()
            .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n)),
        "truncate" => {
            // Sound superset: the truncated support lies inside S.
            match set_expr_valueset(inf, args.get(1).map(|a| a.0)) {
                ValueSet::Unknown => args
                    .first()
                    .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n)),
                set => set,
            }
        }
        // Reweighting never grows the support.
        "normalize" | "bayesupdate" => args
            .first()
            .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n)),
        "weighted" | "logweighted" => args
            .get(1)
            .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n)),
        // Drawing yields a value in the measure's support.
        "draw" => args
            .first()
            .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n)),
        "iid" => {
            let inner = args
                .first()
                .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n));
            match (inner, ty) {
                (ValueSet::Unknown, _) => ValueSet::Unknown,
                (inner, Type::Measure { domain, .. }) => match domain.as_ref() {
                    Type::Array { shape, .. } if shape.len() == 1 => {
                        ValueSet::CartPow(Box::new(inner), shape[0])
                    }
                    _ => ValueSet::Unknown,
                },
                _ => ValueSet::Unknown,
            }
        }
        // Normalization functions (spec §07).
        "softmax" => ValueSet::StdSimplex(vector_dim(arg_ty(args, 0))),
        "l1unit" => {
            let arg_set = args
                .first()
                .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n));
            // `v/‖v‖₁` lies on the simplex only for nonnegative `v`.
            if arg_set.subset_of(&ValueSet::CartPow(
                Box::new(ValueSet::NonNegReals),
                Dim::Dynamic,
            )) {
                ValueSet::StdSimplex(vector_dim(arg_ty(args, 0)))
            } else {
                ValueSet::Unknown
            }
        }
        // Range-constrained scalar functions.
        "exp" => ValueSet::PosReals,
        "abs" | "abs2" | "sqrt" => ValueSet::NonNegReals,
        "invlogit" | "invprobit" => ValueSet::UnitInterval,
        // Vectors lift a common element set; heterogeneous elements widen
        // to the strongest named set containing all of them (literal reals
        // are singleton intervals, so without widening `l1unit`'s simplex
        // guard would never fire on literal weight vectors).
        "vector" => {
            let sets: Vec<ValueSet> = args
                .iter()
                .map(|(n, _, _)| inf.lookup_valueset(*n))
                .collect();
            match join_scalar_sets(&sets) {
                Some(e) => ValueSet::CartPow(Box::new(e), Dim::Static(args.len() as u32)),
                None => ValueSet::Unknown,
            }
        }
        // Distribution constructors: the support column of spec §08.
        _ => distribution_support(inf, &name, args, named),
    }
}

/// The single dim of a vector-typed argument, for simplex sizes.
fn vector_dim(ty: Option<&Type>) -> Dim {
    match ty {
        Some(Type::Array { shape, .. }) if shape.len() == 1 => shape[0],
        _ => Dim::Dynamic,
    }
}

/// A set *expression* (an `elementof` / `truncate` / reference-measure
/// argument) read structurally into a [`ValueSet`].
fn set_expr_valueset(inf: &mut Inferencer<'_>, node: Option<NodeId>) -> ValueSet {
    let Some(node) = node else {
        return ValueSet::Unknown;
    };
    match inf.module.node(node).clone() {
        Node::Const(sym) => match inf.module.resolve(sym) {
            "reals" => ValueSet::Reals,
            "posreals" => ValueSet::PosReals,
            "nonnegreals" => ValueSet::NonNegReals,
            "unitinterval" => ValueSet::UnitInterval,
            "integers" => ValueSet::Integers,
            "posintegers" => ValueSet::PosIntegers,
            "nonnegintegers" => ValueSet::NonNegIntegers,
            "booleans" => ValueSet::Booleans,
            "complexes" => ValueSet::Complexes,
            "rngstates" => ValueSet::RngStates,
            "anything" => ValueSet::Anything,
            _ => ValueSet::Unknown,
        },
        Node::Call(c) => {
            let CallHead::Builtin(op) = c.head else {
                return ValueSet::Unknown;
            };
            match inf.module.resolve(op).to_string().as_str() {
                "interval" => {
                    let bound = |n: Option<&NodeId>| match n.map(|&n| inf.module.node(n).clone()) {
                        Some(Node::Lit(Scalar::Real(r))) => Some(r),
                        Some(Node::Lit(Scalar::Int(i))) => Some(i as f64),
                        Some(Node::Const(sym)) if inf.module.resolve(sym) == "inf" => {
                            Some(f64::INFINITY)
                        }
                        Some(Node::Call(neg))
                            if matches!(neg.head, CallHead::Builtin(op)
                                if inf.module.resolve(op) == "neg") =>
                        {
                            bound_of(inf, neg.args.first().copied()).map(|b| -b)
                        }
                        _ => None,
                    };
                    match (bound(c.args.first()), bound(c.args.get(1))) {
                        (Some(lo), Some(hi)) => ValueSet::Interval(lo, hi),
                        _ => ValueSet::Unknown,
                    }
                }
                "stdsimplex" => ValueSet::StdSimplex(
                    c.args
                        .first()
                        .map_or(Dim::Dynamic, |&n| resolve_dim(inf, n)),
                ),
                "cartpow" => {
                    let elem = set_expr_valueset(inf, c.args.first().copied());
                    if elem == ValueSet::Unknown {
                        return ValueSet::Unknown;
                    }
                    let dim = c.args.get(1).map_or(Dim::Dynamic, |&n| resolve_dim(inf, n));
                    ValueSet::CartPow(Box::new(elem), dim)
                }
                _ => ValueSet::Unknown,
            }
        }
        _ => ValueSet::Unknown,
    }
}

/// A literal numeric bound (used by the `interval` reader above for `neg`).
fn bound_of(inf: &Inferencer<'_>, node: Option<NodeId>) -> Option<f64> {
    match node.map(|n| inf.module.node(n)) {
        Some(Node::Lit(Scalar::Real(r))) => Some(*r),
        Some(Node::Lit(Scalar::Int(i))) => Some(*i as f64),
        Some(Node::Const(sym)) if inf.module.resolve(*sym) == "inf" => Some(f64::INFINITY),
        _ => None,
    }
}

/// The §08 Domain/Support column as a producer table: the support of a
/// distribution constructor, or `Unknown` for a non-distribution.
fn distribution_support(
    inf: &mut Inferencer<'_>,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> ValueSet {
    use ValueSet::*;
    match name {
        "Uniform" => set_expr_valueset(inf, args.first().map(|a| a.0)),
        "Normal" | "GeneralizedNormal" | "Cauchy" | "StudentT" | "Logistic" | "VonMises"
        | "Laplace" => Reals,
        "LogNormal" | "Gamma" | "InverseGamma" | "ChiSquared" => PosReals,
        "Exponential" | "Weibull" => NonNegReals,
        "Beta" => UnitInterval,
        "Bernoulli" => Booleans,
        "Categorical" => PosIntegers,
        "Categorical0" | "Binomial" | "Geometric" | "NegativeBinomial" | "NegativeBinomial2"
        | "Poisson" => NonNegIntegers,
        "MvNormal" => CartPow(Box::new(Reals), param_dim(inf, args, named, "mu")),
        "Dirichlet" => StdSimplex(param_dim(inf, args, named, "alpha")),
        "Multinomial" => CartPow(Box::new(NonNegIntegers), param_dim(inf, args, named, "p")),
        _ => Unknown,
    }
}

// =====================================================================
// Total-mass classes (Level::Normalization) — spec §11
// =====================================================================

/// Fill the `%mass` slot of a measure/kernel-typed call result, per the §06
/// composition rules. `normalize` on a measure with statically known zero or
/// infinite mass is a static error (spec: the result is undefined).
pub(crate) fn fill_mass(
    inf: &mut Inferencer<'_>,
    call: &Call,
    callee: Option<&(NodeId, Type)>,
    ty: Type,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Type {
    // Only measure types carry a deferred mass to fill; kernels and user
    // calls were filled at construction (their mass rides the callee).
    let Type::Measure { domain, mass } = ty else {
        return ty;
    };
    if mass != Mass::Deferred {
        return Type::Measure { domain, mass };
    }
    if callee.is_some() || call.inputs.is_some() {
        return Type::Measure { domain, mass };
    }
    let CallHead::Builtin(op) = call.head else {
        return Type::Measure { domain, mass };
    };
    let name = inf.module.resolve(op).to_string();

    let arg_mass = |i: usize| match arg_ty(args, i) {
        Some(Type::Measure { mass, .. }) => *mass,
        _ => Mass::Unknown,
    };

    let mass = match name.as_str() {
        // Every §08 distribution is a probability measure.
        "lawof" => Mass::Normalized,
        // Reference measures: finite on a bounded support, infinite (but
        // boundedly finite) on an unbounded one.
        "Lebesgue" | "Counting" => {
            match set_expr_valueset(inf, args.first().map(|a| a.0)).is_bounded() {
                Some(true) => Mass::Finite,
                Some(false) => Mass::LocallyFinite,
                None => Mass::Unknown,
            }
        }
        "iid" => match arg_mass(0) {
            Mass::Normalized => Mass::Normalized,
            Mass::Null => Mass::Null,
            Mass::Finite => Mass::Finite,
            Mass::LocallyFinite => Mass::LocallyFinite,
            _ => Mass::Unknown,
        },
        "joint" => {
            let masses: Vec<Mass> = named
                .iter()
                .map(|(_, _, t, _)| match t {
                    Type::Measure { mass, .. } => *mass,
                    _ => Mass::Unknown,
                })
                .collect();
            product_mass(&masses)
        }
        "truncate" => match arg_mass(0) {
            Mass::Null => Mass::Null,
            Mass::Normalized | Mass::Finite => Mass::Finite,
            Mass::LocallyFinite => {
                match set_expr_valueset(inf, args.get(1).map(|a| a.0)).is_bounded() {
                    Some(true) => Mass::Finite,
                    _ => Mass::Unknown,
                }
            }
            _ => Mass::Unknown,
        },
        // A fixed scalar weight rescales: classes survive, except that an
        // unknown constant demotes `%normalized` to `%finite`.
        "weighted" | "logweighted" => {
            let base = arg_mass(1);
            if base == Mass::Null {
                Mass::Null
            } else if matches!(
                (arg_ty(args, 0), args.first().map(|(_, _, p)| *p)),
                (Some(Type::Scalar(_)), Some(Phase::Fixed))
            ) {
                match base {
                    Mass::Normalized | Mass::Finite => Mass::Finite,
                    Mass::LocallyFinite => Mass::LocallyFinite,
                    _ => Mass::Unknown,
                }
            } else {
                Mass::Unknown
            }
        }
        "normalize" => match arg_mass(0) {
            Mass::Null => {
                inf.diags.push(crate::Diagnostic::error(
                    "`normalize` of a measure with zero total mass is undefined (spec §06)",
                ));
                return Type::Failed("normalize of a zero-mass measure".into());
            }
            Mass::LocallyFinite => {
                inf.diags.push(crate::Diagnostic::error(
                    "`normalize` of a measure with infinite total mass is undefined (spec §06)",
                ));
                return Type::Failed("normalize of an infinite-mass measure".into());
            }
            _ => Mass::Normalized,
        },
        "bayesupdate" => Mass::Unknown,
        "jointchain" => match (arg_mass(0), arg_mass(1)) {
            (m, Mass::Normalized) => m,
            _ => Mass::Unknown,
        },
        // A §08 distribution constructor (this arm is reached only for
        // measure-typed results, i.e. recognized distributions).
        _ => Mass::Normalized,
    };
    Type::Measure { domain, mass }
}

/// The mass of an independent product of components.
fn product_mass(masses: &[Mass]) -> Mass {
    use Mass::*;
    if masses.contains(&Null) {
        return Null;
    }
    if masses.iter().all(|m| *m == Normalized) {
        return Normalized;
    }
    if masses.iter().all(|m| matches!(m, Normalized | Finite)) {
        return Finite;
    }
    if masses
        .iter()
        .all(|m| matches!(m, Normalized | Finite | LocallyFinite))
    {
        return LocallyFinite;
    }
    Unknown
}

/// Broadcasting a kernel over data cells: an independent product per cell.
fn broadcast_mass(cell: Mass) -> Mass {
    match cell {
        Mass::Normalized => Mass::Normalized,
        Mass::Null => Mass::Null,
        Mass::Finite => Mass::Finite,
        _ => Mass::Unknown,
    }
}

/// The strongest common containing set of several element sets: their shared
/// value if equal, else the strongest named scalar set that contains all
/// (widening, strongest first). `None` when nothing fits.
fn join_scalar_sets(sets: &[ValueSet]) -> Option<ValueSet> {
    let first = sets.first()?;
    if sets.iter().all(|s| s == first) && *first != ValueSet::Unknown {
        return Some(first.clone());
    }
    const CANDIDATES: &[ValueSet] = &[
        ValueSet::PosIntegers,
        ValueSet::NonNegIntegers,
        ValueSet::Integers,
        ValueSet::UnitInterval,
        ValueSet::PosReals,
        ValueSet::NonNegReals,
        ValueSet::Reals,
        ValueSet::Booleans,
        ValueSet::Complexes,
    ];
    CANDIDATES
        .iter()
        .find(|c| sets.iter().all(|s| s.subset_of(c)))
        .cloned()
}
