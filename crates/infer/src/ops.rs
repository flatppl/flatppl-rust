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
use crate::trace::{Inferencer, join_phase};

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
    inf: &mut Inferencer<'_, '_>,
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
        let ty = user_call_type(inf, callee_node, &callee_ty, args);
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
        // ---- arithmetic (spec §07) — structural: result depends on arg shapes/types ----
        "add" | "sub" => elementwise2(&args.first(), &args.get(1)),
        "mul" => mul_type(args),
        "pow" => promote2(arg_ty(args, 0), arg_ty(args, 1)),
        "neg" => args.first().map_or(Type::Deferred, |(_, t, _)| t.clone()),
        "min" | "max" | "atan2" => promote2(arg_ty(args, 0), arg_ty(args, 1)),
        // `divide(a, b) = a / b` over real OR complex scalars (spec §07):
        // complex if either operand is complex, else real (true division —
        // integer operands still divide to real). NOT a constant Scalar(Real):
        // the complex case must promote. See `divide_type`.
        "divide" => divide_type(arg_ty(args, 0), arg_ty(args, 1)),

        // ---- containers (spec §03) — structural: result type threads arg types ----
        "vector" => vector_type(args),
        "tuple" => Type::Tuple(args.iter().map(|(_, t, _)| t.clone()).collect()),
        "record" => Type::Record(named.iter().map(|(n, _, t, _)| (*n, t.clone())).collect()),
        "rowstack" => rowstack_type(arg_ty(args, 0)),
        "get" => get_type(inf, args, /*base=*/ 1),
        "get0" => get_type(inf, args, /*base=*/ 0),
        "indicesof" | "indicesof0" => Type::Array {
            shape: Box::new([Dim::Dynamic]),
            elem: Box::new(Type::Scalar(ScalarType::Integer)),
        },
        // `sum` / `prod` / `mean` reduce a real/complex array to its element
        // type (spec §07 Reductions): mean of a complex array is complex.
        // (NOT a constant Scalar(Real); legacy ops.rs returned Real always.)
        "sum" | "prod" | "mean" => reduce_type(arg_ty(args, 0)),
        // Vector normalizations: same-shape real vector — shape must thread through.
        "softmax" | "logsoftmax" | "l1unit" | "l2unit" => match arg_ty(args, 0) {
            Some(Type::Array { shape, .. }) if shape.len() == 1 => Type::Array {
                shape: shape.clone(),
                elem: Box::new(Type::Scalar(ScalarType::Real)),
            },
            _ => Type::Deferred,
        },

        // ---- value-preserving assertion (spec §07) ----
        // `checked`/`fixed` are identity for typing (spec §03: `fixed(x)` ≡
        // `identity(x)`, a tooling hint) — the wrapped value's type rides through.
        // (`identity` itself, `ifelse`, `real`, `imag` are catalogue rows —
        // SameAsArg / CommonOf / RealOfArgShape.)
        "checked" | "fixed" => args.first().map_or(Type::Deferred, |(_, t, _)| t.clone()),

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
        // `relabel(M, labels)` (spec §06) renames the variate; the value domain
        // AND total mass are unchanged, so the measure type passes through whole
        // (unlike normalize/truncate, which reset the mass slot).
        "relabel" => arg_ty(args, 0).cloned().unwrap_or(Type::Deferred),
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
        "joint_likelihood" => joint_likelihood_type(args),

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
        "interval" | "cartprod" => Type::Any,
        // `cartpow(S, size)` takes exactly a set and a size; the size is an
        // integer (1-D) or a vector of positive integers (multi-axis), §03
        // "Cartesian power". The legacy variadic `cartpow(S, d1, d2, …)` form
        // is not in the spec — reject it rather than silently reading only the
        // first dimension (a multi-axis power is `cartpow(S, [d1, d2, …])`).
        "cartpow" => {
            if args.len() == 2 {
                Type::Any
            } else {
                inf.diags.push(crate::Diagnostic::error_at(
                    id,
                    "`cartpow` takes a set and a size: `cartpow(S, n)` or, for a \
                     multi-axis power, `cartpow(S, [d1, d2, …])` (a single vector \
                     size). The variadic `cartpow(S, d1, d2, …)` form is not valid \
                     (spec §03).",
                ));
                Type::Failed("cartpow expects (set, size)".into())
            }
        }

        // ---- broadcasting (spec §04) ----
        "broadcast" => broadcast_type(inf, args, named),

        // ---- catalogue dispatch (spec §07 functions + spec §08 distributions) ----
        // Per-name functions whose result is a pure scalar (constant, SameScalarKind,
        // or DomainMap) are declared in catalogue.ron and lowered here.
        // Distribution constructors (Sig::Distribution rows) are also dispatched here.
        // Structural ops above cannot be expressed in ResultSig and stay as code.
        _ => match function_result(&name, args) {
            Some(ty) => ty,
            None => match distribution_domain(inf, &name, args, named) {
                Some(domain) => Type::Measure {
                    domain: Box::new(domain),
                    mass: Mass::Deferred,
                },
                None => {
                    inf.note_gap(op);
                    Type::Deferred
                }
            },
        },
    };

    let phase = match name.as_str() {
        "elementof" => Phase::Parameterized,
        "external" | "load_data" | "load_module" | "standard_module" => Phase::Fixed,
        "draw" => Phase::Stochastic,
        // `lawof` reifies a value into its law; the law is deterministic
        // (parameterized or fixed) — `lawof` absorbs the stochasticity of the
        // `draw` ancestors rather than propagating it (spec §04 "Phase of the
        // reified law"). Trace the argument's law-phase instead of inheriting
        // the stochastic `joined`.
        "lawof" => args
            .first()
            .map_or(Phase::Fixed, |a| law_phase(inf, a.0, 0)),
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

/// `divide(a, b) = a / b` (spec §07): true division over scalars that are real
/// OR complex. The result is complex iff either operand is complex; otherwise
/// it is real — even for integer operands, since `1 / 2 = 0.5` is real (integer
/// floor-division is the separate `div` op). This differs from `promote2`,
/// which would keep integer/integer as integer.
fn divide_type(a: Option<&Type>, b: Option<&Type>) -> Type {
    use ScalarType::*;
    let is_complex = |t: Option<&Type>| matches!(t, Some(Type::Scalar(Complex)));
    let known_scalar = |t: Option<&Type>| {
        matches!(
            t,
            Some(Type::Scalar(Integer | Real | Complex | Boolean)) | Some(Type::Any)
        )
    };
    if is_complex(a) || is_complex(b) {
        Type::Scalar(Complex)
    } else if known_scalar(a) && known_scalar(b) {
        Type::Scalar(Real)
    } else {
        Type::Deferred
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
fn get_type(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo], base: i64) -> Type {
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
fn set_element_type(inf: &mut Inferencer<'_, '_>, node: Option<NodeId>) -> Type {
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
                    // `cartpow(S, size)` where `size` is an integer (1-D) or a
                    // vector of positive integers (multi-axis), §03 "Cartesian
                    // power". `count_dims` reads a `vector` literal as one dim
                    // per element, so `cartpow(reals, [2, 3])` yields a rank-2
                    // (2×3) array — not the rank-1 dynamic a single-dim read
                    // would give (the legacy `cartpow(S, d1, d2, …)` arity is
                    // not in the spec; only arg 1 is the size).
                    let (set_arg, size_arg) = (c.args.first().copied(), c.args.get(1).copied());
                    let elem = set_element_type(inf, set_arg);
                    let shape = size_arg.map_or_else(
                        || Box::new([Dim::Dynamic]) as Box<[Dim]>,
                        |n| count_dims(inf, n),
                    );
                    Type::Array {
                        shape,
                        elem: Box::new(elem),
                    }
                }
                "stdsimplex" => {
                    // `stdsimplex(n)` is the (n-1)-simplex {x ∈ ℝⁿ : xᵢ ≥ 0,
                    // Σxᵢ = 1} embedded in ℝⁿ (§03 "Standard simplex"): an
                    // element is a length-n real vector. The ≥0 / sum-to-1
                    // constraint lives in the value-set slot (`StdSimplex`), not
                    // the scalar type — so the element type is a rank-1 real
                    // array, mirroring `cartpow(reals, n)`.
                    let size_arg = c.args.first().copied();
                    let dim = size_arg.map_or(Dim::Dynamic, |n| resolve_dim(inf, n));
                    Type::Array {
                        shape: Box::new([dim]),
                        elem: Box::new(Type::Scalar(ScalarType::Real)),
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
fn iid_type(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Type {
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
fn count_dims(inf: &mut Inferencer<'_, '_>, node: NodeId) -> Box<[Dim]> {
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
fn resolve_dim(inf: &mut Inferencer<'_, '_>, node: NodeId) -> Dim {
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

/// The phase of `lawof(arg)` — the phase of the **reified law** of `arg`.
///
/// `lawof` absorbs the stochasticity of `draw` ancestors into the law, so the
/// result is deterministic: parameterized if the law depends on a free
/// `elementof` leaf, else fixed; never stochastic (spec §04 "Phase of the
/// reified law"). We re-derive the phase over the argument's ancestor subgraph
/// with two overrides vs the normal join: a `draw` contributes the law-phase of
/// its *measure operand* (absorbing the draw), and the recursion bottoms out at
/// `elementof` (parameterized) / fixed leaves — mirroring how `functionof`
/// traces to parametric leaves. Ref/`draw` cycles are bounded by `depth`.
fn law_phase(inf: &mut Inferencer<'_, '_>, node: NodeId, depth: u32) -> Phase {
    if depth > 64 {
        return Phase::Parameterized; // safe non-stochastic fallback
    }
    match inf.module.node(node).clone() {
        Node::Ref(r) if r.ns == flatppl_core::RefNs::SelfMod => {
            match inf.module.binding_by_name(r.name) {
                Some(b) => {
                    let rhs = inf.module.binding(b).rhs;
                    law_phase(inf, rhs, depth + 1)
                }
                None => Phase::Parameterized,
            }
        }
        Node::Call(c) => match c.head {
            flatppl_core::CallHead::Builtin(op) => match inf.module.resolve(op) {
                // Parametric leaf: the law depends on this free input.
                "elementof" => Phase::Parameterized,
                // Closed-over fixed leaves.
                "external" | "load_data" | "load_module" | "standard_module" => Phase::Fixed,
                // Absorb: the draw's stochasticity collapses into the law of
                // the measure it draws from.
                "draw" => c
                    .args
                    .first()
                    .map_or(Phase::Fixed, |&m| law_phase(inf, m, depth + 1)),
                // Any other node (measure constructor, container, arithmetic,
                // nested `lawof`, …): join the law-phases of its inputs.
                _ => c.args.iter().fold(Phase::Fixed, |acc, &a| {
                    join_phase(acc, law_phase(inf, a, depth + 1))
                }),
            },
            // User-callable application within a law: conservatively
            // parameterized (deterministic, may depend on inputs).
            _ => Phase::Parameterized,
        },
        // Literals and named constants are fixed; anything else (holes,
        // cross-module refs) is conservatively non-stochastic.
        Node::Lit(_) | Node::Const(_) => Phase::Fixed,
        _ => Phase::Parameterized,
    }
}

/// Demand-driven const-eval of a fixed-phase integer expression at a shape
/// position (engine-concepts §17.1, first slice: integers only). "Resolve,
/// don't rewrite" — the IR is read, never modified. Shape observers
/// (`lengthof`) short-circuit off the inferred type instead of recursing
/// into the value, so deferred-by-design computation stays deferred.
/// `None` means not statically resolvable — a non-fixed ancestor, or a
/// value op outside this resolver's reach (legitimately `%dynamic` for now;
/// the op-gap-vs-dynamic distinction hardens with the full §17.1 slice).
fn resolve_fixed_int(inf: &mut Inferencer<'_, '_>, node: NodeId, depth: u32) -> Option<i64> {
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
    inf: &mut Inferencer<'_, '_>,
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
                // §04 auto-trace: discover the body's `elementof` parametric
                // leaves (canonical-sorted by name) and fill the side-table, so
                // the reification types as a kernel/function over those inputs.
                let Some((body, _, _)) = args.first() else {
                    return Type::Deferred;
                };
                let entries = inf.collect_auto_inputs(*body);
                let names: Box<[Symbol]> = entries.iter().map(|(n, _)| *n).collect();
                inf.module.set_auto_inputs(id, entries.into());
                names
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
fn likelihood_type(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo]) -> Type {
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

/// `joint_likelihood(L1, L2, …)` — combine likelihoods by multiplying densities
/// (spec §06). It is *defined* to equal `likelihoodof(joint(model1, …), cat(obs1,
/// …))`, so the combined inputs are the union of the component inputs (order-
/// preserving) and the combined obstype is the §06 cat-composition of the
/// component obstypes — NOT a tuple. Any non-likelihood (or `%deferred`) argument
/// defers the whole result.
fn joint_likelihood_type(args: &[ArgInfo]) -> Type {
    if args.is_empty() {
        return Type::Deferred;
    }
    let mut inputs: Vec<Symbol> = Vec::new();
    let mut obstypes: Vec<Type> = Vec::with_capacity(args.len());
    for (_, t, _) in args {
        let Type::Likelihood {
            inputs: li,
            obstype,
        } = t
        else {
            return Type::Deferred;
        };
        for name in li.iter() {
            if !inputs.contains(name) {
                inputs.push(*name);
            }
        }
        obstypes.push((**obstype).clone());
    }
    Type::Likelihood {
        inputs: inputs.into(),
        obstype: Box::new(cat_compose(&obstypes)),
    }
}

/// The spec §06 "same shape class" composition for `cat`-joined variates: the
/// type of `cat(x1, x2, …)` when the components share a shape class. Used for the
/// obstype of [`joint_likelihood_type`] (the joint observation is `cat(obs…)`);
/// the same rule is what a future `cat` / positional-`joint` / `jointchain`
/// domain rule needs.
///
/// - all scalars    → a length-`n` vector (component scalars promoted);
/// - all 1-D arrays → one concatenated 1-D array (lengths summed, `%dynamic` if
///   any is dynamic; elements promoted);
/// - all records    → a merged record (fields concatenated — the spec requires
///   the component field names be distinct).
///
/// Anything else — an empty list, a `%deferred` component, mixed shape classes, or
/// a higher-rank array — yields `%deferred` (a sound "don't know", never a guess).
fn cat_compose(types: &[Type]) -> Type {
    let Some(first) = types.first() else {
        return Type::Deferred;
    };
    if types.iter().any(|t| matches!(t, Type::Deferred)) {
        return Type::Deferred;
    }
    match first {
        // all scalars → a length-n vector
        Type::Scalar(_) => {
            let mut elem: Option<Type> = None;
            for t in types {
                if !matches!(t, Type::Scalar(_)) {
                    return Type::Deferred; // mixed shape class
                }
                elem = Some(match elem {
                    None => t.clone(),
                    Some(prev) if &prev == t => prev,
                    Some(prev) => match promote2(Some(&prev), Some(t)) {
                        Type::Deferred => return Type::Deferred,
                        p => p,
                    },
                });
            }
            Type::Array {
                shape: Box::new([Dim::Static(types.len() as u32)]),
                elem: Box::new(elem.unwrap_or(Type::Any)),
            }
        }
        // all 1-D arrays → one concatenated 1-D array
        Type::Array { shape, .. } if shape.len() == 1 => {
            let mut total = 0u32;
            let mut dynamic = false;
            let mut elem: Option<Type> = None;
            for t in types {
                let Type::Array { shape, elem: e } = t else {
                    return Type::Deferred; // mixed shape class
                };
                if shape.len() != 1 {
                    return Type::Deferred; // higher-rank cat is not a §06 joint
                }
                match shape[0] {
                    Dim::Static(n) => total += n,
                    Dim::Dynamic => dynamic = true,
                }
                elem = Some(match elem {
                    None => (**e).clone(),
                    Some(prev) if &prev == e.as_ref() => prev,
                    Some(prev) => match promote2(Some(&prev), Some(e.as_ref())) {
                        Type::Deferred => return Type::Deferred,
                        p => p,
                    },
                });
            }
            Type::Array {
                shape: Box::new([if dynamic {
                    Dim::Dynamic
                } else {
                    Dim::Static(total)
                }]),
                elem: Box::new(elem.unwrap_or(Type::Any)),
            }
        }
        // all records → a merged record (component fields assumed distinct)
        Type::Record(_) => {
            let mut fields: Vec<(Symbol, Type)> = Vec::new();
            for t in types {
                let Type::Record(fs) = t else {
                    return Type::Deferred; // mixed shape class
                };
                fields.extend(fs.iter().cloned());
            }
            Type::Record(fields.into())
        }
        _ => Type::Deferred,
    }
}

/// Calling a user-defined callable: a function returns its body's type, a
/// kernel returns the *measure* its body denotes (`kernelof` reifies the law
/// of a value-typed body).
fn user_call_type(
    inf: &mut Inferencer<'_, '_>,
    callee: NodeId,
    callee_ty: &Type,
    args: &[ArgInfo],
) -> Type {
    // §09 standard-module application (`hepphys.CrystalBall(args)` /
    // `specfns.erf(x)`): the callee is a `RefNs::Module` ref whose catalogue
    // signature was stashed at resolution time. Lower it with the concrete
    // call args — a Distribution sig yields the measure with the catalogue
    // `MassTag`-derived mass (Normalized or Finite) already concrete, a
    // Function sig yields the result type. This is the §09 analogue of the
    // base-distribution / per-name-function dispatch in the builtin-call path,
    // which the user-call path bypasses.
    // Surface the honest-degrade note (spec policy): when this catalogue
    // row's support/shape is a sound approximation of the spec entry that
    // the type system cannot express exactly, the user sees why.
    if let Some(cref) = inf.module_catalogue_ref(callee) {
        let note = cref.degraded.clone(); // clone to drop the borrow before &mut inf call
        if let Some(note) = note {
            inf.note_once_str(&note);
        }
        return catalogue_call_type(inf, callee, args);
    }
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

/// Apply a §09 standard-module reference resolved against the catalogue. The
/// catalogue sig (stashed at resolution time, keyed by `callee`) is lowered
/// with a `LowerCtx` built from the concrete call args:
///   - Distribution sig → the lowered `Measure` with the mass from the
///     catalogue `MassTag` preserved (`Normalized` for a probability
///     distribution, `Finite` for a non-probability one such as
///     `ContinuedPoisson`). `fill_mass` leaves a concrete (non-`Deferred`)
///     mass untouched, so this rides through unchanged.
///   - Function sig → the lowered result type (scalar following the arg kind,
///     or a dynamic-dim matrix).
///
/// The support is carried separately by `catalogue_call_valueset`.
fn catalogue_call_type(inf: &mut Inferencer<'_, '_>, callee: NodeId, args: &[ArgInfo]) -> Type {
    // Clone the sig out to drop the immutable borrow before `lower`'s closures
    // re-borrow `args` inside the `&mut inf` call frame.
    let Some(sig) = inf.module_catalogue_ref(callee).map(|c| c.sig.clone()) else {
        return Type::Deferred;
    };
    // Both Distribution and Function sigs return the lowered type directly:
    // `catalogue_lower` already embeds the catalogue `MassTag` mass in the
    // `Measure` for distributions, and the result scalar/matrix type for
    // functions. No per-variant fixup is needed.
    let (ty, _vset) = catalogue_lower(&sig, args);
    ty
}

/// The value set of an applied §09 standard-module reference: a distribution's
/// support (the lowered support ValueSet) or a function result's natural set.
/// Mirrors `distribution_support` / `function_result`'s value-set handling but
/// reads the sig from the catalogue-ref side-table rather than by op name.
fn catalogue_call_valueset(
    inf: &mut Inferencer<'_, '_>,
    callee: NodeId,
    args: &[ArgInfo],
) -> ValueSet {
    let Some(sig) = inf.module_catalogue_ref(callee).map(|c| c.sig.clone()) else {
        return ValueSet::Unknown;
    };
    catalogue_lower(&sig, args).1
}

/// Lower a §09 catalogue sig with a `LowerCtx` built from the concrete
/// positional call args: `arg_scalar`/`arg_dim` read arg `i`'s inferred type,
/// `param_dim` (VectorFromParam) has no named-kwarg source at a `RefNs::Module`
/// application, so it falls back to the first positional arg's vector dim.
/// The `LowerCtx` borrows local closures, so it is built and consumed here in
/// one scope rather than returned.
fn catalogue_lower(sig: &crate::catalogue::Sig, args: &[ArgInfo]) -> (Type, ValueSet) {
    use crate::catalogue::{LowerCtx, lower};

    let first_dim = || match args.first().map(|(_, t, _)| t) {
        Some(Type::Array { shape, .. }) if shape.len() == 1 => shape[0],
        _ => Dim::Dynamic,
    };
    let ctx = LowerCtx {
        arg_scalar: &|i| match arg_ty(args, i) {
            Some(Type::Scalar(s)) => Some(*s),
            _ => None,
        },
        param_dim: &|_| first_dim(),
        arg_dim: &|i| match arg_ty(args, i) {
            Some(Type::Array { shape, .. }) if shape.len() == 1 => shape[0],
            _ => Dim::Dynamic,
        },
        arg_type: &|i| arg_ty(args, i).cloned(),
    };
    lower(sig, &ctx)
}

/// Look through a callable expression (a `%ref` to a binding, or an inline
/// reification) to its reified body node.
fn reified_body(inf: &Inferencer<'_, '_>, mut node: NodeId) -> Option<NodeId> {
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

/// The inferred type of a callable's reified body. For a cross-module callable
/// reference the body lives in the dependency's interner and is unreachable by
/// node here, so its result type was carried over at resolution time and is
/// read from the importer's side-table; for a local callable the body is found
/// by node and looked up in the trace.
fn reified_result_type(inf: &mut Inferencer<'_, '_>, node: NodeId) -> Option<Type> {
    if let Some(result) = inf.module_callable_result(node) {
        return Some(result.clone());
    }
    let body = reified_body(inf, node)?;
    inf.lookup_type(body).cloned()
}

/// `broadcast(f_or_K, args…)` (spec §04 broadcasting): a deterministic head
/// maps elementwise over same-shape arrays (scalars ride along) into an
/// array; a kernel / distribution-constructor head yields a **measure over
/// the array** of per-cell variates — that is why `draw` of a broadcast
/// distribution produces the observation array.
fn broadcast_type(inf: &mut Inferencer<'_, '_>, args: &[ArgInfo], named: &[NamedInfo]) -> Type {
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

    // §09 standard-module head (`hepphys.ContinuedPoisson`, `hepphys.interp_*`):
    // broadcast against the catalogue sig, exactly like a built-in head below.
    // Checked first because the catalogue-ref side-table is populated only for
    // §09 module references, so this never shadows a built-in. The per-cell
    // argument types feed the lowering: an array input contributes its element
    // type, a scalar rides along unchanged (every current §09 sig has a fixed
    // cell type, but lowering against the cell args keeps a future
    // `SameScalarKind` / `DomainMap` row correct).
    if let Some(sig) = inf.module_catalogue_ref(head_node).map(|c| c.sig.clone()) {
        let cell_args: Vec<ArgInfo> = args[1..]
            .iter()
            .map(|(n, t, p)| {
                let cell = match t {
                    Type::Array { elem, .. } => elem.as_ref().clone(),
                    other => other.clone(),
                };
                (*n, cell, *p)
            })
            .collect();
        return match catalogue_lower(&sig, &cell_args).0 {
            // Distribution head: an independent product over the array. Its cell
            // domain and mass come from the catalogue sig, so a non-probability
            // measure like `ContinuedPoisson` stays `Finite`, not forced to
            // `Normalized` (mirrors the built-in distribution path below).
            Type::Measure { domain, mass } => Type::Measure {
                domain: Box::new(Type::Array {
                    shape,
                    elem: domain,
                }),
                mass: broadcast_mass(mass),
            },
            // Deterministic function head (`hepphys.interp_poly6_exp`, …): maps
            // elementwise into an array of the per-cell result, exactly as the
            // built-in deterministic-op path does.
            cell => Type::Array {
                shape,
                elem: Box::new(cell),
            },
        };
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

/// The result type of a per-name function declared in the catalogue as
/// `Sig::Function`, or `None` if the name is not a known function (so the
/// caller can fall through to distribution dispatch, then gap).
///
/// `arg_scalar` is built from the inferred positional argument types so that
/// `SameScalarKind` and `DomainMap` sigs can read the call-site scalar kind.
fn function_result(name: &str, args: &[ArgInfo]) -> Option<Type> {
    use crate::catalogue::{LowerCtx, Sig, lower};

    let sig = crate::catalogue::builtin().base(name)?;
    // Only Function rows here; Distribution rows are handled by distribution_domain.
    let Sig::Function { .. } = sig else {
        return None;
    };
    let ctx = LowerCtx {
        arg_scalar: &|i| match arg_ty(args, i) {
            Some(Type::Scalar(s)) => Some(*s),
            _ => None,
        },
        param_dim: &|_| Dim::Dynamic,
        arg_dim: &|i| match arg_ty(args, i) {
            Some(Type::Array { shape, .. }) if shape.len() == 1 => shape[0],
            _ => Dim::Dynamic,
        },
        arg_type: &|i| arg_ty(args, i).cloned(),
    };
    // `lower` for Sig::Function returns (Type, ValueSet::natural_of(&ty)).
    // We only need the type here; the value-set arm is handled by call_valueset.
    let (ty, _) = lower(sig, &ctx);
    Some(ty)
}

/// The variate domain of a spec-§08 distribution constructor, or `None` when
/// the name is not a known distribution.
///
/// Dispatches via the catalogue for all 30 base distributions; non-distribution
/// names fall through to `None` unchanged.
fn distribution_domain(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> Option<Type> {
    use crate::catalogue::{LowerCtx, Sig, lower};

    let sig = crate::catalogue::builtin().base(name)?;
    // Confirm it's a distribution sig (Function rows are not distributions).
    let Sig::Distribution { .. } = sig else {
        return None;
    };
    // Build a context whose `param_dim` delegates to the existing helper
    // so that VectorFromParam dims (MvNormal/Dirichlet/Multinomial) resolve
    // the same way as before.  The closure borrows `inf` as `&Inferencer`
    // (a shared reborrow of the `&mut`); it is dropped before `inf` is
    // used mutably again.
    let ctx = LowerCtx {
        param_dim: &|kwarg| param_dim(inf, args, named, kwarg),
        arg_scalar: &|_| None,
        arg_dim: &|_| Dim::Dynamic,
        arg_type: &|i| arg_ty(args, i).cloned(),
    };
    let (ty, _vset) = lower(sig, &ctx);
    // `lower` wraps the domain in a `Type::Measure`; unwrap to get the domain.
    if let Type::Measure { domain, .. } = ty {
        Some(*domain)
    } else {
        None
    }
}

/// A dummy `SupportTag::Structural` check helper so `distribution_support` can
/// peek at the raw tag without re-looking up the catalogue entry.
#[inline]
fn support_is_structural(sig: &crate::catalogue::Sig) -> bool {
    use crate::catalogue::{Sig, SupportTag};
    matches!(
        sig,
        Sig::Distribution {
            support: SupportTag::Structural,
            ..
        }
    )
}

/// The static dim of a distribution's length-defining parameter (`mu`,
/// `alpha`, `p`): its inferred type's single array dim, at `Level::Shape`.
fn param_dim(inf: &Inferencer<'_, '_>, args: &[ArgInfo], named: &[NamedInfo], kwarg: &str) -> Dim {
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
    inf: &mut Inferencer<'_, '_>,
    call: &Call,
    callee: Option<&(NodeId, Type)>,
    args: &[ArgInfo],
    named: &[NamedInfo],
    ty: &Type,
) -> ValueSet {
    // User-callable application: the reified body's set rides over (for a
    // kernel call, the body set IS the output measure's support). A §09
    // standard-module reference has no reified body — its support/result set
    // is lowered from the catalogue sig with the call args.
    if let Some((callee_node, _)) = callee {
        if inf.module_catalogue_ref(*callee_node).is_some() {
            return catalogue_call_valueset(inf, *callee_node, args);
        }
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
        // `checked`/`fixed` are identity (spec §03), so the wrapped value's set
        // rides through — otherwise it would be needlessly lost to `Unknown`.
        "checked" | "fixed" => args
            .first()
            .map_or(ValueSet::Unknown, |(n, _, _)| inf.lookup_valueset(*n)),
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
fn set_expr_valueset(inf: &mut Inferencer<'_, '_>, node: Option<NodeId>) -> ValueSet {
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
                    // `ValueSet::CartPow` carries a single dim, so it can only
                    // describe a rank-1 power. A multi-axis size (`[2, 3]`) is a
                    // genuine rank-≥2 array whose value-set has no single-dim
                    // vocabulary — report `Unknown` rather than a misleading
                    // rank-1 set (consistent with `ValueSet::natural_of`, which
                    // also yields `Unknown` for rank-≥2 arrays).
                    let shape = c.args.get(1).map_or_else(
                        || Box::new([Dim::Dynamic]) as Box<[Dim]>,
                        |&n| count_dims(inf, n),
                    );
                    match shape.as_ref() {
                        [dim] => ValueSet::CartPow(Box::new(elem), *dim),
                        _ => ValueSet::Unknown,
                    }
                }
                _ => ValueSet::Unknown,
            }
        }
        _ => ValueSet::Unknown,
    }
}

/// A literal numeric bound (used by the `interval` reader above for `neg`).
fn bound_of(inf: &Inferencer<'_, '_>, node: Option<NodeId>) -> Option<f64> {
    match node.map(|n| inf.module.node(n)) {
        Some(Node::Lit(Scalar::Real(r))) => Some(*r),
        Some(Node::Lit(Scalar::Int(i))) => Some(*i as f64),
        Some(Node::Const(sym)) if inf.module.resolve(*sym) == "inf" => Some(f64::INFINITY),
        _ => None,
    }
}

/// The §08 Domain/Support column as a producer table: the support of a
/// distribution constructor, or `Unknown` for a non-distribution.
///
/// Dispatches via the catalogue for all 30 base distributions.  Distributions
/// with `SupportTag::Structural` (currently only Uniform) retain their live
/// code path so the support is computed from the call argument at inference
/// time rather than from a static tag.
fn distribution_support(
    inf: &mut Inferencer<'_, '_>,
    name: &str,
    args: &[ArgInfo],
    named: &[NamedInfo],
) -> ValueSet {
    use crate::catalogue::{LowerCtx, lower};

    let Some(sig) = crate::catalogue::builtin().base(name) else {
        return ValueSet::Unknown;
    };
    // Structural support: live code path reads the actual set argument.
    // Currently only Uniform; the static catalogue approximation (Unknown) is
    // not used here — the concrete arg-dependent value is what inference needs.
    if support_is_structural(sig) {
        return set_expr_valueset(inf, args.first().map(|a| a.0));
    }
    // All other distributions: lower via the catalogue to get the support ValueSet.
    let ctx = LowerCtx {
        param_dim: &|kwarg| param_dim(inf, args, named, kwarg),
        arg_scalar: &|_| None,
        arg_dim: &|_| Dim::Dynamic,
        arg_type: &|i| arg_ty(args, i).cloned(),
    };
    let (_ty, vs) = lower(sig, &ctx);
    vs
}

// =====================================================================
// Total-mass classes (Level::Normalization) — spec §11
// =====================================================================

/// Fill the `%mass` slot of a measure/kernel-typed call result, per the §06
/// composition rules. `normalize` on a measure with statically known zero or
/// infinite mass is a static error (spec: the result is undefined).
pub(crate) fn fill_mass(
    inf: &mut Inferencer<'_, '_>,
    id: NodeId,
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
        // User calls (including applied §09 catalogue distribution references)
        // had their mass set at construction — a catalogue distribution carries
        // its `MassTag`-derived mass (Normalized/Finite) out of
        // `catalogue_call_type`, so it is already concrete and was returned by
        // the `mass != Deferred` guard above. Reaching here means a deferred
        // mass that the call site cannot refine; pass it through unchanged.
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
                inf.diags.push(crate::Diagnostic::error_at(
                    id,
                    "`normalize` of a measure with zero total mass is undefined (spec §06)",
                ));
                return Type::Failed("normalize of a zero-mass measure".into());
            }
            Mass::LocallyFinite => {
                inf.diags.push(crate::Diagnostic::error_at(
                    id,
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

// =====================================================================
// Test helpers (not part of the normal public API)
// =====================================================================

/// For test use: the variate domain of a named distribution, with
/// `param_dim` provided by the caller rather than inferred from a live Module.
/// Returns `None` for non-distributions, matching the production function.
#[cfg(test)]
pub(crate) fn distribution_domain_static(
    name: &str,
    param_dim: &dyn Fn(&str) -> Dim,
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
        "Normal" | "GeneralizedNormal" | "Cauchy" | "StudentT" | "Logistic" | "LogNormal"
        | "Exponential" | "Gamma" | "Weibull" | "Pareto" | "InverseGamma" | "Beta"
        | "ChiSquared" | "VonMises" | "Laplace" | "Uniform" => scalar(Real),
        // Bernoulli: spec §08 "Domain/Support: integers/booleans".
        // Legacy ops.rs returned Boolean — that is a legacy bug; oracle now
        // reflects the spec-correct value (Integer) to match the catalogue.
        "Bernoulli" => scalar(Integer),
        "Categorical" | "Categorical0" | "Binomial" | "Geometric" | "NegativeBinomial"
        | "NegativeBinomial2" | "Poisson" => scalar(Integer),
        "MvNormal" => Some(Type::Array {
            shape: Box::new([param_dim("mu")]),
            elem: Box::new(Type::Scalar(Real)),
        }),
        "Dirichlet" => Some(Type::Array {
            shape: Box::new([param_dim("alpha")]),
            elem: Box::new(Type::Scalar(Real)),
        }),
        "Multinomial" => Some(Type::Array {
            shape: Box::new([param_dim("p")]),
            elem: Box::new(Type::Scalar(Integer)),
        }),
        "Wishart" | "InverseWishart" | "LKJ" | "LKJCholesky" => dynmat(),
        _ => None,
    }
}

/// For test use: the support `ValueSet` of a named distribution, with
/// `param_dim` provided by the caller. Returns `ValueSet::Unknown` for
/// non-distributions or arg-dependent supports (Uniform, Wishart family).
#[cfg(test)]
pub(crate) fn distribution_support_static(name: &str, param_dim: &dyn Fn(&str) -> Dim) -> ValueSet {
    use ValueSet::*;
    match name {
        // Uniform: support is structural (the set arg passed at the call site,
        // evaluated by set_expr_valueset at inference time). This oracle returns
        // Unknown — the correct static approximation — so the faithfulness test
        // can verify it matches the catalogue's Structural tag (which also lowers
        // to Unknown). The real arg-dependent behavior is guarded by the
        // dedicated `uniform_support_is_the_argument_set` test.
        "Uniform" => Unknown,
        "Normal" | "GeneralizedNormal" | "Cauchy" | "StudentT" | "Logistic" | "VonMises"
        | "Laplace" => Reals,
        "LogNormal" | "Gamma" | "InverseGamma" | "ChiSquared" | "Pareto" => PosReals,
        "Exponential" | "Weibull" => NonNegReals,
        "Beta" => UnitInterval,
        "Bernoulli" => Booleans,
        "Categorical" => PosIntegers,
        "Categorical0" | "Binomial" | "Geometric" | "NegativeBinomial" | "NegativeBinomial2"
        | "Poisson" => NonNegIntegers,
        "MvNormal" => CartPow(Box::new(Reals), param_dim("mu")),
        "Dirichlet" => StdSimplex(param_dim("alpha")),
        "Multinomial" => CartPow(Box::new(NonNegIntegers), param_dim("p")),
        // not in distribution_support — legacy returns Unknown
        "Wishart" | "InverseWishart" | "LKJ" | "LKJCholesky" => Unknown,
        _ => Unknown,
    }
}

/// For test use: the expected result type of a per-name function, mirroring
/// what the old per-name call_rule arms produced.  This is the static oracle
/// for the catalogue faithfulness test: it must match `function_result` for
/// every name in the migrated set.
///
/// `arg_scalar` simulates the caller's arg type at position 0.
/// Returns `None` for names that were never in the per-name arm set.
#[cfg(test)]
pub(crate) fn function_type_static(name: &str, arg0_scalar: Option<ScalarType>) -> Option<Type> {
    use ScalarType::*;
    let real_or_cplx = |s: Option<ScalarType>| match s {
        Some(Complex) => Type::Scalar(Complex),
        _ => Type::Scalar(Real),
    };
    match name {
        // scalar-integer output
        "floor" | "ceil" | "round" | "integer" => Some(Type::Scalar(Integer)),
        "div" | "mod" => Some(Type::Scalar(Integer)),
        "lengthof" | "length" => Some(Type::Scalar(Integer)),
        // scalar-real output
        // (divide and mean are NOT here: they are structural — divide promotes
        // its two operands, mean reduces to the array element type — handled
        // in call_rule, not the catalogue.)
        "logdensityof" | "densityof" => Some(Type::Scalar(Real)),
        "l1norm" | "l2norm" | "logsumexp" => Some(Type::Scalar(Real)),
        // scalar-complex output
        "cis" | "complex" => Some(Type::Scalar(Complex)),
        // scalar-boolean output
        "equal" | "unequal" | "lt" | "le" | "gt" | "ge" | "in" | "land" | "lor" | "lnot"
        | "isfinite" | "isinf" | "isnan" | "iszero" => Some(Type::Scalar(Boolean)),
        // real_or_complex: exp/log/sqrt/trig and friends
        "exp" | "log" | "log2" | "log10" | "sqrt" | "sin" | "cos" | "tan" | "asin" | "acos"
        | "atan" | "sinh" | "cosh" | "tanh" | "asinh" | "acosh" | "atanh" | "log1p" | "expm1"
        | "gamma" | "loggamma" | "logit" | "invlogit" | "probit" | "invprobit" | "conj" => {
            Some(real_or_cplx(arg0_scalar))
        }
        // abs / abs2: |z| and |z|² are always REAL even for complex input
        // (spec §07: |·| maps ℂ → ℝ). Legacy ops.rs used real_or_complex which
        // incorrectly returned Complex for complex input; the catalogue
        // DomainMap(Complex→Real) is spec-correct. The test oracle follows spec.
        "abs" | "abs2" => Some(Type::Scalar(Real)),
        _ => None,
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

#[cfg(test)]
mod cat_compose_tests {
    //! Unit coverage for the §06 cat-composition helper (the `joint_likelihood`
    //! obstype rule). The scalar→vector branch is also exercised end-to-end by
    //! the `joint_likelihood_unions_inputs_and_cats_obstype` golden test; these
    //! cover the array-concat / mixed-class / deferred branches directly.
    use super::*;

    fn arr(n: Dim, elem: ScalarType) -> Type {
        Type::Array {
            shape: Box::new([n]),
            elem: Box::new(Type::Scalar(elem)),
        }
    }

    #[test]
    fn scalars_make_a_length_n_vector() {
        use ScalarType::*;
        assert_eq!(
            cat_compose(&[Type::Scalar(Real), Type::Scalar(Real)]),
            arr(Dim::Static(2), Real)
        );
    }

    #[test]
    fn arrays_concatenate_their_lengths() {
        use ScalarType::*;
        assert_eq!(
            cat_compose(&[arr(Dim::Static(2), Real), arr(Dim::Static(3), Real)]),
            arr(Dim::Static(5), Real)
        );
    }

    #[test]
    fn a_dynamic_length_makes_the_concat_dynamic() {
        use ScalarType::*;
        assert_eq!(
            cat_compose(&[arr(Dim::Dynamic, Real), arr(Dim::Static(3), Real)]),
            arr(Dim::Dynamic, Real)
        );
    }

    #[test]
    fn mixed_shape_classes_defer() {
        use ScalarType::*;
        assert_eq!(
            cat_compose(&[Type::Scalar(Real), arr(Dim::Static(2), Real)]),
            Type::Deferred
        );
    }

    #[test]
    fn a_deferred_component_propagates() {
        use ScalarType::*;
        assert_eq!(
            cat_compose(&[Type::Scalar(Real), Type::Deferred]),
            Type::Deferred
        );
    }

    #[test]
    fn an_empty_list_defers() {
        assert_eq!(cat_compose(&[]), Type::Deferred);
    }
}
