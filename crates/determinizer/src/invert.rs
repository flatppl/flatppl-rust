//! Analytic `(f_inv, logvol)` synthesis for known-bijection forward builtins
//! (spec §06 case 1: the engine MUST recognise the standard invertible maps —
//! exp/log, affine, pow, cis, matrix-affine — analytically). Used by
//! [`lower_pushfwd`] when a `pushfwd`'s forward argument is a bare builtin or a
//! one-op lambda rather than an explicit `bijection(f, f_inv, logvol)` node.
//!
//! `logvol` is the FORWARD log-volume element — `log|f'(x)|` as a function of the
//! forward input `x` — matching the explicit-bijection convention consumed by
//! `lower_pushfwd` (`logdensityof(M, f_inv(v)) - logvol(f_inv(v))`, §06 line 457).
//!
//! Refuse-don't-mislower: an UNRECOGNISED forward function returns `Ok(None)`
//! (the caller tries other cases / refuses); a RECOGNISED-but-non-invertible use
//! (`pow` with exponent 0, `pow` on a non-positive domain) returns `Err`
//! (refuse). A wrong `(f_inv, logvol)` is never synthesised.
//!
//! Task-1 registry (single builtin; verified against the closed-form derivative):
//! | forward   | `f_inv`            | `logvol` (in forward input `x`)          |
//! |-----------|--------------------|------------------------------------------|
//! | `exp`     | `log`              | `x`                    (d/dx eˣ = eˣ)    |
//! | `log`     | `exp`              | `neg(log(x))`          (d/dx ln x = 1/x) |
//! | `neg`     | `neg`              | `0`                    (log|−1| = 0)     |
//! | `pow(_,k)`| `x -> pow(x, 1/k)` | `add(log(abs(k)), mul(k-1, log(x)))`     |
//!
//! (Affine, matrix, and `cis` forms are later tasks.)

use crate::density::{build_call, refuse, resolve_ref_one};
use crate::refuse::RefuseError;
use flatppl_core::{
    Call, CallHead, Inputs, Module, NamedArg, Node, NodeId, Ref, RefNs, Scalar, Symbol, Type,
    ValueSet,
};

/// A synthesised change-of-variables: the inverse map `f_inv` and the FORWARD
/// log-volume element `logvol`, each a single-input FlatPIR callable the caller
/// applies via `build_user_call`.
pub(crate) struct Bijection {
    pub f_inv: NodeId,
    pub logvol: NodeId,
}

/// The recognised forward op denoted by a `pushfwd`'s forward argument.
enum Forward {
    Exp,
    Log,
    Neg,
    /// `pow(_, k)` with the literal exponent node `k`.
    Pow(NodeId),
}

/// Derive `(f_inv, logvol)` for the forward function `f` of a `pushfwd` over a
/// base measure whose variate domain is `domain`.
///
/// * `Ok(Some(_))` — `f` is a recognised, invertible forward map; the derived
///   change-of-variables is returned.
/// * `Ok(None)` — `f` is not a recognised forward function (the caller tries
///   other cases, or refuses).
/// * `Err(_)` — `f` is recognised but not invertible here (refuse).
pub(crate) fn derive_bijection(
    m: &mut Module,
    f: NodeId,
    domain: &Type,
) -> Result<Option<Bijection>, RefuseError> {
    let Some(fwd) = recognise_forward(m, f) else {
        return Ok(None);
    };
    let bij = match fwd {
        // d/dx eˣ = eˣ ⇒ log|f'| = x (identity).
        Forward::Exp => Bijection {
            f_inv: bare_builtin(m, "log"),
            logvol: identity_lambda(m),
        },
        // d/dx ln x = 1/x ⇒ log|f'| = −ln x.
        Forward::Log => {
            let logvol = lambda(m, |m, ph| {
                let logx = build_call(m, "log", &[ph]);
                build_call(m, "neg", &[logx])
            });
            Bijection {
                f_inv: bare_builtin(m, "exp"),
                logvol,
            }
        }
        // f'(x) = −1 ⇒ log|f'| = 0.
        Forward::Neg => {
            let logvol = lambda(m, |m, _ph| m.alloc(Node::Lit(Scalar::Real(0.0))));
            Bijection {
                f_inv: bare_builtin(m, "neg"),
                logvol,
            }
        }
        Forward::Pow(k_node) => return derive_pow(m, f, k_node, domain),
    };
    Ok(Some(bij))
}

/// `pow(_, k)`: f_inv `x -> pow(x, 1/k)`; logvol `x -> add(log(abs(k)), mul(k-1, log(x)))`.
/// Requires a nonzero literal exponent and a strictly-positive domain — the
/// inverse `x^{1/k}` and the log-volume's `log x` are defined only there
/// (d/dx xᵏ = k·xᵏ⁻¹ ⇒ log|f'| = log|k| + (k−1)·log x).
fn derive_pow(
    m: &mut Module,
    f: NodeId,
    k_node: NodeId,
    domain: &Type,
) -> Result<Option<Bijection>, RefuseError> {
    let Some(k) = literal_real(m, k_node) else {
        // A non-literal exponent is not a Task-1 recognised invertible form.
        return Ok(None);
    };
    if k == 0.0 {
        return Err(refuse(f, m, "pow with exponent 0 is not invertible"));
    }
    if !is_positive_domain(domain) {
        return Err(refuse(
            f,
            m,
            "pow forward is invertible (with this log-volume) only on a strictly-positive domain",
        ));
    }
    // f_inv: x -> pow(x, 1/k)
    let f_inv = lambda(m, |m, ph| {
        let inv_exp = m.alloc(Node::Lit(Scalar::Real(1.0 / k)));
        build_call(m, "pow", &[ph, inv_exp])
    });
    // logvol: x -> add(log(abs(k)), mul(k-1, log(x)))
    let logvol = lambda(m, |m, ph| {
        let abs_k = build_call(m, "abs", &[k_node]);
        let log_abs_k = build_call(m, "log", &[abs_k]);
        let km1 = m.alloc(Node::Lit(Scalar::Real(k - 1.0)));
        let logx = build_call(m, "log", &[ph]);
        let term = build_call(m, "mul", &[km1, logx]);
        build_call(m, "add", &[log_abs_k, term])
    });
    Ok(Some(Bijection { f_inv, logvol }))
}

/// Recognise the forward op denoted by a `pushfwd`'s forward argument `f`:
/// a bare builtin ref/const (`pushfwd(exp, M)`), or a one-op lambda whose body
/// is exactly one registry builtin applied to the single input placeholder
/// (`pushfwd(x -> exp(x), M)`). Anything else → `None`.
fn recognise_forward(m: &Module, f: NodeId) -> Option<Forward> {
    // Resolve one level of self-ref (`pushfwd(g, M)` where `g = exp`).
    let (f, _) = resolve_ref_one(m, f);
    match m.node(f) {
        // (a) bare builtin used as a value (`exp` / `log` / `neg`).
        Node::Const(sym) => {
            let name = m.resolve(*sym).to_string();
            builtin_forward(&name, None)
        }
        // (b) one-op lambda `x -> op(x)` / `x -> pow(x, k)`.
        Node::Call(c) => recognise_lambda(m, c),
        _ => None,
    }
}

/// Recognise a single-op `functionof` lambda `x -> op(x)` (or `x -> pow(x, k)`),
/// where `op(...)`'s first argument is exactly the lambda's placeholder.
fn recognise_lambda(m: &Module, c: &Call) -> Option<Forward> {
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    if m.resolve(sym) != "functionof" || c.args.len() != 1 {
        return None;
    }
    let ph = single_placeholder(c)?;
    let Node::Call(body) = m.node(c.args[0]) else {
        return None;
    };
    let CallHead::Builtin(op) = body.head else {
        return None;
    };
    let op_name = m.resolve(op).to_string();
    // The op's first positional argument must be the placeholder itself.
    let first = *body.args.first()?;
    if !is_placeholder_ref(m, first, ph) {
        return None;
    }
    match op_name.as_str() {
        "exp" | "log" | "neg" if body.args.len() == 1 => builtin_forward(&op_name, None),
        "pow" if body.args.len() == 2 => Some(Forward::Pow(body.args[1])),
        _ => None,
    }
}

/// Map a registry builtin name to its [`Forward`]. `pow` needs an exponent, so a
/// bare `pow` (no exponent) is not a recognised invertible form.
fn builtin_forward(name: &str, pow_exp: Option<NodeId>) -> Option<Forward> {
    match name {
        "exp" => Some(Forward::Exp),
        "log" => Some(Forward::Log),
        "neg" => Some(Forward::Neg),
        "pow" => pow_exp.map(Forward::Pow),
        _ => None,
    }
}

/// The placeholder name of a single-input `functionof` `Spec` boundary
/// (`((x (%ref %local _x_)))` → `_x_`), or `None` if not exactly one `%local`
/// input.
fn single_placeholder(c: &Call) -> Option<Symbol> {
    match &c.inputs {
        Some(Inputs::Spec(entries)) if entries.len() == 1 => {
            let (_input_name, r) = &entries[0];
            (r.ns == RefNs::Local).then_some(r.name)
        }
        _ => None,
    }
}

/// Is `id` the placeholder ref `(%ref %local <ph>)`?
fn is_placeholder_ref(m: &Module, id: NodeId, ph: Symbol) -> bool {
    matches!(m.node(id), Node::Ref(Ref { ns: RefNs::Local, name }) if *name == ph)
}

/// A bare builtin symbol node (`exp` / `log` / `neg`) usable directly as `f_inv`.
fn bare_builtin(m: &mut Module, name: &str) -> NodeId {
    let sym = m.intern(name);
    m.alloc(Node::Const(sym))
}

/// Build a single-input `functionof` lambda `x -> <body>`, spelled exactly as the
/// parser emits `x -> …` (input name `x`, placeholder `_x_`). `body(m, ph)`
/// receives the placeholder node id.
fn lambda(m: &mut Module, body: impl FnOnce(&mut Module, NodeId) -> NodeId) -> NodeId {
    let functionof = m.intern("functionof");
    let x = m.intern("x");
    let ph = m.intern("_x_");
    let ph_node = m.alloc(Node::Ref(Ref {
        ns: RefNs::Local,
        name: ph,
    }));
    let body_node = body(m, ph_node);
    m.alloc(Node::Call(Call {
        head: CallHead::Builtin(functionof),
        args: vec![body_node].into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: Some(Inputs::Spec(
            vec![(
                x,
                Ref {
                    ns: RefNs::Local,
                    name: ph,
                },
            )]
            .into(),
        )),
    }))
}

/// The identity lambda `x -> x` (body IS the placeholder) — the forward
/// log-volume of `exp`, spelled as the parser emits `x -> x`.
fn identity_lambda(m: &mut Module) -> NodeId {
    lambda(m, |_m, ph| ph)
}

/// The real value of a numeric literal node (`Int` widens to `Real`), or `None`.
fn literal_real(m: &Module, id: NodeId) -> Option<f64> {
    match m.node(id) {
        Node::Lit(Scalar::Real(r)) => Some(*r),
        Node::Lit(Scalar::Int(i)) => Some(*i as f64),
        _ => None,
    }
}

/// Is the domain's natural extent strictly positive (so `x^{1/k}` and `log x`
/// are defined)? Conservative: only sets that exclude zero and negatives count;
/// an unknown / real / non-negative domain does not.
fn is_positive_domain(domain: &Type) -> bool {
    match ValueSet::natural_of(domain) {
        ValueSet::PosReals | ValueSet::PosIntegers => true,
        ValueSet::Interval(lo, _) => lo > 0.0,
        _ => false,
    }
}
