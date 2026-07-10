//! Analytic `(f_inv, logvol)` synthesis for known-bijection forward functions
//! (spec §06 case 1: the engine MUST recognise the standard invertible maps —
//! exp/log, affine, pow — and their scalar COMPOSITIONS analytically). Used by
//! [`lower_pushfwd`] when a `pushfwd`'s forward argument is a bare builtin, a
//! one-op lambda, or a chain/affine lambda rather than an explicit
//! `bijection(f, f_inv, logvol)` node.
//!
//! `logvol` is the FORWARD log-volume element — `log|f'(x)|` as a function of the
//! forward input `x` — matching the explicit-bijection convention consumed by
//! `lower_pushfwd` (`logdensityof(M, f_inv(v)) - logvol(f_inv(v))`, §06 line 457).
//!
//! ## Scalar-chain inversion
//!
//! The forward body is a linear chain of ops `f = gₙ∘…∘g₁` terminating at the
//! single input placeholder — each op either a registry unary (`exp`/`log`/`neg`)
//! or an AFFINE node with ONE literal operand `c` and ONE sub-expression `u`
//! (`mul`/`divide`/`add`/`sub`). We invert it by:
//!
//! * **`f_inv(y) = g₁_inv(…(gₙ_inv(y))…)`** — apply the per-op inverses
//!   outermost-first to a fresh placeholder (undo `gₙ` first, `g₁` last).
//! * **`logvol(x) = Σᵢ logvolᵢ(gᵢ₋₁∘…∘g₁(x))`** — the chain rule
//!   `log|f'| = Σᵢ log|gᵢ'|`, with each op's LOCAL forward log-derivative
//!   evaluated at its PARTIAL-FORWARD input `gᵢ₋₁∘…∘g₁(x)`. That partial-forward
//!   point is exactly `gᵢ`'s own sub-expression node in the forward body (already
//!   an expression in the input placeholder), so we reuse it directly rather than
//!   re-deriving the composition — only the non-constant local logvols reference
//!   it (`exp`: `z`; `log`: `−log z`); the affine ops contribute a constant
//!   (`mul`: `log|c|`; `divide`: `−log|c|`) or zero (`add`/`sub`, `neg`), so
//!   zero terms are dropped and an all-zero sum collapses to the literal `0`.
//!
//! Per-op table (`acc` = the accumulating inverse argument, `z` = the op's
//! partial-forward input; local logvol = FORWARD `log|gᵢ'|` at `z`):
//! | op            | inverse of `acc`   | local logvol        |
//! |---------------|--------------------|---------------------|
//! | `exp`         | `log(acc)`         | `z`                 |
//! | `log`         | `exp(acc)`         | `neg(log(z))`       |
//! | `neg`         | `neg(acc)`         | `0`                 |
//! | `mul(c, u)`   | `divide(acc, c)`   | `log(abs(c))`       |
//! | `divide(u, c)`| `mul(acc, c)`      | `neg(log(abs(c)))`  |
//! | `add(c, u)`   | `sub(acc, c)`      | `0`                 |
//! | `sub(u, c)`   | `add(acc, c)`      | `0`                 |
//! | `sub(c, u)`   | `sub(c, acc)`      | `0`  (g'(z) = −1)   |
//!
//! Closed-form checks: `x -> 2·x + 1` ⇒ `f_inv = (y−1)/2`, `f'(x) = 2`, `logvol =
//! log 2`. `x -> exp(2·x)` ⇒ `f_inv = log(y)/2`, `f'(x) = 2·e^{2x}`, `logvol =
//! 2x + log 2` (the `2x` is `exp`'s partial-forward point).
//!
//! ## Matrix-affine (vector variate) — the MvNormal construction
//!
//! Over a VECTOR variate, a forward body `mu + L * x` (plain `add`) or `mu .+ L *
//! x` (`broadcast(add, …)`) is a matrix-vector affine map (spec §06 case 1
//! mandates recognising `mu + lower_cholesky(cov) * _`; §08 `MvNormal(mu, cov) ≡
//! pushfwd(fn(mu + lower_cholesky(cov) * _), iid(Normal(0,1), n))`). We synthesise
//! * **`f_inv(y) = linsolve(L, y − mu)`** — solve `L x = y − mu` for `x =
//!   L⁻¹(y − mu)` (§07 `linsolve`), and
//! * **`logvol = logabsdet(L)`** — the forward log-volume `log|det L|`, CONSTANT
//!   in `x` (§07 `logabsdet`), emitted as an argument-ignoring lambda.
//!
//! The map is refused (Err) when `mu` or `L` references the input (a
//! coupled/nonlinear map — Jacobian ≠ constant `L`) or `L` is confirmed
//! non-square; a vector-variate body that is not this shape is also refused
//! (not fallen through to the scalar chain, whose per-op log-volume would not be
//! summed over the vector axes). See [`derive_matrix_affine`] for the MvNormal
//! change-of-variables cross-check.
//!
//! Refuse-don't-mislower: an UNRECOGNISED forward function returns `Ok(None)`
//! (the caller refuses); a RECOGNISED-but-non-invertible shape returns `Err`
//! (refuse) — a non-affine `mul`/`add`/`sub` (both operands non-literal, e.g.
//! `x*x`, or both literal), a `mul`/`divide` whose literal coefficient is
//! ZERO (`0.0 * u` collapses to the constant 0; `u / 0.0` is undefined), a
//! `divide` without a literal denominator, a `pow` inside a composition (its
//! input domain is not verifiable here), or any other recognised builtin op.
//! A wrong `(f_inv, logvol)` is never synthesised.
//!
//! Single-op `pow(_, k)` (`x -> pow(x, k)`) keeps its Task-1 domain-restricted
//! derivation ([`derive_pow`]); a bare builtin value (`pushfwd(exp, M)`) keeps
//! its Task-1 single-op form ([`bare_bijection`], byte-equality-pinned against the
//! explicit `bijection(exp, log, x -> x)`).

use crate::density::{build_call, fold_add, refuse, resolve_ref_one};
use crate::refuse::RefuseError;
use flatppl_core::{
    Call, CallHead, Dim, Inputs, Module, NamedArg, Node, NodeId, Ref, RefNs, Scalar, Symbol, Type,
    ValueSet,
};

/// A synthesised change-of-variables: the inverse map `f_inv` and the FORWARD
/// log-volume element `logvol`, each a single-input FlatPIR callable the caller
/// applies via `build_user_call`.
pub(crate) struct Bijection {
    pub f_inv: NodeId,
    pub logvol: NodeId,
}

/// One op `gᵢ` in a scalar chain, carrying what its inverse and local logvol
/// need: for the two non-constant-logvol unary ops (`exp`/`log`) the PARTIAL-
/// FORWARD sub-expression node (`z`), for the affine ops the literal operand `c`.
enum ChainOp {
    /// `exp(z)`: inverse `log(acc)`; local logvol `z` (the partial-forward node).
    Exp(NodeId),
    /// `log(z)`: inverse `exp(acc)`; local logvol `neg(log(z))`.
    Log(NodeId),
    /// `neg(z)`: inverse `neg(acc)`; local logvol `0`.
    Neg,
    /// `c·z`: inverse `divide(acc, c)`; local logvol `log(abs(c))`.
    MulByLit(NodeId),
    /// `z/c`: inverse `mul(acc, c)`; local logvol `neg(log(abs(c)))`.
    DivByLit(NodeId),
    /// `z+c`: inverse `sub(acc, c)`; local logvol `0`.
    AddLit(NodeId),
    /// `z−c`: inverse `add(acc, c)`; local logvol `0`.
    SubLit(NodeId),
    /// `c−z`: inverse `sub(c, acc)`; local logvol `0` (derivative −1, log|−1| = 0).
    RSubLit(NodeId),
}

/// The recognised surface shape of a `pushfwd`'s forward argument.
enum Recognized {
    /// A bare builtin value used as a function (`pushfwd(exp, M)`).
    BareConst(String),
    /// A one-input `functionof` lambda `x -> body` (chain / affine / single op).
    Lambda {
        body: NodeId,
        input_name: Symbol,
        ph: Symbol,
    },
    /// Anything else — not a recognised forward function.
    Unrecognized,
}

/// Derive `(f_inv, logvol)` for the forward function `f` of a `pushfwd` over a
/// base measure whose variate domain is `domain`.
///
/// * `Ok(Some(_))` — `f` is a recognised, invertible forward map (bare builtin,
///   single-op `pow`, or a scalar chain of unary/affine ops); the derived
///   change-of-variables is returned.
/// * `Ok(None)` — `f` is not a recognised forward function (the caller refuses).
/// * `Err(_)` — `f` is recognised but not invertible here (refuse).
pub(crate) fn derive_bijection(
    m: &mut Module,
    f: NodeId,
    domain: &Type,
) -> Result<Option<Bijection>, RefuseError> {
    // Resolve one level of self-ref (`pushfwd(g, M)` where `g = exp`).
    let (f_resolved, _) = resolve_ref_one(m, f);
    match recognise(m, f_resolved) {
        // Bare builtin value: Task-1 single-op form (byte-equality-pinned).
        Recognized::BareConst(name) => Ok(bare_bijection(m, &name)),
        Recognized::Lambda {
            body,
            input_name,
            ph,
        } => {
            // Matrix-vector affine map `mu + L * x` over a VECTOR variate (§06
            // case 1: the engine MUST recognise maps such as
            // `mu + lower_cholesky(cov) * _`). Keyed on the base measure's variate
            // domain being a vector (1-D array) — this is the MvNormal construction
            // (§08 MvNormal), distinct from the scalar chain below.
            if domain_is_vector(domain) {
                return match derive_matrix_affine(m, body, ph)? {
                    Some(bij) => Ok(Some(bij)),
                    // A vector-variate forward body that is not a recognised
                    // matrix-affine map: refuse rather than fall through to the
                    // scalar chain, whose per-op log-volume is not summed over the
                    // vector's axes and would silently mislower (a scalar-scale
                    // `k·x` over a vector has log-volume `n·log|k|`, not `log|k|`).
                    None => Err(refuse(
                        f,
                        m,
                        "forward map over a vector variate is not a recognised matrix-affine \
                         map (mu + L * x) — refuse rather than mislower",
                    )),
                };
            }
            // Single-op `pow(x, k)` keeps its Task-1 domain-restricted derivation;
            // a `pow` anywhere else in a chain is refused by the chain walk (its
            // input domain is not verifiable here).
            if let Some(k_node) = single_pow(m, body, ph) {
                return derive_pow(m, f, k_node, domain);
            }
            derive_chain(m, body, input_name, ph)
        }
        Recognized::Unrecognized => Ok(None),
    }
}

/// The Task-1 single-builtin bijections for a bare builtin value
/// (`pushfwd(exp, M)`): the `f_inv`/`logvol` forms whose byte-equality against
/// `bijection(exp, log, x -> x)` pins the forward-log-volume convention. Any
/// other bare builtin (including bare `pow`, which needs an exponent) is not a
/// recognised bare bijection → `None`.
fn bare_bijection(m: &mut Module, name: &str) -> Option<Bijection> {
    match name {
        // d/dx eˣ = eˣ ⇒ log|f'| = x (identity).
        "exp" => Some(Bijection {
            f_inv: bare_builtin(m, "log"),
            logvol: identity_lambda(m),
        }),
        // d/dx ln x = 1/x ⇒ log|f'| = −ln x.
        "log" => {
            let logvol = lambda(m, |m, ph| {
                let logx = build_call(m, "log", &[ph]);
                build_call(m, "neg", &[logx])
            });
            Some(Bijection {
                f_inv: bare_builtin(m, "exp"),
                logvol,
            })
        }
        // f'(x) = −1 ⇒ log|f'| = 0.
        "neg" => {
            let logvol = lambda(m, |m, _ph| m.alloc(Node::Lit(Scalar::Real(0.0))));
            Some(Bijection {
                f_inv: bare_builtin(m, "neg"),
                logvol,
            })
        }
        _ => None,
    }
}

/// Derive the change-of-variables for a scalar-chain forward body `f = gₙ∘…∘g₁`
/// (`input_name`/`ph` are the forward lambda's boundary — reused verbatim on the
/// `logvol` so the partial-forward sub-expressions, which reference `ph`, resolve
/// inside it). See the module docs for the inverse / chain-rule construction.
///
/// * `Ok(Some(_))` — every op in the chain is invertible.
/// * `Ok(None)` — the chain hit an unrecognised shape (a non-builtin head, or a
///   leaf that is not the input placeholder).
/// * `Err(_)` — the chain hit a recognised-but-non-invertible op (refuse).
fn derive_chain(
    m: &mut Module,
    body: NodeId,
    input_name: Symbol,
    ph: Symbol,
) -> Result<Option<Bijection>, RefuseError> {
    let Some(ops) = flatten_chain(m, body, ph)? else {
        return Ok(None);
    };

    // f_inv(y) = g₁_inv(…(gₙ_inv(y))…): thread the per-op inverses through a fresh
    // placeholder, outermost-first (the chain is stored outermost-first).
    let f_inv = lambda(m, |m, y| {
        let mut acc = y;
        for op in &ops {
            acc = apply_inverse(m, op, acc);
        }
        acc
    });

    // logvol(x) = Σᵢ logvolᵢ(partial-forward point). Drop the zero contributions
    // (neg / add / sub); an all-zero sum is the constant 0.
    let mut terms = Vec::new();
    for op in &ops {
        if let Some(term) = local_logvol(m, op) {
            terms.push(term);
        }
    }
    let logvol_body = if terms.is_empty() {
        m.alloc(Node::Lit(Scalar::Real(0.0)))
    } else {
        fold_add(m, &terms)
    };
    // Reuse the forward lambda's own input name + placeholder so the reused
    // partial-forward sub-expressions (which reference `ph`) resolve here.
    let logvol = wrap_functionof(m, input_name, ph, logvol_body);

    Ok(Some(Bijection { f_inv, logvol }))
}

/// Flatten the linear forward chain rooted at `body` into its ops, OUTERMOST-
/// FIRST, walking down each op's single sub-expression until the input `ph`.
///
/// * `Ok(Some(ops))` — reached `ph`; every intermediate op was invertible.
/// * `Ok(None)` — hit an unrecognised shape (a non-builtin head or a non-`ph`
///   leaf): the whole forward function is not recognised.
/// * `Err(_)` — hit a recognised-but-non-invertible op (refuse).
fn flatten_chain(
    m: &Module,
    body: NodeId,
    ph: Symbol,
) -> Result<Option<Vec<ChainOp>>, RefuseError> {
    let mut ops = Vec::new();
    let mut cur = body;
    // The forward body is a finite tree; each step descends to a strict subterm.
    loop {
        if is_placeholder_ref(m, cur, ph) {
            return Ok(Some(ops));
        }
        match classify(m, cur)? {
            Some((op, child)) => {
                ops.push(op);
                cur = child;
            }
            None => return Ok(None),
        }
    }
}

/// Classify the single op at `cur`: `Ok(Some((op, child)))` for an invertible
/// unary/affine op (with its sub-expression `child` to descend into), `Ok(None)`
/// for an unrecognised head (a user-function call, or a non-call leaf that is not
/// the placeholder), `Err` for a recognised builtin with no analytic inverse
/// here (refuse-don't-mislower).
fn classify(m: &Module, cur: NodeId) -> Result<Option<(ChainOp, NodeId)>, RefuseError> {
    let (name, args) = match m.node(cur) {
        Node::Call(c) => match c.head {
            CallHead::Builtin(sym) => (m.resolve(sym).to_string(), c.args.to_vec()),
            // A user-function application is not a recognised builtin forward op.
            CallHead::User(_) => return Ok(None),
        },
        // A non-call leaf that is not the placeholder: the chain does not
        // terminate at the input, so this is not a recognised forward function.
        _ => return Ok(None),
    };
    match name.as_str() {
        "exp" if args.len() == 1 => Ok(Some((ChainOp::Exp(args[0]), args[0]))),
        "log" if args.len() == 1 => Ok(Some((ChainOp::Log(args[0]), args[0]))),
        "neg" if args.len() == 1 => Ok(Some((ChainOp::Neg, args[0]))),
        // Affine multiply: exactly one literal operand (the scale `c`), and that
        // literal must be nonzero — `0.0 * u` collapses the forward map to the
        // constant 0, which is not injective (refuse rather than synthesize a
        // degenerate `f_inv = divide(acc, 0.0)`).
        "mul" if args.len() == 2 => {
            match (is_nonzero_lit(m, args[0]), is_nonzero_lit(m, args[1])) {
                (true, false) => Ok(Some((ChainOp::MulByLit(args[0]), args[1]))),
                (false, true) => Ok(Some((ChainOp::MulByLit(args[1]), args[0]))),
                _ => Err(refuse(
                    cur,
                    m,
                    "mul with two non-literal (or two literal, or a literal-zero scale) \
                     operands is not an invertible affine map — refuse rather than mislower",
                )),
            }
        }
        // Affine divide: only `u / c` (literal denominator) is affine; `c / u`
        // (reciprocal) is out of the grammar. The literal denominator must also
        // be nonzero — `u / 0.0` is undefined everywhere (refuse rather than
        // synthesize a degenerate `f_inv = mul(acc, 0.0)`).
        "divide" if args.len() == 2 => match (is_lit(m, args[0]), is_nonzero_lit(m, args[1])) {
            (false, true) => Ok(Some((ChainOp::DivByLit(args[1]), args[0]))),
            _ => Err(refuse(
                cur,
                m,
                "divide is an invertible affine map only with a literal, nonzero denominator \
                 (u / c) — refuse rather than mislower",
            )),
        },
        // Affine add: exactly one literal operand (the shift `c`).
        "add" if args.len() == 2 => match (is_lit(m, args[0]), is_lit(m, args[1])) {
            (true, false) => Ok(Some((ChainOp::AddLit(args[0]), args[1]))),
            (false, true) => Ok(Some((ChainOp::AddLit(args[1]), args[0]))),
            _ => Err(refuse(
                cur,
                m,
                "add with two non-literal (or two literal) operands is not an invertible \
                 affine map — refuse rather than mislower",
            )),
        },
        // Affine subtract: `u − c` (shift) or `c − u` (reflect+shift).
        "sub" if args.len() == 2 => match (is_lit(m, args[0]), is_lit(m, args[1])) {
            (false, true) => Ok(Some((ChainOp::SubLit(args[1]), args[0]))),
            (true, false) => Ok(Some((ChainOp::RSubLit(args[0]), args[1]))),
            _ => Err(refuse(
                cur,
                m,
                "sub with two non-literal (or two literal) operands is not an invertible \
                 affine map — refuse rather than mislower",
            )),
        },
        // `pow` is invertible only as the single top-level op over a strictly-
        // positive base domain ([`derive_pow`], handled before the chain walk); a
        // `pow` reached inside a composition has an unverifiable input domain.
        "pow" => Err(refuse(
            cur,
            m,
            "pow inside a composition is not an invertible shape here (its input domain is \
             not verifiable) — refuse rather than mislower",
        )),
        // A recognised builtin with no analytic inverse in this grammar.
        _ => Err(refuse(
            cur,
            m,
            "forward op is a recognised builtin with no analytic inverse — refuse rather \
             than mislower",
        )),
    }
}

/// Apply `op`'s per-op inverse to the accumulating argument `acc` (see the module
/// per-op table).
fn apply_inverse(m: &mut Module, op: &ChainOp, acc: NodeId) -> NodeId {
    match op {
        ChainOp::Exp(_) => build_call(m, "log", &[acc]),
        ChainOp::Log(_) => build_call(m, "exp", &[acc]),
        ChainOp::Neg => build_call(m, "neg", &[acc]),
        ChainOp::MulByLit(c) => build_call(m, "divide", &[acc, *c]),
        ChainOp::DivByLit(c) => build_call(m, "mul", &[acc, *c]),
        ChainOp::AddLit(c) => build_call(m, "sub", &[acc, *c]),
        ChainOp::SubLit(c) => build_call(m, "add", &[acc, *c]),
        ChainOp::RSubLit(c) => build_call(m, "sub", &[*c, acc]),
    }
}

/// `op`'s LOCAL forward log-derivative at its partial-forward input, or `None`
/// when it is identically zero (`neg` / affine shift). The non-constant terms
/// (`exp`/`log`) reuse the partial-forward sub-expression node directly — it is
/// already the forward composition of the inner ops, expressed in the input
/// placeholder, which is exactly the point `gᵢ`'s derivative is evaluated at.
fn local_logvol(m: &mut Module, op: &ChainOp) -> Option<NodeId> {
    match op {
        // log|d/dz eᶻ| = z, evaluated at the partial-forward point (= the node).
        ChainOp::Exp(z) => Some(*z),
        // log|d/dz ln z| = −log z.
        ChainOp::Log(z) => {
            let logz = build_call(m, "log", &[*z]);
            Some(build_call(m, "neg", &[logz]))
        }
        // log|d/dz (c·z)| = log|c|.
        ChainOp::MulByLit(c) => {
            let absc = build_call(m, "abs", &[*c]);
            Some(build_call(m, "log", &[absc]))
        }
        // log|d/dz (z/c)| = −log|c|.
        ChainOp::DivByLit(c) => {
            let absc = build_call(m, "abs", &[*c]);
            let logabs = build_call(m, "log", &[absc]);
            Some(build_call(m, "neg", &[logabs]))
        }
        // Derivative ±1 ⇒ log|g'| = 0: contributes nothing to the sum.
        ChainOp::Neg | ChainOp::AddLit(_) | ChainOp::SubLit(_) | ChainOp::RSubLit(_) => None,
    }
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

/// Derive `(f_inv, logvol)` for a matrix-vector affine forward body
/// `mu + L * x` (plain `add`) or `mu .+ L * x` (`broadcast(add, …)`) over a
/// VECTOR variate — the MvNormal construction (§06 case 1; §08
/// `MvNormal(mu, cov)` ≡ `pushfwd(fn(mu + lower_cholesky(cov) * _), iid(Normal(0,1), n))`).
///
/// * **`f_inv(y) = linsolve(L, y − mu)`** — solve `L x = y − mu` for the preimage
///   `x = L⁻¹(y − mu)` (spec §07 `linsolve`: square `A`, vector `b`; `inv(L)` is
///   avoided in favour of the direct solve).
/// * **`logvol = logabsdet(L)`** — the FORWARD log-volume `log|det J_f| =
///   log|det L|`, CONSTANT in `x` (a linear map has constant Jacobian `L`; spec
///   §07 `logabsdet(A) = log|det A|`, square matrix → real scalar). Emitted as a
///   lambda that IGNORES its argument, consistent with Tasks 1-2's logvol shape;
///   the caller applies it at the preimage (`logvol(f_inv(v))`), which β-reduces
///   to the constant.
///
/// MvNormal cross-check (Σ = L Lᵀ): the caller emits `logdensityof(iid N(0,1),
/// f_inv(v)) − logvol(f_inv(v))` (§06 line 457) =
/// `−n/2·log 2π − ½‖L⁻¹(v−mu)‖² − log|det L|`. With `‖L⁻¹u‖² = uᵀ(LLᵀ)⁻¹u =
/// uᵀΣ⁻¹u` and `log|det L| = ½·log|det Σ|`, this is exactly
/// `log N(v; mu, Σ)` — the standard-normal inner density plus `−logabsdet(L)`
/// reproduces both the quadratic form AND the `−½log|det Σ|` normaliser. A
/// wrong/absent log-det would be a silently wrong density; `logabsdet(L)`
/// (neither doubled nor halved) is the correct forward log-volume.
///
/// * `Ok(Some(_))` — a recognised, invertible matrix-affine map.
/// * `Ok(None)` — `body` is not an `add`/`broadcast(add, …)` of a shift and a
///   `mul(L, x)` (the caller refuses).
/// * `Err(_)` — recognised-but-non-invertible (refuse): the shift `mu` or the
///   matrix `L` REFERENCES the input placeholder (a coupled/nonlinear map whose
///   Jacobian is not the constant `L`), or `L` is a CONFIRMED non-square matrix
///   (`linsolve`/`logabsdet` need a square matrix).
fn derive_matrix_affine(
    m: &mut Module,
    body: NodeId,
    ph: Symbol,
) -> Result<Option<Bijection>, RefuseError> {
    // All structural reads (immutable) BEFORE the mutable f_inv/logvol builds.
    let Some((a, b)) = affine_add_operands(m, body) else {
        return Ok(None);
    };
    // Identify the linear term `mul(L, x)` (matrix first, placeholder second) and
    // take the OTHER summand as the shift `mu`.
    let (mu, l) = if let Some(l) = matrix_times_ph(m, b, ph) {
        (a, l)
    } else if let Some(l) = matrix_times_ph(m, a, ph) {
        (b, l)
    } else {
        return Ok(None);
    };
    // Coupled/nonlinear guard: a fixed matrix-affine map has `mu` and `L`
    // independent of the input. If either references the placeholder, the
    // forward Jacobian is not the constant `L` — refuse rather than emit a
    // wrong `logabsdet(L)`.
    if refs_placeholder(m, mu, ph) || refs_placeholder(m, l, ph) {
        return Err(refuse(
            body,
            m,
            "coupled/nonlinear multivariate forward map (the shift or matrix depends on the \
             input) is not a fixed matrix-affine map — refuse rather than mislower",
        ));
    }
    // Non-square guard: `linsolve`/`logabsdet` require a square `L` (§07). Only a
    // CONFIRMED non-square matrix refuses; unknown/dynamic dims are the standard
    // (square-by-construction) MvNormal factor and are not over-refused.
    if matrix_confirmed_non_square(m, l) {
        return Err(refuse(
            body,
            m,
            "matrix factor L is not square (linsolve/logabsdet need a square matrix) — \
             refuse rather than mislower",
        ));
    }
    // f_inv(y) = linsolve(L, sub(y, mu)) — solve L x = y − mu.
    let f_inv = lambda(m, |m, y| {
        let diff = build_call(m, "sub", &[y, mu]);
        build_call(m, "linsolve", &[l, diff])
    });
    // logvol(_) = logabsdet(L) — constant; the argument is ignored.
    let logvol = lambda(m, |m, _y| build_call(m, "logabsdet", &[l]));
    Ok(Some(Bijection { f_inv, logvol }))
}

/// The two summands of a plain `add(x, y)` or a `broadcast(add, x, y)` forward
/// body (the two pinned matrix-affine outer forms); `None` for any other head.
/// A `broadcast`'s first arg is the operator constant `(%const add)`.
fn affine_add_operands(m: &Module, body: NodeId) -> Option<(NodeId, NodeId)> {
    let Node::Call(c) = m.node(body) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    match m.resolve(sym) {
        "add" if c.args.len() == 2 => Some((c.args[0], c.args[1])),
        "broadcast" if c.args.len() == 3 && is_const_named(m, c.args[0], "add") => {
            Some((c.args[1], c.args[2]))
        }
        _ => None,
    }
}

/// If `id` is `mul(L, x)` — a matrix-vector product whose SECOND operand is the
/// input placeholder `x` — return the matrix operand `L`; otherwise `None`. (The
/// pinned forward product is `L * x` = `mul(L, ph)`, matrix first.)
fn matrix_times_ph(m: &Module, id: NodeId, ph: Symbol) -> Option<NodeId> {
    let Node::Call(c) = m.node(id) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    if m.resolve(sym) != "mul" || c.args.len() != 2 {
        return None;
    }
    is_placeholder_ref(m, c.args[1], ph).then_some(c.args[0])
}

/// Is `id` the bare builtin-operator constant `(%const <name>)` (e.g. the `add`
/// operator passed as `broadcast`'s first argument)?
fn is_const_named(m: &Module, id: NodeId, name: &str) -> bool {
    matches!(m.node(id), Node::Const(sym) if m.resolve(*sym) == name)
}

/// Does the subtree at `id` reference the input placeholder `(%ref %local ph)`
/// anywhere? A shift `mu` or matrix `L` that does is input-dependent — the map
/// is coupled/nonlinear, not a fixed matrix-affine map.
fn refs_placeholder(m: &Module, id: NodeId, ph: Symbol) -> bool {
    let mut stack = vec![id];
    while let Some(cur) = stack.pop() {
        if is_placeholder_ref(m, cur, ph) {
            return true;
        }
        m.for_each_child(cur, |c| stack.push(c));
    }
    false
}

/// Is `l`'s inferred type a matrix (2-D array) with CONFIRMED unequal static
/// row/column counts? Such an `L` is not invertible. A matrix with
/// dynamic/unknown dims, or an unresolved type, is NOT confirmed non-square (the
/// standard MvNormal factor is square by construction) and is not over-refused.
fn matrix_confirmed_non_square(m: &Module, l: NodeId) -> bool {
    if let Some(Type::Array { shape, .. }) = m.type_of(l) {
        if shape.len() == 2 {
            if let (Dim::Static(rows), Dim::Static(cols)) = (shape[0], shape[1]) {
                return rows != cols;
            }
        }
    }
    false
}

/// Is the base measure's variate domain a VECTOR — a 1-D array? The matrix-
/// affine arm applies only over a vector variate (`mu + L * x`); a scalar domain
/// takes the scalar-chain path, and a higher-rank array is not a recognised
/// matrix-affine variate here.
fn domain_is_vector(domain: &Type) -> bool {
    matches!(domain, Type::Array { shape, .. } if shape.len() == 1)
}

/// Recognise the surface shape of a `pushfwd`'s (ref-resolved) forward argument:
/// a bare builtin value (`Const`), or a one-input `functionof` lambda `x -> body`
/// whose boundary is exactly one `%local` placeholder.
fn recognise(m: &Module, f: NodeId) -> Recognized {
    match m.node(f) {
        Node::Const(sym) => Recognized::BareConst(m.resolve(*sym).to_string()),
        Node::Call(c) => {
            if let CallHead::Builtin(sym) = c.head {
                if m.resolve(sym) == "functionof" && c.args.len() == 1 {
                    if let Some(Inputs::Spec(entries)) = &c.inputs {
                        if entries.len() == 1 && entries[0].1.ns == RefNs::Local {
                            return Recognized::Lambda {
                                body: c.args[0],
                                input_name: entries[0].0,
                                ph: entries[0].1.name,
                            };
                        }
                    }
                }
            }
            Recognized::Unrecognized
        }
        _ => Recognized::Unrecognized,
    }
}

/// If `body` is exactly `pow(<ph>, k)` — a single top-level `pow` applied to the
/// input placeholder — return its exponent node `k`; otherwise `None`.
fn single_pow(m: &Module, body: NodeId, ph: Symbol) -> Option<NodeId> {
    let Node::Call(c) = m.node(body) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    if m.resolve(sym) != "pow" || c.args.len() != 2 {
        return None;
    }
    if !is_placeholder_ref(m, c.args[0], ph) {
        return None;
    }
    Some(c.args[1])
}

/// Is `id` the placeholder ref `(%ref %local <ph>)`?
fn is_placeholder_ref(m: &Module, id: NodeId, ph: Symbol) -> bool {
    matches!(m.node(id), Node::Ref(Ref { ns: RefNs::Local, name }) if *name == ph)
}

/// Is `id` a numeric literal (an affine-operand `c`)?
fn is_lit(m: &Module, id: NodeId) -> bool {
    literal_real(m, id).is_some()
}

/// Is `id` a numeric literal that is also nonzero (an affine `mul`/`divide`
/// coefficient `c`)? `c != 0.0` also rejects `-0.0` — in Rust `f64`,
/// `-0.0 == 0.0`, so a literal-zero-with-negative-sign is caught too. A
/// literal-zero scale/divisor is not a Task-1 recognised invertible affine
/// map: `mul(0.0, u)` collapses to the constant 0 (not injective) and
/// `divide(u, 0.0)` is undefined everywhere.
fn is_nonzero_lit(m: &Module, id: NodeId) -> bool {
    literal_real(m, id).is_some_and(|c| c != 0.0)
}

/// A bare builtin symbol node (`exp` / `log` / `neg`) usable directly as `f_inv`.
fn bare_builtin(m: &mut Module, name: &str) -> NodeId {
    let sym = m.intern(name);
    m.alloc(Node::Const(sym))
}

/// Build a `functionof` lambda `<input_name> -> <body>` with the given boundary
/// (input name + `%local` placeholder symbol).
fn wrap_functionof(m: &mut Module, input_name: Symbol, ph: Symbol, body: NodeId) -> NodeId {
    let functionof = m.intern("functionof");
    m.alloc(Node::Call(Call {
        head: CallHead::Builtin(functionof),
        args: vec![body].into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: Some(Inputs::Spec(
            vec![(
                input_name,
                Ref {
                    ns: RefNs::Local,
                    name: ph,
                },
            )]
            .into(),
        )),
    }))
}

/// Build a single-input `functionof` lambda `x -> <body>`, spelled exactly as the
/// parser emits `x -> …` (input name `x`, placeholder `_x_`). `body(m, ph)`
/// receives the placeholder node id.
fn lambda(m: &mut Module, body: impl FnOnce(&mut Module, NodeId) -> NodeId) -> NodeId {
    let x = m.intern("x");
    let ph = m.intern("_x_");
    let ph_node = m.alloc(Node::Ref(Ref {
        ns: RefNs::Local,
        name: ph,
    }));
    let body_node = body(m, ph_node);
    wrap_functionof(m, x, ph, body_node)
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
