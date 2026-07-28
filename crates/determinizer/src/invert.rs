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
//! ## One registry, two spellings
//!
//! §06 case 1 nowhere distinguishes `pushfwd(g, M)` from `pushfwd(x -> g(x), M)`
//! — `bijection`'s own entry describes its annotated result as "a function that is
//! semantically `f`" — so the spelling must not change the outcome. [`REGISTRY`]
//! is the single table of §06's named unary bijections, and it is reached through
//! ONE lookup ([`unary_entry`]) from both entry points: [`bare_bijection`] for a
//! bare builtin value, [`classify`] for an op inside a lambda body. Each entry
//! carries its inverse, its FORWARD log-volume `log|g'|`, its §06 domain
//! restriction and its IMAGE, with the two emissions kept as BUILDERS parameterised
//! by the point the derivative is taken at — which is what lets one table serve
//! both: a chain needs `log|g'|` at a node it already holds, not a callable to
//! apply. The image is read by [`forward_image`] for the density path's −∞ gate on
//! the query point, from the same table and off the forward map alone, so no
//! spelling of `pushfwd` gates differently from another.
//!
//! ## Scalar-chain inversion
//!
//! The forward body is a linear chain of ops `f = gₙ∘…∘g₁` terminating at the
//! single input placeholder — each op either a [`REGISTRY`] unary or an AFFINE
//! node with ONE literal operand `c` and ONE sub-expression `u`
//! (`mul`/`divide`/`add`/`sub`). We invert it by:
//!
//! * **`f_inv(y) = g₁_inv(…(gₙ_inv(y))…)`** — apply the per-op inverses
//!   outermost-first to a fresh placeholder (undo `gₙ` first, `g₁` last).
//! * **`logvol(x) = Σᵢ logvolᵢ(gᵢ₋₁∘…∘g₁(x))`** — the chain rule
//!   `log|f'| = Σᵢ log|gᵢ'|`, with each op's LOCAL forward log-derivative
//!   evaluated at its PARTIAL-FORWARD input `gᵢ₋₁∘…∘g₁(x)`. That partial-forward
//!   point is exactly `gᵢ`'s own sub-expression node in the forward body (already
//!   an expression in the input placeholder), so we reuse it directly rather than
//!   re-deriving the composition. A registry op's term is its entry's log-volume
//!   built at that node (`exp`: `z`; `log`: `−log z`; `tanh`: `log(1 − tanh(z)²)`);
//!   the affine ops contribute a constant (`mul`: `log|c|`; `divide`: `−log|c|`)
//!   or zero (`add`/`sub`, and a volume-preserving registry op such as `neg`), so
//!   zero terms are dropped and an all-zero sum collapses to the literal `0`.
//!
//! Affine per-op table (`acc` = the accumulating inverse argument, `z` = the op's
//! partial-forward input; local logvol = FORWARD `log|gᵢ'|` at `z`; the unary ops
//! are [`REGISTRY`]'s two columns):
//! | op            | inverse of `acc`   | local logvol        |
//! |---------------|--------------------|---------------------|
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
//! A domain-restricted registry op (`log`, `log10`, `log1p`, `logit`, `probit`,
//! `sqrt`) is admitted in a chain ONLY where its input IS the base variate —
//! innermost — and there the base measure's support decides (§06 case 1). An
//! interior one refuses: the support bounds only the innermost input, and proving
//! `2·x > 0` from `x > 0`, let alone `log x > 0`, needs interval propagation
//! through the chain. This over-refuses maps that are in fact well-defined
//! (`x -> log(2·x)` over a positive base), which is the direction §06 sanctions —
//! "refused rather than yielding a silently sub-probability measure".
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
//! ## Multivariate elementwise (vector variate) — diagonal Jacobian
//!
//! Over a VECTOR variate, a forward body `broadcast(g, x)` (a single scalar-
//! invertible `g` applied to every cell of `x`; spec §06 case 1, the user-requested
//! elementwise extension) has a DIAGONAL Jacobian `diag(g'(x₁), …, g'(xₙ))`, so its
//! log-det is the SUM of the per-cell scalar forward log-derivatives. We derive
//! `(g_inv, g_logvol)` by RECURSING [`derive_bijection`] on the scalar `g` over the
//! vector's ELEMENT domain (`g` then takes the bare-builtin / scalar-chain path —
//! a scalar domain is not a vector, so the recursion never re-enters this arm) and
//! wrap:
//! * **`f_inv(y) = broadcast(g_inv, y)`** — apply the scalar inverse cell-wise;
//! * **`logvol(x) = sum(broadcast(g_logvol, x))`** — `Σᵢ log|g'(xᵢ)|`, the diagonal
//!   log-det (§07 `sum` reduces a real vector to a scalar).
//!
//! A COUPLED broadcast mixing two or more variate slots (`broadcast(add, x, x)`,
//! `broadcast(mul, x, x)`) is refused (Err) — its Jacobian is not diagonal in the
//! single-variate sense; a non-`broadcast(g, x)` vector body returns `Ok(None)`
//! (the caller refuses). See [`derive_elementwise`] for the LogNormal-vector
//! cross-check.
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
//! Single-op `pow(_, k)` (`x -> pow(x, k)`) takes its own domain-restricted
//! derivation ([`derive_pow`]) since the exponent comes from the call rather than
//! the op name; a `pow` reached inside a composition is refused. `sqrt` is the same
//! derivation at a fixed `k = 1/2` ([`pow_inverse`]/[`pow_logvol`]), reached
//! through the registry like any other named op. A bare builtin value
//! (`pushfwd(exp, M)`) is byte-equality-pinned against the explicit
//! `bijection(exp, log, x -> x)`.

use crate::density::{build_call, expect_builtin_call, fold_add, refuse, resolve_ref_one};
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
/// need: for a [`REGISTRY`] unary the PARTIAL-FORWARD sub-expression node (`z`),
/// for the affine ops the literal operand `c`.
enum ChainOp {
    /// A registry unary `g(z)`: inverse and local forward log-volume both come
    /// from the shared [`REGISTRY`] entry, built at `z` — the op's PARTIAL-FORWARD
    /// input, already an expression in the chain's input placeholder.
    Registry {
        op: &'static str,
        entry: &'static UnaryEntry,
        z: NodeId,
    },
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
    support: &ValueSet,
) -> Result<Option<Bijection>, RefuseError> {
    // Resolve one level of self-ref (`pushfwd(g, M)` where `g = exp`).
    let (f_resolved, _) = resolve_ref_one(m, f);
    // The VECTOR variate dispatch comes BEFORE the bare/lambda split, because the
    // bare spelling of an elementwise map is still an elementwise map (§06 nowhere
    // distinguishes `pushfwd(g, M)` from `pushfwd(x -> broadcast(g, x), M)` over a
    // vector base) and its Jacobian is DIAGONAL. Reached after the split, a bare
    // operator took the SCALAR derivation and emitted that op's scalar log-volume
    // against an n-vector — `sub(<scalar>, log(y))` with a vector `log(y)` — where
    // the log-det is `Σᵢ log|g'(yᵢ)|`.
    if domain_is_vector(domain) {
        return derive_vector_bijection(m, f, f_resolved, domain, support);
    }
    match recognise(m, f_resolved) {
        // Bare builtin value: Task-1 single-op form (byte-equality-pinned).
        Recognized::BareConst(name) => bare_bijection(m, &name, f, support),
        Recognized::Lambda {
            body,
            input_name,
            ph,
        } => {
            // Single-op `pow(x, k)` keeps its Task-1 domain-restricted derivation;
            // a `pow` anywhere else in a chain is refused by the chain walk (its
            // input domain is not verifiable here).
            if let Some(k_node) = single_pow(m, body, ph) {
                return derive_pow(m, f, k_node, support);
            }
            derive_chain(m, body, input_name, ph, support)
        }
        Recognized::Unrecognized => Ok(None),
    }
}

/// Derive `(f_inv, logvol)` for a forward map over a VECTOR variate — the two
/// vector forms §06 case 1 mandates, and nothing else:
///
/// * a matrix-vector affine map `mu + L * x` ([`derive_matrix_affine`], the
///   MvNormal construction, §08 `MvNormal`);
/// * an ELEMENTWISE unary map, whose diagonal Jacobian makes `logvol` the SUM of
///   the per-cell scalar forward log-derivatives. Both its spellings land here: the
///   bare operator (`pushfwd(exp, MvNormal)`) derives the per-cell map over the
///   vector's ELEMENT domain and support, and `broadcast(g, x)` recurses on `g` the
///   same way ([`derive_elementwise`]); both then go through [`wrap_elementwise`],
///   so the two emit the identical term.
///
/// Anything else refuses rather than falling through to the scalar chain, whose
/// per-op log-volume is not summed over the vector's axes and would silently
/// mislower (a scalar scale `k·x` over an n-vector has log-volume `n·log|k|`, not
/// `log|k|`).
fn derive_vector_bijection(
    m: &mut Module,
    f: NodeId,
    f_resolved: NodeId,
    domain: &Type,
    support: &ValueSet,
) -> Result<Option<Bijection>, RefuseError> {
    let elem_domain = vector_elem_domain(domain);
    let elem_sup = elem_support(support);
    match recognise(m, f_resolved) {
        // The bare operator IS the per-cell map, so the §06 domain restriction is
        // checked against the ELEMENT support — the same support the `broadcast`
        // spelling's recursion reads.
        Recognized::BareConst(name) => match bare_bijection(m, &name, f, &elem_sup)? {
            Some(per_cell) => Ok(Some(wrap_elementwise(m, &per_cell))),
            None => Ok(None),
        },
        Recognized::Lambda { body, ph, .. } => {
            if let Some(bij) = derive_matrix_affine(m, body, ph)? {
                return Ok(Some(bij));
            }
            if let Some(bij) = derive_elementwise(m, body, ph, &elem_domain, &elem_sup)? {
                return Ok(Some(bij));
            }
            Err(refuse(
                f,
                m,
                "forward map over a vector variate is not a recognised matrix-affine \
                 (mu + L * x) or elementwise (broadcast(g, x)) map — refuse rather \
                 than mislower",
            ))
        }
        Recognized::Unrecognized => Ok(None),
    }
}

/// Lift a PER-CELL scalar change of variables to a vector variate whose forward
/// Jacobian is diagonal:
///
/// * **`f_inv(y) = broadcast(g_inv, y)`** — the scalar inverse applied cell-wise;
/// * **`logvol(x) = sum(broadcast(g_logvol, x))`** — `log|det J_f| = Σᵢ log|g'(xᵢ)|`
///   (§07 `sum` reduces a real vector to a scalar).
///
/// The single emission site for both spellings of the elementwise map, so neither
/// can drift from the other. A logvol that failed to `sum` would be a vector where
/// a scalar log-det belongs.
fn wrap_elementwise(m: &mut Module, per_cell: &Bijection) -> Bijection {
    let (g_inv, g_logvol) = (per_cell.f_inv, per_cell.logvol);
    let f_inv = lambda(m, |m, y| build_call(m, "broadcast", &[g_inv, y]));
    let logvol = lambda(m, |m, x| {
        let per_cell = build_call(m, "broadcast", &[g_logvol, x]);
        build_call(m, "sum", &[per_cell])
    });
    Bijection { f_inv, logvol }
}

/// The IMAGE of a `pushfwd`'s forward map `f`, as a §03 set node for an `in(y, S)`
/// gate — or `None` where this pass cannot determine one (no gate then).
///
/// Read from `f` ALONE, and from the same [`REGISTRY`] entry the change of
/// variables comes from, so the explicit `bijection(f, f_inv, logvol)` spelling
/// gates identically to the synthesised one: the annotation records an inverse and
/// a log-volume but never an image, while §06 makes its result "a function that is
/// semantically `f`".
///
/// For a COMPOSITION the gate is the OUTERMOST op's image, which is a SUPERSET of
/// the composition's own (`image(gₙ∘…∘g₁) ⊆ image(gₙ)`) — exact when the inner ops
/// are affine, hence onto, and never a false −∞ otherwise. Tightening it
/// (`x -> exp(x) + 1` has image (1, ∞) and gates here on (0, ∞)) needs the forward
/// interval propagation the chain domain guard is also waiting on.
pub(crate) fn forward_image(m: &mut Module, f: NodeId, domain: &Type) -> Option<NodeId> {
    let (f_resolved, _) = resolve_ref_one(m, f);
    if domain_is_vector(domain) {
        // An elementwise map applies one scalar `g` per cell, so its image is `g`'s
        // in every cell — `cartpow(image(g), n)` (§03). A dynamic length has no
        // static `n` to build that power with, and a matrix-affine map is onto.
        let g = elementwise_operator(m, f_resolved)?;
        let elem = forward_image(m, g, &vector_elem_domain(domain))?;
        let n = static_vector_len(domain)?;
        let n_node = m.alloc(Node::Lit(Scalar::Int(n)));
        return Some(build_call(m, "cartpow", &[elem, n_node]));
    }
    let image = scalar_image(m, f_resolved)?;
    Some(image.set_node(m))
}

/// A SCALAR forward map's image: the [`REGISTRY`] entry's own for a bare builtin
/// (`pushfwd(exp, M)`) or a one-op lambda, `pow`'s per-exponent range for
/// `pow(_, k)`, and the outermost op's for a longer chain (see
/// [`forward_image`]).
fn scalar_image(m: &Module, f: NodeId) -> Option<Image> {
    match recognise(m, f) {
        Recognized::BareConst(name) => unary_entry(&name)?.1.image,
        Recognized::Lambda { body, ph, .. } => {
            if let Some(k_node) = single_pow(m, body, ph) {
                return pow_image(m, k_node);
            }
            let ops = flatten_chain(m, body, ph).ok().flatten()?;
            match ops.first()? {
                ChainOp::Registry { entry, .. } => entry.image,
                _ => None,
            }
        }
        Recognized::Unrecognized => None,
    }
}

/// `pow(_, k)`'s image: `xᵏ` carries `[0, ∞)` onto `[0, ∞)` for `k > 0` and
/// `(0, ∞)` onto `(0, ∞)` for `k < 0` — the same split as its domain
/// ([`derive_pow`], which refuses `k = 0`).
fn pow_image(m: &Module, k_node: NodeId) -> Option<Image> {
    let k = literal_real(m, k_node)?;
    if k == 0.0 {
        return None;
    }
    Some(Image::Set(if k < 0.0 {
        Domain::PosReals
    } else {
        Domain::NonNegReals
    }))
}

/// The scalar per-cell operator of an elementwise forward map over a vector
/// variate: the map ITSELF for the bare spelling (`pushfwd(exp, MvNormal)`), or
/// `broadcast`'s operator for the lambda spelling (`x -> broadcast(exp, x)`).
/// `None` for any other vector body — a matrix-affine map, or a coupled broadcast
/// (both refused or onto, neither gated).
fn elementwise_operator(m: &Module, f: NodeId) -> Option<NodeId> {
    match recognise(m, f) {
        Recognized::BareConst(_) => Some(f),
        Recognized::Lambda { body, ph, .. } => {
            let c = expect_builtin_call(m, body, "broadcast")?;
            if !c.named.is_empty() || c.args.len() != 2 {
                return None;
            }
            let (g, data) = (c.args[0], c.args[1]);
            is_placeholder_ref(m, data, ph).then_some(g)
        }
        Recognized::Unrecognized => None,
    }
}

/// The STATIC length of a vector `domain`, or `None` for a dynamic one.
fn static_vector_len(domain: &Type) -> Option<i64> {
    match domain {
        Type::Array { shape, .. } if shape.len() == 1 => match shape[0] {
            Dim::Static(n) => Some(i64::from(n)),
            Dim::Dynamic => None,
        },
        _ => None,
    }
}

/// One entry of the §06 case-1 known-bijection registry for a unary builtin
/// forward op `g`, expressed as BUILDERS parameterised by the point `g`'s
/// derivative is taken at. Both `pushfwd` spellings consume the same entry:
///
/// * the BARE spelling (`pushfwd(g, M)`, [`bare_bijection`]) builds at a fresh
///   lambda placeholder, yielding the single-input callables `lower_pushfwd`
///   applies;
/// * the LAMBDA spelling (`pushfwd(x -> g(x), M)` and any deeper composition,
///   [`derive_chain`]) builds at `g`'s own PARTIAL-FORWARD sub-expression — the
///   point the chain rule evaluates `log|gᵢ'|` at.
///
/// Keeping the two columns as builders rather than as finished callables is what
/// lets one table serve both: a chain needs `log|g'|` AT a node it already holds,
/// not a function it must apply.
struct UnaryEntry {
    /// `g⁻¹` — the inverse leg of the change of variables.
    inverse: Inverse,
    /// The FORWARD log-volume `log|g'|` at `point`.
    logvol: LogVol,
    /// The §06 case-1 domain restriction on `g`'s input, if any.
    domain: Option<Domain>,
    /// `g`'s IMAGE, where it is a proper subset of the reals — the set the density
    /// path gates the query point against ([`Image`]). `None` for an onto map
    /// (`log`, `neg`, `sinh`), which needs no gate.
    image: Option<Image>,
}

/// The inverse column of a [`UnaryEntry`], carried so that BOTH spellings a
/// consumer can need are derived from one fact.
enum Inverse {
    /// The inverse is exactly one builtin: `g⁻¹(y) = <name>(y)`. Kept as the NAME
    /// rather than only as a builder because a consumer may need it as an OPERATOR
    /// VALUE — [`derive_elementwise`] passes the per-cell inverse to `broadcast`,
    /// where the bare operator (`broadcast(log, y)`) is the canonical spelling and
    /// a lambda would nest a second `_x_` placeholder inside the outer one.
    Builtin(&'static str),
    /// The inverse needs a compound expression (`log10⁻¹(y) = pow(10, y)`).
    Build(fn(&mut Module, NodeId) -> NodeId),
}

impl Inverse {
    /// The inverse APPLIED at `point` — what a chain accumulates.
    fn at(&self, m: &mut Module, point: NodeId) -> NodeId {
        match self {
            Inverse::Builtin(name) => build_call(m, name, &[point]),
            Inverse::Build(build) => build(m, point),
        }
    }

    /// The inverse as a single-input CALLABLE value — what `lower_pushfwd` applies
    /// and what `broadcast` takes as its operator: the bare builtin where there is
    /// one, else a lambda around the builder.
    fn callable(&self, m: &mut Module) -> NodeId {
        match self {
            Inverse::Builtin(name) => {
                let sym = m.intern(name);
                m.alloc(Node::Const(sym))
            }
            Inverse::Build(build) => lambda(m, *build),
        }
    }
}

/// The forward log-volume column of a [`UnaryEntry`].
enum LogVol {
    /// Identically zero (`|g'| = 1`, so `g` is volume-preserving): the bare
    /// spelling emits the constant-`0` lambda, and a chain DROPS the term from its
    /// sum (an all-zero sum collapses to the literal `0`).
    Zero,
    /// `log|g'|` built at the point.
    At(fn(&mut Module, NodeId) -> NodeId),
}

/// A §06 case-1 domain restriction: "A domain-restricted forward — `log`/`log10`
/// on `posreals`, `sqrt` (and `pow`) on `nonnegreals`, `log1p` on
/// `interval(-1, inf)`, `logit`/`probit` on `interval(0, 1)` — additionally
/// requires the base measure's support to lie within that domain; where it does
/// not, density evaluation is refused rather than yielding a silently
/// sub-probability measure."
///
/// One variant per §06 set, so an entry carries the domain §06 gives it and no
/// other. Containment is decided against the base's refined SUPPORT, and an
/// endpoint the forward sends to ±inf reads differently by measure class: it is a
/// measure-zero boundary under a CONTINUOUS base (full mass survives) but a
/// positive-mass ATOM under a discrete one (that mass is lost), so a discrete
/// support must exclude it outright. Discrete supports are otherwise admitted —
/// `nonnegintegers ⊆ nonnegreals`, and since the pushforward of a discrete base
/// carries no volume element (`crate::density::reference_measure`) there is no
/// Jacobian over it to be wrong.
#[derive(Clone, Copy)]
enum Domain {
    /// `posreals` — `log`, `log10`, and `pow` with a NEGATIVE exponent (`x^k` is
    /// undefined at 0 there). 0 is outside the domain.
    PosReals,
    /// `nonnegreals` — `sqrt`, and `pow` with a positive exponent. 0 is INSIDE
    /// the domain, so a discrete atom at 0 is admitted (`sqrt(0) = 0`).
    NonNegReals,
    /// `interval(-1, inf)` — `log1p`.
    AboveMinusOne,
    /// `interval(0, 1)` — `logit`, `probit`.
    Unit,
}

impl Domain {
    /// Is a base measure whose refined support is `support` PROVABLY inside this
    /// domain? Conservative: every support that does not prove containment
    /// refuses, including `%unknown`/`anything`/`%deferred` and the caller's
    /// `None → Unknown` fallback (refuse-don't-mislower). This reads the inferred
    /// support (`Module::valueset_of`), NOT the coarse structural type of the
    /// variate — a `scalar real` variate has natural extent `reals`, which would
    /// refuse every scalar base, so the caller threads the refined support here.
    fn admits(self, support: &ValueSet) -> bool {
        use ValueSet::*;
        match self {
            // `log 0 = −inf`: a continuous base may touch 0 (measure-zero
            // boundary), a discrete one may not, so only the strictly-positive
            // integers qualify among the discrete supports.
            Domain::PosReals => match support {
                PosReals | NonNegReals | UnitInterval => true,
                Interval(lo, _) => *lo >= 0.0,
                PosIntegers => true,
                _ => false,
            },
            // 0 is in the domain, so `nonnegintegers` (`Poisson`, `Binomial`,
            // `Geometric`) and `booleans` (`Bernoulli`) are admitted alongside the
            // continuous non-negative sets.
            Domain::NonNegReals => match support {
                PosReals | NonNegReals | UnitInterval => true,
                Interval(lo, _) => *lo >= 0.0,
                PosIntegers | NonNegIntegers | Booleans => true,
                _ => false,
            },
            // `log1p(−1) = −inf`; no integer support has −1 as its least element
            // (`integers` is unbounded below and refuses anyway), so the discrete
            // sets need no endpoint carve-out here.
            Domain::AboveMinusOne => match support {
                PosReals | NonNegReals | UnitInterval => true,
                Interval(lo, _) => *lo >= -1.0,
                PosIntegers | NonNegIntegers | Booleans => true,
                _ => false,
            },
            // `logit`/`probit` are ±inf at BOTH endpoints. A continuous base may
            // touch them; `booleans` is exactly {0, 1} — both atoms — so no
            // discrete support lies inside this domain.
            Domain::Unit => match support {
                UnitInterval => true,
                Interval(lo, hi) => *lo >= 0.0 && *hi <= 1.0,
                _ => false,
            },
        }
    }

    /// The §06 set, for a message that only names the domain.
    fn set(self) -> &'static str {
        match self {
            Domain::PosReals => "the positive reals",
            Domain::NonNegReals => "the non-negative reals",
            Domain::AboveMinusOne => "(−1, ∞)",
            Domain::Unit => "(0, 1)",
        }
    }

    /// The §06 set PLUS what [`admits`](Self::admits) actually allows at an
    /// endpoint the forward sends to ±inf — for a message reporting a failed
    /// containment check, where the endpoint rule is usually why it failed.
    fn describe(self) -> &'static str {
        match self {
            Domain::PosReals => {
                "the positive reals (a continuous base may touch 0, a discrete one may not)"
            }
            Domain::NonNegReals => "the non-negative reals",
            Domain::AboveMinusOne => "(−1, ∞) (a continuous base may touch −1)",
            Domain::Unit => "(0, 1) (a continuous base may touch 0 or 1)",
        }
    }

    /// This set as a §03 set-valued node, for an `in(y, S)` membership gate.
    fn set_node(self, m: &mut Module) -> NodeId {
        match self {
            Domain::PosReals => bare_const(m, "posreals"),
            Domain::NonNegReals => bare_const(m, "nonnegreals"),
            Domain::AboveMinusOne => {
                let lo = m.alloc(Node::Lit(Scalar::Real(-1.0)));
                let hi = bare_const(m, "inf");
                build_call(m, "interval", &[lo, hi])
            }
            // §03's `unitinterval` is [0, 1] — the closed form of §06's `interval(0, 1)`.
            Domain::Unit => bare_const(m, "unitinterval"),
        }
    }
}

/// The IMAGE of a recognised forward map `g`: the set outside which the
/// pushforward has no mass. §06 `(f_*M)(Y) = M(f⁻¹(Y))` — at a `y` outside the
/// image the preimage is EMPTY, so the measure is 0 and the log-density −∞. That
/// is a computable value, not an intractable one, so the density path gates on
/// this set instead of refusing.
///
/// A CLOSED superset of an open image is fine (`invlogit` gates on `unitinterval`
/// ⊋ (0, 1), `tanh` on [−1, 1] ⊋ (−1, 1)): the inverse sends the endpoint to ±inf,
/// where the base density is already −∞, so the two agree there.
#[derive(Clone, Copy)]
enum Image {
    /// The §03 set a [`Domain`] variant already names. For an inverse PAIR the
    /// image IS the partner's domain (`exp`'s image is `log`'s domain, `posreals`),
    /// so both columns read one vocabulary.
    Set(Domain),
    /// A range no [`Domain`] variant names, built directly.
    Build(fn(&mut Module) -> NodeId),
}

impl Image {
    fn set_node(&self, m: &mut Module) -> NodeId {
        match self {
            Image::Set(d) => d.set_node(m),
            Image::Build(build) => build(m),
        }
    }
}

/// A bare builtin constant node (`posreals`, `unitinterval`, `inf`, `pi`).
fn bare_const(m: &mut Module, name: &str) -> NodeId {
    let sym = m.intern(name);
    m.alloc(Node::Const(sym))
}

/// `interval(lo, hi)` with literal endpoints.
fn literal_interval(m: &mut Module, lo: f64, hi: f64) -> NodeId {
    let lo = m.alloc(Node::Lit(Scalar::Real(lo)));
    let hi = m.alloc(Node::Lit(Scalar::Real(hi)));
    build_call(m, "interval", &[lo, hi])
}

/// The §06 case-1 known-bijection registry: the built-in unary forwards every
/// conforming engine must recognise by name, each with its inverse, its forward
/// log-volume `log|g'|`, and its domain restriction. This is the SINGLE table
/// both `pushfwd` entry points read (see [`unary_entry`]) — §06 nowhere
/// distinguishes `pushfwd(g, M)` from `pushfwd(x -> g(x), M)`, and `bijection`'s
/// own entry describes its annotated result as "a function that is semantically
/// `f`", so the spelling must not change the outcome.
///
/// `exp`/`log`, `log10`, `log1p`/`expm1`, `logit`/`invlogit`,
/// `probit`/`invprobit`, `atan`, `sinh`/`asinh` and `tanh` are §06's named
/// members; `neg` is the volume-preserving reflection of §06's affine set; `sqrt`
/// is §06's "`pow` with literal exponent (of which `sqrt` = `pow(_, 1/2)` is a
/// case)" and so reuses [`pow_inverse`]/[`pow_logvol`] verbatim rather than
/// carrying a parallel derivation. Each log-volume was cross-checked against
/// numerical differentiation; §06's `cis` (complex) is out of scope here.
static REGISTRY: &[(&str, UnaryEntry)] = &[
    // d/dx eˣ = eˣ ⇒ log|f'| = x (identity).
    (
        "exp",
        UnaryEntry {
            inverse: Inverse::Builtin("log"),
            logvol: LogVol::At(|_m, x| x),
            domain: None,
            image: Some(Image::Set(Domain::PosReals)),
        },
    ),
    // d/dx ln x = 1/x ⇒ log|f'| = −ln x. Domain posreals: over a base whose
    // support is not PROVABLY positive, `f_inv = exp` / `logvol = neg(log(x))`
    // would still typecheck and "lower", but the density is valid only on the
    // positive part of the support — a silently SUB-probability measure.
    (
        "log",
        UnaryEntry {
            inverse: Inverse::Builtin("exp"),
            logvol: LogVol::At(|m, x| {
                let logx = build_call(m, "log", &[x]);
                build_call(m, "neg", &[logx])
            }),
            domain: Some(Domain::PosReals),
            image: None,
        },
    ),
    // f'(x) = −1 ⇒ log|f'| = 0.
    (
        "neg",
        UnaryEntry {
            inverse: Inverse::Builtin("neg"),
            logvol: LogVol::Zero,
            domain: None,
            image: None,
        },
    ),
    // sqrt(x) = pow(x, 0.5) — §06's literal-exponent `pow` case, so the inverse
    // `pow(y, 1/k)` and log-volume `log|k| + (k−1)·log x` come from the shared
    // `pow` builders at k = 0.5. Domain nonnegreals, §06's own set for `sqrt`
    // (`pow` at a positive exponent takes the same one, see `derive_pow`).
    (
        "sqrt",
        UnaryEntry {
            inverse: Inverse::Build(|m, y| pow_inverse(m, SQRT_EXPONENT, y)),
            logvol: LogVol::At(|m, x| {
                let k_node = m.alloc(Node::Lit(Scalar::Real(SQRT_EXPONENT)));
                pow_logvol(m, k_node, SQRT_EXPONENT, x)
            }),
            domain: Some(Domain::NonNegReals),
            image: Some(Image::Set(Domain::NonNegReals)),
        },
    ),
    // log10(x) = ln x / ln 10 ⇒ log|f'| = −ln x − ln(ln 10); inverse
    // 10ˣ = pow(10, x). Domain posreals (same guard as `log`).
    (
        "log10",
        UnaryEntry {
            inverse: Inverse::Build(|m, y| {
                let ten = m.alloc(Node::Lit(Scalar::Real(10.0)));
                build_call(m, "pow", &[ten, y])
            }),
            logvol: LogVol::At(|m, x| {
                let logx = build_call(m, "log", &[x]);
                let ten = m.alloc(Node::Lit(Scalar::Real(10.0)));
                let ln10 = build_call(m, "log", &[ten]);
                let ln_ln10 = build_call(m, "log", &[ln10]);
                let s = build_call(m, "add", &[logx, ln_ln10]);
                build_call(m, "neg", &[s])
            }),
            domain: Some(Domain::PosReals),
            image: None,
        },
    ),
    // log1p(x) = ln(1 + x) ⇒ log|f'| = −ln(1 + x) = −log1p(x); inverse expm1.
    (
        "log1p",
        UnaryEntry {
            inverse: Inverse::Builtin("expm1"),
            logvol: LogVol::At(|m, x| {
                let l = build_call(m, "log1p", &[x]);
                build_call(m, "neg", &[l])
            }),
            domain: Some(Domain::AboveMinusOne),
            image: None,
        },
    ),
    // expm1(x) = eˣ − 1 ⇒ log|f'| = x (identity); inverse log1p. Domain ℝ.
    (
        "expm1",
        UnaryEntry {
            inverse: Inverse::Builtin("log1p"),
            logvol: LogVol::At(|_m, x| x),
            domain: None,
            image: Some(Image::Set(Domain::AboveMinusOne)),
        },
    ),
    // logit(p) = ln(p / (1 − p)) ⇒ log|f'| = −ln p − ln(1 − p); inverse invlogit.
    (
        "logit",
        UnaryEntry {
            inverse: Inverse::Builtin("invlogit"),
            logvol: LogVol::At(|m, x| {
                let logp = build_call(m, "log", &[x]);
                let one = m.alloc(Node::Lit(Scalar::Real(1.0)));
                let omp = build_call(m, "sub", &[one, x]);
                let log_omp = build_call(m, "log", &[omp]);
                let s = build_call(m, "add", &[logp, log_omp]);
                build_call(m, "neg", &[s])
            }),
            domain: Some(Domain::Unit),
            image: None,
        },
    ),
    // invlogit(x) = 1 / (1 + e⁻ˣ) ⇒ log|f'| = ln σ(x) + ln(1 − σ(x)); inverse
    // logit. Domain ℝ.
    (
        "invlogit",
        UnaryEntry {
            inverse: Inverse::Builtin("logit"),
            logvol: LogVol::At(|m, x| {
                let s = build_call(m, "invlogit", &[x]);
                let log_s = build_call(m, "log", &[s]);
                let one = m.alloc(Node::Lit(Scalar::Real(1.0)));
                let oms = build_call(m, "sub", &[one, s]);
                let log_oms = build_call(m, "log", &[oms]);
                build_call(m, "add", &[log_s, log_oms])
            }),
            domain: None,
            image: Some(Image::Set(Domain::Unit)),
        },
    ),
    // probit(p) = Φ⁻¹(p) ⇒ log|f'| = ½ln(2π) + ½·probit(p)²; inverse invprobit (Φ).
    (
        "probit",
        UnaryEntry {
            inverse: Inverse::Builtin("invprobit"),
            logvol: LogVol::At(|m, x| {
                let half_ln2pi = half_ln_two_pi(m);
                let pr = build_call(m, "probit", &[x]);
                let two = m.alloc(Node::Lit(Scalar::Real(2.0)));
                let sq = build_call(m, "pow", &[pr, two]);
                let half = m.alloc(Node::Lit(Scalar::Real(0.5)));
                let half_sq = build_call(m, "mul", &[half, sq]);
                build_call(m, "add", &[half_ln2pi, half_sq])
            }),
            domain: Some(Domain::Unit),
            image: None,
        },
    ),
    // invprobit(x) = Φ(x) ⇒ log|f'| = ln φ(x) = −½ln(2π) − ½x²; inverse probit.
    // Domain ℝ.
    (
        "invprobit",
        UnaryEntry {
            inverse: Inverse::Builtin("probit"),
            logvol: LogVol::At(|m, x| {
                let half_ln2pi = half_ln_two_pi(m);
                let two = m.alloc(Node::Lit(Scalar::Real(2.0)));
                let sq = build_call(m, "pow", &[x, two]);
                let half = m.alloc(Node::Lit(Scalar::Real(0.5)));
                let half_sq = build_call(m, "mul", &[half, sq]);
                let s = build_call(m, "add", &[half_ln2pi, half_sq]);
                build_call(m, "neg", &[s])
            }),
            domain: None,
            image: Some(Image::Set(Domain::Unit)),
        },
    ),
    // atan(x) ⇒ log|f'| = −ln(1 + x²); inverse tan (valid on atan's range
    // (−π/2, π/2), where tan is the single-valued inverse). Domain ℝ.
    (
        "atan",
        UnaryEntry {
            inverse: Inverse::Builtin("tan"),
            logvol: LogVol::At(|m, x| {
                let two = m.alloc(Node::Lit(Scalar::Real(2.0)));
                let sq = build_call(m, "pow", &[x, two]);
                let one = m.alloc(Node::Lit(Scalar::Real(1.0)));
                let onepx2 = build_call(m, "add", &[one, sq]);
                let l = build_call(m, "log", &[onepx2]);
                build_call(m, "neg", &[l])
            }),
            domain: None,
            // `tan` is the single-valued inverse only on (−π/2, π/2) — outside it
            // `tan(y)` is still finite, so an ungated query would read a preimage
            // that is not one.
            image: Some(Image::Build(|m| {
                let pi = bare_const(m, "pi");
                let two = m.alloc(Node::Lit(Scalar::Real(2.0)));
                let half_pi = build_call(m, "divide", &[pi, two]);
                let neg_half_pi = build_call(m, "neg", &[half_pi]);
                build_call(m, "interval", &[neg_half_pi, half_pi])
            })),
        },
    ),
    // sinh(x) ⇒ log|f'| = ln cosh(x); inverse asinh. Domain ℝ.
    (
        "sinh",
        UnaryEntry {
            inverse: Inverse::Builtin("asinh"),
            logvol: LogVol::At(|m, x| {
                let ch = build_call(m, "cosh", &[x]);
                build_call(m, "log", &[ch])
            }),
            domain: None,
            image: None,
        },
    ),
    // asinh(x) ⇒ log|f'| = −½ln(1 + x²); inverse sinh. Domain ℝ.
    (
        "asinh",
        UnaryEntry {
            inverse: Inverse::Builtin("sinh"),
            logvol: LogVol::At(|m, x| {
                let two = m.alloc(Node::Lit(Scalar::Real(2.0)));
                let sq = build_call(m, "pow", &[x, two]);
                let one = m.alloc(Node::Lit(Scalar::Real(1.0)));
                let onepx2 = build_call(m, "add", &[one, sq]);
                let l = build_call(m, "log", &[onepx2]);
                let mhalf = m.alloc(Node::Lit(Scalar::Real(-0.5)));
                build_call(m, "mul", &[mhalf, l])
            }),
            domain: None,
            image: None,
        },
    ),
    // tanh(x) ⇒ log|f'| = ln(1 − tanh(x)²); inverse atanh. Domain ℝ.
    (
        "tanh",
        UnaryEntry {
            inverse: Inverse::Builtin("atanh"),
            logvol: LogVol::At(|m, x| {
                let th = build_call(m, "tanh", &[x]);
                let two = m.alloc(Node::Lit(Scalar::Real(2.0)));
                let sq = build_call(m, "pow", &[th, two]);
                let one = m.alloc(Node::Lit(Scalar::Real(1.0)));
                let omsq = build_call(m, "sub", &[one, sq]);
                build_call(m, "log", &[omsq])
            }),
            domain: None,
            image: Some(Image::Build(|m| literal_interval(m, -1.0, 1.0))),
        },
    ),
];

/// `sqrt` as §06's literal-exponent `pow`: `sqrt(x) = pow(x, 1/2)`.
const SQRT_EXPONENT: f64 = 0.5;

/// Look `name` up in the shared [`REGISTRY`]. The ONE lookup both `pushfwd`
/// entry points use — [`bare_bijection`] for the bare-builtin spelling and
/// [`classify`] for the lambda/chain spelling — so their coverage cannot drift.
fn unary_entry(name: &str) -> Option<(&'static str, &'static UnaryEntry)> {
    REGISTRY
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(n, e)| (*n, e))
}

/// The single-builtin bijection for a bare builtin value (`pushfwd(exp, M)`):
/// the shared [`REGISTRY`] entry's two builders, each wrapped in a fresh lambda
/// to give the single-input callables `lower_pushfwd` applies. Any builtin
/// outside the registry (including bare `pow`, which needs an exponent) is not a
/// recognised bare bijection → `Ok(None)`.
///
/// A registry entry carrying a [`Domain`] refuses here unless the base measure's
/// support provably lies within it (§06 case 1) — refuse rather than mislower a
/// silently sub-probability measure.
fn bare_bijection(
    m: &mut Module,
    name: &str,
    f: NodeId,
    support: &ValueSet,
) -> Result<Option<Bijection>, RefuseError> {
    let Some((op, entry)) = unary_entry(name) else {
        return Ok(None);
    };
    if let Some(domain) = entry.domain {
        if !domain.admits(support) {
            return Err(refuse(
                f,
                m,
                &format!(
                    "pushfwd({op}, M) requires M's support to lie within {op}'s domain, {}; \
                     refuse rather than mislower a sub-probability measure",
                    domain.describe()
                ),
            ));
        }
    }
    let f_inv = entry.inverse.callable(m);
    let logvol = match &entry.logvol {
        LogVol::Zero => lambda(m, |m, _ph| m.alloc(Node::Lit(Scalar::Real(0.0)))),
        LogVol::At(build) => lambda(m, *build),
    };
    Ok(Some(Bijection { f_inv, logvol }))
}

/// Derive the change-of-variables for a scalar-chain forward body `f = gₙ∘…∘g₁`
/// (`input_name`/`ph` are the forward lambda's boundary — reused verbatim on the
/// `logvol` so the partial-forward sub-expressions, which reference `ph`, resolve
/// inside it). See the module docs for the inverse / chain-rule construction.
///
/// * `Ok(Some(_))` — every op in the chain is invertible.
/// * `Ok(None)` — the chain hit an unrecognised shape (a non-builtin head, or a
///   leaf that is not the input placeholder).
/// * `Err(_)` — the chain hit a recognised-but-non-invertible op (refuse), or a
///   `log` anywhere in the chain over a base whose `support` is not provably
///   positive (see the guard below).
fn derive_chain(
    m: &mut Module,
    body: NodeId,
    input_name: Symbol,
    ph: Symbol,
    support: &ValueSet,
) -> Result<Option<Bijection>, RefuseError> {
    let Some(ops) = flatten_chain(m, body, ph)? else {
        return Ok(None);
    };

    // §06 case 1's domain restriction, applied to the chain. A domain-restricted
    // op (`log`, `log10`, `log1p`, `logit`, `probit`, `sqrt`) is undefined outside
    // its domain, and lowering it there yields a silently SUB-probability measure.
    // The base measure's support bounds only the INNERMOST op's input; an interior
    // op receives an intermediate value this pass does not bound. So a
    // domain-restricted op is admitted ONLY where its input IS the base variate —
    // innermost, the last element of the outermost-first chain — and there the base
    // `support` decides. Anywhere else it refuses, which is the direction §06
    // sanctions: "refused rather than yielding a silently sub-probability measure".
    //
    // DELIBERATELY CONSERVATIVE, PENDING INTERVAL PROPAGATION THROUGH THE CHAIN.
    // This rule refuses maps that are in fact perfectly well-defined, whenever the
    // base support already lies inside the op's domain and the intervening ops
    // preserve that: `x -> log(2.0 * x)`, `x -> log(x / 2.0)`, `x -> log(x + 1.0)`,
    // `x -> log(exp(x))`, `x -> sqrt(2.0 * x)` over a positive base are all sound
    // and all refused here. Recovering them needs the propagated input interval at
    // each op (every registry forward is monotone on its own domain, so this is
    // endpoint mapping with orientation tracking, not general interval arithmetic)
    // checked for CONTAINMENT in that op's domain. That containment check is also
    // what keeps `x -> log(neg(x))` refusing once propagation exists — its
    // propagated input lands in the negatives. Until then, read a refusal from this
    // branch as "not proven", NOT as "unsound".
    for (i, op) in ops.iter().enumerate() {
        let ChainOp::Registry {
            op: name, entry, ..
        } = op
        else {
            continue;
        };
        let Some(domain) = entry.domain else { continue };
        if i + 1 != ops.len() {
            return Err(refuse(
                body,
                m,
                &format!(
                    "{name} is restricted to {} and sits inside a composition, where its input \
                     is an intermediate value this pass cannot bound — refuse rather than \
                     mislower a sub-probability measure",
                    domain.set()
                ),
            ));
        }
        if !domain.admits(support) {
            return Err(refuse(
                body,
                m,
                &format!(
                    "a forward map through {name} requires the base measure's support to lie \
                     within {name}'s domain, {}; refuse rather than mislower a sub-probability \
                     measure",
                    domain.describe()
                ),
            ));
        }
    }

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
    // A unary op in the shared §06 registry — the SAME lookup the bare spelling
    // does, so the two cannot cover different sets of ops. Its inverse and local
    // forward log-volume are built later at `args[0]`, this op's partial-forward
    // input.
    if args.len() == 1 {
        if let Some((op, entry)) = unary_entry(&name) {
            return Ok(Some((
                ChainOp::Registry {
                    op,
                    entry,
                    z: args[0],
                },
                args[0],
            )));
        }
    }
    match name.as_str() {
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
        ChainOp::Registry { entry, .. } => entry.inverse.at(m, acc),
        ChainOp::MulByLit(c) => build_call(m, "divide", &[acc, *c]),
        ChainOp::DivByLit(c) => build_call(m, "mul", &[acc, *c]),
        ChainOp::AddLit(c) => build_call(m, "sub", &[acc, *c]),
        ChainOp::SubLit(c) => build_call(m, "add", &[acc, *c]),
        ChainOp::RSubLit(c) => build_call(m, "sub", &[*c, acc]),
    }
}

/// `op`'s LOCAL forward log-derivative at its partial-forward input, or `None`
/// when it is identically zero (a volume-preserving registry op such as `neg`, or
/// an affine shift). A registry op's term is built by its [`UnaryEntry`] AT the
/// partial-forward sub-expression node — that node is already the forward
/// composition of the inner ops, expressed in the input placeholder, which is
/// exactly the point `gᵢ`'s derivative is evaluated at.
fn local_logvol(m: &mut Module, op: &ChainOp) -> Option<NodeId> {
    match op {
        ChainOp::Registry { entry, z, .. } => match &entry.logvol {
            LogVol::Zero => None,
            LogVol::At(build) => Some(build(m, *z)),
        },
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
        ChainOp::AddLit(_) | ChainOp::SubLit(_) | ChainOp::RSubLit(_) => None,
    }
}

/// `pow(_, k)`: f_inv `x -> pow(x, 1/k)`; logvol `x -> add(log(abs(k)), mul(k-1, log(x)))`.
/// Requires a nonzero literal exponent and a base whose `support` lies inside
/// `pow`'s §06 domain — the inverse `x^{1/k}` and the log-volume's `log x` are
/// defined only there (d/dx xᵏ = k·xᵏ⁻¹ ⇒ log|f'| = log|k| + (k−1)·log x).
///
/// §06 gives that domain as `nonnegreals`. A NEGATIVE exponent needs the stricter
/// `posreals`: `x^k` is undefined at 0, so an atom there is not mapped anywhere
/// and its mass is lost (under a continuous base 0 is a measure-zero boundary and
/// either domain admits it, so the split only bites on a discrete base). No
/// spelling currently reaches it — `pow(_, -1.0)` parses as `pow(_, neg(1.0))`,
/// which [`literal_real`] rejects, so the exponent is not a literal and this
/// returns `Ok(None)` first; the split is what keeps `nonnegreals` from becoming
/// wrong if that folds.
fn derive_pow(
    m: &mut Module,
    f: NodeId,
    k_node: NodeId,
    support: &ValueSet,
) -> Result<Option<Bijection>, RefuseError> {
    let Some(k) = literal_real(m, k_node) else {
        // A non-literal exponent is not a Task-1 recognised invertible form.
        return Ok(None);
    };
    if k == 0.0 {
        return Err(refuse(f, m, "pow with exponent 0 is not invertible"));
    }
    let domain = if k < 0.0 {
        Domain::PosReals
    } else {
        Domain::NonNegReals
    };
    if !domain.admits(support) {
        return Err(refuse(
            f,
            m,
            &format!(
                "pow(_, {k}) requires M's support to lie within its domain, {}; refuse rather \
                 than mislower a sub-probability measure",
                domain.describe()
            ),
        ));
    }
    let f_inv = lambda(m, |m, ph| pow_inverse(m, k, ph));
    let logvol = lambda(m, |m, ph| pow_logvol(m, k_node, k, ph));
    Ok(Some(Bijection { f_inv, logvol }))
}

/// `pow(_, k)`'s inverse at `point`: `point^{1/k}`. Shared by [`derive_pow`] (the
/// single-op `pow(x, k)` form) and the registry's `sqrt` entry (`k = 1/2`), so
/// §06's "`sqrt` = `pow(_, 1/2)`" is one derivation, not two.
fn pow_inverse(m: &mut Module, k: f64, point: NodeId) -> NodeId {
    let inv_exp = m.alloc(Node::Lit(Scalar::Real(1.0 / k)));
    build_call(m, "pow", &[point, inv_exp])
}

/// `pow(_, k)`'s FORWARD log-volume at `point`: `log|k| + (k−1)·log point`
/// (d/dx xᵏ = k·xᵏ⁻¹). `k_node` is the exponent node reused inside `abs`, `k` its
/// value. Shared by [`derive_pow`] and the registry's `sqrt` entry.
fn pow_logvol(m: &mut Module, k_node: NodeId, k: f64, point: NodeId) -> NodeId {
    let abs_k = build_call(m, "abs", &[k_node]);
    let log_abs_k = build_call(m, "log", &[abs_k]);
    let km1 = m.alloc(Node::Lit(Scalar::Real(k - 1.0)));
    let logx = build_call(m, "log", &[point]);
    let term = build_call(m, "mul", &[km1, logx]);
    build_call(m, "add", &[log_abs_k, term])
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

/// Derive the change-of-variables for `locscale(m, shift, scale)` — the
/// affine (location-scale) pushforward `pushfwd(x -> scale * x + shift, m)`
/// (spec §06 line 369/402). Rather than materialise the forward node and
/// re-recognise it, we emit `(f_inv, logvol)` directly, which reuses the exact
/// two affine forms [`derive_chain`] (scalar) and [`derive_matrix_affine`]
/// (matrix) synthesise — but WITHOUT their forward-recognition literal
/// constraints, so a SYMBOLIC `shift`/`scale` (a model parameter) lowers too.
///
/// The scalar-vs-matrix split is keyed on `m`'s variate `domain` (mirroring
/// [`derive_bijection`]'s own dispatch), so it stays consistent with the
/// pushfwd path:
///
/// * **Scalar variate** — `f_inv(y) = (y − shift) / scale`, `logvol =
///   log|scale|` (constant; a scalar affine map's forward derivative is
///   `scale`). Cross-check: `locscale(Normal(0,1), s, c)` emits
///   `logdensityof(Normal(0,1), (y−s)/c) − log|c|` = `log N(y; s, c)`.
/// * **Vector variate** — the MvNormal Cholesky case: `scale` must be a square
///   matrix `L`, `f_inv(y) = linsolve(L, y − shift)`, `logvol = logabsdet(L)`
///   (constant; §07). Identical emission to [`derive_matrix_affine`].
///
/// Refuse (never mislower) when: the base variate domain is neither confirmed
/// scalar nor a vector; a vector variate is paired with a non-matrix `scale`
/// (a scalar scale over an n-vector has forward log-volume `n·log|scale|`, not
/// `log|scale|` — the same danger [`derive_bijection`]'s vector guard closes);
/// the matrix `scale` is a CONFIRMED non-square matrix; or a scalar variate is
/// paired with a matrix `scale`. A literal-zero scalar `scale` (a non-injective
/// collapse) also refuses, matching [`classify`]'s affine-`mul` guard.
///
/// `shift` and `scale` are the raw `locscale` argument nodes; they are shared
/// (not cloned) into the emitted callables, exactly as [`derive_matrix_affine`]
/// shares `mu`/`L`.
pub(crate) fn derive_locscale(
    m: &mut Module,
    shift: NodeId,
    scale: NodeId,
    domain: &Type,
) -> Result<Bijection, RefuseError> {
    if domain_is_vector(domain) {
        // Matrix-affine (MvNormal construction): scale is a square matrix L.
        if !type_is_matrix(m, scale) {
            return Err(refuse(
                scale,
                m,
                "locscale over a vector variate requires a matrix scale; a scalar scale would give \
                 the wrong forward log-volume (n·log|scale|, not log|scale|) — refuse rather than mislower",
            ));
        }
        if matrix_confirmed_non_square(m, scale) {
            return Err(refuse(
                scale,
                m,
                "locscale matrix scale is not square (linsolve/logabsdet need a square matrix) — \
                 refuse rather than mislower",
            ));
        }
        // f_inv(y) = linsolve(scale, y − shift); logvol(_) = logabsdet(scale).
        let f_inv = lambda(m, |m, y| {
            let diff = build_call(m, "sub", &[y, shift]);
            build_call(m, "linsolve", &[scale, diff])
        });
        let logvol = lambda(m, |m, _y| build_call(m, "logabsdet", &[scale]));
        return Ok(Bijection { f_inv, logvol });
    }
    if matches!(domain, Type::Scalar(_)) {
        // Scalar affine. A matrix scale is variate-incompatible here — refuse.
        if type_is_matrix(m, scale) {
            return Err(refuse(
                scale,
                m,
                "locscale over a scalar variate requires a scalar scale, not a matrix — \
                 refuse rather than mislower",
            ));
        }
        // A literal-zero scale collapses the forward map to the constant `shift`
        // (not injective) and makes `log|scale| = −∞`; refuse (mirrors the
        // affine-`mul` literal-zero guard in `classify`). A symbolic scale is
        // trusted (as the matrix branch trusts `det L ≠ 0`).
        if literal_real(m, scale) == Some(0.0) {
            return Err(refuse(
                scale,
                m,
                "locscale with a literal-zero scale is not an injective affine map — refuse",
            ));
        }
        // f_inv(y) = (y − shift) / scale; logvol(_) = log|scale|.
        let f_inv = lambda(m, |m, y| {
            let diff = build_call(m, "sub", &[y, shift]);
            build_call(m, "divide", &[diff, scale])
        });
        let logvol = lambda(m, |m, _y| {
            let abss = build_call(m, "abs", &[scale]);
            build_call(m, "log", &[abss])
        });
        return Ok(Bijection { f_inv, logvol });
    }
    Err(refuse(
        scale,
        m,
        "locscale base measure variate domain is not confirmed scalar or vector — refuse rather \
         than guess the affine form",
    ))
}

/// Is `id`'s inferred type a MATRIX — a flat rank-2 array `Array{shape:[r, c]}`
/// or a nested vec-of-vec `Array{shape:[r], elem: Array}`? (Recognises the same
/// two representations as [`matrix_confirmed_non_square`].) A vector
/// (`Array{shape:[n], elem: Scalar}`) or scalar is NOT a matrix; an unresolved
/// type is conservatively NOT a matrix (so a vector-variate `locscale` with an
/// untyped `scale` refuses rather than assumes a matrix).
fn type_is_matrix(m: &Module, id: NodeId) -> bool {
    match m.type_of(id) {
        Some(Type::Array { shape, .. }) if shape.len() == 2 => true,
        Some(Type::Array { shape, elem }) if shape.len() == 1 => {
            matches!(elem.as_ref(), Type::Array { .. })
        }
        _ => false,
    }
}

/// Derive `(f_inv, logvol)` for a multivariate ELEMENTWISE unary forward body
/// `broadcast(g, x)` over a VECTOR variate — a single scalar-invertible `g`
/// applied to EVERY cell of `x` (spec §06 case 1, the user-requested elementwise
/// extension). The forward Jacobian is DIAGONAL (`J_f = diag(g'(x₁), …, g'(xₙ))`),
/// so its log-det is the SUM of the per-cell scalar forward log-derivatives:
///
/// * **`f_inv(y) = broadcast(g_inv, y)`** — apply `g`'s scalar inverse cell-wise.
/// * **`logvol(x) = sum(broadcast(g_logvol, x))`** — `log|det J_f| = Σᵢ log|g'(xᵢ)|`
///   (§07 `sum` reduces a real vector to a scalar; `broadcast` lifts the scalar
///   `g_logvol` over the cells).
///
/// `(g_inv, g_logvol)` are obtained by RECURSING [`derive_bijection`] on the
/// scalar operator `g` over the vector's ELEMENT `domain` — `g` then takes the
/// bare-builtin / scalar-chain path (a scalar domain is not a vector, so the
/// recursion never re-enters this arm), reusing every scalar inversion verbatim.
///
/// LogNormal-vector cross-check: for `g = exp` over an n-vector of iid `N(0,1)`,
/// `g_inv = log`, `g_logvol = identity` (`log|d/dx eˣ| = x`). The caller emits
/// `logdensityof(iid N(0,1), broadcast(log, v)) − sum(broadcast(id, broadcast(log,
/// v)))` = `Σᵢ [logN(0,1)(log vᵢ) − log vᵢ]` — exactly n independent LogNormals
/// (the standard-normal density at `log vᵢ` minus the per-cell `log vᵢ`
/// change-of-variables term, summed by the diagonal log-det). A logvol that failed
/// to `sum` (a vector, not the scalar log-det) or summed at the wrong point would
/// be a silently wrong density; `sum(broadcast(g_logvol, x))` is the correct
/// forward log-volume.
///
/// * `Ok(Some(_))` — `body` is `broadcast(g, x)` with `x` the bare input
///   placeholder and `g` scalar-invertible.
/// * `Ok(None)` — the arm does not apply (not a `broadcast`, a keyword-arg
///   broadcast, a single operand that is not the bare placeholder, or `g` is not a
///   recognised scalar map): the caller refuses via the vector guard.
/// * `Err(_)` — a COUPLED broadcast mixing TWO OR MORE variate slots
///   (`broadcast(add, x, x)`, `broadcast(mul, x, x)`) whose Jacobian is not diagonal
///   in the single-variate sense (refuse); or a recognised-but-non-invertible
///   scalar `g` (the recursion's refuse, propagated).
fn derive_elementwise(
    m: &mut Module,
    body: NodeId,
    ph: Symbol,
    elem_domain: &Type,
    elem_support: &ValueSet,
) -> Result<Option<Bijection>, RefuseError> {
    // Structural read (immutable) BEFORE the recursion / mutable builds. Only the
    // pure positional `broadcast(g, operand…)` form is this arm; a keyword data-arg
    // or a headless broadcast is not the recognised elementwise shape.
    let operands: Vec<NodeId> = {
        let Some(c) = expect_builtin_call(m, body, "broadcast") else {
            return Ok(None); // not a broadcast — this arm does not apply
        };
        if !c.named.is_empty() || c.args.is_empty() {
            return Ok(None);
        }
        c.args.to_vec()
    };
    let g = operands[0];
    let data = &operands[1..];
    // Coupled map: the input feeds two OR MORE distinct broadcast operand slots
    // (`broadcast(add, x, x)` = x .+ x, `broadcast(mul, x, x)` = x .* x). Such a map
    // is not a single-input elementwise unary — its Jacobian is not diagonal in the
    // single-variate sense (a slot-coupling / squaring) — so refuse rather than
    // synthesize a wrong per-cell diagonal log-det.
    let variate_slots = data.iter().filter(|&&a| refs_placeholder(m, a, ph)).count();
    if variate_slots >= 2 {
        return Err(refuse(
            body,
            m,
            "coupled multivariate broadcast (the input feeds two or more operand slots, \
             e.g. broadcast(add, x, x) / broadcast(mul, x, x)) is not a single-input \
             elementwise unary map with a diagonal Jacobian — refuse rather than mislower",
        ));
    }
    // The recognised shape is exactly `broadcast(g, x)`: one operand that IS the
    // bare input placeholder. Anything else (zero operands, a non-placeholder
    // operand such as `broadcast(exp, add(x, 1.0))`, or a lone constant) is not this
    // arm — Ok(None), the caller refuses.
    if data.len() != 1 || !is_placeholder_ref(m, data[0], ph) {
        return Ok(None);
    }
    // Recurse on the scalar operator `g` over the vector's element domain +
    // element support (the per-cell scalar support the domain guard reads for a
    // `log`/`pow` cell op). `None` → arm does not apply; `Err` → propagate (a
    // recognised-but-non-invertible `g`).
    let Some(g_bij) = derive_bijection(m, g, elem_domain, elem_support)? else {
        return Ok(None);
    };
    Ok(Some(wrap_elementwise(m, &g_bij)))
}

/// The element type of a vector (1-D array) `domain` — the SCALAR domain a
/// `broadcast(g, x)`'s per-cell operator `g` acts on (recursed into by
/// [`derive_elementwise`]). Falls back to `Any` for a non-array domain
/// (unreachable here — guarded by [`domain_is_vector`]).
fn vector_elem_domain(domain: &Type) -> Type {
    match domain {
        Type::Array { elem, .. } => (**elem).clone(),
        _ => Type::Any,
    }
}

/// The per-cell SUPPORT of a vector value-set — the scalar support the
/// elementwise operator `g` acts on (`cartpow(elem, n)` → `elem`), threaded into
/// the scalar recursion's [`Domain`] guard. Conservative `Unknown` for any
/// non-power support (so a `log`/`pow` cell op over an unrefined vector base
/// refuses rather than mislowers).
fn elem_support(support: &ValueSet) -> ValueSet {
    match support {
        ValueSet::CartPow(elem, _) => (**elem).clone(),
        _ => ValueSet::Unknown,
    }
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

/// Is `l`'s inferred type a matrix with CONFIRMED unequal static row/column
/// counts? Such an `L` is not invertible. A matrix with dynamic/unknown dims,
/// or an unresolved type, is NOT confirmed non-square (the standard MvNormal
/// factor is square by construction) and is not over-refused.
///
/// Two matrix representations are recognised:
/// * the FLAT rank-2 array `Array{shape: [rows, cols], elem: Real}` — produced
///   by `rowstack`/`colstack`/`lower_cholesky`;
/// * the NESTED vec-of-vec array `Array{shape: [rows], elem: Array{shape:
///   [cols], ..}}` — produced by a bracket-literal matrix `[[..], [..]]`
///   (mirrors how `rowstack_type`, `crates/infer/src/ops.rs`, recognises the
///   same nested shape when converting an array-of-vectors to a matrix).
fn matrix_confirmed_non_square(m: &Module, l: NodeId) -> bool {
    let Some(ty) = m.type_of(l) else {
        return false;
    };
    match ty {
        // Flat rank-2 matrix: shape = [rows, cols].
        Type::Array { shape, .. } if shape.len() == 2 => {
            matches!((shape[0], shape[1]), (Dim::Static(rows), Dim::Static(cols)) if rows != cols)
        }
        // Nested vec-of-vec matrix: outer shape = [rows], element is itself an
        // Array whose own shape = [cols].
        Type::Array { shape, elem } if shape.len() == 1 => {
            let Dim::Static(rows) = shape[0] else {
                return false;
            };
            let Type::Array { shape: inner, .. } = elem.as_ref() else {
                return false;
            };
            if inner.len() != 1 {
                return false;
            }
            matches!(inner[0], Dim::Static(cols) if rows != cols)
        }
        _ => false,
    }
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

/// Build a `functionof` lambda `<input_name> -> <body>` with the given boundary
/// (input name + `%local` placeholder symbol). This is the exact shape
/// [`recognise`] admits as `Recognized::Lambda`, and the shape the parser emits
/// for surface `x -> …`; the density path's record-field pushforward wrapper
/// builds its composed forward maps with it so the two `pushfwd` spellings §06
/// declares equivalent are byte-identical here.
pub(crate) fn wrap_functionof(
    m: &mut Module,
    input_name: Symbol,
    ph: Symbol,
    body: NodeId,
) -> NodeId {
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

/// The real value of a numeric literal node (`Int` widens to `Real`), or `None`.
fn literal_real(m: &Module, id: NodeId) -> Option<f64> {
    match m.node(id) {
        Node::Lit(Scalar::Real(r)) => Some(*r),
        Node::Lit(Scalar::Int(i)) => Some(*i as f64),
        _ => None,
    }
}

/// The node `½·ln(2π) = mul(0.5, log(mul(2, pi)))` — the Gaussian
/// log-normalizing constant, shared by the `probit` / `invprobit` log-volumes.
fn half_ln_two_pi(m: &mut Module) -> NodeId {
    let two = m.alloc(Node::Lit(Scalar::Real(2.0)));
    let pi_sym = m.intern("pi");
    let pi = m.alloc(Node::Const(pi_sym));
    let two_pi = build_call(m, "mul", &[two, pi]);
    let ln_2pi = build_call(m, "log", &[two_pi]);
    let half = m.alloc(Node::Lit(Scalar::Real(0.5)));
    build_call(m, "mul", &[half, ln_2pi])
}
