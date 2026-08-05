//! Analytic `(f_inv, logvol)` synthesis for known-bijection forward functions
//! (spec §06 case 1: the engine MUST recognise the standard invertible maps —
//! exp/log, affine, pow — and their scalar COMPOSITIONS analytically). Used by
//! [`lower_pushfwd`] when a `pushfwd`'s forward argument is a bare builtin, a
//! one-op lambda, or a chain/affine lambda rather than an explicit
//! `bijection(f, f_inv, logvol)` node.
//!
//! `logvol` is the FORWARD log-volume element, spelled in the QUERY POINT: the
//! single-input callable `y -> log|f'(f_inv(y))|`, which `lower_pushfwd` applies at
//! the query point itself (`logdensityof(M, f_inv(v)) - logvol(v)`) rather than at
//! the preimage. An EXPLICIT `bijection(f, f_inv, logvol)` keeps the spec's own
//! convention — a function of the forward input, applied at `f_inv(v)`: §06 →
//! "Engine contract for `pushfwd` density evaluation" says "The forward log-volume
//! is evaluated at the preimage f⁻¹(y) and **subtracted**", and §06 →
//! "Transformation and projection"'s `bijection` entry that "The convention is that
//! `logvolume` describes the forward map". So the two differ, and `lower_pushfwd`
//! names which point each is applied at.
//!
//! ## Why the query point, and not the forward input
//!
//! `log|f'|` composed with `f_inv` is the same real number either way, but the
//! ROUND TRIP through `f` at the inverse point is lossy in float where it
//! saturates, and the emission is a SUBTRACTION: `sub(-inf, -inf)` is NaN and
//! `sub(finite, -inf)` is `+inf` where §06 gives a value. Measured through
//! Enzyme-JAX (f32): `pushfwd(asinh, N(0,1))` at `y = 50` had `1 + sinh(y)²`
//! overflow to `+inf`; `pushfwd(tanh, N(0,1))` at `y = 1 − 1.2e-7` had
//! `tanh(atanh(y))` return exactly `1.0`, so `log(1 − tanh²)` was `-inf`;
//! `pushfwd(log, Gamma)` at `y = 100` had `log(exp(y))` overflow. Each entry's
//! column is therefore the ALGEBRAICALLY IDENTICAL spelling in `y` — `-log(cosh y)`,
//! `log1p(-y²)`, `-y` — which cannot saturate where the round trip did.
//!
//! ## One registry, two spellings
//!
//! §06 case 1 nowhere distinguishes `pushfwd(g, M)` from `pushfwd(x -> g(x), M)`, so
//! the spelling must not change the outcome. [`REGISTRY`] is the single table of §06's
//! named unary bijections, reached through ONE lookup ([`unary_entry`]) from both
//! [`bare_bijection`] (a bare builtin value) and [`classify`] (an op inside a lambda
//! body). Each entry carries its inverse, its log-volume `log|g'∘g⁻¹|` in the op's
//! OUTPUT point, its §06 domain restriction and its ENDPOINT MAP. The two emissions
//! are BUILDERS parameterised by the point they are built at, because a chain needs
//! the term at a node it already holds, not a callable to apply. [`forward_image`]
//! reads the endpoint map from the same table, so no spelling of `pushfwd` gates
//! differently from another.
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
//! * **`logvol(y) = Σᵢ logvolᵢ(gₙ∘…∘gᵢ(f_inv(y)))`** — the chain rule
//!   `log|f'| = Σᵢ log|gᵢ'|`, with each op's term taken at its OWN OUTPUT. Both
//!   legs come out of ONE outermost-first walk: the accumulator holds `gᵢ`'s output
//!   expressed in `y` at the moment `gᵢ⁻¹` is about to be applied to it, so the term
//!   is the entry's column built at the accumulator (`exp`: `log acc`; `log`:
//!   `−acc`; `tanh`: `log1p(−acc²)`), and the accumulator then becomes `gᵢ`'s input
//!   for the next op. The affine ops contribute a constant (`mul`: `log|c|`;
//!   `divide`: `−log|c|`) or zero (`add`/`sub`, and a volume-preserving registry op
//!   such as `neg`), so zero terms are dropped and an all-zero sum collapses to the
//!   literal `0`.
//!
//! Affine per-op table (`acc` = the accumulating inverse argument, which IS the op's
//! output point; local logvol = `log|gᵢ'|` there; the unary ops are [`REGISTRY`]'s
//! two columns):
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
//! log y + log 2` — `exp`'s term sits at `exp`'s own output, which is `y` itself,
//! where the forward-input spelling had `2x` and needed the round trip to reach it.
//!
//! A domain-restricted registry op (`log`, `log10`, `log1p`, `logit`, `probit`,
//! `sqrt`) is admitted in a chain ONLY as the innermost op, where the base measure's
//! support decides (§06 case 1). An interior one refuses, over-refusing well-defined
//! maps like `x -> log(2·x)` — see the chain domain check for what would recover them.
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
//!   in the variate (§07 `logabsdet`), emitted as an argument-ignoring lambda, so
//!   the query-point convention below costs it nothing.
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
//! * **`logvol(y) = sum(broadcast(g_logvol, y))`** — `Σᵢ log|g'(g⁻¹(yᵢ))|`, the
//!   diagonal log-det in the query point (§07 `sum` reduces a real vector to a
//!   scalar).
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
///
/// `logvol` is a function of the QUERY POINT (`y -> log|f'(f_inv(y))|`), so the
/// caller applies it at `v` and NOT at the preimage — see the module docs for why
/// the round trip through `f` at the preimage is the defect this avoids. An
/// explicit `bijection`'s third argument keeps the forward-input convention of §06
/// → "Engine contract for `pushfwd` density evaluation" instead, so
/// `crate::density::lower_pushfwd` applies the two at different points.
pub(crate) struct Bijection {
    pub f_inv: NodeId,
    pub logvol: NodeId,
}

/// One op `gᵢ` in a scalar chain, carrying what its inverse and local logvol
/// need: for a [`REGISTRY`] unary the shared entry, for the affine ops the literal
/// operand `c`. Neither leg needs a node off the FORWARD body — both are built at
/// the op's output point, which [`derive_chain`]'s inverse walk already holds.
enum ChainOp {
    /// A registry unary: inverse and local log-volume both come from the shared
    /// [`REGISTRY`] entry.
    Registry {
        op: &'static str,
        entry: &'static UnaryEntry,
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
    Lambda { body: NodeId, ph: Symbol },
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
/// * `Err(_)` — `f` is recognised but not invertible here, or `domain` is a rank the
///   §06 case-1 forms do not cover (a matrix) — refuse.
pub(crate) fn derive_bijection(
    m: &mut Module,
    f: NodeId,
    domain: &Type,
    support: &ValueSet,
) -> Result<Option<Bijection>, RefuseError> {
    // Resolve one level of self-ref (`pushfwd(g, M)` where `g = exp`).
    let (f_resolved, _) = resolve_ref_one(m, f);
    // A MATRIX variate has no recognised forward map here: §06 case 1's set is
    // scalar, elementwise-over-a-vector, and matrix-VECTOR affine (`mu + L * x`).
    // Unrefused it took the scalar derivation, which emits one scalar log-volume
    // against the whole matrix — the rank-2 analogue of the vector defect the
    // dispatch below closes, where the log-det is the SUM over all r·c cells.
    if domain_is_matrix(domain) {
        return Err(refuse(
            f,
            m,
            "the pushforward base variate is a matrix and §06 case 1 recognises no matrix forward \
             map (scalar, elementwise-over-a-vector, matrix-vector affine only)",
        ));
    }
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
        Recognized::Lambda { body, ph } => {
            // Single-op `pow(x, k)` keeps its Task-1 domain-restricted derivation;
            // a `pow` anywhere else in a chain is refused by the chain walk (its
            // input domain is not verifiable here).
            if let Some(k_node) = single_pow(m, body, ph) {
                return derive_pow(m, f, k_node, support);
            }
            derive_chain(m, body, ph, support)
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
                "forward map over a vector variate is not a recognised matrix-affine (mu + L * x) \
                 or elementwise (broadcast(g, x)) map",
            ))
        }
        Recognized::Unrecognized => Ok(None),
    }
}

/// Lift a PER-CELL scalar change of variables to a vector variate whose forward
/// Jacobian is diagonal:
///
/// * **`f_inv(y) = broadcast(g_inv, y)`** — the scalar inverse applied cell-wise;
/// * **`logvol(y) = sum(broadcast(g_logvol, y))`** — `log|det J_f| =
///   Σᵢ log|g'(g⁻¹(yᵢ))|`, the diagonal log-det in the QUERY point (§07 `sum`
///   reduces a real vector to a scalar). `g_logvol` is the per-cell column, already
///   composed with `g⁻¹`, so the reduction reads the query vector directly rather
///   than the preimage — for `g = exp` that is `sum(broadcast(y -> log y, y))`, and
///   `broadcast(log, …)` appears once in the whole emission (as `f_inv`) instead of
///   twice.
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

/// The −∞ image gate for a `pushfwd`'s forward map `f`: the membership CONDITION on
/// the query point `y`, plus a WITNESS the gated arm is safe to be built over. `None`
/// where this pass cannot determine an image (no gate then).
pub(crate) struct ImageGate {
    /// The boolean condition "`y` is in `f`'s image".
    pub(crate) cond: NodeId,
    /// A point IN the image — the value the gate substitutes for an excluded `y`
    /// (`crate::density::gate_point`), so no dangerous op in the gated arm ever sees
    /// one. Usually interior, but ON the boundary where a CLOSED endpoint comes from a
    /// discrete support's hull (`booleans`' witness is 1, and `exp(1) = e` is the hull
    /// image's own endpoint) — which is enough, since the arm only needs every op in it
    /// to stay finite there. `None` where no point is derivable: the base's support
    /// names none, or it lies outside the forward's own §06 domain.
    pub(crate) witness: Option<NodeId>,
}

/// Read from `f` and the BASE'S SUPPORT alone, through the same [`REGISTRY`] entry
/// the change of variables comes from, so the explicit `bijection(f, f_inv, logvol)`
/// spelling gates identically to the synthesised one: the annotation records an
/// inverse and a log-volume but never an image, while §06 makes its result "a
/// function that is semantically `f`".
///
/// The image is COMPUTED, never tabulated: §06 `(f_*M)(Y) = M(f⁻¹(Y))` makes it
/// `f`'s image of the support, and every §06 case-1 forward is monotone on its
/// domain, so mapping the support's two endpoints gives it ([`Extent`]). A
/// COMPOSITION propagates the support innermost-first, one op at a time, so
/// `x -> 2·exp(x) + 1` over a real base gates on its own (1, ∞) rather than on
/// `exp`'s (0, ∞) or on nothing.
pub(crate) fn forward_image(
    m: &mut Module,
    f: NodeId,
    domain: &Type,
    y: NodeId,
    support: &ValueSet,
) -> Option<ImageGate> {
    let (f_resolved, _) = resolve_ref_one(m, f);
    // A matrix variate reaches here only through an explicit `bijection` (the
    // synthesis refuses it), and the scalar image would gate a matrix against a
    // scalar set — §07 `in` requires "The type of `x` must match the element type
    // of set `S`". No gate rather than an ill-typed one.
    if domain_is_matrix(domain) {
        return None;
    }
    if domain_is_vector(domain) {
        // An elementwise map applies one scalar `g` per cell, so its image is `g`'s in
        // every cell. A dynamic length has no static `n` for the `cartpow` spelling,
        // and a matrix-affine map is onto.
        let g = elementwise_operator(m, f_resolved)?;
        let elem_sup = elem_support(support);
        let (image, per_cell) = scalar_image(m, g, &elem_sup)?;
        let n = static_vector_len(domain)?;
        let cond = image.vector_condition(m, y, n)?;
        // Every cell of the witness is the per-cell one: the image of a vector
        // elementwise map is the per-cell image in every cell.
        let witness = per_cell.map(|w| {
            let n_node = m.alloc(Node::Lit(Scalar::Int(n)));
            build_call(m, "fill", &[w, n_node])
        });
        return Some(ImageGate { cond, witness });
    }
    let (image, witness) = scalar_image(m, f_resolved, support)?;
    let cond = image.condition(m, y)?;
    Some(ImageGate { cond, witness })
}

/// A SCALAR forward map's image and its witness, both walks reading ONE reading of
/// the map's ops ([`image_ops`]) so a spelling cannot change either.
fn scalar_image(m: &mut Module, f: NodeId, support: &ValueSet) -> Option<(Extent, Option<NodeId>)> {
    let ops = image_ops(m, f)?;
    let image = image_extent(&ops, support)?;
    let witness = image_witness(m, &ops, support);
    Some((image, witness))
}

/// The ops of a recognised scalar forward map, OUTERMOST-first: one for a bare
/// builtin (`pushfwd(exp, M)`) or a one-op lambda, `pow`'s own for `pow(_, k)`, and
/// the whole chain for a longer composition. `None` for a map this pass does not
/// recognise (no gate then).
fn image_ops(m: &Module, f: NodeId) -> Option<Vec<ImageOp>> {
    match recognise(m, f) {
        Recognized::BareConst(name) => {
            let (op, entry) = unary_entry(&name)?;
            Some(vec![ImageOp::unary(op, entry)])
        }
        Recognized::Lambda { body, ph, .. } => {
            if let Some(k_node) = single_pow(m, body, ph) {
                return pow_op(m, k_node).map(|op| vec![op]);
            }
            let ops = flatten_chain(m, body, ph).ok().flatten()?;
            ops.iter().map(|op| ImageOp::of_chain_op(m, op)).collect()
        }
        Recognized::Unrecognized => None,
    }
}

/// The image of a base whose support is `support` under `ops`: the support's extent
/// pushed through them INNERMOST-first ([`push_extent`]). `None` where the walk has no
/// op to take a range from and the support proves no extent, and where an op leaves
/// nothing to map ([`Extent::nonempty`]).
fn image_extent(ops: &[ImageOp], support: &ValueSet) -> Option<Extent> {
    let mut ext = Extent::of_support(support);
    for op in ops.iter().rev() {
        ext = Some(push_extent(ext, op)?);
    }
    ext
}

/// Push `ext` through ONE op: intersect the op's §06 domain, map the endpoints, and
/// cap the result at the op's own RANGE — the image over its whole domain, which is
/// also what an unproven support falls back to (exactly the static per-op image this
/// walk replaced) and what supplies a bound the map cannot evaluate
/// ([`EndpointMap::at`]).
///
/// Capping at the range is what makes the walk MONOTONE in the safe direction: every
/// result is a subset of the op's own range, hence of the static image it replaced, so
/// an image can only tighten and never widen.
fn push_extent(ext: Option<Extent>, op: &ImageOp) -> Option<Extent> {
    let domain = op.domain.map_or(Extent::REALS, Domain::extent);
    let range = domain.map(op.map);
    let Some(e) = ext else { return Some(range) };
    let inside = e.intersect(domain).nonempty()?;
    range.intersect(inside.map(op.map)).nonempty()
}

/// A point in the base measure's `support` carried THROUGH the forward: in the image
/// by construction, and with that support point as its preimage, so the gated arm
/// reads the base density and the volume term where both are finite
/// (`crate::density::gate_point`).
///
/// `None` where the support names no point ([`support_witness`]), where the point or
/// an intermediate leaves an op's §06 domain — which the annotated `bijection`
/// spelling reaches with no domain check having run — or where an op carries it to
/// ±inf. The gate then goes unsanitised, exactly as it did before.
fn image_witness(m: &mut Module, ops: &[ImageOp], support: &ValueSet) -> Option<NodeId> {
    let mut x = support_witness(support)?;
    let mut node = m.alloc(Node::Lit(Scalar::Real(x)));
    for op in ops.iter().rev() {
        if op.domain.is_some_and(|d| !d.contains(x)) {
            return None;
        }
        x = op.map.at(x);
        if !x.is_finite() {
            return None;
        }
        node = op.build.at(m, node);
    }
    Some(node)
}

/// One op of a forward map as the IMAGE walks read it. Built from a bare [`REGISTRY`]
/// entry, from `pow`'s literal exponent and from a [`ChainOp`] alike, so the image
/// and the witness are computed by one walk each over every spelling.
struct ImageOp {
    /// The op's §06 case-1 domain restriction, intersected in before it maps.
    domain: Option<Domain>,
    map: EndpointMap,
    build: Build,
}

/// How the witness leg APPLIES an [`ImageOp`] to a node.
enum Build {
    /// `op(x)` — a [`REGISTRY`] unary.
    Unary(&'static str),
    /// `op(x, c)` with the literal operand `c` (`pow(x, k)`, `mul(x, c)`), or
    /// `op(c, x)` where `literal_first` (`sub(c, x)`).
    WithLit {
        op: &'static str,
        c: NodeId,
        literal_first: bool,
    },
}

impl Build {
    fn at(&self, m: &mut Module, x: NodeId) -> NodeId {
        match self {
            Build::Unary(op) => build_call(m, op, &[x]),
            Build::WithLit {
                op,
                c,
                literal_first: false,
            } => build_call(m, op, &[x, *c]),
            Build::WithLit {
                op,
                c,
                literal_first: true,
            } => build_call(m, op, &[*c, x]),
        }
    }
}

impl ImageOp {
    /// A [`REGISTRY`] unary, applied by name.
    fn unary(op: &'static str, entry: &UnaryEntry) -> ImageOp {
        ImageOp {
            domain: entry.domain,
            map: entry.map,
            build: Build::Unary(op),
        }
    }

    /// The image reading of one chain op. `None` where an affine op's coefficient is
    /// not a numeric literal, which [`classify`] admits none of.
    fn of_chain_op(m: &Module, op: &ChainOp) -> Option<ImageOp> {
        let affine = |c: NodeId, map: EndpointMap, op: &'static str, literal_first: bool| {
            Some(ImageOp {
                domain: None,
                map,
                build: Build::WithLit {
                    op,
                    c,
                    literal_first,
                },
            })
        };
        match op {
            ChainOp::Registry { op, entry, .. } => Some(ImageOp::unary(op, entry)),
            ChainOp::MulByLit(c) => {
                affine(*c, EndpointMap::Scale(literal_real(m, *c)?), "mul", false)
            }
            ChainOp::DivByLit(c) => affine(
                *c,
                EndpointMap::Divide(literal_real(m, *c)?),
                "divide",
                false,
            ),
            ChainOp::AddLit(c) => {
                affine(*c, EndpointMap::Shift(literal_real(m, *c)?), "add", false)
            }
            ChainOp::SubLit(c) => {
                affine(*c, EndpointMap::Shift(-literal_real(m, *c)?), "sub", false)
            }
            ChainOp::RSubLit(c) => {
                affine(*c, EndpointMap::Reflect(literal_real(m, *c)?), "sub", true)
            }
        }
    }
}

/// `pow(_, k)` as one [`ImageOp`]: §06's literal-exponent case, whose domain splits
/// on the sign of `k` — `xᵏ` is undefined at 0 for `k < 0` ([`derive_pow`], which
/// refuses `k = 0`).
fn pow_op(m: &Module, k_node: NodeId) -> Option<ImageOp> {
    let k = literal_real(m, k_node)?;
    if k == 0.0 {
        return None;
    }
    let domain = if k < 0.0 {
        Domain::PosReals
    } else {
        Domain::NonNegReals
    };
    Some(ImageOp {
        domain: Some(domain),
        map: EndpointMap::Pow(k),
        build: Build::WithLit {
            op: "pow",
            c: k_node,
            literal_first: false,
        },
    })
}

/// How a forward carries an ENDPOINT. Every §06 case-1 forward is monotone on its own
/// domain, so an interval's image is its two endpoint images ([`Extent::map`]) — the
/// whole mechanism the image comes from, in place of a per-op static image that could
/// not see the base's support.
#[derive(Clone, Copy)]
enum EndpointMap {
    /// A [`REGISTRY`] unary's own numeric forward, INCREASING on its domain (`neg`,
    /// the one reflection in the table, is `Scale(-1)`).
    Unary(fn(f64) -> f64),
    /// `atan`, kept its own variant rather than a [`Unary`](Self::Unary) so the
    /// SPELLING of its range endpoints travels with the op ([`Spelling::HalfPi`]):
    /// ±π/2 emit as §07 `pi / 2`, which a decimal literal would hide. No other op
    /// can pick that spelling up.
    Atan,
    /// `c·x` — `mul` by a literal, and `neg` at `c = −1`.
    Scale(f64),
    /// `x/c` — `divide` by a literal, kept a division so the endpoint is the value
    /// the emitted map produces (`x·(1/c)` is not `x/c`).
    Divide(f64),
    /// `x + c` — `add` a literal, and `sub` one at `−c`.
    Shift(f64),
    /// `c − x` — `sub` FROM a literal.
    Reflect(f64),
    /// `xᵏ` — §06's literal-exponent `pow`, `k ≠ 0`.
    Pow(f64),
}

impl EndpointMap {
    /// The op at an extended-real endpoint. `f64::NAN` where this pass has no
    /// implementation (`probit`/`invprobit` away from their limits, which need an
    /// inverse error function), which drops that bound to the op's range rather than
    /// guessing it.
    fn at(self, x: f64) -> f64 {
        match self {
            EndpointMap::Unary(f) => f(x),
            EndpointMap::Atan => x.atan(),
            EndpointMap::Scale(c) => x * c,
            EndpointMap::Divide(c) => x / c,
            EndpointMap::Shift(c) => x + c,
            EndpointMap::Reflect(c) => c - x,
            EndpointMap::Pow(k) => x.powf(k),
        }
    }

    /// Does the op REVERSE the two endpoints?
    fn decreasing(self) -> bool {
        match self {
            EndpointMap::Unary(_) | EndpointMap::Atan | EndpointMap::Shift(_) => false,
            EndpointMap::Scale(c) | EndpointMap::Divide(c) => c < 0.0,
            EndpointMap::Reflect(_) => true,
            EndpointMap::Pow(k) => k < 0.0,
        }
    }

    /// How the endpoints this op produces are SPELLED — a property of the op, so no
    /// other op's endpoint can take a symbolic spelling by numeric coincidence.
    fn spelling(self) -> Spelling {
        match self {
            EndpointMap::Atan => Spelling::HalfPi,
            _ => Spelling::Literal,
        }
    }
}

/// How an [`Extent`]'s endpoints are SPELLED, decided by the op that produced them
/// ([`EndpointMap::spelling`]).
#[derive(Clone, Copy, PartialEq)]
enum Spelling {
    /// A decimal literal, which `f64`'s shortest-round-trip `Display` recovers
    /// exactly.
    Literal,
    /// The op's range endpoints are ±π/2 and emit as §07 `pi / 2`.
    HalfPi,
}

/// A point strictly inside `support` — what a −∞ gate substitutes for an excluded
/// query point so the gated arm is never differentiated at one
/// (`crate::density::gate_point`). `1` for every support that contains it, which is
/// also the point every gated [`REGISTRY`] entry's domain admits; `0.5` inside the
/// unit interval; the interior of a literal `interval`.
///
/// `None` where the support names no point — `%unknown`/`%deferred`/`anything`, and a
/// simplex or product support whose interior is not a repeated scalar. The gate then
/// goes unsanitised, exactly as it did before.
pub(crate) fn support_witness(support: &ValueSet) -> Option<f64> {
    use ValueSet::*;
    match support {
        Reals | PosReals | NonNegReals | Integers | PosIntegers | NonNegIntegers | Booleans => {
            Some(1.0)
        }
        UnitInterval => Some(0.5),
        Interval(lo, hi) => interval_witness(*lo, *hi),
        // A homogeneous power's cells each take the element support's point.
        CartPow(inner, _) => support_witness(inner),
        _ => None,
    }
}

/// A point strictly inside the OPEN interval `(lo, hi)` — `1` where that lies inside,
/// else the midpoint of a bounded interval or one step in from a half-bounded one.
/// `None` for an empty or degenerate interval, and for one with no finite endpoint at
/// all (`reals` is spelled by its own [`ValueSet`] variant).
fn interval_witness(lo: f64, hi: f64) -> Option<f64> {
    // `partial_cmp` rather than `<`, so a NaN endpoint reports no witness.
    if lo.partial_cmp(&hi) != Some(std::cmp::Ordering::Less) {
        return None;
    }
    if lo < 1.0 && 1.0 < hi {
        return Some(1.0);
    }
    match (lo.is_finite(), hi.is_finite()) {
        (true, true) => Some(0.5 * (lo + hi)),
        (true, false) => Some(lo + 1.0),
        (false, true) => Some(hi - 1.0),
        (false, false) => None,
    }
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
///   [`derive_chain`]) builds at `g`'s own OUTPUT node in the inverse walk — the
///   point the chain rule evaluates `log|gᵢ'∘gᵢ⁻¹|` at.
///
/// Keeping the two columns as builders rather than as finished callables is what
/// lets one table serve both: a chain needs the term AT a node it already holds,
/// not a function it must apply.
struct UnaryEntry {
    /// `g⁻¹` — the inverse leg of the change of variables.
    inverse: Inverse,
    /// `log|g'(g⁻¹(u))|` at the op's OUTPUT point `u` — the log-volume already
    /// composed with the inverse, so no consumer re-applies `g` to reach it.
    logvol_out: LogVol,
    /// The §06 case-1 domain restriction on `g`'s input, if any.
    domain: Option<Domain>,
    /// `g` at an ENDPOINT — the one fact the image gate is computed from. `g` is
    /// monotone on `domain`, so the image of the base's support is its endpoints
    /// mapped ([`Extent::map`]); the image over the whole `domain` is `g`'s range,
    /// which is what a base with an unproven support gates on.
    map: EndpointMap,
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

/// The log-volume column of a [`UnaryEntry`], in the op's OUTPUT point.
enum LogVol {
    /// Identically zero (`|g'| = 1`, so `g` is volume-preserving): the bare
    /// spelling emits the constant-`0` lambda, and a chain DROPS the term from its
    /// sum (an all-zero sum collapses to the literal `0`).
    Zero,
    /// `log|g'(g⁻¹(u))|` built at the output point `u`.
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

    /// Does this domain contain the POINT `x`? Used on a gate's witness, which must
    /// lie inside the forward's domain for the forward to carry it into the image
    /// ([`image_witness`]) — the annotated `bijection` spelling reaches that with no
    /// [`admits`](Self::admits) check having run.
    fn contains(self, x: f64) -> bool {
        match self {
            Domain::PosReals => x > 0.0,
            Domain::NonNegReals => x >= 0.0,
            Domain::AboveMinusOne => x > -1.0,
            Domain::Unit => x > 0.0 && x < 1.0,
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

    /// This domain as an [`Extent`]: the set a base support is intersected with
    /// before it maps, and the set the forward's own RANGE is the image of.
    ///
    /// `Unit` is OPEN at both ends, matching [`contains`](Self::contains) — §06 writes
    /// the domain `interval(0, 1)` and §03's `interval` is closed, but `logit`/`probit`
    /// are ±inf AT the endpoints, so the open reading is the one that maps.
    fn extent(self) -> Extent {
        let (lo, hi) = match self {
            Domain::PosReals => (Bound::Open(0.0), Bound::Unbounded),
            Domain::NonNegReals => (Bound::Closed(0.0), Bound::Unbounded),
            Domain::AboveMinusOne => (Bound::Open(-1.0), Bound::Unbounded),
            Domain::Unit => (Bound::Open(0.0), Bound::Open(1.0)),
        };
        Extent {
            lo,
            hi,
            spell: Spelling::Literal,
        }
    }
}

/// One endpoint of an [`Extent`].
#[derive(Clone, Copy)]
enum Bound {
    /// No bound on this side.
    Unbounded,
    /// The endpoint is IN the set — `nonnegreals`' 0 (§03 `[0, +inf]`).
    Closed(f64),
    /// The endpoint is NOT in the set — `posreals`' 0 (§03 `(0, +inf]`), and every
    /// endpoint a forward reaches only in a limit.
    Open(f64),
}

impl Bound {
    /// The endpoint as an extended real; `side` is the infinity an unbounded side
    /// stands for.
    fn value(self, side: f64) -> f64 {
        match self {
            Bound::Unbounded => side,
            Bound::Closed(x) | Bound::Open(x) => x,
        }
    }

    fn is_closed(self) -> bool {
        matches!(self, Bound::Closed(_))
    }

    /// `point <op> endpoint` — the STRICT op where this bound excludes its endpoint,
    /// the inclusive one where it does not. `None` for an unbounded side.
    fn compare(
        self,
        m: &mut Module,
        point: NodeId,
        [strict, inclusive]: [&str; 2],
        spell: Spelling,
    ) -> Option<NodeId> {
        let (op, x) = match self {
            Bound::Unbounded => return None,
            Bound::Open(x) => (strict, x),
            Bound::Closed(x) => (inclusive, x),
        };
        let bound = bound_node(m, x, spell);
        Some(build_call(m, op, &[point, bound]))
    }
}

/// A connected extent of the reals with per-endpoint openness: a base measure's
/// SUPPORT (for a discrete one, its convex hull) and, mapped through a monotone
/// forward, the pushforward's IMAGE — the set outside which it has no mass. §06
/// `(f_*M)(Y) = M(f⁻¹(Y))`: at a `y` outside the image the preimage is EMPTY, so the
/// measure is 0 and the log-density −∞. That is a computable value, not an
/// intractable one, so the density path gates on this extent instead of refusing.
///
/// The gate must be EXACT at an open endpoint; a closed superset is not harmless
/// there. At the endpoint the inverse is ±inf, so the base density is −∞ AND the
/// volume term diverges, and the emission is a SUBTRACTION: `sub(−∞, −∞)` is NaN
/// where §06 gives −∞ (`pushfwd(invlogit, Normal(0,1))` at `y = 1` — `logit(1) = +∞`,
/// `logvol = log σ(∞) + log(1 − σ(∞)) = −∞`). It also breaks the GRADIENT, which is
/// why the endpoints are computed from the base's support rather than tabulated per
/// op: over `InverseGamma` (§08 support `posreals`) `sqrt`'s image is open at 0, and
/// a closed `nonnegreals` admitted `y = 0`, where the gate was TAKEN and the change
/// of variables differentiated to +inf.
/// An extent is always NON-EMPTY: `lo` below `hi`, or one point both sides include.
/// [`intersect`](Extent::intersect) is the only operation that can violate that, and
/// [`nonempty`](Extent::nonempty) is where it is caught.
#[derive(Clone, Copy)]
struct Extent {
    lo: Bound,
    hi: Bound,
    /// How both endpoints are SPELLED — set by the op that produced them, never read
    /// off their values ([`EndpointMap::spelling`]).
    spell: Spelling,
}

impl Extent {
    /// ℝ — the extent of an unrestricted domain, and the identity of [`intersect`](Self::intersect).
    const REALS: Extent = Extent {
        lo: Bound::Unbounded,
        hi: Bound::Unbounded,
        spell: Spelling::Literal,
    };

    /// The extent of a base measure's inferred support (§03's sets, read via
    /// `Module::valueset_of`). A DISCRETE support gives its convex HULL: the hull's
    /// image is a superset of the atoms' image, and the lattice test
    /// (`crate::density::on_lattice`) is what cuts it back to the atoms.
    ///
    /// `None` for a support that proves no extent — `%unknown`/`%deferred`/`anything`,
    /// a product, a simplex, the complexes — where the image falls back to the
    /// forward's own range ([`push_extent`]).
    fn of_support(support: &ValueSet) -> Option<Extent> {
        use ValueSet::*;
        let (lo, hi) = match support {
            Reals | Integers => (Bound::Unbounded, Bound::Unbounded),
            PosReals => (Bound::Open(0.0), Bound::Unbounded),
            NonNegReals | NonNegIntegers => (Bound::Closed(0.0), Bound::Unbounded),
            PosIntegers => (Bound::Closed(1.0), Bound::Unbounded),
            // §03: `booleans` is {false, true}, whose numeric hull is [0, 1].
            UnitInterval | Booleans => (Bound::Closed(0.0), Bound::Closed(1.0)),
            // §03: `interval(lo, hi)` "denotes the closed interval". An infinite
            // endpoint is no bound; `partial_cmp` rather than `<` so a NaN endpoint
            // proves no extent.
            Interval(lo, hi) => {
                if lo.partial_cmp(hi) != Some(std::cmp::Ordering::Less) {
                    return None;
                }
                (finite_bound(*lo), finite_bound(*hi))
            }
            _ => return None,
        };
        Some(Extent {
            lo,
            hi,
            spell: Spelling::Literal,
        })
    }

    /// This extent mapped through `map`, monotone on it: the two endpoint images,
    /// swapped where `map` decreases. An endpoint the map sends to ±inf (or cannot
    /// evaluate) leaves that side unbounded, and a closed endpoint stays closed only
    /// where its image is finite — the openness §06's image inherits from the
    /// support's own and from the map's endpoint behaviour.
    fn map(self, map: EndpointMap) -> Extent {
        let lo = map_bound(self.lo, f64::NEG_INFINITY, map);
        let hi = map_bound(self.hi, f64::INFINITY, map);
        let (lo, hi) = if map.decreasing() { (hi, lo) } else { (lo, hi) };
        Extent {
            lo,
            hi,
            spell: map.spelling(),
        }
    }

    /// The tighter of two extents on each side. A symbolic spelling survives only
    /// where both sides agree on it — the two partners in [`push_extent`] are one
    /// op's range and its mapped support, so they always do.
    fn intersect(self, other: Extent) -> Extent {
        Extent {
            lo: tighter(self.lo, other.lo, f64::NEG_INFINITY, |a, b| a > b),
            hi: tighter(self.hi, other.hi, f64::INFINITY, |a, b| a < b),
            spell: if self.spell == other.spell {
                self.spell
            } else {
                Spelling::Literal
            },
        }
    }

    /// `None` where this extent contains NO point: `lo` above `hi`, or one point that
    /// either side excludes. Only [`intersect`](Self::intersect) can produce one, and
    /// only where the base's support is disjoint from the op's §06 domain — reachable
    /// through an explicit `bijection`, which skips the `Domain::admits` check
    /// (`crate::density::lower_pushfwd`).
    ///
    /// An inverted extent must not reach [`named_set`](Self::named_set), which would
    /// spell it `interval(lo, hi)` backwards — and the StableHLO `in` lowering reads
    /// that as the COMPLEMENT of the intended set, since its closed-interval identity
    /// `(v − lo)·(hi − v) >= 0` holds BETWEEN an inverted pair. A map UNDEFINED outside
    /// its domain cannot get there (the out-of-domain endpoint maps to NaN or ±inf and
    /// collapses to `Unbounded`), but one merely RESTRICTED there can: `pow`'s domain is
    /// `nonnegreals` while `x³` is finite on negatives.
    ///
    /// No gate is the honest answer, not an always-false one: an empty intersection
    /// means this pass's domain reasoning does not describe the map the user ASSERTED,
    /// on a model §06 case 1 would refuse outright. Emitting nothing falls back to the
    /// ungated change of variables, which is what the same model got before the image
    /// was computed from the support at all.
    fn nonempty(self) -> Option<Extent> {
        let (lo, hi) = (
            self.lo.value(f64::NEG_INFINITY),
            self.hi.value(f64::INFINITY),
        );
        let holds = lo < hi || (lo == hi && self.lo.is_closed() && self.hi.is_closed());
        holds.then_some(self)
    }

    /// The membership condition on a SCALAR query point. `None` for an unbounded
    /// extent — an onto map, which needs no gate.
    fn condition(&self, m: &mut Module, y: NodeId) -> Option<NodeId> {
        if let Some(set) = self.named_set(m) {
            return Some(build_call(m, "in", &[y, set]));
        }
        self.comparisons(m, y, y)
    }

    /// The membership condition on an `n`-cell VECTOR query point, whose image is this
    /// one in EVERY cell: `cartpow(S, n)` (§03) where a set names it, else the same
    /// bounds against the point's extremes — every cell exceeds `lo` exactly when the
    /// minimum does.
    fn vector_condition(&self, m: &mut Module, y: NodeId, n: i64) -> Option<NodeId> {
        if let Some(elem) = self.named_set(m) {
            let n_node = m.alloc(Node::Lit(Scalar::Int(n)));
            let set = build_call(m, "cartpow", &[elem, n_node]);
            return Some(build_call(m, "in", &[y, set]));
        }
        let min = build_call(m, "minimum", &[y]);
        let max = build_call(m, "maximum", &[y]);
        self.comparisons(m, min, max)
    }

    /// The §03 set this extent IS, where one spells it EXACTLY — `posreals` is
    /// (0, +inf] and `nonnegreals` [0, +inf], and `interval(lo, hi)` "denotes the
    /// closed interval". Preferred over comparisons because `in(y, S)` is what the
    /// StableHLO lowering takes for those three sets. §03's `unitinterval` is the
    /// same set as `interval(0, 1)` and is not spelled here: that lowering does not
    /// recognise the name.
    ///
    /// `None` for an extent with a HALF-open or an open FINITE endpoint, which no §03
    /// set spells, and for an unbounded one.
    fn named_set(&self, m: &mut Module) -> Option<NodeId> {
        match (self.lo, self.hi) {
            (Bound::Open(0.0), Bound::Unbounded) => Some(bare_const(m, "posreals")),
            (Bound::Closed(0.0), Bound::Unbounded) => Some(bare_const(m, "nonnegreals")),
            (Bound::Closed(lo), Bound::Closed(hi)) => {
                let lo = bound_node(m, lo, self.spell);
                let hi = bound_node(m, hi, self.spell);
                Some(build_call(m, "interval", &[lo, hi]))
            }
            _ => None,
        }
    }

    /// `lo < lo_point` and `hi_point < hi` (`≤` where the bound includes its
    /// endpoint), conjoined, with an unbounded side dropped. `None` when neither side
    /// is bounded.
    fn comparisons(&self, m: &mut Module, lo_point: NodeId, hi_point: NodeId) -> Option<NodeId> {
        let above = self.lo.compare(m, lo_point, ["gt", "ge"], self.spell);
        let below = self.hi.compare(m, hi_point, ["lt", "le"], self.spell);
        match (above, below) {
            (Some(a), Some(b)) => Some(build_call(m, "land", &[a, b])),
            (Some(c), None) | (None, Some(c)) => Some(c),
            (None, None) => None,
        }
    }
}

/// A §03 `interval` endpoint as a [`Bound`]: closed where finite, no bound where
/// infinite.
fn finite_bound(x: f64) -> Bound {
    if x.is_finite() {
        Bound::Closed(x)
    } else {
        Bound::Unbounded
    }
}

/// One endpoint through `map`; `side` is the infinity an unbounded endpoint stands
/// for. A non-finite image is no bound at all.
fn map_bound(b: Bound, side: f64, map: EndpointMap) -> Bound {
    // `+ 0.0` normalises the `-0.0` a reflection produces at 0 (`neg`'s image of
    // `[0, inf)` is `(-inf, -0.0]`): the two compare equal, so the sign only shows up
    // in the emitted literal and in the `named_set` patterns.
    let y = map.at(b.value(side)) + 0.0;
    if !y.is_finite() {
        Bound::Unbounded
    } else if b.is_closed() {
        Bound::Closed(y)
    } else {
        Bound::Open(y)
    }
}

/// The tighter of two bounds on the same side: the one whose endpoint cuts more, and
/// on an EQUAL endpoint the open one (which excludes it).
fn tighter(a: Bound, b: Bound, side: f64, cuts_more: impl Fn(f64, f64) -> bool) -> Bound {
    let (x, y) = (a.value(side), b.value(side));
    if cuts_more(x, y) || (x == y && b.is_closed()) {
        a
    } else {
        b
    }
}

/// An [`Extent`] endpoint as a node: a decimal literal, except where the OP that
/// produced the endpoint declares a symbolic spelling for it ([`Spelling`]).
///
/// The magnitude check is not the key, it is the CONFIRMATION: `spell` says this op's
/// range endpoints are ±π/2, and the check says this bound is one of them rather than
/// an interior image point. `atan` over a BOUNDED support has finite interior
/// endpoints (`atan([0, 1])` is `[0, 0.7853981633974483]`) and they are literals like
/// any other.
fn bound_node(m: &mut Module, x: f64, spell: Spelling) -> NodeId {
    if spell == Spelling::HalfPi && x.abs() == std::f64::consts::FRAC_PI_2 {
        let h = half_pi(m);
        return if x < 0.0 {
            build_call(m, "neg", &[h])
        } else {
            h
        };
    }
    real_lit(m, x)
}

/// A bare builtin constant node (`posreals`, `unitinterval`, `inf`, `pi`).
fn bare_const(m: &mut Module, name: &str) -> NodeId {
    let sym = m.intern(name);
    m.alloc(Node::Const(sym))
}

/// A real literal node — an [`Extent`] endpoint.
fn real_lit(m: &mut Module, x: f64) -> NodeId {
    m.alloc(Node::Lit(Scalar::Real(x)))
}

/// `pi / 2` — `atan`'s image endpoint.
fn half_pi(m: &mut Module) -> NodeId {
    let pi = bare_const(m, "pi");
    let two = real_lit(m, 2.0);
    build_call(m, "divide", &[pi, two])
}

/// The §06 case-1 known-bijection registry: the built-in unary forwards every
/// conforming engine must recognise by name, each with its inverse, its log-volume
/// `log|g'∘g⁻¹|` in the op's OUTPUT point, and its domain restriction. This is the
/// SINGLE table both `pushfwd` entry points read (see [`unary_entry`]) — §06 nowhere
/// distinguishes `pushfwd(g, M)` from `pushfwd(x -> g(x), M)`, and `bijection`'s
/// own entry describes its annotated result as "a function that is semantically
/// `f`", so the spelling must not change the outcome.
///
/// Every row's column is `log|g'(g⁻¹(u))|`, and each comment carries the algebra
/// that puts it in `u`. Three rows reach `u` through their own inverse and stay that
/// way DELIBERATELY, because the composition is the numerically stable one and the
/// closed form in `u` is worse or absent: `sinh`'s `log(cosh(asinh u))` is
/// `√(1+u²)` computed without ever forming `u²` (which overflows f32 at
/// `u ≈ 1.8e19`, where `cosh∘asinh` survives to `≈ 8e37`); `atan`'s
/// `−log(1 + tan(u)²)` cannot saturate, since `tan` is finite by `1.6e7` at the
/// nearest f32 inside `±π/2`; and `invprobit`'s `log φ(probit u)` has no elementary
/// form in `u` at all. In all three the inverse is the SAME expression the base
/// density already reads, so it is not a round trip through the forward.
///
/// `exp`/`log`, `log10`, `log1p`/`expm1`, `logit`/`invlogit`,
/// `probit`/`invprobit`, `atan`, `sinh`/`asinh` and `tanh` are §06's named
/// members; `neg` is the volume-preserving reflection of §06's affine set; `sqrt`
/// is §06's "`pow` with literal exponent (of which `sqrt` = `pow(_, 1/2)` is a
/// case)" and so reuses [`pow_inverse`]/[`pow_logvol`] verbatim rather than
/// carrying a parallel derivation. Each log-volume was cross-checked against
/// numerical differentiation; §06's `cis` (complex) is out of scope here.
///
/// Every entry's endpoint map is INCREASING on its domain except `neg`, the one
/// reflection, so the image of a base support is its endpoints mapped in place
/// ([`Extent::map`]) — see [`EndpointMap`], which replaced a static per-entry image
/// that could not see the base's support. Mapping an entry's own DOMAIN through its
/// map reproduces that static image exactly, which is what an unproven support still
/// gates on.
static REGISTRY: &[(&str, UnaryEntry)] = &[
    // d/dx eˣ = eˣ ⇒ log|g'(x)| = x; at x = ln u that is ln u.
    (
        "exp",
        UnaryEntry {
            inverse: Inverse::Builtin("log"),
            logvol_out: LogVol::At(|m, u| build_call(m, "log", &[u])),
            domain: None,
            map: EndpointMap::Unary(f64::exp),
        },
    ),
    // d/dx ln x = 1/x ⇒ log|g'(x)| = −ln x; at x = eᵘ that is −u, which is where
    // the round-trip spelling `−log(exp(u))` overflowed (f32 `u ≳ 88.7`).
    // Domain posreals: over a base whose support is not PROVABLY positive,
    // `f_inv = exp` / `logvol = neg(u)` would still typecheck and "lower", but the
    // density is valid only on the positive part of the support — a silently
    // SUB-probability measure.
    (
        "log",
        UnaryEntry {
            inverse: Inverse::Builtin("exp"),
            logvol_out: LogVol::At(|m, u| build_call(m, "neg", &[u])),
            domain: Some(Domain::PosReals),
            map: EndpointMap::Unary(f64::ln),
        },
    ),
    // f'(x) = −1 ⇒ log|f'| = 0. The one DECREASING entry, spelled as its affine
    // scale so the endpoint map needs no direction column.
    (
        "neg",
        UnaryEntry {
            inverse: Inverse::Builtin("neg"),
            logvol_out: LogVol::Zero,
            domain: None,
            map: EndpointMap::Scale(-1.0),
        },
    ),
    // sqrt(x) = pow(x, 0.5) — §06's literal-exponent `pow` case, so the inverse
    // `pow(u, 1/k)` and log-volume `log|k| + ((k−1)/k)·log u` come from the shared
    // `pow` builders at k = 0.5. Domain nonnegreals, §06's own set for `sqrt`
    // (`pow` at a positive exponent takes the same one, see `derive_pow`).
    (
        "sqrt",
        UnaryEntry {
            inverse: Inverse::Build(|m, y| pow_inverse(m, SQRT_EXPONENT, y)),
            logvol_out: LogVol::At(|m, u| {
                let k_node = m.alloc(Node::Lit(Scalar::Real(SQRT_EXPONENT)));
                pow_logvol_out(m, k_node, SQRT_EXPONENT, u)
            }),
            domain: Some(Domain::NonNegReals),
            map: EndpointMap::Unary(f64::sqrt),
        },
    ),
    // log10(x) = ln x / ln 10 ⇒ log|g'(x)| = −ln x − ln(ln 10); at x = 10ᵘ,
    // ln x = u·ln 10, so the term is −(u·ln 10 + ln(ln 10)) and the `10ᵘ` the
    // round trip formed (overflowing at f32 `u ≳ 38.5`) is gone. Domain posreals
    // (same guard as `log`).
    (
        "log10",
        UnaryEntry {
            inverse: Inverse::Build(|m, y| {
                let ten = m.alloc(Node::Lit(Scalar::Real(10.0)));
                build_call(m, "pow", &[ten, y])
            }),
            logvol_out: LogVol::At(|m, u| {
                let ten = m.alloc(Node::Lit(Scalar::Real(10.0)));
                let ln10 = build_call(m, "log", &[ten]);
                let scaled = build_call(m, "mul", &[ln10, u]);
                let ln_ln10 = build_call(m, "log", &[ln10]);
                let s = build_call(m, "add", &[scaled, ln_ln10]);
                build_call(m, "neg", &[s])
            }),
            domain: Some(Domain::PosReals),
            map: EndpointMap::Unary(f64::log10),
        },
    ),
    // log1p(x) = ln(1 + x) ⇒ log|g'(x)| = −log1p(x); at x = expm1(u) that is −u,
    // where the round trip `−log1p(expm1(u))` overflowed (f32 `u ≳ 88.7`).
    (
        "log1p",
        UnaryEntry {
            inverse: Inverse::Builtin("expm1"),
            logvol_out: LogVol::At(|m, u| build_call(m, "neg", &[u])),
            domain: Some(Domain::AboveMinusOne),
            map: EndpointMap::Unary(f64::ln_1p),
        },
    ),
    // expm1(x) = eˣ − 1 ⇒ log|g'(x)| = x; at x = log1p(u) that is log1p(u).
    // Domain ℝ, range (−1, ∞) open at −1 — `exp_m1(−inf) = −1` is a limit, and
    // `log1p(−1) = −∞` with the volume term diverging there too.
    (
        "expm1",
        UnaryEntry {
            inverse: Inverse::Builtin("log1p"),
            logvol_out: LogVol::At(|m, u| build_call(m, "log1p", &[u])),
            domain: None,
            map: EndpointMap::Unary(f64::exp_m1),
        },
    ),
    // logit(p) = ln(p / (1 − p)) ⇒ log|g'(p)| = −ln p − ln(1 − p); at p = σ(u) that
    // is softplus(u) + softplus(−u) — `−ln σ(u) = softplus(−u)` and
    // `−ln(1 − σ(u)) = −ln σ(−u) = softplus(u)` — which is even in `u` and equals
    // `|u| + 2·log1p(exp(−|u|))`. The `σ(u)` spelling saturated to exactly 1.0 at
    // f32 `u ≳ 16.6`, where `ln(1 − σ)` is `−inf`.
    (
        "logit",
        UnaryEntry {
            inverse: Inverse::Builtin("invlogit"),
            logvol_out: LogVol::At(|m, u| {
                let a = build_call(m, "abs", &[u]);
                let neg_a = build_call(m, "neg", &[a]);
                let e = build_call(m, "exp", &[neg_a]);
                let l = build_call(m, "log1p", &[e]);
                let two = m.alloc(Node::Lit(Scalar::Real(2.0)));
                let doubled = build_call(m, "mul", &[two, l]);
                build_call(m, "add", &[a, doubled])
            }),
            domain: Some(Domain::Unit),
            map: EndpointMap::Unary(logit_at),
        },
    ),
    // invlogit(x) = 1 / (1 + e⁻ˣ) ⇒ log|g'(x)| = ln σ(x) + ln(1 − σ(x)); at
    // x = logit(u), σ(x) IS u, so the term is `ln u + log1p(−u)`. The round trip
    // returned exactly 1.0 from `σ(logit u)` at f32 `u = 1 − 6e-8`, making
    // `ln(1 − σ)` `−inf` and the whole density `+inf`. Domain ℝ.
    (
        "invlogit",
        UnaryEntry {
            inverse: Inverse::Builtin("logit"),
            logvol_out: LogVol::At(|m, u| {
                let log_u = build_call(m, "log", &[u]);
                let neg_u = build_call(m, "neg", &[u]);
                let log1m = build_call(m, "log1p", &[neg_u]);
                build_call(m, "add", &[log_u, log1m])
            }),
            domain: None,
            // Range (0, 1), open at BOTH endpoints (both are limits): the inverse is
            // ±∞ there and the volume term diverges with it.
            map: EndpointMap::Unary(invlogit_at),
        },
    ),
    // probit(p) = Φ⁻¹(p) ⇒ log|g'(p)| = ½ln(2π) + ½·probit(p)²; at p = Φ(u),
    // probit(p) IS u, so the term is `½ln(2π) + ½u²`. The round trip
    // `probit(invprobit(u))` returned ±inf once `Φ(u)` saturated to 0 or 1
    // (f32 `|u| ≳ 5.2`).
    (
        "probit",
        UnaryEntry {
            inverse: Inverse::Builtin("invprobit"),
            logvol_out: LogVol::At(|m, u| {
                let half_ln2pi = half_ln_two_pi(m);
                let two = m.alloc(Node::Lit(Scalar::Real(2.0)));
                let sq = build_call(m, "pow", &[u, two]);
                let half = m.alloc(Node::Lit(Scalar::Real(0.5)));
                let half_sq = build_call(m, "mul", &[half, sq]);
                build_call(m, "add", &[half_ln2pi, half_sq])
            }),
            domain: Some(Domain::Unit),
            map: EndpointMap::Unary(probit_at),
        },
    ),
    // invprobit(x) = Φ(x) ⇒ log|g'(x)| = ln φ(x) = −½ln(2π) − ½x²; at x = probit(u)
    // that is `ln φ(probit u)`, which has no elementary form in `u`. The `probit(u)`
    // that stays is the INVERSE — the same expression the base density is scored at
    // — not the forward, so this row is not a round trip. It is the one row where a
    // saturating inverse still poisons both terms: §07 gives `probit` as `-inf` only
    // "at p = 0", so a finite `u` reaching `-inf` is the emitter's
    // `√2·erf_inv(2u − 1)` losing `u` entirely once `2u − 1` rounds to `−1`, which
    // no spelling here can undo. Domain ℝ.
    (
        "invprobit",
        UnaryEntry {
            inverse: Inverse::Builtin("probit"),
            logvol_out: LogVol::At(|m, u| {
                let half_ln2pi = half_ln_two_pi(m);
                let pr = build_call(m, "probit", &[u]);
                let two = m.alloc(Node::Lit(Scalar::Real(2.0)));
                let sq = build_call(m, "pow", &[pr, two]);
                let half = m.alloc(Node::Lit(Scalar::Real(0.5)));
                let half_sq = build_call(m, "mul", &[half, sq]);
                let s = build_call(m, "add", &[half_ln2pi, half_sq]);
                build_call(m, "neg", &[s])
            }),
            domain: None,
            // Range (0, 1), open at BOTH endpoints (both are limits): the inverse is
            // ±∞ there and the volume term diverges with it.
            map: EndpointMap::Unary(invprobit_at),
        },
    ),
    // atan(x) ⇒ log|g'(x)| = −ln(1 + x²); at x = tan(u) that is `−ln(1 + tan(u)²)`,
    // kept over the equivalent `2·ln(cos u)` because it cannot saturate: `tan` at the
    // nearest f32 inside `±π/2` is only `1.6e7`, so `tan(u)²` never overflows, and
    // `tan(u)` is the inverse the base density is scored at anyway. Inverse `tan` is
    // valid on atan's range (−π/2, π/2), where it is the single-valued inverse.
    // Domain ℝ.
    (
        "atan",
        UnaryEntry {
            inverse: Inverse::Builtin("tan"),
            logvol_out: LogVol::At(|m, u| {
                let t = build_call(m, "tan", &[u]);
                let two = m.alloc(Node::Lit(Scalar::Real(2.0)));
                let sq = build_call(m, "pow", &[t, two]);
                let one = m.alloc(Node::Lit(Scalar::Real(1.0)));
                let onepx2 = build_call(m, "add", &[one, sq]);
                let l = build_call(m, "log", &[onepx2]);
                build_call(m, "neg", &[l])
            }),
            domain: None,
            // `tan` is the single-valued inverse only on `atan`'s range (−π/2, π/2) —
            // outside it `tan(y)` is still finite, so an ungated query would read a
            // preimage that is not one. Open at both: `tan(±π/2) = ±∞`. Its own
            // [`EndpointMap`] variant, which is what carries the symbolic `pi / 2`
            // spelling of those two endpoints.
            map: EndpointMap::Atan,
        },
    ),
    // sinh(x) ⇒ log|g'(x)| = ln cosh(x); at x = asinh(u) that is `ln(cosh(asinh u))`,
    // i.e. `½·log1p(u²)`. The composition is kept because it is the STABLER of the
    // two: `cosh∘asinh` computes `√(1+u²)` without forming `u²`, so it survives to
    // f32 `u ≈ 8e37` where `log1p(u²)` overflows at `1.8e19`. Domain ℝ.
    (
        "sinh",
        UnaryEntry {
            inverse: Inverse::Builtin("asinh"),
            logvol_out: LogVol::At(|m, u| {
                let a = build_call(m, "asinh", &[u]);
                let ch = build_call(m, "cosh", &[a]);
                build_call(m, "log", &[ch])
            }),
            domain: None,
            map: EndpointMap::Unary(f64::sinh),
        },
    ),
    // asinh(x) ⇒ log|g'(x)| = −½ln(1 + x²); at x = sinh(u), `1 + sinh²u = cosh²u`, so
    // the term is `−ln cosh u` = `−(|u| + log1p(exp(−2|u|)) − ln 2)` (from
    // `cosh u = e^{|u|}(1 + e^{−2|u|})/2`). Overflow-free, where the round trip's
    // `sinh(u)²` overflowed f32 at `|u| ≳ 45.5` and `−ln(cosh u)` alone would only
    // move that to `≈ 88.7`.
    (
        "asinh",
        UnaryEntry {
            inverse: Inverse::Builtin("sinh"),
            logvol_out: LogVol::At(|m, u| {
                let a = build_call(m, "abs", &[u]);
                let mtwo = m.alloc(Node::Lit(Scalar::Real(-2.0)));
                let scaled = build_call(m, "mul", &[mtwo, a]);
                let e = build_call(m, "exp", &[scaled]);
                let l = build_call(m, "log1p", &[e]);
                let s = build_call(m, "add", &[a, l]);
                let two = m.alloc(Node::Lit(Scalar::Real(2.0)));
                let ln2 = build_call(m, "log", &[two]);
                let ln_cosh = build_call(m, "sub", &[s, ln2]);
                build_call(m, "neg", &[ln_cosh])
            }),
            domain: None,
            map: EndpointMap::Unary(f64::asinh),
        },
    ),
    // tanh(x) ⇒ log|g'(x)| = ln(1 − tanh(x)²); at x = atanh(u), tanh(x) IS u, so the
    // term is `log1p(−u²)`. The round trip returned exactly 1.0 from
    // `tanh(atanh u)` at f32 `u = 1 − 1.2e-7`, making the term `−inf` and the whole
    // density `+inf`. Inverse atanh. Domain ℝ.
    (
        "tanh",
        UnaryEntry {
            inverse: Inverse::Builtin("atanh"),
            logvol_out: LogVol::At(|m, u| {
                let two = m.alloc(Node::Lit(Scalar::Real(2.0)));
                let sq = build_call(m, "pow", &[u, two]);
                let neg_sq = build_call(m, "neg", &[sq]);
                build_call(m, "log1p", &[neg_sq])
            }),
            domain: None,
            // Range (−1, 1), open at both (limits): `atanh(±1) = ±∞` and
            // `log(1 − tanh²)` diverges there.
            map: EndpointMap::Unary(f64::tanh),
        },
    ),
];

/// `logit(p) = ln(p / (1 − p))` at an endpoint: −inf at 0 and +inf at 1, so
/// `logit`'s range over its (0, 1) domain is ℝ and it carries no gate.
fn logit_at(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// `invlogit(x) = 1 / (1 + e⁻ˣ)` at an endpoint: 0 and 1 in the two limits.
fn invlogit_at(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `probit(p) = Φ⁻¹(p)` at an endpoint. Only the two DOMAIN endpoints are reachable
/// without an inverse error function, which this pass has no implementation of; a
/// finite interior point reports NaN, which leaves the image bound at `probit`'s own
/// range ([`push_extent`]) rather than guessing one.
fn probit_at(p: f64) -> f64 {
    if p <= 0.0 {
        f64::NEG_INFINITY
    } else if p >= 1.0 {
        f64::INFINITY
    } else {
        f64::NAN
    }
}

/// `invprobit(x) = Φ(x)` at an endpoint — 0 and 1 in the two limits, and NaN at a
/// finite point, on the same argument as [`probit_at`].
fn invprobit_at(x: f64) -> f64 {
    if x == f64::NEG_INFINITY {
        0.0
    } else if x == f64::INFINITY {
        1.0
    } else {
        f64::NAN
    }
}

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
                    "pushfwd({op}, M) requires M's support to lie within {op}'s domain, {}",
                    domain.describe()
                ),
            ));
        }
    }
    let f_inv = entry.inverse.callable(m);
    let logvol = match &entry.logvol_out {
        LogVol::Zero => lambda(m, |m, _ph| m.alloc(Node::Lit(Scalar::Real(0.0)))),
        LogVol::At(build) => lambda(m, *build),
    };
    Ok(Some(Bijection { f_inv, logvol }))
}

/// Derive the change-of-variables for a scalar-chain forward body `f = gₙ∘…∘g₁`
/// (`ph` is the forward lambda's placeholder, read only to flatten the chain — both
/// emitted callables carry their own fresh one). See the module docs for the
/// inverse / chain-rule construction.
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
    ph: Symbol,
    support: &ValueSet,
) -> Result<Option<Bijection>, RefuseError> {
    let Some(ops) = flatten_chain(m, body, ph)? else {
        return Ok(None);
    };

    // §06 case 1's domain restriction, applied to the chain. A domain-restricted op
    // (`log`, `log10`, `log1p`, `logit`, `probit`, `sqrt`) lowered outside its domain
    // yields a silently SUB-probability measure. The base `support` bounds only the
    // INNERMOST op's input, so a domain-restricted op is admitted ONLY as the last
    // element of the outermost-first chain; anywhere else it refuses, per §06:
    // "refused rather than yielding a silently sub-probability measure".
    //
    // CONSERVATIVE, pending interval propagation through the chain. Sound maps refuse
    // here too — `x -> log(2.0 * x)`, `x -> log(x + 1.0)`, `x -> sqrt(2.0 * x)` over a
    // positive base. Recovering them needs each op's propagated input interval checked
    // for CONTAINMENT in its domain (every registry forward is monotone on its own
    // domain, so endpoint mapping with orientation tracking suffices); the same check
    // keeps `x -> log(neg(x))` refusing. Read a refusal here as "not proven", NOT as
    // "unsound".
    for (i, op) in ops.iter().enumerate() {
        let ChainOp::Registry { op: name, entry } = op else {
            continue;
        };
        let Some(domain) = entry.domain else { continue };
        if i + 1 != ops.len() {
            return Err(refuse(
                body,
                m,
                &format!(
                    "{name} is restricted to {} and sits inside a composition, where its input is \
                     an intermediate value this pass cannot bound",
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
                     within {name}'s domain, {}",
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

    // logvol(y) = Σᵢ logvolᵢ(gᵢ's own output). The SAME outermost-first walk as
    // `f_inv`: at each step the accumulator is the op's output expressed in `y`, so
    // the term is built there BEFORE the inverse is applied. Drop the zero
    // contributions (neg / add / sub); an all-zero sum is the constant 0.
    let logvol = lambda(m, |m, y| {
        let mut acc = y;
        let mut terms = Vec::new();
        for op in &ops {
            if let Some(term) = local_logvol(m, op, acc) {
                terms.push(term);
            }
            acc = apply_inverse(m, op, acc);
        }
        if terms.is_empty() {
            m.alloc(Node::Lit(Scalar::Real(0.0)))
        } else {
            fold_add(m, &terms)
        }
    });

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
    // log-volume are built later, at the op's OUTPUT in the inverse walk; `args[0]`
    // is returned only as the subterm to descend into.
    if args.len() == 1 {
        if let Some((op, entry)) = unary_entry(&name) {
            return Ok(Some((ChainOp::Registry { op, entry }, args[0])));
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

/// `op`'s LOCAL log-volume at `out`, its own output point, or `None` when the term
/// is identically zero (a volume-preserving registry op such as `neg`, or an affine
/// shift). A registry op's term is built by its [`UnaryEntry`]'s already-composed
/// column, so `out` is the accumulating inverse argument and no forward
/// sub-expression is read.
fn local_logvol(m: &mut Module, op: &ChainOp, out: NodeId) -> Option<NodeId> {
    match op {
        ChainOp::Registry { entry, .. } => match &entry.logvol_out {
            LogVol::Zero => None,
            LogVol::At(build) => Some(build(m, out)),
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

/// `pow(_, k)`: f_inv `y -> pow(y, 1/k)`; logvol `y -> add(log(abs(k)), mul((k-1)/k, log(y)))`.
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
                "pow(_, {k}) requires M's support to lie within its domain, {}",
                domain.describe()
            ),
        ));
    }
    let f_inv = lambda(m, |m, ph| pow_inverse(m, k, ph));
    let logvol = lambda(m, |m, ph| pow_logvol_out(m, k_node, k, ph));
    Ok(Some(Bijection { f_inv, logvol }))
}

/// `pow(_, k)`'s inverse at `point`: `point^{1/k}`. Shared by [`derive_pow`] (the
/// single-op `pow(x, k)` form) and the registry's `sqrt` entry (`k = 1/2`), so
/// §06's "`sqrt` = `pow(_, 1/2)`" is one derivation, not two.
fn pow_inverse(m: &mut Module, k: f64, point: NodeId) -> NodeId {
    let inv_exp = m.alloc(Node::Lit(Scalar::Real(1.0 / k)));
    build_call(m, "pow", &[point, inv_exp])
}

/// `pow(_, k)`'s log-volume at its OUTPUT `point`: `log|k| + ((k−1)/k)·log point`.
/// From `log|g'(x)| = log|k| + (k−1)·log x` (d/dx xᵏ = k·xᵏ⁻¹) at `x = point^{1/k}`,
/// where `log x = (1/k)·log point`. That folds the inverse into the coefficient, so
/// the `point^{1/k}` the forward-input spelling formed — overflowing f32 at
/// `point ≳ 1.8e19` for `k = 1/2` — is gone. `k_node` is the exponent node reused
/// inside `abs`, `k` its value. Shared by [`derive_pow`] and the registry's `sqrt`
/// entry.
fn pow_logvol_out(m: &mut Module, k_node: NodeId, k: f64, point: NodeId) -> NodeId {
    let abs_k = build_call(m, "abs", &[k_node]);
    let log_abs_k = build_call(m, "log", &[abs_k]);
    let coeff = m.alloc(Node::Lit(Scalar::Real((k - 1.0) / k)));
    let logx = build_call(m, "log", &[point]);
    let term = build_call(m, "mul", &[coeff, logx]);
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
///   log|det L|`, CONSTANT in the variate (a linear map has constant Jacobian `L`;
///   spec §07 `logabsdet(A) = log|det A|`, square matrix → real scalar). Emitted as
///   a lambda that IGNORES its argument, so the point it is applied at cannot
///   matter: the caller applies a synthesised logvol at the QUERY point
///   ([`Bijection`]), and `logvol(v)` β-reduces to the same constant `logvol(f_inv(v))`
///   would have.
///
/// MvNormal cross-check (Σ = L Lᵀ): the caller emits `logdensityof(iid N(0,1),
/// f_inv(v)) − logvol(v)` (§06 → "Engine contract for `pushfwd` density
/// evaluation", whose change-of-variables formula subtracts the forward log-volume;
/// its own spelling evaluates that at the preimage, which for a constant is the
/// same number) =
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
/// * **`logvol(y) = sum(broadcast(g_logvol, y))`** — `log|det J_f| =
///   Σᵢ log|g'(g⁻¹(yᵢ))|` in the QUERY point (§07 `sum` reduces a real vector to a
///   scalar; `broadcast` lifts the scalar `g_logvol` over the cells).
///
/// `(g_inv, g_logvol)` are obtained by RECURSING [`derive_bijection`] on the
/// scalar operator `g` over the vector's ELEMENT `domain` — `g` then takes the
/// bare-builtin / scalar-chain path (a scalar domain is not a vector, so the
/// recursion never re-enters this arm), reusing every scalar inversion verbatim.
/// `g_logvol` therefore arrives already composed with `g⁻¹`.
///
/// LogNormal-vector cross-check: for `g = exp` over an n-vector of iid `N(0,1)`,
/// `g_inv = log` and `g_logvol = y -> log y` (`log|d/dx eˣ| = x` at `x = log y`).
/// The caller emits `logdensityof(iid N(0,1), broadcast(log, v)) −
/// sum(broadcast(y -> log y, v))` = `Σᵢ [logN(0,1)(log vᵢ) − log vᵢ]` — exactly n
/// independent LogNormals (the standard-normal density at `log vᵢ` minus the
/// per-cell `log vᵢ` change-of-variables term, summed by the diagonal log-det).
/// The volume reduction reads `v` itself, so `broadcast(log, …)` appears once in
/// the emission rather than nested inside the sum as well. A logvol that failed to
/// `sum` (a vector, not the scalar log-det) or read the wrong point would be a
/// silently wrong density.
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

/// Is the base measure's variate domain a MATRIX — rank 2 or higher? Recognises
/// the same two representations as [`type_is_matrix`]: a flat rank-2+ array
/// (`Wishart`, `LKJ`: `Array{shape:[dyn, dyn]}`) and a nested array of arrays.
/// An UNRESOLVED domain is conservatively not a matrix, so the callers' existing
/// unknown-domain paths (refused later by `density::reference_measure`) are
/// unchanged.
fn domain_is_matrix(domain: &Type) -> bool {
    match domain {
        Type::Array { shape, .. } if shape.len() >= 2 => true,
        Type::Array { shape, elem } if shape.len() == 1 => {
            matches!(elem.as_ref(), Type::Array { .. })
        }
        _ => false,
    }
}

/// Recognise the surface shape of a `pushfwd`'s (ref-resolved) forward argument:
/// a bare builtin value (`Const`), or a one-input `functionof` lambda `x -> body`
/// whose boundary is exactly one `%local` placeholder.
///
/// A CLOSED reification wrapping either of those — `functionof(v -> get(v, [1]))`,
/// which has no boundary input of its own — is unwrapped first and recognised as
/// what it wraps. §04 forbids a nullary callable "as this would make them
/// equivalent to known values", so the wrapper carries no meaning the shapes below
/// need; without this the reified spelling misses the recogniser its plain spelling
/// reaches, and refuses with a bijection-annotation misdiagnosis.
/// [`crate::kernel::classify_reification`] unwraps nested wrappers to a fixpoint, so
/// no recursion is needed here.
fn recognise(m: &Module, f: NodeId) -> Recognized {
    let f = crate::kernel::resolve_closed_reification(m, f).unwrap_or(f);
    match m.node(f) {
        Node::Const(sym) => Recognized::BareConst(m.resolve(*sym).to_string()),
        Node::Call(c) => {
            if let CallHead::Builtin(sym) = c.head {
                if m.resolve(sym) == "functionof" && c.args.len() == 1 {
                    if let Some(Inputs::Spec(entries)) = &c.inputs {
                        if entries.len() == 1 && entries[0].1.ns == RefNs::Local {
                            return Recognized::Lambda {
                                body: c.args[0],
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
