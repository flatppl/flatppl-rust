//! The `kchain` marginal-density rule and its discrete/continuous classifier
//! (spec §06, "Density of composed measures", the `kchain` row).
//!
//! `kchain(M, K)` is Kleisli bind: it marginalizes the intermediate latent `a`,
//! keeping the kernel `K`'s variate. Its density at `x` is the marginal integral
//!
//! ```text
//! densityof(kchain(M, K), x) = ∫ densityof(K(a), x) dM(a)
//! ```
//!
//! which is **generally intractable**. The spec says an engine evaluates it "in
//! closed form, or by enumeration of a discrete latent, and otherwise reports a
//! static error." This module implements the **discrete-enumeration** branch and
//! refuses everything else:
//!
//! - **Discrete-finite latent** (a statically-known, small atom set `{a₀, …, a_{N-1}}`):
//!   the integral becomes a finite **mass-weighted** sum, in log space the
//!   logsumexp
//!   ```text
//!   logsumexpᵢ[ logdensityof(M, aᵢ) + logdensityof(K(aᵢ), x) ]
//!   ```
//!   where `logdensityof(M, aᵢ)` is the latent's log-pmf at atom `aᵢ` and `K(aᵢ)`
//!   is the kernel applied to the pinned latent. This is the *mass-weighted* form
//!   — NOT `logsumexp − logN`, which is only correct for a uniform latent (the
//!   biased Monte-Carlo form the design explicitly rejects).
//!
//! - **Continuous, infinite-discrete, or otherwise non-enumerable latent**
//!   (`Normal`, `Poisson`, an unbounded integer range, …): **refused**. The
//!   conjugate / quadrature closed-form table is a deliberate follow-on.
//!
//! ## What is enumerable
//!
//! A latent is enumerable here only when its variate is a finite atom set whose
//! cardinality is **statically known and small**. We read this from the latent's
//! distribution constructor (the support `ValueSet` alone is insufficient — a
//! finite `Categorical`'s support infers to the infinite `posintegers`, and a
//! `Binomial`'s to `nonnegintegers`; the finite bound lives in the constructor's
//! arguments):
//!
//! | constructor       | support     | atoms                       | finite when            |
//! |-------------------|-------------|-----------------------------|------------------------|
//! | `Bernoulli(p)`    | `booleans`  | `{0, 1}` (integer variate)  | always (2 atoms)       |
//! | `Categorical(p)`  | `[1, n]`    | `{1, …, n}`                 | `p` a static vector(n) |
//! | `Categorical0(p)` | `[0, n-1]`  | `{0, …, n-1}`               | `p` a static vector(n) |
//! | `Binomial(n, p)`  | `[0, n]`    | `{0, …, n}`                 | `n` a static int       |
//!
//! `Poisson`, `Geometric`, `NegativeBinomial*` (support `nonnegintegers`) and any
//! continuous distribution are **not** enumerable → refused.

use crate::density::{
    build_call, build_density_term, build_record, draw_argument, expect_builtin_call,
    lower_measure_density, refuse, resolve_ref_one, split_kernel_constructor,
};
use crate::refuse::RefuseError;
use flatppl_core::{
    Call, CallHead, Module, NamedArg, NamedKind, Node, NodeId, Ref, RefNs, Scalar, Symbol, ValueSet,
};

/// Above this many atoms, refuse: an enumerated logsumexp must stay small (the
/// determiniser emits one density sub-tree per atom). A finite but large latent
/// is treated as non-enumerable.
const MAX_ATOMS: i64 = 256;

/// Lower `logdensityof(kchain(M, K), v)` at the `kchain` node `node` to a
/// deterministic mass-weighted `logsumexp`, or refuse.
pub(crate) fn lower_kchain_marginal(
    m: &mut Module,
    node: NodeId,
    v: NodeId,
) -> Result<NodeId, RefuseError> {
    // --- 1. Match the kchain node: exactly one base measure + one kernel. ---
    let (m_arg, k_arg) = {
        let c = expect_builtin_call(m, node, "kchain")
            .ok_or_else(|| refuse(node, m, "expected kchain"))?;
        // A multi-step chain `kchain(M, K1, K2, …)` marginalizes several
        // intermediate latents; only the single-step case is in scope here.
        if c.args.len() != 2 {
            return Err(refuse_kchain(
                node,
                "single-step kchain(M, K) only; multi-step chains are a follow-on",
            ));
        }
        (c.args[0], c.args[1])
    };

    // --- 2. Find the latent's distribution constructor and its variate name. ---
    // `M` is `lawof(record(name = draw(dist)))`, `lawof(draw(dist))`, or a bare
    // `dist` constructor. We need the dist node (for the pmf + classification)
    // and the latent variate name (the field name, if any) to know what value
    // shape `K(aᵢ)` / `logdensityof(M, aᵢ)` consume.
    let latent = resolve_latent(m, m_arg)
        .ok_or_else(|| refuse_kchain(node, "latent measure is not a recognisable single draw"))?;

    // --- 3. Resolve the kernel: kernelof(body, %specinputs([(input, ref)])). ---
    // Resolved before classification because both the discrete-enumeration path
    // and the continuous conjugate path need the kernel body.
    let kernel = resolve_kernel(m, k_arg)
        .ok_or_else(|| refuse_kchain(node, "kchain kernel is not a recognisable kernelof(...)"))?;

    // --- 4. Classify the latent. A discrete-finite latent enumerates (below); a
    //        continuous / infinite-discrete latent first tries the closed-form
    //        conjugate table, and only refuses if no conjugate row applies. ---
    let atoms = match classify_atoms(m, latent.dist) {
        Some(atoms) => atoms,
        None => {
            if let Some(result) = try_conjugate_marginal(m, latent.dist, &kernel, v) {
                return result;
            }
            return Err(refuse_kchain(
                node,
                "non-enumerable marginal (continuous / infinite-discrete); \
                 no conjugate closed-form applies",
            ));
        }
    };

    // --- 5. Per atom: mass term + kernel term, summed; then logsumexp. ---
    let mut branches: Vec<NodeId> = Vec::with_capacity(atoms.len());
    for &atom_val in &atoms {
        let atom_node = m.alloc(Node::Lit(Scalar::Int(atom_val)));

        // logdensityof(M, aᵢ): the latent's log-pmf at the atom, scored against
        // its OWN distribution constructor. `build_density_term` emits
        // `builtin_logdensityof(dist, dist_input, atom)`.
        let mass_term = build_density_term(m, latent.dist, atom_node)?;

        // K(aᵢ): substitute the atom for the kernel's boundary-input ref inside a
        // fresh copy of the kernel body, then score that measure at `v`.
        let applied_body = substitute_ref(m, kernel.body, kernel.input, atom_node);
        let kernel_term = lower_measure_density(m, applied_body, v)?;

        branches.push(build_call(m, "add", &[mass_term, kernel_term]));
    }

    // logsumexp over the per-atom mass-weighted branches. A single atom degenerates
    // to that one branch (logsumexp of one term = identity), which is still correct.
    if branches.len() == 1 {
        return Ok(branches[0]);
    }
    // §07 `logsumexp(v)` takes a single real VECTOR, not variadic scalars: wrap the
    // per-atom branches in a `vector` literal so the emitted call is `logsumexp([…])`.
    let branches_vec = build_call(m, "vector", &branches);
    Ok(build_call(m, "logsumexp", &[branches_vec]))
}

// ---------------------------------------------------------------------------
// Latent identification
// ---------------------------------------------------------------------------

/// The latent of a `kchain`: its distribution-constructor node (for both the
/// pmf and the discrete-finite classification).
struct Latent {
    /// The distribution-constructor node, e.g. `Bernoulli(p = 0.3)`.
    dist: NodeId,
}

/// Resolve `M` (the kchain's first argument) to the latent's distribution
/// constructor. Accepts `lawof(record(name = draw(dist)))`, `lawof(draw(dist))`,
/// `lawof(dist)`, a bare `draw(dist)`, or a bare `dist`. Returns `None` for any
/// shape we cannot pin to a single primitive constructor (e.g. a multi-field
/// record latent, or a combinator).
fn resolve_latent(m: &Module, m_arg: NodeId) -> Option<Latent> {
    let (resolved, _) = resolve_ref_one(m, m_arg);

    // Strip an optional `lawof(...)`.
    let inner = match expect_builtin_call(m, resolved, "lawof") {
        Some(law) if law.args.len() == 1 => {
            let (i, _) = resolve_ref_one(m, law.args[0]);
            i
        }
        Some(_) => return None,
        None => resolved,
    };

    // `record(name = X)` with exactly one field → X is the latent value.
    let value = if let Some(rec) = expect_builtin_call(m, inner, "record") {
        if !rec.args.is_empty() || rec.named.len() != 1 {
            return None;
        }
        let (v, _) = resolve_ref_one(m, rec.named[0].value);
        v
    } else {
        inner
    };

    // `draw(dist)` → dist; or a bare `dist` constructor.
    let dist = if let Some(draw) = expect_builtin_call(m, value, "draw") {
        if draw.args.len() != 1 {
            return None;
        }
        let (d, _) = resolve_ref_one(m, draw.args[0]);
        d
    } else {
        value
    };

    // Must be a builtin distribution-constructor call.
    if !matches!(m.node(dist), Node::Call(c) if matches!(c.head, CallHead::Builtin(_))) {
        return None;
    }
    Some(Latent { dist })
}

// ---------------------------------------------------------------------------
// Discrete-finite classification + atom enumeration
// ---------------------------------------------------------------------------

/// Classify the latent distribution `dist` and, if it is discrete-finite with a
/// statically-known small atom count, return its atoms (as integer variate
/// values). Returns `None` for any non-enumerable latent — continuous,
/// infinite-discrete, dynamically-sized, or oversized.
///
/// The atom set is read from the constructor (its name + arguments), not from
/// the support `ValueSet` alone: a finite `Categorical`'s support infers to the
/// *infinite* `posintegers`, so the bound must come from `p`'s length. We *do*
/// cross-check the support against `booleans` for `Bernoulli` as a guard.
fn classify_atoms(m: &Module, dist: NodeId) -> Option<Vec<i64>> {
    let Node::Call(c) = m.node(dist) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    let name = m.resolve(sym);

    match name {
        // Bernoulli: support `booleans`, integer variate {0, 1} — always finite.
        // Cross-check the inferred support to guard against a mis-typed node.
        "Bernoulli" => {
            if support_subset_of(m, dist, &ValueSet::Booleans) {
                Some(vec![0, 1])
            } else {
                None
            }
        }
        // Categorical(p): atoms {1, …, n}; Categorical0(p): atoms {0, …, n-1}.
        // n = the static length of `p`. The support is the infinite `posintegers`
        // (Categorical) — finiteness comes from `p`'s vector length.
        "Categorical" | "Categorical0" => {
            let n = static_vector_len(m, kwarg(m, c, "p")?)?;
            bounded(n).then(|| {
                let base = if name == "Categorical" { 1 } else { 0 };
                (0..n).map(|i| base + i).collect()
            })
        }
        // Binomial(n, p): atoms {0, …, n}, n+1 of them; n must be a static int.
        "Binomial" => {
            let n = static_int(m, kwarg(m, c, "n")?)?;
            // n+1 atoms (inclusive of 0 and n).
            bounded(n + 1).then(|| (0..=n).collect())
        }
        // Everything else — continuous (`Normal`, `Beta`, …) or infinite-discrete
        // (`Poisson`, `Geometric`, `NegativeBinomial*`, `Categorical` with a
        // dynamic `p`) — is not enumerable.
        _ => None,
    }
}

/// `true` iff the count `n` is a usable finite enumeration bound: positive and
/// not larger than [`MAX_ATOMS`].
fn bounded(n: i64) -> bool {
    n > 0 && n <= MAX_ATOMS
}

/// Is the inferred support of `node` a proven subset of `want`? Conservative:
/// `false` when the value-set is missing or unproven.
fn support_subset_of(m: &Module, node: NodeId, want: &ValueSet) -> bool {
    m.valueset_of(node)
        .map(|vs| vs.subset_of(want))
        .unwrap_or(false)
}

/// The value node of a `%kwarg` named `name` on call `c`, if present.
fn kwarg(m: &Module, c: &Call, name: &str) -> Option<NodeId> {
    c.named
        .iter()
        .find(|na| m.resolve(na.name) == name)
        .map(|na| na.value)
}

/// If `id` (through one ref level) is a `vector(...)` literal, its statically
/// known element count; otherwise `None` (a dynamically-sized / non-literal `p`).
fn static_vector_len(m: &Module, id: NodeId) -> Option<i64> {
    let (resolved, _) = resolve_ref_one(m, id);
    let vec = expect_builtin_call(m, resolved, "vector")?;
    // A vector literal carries its elements as positional args.
    if vec.named.is_empty() {
        Some(vec.args.len() as i64)
    } else {
        None
    }
}

/// If `id` (through one ref level) is a static integer literal, its value.
fn static_int(m: &Module, id: NodeId) -> Option<i64> {
    let (resolved, _) = resolve_ref_one(m, id);
    match m.node(resolved) {
        Node::Lit(Scalar::Int(n)) => Some(*n),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Kernel identification + application
// ---------------------------------------------------------------------------

/// A resolved kernel: its reified body and the single boundary-input symbol that
/// `K(a)` substitutes.
struct Kernel {
    /// The reified body node (e.g. `record(y = draw(Normal(mu = z, …)))`).
    body: NodeId,
    /// The single boundary-input name (e.g. `z`); references to it inside the
    /// body are `(%ref self z)` (or `(%ref %local z)`), substituted by `K(a)`.
    input: Symbol,
}

/// Resolve `K` (the kchain's second argument) to a `kernelof(body, %specinputs([
/// (input, _)]))` with exactly one boundary input. Returns `None` for a kernel
/// with no / multiple boundary inputs (those need a record-shaped application we
/// do not yet emit) or any non-`kernelof` shape.
fn resolve_kernel(m: &Module, k_arg: NodeId) -> Option<Kernel> {
    let (resolved, _) = resolve_ref_one(m, k_arg);
    let Node::Call(c) = m.node(resolved) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    if m.resolve(sym) != "kernelof" || c.args.len() != 1 {
        return None;
    }
    let body = c.args[0];
    // Read the single boundary input from the `%specinputs` list.
    let input = match &c.inputs {
        Some(flatppl_core::Inputs::Spec(entries)) if entries.len() == 1 => entries[0].0,
        _ => return None,
    };
    Some(Kernel { body, input })
}

/// Replace every `(%ref self name)` / `(%ref %local name)` in the subtree rooted
/// at `root` with `new_id`, returning the (possibly new) root. Append-only: nodes
/// that need no rewrite are reused; only rewritten `Call` ancestors are realloc'd.
///
/// This is `K(a)`: it pins the kernel's boundary input to the atom value `a`
/// inside a fresh copy of the kernel body, so the body's draws score against the
/// pinned latent and no reference to the (now marginalized) latent survives.
///
/// INVARIANT: this is scope-UNAWARE — it rewrites *every* matching `(%ref … name)`
/// in the subtree, with no notion of an inner binder shadowing `name`. Sound only
/// under the workspace no-shadowing assumption: the kernel-body substitution only
/// ever targets the single boundary-input symbol, which is never rebound inside
/// the body.
fn substitute_ref(m: &mut Module, root: NodeId, name: Symbol, new_id: NodeId) -> NodeId {
    if let Node::Ref(Ref { ns, name: rname }) = m.node(root) {
        if matches!(ns, RefNs::SelfMod | RefNs::Local) && *rname == name {
            return new_id;
        }
    }

    let children: Vec<NodeId> = m.node(root).children();
    if children.is_empty() {
        return root;
    }
    let new_children: Vec<NodeId> = children
        .iter()
        .map(|&c| substitute_ref(m, c, name, new_id))
        .collect();
    if new_children == children {
        return root;
    }

    // Rebuild the Call node with substituted children. Only `Call` nodes have
    // children, so this is always a Call.
    let Node::Call(orig) = m.node(root) else {
        unreachable!("non-call node with children is impossible in this IR");
    };
    let head = orig.head;
    let inputs = orig.inputs.clone();
    let n_args = orig.args.len();

    let (new_head, slice) = match head {
        // children = [callee, args…, named-values…]
        CallHead::User(_) => (CallHead::User(new_children[0]), &new_children[1..]),
        CallHead::Builtin(s) => (CallHead::Builtin(s), &new_children[..]),
    };
    let new_args: Vec<NodeId> = slice[..n_args].to_vec();
    let new_named_values = &slice[n_args..];
    let new_named: Vec<NamedArg> = orig
        .named
        .iter()
        .zip(new_named_values.iter())
        .map(|(na, &val)| NamedArg {
            kind: na.kind,
            name: na.name,
            value: val,
        })
        .collect();

    m.alloc(Node::Call(Call {
        head: new_head,
        args: new_args.into(),
        named: new_named.into(),
        inputs,
    }))
}

// ---------------------------------------------------------------------------
// Continuous latent: closed-form conjugate marginal
// ---------------------------------------------------------------------------
//
// A `kchain(prior, K)` whose latent is *continuous* has no discrete enumeration.
// For a handful of conjugate prior/likelihood pairs the marginal integral
// `∫ densityof(K(a), x) dprior(a)` collapses to a single closed-form
// distribution — e.g. the Normal–Normal (mean) pair
//
// ```text
// ∫ N(y; μ, σ)·N(μ; μ0, σ0) dμ = N(y; μ0, sqrt(σ0² + σ²)).
// ```
//
// This is a closed-form density rule for a SPECIFIC recognised shape, NOT
// general integration: a row matches only when the prior/likelihood
// constructors and the "conjugating" likelihood parameter (the one the latent
// feeds) line up exactly, and every OTHER likelihood parameter is
// latent-independent. Anything else keeps the non-enumerable refuse
// (refuse-don't-mislower).

/// A conjugate-marginal builder: from the prior's and likelihood's keyword
/// arguments (`(param, value)` pairs), build the closed-form marginal
/// distribution-constructor node, or `None` if a required parameter is absent.
type MarginalBuilder = fn(&mut Module, &[(Symbol, NodeId)], &[(Symbol, NodeId)]) -> Option<NodeId>;

/// One conjugate prior/likelihood pair whose `kchain` marginal is closed-form.
struct ConjugateRow {
    /// Prior distribution-constructor name (the latent's law), e.g. `"Normal"`.
    prior: &'static str,
    /// Likelihood distribution-constructor name (the kernel body), e.g. `"Normal"`.
    likelihood: &'static str,
    /// The likelihood parameter the latent feeds (the "conjugating" parameter),
    /// e.g. `"mu"`. Its value must be exactly the kernel's boundary-input ref.
    conjugating_param: &'static str,
    /// Build the closed-form marginal distribution-constructor node from the
    /// prior's keyword arguments and the likelihood's keyword arguments. Returns
    /// `None` if a required parameter is absent (a matched-but-malformed pair).
    build_marginal: MarginalBuilder,
}

/// The conjugate-marginal table. Data-driven and extensible: a new conjugate
/// pair is one more row (Gamma–Poisson is a follow-on task).
const CONJUGATE_TABLE: &[ConjugateRow] = &[ConjugateRow {
    prior: "Normal",
    likelihood: "Normal",
    conjugating_param: "mu",
    build_marginal: build_normal_normal_marginal,
}];

/// Try to lower a continuous-latent `kchain` as a closed-form conjugate marginal.
///
/// * `Some(Ok(node))` — a conjugate row matched and the marginal density was
///   emitted (a single `builtin_logdensityof` scoring the closed-form marginal at
///   the observation, through the kernel's variate wrapper).
/// * `Some(Err(..))` — a row matched but the pair is malformed (a required
///   distribution parameter is missing).
/// * `None` — no row matches; the caller falls through to the non-enumerable
///   refuse.
///
/// Detection (refuse-don't-mislower): a row matches only when (a) `latent_dist`
/// is the prior constructor, (b) the kernel body resolves to the likelihood
/// constructor whose conjugating-parameter value is *exactly* the kernel's
/// boundary-input ref, and (c) every OTHER likelihood parameter is
/// latent-independent (does not reference the boundary input).
fn try_conjugate_marginal(
    m: &mut Module,
    latent_dist: NodeId,
    kernel: &Kernel,
    v: NodeId,
) -> Option<Result<NodeId, RefuseError>> {
    // (a) The prior must be a bare distribution constructor (kwargs only).
    let (prior_sym, prior_kwargs) = split_kernel_constructor(m, latent_dist)?;
    let prior_name = m.resolve(prior_sym);

    // Resolve the likelihood constructor from the kernel body, remembering any
    // single-field record wrapper so the marginal is scored at the SAME variate
    // shape as the kernel (a record `{y}` vs. a bare scalar).
    let lik = resolve_likelihood(m, kernel.body)?;
    let (lik_sym, lik_kwargs) = split_kernel_constructor(m, lik.dist)?;
    let lik_name = m.resolve(lik_sym);

    // Find the row whose prior + likelihood families both match.
    let row = CONJUGATE_TABLE
        .iter()
        .find(|r| r.prior == prior_name && r.likelihood == lik_name)?;

    // (b) The conjugating parameter's value must be EXACTLY the boundary-input
    // ref `(%ref self|%local kernel.input)` — the latent feeding that parameter,
    // unresolved. Anything else (a constant, a derived expression) is not this
    // conjugate shape.
    let conj_val = find_kwarg(m, &lik_kwargs, row.conjugating_param)?;
    if !is_input_ref(m, conj_val, kernel.input) {
        return None;
    }

    // (c) Every OTHER likelihood parameter must be latent-independent. A second
    // parameter that also references the latent (e.g. both `mu` and `sigma`
    // depending on the latent) is not a Normal–Normal (mean-only) conjugacy.
    for (psym, pval) in &lik_kwargs {
        if m.resolve(*psym) == row.conjugating_param {
            continue;
        }
        if references_input(m, *pval, kernel.input) {
            return None;
        }
    }

    // Build the closed-form marginal distribution constructor.
    let marginal = match (row.build_marginal)(m, &prior_kwargs, &lik_kwargs) {
        Some(node) => node,
        None => {
            return Some(Err(refuse_kchain(
                latent_dist,
                "conjugate pair matched but a required distribution parameter is missing",
            )));
        }
    };

    // Score the marginal at `v` through the kernel's variate wrapper: for a
    // record-shaped kernel body this descends `record{field}` → scalar and scores
    // the marginal at `v.field`; for a bare scalar body it scores directly. Both
    // reach `build_density_term`, emitting one `builtin_logdensityof(marginal, …)`.
    let marginal_measure = wrap_like_kernel(m, marginal, lik.record_field);
    Some(lower_measure_density(m, marginal_measure, v))
}

/// The likelihood constructor resolved out of a kernel body, plus any
/// single-field `record(field = draw(dist))` wrapper around it.
struct Likelihood {
    /// The likelihood distribution-constructor node (e.g. `Normal(mu = z, …)`).
    dist: NodeId,
    /// `Some(field)` when the kernel body is `record(field = draw(dist))`; `None`
    /// for a bare scalar body. Drives how the marginal is scored at the variate.
    record_field: Option<Symbol>,
}

/// Resolve a kernel body to its likelihood distribution constructor, mirroring
/// how [`resolve_latent`] peels a latent measure: strip an optional single-field
/// `record(...)` wrapper, then an optional `draw(...)`, down to a builtin
/// distribution-constructor call. Returns `None` for any other shape.
fn resolve_likelihood(m: &Module, body: NodeId) -> Option<Likelihood> {
    let (resolved, _) = resolve_ref_one(m, body);

    // Optional single-field `record(field = X)` wrapper → remember the field.
    let (inner, record_field) = if let Some(rec) = expect_builtin_call(m, resolved, "record") {
        if !rec.args.is_empty() || rec.named.len() != 1 {
            return None;
        }
        let (val, _) = resolve_ref_one(m, rec.named[0].value);
        (val, Some(rec.named[0].name))
    } else {
        (resolved, None)
    };

    // Optional `draw(dist)` → dist.
    let dist = if let Some(inner_dist) = draw_argument(m, inner) {
        let (d, _) = resolve_ref_one(m, inner_dist);
        d
    } else {
        inner
    };

    // Must be a builtin distribution-constructor call.
    if !matches!(m.node(dist), Node::Call(c) if matches!(c.head, CallHead::Builtin(_))) {
        return None;
    }
    Some(Likelihood { dist, record_field })
}

/// Wrap a marginal distribution constructor in the kernel body's variate shape so
/// it can be scored by [`lower_measure_density`]: a `record(field = draw(marg))`
/// for a record-shaped kernel, or the bare constructor for a scalar kernel.
fn wrap_like_kernel(m: &mut Module, marginal: NodeId, record_field: Option<Symbol>) -> NodeId {
    match record_field {
        Some(field) => {
            let drawn = build_call(m, "draw", &[marginal]);
            build_record(m, &[(field, drawn)])
        }
        None => marginal,
    }
}

/// The value of the keyword argument `name` among `kwargs`, if present.
fn find_kwarg(m: &Module, kwargs: &[(Symbol, NodeId)], name: &str) -> Option<NodeId> {
    kwargs
        .iter()
        .find(|(sym, _)| m.resolve(*sym) == name)
        .map(|(_, val)| *val)
}

/// Is `node` exactly the boundary-input reference `(%ref self|%local input)` —
/// the latent feeding a parameter directly (not a derived expression)?
fn is_input_ref(m: &Module, node: NodeId, input: Symbol) -> bool {
    matches!(
        m.node(node),
        Node::Ref(Ref { ns, name })
            if matches!(ns, RefNs::SelfMod | RefNs::Local) && *name == input
    )
}

/// Does the subtree rooted at `node` reference the boundary input `input`
/// anywhere (a `(%ref self|%local input)`)? Used to prove a likelihood parameter
/// is latent-independent.
fn references_input(m: &Module, node: NodeId, input: Symbol) -> bool {
    if is_input_ref(m, node, input) {
        return true;
    }
    m.node(node)
        .children()
        .into_iter()
        .any(|child| references_input(m, child, input))
}

/// Allocate a distribution-constructor call `Ctor(param = value, …)` with only
/// keyword arguments — the shape [`split_kernel_constructor`] /
/// [`build_density_term`] consume.
fn build_constructor(m: &mut Module, ctor: &str, params: &[(&str, NodeId)]) -> NodeId {
    let mut named = Vec::with_capacity(params.len());
    for &(name, value) in params {
        let name = m.intern(name);
        named.push(NamedArg {
            kind: NamedKind::Kwarg,
            name,
            value,
        });
    }
    let head = m.intern(ctor);
    m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Vec::<NodeId>::new().into(),
        named: named.into(),
        inputs: None,
    }))
}

/// Normal–Normal (conjugate mean) marginal builder:
/// `Normal(mu = μ0, sigma = sqrt(add(pow(σ0, 2), pow(σ, 2))))` where `μ0`, `σ0`
/// are the prior's `mu`/`sigma` and `σ` is the likelihood's `sigma`.
fn build_normal_normal_marginal(
    m: &mut Module,
    prior_kwargs: &[(Symbol, NodeId)],
    lik_kwargs: &[(Symbol, NodeId)],
) -> Option<NodeId> {
    let mu0 = find_kwarg(m, prior_kwargs, "mu")?;
    let sigma0 = find_kwarg(m, prior_kwargs, "sigma")?;
    let sigma = find_kwarg(m, lik_kwargs, "sigma")?;

    // sqrt(add(pow(σ0, 2), pow(σ, 2))): the marginal stddev is the root of the
    // summed variances (prior + likelihood).
    let two_a = m.alloc(Node::Lit(Scalar::Real(2.0)));
    let var0 = build_call(m, "pow", &[sigma0, two_a]);
    let two_b = m.alloc(Node::Lit(Scalar::Real(2.0)));
    let var = build_call(m, "pow", &[sigma, two_b]);
    let var_sum = build_call(m, "add", &[var0, var]);
    let sigma_marginal = build_call(m, "sqrt", &[var_sum]);

    Some(build_constructor(
        m,
        "Normal",
        &[("mu", mu0), ("sigma", sigma_marginal)],
    ))
}

// ---------------------------------------------------------------------------
// Refusal
// ---------------------------------------------------------------------------

/// A refusal that names `kchain` with the given reason.
fn refuse_kchain(node: NodeId, reason: &str) -> RefuseError {
    RefuseError {
        node,
        construct: "kchain".to_string(),
        reason: reason.to_string(),
    }
}
