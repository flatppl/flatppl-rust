//! The `kchain` marginal-density rule, its discrete/continuous classifier, and the
//! closed-form conjugate table (spec §06, "Density of composed measures", the `kchain`
//! row).
//!
//! **The maths of every conjugate row — the integral, its closed form, the §08 name or the
//! emitted expression, the test point and the wrong answer that point discriminates
//! against — is in `marginal.md`, beside this file.** Check a row there rather than
//! reverse-engineering its `build_*_marginal`. Not every row has a §08 name: two answer
//! with a log-density expression, because §08 names no constructor for them.
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
//! static error." This module implements two of the three branches and refuses the
//! rest:
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
//! - **A recognised conjugate prior/likelihood pair** (`CONJUGATE_TABLE`), whose
//!   integral collapses to one closed-form distribution. Continuity is no obstacle
//!   here: the Normal–Normal row integrates a `Normal` latent exactly.
//!
//! - **Any other continuous or infinite-discrete latent** (`Normal` on a scale
//!   parameter, `Poisson` with no matching row, …): **refused**.
//!
//! Both spellings of the integral reach the table. `kchain(prior, kernelof(…))` is the
//! explicit one; `lawof(y)` over a `y ~ Dist(param = z, …)` with `z` latent is the
//! implicit one §04 "Reification to measures" defines as the same marginal, routed here
//! from `density.rs`'s marginalization guard by [`conjugate_marginal_measure`].
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
    Ancestor, build_call, build_density_term, build_record, draw_argument, expect_builtin_call,
    lower_measure_density, measure_stochastic_ancestors, refuse, resolve_ref_one,
    split_kernel_constructor,
};
use crate::kernel::{Kernel, resolve_kernel, substitute_ref};
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
    // The kchain marginal substitutes the enumerated latent's atoms into
    // `kernel.inputs[0]`, ASSUMING that boundary input IS the latent dependency —
    // which holds only for a `%specinputs` boundary. An `%autoinputs` boundary
    // traces the reified body's `elementof` FREE parameters (never the
    // `draw`-bound latent), so substituting the latent's atoms into it would
    // replace the WRONG node — a free parameter that must stay symbolic — and
    // emit a `logsumexp` that silently eliminates it (a wrong density
    // `is_flatpdl` cannot catch). Refuse rather than mislower.
    if kernel.auto {
        return Err(refuse_kchain(
            node,
            "kchain marginal over an %autoinputs kernel: the auto-traced boundary is a free \
             parameter, not the enumerated latent — refuse rather than mislower",
        ));
    }
    // The kchain marginal path substitutes ONE latent into the kernel; a
    // multi-input kernel is not the single-step shape it handles.
    if kernel.inputs.len() != 1 {
        return Err(refuse_kchain(
            node,
            "single-input kernel only; multi-input kchain kernels are out of scope",
        ));
    }
    let kernel_input_sym = kernel.inputs[0].1.name;

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
        let applied_body = substitute_ref(m, kernel.body, kernel_input_sym, atom_node);
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

/// The closed form a conjugate row returns.
///
/// A row does **not** have to name a §08 distribution. Determinised output is a
/// deterministic expression, so a row whose answer §08 has no constructor for emits the
/// closed-form log-density instead: the beta-binomial pmf (Row 3) and the scaled Student t
/// (Row 5) are both that case, and both are built from §07 builtins only.
pub(crate) enum MarginalForm {
    /// A §08 distribution-constructor node, scored by the ordinary density path.
    Measure(NodeId),
    /// A closed-form log-density, applied to the variate by [`LogDensity::at`].
    LogDensity(LogDensity),
}

/// A row's closed-form log-density, resolved down to its parameter nodes and waiting for
/// the variate.
///
/// Not a boxed closure: the parameters travel as a `Vec` in the order the row's `build`
/// reads them, so the builder stays a plain `fn` pointer like [`MarginalBuilder`].
pub(crate) struct LogDensity {
    build: fn(&mut Module, &[NodeId], &[NodeId]) -> NodeId,
    params: Vec<NodeId>,
}

impl LogDensity {
    /// The log-density expression at `variates` — ONE variate for a `CONJUGATE_TABLE` row,
    /// whose law is over a single variate, and one per record field for the shared-latent
    /// record law ([`shared_latent_record_law`]), whose law is over all of them jointly.
    pub(crate) fn at(&self, m: &mut Module, variates: &[NodeId]) -> NodeId {
        (self.build)(m, &self.params, variates)
    }
}

/// The sole variate of a scalar row's law. A row that reads its variate through this cannot
/// silently score the wrong one if a caller ever hands it a record's whole field list.
fn sole_variate(variates: &[NodeId]) -> NodeId {
    match variates {
        [v] => *v,
        _ => unreachable!("a CONJUGATE_TABLE row's law is over exactly one variate"),
    }
}

/// How the latent reaches the conjugating parameter.
///
/// The row records this because §08 parameterizes by the quantity it names, not by
/// whatever a conjugacy is stated in: `Normal` takes `sigma`, so a prior on the VARIANCE
/// reaches it through a `sqrt`. A row that accepted a bare ref where it wanted `sqrt`
/// would score a LOCATION mixture as a scale mixture, and those two agree to 0.017 nats
/// at `y = 0.5` (`marginal.md`, Rows 4 and 5).
enum LatentPath {
    /// The parameter's value IS the latent's ref.
    Direct,
    /// The parameter's value is `sqrt(<the latent's ref>)`.
    Sqrt,
}

/// A conjugate-marginal builder: from the prior's and likelihood's keyword
/// arguments (`(param, value)` pairs), build the closed form, or `None` if a required
/// parameter is absent.
type MarginalBuilder =
    fn(&mut Module, &[(Symbol, NodeId)], &[(Symbol, NodeId)]) -> Option<MarginalForm>;

/// One conjugate prior/likelihood pair whose `kchain` marginal is closed-form.
struct ConjugateRow {
    /// Prior distribution-constructor name (the latent's law), e.g. `"Normal"`.
    prior: &'static str,
    /// Likelihood distribution-constructor name (the kernel body), e.g. `"Normal"`.
    likelihood: &'static str,
    /// The likelihood parameter the latent feeds (the "conjugating" parameter),
    /// e.g. `"mu"`.
    conjugating_param: &'static str,
    /// The transformation the latent passes through on its way to that parameter.
    latent_path: LatentPath,
    /// Build the closed form from the prior's keyword arguments and the likelihood's
    /// keyword arguments. Returns `None` if a required parameter is absent (a
    /// matched-but-malformed pair).
    build_marginal: MarginalBuilder,
}

/// The conjugate-marginal table. Data-driven and extensible: a new conjugate
/// pair is one more row.
const CONJUGATE_TABLE: &[ConjugateRow] = &[
    ConjugateRow {
        prior: "Normal",
        likelihood: "Normal",
        conjugating_param: "mu",
        latent_path: LatentPath::Direct,
        build_marginal: build_normal_normal_marginal,
    },
    ConjugateRow {
        prior: "Gamma",
        likelihood: "Poisson",
        conjugating_param: "rate",
        latent_path: LatentPath::Direct,
        build_marginal: build_gamma_poisson_marginal,
    },
    ConjugateRow {
        prior: "Beta",
        likelihood: "Binomial",
        conjugating_param: "p",
        latent_path: LatentPath::Direct,
        build_marginal: build_beta_binomial_marginal,
    },
    ConjugateRow {
        prior: "Exponential",
        likelihood: "Normal",
        conjugating_param: "sigma",
        latent_path: LatentPath::Sqrt,
        build_marginal: build_exponential_variance_marginal,
    },
    ConjugateRow {
        prior: "InverseGamma",
        likelihood: "Normal",
        conjugating_param: "sigma",
        latent_path: LatentPath::Sqrt,
        build_marginal: build_inverse_gamma_variance_marginal,
    },
];

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
    // Resolve the likelihood constructor from the kernel body, remembering any
    // single-field record wrapper so the marginal is scored at the SAME variate
    // shape as the kernel (a record `{y}` vs. a bare scalar).
    let lik = resolve_likelihood(m, kernel.body)?;
    // `try_conjugate_marginal` is only reached after the caller's single-input
    // length check, so the single kernel input here is guaranteed.
    let kernel_input_sym = kernel.inputs[0].1.name;

    let marginal = match build_conjugate_marginal(m, latent_dist, lik.dist, kernel_input_sym)? {
        Ok(form) => form,
        Err(e) => return Some(Err(e)),
    };

    match marginal {
        // Score the marginal at `v` through the kernel's variate wrapper: for a
        // record-shaped kernel body this descends `record{field}` → scalar and scores
        // the marginal at `v.field`; for a bare scalar body it scores directly. Both
        // reach `build_density_term`, emitting one `builtin_logdensityof(marginal, …)`.
        MarginalForm::Measure(marginal) => {
            let marginal_measure = wrap_like_kernel(m, marginal, lik.record_field);
            Some(lower_measure_density(m, marginal_measure, v))
        }
        // A log-density form has no measure to wrap, so the kernel's variate wrapper is
        // applied to the VALUE instead: the row's expression consumes the same scalar the
        // measure path would have descended to.
        MarginalForm::LogDensity(ld) => match variate_like_kernel(m, v, lik.record_field) {
            Some(scalar) => Some(Ok(ld.at(m, &[scalar]))),
            None => Some(Err(refuse_kchain(
                v,
                "closed-form log-density marginal over a record-shaped kernel needs the \
                 variate's field, and this variate is not a record literal carrying it",
            ))),
        },
    }
}

/// The scalar the kernel's variate wrapper descends `v` to: `v` itself for a bare scalar
/// kernel body, or `v`'s matching field for a `record(field = …)` one.
///
/// The field is read from a record LITERAL rather than emitted as a `get`, mirroring what
/// [`lower_measure_density`]'s record descent does on the measure path — so the two forms
/// score at the same node and neither leaves a `get` over a literal behind.
fn variate_like_kernel(m: &Module, v: NodeId, record_field: Option<Symbol>) -> Option<NodeId> {
    let Some(field) = record_field else {
        return Some(v);
    };
    let (resolved, _) = resolve_ref_one(m, v);
    let rec = expect_builtin_call(m, resolved, "record")?;
    rec.named
        .iter()
        .find(|na| na.name == field)
        .map(|na| na.value)
}

/// The conjugate table's detection + build, shared by the explicit `kchain` spelling
/// ([`try_conjugate_marginal`]) and the implicit `lawof` one
/// ([`conjugate_marginal_measure`]). `latent_sym` is the symbol the likelihood
/// references the latent by — a `kernelof` boundary input for the explicit spelling,
/// the latent's own binding name for the implicit one.
///
/// * `None` — no row matches (the caller keeps its own refusal).
/// * `Some(Err(..))` — a row matched but a required distribution parameter is missing.
/// * `Some(Ok(node))` — the closed-form marginal distribution-constructor node.
fn build_conjugate_marginal(
    m: &mut Module,
    latent_dist: NodeId,
    lik_dist: NodeId,
    latent_sym: Symbol,
) -> Option<Result<MarginalForm, RefuseError>> {
    // (a) The prior must be a bare distribution constructor (positional or
    // keyword arguments).
    let (prior_sym, prior_kwargs) = split_kernel_constructor(m, latent_dist)?;
    let (lik_sym, lik_kwargs) = split_kernel_constructor(m, lik_dist)?;

    // Resolve the constructor names to owned strings (the `split_*` calls above
    // borrow `m` mutably to intern positional-arg names, so we cannot hold a
    // `resolve` borrow across them).
    let prior_name = m.resolve(prior_sym).to_string();
    let lik_name = m.resolve(lik_sym).to_string();

    // EVERY row whose prior + likelihood families match is tried, not just the first: two
    // rows may share a family pair and differ only in which parameter the latent feeds, or
    // along which path it gets there (`Normal`–`Normal` on `mu` vs. on `sqrt`-of-variance).
    for row in CONJUGATE_TABLE
        .iter()
        .filter(|r| r.prior == prior_name.as_str() && r.likelihood == lik_name.as_str())
    {
        // (b) The conjugating parameter's value must carry the latent's ref
        // `(%ref self|%local latent_sym)` along the row's own path — bare for `Direct`,
        // under a `sqrt` for `Sqrt`. Anything else (a constant, some other derived
        // expression) is not this conjugate shape.
        let Some(conj_val) = find_kwarg(m, &lik_kwargs, row.conjugating_param) else {
            continue;
        };
        if !matches_latent_path(m, conj_val, &row.latent_path, latent_sym) {
            continue;
        }

        // (c) Every OTHER likelihood parameter must be latent-independent. A second
        // parameter that also references the latent (e.g. both `mu` and `sigma`
        // depending on the latent) is not a Normal–Normal (mean-only) conjugacy.
        if lik_kwargs.iter().any(|(psym, pval)| {
            m.resolve(*psym) != row.conjugating_param && references_input(m, *pval, latent_sym)
        }) {
            continue;
        }

        return Some(
            (row.build_marginal)(m, &prior_kwargs, &lik_kwargs).ok_or_else(|| {
                refuse_kchain(
                    latent_dist,
                    "conjugate pair matched but a required distribution parameter is missing",
                )
            }),
        );
    }
    None
}

/// Does the conjugating parameter's value `value` carry the latent along `path`?
fn matches_latent_path(m: &Module, value: NodeId, path: &LatentPath, latent: Symbol) -> bool {
    match path {
        LatentPath::Direct => is_input_ref(m, value, latent),
        // The ref resolution is DEFENSIVE, and currently unreachable: it would admit a named
        // intermediate (`s = sqrt(v)`; `sigma = s`), but that whole shape refuses earlier, at
        // the driver's residual-`draw` scan — `s` keeps referencing `v`, so `v = draw(…)` is
        // never swept and survives to exit. Kept rather than dropped so the arm still matches
        // if that upstream refusal is ever lifted; pinned as a refusal by
        // `a_named_sqrt_intermediate_refuses_upstream_not_in_the_row`, which records that the
        // refusal is not this row's.
        LatentPath::Sqrt => {
            let (resolved, _) = resolve_ref_one(m, value);
            match expect_builtin_call(m, resolved, "sqrt") {
                Some(c) if c.args.len() == 1 && c.named.is_empty() => {
                    is_input_ref(m, c.args[0], latent)
                }
                _ => false,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The implicit `lawof` spelling
// ---------------------------------------------------------------------------

/// A marginal this module built for the implicit spelling: the closed form, and the latent
/// it integrated out.
///
/// The `latent` is not diagnostics. A caller building a PRODUCT of several marginals must
/// know which latent each one integrated: two marginals over the SAME latent are not
/// independent, and adding their log-densities is a wrong number
/// ([`crate::density::marginalize_or_refuse_record_law`]).
pub(crate) struct ImplicitMarginal {
    pub form: MarginalForm,
    pub latent: Symbol,
}

/// The closed-form conjugate marginal of `likelihood` — a distribution constructor one
/// of whose parameters is a latent of the model — as a distribution-constructor node plus
/// the latent it integrated out.
///
/// The implicit spelling of the integral [`lower_kchain_marginal`] does for an explicit
/// `kchain`. §04 "Reification to measures" makes `lawof(y)` y's TOTAL law — internal
/// stochastic nodes are "not boundary inputs, so `lawof` integrates them out" — so
/// `y ~ Normal(mu = z, sigma = 1)` over a latent `z` asks for the Normal–Normal row
/// without the user writing the `kchain`. `None` means no row applies, never that the
/// conditional may be scored instead.
///
/// `exempt` passes the caller's own accounted-for draw sites through to
/// [`measure_stochastic_ancestors`] — the sibling fields of a record product, whose
/// dependence is the chain rule, not a marginal.
///
/// **One latent, whose prior is itself ancestor-free.** Two latents feeding two
/// parameters is not a single conjugate integral, and neither is a two-level hierarchy
/// (`z ~ Normal(mu = w, …)` with `w` latent) — the row's closed form integrates ONE
/// prior. Both return `None` rather than a marginal that silently conditions on the
/// ancestor it did not integrate.
///
/// **A marginal is correct only for the ONE variate it is the law of.** This returns a
/// marginal, never a licence to multiply marginals together: the returned [`ImplicitMarginal`]
/// carries the latent so a caller assembling a product can refuse when two of them share it.
pub(crate) fn conjugate_marginal_measure(
    m: &mut Module,
    likelihood: NodeId,
    exempt: &[NodeId],
) -> Option<Result<ImplicitMarginal, RefuseError>> {
    let (latent, prior) = sole_named_ancestor(m, likelihood, exempt)?;
    if !measure_stochastic_ancestors(m, prior, &[]).is_empty() {
        return None;
    }
    Some(
        build_conjugate_marginal(m, prior, likelihood, latent)?
            .map(|form| ImplicitMarginal { form, latent }),
    )
}

/// The single named stochastic ancestor of `likelihood`, as `(symbol, prior)`. `None`
/// when there is none, when two DISTINCT latents are reached, or when any ancestor has
/// no recoverable `(name, prior)` pair ([`Ancestor::Opaque`]) — each of which is outside
/// the table's one-prior integral.
fn sole_named_ancestor(
    m: &Module,
    likelihood: NodeId,
    exempt: &[NodeId],
) -> Option<(Symbol, NodeId)> {
    let mut found: Option<(Symbol, NodeId)> = None;
    for ancestor in measure_stochastic_ancestors(m, likelihood, exempt) {
        let Ancestor::Named { name, prior } = ancestor else {
            return None;
        };
        match found {
            None => found = Some((name, prior)),
            // The same latent feeding two parameters is still one ancestor; check (c)
            // in `build_conjugate_marginal` is what rejects that shape.
            Some((seen, _)) if seen == name => {}
            Some(_) => return None,
        }
    }
    found
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

/// A real literal node.
fn real(m: &mut Module, v: f64) -> NodeId {
    m.alloc(Node::Lit(Scalar::Real(v)))
}

/// `log B(x, y) = loggamma(x) + loggamma(y) − loggamma(x + y)`. `loggamma` is a §07
/// builtin ("Elementary functions", domain `posreals`); §07 names no log-beta.
fn build_logbeta(m: &mut Module, x: NodeId, y: NodeId) -> NodeId {
    let lx = build_call(m, "loggamma", &[x]);
    let ly = build_call(m, "loggamma", &[y]);
    let sum = build_call(m, "add", &[x, y]);
    let lxy = build_call(m, "loggamma", &[sum]);
    let num = build_call(m, "add", &[lx, ly]);
    build_call(m, "sub", &[num, lxy])
}

/// Normal–Normal (conjugate mean) marginal builder:
/// `Normal(mu = μ0, sigma = sqrt(add(pow(σ0, 2), pow(σ, 2))))` where `μ0`, `σ0`
/// are the prior's `mu`/`sigma` and `σ` is the likelihood's `sigma`.
fn build_normal_normal_marginal(
    m: &mut Module,
    prior_kwargs: &[(Symbol, NodeId)],
    lik_kwargs: &[(Symbol, NodeId)],
) -> Option<MarginalForm> {
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

    Some(MarginalForm::Measure(build_constructor(
        m,
        "Normal",
        &[("mu", mu0), ("sigma", sigma_marginal)],
    )))
}

/// Gamma–Poisson (conjugate rate) marginal builder:
/// `NegativeBinomial(alpha, beta)` (§08) IS the Gamma(shape=α, rate=β)–
/// Poisson(rate=λ) mixture `∫ Poisson(N; λ)·Gamma(λ; α, β) dλ`, so the marginal
/// is an IDENTITY parameter map — `alpha`/`beta` are the prior's `shape`/`rate`
/// value nodes, reused unchanged, no arithmetic.
fn build_gamma_poisson_marginal(
    m: &mut Module,
    prior_kwargs: &[(Symbol, NodeId)],
    _lik_kwargs: &[(Symbol, NodeId)],
) -> Option<MarginalForm> {
    let alpha = find_kwarg(m, prior_kwargs, "shape")?;
    let beta = find_kwarg(m, prior_kwargs, "rate")?;

    Some(MarginalForm::Measure(build_constructor(
        m,
        "NegativeBinomial",
        &[("alpha", alpha), ("beta", beta)],
    )))
}

/// Beta–Binomial (conjugate success probability) marginal builder. §08 names no
/// `BetaBinomial` constructor, so the row returns the log-pmf
/// ([`build_beta_binomial_logpmf`]) rather than a measure.
fn build_beta_binomial_marginal(
    m: &mut Module,
    prior_kwargs: &[(Symbol, NodeId)],
    lik_kwargs: &[(Symbol, NodeId)],
) -> Option<MarginalForm> {
    let alpha = find_kwarg(m, prior_kwargs, "alpha")?;
    let beta = find_kwarg(m, prior_kwargs, "beta")?;
    let n = find_kwarg(m, lik_kwargs, "n")?;

    Some(MarginalForm::LogDensity(LogDensity {
        build: build_beta_binomial_logpmf,
        params: vec![alpha, beta, n],
    }))
}

/// The beta-binomial log-pmf at `k`, from `params = [α, β, n]`:
///
/// ```text
/// log C(n, k) + log B(k + α, n − k + β) − log B(α, β)
/// log C(n, k) = loggamma(n+1) − loggamma(k+1) − loggamma(n−k+1)
/// ```
fn build_beta_binomial_logpmf(m: &mut Module, params: &[NodeId], variates: &[NodeId]) -> NodeId {
    let k = sole_variate(variates);
    let [alpha, beta, n] = [params[0], params[1], params[2]];

    let one = real(m, 1.0);
    let n1 = build_call(m, "add", &[n, one]);
    let k1 = build_call(m, "add", &[k, one]);
    // `n − k` serves both the binomial coefficient and the posterior beta's second shape.
    let nk = build_call(m, "sub", &[n, k]);
    let nk1 = build_call(m, "add", &[nk, one]);
    let lg_n1 = build_call(m, "loggamma", &[n1]);
    let lg_k1 = build_call(m, "loggamma", &[k1]);
    let lg_nk1 = build_call(m, "loggamma", &[nk1]);
    let coeff = build_call(m, "sub", &[lg_n1, lg_k1]);
    let coeff = build_call(m, "sub", &[coeff, lg_nk1]);

    let post_a = build_call(m, "add", &[k, alpha]);
    let post_b = build_call(m, "add", &[nk, beta]);
    let post = build_logbeta(m, post_a, post_b);
    let prior = build_logbeta(m, alpha, beta);
    let ratio = build_call(m, "sub", &[post, prior]);

    build_call(m, "add", &[coeff, ratio])
}

/// Exponential-prior-on-the-VARIANCE marginal builder:
/// `Laplace(location = μ, scale = 1/sqrt(2λ))` where `λ` is the prior's `rate` and `μ` the
/// likelihood's `mu`. §08 parameterizes `Exponential` by **rate**, so a prior of mean
/// `2b²` is `rate = 1/(2b²)` and the map inverts that to `b`.
fn build_exponential_variance_marginal(
    m: &mut Module,
    prior_kwargs: &[(Symbol, NodeId)],
    lik_kwargs: &[(Symbol, NodeId)],
) -> Option<MarginalForm> {
    let rate = find_kwarg(m, prior_kwargs, "rate")?;
    let mu = find_kwarg(m, lik_kwargs, "mu")?;

    let two = real(m, 2.0);
    let two_rate = build_call(m, "mul", &[two, rate]);
    let root = build_call(m, "sqrt", &[two_rate]);
    let one = real(m, 1.0);
    let scale = build_call(m, "divide", &[one, root]);

    Some(MarginalForm::Measure(build_constructor(
        m,
        "Laplace",
        &[("location", mu), ("scale", scale)],
    )))
}

/// InverseGamma-prior-on-the-VARIANCE marginal builder. The answer is the scaled Student t
/// with location `μ`, scale `sqrt(β/α)` and `ν = 2α`; §08's `StudentT(nu)` is the standard
/// form only ("The location-scale form is obtained via `pushfwd(fn(mu + sigma * _),
/// StudentT(nu))`"), and a `pushfwd` is not a bare constructor, so the row returns the
/// log-density ([`build_scaled_t_logpdf`]).
///
/// §08 parameterizes `InverseGamma` by `shape`/`scale`, and its `scale` "plays the same
/// numerical role as the `rate` parameter of `Gamma`" — the `β` in `e^{-β/x}`. So the map
/// reads the prior's `scale` as β, NOT as a multiplicative scale.
fn build_inverse_gamma_variance_marginal(
    m: &mut Module,
    prior_kwargs: &[(Symbol, NodeId)],
    lik_kwargs: &[(Symbol, NodeId)],
) -> Option<MarginalForm> {
    let shape = find_kwarg(m, prior_kwargs, "shape")?;
    let scale = find_kwarg(m, prior_kwargs, "scale")?;
    let mu = find_kwarg(m, lik_kwargs, "mu")?;

    Some(MarginalForm::LogDensity(LogDensity {
        build: build_scaled_t_logpdf,
        params: vec![shape, scale, mu],
    }))
}

/// The scaled Student t log-density at `y`, from `params = [α, β, μ]` — location `μ`,
/// scale `s = sqrt(β/α)`, `ν = 2α`:
///
/// ```text
/// −[ log s + log sqrt(ν) + log B(ν/2, 1/2) + ((ν+1)/2)·log1p(z²/ν) ],   z = (y − μ)/s
/// ```
///
/// The normalizer is written with `log B(ν/2, 1/2)` rather than the gamma ratio and
/// `log(νπ)/2` because `B(ν/2, 1/2) = Γ(ν/2)Γ(1/2)/Γ((ν+1)/2)` absorbs the `Γ(1/2) = √π`,
/// so no `pi` constant is needed and [`build_logbeta`] is reused.
fn build_scaled_t_logpdf(m: &mut Module, params: &[NodeId], variates: &[NodeId]) -> NodeId {
    let y = sole_variate(variates);
    let [shape, beta, mu] = [params[0], params[1], params[2]];

    let two = real(m, 2.0);
    let nu = build_call(m, "mul", &[two, shape]);
    let ratio = build_call(m, "divide", &[beta, shape]);
    let s = build_call(m, "sqrt", &[ratio]);

    let one = real(m, 1.0);
    let nu1 = build_call(m, "add", &[nu, one]);
    let half_nu1 = build_call(m, "divide", &[nu1, two]);
    let half_nu = build_call(m, "divide", &[nu, two]);
    let half = real(m, 0.5);

    let log_s = build_call(m, "log", &[s]);
    let root_nu = build_call(m, "sqrt", &[nu]);
    let log_root_nu = build_call(m, "log", &[root_nu]);
    let logb = build_logbeta(m, half_nu, half);
    let norm = build_call(m, "add", &[log_s, log_root_nu]);
    let norm = build_call(m, "add", &[norm, logb]);

    let dev = build_call(m, "sub", &[y, mu]);
    let z = build_call(m, "divide", &[dev, s]);
    let z2 = build_call(m, "pow", &[z, two]);
    let z2_nu = build_call(m, "divide", &[z2, nu]);
    let shaped = build_call(m, "log1p", &[z2_nu]);
    let tail = build_call(m, "mul", &[half_nu1, shaped]);

    let total = build_call(m, "add", &[norm, tail]);
    build_call(m, "neg", &[total])
}

// ---------------------------------------------------------------------------
// The shared-latent record law
// ---------------------------------------------------------------------------

/// `log(2π)`, the Gaussian normalizer's constant. Hardcoded rather than computed from a
/// `PI` const so the emitted literal is the same on every platform: `canon::fold` folds
/// `mul(N, LOG_2PI)` into the output, and `f64::ln` is not rounding-mandated.
/// `log_2pi_matches_the_computed_constant` pins it against `(2·π).ln()`.
const LOG_2PI: f64 = 1.8378770664093453;

/// The joint law of a record whose fields all draw `Normal(mu = z, sigma = σᵢ)` over ONE
/// shared latent `z ~ Normal(μ₀, s₀)`.
///
/// Not a `CONJUGATE_TABLE` row: a row's closed form is the law of ONE variate, and this law
/// is over all N of them jointly. That is the whole point — each field's own Row 1 marginal
/// is correct, and their product is a different measure, because
/// `Cov(yᵢ, yⱼ) = Var(z) = s₀²`. `marginal.md`'s *The shared-latent record law* has the
/// derivation, the emitted expression and the pinned points.
pub(crate) struct SharedLatentRecordLaw {
    /// The shared latent this law integrated out, so the caller can report it.
    pub latent: Symbol,
    /// How many fields the law was built over, so the emitter can prove it has one variate
    /// per sigma rather than `zip` a mismatch down to the shorter list.
    pub field_count: usize,
    /// The joint log-density, awaiting one variate per field in the SAME order the field
    /// measures were supplied in. §08 names no constructor for it, and a §08 `MvNormal`
    /// would force the record variate into a vector, so the form is an expression (§13
    /// admits "any other **deterministic expression** over the inputs").
    pub form: LogDensity,
}

/// Recognise the shared-latent record law over `field_measures` — the record's per-field
/// distribution-constructor nodes, in field order — or `None`.
///
/// `exempt` is the caller's own accounted-for draw sites (the record's sibling fields),
/// threaded to [`measure_stochastic_ancestors`] exactly as
/// [`conjugate_marginal_measure`] threads it: a latent the record CARRIES is scored by the
/// chain rule, not integrated, so only an uncarried ancestor reaches this law.
///
/// The shape is narrow on purpose, and every rejection here keeps the caller's
/// shared-latent refusal (refuse-don't-mislower):
///
/// * at least TWO fields — one field is Row 1, which already lowers, and routing it here
///   would change output for no gain;
/// * every field's measure is `Normal`, its `mu` EXACTLY the shared latent's ref
///   ([`LatentPath::Direct`] — a derived `mul(2, z)` mean has a different joint), and its
///   `sigma` latent-independent;
/// * ONE shared latent across all fields, whose prior is a `Normal` with both parameters
///   and is itself ancestor-free (a two-level hierarchy is not this integral).
///
/// **The checks are deliberately over-determined, and most are not individually reachable
/// today.** Only a shape whose every field ALREADY matched a `CONJUGATE_TABLE` row gets
/// here — the caller detects the repeated latent from those rows — so the caller's upstream
/// filtering happens to imply several of them. Verified by mutation: removing any ONE of the
/// family, path and cross-field-agreement checks reddens nothing, and removing the mean-path
/// and latent-agreement checks TOGETHER lets a partly-shared record mislower.
///
/// **The masking mechanism is `find_kwarg`, not the family check.** `CONJUGATE_TABLE` admits
/// the priors `Normal`, `Gamma`, `Beta`, `Exponential` and `InverseGamma`, and of the
/// non-`Normal` four NONE takes both `mu` and `sigma` — so `find_kwarg(prior_kwargs, "mu")?`
/// turns them away before `prior_sym` is ever compared. The field-family check is masked the
/// same way (`Poisson` takes `rate`, `Binomial` takes `n`/`p`). The one shape that does reach
/// here with a non-`Normal` prior is Rows 4 and 5's shared VARIANCE, and it is turned away
/// twice — by `find_kwarg("mu")` on the prior and, independently, by the `Direct` mean path.
///
/// That masking is an accident of the current table, not a guarantee, which is why the
/// explicit checks stay. Two changes would make them live, and each needs a probe that
/// isolates the guard it turns on: a §08 prior taking both `mu` and `sigma` (`LogNormal` is
/// the candidate), or a second Normal-mean row.
pub(crate) fn shared_latent_record_law(
    m: &mut Module,
    field_measures: &[NodeId],
    exempt: &[NodeId],
) -> Option<SharedLatentRecordLaw> {
    if field_measures.len() < 2 {
        return None;
    }

    // One shared latent across every field. `sole_named_ancestor` already rejects a field
    // reaching two DISTINCT latents, so this only has to check the fields agree.
    let (latent, prior) = sole_named_ancestor(m, field_measures[0], exempt)?;
    for &measure in &field_measures[1..] {
        let (other, other_prior) = sole_named_ancestor(m, measure, exempt)?;
        if other != latent || other_prior != prior {
            return None;
        }
    }
    if !measure_stochastic_ancestors(m, prior, &[]).is_empty() {
        return None;
    }

    let (prior_sym, prior_kwargs) = split_kernel_constructor(m, prior)?;
    if m.resolve(prior_sym) != "Normal" {
        return None;
    }
    let mu0 = find_kwarg(m, &prior_kwargs, "mu")?;
    let s0 = find_kwarg(m, &prior_kwargs, "sigma")?;

    let mut sigmas = Vec::with_capacity(field_measures.len());
    for &measure in field_measures {
        let (lik_sym, lik_kwargs) = split_kernel_constructor(m, measure)?;
        if m.resolve(lik_sym) != "Normal" {
            return None;
        }
        let mu = find_kwarg(m, &lik_kwargs, "mu")?;
        if !matches_latent_path(m, mu, &LatentPath::Direct, latent) {
            return None;
        }
        let sigma = find_kwarg(m, &lik_kwargs, "sigma")?;
        if references_input(m, sigma, latent) {
            return None;
        }
        sigmas.push(sigma);
    }

    let mut params = vec![mu0, s0];
    params.extend(sigmas);
    Some(SharedLatentRecordLaw {
        latent,
        field_count: field_measures.len(),
        form: LogDensity {
            build: build_shared_latent_normal_logpdf,
            params,
        },
    })
}

/// The shared-latent record law's log-density at `variates`, from
/// `params = [μ₀, s₀, σ₁, …, σ_N]`.
///
/// `Σ = s₀²·11ᵀ + diag(σᵢ²)` is diagonal plus rank one, so Sherman–Morrison gives the
/// inverse's action and the matrix determinant lemma the log-det, and BOTH corrections
/// carry the same `1 + k`:
///
/// ```text
/// dᵢ = 1/σᵢ²   rᵢ = xᵢ − μ₀   S = Σ dᵢ   T = Σ dᵢrᵢ   k = s₀²S
/// rᵀΣ⁻¹r  = Σ dᵢrᵢ² − s₀²T²/(1 + k)
/// logdetΣ = Σ log σᵢ² + log(1 + k)
/// out     = −½[ N·log2π + logdetΣ + rᵀΣ⁻¹r ]
/// ```
///
/// Every op is §07 ("Elementary functions") — no matrix op and no `MvNormal` constructor.
/// The sum is emitted flat, with `quad` last and on its own, so an all-literal model folds
/// it to a single legible literal (`canon::fold` leaves `log`/`log1p` alone).
fn build_shared_latent_normal_logpdf(
    m: &mut Module,
    params: &[NodeId],
    variates: &[NodeId],
) -> NodeId {
    let [mu0, s0] = [params[0], params[1]];
    let sigmas = &params[2..];
    // The emitter refuses a count mismatch before reaching here
    // ([`crate::density::lower_shared_latent_record_law`]), so the `zip`s below cannot
    // truncate. Restated as an assert because this builder's own correctness needs it.
    debug_assert_eq!(
        sigmas.len(),
        variates.len(),
        "one sigma and one variate per record field"
    );

    let two = real(m, 2.0);
    let one = real(m, 1.0);

    // Per field: the variance vᵢ (shared between dᵢ and the log-det), the precision dᵢ,
    // and the deviation rᵢ.
    let mut vars = Vec::with_capacity(sigmas.len());
    let mut precisions = Vec::with_capacity(sigmas.len());
    let mut devs = Vec::with_capacity(sigmas.len());
    for (&sigma, &x) in sigmas.iter().zip(variates) {
        let var = build_call(m, "pow", &[sigma, two]);
        vars.push(var);
        precisions.push(build_call(m, "divide", &[one, var]));
        devs.push(build_call(m, "sub", &[x, mu0]));
    }

    // k = s₀²·Σdᵢ, the rank-one term both corrections divide by.
    let prior_var = build_call(m, "pow", &[s0, two]);
    let precision_sum = crate::density::fold_add(m, &precisions);
    let k = build_call(m, "mul", &[prior_var, precision_sum]);

    // rᵀΣ⁻¹r = Σdᵢrᵢ² − s₀²T²/(1 + k).
    let weighted: Vec<NodeId> = precisions
        .iter()
        .zip(&devs)
        .map(|(&d, &r)| build_call(m, "mul", &[d, r]))
        .collect();
    let t = crate::density::fold_add(m, &weighted);
    let squared: Vec<NodeId> = precisions
        .iter()
        .zip(&devs)
        .map(|(&d, &r)| {
            let r2 = build_call(m, "pow", &[r, two]);
            build_call(m, "mul", &[d, r2])
        })
        .collect();
    let diag_quad = crate::density::fold_add(m, &squared);
    let t2 = build_call(m, "pow", &[t, two]);
    let correction_num = build_call(m, "mul", &[prior_var, t2]);
    let denom = build_call(m, "add", &[one, k]);
    let correction = build_call(m, "divide", &[correction_num, denom]);
    let quad = build_call(m, "sub", &[diag_quad, correction]);

    // −½[ N·log2π + Σ log vᵢ + log1p(k) + quad ], flat so `quad` stays a lone literal.
    let count = real(m, sigmas.len() as f64);
    let log_2pi = real(m, LOG_2PI);
    let mut terms = vec![build_call(m, "mul", &[count, log_2pi])];
    for &var in &vars {
        terms.push(build_call(m, "log", &[var]));
    }
    terms.push(build_call(m, "log1p", &[k]));
    terms.push(quad);
    let total = crate::density::fold_add(m, &terms);
    let neg_half = real(m, -0.5);
    build_call(m, "mul", &[neg_half, total])
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

#[cfg(test)]
mod tests {
    use super::LOG_2PI;

    // `LOG_2PI` is hardcoded so the emitted literal cannot vary with the platform's `ln`,
    // and this is what keeps it honest: it must still BE log(2π).
    #[test]
    fn log_2pi_matches_the_computed_constant() {
        assert_eq!(LOG_2PI, (2.0 * std::f64::consts::PI).ln());
    }
}
