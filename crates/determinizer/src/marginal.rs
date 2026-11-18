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
//! static error." Exact rules stay first. An opt-in numerical extension handles
//! a supported scalar continuous prior when no exact rule applies:
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
//! - **A supported scalar continuous prior**: integrate the conditional density
//!   against the prior. Ordinary FlatPDL conformance rejects this extension.
//!
//! - **Other continuous or infinite-discrete latents**: refused.
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
//! | `Categorical(p)`  | `[1, n]`    | `{1, …, n}`                 | `p` has fixed length n |
//! | `Categorical0(p)` | `[0, n-1]`  | `{0, …, n-1}`               | `p` has fixed length n |
//! | `Binomial(n, p)`  | `[0, n]`    | `{0, …, n}`                 | `n` a static int       |
//!
//! `Poisson`, `Geometric`, `NegativeBinomial*` (support `nonnegintegers`) and any
//! continuous distribution are **not** enumerable → refused.

use crate::density::{
    build_call, build_record, draw_argument, expect_builtin_call, lower_measure_density, refuse,
    resolve_ref, split_kernel_constructor,
};
use crate::kernel::{Kernel, resolve_reified};
use crate::refuse::RefuseError;
use flatppl_core::graph::is_input_ref;
use flatppl_core::{
    Call, CallHead, Dim, Inputs, Module, NamedArg, NamedKind, Node, NodeId, Ref, RefNs, Scalar,
    Symbol, Type, ValueSet,
};
use flatppl_infer::constructors::{constructor_symbol, find_parameter as find_kwarg};
use flatppl_model_analysis::record_marginals::{CountObservation, SharedRecordLaw};
use flatppl_model_analysis::{
    ConjugateMatch, ConjugateRowId, measure_stochastic_ancestors, sole_named_ancestor,
};

mod beta;
mod dirichlet;
pub(crate) mod gaussian;

/// Above this many atoms, refuse: an enumerated logsumexp must stay small (the
/// determiniser emits one density sub-tree per atom). A finite but large latent
/// is treated as non-enumerable.
pub(crate) const MAX_ATOMS: i64 = 256;

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
    let kernel = resolve_reified(m, k_arg)
        .ok_or_else(|| refuse_kchain(node, "kchain kernel is not a recognisable kernelof(...)"))?;
    if kernel.inputs.len() != 1 {
        return Err(refuse_kchain(
            node,
            "single-input kernel only; multi-input kchain kernels are out of scope",
        ));
    }
    // Normalize the ordinary application before recognizing conjugacy. This
    // keeps one-field record adapters equivalent to their scalar conditional.
    let reference = Ref {
        ns: RefNs::Local,
        name: m.intern(&format!("_kchain_{}_", m.node_count())),
    };
    let coordinate = m.alloc(Node::Ref(reference));
    let point = latent
        .field
        .map(|name| build_record(m, &[(name, coordinate)]))
        .unwrap_or(coordinate);
    let names: Vec<_> = kernel.inputs.iter().map(|(name, _)| *name).collect();
    let values = crate::kernel::bind_call_args(m, &names, &[point], &[], kernel.auto)
        .ok_or_else(|| refuse_kchain(node, "latent point does not bind the kernel interface"))?;
    let bound: Vec<_> = kernel
        .inputs
        .iter()
        .zip(values)
        .map(|((_, target), value)| (*target, value))
        .collect();
    let absorbed = crate::kernel::resolve_kernel(m, k_arg).is_some();
    let body = crate::kernel::substitute_admitted(
        m,
        kernel.body,
        &bound,
        crate::kernel::Substitute::All,
        absorbed,
    );
    let body = crate::canon::inline_expression(m, body);
    let label = m.intern("latent");
    let kernel = Kernel {
        body,
        inputs: vec![(label, reference)],
        auto: false,
    };
    if let Some(result) = try_conjugate_marginal(m, latent.dist, &kernel, v) {
        return result;
    }
    if let Some(result) =
        crate::density::integral::kernel_marginal(m, latent.dist, k_arg, &kernel, v)
    {
        return result;
    }
    Err(refuse_kchain(
        node,
        "non-enumerable marginal (continuous / infinite-discrete); \
         no conjugate closed-form applies",
    ))
}

// ---------------------------------------------------------------------------
// Latent identification
// ---------------------------------------------------------------------------

/// The latent of a `kchain`: its distribution-constructor node (for both the
/// pmf and the discrete-finite classification).
struct Latent {
    /// The distribution-constructor node, e.g. `Bernoulli(p = 0.3)`.
    dist: NodeId,
    field: Option<Symbol>,
}

/// Resolve `M` (the kchain's first argument) to the latent's distribution
/// constructor. Accepts `lawof(record(name = draw(dist)))`, `lawof(draw(dist))`,
/// `lawof(dist)`, a bare `draw(dist)`, or a bare `dist`. Returns `None` for any
/// shape we cannot pin to a single primitive constructor (e.g. a multi-field
/// record latent, or a combinator).
fn resolve_latent(m: &Module, m_arg: NodeId) -> Option<Latent> {
    let (resolved, _) = resolve_ref(m, m_arg);

    // Strip an optional `lawof(...)`.
    let inner = match expect_builtin_call(m, resolved, "lawof") {
        Some(law) if law.args.len() == 1 => {
            let (i, _) = resolve_ref(m, law.args[0]);
            i
        }
        Some(_) => return None,
        None => resolved,
    };

    // `record(name = X)` with exactly one field → X is the latent value.
    let (value, field) = if let Some(rec) = expect_builtin_call(m, inner, "record") {
        if !rec.args.is_empty() || rec.named.len() != 1 {
            return None;
        }
        let (v, _) = resolve_ref(m, rec.named[0].value);
        (v, Some(rec.named[0].name))
    } else {
        (inner, None)
    };

    // `draw(dist)` → dist; or a bare `dist` constructor.
    let dist = if let Some(draw) = expect_builtin_call(m, value, "draw") {
        if draw.args.len() != 1 {
            return None;
        }
        let (d, _) = resolve_ref(m, draw.args[0]);
        d
    } else {
        value
    };

    // Must be a builtin distribution-constructor call.
    if !matches!(m.node(dist), Node::Call(c) if matches!(c.head, CallHead::Builtin(_))) {
        return None;
    }
    Some(Latent { dist, field })
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
pub(crate) fn classify_atoms(m: &mut Module, dist: NodeId) -> Option<Vec<i64>> {
    let (sym, parameters) = split_kernel_constructor(m, dist)?;
    let parameter = |name: &str| {
        parameters
            .iter()
            .find(|(key, _)| m.resolve(*key) == name)
            .map(|(_, value)| *value)
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
            let p = parameter("p")?;
            let base = if name == "Categorical" { 1 } else { 0 };
            let n = static_vector_len(m, p)?;
            bounded(n).then(|| (0..n).map(|i| base + i).collect())
        }
        // Binomial(n, p): atoms {0, …, n}, n+1 of them; n must be a static int.
        "Binomial" => {
            let n = static_int(m, parameter("n")?)?;
            // n+1 atoms (inclusive of 0 and n).
            (0..MAX_ATOMS).contains(&n).then(|| (0..=n).collect())
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

/// The ABI fixes a vector's length even when its weights remain runtime inputs.
fn static_vector_len(m: &mut Module, id: NodeId) -> Option<i64> {
    let ty = m.type_of(id).cloned().or_else(|| {
        // Constructor record splats may have just created this projection.
        flatppl_infer::infer_expression(m, id, &[]).0.remove(&id)
    })?;
    let Type::Array { shape, .. } = ty else {
        return None;
    };
    let [Dim::Static(n)] = shape.as_ref() else {
        return None;
    };
    Some(i64::from(*n))
}

/// Follow constant aliases without folding through a promoted runtime input.
fn static_int(m: &Module, id: NodeId) -> Option<i64> {
    let mut dynamic = crate::canon::promoted_input_names(m);
    // A surviving callable may rebind a module name that has a literal default.
    // Its cardinality must not be fixed before that boundary is applied.
    let mut pending: Vec<_> = m.bindings().map(|(_, binding)| binding.rhs).collect();
    let mut seen = std::collections::HashSet::new();
    while let Some(node) = pending.pop() {
        if !seen.insert(node) {
            continue;
        }
        if let Node::Call(call) = m.node(node)
            && let Some(Inputs::Spec(inputs)) = &call.inputs
        {
            dynamic.extend(
                inputs
                    .iter()
                    .filter_map(|(_, r)| (r.ns == RefNs::SelfMod).then_some(r.name)),
            );
        }
        m.for_each_child(node, |child| pending.push(child));
    }
    let mut id = id;
    seen.clear();
    while seen.insert(id) {
        match m.node(id) {
            Node::Lit(Scalar::Int(n)) => return Some(*n),
            Node::Ref(r) if r.ns == RefNs::SelfMod && !dynamic.contains(&r.name) => {
                id = m.binding(m.binding_by_name(r.name)?).rhs;
            }
            _ => return None,
        }
    }
    None
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

impl MarginalForm {
    pub(crate) fn map_parameters(
        &mut self,
        m: &mut Module,
        mut map: impl FnMut(&mut Module, NodeId) -> NodeId,
    ) {
        match self {
            Self::Measure(measure) => *measure = map(m, *measure),
            Self::LogDensity(form) => form.map_parameters(m, map),
        }
    }
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
    pub(crate) fn map_parameters(
        &mut self,
        m: &mut Module,
        mut map: impl FnMut(&mut Module, NodeId) -> NodeId,
    ) {
        for parameter in &mut self.params {
            *parameter = map(m, *parameter);
        }
    }
    /// The log-density expression at `variates` — ONE variate for a `CONJUGATE_TABLE` row,
    /// whose law is over a single variate, and one per record field for the shared-latent
    /// record law ([`shared_record_density`]), whose law is over all of them jointly.
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

/// A conjugate-marginal builder: from the prior's and likelihood's keyword
/// arguments (`(param, value)` pairs), build the closed form, or `None` if a required
/// parameter is absent.
type MarginalBuilder =
    fn(&mut Module, &[(Symbol, NodeId)], &[(Symbol, NodeId)]) -> Option<MarginalForm>;

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

    let marginal = match build_conjugate_marginal(m, latent_dist, lik.dist, kernel_input_sym, &[])?
    {
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
    let (resolved, _) = resolve_ref(m, v);
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
    exempt: &[NodeId],
) -> Option<Result<MarginalForm, RefuseError>> {
    let prior = constructor_symbol(m, latent_dist)?;
    if matches!(m.resolve(prior), "MvNormal" | "Dirichlet") {
        let (_, prior_kwargs) = split_kernel_constructor(m, latent_dist)?;
        let (lik, lik_kwargs) = split_kernel_constructor(m, lik_dist)?;
        if m.resolve(prior) == "MvNormal" {
            if m.resolve(lik) != "MvNormal" {
                return None;
            }
            return gaussian::marginal(m, latent_sym, &prior_kwargs, &lik_kwargs);
        }
        return dirichlet::marginal(m, latent_sym, &prior_kwargs, &lik_kwargs, lik, exempt);
    }

    let matched = match flatppl_model_analysis::match_conjugate_marginal(
        m,
        latent_dist,
        lik_dist,
        latent_sym,
        exempt,
    )? {
        Ok(matched) => matched,
        Err(err) => return Some(Err(refuse_kchain(latent_dist, &err.0))),
    };
    let build: MarginalBuilder = match matched.row {
        ConjugateRowId::NormalMean => build_normal_normal_marginal,
        ConjugateRowId::GammaPoissonRate => build_gamma_poisson_marginal,
        ConjugateRowId::BetaBernoulli => beta::bernoulli_marginal,
        ConjugateRowId::BetaBinomial => build_beta_binomial_marginal,
        ConjugateRowId::ExponentialVariance => build_exponential_variance_marginal,
        ConjugateRowId::InverseGammaVariance => build_inverse_gamma_variance_marginal,
        ConjugateRowId::ExponentialMean => build_exponential_mean_marginal,
    };
    Some(
        build(m, &matched.prior, &matched.likelihood).ok_or_else(|| {
            refuse_kchain(
                latent_dist,
                "conjugate pair matched but a required distribution parameter is missing",
            )
        }),
    )
}

/// Recognize a scalar conjugate marginal with no observed sibling draws.
/// See [`flatppl_model_analysis::match_conjugate_marginal`] for the rule contract.
pub fn match_conjugate_marginal(
    m: &mut Module,
    latent_dist: NodeId,
    lik_dist: NodeId,
    latent: Symbol,
) -> Option<Result<ConjugateMatch, RefuseError>> {
    flatppl_model_analysis::match_conjugate_marginal(m, latent_dist, lik_dist, latent, &[])
        .map(|result| result.map_err(|err| refuse_kchain(latent_dist, &err.0)))
}

// ---------------------------------------------------------------------------
// The implicit `lawof` spelling
// ---------------------------------------------------------------------------

/// Integrate the sole named latent, recursively composing measure-valued marginals.
/// The caller supplies observed draws at which the ancestry walk stops.
pub(crate) fn conjugate_marginal_measure(
    m: &mut Module,
    likelihood: NodeId,
    exempt: &[NodeId],
) -> Option<Result<MarginalForm, RefuseError>> {
    let (latent, prior) =
        sole_named_ancestor(&measure_stochastic_ancestors(m, likelihood, exempt))?;
    let above = measure_stochastic_ancestors(m, prior, &[]);
    let prior = if above.is_empty() {
        prior
    } else {
        // A carried draw above the latent makes the prior conditional on the query.
        if measure_stochastic_ancestors(m, prior, exempt).len() != above.len() {
            return None;
        }
        match conjugate_marginal_measure(m, prior, exempt)? {
            Ok(MarginalForm::Measure(marginal)) => marginal,
            Ok(MarginalForm::LogDensity(_)) => return None,
            Err(err) => return Some(Err(err)),
        }
    };
    build_conjugate_marginal(m, prior, likelihood, latent, exempt)
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
    let (resolved, _) = resolve_ref(m, body);

    // Optional single-field `record(field = X)` wrapper → remember the field.
    let (inner, record_field) = if let Some(rec) = expect_builtin_call(m, resolved, "record") {
        if !rec.args.is_empty() || rec.named.len() != 1 {
            return None;
        }
        let (val, _) = resolve_ref(m, rec.named[0].value);
        (val, Some(rec.named[0].name))
    } else {
        (resolved, None)
    };

    // Optional `draw(dist)` → dist.
    let dist = if let Some(inner_dist) = draw_argument(m, inner) {
        let (d, _) = resolve_ref(m, inner_dist);
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

/// Exponential-prior-on-the-MEAN marginal builder. `y = v + σ·ε` with `v ~ Exponential(λ)`
/// is the exponentially modified Gaussian. §08 has no constructor for it, so the row
/// returns the log-density ([`build_exponentially_modified_normal_logpdf`]).
fn build_exponential_mean_marginal(
    m: &mut Module,
    prior_kwargs: &[(Symbol, NodeId)],
    lik_kwargs: &[(Symbol, NodeId)],
) -> Option<MarginalForm> {
    let rate = find_kwarg(m, prior_kwargs, "rate")?;
    let sigma = find_kwarg(m, lik_kwargs, "sigma")?;
    Some(MarginalForm::LogDensity(LogDensity {
        build: build_exponentially_modified_normal_logpdf,
        params: vec![rate, sigma],
    }))
}

fn build_exponentially_modified_normal_logpdf(
    m: &mut Module,
    params: &[NodeId],
    variates: &[NodeId],
) -> NodeId {
    exponentially_modified_normal_logpdf(m, params[0], params[1], sole_variate(variates))
}

/// Build the log-density of the sum of Exponential(λ) and Normal(0, σ) at `y`,
/// with `z = (λσ² − y)/(√2·σ)`:
///
/// ```text
/// z < 0:  log λ − log 2 − λy + (λσ)²/2 + log erfc(z)
/// z ≥ 0:  log λ − log 2 − y²/(2σ²)    + log erfcx(z)
/// ```
///
/// Both are `log λ − log 2 − λy + (λσ)²/2 − z² + log erfcx(z)`, since `erfcx(z) =
/// exp(z²)·erfc(z)`. `erfc(z)` underflows for large positive `z` and `erfcx(z)`
/// overflows for large negative `z`, so each side uses the function that stays in
/// range. Each side reads `z` clamped to its own half-line, so the side `ifelse`
/// drops is finite and adds no `NaN` to a gradient. Both functions are §09
/// `special-functions` members, reached through [`flatppl_core::graph::special_functions_alias`].
pub fn exponentially_modified_normal_logpdf(
    m: &mut Module,
    rate: NodeId,
    sigma: NodeId,
    y: NodeId,
) -> NodeId {
    let rate_sigma = build_call(m, "mul", &[rate, sigma]);
    let center = build_call(m, "mul", &[rate_sigma, sigma]);
    let dev = build_call(m, "sub", &[center, y]);
    let sqrt2 = real(m, std::f64::consts::SQRT_2);
    let width = build_call(m, "mul", &[sqrt2, sigma]);
    let z = build_call(m, "divide", &[dev, width]);
    let zero = real(m, 0.0);
    let left = build_call(m, "lt", &[z, zero]);
    let z_left = build_call(m, "ifelse", &[left, z, zero]);
    let z_right = build_call(m, "ifelse", &[left, zero, z]);
    let alias = flatppl_core::graph::special_functions_alias(m);

    let half = real(m, 0.5);
    let square = build_call(m, "mul", &[rate_sigma, rate_sigma]);
    let shift = build_call(m, "mul", &[half, square]);
    let decay = build_call(m, "mul", &[rate, y]);
    let exponent = build_call(m, "sub", &[shift, decay]);
    let erfc = special_function_call(m, alias, "erfc", z_left);
    let log_erfc = build_call(m, "log", &[erfc]);
    let left_tail = build_call(m, "add", &[exponent, log_erfc]);

    let y2 = build_call(m, "mul", &[y, y]);
    let two = real(m, 2.0);
    let var = build_call(m, "mul", &[sigma, sigma]);
    let twice_var = build_call(m, "mul", &[two, var]);
    let gauss = build_call(m, "divide", &[y2, twice_var]);
    let erfcx = special_function_call(m, alias, "erfcx", z_right);
    let log_erfcx = build_call(m, "log", &[erfcx]);
    let right_tail = build_call(m, "sub", &[log_erfcx, gauss]);

    let tail = build_call(m, "ifelse", &[left, left_tail, right_tail]);
    let log_rate = build_call(m, "log", &[rate]);
    let total = build_call(m, "add", &[log_rate, tail]);
    let ln2 = real(m, std::f64::consts::LN_2);
    build_call(m, "sub", &[total, ln2])
}

/// `alias.member(arg)` for a `special-functions` member.
fn special_function_call(m: &mut Module, alias: Symbol, member: &str, arg: NodeId) -> NodeId {
    let name = m.intern(member);
    let callee = m.alloc(Node::Ref(Ref {
        ns: RefNs::Module(alias),
        name,
    }));
    crate::density::build_user_call_n(m, callee, vec![arg])
}

// ---------------------------------------------------------------------------
// The shared-latent record law
// ---------------------------------------------------------------------------

/// `log(2π)`, the Gaussian normalizer's constant. Hardcoded rather than computed from a
/// `PI` const so the emitted literal is the same on every platform: `canon::fold` folds
/// `mul(N, LOG_2PI)` into the output, and `f64::ln` is not rounding-mandated.
/// `log_2pi_matches_the_computed_constant` pins it against `(2·π).ln()`.
const LOG_2PI: f64 = 1.8378770664093453;

/// Build the joint score chosen by structural model analysis.
pub(crate) fn shared_record_density(m: &mut Module, law: SharedRecordLaw) -> LogDensity {
    match law {
        SharedRecordLaw::NormalMean {
            mean,
            sigma,
            field_sigmas,
        } => {
            let mut params = vec![mean, sigma];
            params.extend(field_sigmas);
            LogDensity {
                build: build_shared_latent_normal_logpdf,
                params,
            }
        }
        SharedRecordLaw::GammaPoisson { shape, rate } => LogDensity {
            build: build_shared_gamma_poisson_logpmf,
            params: vec![shape, rate],
        },
        SharedRecordLaw::Beta {
            alpha,
            beta,
            counts,
        } => LogDensity {
            build: beta::shared_logpmf,
            params: count_parameters(m, vec![alpha, beta], counts),
        },
        SharedRecordLaw::Dirichlet {
            alpha,
            categories,
            counts,
        } => {
            let size = m.alloc(Node::Lit(Scalar::Int(i64::from(categories))));
            LogDensity {
                build: dirichlet::shared_logpmf,
                params: count_parameters(m, vec![alpha, size], counts),
            }
        }
    }
}

fn count_parameters(
    m: &mut Module,
    mut params: Vec<NodeId>,
    counts: Vec<CountObservation>,
) -> Vec<NodeId> {
    params.reserve(2 * counts.len());
    for count in counts {
        let n = count
            .trials
            .unwrap_or_else(|| m.alloc(Node::Lit(Scalar::Int(1))));
        params.push(m.alloc(Node::Const(count.family)));
        params.push(n);
    }
    params
}

/// Sequential Gamma-Poisson predictives retain native count coefficients.
/// Unlike a product of marginal NBs, each predictive updates both parameters.
fn build_shared_gamma_poisson_logpmf(
    m: &mut Module,
    params: &[NodeId],
    variates: &[NodeId],
) -> NodeId {
    let [mut alpha, mut beta] = [params[0], params[1]];
    let zero = real(m, 0.0);
    let one = real(m, 1.0);
    let mut terms = Vec::with_capacity(variates.len());
    for &value in variates {
        let alpha_name = m.intern("alpha");
        let beta_name = m.intern("beta");
        let record = build_record(m, &[(alpha_name, alpha), (beta_name, beta)]);
        let family = m.intern("NegativeBinomial");
        let kernel = m.alloc(Node::Const(family));
        let term = build_call(m, "builtin_logdensityof", &[kernel, record, value]);
        terms.push(term);
        // Widen before adding. Invalid earlier counts must not turn a later
        // predictive's positive shape into a new parameter-domain error.
        let count = build_call(m, "real", &[value]);
        let nonnegative = build_call(m, "ge", &[count, zero]);
        let finite = build_call(m, "isfinite", &[count]);
        let valid = build_call(m, "land", &[finite, nonnegative]);
        let safe = build_call(m, "ifelse", &[valid, count, zero]);
        alpha = build_call(m, "add", &[alpha, safe]);
        beta = build_call(m, "add", &[beta, one]);
    }
    crate::density::fold_add(m, &terms)
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
    // The plan supplies one scale and one variate per field in the same order.
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
