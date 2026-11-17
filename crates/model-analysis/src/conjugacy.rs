//! Scalar conjugacy rules for the marginal integral in §06.

use crate::{AnalysisError, references_latent};
use flatppl_core::graph::{confirmed_non_scalar, expect_builtin_call, is_input_ref, resolve_ref};
use flatppl_core::{Module, NodeId, Symbol};
use flatppl_infer::constructors::{find_parameter, split_kernel_constructor};

/// How the latent reaches the conjugating parameter. Normal takes a standard
/// deviation, so a prior on its variance must reach `sigma` through `sqrt`.
enum LatentPath {
    Direct,
    Sqrt,
}

/// A row of the conjugate table (`marginal.md`). Each names one prior/likelihood
/// pair and the parameter path the latent takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConjugateRowId {
    /// Row 1: `Normal` prior on a `Normal` mean.
    NormalMean,
    /// Row 2: `Gamma` prior on a `Poisson` rate.
    GammaPoissonRate,
    /// `Beta` prior on a `Bernoulli` probability.
    BetaBernoulli,
    /// Row 3: `Beta` prior on a `Binomial` probability.
    BetaBinomial,
    /// Row 4: `Exponential` prior on a `Normal` variance.
    ExponentialVariance,
    /// Row 5: `InverseGamma` prior on a `Normal` variance.
    InverseGammaVariance,
    /// Row 6: `Exponential` prior on a `Normal` mean.
    ExponentialMean,
}

/// A recognised conjugate pair, before any closed form is built. The parameter
/// lists hold the constructors' keyword arguments as nodes of the matched module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConjugateMatch {
    pub row: ConjugateRowId,
    /// The symbol the likelihood references the latent by.
    pub latent: Symbol,
    pub prior: Vec<(Symbol, NodeId)>,
    pub likelihood: Vec<(Symbol, NodeId)>,
}

/// One conjugate prior/likelihood pair whose `kchain` marginal is closed-form.
struct ConjugateRow {
    id: ConjugateRowId,
    /// Prior distribution-constructor name (the latent's law), e.g. `"Normal"`.
    prior: &'static str,
    /// Likelihood distribution-constructor name (the kernel body), e.g. `"Normal"`.
    likelihood: &'static str,
    /// The likelihood parameter the latent feeds (the "conjugating" parameter),
    /// e.g. `"mu"`.
    conjugating_param: &'static str,
    /// The transformation the latent passes through on its way to that parameter.
    latent_path: LatentPath,
}

/// Scalar conjugate pairs in match order.
const CONJUGATE_TABLE: &[ConjugateRow] = &[
    ConjugateRow {
        id: ConjugateRowId::NormalMean,
        prior: "Normal",
        likelihood: "Normal",
        conjugating_param: "mu",
        latent_path: LatentPath::Direct,
    },
    ConjugateRow {
        id: ConjugateRowId::GammaPoissonRate,
        prior: "Gamma",
        likelihood: "Poisson",
        conjugating_param: "rate",
        latent_path: LatentPath::Direct,
    },
    ConjugateRow {
        id: ConjugateRowId::BetaBernoulli,
        prior: "Beta",
        likelihood: "Bernoulli",
        conjugating_param: "p",
        latent_path: LatentPath::Direct,
    },
    ConjugateRow {
        id: ConjugateRowId::BetaBinomial,
        prior: "Beta",
        likelihood: "Binomial",
        conjugating_param: "p",
        latent_path: LatentPath::Direct,
    },
    ConjugateRow {
        id: ConjugateRowId::ExponentialVariance,
        prior: "Exponential",
        likelihood: "Normal",
        conjugating_param: "sigma",
        latent_path: LatentPath::Sqrt,
    },
    ConjugateRow {
        id: ConjugateRowId::InverseGammaVariance,
        prior: "InverseGamma",
        likelihood: "Normal",
        conjugating_param: "sigma",
        latent_path: LatentPath::Sqrt,
    },
    ConjugateRow {
        id: ConjugateRowId::ExponentialMean,
        prior: "Exponential",
        likelihood: "Normal",
        conjugating_param: "mu",
        latent_path: LatentPath::Direct,
    },
];

/// Recognize a scalar conjugate prior and likelihood in a typed module.
/// `latent` names the reference feeding the likelihood. Draws in `observed`
/// are fixed by this query when checking the other likelihood parameters.
///
/// Returns `None` for an unmatched family, argument binding or dependency,
/// and an error for a matched row with a known non-scalar parameter.
/// Binding a positional or splatted constructor can intern names or allocate
/// field projections. The matcher returns parameter nodes before lowering.
pub fn match_conjugate_marginal(
    m: &mut Module,
    latent_dist: NodeId,
    lik_dist: NodeId,
    latent: Symbol,
    observed: &[NodeId],
) -> Option<Result<ConjugateMatch, AnalysisError>> {
    let (prior_sym, prior) = split_kernel_constructor(m, latent_dist)?;
    let (lik_sym, likelihood) = split_kernel_constructor(m, lik_dist)?;
    let row = match_scalar_row(
        m,
        (prior_sym, &prior),
        (lik_sym, &likelihood),
        latent,
        observed,
    )?;
    Some(row.map(|row| ConjugateMatch {
        row,
        latent,
        prior,
        likelihood,
    }))
}

pub(super) fn match_scalar_row(
    m: &Module,
    (prior_family, prior): (Symbol, &[(Symbol, NodeId)]),
    (family, likelihood): (Symbol, &[(Symbol, NodeId)]),
    latent: Symbol,
    observed: &[NodeId],
) -> Option<Result<ConjugateRowId, AnalysisError>> {
    let families = (m.resolve(prior_family), m.resolve(family));

    // Two rows can share the family pair but place the latent on different parameters.
    for row in CONJUGATE_TABLE
        .iter()
        .filter(|r| (r.prior, r.likelihood) == families)
    {
        let Some(value) = find_parameter(m, likelihood, row.conjugating_param) else {
            continue;
        };
        if !matches_latent_path(m, value, &row.latent_path, latent) {
            continue;
        }
        if likelihood.iter().any(|(p, v)| {
            m.resolve(*p) != row.conjugating_param && references_latent(m, *v, latent, observed)
        }) {
            continue;
        }
        // Each row describes a scalar law, including rows that reuse parameters verbatim.
        if let Some(error) = non_scalar_parameter(m, prior, likelihood) {
            return Some(Err(error));
        }
        return Some(Ok(row.id));
    }
    None
}

fn matches_latent_path(m: &Module, value: NodeId, path: &LatentPath, latent: Symbol) -> bool {
    match path {
        LatentPath::Direct => is_input_ref(m, value, latent),
        LatentPath::Sqrt => {
            let (resolved, _) = resolve_ref(m, value);
            let Some(call) = expect_builtin_call(m, resolved, "sqrt") else {
                return false;
            };
            call.named.is_empty()
                && matches!(call.args.as_ref(), [arg] if is_input_ref(m, *arg, latent))
        }
    }
}

fn non_scalar_parameter(
    m: &Module,
    prior: &[(Symbol, NodeId)],
    likelihood: &[(Symbol, NodeId)],
) -> Option<AnalysisError> {
    let (which, psym, kind) = prior
        .iter()
        .map(|p| ("prior", p))
        .chain(likelihood.iter().map(|p| ("likelihood", p)))
        .find_map(|(which, (p, v))| confirmed_non_scalar(m, *v).map(|kind| (which, *p, kind)))?;
    Some(AnalysisError(format!(
        "conjugate pair matched but the {which}'s `{}` is {} — a row's closed form is the \
             law of one SCALAR variate, and its parameter map is §07 scalar arithmetic \
             (`pow`'s domain is \"scalars\"). The vector-parameter spelling is \
             `broadcast(Dist, …)` (§04 \"Broadcasting\"), an array-valued measure no row \
             covers — refuse rather than mislower",
        m.resolve(psym),
        kind.article_noun(),
    )))
}
