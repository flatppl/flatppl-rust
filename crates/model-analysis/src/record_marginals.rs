//! Joint marginal rules for record fields with overlapping stochastic ancestry.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::conjugacy::match_scalar_row;
use crate::{
    ConjugateRowId, StochasticAncestor, measure_stochastic_ancestors, references_latent,
    sole_named_ancestor,
};
use flatppl_core::graph::is_input_ref;
use flatppl_core::{Dim, Module, NodeId, ScalarType, Symbol, Type};
use flatppl_infer::constructors::{find_parameter, split_kernel_constructor};

/// Distribution of a record field's draw and its optional transformation.
#[derive(Clone, Copy, Debug)]
pub struct RecordField {
    pub measure: NodeId,
    pub transform: Option<NodeId>,
}

/// A connected component of fields requiring marginalization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordMarginalBlock {
    /// Indices into the supplied field list, in source order.
    pub fields: Vec<usize>,
    /// An exact joint rule, when the block shares one supported latent.
    pub shared: Option<SharedRecordMatch>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedRecordMatch {
    pub ancestor: StochasticAncestor,
    pub law: SharedRecordLaw,
}

/// Parameters of the exact joint score. Per-field entries follow the block's field order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SharedRecordLaw {
    NormalMean {
        mean: NodeId,
        sigma: NodeId,
        field_sigmas: Vec<NodeId>,
    },
    GammaPoisson {
        shape: NodeId,
        rate: NodeId,
    },
    Beta {
        alpha: NodeId,
        beta: NodeId,
        counts: Vec<CountObservation>,
    },
    Dirichlet {
        alpha: NodeId,
        categories: u32,
        counts: Vec<CountObservation>,
    },
}

/// The native count likelihood retains its support and combinatorial coefficient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CountObservation {
    pub family: Symbol,
    /// A Binomial or Multinomial trial count. Other count observations have one trial.
    pub trials: Option<NodeId>,
}

/// Partition fields by all unobserved draws they reach, including through their priors.
/// Supply one field per distinct observation site and include those draw sites in `observed`.
/// Fields with no such draw use their conditional law and are omitted. Each block is
/// ordered by its first field. An unmatched multi-field block still requires a joint law.
pub fn plan_record_marginals(
    m: &mut Module,
    fields: &[RecordField],
    observed: &[NodeId],
) -> Vec<RecordMarginalBlock> {
    let mut parents: Vec<_> = (0..fields.len()).collect();
    let mut owners = HashMap::new();
    let mut active = Vec::new();
    for (index, field) in fields.iter().enumerate() {
        let mut draws = measure_stochastic_ancestors(m, field.measure, observed);
        if let Some(transform) = field.transform {
            draws.extend(measure_stochastic_ancestors(m, transform, observed));
        }
        if draws.is_empty() {
            continue;
        }
        let direct = if field.transform.is_none() {
            sole_named_ancestor(&draws).map(|_| draws[0])
        } else {
            None
        };
        active.push((index, direct));
        while let Some(ancestor) = draws.pop() {
            match owners.entry(ancestor.draw) {
                Entry::Vacant(slot) => {
                    slot.insert(index);
                    draws.extend(measure_stochastic_ancestors(m, ancestor.prior, observed));
                }
                Entry::Occupied(slot) => {
                    // The first owner already links this draw to all its ancestors.
                    let left = root(&mut parents, index);
                    let right = root(&mut parents, *slot.get());
                    parents[left.max(right)] = left.min(right);
                }
            }
        }
    }

    let mut groups = vec![Vec::new(); fields.len()];
    for (index, direct) in active {
        groups[root(&mut parents, index)].push((index, direct));
    }
    groups
        .into_iter()
        .filter(|group| !group.is_empty())
        .map(|group| {
            let ancestor = group[0]
                .1
                .filter(|a| group.len() > 1 && group.iter().all(|(_, direct)| *direct == Some(*a)));
            let indices: Vec<_> = group.into_iter().map(|(i, _)| i).collect();
            let shared = ancestor.and_then(|a| {
                if !measure_stochastic_ancestors(m, a.prior, &[]).is_empty() {
                    return None;
                }
                let law = shared_law(m, a, indices.iter().map(|&i| fields[i].measure), observed)?;
                Some(SharedRecordMatch { ancestor: a, law })
            });
            RecordMarginalBlock {
                fields: indices,
                shared,
            }
        })
        .collect()
}

fn root(parents: &mut [usize], mut index: usize) -> usize {
    while parents[index] != index {
        parents[index] = parents[parents[index]];
        index = parents[index];
    }
    index
}

fn shared_law(
    m: &mut Module,
    ancestor: StochasticAncestor,
    measures: impl Iterator<Item = NodeId>,
    observed: &[NodeId],
) -> Option<SharedRecordLaw> {
    let latent = ancestor.name?;
    let (prior, params) = split_kernel_constructor(m, ancestor.prior)?;
    let param = |name| find_parameter(m, &params, name);
    let mut law = match m.resolve(prior) {
        "Normal" => SharedRecordLaw::NormalMean {
            mean: param("mu")?,
            sigma: param("sigma")?,
            field_sigmas: Vec::new(),
        },
        "Gamma" => SharedRecordLaw::GammaPoisson {
            shape: param("shape")?,
            rate: param("rate")?,
        },
        "Beta" => SharedRecordLaw::Beta {
            alpha: param("alpha")?,
            beta: param("beta")?,
            counts: Vec::new(),
        },
        "Dirichlet" => {
            let alpha = param("alpha")?;
            SharedRecordLaw::Dirichlet {
                alpha,
                categories: category_count(m, alpha)?,
                counts: Vec::new(),
            }
        }
        _ => return None,
    };
    for measure in measures {
        let (family, args) = split_kernel_constructor(m, measure)?;
        if let SharedRecordLaw::Dirichlet { counts, .. } = &mut law {
            counts.push(dirichlet_count(m, latent, family, &args, observed)?);
            continue;
        }
        let row = match_scalar_row(m, (prior, &params), (family, &args), latent, observed)?.ok()?;
        match (&mut law, row) {
            (SharedRecordLaw::NormalMean { field_sigmas, .. }, ConjugateRowId::NormalMean) => {
                field_sigmas.push(find_parameter(m, &args, "sigma")?);
            }
            (SharedRecordLaw::GammaPoisson { .. }, ConjugateRowId::GammaPoissonRate) => {}
            (SharedRecordLaw::Beta { counts, .. }, ConjugateRowId::BetaBernoulli) => {
                counts.push(CountObservation {
                    family,
                    trials: None,
                });
            }
            (SharedRecordLaw::Beta { counts, .. }, ConjugateRowId::BetaBinomial) => {
                counts.push(CountObservation {
                    family,
                    trials: Some(find_parameter(m, &args, "n")?),
                });
            }
            _ => return None,
        }
    }
    Some(law)
}

/// A one-field Dirichlet marginal uses the same category and count checks as a joint block.
pub fn match_dirichlet_observation(
    m: &Module,
    latent: Symbol,
    prior: &[(Symbol, NodeId)],
    likelihood: (Symbol, &[(Symbol, NodeId)]),
    observed: &[NodeId],
) -> Option<(NodeId, u32, CountObservation)> {
    let alpha = find_parameter(m, prior, "alpha")?;
    Some((
        alpha,
        category_count(m, alpha)?,
        dirichlet_count(m, latent, likelihood.0, likelihood.1, observed)?,
    ))
}

fn category_count(m: &Module, alpha: NodeId) -> Option<u32> {
    let Type::Array { shape, elem } = m.type_of(alpha)? else {
        return None;
    };
    let [Dim::Static(n)] = shape.as_ref() else {
        return None;
    };
    // Category labels must be exactly representable at both supported real precisions.
    (matches!(
        elem.as_ref(),
        Type::Scalar(ScalarType::Real | ScalarType::Integer)
    ) && *n > 0
        && *n <= 1 << 24)
        .then_some(*n)
}

fn dirichlet_count(
    m: &Module,
    latent: Symbol,
    family: Symbol,
    args: &[(Symbol, NodeId)],
    observed: &[NodeId],
) -> Option<CountObservation> {
    if !is_input_ref(m, find_parameter(m, args, "p")?, latent)
        || args.iter().any(|(p, value)| {
            m.resolve(*p) != "p" && references_latent(m, *value, latent, observed)
        })
    {
        return None;
    }
    let trials = match m.resolve(family) {
        "Categorical" | "Categorical0" => None,
        "Multinomial" => {
            let n = find_parameter(m, args, "n")?;
            if !matches!(
                m.type_of(n),
                Some(Type::Scalar(ScalarType::Integer | ScalarType::Real))
            ) {
                return None;
            }
            Some(n)
        }
        _ => return None,
    };
    Some(CountObservation { family, trials })
}
