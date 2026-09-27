//! Preserve auxiliary density blocks across HistFactory conversion.

use flatppl_core::NodeId;

use crate::builder::Builder;

pub(crate) struct Constraint {
    pub likelihood: NodeId,
    pub normal: Option<Normal>,
}

pub(crate) struct Normal {
    /// Every density input and the observation is a vector, including scalars
    /// wrapped as one-element vectors. This makes concatenation shape-safe.
    pub mean: NodeId,
    pub sigma: NodeId,
    pub observed: NodeId,
}

pub(crate) fn likelihoods(b: &mut Builder, constraints: &[Constraint]) -> Vec<NodeId> {
    let normal: Vec<_> = constraints
        .iter()
        .filter_map(|term| {
            term.normal
                .as_ref()
                .map(|density| (term.likelihood, density))
        })
        .collect();
    let mut result = Vec::new();
    if normal.len() < 2 {
        result.extend(normal.iter().map(|(likelihood, _)| *likelihood));
    } else {
        let head = b.call_head("Normal");
        let means = b.call(
            "cat",
            &normal.iter().map(|(_, n)| n.mean).collect::<Vec<_>>(),
        );
        let sigmas = b.call(
            "cat",
            &normal.iter().map(|(_, n)| n.sigma).collect::<Vec<_>>(),
        );
        let distribution = b.call("broadcast", &[head, means, sigmas]);
        let kernel = b.functionof(distribution);
        let observed = b.call(
            "cat",
            &normal.iter().map(|(_, n)| n.observed).collect::<Vec<_>>(),
        );
        let likelihood = b.call("likelihoodof", &[kernel, observed]);
        let name = b.bind_unique_doc(
            "normal_constraints_likelihood",
            likelihood,
            "Independent auxiliary measurements evaluated as one tensor block.",
        );
        result.push(b.self_ref(&name));
    }
    // Poisson constraints already have vector blocks. Preserve those partitions
    // instead of forcing all auxiliary densities into one reduction.
    result.extend(
        constraints
            .iter()
            .filter(|term| term.normal.is_none())
            .map(|term| term.likelihood),
    );
    result
}
