//! Source posterior laws and the sampling decisions of the Rust StableHLO lowering.

use std::collections::{HashMap, HashSet};
use std::fmt;

use flatppl_core::graph::expect_builtin_call;
use flatppl_core::{CallHead, Module, Node, NodeId, RefNs, Symbol};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PosteriorSampling {
    ExactIid,
    NotUsed,
    LoweringRefused,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PosteriorUpdate {
    pub binding: String,
    pub backend: &'static str,
    pub field: Option<String>,
    /// The common prior and posterior family when every likelihood factor matches.
    pub family: Option<String>,
    pub likelihoods: Vec<String>,
    pub parameters: Option<Vec<f64>>,
    pub sampling: PosteriorSampling,
}

impl fmt::Display for PosteriorUpdate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.binding)?;
        if let Some(family) = &self.family {
            if let Some(field) = &self.field {
                write!(f, ".{field}")?;
            }
            write!(
                f,
                "  {family} prior × {} → {family}",
                self.likelihoods.join(" × ")
            )?;
            if let Some(parameters) = &self.parameters {
                write!(
                    f,
                    "({})",
                    parameters
                        .iter()
                        .map(f64::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                )?;
            }
        } else {
            write!(f, "  no matching rule")?;
        }
        write!(
            f,
            "  [Rust StableHLO: {}]",
            match self.sampling {
                PosteriorSampling::ExactIid => "exact IID draw",
                PosteriorSampling::NotUsed => "conjugate sampler unused",
                PosteriorSampling::LoweringRefused => "query lowering refused",
            }
        )
    }
}

/// Visit the selected output graph. A promoted input hides its source declaration.
fn output_nodes(m: &Module, mut visit: impl FnMut(NodeId, Symbol, bool)) {
    let Some((_, output)) = m.bindings().find(|(_, b)| m.resolve(b.name) == "outputs") else {
        return;
    };
    let promoted = flatppl_core::query::promoted_input_names(m);
    let mut seen = HashSet::new();
    let mut todo = vec![(output.rhs, output.name, false)];
    while let Some((node, owner, reified)) = todo.pop() {
        if !seen.insert((node, reified)) {
            continue;
        }
        visit(node, owner, reified);
        if let Some(value) = flatppl_determinizer::projected_child(m, node, &promoted) {
            todo.push((value, owner, reified));
        } else if let Node::Ref(reference) = m.node(node) {
            if reference.ns == RefNs::SelfMod
                && !promoted.contains(&reference.name)
                && let Some(binding) = m.binding_by_name(reference.name)
            {
                let b = m.binding(binding);
                todo.push((
                    b.rhs,
                    if b.synthetic || !b.public {
                        owner
                    } else {
                        b.name
                    },
                    reified,
                ));
            }
        } else {
            let reified = reified
                || ["functionof", "kernelof"]
                    .iter()
                    .any(|op| expect_builtin_call(m, node, op).is_some());
            m.for_each_child(node, |child| todo.push((child, owner, reified)));
        }
    }
}

/// One source can produce several distinct native laws when a reified function
/// is called with different arguments. Preserve the instantiated proof for each.
#[derive(Default)]
pub(crate) struct PosteriorTrace {
    sources: Vec<Vec<PosteriorUpdate>>,
    by_node: HashMap<NodeId, usize>,
    draws: HashMap<NodeId, (usize, Vec<PosteriorUpdate>)>,
}

fn describe(m: &mut Module, node: Option<NodeId>, binding: String) -> Vec<PosteriorUpdate> {
    let product = node.and_then(|node| flatppl_determinizer::conjugate_product(m, node));
    let update = PosteriorUpdate {
        binding,
        backend: "rust-stablehlo",
        field: None,
        family: None,
        likelihoods: Vec::new(),
        parameters: None,
        sampling: PosteriorSampling::NotUsed,
    };
    let Some(product) = product else {
        if let Some(block) =
            node.and_then(|node| flatppl_determinizer::gaussian_posterior_block(m, node))
        {
            return vec![PosteriorUpdate {
                family: Some("MvNormal".into()),
                likelihoods: vec!["Normal".into(); block.observations],
                ..update
            }];
        }
        return vec![update];
    };
    product
        .fields
        .into_iter()
        .map(|law| {
            let mut field = update.clone();
            if let Node::Call(distribution) = m.node(law.distribution)
                && let CallHead::Builtin(family) = distribution.head
            {
                field.field = Some(m.resolve(law.field).into());
                field.family = Some(m.resolve(family).into());
                field.likelihoods = law
                    .likelihoods
                    .iter()
                    .map(|&f| m.resolve(f).into())
                    .collect();
                field.parameters = flatppl_stablehlo::Emitter::fixed_distribution_parameters(
                    m,
                    law.valid,
                    &distribution.args,
                );
            }
            field
        })
        .collect()
}

impl flatppl_determinizer::SamplingObserver for PosteriorTrace {
    fn source_measures(&mut self, m: &Module) -> Vec<NodeId> {
        let mut candidates = Vec::new();
        output_nodes(m, |node, owner, reified| {
            if expect_builtin_call(m, node, "bayesupdate").is_some()
                && !self
                    .by_node
                    .contains_key(&m.sample_origin(node).unwrap_or(node))
            {
                candidates.push((node, owner, reified));
            }
        });
        if candidates.is_empty() {
            return Vec::new();
        }
        // Prefer a direct use. A reified body can replace the declared value
        // of an observation or an entire likelihood when the caller applies it.
        candidates.sort_by_key(|&(node, _, reified)| (node, reified));
        candidates.dedup_by_key(|candidate| candidate.0);
        let mut scratch = m.clone();
        candidates
            .into_iter()
            .map(|(node, owner, reified)| {
                self.by_node.insert(node, self.sources.len());
                self.sources.push(describe(
                    &mut scratch,
                    (!reified).then_some(node),
                    m.resolve(owner).into(),
                ));
                node
            })
            .collect()
    }

    fn sampled_measures(&mut self, m: &Module, owner: Symbol, measures: &[NodeId]) {
        let mut scratch = m.clone();
        for &node in measures {
            let source = m.sample_origin(node).unwrap_or(node);
            let entry = self.by_node.get(&source).copied();
            let name = entry
                .map(|i| self.sources[i][0].binding.clone())
                .unwrap_or_else(|| m.resolve(owner).into());
            let mut actual = describe(&mut scratch, Some(node), name);
            let i = entry.unwrap_or_else(|| {
                let index = self.sources.len();
                self.by_node.insert(source, index);
                self.sources.push(actual.clone());
                index
            });
            for update in &mut actual {
                update.sampling = PosteriorSampling::ExactIid;
            }
            self.draws.insert(node, (i, actual));
        }
    }
}

impl PosteriorTrace {
    pub(crate) fn finish(mut self, lowered: Option<&Module>) -> Vec<PosteriorUpdate> {
        let Some(m) = lowered else {
            for update in self.sources.iter_mut().flatten() {
                update.sampling = PosteriorSampling::LoweringRefused;
            }
            return self.sources.into_iter().flatten().collect();
        };
        let mut selected = vec![Vec::new(); self.sources.len()];
        output_nodes(m, |node, _, _| {
            if let Some(instance) = m.sampled_measure(node)
                && let Some((i, actual)) = self.draws.remove(&instance)
            {
                for update in actual {
                    if !selected[i].contains(&update) {
                        selected[i].push(update);
                    }
                }
            }
        });
        self.sources
            .into_iter()
            .zip(selected)
            .flat_map(
                |(source, draws)| {
                    if draws.is_empty() { source } else { draws }
                },
            )
            .collect()
    }
}
