//! Shared posterior updates and optional predictive log mass for each conjugate factor.

use super::{
    Data, MatchedUpdate, Update, UpdatedLaw, match_single_update, match_updates, op, real,
};
use crate::sufficient_statistics::{self as stats, ProofCache};
use flatppl_core::graph::{build_record, special_functions_alias};
use flatppl_core::{Call, CallHead, Module, Node, NodeId, Ref, RefNs, Scalar, Symbol};
use std::collections::HashSet;

mod beta;
mod counts;
mod gaussian;

/// Log mass of an unnormalized Beta update. An impossible observation has zero mass.
pub fn beta_log_evidence(m: &mut Module, posterior: NodeId) -> Option<NodeId> {
    let (_, update) = match_single_update(m, posterior)?;
    if !matches!(update.prior, UpdatedLaw::Beta(..)) {
        return None;
    }
    log_evidence(m, &[update])
}

/// Exact log evidence of independent conjugate fields, relative to the observation measure.
pub fn conjugate_log_evidence(m: &mut Module, posterior: NodeId) -> Option<NodeId> {
    if let Some((_, updates)) = match_updates(m, posterior) {
        return log_evidence(m, &updates);
    }
    Some(super::gaussian_block::update(m, posterior)?.log_evidence(m))
}

struct UpdatePlan {
    with_evidence: bool,
    batch: bool,
    serial: usize,
    logbeta: Option<NodeId>,
    boundary: HashSet<Symbol>,
    proofs: ProofCache,
    bound: Vec<(Symbol, NodeId)>,
}

impl UpdatePlan {
    fn summary(&mut self, m: &mut Module, rhs: NodeId) -> NodeId {
        if !self.batch
            || !stats::data_only_cached(m, rhs, &self.boundary, &self.boundary, &mut self.proofs)
        {
            return rhs;
        }
        // Another query in the same module can use the same observation statistic.
        if let Some(&(name, _)) = self
            .bound
            .iter()
            .find(|&&(_, other)| m.structural_eq(rhs, other))
        {
            return stats::local_ref(m, name);
        }
        let (name, reference) = stats::bind_summary(m, rhs, &mut self.serial);
        self.bound.push((name, rhs));
        reference
    }

    fn logbeta(&mut self, m: &mut Module) -> NodeId {
        *self.logbeta.get_or_insert_with(|| special_logbeta(m))
    }
}

pub(super) struct AppliedUpdate {
    pub law: UpdatedLaw,
    pub supported: NodeId,
    terms: Vec<NodeId>,
}

pub(super) fn apply(
    m: &mut Module,
    matched: &MatchedUpdate,
    with_evidence: bool,
) -> Option<AppliedUpdate> {
    let mut plan = UpdatePlan {
        with_evidence,
        batch: false,
        serial: 0,
        logbeta: None,
        boundary: flatppl_core::query::promoted_input_names(m),
        proofs: ProofCache::new(),
        bound: stats::summary_bindings(m).collect(),
    };
    let mut law = matched.prior;
    let mut supported = m.alloc(Node::Lit(Scalar::Bool(true)));
    let mut terms = Vec::new();
    for factor in &matched.factors {
        let data = factor.data;
        if data.count == Some(0) {
            continue;
        }
        plan.batch = data.count.is_some();
        let valid = op(m, "land", &[matched.model_valid, supported]);
        let (score, observed) = match (law, factor.row) {
            (UpdatedLaw::Beta(a, b), Update::Binomial(trials)) => {
                beta::factor(m, &mut plan, &mut law, [a, b], trials, data, valid)
            }
            (UpdatedLaw::Gamma(a, b), Update::Poisson) => {
                counts::poisson(m, &mut plan, &mut law, [a, b], data, valid)
            }
            (
                UpdatedLaw::Dirichlet(alpha),
                Update::Categories {
                    dimensions,
                    trials,
                    one_based,
                },
            ) => counts::categories(
                m,
                &mut plan,
                &mut law,
                alpha,
                data,
                (dimensions, trials, one_based),
                valid,
            ),
            (UpdatedLaw::Normal(..) | UpdatedLaw::InverseGamma(..), _) => {
                gaussian::factor(m, &mut plan, &mut law, factor.row, data, valid)?
            }
            _ => return None,
        };
        supported = op(m, "land", &[supported, observed]);
        if with_evidence {
            terms.push(score);
        }
    }
    Some(AppliedUpdate {
        law,
        supported,
        terms,
    })
}

fn log_evidence(m: &mut Module, matched: &[MatchedUpdate]) -> Option<NodeId> {
    let (first, rest) = matched.split_first()?;
    let AppliedUpdate {
        mut supported,
        mut terms,
        ..
    } = apply(m, first, true)?;
    let mut model_valid = first.model_valid;
    for field in rest {
        let applied = apply(m, field, true)?;
        supported = op(m, "land", &[supported, applied.supported]);
        model_valid = op(m, "land", &[model_valid, field.model_valid]);
        terms.extend(applied.terms);
    }
    let sum = if terms.is_empty() {
        real(m, 0.0)
    } else {
        flatppl_core::graph::fold_add(m, &terms)
    };
    let minus_inf = real(m, f64::NEG_INFINITY);
    let log_mass = op(m, "ifelse", &[supported, sum, minus_inf]);
    let anchor = parameter_anchor(m, model_valid);
    Some(op(m, "add", &[log_mass, anchor]))
}

/// A unit-probability event preserves the source parameter check.
pub(super) fn parameter_anchor(m: &mut Module, model_valid: NodeId) -> NodeId {
    let one = real(m, 1.0);
    let invalid = real(m, 2.0);
    let p = op(m, "ifelse", &[model_valid, one, invalid]);
    let name = m.intern("p");
    let parameters = build_record(m, &[(name, p)]);
    let name = m.intern("Bernoulli");
    let kernel = m.alloc(Node::Const(name));
    op(m, "builtin_logdensityof", &[kernel, parameters, one])
}

fn safe(m: &mut Module, valid: NodeId, value: NodeId, fallback: f64) -> NodeId {
    let fallback = real(m, fallback);
    op(m, "ifelse", &[valid, value, fallback])
}

fn special_logbeta(m: &mut Module) -> NodeId {
    let alias = special_functions_alias(m);
    let name = m.intern("logbeta");
    m.alloc(Node::Ref(Ref {
        ns: RefNs::Module(alias),
        name,
    }))
}

fn log_beta(m: &mut Module, function: NodeId, x: NodeId, y: NodeId) -> NodeId {
    m.alloc(Node::Call(Call {
        head: CallHead::User(function),
        args: Box::new([x, y]),
        named: Box::new([]),
        inputs: None,
    }))
}
