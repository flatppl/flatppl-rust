//! Decisions from the actual reduction pass, separate from the query ABI.

use std::fmt;

use super::{call, resolve};
use flatppl_core::{Inputs, Module, Node, NodeId, Symbol};
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupPartition {
    pub observations: usize,
    pub groups: usize,
    pub singletons: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReductionOutcome {
    Reduced {
        rule: &'static str,
        partition: Option<GroupPartition>,
    },
    Retained {
        reason: &'static str,
    },
    Disabled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReductionDecision {
    pub binding: String,
    /// Node in the module at this pass, not a surface source location.
    pub node: NodeId,
    pub family: String,
    pub outcome: ReductionOutcome,
}

/// Decisions for recognized independent primitive-density sums. Other expression
/// shapes do not produce a decision.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReductionReport {
    pub decisions: Vec<ReductionDecision>,
}

impl ReductionReport {
    /// Inspect disabled candidates without changing the supplied module.
    pub fn record_disabled(&mut self, m: &Module) {
        super::reduce(&mut m.clone(), HashSet::new(), false, Some(self), false);
    }
}

impl fmt::Display for ReductionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for decision in &self.decisions {
            write!(
                f,
                "{} [{:?}] {}: ",
                decision.binding, decision.node, decision.family
            )?;
            match &decision.outcome {
                ReductionOutcome::Reduced { rule, partition } => {
                    write!(f, "reduced by {rule}")?;
                    if let Some(p) = partition {
                        write!(
                            f,
                            " ({} observations, {} occupied groups, {} singletons)",
                            p.observations, p.groups, p.singletons
                        )?;
                    }
                    writeln!(f)?;
                }
                ReductionOutcome::Retained { reason } => writeln!(f, "retained: {reason}")?,
                ReductionOutcome::Disabled => writeln!(f, "disabled")?,
            }
        }
        Ok(())
    }
}

// Recognize candidates without normalizing or allocating expressions. This
// keeps tracing from changing emission, including source and FlatPIR output.
pub(super) fn family(m: &Module, id: NodeId, boundary: &HashSet<Symbol>) -> Option<Symbol> {
    let sum = call(m, id, "sum")?;
    let [arg] = *sum.args else { return None };
    let c = call(m, resolve(m, arg, boundary), "broadcast")?;
    let (head, args) = c.args.split_first()?;
    let kernel = if matches!(m.node(*head),Node::Const(s) if m.resolve(*s)=="builtin_logdensityof")
    {
        *args.first()?
    } else {
        let function = call(m, resolve(m, *head, boundary), "functionof")?;
        if !matches!(function.inputs, Some(Inputs::Spec(_))) {
            return None;
        }
        let [body] = *function.args else { return None };
        *call(m, body, "builtin_logdensityof")?.args.first()?
    };
    match m.node(kernel) {
        Node::Const(s) => Some(*s),
        _ => None,
    }
}
