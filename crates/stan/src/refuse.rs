//! Refusals. A construct either has no sensible Stan form (`Never`) or has one
//! this emitter does not write yet (`NotYet`). The testsuite counts them apart.

use flatppl_core::NodeId;
use std::fmt;

/// Why a target cannot be emitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RefusalKind {
    /// The module names no single target.
    NoTarget,
    /// More than one binding qualifies as the default target.
    AmbiguousTarget,
    /// A continuous draw the target integrates out.
    LatentMarginalized,
    /// A discrete draw with unbounded support the target integrates out.
    UnboundedDiscreteLatent,
    /// A prior field with a discrete law.
    DiscreteParameter,
    /// `kchain` with a continuous intermediate variate.
    KchainMarginal,
    /// `normalize` of a weighted Lebesgue measure with no closed-form mass.
    NumericNormalizer,
    /// `pushfwd` through a map that is not a bijection.
    NonBijectivePushfwd,
    /// A function with no Stan counterpart.
    NoStanFunction,
    /// A construct with a Stan form this emitter does not write yet.
    NotYetSupported,
}

impl RefusalKind {
    /// `true` for a construct Stan cannot express at all.
    pub fn is_never(&self) -> bool {
        !matches!(
            self,
            RefusalKind::NotYetSupported | RefusalKind::NoTarget | RefusalKind::AmbiguousTarget
        )
    }

    /// The class a message leads with.
    pub fn class(&self) -> &'static str {
        match self {
            RefusalKind::NotYetSupported => "not yet supported",
            RefusalKind::NoTarget | RefusalKind::AmbiguousTarget => "no target",
            _ => "no Stan form",
        }
    }
}

/// A refusal, localized to the node of the user's construct when one exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub kind: RefusalKind,
    pub node: Option<NodeId>,
    /// The construct as the user writes it (`superpose`, `draw` of `z`, …).
    pub construct: String,
    pub reason: String,
}

impl Refusal {
    pub(crate) fn new(
        kind: RefusalKind,
        node: Option<NodeId>,
        construct: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Refusal {
            kind,
            node,
            construct: construct.into(),
            reason: reason.into(),
        }
    }

    pub(crate) fn not_yet(node: Option<NodeId>, construct: impl Into<String>) -> Self {
        let construct = construct.into();
        let reason = format!("`{construct}` is not supported by the Stan emitter yet");
        Refusal::new(RefusalKind::NotYetSupported, node, construct, reason)
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind.class(), self.reason)
    }
}

impl std::error::Error for Refusal {}

pub(crate) type Result<T> = std::result::Result<T, Refusal>;
