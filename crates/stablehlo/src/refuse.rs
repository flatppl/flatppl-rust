//! `EmitError` — refuse-don't-mislower for the StableHLO emitter: a construct
//! the emitter cannot lower is reported with a precise message, never guessed.

use flatppl_core::NodeId;

/// A construct `emit` cannot lower to StableHLO — reported, never mis-lowered.
///
/// `node` localizes the error to a specific IR node when one is available;
/// `whole` (`node: None`) is used for module-level refusals (e.g. the input
/// is not FlatPDL at all).
#[derive(Debug)]
pub struct EmitError {
    pub msg: String,
    pub node: Option<NodeId>,
}

impl EmitError {
    /// A refusal localized to `node`.
    pub fn at(node: NodeId, msg: impl Into<String>) -> Self {
        EmitError {
            msg: msg.into(),
            node: Some(node),
        }
    }

    /// A refusal with no single localizing node (e.g. a module-level check).
    pub fn whole(msg: impl Into<String>) -> Self {
        EmitError {
            msg: msg.into(),
            node: None,
        }
    }
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stablehlo: {}", self.msg)
    }
}
