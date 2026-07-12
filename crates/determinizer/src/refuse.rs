use flatppl_core::NodeId;

/// A construct the determiniser cannot legalize to FlatPDL — reported, never mis-lowered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefuseError {
    pub node: NodeId,
    pub construct: String,
    pub reason: String,
}

/// A FlatPDL-conformance violation found by `is_flatpdl`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonConformance {
    pub node: NodeId,
    pub kind: NonConformKind,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonConformKind {
    MeasureTyped,
    LikelihoodTyped,
    StochasticPhase,
    KernelNotBuiltinArg,
    /// A node `flatppl-infer` could not type (`Type::Failed` — "inference
    /// attempted but failed; the module is ill-formed", `flatppl_core::ty`)
    /// survived into what should be FlatPDL output. Generic backstop: an
    /// ill-formed node must never pass as valid FlatPDL, whatever produced it.
    Failed,
}
