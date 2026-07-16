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
    /// A `(%ref self <name>)` — as an ordinary body sub-node OR a `functionof`/
    /// `kernelof` reification `Inputs` boundary entry — names a binding that is
    /// not present in the module. Permanent self-check against any
    /// binding-removal pass (root-based DCE, Buffy #263 Pass 4-A, is the first
    /// one) dropping a binding something still points at.
    DanglingSelfRef,
}
