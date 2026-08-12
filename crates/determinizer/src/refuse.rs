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
    /// A `CallHead::User` application survived into FlatPDL. FlatPDL is
    /// deterministic ops plus the six `builtin_*` primitives (§07 "Measure
    /// kernel evaluation primitives"); an application of a user-defined callable
    /// is neither, and no consumer can evaluate one.
    ResidualUserCall,
    /// A call to one of the six `builtin_*` primitives carries the wrong number
    /// of arguments for its §07 signature. `flatppl-infer` has no arity rule for
    /// these, so a mis-arity primitive is typed, not `Type::Failed`.
    BuiltinArity,
    /// A bare atom (`Node::Const`) or a builtin call head
    /// (`CallHead::Builtin`) names nothing in the `base` namespace: a FREE
    /// VARIABLE, or a call to a function that does not exist, in the emitted
    /// FlatPDL — neither of which any consumer can evaluate. `flatppl-infer`
    /// rejects both at their source (spec §04 "Name resolution"), so this is the
    /// structural backstop for any future path that synthesises or re-admits
    /// one — it reads the name, not the type table, so it holds even when the
    /// node is typed rather than `Type::Failed`.
    FreeBareName,
}
