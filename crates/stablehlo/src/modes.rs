//! The mode builders — `emit_logdensity` (this task) and, from Task 6,
//! `emit_sample` — that turn a determinized FlatPDL [`Module`] into one
//! complete `func.func` StableHLO module.
//!
//! **Free parameters vs. fixed data.** A determinized module's top-level
//! bindings carry a `Phase` (spec §04): `elementof(...)`-declared parameters
//! are `Phase::Parameterized`, everything derived from them (including the
//! logdensity query itself) is *also* `Parameterized` (phase is a taint over
//! the whole dependent subtree, not a leaf marker), and already-pinned
//! observed data is `Phase::Fixed`. [`emit_logdensity`] therefore cannot use
//! "phase is `Parameterized`" alone to find the free parameters — it also
//! checks that the binding's RHS is *structurally* a bare `elementof(...)`
//! call, i.e. a parameter *declaration* rather than a computation that
//! merely depends on one (see [`is_free_param`]).
//!
//! Each free parameter becomes a fresh `func.func` argument
//! (`%argN : tensor<...>`, in top-level binding/source order — a
//! deterministic order derived from the module itself), and its RHS
//! `NodeId` is [`Emitter::bind`]-seeded to that argument's [`Value`] *before*
//! the query is walked. This is essential, not cosmetic: `elementof` has no
//! op-map lowering (it is a declaration, not a computation), so if the query
//! walk ever reached an unbound `elementof(...)` node directly, it would
//! refuse. Pre-binding means a `Ref` back to the parameter resolves straight
//! to the pre-allocated `Value` via `Emitter::lower_node`'s memo, and the
//! `elementof` node itself is never visited.
//!
//! Fixed data needs no special handling here: `Emitter::lower_node`'s
//! ordinary `Lit` dispatch already turns a fixed scalar leaf into a
//! `stablehlo.constant` when the query walk reaches it.
//!
//! **Finding the query.** Nothing in FlatPDL marks a binding as "the"
//! logdensity output — constructing the query is a step upstream of this
//! crate (the CLI verb / the testsuite harness, per the design doc). Every
//! `flatppl-determinizer` density fixture and golden test follows the same
//! shape, though: the density expression (`lp = logdensityof(...)`, or
//! equivalent) is the LAST public top-level binding in source order. This
//! module relies on that convention rather than re-deriving one.

use flatppl_core::{CallHead, Module, Node, NodeId, Phase};

use crate::EmitOptions;
use crate::emitter::Emitter;
use crate::mlir::{MlirTy, Value};
use crate::refuse::EmitError;
use crate::types::mlir_type_of;

/// Emit `@logdensity` for a determinized module `m` (see the module doc
/// comment for the free-param/fixed-data/query-finding rules). `m` is
/// assumed already FlatPDL-conformant — [`crate::emit`] (the mode router)
/// checks that once, up front.
pub fn emit_logdensity(m: &Module, opts: &EmitOptions) -> Result<String, EmitError> {
    let mut e = Emitter::new(m, opts.dtype);

    // Free parameters, in binding (source) order: bind each BEFORE the query
    // is walked (see module doc comment).
    let mut args: Vec<(String, MlirTy)> = Vec::new();
    for (_, binding) in m.bindings() {
        if !is_free_param(m, binding.rhs) {
            continue;
        }
        let name = format!("%arg{}", args.len());
        let ty = mlir_type_of(m, binding.rhs, opts.dtype)?;
        e.bind(
            binding.rhs,
            Value {
                ssa: name.clone(),
                ty: ty.clone(),
            },
        );
        args.push((name, ty));
    }

    let query = m.public_bindings().last().ok_or_else(|| {
        EmitError::whole("module has no public binding to emit as the logdensity query")
    })?;
    let result = e.lower_node(query.1.rhs)?;
    Ok(e.finish("logdensity", &args, &result))
}

/// A free-parameter declaration: `Phase::Parameterized` (spec §04 "Phase of
/// an expression") AND structurally a bare `elementof(...)` call. The phase
/// check alone is not enough — see the module doc comment on why phase is a
/// taint over the whole dependent subtree, not a parameter-leaf marker.
fn is_free_param(m: &Module, rhs: NodeId) -> bool {
    m.phase_of(rhs) == Some(Phase::Parameterized) && is_elementof_call(m, rhs)
}

fn is_elementof_call(m: &Module, id: NodeId) -> bool {
    matches!(
        m.node(id),
        Node::Call(c) if matches!(
            c.head,
            CallHead::Builtin(sym) if m.resolve(sym) == "elementof"
        )
    )
}
