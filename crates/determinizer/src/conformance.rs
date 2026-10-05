use crate::refuse::{NonConformKind, NonConformance};
use flatppl_core::{CallHead, Inputs, Module, Node, NodeId, Phase, Ref, RefNs, Type};
use std::collections::HashSet;

/// FlatPDL conformance over `flatppl-infer` output: no `Measure`/`Likelihood`-typed node;
/// `Kernel` type only as an argument of a `builtin_*` primitive (the constructor-tag arg —
/// its position varies: arg 0 for `builtin_logdensityof` and the transports, arg 1 for
/// `builtin_sample`); no `Stochastic` phase; no residual `Type::Failed` node (a
/// generic backstop — any node `flatppl-infer` could not type is ill-formed, whatever
/// produced it); and no dangling `(%ref self <name>)` — a body reference or a
/// `functionof`/`kernelof` reification `Inputs` boundary entry naming a binding
/// that is not present in the module (the self-check against any binding-removal
/// pass, e.g. root-based DCE, leaving a stranded pointer).
/// Three further rejections: no residual `CallHead::User` application, no
/// wrong-arity call to one of the six `builtin_*` primitives (§07), and no bare
/// atom or unresolvable call head naming something outside the `base`
/// namespace — a free variable, which `flatppl-infer` already refuses at source
/// but which this scan re-checks structurally. Mostly a read
/// of the inferred side-tables — run `infer` first — plus those structural
/// checks, which read the call shape because `flatppl-infer` types both shapes
/// without complaint.
pub fn is_flatpdl(m: &Module) -> Result<(), Vec<NonConformance>> {
    is_flatpdl_with_options(m, &crate::LoweringOptions::default())
}

/// Check the same deterministic contract while admitting the target's resolved
/// standard functions. This does not admit arbitrary residual user calls.
pub fn is_flatpdl_with_options(
    m: &Module,
    options: &crate::LoweringOptions<'_>,
) -> Result<(), Vec<NonConformance>> {
    let mut bad = Vec::new();
    let tags = kernel_tag_slots(m);
    let mut seen = HashSet::new();
    let mut pending = Vec::new();
    for (_bid, binding) in m.bindings() {
        pending.push((binding.rhs, false));
        while let Some((id, kernel_allowed)) = pending.pop() {
            // A shared kernel can occur both inside and outside a builtin_* call.
            // Its two contexts must be checked independently.
            if !seen.insert((id, kernel_allowed)) {
                continue;
            }
            check_node(m, id, kernel_allowed, &tags, options, &mut bad);
            let node = m.node(id);
            let child_kernel_allowed = matches!(node, Node::Call(c)
                if matches!(c.head, CallHead::Builtin(op)
                    if m.resolve(op).starts_with("builtin_")));
            let start = pending.len();
            node.for_each_child(|child| pending.push((child, child_kernel_allowed)));
            pending[start..].reverse();
        }
    }
    if bad.is_empty() { Ok(()) } else { Err(bad) }
}

/// The exact node ids in a kernel-TAG SLOT, from
/// `flatppl_infer::builtins::kernel_tag_node` — the same table spec-§04 name
/// resolution uses, so the resolver and this scan cannot disagree about which
/// argument is a tag.
fn kernel_tag_slots(m: &Module) -> HashSet<NodeId> {
    let mut out = HashSet::new();
    let mut seen = HashSet::new();
    let mut pending = Vec::new();
    for (_bid, b) in m.bindings() {
        pending.push(b.rhs);
        while let Some(id) = pending.pop() {
            if !seen.insert(id) {
                continue;
            }
            if let Node::Call(c) = m.node(id) {
                out.extend(flatppl_infer::builtins::kernel_tag_node(m, c));
            }
            m.node(id).for_each_child(|child| pending.push(child));
        }
    }
    out
}

/// Required argument count of a `builtin_*` primitive, and whether that count is
/// a minimum. §07 "Measure kernel evaluation primitives" fixes each signature;
/// only `builtin_sample` is variadic, its trailing sample shape `n, m, …` being
/// optional ("or a scalar `X` if no `n, m, ...` are given"). A name that is not one
/// of the six is not checked here — there is no arity table for the rest of the
/// builtins, and duplicating `flatppl-infer`'s per-op rules would drift from them.
///
/// The tag-INDEX counterpart is `flatppl_infer::builtins::kernel_tag_index`. It
/// lives in `flatppl-infer` rather than beside this table because name resolution
/// needs it and the crate direction only runs one way.
fn builtin_primitive_arity(name: &str) -> Option<(usize, bool)> {
    match name {
        "builtin_logdensityof"
        | "builtin_touniform"
        | "builtin_fromuniform"
        | "builtin_tonormal"
        | "builtin_fromnormal" => Some((3, false)),
        "builtin_sample" => Some((3, true)),
        _ => None,
    }
}

// A `Kernel`-typed node is allowed inside a `builtin_*` call. Its argument
// position varies by primitive, so the enclosing call supplies the context.
fn check_node(
    m: &Module,
    id: NodeId,
    kernel_allowed: bool,
    tags: &HashSet<NodeId>,
    options: &crate::LoweringOptions<'_>,
    bad: &mut Vec<NonConformance>,
) {
    if matches!(m.phase_of(id), Some(Phase::Stochastic)) {
        bad.push(NonConformance {
            node: id,
            kind: NonConformKind::StochasticPhase,
            reason: "stochastic-phase node (a `draw` survives)".into(),
        });
    }
    match m.type_of(id) {
        Some(Type::Measure { .. }) => bad.push(NonConformance {
            node: id,
            kind: NonConformKind::MeasureTyped,
            reason: "measure-typed node".into(),
        }),
        Some(Type::Likelihood { .. }) => bad.push(NonConformance {
            node: id,
            kind: NonConformKind::LikelihoodTyped,
            reason: "likelihood-typed node".into(),
        }),
        Some(Type::Kernel { .. }) if !kernel_allowed => bad.push(NonConformance {
            node: id,
            kind: NonConformKind::KernelNotBuiltinArg,
            reason: "kernel outside a builtin_* argument".into(),
        }),
        Some(Type::Failed(reason)) => bad.push(NonConformance {
            node: id,
            kind: NonConformKind::Failed,
            reason: reason.to_string(),
        }),
        _ => {}
    }

    // Dangling self-ref check: a `(%ref self <name>)` — as this node itself, or
    // as one of the current node's reification `Inputs` boundary entries (which
    // `children()`/`for_each_child` deliberately exclude, see
    // `driver::collect_referenced_names`) — must name a binding still present
    // in the module. A miss here means some earlier pass (a binding-removal
    // pass, first introduced by root-based DCE) dropped a binding something
    // else still points at.
    if let Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }) = m.node(id)
        && m.binding_by_name(*name).is_none()
    {
        bad.push(NonConformance {
            node: id,
            kind: NonConformKind::DanglingSelfRef,
            reason: format!(
                "dangling (%ref self {}) — no binding of that name in the module",
                m.resolve(*name)
            ),
        });
    }
    // Free-name check, over BOTH shapes the `base` namespace can be spelled in:
    // a bare atom (`Node::Const`) and a builtin call head
    // (`CallHead::Builtin`). Either one naming nothing in `base` binds nowhere —
    // a free variable, or a call to a function that does not exist. Structural on
    // purpose: `Failed` above catches the same node only while `flatppl-infer`
    // types it that way, and the whole point of this arm is to hold if a future
    // path types one as an ordinary value again.
    let free_name = match m.node(id) {
        Node::Const(sym) => Some((*sym, "free variable")),
        Node::Call(c) => match c.head {
            // A reification's head is the construct itself (`functionof` /
            // `kernelof`), always a real builtin, so no special case is needed.
            CallHead::Builtin(op) => Some((op, "free call head")),
            CallHead::User(_) => None,
        },
        _ => None,
    };
    if let Some((sym, what)) = free_name {
        let name = m.resolve(sym);
        // A distribution constructor IN THE TAG SLOT is exempt: the determiniser
        // deliberately emits a §09 constructor BARE as a kernel tag, dropping its
        // module qualification because both engines key the registry bare. A tag
        // is not a variable reference. Position AND name, both required — the
        // name alone would exempt a constructor sitting in the observed-value,
        // params or rngstate argument, and `kallen` (a §09 non-constructor) is
        // caught in every position.
        let is_tag = tags.contains(&id) && flatppl_infer::builtins::is_kernel_tag_name(name);
        if !is_tag && !flatppl_infer::builtins::is_base_name(name) {
            bad.push(NonConformance {
                node: id,
                kind: NonConformKind::FreeBareName,
                reason: format!("{what} `{name}` — names no built-in and no binding"),
            });
        }
    }
    if let Node::Call(c) = m.node(id) {
        // A residual `CallHead::User` application. `canon::inline` leaves a call it
        // cannot reduce in place refuse-free, so an unresolved callee or an arity mismatch
        // reaches exit as a live `(%call (%ref self f) …)`. FlatPDL admits deterministic
        // ops and the six `builtin_*` primitives (§07 "Measure kernel evaluation
        // primitives"), and the surface printer spells a user call `f(x)` exactly like a
        // builtin one, so nothing downstream can tell them apart.
        let retained = match c.head {
            CallHead::User(callee) if !options.retain_standard_functions.is_empty() => {
                crate::stdfn::standard_function(m, callee)
                    .is_some_and(|(module, member)| options.retains(&module, &member))
            }
            _ => false,
        };
        if matches!(c.head, CallHead::User(_)) && !retained {
            // A §09 standard-module FUNCTION member with no base-op form
            // (`special-functions`, `ext-linear-algebra`, the Wigner functions, the
            // higher-order `distances` members) reaches here as an ordinary
            // residual user call. Name it: the generic reason gives no clue which
            // member needs an engine primitive. `crate::stdfn` rewrites the
            // members that do have one before the loop.
            let reason = match crate::stdfn::residual_std_member(m, id) {
                Some(name) => format!(
                    "§09 standard-module function `{name}` has no lowering to FlatPDL base \
                     operations — it needs an engine primitive"
                ),
                None => "residual user call — FlatPDL admits deterministic ops and the six \
                         builtin_* primitives only"
                    .to_string(),
            };
            bad.push(NonConformance {
                node: id,
                kind: NonConformKind::ResidualUserCall,
                reason,
            });
        }
        // Arity of the six `builtin_*` primitives. `flatppl-infer` has no arity
        // rule for them — `builtin_logdensityof(1.0, 2.0)` types as
        // `Scalar(Real)`, not `Type::Failed` — so the generic `Failed` backstop
        // never fires and a mis-arity primitive would pass the gate.
        if let CallHead::Builtin(op) = c.head {
            let name = m.resolve(op);
            if let Some((want, variadic)) = builtin_primitive_arity(name) {
                let got = c.args.len() + c.named.len();
                if got < want || (!variadic && got > want) {
                    bad.push(NonConformance {
                        node: id,
                        kind: NonConformKind::BuiltinArity,
                        reason: format!(
                            "{name} takes {}{want} arguments, got {got}",
                            if variadic { "at least " } else { "" }
                        ),
                    });
                }
            }
        }
        if let Some(Inputs::Spec(entries)) = &c.inputs {
            for (input_name, r) in entries.iter() {
                if r.ns == RefNs::SelfMod && m.binding_by_name(r.name).is_none() {
                    bad.push(NonConformance {
                        node: id,
                        kind: NonConformKind::DanglingSelfRef,
                        reason: format!(
                            "dangling reification input `{}` = (%ref self {}) — no binding of \
                             that name in the module",
                            m.resolve(*input_name),
                            m.resolve(r.name)
                        ),
                    });
                }
            }
        }
    }
}
