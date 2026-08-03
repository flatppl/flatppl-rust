use crate::refuse::{NonConformKind, NonConformance};
use flatppl_core::{CallHead, Inputs, Module, Node, NodeId, Phase, Ref, RefNs, Type};

/// FlatPDL conformance over `flatppl-infer` output: no `Measure`/`Likelihood`-typed node;
/// `Kernel` type only as an argument of a `builtin_*` primitive (the constructor-tag arg —
/// its position varies: arg 0 for `builtin_logdensityof` and the transports, arg 1 for
/// `builtin_sample`); no `Stochastic` phase; no residual `Type::Failed` node (a
/// generic backstop — any node `flatppl-infer` could not type is ill-formed, whatever
/// produced it); and no dangling `(%ref self <name>)` — a body reference or a
/// `functionof`/`kernelof` reification `Inputs` boundary entry naming a binding
/// that is not present in the module (the self-check against any binding-removal
/// pass, e.g. root-based DCE, Buffy #263 Pass 4-A, leaving a stranded pointer);
/// no residual `CallHead::User` application; and no wrong-arity call to one of
/// the six `builtin_*` primitives (§07). Mostly a read of the inferred
/// side-tables — run `infer` first — plus the two structural checks, which read
/// the call shape because `flatppl-infer` types both shapes without complaint.
pub fn is_flatpdl(m: &Module) -> Result<(), Vec<NonConformance>> {
    let mut bad = Vec::new();
    for (_bid, binding) in m.bindings() {
        visit(m, binding.rhs, None, &mut bad);
    }
    if bad.is_empty() { Ok(()) } else { Err(bad) }
}

/// Required argument count of a `builtin_*` primitive, and whether that count is
/// a minimum. §07 "Measure kernel evaluation primitives" fixes each signature;
/// only `builtin_sample` is variadic, its trailing sample shape `n, m, …` being
/// optional ("or a scalar `X` if no `n, m, ...` are given"). A name that is not one
/// of the six is not checked here — there is no arity table for the rest of the
/// builtins, and duplicating `flatppl-infer`'s per-op rules would drift from them.
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

// `parent_builtin`: the interned name of the enclosing builtin call head, so a
// `Kernel`-typed node is allowed iff it sits inside a `builtin_*` call. The kernel arg's
// position varies by primitive, so the check is by-enclosing-call, not by-index; non-kernel
// args are never `Kernel`-typed, so this admits no stray kernel.

fn visit(m: &Module, id: NodeId, parent_builtin: Option<&str>, bad: &mut Vec<NonConformance>) {
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
        Some(Type::Kernel { .. }) if !parent_builtin.is_some_and(|h| h.starts_with("builtin_")) => {
            bad.push(NonConformance {
                node: id,
                kind: NonConformKind::KernelNotBuiltinArg,
                reason: "kernel outside a builtin_* argument".into(),
            })
        }
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
    {
        if m.binding_by_name(*name).is_none() {
            bad.push(NonConformance {
                node: id,
                kind: NonConformKind::DanglingSelfRef,
                reason: format!(
                    "dangling (%ref self {}) — no binding of that name in the module",
                    m.resolve(*name)
                ),
            });
        }
    }
    if let Node::Call(c) = m.node(id) {
        // A residual `CallHead::User` application. `canon::inline` beta-reduces
        // every user call it CAN reduce and leaves the rest in place refuse-free,
        // so an unresolved callee or an arity mismatch reaches exit as a live
        // `(%call (%ref self f) …)`. FlatPDL admits deterministic ops and the six
        // `builtin_*` primitives (§07 "Measure kernel evaluation primitives");
        // an application of a user-defined callable is neither, and the surface
        // printer spells it `f(x)` exactly like a builtin call, so nothing
        // downstream can tell the two apart.
        if matches!(c.head, CallHead::User(_)) {
            bad.push(NonConformance {
                node: id,
                kind: NonConformKind::ResidualUserCall,
                reason: "residual user call — FlatPDL admits deterministic ops and the six \
                         builtin_* primitives only"
                    .into(),
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

    // Collect children and determine the builtin head symbol before recursing,
    // keeping the `m.node(id)` borrow scoped so it doesn't conflict with
    // `m.resolve(sym)` in the recursive call.
    let (children, head_sym) = {
        let node = m.node(id);
        let sym = match node {
            Node::Call(c) => match c.head {
                CallHead::Builtin(op) => Some(op),
                _ => None,
            },
            _ => None,
        };
        (node.children(), sym)
    };
    let this_builtin: Option<&str> = head_sym.map(|op| m.resolve(op));

    for child in children {
        visit(m, child, this_builtin, bad);
    }
}
