//! Spec §05 axis-position checks: where an `Axis` and an `AxisList` may appear.
//!
//! §05 "Axis names and aggregation": "an axis name is legal only as an entry in
//! `aggregate`'s `output_axes` axis list, as an index inside `[...]` within the
//! body, or as a binder on the left-hand side of `:=`. Used anywhere else it is
//! a static error."
//!
//! §05 "Note on axis names": "The grammar likewise admits `AxisList` as a
//! `Primary`, but it is legal only as the `output_axes` argument of an
//! `aggregate` or `metricsum` call and as the axis-list binder of an
//! `AggregateBinding` or `MetricsumBinding`; anywhere else it is a static
//! error. Unlike `ArrayLiteral`, `AxisList` may be empty".
//!
//! Both rules are *positional*, so they need the parent chain the memoised
//! type trace does not carry — hence a separate structural pre-pass.
//!
//! The two `:=` statement forms need no case of their own: the parser lowers
//! `C[.i] := e` to `aggregate(sum, [.i], e)` and `g: C[.i^] := e` to
//! `metricsum(g, [.i^], e)`, so a binder and a spelled-out `output_axes`
//! argument are the same IR position.

use flatppl_core::{CallHead, Module, Node, NodeId};

use crate::Diagnostic;

/// The position a node occupies, for the two §05 rules.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// The `output_axes` argument of an `aggregate` / `metricsum` call — the
    /// only place an `AxisList` is legal, and where both `:=` binders land.
    OutputAxes,
    /// An index argument of a `get` lexically inside an aggregation body — the
    /// only place a bare axis name is legal. §04 "Multi-axis aggregation" spells
    /// both surface forms out: "array indexing may contain axis names, like
    /// `A[.i, 1, .j]` or `get(A, .i, 1, .j)`".
    Index,
    /// Any other expression position: neither an `Axis` nor an `AxisList`.
    Value,
}

/// Every §05 axis-position violation in `module`, as `(offending node,
/// diagnostic)`. The caller marks each node `Failed` before the type trace runs,
/// so a refused bracket raises exactly one error rather than a cascade.
pub(crate) fn position_errors(module: &Module) -> Vec<(NodeId, Diagnostic)> {
    let mut out = Vec::new();
    for (_id, b) in module.bindings() {
        scan(module, b.rhs, Slot::Value, false, &mut out);
    }
    out
}

fn scan(m: &Module, id: NodeId, slot: Slot, in_body: bool, out: &mut Vec<(NodeId, Diagnostic)>) {
    let node = m.node(id);
    let Node::Call(c) = node else {
        if let Node::Axis(ax) = node {
            if slot != Slot::Index {
                out.push((id, axis_error(m, id, ax.name)));
            }
        }
        return;
    };

    if is_axis_list(m, c) {
        if slot != Slot::OutputAxes {
            out.push((id, axis_list_error(id, c.args.is_empty())));
        }
        // Reported through the bracket either way: the entries of a legal list
        // are legal, and the entries of a refused one add nothing.
        return;
    }

    let head = match c.head {
        CallHead::Builtin(h) => Some(m.resolve(h)),
        CallHead::User(callee) => {
            scan(m, callee, Slot::Value, in_body, out);
            None
        }
    };

    match head {
        // `aggregate(f_reduction, output_axes, expr)` /
        // `metricsum(metric, output_axes, expr)` (spec §04).
        Some("aggregate" | "metricsum") => {
            for (i, &a) in c.args.iter().enumerate() {
                let child = if i == 1 {
                    Slot::OutputAxes
                } else {
                    Slot::Value
                };
                scan(m, a, child, in_body || i == 2, out);
            }
            // No keyword branch: the keyword spelling of a distinguished input
            // is a static error (§04 "Calling conventions": "A distinguished
            // input has no name and so cannot be passed by keyword"), refused by
            // `ops::special_arity_check` before any axis scan runs. Where §04
            // refers to such an input by a name, the name "identifies the input
            // in prose only" — adjudicated 2026-09-03, merged as
            // flatppl-design PR #109; see
            // `flatppl-dev/adjudication-keyword-distinguished-inputs.md`. An
            // earlier revision mapped `output_axes`/`expr` by keyword here, which
            // made this the one place an illegal spelling was given a meaning.
        }
        // `A[.i, 1, .j]` and its `get(A, .i, 1, .j)` spelling: the indices are
        // index positions, the indexed object is not.
        Some("get") => {
            let index_slot = if in_body { Slot::Index } else { Slot::Value };
            for (i, &a) in c.args.iter().enumerate() {
                let child = if i == 0 { Slot::Value } else { index_slot };
                scan(m, a, child, in_body, out);
            }
            for n in c.named.iter() {
                scan(m, n.value, Slot::Value, in_body, out);
            }
        }
        _ => {
            for &a in c.args.iter() {
                scan(m, a, Slot::Value, in_body, out);
            }
            for n in c.named.iter() {
                scan(m, n.value, Slot::Value, in_body, out);
            }
        }
    }
}

/// Is this `vector` call an `AxisList` rather than an `ArrayLiteral`?
///
/// Both print as the same `vector` call, so the entries decide. An empty one is
/// an `AxisList`: §05's `ArrayLiteral ::= "[" Expression ("," Expression)* ","?
/// "]"` admits no empty form, while "Unlike `ArrayLiteral`, `AxisList` may be
/// empty".
fn is_axis_list(m: &Module, c: &flatppl_core::Call) -> bool {
    if !matches!(c.head, CallHead::Builtin(h) if m.resolve(h) == "vector") {
        return false;
    }
    if !c.named.is_empty() {
        return false;
    }
    c.args.iter().all(|&a| matches!(m.node(a), Node::Axis(_)))
}

fn axis_list_error(id: NodeId, empty: bool) -> Diagnostic {
    let empty_note = if empty {
        " `[]` has no other reading: §05's `ArrayLiteral` admits no empty form, so this is an \
         empty axis list, not an empty vector — a derived size (`filter`, `selectbins`, \
         `lengthof`) is how an empty array arises."
    } else {
        ""
    };
    Diagnostic::error_at(
        id,
        format!(
            "an axis list is legal only as `output_axes` (spec §05 \"Note on axis names\": \"The \
             grammar likewise admits `AxisList` as a `Primary`, but it is legal only as the \
             `output_axes` argument of an `aggregate` or `metricsum` call and as the axis-list \
             binder of an `AggregateBinding` or `MetricsumBinding`; anywhere else it is a static \
             error\").{empty_note}"
        ),
    )
}

fn axis_error(m: &Module, id: NodeId, name: flatppl_core::Symbol) -> Diagnostic {
    Diagnostic::error_at(
        id,
        format!(
            "axis name `.{}` is out of position (spec §05 \"Axis names and aggregation\": \"an \
             axis name is legal only as an entry in `aggregate`'s `output_axes` axis list, as an \
             index inside `[...]` within the body, or as a binder on the left-hand side of `:=`. \
             Used anywhere else it is a static error\"); axis names are not values",
            m.resolve(name)
        ),
    )
}
