//! Structural disintegration split for the explicit-DAG case (spec §06
//! "Structural disintegration").
//!
//! `disintegrate(selector, lawof(record(field…)))` returns `(kernel,
//! marginal)`, the structural inverse of `jointchain(marginal, kernel) ≡
//! joint`. The SELECTED fields (named by the selector) become the kernel's
//! output law, conditioned on the NON-selected fields, which are the kernel's
//! `%specinputs` boundary; the NON-selected fields also form the marginal law.
//!
//! Only the explicit `lawof(record(…))` DAG case is handled here — every other
//! shape (a non-`lawof(record)` measure, a selector that is not a clean
//! field-name partition of the record, an empty selected or non-selected set)
//! yields `None`, and the caller refuses (§06 permits refusing intractable /
//! non-explicit disintegrations; refuse-don't-mislower).

use crate::density::{expect_builtin_call, resolve_ref_one};
use flatppl_core::{
    Call, CallHead, Inputs, Module, NamedArg, NamedKind, Node, NodeId, Ref, RefNs, Scalar, Symbol,
};
use std::collections::HashSet;

/// Split `disint_node = disintegrate(selector, lawof(record(fields)))` into a
/// `(kernel, marginal)` pair, structurally (spec §06 "Structural
/// disintegration").
///
/// - `kernel` = `kernelof(record(<selected %fields, verbatim value nodes>),
///   %specinputs([(non_sel_name, non_sel_value_ref), …]))` — the conditional of
///   the SELECTED fields, with the NON-selected fields as boundary inputs (so
///   `likelihoodof(kernel, data)` scores the selected fields given the rest).
/// - `marginal` = `lawof(record(<non-selected %fields, verbatim value nodes>))`
///   — the joint law over the NON-selected fields.
///
/// Returns `None` (caller refuses) for any shape outside the explicit DAG case:
/// - `disint_node` is not a 2-arg `disintegrate` call;
/// - the measure (after one ref hop) is not `lawof(record(field…))`, or the
///   record carries positional args / a non-`%field` named entry;
/// - the selector is not a string literal or `vector` of string literals
///   naming fields of the record (a name not present in the record, or an
///   empty selector, refuses);
/// - the selected or the non-selected set is empty (a vacuous split);
/// - a NON-selected field's value is not a bare `%ref` (its boundary-input
///   `Ref` cannot be formed — outside the explicit-DAG shape);
/// - the NON-selected marginal is not closed: some non-selected field
///   transitively DEPENDS on a selected field (a causally REVERSED selector), so
///   `jointchain(marginal, kernel) ≢ joint` (§06 "Structural disintegration").
pub(crate) fn split_disintegrate(m: &mut Module, disint_node: NodeId) -> Option<(NodeId, NodeId)> {
    // 1. Recognize `disintegrate(selector, M)` with exactly two arguments.
    let c = expect_builtin_call(m, disint_node, "disintegrate")?;
    if c.args.len() != 2 {
        return None;
    }
    let selector = c.args[0];
    let measure = c.args[1];

    // 2. Parse the selector into the SELECTED field-name set, then run the shared
    //    structural split on `M`.
    let selected_names = selector_names(m, selector)?;
    split_law_record(m, &selected_names, measure)
}

/// The core structural split (spec §06 "Structural disintegration"), factored out
/// of [`split_disintegrate`] so the `restrict` desugaring
/// ([`rewrite_restrict`]) can reuse it directly with the field-names of the
/// observed record — without fabricating a synthetic `disintegrate(vector(…), M)`
/// node. `selected_names` are the already-parsed SELECTED field names; `measure`
/// is `M` (resolved one ref hop below to `lawof(record(field…))`).
///
/// Returns the same `(kernel, marginal)` pair as [`split_disintegrate`], and
/// `None` (caller refuses) for every shape outside the explicit DAG case — see
/// [`split_disintegrate`]'s doc for the exhaustive list.
fn split_law_record(
    m: &mut Module,
    selected_names: &[Box<str>],
    measure: NodeId,
) -> Option<(NodeId, NodeId)> {
    // Resolve `M` one ref hop to `lawof(record(field…))`.
    let (measure_resolved, _) = resolve_ref_one(m, measure);
    let lawof = expect_builtin_call(m, measure_resolved, "lawof")?;
    if lawof.args.len() != 1 {
        return None;
    }
    let record_arg = lawof.args[0];
    let (record_resolved, _) = resolve_ref_one(m, record_arg);
    let record = expect_builtin_call(m, record_resolved, "record")?;
    if !record.args.is_empty() {
        // A record with positional args is not a field-keyed product.
        return None;
    }
    // Gather the record's `%field name value` entries (verbatim value nodes),
    // in order; refuse any non-`%field` named entry.
    let mut fields: Vec<(Symbol, NodeId)> = Vec::with_capacity(record.named.len());
    for na in record.named.iter() {
        if na.kind != NamedKind::Field {
            return None;
        }
        fields.push((na.name, na.value));
    }
    if fields.is_empty() {
        return None;
    }

    // Every selected name must be a field of the record.
    let is_selected = |name: Symbol| selected_names.iter().any(|s| m.resolve(name) == &**s);
    let all_present = selected_names
        .iter()
        .all(|s| fields.iter().any(|(n, _)| m.resolve(*n) == &**s));
    if !all_present {
        return None;
    }

    // Partition the record fields into SELECTED and NON-selected, preserving
    //    record order within each group.
    let mut selected_fields: Vec<(Symbol, NodeId)> = Vec::new();
    let mut nonselected_fields: Vec<(Symbol, NodeId)> = Vec::new();
    for &(name, value) in &fields {
        if is_selected(name) {
            selected_fields.push((name, value));
        } else {
            nonselected_fields.push((name, value));
        }
    }
    // A vacuous split (nothing selected, or nothing left to condition on) is not
    // a structural disintegration.
    if selected_fields.is_empty() || nonselected_fields.is_empty() {
        return None;
    }

    // The kernel's boundary inputs are the NON-selected fields. Each entry is
    // `(name, ref)` where `ref` is how the value is referenced — a bare `%ref`
    // in the explicit-DAG shape (`(%field theta1 (%ref self theta1))`). A
    // non-ref value falls outside the case → refuse.
    let mut spec_inputs: Vec<(Symbol, Ref)> = Vec::with_capacity(nonselected_fields.len());
    for &(name, value) in &nonselected_fields {
        let Node::Ref(r) = m.node(value) else {
            return None;
        };
        spec_inputs.push((name, *r));
    }

    // Enforce the closed-marginal invariant (spec §06 "Structural disintegration":
    // the split is valid only when `jointchain(marginal, kernel) ≡ joint`). The
    // NON-selected fields form the marginal; for that equivalence the SELECTED
    // fields must be causally DOWNSTREAM — so no NON-selected field may
    // (transitively) DEPEND on a SELECTED variate.
    //
    // The comparison is between RESOLVED BINDING NAMES on both sides, NOT the
    // selector's surface field LABELS. A record may alias a variate under a field
    // whose label differs from its binding (`(%field mu_param (%ref self theta1))`);
    // intersecting the non-selected closure (a set of BINDING names) against the
    // selector's field-name strings — a DIFFERENT namespace — would then miss the
    // dependency and wrongly emit a vacuous kernel + non-closed marginal for a
    // causally REVERSED selector (a silent-wrong-density). So compare like with
    // like:
    //   - `selected_bindings`: the binding names each SELECTED field's value
    //     DIRECTLY references (one hop — a transform value's whole subtree is
    //     scanned, but a referenced binding's rhs is NOT descended into, since we
    //     want the selected variates themselves, not their upstream closure).
    //     `(%field mu_param (%ref self theta1))` ⇒ `theta1`.
    //   - `reachable`: the transitive generative closure of the NON-selected
    //     fields (following `(%ref self …)` into each binding's rhs AND across a
    //     reified callable's `%specinputs` cut).
    // Symbols come from the same module interner on both sides, so a bare-name
    // intersection is exact. If it is non-empty, a non-selected field
    // (transitively) depends on a selected variate → the marginal is not closed
    // and the selector is causally REVERSED → refuse (fail-closed,
    // refuse-don't-mislower). The reverse-direction disintegrate (§06 "two
    // formulations") is a separate follow-up, out of scope here.
    // NOTE: one-hop resolution here can miss a chain-alias selected field (a
    // bare-ref binding to another draw), so this guard alone isn't sufficient.
    // Fail-closed by composition: `resolve_component_draw` (density.rs) refuses
    // a selected field that is not a draw or a reference to a draw.
    let mut selected_bindings: HashSet<Symbol> = HashSet::new();
    for &(_, value) in &selected_fields {
        collect_selected_bindings(m, value, &mut selected_bindings);
    }
    let mut reachable: HashSet<Symbol> = HashSet::new();
    for &(_, value) in &nonselected_fields {
        collect_reachable_bindings(m, value, &mut reachable);
    }
    if reachable
        .iter()
        .any(|name| selected_bindings.contains(name))
    {
        return None;
    }

    // Build the kernel: `kernelof(record(<selected fields>), %specinputs(…))`.
    let record_sym = m.intern("record");
    let kernel_body = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(record_sym),
        args: Vec::<NodeId>::new().into(),
        named: selected_fields
            .iter()
            .map(|&(name, value)| NamedArg {
                kind: NamedKind::Field,
                name,
                value,
            })
            .collect::<Vec<_>>()
            .into(),
        inputs: None,
    }));
    let kernelof_sym = m.intern("kernelof");
    let kernel = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(kernelof_sym),
        args: vec![kernel_body].into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: Some(Inputs::Spec(spec_inputs.into())),
    }));

    // Build the marginal: `lawof(record(<non-selected fields>))`.
    let marginal_body = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(record_sym),
        args: Vec::<NodeId>::new().into(),
        named: nonselected_fields
            .iter()
            .map(|&(name, value)| NamedArg {
                kind: NamedKind::Field,
                name,
                value,
            })
            .collect::<Vec<_>>()
            .into(),
        inputs: None,
    }));
    let lawof_sym = m.intern("lawof");
    let marginal = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(lawof_sym),
        args: vec![marginal_body].into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: None,
    }));

    Some((kernel, marginal))
}

/// Collect (into `out`) the names of every top-level binding transitively
/// reachable from `start` by following `(%ref self …)` edges into each
/// referenced binding's rhs — the generative closure of `start` (spec §06,
/// the sub-DAG a field depends on). Used by [`split_law_record`] to test the
/// closed-marginal invariant: the non-selected marginal is closed iff none of
/// its fields' closures reach a selected variate's binding.
///
/// `out` doubles as the visited-set for bindings, so a binding's rhs is walked
/// at most once — the walk terminates even on a (malformed) cyclic self-ref
/// graph. Only `(%ref self …)` edges are followed (module/local refs cannot name
/// a top-level selected field of this record, so ignoring them is sound and
/// conservative).
///
/// The walk also follows a `Call`'s reification boundary inputs
/// (`Inputs::Spec` — the `%specinputs`/`kernelof`/`functionof` cut). Those
/// `(Symbol, Ref)` source refs are a real dependency edge but are NOT visited by
/// `Node::children()` (core `node.rs`, `for_each_child` — the `Inputs` bucket is
/// name/ref leaves, not child sub-nodes). Without following them, a non-selected
/// field whose closure passes through a reified callable whose cut references a
/// SELECTED variate would be invisible here → under-refuse. Each boundary
/// `Ref(SelfMod, …)` is treated exactly like a `(%ref self …)` edge (a `%local`
/// placeholder is not a binding ref, so it is skipped).
fn collect_reachable_bindings(m: &Module, start: NodeId, out: &mut HashSet<Symbol>) {
    let mut stack = vec![start];
    while let Some(id) = stack.pop() {
        let node = m.node(id);
        if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = node
        {
            let name = *name;
            // First visit of this binding: record it, then descend into its rhs.
            if out.insert(name) {
                if let Some(bid) = m.binding_by_name(name) {
                    stack.push(m.binding(bid).rhs);
                }
            }
            continue;
        }
        // Follow a reification's `%specinputs` boundary source refs (invisible to
        // `children()`), enqueuing each `(%ref self …)` cut like a body edge.
        if let Node::Call(c) = node {
            if let Some(Inputs::Spec(entries)) = &c.inputs {
                for (_, r) in entries.iter() {
                    if r.ns == RefNs::SelfMod && out.insert(r.name) {
                        if let Some(bid) = m.binding_by_name(r.name) {
                            stack.push(m.binding(bid).rhs);
                        }
                    }
                }
            }
        }
        for c in node.children() {
            stack.push(c);
        }
    }
}

/// Collect (into `out`) the binding names a SELECTED field's value DIRECTLY
/// references — the selected variates themselves. Unlike
/// [`collect_reachable_bindings`], this does NOT descend into a referenced
/// binding's rhs: it scans only the syntactic subtree of `start`, recording each
/// `(%ref self …)` name (and each `%specinputs` boundary `Ref(SelfMod, …)`) as a
/// leaf. Descending would fold the selected variate's own upstream closure into
/// the set and OVER-refuse the valid direction (e.g. selecting the downstream
/// `obs` whose rhs references the non-selected `theta1`/`theta2` would then
/// wrongly intersect them). `seen` bounds the syntactic walk against shared DAG
/// nodes.
fn collect_selected_bindings(m: &Module, start: NodeId, out: &mut HashSet<Symbol>) {
    let mut stack = vec![start];
    let mut seen: HashSet<NodeId> = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = m.node(id);
        if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = node
        {
            // A selected variate: record it, but do NOT descend into its rhs.
            out.insert(*name);
            continue;
        }
        if let Node::Call(c) = node {
            if let Some(Inputs::Spec(entries)) = &c.inputs {
                for (_, r) in entries.iter() {
                    if r.ns == RefNs::SelfMod {
                        out.insert(r.name);
                    }
                }
            }
        }
        for c in node.children() {
            stack.push(c);
        }
    }
}

/// Desugar `restrict(M, x)` into `bayesupdate(likelihoodof(kernel, x), marginal)`
/// where `(kernel, marginal) = disintegrate([field-names of x], M)` (spec §06
/// "Measure restriction": `restrict(M, x)` is the non-normalized conditional of
/// `M` given the observed values `x`). Returns the desugared node — the driver
/// substitutes it for the `restrict` binding's RHS, and the resulting
/// `bayesupdate` lowers via the existing posterior path.
///
/// `x` may be given either as an explicit `record(field…)` positional argument,
/// or as the spec's idiomatic keyword-splat — `restrict(M, a = …, b = …)`,
/// auto-splat-equivalent to `restrict(M, record(a = …, b = …))` (spec §06
/// "Measure restriction"). The parser leaves a splat's `field = value` pairs as
/// bare `%kwarg` entries directly on the `restrict` call (`named_kind_for` in
/// `crates/syntax/src/parser.rs` only tags `%field` for
/// record/table/joint/jointchain/cartprod, not `restrict`) rather than
/// synthesizing a `record(...)` node, so this normalizes them into one,
/// re-tagged `NamedKind::Field`, before the rest of the desugar runs — both
/// forms then share the same downstream path.
///
/// The selector is exactly the `%field` names of `x`: the observed record names
/// which variates of `M` are conditioned on (bi4 ⇒ `["obs"]`), so the split is
/// the SAME `(kernel, marginal)` the bi3 explicit `disintegrate(["obs"], M)`
/// produces, and the emitted `likelihoodof(kernel, x)` scores that kernel at the
/// observed `x`.
///
/// Returns `None` (caller refuses — refuse-don't-mislower) for any shape outside
/// the explicit-DAG case this handles:
/// - `restrict_node` is not a 2-arg `restrict` call with no stray named args,
///   AND not a 1-arg `restrict` call carrying at least one keyword-splat entry
///   (a 2-arg call that ALSO carries a named arg — e.g. a malformed `restrict(M,
///   x, bogus = …)` — refuses rather than silently dropping `bogus`);
/// - `x` (arg1, explicit form) is not a `record(field…)` of observed values
///   (positional args or a non-`%field` named entry refuse);
/// - the disintegration on `x`'s field names does not split
///   ([`split_law_record`] returns `None`) — `M` is not a `lawof(record(…))`, or
///   a field of `x` names no variate of `M`.
pub(crate) fn rewrite_restrict(m: &mut Module, restrict_node: NodeId) -> Option<NodeId> {
    /// The observed argument `x`, before normalization: either the explicit
    /// positional node, or the keyword-splat's `(name, value)` pairs (owned,
    /// copied out of the `restrict` call's `named` entries).
    enum XArg {
        Explicit(NodeId),
        Splat(Vec<(Symbol, NodeId)>),
    }

    // 1. Recognize `restrict(M, x)` (explicit form, 2 positional args, and NO
    //    stray named args — a `restrict(M, x, bogus = …)` carries a kwarg this
    //    desugaring does not understand, and silently taking just the 2
    //    positionals would mislower the malformed call) or `restrict(M, a = …, b
    //    = …)` (keyword-splat form: 1 positional arg + at least one `%kwarg`).
    let (measure, x_arg) = {
        let c = expect_builtin_call(m, restrict_node, "restrict")?;
        if c.args.len() == 2 && c.named.is_empty() {
            (c.args[0], XArg::Explicit(c.args[1]))
        } else if c.args.len() == 1 && !c.named.is_empty() {
            (
                c.args[0],
                XArg::Splat(c.named.iter().map(|na| (na.name, na.value)).collect()),
            )
        } else {
            return None;
        }
    };

    // 2. `x` must be a `record(field…)` — either the explicit positional node,
    //    or synthesized here from the keyword-splat's pairs, re-tagged
    //    `NamedKind::Field` (the SAME shape `record_field_names` expects from an
    //    explicit `record(...)` argument, so the rest of the desugar —
    //    selector = field-names, `likelihoodof(kernel, x)`, `bayesupdate` — is
    //    unchanged for both forms). The disintegration selector is `x`'s
    //    field-names (resolve one ref hop for the explicit form, in case `x` is
    //    bound by name).
    let x = match x_arg {
        XArg::Explicit(x) => x,
        XArg::Splat(fields) => {
            let record_sym = m.intern("record");
            m.alloc(Node::Call(Call {
                head: CallHead::Builtin(record_sym),
                args: Vec::<NodeId>::new().into(),
                named: fields
                    .into_iter()
                    .map(|(name, value)| NamedArg {
                        kind: NamedKind::Field,
                        name,
                        value,
                    })
                    .collect::<Vec<_>>()
                    .into(),
                inputs: None,
            }))
        }
    };
    let (x_resolved, _) = resolve_ref_one(m, x);
    let x_field_names = record_field_names(m, x_resolved)?;

    // 3. Split `M` on `x`'s field-names → the SAME (kernel, marginal) the
    //    equivalent `disintegrate([field-names of x], M)` yields. A field of `x`
    //    that is not a variate of `M`, or a non-`lawof(record)` `M`, refuses here.
    let (kernel, marginal) = split_law_record(m, &x_field_names, measure)?;

    // 4. Build `bayesupdate(likelihoodof(kernel, x), marginal)`. `likelihoodof`
    //    scores the kernel at the observed values `x` (the original arg1 node,
    //    verbatim); `bayesupdate` reweights the marginal by that likelihood.
    let likelihoodof_sym = m.intern("likelihoodof");
    let likelihood = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(likelihoodof_sym),
        args: vec![kernel, x].into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: None,
    }));
    let bayesupdate_sym = m.intern("bayesupdate");
    let posterior = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(bayesupdate_sym),
        args: vec![likelihood, marginal].into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: None,
    }));
    Some(posterior)
}

/// The `%field` names of a `record(field…)` node, in order (as `Box<str>`, to
/// feed [`split_law_record`]'s selector). `None` when `id` is not a `record`
/// builtin call, carries positional args, or has a non-`%field` named entry —
/// the caller refuses (the observed `x` must be a clean field-keyed record).
fn record_field_names(m: &Module, id: NodeId) -> Option<Vec<Box<str>>> {
    let rec = expect_builtin_call(m, id, "record")?;
    if !rec.args.is_empty() {
        return None;
    }
    let mut names = Vec::with_capacity(rec.named.len());
    for na in rec.named.iter() {
        if na.kind != NamedKind::Field {
            return None;
        }
        names.push(Box::from(m.resolve(na.name)));
    }
    Some(names)
}

/// The field names a `disintegrate` selector picks (spec §06: works like `get`
/// — `"obs"` selects field `obs`; `["obs", …]` selects each named field).
/// `Some` only when every entry is a literal string (a bare `Scalar::Str`, or a
/// `vector(...)` of `Scalar::Str`); an index selector, a non-literal, or an
/// empty `vector` ⇒ `None` (mirrors `flatppl_infer`'s `selector_field_names`).
fn selector_names(m: &Module, node: NodeId) -> Option<Vec<Box<str>>> {
    match m.node(node) {
        Node::Lit(Scalar::Str(s)) => Some(vec![s.clone()]),
        Node::Call(c) => {
            let CallHead::Builtin(op) = c.head else {
                return None;
            };
            if m.resolve(op) != "vector" {
                return None;
            }
            let names: Option<Vec<Box<str>>> = c
                .args
                .iter()
                .map(|&a| match m.node(a) {
                    Node::Lit(Scalar::Str(s)) => Some(s.clone()),
                    _ => None,
                })
                .collect();
            // An empty selector (`[]` → `(vector)`) has no defined meaning.
            names.filter(|v| !v.is_empty())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::split_disintegrate;
    use flatppl_core::{Call, CallHead, Inputs, Module, NamedKind, Node, NodeId, Ref, RefNs};

    fn parse_infer(src: &str) -> Module {
        let mut m = flatppl_syntax::parse(src).unwrap();
        let _ = flatppl_infer::infer(&mut m);
        m
    }

    /// Locate the `disintegrate(...)` call node in an inferred module (the rhs of
    /// the desugared `__0xN` binding).
    fn find_disintegrate(m: &Module) -> NodeId {
        for (_, b) in m.bindings() {
            if let Node::Call(c) = m.node(b.rhs) {
                if let CallHead::Builtin(op) = c.head {
                    if m.resolve(op) == "disintegrate" {
                        return b.rhs;
                    }
                }
            }
        }
        panic!("no disintegrate node in module");
    }

    /// The builtin-call node at `id`, asserting its head name.
    fn expect_call<'a>(m: &'a Module, id: NodeId, name: &str) -> &'a Call {
        let Node::Call(c) = m.node(id) else {
            panic!("node is not a call: {:?}", m.node(id));
        };
        let CallHead::Builtin(op) = c.head else {
            panic!("call is not a builtin");
        };
        assert_eq!(m.resolve(op), name, "unexpected head");
        c
    }

    /// The `%field` entries of a `record(...)` node as `(name, value)` pairs.
    fn record_fields(m: &Module, id: NodeId) -> Vec<(String, NodeId)> {
        let rec = expect_call(m, id, "record");
        assert!(rec.args.is_empty(), "record has positional args");
        rec.named
            .iter()
            .map(|na| {
                assert_eq!(na.kind, NamedKind::Field, "non-field entry in record");
                (m.resolve(na.name).to_string(), na.value)
            })
            .collect()
    }

    /// A `(%ref self <name>)` value node → its bound name.
    fn self_ref_name(m: &Module, id: NodeId) -> String {
        match m.node(id) {
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) => m.resolve(*name).to_string(),
            other => panic!("expected `(%ref self …)`, got {other:?}"),
        }
    }

    /// A minimal bi3-shape joint: two prior variates and an iid observation,
    /// bundled into a joint law and split with `disintegrate(["obs"], …)`.
    const BI3: &str = "\
theta1 ~ Normal(mu = 0, sigma = 1)
theta2 ~ Gamma(alpha = 2, beta = 1)
obs ~ iid(Normal(mu = theta1, sigma = theta2), 10)
joint_model = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))
forward_kernel, prior = disintegrate([\"obs\"], joint_model)";

    #[test]
    fn splits_bi3_into_kernel_and_marginal() {
        let mut m = parse_infer(BI3);
        let disint = find_disintegrate(&m);
        let (kernel, marginal) =
            split_disintegrate(&mut m, disint).expect("bi3 explicit-DAG disintegration must split");

        // Kernel = kernelof(record(obs = <obs rhs>), %specinputs(theta1, theta2)):
        // the SELECTED field `obs` is the body, the NON-selected fields are the
        // boundary inputs (verbatim value refs).
        let k = expect_call(&m, kernel, "kernelof");
        assert_eq!(k.args.len(), 1, "kernelof takes one positional body");
        let body_fields = record_fields(&m, k.args[0]);
        assert_eq!(
            body_fields
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["obs"],
            "kernel body must be exactly the selected `obs` field"
        );
        assert_eq!(
            self_ref_name(&m, body_fields[0].1),
            "obs",
            "kernel body `obs` field must carry the verbatim `(%ref self obs)`"
        );
        // Boundary inputs are exactly the NON-selected fields, as `(name, ref)`.
        let Some(Inputs::Spec(entries)) = &k.inputs else {
            panic!("kernel must carry %specinputs");
        };
        let got: Vec<(String, RefNs, String)> = entries
            .iter()
            .map(|(nm, r)| {
                (
                    m.resolve(*nm).to_string(),
                    r.ns,
                    m.resolve(r.name).to_string(),
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                ("theta1".into(), RefNs::SelfMod, "theta1".into()),
                ("theta2".into(), RefNs::SelfMod, "theta2".into()),
            ],
            "boundary inputs must be exactly the non-selected fields bound to their value refs"
        );

        // Marginal = lawof(record(theta1 = …, theta2 = …)) over non-selected only.
        let law = expect_call(&m, marginal, "lawof");
        assert_eq!(law.args.len(), 1, "lawof takes one positional body");
        let marg_fields = record_fields(&m, law.args[0]);
        assert_eq!(
            marg_fields
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["theta1", "theta2"],
            "marginal must carry exactly the non-selected fields, in order"
        );
        assert_eq!(self_ref_name(&m, marg_fields[0].1), "theta1");
        assert_eq!(self_ref_name(&m, marg_fields[1].1), "theta2");
    }

    #[test]
    fn refuses_selector_naming_a_non_field() {
        // `["bogus"]` names no field of the record → refuse (None).
        let src = "\
theta1 ~ Normal(mu = 0, sigma = 1)
obs ~ Normal(mu = theta1, sigma = 1)
joint_model = lawof(record(theta1 = theta1, obs = obs))
fk, pr = disintegrate([\"bogus\"], joint_model)";
        let mut m = parse_infer(src);
        let disint = find_disintegrate(&m);
        assert!(
            split_disintegrate(&mut m, disint).is_none(),
            "a selector naming no field must refuse"
        );
    }

    #[test]
    fn refuses_selecting_all_fields() {
        // Selecting every field leaves an empty non-selected (conditioning) set
        // → a vacuous split → refuse.
        let src = "\
theta1 ~ Normal(mu = 0, sigma = 1)
obs ~ Normal(mu = theta1, sigma = 1)
joint_model = lawof(record(theta1 = theta1, obs = obs))
fk, pr = disintegrate([\"theta1\", \"obs\"], joint_model)";
        let mut m = parse_infer(src);
        let disint = find_disintegrate(&m);
        assert!(
            split_disintegrate(&mut m, disint).is_none(),
            "selecting all fields (empty conditioning set) must refuse"
        );
    }

    #[test]
    fn refuses_reversed_selector_under_field_aliasing() {
        // The record ALIASES the upstream roots under labels (`mu_param`,
        // `sigma_param`) that differ from their bindings (`theta1`, `theta2`),
        // and the selector names those LABELS. The closed-marginal guard must
        // compare RESOLVED BINDING NAMES: selected {mu_param→theta1,
        // sigma_param→theta2} ⇒ {theta1, theta2}; non-selected `obs` closure
        // {obs, theta1, theta2}; intersection {theta1, theta2} ≠ ∅ → REFUSE. A
        // guard comparing surface labels would see an empty intersection and
        // wrongly split (a reopened silent-wrong-density).
        let src = "\
theta1 ~ Normal(mu = 0, sigma = 1)
theta2 ~ Exponential(rate = 1)
obs ~ iid(Normal(mu = theta1, sigma = theta2), 10)
joint_model = lawof(record(mu_param = theta1, sigma_param = theta2, obs = obs))
fk, pr = disintegrate([\"mu_param\", \"sigma_param\"], joint_model)";
        let mut m = parse_infer(src);
        let disint = find_disintegrate(&m);
        assert!(
            split_disintegrate(&mut m, disint).is_none(),
            "a reversed selector under field aliasing (non-closed marginal) must refuse"
        );
    }

    #[test]
    fn splits_valid_aliased_marginal_fields() {
        // The VALID direction with the MARGINAL fields aliased too
        // (`mu_param = theta1`, `sigma_param = theta2`, `data = obs`), selecting
        // the downstream `obs` under the label `data`. The fix must NOT
        // over-refuse: selected {data→obs} ⇒ {obs}; non-selected
        // {mu_param→theta1, sigma_param→theta2} closure {theta1, theta2};
        // intersection ∅ → SPLIT. (This shape splits correctly but does not lower
        // end-to-end — its posterior θ point over the aliased marginal cannot bind
        // in the likelihood path, which still requires θ field names to name
        // module bindings; that is a separate, out-of-scope likelihood-path limit,
        // so this asserts at the split level.)
        let src = "\
theta1 ~ Normal(mu = 0, sigma = 1)
theta2 ~ Exponential(rate = 1)
obs ~ iid(Normal(mu = theta1, sigma = theta2), 10)
joint_model = lawof(record(mu_param = theta1, sigma_param = theta2, data = obs))
fk, pr = disintegrate([\"data\"], joint_model)";
        let mut m = parse_infer(src);
        let disint = find_disintegrate(&m);
        let (kernel, marginal) = split_disintegrate(&mut m, disint)
            .expect("the valid direction with aliased marginal fields must split, not over-refuse");

        // Kernel body = the selected `data` field (aliasing `obs`); its boundary
        // inputs are the non-selected aliased fields, bound to their value refs.
        let k = expect_call(&m, kernel, "kernelof");
        let body_fields = record_fields(&m, k.args[0]);
        assert_eq!(
            body_fields
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["data"],
            "kernel body must be exactly the selected aliased field `data`"
        );
        assert_eq!(
            self_ref_name(&m, body_fields[0].1),
            "obs",
            "the selected field `data` must carry the verbatim `(%ref self obs)`"
        );
        let Some(Inputs::Spec(entries)) = &k.inputs else {
            panic!("kernel must carry %specinputs");
        };
        let got: Vec<(String, String)> = entries
            .iter()
            .map(|(nm, r)| (m.resolve(*nm).to_string(), m.resolve(r.name).to_string()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("mu_param".into(), "theta1".into()),
                ("sigma_param".into(), "theta2".into()),
            ],
            "boundary inputs keep the aliased field labels, sourced from their binding refs"
        );

        // Marginal = lawof(record(mu_param = …, sigma_param = …)) over the
        // non-selected aliased fields.
        let law = expect_call(&m, marginal, "lawof");
        let marg_fields = record_fields(&m, law.args[0]);
        assert_eq!(
            marg_fields
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["mu_param", "sigma_param"],
            "marginal must carry exactly the non-selected aliased fields, in order"
        );
        assert_eq!(self_ref_name(&m, marg_fields[0].1), "theta1");
        assert_eq!(self_ref_name(&m, marg_fields[1].1), "theta2");
    }
}
