//! Build and rewrite the binding graph of a module.

use std::collections::HashSet;

use crate::ty::{VariateKind, variate_kind};
use crate::{
    BindingId, Call, CallHead, Module, NamedArg, NamedKind, Node, NodeId, Ref, RefNs, Symbol,
};

/// Follow `(%ref self x)` indirection through every bare-name alias: return the
/// first right-hand side that is not itself a self-ref, with the binding that
/// holds it, or `(id, None)` when `id` is not a self-ref.
///
/// §04 *Name resolution* makes `M = a` name the very expression `a` names, so a
/// caller that stopped one hop off `M` would hold the ref node `a` and refuse a
/// legal alias. Bounded by the binding count, so a cyclic module cannot hang.
pub fn resolve_ref(m: &Module, id: NodeId) -> (NodeId, Option<BindingId>) {
    let (mut node, mut bid) = resolve_ref_one(m, id);
    for _ in 0..m.binding_count() {
        match resolve_ref_one(m, node) {
            (next, Some(next_bid)) => (node, bid) = (next, Some(next_bid)),
            (_, None) => break,
        }
    }
    (node, bid)
}

/// Follow aliases to the model declaration, even after a query pinned a draw.
/// The last binding in the chain owns the declaration.
pub fn declared_binding(m: &Module, mut node: NodeId) -> (NodeId, Option<BindingId>) {
    let mut holder = None;
    for _ in 0..m.binding_count() {
        let Node::Ref(reference) = m.node(node) else {
            break;
        };
        if reference.ns != RefNs::SelfMod {
            break;
        }
        let Some(binding) = m.binding_by_name(reference.name) else {
            break;
        };
        node = m.declared_binding_rhs(binding);
        holder = Some(binding);
    }
    (node, holder)
}

/// Follow one level of `(%ref self x)` indirection: if `id` is a self-ref,
/// return `(binding.rhs, Some(bid))`; otherwise return `(id, None)`. Only a
/// caller that inspects each hop wants this; the rest use [`resolve_ref`].
pub fn resolve_ref_one(m: &Module, id: NodeId) -> (NodeId, Option<BindingId>) {
    if let Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }) = m.node(id)
        && let Some(bid) = m.binding_by_name(*name)
    {
        return (m.binding(bid).rhs, Some(bid));
    }
    (id, None)
}

/// Is `node` exactly the boundary-input reference `(%ref self|%local input)` —
/// the latent feeding a parameter directly (not a derived expression)?
pub fn is_input_ref(m: &Module, node: NodeId, input: Symbol) -> bool {
    matches!(
        m.node(node),
        Node::Ref(Ref { ns, name })
            if matches!(ns, RefNs::SelfMod | RefNs::Local) && *name == input
    )
}

/// The structural kind of a value known to be non-scalar, following its aliases.
pub fn confirmed_non_scalar(m: &Module, value: NodeId) -> Option<VariateKind> {
    let (resolved, _) = resolve_ref(m, value);
    match variate_kind(m.type_of(resolved)?)? {
        VariateKind::Scalar => None,
        kind => Some(kind),
    }
}

/// If `id` is a builtin call with head named `name`, return its [`Call`].
pub fn expect_builtin_call<'a>(m: &'a Module, id: NodeId, name: &str) -> Option<&'a Call> {
    let Node::Call(c) = m.node(id) else {
        return None;
    };
    let CallHead::Builtin(sym) = c.head else {
        return None;
    };
    if m.resolve(sym) == name {
        Some(c)
    } else {
        None
    }
}

/// If `id` is `draw(mᵢ)`, return `mᵢ`; otherwise `None`.
pub fn draw_argument(m: &Module, id: NodeId) -> Option<NodeId> {
    let c = expect_builtin_call(m, id, "draw")?;
    if c.args.len() != 1 {
        return None;
    }
    Some(c.args[0])
}

/// Allocate a positional builtin call `head(args…)`.
pub fn build_call(m: &mut Module, head: &str, args: &[NodeId]) -> NodeId {
    let sym = m.intern(head);
    m.alloc(Node::Call(Call {
        head: CallHead::Builtin(sym),
        args: args.to_vec().into(),
        named: Vec::<NamedArg>::new().into(),
        inputs: None,
    }))
}

/// Allocate a `record(%field name value ...)` call from `(name, value)` pairs.
pub fn build_record(m: &mut Module, fields: &[(Symbol, NodeId)]) -> NodeId {
    let named: Vec<NamedArg> = fields
        .iter()
        .map(|&(name, value)| NamedArg {
            kind: NamedKind::Field,
            name,
            value,
        })
        .collect();
    let record = m.intern("record");
    m.alloc(Node::Call(Call {
        head: CallHead::Builtin(record),
        args: Vec::<NodeId>::new().into(),
        named: named.into(),
        inputs: None,
    }))
}

/// Rebuild a call using the child order of [`Node::for_each_child`]. The new call
/// keeps its parameter names, callable boundary and sample-result provenance.
///
/// # Panics
///
/// The caller must supply a call node and one replacement per child.
pub fn rebuild_with_children(m: &mut Module, id: NodeId, new_children: &[NodeId]) -> NodeId {
    let Node::Call(orig_call) = m.node(id) else {
        panic!("rebuild_with_children requires a call");
    };
    let head = orig_call.head;
    let inputs = orig_call.inputs.clone();
    let pos_count = orig_call.args.len();
    let named_count = orig_call.named.len();

    // `for_each_child` visits: callee (User head) first, then positional args,
    // then named values — mirror that partition here.
    let (new_head, child_slice) = match head {
        CallHead::User(_) => (CallHead::User(new_children[0]), &new_children[1..]),
        CallHead::Builtin(s) => (CallHead::Builtin(s), new_children),
    };

    assert_eq!(child_slice.len(), pos_count + named_count);
    let new_args: Vec<NodeId> = child_slice[..pos_count].to_vec();
    let new_named_values: Vec<NodeId> = child_slice[pos_count..pos_count + named_count].to_vec();

    // Rebuild named args with updated values but original names/kinds.
    let new_named: Vec<NamedArg> = orig_call
        .named
        .iter()
        .zip(new_named_values.iter())
        .map(|(n, &v)| NamedArg {
            kind: n.kind,
            name: n.name,
            value: v,
        })
        .collect();

    let auto_inputs = m.auto_inputs_of(id).map(<[_]>::to_vec);
    let rebuilt = m.alloc(Node::Call(Call {
        head: new_head,
        args: new_args.into(),
        named: new_named.into(),
        inputs,
    }));
    if let Some(entries) = auto_inputs {
        m.set_auto_inputs(rebuilt, entries.into());
    }
    if m.is_sample_result(id) {
        m.mark_sample_result(rebuilt);
    }
    rebuilt
}

/// Keep the bindings reachable from the named roots through self references
/// and reification inputs. Absent roots are ignored.
pub fn retain_reachable(m: &mut Module, roots: &[Symbol]) {
    // Seed the worklist with the root bindings that exist.
    let mut keep: HashSet<BindingId> = HashSet::new();
    let mut work: Vec<BindingId> = Vec::new();
    for &r in roots {
        if let Some(bid) = m.binding_by_name(r)
            && keep.insert(bid)
        {
            work.push(bid);
        }
    }
    // Transitive closure: for each kept binding, mark every binding it references.
    while let Some(bid) = work.pop() {
        let rhs = m.binding(bid).rhs;
        let mut names: HashSet<Symbol> = HashSet::new();
        m.collect_referenced_names(rhs, &mut names);
        for name in names {
            if let Some(dep) = m.binding_by_name(name)
                && keep.insert(dep)
            {
                work.push(dep);
            }
        }
    }
    m.retain_bindings(&keep);
}

/// Add the terms in a balanced tree. Each pair contains adjacent terms in
/// their original order. This keeps the depth logarithmic.
///
/// # Panics
///
/// Panics if `terms` is empty.
pub fn fold_add(m: &mut Module, terms: &[NodeId]) -> NodeId {
    debug_assert!(!terms.is_empty(), "fold_add requires at least one term");
    // Pair front-to-back into `level`, then repeat over that. An odd tail term
    // carries forward unpaired, which keeps the sequence order and leaves
    // N <= 2 byte-identical to the old left fold.
    let mut level: Vec<NodeId> = terms.to_vec();
    let mut next: Vec<NodeId> = Vec::new();
    while level.len() > 1 {
        next.clear();
        next.reserve(level.len().div_ceil(2));
        let (pairs, remainder) = level.as_chunks::<2>();
        for pair in pairs {
            next.push(build_call(m, "add", &[pair[0], pair[1]]));
        }
        next.extend_from_slice(remainder);
        std::mem::swap(&mut level, &mut next);
    }
    level[0]
}
