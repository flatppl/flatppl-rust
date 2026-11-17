//! Build and rewrite the binding graph of a module.

use std::collections::HashSet;

use crate::{BindingId, Call, CallHead, Module, NamedArg, NamedKind, Node, NodeId, Symbol};

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
