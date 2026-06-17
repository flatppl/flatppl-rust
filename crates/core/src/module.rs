//! The [`Module`] — owns the IR arena, the bindings, the interner, and the
//! (optional) annotation side-tables; exposes the read / build / annotate API.

use crate::id::{Arena, BindingId, Interner, NodeId, SecondaryMap, Symbol};
use crate::node::{Node, Ref};
use crate::ty::{Phase, Type, ValueSet};
use std::collections::HashMap;

/// A single FlatPPL module: a flat, order-irrelevant set of named bindings over a
/// shared node arena, plus the interner and analysis side-tables.
///
/// **v1 stage model** (see `ARCHITECTURE.md` "Open"): a single `Module` with
/// *optional* annotation tables (filled by passes) + debug asserts, rather than a
/// type-state `Module<Stage>`. Promote to type-state only if drift forces it.
#[derive(Clone, Debug, Default)]
pub struct Module {
    interner: Interner,
    nodes: Arena<NodeId, Node>,
    bindings: Arena<BindingId, Binding>,
    /// Source / insertion order of bindings. FlatPPL is order-irrelevant; this is
    /// kept only for faithful round-tripping.
    order: Vec<BindingId>,
    by_name: HashMap<Symbol, BindingId>,

    // ---- annotation side-tables (empty until the producing pass fills them) ----
    /// Inferred type per node (`flatppl-infer`) — the FlatPIR `%meta` type slot.
    /// `%meta` is a transparent wrapper around any expression (spec §11), so
    /// annotations are keyed per node, not per call.
    types: SecondaryMap<NodeId, Type>,
    /// Inferred phase per node (`flatppl-infer`). Per-node, not per-binding: a
    /// binding's phase is just its RHS node's phase (see [`Module::binding_phase`]).
    phases: SecondaryMap<NodeId, Phase>,
    /// Inferred value set per node (`flatppl-infer`): a sound set containing the
    /// node's value (a measure node's support). The third `%meta` slot.
    valuesets: SecondaryMap<NodeId, ValueSet>,
    /// Source span per node (`flatppl-syntax`), for diagnostics / DAG back-refs.
    spans: SecondaryMap<NodeId, Span>,
    /// Filled `%autoinputs` lists per boundary-less reification node
    /// (`flatppl-infer`, phase inference). Inference metadata: projected into
    /// the `%autoinputs` slot on FlatPIR write, dropped on FlatPPL write.
    auto_inputs: SecondaryMap<NodeId, Box<[(Symbol, Ref)]>>,
}

/// A top-level binding `name = rhs`.
#[derive(Clone, Debug, PartialEq)]
pub struct Binding {
    pub name: Symbol,
    pub rhs: NodeId,
    pub doc: Option<Doc>,
    /// Public interface (the name does not start with `_`).
    pub public: bool,
    /// Engine-generated (a lifted anon, a `%mlhs` multi-LHS intermediate, …).
    pub synthetic: bool,
}

/// A doc-comment attached to a binding (spec §11 `(%doc <markup> <line>…)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Doc {
    pub markup: Markup,
    pub lines: Box<[Box<str>]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Markup {
    Md,
    Typ,
}

/// A half-open source byte range (0-based), for diagnostics and DAG back-refs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Module {
    pub fn new() -> Self {
        Self::default()
    }

    // ---- build ----

    /// Intern a name into a [`Symbol`].
    pub fn intern(&mut self, name: &str) -> Symbol {
        self.interner.intern(name)
    }

    /// Allocate an expression node, returning its handle.
    pub fn alloc(&mut self, node: Node) -> NodeId {
        self.nodes.alloc(node)
    }

    /// Add a top-level binding (appends to source order; indexes it by name).
    pub fn add_binding(&mut self, binding: Binding) -> BindingId {
        let name = binding.name;
        let id = self.bindings.alloc(binding);
        self.order.push(id);
        self.by_name.insert(name, id);
        id
    }

    // ---- read ----

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }
    pub fn binding(&self, id: BindingId) -> &Binding {
        &self.bindings[id]
    }

    /// Number of allocated nodes — an upper bound on `NodeId` indices, for
    /// sizing dense per-node side structures (e.g. a visited bitmap).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of top-level bindings.
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }

    /// Resolve a symbol to its name.
    pub fn resolve(&self, sym: Symbol) -> &str {
        self.interner.resolve(sym)
    }

    /// Look up a top-level binding by name.
    pub fn binding_by_name(&self, name: Symbol) -> Option<BindingId> {
        self.by_name.get(&name).copied()
    }

    /// Bindings in source order.
    pub fn bindings(&self) -> impl Iterator<Item = (BindingId, &Binding)> {
        self.order.iter().map(move |&id| (id, &self.bindings[id]))
    }

    /// Public bindings (the module interface), in source order.
    pub fn public_bindings(&self) -> impl Iterator<Item = (BindingId, &Binding)> {
        self.bindings().filter(|(_, b)| b.public)
    }

    /// Visit the child sub-nodes of a node (delegates to [`Node::for_each_child`]).
    pub fn for_each_child(&self, id: NodeId, f: impl FnMut(NodeId)) {
        self.node(id).for_each_child(f);
    }

    // ---- annotate (side-tables) ----

    pub fn type_of(&self, id: NodeId) -> Option<&Type> {
        self.types.get(id)
    }
    pub fn set_type(&mut self, id: NodeId, ty: Type) {
        self.types.insert(id, ty);
    }

    pub fn phase_of(&self, id: NodeId) -> Option<Phase> {
        self.phases.get(id).copied()
    }
    pub fn set_phase(&mut self, id: NodeId, phase: Phase) {
        self.phases.insert(id, phase);
    }

    pub fn valueset_of(&self, id: NodeId) -> Option<&ValueSet> {
        self.valuesets.get(id)
    }
    pub fn set_valueset(&mut self, id: NodeId, set: ValueSet) {
        self.valuesets.insert(id, set);
    }

    /// The phase of a binding — i.e. the phase of its right-hand-side node.
    pub fn binding_phase(&self, id: BindingId) -> Option<Phase> {
        self.phase_of(self.binding(id).rhs)
    }

    pub fn span_of(&self, id: NodeId) -> Option<Span> {
        self.spans.get(id).copied()
    }
    pub fn set_span(&mut self, id: NodeId, span: Span) {
        self.spans.insert(id, span);
    }

    /// The inferred input list of a boundary-less reification
    /// ([`Inputs::Auto`](crate::Inputs::Auto)), if phase inference has filled it.
    pub fn auto_inputs_of(&self, id: NodeId) -> Option<&[(Symbol, Ref)]> {
        self.auto_inputs.get(id).map(|entries| entries.as_ref())
    }
    pub fn set_auto_inputs(&mut self, id: NodeId, entries: Box<[(Symbol, Ref)]>) {
        self.auto_inputs.insert(id, entries);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Call, CallHead, Node, Scalar};
    use crate::ty::{ScalarType, Type};

    #[test]
    fn build_access_traverse_annotate() {
        let mut m = Module::new();

        // Build `x = add(1, 2)`.
        let one = m.alloc(Node::Lit(Scalar::Int(1)));
        let two = m.alloc(Node::Lit(Scalar::Int(2)));
        let add = m.intern("add");
        let call = m.alloc(Node::Call(Call {
            head: CallHead::Builtin(add),
            args: vec![one, two].into(),
            named: Vec::new().into(),
            inputs: None,
        }));
        let xname = m.intern("x");
        let x = m.add_binding(Binding {
            name: xname,
            rhs: call,
            doc: None,
            public: true,
            synthetic: false,
        });

        // Access.
        assert_eq!(m.binding_by_name(xname), Some(x));
        assert_eq!(m.resolve(xname), "x");
        let rhs = m.binding(x).rhs;

        // Traverse the call's children.
        let mut kids = Vec::new();
        m.for_each_child(rhs, |c| kids.push(c));
        assert_eq!(kids, vec![one, two]);

        // Annotate (side-table).
        m.set_type(rhs, Type::Scalar(ScalarType::Integer));
        assert_eq!(m.type_of(rhs), Some(&Type::Scalar(ScalarType::Integer)));
        assert_eq!(m.type_of(one), None); // unannotated
    }
}
