//! Keep repeated expressions as graph edges when writing a tree-shaped format.

use std::borrow::Cow;

use crate::{Binding, Call, CallHead, Idx, Module, Node, NodeId, Ref, RefNs};

// Keep short expressions inline. Count their cost after any inner cuts, so
// nested diamonds introduce a name only when they would grow large again.
const INLINE_NODES: usize = 16;
// A call can nest a named argument and a FlatPIR metadata wrapper. Leave
// space for the binding, module and type annotation inside the reader limit.
const INLINE_DEPTH: usize = crate::DEFAULT_MAX_DEPTH / 4;

#[derive(Clone, Copy, Default)]
struct Usage {
    incoming: u8,
    has_parent: bool,
    visited: bool,
    keep_inline: bool,
}

impl Module {
    /// Internal graph view used by the FlatPPL and FlatPIR module writers.
    #[doc(hidden)]
    pub fn for_serialization(&self) -> Cow<'_, Self> {
        let (order, mut uses) = self.serialization_order();
        if !order
            .iter()
            .any(|&id| uses[id.index()].incoming > 1 && matches!(self.node(id), Node::Call(_)))
        {
            return Cow::Borrowed(self);
        }
        self.protect_serialization_scopes(&order, &mut uses);
        let selected = self.serialization_bindings(&order, &uses);
        if selected.is_empty() {
            return Cow::Borrowed(self);
        }
        let mut out = self.clone();
        let mut refs = vec![None; self.node_count()];
        let mut suffix = 0;
        for id in selected {
            let name = loop {
                let name = format!("__shared_{suffix}");
                suffix += 1;
                if !out.interner.contains_name(&name) {
                    break out.intern(&name);
                }
            };
            out.add_binding(Binding {
                name,
                rhs: id,
                doc: None,
                public: false,
                synthetic: true,
            });
            refs[id.index()] = Some(out.alloc(Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            })));
        }
        // Redirect only child edges in the copy. The original IDs still own
        // all inferred metadata, including each named RHS's FlatPIR annotation.
        for id in order {
            if let Node::Call(call) = out.nodes.get_mut(id) {
                if let CallHead::User(callee) = &mut call.head {
                    *callee = refs[callee.index()].unwrap_or(*callee);
                }
                for child in &mut call.args {
                    *child = refs[child.index()].unwrap_or(*child);
                }
                for arg in &mut call.named {
                    arg.value = refs[arg.value.index()].unwrap_or(arg.value);
                }
            }
        }
        Cow::Owned(out)
    }

    fn serialization_order(&self) -> (Vec<NodeId>, Vec<Usage>) {
        let mut uses = vec![Usage::default(); self.node_count()];
        let mut order = Vec::new();
        let mut pending = Vec::new();
        for (_, binding) in self.bindings() {
            uses[binding.rhs.index()].incoming = 1;
            pending.push((binding.rhs, false));
        }
        while let Some((id, finish)) = pending.pop() {
            if finish {
                order.push(id);
                continue;
            }
            if !std::mem::replace(&mut uses[id.index()].visited, true) {
                pending.push((id, true));
                self.for_each_child(id, |child| {
                    let usage = &mut uses[child.index()];
                    usage.incoming = usage.incoming.saturating_add(1);
                    usage.has_parent = true;
                    pending.push((child, false));
                });
            }
        }

        (order, uses)
    }

    fn protect_serialization_scopes(&self, order: &[NodeId], uses: &mut [Usage]) {
        // A callable can cut through several named bindings. Its whole body
        // belongs to its evaluation scope, as do axis and quadrature bodies.
        let mut scoped = Vec::new();
        for &id in order {
            if let Node::Call(c) = self.node(id)
                && (c.inputs.is_some()
                    || self.is_guarded_likelihood(id)
                    || self.is_sample_result(id)
                    || matches!(c.head, CallHead::Builtin(op) if matches!(self.resolve(op),
                        "aggregate" | "metricsum" | crate::LOG_INTEGRAL | crate::MASS_INTEGRAL)))
            {
                scoped.push(id);
            }
        }
        while let Some(id) = scoped.pop() {
            if std::mem::replace(&mut uses[id.index()].keep_inline, true) {
                continue;
            }
            if let Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) = self.node(id)
                && let Some(binding) = self.binding_by_name(*name)
            {
                scoped.push(self.binding(binding).rhs);
            }
            self.for_each_child(id, |child| scoped.push(child));
        }
        // Signature entries are references to the caller's bindings. Keep even
        // a malformed raw entry intact so validation reports the same error.
        for (_, b) in self.bindings() {
            if matches!(self.resolve(b.name), "inputs" | "outputs") {
                uses[b.rhs.index()].keep_inline = true;
                self.for_each_child(b.rhs, |child| uses[child.index()].keep_inline = true);
            }
        }
    }

    fn serialization_bindings(&self, order: &[NodeId], uses: &[Usage]) -> Vec<NodeId> {
        let mut sizes = vec![0usize; self.node_count()];
        let mut depths = vec![0usize; self.node_count()];
        let mut selected = Vec::new();
        for &id in order {
            let op = match self.node(id) {
                Node::Lit(_)
                | Node::Const(_)
                | Node::Ref(Ref {
                    ns: RefNs::SelfMod | RefNs::Module(_),
                    ..
                }) => {
                    sizes[id.index()] = 1;
                    continue;
                }
                Node::Call(Call {
                    head: CallHead::Builtin(op),
                    inputs: None,
                    ..
                }) => self.resolve(*op),
                _ => continue,
            };
            if !value_operation(op) {
                continue;
            }
            let mut size = 1usize;
            let mut depth = 1usize;
            self.for_each_child(id, |child| {
                let child = child.index();
                size = if sizes[child] == 0 || size == 0 {
                    0
                } else {
                    size.saturating_add(sizes[child])
                };
                depth = depth.max(depths[child] + 1);
            });
            let usage = uses[id.index()];
            // Set descriptors are read structurally, including the bounds of
            // an interval used in a membership test.
            if size > 0
                && usage.has_parent
                && !usage.keep_inline
                && op != "interval"
                && ((size > INLINE_NODES && usage.incoming > 1) || depth >= INLINE_DEPTH)
            {
                selected.push(id);
                sizes[id.index()] = 1;
            } else {
                sizes[id.index()] = size;
                depths[id.index()] = depth;
            }
        }
        selected
    }
}

// These calls denote ordinary values. Keep declarations, draws, reification
// boundaries, axis expressions and compiler-only operations in their scope.
fn value_operation(op: &str) -> bool {
    matches!(
        op,
        "add"
            | "sub"
            | "mul"
            | "divide"
            | "pow"
            | "neg"
            | "abs"
            | "abs2"
            | "sqrt"
            | "exp"
            | "log"
            | "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "atan2"
            | "sinh"
            | "cosh"
            | "tanh"
            | "min"
            | "max"
            | "real"
            | "imag"
            | "complex"
            | "conj"
            | "cis"
            | "equal"
            | "unequal"
            | "lt"
            | "le"
            | "gt"
            | "ge"
            | "land"
            | "lor"
            | "lnot"
            | "ifelse"
            | "in"
            | "interval"
            | "polynomial"
            | "vector"
            | "tuple"
            | "record"
            | "get"
            | "get0"
            | "get_field"
            | "tuple_get"
            | "sum"
            | "prod"
    )
}
