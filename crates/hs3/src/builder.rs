//! Ergonomic helpers for building flatppl_core IR.
use flatppl_core::id::{NodeId, Symbol};
use flatppl_core::node::{Call, CallHead, NamedArg, NamedKind, Node, Ref, RefNs, Scalar};
use flatppl_core::{Binding, Doc, Markup, Module};

pub struct Builder<'m> {
    pub m: &'m mut Module,
}

impl<'m> Builder<'m> {
    pub fn new(m: &'m mut Module) -> Self {
        Builder { m }
    }

    pub fn sym(&mut self, s: &str) -> Symbol {
        self.m.intern(s)
    }

    pub fn lit_real(&mut self, v: f64) -> NodeId {
        self.m.alloc(Node::Lit(Scalar::Real(v)))
    }

    pub fn lit_int(&mut self, v: i64) -> NodeId {
        self.m.alloc(Node::Lit(Scalar::Int(v)))
    }

    pub fn str_lit(&mut self, s: &str) -> NodeId {
        self.m.alloc(Node::Lit(Scalar::Str(s.into())))
    }

    pub fn self_ref(&mut self, name: &str) -> NodeId {
        let name = self.sym(name);
        self.m.alloc(Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }))
    }

    pub fn call(&mut self, head: &str, args: &[NodeId]) -> NodeId {
        let head = self.sym(head);
        self.m.alloc(Node::Call(Call {
            head: CallHead::Builtin(head),
            args: args.to_vec().into(),
            named: Vec::new().into(),
            inputs: None,
        }))
    }

    pub fn call_kw(&mut self, head: &str, kw: &[(&str, NodeId)]) -> NodeId {
        let head = self.sym(head);
        let named = kw
            .iter()
            .map(|(k, v)| NamedArg {
                kind: NamedKind::Kwarg,
                name: self.m.intern(k),
                value: *v,
            })
            .collect::<Vec<_>>();
        self.m.alloc(Node::Call(Call {
            head: CallHead::Builtin(head),
            args: Vec::new().into(),
            named: named.into(),
            inputs: None,
        }))
    }

    /// Array literal `[a,b,...]`. Uses `vector` (the canonical FlatPPL builtin).
    pub fn array(&mut self, elems: &[NodeId]) -> NodeId {
        self.call("vector", elems)
    }

    pub fn bind(&mut self, name: &str, rhs: NodeId) {
        let name = self.sym(name);
        self.m.add_binding(Binding {
            name,
            rhs,
            doc: None,
            public: true,
            synthetic: false,
        });
    }

    /// Like [`bind`] but attaches a Markdown doc-comment to the binding.
    ///
    /// Used for non-1:1 HS3 → FlatPPL lowerings so that the emitted FlatPPL
    /// carries a human-readable `% HS3 <type> → <what was emitted>` provenance
    /// note (spec §05 / §11 doc-comment syntax).
    pub fn bind_doc(&mut self, name: &str, rhs: NodeId, doc_lines: &[&str]) {
        let name = self.sym(name);
        let lines: Box<[Box<str>]> = doc_lines.iter().map(|s| Box::from(*s)).collect();
        self.m.add_binding(Binding {
            name,
            rhs,
            doc: Some(Doc {
                markup: Markup::Md,
                lines,
            }),
            public: true,
            synthetic: false,
        });
    }

    /// `name = elementof(<set_node>)`.
    pub fn bind_set(&mut self, name: &str, set: NodeId) {
        let eo = self.call("elementof", &[set]);
        self.bind(name, eo);
    }

    /// A built-in used as a value (head passed positionally, e.g. into broadcast).
    pub fn call_head(&mut self, name: &str) -> NodeId {
        let sym = self.sym(name);
        self.m.alloc(Node::Const(sym))
    }

    /// A module-qualified callable `alias.name` (e.g. `hepphys.ContinuedPoisson`) as a value.
    pub fn module_call(&mut self, alias: &str, name: &str) -> NodeId {
        let alias = self.sym(alias);
        let name = self.sym(name);
        self.m.alloc(Node::Ref(Ref {
            ns: RefNs::Module(alias),
            name,
        }))
    }

    /// A module-member application `alias.name(args...)` — emits `CallHead::User` over a
    /// module ref, which prints as `alias.name(arg0, arg1, ...)` and round-trips correctly.
    pub fn module_user_call(&mut self, alias: &str, name: &str, args: &[NodeId]) -> NodeId {
        let callee = self.module_call(alias, name);
        self.m.alloc(Node::Call(Call {
            head: CallHead::User(callee),
            args: args.to_vec().into(),
            named: Vec::new().into(),
            inputs: None,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flatppl_syntax::{Syntax, print_with};

    #[test]
    fn builds_normal_call() {
        let mut m = flatppl_core::Module::new();
        {
            let mut b = Builder::new(&mut m);
            let mu = b.self_ref("mu_param");
            let sigma = b.self_ref("sigma_param");
            let normal = b.call_kw("Normal", &[("mu", mu), ("sigma", sigma)]);
            b.bind("mass", normal);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("Normal"), "got: {text}");
        assert!(text.contains("mu_param"), "got: {text}");
    }
}
