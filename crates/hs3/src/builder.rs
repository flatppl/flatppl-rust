//! Ergonomic helpers for building flatppl_core IR.
use flatppl_core::id::{NodeId, Symbol};
use flatppl_core::node::{Call, CallHead, Inputs, NamedArg, NamedKind, Node, Ref, RefNs, Scalar};
use flatppl_core::{Binding, Doc, Markup, Module};

/// FlatPPL reserved words that may not be a binding name (spec §05 / the parser's
/// `check_binding_name`). The name allocator must never hand one of these back.
const RESERVED: &[&str] = &["true", "false", "in", "all", "only", "self", "base"];

/// Sanitize an arbitrary source string (a pyhf/HS3 channel, sample, or modifier
/// name) into a valid FlatPPL identifier: every character outside
/// `[A-Za-z0-9_]` becomes `_`, a leading digit is prefixed with `_`, and the
/// empty string becomes `_`. Distinct inputs may sanitize to the same identifier
/// (e.g. `a-b` and `a.b`); the allocator's dedup step resolves that.
fn sanitize_ident(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() {
        out.push('_');
    }
    if out.as_bytes()[0].is_ascii_digit() {
        out.insert(0, '_');
    }
    out
}

pub(crate) struct Builder<'m> {
    pub(crate) m: &'m mut Module,
}

impl<'m> Builder<'m> {
    pub(crate) fn new(m: &'m mut Module) -> Self {
        Builder { m }
    }

    pub(crate) fn sym(&mut self, s: &str) -> Symbol {
        self.m.intern(s)
    }

    pub(crate) fn lit_real(&mut self, v: f64) -> NodeId {
        self.m.alloc(Node::Lit(Scalar::Real(v)))
    }

    pub(crate) fn lit_int(&mut self, v: i64) -> NodeId {
        self.m.alloc(Node::Lit(Scalar::Int(v)))
    }

    pub(crate) fn str_lit(&mut self, s: &str) -> NodeId {
        self.m.alloc(Node::Lit(Scalar::Str(s.into())))
    }

    pub(crate) fn self_ref(&mut self, name: &str) -> NodeId {
        let name = self.sym(name);
        self.m.alloc(Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }))
    }

    pub(crate) fn call(&mut self, head: &str, args: &[NodeId]) -> NodeId {
        let head = self.sym(head);
        self.m.alloc(Node::Call(Call {
            head: CallHead::Builtin(head),
            args: args.to_vec().into(),
            named: Vec::new().into(),
            inputs: None,
        }))
    }

    pub(crate) fn call_kw(&mut self, head: &str, kw: &[(&str, NodeId)]) -> NodeId {
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

    /// A call with `name = value` **field**-named arguments, e.g.
    /// `table(x = …, y = …)`, `cartprod(a = …, b = …)`, `record(p = …)`.
    /// Distinct from [`Builder::call_kw`], which emits function **kwargs**
    /// (`Normal(mu = …)`); the field form is the record/table/set-product
    /// constructor syntax (spec §03).
    pub(crate) fn call_fields(&mut self, head: &str, fields: &[(&str, NodeId)]) -> NodeId {
        let head = self.sym(head);
        let named = fields
            .iter()
            .map(|(k, v)| NamedArg {
                kind: NamedKind::Field,
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
    pub(crate) fn array(&mut self, elems: &[NodeId]) -> NodeId {
        self.call("vector", elems)
    }

    /// Stamp the module with `flatppl_compat = "<version>"` (spec §11: an ordinary
    /// string binding declaring the targeted FlatPPL language version). Call this
    /// **first** when building a generated module so it lands as the first binding.
    /// The version is [`flatppl_core::FLATPPL_COMPAT`], the ecosystem-wide pin.
    pub(crate) fn stamp_compat(&mut self) {
        let v = self.str_lit(flatppl_core::FLATPPL_COMPAT);
        self.bind("flatppl_compat", v);
    }

    pub(crate) fn bind(&mut self, name: &str, rhs: NodeId) {
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
    pub(crate) fn bind_doc(&mut self, name: &str, rhs: NodeId, doc_lines: &[&str]) {
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
    pub(crate) fn bind_set(&mut self, name: &str, set: NodeId) {
        let eo = self.call("elementof", &[set]);
        self.bind(name, eo);
    }

    /// A built-in used as a value (head passed positionally, e.g. into broadcast).
    pub(crate) fn call_head(&mut self, name: &str) -> NodeId {
        let sym = self.sym(name);
        self.m.alloc(Node::Const(sym))
    }

    /// A module-qualified callable `alias.name` (e.g. `hepphys.ContinuedPoisson`) as a value.
    pub(crate) fn module_call(&mut self, alias: &str, name: &str) -> NodeId {
        let alias = self.sym(alias);
        let name = self.sym(name);
        self.m.alloc(Node::Ref(Ref {
            ns: RefNs::Module(alias),
            name,
        }))
    }

    /// A module-member application `alias.name(args...)` — emits `CallHead::User` over a
    /// module ref, which prints as `alias.name(arg0, arg1, ...)` and round-trips correctly.
    pub(crate) fn module_user_call(&mut self, alias: &str, name: &str, args: &[NodeId]) -> NodeId {
        let callee = self.module_call(alias, name);
        self.m.alloc(Node::Call(Call {
            head: CallHead::User(callee),
            args: args.to_vec().into(),
            named: Vec::new().into(),
            inputs: None,
        }))
    }

    /// A clash-free binding name derived from `desired`: sanitize it to a valid
    /// identifier, then — if that name is already a module binding (params,
    /// `flatppl_compat`, `hepphys`, an earlier intermediate) or a reserved word —
    /// append `_2`, `_3`, … until free. Guarantees a unique, valid name for any
    /// input, so generated intermediate names never collide regardless of how the
    /// source document is written. The check is against the live module, so it
    /// works across the several `Builder`s the HS3 path creates (each new binding
    /// is registered immediately, so the next call sees it).
    pub(crate) fn alloc_name(&mut self, desired: &str) -> String {
        let base = sanitize_ident(desired);
        let mut cand = base.clone();
        let mut i = 2u32;
        loop {
            let reserved = RESERVED.contains(&cand.as_str());
            let sym = self.m.intern(&cand);
            if !reserved && self.m.binding_by_name(sym).is_none() {
                return cand;
            }
            cand = format!("{base}_{i}");
            i += 1;
        }
    }

    /// [`alloc_name`] + [`bind_doc`]: bind `rhs` under a clash-free name derived
    /// from `desired`, with a one-line `%` doc-comment (spec §05). Returns the
    /// clash-free name actually used (for `self_ref`s to it).
    pub(crate) fn bind_unique_doc(&mut self, desired: &str, rhs: NodeId, doc: &str) -> String {
        let name = self.alloc_name(desired);
        self.bind_doc(&name, rhs, &[doc]);
        name
    }

    /// A bare reification `functionof(<body>)` with auto-inferred inputs
    /// (`%autoinputs`). Wraps a parameter-dependent measure into a kernel so it
    /// can feed `likelihoodof` (spec §06: `likelihoodof` takes a kernel, not a
    /// measure). The input list is left for inference's auto-trace to fill — the
    /// converter deliberately imposes no parameter order (a pyhf workspace has no
    /// canonical one).
    pub(crate) fn functionof(&mut self, body: NodeId) -> NodeId {
        let head = self.m.intern("functionof");
        self.m.alloc(Node::Call(Call {
            head: CallHead::Builtin(head),
            args: vec![body].into(),
            named: Vec::new().into(),
            inputs: Some(Inputs::Auto),
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

    #[test]
    fn sanitize_handles_illegal_chars_and_leading_digit() {
        assert_eq!(sanitize_ident("a-b.c"), "a_b_c");
        assert_eq!(sanitize_ident("2bins"), "_2bins");
        assert_eq!(sanitize_ident(""), "_");
        assert_eq!(sanitize_ident("ok_name"), "ok_name");
    }

    #[test]
    fn alloc_name_dedups_and_avoids_reserved() {
        let mut m = flatppl_core::Module::new();
        let mut b = Builder::new(&mut m);
        // A colliding desired name gets a `_2`, `_3`, … suffix.
        let a = b.lit_real(1.0);
        assert_eq!(b.bind_unique_doc("expected", a, "d"), "expected");
        let c = b.lit_real(2.0);
        assert_eq!(b.bind_unique_doc("expected", c, "d"), "expected_2");
        let d = b.lit_real(3.0);
        assert_eq!(b.bind_unique_doc("expected", d, "d"), "expected_3");
        // Reserved words are never handed back.
        let e = b.lit_real(4.0);
        assert_eq!(b.bind_unique_doc("self", e, "d"), "self_2");
    }

    #[test]
    fn functionof_wraps_body_as_reification() {
        let mut m = flatppl_core::Module::new();
        {
            let mut b = Builder::new(&mut m);
            let mu = b.lit_real(0.0);
            let sigma = b.lit_real(1.0);
            let normal = b.call_kw("Normal", &[("mu", mu), ("sigma", sigma)]);
            let k = b.functionof(normal);
            b.bind("k", k);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("functionof"), "got: {text}");
        assert!(text.contains("Normal"), "got: {text}");
    }
}
