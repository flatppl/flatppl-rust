//! The [`Module`] — owns the IR arena, the bindings, the interner, and the
//! (optional) annotation side-tables; exposes the read / build / annotate API.

use crate::id::{Arena, BindingId, Idx, Interner, NodeId, SecondaryMap, Symbol};
use crate::node::{Node, Ref};
use crate::ty::{Mass, Phase, Type, ValueSet};
use std::collections::HashMap;

/// The trailing ` · <mass>` annotation for a measure/kernel type in
/// [`Module::display_type`]. An unknown or not-yet-inferred mass adds no
/// annotation (it would only be noise).
fn mass_suffix(mass: Mass) -> &'static str {
    match mass {
        Mass::Normalized => " · normalized",
        Mass::Finite => " · finite",
        Mass::LocallyFinite => " · locally-finite",
        Mass::Null => " · null",
        Mass::Unknown | Mass::Deferred => "",
    }
}

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

    /// Redirect a binding's right-hand side to a freshly-`alloc`'d node.
    /// Used by rewriting passes (e.g. the determiniser) that replace one expression
    /// with another inside an already-built module.
    pub fn set_binding_rhs(&mut self, id: BindingId, rhs: NodeId) {
        self.bindings.get_mut(id).rhs = rhs;
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

    /// Render a [`Type`] in a concise, code-like notation, resolving interned
    /// field/input names to their source text.
    ///
    /// Unlike the derived `Debug` — which prints Rust struct syntax and shows
    /// interned names as opaque `Symbol(n)` integers — this produces the
    /// readable form used for human-facing surfaces (e.g. LSP hover):
    /// `{mu: real, n: integer}`, `kernel(theta) → measure · normalized`,
    /// `measure<real> · normalized`, `real[3]`, `likelihood(theta) over real`.
    pub fn display_type(&self, ty: &Type) -> String {
        let mut s = String::new();
        self.write_type(&mut s, ty);
        s
    }

    /// Render a [`ValueSet`] in the value-set surface vocabulary, resolving
    /// interned `RecordSet` field names to their source text. For the
    /// symbol-free variants this matches the plain `Display` impl; only
    /// `RecordSet` (and nestings containing it) need the interner.
    pub fn display_valueset(&self, vs: &ValueSet) -> String {
        match vs {
            ValueSet::CartPow(elem, d) => {
                format!("cartpow({}, {d})", self.display_valueset(elem))
            }
            ValueSet::CartProd(parts) => {
                let inner: Vec<String> = parts.iter().map(|s| self.display_valueset(s)).collect();
                format!("cartprod({})", inner.join(", "))
            }
            ValueSet::RecordSet(fields) => {
                let inner: Vec<String> = fields
                    .iter()
                    .map(|(n, s)| format!("{}: {}", self.resolve(*n), self.display_valueset(s)))
                    .collect();
                format!("record({})", inner.join(", "))
            }
            // Symbol-free atoms: defer to the plain Display impl.
            other => other.to_string(),
        }
    }

    /// Render the full inferred *specification* of a node — its type plus the
    /// value-set and phase — as one compact line for an inline surface (LSP
    /// inlay hint), e.g. `real {nonnegreals, stochastic}`.
    ///
    /// The brace group carries the facts the bare type omits: the value-set
    /// when it is tighter than the type's natural extent (a plain `reals` over
    /// a `real` adds nothing and is dropped), and the phase. Returns `None`
    /// when the node has no inferred type.
    pub fn display_meta(&self, id: NodeId) -> Option<String> {
        let ty = self.type_of(id)?;
        let mut facts: Vec<String> = Vec::new();
        if let Some(vs) = self.valueset_of(id) {
            if !matches!(vs, ValueSet::Unknown | ValueSet::Deferred)
                && *vs != ValueSet::natural_of(ty)
            {
                facts.push(self.display_valueset(vs));
            }
        }
        if let Some(phase) = self.phase_of(id) {
            facts.push(phase.to_string());
        }
        let base = self.display_type(ty);
        Some(if facts.is_empty() {
            base
        } else {
            format!("{base} {{{}}}", facts.join(", "))
        })
    }

    fn write_type(&self, out: &mut String, ty: &Type) {
        use std::fmt::Write as _;
        match ty {
            Type::Deferred => out.push_str("deferred"),
            Type::Failed(reason) => {
                let _ = write!(out, "failed(\"{reason}\")");
            }
            Type::Any => out.push_str("any"),
            Type::Scalar(st) => {
                let _ = write!(out, "{st}");
            }
            Type::Array { shape, elem } => {
                self.write_type(out, elem);
                out.push('[');
                for (i, d) in shape.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "{d}");
                }
                out.push(']');
            }
            Type::TVector { len, elem } => {
                self.write_type(out, elem);
                let _ = write!(out, "[{len}]ᵀ");
            }
            Type::Record(fields) => {
                out.push('{');
                for (i, (name, fty)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "{}: ", self.resolve(*name));
                    self.write_type(out, fty);
                }
                out.push('}');
            }
            Type::Tuple(parts) => {
                out.push('(');
                for (i, p) in parts.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    self.write_type(out, p);
                }
                out.push(')');
            }
            Type::Table { columns, nrows } => {
                out.push_str("table {");
                for (i, (name, cty)) in columns.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "{}: ", self.resolve(*name));
                    self.write_type(out, cty);
                }
                let _ = write!(out, "}}[{nrows}]");
            }
            Type::Measure { domain, mass } => {
                out.push_str("measure<");
                self.write_type(out, domain);
                out.push('>');
                out.push_str(mass_suffix(*mass));
            }
            Type::Kernel { inputs, mass } => {
                out.push_str("kernel(");
                for (i, inp) in inputs.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(self.resolve(*inp));
                }
                out.push_str(") → measure");
                out.push_str(mass_suffix(*mass));
            }
            Type::Function { inputs } => {
                out.push_str("fn(");
                for (i, inp) in inputs.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(self.resolve(*inp));
                }
                out.push(')');
            }
            Type::Likelihood { inputs, obstype } => {
                out.push_str("likelihood(");
                for (i, inp) in inputs.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(self.resolve(*inp));
                }
                out.push_str(") over ");
                self.write_type(out, obstype);
            }
            Type::RngState => out.push_str("rngstate"),
            Type::Module => out.push_str("module"),
            Type::Var(n) => {
                let _ = write!(out, "?{n}");
            }
        }
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

    /// The innermost node whose span contains the byte `offset` — the node a
    /// cursor at `offset` is "on". Returns the span with the smallest width
    /// among all containing spans (so an argument wins over its enclosing
    /// call). `None` if no node's span contains `offset`.
    pub fn node_at_offset(&self, offset: u32) -> Option<NodeId> {
        let mut best: Option<(NodeId, u32)> = None; // (node, width)
        for i in 0..self.node_count() {
            let id = NodeId::from_usize(i);
            let Some(span) = self.span_of(id) else {
                continue;
            };
            if offset >= span.start && offset < span.end {
                let width = span.end - span.start;
                if best.is_none_or(|(_, w)| width < w) {
                    best = Some((id, width));
                }
            }
        }
        best.map(|(id, _)| id)
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
    fn node_at_offset_returns_innermost() {
        use crate::node::{Call, CallHead, Node, Scalar};
        let mut m = Module::new();
        // Build `add(7, 9)` with spans: call [0,9), `7` [4,5), `9` [7,8).
        let seven = m.alloc(Node::Lit(Scalar::Int(7)));
        m.set_span(seven, Span { start: 4, end: 5 });
        let nine = m.alloc(Node::Lit(Scalar::Int(9)));
        m.set_span(nine, Span { start: 7, end: 8 });
        let add = m.intern("add");
        let call = m.alloc(Node::Call(Call {
            head: CallHead::Builtin(add),
            args: vec![seven, nine].into(),
            named: Vec::new().into(),
            inputs: None,
        }));
        m.set_span(call, Span { start: 0, end: 9 });

        assert_eq!(m.node_at_offset(4), Some(seven)); // inside `7` -> the literal
        assert_eq!(m.node_at_offset(1), Some(call)); // on `add` text -> the call
        assert_eq!(m.node_at_offset(99), None); // past end -> nothing
    }

    #[test]
    fn display_type_resolves_names_and_uses_code_notation() {
        use crate::ty::{Dim, Mass, ScalarType, Type};
        let mut m = Module::new();
        let mu = m.intern("mu");
        let n = m.intern("n");
        let theta = m.intern("theta");
        let a = m.intern("a");
        let b = m.intern("b");
        let real = || Type::Scalar(ScalarType::Real);
        let int = || Type::Scalar(ScalarType::Integer);

        // Record field names are resolved (not `Symbol(n)`); scalars are bare.
        let rec = Type::Record(vec![(mu, real()), (n, int())].into());
        assert_eq!(m.display_type(&rec), "{mu: real, n: integer}");

        // Kernel / likelihood / measure — the constructs whose Debug was worst.
        assert_eq!(
            m.display_type(&Type::Kernel {
                inputs: vec![theta].into(),
                mass: Mass::Normalized
            }),
            "kernel(theta) → measure · normalized"
        );
        assert_eq!(
            m.display_type(&Type::Likelihood {
                inputs: vec![theta].into(),
                obstype: Box::new(real())
            }),
            "likelihood(theta) over real"
        );
        assert_eq!(
            m.display_type(&Type::Measure {
                domain: Box::new(real()),
                mass: Mass::Finite
            }),
            "measure<real> · finite"
        );
        // Unknown / deferred mass adds no annotation (avoids noise).
        assert_eq!(
            m.display_type(&Type::Measure {
                domain: Box::new(real()),
                mass: Mass::Unknown
            }),
            "measure<real>"
        );

        // Arrays: static, multi-dim, and dynamic (`?`); transposed vector.
        assert_eq!(
            m.display_type(&Type::Array {
                shape: vec![Dim::Static(3)].into(),
                elem: Box::new(real())
            }),
            "real[3]"
        );
        assert_eq!(
            m.display_type(&Type::Array {
                shape: vec![Dim::Static(2), Dim::Dynamic].into(),
                elem: Box::new(int())
            }),
            "integer[2, ?]"
        );
        assert_eq!(
            m.display_type(&Type::TVector {
                len: Dim::Static(3),
                elem: Box::new(real())
            }),
            "real[3]ᵀ"
        );

        // Function / tuple / table / type-var / bare keywords.
        assert_eq!(
            m.display_type(&Type::Function {
                inputs: vec![a, b].into()
            }),
            "fn(a, b)"
        );
        assert_eq!(
            m.display_type(&Type::Tuple(vec![real(), int()].into())),
            "(real, integer)"
        );
        assert_eq!(
            m.display_type(&Type::Table {
                columns: vec![(a, real())].into(),
                nrows: Dim::Static(100)
            }),
            "table {a: real}[100]"
        );
        assert_eq!(m.display_type(&Type::Var(0)), "?0");
        assert_eq!(m.display_type(&Type::Any), "any");
        assert_eq!(m.display_type(&Type::Deferred), "deferred");
        assert_eq!(m.display_type(&Type::RngState), "rngstate");
        assert_eq!(m.display_type(&Type::Module), "module");
    }

    #[test]
    fn display_valueset_shows_record_field_names() {
        use crate::ty::Dim;
        use crate::ty::ValueSet::*;
        let mut m = Module::new();
        let a = m.intern("alpha");
        let b = m.intern("beta");
        let rs = RecordSet(Box::new([(a, Reals), (b, UnitInterval)]));
        assert_eq!(
            m.display_valueset(&rs),
            "record(alpha: reals, beta: unitinterval)"
        );
        // positional + nested power render too
        let cp = CartProd(Box::new([Reals, PosReals]));
        assert_eq!(m.display_valueset(&cp), "cartprod(reals, posreals)");
        let pow = CartPow(
            Box::new(CartPow(Box::new(Reals), Dim::Static(3))),
            Dim::Static(2),
        );
        assert_eq!(m.display_valueset(&pow), "cartpow(cartpow(reals, 3), 2)");
    }

    #[test]
    fn display_valueset_recurses_record_inside_cartprod() {
        use crate::ty::Dim;
        use crate::ty::ValueSet::*;
        let mut m = Module::new();
        let a = m.intern("a");
        let b = m.intern("b");
        // RecordSet nested as the first element of a CartProd — the interner
        // must be threaded into the recursive call so field names resolve.
        let rs = RecordSet(Box::new([(a, Reals), (b, UnitInterval)]));
        let cp = CartProd(Box::new([rs, PosReals]));
        assert_eq!(
            m.display_valueset(&cp),
            "cartprod(record(a: reals, b: unitinterval), posreals)"
        );
        // RecordSet nested inside CartPow — same recursion path, different arm.
        let a2 = m.intern("a");
        let b2 = m.intern("b");
        let rs2 = RecordSet(Box::new([(a2, Reals), (b2, UnitInterval)]));
        let cpow = CartPow(Box::new(rs2), Dim::Static(3));
        assert_eq!(
            m.display_valueset(&cpow),
            "cartpow(record(a: reals, b: unitinterval), 3)"
        );
    }

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
