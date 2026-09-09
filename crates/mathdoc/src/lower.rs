//! Lowering: IR bindings → [`Statement`]s (`NOTATION.md`).
//!
//! One binding becomes one [`Row`] in source order. The lowering reads the
//! module's inferred types, phases and filled `%autoinputs` where they are
//! present and degrades structurally where they are not: an untyped module
//! still renders, but index ranges, kernel input lists and the
//! measure/callable kinds fall back to what the syntax alone says.
//!
//! Two kinds of state are threaded: **scopes** (the parameters of a
//! reification being lowered — a body reference to a boundary node prints as
//! its input name) and **index letters** (fresh per statement, skipping every
//! name the module binds).
//!
//! Nesting is bounded by [`flatppl_core::DEFAULT_MAX_DEPTH`] at both ends: a
//! right-hand side nested deeper than that is not lowered at all, and a lowered
//! statement that ends up deeper (a 200-term `superpose` folds into a 200-deep
//! chain) is discarded; either way the row shows the binding's source text
//! with a diagnostic. Every printer over a [`Statement`] may therefore recurse
//! freely.

use std::collections::{HashMap, HashSet};

use flatppl_core::{
    BindingId, Call, CallHead, DEFAULT_MAX_DEPTH, Dim, Inputs, Module, Node, NodeId, Phase, Ref,
    RefNs, Scalar, Symbol, Type, Variance,
};

use crate::ast::{BigOp, BinOp, Fence, Math, Op, Rel, Statement, Sym};

/// Arrays longer than this print as a membership statement, their values
/// going to the document's data appendix.
pub const INLINE_ARRAY_LIMIT: usize = 12;

/// The structural kind of a row, for the viewer (`NOTATION.md`, contract §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Draw,
    Value,
    Measure,
    Callable,
    Likelihood,
    Module,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Draw => "draw",
            Kind::Value => "value",
            Kind::Measure => "measure",
            Kind::Callable => "callable",
            Kind::Likelihood => "likelihood",
            Kind::Module => "module",
        }
    }
}

/// One rendered binding (or one decomposition group of bindings).
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// The binding the row stands for; for a decomposition group, the first
    /// consumer in source order.
    pub binding: BindingId,
    /// Every name this row binds (one, or the decomposition's targets).
    pub names: Vec<String>,
    pub kind: Kind,
    pub statement: Statement,
    /// A short annotation beside the row (`external input`, `120 values`, …).
    pub annotation: Option<String>,
    /// The row shows a membership in place of a long literal array whose
    /// values belong in a data appendix ([`Lowerer::full_value`]).
    pub elided: bool,
    /// Constructs this row could not render and printed as source text.
    pub diagnostics: Vec<String>,
}

/// Lower every binding of `module` in source order.
///
/// Parser-generated (synthetic) bindings get no row of their own: a
/// decomposition whose projections cover positions `1..n` (`n ≥ 2`) renders as
/// one tuple row, every other use of a synthetic binding is inlined where it
/// is referenced, and an unreferenced one (`_ = expr`) renders nothing.
pub fn lower_module(module: &Module) -> Vec<Row> {
    lower_module_with(module, None)
}

/// [`lower_module`] with the module's source text at hand, so a row that
/// falls back to source text shows the expression as written rather than
/// its canonical print.
pub fn lower_module_with(module: &Module, source: Option<&str>) -> Vec<Row> {
    let mut lowerer = Lowerer::with_source(module, source);
    let groups = decomposition_groups(module);
    let mut done: HashSet<BindingId> = HashSet::new();
    let mut rows = Vec::new();
    for (id, binding) in module.bindings() {
        if module.resolve(binding.name) == "flatppl_compat" || binding.synthetic {
            continue;
        }
        if done.contains(&id) {
            continue;
        }
        // A consumer of a complete decomposition renders the whole group once.
        if let Some((source_binding, consumers)) =
            groups.iter().find(|(_, c)| c.iter().any(|(_, b)| *b == id))
        {
            for (_, b) in consumers {
                done.insert(*b);
            }
            rows.push(lowerer.decomposition_row(*source_binding, consumers));
            continue;
        }
        rows.push(lowerer.row(id));
    }
    rows
}

/// Synthetic decomposition sources whose projections cover positions
/// `1..n` for some `n ≥ 2` → their consumers `(position, binding)` in position
/// order.
fn decomposition_groups(module: &Module) -> HashMap<BindingId, Vec<(i64, BindingId)>> {
    let mut groups = HashMap::new();
    for (id, binding) in module.bindings() {
        if !binding.synthetic {
            continue;
        }
        let mut consumers: Vec<(i64, BindingId)> = module
            .bindings()
            .filter_map(|(cid, c)| match projection_source(module, c.rhs) {
                Some((source, k)) if source == id => Some((k, cid)),
                _ => None,
            })
            .collect();
        consumers.sort_by_key(|(k, _)| *k);
        let complete = consumers.len() >= 2
            && consumers
                .iter()
                .enumerate()
                .all(|(i, (k, _))| *k == i as i64 + 1);
        if complete {
            groups.insert(id, consumers);
        }
    }
    groups
}

/// A projection `get(<ref to a synthetic source>, k)` → `(source, k)`.
fn projection_source(module: &Module, rhs: NodeId) -> Option<(BindingId, i64)> {
    let Node::Call(call) = module.node(rhs) else {
        return None;
    };
    let CallHead::Builtin(head) = call.head else {
        return None;
    };
    if module.resolve(head) != "get" || call.args.len() != 2 {
        return None;
    }
    let Node::Ref(r) = module.node(call.args[0]) else {
        return None;
    };
    if r.ns != RefNs::SelfMod {
        return None;
    }
    let source = module.binding_by_name(r.name)?;
    if !module.binding(source).synthetic {
        return None;
    }
    match module.node(call.args[1]) {
        Node::Lit(Scalar::Int(k)) => Some((source, *k)),
        _ => None,
    }
}

/// A reification parameter in scope: the node it stands for and the name it
/// prints as. `linked` keeps the back-reference when the parameter keeps the
/// binding's own name.
#[derive(Clone, Debug)]
struct Param {
    target: Ref,
    name: String,
    linked: bool,
}

pub struct Lowerer<'m> {
    m: &'m Module,
    scopes: Vec<Vec<Param>>,
    diagnostics: Vec<String>,
    /// Names the module binds: index letters avoid them.
    bound: HashSet<String>,
    /// Index letters handed out for the statement being lowered.
    indices: Vec<String>,
    /// Synthetic bindings being inlined, so a malformed self-reference
    /// prints as a name rather than recursing.
    inlining: Vec<BindingId>,
    /// The module's source text, for fallback rows.
    source: Option<&'m str>,
}

/// What a binding's right-hand side lowered to.
struct Lowered {
    statement: Statement,
    annotation: Option<String>,
    elided: bool,
}

impl<'m> Lowerer<'m> {
    pub fn new(m: &'m Module) -> Self {
        Self::with_source(m, None)
    }

    /// A lowerer that can quote the module's source text in fallback rows.
    pub fn with_source(m: &'m Module, source: Option<&'m str>) -> Self {
        let bound = m
            .bindings()
            .map(|(_, b)| m.resolve(b.name).to_string())
            .collect();
        Lowerer {
            m,
            scopes: Vec::new(),
            diagnostics: Vec::new(),
            bound,
            indices: Vec::new(),
            inlining: Vec::new(),
            source,
        }
    }

    // ── rows ───────────────────────────────────────────────────────────────

    /// The row for one ordinary binding.
    pub fn row(&mut self, id: BindingId) -> Row {
        self.indices.clear();
        self.diagnostics.clear();
        let binding = self.m.binding(id);
        let name = self.m.resolve(binding.name).to_string();
        let mut lowered = if too_deep(self.m, binding.rhs) {
            None
        } else {
            Some(self.statement(&name, binding.rhs))
        };
        if lowered
            .as_ref()
            .is_some_and(|l| l.statement.depth() > DEFAULT_MAX_DEPTH)
        {
            lowered = None;
        }
        let lowered = lowered.unwrap_or_else(|| self.source_fallback(id, &name));
        Row {
            binding: id,
            names: vec![name],
            kind: self.kind(binding.rhs),
            statement: lowered.statement,
            annotation: lowered.annotation,
            elided: lowered.elided,
            diagnostics: std::mem::take(&mut self.diagnostics),
        }
    }

    /// The row for a decomposition `a, b = source`.
    fn decomposition_row(
        &mut self,
        source_binding: BindingId,
        consumers: &[(i64, BindingId)],
    ) -> Row {
        self.indices.clear();
        self.diagnostics.clear();
        let source = self.m.binding(source_binding).rhs;
        let names: Vec<String> = consumers
            .iter()
            .map(|(_, b)| self.m.resolve(self.m.binding(*b).name).to_string())
            .collect();
        let lhs = Math::paren(names.iter().map(|n| Math::binding(n)).collect());
        let mut statement = if too_deep(self.m, source) {
            None
        } else {
            let (rel, rhs) = match self.m.node(source) {
                Node::Call(call) if self.head_is(call, "draw") && call.args.len() == 1 => {
                    (Rel::Sim, self.expr(call.args[0]))
                }
                _ => (Rel::Eq, self.expr(source)),
            };
            Some(Statement {
                lhs: lhs.clone(),
                rel,
                rhs,
            })
        };
        if statement
            .as_ref()
            .is_some_and(|s| s.depth() > DEFAULT_MAX_DEPTH)
        {
            statement = None;
        }
        let statement = statement.unwrap_or_else(|| {
            self.diagnostics.push(too_deep_message(&names.join(", ")));
            let (rel, text) = self.rhs_text(source_binding);
            Statement {
                lhs,
                rel,
                rhs: Math::Code(text),
            }
        });
        Row {
            binding: consumers[0].1,
            names,
            kind: self.kind(source),
            statement,
            annotation: None,
            elided: false,
            diagnostics: std::mem::take(&mut self.diagnostics),
        }
    }

    /// The full value of a binding whose row elided it (the data appendix).
    pub fn full_value(&mut self, id: BindingId) -> Math {
        self.indices.clear();
        let rhs = self.m.binding(id).rhs;
        if too_deep(self.m, rhs) {
            return Math::Code(self.rhs_text(id).1);
        }
        let value = self.expr(rhs);
        if value.depth() > DEFAULT_MAX_DEPTH {
            return Math::Code(self.rhs_text(id).1);
        }
        value
    }

    /// A row that shows the binding's source text because its expression is
    /// too deeply nested to lower. A projection of a decomposition source
    /// (`p, _ = v`) shows the source with the component index, never the
    /// generated name: `p = v[1]`, or `p ∼ M` annotated "component 1".
    fn source_fallback(&mut self, id: BindingId, name: &str) -> Lowered {
        self.diagnostics.push(too_deep_message(name));
        let lhs = Math::binding(name);
        let (rel, text, annotation) = match projection_source(self.m, self.m.binding(id).rhs) {
            Some((source, k)) => match self.rhs_text(source) {
                (Rel::Sim, text) => (Rel::Sim, text, Some(format!("component {k}"))),
                (_, text) => (Rel::Eq, format!("{text}[{k}]"), None),
            },
            None => {
                let (rel, text) = self.rhs_text(id);
                (rel, text, None)
            }
        };
        Lowered {
            statement: Statement {
                lhs,
                rel,
                rhs: Math::Code(text),
            },
            annotation,
            elided: false,
        }
    }

    /// A binding's right-hand side as FlatPPL text, with the relation it
    /// binds under (`∼` for a `draw`, whose measure is then the text): the
    /// source slice when the source is at hand, the canonical print of that
    /// one binding otherwise. The canonical printer recurses, so the slice
    /// is what keeps a fallback row safe on a deep expression; a module
    /// without source text is a programmatic caller's.
    fn rhs_text(&self, id: BindingId) -> (Rel, String) {
        let rhs = self.m.binding(id).rhs;
        let (rel, node) = match self.m.node(rhs) {
            Node::Call(c) if self.head_is(c, "draw") && c.args.len() == 1 => (Rel::Sim, c.args[0]),
            _ => (Rel::Eq, rhs),
        };
        if let Some(src) = self.source
            && let Some(span) = self.m.span_of(node)
            && let Some(slice) = src.get(span.start as usize..span.end as usize)
        {
            return (rel, slice.trim().to_string());
        }
        let name = self.m.resolve(self.m.binding(id).name);
        let mut only = self.m.clone();
        only.retain_bindings(&[id].into_iter().collect());
        let text = flatppl_syntax::print(&only);
        // Drop doc-comment lines and the `name =` / `name ~` prefix.
        let body: Vec<&str> = text
            .lines()
            .filter(|l| !l.trim_start().starts_with('%'))
            .collect();
        let joined = body.join("\n");
        let trimmed = joined.trim();
        let rest = trimmed
            .strip_prefix(name)
            .and_then(|r| r.strip_prefix(" = ").or_else(|| r.strip_prefix(" ~ ")))
            .unwrap_or(trimmed);
        (rel, rest.to_string())
    }

    fn kind(&self, rhs: NodeId) -> Kind {
        if let Node::Call(call) = self.m.node(rhs)
            && self.head_is(call, "draw")
        {
            return Kind::Draw;
        }
        match self.m.type_of(rhs) {
            Some(Type::Measure { .. }) => Kind::Measure,
            Some(Type::Kernel { .. } | Type::Function { .. }) => Kind::Callable,
            Some(Type::Likelihood { .. }) => Kind::Likelihood,
            Some(Type::Module) => Kind::Module,
            Some(_) => Kind::Value,
            None => match self.m.node(rhs) {
                Node::Call(call) => match self.head_name(call) {
                    Some("functionof" | "kernelof") => Kind::Callable,
                    Some("likelihoodof" | "joint_likelihood") => Kind::Likelihood,
                    Some("load_module" | "standard_module") => Kind::Module,
                    Some(h) if self.is_measure_head(h) => Kind::Measure,
                    _ => Kind::Value,
                },
                _ => Kind::Value,
            },
        }
    }

    fn is_measure_head(&self, head: &str) -> bool {
        matches!(
            head,
            "weighted"
                | "logweighted"
                | "superpose"
                | "normalize"
                | "truncate"
                | "pushfwd"
                | "locscale"
                | "joint"
                | "iid"
                | "lawof"
                | "bayesupdate"
                | "restrict"
                | "kchain"
                | "jointchain"
                | "markovchain"
                | "kscan"
                | "Lebesgue"
                | "Counting"
                | "Dirac"
                | "PoissonProcess"
                | "BinnedPoissonProcess"
        ) || flatppl_infer::builtin_catalogue().base_is_distribution(head)
    }

    /// `name = rhs` as a statement, with the statement forms of `NOTATION.md`.
    fn statement(&mut self, name: &str, rhs: NodeId) -> Lowered {
        let lhs = Math::binding(name);
        let plain = |this: &mut Self, lhs: Math| Lowered {
            statement: eq(lhs, this.expr(rhs)),
            annotation: None,
            elided: false,
        };
        let Node::Call(call) = self.m.node(rhs) else {
            return plain(self, lhs);
        };
        let Some(head) = self.head_name(call).map(str::to_string) else {
            return plain(self, lhs);
        };
        match head.as_str() {
            "elementof" | "external" if call.args.len() == 1 => {
                let set = self.expr(call.args[0]);
                Lowered {
                    statement: Statement {
                        lhs,
                        rel: Rel::In,
                        rhs: set,
                    },
                    annotation: (head == "external").then(|| "external input".to_string()),
                    elided: false,
                }
            }
            "draw" if call.args.len() == 1 => {
                let measure = self.expr(call.args[0]);
                Lowered {
                    statement: Statement {
                        lhs,
                        rel: Rel::Sim,
                        rhs: measure,
                    },
                    annotation: None,
                    elided: false,
                }
            }
            "fixed" if call.args.len() == 1 && call.named.is_empty() => {
                let value = self.expr(call.args[0]);
                Lowered {
                    statement: eq(lhs, value),
                    annotation: Some("fixed".to_string()),
                    elided: false,
                }
            }
            "functionof" | "kernelof" if !call.args.is_empty() => {
                let statement = self.reification_statement(name, rhs, call, &head);
                Lowered {
                    statement,
                    annotation: None,
                    elided: false,
                }
            }
            "likelihoodof" if call.args.len() == 2 && call.named.is_empty() => {
                let inputs = self.callable_inputs(rhs).unwrap_or_default();
                let lhs = if inputs.is_empty() {
                    lhs
                } else {
                    Math::apply(lhs, inputs.clone())
                };
                let rhs = self.likelihood(call.args[0], call.args[1], &inputs);
                Lowered {
                    statement: eq(lhs, rhs),
                    annotation: None,
                    elided: false,
                }
            }
            "aggregate" | "metricsum" if call.args.len() == 3 => {
                let agg = self.aggregation(call, &head);
                let lhs = decorate_with_axes(lhs, &agg.out_axes);
                Lowered {
                    statement: eq(lhs, agg.body),
                    annotation: agg.annotation,
                    elided: false,
                }
            }
            "vector" if call.args.len() > INLINE_ARRAY_LIMIT && self.all_literals(call) => {
                let n = call.args.len();
                let set = self.literal_array_set(call, rhs);
                Lowered {
                    statement: Statement {
                        lhs,
                        rel: Rel::In,
                        rhs: set,
                    },
                    annotation: Some(format!("{n} values, see the data appendix")),
                    elided: true,
                }
            }
            _ => plain(self, lhs),
        }
    }

    fn all_literals(&self, call: &Call) -> bool {
        call.args.iter().all(|a| match self.m.node(*a) {
            Node::Lit(_) => true,
            Node::Call(c) => {
                self.head_is(c, "neg")
                    && c.args.len() == 1
                    && matches!(self.m.node(c.args[0]), Node::Lit(_))
            }
            _ => false,
        })
    }

    /// `ℝ^{n}` / `ℤ^{n}` for an elided literal array.
    fn literal_array_set(&mut self, call: &Call, node: NodeId) -> Math {
        let elem = match self.m.type_of(node) {
            Some(Type::Array { elem, .. }) => match elem.as_ref() {
                Type::Scalar(flatppl_core::ScalarType::Integer) => Math::Sym(Sym::Integers),
                Type::Scalar(flatppl_core::ScalarType::Boolean) => Math::Sym(Sym::Booleans),
                _ => Math::Sym(Sym::Reals),
            },
            _ => {
                let all_int = call.args.iter().all(|a| match self.m.node(*a) {
                    Node::Lit(Scalar::Int(_)) => true,
                    Node::Call(c) => c
                        .args
                        .first()
                        .is_some_and(|x| matches!(self.m.node(*x), Node::Lit(Scalar::Int(_)))),
                    _ => false,
                });
                Math::Sym(if all_int { Sym::Integers } else { Sym::Reals })
            }
        };
        Math::pow(elem, Math::int(call.args.len() as i64))
    }

    // ── reifications ───────────────────────────────────────────────────────

    /// `F = functionof(e, …)` → `F(params) = e`; `K = kernelof(x, …)` →
    /// `K(params) = Law(x | params)`. A reification with no parameters is
    /// `F() = e`.
    fn reification_statement(
        &mut self,
        name: &str,
        node: NodeId,
        call: &Call,
        head: &str,
    ) -> Statement {
        let Some(params) = self.reification_params(node, call) else {
            // Unfilled `%autoinputs`: the boundary is unknown.
            let body = self.expr(call.args[0]);
            return eq(Math::binding(name), Math::call(head, vec![body]));
        };
        self.scopes.push(params.clone());
        let param_maths: Vec<Math> = params.iter().map(|p| self.param_math(p)).collect();
        let lhs = Math::apply(Math::binding(name), param_maths.clone());
        let rhs = if head == "kernelof" {
            self.law_given(call.args[0], &param_maths)
        } else {
            self.expr(call.args[0])
        };
        self.scopes.pop();
        eq(lhs, rhs)
    }

    /// A reification in expression position: `params ↦ body`, or the body
    /// alone when it has no parameters.
    fn reification_expr(&mut self, node: NodeId, call: &Call, head: &str) -> Math {
        let Some(params) = self.reification_params(node, call) else {
            let body = self.expr(call.args[0]);
            return Math::call(head, vec![body]);
        };
        self.scopes.push(params.clone());
        let param_maths: Vec<Math> = params.iter().map(|p| self.param_math(p)).collect();
        let body = if head == "kernelof" {
            self.law_given(call.args[0], &param_maths)
        } else {
            self.expr(call.args[0])
        };
        self.scopes.pop();
        match param_maths.len() {
            0 => body,
            1 => Math::relation(param_maths.into_iter().next().unwrap(), Rel::MapsTo, body),
            _ => Math::relation(Math::paren(param_maths), Rel::MapsTo, body),
        }
    }

    /// The parameters of a reification: authored boundary entries, or the
    /// `%autoinputs` list inference filled. `None` when neither is available.
    fn reification_params(&self, node: NodeId, call: &Call) -> Option<Vec<Param>> {
        let entries: Vec<(Symbol, Ref)> = match &call.inputs {
            Some(Inputs::Spec(entries)) => entries.to_vec(),
            Some(Inputs::Auto) => self.m.auto_inputs_of(node)?.to_vec(),
            None => return None,
        };
        Some(
            entries
                .iter()
                .map(|(name, target)| {
                    let name = self.m.resolve(*name).to_string();
                    let linked = target.ns == RefNs::SelfMod && self.m.resolve(target.name) == name;
                    Param {
                        target: *target,
                        name,
                        linked,
                    }
                })
                .collect(),
        )
    }

    fn param_math(&self, p: &Param) -> Math {
        Math::ident(&p.name, p.linked.then_some(p.name.as_str()))
    }

    /// `Law(x | params)` for a kernel body, with `x` flattened by the record
    /// rule; no bar when there are no parameters.
    fn law_given(&mut self, output: NodeId, params: &[Math]) -> Math {
        let mut items = self.law_items(output);
        if !params.is_empty() {
            items.push(Math::Op(Op::Bar));
            items.extend(intersperse_commas(params.to_vec()));
        }
        Math::apply(Math::Sym(Sym::Law), vec![Math::row(items)])
    }

    /// The arguments of `Law(…)`: a record's fields by the same-name rule,
    /// anything else as itself.
    fn law_items(&mut self, output: NodeId) -> Vec<Math> {
        if let Node::Call(call) = self.m.node(output)
            && self.head_is(call, "record")
            && call.args.is_empty()
        {
            let fields = self.record_items(call);
            return intersperse_commas(fields);
        }
        vec![self.expr(output)]
    }

    /// `p_K(data | inputs)` — the likelihood body; no bar when the kernel has
    /// no inputs.
    fn likelihood(&mut self, kernel: NodeId, data: NodeId, inputs: &[Math]) -> Math {
        let k = self.expr(kernel);
        let mut items = self.law_items(data);
        if !inputs.is_empty() {
            items.push(Math::Op(Op::Bar));
            items.extend(intersperse_commas(inputs.to_vec()));
        }
        Math::apply(
            Math::subscript(Math::letter('p'), k),
            vec![Math::row(items)],
        )
    }

    /// The input names of a callable-typed or likelihood-typed node, as
    /// identifiers linked to the bindings of the same name.
    fn callable_inputs(&self, node: NodeId) -> Option<Vec<Math>> {
        let inputs = match self.m.type_of(node)? {
            Type::Kernel { inputs, .. }
            | Type::Function { inputs }
            | Type::Likelihood { inputs, .. } => inputs,
            _ => return None,
        };
        Some(
            inputs
                .iter()
                .map(|s| {
                    let name = self.m.resolve(*s);
                    let linked = self.bound.contains(name);
                    Math::ident(name, linked.then_some(name))
                })
                .collect(),
        )
    }

    // ── expressions ────────────────────────────────────────────────────────

    pub fn expr(&mut self, id: NodeId) -> Math {
        match self.m.node(id) {
            Node::Lit(s) => match s {
                Scalar::Int(i) => Math::int(*i),
                Scalar::Real(r) => Math::real(*r),
                Scalar::Bool(b) => Math::text(if *b { "true" } else { "false" }),
                Scalar::Str(s) => Math::Str(s.to_string()),
            },
            Node::Const(sym) => {
                let name = self.m.resolve(*sym);
                self.constant(name)
            }
            Node::Ref(r) => self.reference(*r),
            Node::Hole => {
                self.diagnostics
                    .push("a bare hole `_` outside `fn(…)`; shown as source text".to_string());
                Math::Code("_".into())
            }
            Node::Axis(axis) => Math::ident(self.m.resolve(axis.name), None),
            Node::Call(call) => self.call(id, call),
        }
    }

    fn constant(&self, name: &str) -> Math {
        match name {
            "pi" => Math::Sym(Sym::Pi),
            "inf" => Math::Sym(Sym::Infinity),
            "im" => Math::Sym(Sym::ImagUnit),
            "true" | "false" => Math::text(name),
            "reals" => Math::Sym(Sym::ExtendedReals),
            "posreals" => half_line(Fence::Paren, Math::int(0)),
            "nonnegreals" => half_line(Fence::Bracket, Math::int(0)),
            "unitinterval" => Math::bracket(vec![Math::int(0), Math::int(1)]),
            "integers" => Math::Sym(Sym::Integers),
            "posintegers" => Math::Sym(Sym::PosIntegers),
            "nonnegintegers" => Math::Sym(Sym::NonNegIntegers),
            "booleans" => Math::Sym(Sym::Booleans),
            "complexes" => Math::Sym(Sym::Complexes),
            "all" => Math::Sym(Sym::Placeholder),
            // A builtin used as a value (`reduce(sum, xs)`, `pushfwd(exp, M)`).
            other => Math::text(other),
        }
    }

    fn reference(&mut self, r: Ref) -> Math {
        // A reification parameter in scope prints as its input name.
        for scope in self.scopes.iter().rev() {
            if let Some(p) = scope.iter().find(|p| p.target == r) {
                return self.param_math(p);
            }
        }
        match r.ns {
            RefNs::SelfMod => {
                // A synthetic binding has no row: inline it where it is used.
                if let Some(b) = self.m.binding_by_name(r.name)
                    && self.m.binding(b).synthetic
                    && !self.inlining.contains(&b)
                {
                    self.inlining.push(b);
                    let inlined = self.expr(self.m.binding(b).rhs);
                    self.inlining.pop();
                    return inlined;
                }
                Math::binding(self.m.resolve(r.name))
            }
            RefNs::Local => {
                // A placeholder no boundary declares: strip its underscores.
                let raw = self.m.resolve(r.name);
                Math::ident(raw.trim_matches('_'), None)
            }
            RefNs::Module(alias) => {
                let alias = self.m.resolve(alias);
                Math::row(vec![
                    Math::binding(alias),
                    Math::Op(Op::Dot),
                    Math::ident(self.m.resolve(r.name), None),
                ])
            }
        }
    }

    fn head_name(&self, call: &Call) -> Option<&'m str> {
        match call.head {
            CallHead::Builtin(sym) => Some(self.m.resolve(sym)),
            CallHead::User(_) => None,
        }
    }

    fn head_is(&self, call: &Call, name: &str) -> bool {
        self.head_name(call) == Some(name)
    }

    fn args(&mut self, call: &Call) -> Vec<Math> {
        call.args.iter().map(|a| self.expr(*a)).collect()
    }

    /// Keyword arguments as `(name, value)`, in source order.
    fn kwargs(&mut self, call: &Call) -> Vec<(String, Math)> {
        call.named
            .iter()
            .map(|n| (self.m.resolve(n.name).to_string(), self.expr(n.value)))
            .collect()
    }

    /// `name = value` entries.
    fn labelled(&mut self, call: &Call) -> Vec<Math> {
        call.named
            .iter()
            .map(|n| {
                let value = self.expr(n.value);
                Math::relation(Math::ident(self.m.resolve(n.name), None), Rel::Eq, value)
            })
            .collect()
    }

    /// Record fields by the same-name rule: a field whose value is the binding
    /// of the same name prints as that symbol, anything else as `name = value`.
    fn record_items(&mut self, call: &Call) -> Vec<Math> {
        call.named
            .iter()
            .map(|n| {
                let field = self.m.resolve(n.name);
                if let Node::Ref(r) = self.m.node(n.value)
                    && r.ns == RefNs::SelfMod
                    && self.m.resolve(r.name) == field
                    && !self.in_scope(*r)
                {
                    return Math::binding(field);
                }
                let value = self.expr(n.value);
                Math::relation(Math::ident(field, None), Rel::Eq, value)
            })
            .collect()
    }

    fn in_scope(&self, r: Ref) -> bool {
        self.scopes.iter().any(|s| s.iter().any(|p| p.target == r))
    }

    /// Positional arguments followed by keyword arguments moved into their
    /// declared positions (see [`order_kwargs`]).
    fn ordered_args(&mut self, name: &str, call: &Call) -> Vec<Math> {
        let args = self.args(call);
        let kwargs = self.kwargs(call);
        order_kwargs(param_names(name), args, kwargs)
    }

    fn call(&mut self, id: NodeId, call: &Call) -> Math {
        let Some(head) = self.head_name(call).map(str::to_string) else {
            return self.user_call(call);
        };
        let head = head.as_str();
        match head {
            // ── values ───────────────────────────────────────────────────
            "vector" => Math::paren(self.args(call)),
            "tuple" => Math::paren(self.args(call)),
            "record" => Math::paren(self.record_items(call)),
            "table" => Math::apply(Math::text("table"), self.labelled(call)),
            "rowstack" | "colstack" if call.args.len() == 1 => self.stack(head, call.args[0]),
            // ── sets ─────────────────────────────────────────────────────
            "cartpow" if call.args.len() == 2 => {
                let set = self.expr(call.args[0]);
                let size = self.size_exponent(call.args[1]);
                Math::pow(set, size)
            }
            "cartprod" if !call.named.is_empty() && call.args.is_empty() => {
                let items = call
                    .named
                    .iter()
                    .map(|n| {
                        let set = self.expr(n.value);
                        Math::relation(Math::ident(self.m.resolve(n.name), None), Rel::In, set)
                    })
                    .collect();
                Math::brace(items)
            }
            "cartprod" if !call.args.is_empty() && call.named.is_empty() => {
                let sets = self.args(call);
                chain(BinOp::Times, sets).unwrap_or_else(|| Math::call(head, vec![]))
            }
            "stdsimplex" if call.args.len() == 1 => {
                let n = match self.m.node(call.args[0]) {
                    Node::Lit(Scalar::Int(n)) => Math::int(n - 1),
                    _ => Math::minus(self.expr(call.args[0]), Math::int(1)),
                };
                Math::pow(Math::Sym(Sym::Simplex), n)
            }
            // ── access ───────────────────────────────────────────────────
            "get" if call.args.len() >= 2 => self.get(call),
            // ── reification and laws ─────────────────────────────────────
            "functionof" | "kernelof" if !call.args.is_empty() => {
                self.reification_expr(id, call, head)
            }
            "lawof" if call.args.len() == 1 && call.named.is_empty() => {
                let items = self.law_items(call.args[0]);
                Math::apply(Math::Sym(Sym::Law), vec![Math::row(items)])
            }
            "likelihoodof" if call.args.len() == 2 && call.named.is_empty() => {
                let inputs = self.callable_inputs(id).unwrap_or_default();
                self.likelihood(call.args[0], call.args[1], &inputs)
            }
            "disintegrate" if call.args.len() == 2 && self.is_string_vector(call.args[0]) => {
                let selector = self
                    .string_vector(call.args[0])
                    .iter()
                    .map(|s| Math::ident(s, None))
                    .collect();
                let m = self.expr(call.args[1]);
                Math::apply(
                    Math::subscript(
                        Math::text("disintegrate"),
                        Math::row(intersperse_commas(selector)),
                    ),
                    vec![m],
                )
            }
            // ── measure algebra with node-level structure ────────────────
            "restrict" if call.args.len() == 2 && call.named.is_empty() => {
                let m = self.expr(call.args[0]);
                let cond = match self.m.node(call.args[1]) {
                    Node::Call(c) if self.head_is(c, "record") => {
                        Math::row(intersperse_commas(self.labelled(c)))
                    }
                    _ => self.expr(call.args[1]),
                };
                conditional(m, cond)
            }
            "restrict" if call.args.len() == 1 && !call.named.is_empty() => {
                let m = self.expr(call.args[0]);
                let cond = Math::row(intersperse_commas(self.labelled(call)));
                conditional(m, cond)
            }
            "joint" => self.joint(call),
            "iid" if call.args.len() == 2 && call.named.is_empty() => {
                let m = self.expr(call.args[0]);
                let n = self.size_exponent(call.args[1]);
                Math::pow(m, n)
            }
            "relabel" if call.args.len() == 2 && self.is_string_vector(call.args[1]) => {
                let m = self.expr(call.args[0]);
                let labels = self.string_vector(call.args[1]);
                let ds = labels
                    .iter()
                    .map(|l| Math::row(vec![Math::Op(Op::Differential), Math::ident(l, None)]))
                    .collect();
                Math::row(vec![unit(m), Math::paren(ds)])
            }
            // ── collections ──────────────────────────────────────────────
            "broadcast" if !call.args.is_empty() => self.broadcast(id, call, None),
            "sum" | "prod" | "maximum" | "minimum"
                if call.args.len() == 1 && call.named.is_empty() =>
            {
                let op = match head {
                    "sum" => BigOp::Sum,
                    "prod" => BigOp::Prod,
                    "maximum" => BigOp::Max,
                    _ => BigOp::Min,
                };
                let i = self.fresh_index();
                let body = self.indexed(call.args[0], &i);
                Math::big(op, Some(i), None, body)
            }
            "aggregate" | "metricsum" if call.args.len() == 3 => {
                let agg = self.aggregation(call, head);
                if agg.out_axes.is_empty() {
                    agg.body
                } else {
                    let index = Math::row(
                        agg.out_axes
                            .iter()
                            .map(|(n, _)| Math::ident(n, None))
                            .collect(),
                    );
                    Math::family(agg.body, index, None)
                }
            }
            "fchain" if call.args.len() >= 2 && call.named.is_empty() => {
                let mut fs = self.args(call);
                fs.reverse();
                chain(BinOp::Compose, fs).expect("fchain has two or more operands")
            }
            // ── everything with a math-level rendering ───────────────────
            _ => {
                let args = self.ordered_args(head, call);
                // A keyword spelling on a head with no declared parameter
                // order keeps its construct name: the glyph forms apply to
                // positional operands only.
                if !call.named.is_empty() && param_names(head).is_none() {
                    return Math::call(head, args);
                }
                apply_builtin(head, args)
            }
        }
    }

    /// `callee(args, name = value)` for a user-defined or module callable. A
    /// standard-module distribution's keyword arguments follow its declared
    /// parameter order, like a base distribution's.
    fn user_call(&mut self, call: &Call) -> Math {
        let CallHead::User(callee) = call.head else {
            unreachable!("user_call is reached for CallHead::User only");
        };
        let head = self.expr(callee);
        let args = self.args(call);
        let kwargs = self.kwargs(call);
        let order = self.standard_module_params(callee);
        Math::apply(head, order_kwargs(order, args, kwargs))
    }

    /// The declared parameter names of `callee` when it is a member of a
    /// standard module (`hep.CrystalBall`), from the catalogue.
    fn standard_module_params(&self, callee: NodeId) -> Option<Vec<String>> {
        let Node::Ref(r) = self.m.node(callee) else {
            return None;
        };
        let RefNs::Module(alias) = r.ns else {
            return None;
        };
        let module = self.standard_module_name(alias)?;
        flatppl_infer::builtin_catalogue()
            .module_param_names(module, self.m.resolve(r.name))
            .map(|names| names.to_vec())
    }

    /// The name of the standard module bound to `alias`
    /// (`h = standard_module("particle-physics", …)` → `particle-physics`);
    /// `None` for a loaded module.
    fn standard_module_name(&self, alias: Symbol) -> Option<&'m str> {
        let alias_binding = self.m.binding_by_name(alias)?;
        let Node::Call(load) = self.m.node(self.m.binding(alias_binding).rhs) else {
            return None;
        };
        if !self.head_is(load, "standard_module") {
            return None;
        }
        match self.m.node(*load.args.first()?) {
            Node::Lit(Scalar::Str(module)) => Some(&**module),
            _ => None,
        }
    }

    /// `joint`: `⊗` of the components when they provably share no stochastic
    /// ancestor (spec §06 "Joint composition"), the construct name otherwise.
    fn joint(&mut self, call: &Call) -> Math {
        let components: Vec<NodeId> = call
            .args
            .iter()
            .copied()
            .chain(call.named.iter().map(|n| n.value))
            .collect();
        // A spelling that mixes positional and keyword components is not a
        // §06 form; it keeps the construct name with every component.
        let mixed = !call.args.is_empty() && !call.named.is_empty();
        let independent = !components.is_empty()
            && !mixed
            && components.iter().all(|c| self.independent_component(*c));
        if !independent {
            let mut args = self.args(call);
            args.extend(self.labelled(call));
            return Math::call("joint", args);
        }
        if !call.named.is_empty() {
            let factors = call
                .named
                .iter()
                .map(|n| {
                    let m = self.expr(n.value);
                    let label = Math::ident(self.m.resolve(n.name), None);
                    Math::row(vec![
                        unit(m),
                        Math::paren(vec![Math::row(vec![Math::Op(Op::Differential), label])]),
                    ])
                })
                .collect();
            return chain(BinOp::Otimes, factors).expect("joint has a component");
        }
        let args = self.args(call);
        chain(BinOp::Otimes, args).expect("joint has a component")
    }

    /// A `joint` component contributes a fresh coordinate with no shared
    /// stochastic node when it is not of stochastic phase and its subtree
    /// (through this module's bindings) holds no reification or draw — the
    /// two channels §06 names for sharing. A reference into a loaded module
    /// may reach a draw this module cannot see, so it counts as sharing; a
    /// standard module is a catalogue of pure functions and distributions and
    /// holds none.
    fn independent_component(&self, component: NodeId) -> bool {
        if self.m.phase_of(component) == Some(Phase::Stochastic) {
            return false;
        }
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut stack = vec![component];
        while let Some(node) = stack.pop() {
            if !visited.insert(node) {
                continue;
            }
            match self.m.node(node) {
                Node::Call(c) => {
                    if matches!(
                        self.head_name(c),
                        Some("lawof" | "kernelof" | "functionof" | "draw" | "rand")
                    ) {
                        return false;
                    }
                    c.args.iter().for_each(|a| stack.push(*a));
                    c.named.iter().for_each(|n| stack.push(n.value));
                    if let CallHead::User(callee) = c.head {
                        stack.push(callee);
                    }
                }
                Node::Ref(r) => match r.ns {
                    RefNs::SelfMod => {
                        if let Some(b) = self.m.binding_by_name(r.name) {
                            stack.push(self.m.binding(b).rhs);
                        }
                    }
                    RefNs::Module(alias) => {
                        if self.standard_module_name(alias).is_none() {
                            return false;
                        }
                    }
                    RefNs::Local => {}
                },
                _ => {}
            }
        }
        true
    }

    /// `rowstack([[…], […]])` / `colstack` as a matrix when the argument is a
    /// literal vector of vectors.
    fn stack(&mut self, head: &str, arg: NodeId) -> Math {
        let rows: Option<Vec<Vec<Math>>> = match self.m.node(arg) {
            Node::Call(outer) if self.head_is(outer, "vector") => outer
                .args
                .iter()
                .map(|r| match self.m.node(*r) {
                    Node::Call(inner) if self.head_is(inner, "vector") => {
                        let inner = inner.clone();
                        Some(self.args(&inner))
                    }
                    _ => None,
                })
                .collect(),
            _ => None,
        };
        match rows {
            Some(rows) if head == "rowstack" => Math::Matrix(rows),
            Some(cols) if !cols.is_empty() && cols.iter().all(|c| c.len() == cols[0].len()) => {
                let n = cols[0].len();
                let rows = (0..n)
                    .map(|i| cols.iter().map(|c| c[i].clone()).collect())
                    .collect();
                Math::Matrix(rows)
            }
            _ => {
                let a = self.expr(arg);
                Math::call(head, vec![a])
            }
        }
    }

    /// The exponent of `cartpow(S, size)` / `iid(M, size)`: `n`, or `m×n`
    /// for a vector of sizes.
    fn size_exponent(&mut self, size: NodeId) -> Math {
        if let Node::Call(c) = self.m.node(size)
            && self.head_is(c, "vector")
            && c.args.len() >= 2
        {
            let c = c.clone();
            let dims = self.args(&c);
            let mut items = Vec::new();
            for (i, d) in dims.into_iter().enumerate() {
                if i > 0 {
                    items.push(Math::Op(Op::Times));
                }
                items.push(d);
            }
            return Math::row(items);
        }
        self.expr(size)
    }

    fn is_string_vector(&self, id: NodeId) -> bool {
        match self.m.node(id) {
            Node::Call(c) if self.head_is(c, "vector") && !c.args.is_empty() => c
                .args
                .iter()
                .all(|a| matches!(self.m.node(*a), Node::Lit(Scalar::Str(_)))),
            _ => false,
        }
    }

    fn string_vector(&self, id: NodeId) -> Vec<String> {
        match self.m.node(id) {
            Node::Call(c) => c
                .args
                .iter()
                .filter_map(|a| match self.m.node(*a) {
                    Node::Lit(Scalar::Str(s)) => Some(s.to_string()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    // ── get ────────────────────────────────────────────────────────────────

    fn get(&mut self, call: &Call) -> Math {
        let base = self.expr(call.args[0]);
        let selectors = &call.args[1..];
        // Field access `r.a`.
        if selectors.len() == 1
            && let Node::Lit(Scalar::Str(field)) = self.m.node(selectors[0])
        {
            return Math::row(vec![
                unit(base),
                Math::Op(Op::Dot),
                Math::ident(field, None),
            ]);
        }
        // Subset selectors: a vector of names or indices.
        if selectors.len() == 1 && self.is_string_vector(selectors[0]) {
            let names = self
                .string_vector(selectors[0])
                .iter()
                .map(|n| Math::ident(n, None))
                .collect();
            return Math::subscript(base, Math::brace(names));
        }
        // Axis-labelled access (aggregate / metricsum): uppers as superscript,
        // lowers and neutral axes as subscript, no separators.
        let axes: Vec<Option<Option<Variance>>> = selectors
            .iter()
            .map(|s| match self.m.node(*s) {
                Node::Axis(a) => Some(a.variance),
                _ => None,
            })
            .collect();
        if axes.iter().all(Option::is_some) {
            let mut upper = Vec::new();
            let mut lower = Vec::new();
            for (s, v) in selectors.iter().zip(axes) {
                let ident = self.expr(*s);
                match v.flatten() {
                    Some(Variance::Upper) => upper.push(ident),
                    _ => lower.push(ident),
                }
            }
            return match (lower.is_empty(), upper.is_empty()) {
                (false, true) => Math::subscript(base, Math::row(lower)),
                (true, false) => Math::pow(base, Math::row(upper)),
                (false, false) => Math::SubSup(
                    Box::new(base),
                    Box::new(Math::row(lower)),
                    Box::new(Math::row(upper)),
                ),
                (true, true) => base,
            };
        }
        // Ordinary indices, comma-separated; `only` selects the sole element.
        let mut parts = Vec::new();
        for s in selectors {
            if let Node::Const(c) = self.m.node(*s)
                && self.m.resolve(*c) == "only"
            {
                continue;
            }
            parts.push(self.expr(*s));
        }
        if parts.is_empty() {
            return base;
        }
        Math::subscript(base, Math::row(intersperse_commas(parts)))
    }

    // ── broadcasting and indexing ──────────────────────────────────────────

    fn fresh_index(&mut self) -> Math {
        for c in [
            'i', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
        ] {
            let s = c.to_string();
            if self.bound.contains(&s) || self.indices.contains(&s) || self.scope_has(&s) {
                continue;
            }
            self.indices.push(s);
            return Math::ident(&c.to_string(), None);
        }
        // Fourteen letters exhausted: number them.
        let s = format!("i{}", self.indices.len());
        self.indices.push(s.clone());
        Math::ident(&s, None)
    }

    fn scope_has(&self, name: &str) -> bool {
        self.scopes.iter().any(|s| s.iter().any(|p| p.name == name))
    }

    fn is_collection(&self, id: NodeId) -> bool {
        match self.m.type_of(id) {
            Some(Type::Array { .. } | Type::TVector { .. } | Type::Table { .. }) => true,
            Some(_) => false,
            // Untyped: a literal vector is a collection by syntax.
            None => matches!(self.m.node(id), Node::Call(c) if self.head_is(c, "vector")),
        }
    }

    /// The static length of a rank-1 collection, as math: a named `iid` size
    /// where the source gives one, else the inferred static dimension.
    fn collection_len(&mut self, id: NodeId) -> Option<Math> {
        if let Node::Ref(r) = self.m.node(id)
            && r.ns == RefNs::SelfMod
            && let Some(b) = self.m.binding_by_name(r.name)
            && let Some(size) = self.iid_size(self.m.binding(b).rhs)
        {
            return Some(self.expr(size));
        }
        match self.m.type_of(id) {
            Some(Type::Array { shape, .. }) if shape.len() == 1 => match shape[0] {
                Dim::Static(n) => Some(Math::int(n as i64)),
                Dim::Dynamic => None,
            },
            Some(Type::TVector {
                len: Dim::Static(n),
                ..
            }) => Some(Math::int(*n as i64)),
            Some(Type::Table {
                nrows: Dim::Static(n),
                ..
            }) => Some(Math::int(*n as i64)),
            _ => match self.m.node(id) {
                Node::Call(c) if self.head_is(c, "vector") => Some(Math::int(c.args.len() as i64)),
                _ => None,
            },
        }
    }

    /// The scalar size of an `iid(M, n)` at `node`, also through a `draw`
    /// (`x ~ iid(M, n)`); `None` for a multi-axis size or any other shape.
    fn iid_size(&self, node: NodeId) -> Option<NodeId> {
        let Node::Call(c) = self.m.node(node) else {
            return None;
        };
        if self.head_is(c, "draw") && c.args.len() == 1 {
            return self.iid_size(c.args[0]);
        }
        if self.head_is(c, "iid")
            && c.args.len() == 2
            && !matches!(self.m.node(c.args[1]), Node::Call(v) if self.head_is(v, "vector"))
        {
            return Some(c.args[1]);
        }
        None
    }

    /// `collection` at index `i`: a nested dotted expression shares the index,
    /// a gather `a[idx]` becomes `a_{idx_i}`, anything else is subscripted.
    fn indexed(&mut self, collection: NodeId, i: &Math) -> Math {
        if let Node::Call(c) = self.m.node(collection) {
            if self.head_is(c, "broadcast") && !c.args.is_empty() {
                let c = c.clone();
                return self.broadcast(collection, &c, Some(i));
            }
            if self.head_is(c, "get") && c.args.len() == 2 && self.is_collection(c.args[1]) {
                let base = self.expr(c.args[0]);
                let idx = self.indexed(c.args[1], i);
                return Math::subscript(base, idx);
            }
        }
        let whole = self.expr(collection);
        Math::subscript(whole, i.clone())
    }

    /// `broadcast(f, args…)`: the elementwise body under a fresh index, wrapped
    /// as an independent product (kernel) or a family (function). With
    /// `shared` set, only the body is returned (a nested dotted expression).
    fn broadcast(&mut self, node: NodeId, call: &Call, shared: Option<&Math>) -> Math {
        let head = call.args[0];
        let rest: Vec<NodeId> = call.args[1..].to_vec();
        let any_typed = rest
            .iter()
            .chain(call.named.iter().map(|n| &n.value))
            .any(|a| {
                self.m.type_of(*a).is_some()
                    || matches!(self.m.node(*a), Node::Call(c) if self.head_is(c, "vector"))
            });
        if !any_typed {
            let mut args = vec![self.expr(head)];
            args.extend(self.args_from(&rest));
            args.extend(self.labelled(call));
            return Math::call("broadcast", args);
        }
        let index = match shared {
            Some(i) => i.clone(),
            None => self.fresh_index(),
        };
        let mut range_hi = None;
        let mut args = Vec::new();
        for a in &rest {
            if self.is_collection(*a) {
                if range_hi.is_none() {
                    range_hi = self.collection_len(*a);
                }
                args.push(self.indexed(*a, &index));
            } else {
                args.push(self.expr(*a));
            }
        }
        let mut kwargs = Vec::new();
        for n in call.named.iter() {
            let name = self.m.resolve(n.name).to_string();
            let v = if self.is_collection(n.value) {
                if range_hi.is_none() {
                    range_hi = self.collection_len(n.value);
                }
                self.indexed(n.value, &index)
            } else {
                self.expr(n.value)
            };
            kwargs.push((name, v));
        }
        let body = self.apply_head(head, args, kwargs);
        if shared.is_some() {
            return body;
        }
        let is_measure = match self.m.type_of(node) {
            Some(Type::Measure { .. }) => true,
            Some(_) => false,
            None => {
                matches!(self.m.node(head), Node::Const(s) if self.is_measure_head(self.m.resolve(*s)))
            }
        };
        let range = range_hi.map(|hi| (Math::int(1), hi));
        if is_measure {
            let (sub, sup) = match range {
                Some((lo, hi)) => (Some(Math::relation(index, Rel::Eq, lo)), Some(hi)),
                None => (Some(index), None),
            };
            Math::big(BigOp::Otimes, sub, sup, body)
        } else {
            Math::family(body, index, range)
        }
    }

    fn args_from(&mut self, ids: &[NodeId]) -> Vec<Math> {
        ids.iter().map(|a| self.expr(*a)).collect()
    }

    /// Apply a broadcast head to already-lowered arguments: a builtin by its
    /// math rendering, a user callable by application.
    fn apply_head(&mut self, head: NodeId, args: Vec<Math>, kwargs: Vec<(String, Math)>) -> Math {
        if let Node::Const(sym) = self.m.node(head) {
            let name = self.m.resolve(*sym);
            let keyword_form = !kwargs.is_empty() && param_names(name).is_none();
            let args = order_kwargs(param_names(name), args, kwargs);
            if keyword_form {
                return Math::call(name, args);
            }
            return apply_builtin(name, args);
        }
        let f = self.expr(head);
        let order = self.standard_module_params(head);
        Math::apply(f, order_kwargs(order, args, kwargs))
    }

    // ── aggregation ────────────────────────────────────────────────────────

    /// `aggregate(f, [out axes], body)` / `metricsum(g, [out axes], body)`:
    /// the output axes, the reduced body, and an annotation.
    fn aggregation(&mut self, call: &Call, head: &str) -> Aggregation {
        let out_axes = self.axis_list(call.args[1]);
        let body_node = call.args[2];
        let body = self.expr(body_node);
        let body_axes = self.collect_axes(body_node);
        if head == "metricsum" {
            let metric = self.expr(call.args[0]);
            return Aggregation {
                out_axes,
                body,
                annotation: Some(format!(
                    "indices lowered with the metric {}",
                    plain_name(&metric)
                )),
            };
        }
        let reduced: Vec<Math> = body_axes
            .iter()
            .filter(|a| !out_axes.iter().any(|(o, _)| o == *a))
            .map(|a| Math::ident(a, None))
            .collect();
        let op = match self.m.node(call.args[0]) {
            Node::Const(s) => reduction_op(self.m.resolve(*s)),
            _ => BigOp::Named("aggregate"),
        };
        let body = if reduced.is_empty() {
            body
        } else {
            Math::big(op, Some(Math::row(intersperse_commas(reduced))), None, body)
        };
        Aggregation {
            out_axes,
            body,
            annotation: None,
        }
    }

    fn axis_list(&self, id: NodeId) -> Vec<(String, Option<Variance>)> {
        match self.m.node(id) {
            Node::Call(c) if self.head_is(c, "vector") => c
                .args
                .iter()
                .filter_map(|a| match self.m.node(*a) {
                    Node::Axis(ax) => Some((self.m.resolve(ax.name).to_string(), ax.variance)),
                    _ => None,
                })
                .collect(),
            Node::Axis(ax) => vec![(self.m.resolve(ax.name).to_string(), ax.variance)],
            _ => Vec::new(),
        }
    }

    /// Axis names occurring in `id`, in first-appearance order.
    fn collect_axes(&self, id: NodeId) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut stack = vec![id];
        while let Some(node) = stack.pop() {
            if let Node::Axis(ax) = self.m.node(node) {
                let name = self.m.resolve(ax.name).to_string();
                if !out.contains(&name) {
                    out.push(name);
                }
                continue;
            }
            // Children in source order: push them reversed.
            let mut children = Vec::new();
            self.m.for_each_child(node, |c| children.push(c));
            stack.extend(children.into_iter().rev());
        }
        out
    }
}

/// The pieces of a lowered `aggregate` / `metricsum`.
struct Aggregation {
    out_axes: Vec<(String, Option<Variance>)>,
    body: Math,
    annotation: Option<String>,
}

/// Whether the expression at `id` nests deeper than
/// [`DEFAULT_MAX_DEPTH`] (iterative, so the check itself cannot overflow).
/// A reference to a synthetic binding counts the binding's right-hand side,
/// which the lowering inlines at that point.
fn too_deep(m: &Module, id: NodeId) -> bool {
    let mut stack = vec![(id, 0usize)];
    while let Some((node, depth)) = stack.pop() {
        if depth > DEFAULT_MAX_DEPTH {
            return true;
        }
        if let Node::Ref(r) = m.node(node)
            && r.ns == RefNs::SelfMod
            && let Some(b) = m.binding_by_name(r.name)
            && m.binding(b).synthetic
        {
            stack.push((m.binding(b).rhs, depth + 1));
            continue;
        }
        m.for_each_child(node, |c| stack.push((c, depth + 1)));
    }
    false
}

fn too_deep_message(name: &str) -> String {
    format!(
        "`{name}` nests deeper than {DEFAULT_MAX_DEPTH} levels of mathematics; shown as source text"
    )
}

/// `lhs` decorated with the output axes: lowers and neutral axes as a
/// subscript, uppers as a superscript.
fn decorate_with_axes(lhs: Math, out_axes: &[(String, Option<Variance>)]) -> Math {
    let (lower, upper): (Vec<_>, Vec<_>) = out_axes
        .iter()
        .partition(|(_, v)| !matches!(v, Some(Variance::Upper)));
    let lower: Vec<Math> = lower
        .into_iter()
        .map(|(n, _)| Math::ident(n, None))
        .collect();
    let upper: Vec<Math> = upper
        .into_iter()
        .map(|(n, _)| Math::ident(n, None))
        .collect();
    match (lower.is_empty(), upper.is_empty()) {
        (false, true) => Math::subscript(lhs, Math::row(lower)),
        (true, false) => Math::pow(lhs, Math::row(upper)),
        (false, false) => Math::SubSup(
            Box::new(lhs),
            Box::new(Math::row(lower)),
            Box::new(Math::row(upper)),
        ),
        (true, true) => lhs,
    }
}

/// The big-operator form of an `aggregate` reduction (spec §04 lists the
/// eligible reductions).
fn reduction_op(name: &str) -> BigOp {
    match name {
        "sum" => BigOp::Sum,
        "prod" => BigOp::Prod,
        "maximum" => BigOp::Max,
        "minimum" => BigOp::Min,
        "mean" => BigOp::Named("mean"),
        "var" => BigOp::Named("var"),
        "std" => BigOp::Named("std"),
        "median" => BigOp::Named("median"),
        "lany" => BigOp::Named("lany"),
        "lall" => BigOp::Named("lall"),
        _ => BigOp::Named("aggregate"),
    }
}

/// The declared parameter names of a builtin: §08/§06 constructors first,
/// then the §07 catalogue rows that carry names.
fn param_names(name: &str) -> Option<Vec<String>> {
    flatppl_infer::constructor_param_names(name).or_else(|| {
        flatppl_infer::builtin_catalogue()
            .base_param_names(name)
            .map(|n| n.to_vec())
    })
}

/// Positional `args` followed by `kwargs` moved into their declared
/// positions. When every declared slot up to the last supplied one is
/// filled the result is purely positional; otherwise — a missing argument, an
/// undeclared name, or no declared order at all — the keyword arguments stay
/// labelled `name = value`, so a misspelt or missing argument never shifts
/// into another parameter's place.
pub fn order_kwargs(
    order: Option<Vec<String>>,
    mut args: Vec<Math>,
    kwargs: Vec<(String, Math)>,
) -> Vec<Math> {
    if kwargs.is_empty() {
        return args;
    }
    let labelled = |args: &mut Vec<Math>, kwargs: Vec<(String, Math)>| {
        args.extend(
            kwargs
                .into_iter()
                .map(|(k, v)| Math::relation(Math::ident(&k, None), Rel::Eq, v)),
        );
    };
    let Some(order) = order else {
        labelled(&mut args, kwargs);
        return args;
    };
    if args.len() > order.len() {
        labelled(&mut args, kwargs);
        return args;
    }
    let mut slots: Vec<Option<Math>> = vec![None; order.len()];
    for (i, a) in args.iter().enumerate() {
        slots[i] = Some(a.clone());
    }
    for (k, v) in &kwargs {
        match order.iter().position(|p| p == k) {
            Some(i) if slots[i].is_none() => slots[i] = Some(v.clone()),
            _ => {
                labelled(&mut args, kwargs);
                return args;
            }
        }
    }
    let last = slots.iter().rposition(Option::is_some).map_or(0, |i| i + 1);
    if slots[..last].iter().any(Option::is_none) {
        labelled(&mut args, kwargs);
        return args;
    }
    slots.into_iter().flatten().collect()
}

/// A builtin applied to lowered arguments: operators and the functions with a
/// standard glyph get it; every other builtin is `name(args)` in roman. Used
/// by direct calls and by the elementwise body of a broadcast alike.
pub fn apply_builtin(name: &str, mut args: Vec<Math>) -> Math {
    // Unresolved keyword slots keep their names, not a guessed parameter role.
    if matches!(
        name,
        "Normal"
            | "MvNormal"
            | "StudentT"
            | "ChiSquared"
            | "Uniform"
            | "Gamma"
            | "Exponential"
            | "InverseGamma"
            | "Weibull"
    ) && args
        .iter()
        .any(|a| matches!(a, Math::Relation { rel: Rel::Eq, .. }))
    {
        return Math::call(name, args);
    }
    let n = args.len();
    let mut take = |k: usize| std::mem::replace(&mut args[k], Math::Num(String::new()));
    match (name, n) {
        // §08 takes a standard deviation. Show its square without folding,
        // including literal scales, so the displayed parameterisation is clear.
        ("Normal", 2) => Math::apply(
            Math::Sym(Sym::Normal),
            vec![take(0), Math::pow(take(1), Math::int(2))],
        ),
        ("MvNormal", 2) => Math::apply(Math::Sym(Sym::Normal), args),
        ("StudentT", 1) => Math::subscript(Math::letter('t'), take(0)),
        ("ChiSquared", 1) => Math::SubSup(
            Box::new(Math::letter('χ')),
            Box::new(take(0)),
            Box::new(Math::int(2)),
        ),
        ("Uniform", 1) => Math::apply(Math::Sym(Sym::Uniform), args),
        ("Exponential", 1) => Math::call("Exp", args),
        ("add", 2) => Math::plus(take(0), take(1)),
        ("sub", 2) => Math::minus(take(0), take(1)),
        ("mul", 2) => Math::times(take(0), take(1)),
        ("divide", 2) => Math::frac(take(0), take(1)),
        ("neg", 1) => Math::negate(take(0)),
        ("pow", 2) => Math::pow(take(0), take(1)),
        ("sqrt", 1) => Math::sqrt(take(0)),
        ("abs", 1) => Math::abs(take(0)),
        ("abs2", 1) => Math::pow(Math::abs(take(0)), Math::int(2)),
        ("floor", 1) => fenced(Fence::Floor, take(0)),
        ("ceil", 1) => fenced(Fence::Ceil, take(0)),
        ("equal", 2) => Math::relation(take(0), Rel::Eq, take(1)),
        ("unequal", 2) => Math::relation(take(0), Rel::Ne, take(1)),
        ("lt", 2) => Math::relation(take(0), Rel::Lt, take(1)),
        ("le", 2) => Math::relation(take(0), Rel::Le, take(1)),
        ("gt", 2) => Math::relation(take(0), Rel::Gt, take(1)),
        ("ge", 2) => Math::relation(take(0), Rel::Ge, take(1)),
        ("in", 2) => Math::relation(take(0), Rel::In, take(1)),
        ("land", 2) => Math::binary(BinOp::And, take(0), take(1)),
        ("lor", 2) => Math::binary(BinOp::Or, take(0), take(1)),
        ("lnot", 1) => Math::Unary {
            op: crate::ast::UnOp::Not,
            arg: Box::new(take(0)),
        },
        ("identity", 1) => take(0),
        ("transpose", 1) => Math::pow(take(0), Math::Op(Op::Transpose)),
        ("adjoint", 1) => Math::pow(take(0), Math::Op(Op::Dagger)),
        ("inv", 1) => Math::pow(take(0), Math::int(-1)),
        ("det", 1) => Math::call("det", vec![take(0)]),
        ("trace", 1) => Math::call("tr", vec![take(0)]),
        ("eye", 1) => Math::subscript(Math::letter('I'), take(0)),
        ("onehot", 2) => Math::subscript(Math::letter('e'), take(0)),
        ("diagmat", 1) => Math::call("diag", vec![take(0)]),
        ("quadform", 2) => {
            let a = take(0);
            let x = take(1);
            Math::times(
                Math::times(Math::pow(x.clone(), Math::Op(Op::Dagger)), a),
                x,
            )
        }
        ("cross", 2) => Math::binary(BinOp::Times, take(0), take(1)),
        ("l1norm", 1) => norm(take(0), Math::int(1)),
        ("l2norm", 1) => norm(take(0), Math::int(2)),
        ("linfnorm", 1) => norm(take(0), Math::Sym(Sym::Infinity)),
        ("min", 2) => Math::call("min", vec![take(0), take(1)]),
        ("max", 2) => Math::call("max", vec![take(0), take(1)]),
        ("ifelse", 3) => Math::Cases(vec![(take(1), Some(take(0))), (take(2), None)]),
        ("complex", 2) => Math::plus(take(0), Math::times(take(1), Math::Sym(Sym::ImagUnit))),
        ("cis", 1) => Math::pow(Math::Sym(Sym::Euler), imag_times(take(0))),
        ("conj", 1) => Math::Overline(Box::new(take(0))),
        ("real", 1) => Math::call("Re", vec![take(0)]),
        ("imag", 1) => Math::call("Im", vec![take(0)]),
        ("interval", 2) => Math::bracket(vec![take(0), take(1)]),
        // ── measure algebra (§06) ───────────────────────────────────────
        ("weighted", 2) => Math::dot(take(0), take(1)),
        ("logweighted", 2) => Math::dot(Math::pow(Math::Sym(Sym::Euler), take(0)), take(1)),
        ("superpose", _) if n > 0 => chain(BinOp::Add, args).expect("one or more operands"),
        ("truncate", 2) => restrict(take(0), take(1)),
        ("iid", 2) => Math::pow(take(0), take(1)),
        ("locscale", 3) => {
            let m = take(0);
            let shift = take(1);
            let scale = take(2);
            Math::plus(shift, Math::dot(scale, m))
        }
        ("pushfwd", 2) => Math::binary(
            BinOp::Juxtapose,
            Math::subscript(take(0), Math::Op(Op::Star)),
            take(1),
        ),
        ("Lebesgue", 0) => Math::Sym(Sym::Lebesgue),
        ("Lebesgue", 1) => match take(0) {
            Math::Sym(Sym::Reals | Sym::ExtendedReals) => Math::Sym(Sym::Lebesgue),
            s => Math::subscript(Math::Sym(Sym::Lebesgue), s),
        },
        ("Dirac", 1) => Math::subscript(Math::Sym(Sym::Dirac), take(0)),
        ("lawof", 1) => Math::apply(Math::Sym(Sym::Law), vec![take(0)]),
        ("densityof", 2) => Math::apply(Math::subscript(Math::letter('p'), take(0)), vec![take(1)]),
        ("logdensityof", 2) => Math::row(vec![
            Math::text("log"),
            Math::apply(Math::subscript(Math::letter('p'), take(0)), vec![take(1)]),
        ]),
        ("bayesupdate", 2) => Math::dot(take(0), take(1)),
        ("joint_likelihood", _) if n > 0 => chain(BinOp::Dot, args).expect("one or more operands"),
        _ => Math::call(name, args),
    }
}

/// `i t` for a symbolic `t`, `t i` for a numeric one (`e^{iθ}`, `e^{0.7 i}`).
fn imag_times(t: Math) -> Math {
    if matches!(t, Math::Num(_)) {
        Math::times(t, Math::Sym(Sym::ImagUnit))
    } else {
        Math::times(Math::Sym(Sym::ImagUnit), t)
    }
}

fn fenced(fence: Fence, item: Math) -> Math {
    Math::Fenced {
        open: fence,
        close: fence,
        items: vec![item],
    }
}

fn norm(v: Math, which: Math) -> Math {
    Math::subscript(fenced(Fence::Norm, v), which)
}

/// `m` as a single unit: bracketed unless it already reads as one.
fn unit(m: Math) -> Math {
    if m.prec() < 9 {
        Math::paren(vec![m])
    } else {
        m
    }
}

/// `M|_S` — `truncate`, the support restriction.
fn restrict(m: Math, s: Math) -> Math {
    Math::subscript(Math::row(vec![unit(m), Math::Op(Op::Restrict)]), s)
}

/// `M(· | a = v)` — `restrict`, the unnormalised conditional measure.
fn conditional(m: Math, cond: Math) -> Math {
    Math::apply(
        unit(m),
        vec![Math::row(vec![
            Math::Sym(Sym::Placeholder),
            Math::Op(Op::Bar),
            cond,
        ])],
    )
}

/// `(0, ∞]` / `[0, ∞]`: the §03 positive and non-negative half-lines.
fn half_line(open: Fence, lo: Math) -> Math {
    Math::Fenced {
        open,
        close: Fence::Bracket,
        items: vec![lo, Math::Sym(Sym::Infinity)],
    }
}

/// Left-fold `items` under `op`; a single item is returned as itself, no
/// items as `None`.
fn chain(op: BinOp, items: Vec<Math>) -> Option<Math> {
    let mut iter = items.into_iter();
    let first = iter.next()?;
    Some(iter.fold(first, |acc, x| Math::binary(op, acc, x)))
}

fn intersperse_commas(items: Vec<Math>) -> Vec<Math> {
    let mut out = Vec::with_capacity(items.len() * 2);
    for (i, item) in items.into_iter().enumerate() {
        if i > 0 {
            out.push(Math::Op(Op::Comma));
        }
        out.push(item);
    }
    out
}

fn eq(lhs: Math, rhs: Math) -> Statement {
    Statement {
        lhs,
        rel: Rel::Eq,
        rhs,
    }
}

/// The source name of an identifier, for annotations.
fn plain_name(m: &Math) -> String {
    match m {
        Math::Ident(id) => id.name.clone(),
        Math::Text(t) => t.clone(),
        _ => "g".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mathml;

    fn rows(src: &str) -> Vec<Row> {
        let mut module = flatppl_syntax::parse(src).expect("parses");
        flatppl_infer::infer(&mut module);
        lower_module_with(&module, Some(src))
    }

    /// As [`rows`], for a caller that has no source text (fallbacks print).
    fn rows_without_source(src: &str) -> Vec<Row> {
        let mut module = flatppl_syntax::parse(src).expect("parses");
        flatppl_infer::infer(&mut module);
        lower_module(&module)
    }

    fn row_named<'a>(rows: &'a [Row], name: &str) -> &'a Row {
        rows.iter()
            .find(|r| r.names.iter().any(|n| n == name))
            .unwrap_or_else(|| panic!("no row {name}"))
    }

    /// The MathML of a row's right-hand side.
    fn rhs(src: &str, name: &str) -> String {
        let rows = rows(src);
        mathml::expr(&row_named(&rows, name).statement.rhs)
    }

    #[test]
    fn declarations_draws_and_values_take_their_relation() {
        let rows = rows(
            "flatppl_compat = \"0.1\"\nmu = elementof(reals)\nc = external(posreals)\nx ~ Normal(mu, 1.0)\ny = 2 * x + 1",
        );
        assert_eq!(rows.len(), 4, "flatppl_compat has no row");
        let mu = row_named(&rows, "mu");
        assert_eq!(mu.statement.rel, Rel::In);
        assert_eq!(
            mathml::expr(&mu.statement.rhs),
            "<mover><mi>ℝ</mi><mo>¯</mo></mover>"
        );
        let c = row_named(&rows, "c");
        assert_eq!(c.annotation.as_deref(), Some("external input"));
        assert!(mathml::expr(&c.statement.rhs).starts_with(
            "<mrow><mo stretchy=\"false\">(</mo><mn>0</mn><mo>,</mo><mi>∞</mi><mo stretchy=\"false\">]</mo>"
        ));
        let x = row_named(&rows, "x");
        assert_eq!(x.statement.rel, Rel::Sim);
        assert_eq!(x.kind, Kind::Draw);
        assert_eq!(
            mathml::expr(&x.statement.rhs),
            "<mrow><mi>𝒩</mi><mo>&#x2061;</mo><mrow><mo stretchy=\"false\">(</mo><mi data-flatppl-ref=\"mu\">μ</mi><mo>,</mo><msup><mn>1</mn><mn>2</mn></msup><mo stretchy=\"false\">)</mo></mrow></mrow>"
        );
        let y = row_named(&rows, "y");
        assert_eq!(y.kind, Kind::Value);
        assert_eq!(y.statement.refs(), vec!["y", "x"]);
    }

    #[test]
    fn keyword_arguments_take_their_declared_positions_or_stay_labelled() {
        assert_eq!(
            rhs("d = Normal(sigma = 2.0, mu = 1.0)", "d"),
            "<mrow><mi>𝒩</mi><mo>&#x2061;</mo><mrow><mo stretchy=\"false\">(</mo><mn>1</mn><mo>,</mo><msup><mn>2</mn><mn>2</mn></msup><mo stretchy=\"false\">)</mo></mrow></mrow>"
        );
        assert!(
            rhs("d = Gamma(shape = 4.0, rate = 2.0)", "d")
                .contains("<mn>4</mn><mo>,</mo><mn>2</mn>")
        );
        // A missing argument never shifts into another slot.
        let d = rhs("d = Normal(sigma = 2.0)", "d");
        assert!(d.contains("<mi>σ</mi><mo>=</mo><mn>2</mn>"), "{d}");
        // A keyword spelling of a measure operator keeps the construct name.
        let p = rhs(
            "pr = Normal(0, 1)\nL = likelihoodof(functionof(Normal(0, 1)), 0.5)\npost = bayesupdate(prior = pr, L = L)",
            "post",
        );
        assert!(p.starts_with("<mrow><mi>bayesupdate</mi>"), "{p}");
        assert!(p.contains("<mi>prior</mi><mo>=</mo>"), "{p}");
    }

    #[test]
    fn measure_algebra_reads_as_measure_notation() {
        let tau = rhs(
            "tau ~ normalize(truncate(Cauchy(0, 5), interval(0, inf)))",
            "tau",
        );
        assert!(
            tau.starts_with(
                "<mrow><mi>normalize</mi><mo>&#x2061;</mo><mrow><mo>(</mo><msub><mrow>"
            )
        );
        assert!(tau.contains("<mo stretchy=\"false\">|</mo></mrow><mrow><mo stretchy=\"false\">[</mo><mn>0</mn><mo>,</mo><mi>∞</mi><mo stretchy=\"false\">]</mo></mrow></msub>"));
        let mix = rhs(
            "p = elementof(unitinterval)\nmix = normalize(superpose(weighted(p, Normal(0, 1)), weighted(1 - p, Gamma(2, 1))))",
            "mix",
        );
        assert!(mix.contains("<mi data-flatppl-ref=\"p\">p</mi><mo>⋅</mo>"));
        assert!(mix.contains(
            "<mo>+</mo><mrow><mrow><mo stretchy=\"false\">(</mo><mrow><mn>1</mn><mo>−</mo>"
        ));
        let pf = rhs("m = pushfwd(exp, Normal(0, 1))", "m");
        assert!(pf.starts_with("<mrow><msub><mi>exp</mi><mo>∗</mo></msub><mo>&#x2062;</mo>"));
        let ls = rhs(
            "nu = elementof(posreals)\nm = locscale(StudentT(nu), 0.0, 2.5)",
            "m",
        );
        assert!(ls.starts_with("<mrow><mn>0</mn><mo>+</mo><mrow><mn>2.5</mn><mo>⋅</mo>"));
        assert_eq!(rhs("l = Lebesgue(support = reals)", "l"), "<mi>λ</mi>");
        assert!(rhs("d = Dirac(0)", "d").starts_with("<msub><mi>δ</mi>"));
    }

    #[test]
    fn postfix_measure_operators_bracket_a_compound_operand() {
        let src = "M1 = Normal(0, 1)\nM2 = Normal(1, 2)\nt = truncate(superpose(M1, M2), interval(0, 1))\nr = relabel(superpose(M1, M2), [\"x\"])";
        let rows = rows(src);
        let t = mathml::expr(&row_named(&rows, "t").statement.rhs);
        assert!(
            t.starts_with("<msub><mrow><mrow><mo stretchy=\"false\">(</mo><mrow><msub data-flatppl-ref=\"M1\">"),
            "{t}"
        );
        let r = mathml::expr(&row_named(&rows, "r").statement.rhs);
        assert!(
            r.starts_with("<mrow><mrow><mo stretchy=\"false\">(</mo>"),
            "{r}"
        );
        assert!(
            r.contains("<mi mathvariant=\"normal\">d</mi><mi>x</mi>"),
            "{r}"
        );
    }

    #[test]
    fn joint_is_a_tensor_product_only_without_shared_stochastic_ancestry() {
        let independent = rhs("pr = joint(a = Normal(0, 1), b = Exponential(1))", "pr");
        assert!(independent.contains("<mo>⊗</mo>"), "{independent}");
        assert!(independent.contains("<mi mathvariant=\"normal\">d</mi><mi>a</mi>"));
        let positional = rhs("pr = joint(Normal(0, 1), Exponential(1))", "pr");
        assert!(positional.contains("<mo>⊗</mo>"), "{positional}");
        // Reified components share their ancestry (§06): no product claim.
        let shared = rhs(
            "z ~ Normal(0, 1)\na ~ Normal(z, 1)\nb ~ Normal(z, 2)\ndep = joint(a = lawof(a), b = lawof(b))",
            "dep",
        );
        assert!(shared.starts_with("<mrow><mi>joint</mi>"), "{shared}");
        assert!(!shared.contains("<mo>⊗</mo>"), "{shared}");
        // A constructor with a stochastic parameter is a sharing channel too.
        let param = rhs(
            "z ~ Normal(0, 1)\ndep = joint(a = Normal(z, 1), b = Normal(z, 2))",
            "dep",
        );
        assert!(param.starts_with("<mrow><mi>joint</mi>"), "{param}");
        // A mixed spelling keeps every component under the construct name.
        let mixed = rhs("m = joint(Normal(0, 1), b = Exponential(1))", "m");
        assert!(mixed.starts_with("<mrow><mi>joint</mi>"), "{mixed}");
        assert!(mixed.contains("<mi>𝒩</mi>") && mixed.contains("<mi>Exp</mi>"));
        // A standard-module distribution holds no draw: still a product.
        let std = rhs(
            "h = standard_module(\"particle-physics\", \"0.1\")\nm = joint(h.CrystalBall(5.0, 0.3, 1.5, 2.0), Normal(0, 1))",
            "m",
        );
        assert!(std.contains("<mo>⊗</mo>"), "{std}");
        // A kernel-typed joint is a product of its (independent) components.
        let kernel = rhs("K = joint(kernelof(Normal(0, 1)), Exponential(1))", "K");
        assert!(kernel.starts_with("<mrow><mi>joint</mi>"), "{kernel}");
    }

    #[test]
    fn restrict_reads_as_a_conditional_measure() {
        let src = "z ~ Normal(0, 1)\nobs ~ Normal(z, 1)\nm = lawof(record(z = z, obs = obs))\npost = restrict(m, record(obs = 2.1))";
        let post = rhs(src, "post");
        assert!(
            post.starts_with("<mrow><mi data-flatppl-ref=\"m\">m</mi><mo>&#x2061;</mo><mrow><mo stretchy=\"false\">(</mo><mrow><mo>·</mo><mo stretchy=\"false\">|</mo>"),
            "{post}"
        );
        assert!(
            post.contains("<mi>obs</mi><mo>=</mo><mn>2.1</mn>"),
            "{post}"
        );
    }

    #[test]
    fn iid_is_a_tensor_power_and_broadcasts_are_products_or_families() {
        let src = "J = 8\nmu ~ Normal(0, 5)\ntau = elementof(posreals)\ntheta ~ iid(Normal(mu, tau), J)\ns = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]\ny ~ Normal.(theta, s)\nalpha = elementof(reals)\nbeta = elementof(reals)\nx = [1.1, 1.5, 1.3, 1.4]\nmeans = alpha .+ beta .* x";
        let rows = rows(src);
        let theta = mathml::expr(&row_named(&rows, "theta").statement.rhs);
        // The n-fold product measure is the bare power `Mⁿ` (van der Vaart's
        // `Pⁿ`), not `M^{⊗n}`; a vector size is `M^{m×n}`.
        assert!(
            theta.ends_with("<mo stretchy=\"false\">)</mo></mrow></mrow><mi data-flatppl-ref=\"J\">J</mi></msup>"),
            "{theta}"
        );
        let grid = rhs("w ~ iid(Normal(0, 1), [2, 3])", "w");
        assert!(
            grid.ends_with("<mrow><mn>2</mn><mo>×</mo><mn>3</mn></mrow></msup>"),
            "{grid}"
        );
        let y = mathml::expr(&row_named(&rows, "y").statement.rhs);
        assert!(y.starts_with("<mrow><munderover><mo>⨂</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi data-flatppl-ref=\"J\">J</mi></munderover>"), "{y}");
        assert!(y.contains("<msub><mi data-flatppl-ref=\"theta\">θ</mi><mi>i</mi></msub><mo>,</mo><msup><msub><mi data-flatppl-ref=\"s\">s</mi><mi>i</mi></msub><mn>2</mn></msup>"));
        let means = mathml::expr(&row_named(&rows, "means").statement.rhs);
        assert!(
            means.starts_with("<msubsup><mrow><mo stretchy=\"false\">(</mo>"),
            "{means}"
        );
        assert!(
            means.contains("<mi>i</mi><mo>=</mo><mn>1</mn></mrow><mn>4</mn></msubsup>"),
            "{means}"
        );
        assert!(means.contains("<mi data-flatppl-ref=\"beta\">β</mi><mo>&#x2062;</mo><msub><mi data-flatppl-ref=\"x\">x</mi><mi>i</mi></msub>"), "{means}");
    }

    #[test]
    fn gathers_nest_under_the_family_index() {
        let src = "G = 3\na ~ iid(Normal(0, 1), G)\ng = [1, 2, 3, 1, 2, 3]\nxs = [-1.2, 0.4, 1.1, -0.3, 0.8, 2.0]\nb = elementof(reals)\neta = a[g] .+ b .* xs\np = invlogit.(eta)";
        let rows = rows(src);
        let eta = mathml::expr(&row_named(&rows, "eta").statement.rhs);
        assert!(eta.contains("<msub><mi data-flatppl-ref=\"a\">a</mi><msub><mi data-flatppl-ref=\"g\">g</mi><mi>i</mi></msub></msub>"), "{eta}");
        let p = mathml::expr(&row_named(&rows, "p").statement.rhs);
        assert!(p.contains("<mi>invlogit</mi><mo>&#x2061;</mo><mrow><mo stretchy=\"false\">(</mo><msub><mi data-flatppl-ref=\"eta\">η</mi><mi>i</mi></msub>"), "{p}");
    }

    #[test]
    fn reifications_print_as_named_functions_and_kernels() {
        let src = "mu = elementof(reals)\ntau = elementof(posreals)\ntheta ~ iid(Normal(mu, tau), 8)\ny ~ Normal.(theta, 1.0)\nprior = lawof(record(mu = mu, tau = tau, theta = theta))\nK = kernelof(record(y = y), mu = mu, tau = tau, theta = theta)\nL = likelihoodof(K, record(y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))\npost = bayesupdate(L, prior)\nf = x -> 2 * x\nh = fn(abs(_) * _)\nsq(a) = a^2";
        let rows = rows(src);
        let prior = mathml::expr(&row_named(&rows, "prior").statement.rhs);
        assert_eq!(
            prior,
            "<mrow><mi>ℒ</mi><mo>&#x2061;</mo><mrow><mo stretchy=\"false\">(</mo><mrow><mi data-flatppl-ref=\"mu\">μ</mi><mo>,</mo><mi data-flatppl-ref=\"tau\">τ</mi><mo>,</mo><mi data-flatppl-ref=\"theta\">θ</mi></mrow><mo stretchy=\"false\">)</mo></mrow></mrow>"
        );
        let k = row_named(&rows, "K");
        assert_eq!(k.kind, Kind::Callable);
        let lhs = mathml::expr(&k.statement.lhs);
        assert!(lhs.starts_with("<mrow><mi data-flatppl-ref=\"K\">K</mi><mo>&#x2061;</mo><mrow><mo stretchy=\"false\">(</mo><mi data-flatppl-ref=\"mu\">μ</mi>"), "{lhs}");
        let rhs = mathml::expr(&k.statement.rhs);
        assert!(rhs.contains("<mi>ℒ</mi>"), "{rhs}");
        assert!(rhs.contains("<mi data-flatppl-ref=\"y\">y</mi><mo stretchy=\"false\">|</mo><mi data-flatppl-ref=\"mu\">μ</mi><mo>,</mo><mi data-flatppl-ref=\"tau\">τ</mi>"), "{rhs}");
        let l = row_named(&rows, "L");
        assert_eq!(l.kind, Kind::Likelihood);
        let lhs = mathml::expr(&l.statement.lhs);
        assert!(lhs.starts_with("<mrow><mi data-flatppl-ref=\"L\">L</mi><mo>&#x2061;</mo><mrow><mo stretchy=\"false\">(</mo><mi data-flatppl-ref=\"mu\">μ</mi>"), "{lhs}");
        let rhs = mathml::expr(&l.statement.rhs);
        assert!(rhs.starts_with("<mrow><msub><mi>p</mi><mi data-flatppl-ref=\"K\">K</mi></msub><mo>&#x2061;</mo><mrow><mo stretchy=\"false\">(</mo><mrow><mrow><mi>y</mi><mo>=</mo>"), "{rhs}");
        assert!(
            rhs.contains("<mo stretchy=\"false\">|</mo><mi data-flatppl-ref=\"mu\">μ</mi>"),
            "{rhs}"
        );
        let post = mathml::expr(&row_named(&rows, "post").statement.rhs);
        assert_eq!(
            post,
            "<mrow><mi data-flatppl-ref=\"L\">L</mi><mo>⋅</mo><mi data-flatppl-ref=\"prior\">prior</mi></mrow>"
        );
        let f = row_named(&rows, "f");
        assert_eq!(
            mathml::expr(&f.statement.lhs),
            "<mrow><mi data-flatppl-ref=\"f\">f</mi><mo>&#x2061;</mo><mrow><mo stretchy=\"false\">(</mo><mi>x</mi><mo stretchy=\"false\">)</mo></mrow></mrow>"
        );
        assert_eq!(
            mathml::expr(&f.statement.rhs),
            "<mrow><mn>2</mn><mo>&#x2062;</mo><mi>x</mi></mrow>"
        );
        let h = row_named(&rows, "h");
        assert!(mathml::expr(&h.statement.lhs).contains("<mi>arg1</mi><mo>,</mo><mi>arg2</mi>"));
        let sq = row_named(&rows, "sq");
        assert_eq!(
            mathml::expr(&sq.statement.rhs),
            "<msup><mi>a</mi><mn>2</mn></msup>"
        );
    }

    #[test]
    fn a_reified_named_output_renders_by_reference() {
        let src = "a = elementof(nonnegreals)\nb = a^0.5\nf_sqrt = functionof(b)";
        let rows = rows(src);
        let f = row_named(&rows, "f_sqrt");
        let lhs = mathml::expr(&f.statement.lhs);
        assert!(
            lhs.contains("<mo stretchy=\"false\">(</mo><mi data-flatppl-ref=\"a\">a</mi><mo stretchy=\"false\">)</mo>"),
            "{lhs}"
        );
        assert_eq!(
            mathml::expr(&f.statement.rhs),
            "<mi data-flatppl-ref=\"b\">b</mi>"
        );
    }

    #[test]
    fn zero_input_likelihoods_and_closed_reifications_carry_no_bar() {
        let src = "L0 = likelihoodof(functionof(Normal(0, 1)), 1.5)\ng = functionof(Normal(0, 1))\nh = weighted(2.0, functionof(Normal(0, 1)))";
        let rows = rows(src);
        let l = mathml::expr(&row_named(&rows, "L0").statement.rhs);
        assert!(!l.contains('|'), "{l}");
        assert!(!l.contains("<mo>·</mo>"), "{l}");
        let g = &row_named(&rows, "g").statement;
        assert_eq!(
            mathml::expr(&g.lhs),
            "<mrow><mi data-flatppl-ref=\"g\">g</mi><mo>&#x2061;</mo><mrow><mo stretchy=\"false\">(</mo><mo stretchy=\"false\">)</mo></mrow></mrow>"
        );
        assert!(mathml::expr(&g.rhs).starts_with("<mrow><mi>𝒩</mi>"));
        // In expression position a closed reification is its body.
        let h = mathml::expr(&row_named(&rows, "h").statement.rhs);
        assert!(!h.contains("↦"), "{h}");
    }

    #[test]
    fn fixed_values_keep_their_value_and_gain_an_annotation() {
        let rows = rows("c = fixed(8.0)\nr = record(a = 2.0, c = fixed(8.0))");
        let c = row_named(&rows, "c");
        assert_eq!(mathml::expr(&c.statement.rhs), "<mn>8</mn>");
        assert_eq!(c.annotation.as_deref(), Some("fixed"));
        let r = mathml::expr(&row_named(&rows, "r").statement.rhs);
        assert!(r.contains("<mi>fixed</mi>"), "{r}");
    }

    #[test]
    fn decompositions_fold_into_one_row_and_partial_ones_inline_the_source() {
        let src = "a, b ~ MvNormal(mu = [0.0, 0.0], cov = eye(2))\nm = lawof(record(a = a, b = b))\nfk, pr = disintegrate([\"a\"], m)\np, _ = (3.0, 4.0)\n_ = 1.0 + 2.0";
        let rows = rows(src);
        let names: Vec<&str> = rows.iter().map(|r| r.names[0].as_str()).collect();
        assert_eq!(names, vec!["a", "m", "fk", "p"]);
        let ab = &rows[0];
        assert_eq!(ab.names, vec!["a", "b"]);
        assert_eq!(ab.statement.rel, Rel::Sim);
        assert_eq!(ab.kind, Kind::Draw);
        assert!(mathml::expr(&ab.statement.lhs).starts_with(
            "<mrow><mo stretchy=\"false\">(</mo><mi data-flatppl-ref=\"a\">a</mi><mo>,</mo>"
        ));
        let d = &rows[2];
        assert_eq!(d.names, vec!["fk", "pr"]);
        let drhs = mathml::expr(&d.statement.rhs);
        assert!(
            drhs.starts_with("<mrow><msub><mi>disintegrate</mi><mi>a</mi></msub><mo>&#x2061;</mo>"),
            "{drhs}"
        );
        // A partial decomposition keeps the component index on the inlined source.
        let p = &rows[3];
        assert_eq!(p.names, vec!["p"]);
        assert_eq!(
            mathml::expr(&p.statement.rhs),
            "<msub><mrow><mo stretchy=\"false\">(</mo><mn>3</mn><mo>,</mo><mn>4</mn><mo stretchy=\"false\">)</mo></mrow><mn>1</mn></msub>"
        );
    }

    #[test]
    fn aggregates_are_sums_over_the_reduced_axes_and_metricsums_keep_index_positions() {
        let src = "A = rowstack([[1, 3, 5], [9, 5, 1]])\nB = rowstack([[1, 0], [0, 1], [1, 1]])\nC = aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])\nV = aggregate(var, [.j], A[.i, .j])\nP = aggregate(prod, [.i, .k], A[.i, .j] + B[.j, .k])\nv = [1.0, 2.0]\ny = 2.0 * aggregate(sum, [], v[.i])\nW = 2.0 * aggregate(sum, [.i], A[.i, .j])\ng = rowstack([[1.0, 0.0], [0.0, -1.0]])\nr = [1.0, 2.0]\ng: s[] := r[.mu^] * r[.mu_]";
        let rows = rows(src);
        let a = mathml::expr(&row_named(&rows, "A").statement.rhs);
        assert!(a.starts_with("<mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd>"));
        let c = row_named(&rows, "C");
        assert_eq!(
            mathml::expr(&c.statement.lhs),
            "<msub><mi data-flatppl-ref=\"C\">C</mi><mrow><mi>i</mi><mi>k</mi></mrow></msub>"
        );
        let body = mathml::expr(&c.statement.rhs);
        assert!(
            body.starts_with("<mrow><munder><mo>∑</mo><mi>j</mi></munder>"),
            "{body}"
        );
        assert!(
            body.contains(
                "<msub><mi data-flatppl-ref=\"A\">A</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub>"
            ),
            "{body}"
        );
        let v = mathml::expr(&row_named(&rows, "V").statement.rhs);
        assert!(
            v.starts_with("<mrow><munder><mi>var</mi><mi>i</mi></munder>"),
            "{v}"
        );
        // An additive body is bracketed under the big operator.
        let p = mathml::expr(&row_named(&rows, "P").statement.rhs);
        assert!(
            p.starts_with(
                "<mrow><munder><mo>∏</mo><mi>j</mi></munder><mrow><mo stretchy=\"false\">(</mo>"
            ),
            "{p}"
        );
        // Nested aggregation: a scalar reduction inline, an indexed one as a family.
        let y = mathml::expr(&row_named(&rows, "y").statement.rhs);
        assert!(
            y.contains("<mo>&#x2062;</mo><mrow><munder><mo>∑</mo><mi>i</mi></munder>"),
            "{y}"
        );
        let w = mathml::expr(&row_named(&rows, "W").statement.rhs);
        assert!(
            w.contains("<msub><mrow><mo>(</mo><mrow><munder><mo>∑</mo><mi>j</mi></munder>"),
            "{w}"
        );
        let s = row_named(&rows, "s");
        let body = mathml::expr(&s.statement.rhs);
        assert!(
            body.contains("<msup><mi data-flatppl-ref=\"r\">r</mi><mi>μ</mi></msup>"),
            "{body}"
        );
        assert!(
            body.contains("<msub><mi data-flatppl-ref=\"r\">r</mi><mi>μ</mi></msub>"),
            "{body}"
        );
        assert_eq!(
            s.annotation.as_deref(),
            Some("indices lowered with the metric g")
        );
    }

    #[test]
    fn long_literal_arrays_become_memberships() {
        let src = "xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]\nks = [1, 2, 3]";
        let rows = rows(src);
        let xs = row_named(&rows, "xs");
        assert_eq!(xs.statement.rel, Rel::In);
        assert!(xs.elided);
        assert_eq!(
            mathml::expr(&xs.statement.rhs),
            "<msup><mi>ℝ</mi><mn>13</mn></msup>"
        );
        assert_eq!(
            xs.annotation.as_deref(),
            Some("13 values, see the data appendix")
        );
        let ks = row_named(&rows, "ks");
        assert_eq!(ks.statement.rel, Rel::Eq);
        assert!(!ks.elided);
        assert_eq!(
            mathml::expr(&ks.statement.rhs),
            "<mrow><mo stretchy=\"false\">(</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo stretchy=\"false\">)</mo></mrow>"
        );
    }

    #[test]
    fn access_forms_and_sums() {
        let src = "r = record(a = 1.0, b = 2.0)\nv = [1.0, 2.0, 3.0]\nx = r.a\ny = v[2]\nz = sum(v)\nt = table(c = [1.0, 2.0], d = [3.0, 4.0])\nw = t.c\nq = get(r, [\"a\", \"b\"])\ns = sum(v) + 1.0\nu = (sum(v) + 1.0) * 2.0";
        let rows = rows(src);
        assert_eq!(
            mathml::expr(&row_named(&rows, "x").statement.rhs),
            "<mrow><mi data-flatppl-ref=\"r\">r</mi><mo>.</mo><mi>a</mi></mrow>"
        );
        assert_eq!(
            mathml::expr(&row_named(&rows, "y").statement.rhs),
            "<msub><mi data-flatppl-ref=\"v\">v</mi><mn>2</mn></msub>"
        );
        let z = mathml::expr(&row_named(&rows, "z").statement.rhs);
        assert_eq!(
            z,
            "<mrow><munder><mo>∑</mo><mi>i</mi></munder><msub><mi data-flatppl-ref=\"v\">v</mi><mi>i</mi></msub></mrow>"
        );
        assert!(
            mathml::expr(&row_named(&rows, "q").statement.rhs).ends_with(
                "<mrow><mo stretchy=\"false\">{</mo><mi>a</mi><mo>,</mo><mi>b</mi><mo stretchy=\"false\">}</mo></mrow></msub>"
            )
        );
        // A big operator that is not the last thing in a row is bracketed.
        let s = mathml::expr(&row_named(&rows, "s").statement.rhs);
        assert!(
            s.starts_with("<mrow><mrow><mo>(</mo><mrow><munder><mo>∑</mo>"),
            "{s}"
        );
        let u = mathml::expr(&row_named(&rows, "u").statement.rhs);
        assert!(
            u.starts_with("<mrow><mrow><mo>(</mo><mrow><mrow><mo>(</mo><mrow><munder>"),
            "{u}"
        );
    }

    #[test]
    fn module_references_and_unknown_builtins_stay_roman() {
        let src = "h = standard_module(\"particle-physics\", \"0.1\")\nk = h.kallen(1.0, 2.0, 3.0)\nd = load_data(\"x.csv\", cartpow(reals, 4))\nb = bincounts([0.0, 1.0, 2.0], d)\ncb = h.CrystalBall(n = 2.0, alpha = 1.5, sigma = 0.3, m0 = 5.0)";
        let rows = rows(src);
        let k = mathml::expr(&row_named(&rows, "k").statement.rhs);
        assert!(k.starts_with("<mrow><mrow><mi data-flatppl-ref=\"h\">h</mi><mo>.</mo><mi>kallen</mi></mrow><mo>&#x2061;</mo>"), "{k}");
        assert_eq!(row_named(&rows, "h").kind, Kind::Module);
        let d = mathml::expr(&row_named(&rows, "d").statement.rhs);
        assert!(d.starts_with("<mrow><mi>load_data</mi><mo>&#x2061;</mo><mrow><mo stretchy=\"false\">(</mo><mtext>\"x.csv\"</mtext><mo>,</mo><msup><mover><mi>ℝ</mi><mo>¯</mo></mover><mn>4</mn></msup>"), "{d}");
        assert!(
            mathml::expr(&row_named(&rows, "b").statement.rhs)
                .starts_with("<mrow><mi>bincounts</mi>")
        );
        // A standard-module distribution's keyword arguments follow §09 order.
        let cb = mathml::expr(&row_named(&rows, "cb").statement.rhs);
        assert!(
            cb.ends_with("<mo stretchy=\"false\">(</mo><mn>5</mn><mo>,</mo><mn>0.3</mn><mo>,</mo><mn>1.5</mn><mo>,</mo><mn>2</mn><mo stretchy=\"false\">)</mo></mrow></mrow>"),
            "{cb}"
        );
    }

    #[test]
    fn arithmetic_and_complex_forms() {
        let src = "m = elementof(posreals)\nmD = 1.8\nmpi = 0.14\nE1 = (mD^2 - m^2 - mpi^2) / (2 * m)\nz = 0.8 * cis(0.7)\nw = abs2(z)\nc = ifelse(m > 1.0, m, 1.0)";
        let rows = rows(src);
        let e1 = mathml::expr(&row_named(&rows, "E1").statement.rhs);
        assert!(
            e1.starts_with(
                "<mfrac><mrow><mrow><msup><mi data-flatppl-ref=\"mD\">mD</mi><mn>2</mn></msup>"
            ),
            "{e1}"
        );
        let z = mathml::expr(&row_named(&rows, "z").statement.rhs);
        assert!(z.contains("<msup><mi>e</mi><mrow><mn>0.7</mn><mo>&#x2062;</mo><mi mathvariant=\"normal\">i</mi></mrow></msup>"), "{z}");
        let w = mathml::expr(&row_named(&rows, "w").statement.rhs);
        assert_eq!(
            w,
            "<msup><mrow><mo stretchy=\"false\">|</mo><mi data-flatppl-ref=\"z\">z</mi><mo stretchy=\"false\">|</mo></mrow><mn>2</mn></msup>"
        );
        let c = mathml::expr(&row_named(&rows, "c").statement.rhs);
        assert!(c.contains("<mtext>if&#xa0;</mtext><mrow><mi data-flatppl-ref=\"m\">m</mi><mo>&gt;</mo><mn>1</mn></mrow>"), "{c}");
    }

    #[test]
    fn too_deep_mathematics_renders_as_source_text_with_a_diagnostic() {
        // A wide `superpose` folds into a chain as deep as it is wide.
        let terms: Vec<String> = (0..DEFAULT_MAX_DEPTH + 20)
            .map(|i| format!("Normal({i}, 1)"))
            .collect();
        let src = format!("m = superpose({})", terms.join(", "));
        let deep = rows(&src);
        let m = row_named(&deep, "m");
        assert!(
            matches!(m.statement.rhs, Math::Code(ref t) if t.starts_with("superpose(") && t.contains("Normal(0, 1)")),
            "{:?}",
            m.statement.rhs
        );
        assert_eq!(m.diagnostics.len(), 1);
        assert!(
            m.diagnostics[0].contains("deeper than"),
            "{}",
            m.diagnostics[0]
        );
        // A shallower one lowers normally.
        let shallow = rows("m = superpose(Normal(0, 1), Normal(1, 1), Normal(2, 1))");
        assert!(row_named(&shallow, "m").diagnostics.is_empty());
    }

    #[test]
    fn a_too_deep_decomposition_source_falls_back_without_generated_names() {
        let terms: Vec<&str> = std::iter::repeat_n("1", DEFAULT_MAX_DEPTH + 20).collect();
        let sum = terms.join(" + ");
        // A partial decomposition inlines the source; the guard sees through
        // the synthetic binding it inlines.
        let src = format!("p, _ = ({sum}, 2)\nq = p * 2");
        // With the source at hand the row quotes it as written; without, the
        // canonical print (line-broken) stands in. Neither names `__0x…`.
        for (label, rows) in [
            ("source", rows(&src)),
            ("printed", rows_without_source(&src)),
        ] {
            let p = row_named(&rows, "p");
            let Math::Code(text) = &p.statement.rhs else {
                panic!("{label}: {:?}", p.statement.rhs);
            };
            let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
            assert!(
                flat.starts_with("( 1 + 1") || flat.starts_with("(1 + 1"),
                "{label}: {flat}"
            );
            assert!(flat.ends_with(")[1]"), "{label}: {flat}");
            assert!(!text.contains("__0x"), "{label}: {text}");
            assert_eq!(p.diagnostics.len(), 1, "{label}");
            assert!(row_named(&rows, "q").diagnostics.is_empty(), "{label}");
        }
        // A complete decomposition of a draw keeps its relation.
        let src = format!("a, b ~ MvNormal([{sum}, 0.0], eye(2))");
        let rows = rows(&src);
        let ab = row_named(&rows, "a");
        assert_eq!(ab.names, vec!["a", "b"]);
        assert_eq!(ab.statement.rel, Rel::Sim);
        assert!(
            matches!(&ab.statement.rhs, Math::Code(t) if t.starts_with("MvNormal([1 + 1")),
            "{:?}",
            ab.statement.rhs
        );
    }
}
