//! Pretty-print a [`Module`] back to canonical FlatPPL.
//!
//! Two syntax levels, both canonical (the printer is *canonicalizing*, not
//! byte-preserving — parse → print is semantically faithful and idempotent
//! after the first print at either level):
//!
//! - [`Syntax::Full`] (the default): re-applies every spec §05 sugar form —
//!   precedence-aware infix operators (including dotted broadcasting and
//!   comparison chains), indexing with `:` / `!` slicing, field access,
//!   lambdas (a `functionof` whose boundary is all placeholders), the `~` and
//!   `:=` / `metric: …` statement forms, and array / tuple literals.
//! - [`Syntax::Minimal`]: the spec §04 lowered linear form — every right-hand
//!   side is a literal or a function call, with only the `~` statement
//!   re-sugar and array / tuple literal forms.
//!
//! Sugar is re-applied only where the re-parse provably inverts it; anything
//! else keeps the call form. E.g. `get` on a module binding does *not* print
//! as `m.x` (which would re-parse as a cross-module ref), a non-`sum`
//! `aggregate` has no `:=` form, and `kernelof` has no lambda form.

use std::collections::HashSet;

use flatppl_core::{
    Axis, Call, CallHead, Doc, Inputs, Markup, Module, Node, NodeId, Ref, RefNs, Scalar, Symbol,
    Variance,
};

use crate::parser::is_placeholder;

/// Surface-syntax level for printed FlatPPL.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Syntax {
    /// All spec §05 sugar re-applied (operators, indexing, lambdas, `:=`, …).
    #[default]
    Full,
    /// The lowered linear form: function-call syntax only (plus `~` and
    /// array / tuple literals).
    Minimal,
}

/// Render `module` as canonical FlatPPL text with full sugar
/// (no trailing newline).
pub fn print(module: &Module) -> String {
    print_with(module, Syntax::Full)
}

/// Render `module` as canonical FlatPPL text at the given [`Syntax`] level
/// (no trailing newline).
pub fn print_with(module: &Module, syntax: Syntax) -> String {
    let printer = Printer::new(module, syntax);
    let mut out = String::new();
    for (idx, (_, binding)) in module.bindings().enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        if let Some(doc) = &binding.doc {
            out.push_str(&print_doc(doc));
            out.push('\n');
        }
        out.push_str(&printer.binding_text(binding));
    }
    out
}

// ---- precedence levels (spec §05 grammar nonterminals, low → high) ----
//
// A node prints bare in a context accepting its level and parenthesized in a
// tighter one; the levels mirror the grammar so the re-parse reproduces the
// tree exactly.

/// `Expression` — lambdas bind loosest (the body extends maximally right).
const EXPR: u8 = 0;
const OR: u8 = 1;
const AND: u8 = 2;
/// Comparisons are non-associative; adjacent comparisons form a *chain*
/// statement-lowering instead, so comparison operands print at [`ADD`].
const CMP: u8 = 3;
const ADD: u8 = 4;
const MUL: u8 = 5;
const UNARY: u8 = 6;
/// `^` — right-associative, binds tighter than unary minus.
const EXP: u8 = 7;
const POSTFIX: u8 = 8;
const ATOM: u8 = 9;

/// An infix operator's surface forms and grammar level (spec §05).
struct BinOp {
    plain: &'static str,
    dotted: Option<&'static str>,
    prec: u8,
}

/// The lowered builtin → infix operator map (the inverse of the parser's
/// operator lowering).
fn binop(func: &str) -> Option<BinOp> {
    let (plain, dotted, prec) = match func {
        "lor" => ("||", Some(".||"), OR),
        "land" => ("&&", Some(".&&"), AND),
        "lt" => ("<", Some(".<"), CMP),
        "gt" => (">", Some(".>"), CMP),
        "le" => ("<=", Some(".<="), CMP),
        "ge" => (">=", Some(".>="), CMP),
        "equal" => ("==", Some(".=="), CMP),
        "unequal" => ("!=", Some(".!="), CMP),
        // Membership has no dotted form (spec §05).
        "in" => ("in", None, CMP),
        "add" => ("+", Some(".+"), ADD),
        "sub" => ("-", Some(".-"), ADD),
        "mul" => ("*", Some(".*"), MUL),
        "divide" => ("/", Some("./"), MUL),
        "pow" => ("^", Some(".^"), EXP),
        _ => return None,
    };
    Some(BinOp {
        plain,
        dotted,
        prec,
    })
}

/// The `(plain, dotted)` prefix spellings of a lowered unary builtin.
fn unop(func: &str) -> Option<(&'static str, &'static str)> {
    match func {
        "neg" => Some(("-", ".-")),
        "lnot" => Some(("!", ".!")),
        _ => None,
    }
}

/// Operand contexts for an infix level. Left-associative levels accept their
/// own level on the left and one tighter on the right; `^` is
/// right-associative with a `Postfix` left and `Unary` right (spec §05);
/// comparisons are non-associative (both sides `Additive`).
fn operand_mins(prec: u8) -> (u8, u8) {
    match prec {
        EXP => (POSTFIX, UNARY),
        CMP => (ADD, ADD),
        p => (p, p + 1),
    }
}

struct Printer<'m> {
    module: &'m Module,
    syntax: Syntax,
    /// All binding names. A built-in op or constant sharing a binding's name
    /// cannot print bare — name resolution would capture it as a reference /
    /// user call — so it prints through the reserved `base` namespace, which
    /// always denotes the built-in (spec §04 / §11).
    bound: HashSet<Symbol>,
    /// Names bound to `load_module` / `standard_module`. These open
    /// namespaces, so field-access sugar on them is suppressed (`m.x` would
    /// re-parse as a cross-module ref, not `get`).
    modules: HashSet<Symbol>,
}

impl<'m> Printer<'m> {
    fn new(module: &'m Module, syntax: Syntax) -> Self {
        let mut bound = HashSet::new();
        let mut modules = HashSet::new();
        for (_, b) in module.bindings() {
            bound.insert(b.name);
            if let Node::Call(c) = module.node(b.rhs) {
                if let CallHead::Builtin(op) = c.head {
                    if matches!(module.resolve(op), "load_module" | "standard_module") {
                        modules.insert(b.name);
                    }
                }
            }
        }
        Printer {
            module,
            syntax,
            bound,
            modules,
        }
    }

    /// A built-in name, spelled so it re-resolves to the built-in:
    /// `base.{name}` when a module binding shadows it, bare otherwise.
    fn builtin_name(&self, sym: Symbol) -> String {
        let name = self.module.resolve(sym);
        if self.bound.contains(&sym) {
            format!("base.{name}")
        } else {
            name.to_string()
        }
    }

    // ---- statement level ----

    fn binding_text(&self, binding: &flatppl_core::Binding) -> String {
        let name = self.module.resolve(binding.name);
        if let Node::Call(call) = self.module.node(binding.rhs) {
            // `name = draw(M)` re-sugars to `name ~ M` (both levels).
            if let CallHead::Builtin(op) = call.head {
                if self.module.resolve(op) == "draw"
                    && call.args.len() == 1
                    && call.named.is_empty()
                    && call.inputs.is_none()
                {
                    return format!("{name} ~ {}", self.expr_top(call.args[0]));
                }
            }
            if self.syntax == Syntax::Full {
                if let Some(text) = self.aggregate_stmt(name, call) {
                    return text;
                }
                if let Some(text) = self.metricsum_stmt(name, call) {
                    return text;
                }
            }
        }
        format!("{name} = {}", self.expr_top(binding.rhs))
    }

    fn expr_top(&self, id: NodeId) -> String {
        match self.syntax {
            Syntax::Full => self.full(id, EXPR, &[]),
            Syntax::Minimal => self.minimal(id),
        }
    }

    /// `name[.axes…] := body` — only the `sum` reduction has the statement
    /// form (spec §04).
    fn aggregate_stmt(&self, name: &str, call: &Call) -> Option<String> {
        let CallHead::Builtin(op) = call.head else {
            return None;
        };
        if self.module.resolve(op) != "aggregate"
            || call.args.len() != 3
            || !call.named.is_empty()
            || call.inputs.is_some()
        {
            return None;
        }
        match self.module.node(call.args[0]) {
            Node::Const(s) if self.module.resolve(*s) == "sum" => {}
            _ => return None,
        }
        let axes = self.axis_list_text(call.args[1])?;
        Some(format!(
            "{name}[{axes}] := {}",
            self.full(call.args[2], EXPR, &[])
        ))
    }

    /// `metric: name[.axes…] := body` — the statement form's metric is a
    /// single bare `Name` (grammar §05); a module-qualified or computed
    /// metric keeps the call form.
    fn metricsum_stmt(&self, name: &str, call: &Call) -> Option<String> {
        let CallHead::Builtin(op) = call.head else {
            return None;
        };
        if self.module.resolve(op) != "metricsum"
            || call.args.len() != 3
            || !call.named.is_empty()
            || call.inputs.is_some()
        {
            return None;
        }
        let metric = match self.module.node(call.args[0]) {
            Node::Ref(r) if matches!(r.ns, RefNs::SelfMod) => self.module.resolve(r.name),
            // A constant metric name that a binding shadows would re-parse
            // as a reference to that binding — keep the call form.
            Node::Const(s) if !self.bound.contains(s) => self.module.resolve(*s),
            _ => return None,
        };
        if !is_name_token(metric) {
            return None;
        }
        let axes = self.axis_list_text(call.args[1])?;
        Some(format!(
            "{metric}: {name}[{axes}] := {}",
            self.full(call.args[2], EXPR, &[])
        ))
    }

    /// The `[.i, .k^]` axis list of a `:=` statement: a `vector` literal whose
    /// entries are all axis labels.
    fn axis_list_text(&self, id: NodeId) -> Option<String> {
        let Node::Call(c) = self.module.node(id) else {
            return None;
        };
        let CallHead::Builtin(op) = c.head else {
            return None;
        };
        if self.module.resolve(op) != "vector" || !c.named.is_empty() || c.inputs.is_some() {
            return None;
        }
        let mut parts = Vec::with_capacity(c.args.len());
        for &a in c.args.iter() {
            let Node::Axis(axis) = self.module.node(a) else {
                return None;
            };
            parts.push(print_axis(self.module, axis));
        }
        Some(parts.join(", "))
    }

    // ---- full syntax (precedence-aware, all sugar) ----

    /// Print `id` for a context accepting precedence ≥ `min`, parenthesizing
    /// if the node binds looser. `lambda` is the innermost enclosing lambda's
    /// placeholder set (those print bare); it is REPLACED, not extended, at
    /// each reification boundary — placeholders are innermost-scoped
    /// (spec §04), so an outer placeholder inside a nested reification keeps
    /// its `_x_` spelling.
    fn full(&self, id: NodeId, min: u8, lambda: &[Symbol]) -> String {
        let (text, prec) = self.full_node(id, lambda);
        if prec < min {
            format!("({text})")
        } else {
            text
        }
    }

    fn full_node(&self, id: NodeId, lambda: &[Symbol]) -> (String, u8) {
        match self.module.node(id) {
            // A negative numeric literal prints like the unary-minus
            // expression it re-parses as.
            Node::Lit(s) => (print_scalar(s), scalar_prec(s)),
            Node::Const(sym) => (self.builtin_name(*sym), ATOM),
            Node::Hole => ("_".to_string(), ATOM),
            Node::Ref(r) => self.ref_form(r, lambda),
            Node::Axis(a) => (print_axis(self.module, a), ATOM),
            Node::Call(c) => self.full_call(id, c, lambda),
        }
    }

    fn ref_form(&self, r: &Ref, lambda: &[Symbol]) -> (String, u8) {
        if matches!(r.ns, RefNs::Local) && lambda.contains(&r.name) {
            // The innermost lambda's placeholder prints as its bare argument
            // name (`_x_` → `x`); re-parsing rewrites it back.
            let spelled = self.module.resolve(r.name);
            return (spelled[1..spelled.len() - 1].to_string(), ATOM);
        }
        match r.ns {
            RefNs::SelfMod | RefNs::Local => (print_ref(self.module, r), ATOM),
            RefNs::Module(_) => (print_ref(self.module, r), POSTFIX),
        }
    }

    fn full_call(&self, id: NodeId, call: &Call, lambda: &[Symbol]) -> (String, u8) {
        if call.inputs.is_some() {
            if let Some(text) = self.lambda_form(call) {
                return (text, EXPR);
            }
            return (self.reified_text(call, Syntax::Full), POSTFIX);
        }
        match &call.head {
            CallHead::Builtin(op) => {
                let name = self.module.resolve(*op);
                if call.named.is_empty() && call.args.len() == 2 {
                    if let Some(op) = binop(name) {
                        if op.prec == AND {
                            // `land` may be a lowered comparison chain.
                            if let Some(chain) = self.comparison_chain(id, lambda) {
                                return (chain, CMP);
                            }
                        }
                        let (lmin, rmin) = operand_mins(op.prec);
                        let text = format!(
                            "{} {} {}",
                            self.full(call.args[0], lmin, lambda),
                            op.plain,
                            self.full(call.args[1], rmin, lambda)
                        );
                        return (text, op.prec);
                    }
                }
                if call.named.is_empty() && call.args.len() == 1 {
                    if let Some((plain, _)) = unop(name) {
                        let text = format!("{plain}{}", self.full(call.args[0], UNARY, lambda));
                        return (text, UNARY);
                    }
                }
                match name {
                    "get" if call.named.is_empty() && call.args.len() >= 2 => {
                        self.get_form(call, lambda)
                    }
                    "broadcast" if call.args.len() >= 2 => self.broadcast_form(call, lambda),
                    "vector" if call.named.is_empty() => {
                        (format!("[{}]", self.expr_list(&call.args, lambda)), ATOM)
                    }
                    "tuple" if call.named.is_empty() && call.args.len() >= 2 => {
                        (format!("({})", self.expr_list(&call.args, lambda)), ATOM)
                    }
                    _ => (
                        format!(
                            "{}({})",
                            self.builtin_name(*op),
                            self.args_text(call, lambda)
                        ),
                        POSTFIX,
                    ),
                }
            }
            // The callee is an expression (a ref prints as its bare/dotted
            // name; an inline callable as itself, e.g. `functionof(…)(args)`).
            CallHead::User(callee) => (
                format!(
                    "{}({})",
                    self.full(*callee, POSTFIX, lambda),
                    self.args_text(call, lambda)
                ),
                POSTFIX,
            ),
        }
    }

    /// Indexing `obj[i, :, !]` or field access `obj.name` (spec §05). Field
    /// access applies only when the printed form re-parses as one: the key
    /// must be a non-reserved `Name` and the object must not be a namespace
    /// (`self` / `base` / a module binding — dot syntax there means member
    /// access).
    fn get_form(&self, call: &Call, lambda: &[Symbol]) -> (String, u8) {
        if call.args.len() == 2 {
            if let Node::Lit(Scalar::Str(key)) = self.module.node(call.args[1]) {
                if is_field_name(key) && !self.is_namespace(call.args[0]) {
                    let object = self.dot_object(call.args[0], lambda);
                    return (format!("{object}.{key}"), POSTFIX);
                }
            }
        }
        let object = self.full(call.args[0], POSTFIX, lambda);
        let entries: Vec<String> = call.args[1..]
            .iter()
            .map(|&a| match self.module.node(a) {
                // The `all` / `only` selectors print as `:` / `!` slices; a
                // top-of-entry `!` is unambiguous (always before `,` / `]`).
                Node::Const(s) if self.module.resolve(*s) == "all" => ":".to_string(),
                Node::Const(s) if self.module.resolve(*s) == "only" => "!".to_string(),
                _ => self.full(a, EXPR, lambda),
            })
            .collect();
        (format!("{object}[{}]", entries.join(", ")), POSTFIX)
    }

    fn is_namespace(&self, id: NodeId) -> bool {
        match self.module.node(id) {
            Node::Const(s) => matches!(self.module.resolve(*s), "self" | "base"),
            Node::Ref(r) => matches!(r.ns, RefNs::SelfMod) && self.modules.contains(&r.name),
            _ => false,
        }
    }

    /// The object of a `.field` / `.(…)` postfix. A non-negative numeric
    /// literal must be parenthesized or the following `.` would lex into the
    /// number by maximal munch (`1.foo` lexes as the real `1.` then `foo`).
    fn dot_object(&self, id: NodeId, lambda: &[Symbol]) -> String {
        let text = self.full(id, POSTFIX, lambda);
        match self.module.node(id) {
            Node::Lit(Scalar::Int(n)) if *n >= 0 => format!("({text})"),
            Node::Lit(Scalar::Real(r)) if !r.is_sign_negative() => format!("({text})"),
            _ => text,
        }
    }

    /// `broadcast` re-sugars to the dotted forms (spec §05): a dotted
    /// operator when the head is that operator's name, a dot-call `f.(…)`
    /// otherwise.
    fn broadcast_form(&self, call: &Call, lambda: &[Symbol]) -> (String, u8) {
        if call.named.is_empty() {
            if let Node::Const(f) = self.module.node(call.args[0]) {
                let f = self.module.resolve(*f);
                if call.args.len() == 3 {
                    if let Some(op) = binop(f) {
                        if let Some(dotted) = op.dotted {
                            let (lmin, rmin) = operand_mins(op.prec);
                            let text = format!(
                                "{} {dotted} {}",
                                self.full(call.args[1], lmin, lambda),
                                self.full(call.args[2], rmin, lambda)
                            );
                            return (text, op.prec);
                        }
                    }
                }
                if call.args.len() == 2 {
                    if let Some((_, dotted)) = unop(f) {
                        let text = format!("{dotted}{}", self.full(call.args[1], UNARY, lambda));
                        return (text, UNARY);
                    }
                }
            }
        }
        let mut parts: Vec<String> = call.args[1..]
            .iter()
            .map(|&a| self.full(a, EXPR, lambda))
            .collect();
        for n in call.named.iter() {
            parts.push(format!(
                "{} = {}",
                self.module.resolve(n.name),
                self.full(n.value, EXPR, lambda)
            ));
        }
        let head = self.dot_object(call.args[0], lambda);
        (format!("{head}.({})", parts.join(", ")), POSTFIX)
    }

    /// A `functionof` whose authored boundary is exclusively placeholder
    /// declarations is exactly the lambda desugaring (spec §04) — print it
    /// back as `x -> body` / `(x, y) -> body` with the placeholders bare.
    /// (`fn`-hole reifications are indistinguishable after lowering and
    /// canonicalize to a lambda too. `kernelof` has no lambda form.)
    fn lambda_form(&self, call: &Call) -> Option<String> {
        let CallHead::Builtin(op) = call.head else {
            return None;
        };
        if self.module.resolve(op) != "functionof" {
            return None;
        }
        let Some(Inputs::Spec(entries)) = &call.inputs else {
            return None;
        };
        if entries.is_empty() || call.args.len() != 1 || !call.named.is_empty() {
            return None;
        }
        let mut params = Vec::with_capacity(entries.len());
        let mut placeholders = Vec::with_capacity(entries.len());
        for (name, r) in entries.iter() {
            if !matches!(r.ns, RefNs::Local) {
                return None;
            }
            let param = self.module.resolve(*name);
            if self.module.resolve(r.name) != format!("_{param}_") {
                return None;
            }
            if !is_lambda_param_name(param) {
                return None;
            }
            params.push(param);
            placeholders.push(r.name);
        }
        let body = self.full(call.args[0], EXPR, &placeholders);
        Some(if params.len() == 1 {
            format!("{} -> {body}", params[0])
        } else {
            format!("({}) -> {body}", params.join(", "))
        })
    }

    /// Re-sugar a lowered comparison chain. `a < b <= c` lowers to
    /// `land(land(lt(a, b), le(b, c)), …)` re-using the middle operands, so a
    /// left `land`-spine of plain comparisons whose adjacent operands are
    /// structurally equal prints back as the chain.
    fn comparison_chain(&self, id: NodeId, lambda: &[Symbol]) -> Option<String> {
        let mut elems = Vec::new();
        self.land_spine(id, &mut elems);
        if elems.len() < 2 {
            return None;
        }
        let mut cmps = Vec::with_capacity(elems.len());
        for &e in &elems {
            let Node::Call(c) = self.module.node(e) else {
                return None;
            };
            let CallHead::Builtin(op) = c.head else {
                return None;
            };
            let op = binop(self.module.resolve(op)).filter(|o| o.prec == CMP)?;
            if c.args.len() != 2 || !c.named.is_empty() || c.inputs.is_some() {
                return None;
            }
            cmps.push((op.plain, c.args[0], c.args[1]));
        }
        for pair in cmps.windows(2) {
            if !self.structural_eq(pair[0].2, pair[1].1) {
                return None;
            }
        }
        let mut text = self.full(cmps[0].1, ADD, lambda);
        for (op, _, rhs) in &cmps {
            text.push_str(&format!(" {op} {}", self.full(*rhs, ADD, lambda)));
        }
        Some(text)
    }

    /// Flatten the left-leaning spine of plain binary `land` calls.
    fn land_spine(&self, id: NodeId, out: &mut Vec<NodeId>) {
        if let Node::Call(c) = self.module.node(id) {
            if let CallHead::Builtin(op) = c.head {
                if self.module.resolve(op) == "land"
                    && c.args.len() == 2
                    && c.named.is_empty()
                    && c.inputs.is_none()
                {
                    self.land_spine(c.args[0], out);
                    out.push(c.args[1]);
                    return;
                }
            }
        }
        out.push(id);
    }

    /// Structural equality of two expression trees (chain middles are shared
    /// node ids straight from the parser, but FlatPIR round-trips duplicate
    /// them).
    fn structural_eq(&self, a: NodeId, b: NodeId) -> bool {
        if a == b {
            return true;
        }
        match (self.module.node(a), self.module.node(b)) {
            (Node::Lit(x), Node::Lit(y)) => x == y,
            (Node::Const(x), Node::Const(y)) => x == y,
            (Node::Hole, Node::Hole) => true,
            (Node::Ref(x), Node::Ref(y)) => x == y,
            (Node::Axis(x), Node::Axis(y)) => x == y,
            (Node::Call(x), Node::Call(y)) => {
                let heads = match (x.head, y.head) {
                    (CallHead::Builtin(p), CallHead::Builtin(q)) => p == q,
                    (CallHead::User(p), CallHead::User(q)) => self.structural_eq(p, q),
                    _ => false,
                };
                heads
                    && x.inputs == y.inputs
                    && x.args.len() == y.args.len()
                    && x.args
                        .iter()
                        .zip(y.args.iter())
                        .all(|(&p, &q)| self.structural_eq(p, q))
                    && x.named.len() == y.named.len()
                    && x.named.iter().zip(y.named.iter()).all(|(p, q)| {
                        p.kind == q.kind && p.name == q.name && self.structural_eq(p.value, q.value)
                    })
            }
            _ => false,
        }
    }

    fn expr_list(&self, ids: &[NodeId], lambda: &[Symbol]) -> String {
        ids.iter()
            .map(|&e| self.full(e, EXPR, lambda))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Positional args, then named entries (each `name = value`).
    fn args_text(&self, call: &Call, lambda: &[Symbol]) -> String {
        let mut parts: Vec<String> = call
            .args
            .iter()
            .map(|&a| self.full(a, EXPR, lambda))
            .collect();
        for n in call.named.iter() {
            parts.push(format!(
                "{} = {}",
                self.module.resolve(n.name),
                self.full(n.value, EXPR, lambda)
            ));
        }
        parts.join(", ")
    }

    // ---- reified callables (shared by both syntaxes) ----

    /// Explicit reification form: output first, then — for an authored
    /// boundary — the entries as boundary kwargs (`p = a`, `x = _x_`). An
    /// `%autoinputs` cut prints as the bare boundary-less form (a filled list
    /// is inference metadata and is dropped on conversion to FlatPPL,
    /// spec §11). The whole call is a placeholder-scope boundary, so the body
    /// prints with no enclosing-lambda strip set.
    fn reified_text(&self, call: &Call, syntax: Syntax) -> String {
        let render = |id: NodeId| match syntax {
            Syntax::Full => self.full(id, EXPR, &[]),
            Syntax::Minimal => self.minimal(id),
        };
        let head = match &call.head {
            CallHead::Builtin(op) => self.module.resolve(*op).to_string(),
            CallHead::User(callee) => render(*callee),
        };
        let mut parts = vec![render(call.args[0])];
        if let Some(Inputs::Spec(entries)) = &call.inputs {
            for (pname, r) in entries.iter() {
                parts.push(format!(
                    "{} = {}",
                    self.module.resolve(*pname),
                    print_ref(self.module, r)
                ));
            }
        }
        format!("{head}({})", parts.join(", "))
    }

    // ---- minimal syntax (lowered linear form) ----

    fn minimal(&self, id: NodeId) -> String {
        match self.module.node(id) {
            Node::Lit(s) => print_scalar(s),
            Node::Const(sym) => self.builtin_name(*sym),
            Node::Hole => "_".to_string(),
            Node::Ref(r) => print_ref(self.module, r),
            Node::Axis(a) => print_axis(self.module, a),
            Node::Call(c) => self.minimal_call(c),
        }
    }

    fn minimal_call(&self, call: &Call) -> String {
        if call.inputs.is_some() {
            return self.reified_text(call, Syntax::Minimal);
        }
        match &call.head {
            CallHead::Builtin(op) => {
                let name = self.module.resolve(*op);
                match name {
                    // Canonical literal surface forms (kept in minimal too —
                    // they are the only spelling of array / tuple literals).
                    "vector" if call.named.is_empty() => {
                        format!("[{}]", self.minimal_list(&call.args))
                    }
                    "tuple" if call.named.is_empty() && call.args.len() >= 2 => {
                        format!("({})", self.minimal_list(&call.args))
                    }
                    _ => format!("{}({})", self.builtin_name(*op), self.minimal_args(call)),
                }
            }
            CallHead::User(callee) => {
                format!("{}({})", self.minimal(*callee), self.minimal_args(call))
            }
        }
    }

    fn minimal_list(&self, ids: &[NodeId]) -> String {
        ids.iter()
            .map(|&e| self.minimal(e))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn minimal_args(&self, call: &Call) -> String {
        let mut parts: Vec<String> = call.args.iter().map(|&a| self.minimal(a)).collect();
        for n in call.named.iter() {
            parts.push(format!(
                "{} = {}",
                self.module.resolve(n.name),
                self.minimal(n.value)
            ));
        }
        parts.join(", ")
    }
}

fn scalar_prec(s: &Scalar) -> u8 {
    match s {
        Scalar::Int(n) if *n < 0 => UNARY,
        Scalar::Real(r) if r.is_sign_negative() => UNARY,
        _ => ATOM,
    }
}

/// Is `s` lexically a `Name` token (so it re-lexes as one when printed bare)?
fn is_name_token(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty()
        && (b[0].is_ascii_alphabetic() || b[0] == b'_')
        && b.iter().all(|&c| c.is_ascii_alphanumeric() || c == b'_')
}

/// Can `key` print as a surface field name? Reserved words are recognized
/// before `Name` (spec §05), so those keep the `get` call form.
fn is_field_name(key: &str) -> bool {
    is_name_token(key) && !matches!(key, "true" | "false" | "in" | "all" | "only")
}

/// Can `s` be a surface lambda parameter (the parser's `check_lambda_param`
/// plus the `Name` lexical rule)?
fn is_lambda_param_name(s: &str) -> bool {
    is_name_token(s)
        && !matches!(
            s,
            "_" | "true" | "false" | "in" | "all" | "only" | "self" | "base"
        )
        && !is_placeholder(s)
}

fn print_ref(module: &Module, r: &Ref) -> String {
    let name = module.resolve(r.name);
    match r.ns {
        // SelfMod / Local print as the bare name (a `%local` body ref prints as
        // its input name); module-member access uses dot syntax.
        RefNs::SelfMod | RefNs::Local => name.to_string(),
        RefNs::Module(alias) => format!("{}.{name}", module.resolve(alias)),
    }
}

fn print_axis(module: &Module, a: &Axis) -> String {
    let name = module.resolve(a.name);
    match a.variance {
        None => format!(".{name}"),
        Some(Variance::Upper) => format!(".{name}^"),
        Some(Variance::Lower) => format!(".{name}_"),
    }
}

fn print_scalar(s: &Scalar) -> String {
    match s {
        Scalar::Int(n) => n.to_string(),
        Scalar::Real(r) => print_real(*r),
        Scalar::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Scalar::Str(s) => quote_string(s),
    }
}

/// Ensure a real prints with a `.`/`e` so it re-reads as a real (the shortest
/// repr of `2.0` is `"2"`, which would re-parse as an integer).
fn print_real(r: f64) -> String {
    let s = format!("{r}");
    if s.contains(['.', 'e', 'E']) || s.contains("inf") || s.contains("NaN") {
        s
    } else {
        format!("{s}.0")
    }
}

fn quote_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\0' => out.push_str("\\0"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

fn print_doc(doc: &Doc) -> String {
    let tag = match doc.markup {
        Markup::Md => "", // md is the default; omit the tag
        Markup::Typ => "typ",
    };
    if doc.lines.len() <= 1 {
        let content = doc.lines.first().map(|s| s.as_ref()).unwrap_or("");
        if tag.is_empty() {
            format!("% {content}")
        } else {
            format!("%{tag} {content}")
        }
    } else {
        let mut s = if tag.is_empty() {
            String::from("%%%\n")
        } else {
            format!("%%%{tag}\n")
        };
        for line in doc.lines.iter() {
            s.push_str(line);
            s.push('\n');
        }
        s.push_str("%%%");
        s
    }
}
