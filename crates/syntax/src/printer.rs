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
use std::rc::Rc;

use flatppl_core::{
    Axis, Call, CallHead, Doc, Inputs, Markup, Module, NamedArg, Node, NodeId, Ref, RefNs, Scalar,
    Symbol, Variance,
};

use crate::parser::is_placeholder;

/// Target line width for [`Syntax::Full`]. A statement whose single-line form
/// would exceed this is broken across lines at its outermost composition
/// boundary (call argument lists, dotted-broadcast operands, literals).
const WIDTH: usize = 100;
/// Continuation indent (spaces) per nesting level when a line is broken.
const INDENT: usize = 2;

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

/// A node viewed as an infix operator application, for line wrapping: the
/// surface operator spelling, its operands, and the precedence floors each
/// operand prints at (so a broken chain re-parses to the same tree).
struct Infix {
    op: String,
    lmin: u8,
    rmin: u8,
    left: NodeId,
    right: NodeId,
}

/// Rendering state belongs to an occurrence: the same node can be printed in
/// different precedence positions or under different lambda boundaries.
#[derive(Clone)]
struct PrintExpr {
    id: NodeId,
    min: u8,
    syntax: Syntax,
    lambda: Rc<[Symbol]>,
}

impl PrintExpr {
    fn child(&self, id: NodeId, min: u8) -> Self {
        Self {
            id,
            min,
            syntax: self.syntax,
            lambda: self.lambda.clone(),
        }
    }
}

enum PrintPiece {
    Text(String),
    Flat(PrintExpr),
    Wide {
        expr: PrintExpr,
        col: usize,
        indent: usize,
        bracketed: bool,
    },
}

impl PrintPiece {
    fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }
    fn child(expr: &PrintExpr, id: NodeId, min: u8) -> Self {
        Self::Flat(expr.child(id, min))
    }
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
            if let Node::Call(c) = module.node(b.rhs)
                && let CallHead::Builtin(op) = c.head
                && matches!(module.resolve(op), "load_module" | "standard_module")
            {
                modules.insert(b.name);
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
            if let CallHead::Builtin(op) = call.head
                && self.module.resolve(op) == "draw"
                && call.args.len() == 1
                && call.named.is_empty()
                && call.inputs.is_none()
            {
                let prefix = format!("{name} ~ ");
                let body = self.body_w(call.args[0], prefix.chars().count());
                return format!("{prefix}{body}");
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
        let prefix = format!("{name} = ");
        let body = self.body_w(binding.rhs, prefix.chars().count());
        format!("{prefix}{body}")
    }

    /// A statement's right-hand side, width-aware: [`Syntax::Full`] breaks an
    /// over-wide composition across lines (starting at column `col`, the width
    /// already consumed by the statement's `lhs = ` / `lhs ~ ` / `… :=` prefix);
    /// [`Syntax::Minimal`] always emits the flat linear form.
    fn body_w(&self, id: NodeId, col: usize) -> String {
        let expr = PrintExpr {
            id,
            min: EXPR,
            syntax: self.syntax,
            lambda: Rc::from([]),
        };
        match self.syntax {
            Syntax::Full => self.render(PrintPiece::Wide {
                expr,
                col,
                indent: 0,
                bracketed: false,
            }),
            Syntax::Minimal => self.render(PrintPiece::Flat(expr)),
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
        let prefix = format!("{name}[{axes}] := ");
        let body = self.body_w(call.args[2], prefix.chars().count());
        Some(format!("{prefix}{body}"))
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
        let prefix = format!("{metric}: {name}[{axes}] := ");
        let body = self.body_w(call.args[2], prefix.chars().count());
        Some(format!("{prefix}{body}"))
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

    // ---- expression output ----

    /// Stream text and expression continuations. Width probes consume this same
    /// flat stream only up to WIDTH; no rendered subtree strings are cached.
    fn render(&self, first: PrintPiece) -> String {
        let mut pending = vec![first];
        let mut out = String::new();
        while let Some(piece) = pending.pop() {
            match piece {
                PrintPiece::Text(text) => out.push_str(&text),
                PrintPiece::Flat(expr) => pending.extend(self.flat_pieces(&expr).into_iter().rev()),
                PrintPiece::Wide {
                    expr,
                    col,
                    indent,
                    bracketed,
                } => {
                    if col <= WIDTH && self.fits(&expr, WIDTH - col) {
                        pending.push(PrintPiece::Flat(expr));
                        continue;
                    }
                    let (_, prec) = self.expression_parts(&expr);
                    let grouped = prec < expr.min;
                    let inner_indent = if grouped { indent + INDENT } else { indent };
                    if let Some(mut parts) =
                        self.break_parts(&expr, inner_indent, bracketed || grouped)
                    {
                        if grouped {
                            parts.insert(
                                0,
                                PrintPiece::text(format!("(\n{}", " ".repeat(inner_indent))),
                            );
                            parts.push(PrintPiece::text(format!("\n{})", " ".repeat(indent))));
                        }
                        pending.extend(parts.into_iter().rev());
                    } else {
                        pending.push(PrintPiece::Flat(expr));
                    }
                }
            }
        }
        out
    }

    fn fits(&self, expr: &PrintExpr, mut remaining: usize) -> bool {
        let mut pending = vec![PrintPiece::Flat(expr.clone())];
        while let Some(piece) = pending.pop() {
            match piece {
                PrintPiece::Text(text) => {
                    let width = text.chars().take(remaining + 1).count();
                    if width > remaining {
                        return false;
                    }
                    remaining -= width;
                }
                PrintPiece::Flat(expr) => pending.extend(self.flat_pieces(&expr).into_iter().rev()),
                PrintPiece::Wide { .. } => {
                    unreachable!("flat width probes contain no layout requests")
                }
            }
        }
        true
    }

    fn flat_pieces(&self, expr: &PrintExpr) -> Vec<PrintPiece> {
        let (mut parts, prec) = self.expression_parts(expr);
        if expr.syntax == Syntax::Full && prec < expr.min {
            parts.insert(0, PrintPiece::text("("));
            parts.push(PrintPiece::text(")"));
        }
        parts
    }

    fn expression_parts(&self, expr: &PrintExpr) -> (Vec<PrintPiece>, u8) {
        let (text, prec) = match self.module.node(expr.id) {
            Node::Lit(s) => (print_scalar(s), scalar_prec(s)),
            Node::Const(sym) => (self.builtin_name(*sym), ATOM),
            Node::Hole => ("_".to_string(), ATOM),
            Node::Ref(r) => self.ref_form(r, &expr.lambda),
            Node::Axis(a) => (print_axis(self.module, a), ATOM),
            Node::Call(call) => return self.call_parts(expr, call),
        };
        (vec![PrintPiece::text(text)], prec)
    }

    fn ref_form(&self, r: &Ref, lambda: &[Symbol]) -> (String, u8) {
        if matches!(r.ns, RefNs::Local) && lambda.contains(&r.name) {
            let spelled = self.module.resolve(r.name);
            return (spelled[1..spelled.len() - 1].to_string(), ATOM);
        }
        match r.ns {
            RefNs::SelfMod | RefNs::Local => (print_ref(self.module, r), ATOM),
            RefNs::Module(_) => (print_ref(self.module, r), POSTFIX),
        }
    }

    fn call_parts(&self, expr: &PrintExpr, call: &Call) -> (Vec<PrintPiece>, u8) {
        if call.inputs.is_some() {
            if expr.syntax == Syntax::Full
                && let Some((head, lambda)) = self.lambda_head(call)
            {
                let body = PrintExpr {
                    id: call.args[0],
                    min: EXPR,
                    syntax: Syntax::Full,
                    lambda,
                };
                return (vec![PrintPiece::text(head), PrintPiece::Flat(body)], EXPR);
            }
            return (self.reified_parts(expr, call), POSTFIX);
        }

        if let CallHead::Builtin(op) = call.head {
            let name = self.module.resolve(op);
            if expr.syntax == Syntax::Full {
                if call.named.is_empty()
                    && call.args.len() == 2
                    && let Some(op) = binop(name)
                {
                    if op.prec == AND
                        && let Some(parts) = self.comparison_parts(expr)
                    {
                        return (parts, CMP);
                    }
                    let (lmin, rmin) = operand_mins(op.prec);
                    return (
                        vec![
                            PrintPiece::child(expr, call.args[0], lmin),
                            PrintPiece::text(format!(" {} ", op.plain)),
                            PrintPiece::child(expr, call.args[1], rmin),
                        ],
                        op.prec,
                    );
                }
                if call.named.is_empty()
                    && call.args.len() == 1
                    && let Some((plain, _)) = unop(name)
                {
                    return (
                        vec![
                            PrintPiece::text(plain),
                            PrintPiece::child(expr, call.args[0], UNARY),
                        ],
                        UNARY,
                    );
                }
                match name {
                    "get" if call.named.is_empty() && call.args.len() >= 2 => {
                        return (self.get_parts(expr, call), POSTFIX);
                    }
                    "broadcast" if call.args.len() >= 2 => return self.broadcast_parts(expr, call),
                    _ => {}
                }
            }
            let literal = match name {
                "vector" if call.named.is_empty() => Some(("[", "]")),
                "tuple" if call.named.is_empty() && call.args.len() >= 2 => Some(("(", ")")),
                _ => None,
            };
            if let Some((open, close)) = literal {
                let mut parts = vec![PrintPiece::text(open)];
                self.argument_parts(&mut parts, expr, &call.args, &[]);
                parts.push(PrintPiece::text(close));
                return (parts, ATOM);
            }
            let mut parts = vec![PrintPiece::text(format!("{}(", self.builtin_name(op)))];
            self.argument_parts(&mut parts, expr, &call.args, &call.named);
            parts.push(PrintPiece::text(")"));
            return (parts, POSTFIX);
        }
        let CallHead::User(callee) = call.head else {
            unreachable!()
        };
        let mut parts = vec![
            PrintPiece::child(expr, callee, POSTFIX),
            PrintPiece::text("("),
        ];
        self.argument_parts(&mut parts, expr, &call.args, &call.named);
        parts.push(PrintPiece::text(")"));
        (parts, POSTFIX)
    }

    fn argument_parts(
        &self,
        parts: &mut Vec<PrintPiece>,
        expr: &PrintExpr,
        positional: &[NodeId],
        named: &[NamedArg],
    ) {
        for (i, &arg) in positional.iter().enumerate() {
            if i > 0 {
                parts.push(PrintPiece::text(", "));
            }
            parts.push(PrintPiece::child(expr, arg, EXPR));
        }
        for (i, arg) in named.iter().enumerate() {
            if i > 0 || !positional.is_empty() {
                parts.push(PrintPiece::text(", "));
            }
            parts.push(PrintPiece::text(format!(
                "{} = ",
                self.module.resolve(arg.name)
            )));
            parts.push(PrintPiece::child(expr, arg.value, EXPR));
        }
    }

    fn get_parts(&self, expr: &PrintExpr, call: &Call) -> Vec<PrintPiece> {
        if call.args.len() == 2
            && let Node::Lit(Scalar::Str(key)) = self.module.node(call.args[1])
            && is_field_name(key)
            && !self.is_namespace(call.args[0])
        {
            let mut parts = self.dot_parts(expr, call.args[0]);
            parts.push(PrintPiece::text(format!(".{key}")));
            return parts;
        }
        let mut parts = vec![
            PrintPiece::child(expr, call.args[0], POSTFIX),
            PrintPiece::text("["),
        ];
        for (i, &arg) in call.args[1..].iter().enumerate() {
            if i > 0 {
                parts.push(PrintPiece::text(", "));
            }
            parts.push(match self.module.node(arg) {
                Node::Const(s) if self.module.resolve(*s) == "all" => PrintPiece::text(":"),
                Node::Const(s) if self.module.resolve(*s) == "only" => PrintPiece::text("!"),
                _ => PrintPiece::child(expr, arg, EXPR),
            });
        }
        parts.push(PrintPiece::text("]"));
        parts
    }

    fn is_namespace(&self, id: NodeId) -> bool {
        match self.module.node(id) {
            Node::Const(s) => matches!(self.module.resolve(*s), "self" | "base"),
            Node::Ref(r) => matches!(r.ns, RefNs::SelfMod) && self.modules.contains(&r.name),
            _ => false,
        }
    }

    /// Keep a following dot out of a nonnegative numeric literal's token.
    fn dot_parts(&self, expr: &PrintExpr, id: NodeId) -> Vec<PrintPiece> {
        let numeric = match self.module.node(id) {
            Node::Lit(Scalar::Int(n)) => *n >= 0,
            Node::Lit(Scalar::Real(r)) => !r.is_sign_negative(),
            _ => false,
        };
        let child = PrintPiece::child(expr, id, POSTFIX);
        if numeric {
            vec![PrintPiece::text("("), child, PrintPiece::text(")")]
        } else {
            vec![child]
        }
    }

    fn broadcast_parts(&self, expr: &PrintExpr, call: &Call) -> (Vec<PrintPiece>, u8) {
        if call.named.is_empty()
            && let Node::Const(f) = self.module.node(call.args[0])
        {
            let f = self.module.resolve(*f);
            if call.args.len() == 3
                && let Some(op) = binop(f)
                && let Some(dotted) = op.dotted
            {
                let (lmin, rmin) = operand_mins(op.prec);
                return (
                    vec![
                        PrintPiece::child(expr, call.args[1], lmin),
                        PrintPiece::text(format!(" {dotted} ")),
                        PrintPiece::child(expr, call.args[2], rmin),
                    ],
                    op.prec,
                );
            }
            if call.args.len() == 2
                && let Some((_, dotted)) = unop(f)
            {
                return (
                    vec![
                        PrintPiece::text(dotted),
                        PrintPiece::child(expr, call.args[1], UNARY),
                    ],
                    UNARY,
                );
            }
        }
        let mut parts = self.dot_parts(expr, call.args[0]);
        parts.push(PrintPiece::text(".("));
        self.argument_parts(&mut parts, expr, &call.args[1..], &call.named);
        parts.push(PrintPiece::text(")"));
        (parts, POSTFIX)
    }

    /// A lambda replaces the enclosing placeholder scope, never extends it.
    fn lambda_head(&self, call: &Call) -> Option<(String, Rc<[Symbol]>)> {
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
            if self.module.resolve(r.name) != format!("_{param}_") || !is_lambda_param_name(param) {
                return None;
            }
            params.push(param);
            placeholders.push(r.name);
        }
        let head = if params.len() == 1 {
            params[0].to_string()
        } else {
            format!("({})", params.join(", "))
        };
        Some((format!("{head} -> "), placeholders.into()))
    }

    fn reified_parts(&self, expr: &PrintExpr, call: &Call) -> Vec<PrintPiece> {
        let body = PrintExpr {
            id: call.args[0],
            min: EXPR,
            syntax: expr.syntax,
            lambda: Rc::from([]),
        };
        let mut parts = match call.head {
            CallHead::Builtin(op) => vec![PrintPiece::text(self.module.resolve(op))],
            CallHead::User(callee) => vec![PrintPiece::child(&body, callee, EXPR)],
        };
        parts.push(PrintPiece::text("("));
        parts.push(PrintPiece::Flat(body));
        if let Some(Inputs::Spec(entries)) = &call.inputs {
            for (name, r) in entries.iter() {
                parts.push(PrintPiece::text(format!(
                    ", {} = {}",
                    self.module.resolve(*name),
                    print_ref(self.module, r)
                )));
            }
        }
        parts.push(PrintPiece::text(")"));
        parts
    }

    fn comparison_parts(&self, expr: &PrintExpr) -> Option<Vec<PrintPiece>> {
        let mut elems = Vec::new();
        let mut id = expr.id;
        // Reverse the left spine once to retain the original comparison order.
        loop {
            if let Node::Call(c) = self.module.node(id)
                && let CallHead::Builtin(op) = c.head
                && self.module.resolve(op) == "land"
                && c.args.len() == 2
                && c.named.is_empty()
                && c.inputs.is_none()
            {
                elems.push(c.args[1]);
                id = c.args[0];
            } else {
                elems.push(id);
                break;
            }
        }
        if elems.len() < 2 {
            return None;
        }
        elems.reverse();
        let mut cmps = Vec::with_capacity(elems.len());
        for id in elems {
            let Node::Call(c) = self.module.node(id) else {
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
            if !self.module.structural_eq(pair[0].2, pair[1].1) {
                return None;
            }
        }
        let mut parts = vec![PrintPiece::child(expr, cmps[0].1, ADD)];
        for (op, _, rhs) in cmps {
            parts.push(PrintPiece::text(format!(" {op} ")));
            parts.push(PrintPiece::child(expr, rhs, ADD));
        }
        Some(parts)
    }

    // ---- width-aware layout ----

    /// None means that this shape stays flat. Every returned layout contains
    /// a newline, so the caller need not retain an ancestor's flat fallback.
    fn break_parts(
        &self,
        expr: &PrintExpr,
        indent: usize,
        bracketed: bool,
    ) -> Option<Vec<PrintPiece>> {
        if bracketed && let Some(inf) = self.as_infix(expr.id) {
            let mut elems = Vec::new();
            let mut id = expr.id;
            while let Some(next) = self.as_infix(id)
                && next.op == inf.op
            {
                elems.push(next.right);
                id = next.left;
            }
            elems.push(id);
            elems.reverse();
            let mut parts = vec![PrintPiece::Wide {
                expr: expr.child(elems[0], inf.lmin),
                col: indent,
                indent,
                bracketed: true,
            }];
            let opcol = indent + inf.op.chars().count() + 1;
            for &id in &elems[1..] {
                parts.push(PrintPiece::text(format!(
                    "\n{}{} ",
                    " ".repeat(indent),
                    inf.op
                )));
                parts.push(PrintPiece::Wide {
                    expr: expr.child(id, inf.rmin),
                    col: opcol,
                    indent,
                    bracketed: true,
                });
            }
            return Some(parts);
        }
        let Node::Call(call) = self.module.node(expr.id) else {
            return None;
        };
        if call.inputs.is_some() {
            return None;
        }
        let (mut parts, close, args) = match call.head {
            CallHead::Builtin(op) => match self.module.resolve(op) {
                "vector" if call.named.is_empty() => {
                    (vec![PrintPiece::text("[")], "]", &call.args[..])
                }
                "tuple" if call.named.is_empty() && call.args.len() >= 2 => {
                    (vec![PrintPiece::text("(")], ")", &call.args[..])
                }
                "broadcast" if call.args.len() >= 2 => {
                    let mut head = self.dot_parts(expr, call.args[0]);
                    head.push(PrintPiece::text(".("));
                    (head, ")", &call.args[1..])
                }
                "get" => return None,
                _ => (
                    vec![PrintPiece::text(format!("{}(", self.builtin_name(op)))],
                    ")",
                    &call.args[..],
                ),
            },
            CallHead::User(callee) => (
                vec![
                    PrintPiece::child(expr, callee, POSTFIX),
                    PrintPiece::text("("),
                ],
                ")",
                &call.args[..],
            ),
        };
        let child = indent + INDENT;
        parts.push(PrintPiece::text("\n"));
        for &id in args {
            parts.push(PrintPiece::text(" ".repeat(child)));
            parts.push(PrintPiece::Wide {
                expr: expr.child(id, EXPR),
                col: child,
                indent: child,
                bracketed: true,
            });
            parts.push(PrintPiece::text(",\n"));
        }
        for named in &call.named {
            let prefix = format!("{} = ", self.module.resolve(named.name));
            let col = child + prefix.chars().count();
            parts.push(PrintPiece::text(format!("{}{prefix}", " ".repeat(child))));
            parts.push(PrintPiece::Wide {
                expr: expr.child(named.value, EXPR),
                col,
                indent: child,
                bracketed: true,
            });
            parts.push(PrintPiece::text(",\n"));
        }
        parts.push(PrintPiece::text(format!("{}{close}", " ".repeat(indent))));
        Some(parts)
    }

    /// View a node as a breakable infix operator application: a plain binary
    /// builtin (`add`/`mul`/…) or a dotted broadcast `broadcast(op, l, r)`.
    /// `land` is excluded — a 2-arg `land` may be a comparison-chain sugar
    /// (`a < b < c`), and chains are short enough never to wrap.
    fn as_infix(&self, id: NodeId) -> Option<Infix> {
        let Node::Call(c) = self.module.node(id) else {
            return None;
        };
        if !c.named.is_empty() || c.inputs.is_some() {
            return None;
        }
        let CallHead::Builtin(op) = c.head else {
            return None;
        };
        let name = self.module.resolve(op);
        if c.args.len() == 2
            && let Some(b) = binop(name)
        {
            if b.prec == AND {
                return None;
            }
            let (lmin, rmin) = operand_mins(b.prec);
            return Some(Infix {
                op: b.plain.to_string(),
                lmin,
                rmin,
                left: c.args[0],
                right: c.args[1],
            });
        }
        if name == "broadcast"
            && c.args.len() == 3
            && let Node::Const(f) = self.module.node(c.args[0])
            && let Some(b) = binop(self.module.resolve(*f))
            && let Some(dotted) = b.dotted
        {
            let (lmin, rmin) = operand_mins(b.prec);
            return Some(Infix {
                op: dotted.to_string(),
                lmin,
                rmin,
                left: c.args[1],
                right: c.args[2],
            });
        }
        None
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
