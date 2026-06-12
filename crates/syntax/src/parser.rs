//! Parse canonical FlatPPL surface text into a [`Module`], stripping syntactic
//! sugar (spec §05 → §04 lowered form).
//!
//! Two passes over the statement list: a **name pre-pass** collects every
//! binding's LHS name (FlatPPL is order-irrelevant, so a call `f(x)` can't be
//! classified built-in-vs-user until all names are known), then each statement
//! is parsed and lowered with that name set in hand. A call head bound in the
//! module lowers to `(%call (%ref self f) …)`; otherwise it is a bare built-in.
//!
//! **Covered:** the full spec §05 statement grammar — `=` / `~` bindings,
//! decomposition, `:=` sum-aggregate and `metric: …` metricsum statements,
//! calls (positional + keyword), operators with precedence + comparison
//! chaining (incl. `in` membership), member access (`self.` / `base.` /
//! `mod.`), field access and indexing with `:` / `!` slicing, axis names with
//! variance markers, arrays / records / tuples, dot-broadcast (`a .+ b`,
//! `f.(x)`), lambdas and `fn`/holes (both lower to `functionof`), and the
//! reification ops `functionof` / `kernelof` / `lawof` as **un-traced
//! construct calls**. Printer operator/indexing re-sugaring is a deferred
//! enhancement (the lowered linear form is canonical).

use std::collections::HashSet;

use flatppl_core::{
    Axis, Binding, Call, CallHead, Doc as CoreDoc, Inputs, Markup, Module, NamedArg, NamedKind,
    Node, NodeId, Ref, RefNs, Scalar, Variance,
};

use crate::error::{Error, Result};
use crate::token::{self, Doc, Token, TokenKind};

/// An error anchored to `tok`'s span (widened to one byte for the
/// zero-width Eof token).
fn err_at(tok: &Token, message: impl Into<String>) -> Error {
    Error::at_span(tok.line, (tok.start, tok.end.max(tok.start + 1)), message)
}

/// Parse canonical FlatPPL surface text into a [`Module`].
pub fn parse(input: &str) -> Result<Module> {
    let tokens = token::tokenize(input)?;
    let statements = split_statements(&tokens)?;

    // Name pre-pass: every binding LHS name (for user-vs-builtin call
    // resolution) and every module-binding name (RHS head `load_module` /
    // `standard_module` — member access `m.x` resolves to a cross-module ref).
    let mut names = Names::default();
    for st in &statements {
        collect_lhs_names(st, &mut names);
    }

    let mut module = Module::new();
    let mut synth: u32 = 0;
    for st in &statements {
        lower_statement(&mut module, st, &names, &mut synth)?;
    }
    Ok(module)
}

/// The pre-pass result: which names the module binds, and which of those are
/// module bindings (their RHS is a `load_module` / `standard_module` call).
#[derive(Default)]
struct Names {
    bound: HashSet<String>,
    modules: HashSet<String>,
}

/// One surface statement: its body tokens (no separators, no doc-comments) plus
/// an optional attached doc-comment (leading or trailing).
struct Stmt {
    tokens: Vec<Token>,
    doc: Option<Doc>,
}

fn split_statements(tokens: &[Token]) -> Result<Vec<Stmt>> {
    let mut statements = Vec::new();
    let mut cur: Vec<Token> = Vec::new();
    let mut trailing: Option<Doc> = None;
    let mut pending_leading: Option<Doc> = None;

    let flush = |cur: &mut Vec<Token>,
                 trailing: &mut Option<Doc>,
                 pending: &mut Option<Doc>,
                 out: &mut Vec<Stmt>| {
        if cur.is_empty() {
            return;
        }
        let doc = trailing.take().or_else(|| pending.take());
        out.push(Stmt {
            tokens: std::mem::take(cur),
            doc,
        });
    };

    for tok in tokens {
        match &tok.kind {
            TokenKind::Eof => break,
            TokenKind::Newline | TokenKind::Semi => {
                flush(
                    &mut cur,
                    &mut trailing,
                    &mut pending_leading,
                    &mut statements,
                );
            }
            // A binding carries at most ONE doc-comment, leading or trailing
            // but not both (spec §04 Documentation) — accepting more would
            // silently drop content.
            TokenKind::Doc(doc) => {
                if doc.trailing {
                    if pending_leading.is_some() || trailing.is_some() {
                        return Err(err_at(
                            tok,
                            "a binding may carry at most one doc-comment \
                             (leading or trailing, not both)",
                        ));
                    }
                    trailing = Some(doc.clone());
                } else if !cur.is_empty() {
                    return Err(err_at(tok, "a doc-comment may not interrupt a statement"));
                } else if pending_leading.is_some() {
                    return Err(err_at(
                        tok,
                        "only one doc-comment may precede a binding \
                         (use a `%%%` block for multi-line content)",
                    ));
                } else {
                    // A standalone leading doc-comment; attaches to the next stmt.
                    pending_leading = Some(doc.clone());
                }
            }
            _ => cur.push(tok.clone()),
        }
    }
    flush(
        &mut cur,
        &mut trailing,
        &mut pending_leading,
        &mut statements,
    );
    Ok(statements)
}

/// Collect the binding name(s) on a statement's LHS: the leading
/// `Name (',' Name)*` run, up to the binding operator / `[` / `:`.
/// `_` is the discard name, not a binding. A `metric: result[…] := …`
/// statement binds the name *after* the colon (the metric is a reference).
fn collect_lhs_names(stmt: &Stmt, names: &mut Names) {
    let toks = &stmt.tokens;

    // Metricsum statement: `Name ':' Name '[' …` — only the second name binds.
    if matches!(toks.first().map(|t| &t.kind), Some(TokenKind::Name(_)))
        && matches!(toks.get(1).map(|t| &t.kind), Some(TokenKind::Colon))
    {
        if let Some(TokenKind::Name(result)) = toks.get(2).map(|t| &t.kind) {
            if result != "_" {
                names.bound.insert(result.clone());
            }
        }
        return;
    }

    let mut count = 0;
    let mut i = 0;
    while let Some(TokenKind::Name(n)) = toks.get(i).map(|t| &t.kind) {
        if n != "_" {
            names.bound.insert(n.clone());
        }
        count += 1;
        i += 1;
        if matches!(toks.get(i).map(|t| &t.kind), Some(TokenKind::Comma)) {
            i += 1;
        } else {
            break;
        }
    }

    // `m = load_module(…)` / `m = standard_module(…)` makes `m` a module binding.
    if count == 1
        && matches!(toks.get(1).map(|t| &t.kind), Some(TokenKind::Assign))
        && matches!(toks.get(2).map(|t| &t.kind),
            Some(TokenKind::Name(h)) if h == "load_module" || h == "standard_module")
        && matches!(toks.get(3).map(|t| &t.kind), Some(TokenKind::LParen))
    {
        if let Some(TokenKind::Name(m)) = toks.first().map(|t| &t.kind) {
            names.modules.insert(m.clone());
        }
    }
}

fn lower_statement(module: &mut Module, stmt: &Stmt, names: &Names, synth: &mut u32) -> Result<()> {
    let toks = &stmt.tokens;

    // LHS: leading `Name (',' Name)*`, then the binding operator. The token
    // of each name rides along to anchor binding-name diagnostics.
    let mut lhs = Vec::new();
    let mut lhs_toks = Vec::new();
    let mut i = 0;
    while let Some(TokenKind::Name(n)) = toks.get(i).map(|t| &t.kind) {
        lhs.push(n.clone());
        lhs_toks.push(&toks[i]);
        i += 1;
        if matches!(toks.get(i).map(|t| &t.kind), Some(TokenKind::Comma)) {
            i += 1;
        } else {
            break;
        }
    }
    match toks.get(i).map(|t| &t.kind) {
        // `C[.i, .k] := expr` — sum-aggregate.
        Some(TokenKind::LBracket) => {
            if lhs.len() != 1 {
                return Err(err_at(
                    &toks[0],
                    "an aggregate target must be a single name",
                ));
            }
            check_binding_name(&lhs[0], lhs_toks[0])?;
            let mut ep = ExprParser::new(&toks[i..], module, names);
            let rhs = ep.parse_aggregate()?;
            ep.expect_end()?;
            bind_name(module, &lhs[0], rhs, stmt.doc.as_ref(), synth);
            Ok(())
        }
        // `metric: result[.axes…] := expr` — metric-aware Einstein summation
        // (spec §04), lowering to `metricsum(metric, [.axes…], expr)`. The
        // leading name is a *reference* to the metric, not a binding.
        Some(TokenKind::Colon) => {
            if lhs.len() != 1 {
                return Err(err_at(&toks[0], "a metricsum metric must be a single name"));
            }
            let mut ep = ExprParser::new(&toks[i + 1..], module, names);
            let result = ep.expect_name("a result name after the metric `:`")?;
            check_binding_name(&result, ep.prev_token().unwrap_or(&toks[0]))?;
            let rhs = ep.parse_metricsum(&lhs[0])?;
            ep.expect_end()?;
            bind_name(module, &result, rhs, stmt.doc.as_ref(), synth);
            Ok(())
        }
        Some(op @ (TokenKind::Assign | TokenKind::Tilde)) => {
            for (name, tok) in lhs.iter().zip(&lhs_toks) {
                check_binding_name(name, tok)?;
            }
            let is_tilde = matches!(op, TokenKind::Tilde);
            let mut ep = ExprParser::new(&toks[i + 1..], module, names);
            let mut rhs = ep.parse_expr()?;
            ep.expect_end()?;
            if is_tilde {
                rhs = wrap_draw(module, rhs);
            }
            if lhs.len() == 1 {
                bind_name(module, &lhs[0], rhs, stmt.doc.as_ref(), synth);
            } else if lhs.is_empty() {
                return Err(err_at(&toks[0], "binding has no name"));
            } else {
                lower_decomposition(module, &lhs, rhs, synth);
            }
            Ok(())
        }
        _ => Err(err_at(
            toks.get(i).unwrap_or(&toks[0]),
            "expected `=`, `~`, or `[…] :=` after the binding name",
        )),
    }
}

/// Reject names that cannot be bound: reserved words (spec §05), the reserved
/// modules `self` / `base` (spec §04), and the placeholder lexical class
/// (spec §04 binding names). `_` is allowed — it discards (see [`bind_name`]).
fn check_binding_name(name: &str, tok: &Token) -> Result<()> {
    match name {
        "_" => Ok(()),
        "true" | "false" | "in" | "all" | "only" => Err(err_at(
            tok,
            format!("`{name}` is a reserved word and cannot be bound"),
        )),
        "self" | "base" => Err(err_at(
            tok,
            format!("`{name}` is a reserved module name and cannot be bound"),
        )),
        _ if is_placeholder(name) => Err(err_at(
            tok,
            format!("placeholder `{name}` cannot be a module-level binding"),
        )),
        _ => Ok(()),
    }
}

/// Bind `rhs` to `name`; `_` discards by binding to a fresh auto-generated
/// private name instead (spec §04 binding names).
fn bind_name(module: &mut Module, name: &str, rhs: NodeId, doc: Option<&Doc>, synth: &mut u32) {
    if name == "_" {
        *synth += 1;
        let tmp = format!("__0x{:x}", *synth);
        let tmp_sym = module.intern(&tmp);
        module.add_binding(Binding {
            name: tmp_sym,
            rhs,
            doc: doc.map(lower_doc),
            public: false,
            synthetic: true,
        });
    } else {
        add_simple_binding(module, name, rhs, doc);
    }
}

/// Wrap a measure node in `draw(…)` (the lowering of a `~` binding).
fn wrap_draw(module: &mut Module, measure: NodeId) -> NodeId {
    let draw = CallHead::Builtin(module.intern("draw"));
    module.alloc(Node::Call(Call {
        head: draw,
        args: Box::new([measure]),
        named: Box::new([]),
        inputs: None,
    }))
}

fn add_simple_binding(module: &mut Module, name: &str, rhs: NodeId, doc: Option<&Doc>) {
    let public = !name.starts_with('_');
    let name_sym = module.intern(name);
    let doc = doc.map(lower_doc);
    module.add_binding(Binding {
        name: name_sym,
        rhs,
        doc,
        public,
        synthetic: false,
    });
}

/// `a, b, _ = expr`: bind the source to a fresh synthetic name, then project
/// each non-`_` target by 1-based position (`get(tmp, k)`). The shared `tmp`
/// keeps every projection reading the *same* value — essential when the source
/// is stochastic (re-evaluating it would redraw).
fn lower_decomposition(module: &mut Module, names: &[String], source: NodeId, synth: &mut u32) {
    *synth += 1;
    let tmp_name = format!("__0x{:x}", *synth);
    let tmp_sym = module.intern(&tmp_name);
    module.add_binding(Binding {
        name: tmp_sym,
        rhs: source,
        doc: None,
        public: false,
        synthetic: true,
    });

    for (k, name) in names.iter().enumerate() {
        if name == "_" {
            continue; // discard this component
        }
        let idx = module.alloc(Node::Lit(Scalar::Int((k + 1) as i64)));
        let tmp_ref = module.alloc(Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name: tmp_sym,
        }));
        let get = CallHead::Builtin(module.intern("get"));
        let proj = module.alloc(Node::Call(Call {
            head: get,
            args: Box::new([tmp_ref, idx]),
            named: Box::new([]),
            inputs: None,
        }));
        add_simple_binding(module, name, proj, None);
    }
}

fn lower_doc(doc: &Doc) -> CoreDoc {
    let markup = match doc.tag.as_deref() {
        Some("typ") => Markup::Typ,
        _ => Markup::Md,
    };
    CoreDoc {
        markup,
        lines: doc
            .lines
            .iter()
            .map(|l| l.clone().into_boxed_str())
            .collect(),
    }
}

/// Recursive-descent / precedence-climbing expression parser. Lowers directly
/// into `module`'s node arena as it parses.
struct ExprParser<'a> {
    toks: &'a [Token],
    pos: usize,
    module: &'a mut Module,
    names: &'a Names,
    /// The enclosing reification frames (innermost last). Placeholders are
    /// scoped to the **nearest** enclosing reification (spec §04), so the
    /// parser-introduced ones (lambda args, holes) resolve only in the
    /// innermost frame; a match in an outer frame is the spec's DISALLOWED
    /// cross-reification capture and errors.
    reify_frames: Vec<ReifyFrame>,
}

/// One enclosing reification while its body is being parsed. Every variant is
/// a placeholder-scope boundary; lambdas and `fn` additionally bind names.
enum ReifyFrame {
    /// Lambda argument names: occurrences lower to the placeholder `_name_`.
    Lambda(Vec<String>),
    /// `fn(…)`: collects one `(arg<n>, %local _arg<n>_)` entry per `_` hole.
    Fn(Vec<(flatppl_core::Symbol, Ref)>),
    /// An explicit `functionof(…)` / `kernelof(…)` call. Binds nothing the
    /// parser can know (its boundary kwargs come after the body and are
    /// validated later), but still cuts off outer lambda args and holes.
    Explicit,
}

/// The namespace a `<name>.<member>` access goes through (spec §04 name
/// resolution): the current module, the built-ins, or a loaded module.
enum MemberNs {
    SelfMod,
    Base,
    Module(flatppl_core::Symbol),
}

impl<'a> ExprParser<'a> {
    fn new(toks: &'a [Token], module: &'a mut Module, names: &'a Names) -> Self {
        ExprParser {
            toks,
            pos: 0,
            module,
            names,
            reify_frames: Vec::new(),
        }
    }

    fn peek(&self) -> Option<&TokenKind> {
        self.toks.get(self.pos).map(|t| &t.kind)
    }

    fn peek_at(&self, n: usize) -> Option<&TokenKind> {
        self.toks.get(self.pos + n).map(|t| &t.kind)
    }

    /// An error anchored to the token at the cursor (or the last token when
    /// the cursor has run off the end of the slice).
    fn err_here(&self, message: impl Into<String>) -> Error {
        match self.toks.get(self.pos).or_else(|| self.toks.last()) {
            Some(t) => err_at(t, message),
            None => Error::new(message),
        }
    }

    /// An error anchored to the most recently consumed token — for
    /// "this thing you just wrote is invalid" diagnostics.
    fn err_prev(&self, message: impl Into<String>) -> Error {
        match self.prev_token() {
            Some(t) => err_at(t, message),
            None => Error::new(message),
        }
    }

    /// The most recently consumed token, if any.
    fn prev_token(&self) -> Option<&Token> {
        self.toks.get(self.pos.saturating_sub(1))
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn eat(&mut self, kind: &TokenKind) -> bool {
        if self.peek() == Some(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: &TokenKind, what: &str) -> Result<()> {
        if self.eat(kind) {
            Ok(())
        } else {
            Err(self.err_here(format!("expected {what}")))
        }
    }

    fn expect_end(&self) -> Result<()> {
        if self.pos >= self.toks.len() {
            Ok(())
        } else {
            Err(self.err_here("unexpected trailing tokens in expression"))
        }
    }

    // ---- node constructors ----

    fn builtin_call(&mut self, op: &str, args: Vec<NodeId>) -> NodeId {
        let head = CallHead::Builtin(self.module.intern(op));
        self.module.alloc(Node::Call(Call {
            head,
            args: args.into(),
            named: Box::new([]),
            inputs: None,
        }))
    }

    // ---- precedence levels (low → high) ----

    fn parse_expr(&mut self) -> Result<NodeId> {
        // Lambdas sit at the lowest precedence; the body extends as far right
        // as possible (spec §05).
        if let Some(params) = self.try_lambda_params()? {
            return self.lower_lambda(params);
        }
        self.parse_or()
    }

    /// Bounded lookahead for a lambda head — `name ->` or `(name, name, …) ->`
    /// (spec §05). Consumes the head through `->` and returns the parameter
    /// names iff it is one; otherwise consumes nothing.
    fn try_lambda_params(&mut self) -> Result<Option<Vec<String>>> {
        if let (Some(TokenKind::Name(n)), Some(TokenKind::Arrow)) = (self.peek(), self.peek_at(1)) {
            let n = n.clone();
            self.check_lambda_param(&n)?;
            self.advance();
            self.advance();
            return Ok(Some(vec![n]));
        }
        if !matches!(self.peek(), Some(TokenKind::LParen)) {
            return Ok(None);
        }
        // Scan `( Name (, Name)* ) ->` without consuming.
        let mut k = 1;
        let mut params = Vec::new();
        loop {
            match self.peek_at(k) {
                Some(TokenKind::Name(n)) => {
                    params.push(n.clone());
                    k += 1;
                }
                _ => return Ok(None),
            }
            match self.peek_at(k) {
                Some(TokenKind::Comma) => k += 1,
                Some(TokenKind::RParen) => {
                    k += 1;
                    break;
                }
                _ => return Ok(None),
            }
        }
        if !matches!(self.peek_at(k), Some(TokenKind::Arrow)) {
            return Ok(None);
        }
        if params.len() == 1 {
            return Err(
                self.err_here("`(arg) -> expr` is not valid lambda syntax; write `arg -> expr`")
            );
        }
        for p in &params {
            self.check_lambda_param(p)?;
        }
        self.pos += k + 1; // consume `( … ) ->`
        Ok(Some(params))
    }

    fn check_lambda_param(&self, name: &str) -> Result<()> {
        let reserved = matches!(
            name,
            "_" | "true" | "false" | "in" | "all" | "only" | "self" | "base"
        );
        if reserved || is_placeholder(name) {
            return Err(self.err_here(format!("`{name}` cannot be a lambda argument name")));
        }
        Ok(())
    }

    /// Lambda sugar (spec §04): `x -> expr` resolves to
    /// `functionof(expr', x = _x_)` where free occurrences of `x` in `expr`
    /// are rewritten to the placeholder `_x_`.
    fn lower_lambda(&mut self, params: Vec<String>) -> Result<NodeId> {
        self.reify_frames.push(ReifyFrame::Lambda(params.clone()));
        let body = self.parse_expr();
        self.reify_frames.pop();
        let body = body?;

        let mut entries = Vec::with_capacity(params.len());
        for p in &params {
            let name_sym = self.module.intern(p);
            let ph_sym = self.module.intern(&format!("_{p}_"));
            entries.push((
                name_sym,
                Ref {
                    ns: RefNs::Local,
                    name: ph_sym,
                },
            ));
        }
        let head = CallHead::Builtin(self.module.intern("functionof"));
        Ok(self.module.alloc(Node::Call(Call {
            head,
            args: Box::new([body]),
            named: Box::new([]),
            inputs: Some(Inputs::Spec(entries.into())),
        })))
    }

    /// `fn(expr)` hole sugar (spec §04): each `_` in `expr` becomes a distinct
    /// positional input `arg<n>` in left-to-right reading order, lowering to
    /// `functionof(expr', arg1 = _arg1_, …)`. The `fn(…)` call delimits the
    /// hole scope. The opening `(` has already been consumed.
    fn lower_fn(&mut self) -> Result<NodeId> {
        self.reify_frames.push(ReifyFrame::Fn(Vec::new()));
        let body = self.parse_expr();
        let Some(ReifyFrame::Fn(entries)) = self.reify_frames.pop() else {
            unreachable!("Fn frame pushed above");
        };
        let body = body?;
        self.expect(&TokenKind::RParen, "`)` to close `fn(…)`")?;
        if entries.is_empty() {
            return Err(self.err_prev("`fn(…)` requires at least one `_` hole in its expression"));
        }
        let head = CallHead::Builtin(self.module.intern("functionof"));
        Ok(self.module.alloc(Node::Call(Call {
            head,
            args: Box::new([body]),
            named: Box::new([]),
            inputs: Some(Inputs::Spec(entries.into())),
        })))
    }

    fn parse_or(&mut self) -> Result<NodeId> {
        let mut lhs = self.parse_and()?;
        loop {
            match self.peek() {
                Some(TokenKind::PipePipe) => {
                    self.advance();
                    let rhs = self.parse_and()?;
                    lhs = self.builtin_call("lor", vec![lhs, rhs]);
                }
                Some(TokenKind::DotPipePipe) => {
                    self.advance();
                    let rhs = self.parse_and()?;
                    lhs = self.broadcast_op("lor", lhs, rhs);
                }
                _ => return Ok(lhs),
            }
        }
    }

    fn parse_and(&mut self) -> Result<NodeId> {
        let mut lhs = self.parse_cmp()?;
        loop {
            match self.peek() {
                Some(TokenKind::AmpAmp) => {
                    self.advance();
                    let rhs = self.parse_cmp()?;
                    lhs = self.builtin_call("land", vec![lhs, rhs]);
                }
                Some(TokenKind::DotAmpAmp) => {
                    self.advance();
                    let rhs = self.parse_cmp()?;
                    lhs = self.broadcast_op("land", lhs, rhs);
                }
                _ => return Ok(lhs),
            }
        }
    }

    /// Comparisons chain: `a < b <= c` lowers to `land(lt(a,b), le(b,c))`.
    fn parse_cmp(&mut self) -> Result<NodeId> {
        let first = self.parse_add()?;
        let mut operands = vec![first];
        let mut ops: Vec<&'static str> = Vec::new();
        let mut dotted = false;
        while let Some(op) = self.peek().and_then(cmp_op) {
            if op.dotted {
                dotted = true;
            }
            self.advance();
            let rhs = self.parse_add()?;
            ops.push(op.func);
            operands.push(rhs);
        }
        if ops.is_empty() {
            return Ok(operands.pop().unwrap());
        }
        if ops.len() == 1 {
            let rhs = operands.pop().unwrap();
            let lhs = operands.pop().unwrap();
            return Ok(if dotted {
                self.broadcast_op(ops[0], lhs, rhs)
            } else {
                self.builtin_call(ops[0], vec![lhs, rhs])
            });
        }
        // Chained: AND together each adjacent comparison (plain only; dotted
        // chains are unusual and not lowered here).
        if dotted {
            return Err(self.err_here("chained dotted comparisons are not supported"));
        }
        let mut terms = Vec::with_capacity(ops.len());
        for (k, func) in ops.iter().enumerate() {
            let pair = self.builtin_call(func, vec![operands[k], operands[k + 1]]);
            terms.push(pair);
        }
        let mut acc = terms[0];
        for &t in &terms[1..] {
            acc = self.builtin_call("land", vec![acc, t]);
        }
        Ok(acc)
    }

    fn parse_add(&mut self) -> Result<NodeId> {
        let mut lhs = self.parse_mul()?;
        loop {
            let (func, dotted) = match self.peek() {
                Some(TokenKind::Plus) => ("add", false),
                Some(TokenKind::Minus) => ("sub", false),
                Some(TokenKind::DotPlus) => ("add", true),
                Some(TokenKind::DotMinus) => ("sub", true),
                _ => return Ok(lhs),
            };
            self.advance();
            let rhs = self.parse_mul()?;
            lhs = self.apply_binop(func, dotted, lhs, rhs);
        }
    }

    fn parse_mul(&mut self) -> Result<NodeId> {
        let mut lhs = self.parse_unary()?;
        loop {
            let (func, dotted) = match self.peek() {
                Some(TokenKind::Star) => ("mul", false),
                Some(TokenKind::Slash) => ("divide", false),
                Some(TokenKind::DotStar) => ("mul", true),
                Some(TokenKind::DotSlash) => ("divide", true),
                _ => return Ok(lhs),
            };
            self.advance();
            let rhs = self.parse_unary()?;
            lhs = self.apply_binop(func, dotted, lhs, rhs);
        }
    }

    fn parse_unary(&mut self) -> Result<NodeId> {
        match self.peek() {
            Some(TokenKind::Minus) => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(self.builtin_call("neg", vec![operand]))
            }
            Some(TokenKind::Bang) => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(self.builtin_call("lnot", vec![operand]))
            }
            Some(TokenKind::DotMinus) => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(self.broadcast_unop("neg", operand))
            }
            Some(TokenKind::DotBang) => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(self.broadcast_unop("lnot", operand))
            }
            _ => self.parse_exp(),
        }
    }

    /// `^` is right-associative and binds tighter than unary minus (its RHS is a
    /// `Unary`, so `a ^ -b` and `a ^ b ^ c` parse as in the spec).
    fn parse_exp(&mut self) -> Result<NodeId> {
        let base = self.parse_postfix()?;
        match self.peek() {
            Some(TokenKind::Caret) => {
                self.advance();
                let exp = self.parse_unary()?;
                Ok(self.builtin_call("pow", vec![base, exp]))
            }
            Some(TokenKind::DotCaret) => {
                self.advance();
                let exp = self.parse_unary()?;
                Ok(self.broadcast_op("pow", base, exp))
            }
            _ => Ok(base),
        }
    }

    fn parse_postfix(&mut self) -> Result<NodeId> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek() {
                Some(TokenKind::Dot) => {
                    self.advance();
                    let name = self.expect_name("a member or field name after `.`")?;
                    // Module-member access (`self.x`, `base.x`, `mod.x`) is a
                    // separate syntactic category from field access: modules
                    // are namespaces, so the access lowers to a ref or call
                    // head, never to `get` (spec §04 / §07).
                    if let Some(ns) = self.member_namespace(expr) {
                        expr = self.lower_member(ns, &name)?;
                    } else {
                        // Field access `obj.name` → `get(obj, "name")`.
                        let key = self
                            .module
                            .alloc(Node::Lit(Scalar::Str(name.into_boxed_str())));
                        expr = self.builtin_call("get", vec![expr, key]);
                    }
                }
                Some(TokenKind::LBracket) => {
                    // Indexing `obj[i, …]` → `get(obj, i, …)`.
                    self.advance();
                    let mut args = vec![expr];
                    self.parse_index_args(&mut args)?;
                    self.expect(&TokenKind::RBracket, "`]` to close indexing")?;
                    expr = self.builtin_call("get", args);
                }
                Some(TokenKind::DotLParen) => {
                    // Dot-call `f.(args)` → `broadcast(f, args…)`.
                    self.advance();
                    let (mut positional, named) = self.parse_call_args()?;
                    self.expect(&TokenKind::RParen, "`)` to close dot-call")?;
                    let mut args = vec![expr];
                    args.append(&mut positional);
                    let head = CallHead::Builtin(self.module.intern("broadcast"));
                    expr = self.module.alloc(Node::Call(Call {
                        head,
                        args: args.into(),
                        named: named.into(),
                        inputs: None,
                    }));
                }
                Some(TokenKind::LParen) => {
                    // Postfix application (spec §05 `Postfix Call`): the
                    // expression so far is the callee — `(%call <callable> …)`
                    // with an expression head (spec §11). Named-callee calls
                    // never reach here (consumed in `parse_name_tail` /
                    // `lower_member`); this covers inline callables like
                    // `functionof(…)(v)` and chained calls `f(x)(y)`.
                    self.advance();
                    let (positional, named) = self.parse_call_args()?;
                    self.expect(&TokenKind::RParen, "`)` to close the call")?;
                    expr = self.make_applied_call(expr, positional, named);
                }
                _ => return Ok(expr),
            }
        }
    }

    fn parse_index_args(&mut self, args: &mut Vec<NodeId>) -> Result<()> {
        loop {
            match self.peek() {
                Some(TokenKind::RBracket) => return Ok(()),
                // `:` is the `all` selector (entire axis, spec §05).
                Some(TokenKind::Colon) => {
                    self.advance();
                    let sym = self.module.intern("all");
                    args.push(self.module.alloc(Node::Const(sym)));
                }
                // `!` immediately before `,` / `]` is the `only` selector
                // (unique element of a length-1 axis); otherwise it starts a
                // unary logical-not expression (spec §05 disambiguation note).
                Some(TokenKind::Bang)
                    if matches!(
                        self.peek_at(1),
                        Some(TokenKind::Comma | TokenKind::RBracket)
                    ) =>
                {
                    self.advance();
                    let sym = self.module.intern("only");
                    args.push(self.module.alloc(Node::Const(sym)));
                }
                _ => {
                    let idx = self.parse_expr()?;
                    args.push(idx);
                }
            }
            if !self.eat(&TokenKind::Comma) {
                return Ok(());
            }
        }
    }

    fn parse_primary(&mut self) -> Result<NodeId> {
        match self.peek() {
            Some(TokenKind::Int(n)) => {
                let n = *n;
                self.advance();
                Ok(self.module.alloc(Node::Lit(Scalar::Int(n))))
            }
            Some(TokenKind::Real(r)) => {
                let r = *r;
                self.advance();
                Ok(self.module.alloc(Node::Lit(Scalar::Real(r))))
            }
            Some(TokenKind::Str(s)) => {
                let s = s.clone();
                self.advance();
                Ok(self
                    .module
                    .alloc(Node::Lit(Scalar::Str(s.into_boxed_str()))))
            }
            Some(TokenKind::Name(n)) => {
                let n = n.clone();
                self.advance();
                self.parse_name_tail(n)
            }
            Some(TokenKind::LParen) => self.parse_paren_or_tuple(),
            Some(TokenKind::LBracket) => self.parse_array(),
            // A leading `.name` is an axis label (legal inside aggregate bodies /
            // indexing); a `.name` *after* an expression is field access (postfix).
            Some(TokenKind::Dot) => self.parse_axis_node(),
            other => {
                Err(self.err_here(format!("expected an expression, found {}", describe(other))))
            }
        }
    }

    fn parse_axis_node(&mut self) -> Result<NodeId> {
        self.advance(); // .
        let mut name = self.expect_name("an axis name after `.`")?;
        // Variance markers (spec §05 axis names): a trailing `_` in the lexed
        // name is the lower marker (axis names themselves may not end in `_`);
        // an immediately following `^` is the upper marker.
        let mut variance = None;
        if let Some(stripped) = name.strip_suffix('_') {
            name = stripped.to_string();
            variance = Some(Variance::Lower);
        } else if self.eat(&TokenKind::Caret) {
            variance = Some(Variance::Upper);
        }
        let valid =
            !name.is_empty() && name.as_bytes()[0].is_ascii_alphabetic() && !name.ends_with('_');
        if !valid {
            return Err(self.err_prev(format!(
                "invalid axis name `.{name}`: must start with a letter and not end in `_`"
            )));
        }
        let sym = self.module.intern(&name);
        Ok(self.module.alloc(Node::Axis(Axis {
            name: sym,
            variance,
        })))
    }

    /// Parse `[.i, .k] := <body>` (the LHS bracket onward) into
    /// `aggregate(sum, [.i, .k], <body>)`. Position starts at the `[`.
    fn parse_aggregate(&mut self) -> Result<NodeId> {
        let (axes_vec, body) = self.parse_axes_walrus_body()?;
        let sum_sym = self.module.intern("sum");
        let sum = self.module.alloc(Node::Const(sum_sym));
        Ok(self.builtin_call("aggregate", vec![sum, axes_vec, body]))
    }

    /// Parse `[.axes…] := <body>` (the result-name bracket onward) into
    /// `metricsum(metric, [.axes…], <body>)` (spec §04 metricsum shorthand).
    fn parse_metricsum(&mut self, metric: &str) -> Result<NodeId> {
        let metric_node = self.resolve_bare_name(metric);
        let (axes_vec, body) = self.parse_axes_walrus_body()?;
        Ok(self.builtin_call("metricsum", vec![metric_node, axes_vec, body]))
    }

    /// The shared `[.i, .k] := <body>` tail of `:=` statements: the axis list
    /// (possibly empty) as a `vector` literal, and the body expression.
    fn parse_axes_walrus_body(&mut self) -> Result<(NodeId, NodeId)> {
        self.expect(&TokenKind::LBracket, "`[` to open the axis list")?;
        let mut axes = Vec::new();
        while !matches!(self.peek(), Some(TokenKind::RBracket)) {
            axes.push(self.parse_axis_node()?);
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RBracket, "`]` to close the axis list")?;
        self.expect(&TokenKind::Walrus, "`:=`")?;
        let body = self.parse_expr()?;
        let axes_vec = self.builtin_call("vector", axes);
        Ok((axes_vec, body))
    }

    /// Resolve a bare (non-call) name: a module binding is a self-ref,
    /// anything else a built-in constant.
    fn resolve_bare_name(&mut self, name: &str) -> NodeId {
        let sym = self.module.intern(name);
        let node = if self.names.bound.contains(name) {
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name: sym,
            })
        } else {
            Node::Const(sym)
        };
        self.module.alloc(node)
    }

    /// If `expr` is a bare name that opens a namespace — the reserved `self` /
    /// `base` modules or a `load_module` / `standard_module` binding — return
    /// that namespace.
    fn member_namespace(&self, expr: NodeId) -> Option<MemberNs> {
        match self.module.node(expr) {
            Node::Const(sym) => match self.module.resolve(*sym) {
                "self" => Some(MemberNs::SelfMod),
                "base" => Some(MemberNs::Base),
                _ => None,
            },
            Node::Ref(r) if matches!(r.ns, RefNs::SelfMod) => {
                let name = self.module.resolve(r.name);
                if self.names.modules.contains(name) {
                    Some(MemberNs::Module(r.name))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Lower `<ns>.<name>` — optionally followed by a call — to a ref, a bare
    /// built-in constant, or a call through the namespace. `base.foo` always
    /// denotes the built-in and lowers to the bare form (spec §11).
    fn lower_member(&mut self, ns: MemberNs, name: &str) -> Result<NodeId> {
        let is_call = self.eat(&TokenKind::LParen);
        if let MemberNs::Base = ns {
            return if is_call {
                // `base.functionof(…)` is still the reification construct —
                // same placeholder-scope boundary as the bare form.
                let reify = name == "functionof" || name == "kernelof";
                if reify {
                    self.reify_frames.push(ReifyFrame::Explicit);
                }
                let args = self.parse_call_args();
                if reify {
                    self.reify_frames.pop();
                }
                let (positional, named) = args?;
                self.expect(&TokenKind::RParen, "`)` to close the call")?;
                self.make_builtin_call(name, positional, named)
            } else {
                let sym = self.module.intern(name);
                Ok(self.module.alloc(Node::Const(sym)))
            };
        }
        let name_sym = self.module.intern(name);
        let r = Ref {
            ns: match ns {
                MemberNs::SelfMod => RefNs::SelfMod,
                MemberNs::Module(alias) => RefNs::Module(alias),
                MemberNs::Base => unreachable!(),
            },
            name: name_sym,
        };
        if is_call {
            let (positional, named) = self.parse_call_args()?;
            self.expect(&TokenKind::RParen, "`)` to close the call")?;
            Ok(self.make_user_call(r, positional, named))
        } else {
            Ok(self.module.alloc(Node::Ref(r)))
        }
    }

    /// A bare name has already been consumed; decide literal / call / reference.
    fn parse_name_tail(&mut self, name: String) -> Result<NodeId> {
        // `true` / `false` are boolean literals (never callable).
        if name == "true" {
            return Ok(self.module.alloc(Node::Lit(Scalar::Bool(true))));
        }
        if name == "false" {
            return Ok(self.module.alloc(Node::Lit(Scalar::Bool(false))));
        }

        // `_` is a hole (spec §04): valid only when the *innermost* enclosing
        // reification is the `fn(…)` that binds it — an intervening lambda or
        // `functionof` would leave the generated placeholder unbound there.
        if name == "_" {
            if !matches!(self.reify_frames.last(), Some(ReifyFrame::Fn(_))) {
                let separated = self
                    .reify_frames
                    .iter()
                    .any(|f| matches!(f, ReifyFrame::Fn(_)));
                let msg = if separated {
                    "`_` hole is separated from its `fn(…)` by an enclosing reification \
                     (placeholders are scoped to the nearest enclosing `functionof`)"
                } else {
                    "`_` hole is only valid inside `fn(…)`"
                };
                return Err(self.err_prev(msg));
            }
            let Some(ReifyFrame::Fn(entries)) = self.reify_frames.last_mut() else {
                unreachable!("checked above");
            };
            let n = entries.len() + 1;
            let arg_sym = self.module.intern(&format!("arg{n}"));
            let ph_sym = self.module.intern(&format!("_arg{n}_"));
            let r = Ref {
                ns: RefNs::Local,
                name: ph_sym,
            };
            entries.push((arg_sym, r));
            return Ok(self.module.alloc(Node::Ref(r)));
        }

        // A lambda argument shadows module bindings and built-ins inside its
        // body (spec §05), lowering to the placeholder `_name_` — but only in
        // the *innermost* reification: referencing an enclosing lambda's
        // argument from inside a nested lambda / `fn` / `functionof` would
        // leave the placeholder unbound there (spec §04's DISALLOWED case).
        let lambda_arg =
            |f: &ReifyFrame| matches!(f, ReifyFrame::Lambda(args) if args.contains(&name));
        if self.reify_frames.last().is_some_and(lambda_arg) {
            let ph_sym = self.module.intern(&format!("_{name}_"));
            let r = Ref {
                ns: RefNs::Local,
                name: ph_sym,
            };
            if self.eat(&TokenKind::LParen) {
                let (positional, named) = self.parse_call_args()?;
                self.expect(&TokenKind::RParen, "`)` to close the call")?;
                return Ok(self.make_user_call(r, positional, named));
            }
            return Ok(self.module.alloc(Node::Ref(r)));
        }
        if self.reify_frames.iter().any(lambda_arg) {
            return Err(self.err_prev(format!(
                "`{name}` is an argument of an enclosing lambda and cannot be referenced \
                     inside a nested reification (placeholders are scoped to the nearest \
                     enclosing `functionof`)"
            )));
        }

        if matches!(self.peek(), Some(TokenKind::LParen)) {
            self.advance();
            // `fn(expr)` is the hole-sugar special operation (spec §05).
            if name == "fn" {
                return self.lower_fn();
            }
            // An explicit reification's whole argument list is a placeholder-
            // scope boundary (body and boundary kwargs alike).
            let reify = name == "functionof" || name == "kernelof";
            if reify {
                self.reify_frames.push(ReifyFrame::Explicit);
            }
            let args = self.parse_call_args();
            if reify {
                self.reify_frames.pop();
            }
            let (positional, named) = args?;
            self.expect(&TokenKind::RParen, "`)` to close the call")?;
            return self.make_call(&name, positional, named);
        }

        // A bare reference: a placeholder (`_x_`, reserved lexical class) is a
        // `%local` ref; a module binding is a self-ref; anything else is a
        // built-in constant / set / function-as-value.
        let sym = self.module.intern(&name);
        let node = if is_placeholder(&name) {
            Node::Ref(flatppl_core::Ref {
                ns: RefNs::Local,
                name: sym,
            })
        } else if self.names.bound.contains(&name) {
            Node::Ref(flatppl_core::Ref {
                ns: RefNs::SelfMod,
                name: sym,
            })
        } else {
            Node::Const(sym)
        };
        Ok(self.module.alloc(node))
    }

    fn make_call(
        &mut self,
        name: &str,
        positional: Vec<NodeId>,
        named: Vec<NamedArg>,
    ) -> Result<NodeId> {
        // The reification constructs are special operations with their own
        // syntax (spec §05), recognised before name resolution.
        let special = name == "functionof" || name == "kernelof";
        if !special && self.names.bound.contains(name) {
            let sym = self.module.intern(name);
            let r = Ref {
                ns: RefNs::SelfMod,
                name: sym,
            };
            return Ok(self.make_user_call(r, positional, named));
        }
        self.make_builtin_call(name, positional, named)
    }

    /// A call to a user-defined callable named by `r` — `(%call (%ref …) …)`.
    fn make_user_call(&mut self, r: Ref, positional: Vec<NodeId>, named: Vec<NamedArg>) -> NodeId {
        let callee = self.module.alloc(Node::Ref(r));
        self.make_applied_call(callee, positional, named)
    }

    /// Apply a callable-valued expression — `(%call <callable> …)` with an
    /// expression callee (spec §11). Named arguments stay `%kwarg`.
    fn make_applied_call(
        &mut self,
        callee: NodeId,
        positional: Vec<NodeId>,
        named: Vec<NamedArg>,
    ) -> NodeId {
        self.module.alloc(Node::Call(Call {
            head: CallHead::User(callee),
            args: positional.into(),
            named: named.into(),
            inputs: None,
        }))
    }

    fn make_builtin_call(
        &mut self,
        name: &str,
        positional: Vec<NodeId>,
        named: Vec<NamedArg>,
    ) -> Result<NodeId> {
        // Reification (spec §11 "Reified callables"): the single positional is
        // the reified output; boundary kwargs become the `%specinputs` entries
        // (each value must be a module binding or a placeholder — already
        // lowered to a Ref node); no kwargs means `%autoinputs` (cut deferred
        // to phase inference).
        if name == "functionof" || name == "kernelof" {
            if positional.len() != 1 {
                return Err(
                    self.err_here(format!("`{name}` takes exactly one expression to reify"))
                );
            }
            let inputs = if named.is_empty() {
                Inputs::Auto
            } else {
                let mut entries = Vec::with_capacity(named.len());
                for n in &named {
                    let Node::Ref(r) = self.module.node(n.value) else {
                        return Err(self.err_here(format!(
                                "`{name}` boundary input `{}` must be a module binding or a placeholder",
                                self.module.resolve(n.name)
                            ),
                        ));
                    };
                    entries.push((n.name, *r));
                }
                Inputs::Spec(entries.into())
            };
            let head = CallHead::Builtin(self.module.intern(name));
            return Ok(self.module.alloc(Node::Call(Call {
                head,
                args: Box::new([positional[0]]),
                named: Box::new([]),
                inputs: Some(inputs),
            })));
        }

        // Built-in data/binding constructors use ordered `%field` / `%assign`
        // entries, not order-insignificant `%kwarg` (spec §11).
        let sym = self.module.intern(name);
        let kind = named_kind_for(name);
        let named: Vec<NamedArg> = named.into_iter().map(|a| NamedArg { kind, ..a }).collect();
        Ok(self.module.alloc(Node::Call(Call {
            head: CallHead::Builtin(sym),
            args: positional.into(),
            named: named.into(),
            inputs: None,
        })))
    }

    /// Parse `pos, …, kw = val, …` up to (not consuming) the closing delimiter.
    fn parse_call_args(&mut self) -> Result<(Vec<NodeId>, Vec<NamedArg>)> {
        let mut positional = Vec::new();
        let mut named = Vec::new();
        loop {
            if matches!(self.peek(), Some(TokenKind::RParen)) {
                break;
            }
            // Keyword argument: `Name = expr`.
            if let (Some(TokenKind::Name(n)), Some(TokenKind::Assign)) =
                (self.peek(), self.peek_at(1))
            {
                let key = n.clone();
                self.advance();
                self.advance();
                let value = self.parse_expr()?;
                named.push(NamedArg {
                    kind: NamedKind::Kwarg,
                    name: self.module.intern(&key),
                    value,
                });
            } else {
                positional.push(self.parse_expr()?);
            }
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        Ok((positional, named))
    }

    fn parse_paren_or_tuple(&mut self) -> Result<NodeId> {
        self.advance(); // (
        let first = self.parse_expr()?;
        if matches!(self.peek(), Some(TokenKind::Comma)) {
            let mut elems = vec![first];
            while self.eat(&TokenKind::Comma) {
                if matches!(self.peek(), Some(TokenKind::RParen)) {
                    break;
                }
                elems.push(self.parse_expr()?);
            }
            self.expect(&TokenKind::RParen, "`)` to close the tuple")?;
            Ok(self.builtin_call("tuple", elems))
        } else {
            self.expect(&TokenKind::RParen, "`)` to close the group")?;
            Ok(first)
        }
    }

    fn parse_array(&mut self) -> Result<NodeId> {
        self.advance(); // [
        let mut elems = Vec::new();
        loop {
            if matches!(self.peek(), Some(TokenKind::RBracket)) {
                break;
            }
            elems.push(self.parse_expr()?);
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RBracket, "`]` to close the array")?;
        Ok(self.builtin_call("vector", elems))
    }

    // ---- broadcast helpers (dotted operators) ----

    fn broadcast_op(&mut self, func: &str, lhs: NodeId, rhs: NodeId) -> NodeId {
        let fsym = self.module.intern(func);
        let f = self.module.alloc(Node::Const(fsym));
        self.builtin_call("broadcast", vec![f, lhs, rhs])
    }

    fn broadcast_unop(&mut self, func: &str, operand: NodeId) -> NodeId {
        let fsym = self.module.intern(func);
        let f = self.module.alloc(Node::Const(fsym));
        self.builtin_call("broadcast", vec![f, operand])
    }

    fn apply_binop(&mut self, func: &str, dotted: bool, lhs: NodeId, rhs: NodeId) -> NodeId {
        if dotted {
            self.broadcast_op(func, lhs, rhs)
        } else {
            self.builtin_call(func, vec![lhs, rhs])
        }
    }

    fn expect_name(&mut self, what: &str) -> Result<String> {
        match self.peek() {
            Some(TokenKind::Name(n)) => {
                let n = n.clone();
                self.advance();
                Ok(n)
            }
            _ => Err(self.err_here(format!("expected {what}"))),
        }
    }
}

/// A comparison operator's lowered function name and whether it is dotted.
struct CmpOp {
    func: &'static str,
    dotted: bool,
}

fn cmp_op(kind: &TokenKind) -> Option<CmpOp> {
    let (func, dotted) = match kind {
        TokenKind::Lt => ("lt", false),
        TokenKind::Gt => ("gt", false),
        TokenKind::EqEq => ("equal", false),
        TokenKind::BangEq => ("unequal", false),
        TokenKind::Le => ("le", false),
        TokenKind::Ge => ("ge", false),
        // Set membership `x in S` (reserved word; no dotted form, spec §05).
        TokenKind::Name(n) if n == "in" => ("in", false),
        TokenKind::DotLt => ("lt", true),
        TokenKind::DotGt => ("gt", true),
        TokenKind::DotEqEq => ("equal", true),
        TokenKind::DotBangEq => ("unequal", true),
        TokenKind::DotLe => ("le", true),
        TokenKind::DotGe => ("ge", true),
        _ => return None,
    };
    Some(CmpOp { func, dotted })
}

/// Is `name` a placeholder (spec §04: `^_[A-Za-z]([A-Za-z0-9_]*[A-Za-z0-9])?_$`
/// — single leading and trailing underscore)? Placeholders are a reserved
/// lexical class: they may not be module bindings, so the classification is
/// context-free. (The printer shares this to exclude placeholder-shaped names
/// from lambda re-sugaring.)
pub(crate) fn is_placeholder(name: &str) -> bool {
    let b = name.as_bytes();
    b.len() >= 3
        && b[0] == b'_'
        && b[b.len() - 1] == b'_'
        && b[1].is_ascii_alphabetic()
        && (b.len() == 3 || b[b.len() - 2].is_ascii_alphanumeric())
}

/// Which FlatPIR named-entry kind a built-in's `name = value` arguments use.
fn named_kind_for(op: &str) -> NamedKind {
    match op {
        "record" | "table" | "joint" | "jointchain" | "cartprod" => NamedKind::Field,
        "load_module" | "standard_module" => NamedKind::Assign,
        _ => NamedKind::Kwarg,
    }
}

fn describe(kind: Option<&TokenKind>) -> String {
    match kind {
        None => "end of input".to_string(),
        Some(k) => format!("{k:?}"),
    }
}
