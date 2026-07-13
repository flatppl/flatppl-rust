//! Small recursive-descent / Pratt parser for HS3 C-like expression strings.
//!
//! `parse_expr_as_fn` is the primary entry point: it wraps the parsed
//! expression in a FlatPPL lambda `x -> <expr>` so that the result is a
//! callable usable as the weight argument to `weighted(w, Lebesgue(reals))`.
//!
//! `parse_expr_inline` returns the raw expression node (no lambda wrapper),
//! for cases where a plain value is needed (e.g. `functions` block entries
//! in `generic_function`).
//!
//! Grammar summary (precedence, lowest to highest):
//!   expr        = add_expr
//!   add_expr    = mul_expr ( ('+' | '-') mul_expr )*
//!   mul_expr    = unary   ( ('*' | '/') unary   )*
//!   unary       = '-' unary  |  pow_expr
//!   pow_expr    = atom ('^' unary)*    — right-associative
//!   atom        = NUMBER | IDENT | '(' expr ')' | IDENT '(' args ')'
//!
//! FlatPPL builtin mapping:
//!   `+` → `add`, `-` → `sub`, `*` → `mul`, `/` → `divide`, `^` → `pow`
//!   unary `-` → `neg`
//!   Math fns: `exp log sqrt abs sin cos tan asin acos atan` (1-arg)
//!             `min max pow` (2-arg)
//!   Constants: `PI` → lit_real(π), `EULER` → lit_real(e)
//!
//! NOT YET supported (rejected with `Error::Unimplemented`, never silently
//! parsed): comparisons `== != < <= > >=`, boolean `&& || !`, and the ternary
//! conditional `a ? b : c` — there is no ternary/ifelse parse rule.

use crate::builder::Builder;
use crate::error::{Error, Result};
use flatppl_core::id::NodeId;
use flatppl_core::node::{Call, CallHead, Inputs, Node, Ref, RefNs};

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Parse an HS3 expression string, returning a FlatPPL *function* node
/// `<obs_name> -> <expr>` (a lambda over the observable variable name).
///
/// The observable variable name must match the one used in the expression;
/// by HS3 convention this is `"x"` for most distributions.
pub fn parse_expr_as_fn(b: &mut Builder, src: &str, obs_name: &str) -> Result<NodeId> {
    let body = parse_expr_inline(b, src)?;
    Ok(wrap_lambda(b, body, obs_name))
}

/// Parse an HS3 expression string into an inline FlatPPL expression node.
///
/// Identifier references become `self_ref` nodes (resolved at module level).
/// Numeric literals become `lit_real` nodes.
pub fn parse_expr_inline(b: &mut Builder, src: &str) -> Result<NodeId> {
    let tokens = tokenize(src)?;
    let mut p = Parser::new(&tokens, b);
    let node = p.parse_expr()?;
    if p.pos < p.tokens.len() {
        return Err(Error::Unsupported(format!(
            "expression parser: unexpected token `{}` at position {}",
            token_repr(&p.tokens[p.pos]),
            p.pos
        )));
    }
    Ok(node)
}

/// Collect the *variable* identifiers referenced in an HS3 expression string.
///
/// Tokenizes `src` with the same tokenizer the parser uses and returns each
/// `Tok::Ident` that is NOT immediately followed by `(` — i.e. a value
/// reference, never a function call (`sqrt`, `abs`, `sin`, …). Results are
/// deduplicated, preserving first-occurrence order.
///
/// The constants `PI`/`Pi`/`pi` and `EULER`/`Euler` are inlined as literals by
/// the parser (never module references), so they are excluded here too.
///
/// A malformed expression that fails to tokenize yields an empty list rather
/// than an error: this helper is used to *discover* declarations, and any real
/// tokenization failure surfaces later when the expression itself is parsed.
pub fn free_identifiers(src: &str) -> Vec<String> {
    let Ok(toks) = tokenize(src) else {
        return Vec::new();
    };
    let mut out: Vec<String> = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        if let Tok::Ident(name) = t {
            // A following `(` makes this a function call, not a value reference.
            if matches!(toks.get(i + 1), Some(Tok::LParen)) {
                continue;
            }
            // Parser-inlined constants are never module references.
            if matches!(name.as_str(), "PI" | "Pi" | "pi" | "EULER" | "Euler") {
                continue;
            }
            if !out.iter().any(|s| s == name) {
                out.push(name.clone());
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Lambda wrapper — builds `obs_name -> body` in the IR
// ---------------------------------------------------------------------------

// Lambda wrapper — builds `(name0, name1, ...) -> body` in the IR.
pub(crate) fn wrap_lambda_multi(b: &mut Builder, body: NodeId, obs_names: &[&str]) -> NodeId {
    let mut rewritten = body;
    let mut entries: Vec<(flatppl_core::id::Symbol, Ref)> = Vec::with_capacity(obs_names.len());
    for name in obs_names {
        let name_sym = b.m.intern(name);
        let ph_sym = b.m.intern(&format!("_{name}_"));
        rewritten = rewrite_self_to_local(b, rewritten, name_sym, ph_sym);
        entries.push((
            name_sym,
            Ref {
                ns: RefNs::Local,
                name: ph_sym,
            },
        ));
    }
    let head = CallHead::Builtin(b.m.intern("functionof"));
    b.m.alloc(Node::Call(Call {
        head,
        args: Box::new([rewritten]),
        named: Box::new([]),
        inputs: Some(Inputs::Spec(entries.into_boxed_slice())),
    }))
}

fn wrap_lambda(b: &mut Builder, body: NodeId, obs_name: &str) -> NodeId {
    wrap_lambda_multi(b, body, &[obs_name])
}

/// Recursively copy the sub-graph rooted at `node`, replacing every
/// `SelfMod` ref named `target_sym` with a `Local` ref named `ph_sym`.
/// Returns the (possibly new) root NodeId.
fn rewrite_self_to_local(
    b: &mut Builder,
    node: NodeId,
    target_sym: flatppl_core::id::Symbol,
    ph_sym: flatppl_core::id::Symbol,
) -> NodeId {
    // Only Ref and Call nodes need handling. Inspect by reference and clone the
    // Call (which owns boxed children) lazily — leaves (Lit/Const/Hole/Axis) carry
    // no refs and are returned unchanged without an allocation.
    match b.m.node(node) {
        Node::Ref(r) if r.ns == RefNs::SelfMod && r.name == target_sym => {
            // Replace: emit a Local ref (the lambda placeholder).
            b.m.alloc(Node::Ref(Ref {
                ns: RefNs::Local,
                name: ph_sym,
            }))
        }
        Node::Call(_) => {
            // Clone the Call so we can recurse without holding a borrow of `b.m`.
            let Node::Call(call) = b.m.node(node).clone() else {
                unreachable!("matched Node::Call above")
            };
            // Rewrite head if it is a User(callee) node.
            let new_head = match call.head {
                CallHead::User(callee_id) => {
                    CallHead::User(rewrite_self_to_local(b, callee_id, target_sym, ph_sym))
                }
                other => other,
            };
            let new_args: Vec<NodeId> = call
                .args
                .iter()
                .map(|&a| rewrite_self_to_local(b, a, target_sym, ph_sym))
                .collect();
            let new_named: Vec<_> = call
                .named
                .iter()
                .map(|na| {
                    let mut na2 = *na;
                    na2.value = rewrite_self_to_local(b, na.value, target_sym, ph_sym);
                    na2
                })
                .collect();
            // Preserve inputs (boundary spec) unchanged — inputs are lambda/fn
            // declarations, not expression sub-nodes.
            b.m.alloc(Node::Call(Call {
                head: new_head,
                args: new_args.into_boxed_slice(),
                named: new_named.into_boxed_slice(),
                inputs: call.inputs,
            }))
        }
        // Ref (non-matching), Lit, Const, Hole, Axis — no rewritable refs inside.
        _ => node,
    }
}

// ---------------------------------------------------------------------------
// Tokeniser
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
    // error tokens for unsupported operators
    EqEq,
    BangEq,
    Lt,
    Le,
    Gt,
    Ge,
    AmpAmp,
    PipePipe,
    Bang,
    Question,
    Colon,
}

fn token_repr(t: &Tok) -> &'static str {
    match t {
        Tok::Num(_) => "<number>",
        Tok::Ident(_) => "<ident>",
        Tok::Plus => "+",
        Tok::Minus => "-",
        Tok::Star => "*",
        Tok::Slash => "/",
        Tok::Caret => "^",
        Tok::LParen => "(",
        Tok::RParen => ")",
        Tok::Comma => ",",
        Tok::EqEq => "==",
        Tok::BangEq => "!=",
        Tok::Lt => "<",
        Tok::Le => "<=",
        Tok::Gt => ">",
        Tok::Ge => ">=",
        Tok::AmpAmp => "&&",
        Tok::PipePipe => "||",
        Tok::Bang => "!",
        Tok::Question => "?",
        Tok::Colon => ":",
    }
}

fn tokenize(src: &str) -> Result<Vec<Tok>> {
    let mut toks = Vec::new();
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' | '\r' | '\n' => {
                i += 1;
            }
            '0'..='9' | '.' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_digit()
                        || chars[i] == '.'
                        || chars[i] == 'e'
                        || chars[i] == 'E')
                {
                    // Allow sign after e/E
                    if (chars[i] == 'e' || chars[i] == 'E')
                        && i + 1 < chars.len()
                        && (chars[i + 1] == '+' || chars[i + 1] == '-')
                    {
                        i += 2;
                        continue;
                    }
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                let v = s.parse::<f64>().map_err(|_| {
                    Error::Unsupported(format!("expression parser: bad number literal `{s}`"))
                })?;
                toks.push(Tok::Num(v));
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let name: String = chars[start..i].iter().collect();
                toks.push(Tok::Ident(name));
            }
            '+' => {
                toks.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                toks.push(Tok::Minus);
                i += 1;
            }
            '*' => {
                toks.push(Tok::Star);
                i += 1;
            }
            '/' => {
                toks.push(Tok::Slash);
                i += 1;
            }
            '^' => {
                toks.push(Tok::Caret);
                i += 1;
            }
            '(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            ',' => {
                toks.push(Tok::Comma);
                i += 1;
            }
            '=' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    toks.push(Tok::EqEq);
                    i += 2;
                } else {
                    return Err(Error::Unsupported(
                        "expression operator '=' not supported (did you mean `==`?)".into(),
                    ));
                }
            }
            '!' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    toks.push(Tok::BangEq);
                    i += 2;
                } else {
                    toks.push(Tok::Bang);
                    i += 1;
                }
            }
            '<' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    toks.push(Tok::Le);
                    i += 2;
                } else {
                    toks.push(Tok::Lt);
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    toks.push(Tok::Ge);
                    i += 2;
                } else {
                    toks.push(Tok::Gt);
                    i += 1;
                }
            }
            '&' => {
                if i + 1 < chars.len() && chars[i + 1] == '&' {
                    toks.push(Tok::AmpAmp);
                    i += 2;
                } else {
                    return Err(Error::Unsupported(
                        "expression operator '&' not supported".into(),
                    ));
                }
            }
            '|' => {
                if i + 1 < chars.len() && chars[i + 1] == '|' {
                    toks.push(Tok::PipePipe);
                    i += 2;
                } else {
                    return Err(Error::Unsupported(
                        "expression operator '|' not supported".into(),
                    ));
                }
            }
            '?' => {
                toks.push(Tok::Question);
                i += 1;
            }
            ':' => {
                toks.push(Tok::Colon);
                i += 1;
            }
            c => {
                return Err(Error::Unsupported(format!(
                    "expression parser: unexpected character `{c}`"
                )));
            }
        }
    }
    Ok(toks)
}

// ---------------------------------------------------------------------------
// Pratt / recursive-descent parser
// ---------------------------------------------------------------------------

struct Parser<'b, 'm> {
    tokens: &'b [Tok],
    pos: usize,
    b: &'b mut Builder<'m>,
}

impl<'b, 'm> Parser<'b, 'm> {
    fn new(tokens: &'b [Tok], b: &'b mut Builder<'m>) -> Self {
        Parser { tokens, pos: 0, b }
    }

    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }

    /// Advance past the current token, returning it (or `None` at end of input).
    fn advance(&mut self) -> Option<&Tok> {
        let t = self.tokens.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    /// If the current token is a number, consume and return its value.
    fn next_num(&mut self) -> Option<f64> {
        match self.tokens.get(self.pos) {
            Some(Tok::Num(v)) => {
                let v = *v;
                self.pos += 1;
                Some(v)
            }
            _ => None,
        }
    }

    /// If the current token is an identifier, consume and return its name.
    fn next_ident(&mut self) -> Option<String> {
        match self.tokens.get(self.pos) {
            Some(Tok::Ident(s)) => {
                let s = s.clone();
                self.pos += 1;
                Some(s)
            }
            _ => None,
        }
    }

    fn check_unsupported(&self, t: &Tok) -> Result<()> {
        match t {
            Tok::EqEq => Err(Error::Unimplemented(
                "expression operator '==' not supported".into(),
            )),
            Tok::BangEq => Err(Error::Unimplemented(
                "expression operator '!=' not supported".into(),
            )),
            Tok::Lt => Err(Error::Unimplemented(
                "expression operator '<' not supported".into(),
            )),
            Tok::Le => Err(Error::Unimplemented(
                "expression operator '<=' not supported".into(),
            )),
            Tok::Gt => Err(Error::Unimplemented(
                "expression operator '>' not supported".into(),
            )),
            Tok::Ge => Err(Error::Unimplemented(
                "expression operator '>=' not supported".into(),
            )),
            Tok::AmpAmp => Err(Error::Unimplemented(
                "expression operator '&&' not supported".into(),
            )),
            Tok::PipePipe => Err(Error::Unimplemented(
                "expression operator '||' not supported".into(),
            )),
            Tok::Bang => Err(Error::Unimplemented(
                "expression operator '!' not supported".into(),
            )),
            Tok::Question => Err(Error::Unimplemented(
                "expression operator '?' (ternary) not supported".into(),
            )),
            Tok::Colon => Err(Error::Unimplemented(
                "expression operator ':' (ternary) not supported".into(),
            )),
            _ => Ok(()),
        }
    }

    // expr = add_expr
    fn parse_expr(&mut self) -> Result<NodeId> {
        let node = self.parse_add()?;
        // Reject unsupported trailing operators
        if let Some(t) = self.peek() {
            self.check_unsupported(t)?;
        }
        Ok(node)
    }

    // add_expr = mul_expr ( ('+' | '-') mul_expr )*
    fn parse_add(&mut self) -> Result<NodeId> {
        let mut lhs = self.parse_mul()?;
        loop {
            match self.peek() {
                Some(Tok::Plus) => {
                    self.advance();
                    let rhs = self.parse_mul()?;
                    lhs = self.b.call("add", &[lhs, rhs]);
                }
                Some(Tok::Minus) => {
                    self.advance();
                    let rhs = self.parse_mul()?;
                    lhs = self.b.call("sub", &[lhs, rhs]);
                }
                Some(t) => {
                    self.check_unsupported(t)?;
                    break;
                }
                None => break,
            }
        }
        Ok(lhs)
    }

    // mul_expr = unary ( ('*' | '/') unary )*
    fn parse_mul(&mut self) -> Result<NodeId> {
        let mut lhs = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(Tok::Star) => {
                    self.advance();
                    let rhs = self.parse_unary()?;
                    lhs = self.b.call("mul", &[lhs, rhs]);
                }
                Some(Tok::Slash) => {
                    self.advance();
                    let rhs = self.parse_unary()?;
                    lhs = self.b.call("divide", &[lhs, rhs]);
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    // unary = '-' unary | pow_expr
    fn parse_unary(&mut self) -> Result<NodeId> {
        if matches!(self.peek(), Some(Tok::Minus)) {
            self.advance();
            let operand = self.parse_unary()?;
            Ok(self.b.call("neg", &[operand]))
        } else {
            self.parse_pow()
        }
    }

    // pow_expr = atom ('^' unary)*   — right-assoc via recursion
    fn parse_pow(&mut self) -> Result<NodeId> {
        let base = self.parse_atom()?;
        if matches!(self.peek(), Some(Tok::Caret)) {
            self.advance();
            // right-associative: exponent parsed as unary (allows `a^-b`)
            let exp = self.parse_unary()?;
            Ok(self.b.call("pow", &[base, exp]))
        } else {
            Ok(base)
        }
    }

    // atom = NUMBER | '(' expr ')' | IDENT | IDENT '(' args... ')'
    fn parse_atom(&mut self) -> Result<NodeId> {
        match self.peek() {
            Some(Tok::Num(_)) => {
                let v = self.next_num().expect("peeked Num");
                Ok(self.b.lit_real(v))
            }
            Some(Tok::LParen) => {
                self.advance();
                let inner = self.parse_expr()?;
                match self.peek() {
                    Some(Tok::RParen) => {
                        self.advance();
                    }
                    _ => {
                        return Err(Error::Unsupported(
                            "expression parser: missing `)` after sub-expression".into(),
                        ));
                    }
                }
                Ok(inner)
            }
            Some(Tok::Ident(_)) => {
                let name = self.next_ident().expect("peeked Ident");
                // Check for function call: IDENT '('
                if matches!(self.peek(), Some(Tok::LParen)) {
                    self.advance(); // consume '('
                    self.parse_call(&name)
                } else {
                    // Variable reference or constant
                    Ok(self.ident_node(&name))
                }
            }
            Some(t) => {
                let msg = format!(
                    "expression parser: unexpected token `{}` at start of atom",
                    token_repr(t)
                );
                Err(Error::Unsupported(msg))
            }
            None => Err(Error::Unsupported(
                "expression parser: unexpected end of input".into(),
            )),
        }
    }

    /// Map an identifier to a FlatPPL node.
    /// Constants `PI` and `EULER` are inlined as real literals.
    /// All other identifiers become `self_ref` nodes.
    fn ident_node(&mut self, name: &str) -> NodeId {
        match name {
            "PI" | "Pi" | "pi" => self.b.lit_real(std::f64::consts::PI),
            "EULER" | "Euler" => self.b.lit_real(std::f64::consts::E),
            other => self.b.self_ref(other),
        }
    }

    /// Parse a function call `name(arg, ...)` where the opening `(` is already consumed.
    fn parse_call(&mut self, name: &str) -> Result<NodeId> {
        // Collect arguments
        let mut args = Vec::new();
        if !matches!(self.peek(), Some(Tok::RParen)) {
            args.push(self.parse_expr()?);
            while matches!(self.peek(), Some(Tok::Comma)) {
                self.advance();
                args.push(self.parse_expr()?);
            }
        }
        match self.peek() {
            Some(Tok::RParen) => {
                self.advance();
            }
            _ => {
                return Err(Error::Unsupported(format!(
                    "expression parser: missing `)` after arguments to `{name}`"
                )));
            }
        }

        // Map to FlatPPL builtins
        let flatppl_name = match name {
            // 1-argument math functions
            "exp" | "log" | "sqrt" | "abs" | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" => {
                if args.len() != 1 {
                    return Err(Error::Unsupported(format!(
                        "expression parser: `{name}` expects 1 argument, got {}",
                        args.len()
                    )));
                }
                name
            }
            // 2-argument functions
            "min" | "max" | "pow" => {
                if args.len() != 2 {
                    return Err(Error::Unsupported(format!(
                        "expression parser: `{name}` expects 2 arguments, got {}",
                        args.len()
                    )));
                }
                name
            }
            "erf" | "erfc" => {
                if args.len() != 1 {
                    return Err(Error::Unsupported(format!(
                        "expression parser: `{name}` expects 1 argument, got {}",
                        args.len()
                    )));
                }
                // special-functions module member: alias `specfun`.
                return Ok(self.b.module_user_call("specfun", name, &args));
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "expression parser: unknown function `{other}`"
                )));
            }
        };

        Ok(self.b.call(flatppl_name, &args))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::Builder;
    use flatppl_core::Module;
    use flatppl_syntax::{Syntax, parse, print_with};

    fn parsed_text(src: &str) -> String {
        let mut m = Module::new();
        let node = {
            let mut b = Builder::new(&mut m);
            parse_expr_inline(&mut b, src).expect("parse_expr_inline failed")
        };
        {
            let mut b = Builder::new(&mut m);
            b.bind("result", node);
        }
        print_with(&m, Syntax::Minimal)
    }

    fn parsed_fn_text(src: &str, obs: &str) -> String {
        let mut m = Module::new();
        let node = {
            let mut b = Builder::new(&mut m);
            parse_expr_as_fn(&mut b, src, obs).expect("parse_expr_as_fn failed")
        };
        {
            let mut b = Builder::new(&mut m);
            b.bind("result", node);
        }
        print_with(&m, Syntax::Minimal)
    }

    #[test]
    fn linear_expr_structure() {
        // "mu + sigma * x" → add(mu, mul(sigma, x))
        let text = parsed_text("mu + sigma * x");
        eprintln!("linear_expr: {text}");
        assert!(text.contains("mu"), "got: {text}");
        assert!(text.contains("sigma"), "got: {text}");
        assert!(text.contains("x"), "got: {text}");
        // round-trip parse
        let rt = parse(&text);
        assert!(rt.is_ok(), "round-trip failed: {:?}\n{text}", rt.err());
    }

    #[test]
    fn gaussian_exponent_structure() {
        // "exp(-0.5*((x-mu)/sigma)^2)"
        let text = parsed_text("exp(-0.5*((x-mu)/sigma)^2)");
        eprintln!("gaussian_exponent: {text}");
        assert!(text.contains("exp"), "missing exp, got: {text}");
        assert!(text.contains("mu"), "missing mu, got: {text}");
        assert!(text.contains("sigma"), "missing sigma, got: {text}");
        assert!(text.contains("0.5"), "missing 0.5, got: {text}");
        let rt = parse(&text);
        assert!(rt.is_ok(), "round-trip failed: {:?}\n{text}", rt.err());
    }

    #[test]
    fn pi_inlined_as_real() {
        let text = parsed_text("PI * x");
        eprintln!("pi_inline: {text}");
        // PI must NOT appear as an identifier — it is inlined as a literal
        assert!(!text.contains("PI"), "PI should be inlined, got: {text}");
        assert!(
            text.contains("3.14") || text.contains("3.1415"),
            "expected π literal, got: {text}"
        );
        let rt = parse(&text);
        assert!(rt.is_ok(), "round-trip failed: {:?}\n{text}", rt.err());
    }

    #[test]
    fn euler_inlined_as_real() {
        let text = parsed_text("EULER ^ x");
        eprintln!("euler_inline: {text}");
        assert!(
            !text.contains("EULER"),
            "EULER should be inlined, got: {text}"
        );
        let rt = parse(&text);
        assert!(rt.is_ok(), "round-trip failed: {:?}\n{text}", rt.err());
    }

    #[test]
    fn as_fn_produces_lambda() {
        // parse_expr_as_fn wraps in x -> <body>
        let text = parsed_fn_text("mu + sigma * x", "x");
        eprintln!("as_fn: {text}");
        // Lambda form should contain `x ->` or be printed as a functionof
        assert!(
            text.contains("x ->") || text.contains("functionof"),
            "missing lambda, got: {text}"
        );
        // x should NOT appear as a free self-ref (it's the lambda param)
        // mu and sigma still reference the module scope
        assert!(text.contains("mu"), "missing mu, got: {text}");
        assert!(text.contains("sigma"), "missing sigma, got: {text}");
        let rt = parse(&text);
        assert!(rt.is_ok(), "round-trip failed: {:?}\n{text}", rt.err());
    }

    #[test]
    fn normalize_weighted_lebesgue_roundtrips() {
        // Full pattern: normalize(weighted(<fn>, Lebesgue(reals)))
        let mut m = Module::new();
        let fn_node = {
            let mut b = Builder::new(&mut m);
            parse_expr_as_fn(&mut b, "exp(-0.5*((x-mu)/sigma)^2)", "x").unwrap()
        };
        {
            let mut b = Builder::new(&mut m);
            // Lebesgue(reals)
            let reals_sym = b.m.intern("reals");
            let reals_node = b.m.alloc(flatppl_core::node::Node::Const(reals_sym));
            let lebesgue = b.call("Lebesgue", &[reals_node]);
            let weighted = b.call("weighted", &[fn_node, lebesgue]);
            let norm = b.call("normalize", &[weighted]);
            b.bind("gauss_density", norm);
        }
        let text = print_with(&m, Syntax::Minimal);
        eprintln!("normalize_weighted_lebesgue:\n{text}");
        assert!(text.contains("normalize"), "missing normalize, got: {text}");
        assert!(text.contains("weighted"), "missing weighted, got: {text}");
        assert!(text.contains("Lebesgue"), "missing Lebesgue, got: {text}");
        let rt = parse(&text);
        assert!(rt.is_ok(), "round-trip failed: {:?}\n{text}", rt.err());
    }

    #[test]
    fn unsupported_comparison_fails_loud() {
        let mut m = Module::new();
        let mut b = Builder::new(&mut m);
        let result = parse_expr_inline(&mut b, "x == 1");
        assert!(
            matches!(result, Err(Error::Unimplemented(_))),
            "expected Unimplemented error for `==`, got: {:?}",
            result
        );
    }

    #[test]
    fn free_identifiers_excludes_calls_and_constants() {
        // `sqrt`, `abs`, `sin` are calls (followed by `(`) — excluded.
        // `x`, `alpha` are value references — included, in first-occurrence order.
        // `PI` is an inlined constant — excluded.
        let ids = free_identifiers("(1 + 0.1 * abs(x) + sin(sqrt(abs(x * alpha + 0.1)))) / PI");
        assert_eq!(ids, vec!["x".to_string(), "alpha".to_string()]);
    }

    #[test]
    fn free_identifiers_dedups_preserving_order() {
        let ids = free_identifiers("mean2 + sqrt(mean2) - other");
        assert_eq!(ids, vec!["mean2".to_string(), "other".to_string()]);
    }

    #[test]
    fn free_identifiers_bad_expr_is_empty() {
        // Unterminated/garbage tokenization → no discovered identifiers (the real
        // failure surfaces when the expression is parsed).
        assert!(free_identifiers("@@@").is_empty());
    }

    #[test]
    fn unsupported_ternary_fails_loud() {
        let mut m = Module::new();
        let mut b = Builder::new(&mut m);
        let result = parse_expr_inline(&mut b, "x > 0 ? x : 0");
        assert!(
            matches!(result, Err(Error::Unimplemented(_))),
            "expected Unimplemented error for ternary, got: {:?}",
            result
        );
    }

    #[test]
    fn wrap_lambda_multi_builds_two_param_lambda() {
        let mut m = flatppl_core::Module::new();
        {
            let mut b = Builder::new(&mut m);
            // body: logdensityof(Normal(mu = y, sigma = 1.0), x)  with x,y as self-refs
            let y = b.self_ref("y");
            let one = b.lit_real(1.0);
            let normal = b.call_kw("Normal", &[("mu", y), ("sigma", one)]);
            let x = b.self_ref("x");
            let body = b.call("logdensityof", &[normal, x]);
            let lam = super::wrap_lambda_multi(&mut b, body, &["x", "y"]);
            b.bind("w", lam);
        }
        let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
        // Minimal syntax prints the lambda as `functionof(<body>, x = _x_, y = _y_)`
        // (the `(x, y) ->` arrow is the canonical-printer sugar, not Minimal).
        assert!(text.contains("functionof("), "missing lambda: {text}");
        assert!(
            text.contains("x = _x_") && text.contains("y = _y_"),
            "both params not bound as locals: {text}"
        );
        // Body self-refs to x and y were rewritten to the lambda placeholders.
        assert!(
            text.contains("logdensityof(Normal(mu = _y_"),
            "body not rewritten to placeholders: {text}"
        );
    }
}
