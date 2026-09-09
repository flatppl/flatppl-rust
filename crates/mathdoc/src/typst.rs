//! Native Typst math source over the semantic [`ast`](crate::ast).
//!
//! Expressions and statements omit `$` delimiters, so callers can place them
//! in inline math, display equations, or a document. Author text is always
//! quoted; the generated source does not evaluate it as Typst code.

use std::fmt::Write;

use crate::ast::{BigOp, BinOp, Fence, Math, Op, Rel, Slot, Statement, Sym, UnOp};
use crate::names::Atom;

/// A statement in native Typst math syntax, without `$` delimiters.
pub fn statement(stmt: &Statement) -> String {
    format!(
        "{} {} {}",
        expr(&stmt.lhs),
        relation(stmt.rel),
        expr(&stmt.rhs)
    )
}

/// An expression in native Typst math syntax, without `$` delimiters.
pub fn expr(m: &Math) -> String {
    let lines = crate::layout::sum_lines(m);
    if !lines.is_empty() {
        let rows = lines
            .iter()
            .map(|line| {
                let term = expr(line.term);
                let term = if line.parens {
                    fenced(Fence::Paren, Fence::Paren, &term)
                } else {
                    term
                };
                let sign = match line.sign {
                    Some(BinOp::Sub) => "− ",
                    Some(_) => "+ ",
                    None => "",
                };
                format!("{sign}{term}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        return format!("vec(delim: #none, align: #left, {rows})");
    }
    match m {
        Math::Ident(id) => {
            let head = atom(&id.display.head);
            if id.display.subs.is_empty() {
                head
            } else {
                format!(
                    "attach({head}, br: {})",
                    id.display
                        .subs
                        .iter()
                        .map(atom)
                        .collect::<Vec<_>>()
                        .join(" \\, ")
                )
            }
        }
        Math::Num(n) => n.clone(),
        Math::Text(t) => quote(t),
        Math::Str(s) => quote(&format!("\"{s}\"")),
        Math::Sym(s) => symbol(*s).into(),
        Math::Op(op) => operator(*op).into(),
        Math::Row(items) => list(items, " "),
        Math::Sub(base, sub) => format!(
            "attach({}, br: {})",
            operand(m, base, Slot::Base),
            expr(sub)
        ),
        Math::Sup(base, sup) => format!(
            "attach({}, tr: {})",
            operand(m, base, Slot::Base),
            expr(sup)
        ),
        Math::SubSup(base, sub, sup) => format!(
            "attach({}, br: {}, tr: {})",
            operand(m, base, Slot::Base),
            expr(sub),
            expr(sup)
        ),
        Math::Frac(num, den) => format!("frac({}, {})", expr(num), expr(den)),
        Math::Sqrt(arg) => format!("sqrt({})", expr(arg)),
        Math::Fenced { open, close, items } => fenced(*open, *close, &list(items, " \\, ")),
        Math::Apply { head, args } => format!(
            "{} {}",
            expr(head),
            fenced(Fence::Paren, Fence::Paren, &list(args, " \\, "))
        ),
        Math::Binary { op, lhs, rhs } => {
            let glyph = match op {
                BinOp::Add => "+",
                BinOp::Sub => "−",
                BinOp::Juxtapose => "",
                BinOp::Dot => "⋅",
                BinOp::Times => "×",
                BinOp::Otimes => "⊗",
                BinOp::Compose => "∘",
                BinOp::And => "∧",
                BinOp::Or => "∨",
            };
            let lhs = operand(m, lhs, Slot::Left);
            let rhs = operand(m, rhs, Slot::Right);
            if glyph.is_empty() {
                format!("{lhs} {rhs}")
            } else {
                format!("{lhs} {glyph} {rhs}")
            }
        }
        Math::Unary { op, arg } => format!(
            "{} {}",
            match op {
                UnOp::Neg => "−",
                UnOp::Not => "¬",
            },
            operand(m, arg, Slot::Right)
        ),
        Math::Relation { lhs, rel, rhs } => format!(
            "{} {} {}",
            operand(m, lhs, Slot::Left),
            relation(*rel),
            operand(m, rhs, Slot::Right)
        ),
        Math::BigOp { op, sub, sup, body } => {
            let head = match op {
                BigOp::Sum => "sum".into(),
                BigOp::Prod => "product".into(),
                BigOp::Otimes => "times.o.big".into(),
                BigOp::Max => "op(\"max\", limits: #true)".into(),
                BigOp::Min => "op(\"min\", limits: #true)".into(),
                BigOp::Named(name) => format!("op({}, limits: #true)", quote(name)),
            };
            let mut head = format!("limits({head})");
            if let Some(sub) = sub {
                let _ = write!(head, "_({})", expr(sub));
            }
            if let Some(sup) = sup {
                let _ = write!(head, "^({})", expr(sup));
            }
            format!("{head} {}", operand(m, body, Slot::Right))
        }
        Math::Family { body, index, range } => {
            let base = fenced(Fence::Paren, Fence::Paren, &expr(body));
            match range {
                Some((lo, hi)) => format!(
                    "attach({base}, br: {} = {}, tr: {})",
                    expr(index),
                    expr(lo),
                    expr(hi)
                ),
                None => format!("attach({base}, br: {})", expr(index)),
            }
        }
        Math::Matrix(rows) => format!(
            "mat(delim: \"[\", {})",
            rows.iter()
                .map(|row| list(row, ", "))
                .collect::<Vec<_>>()
                .join("; ")
        ),
        Math::Cases(rows) => format!(
            "cases({})",
            rows.iter()
                .map(|(value, condition)| {
                    let condition = condition
                        .as_ref()
                        .map_or_else(|| "\"otherwise\"".into(), |c| format!("\"if\" {}", expr(c)));
                    format!("{} & {condition}", expr(value))
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Math::Overline(arg) => format!("overline({})", expr(arg)),
        Math::Code(source) => format!("#raw({})", quote(source)),
    }
}

/// Quote plain text as a Typst string, including the surrounding quotes.
/// Use `#` before this string when inserting it into markup prose.
pub(crate) fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{{{:x}}}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn atom(atom: &Atom) -> String {
    match atom {
        Atom::Greek(c) | Atom::Letter(c) => c.to_string(),
        Atom::Digits(d) => d.clone(),
        Atom::Word(w) => quote(w),
    }
}

fn list(items: &[Math], separator: &str) -> String {
    items.iter().map(expr).collect::<Vec<_>>().join(separator)
}

fn operand(parent: &Math, child: &Math, slot: Slot) -> String {
    let body = expr(child);
    if parent.needs_parens(child, slot) {
        fenced(Fence::Paren, Fence::Paren, &body)
    } else {
        body
    }
}

fn fenced(open: Fence, close: Fence, body: &str) -> String {
    format!("lr({} {body} {})", fence(open, true), fence(close, false))
}

fn fence(fence: Fence, open: bool) -> &'static str {
    match (fence, open) {
        (Fence::Paren, true) => "\\(",
        (Fence::Paren, false) => "\\)",
        (Fence::Bracket, true) => "\\[",
        (Fence::Bracket, false) => "\\]",
        (Fence::Brace, true) => "\\{",
        (Fence::Brace, false) => "\\}",
        (Fence::Bar, _) => "\\|",
        (Fence::Norm, _) => "‖",
        (Fence::Floor, true) => "⌊",
        (Fence::Floor, false) => "⌋",
        (Fence::Ceil, true) => "⌈",
        (Fence::Ceil, false) => "⌉",
    }
}

pub(crate) fn relation(rel: Rel) -> &'static str {
    match rel {
        Rel::Eq => "=",
        Rel::Ne => "≠",
        Rel::Sim => "∼",
        Rel::In => "∈",
        Rel::Lt => "<",
        Rel::Le => "≤",
        Rel::Gt => ">",
        Rel::Ge => "≥",
        Rel::MapsTo => "↦",
    }
}

fn symbol(sym: Sym) -> &'static str {
    match sym {
        Sym::Infinity => "∞",
        Sym::Pi => "π",
        Sym::ImagUnit => "upright(i)",
        Sym::Reals => "ℝ",
        Sym::ExtendedReals => "overline(ℝ)",
        Sym::Normal => "cal(N)",
        Sym::Uniform => "cal(U)",
        Sym::Law => "op(\"Law\")",
        Sym::Integers => "ℤ",
        Sym::Complexes => "ℂ",
        Sym::Booleans => "𝔹",
        Sym::NonNegIntegers => "ℕ_(0)",
        Sym::PosIntegers => "ℤ_(> 0)",
        Sym::Lebesgue => "λ",
        Sym::Dirac => "δ",
        Sym::Simplex => "Δ",
        Sym::Ellipsis => "…",
        Sym::Placeholder => "⋅",
        Sym::Euler => "e",
    }
}

fn operator(op: Op) -> &'static str {
    match op {
        Op::Bar | Op::Restrict => "\\|",
        Op::Star => "∗",
        Op::Times => "×",
        Op::Dot => ".",
        Op::Cdot => "⋅",
        Op::Comma => "\\,",
        Op::Differential => "upright(d)",
        Op::Transpose => "upright(T)",
        Op::Dagger => "†",
    }
}
