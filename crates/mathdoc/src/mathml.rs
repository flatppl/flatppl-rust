//! MathML printer over the [`ast`](crate::ast).
//!
//! Emits MathML Core. Every identifier that denotes a module binding carries
//! `data-flatppl-ref="<name>"` on its outermost element (`<mi>`, or `<msub>`
//! when it has subscripts), and a statement fragment carries
//! `data-flatppl-binding="<name>"` on its `<math>` root — the two hooks the
//! viewer wires click-to-focus and click-to-source from. All text is escaped
//! here; the output is trusted markup.
//!
//! The printer recurses over the tree. The lowering guarantees every
//! [`Statement`] it hands out is at most [`flatppl_core::DEFAULT_MAX_DEPTH`]
//! levels deep (deeper ones come back as source text), so the recursion is
//! bounded.

use std::fmt::Write;

use crate::ast::{BigOp, BinOp, Fence, Ident, Math, Op, Rel, Slot, Statement, Sym, UnOp};
use crate::names::Atom;

/// A whole statement as a display-math fragment rooted at `<math>`.
pub fn fragment(binding: &str, stmt: &Statement) -> String {
    format!(
        "<math display=\"block\" data-flatppl-binding=\"{}\">{}</math>",
        escape(binding),
        statement(stmt)
    )
}

/// The statement's row without the `<math>` root: `<mrow>lhs <mo>rel</mo> rhs</mrow>`.
pub fn statement(stmt: &Statement) -> String {
    format!(
        "<mrow>{}<mo>{}</mo>{}</mrow>",
        expr(&stmt.lhs),
        rel_glyph(stmt.rel),
        expr(&stmt.rhs)
    )
}

/// The three aligned parts of a statement (`lhs`, relation glyph, `rhs`), for
/// a document that lays rows out in one `<mtable>`.
pub fn statement_parts(stmt: &Statement) -> (String, &'static str, String) {
    (expr(&stmt.lhs), rel_glyph(stmt.rel), expr(&stmt.rhs))
}

/// An expression as MathML content (no `<math>` root).
pub fn expr(m: &Math) -> String {
    let mut out = String::new();
    write_expr(&mut out, m);
    out
}

pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            c => out.push(c),
        }
    }
    out
}

fn rel_glyph(rel: Rel) -> &'static str {
    match rel {
        Rel::Eq => "=",
        Rel::Ne => "≠",
        Rel::Sim => "∼",
        Rel::In => "∈",
        Rel::Lt => "&lt;",
        Rel::Le => "≤",
        Rel::Gt => "&gt;",
        Rel::Ge => "≥",
        Rel::MapsTo => "↦",
    }
}

fn binop_glyph(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "−",
        BinOp::Juxtapose => "&#x2062;",
        BinOp::Dot => "⋅",
        BinOp::Times => "×",
        BinOp::Otimes => "⊗",
        BinOp::Compose => "∘",
        BinOp::And => "∧",
        BinOp::Or => "∨",
    }
}

fn fence_glyph(f: Fence, open: bool) -> &'static str {
    match (f, open) {
        (Fence::Paren, true) => "(",
        (Fence::Paren, false) => ")",
        (Fence::Bracket, true) => "[",
        (Fence::Bracket, false) => "]",
        (Fence::Brace, true) => "{",
        (Fence::Brace, false) => "}",
        (Fence::Bar, _) => "|",
        (Fence::Norm, _) => "‖",
        (Fence::Floor, true) => "⌊",
        (Fence::Floor, false) => "⌋",
        (Fence::Ceil, true) => "⌈",
        (Fence::Ceil, false) => "⌉",
    }
}

fn write_expr(out: &mut String, m: &Math) {
    let lines = crate::layout::sum_lines(m);
    if !lines.is_empty() {
        out.push_str("<mtable class=\"flatppl-sum\" columnalign=\"left\" rowspacing=\"0.25em\">");
        for line in lines {
            out.push_str("<mtr><mtd style=\"text-align: left\"><mrow>");
            if let Some(sign) = line.sign {
                let glyph = if sign == BinOp::Sub { "−" } else { "+" };
                let _ = write!(out, "<mo form=\"infix\">{glyph}</mo>");
            }
            if line.parens {
                write_fenced(out, "(", ")", is_tall(line.term), |out| {
                    write_expr(out, line.term)
                });
            } else {
                write_expr(out, line.term);
            }
            out.push_str("</mrow></mtd></mtr>");
        }
        out.push_str("</mtable>");
        return;
    }
    match m {
        Math::Ident(id) => write_ident(out, id),
        Math::Num(n) => {
            let _ = write!(out, "<mn>{}</mn>", escape(n));
        }
        Math::Text(t) => {
            let _ = write!(out, "<mi>{}</mi>", escape(t));
        }
        Math::Str(s) => {
            let _ = write!(out, "<mtext>\"{}\"</mtext>", escape(s));
        }
        Math::Sym(s) => write_sym(out, *s),
        Math::Op(op) => write_op(out, *op),
        Math::Row(items) => {
            out.push_str("<mrow>");
            for item in items {
                write_expr(out, item);
            }
            out.push_str("</mrow>");
        }
        Math::Sub(base, sub) => {
            out.push_str("<msub>");
            write_base(out, m, base);
            write_expr(out, sub);
            out.push_str("</msub>");
        }
        Math::Sup(base, sup) => {
            out.push_str("<msup>");
            write_base(out, m, base);
            write_expr(out, sup);
            out.push_str("</msup>");
        }
        Math::SubSup(base, sub, sup) => {
            out.push_str("<msubsup>");
            write_base(out, m, base);
            write_expr(out, sub);
            write_expr(out, sup);
            out.push_str("</msubsup>");
        }
        Math::Frac(num, den) => {
            out.push_str("<mfrac>");
            write_wrapped(out, num);
            write_wrapped(out, den);
            out.push_str("</mfrac>");
        }
        Math::Sqrt(arg) => {
            out.push_str("<msqrt>");
            write_expr(out, arg);
            out.push_str("</msqrt>");
        }
        Math::Fenced { open, close, items } => {
            let tall = items.iter().any(is_tall);
            write_fenced(
                out,
                fence_glyph(*open, true),
                fence_glyph(*close, false),
                tall,
                |out| write_list(out, items),
            );
        }
        Math::Apply { head, args } => {
            out.push_str("<mrow>");
            write_expr(out, head);
            out.push_str("<mo>&#x2061;</mo>");
            let tall = args.iter().any(is_tall);
            write_fenced(out, "(", ")", tall, |out| write_list(out, args));
            out.push_str("</mrow>");
        }
        Math::Binary { op, lhs, rhs } => {
            out.push_str("<mrow>");
            write_operand(out, m, lhs, Slot::Left);
            let _ = write!(out, "<mo>{}</mo>", binop_glyph(*op));
            write_operand(out, m, rhs, Slot::Right);
            out.push_str("</mrow>");
        }
        Math::Unary { op, arg } => {
            let glyph = match op {
                UnOp::Neg => "−",
                UnOp::Not => "¬",
            };
            let _ = write!(out, "<mrow><mo>{glyph}</mo>");
            write_operand(out, m, arg, Slot::Right);
            out.push_str("</mrow>");
        }
        Math::Relation { lhs, rel, rhs } => {
            out.push_str("<mrow>");
            write_operand(out, m, lhs, Slot::Left);
            let _ = write!(out, "<mo>{}</mo>", rel_glyph(*rel));
            write_operand(out, m, rhs, Slot::Right);
            out.push_str("</mrow>");
        }
        Math::BigOp { op, sub, sup, body } => {
            out.push_str("<mrow>");
            let glyph = match op {
                BigOp::Sum => "<mo>∑</mo>".to_string(),
                BigOp::Prod => "<mo>∏</mo>".to_string(),
                BigOp::Otimes => "<mo>⨂</mo>".to_string(),
                BigOp::Max => "<mi>max</mi>".to_string(),
                BigOp::Min => "<mi>min</mi>".to_string(),
                BigOp::Named(name) => format!("<mi>{}</mi>", escape(name)),
            };
            match (sub, sup) {
                (Some(sub), Some(sup)) => {
                    let _ = write!(out, "<munderover>{glyph}");
                    write_wrapped(out, sub);
                    write_wrapped(out, sup);
                    out.push_str("</munderover>");
                }
                (Some(sub), None) => {
                    let _ = write!(out, "<munder>{glyph}");
                    write_wrapped(out, sub);
                    out.push_str("</munder>");
                }
                (None, Some(sup)) => {
                    let _ = write!(out, "<mover>{glyph}");
                    write_wrapped(out, sup);
                    out.push_str("</mover>");
                }
                (None, None) => out.push_str(&glyph),
            }
            write_operand(out, m, body, Slot::Right);
            out.push_str("</mrow>");
        }
        Math::Family { body, index, range } => {
            let tag = if range.is_some() { "msubsup" } else { "msub" };
            let _ = write!(out, "<{tag}>");
            write_fenced(out, "(", ")", is_tall(body), |out| write_expr(out, body));
            match range {
                Some((lo, hi)) => {
                    out.push_str("<mrow>");
                    write_expr(out, index);
                    out.push_str("<mo>=</mo>");
                    write_expr(out, lo);
                    out.push_str("</mrow>");
                    write_wrapped(out, hi);
                }
                None => write_wrapped(out, index),
            }
            let _ = write!(out, "</{tag}>");
        }
        Math::Matrix(rows) => {
            out.push_str("<mrow><mo>[</mo><mtable>");
            for row in rows {
                out.push_str("<mtr>");
                for cell in row {
                    out.push_str("<mtd>");
                    write_expr(out, cell);
                    out.push_str("</mtd>");
                }
                out.push_str("</mtr>");
            }
            out.push_str("</mtable><mo>]</mo></mrow>");
        }
        Math::Cases(rows) => {
            out.push_str("<mrow><mo>{</mo><mtable columnalign=\"left left\">");
            for (value, cond) in rows {
                out.push_str("<mtr><mtd>");
                write_expr(out, value);
                out.push_str("</mtd><mtd>");
                match cond {
                    Some(c) => {
                        out.push_str("<mtext>if&#xa0;</mtext>");
                        write_expr(out, c);
                    }
                    None => out.push_str("<mtext>otherwise</mtext>"),
                }
                out.push_str("</mtd></mtr>");
            }
            out.push_str("</mtable></mrow>");
        }
        Math::Overline(arg) => {
            out.push_str("<mover>");
            write_wrapped(out, arg);
            out.push_str("<mo>‾</mo></mover>");
        }
        Math::Code(text) => {
            let _ = write!(
                out,
                "<mtext class=\"flatppl-code\">{}</mtext>",
                escape(text)
            );
        }
    }
}

/// An identifier: head `<mi>` plus an optional `<msub>` of its parts, the
/// back-reference attribute on the outermost element.
fn write_ident(out: &mut String, id: &Ident) {
    let attr = match &id.target {
        Some(t) => format!(" data-flatppl-ref=\"{}\"", escape(t)),
        None => String::new(),
    };
    if id.display.subs.is_empty() {
        let _ = write!(out, "<mi{attr}>{}</mi>", atom_text(&id.display.head));
        return;
    }
    let _ = write!(out, "<msub{attr}><mi>{}</mi>", atom_text(&id.display.head));
    if id.display.subs.len() == 1 {
        write_atom(out, &id.display.subs[0]);
    } else {
        out.push_str("<mrow>");
        for (i, part) in id.display.subs.iter().enumerate() {
            if i > 0 {
                out.push_str("<mo>,</mo>");
            }
            write_atom(out, part);
        }
        out.push_str("</mrow>");
    }
    out.push_str("</msub>");
}

fn atom_text(atom: &Atom) -> String {
    match atom {
        Atom::Greek(c) | Atom::Letter(c) => c.to_string(),
        Atom::Digits(d) => escape(d),
        Atom::Word(w) => escape(w),
    }
}

fn write_atom(out: &mut String, atom: &Atom) {
    match atom {
        Atom::Digits(d) => {
            let _ = write!(out, "<mn>{}</mn>", escape(d));
        }
        other => {
            let _ = write!(out, "<mi>{}</mi>", atom_text(other));
        }
    }
}

fn write_sym(out: &mut String, s: Sym) {
    out.push_str(match s {
        Sym::Infinity => "<mi>∞</mi>",
        Sym::Pi => "<mi>π</mi>",
        Sym::ImagUnit => "<mi mathvariant=\"normal\">i</mi>",
        Sym::Reals => "<mi>ℝ</mi>",
        Sym::ExtendedReals => "<mover><mi>ℝ</mi><mo>¯</mo></mover>",
        Sym::Normal => "<mi>𝒩</mi>",
        Sym::Uniform => "<mi>𝒰</mi>",
        Sym::Law => "<mi>ℒ</mi>",
        Sym::Integers => "<mi>ℤ</mi>",
        Sym::Complexes => "<mi>ℂ</mi>",
        Sym::Booleans => "<mi>𝔹</mi>",
        Sym::NonNegIntegers => "<msub><mi>ℕ</mi><mn>0</mn></msub>",
        Sym::PosIntegers => "<msub><mi>ℤ</mi><mrow><mo>&gt;</mo><mn>0</mn></mrow></msub>",
        Sym::Lebesgue => "<mi>λ</mi>",
        Sym::Dirac => "<mi>δ</mi>",
        Sym::Simplex => "<mi>Δ</mi>",
        Sym::Ellipsis => "<mi>…</mi>",
        Sym::Placeholder => "<mo>·</mo>",
        Sym::Euler => "<mi>e</mi>",
    });
}

fn write_op(out: &mut String, op: Op) {
    out.push_str(match op {
        Op::Bar => "<mo stretchy=\"false\">|</mo>",
        Op::Star => "<mo>∗</mo>",
        Op::Times => "<mo>×</mo>",
        Op::Dot => "<mo>.</mo>",
        Op::Cdot => "<mo>⋅</mo>",
        Op::Comma => "<mo>,</mo>",
        Op::Restrict => "<mo stretchy=\"false\">|</mo>",
        Op::Differential => "<mi mathvariant=\"normal\">d</mi>",
        Op::Transpose => "<mi mathvariant=\"normal\">T</mi>",
        Op::Dagger => "<mo>†</mo>",
    });
}

/// `<mrow>` `open` … `close` `</mrow>` around what `body` writes. The fences
/// stretch only around content taller than a line: MathML Core stretches
/// every fence by default, and Chromium gives a stretchy fence a wider
/// advance even around plain content, which read as `Normal ( μ, τ )`.
/// LaTeX has the same rule — `(` is rigid unless written `\left(`.
fn write_fenced(
    out: &mut String,
    open: &str,
    close: &str,
    tall: bool,
    body: impl FnOnce(&mut String),
) {
    let attr = if tall { "" } else { " stretchy=\"false\"" };
    let _ = write!(out, "<mrow><mo{attr}>{open}</mo>");
    body(out);
    let _ = write!(out, "<mo{attr}>{close}</mo></mrow>");
}

/// Whether `m` is taller than a line of text: a fraction, matrix, case
/// distinction or big operator anywhere in it, or a script that itself
/// holds a fence or fraction (`M|_{[0, ∞]}` hangs below the line).
fn is_tall(m: &Math) -> bool {
    let mut stack = vec![m];
    while let Some(n) = stack.pop() {
        if !crate::layout::sum_lines(n).is_empty() {
            return true;
        }
        match n {
            Math::Frac(..) | Math::Matrix(_) | Math::Cases(_) | Math::BigOp { .. } => return true,
            Math::Sub(_, s) | Math::Sup(_, s) if holds_fence(s) => return true,
            Math::SubSup(_, a, b) if holds_fence(a) || holds_fence(b) => return true,
            _ => {}
        }
        stack.extend(n.children());
    }
    false
}

fn holds_fence(m: &Math) -> bool {
    let mut stack = vec![m];
    while let Some(n) = stack.pop() {
        if matches!(n, Math::Fenced { .. } | Math::Frac(..)) {
            return true;
        }
        stack.extend(n.children());
    }
    false
}

/// Comma-separated items.
fn write_list(out: &mut String, items: &[Math]) {
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push_str("<mo>,</mo>");
        }
        write_expr(out, item);
    }
}

/// A child in a slot that must hold exactly one element (`<mfrac>`,
/// `<munder>`, a script). Every [`Math`] variant prints as exactly one
/// element — a multi-token expression is already an `<mrow>` — so this is
/// [`write_expr`] under a name that states the requirement.
fn write_wrapped(out: &mut String, m: &Math) {
    write_expr(out, m);
}

/// The base of the script `parent`: bracketed when the tree rules say so.
fn write_base(out: &mut String, parent: &Math, base: &Math) {
    if parent.needs_parens(base, Slot::Base) {
        write_fenced(out, "(", ")", is_tall(base), |out| write_expr(out, base));
    } else {
        write_wrapped(out, base);
    }
}

/// An operand of `parent`, bracketed when the tree's precedence rules say so.
fn write_operand(out: &mut String, parent: &Math, child: &Math, slot: Slot) {
    if parent.needs_parens(child, slot) {
        write_fenced(out, "(", ")", is_tall(child), |out| write_expr(out, child));
    } else {
        write_expr(out, child);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Math;

    fn b(name: &str) -> Math {
        Math::binding(name)
    }

    #[test]
    fn identifiers_carry_the_back_reference_on_the_outermost_element() {
        assert_eq!(expr(&b("mu")), "<mi data-flatppl-ref=\"mu\">μ</mi>");
        assert_eq!(
            expr(&b("theta1")),
            "<msub data-flatppl-ref=\"theta1\"><mi>θ</mi><mn>1</mn></msub>"
        );
        assert_eq!(
            expr(&b("E1_data")),
            "<msub data-flatppl-ref=\"E1_data\"><mi>E</mi><mrow><mn>1</mn><mo>,</mo><mi>data</mi></mrow></msub>"
        );
        assert_eq!(expr(&Math::ident("par", None)), "<mi>par</mi>");
    }

    #[test]
    fn juxtaposition_uses_the_invisible_times_and_the_dot_is_explicit() {
        let m = Math::plus(Math::times(Math::int(2), b("x")), Math::int(1));
        assert_eq!(
            expr(&m),
            "<mrow><mrow><mn>2</mn><mo>&#x2062;</mo><mi data-flatppl-ref=\"x\">x</mi></mrow><mo>+</mo><mn>1</mn></mrow>"
        );
        let m = Math::dot(b("x"), Math::int(2));
        assert!(expr(&m).contains("<mo>⋅</mo>"));
    }

    #[test]
    fn brackets_follow_the_tree_rules() {
        let inner = Math::minus(b("b"), b("c"));
        let m = Math::minus(b("a"), inner.clone());
        assert!(expr(&m).contains("<mo stretchy=\"false\">(</mo>"));
        let m = Math::minus(inner, b("a"));
        assert!(!expr(&m).contains("<mo stretchy=\"false\">(</mo>"));
        let m = Math::times(Math::plus(b("a"), b("b")), b("c"));
        assert!(expr(&m).starts_with("<mrow><mrow><mo stretchy=\"false\">(</mo>"));
        let m = Math::pow(Math::negate(Math::int(2)), Math::int(3));
        assert_eq!(
            expr(&m),
            "<msup><mrow><mo stretchy=\"false\">(</mo><mrow><mo>−</mo><mn>2</mn></mrow><mo stretchy=\"false\">)</mo></mrow><mn>3</mn></msup>"
        );
        let m = Math::pow(
            Math::subscript(b("x"), Math::ident("i", None)),
            Math::int(2),
        );
        assert!(!expr(&m).contains("<mo stretchy=\"false\">(</mo>"));
        // A compound subscript base is bracketed too: `(a + b)_2`.
        let m = Math::subscript(Math::plus(b("a"), b("b")), Math::int(2));
        assert!(expr(&m).starts_with("<msub><mrow><mo stretchy=\"false\">(</mo>"));
    }

    #[test]
    fn a_statement_fragment_is_rooted_at_math_with_the_binding_attribute() {
        let stmt = Statement {
            lhs: b("mu"),
            rel: Rel::Sim,
            rhs: Math::call("Normal", vec![Math::int(0), Math::int(5)]),
        };
        assert_eq!(
            fragment("mu", &stmt),
            "<math display=\"block\" data-flatppl-binding=\"mu\"><mrow><mi data-flatppl-ref=\"mu\">μ</mi><mo>∼</mo>\
             <mrow><mi>Normal</mi><mo>&#x2061;</mo><mrow><mo stretchy=\"false\">(</mo><mn>0</mn><mo>,</mo><mn>5</mn><mo stretchy=\"false\">)</mo></mrow></mrow></mrow></math>"
        );
    }

    #[test]
    fn text_is_escaped_everywhere() {
        let m = Math::Str("a<b&\"c\"".into());
        assert_eq!(expr(&m), "<mtext>\"a&lt;b&amp;&quot;c&quot;\"</mtext>");
        let m = Math::relation(b("x"), Rel::Lt, Math::int(0));
        assert!(expr(&m).contains("<mo>&lt;</mo>"));
        let m = Math::Code("y = f(x) < 2".into());
        assert_eq!(
            expr(&m),
            "<mtext class=\"flatppl-code\">y = f(x) &lt; 2</mtext>"
        );
        assert!(
            fragment(
                "a\"b",
                &Statement {
                    lhs: b("x"),
                    rel: Rel::Eq,
                    rhs: Math::int(1)
                }
            )
            .contains("data-flatppl-binding=\"a&quot;b\"")
        );
    }

    #[test]
    fn families_and_big_operators_lay_out_their_limits() {
        let fam = Math::family(
            Math::plus(b("a"), Math::subscript(b("x"), Math::ident("i", None))),
            Math::ident("i", None),
            Some((Math::int(1), b("N"))),
        );
        let s = expr(&fam);
        assert!(s.starts_with("<msubsup><mrow><mo stretchy=\"false\">(</mo>"));
        assert!(s.contains(
            "<mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi data-flatppl-ref=\"N\">N</mi></msubsup>"
        ));
        let sum = Math::big(
            BigOp::Sum,
            Some(Math::ident("j", None)),
            None,
            Math::times(
                Math::subscript(b("A"), Math::ident("ij", None)),
                Math::subscript(b("B"), Math::ident("jk", None)),
            ),
        );
        assert!(expr(&sum).starts_with("<mrow><munder><mo>∑</mo><mi>j</mi></munder>"));
        let prod = Math::big(
            BigOp::Otimes,
            Some(Math::relation(
                Math::ident("i", None),
                Rel::Eq,
                Math::int(1),
            )),
            Some(b("n")),
            Math::call(
                "Normal",
                vec![Math::subscript(b("mu"), Math::ident("i", None)), b("s")],
            ),
        );
        assert!(expr(&prod).contains("<munderover><mo>⨂</mo>"));
    }

    #[test]
    fn conditional_densities_and_restrictions_use_bare_operators() {
        let p = Math::apply(
            Math::subscript(Math::letter('p'), b("K")),
            vec![Math::row(vec![b("y_data"), Math::Op(Op::Bar), b("theta")])],
        );
        let s = expr(&p);
        assert!(s.starts_with(
            "<mrow><msub><mi>p</mi><mi data-flatppl-ref=\"K\">K</mi></msub><mo>&#x2061;</mo>"
        ));
        assert!(s.contains("<mo stretchy=\"false\">|</mo>"));
        let r = Math::subscript(
            Math::row(vec![
                Math::call("Cauchy", vec![Math::int(0), Math::int(5)]),
                Math::Op(Op::Restrict),
            ]),
            Math::bracket(vec![Math::int(0), Math::Sym(Sym::Infinity)]),
        );
        assert!(
            expr(&r).ends_with("<mo stretchy=\"false\">[</mo><mn>0</mn><mo>,</mo><mi>∞</mi><mo stretchy=\"false\">]</mo></mrow></msub>")
        );
    }

    #[test]
    fn cases_and_matrices_render_as_tables() {
        let c = Math::Cases(vec![
            (b("a"), Some(Math::relation(b("x"), Rel::Gt, Math::int(0)))),
            (b("b"), None),
        ]);
        let s = expr(&c);
        assert!(s.starts_with("<mrow><mo>{</mo><mtable columnalign=\"left left\">"));
        assert!(s.contains("<mtext>otherwise</mtext>"));
        let m = Math::Matrix(vec![
            vec![Math::int(1), Math::int(2)],
            vec![Math::int(3), Math::int(4)],
        ]);
        assert!(expr(&m).contains("<mtr><mtd><mn>3</mn></mtd><mtd><mn>4</mn></mtd></mtr>"));
    }

    #[test]
    fn single_element_slots_hold_one_element() {
        let f = Math::frac(Math::plus(b("a"), b("b")), b("c"));
        assert_eq!(
            expr(&f),
            "<mfrac><mrow><mi data-flatppl-ref=\"a\">a</mi><mo>+</mo><mi data-flatppl-ref=\"b\">b</mi></mrow><mi data-flatppl-ref=\"c\">c</mi></mfrac>"
        );
    }

    #[test]
    fn big_operators_bracket_where_something_follows_them() {
        let sum = Math::big(
            BigOp::Sum,
            Some(Math::ident("i", None)),
            None,
            Math::subscript(b("x"), Math::ident("i", None)),
        );
        // Left operand: bracketed. Right operand: not.
        let left = expr(&Math::plus(sum.clone(), Math::int(1)));
        assert!(
            left.starts_with("<mrow><mrow><mo>(</mo><mrow><munder>"),
            "{left}"
        );
        let right = expr(&Math::plus(Math::int(1), sum.clone()));
        assert!(
            right.starts_with("<mrow><mn>1</mn><mo>+</mo><mrow><munder>"),
            "{right}"
        );
        // Negated and nested: not bracketed.
        let neg = expr(&Math::negate(sum.clone()));
        assert!(neg.starts_with("<mrow><mo>−</mo><mrow><munder>"), "{neg}");
        // As a power's base: bracketed.
        let sq = expr(&Math::pow(sum.clone(), Math::int(2)));
        assert!(sq.starts_with("<msup><mrow><mo>(</mo>"), "{sq}");
        // On the left of a relation: bracketed; on the right: not.
        let rel = expr(&Math::relation(sum.clone(), Rel::Eq, Math::int(1)));
        assert!(
            rel.starts_with("<mrow><mrow><mo>(</mo><mrow><munder>"),
            "{rel}"
        );
        let rel = expr(&Math::relation(Math::int(1), Rel::Eq, sum.clone()));
        assert!(
            rel.starts_with("<mrow><mn>1</mn><mo>=</mo><mrow><munder>"),
            "{rel}"
        );
        // As an argument: the call's own parentheses and the comma delimit it.
        let arg = expr(&Math::call("Normal", vec![sum.clone(), Math::int(1)]));
        assert_eq!(arg.matches(">(</mo>").count(), 1, "{arg}");
        // An additive body is bracketed, a product body is not.
        let add_body = Math::big(
            BigOp::Prod,
            Some(Math::ident("j", None)),
            None,
            Math::plus(b("a"), b("c")),
        );
        assert!(
            expr(&add_body).contains("</munder><mrow><mo stretchy=\"false\">(</mo>"),
            "{}",
            expr(&add_body)
        );
        assert!(!expr(&sum).contains("<mo stretchy=\"false\">(</mo>"));
    }
}
