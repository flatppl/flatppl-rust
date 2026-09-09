//! Plain TeX export from the semantic tree, without math delimiters.

use crate::ast::{BigOp, BinOp, Fence, Math, Op, Rel, Slot, Statement, Sym, UnOp};
use crate::names::Atom;

pub fn statement(stmt: &Statement) -> String {
    format!(
        "{} {} {}",
        expr(&stmt.lhs),
        relation(stmt.rel),
        expr(&stmt.rhs)
    )
}

pub fn expr(m: &Math) -> String {
    let lines = crate::layout::sum_lines(m);
    if !lines.is_empty() {
        let rows = lines
            .iter()
            .map(|line| {
                let term = expr(line.term);
                let term = if line.parens {
                    format!(r"\left({term}\right)")
                } else {
                    term
                };
                let sign = match line.sign {
                    Some(BinOp::Sub) => "- ",
                    Some(_) => "+ ",
                    None => "",
                };
                format!("&{sign}{term}")
            })
            .collect::<Vec<_>>()
            .join(r" \\ ");
        return format!(r"\begin{{aligned}}{rows}\end{{aligned}}");
    }
    let render = |m: &Math| expr(m);
    let operand = |child: &Math, slot| {
        let s = render(child);
        if m.needs_parens(child, slot) {
            format!(r"\left({s}\right)")
        } else {
            s
        }
    };
    let list = |items: &[Math]| items.iter().map(render).collect::<Vec<_>>().join(", ");
    match m {
        Math::Ident(id) => {
            let mut s = atom(&id.display.head);
            if !id.display.subs.is_empty() {
                s.push_str(&format!(
                    "_{{{}}}",
                    id.display
                        .subs
                        .iter()
                        .map(atom)
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            s
        }
        Math::Num(n) => n.clone(),
        Math::Text(t) => format!(r"\operatorname{{{}}}", escape(t)),
        Math::Str(s) => format!(r#"\text{{"{}"}}"#, escape(s)),
        Math::Sym(s) => symbol(*s).into(),
        Math::Op(op) => match op {
            Op::Bar => r"\mid",
            Op::Star => r"\ast",
            Op::Times => r"\times",
            Op::Dot => ".",
            Op::Cdot => r"\cdot",
            Op::Comma => ",",
            Op::Restrict => r"\vert",
            Op::ThinSpace => r"\,",
            Op::Differential => r"\mathrm{d}",
            Op::Transpose => r"\mathrm{T}",
            Op::Dagger => r"\dagger",
        }
        .into(),
        Math::Row(items) => items.iter().map(render).collect::<Vec<_>>().join(" "),
        Math::Sub(base, sub) => format!("{{{}}}_{{{}}}", operand(base, Slot::Base), render(sub)),
        Math::Sup(base, sup) => format!("{{{}}}^{{{}}}", operand(base, Slot::Base), render(sup)),
        Math::SubSup(base, sub, sup) => format!(
            "{{{}}}_{{{}}}^{{{}}}",
            operand(base, Slot::Base),
            render(sub),
            render(sup)
        ),
        Math::Frac(num, den) => format!(r"\frac{{{}}}{{{}}}", render(num), render(den)),
        Math::Sqrt(arg) => format!(r"\sqrt{{{}}}", render(arg)),
        Math::Fenced { open, close, items } => format!(
            r"\left{}{}\right{}",
            fence(*open, true),
            list(items),
            fence(*close, false)
        ),
        Math::Apply { head, args } => format!(r"{}\left({}\right)", render(head), list(args)),
        Math::Binary { op, lhs, rhs } => {
            let op = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Juxtapose => "",
                BinOp::Dot => r"\cdot",
                BinOp::Times => r"\times",
                BinOp::Otimes => r"\otimes",
                BinOp::Compose => r"\circ",
                BinOp::And => r"\land",
                BinOp::Or => r"\lor",
            };
            format!(
                "{} {op} {}",
                operand(lhs, Slot::Left),
                operand(rhs, Slot::Right)
            )
        }
        Math::Unary { op, arg } => format!(
            "{} {}",
            match op {
                UnOp::Neg => "-",
                UnOp::Not => r"\neg",
            },
            operand(arg, Slot::Right)
        ),
        Math::Relation { lhs, rel, rhs } => format!(
            "{} {} {}",
            operand(lhs, Slot::Left),
            relation(*rel),
            operand(rhs, Slot::Right)
        ),
        Math::BigOp { op, sub, sup, body } => {
            let mut s = match op {
                BigOp::Sum => r"\sum".into(),
                BigOp::Prod => r"\prod".into(),
                BigOp::Otimes => r"\bigotimes".into(),
                BigOp::Max => r"\max".into(),
                BigOp::Min => r"\min".into(),
                BigOp::Named(name) => format!(r"\operatorname*{{{}}}", escape(name)),
                BigOp::Integral => r"\int".into(),
            };
            if let Some(sub) = sub {
                s.push_str(&format!("_{{{}}}", render(sub)));
            }
            if let Some(sup) = sup {
                s.push_str(&format!("^{{{}}}", render(sup)));
            }
            format!("{s} {}", operand(body, Slot::Right))
        }
        Math::Family { body, index, range } => {
            let index = render(index);
            let body = render(body);
            match range {
                Some((lo, hi)) => format!(
                    r"\left({body}\right)_{{{index} = {}}}^{{{}}}",
                    render(lo),
                    render(hi)
                ),
                None => format!(r"\left({body}\right)_{{{index}}}"),
            }
        }
        Math::Matrix(rows) => format!(
            r"\begin{{bmatrix}}{}\end{{bmatrix}}",
            rows.iter()
                .map(|row| row.iter().map(render).collect::<Vec<_>>().join(" & "))
                .collect::<Vec<_>>()
                .join(r" \\ ")
        ),
        Math::Cases(rows) => format!(
            r"\begin{{cases}}{}\end{{cases}}",
            rows.iter()
                .map(|(value, cond)| {
                    format!(
                        "{} & {}",
                        render(value),
                        cond.as_ref().map_or_else(
                            || r"\text{otherwise}".into(),
                            |c| format!(r"\text{{if }}{}", render(c))
                        )
                    )
                })
                .collect::<Vec<_>>()
                .join(r" \\ ")
        ),
        Math::Overline(arg) => format!(r"\overline{{{}}}", render(arg)),
        Math::Code(s) => format!(r"\texttt{{{}}}", escape(s)),
    }
}

fn atom(a: &Atom) -> String {
    match a {
        Atom::Greek(c) | Atom::Letter(c) => c.to_string(),
        Atom::Digits(s) => s.clone(),
        Atom::Word(s) => format!(r"\mathrm{{{}}}", escape(s)),
    }
}

/// Escape source text inside a TeX text argument, never as commands.
pub(crate) fn escape(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\\' => r"\textbackslash{}".into(),
            '{' => r"\{".into(),
            '}' => r"\}".into(),
            '$' => r"\$".into(),
            '&' => r"\&".into(),
            '#' => r"\#".into(),
            '%' => r"\%".into(),
            '_' => r"\_".into(),
            '^' => r"\textasciicircum{}".into(),
            '~' => r"\textasciitilde{}".into(),
            // Text is single-line even when a fallback quotes several source lines.
            c if c.is_whitespace() => " ".into(),
            c => c.to_string(),
        })
        .collect()
}

pub(crate) fn relation(rel: Rel) -> &'static str {
    match rel {
        Rel::Eq => "=",
        Rel::Ne => r"\ne",
        Rel::Sim => r"\sim",
        Rel::In => r"\in",
        Rel::Lt => "<",
        Rel::Le => r"\le",
        Rel::Gt => ">",
        Rel::Ge => r"\ge",
        Rel::MapsTo => r"\mapsto",
    }
}

fn fence(f: Fence, open: bool) -> &'static str {
    match (f, open) {
        (Fence::Paren, true) => "(",
        (Fence::Paren, false) => ")",
        (Fence::Bracket, true) => "[",
        (Fence::Bracket, false) => "]",
        (Fence::Brace, true) => r"\{",
        (Fence::Brace, false) => r"\}",
        (Fence::Bar, _) => "|",
        // TeX control words need a boundary before a letter-valued body.
        (Fence::Norm, _) => "\\Vert ",
        (Fence::Floor, true) => "\\lfloor ",
        (Fence::Floor, false) => "\\rfloor ",
        (Fence::Ceil, true) => "\\lceil ",
        (Fence::Ceil, false) => "\\rceil ",
    }
}

fn symbol(s: Sym) -> &'static str {
    match s {
        Sym::Infinity => r"\infty",
        Sym::Pi => r"\pi",
        Sym::ImagUnit => r"\mathrm{i}",
        Sym::Reals => r"\mathbb{R}",
        Sym::ExtendedReals => r"\overline{\mathbb{R}}",
        Sym::Normal => r"\mathcal{N}",
        Sym::Uniform => r"\mathcal{U}",
        Sym::Law => r"\operatorname{Law}",
        Sym::Integers => r"\mathbb{Z}",
        Sym::Complexes => r"\mathbb{C}",
        Sym::Booleans => r"\mathbb{B}",
        Sym::NonNegIntegers => r"\mathbb{N}_{0}",
        Sym::PosIntegers => r"\mathbb{Z}_{>0}",
        Sym::Lebesgue => r"\lambda",
        Sym::Dirac => r"\delta",
        Sym::Simplex => r"\Delta",
        Sym::Ellipsis => r"\ldots",
        Sym::Placeholder => r"\cdot",
        Sym::Euler => "e",
    }
}
