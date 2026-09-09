//! Shared display decisions, without changing the semantic expression tree.

use crate::ast::{BinOp, Math, Slot};
use crate::names::Atom;

pub(crate) const LINE_WIDTH: usize = 72;

/// Approximate printed glyph count, not a font or viewport measurement.
pub(crate) fn width(m: &Math) -> usize {
    let atom = |a: &Atom| match a {
        Atom::Greek(_) | Atom::Letter(_) | Atom::Script(_) => 1,
        Atom::Digits(s) | Atom::Word(s) => s.chars().count(),
    };
    match m {
        Math::Ident(id) => {
            atom(&id.display.head) + id.display.subs.iter().map(atom).sum::<usize>().div_ceil(2)
        }
        Math::Num(s) | Math::Text(s) | Math::Str(s) | Math::Code(s) => s.chars().count(),
        Math::Frac(a, b) => width(a).max(width(b)) + 2,
        Math::Sub(a, b) | Math::Sup(a, b) => width(a) + width(b).div_ceil(2),
        Math::SubSup(a, b, c) => width(a) + width(b).max(width(c)).div_ceil(2),
        _ => m.children().iter().map(|child| width(child)).sum::<usize>() + 2,
    }
}

pub(crate) struct SumLine<'a> {
    pub(crate) sign: Option<BinOp>,
    pub(crate) term: &'a Math,
    pub(crate) parens: bool,
}

/// Split only the left-associated chain. Right-hand groups stay whole, so
/// `a - (b - c)` never becomes `a - b - c`. The enclosing AST keeps its fences.
pub(crate) fn sum_lines(m: &Math) -> Vec<SumLine<'_>> {
    if !matches!(
        m,
        Math::Binary {
            op: BinOp::Add | BinOp::Sub,
            ..
        }
    ) || width(m) <= LINE_WIDTH
    {
        return Vec::new();
    }
    let mut lines = Vec::new();
    let mut next = m;
    while let Math::Binary {
        op: op @ (BinOp::Add | BinOp::Sub),
        lhs,
        rhs,
    } = next
    {
        lines.push(SumLine {
            sign: Some(*op),
            term: rhs,
            parens: next.needs_parens(rhs, Slot::Right),
        });
        next = lhs;
    }
    lines.push(SumLine {
        sign: None,
        term: next,
        parens: next.prec() <= 3,
    });
    if lines.len() < 3 {
        return Vec::new();
    }
    lines.reverse();
    lines
}
