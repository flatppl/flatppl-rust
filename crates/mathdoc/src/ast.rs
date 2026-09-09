//! The math AST — target-independent mathematical structure.
//!
//! The lowering (`lower`) builds this tree from the IR; the printers
//! (`mathml`, `tex`, `typst`) only choose glyphs. Nothing here carries a target
//! escape or a target-specific layout decision. Parenthesisation is a property
//! of the tree ([`Math::prec`] + the slot rules in [`Math::needs_parens`]) so
//! every printer brackets identically.

use crate::names::{DisplayName, display_name};

/// An identifier leaf: the source name, its rendering, and the binding it
/// refers to (for the viewer's `data-flatppl-ref` back-reference).
#[derive(Clone, Debug, PartialEq)]
pub struct Ident {
    /// The name as spelled in the source (binding, parameter or field name).
    pub name: String,
    pub display: DisplayName,
    /// The module binding this identifier denotes, when it is one. `None` for
    /// lambda parameters, placeholders, indices and record fields.
    pub target: Option<String>,
}

/// A mathematical expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Math {
    Ident(Ident),
    /// A non-negative number, already formatted.
    Num(String),
    /// Upright text: operator names (`Law`, `normalize`), words.
    Text(String),
    /// A quoted string literal.
    Str(String),
    Sym(Sym),
    /// A bare operator glyph inside a [`Math::Row`] (`|` in `p(x | θ)`).
    Op(Op),
    /// Juxtaposition.
    Row(Vec<Math>),
    Sub(Box<Math>, Box<Math>),
    Sup(Box<Math>, Box<Math>),
    SubSup(Box<Math>, Box<Math>, Box<Math>),
    Frac(Box<Math>, Box<Math>),
    Sqrt(Box<Math>),
    /// A delimited, comma-separated list: `(a, b)`, `[x]`, `{…}`, `|x|`, `‖v‖`.
    Fenced {
        open: Fence,
        close: Fence,
        items: Vec<Math>,
    },
    /// Function application `head(args)`.
    Apply {
        head: Box<Math>,
        args: Vec<Math>,
    },
    Binary {
        op: BinOp,
        lhs: Box<Math>,
        rhs: Box<Math>,
    },
    Unary {
        op: UnOp,
        arg: Box<Math>,
    },
    Relation {
        lhs: Box<Math>,
        rel: Rel,
        rhs: Box<Math>,
    },
    /// `Σ_{sub}^{sup} body` and friends.
    BigOp {
        op: BigOp,
        sub: Option<Box<Math>>,
        sup: Option<Box<Math>>,
        body: Box<Math>,
    },
    /// An indexed family `(body)_{index = lo}^{hi}`; `range` absent gives
    /// `(body)_{index}`.
    Family {
        body: Box<Math>,
        index: Box<Math>,
        range: Option<(Box<Math>, Box<Math>)>,
    },
    /// A bracketed matrix.
    Matrix(Vec<Vec<Math>>),
    /// A case distinction: `(value, condition)` rows, `None` = otherwise.
    Cases(Vec<(Math, Option<Math>)>),
    /// An overline (complex conjugate).
    Overline(Box<Math>),
    /// Fallback: source text in monospace.
    Code(String),
}

/// Fixed symbols.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sym {
    Infinity,
    Pi,
    /// The imaginary unit, upright.
    ImagUnit,
    Reals,
    /// Extended reals, the predefined FlatPPL `reals` set (§03).
    ExtendedReals,
    /// Distribution and law symbols, distinct from author-named variables.
    Normal,
    Uniform,
    Law,
    Integers,
    Complexes,
    Booleans,
    /// ℕ₀.
    NonNegIntegers,
    /// ℤ_{>0}.
    PosIntegers,
    /// Lebesgue measure λ.
    Lebesgue,
    /// Dirac δ.
    Dirac,
    /// Δ for the standard simplex.
    Simplex,
    /// `…`
    Ellipsis,
    /// `·` as a placeholder argument (`A_{·,j}`, `p_K(data | ·)`).
    Placeholder,
    /// Euler's e.
    Euler,
}

/// Operator glyphs that appear bare in a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// Conditioning bar.
    Bar,
    /// Lower-star for the pushforward `f_*`.
    Star,
    /// `×` inside a superscript (`S^{m×n}`, `M^{m×n}`).
    Times,
    /// A member-access dot (`r.a`).
    Dot,
    /// A multiplication dot in a row (`1 · 10^{-3}`).
    Cdot,
    /// A separating comma inside a script (`A_{i,j}`).
    Comma,
    /// Restriction bar before a subscript (`M|_S`).
    Restrict,
    /// Differential `d` in `M(dx)`.
    Differential,
    /// Transpose `ᵀ`, as a superscript body.
    Transpose,
    /// Adjoint `†`.
    Dagger,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fence {
    Paren,
    Bracket,
    Brace,
    Bar,
    Norm,
    Floor,
    Ceil,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    /// Multiplication by juxtaposition (`2x`, `αβ`).
    Juxtapose,
    /// Multiplication with an explicit `·`.
    Dot,
    /// Cartesian / cross product `×`.
    Times,
    /// Product measure `⊗`.
    Otimes,
    /// Function composition `∘`.
    Compose,
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rel {
    Eq,
    Ne,
    /// `∼` (distributed as).
    Sim,
    In,
    Lt,
    Le,
    Gt,
    Ge,
    /// `↦`.
    MapsTo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BigOp {
    Sum,
    Prod,
    /// `⨂`.
    Otimes,
    Max,
    Min,
    /// A named reduction (`var`, `mean`, …) written as an operator with
    /// the reduced index below it.
    Named(&'static str),
}

/// One rendered row: `lhs rel rhs`.
#[derive(Clone, Debug, PartialEq)]
pub struct Statement {
    pub lhs: Math,
    pub rel: Rel,
    pub rhs: Math,
}

impl Statement {
    /// Every binding the row refers to, left-hand side first, each once
    /// (including the row's own names, when the left-hand side carries them).
    pub fn refs(&self) -> Vec<String> {
        let mut out = self.lhs.refs();
        for r in self.rhs.refs() {
            if !out.contains(&r) {
                out.push(r);
            }
        }
        out
    }

    /// The deeper of the two sides ([`Math::depth`]).
    pub fn depth(&self) -> usize {
        self.lhs.depth().max(self.rhs.depth())
    }
}

impl Math {
    // ── constructors ───────────────────────────────────────────────────────

    /// An identifier by the name rules, optionally pointing at a binding.
    pub fn ident(name: &str, target: Option<&str>) -> Math {
        Math::Ident(Ident {
            name: name.to_string(),
            display: display_name(name),
            target: target.map(str::to_string),
        })
    }

    /// An identifier that refers to the binding of the same name.
    pub fn binding(name: &str) -> Math {
        Math::ident(name, Some(name))
    }

    /// A single italic letter with no binding (the `p` of `p_M(x)`).
    pub fn letter(c: char) -> Math {
        Math::Ident(Ident {
            name: c.to_string(),
            display: DisplayName {
                head: crate::names::Atom::Letter(c),
                subs: Vec::new(),
            },
            target: None,
        })
    }

    pub fn text(s: impl Into<String>) -> Math {
        Math::Text(s.into())
    }

    pub fn int(i: i64) -> Math {
        if i < 0 {
            Math::negate(Math::Num(i.unsigned_abs().to_string()))
        } else {
            Math::Num(i.to_string())
        }
    }

    /// A real literal. Very small or very large magnitudes print as
    /// `m · 10^e`; everything else uses the shortest round-trip decimal.
    pub fn real(x: f64) -> Math {
        if x.is_nan() {
            return Math::text("NaN");
        }
        if x.is_infinite() {
            let inf = Math::Sym(Sym::Infinity);
            return if x < 0.0 { Math::negate(inf) } else { inf };
        }
        if x < 0.0 {
            return Math::negate(Math::real(-x));
        }
        let mag = x.abs();
        if mag != 0.0 && !(1e-4..1e16).contains(&mag) {
            let mut exp = mag.log10().floor() as i32;
            let mut mant = mag / 10f64.powi(exp);
            // `log10` of an exact power of ten can land a hair under the
            // integer (`1e-5` → `−5.000…01`): renormalise into `[1, 10)`.
            if mant >= 10.0 {
                mant /= 10.0;
                exp += 1;
            } else if mant < 1.0 {
                mant *= 10.0;
                exp -= 1;
            }
            // Twelve significant digits, trailing zeros dropped.
            let mant = format!("{}", (mant * 1e12).round() / 1e12);
            return Math::Row(vec![
                Math::Num(mant),
                Math::Op(Op::Cdot),
                Math::Sup(
                    Box::new(Math::Num("10".into())),
                    Box::new(Math::int(exp as i64)),
                ),
            ]);
        }
        Math::Num(format!("{x}"))
    }

    pub fn negate(arg: Math) -> Math {
        Math::Unary {
            op: UnOp::Neg,
            arg: Box::new(arg),
        }
    }

    pub fn binary(op: BinOp, lhs: Math, rhs: Math) -> Math {
        Math::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn plus(lhs: Math, rhs: Math) -> Math {
        Math::binary(BinOp::Add, lhs, rhs)
    }

    pub fn minus(lhs: Math, rhs: Math) -> Math {
        Math::binary(BinOp::Sub, lhs, rhs)
    }

    /// Ordinary arithmetic multiplication: numeric coefficients read first,
    /// and multiplication by a unit reciprocal reads as a fraction.
    /// Symbolic factors retain their order; neither rule evaluates constants.
    /// Measure weighting uses [`Math::dot`] instead.
    pub fn times(lhs: Math, rhs: Math) -> Math {
        let number = |m: &Math| {
            matches!(m, Math::Num(_))
                || matches!(m, Math::Unary { op: UnOp::Neg, arg } if matches!(**arg, Math::Num(_)))
        };
        if number(&rhs) && !number(&lhs) {
            return Math::times(rhs, lhs);
        }
        let rhs = match rhs {
            // FlatPPL's ordinary division has a scalar denominator (§07).
            // Keep that denominator intact: no cancellation or reassociation
            // inside it, even when it contains the same symbol as the numerator.
            Math::Frac(num, den) if matches!(&*num, Math::Num(n) if n == "1") => {
                return Math::frac(lhs, *den);
            }
            other => other,
        };
        let explicit = rhs.starts_with_number()
            || matches!(rhs, Math::Unary { .. })
            || (lhs.ends_with_number() && rhs.starts_with_number());
        let op = if explicit {
            BinOp::Dot
        } else {
            BinOp::Juxtapose
        };
        Math::binary(op, lhs, rhs)
    }

    /// Multiplication with an explicit `·` (measures: `w · M`).
    pub fn dot(lhs: Math, rhs: Math) -> Math {
        Math::binary(BinOp::Dot, lhs, rhs)
    }

    pub fn frac(num: Math, den: Math) -> Math {
        Math::Frac(Box::new(num), Box::new(den))
    }

    pub fn pow(base: Math, exp: Math) -> Math {
        Math::Sup(Box::new(base), Box::new(exp))
    }

    pub fn subscript(base: Math, sub: Math) -> Math {
        Math::Sub(Box::new(base), Box::new(sub))
    }

    pub fn sqrt(arg: Math) -> Math {
        Math::Sqrt(Box::new(arg))
    }

    pub fn paren(items: Vec<Math>) -> Math {
        Math::Fenced {
            open: Fence::Paren,
            close: Fence::Paren,
            items,
        }
    }

    pub fn bracket(items: Vec<Math>) -> Math {
        Math::Fenced {
            open: Fence::Bracket,
            close: Fence::Bracket,
            items,
        }
    }

    pub fn brace(items: Vec<Math>) -> Math {
        Math::Fenced {
            open: Fence::Brace,
            close: Fence::Brace,
            items,
        }
    }

    pub fn abs(arg: Math) -> Math {
        Math::Fenced {
            open: Fence::Bar,
            close: Fence::Bar,
            items: vec![arg],
        }
    }

    pub fn apply(head: Math, args: Vec<Math>) -> Math {
        Math::Apply {
            head: Box::new(head),
            args,
        }
    }

    /// `name(args)` with a roman operator name.
    pub fn call(name: &str, args: Vec<Math>) -> Math {
        Math::apply(Math::text(name), args)
    }

    pub fn relation(lhs: Math, rel: Rel, rhs: Math) -> Math {
        Math::Relation {
            lhs: Box::new(lhs),
            rel,
            rhs: Box::new(rhs),
        }
    }

    pub fn big(op: BigOp, sub: Option<Math>, sup: Option<Math>, body: Math) -> Math {
        Math::BigOp {
            op,
            sub: sub.map(Box::new),
            sup: sup.map(Box::new),
            body: Box::new(body),
        }
    }

    pub fn family(body: Math, index: Math, range: Option<(Math, Math)>) -> Math {
        Math::Family {
            body: Box::new(body),
            index: Box::new(index),
            range: range.map(|(lo, hi)| (Box::new(lo), Box::new(hi))),
        }
    }

    /// Flatten nested rows; a single item is returned as itself.
    pub fn row(items: Vec<Math>) -> Math {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            match item {
                Math::Row(inner) => out.extend(inner),
                other => out.push(other),
            }
        }
        if out.len() == 1 {
            out.pop().unwrap()
        } else {
            Math::Row(out)
        }
    }

    // ── structure queries ─────────────────────────────────────────────────

    /// Binding precedence: higher binds tighter. Relations are loosest (0),
    /// logical connectives 1–2, additive 3, multiplicative 4, prefix 5, and
    /// anything that reads as a single unit 9.
    pub fn prec(&self) -> u8 {
        match self {
            Math::Relation { .. } => 0,
            Math::Binary { op: BinOp::Or, .. } => 1,
            Math::Binary { op: BinOp::And, .. } => 2,
            Math::Binary {
                op: BinOp::Add | BinOp::Sub,
                ..
            } => 3,
            Math::Binary { .. } => 4,
            Math::Unary { .. } => 5,
            // A big operator swallows everything to its right.
            Math::BigOp { .. } => 3,
            Math::Cases(_) => 9,
            _ => 9,
        }
    }

    /// Whether `child` in the given slot of `self` must be bracketed.
    pub fn needs_parens(&self, child: &Math, slot: Slot) -> bool {
        // A big operator swallows everything to its right: it is bracketed
        // wherever an operator or a script follows it (`(∑ᵢ xᵢ) + 1`,
        // `(∑ᵢ xᵢ) = 1`, `(∑ᵢ xᵢ)²`) and never where it ends the operand
        // (`1 + ∑ᵢ xᵢ`, `−∑ᵢ xᵢ`, `∑ᵢ ∑ⱼ xᵢⱼ`). A comma or a fence delimits
        // it by itself (`Normal(∑ᵢ xᵢ, 1)`).
        if matches!(child, Math::BigOp { .. }) {
            return matches!(slot, Slot::Left | Slot::Base);
        }
        match self {
            Math::Binary { op, .. } => {
                // A negative coefficient inside a product still needs a
                // fence on the right: `a(-3b)`, never `a - 3b`.
                if slot == Slot::Right
                    && matches!(op, BinOp::Juxtapose | BinOp::Dot)
                    && child.starts_with_minus()
                {
                    return true;
                }
                let p = self.prec();
                let c = child.prec();
                if c < p {
                    return true;
                }
                if c == p {
                    // Left-associative: the right operand of `−` keeps its
                    // parentheses (`a − (b − c)`).
                    return slot == Slot::Right && matches!(op, BinOp::Sub);
                }
                false
            }
            Math::Unary { .. } => child.prec() < self.prec(),
            // The body of a big operator: an additive or logical body is
            // bracketed (`∏ⱼ (Aᵢⱼ + Bⱼₖ)`), a product is not (`∑ⱼ Aᵢⱼ Bⱼₖ`).
            Math::BigOp { .. } => child.prec() <= 3,
            // A second superscript must not look like an exponent tower.
            Math::Sup(..) | Math::SubSup(..)
                if slot == Slot::Base && matches!(child, Math::Sup(..) | Math::SubSup(..)) =>
            {
                true
            }
            // The base of a script needs brackets unless it reads as one unit
            // (`x_i^2`, `f(x)^2`, `2^3`; but `(−2)^3`, `(2x)^3`, `(a + b)_2`).
            Math::Sub(..) | Math::Sup(..) | Math::SubSup(..) if slot == Slot::Base => {
                child.prec() < 9
            }
            _ => false,
        }
    }

    fn starts_with_minus(&self) -> bool {
        match self {
            Math::Unary { op: UnOp::Neg, .. } => true,
            Math::Binary {
                op: BinOp::Juxtapose | BinOp::Dot,
                lhs,
                ..
            } => lhs.starts_with_minus(),
            _ => false,
        }
    }

    fn starts_with_number(&self) -> bool {
        match self {
            Math::Num(_) => true,
            Math::Row(items) => items.first().is_some_and(Math::starts_with_number),
            Math::Binary { lhs, .. } => lhs.starts_with_number(),
            Math::Sub(b, _) | Math::Sup(b, _) | Math::SubSup(b, _, _) => b.starts_with_number(),
            Math::Frac(..) => true,
            _ => false,
        }
    }

    fn ends_with_number(&self) -> bool {
        match self {
            Math::Num(_) => true,
            Math::Row(items) => items.last().is_some_and(Math::ends_with_number),
            Math::Binary { rhs, .. } => rhs.ends_with_number(),
            Math::Frac(..) => true,
            _ => false,
        }
    }

    /// The direct sub-expressions, in reading order.
    pub fn children(&self) -> Vec<&Math> {
        match self {
            Math::Row(items) | Math::Fenced { items, .. } => items.iter().collect(),
            Math::Sub(a, b) | Math::Sup(a, b) | Math::Frac(a, b) => vec![a, b],
            Math::SubSup(a, b, c) => vec![a, b, c],
            Math::Sqrt(a) | Math::Overline(a) | Math::Unary { arg: a, .. } => vec![a],
            Math::Apply { head, args } => {
                std::iter::once(head.as_ref()).chain(args.iter()).collect()
            }
            Math::Binary { lhs, rhs, .. } | Math::Relation { lhs, rhs, .. } => vec![lhs, rhs],
            Math::BigOp { sub, sup, body, .. } => sub
                .iter()
                .chain(sup.iter())
                .map(Box::as_ref)
                .chain(std::iter::once(body.as_ref()))
                .collect(),
            Math::Family { body, index, range } => {
                let mut out = vec![body.as_ref(), index.as_ref()];
                if let Some((lo, hi)) = range {
                    out.push(lo);
                    out.push(hi);
                }
                out
            }
            Math::Matrix(rows) => rows.iter().flatten().collect(),
            Math::Cases(rows) => rows
                .iter()
                .flat_map(|(v, c)| std::iter::once(v).chain(c.iter()))
                .collect(),
            Math::Ident(_)
            | Math::Num(_)
            | Math::Text(_)
            | Math::Str(_)
            | Math::Sym(_)
            | Math::Op(_)
            | Math::Code(_) => Vec::new(),
        }
    }

    /// Every binding name referenced by identifiers in this tree, in order of
    /// first appearance.
    pub fn refs(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        // Depth-first in reading order: children are pushed reversed.
        let mut stack = vec![self];
        while let Some(m) = stack.pop() {
            if let Math::Ident(id) = m
                && let Some(t) = &id.target
                && !out.iter().any(|o| o == t)
            {
                out.push(t.clone());
            }
            stack.extend(children_reversed(m));
        }
        out
    }

    /// The nesting depth of the tree (a leaf is 1), computed without
    /// recursion so it is safe on any input.
    pub fn depth(&self) -> usize {
        let mut max = 0;
        let mut stack = vec![(self, 1usize)];
        while let Some((m, d)) = stack.pop() {
            max = max.max(d);
            stack.extend(children_reversed(m).into_iter().map(|c| (c, d + 1)));
        }
        max
    }
}

fn children_reversed(m: &Math) -> Vec<&Math> {
    let mut children = m.children();
    children.reverse();
    children
}

/// Which operand slot a child occupies, for [`Math::needs_parens`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Left,
    Right,
    /// The base of a script.
    Base,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplication_keeps_numeric_products_explicit_and_symbolic_order() {
        let x = Math::binding("x");
        assert_eq!(
            crate::tex::expr(&Math::times(Math::int(2), x.clone())),
            "2  x"
        );
        assert_eq!(
            crate::tex::expr(&Math::times(x.clone(), Math::int(2))),
            "2  x"
        );
        assert_eq!(
            crate::tex::expr(&Math::times(Math::int(3), Math::real(4.5))),
            r"3 \cdot 4.5"
        );
        assert_eq!(
            crate::tex::expr(&Math::times(x, Math::negate(Math::binding("y")))),
            r"x \cdot \left(- y\right)"
        );
    }

    #[test]
    fn reals_print_shortest_and_switch_to_powers_of_ten_at_the_extremes() {
        assert_eq!(Math::real(5.0), Math::Num("5".into()));
        assert_eq!(Math::real(0.1), Math::Num("0.1".into()));
        assert_eq!(Math::real(1e-10), Math::real(1e-10));
        match Math::real(1e-10) {
            Math::Row(items) => assert!(matches!(items[0], Math::Num(ref m) if m == "1")),
            other => panic!("{other:?}"),
        }
        assert_eq!(Math::real(-2.5), Math::negate(Math::Num("2.5".into())));
        assert_eq!(Math::real(f64::INFINITY), Math::Sym(Sym::Infinity));
    }

    #[test]
    fn subtraction_keeps_brackets_on_its_right_operand_only() {
        let a = Math::binding("a");
        let b = Math::binding("b");
        let c = Math::binding("c");
        let inner = Math::minus(b.clone(), c.clone());
        let left = Math::minus(inner.clone(), a.clone());
        let right = Math::minus(a, inner.clone());
        assert!(!left.needs_parens(&inner, Slot::Left));
        assert!(right.needs_parens(&inner, Slot::Right));
        let sum = Math::plus(b, c);
        let prod = Math::times(sum.clone(), Math::binding("d"));
        assert!(prod.needs_parens(&sum, Slot::Left));
    }

    #[test]
    fn refs_collects_binding_targets_once_in_order() {
        let m = Math::plus(
            Math::times(Math::binding("a"), Math::binding("b")),
            Math::apply(
                Math::text("exp"),
                vec![Math::binding("a"), Math::ident("p", None)],
            ),
        );
        assert_eq!(m.refs(), vec!["a", "b"]);
    }
}
