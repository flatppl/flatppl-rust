//! The math AST — target-independent mathematical structure.
//!
//! The lowering (`lower`) builds this tree from the IR; the printers
//! (`mathml`, later `typst`) only choose glyphs. Nothing here carries a target
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
    /// `⊗` inside a superscript (`M^{⊗n}`).
    Otimes,
    /// `×` inside a superscript (`S^{m×n}`).
    Times,
    /// A member-access dot (`r.a`).
    Dot,
    /// A multiplication dot in a row (`1 · 10^{-3}`).
    Cdot,
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
            Math::neg(Math::Num(i.unsigned_abs().to_string()))
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
            return if x < 0.0 { Math::neg(inf) } else { inf };
        }
        if x < 0.0 {
            return Math::neg(Math::real(-x));
        }
        let mag = x.abs();
        if mag != 0.0 && !(1e-4..1e16).contains(&mag) {
            let exp = mag.log10().floor() as i32;
            let mant = mag / 10f64.powi(exp);
            // Shortest decimal of the mantissa (round-trip of x is what matters).
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

    pub fn neg(arg: Math) -> Math {
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

    pub fn add(lhs: Math, rhs: Math) -> Math {
        Math::binary(BinOp::Add, lhs, rhs)
    }

    pub fn sub(lhs: Math, rhs: Math) -> Math {
        Math::binary(BinOp::Sub, lhs, rhs)
    }

    /// Multiplication, choosing juxtaposition unless the right operand starts
    /// with a number (`x · 2`, `2 · 3`) or is negated.
    pub fn mul(lhs: Math, rhs: Math) -> Math {
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

    pub fn sub_(base: Math, sub: Math) -> Math {
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
        match self {
            Math::Binary { op, .. } => {
                let p = self.prec();
                let c = child.prec();
                if c < p {
                    return true;
                }
                if c == p {
                    // Left-associative operators: the right operand of `−`
                    // and `÷`-like operators keeps its parentheses.
                    return slot == Slot::Right && matches!(op, BinOp::Sub);
                }
                false
            }
            Math::Unary { .. } => child.prec() < self.prec(),
            // The base of a power needs brackets unless it reads as one unit
            // (`x_i^2`, `f(x)^2`, `2^3`; but `(−2)^3`, `(2x)^3`).
            Math::Sup(..) if slot == Slot::Base => child.prec() < 9,
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

    /// Every binding name referenced by identifiers in this tree, in order of
    /// first appearance.
    pub fn refs(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_refs(&mut out);
        out
    }

    fn collect_refs(&self, out: &mut Vec<String>) {
        let mut push = |m: &Math| m.collect_refs(out);
        match self {
            Math::Ident(id) => {
                if let Some(t) = &id.target
                    && !out.iter().any(|o| o == t)
                {
                    out.push(t.clone());
                }
            }
            Math::Row(items) | Math::Fenced { items, .. } => items.iter().for_each(&mut push),
            Math::Sub(a, b) | Math::Sup(a, b) | Math::Frac(a, b) => {
                push(a);
                push(b);
            }
            Math::SubSup(a, b, c) => {
                push(a);
                push(b);
                push(c);
            }
            Math::Sqrt(a) | Math::Overline(a) => push(a),
            Math::Apply { head, args } => {
                push(head);
                args.iter().for_each(&mut push);
            }
            Math::Binary { lhs, rhs, .. } | Math::Relation { lhs, rhs, .. } => {
                push(lhs);
                push(rhs);
            }
            Math::Unary { arg, .. } => push(arg),
            Math::BigOp { sub, sup, body, .. } => {
                if let Some(s) = sub {
                    push(s);
                }
                if let Some(s) = sup {
                    push(s);
                }
                push(body);
            }
            Math::Family { body, index, range } => {
                push(body);
                push(index);
                if let Some((lo, hi)) = range {
                    push(lo);
                    push(hi);
                }
            }
            Math::Matrix(rows) => rows.iter().flatten().for_each(&mut push),
            Math::Cases(rows) => {
                for (v, c) in rows {
                    push(v);
                    if let Some(c) = c {
                        push(c);
                    }
                }
            }
            Math::Num(_)
            | Math::Text(_)
            | Math::Str(_)
            | Math::Sym(_)
            | Math::Op(_)
            | Math::Code(_) => {}
        }
    }
}

/// Which operand slot a child occupies, for [`Math::needs_parens`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Left,
    Right,
    /// The base of a power.
    Base,
    /// Any other position (arguments, subscripts, fenced items).
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_picks_juxtaposition_unless_a_number_follows() {
        let x = Math::binding("x");
        assert!(matches!(
            Math::mul(Math::int(2), x.clone()),
            Math::Binary {
                op: BinOp::Juxtapose,
                ..
            }
        ));
        assert!(matches!(
            Math::mul(x.clone(), Math::int(2)),
            Math::Binary { op: BinOp::Dot, .. }
        ));
        assert!(matches!(
            Math::mul(Math::int(3), Math::real(4.5)),
            Math::Binary { op: BinOp::Dot, .. }
        ));
        assert!(matches!(
            Math::mul(x, Math::neg(Math::binding("y"))),
            Math::Binary { op: BinOp::Dot, .. }
        ));
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
        assert_eq!(Math::real(-2.5), Math::neg(Math::Num("2.5".into())));
        assert_eq!(Math::real(f64::INFINITY), Math::Sym(Sym::Infinity));
    }

    #[test]
    fn subtraction_keeps_brackets_on_its_right_operand_only() {
        let a = Math::binding("a");
        let b = Math::binding("b");
        let c = Math::binding("c");
        let inner = Math::sub(b.clone(), c.clone());
        let left = Math::sub(inner.clone(), a.clone());
        let right = Math::sub(a, inner.clone());
        assert!(!left.needs_parens(&inner, Slot::Left));
        assert!(right.needs_parens(&inner, Slot::Right));
        let sum = Math::add(b, c);
        let prod = Math::mul(sum.clone(), Math::binding("d"));
        assert!(prod.needs_parens(&sum, Slot::Left));
    }

    #[test]
    fn refs_collects_binding_targets_once_in_order() {
        let m = Math::add(
            Math::mul(Math::binding("a"), Math::binding("b")),
            Math::apply(
                Math::text("exp"),
                vec![Math::binding("a"), Math::ident("p", None)],
            ),
        );
        assert_eq!(m.refs(), vec!["a", "b"]);
    }
}
