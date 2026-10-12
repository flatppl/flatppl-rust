use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Formula {
    pub response: String,
    pub intercept: bool,
    pub variables: Vec<String>,
    pub fixed: Vec<Term>,
    pub groups: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Term {
    pub factors: Vec<String>,
}

const MAX_DEPTH: usize = 128;
const MAX_TERMS: usize = 4096;

/// Expand R formula operators over column names and `(1 | group)` intercepts.
///
/// Fixed terms follow brms's `terms_fe` reconstruction and second `terms` pass:
/// removed variables disappear, and interactions use the surviving variable order.
/// Group terms use C collation. Expression calls and random slopes are unsupported.
pub fn parse_formula(text: &str) -> Result<Formula> {
    let (response, rhs) = text
        .split_once('~')
        .ok_or_else(|| Error("formula must contain `response ~ terms`".into()))?;
    let response = name(response.trim())?;
    let mut parser = Parser {
        tokens: tokenize(rhs)?,
        at: 0,
        atoms: Vec::new(),
    };
    let expr = parser.expression(0, 0)?;
    if parser.peek() != Token::End {
        return Err(Error("unexpected token in formula".into()));
    }
    let mut intercept = true;
    let terms = expand(&expr, &parser.atoms, &mut intercept, true)?;
    let mut fixed = Vec::new();
    let mut groups = Vec::new();
    for term in terms {
        match &parser.atoms[term[0]] {
            Atom::Group(group, independent) => groups.push((group.clone(), *independent)),
            Atom::Variable(_) => fixed.push(term),
        }
    }
    fixed.sort_by_key(Vec::len);
    let mut order = Vec::new();
    for term in &fixed {
        for &variable in term {
            if !order.contains(&variable) {
                order.push(variable);
            }
        }
    }
    let variable = |id| match &parser.atoms[id] {
        Atom::Variable(name) => name.clone(),
        Atom::Group(..) => unreachable!(),
    };
    let variables = order.iter().map(|&id| variable(id)).collect();
    let fixed = fixed
        .into_iter()
        .map(|mut term| {
            term.sort_by_key(|id| order.iter().position(|v| v == id));
            Term {
                factors: term.into_iter().map(variable).collect(),
            }
        })
        .collect();
    groups.sort();
    if let Some(pair) = groups.windows(2).find(|pair| pair[0].0 == pair[1].0) {
        return Err(Error(format!(
            "duplicated group-level intercept for `{}`",
            pair[0].0
        )));
    }
    Ok(Formula {
        response,
        intercept,
        variables,
        fixed,
        groups: groups.into_iter().map(|(group, _)| group).collect(),
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Add,
    Subtract,
    Interact,
    Cross,
    Nest,
    In,
    Power,
}

impl Op {
    fn precedence(self) -> u8 {
        match self {
            Self::Add | Self::Subtract => 1,
            Self::Cross | Self::Nest => 2,
            Self::In => 3,
            Self::Interact => 4,
            Self::Power => 6,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Token<'a> {
    Name(&'a str),
    Number(f64),
    Operator(Op),
    Left,
    Right,
    Bar(bool),
    End,
}

fn tokenize(text: &str) -> Result<Vec<Token<'_>>> {
    let mut tokens = Vec::new();
    let bytes = text.as_bytes();
    let mut at = 0;
    while at < bytes.len() {
        let start = at;
        let ch = bytes[at];
        at += 1;
        let token = match ch {
            b' ' | b'\t' | b'\r' | b'\n' => continue,
            b'+' => Token::Operator(Op::Add),
            b'-' => Token::Operator(Op::Subtract),
            b':' => Token::Operator(Op::Interact),
            b'*' => Token::Operator(Op::Cross),
            b'/' => Token::Operator(Op::Nest),
            b'^' => Token::Operator(Op::Power),
            b'(' => Token::Left,
            b')' => Token::Right,
            b'|' => {
                let independent = bytes.get(at) == Some(&b'|');
                at += usize::from(independent);
                Token::Bar(independent)
            }
            b'%' if text[start..].starts_with("%in%") => {
                at = start + 4;
                Token::Operator(Op::In)
            }
            b'a'..=b'z' | b'A'..=b'Z' => {
                while bytes
                    .get(at)
                    .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_' || *c == b'.')
                {
                    at += 1;
                }
                Token::Name(&text[start..at])
            }
            b'0'..=b'9' | b'.' => {
                while bytes
                    .get(at)
                    .is_some_and(|c| c.is_ascii_digit() || *c == b'.')
                {
                    at += 1;
                }
                if bytes.get(at).is_some_and(|c| *c == b'e' || *c == b'E') {
                    at += 1;
                    if bytes.get(at).is_some_and(|c| *c == b'+' || *c == b'-') {
                        at += 1;
                    }
                    while bytes.get(at).is_some_and(u8::is_ascii_digit) {
                        at += 1;
                    }
                }
                Token::Number(text[start..at].parse().map_err(|_| {
                    Error(format!("invalid number `{}` in formula", &text[start..at]))
                })?)
            }
            _ => return Err(Error("unsupported character in formula".into())),
        };
        tokens.push(token);
    }
    tokens.push(Token::End);
    Ok(tokens)
}

#[derive(PartialEq)]
enum Atom {
    Variable(String),
    Group(String, bool),
}

struct Expr {
    kind: ExprKind,
    depth: usize,
}

enum ExprKind {
    Number(f64),
    Atom(usize),
    Unary(Op, Box<Expr>),
    Binary(Op, Box<Expr>, Box<Expr>),
    Parentheses(Box<Expr>),
}

impl Expr {
    fn new(kind: ExprKind) -> Result<Self> {
        let depth = 1 + match &kind {
            ExprKind::Unary(_, child) | ExprKind::Parentheses(child) => child.depth,
            ExprKind::Binary(_, a, b) => a.depth.max(b.depth),
            _ => 0,
        };
        if depth > MAX_DEPTH {
            return Err(Error(format!("formula exceeds nesting limit {MAX_DEPTH}")));
        }
        Ok(Self { kind, depth })
    }
}

struct Parser<'a> {
    tokens: Vec<Token<'a>>,
    at: usize,
    atoms: Vec<Atom>,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Token<'a> {
        self.tokens[self.at]
    }

    fn atom(&mut self, atom: Atom) -> Result<Expr> {
        let id = self
            .atoms
            .iter()
            .position(|a| a == &atom)
            .unwrap_or_else(|| {
                self.atoms.push(atom);
                self.atoms.len() - 1
            });
        Expr::new(ExprKind::Atom(id))
    }

    fn expression(&mut self, min_precedence: u8, depth: usize) -> Result<Expr> {
        if depth >= MAX_DEPTH {
            return Err(Error(format!("formula exceeds nesting limit {MAX_DEPTH}")));
        }
        let token = self.peek();
        self.at += 1;
        let mut left = match token {
            Token::Name(text) => {
                if self.peek() == Token::Left {
                    return Err(Error(format!(
                        "formula call `{text}(...)` is not supported"
                    )));
                }
                self.atom(Atom::Variable(name(text)?))?
            }
            Token::Number(value) => Expr::new(ExprKind::Number(value))?,
            Token::Operator(op @ (Op::Add | Op::Subtract)) => {
                let child = self.expression(5, depth + 1)?;
                Expr::new(ExprKind::Unary(op, Box::new(child)))?
            }
            Token::Left => {
                let child = self.expression(0, depth + 1)?;
                let inner = if let Token::Bar(independent) = self.peek() {
                    if !matches!(child.kind, ExprKind::Number(1.0)) {
                        return Err(Error(
                            "only `(1 | group)` random intercepts are supported".into(),
                        ));
                    }
                    self.at += 1;
                    let Token::Name(group) = self.peek() else {
                        return Err(Error("expected a plain grouping column".into()));
                    };
                    let group = name(group)?;
                    self.at += 1;
                    self.atom(Atom::Group(group, independent))?
                } else {
                    Expr::new(ExprKind::Parentheses(Box::new(child)))?
                };
                if self.peek() != Token::Right {
                    return Err(Error("expected `)` in formula".into()));
                }
                self.at += 1;
                inner
            }
            _ => {
                return Err(Error(
                    "expected a column, intercept or parenthesised formula".into(),
                ));
            }
        };
        while let Token::Operator(op) = self.peek() {
            let precedence = op.precedence();
            if precedence < min_precedence {
                break;
            }
            self.at += 1;
            let right = self.expression(precedence + u8::from(op != Op::Power), depth + 1)?;
            left = Expr::new(ExprKind::Binary(op, Box::new(left), Box::new(right)))?;
        }
        Ok(left)
    }
}

type Terms = Vec<Vec<usize>>;

fn expand(expr: &Expr, atoms: &[Atom], intercept: &mut bool, positive: bool) -> Result<Terms> {
    let (op, left, right) = match &expr.kind {
        ExprKind::Number(value) => {
            *intercept = match *value {
                0.0 => !positive,
                1.0 => positive,
                _ => return Err(Error("only 0 and 1 are valid numeric formula terms".into())),
            };
            return Ok(Vec::new());
        }
        ExprKind::Atom(id) => return Ok(vec![vec![*id]]),
        ExprKind::Parentheses(child) => return expand(child, atoms, intercept, positive),
        ExprKind::Unary(op, child) => {
            let terms = expand(child, atoms, intercept, positive == (*op == Op::Add))?;
            return Ok(if *op == Op::Add { terms } else { Vec::new() });
        }
        ExprKind::Binary(op, left, right) => (*op, left, right),
    };
    if op == Op::Power {
        // R requires a numeric literal here; 2.5 truncates to 2, but (2) is a call.
        let ExprKind::Number(power) = right.kind else {
            return Err(Error(
                "invalid power in formula; expected a numeric literal greater than 1".into(),
            ));
        };
        if !(2.0..=i32::MAX as f64).contains(&power) {
            return Err(Error(
                "invalid power in formula; expected a numeric literal greater than 1".into(),
            ));
        }
        let base = expand(left, atoms, intercept, positive)?;
        reject_group_interaction(&base, atoms)?;
        let mut terms = base.clone();
        for _ in 1..power as i32 {
            let next = interact(&base, &terms)?;
            if next == terms {
                break;
            }
            terms = next;
        }
        return Ok(terms);
    }
    let left = expand(left, atoms, intercept, positive)?;
    let right = expand(right, atoms, intercept, positive != (op == Op::Subtract))?;
    if op == Op::Add {
        return unique(left.into_iter().chain(right));
    }
    if op == Op::Subtract {
        let remove: HashSet<_> = right.into_iter().collect();
        return Ok(left
            .into_iter()
            .filter(|term| !remove.contains(term))
            .collect());
    }
    reject_group_interaction(&left, atoms)?;
    reject_group_interaction(&right, atoms)?;
    // R keeps an empty left term set for `*` and `/`; both operands still
    // affect the intercept before that term-set rule is applied.
    if left.is_empty() && matches!(op, Op::Cross | Op::Nest) {
        return Ok(left);
    }
    match op {
        Op::Interact => interact(&left, &right),
        Op::Cross => {
            let interactions = interact(&left, &right)?;
            unique(left.into_iter().chain(right).chain(interactions))
        }
        Op::Nest => {
            let parents = union_all(&left);
            unique(
                left.into_iter()
                    .chain(right.iter().map(|term| union(&parents, term))),
            )
        }
        Op::In => {
            let parents = union_all(&right);
            unique(left.iter().map(|term| union(term, &parents)))
        }
        _ => unreachable!(),
    }
}

fn reject_group_interaction(terms: &Terms, atoms: &[Atom]) -> Result<()> {
    if terms
        .iter()
        .flatten()
        .any(|&id| matches!(atoms[id], Atom::Group(..)))
    {
        return Err(Error(
            "random intercepts may only be added or removed".into(),
        ));
    }
    Ok(())
}

fn unique(terms: impl IntoIterator<Item = Vec<usize>>) -> Result<Terms> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for term in terms {
        if seen.insert(term.clone()) {
            if out.len() == MAX_TERMS {
                return Err(Error(format!(
                    "formula exceeds expanded term limit {MAX_TERMS}"
                )));
            }
            out.push(term);
        }
    }
    Ok(out)
}

fn interact(left: &Terms, right: &Terms) -> Result<Terms> {
    unique(
        left.iter()
            .flat_map(|a| right.iter().map(move |b| union(a, b))),
    )
}

fn union(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut out: Vec<_> = left.iter().chain(right).copied().collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn union_all(terms: &Terms) -> Vec<usize> {
    let mut out: Vec<_> = terms.iter().flatten().copied().collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn name(text: &str) -> Result<String> {
    let valid = text.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.');
    if !valid
        || [
            "Intercept",
            "TRUE",
            "FALSE",
            "NULL",
            "Inf",
            "NaN",
            "NA",
            "NA_integer_",
            "NA_real_",
            "NA_complex_",
            "NA_character_",
        ]
        .contains(&text)
    {
        return Err(Error(format!(
            "unsupported formula term `{text}`; expected a plain column name"
        )));
    }
    Ok(text.into())
}
