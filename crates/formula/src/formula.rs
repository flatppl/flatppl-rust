use crate::{Arithmetic, ColumnExpression, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Formula {
    pub response: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trials: Option<Trials>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub se: Option<StandardError>,
    pub intercept: bool,
    pub variables: Vec<String>,
    pub fixed: Vec<Term>,
    pub groups: Vec<GroupTerm>,
    #[serde(default)]
    pub smooths: Vec<SmoothTerm>,
    pub expressions: BTreeMap<String, ColumnExpression>,
    pub offsets: Vec<ColumnExpression>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Trials {
    Constant(i64),
    Column(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct StandardError {
    pub value: ColumnExpression,
    pub sigma: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct SmoothTerm {
    pub column: String,
    pub rank: Option<usize>,
}

impl SmoothTerm {
    pub fn label(&self) -> String {
        match self.rank {
            Some(rank) => format!("s({}, k = {rank})", self.column),
            None => format!("s({})", self.column),
        }
    }

    pub(crate) fn coefficient(&self) -> String {
        format!("s{}_1", crate::emit::coefficient_name(&self.column))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Term {
    pub factors: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct GroupTerm {
    pub columns: Vec<String>,
    pub intercept: bool,
    pub variables: Vec<String>,
    pub fixed: Vec<Term>,
    pub expressions: BTreeMap<String, ColumnExpression>,
    pub correlated: bool,
}

impl GroupTerm {
    pub fn name(&self) -> String {
        self.columns.join(":")
    }
}

const MAX_DEPTH: usize = 128;
const MAX_TERMS: usize = 4096;

/// Expand R formula operators over column expressions and group effects.
///
/// Fixed terms follow brms's `terms_fe` reconstruction and second `terms` pass:
/// removed variables disappear, and interactions use the surviving variable order.
/// Group terms use C collation and preserve separate group-effect blocks.
pub fn parse_formula(text: &str) -> Result<Formula> {
    let (response, rhs) = text
        .split_once('~')
        .ok_or_else(|| Error("formula must contain `response ~ terms`".into()))?;
    let (response, trials, weights, se) = response_additions(response)?;
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
    let mut smooths = Vec::new();
    // R records offset calls separately from coefficient terms, including
    // calls appearing after subtraction. Negation belongs inside offset().
    let offsets = parser
        .atoms
        .iter()
        .filter_map(|atom| match atom {
            Atom::Offset(expression) => Some(expression.clone()),
            _ => None,
        })
        .collect();
    for term in terms {
        match &parser.atoms[term[0]] {
            Atom::Groups(terms) => groups.extend(terms.iter().cloned()),
            Atom::Variable(_) | Atom::Expression(_) => fixed.push(term),
            Atom::Offset(_) => {}
            Atom::Smooth(smooth) => {
                if smooths
                    .iter()
                    .any(|s: &SmoothTerm| s.column == smooth.column)
                {
                    return Err(Error(format!(
                        "repeated smooths of `{}` are not supported",
                        smooth.column
                    )));
                }
                smooths.push(smooth.clone());
            }
        }
    }
    let (variables, fixed, expressions) = fixed_terms(fixed, &parser.atoms);
    groups.sort_by_key(GroupTerm::name);
    for (i, group) in groups.iter().enumerate() {
        if group.intercept
            && groups[..i]
                .iter()
                .any(|previous| previous.columns == group.columns && previous.intercept)
        {
            return Err(Error(format!(
                "duplicated group effect `Intercept` for `{}`",
                group.name()
            )));
        }
    }
    let formula = Formula {
        response,
        trials,
        weights,
        se,
        intercept,
        variables,
        fixed,
        groups,
        smooths,
        expressions,
        offsets,
    };
    if formula.intercept && formula.uses_intercept_column() {
        return Err(Error(
            "reserved `Intercept` requires removing the population intercept with `0 +` or `- 1`"
                .into(),
        ));
    }
    if formula.uses_intercept_column()
        && !formula
            .expressions
            .values()
            .any(ColumnExpression::uses_intercept)
    {
        return Err(Error(
            "reserved `Intercept` must appear in population-level terms".into(),
        ));
    }
    Ok(formula)
}

type FixedTerms = (Vec<String>, Vec<Term>, BTreeMap<String, ColumnExpression>);

fn fixed_terms(mut fixed: Terms, atoms: &[Atom]) -> FixedTerms {
    fixed.sort_by_key(Vec::len);
    let mut order = Vec::new();
    for term in &fixed {
        for &variable in term {
            if !order.contains(&variable) {
                order.push(variable);
            }
        }
    }
    let variable = |id| match &atoms[id] {
        Atom::Variable(name) => name.clone(),
        Atom::Expression(expression) => expression.to_string(),
        Atom::Groups(_) | Atom::Offset(_) | Atom::Smooth(_) => unreachable!(),
    };
    let variables = order.iter().map(|&id| variable(id)).collect();
    let expressions = order
        .iter()
        .filter_map(|&id| match &atoms[id] {
            Atom::Expression(expression) => Some((expression.to_string(), expression.clone())),
            _ => None,
        })
        .collect();
    let fixed = fixed
        .into_iter()
        .map(|mut term| {
            term.sort_by_key(|id| order.iter().position(|v| v == id));
            Term {
                factors: term.into_iter().map(variable).collect(),
            }
        })
        .collect();
    (variables, fixed, expressions)
}

type ResponseAdditions = (
    String,
    Option<Trials>,
    Option<String>,
    Option<StandardError>,
);

fn response_additions(text: &str) -> Result<ResponseAdditions> {
    let Some((response, addition)) = text.split_once('|') else {
        return Ok((name(text.trim())?, None, None, None));
    };
    let fail = || {
        Error("response additions support `trials(...)`, `weights(column)` and `se(..., sigma=FALSE)`".into())
    };
    let mut trials = None;
    let mut weights = None;
    let mut se = None;
    let mut depth = 0i32;
    for addition in addition.split(|c| {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ => {}
        }
        c == '+' && depth == 0
    }) {
        let (head, argument) = addition.trim().split_once('(').ok_or_else(fail)?;
        let argument = argument.strip_suffix(')').ok_or_else(fail)?.trim();
        match head.trim() {
            "trials" => {
                let value = if let Ok(value) = argument.parse::<i64>() {
                    if value < 0 {
                        return Err(Error("binomial trials must be nonnegative".into()));
                    }
                    Trials::Constant(value)
                } else {
                    Trials::Column(name(argument).map_err(|_| fail())?)
                };
                if trials.as_ref().is_some_and(|previous| previous != &value) {
                    return Err(Error("conflicting trials additions".into()));
                }
                trials = Some(value);
            }
            "weights" => {
                let value = name(argument).map_err(|_| fail())?;
                if weights.as_ref().is_some_and(|previous| previous != &value) {
                    return Err(Error("conflicting weights additions".into()));
                }
                weights = Some(value);
            }
            "se" => {
                let value = standard_error(argument)?;
                if se.as_ref().is_some_and(|old| *old != value) {
                    return Err(Error("conflicting se additions".into()));
                }
                se = Some(value);
            }
            _ => return Err(fail()),
        }
    }
    Ok((name(response.trim())?, trials, weights, se))
}

fn standard_error(text: &str) -> Result<StandardError> {
    let (value, sigma) = text
        .split_once(',')
        .map_or((text, "FALSE"), |(value, flag)| (value, flag.trim()));
    let sigma = match sigma.split_once('=') {
        Some((name, value)) if name.trim() == "sigma" => value.trim(),
        _ => sigma,
    };
    let sigma = match sigma {
        "TRUE" => true,
        "FALSE" => false,
        _ => return Err(Error("se() sigma must be TRUE or FALSE".into())),
    };
    let mut parser = Parser {
        tokens: tokenize(value)?,
        at: 0,
        atoms: Vec::new(),
    };
    let expr = parser.expression(0, 0)?;
    if parser.peek() != Token::End {
        return Err(Error("unexpected token in se() expression".into()));
    }
    Ok(StandardError {
        value: column_expression(&expr, &parser.atoms)?,
        sigma,
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
    Comma,
    Equals,
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
            b',' => Token::Comma,
            b'=' => Token::Equals,
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
    Expression(ColumnExpression),
    Offset(ColumnExpression),
    Groups(Vec<GroupTerm>),
    Smooth(SmoothTerm),
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

    fn smooth(&mut self) -> Result<Expr> {
        let fail = || {
            Error("smooths support `s(column)` and `s(column, k=integer)` with the default thin-plate basis".into())
        };
        self.at += 1;
        let Token::Name(column) = self.peek() else {
            return Err(fail());
        };
        let column = name(column)?;
        self.at += 1;
        let rank = if self.peek() == Token::Comma {
            self.at += 1;
            if self.peek() != Token::Name("k") {
                return Err(fail());
            }
            self.at += 1;
            if self.peek() != Token::Equals {
                return Err(fail());
            }
            self.at += 1;
            let Token::Number(rank) = self.peek() else {
                return Err(fail());
            };
            if !(3.0..=2000.0).contains(&rank) || rank.fract() != 0.0 {
                return Err(fail());
            }
            self.at += 1;
            Some(rank as usize)
        } else {
            None
        };
        if self.peek() != Token::Right {
            return Err(fail());
        }
        self.at += 1;
        self.atom(Atom::Smooth(SmoothTerm { column, rank }))
    }

    fn expression(&mut self, min_precedence: u8, depth: usize) -> Result<Expr> {
        if depth >= MAX_DEPTH {
            return Err(Error(format!("formula exceeds nesting limit {MAX_DEPTH}")));
        }
        let token = self.peek();
        self.at += 1;
        let mut left = match token {
            Token::Name(text) => {
                if self.peek() == Token::Left && text == "s" {
                    self.smooth()?
                } else if self.peek() == Token::Left {
                    if !matches!(text, "I" | "log" | "offset") {
                        return Err(Error(format!(
                            "formula call `{text}(...)` is not supported"
                        )));
                    }
                    self.at += 1;
                    let argument = self.expression(0, depth + 1)?;
                    if self.peek() != Token::Right {
                        return Err(Error(format!(
                            "expected one numeric expression in `{text}`"
                        )));
                    }
                    self.at += 1;
                    let expression = column_expression(&argument, &self.atoms)?;
                    self.atom(match text {
                        "I" => Atom::Expression(ColumnExpression::Identity(Box::new(expression))),
                        "log" => Atom::Expression(ColumnExpression::Log(Box::new(expression))),
                        "offset" => Atom::Offset(expression),
                        _ => unreachable!(),
                    })?
                } else if text == "Intercept" {
                    self.atom(Atom::Expression(ColumnExpression::Intercept))?
                } else {
                    self.atom(Atom::Variable(name(text)?))?
                }
            }
            Token::Number(value) => Expr::new(ExprKind::Number(value))?,
            Token::Operator(op @ (Op::Add | Op::Subtract)) => {
                let child = self.expression(5, depth + 1)?;
                Expr::new(ExprKind::Unary(op, Box::new(child)))?
            }
            Token::Left => {
                let child = self.expression(0, depth + 1)?;
                let inner = if let Token::Bar(independent) = self.peek() {
                    self.at += 1;
                    let grouping = self.expression(0, depth + 1)?;
                    let mut group_order = Vec::new();
                    grouping_columns(&grouping, &self.atoms, &mut group_order)?;
                    group_effect_terms(&child, &self.atoms)?;
                    let mut intercept = true;
                    let fixed = expand(&child, &self.atoms, &mut intercept, true)?;
                    reject_group_interaction(&fixed, &self.atoms)?;
                    let (variables, fixed, expressions) = fixed_terms(fixed, &self.atoms);
                    let mut unused_intercept = false;
                    let columns = expand(&grouping, &self.atoms, &mut unused_intercept, true)?;
                    if columns.is_empty() {
                        return Err(Error("group effects require a grouping column".into()));
                    }
                    let groups = columns
                        .into_iter()
                        .map(|mut term| {
                            term.sort_by_key(|id| group_order.iter().position(|v| v == id));
                            let columns = term
                                .into_iter()
                                .map(|id| match &self.atoms[id] {
                                    Atom::Variable(name) => Ok(name.clone()),
                                    _ => Err(Error(
                                        "grouping factors must be columns or their interactions"
                                            .into(),
                                    )),
                                })
                                .collect::<Result<_>>()?;
                            Ok(GroupTerm {
                                columns,
                                intercept,
                                variables: variables.clone(),
                                fixed: fixed.clone(),
                                expressions: expressions.clone(),
                                correlated: !independent,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    self.atom(Atom::Groups(groups))?
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
    if terms.iter().flatten().any(|&id| {
        matches!(
            atoms[id],
            Atom::Groups(_) | Atom::Offset(_) | Atom::Smooth(_)
        )
    }) {
        return Err(Error(
            "group terms, offsets and smooths may only be added or removed".into(),
        ));
    }
    Ok(())
}

fn grouping_columns(expr: &Expr, atoms: &[Atom], order: &mut Vec<usize>) -> Result<()> {
    match &expr.kind {
        ExprKind::Atom(id) if matches!(atoms[*id], Atom::Variable(_)) => {
            if !order.contains(id) {
                order.push(*id);
            }
            Ok(())
        }
        ExprKind::Parentheses(child) => grouping_columns(child, atoms, order),
        ExprKind::Binary(Op::Interact | Op::Nest, left, right) => {
            grouping_columns(left, atoms, order)?;
            grouping_columns(right, atoms, order)
        }
        _ => Err(Error(
            "grouping factors support columns, `:` and `/`".into(),
        )),
    }
}

fn group_effect_terms(expr: &Expr, atoms: &[Atom]) -> Result<()> {
    match &expr.kind {
        ExprKind::Atom(id)
            if matches!(
                atoms[*id],
                Atom::Groups(_) | Atom::Offset(_) | Atom::Smooth(_)
            ) =>
        {
            Err(Error(
                "group effects cannot contain nested group terms, offsets or smooths".into(),
            ))
        }
        ExprKind::Parentheses(child) | ExprKind::Unary(_, child) => {
            group_effect_terms(child, atoms)
        }
        ExprKind::Binary(_, left, right) => {
            group_effect_terms(left, atoms)?;
            group_effect_terms(right, atoms)
        }
        _ => Ok(()),
    }
}

fn column_expression(expr: &Expr, atoms: &[Atom]) -> Result<ColumnExpression> {
    Ok(match &expr.kind {
        ExprKind::Number(value) if value.is_finite() => ColumnExpression::Number(*value),
        ExprKind::Atom(id) => match &atoms[*id] {
            Atom::Variable(name) => ColumnExpression::Column(name.clone()),
            Atom::Expression(expression) => expression.clone(),
            _ => {
                return Err(Error(
                    "group effects and offsets cannot be nested in column expressions".into(),
                ));
            }
        },
        ExprKind::Parentheses(x) => {
            ColumnExpression::Parentheses(Box::new(column_expression(x, atoms)?))
        }
        ExprKind::Unary(Op::Add, x) => {
            ColumnExpression::Positive(Box::new(column_expression(x, atoms)?))
        }
        ExprKind::Unary(Op::Subtract, x) => {
            ColumnExpression::Negate(Box::new(column_expression(x, atoms)?))
        }
        ExprKind::Binary(op, x, y) => {
            let op = match op {
                Op::Add => Arithmetic::Add,
                Op::Subtract => Arithmetic::Subtract,
                Op::Cross => Arithmetic::Multiply,
                Op::Nest => Arithmetic::Divide,
                Op::Power => Arithmetic::Power,
                _ => {
                    return Err(Error(
                        "unsupported operator in numeric column expression".into(),
                    ));
                }
            };
            ColumnExpression::Binary(
                op,
                Box::new(column_expression(x, atoms)?),
                Box::new(column_expression(y, atoms)?),
            )
        }
        _ => return Err(Error("unsupported numeric column expression".into())),
    })
}

impl Formula {
    pub(crate) fn uses_intercept_column(&self) -> bool {
        self.expressions
            .values()
            .chain(&self.offsets)
            .chain(self.se.as_ref().map(|se| &se.value))
            .chain(
                self.groups
                    .iter()
                    .flat_map(|group| group.expressions.values()),
            )
            .any(ColumnExpression::uses_intercept)
    }

    pub(crate) fn predictor_columns(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for (variables, expressions) in std::iter::once((&self.variables, &self.expressions)).chain(
            self.groups
                .iter()
                .map(|group| (&group.variables, &group.expressions)),
        ) {
            for name in variables {
                if let Some(expression) = expressions.get(name) {
                    expression.columns(&mut names);
                } else if !names.contains(&name.as_str()) {
                    names.push(name);
                }
            }
        }
        for expression in &self.offsets {
            expression.columns(&mut names);
        }
        for smooth in &self.smooths {
            if !names.contains(&smooth.column.as_str()) {
                names.push(&smooth.column);
            }
        }
        names
    }

    pub(crate) fn offset(&self) -> Option<ColumnExpression> {
        self.offsets
            .iter()
            .cloned()
            .reduce(|a, b| ColumnExpression::Binary(Arithmetic::Add, Box::new(a), Box::new(b)))
    }
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
