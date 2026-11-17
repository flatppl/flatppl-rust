//! Matching `distributions.ron` patterns (`Bernoulli(invlogit(a))`,
//! `locscale(StudentT(nu), mu, sigma)`) against FlatPPL law nodes, and filling
//! the Stan argument templates.

use super::{Emitter, Ex, P_ATOM};
use crate::refuse::Result;
use flatppl_core::{Node, NodeId, Scalar};
use std::collections::HashMap;

/// A parsed table pattern. Operators become their FlatPPL call names.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Pat {
    Var(String),
    Num(f64),
    Call(String, Vec<Pat>),
}

/// What a pattern variable is bound to.
#[derive(Clone, Debug)]
pub(crate) enum Bound {
    Node(NodeId),
    /// A literal the pattern implies (`unitinterval` as `interval(0, 1)`).
    Lit(&'static str),
    /// An expression already printed.
    Expr(Ex),
}

pub(crate) fn parse(s: &str) -> Pat {
    let toks = tokenize(s);
    let mut p = Parser { toks, i: 0 };
    let pat = p.sum();
    assert_eq!(p.i, p.toks.len(), "pattern `{s}`: trailing input");
    pat
}

fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if c.is_alphanumeric() || c == '_' || c == '.' {
            cur.push(c);
        } else {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            if !c.is_whitespace() {
                out.push(c.to_string());
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

struct Parser {
    toks: Vec<String>,
    i: usize,
}

impl Parser {
    fn peek(&self) -> Option<&str> {
        self.toks.get(self.i).map(String::as_str)
    }
    fn next(&mut self) -> String {
        self.i += 1;
        self.toks[self.i - 1].clone()
    }
    fn sum(&mut self) -> Pat {
        let mut l = self.product();
        while let Some(op @ ("+" | "-")) = self.peek() {
            let name = if op == "+" { "add" } else { "sub" };
            self.next();
            let r = self.product();
            l = Pat::Call(name.into(), vec![l, r]);
        }
        l
    }
    fn product(&mut self) -> Pat {
        let mut l = self.atom();
        while let Some(op @ ("*" | "/")) = self.peek() {
            let name = if op == "*" { "mul" } else { "divide" };
            self.next();
            let r = self.atom();
            l = Pat::Call(name.into(), vec![l, r]);
        }
        l
    }
    fn atom(&mut self) -> Pat {
        let t = self.next();
        if t == "(" {
            let p = self.sum();
            assert_eq!(self.next(), ")");
            return p;
        }
        if let Ok(x) = t.parse::<f64>() {
            return Pat::Num(x);
        }
        if self.peek() == Some("(") {
            self.next();
            let mut args = Vec::new();
            if self.peek() != Some(")") {
                loop {
                    args.push(self.sum());
                    if self.peek() == Some(",") {
                        self.next();
                    } else {
                        break;
                    }
                }
            }
            assert_eq!(self.next(), ")");
            return Pat::Call(t, args);
        }
        Pat::Var(t)
    }
}

impl<'m> Emitter<'m> {
    /// Match the arguments of a law call `name(args…)` against a pattern whose
    /// head is `name`.
    pub(super) fn match_call(
        &self,
        pat: &Pat,
        args: &[NodeId],
        binds: &mut HashMap<String, Bound>,
    ) -> bool {
        let Pat::Call(_, pargs) = pat else {
            return false;
        };
        pargs.len() == args.len()
            && pargs
                .iter()
                .zip(args)
                .all(|(p, &a)| self.match_pat(p, a, binds))
    }

    fn match_pat(&self, pat: &Pat, node: NodeId, binds: &mut HashMap<String, Bound>) -> bool {
        match pat {
            Pat::Var(v) => match binds.get(v) {
                Some(Bound::Node(b)) => self.m.structural_eq(*b, node),
                Some(Bound::Lit(_) | Bound::Expr(_)) => false,
                None => {
                    binds.insert(v.clone(), Bound::Node(node));
                    true
                }
            },
            Pat::Num(x) => match self.m.node(self.resolve(node)) {
                Node::Lit(Scalar::Int(n)) => *n as f64 == *x,
                Node::Lit(Scalar::Real(r)) => r == x,
                _ => false,
            },
            Pat::Call(name, pargs) => {
                let r = self.resolve(node);
                // A set constant in its interval form.
                if name == "interval"
                    && let Node::Const(c) = self.m.node(r)
                    && self.name(*c) == "unitinterval"
                    && let [Pat::Var(a), Pat::Var(b)] = pargs.as_slice()
                    && !binds.contains_key(a)
                    && !binds.contains_key(b)
                {
                    binds.insert(a.clone(), Bound::Lit("0"));
                    binds.insert(b.clone(), Bound::Lit("1"));
                    return true;
                }
                let Some(c) = self.call(r) else {
                    return false;
                };
                let args: Vec<NodeId> = match self.head(r) {
                    Some(h) if h == name => {
                        if name.starts_with(char::is_uppercase) {
                            match self.law_args(name, r) {
                                Ok(a) => a,
                                Err(_) => return false,
                            }
                        } else if c.named.is_empty() {
                            c.args.to_vec()
                        } else {
                            return false;
                        }
                    }
                    // An elementwise call matches its scalar pattern.
                    Some("broadcast") => match self.m.node(c.args[0]) {
                        Node::Const(f) if self.name(*f) == name && c.named.is_empty() => {
                            c.args[1..].to_vec()
                        }
                        _ => return false,
                    },
                    _ => return false,
                };
                pargs.len() == args.len()
                    && pargs
                        .iter()
                        .zip(&args)
                        .all(|(p, &a)| self.match_pat(p, a, binds))
            }
        }
    }

    /// Fill a template. A placeholder that is a whole call argument (or the
    /// whole template) prints as is; one inside a larger expression prints
    /// atomically. `index` subscripts the variables `per_element` marks.
    pub(super) fn fill(
        &mut self,
        template: &str,
        binds: &HashMap<String, Bound>,
        per_element: &dyn Fn(&str) -> bool,
        index: Option<&str>,
    ) -> Result<String> {
        let mut out = String::new();
        let mut rest = template;
        while let Some(open) = rest.find('{') {
            let before = &rest[..open];
            let close = rest[open..].find('}').expect("closed placeholder") + open;
            let var = &rest[open + 1..close];
            let after = &rest[close + 1..];
            out.push_str(before);
            let prev = out.trim_end().chars().last();
            let next = after.trim_start().chars().next();
            let whole =
                matches!(prev, None | Some('(' | ',')) && matches!(next, None | Some(')' | ','));
            let e = match binds.get(var) {
                Some(Bound::Node(n)) => self.expr(*n)?,
                Some(Bound::Lit(l)) => Ex::atom(*l),
                Some(Bound::Expr(e)) => e.clone(),
                None => panic!("template variable `{var}` unbound"),
            };
            let text = match index.filter(|_| per_element(var)) {
                Some(i) => format!("{}[{i}]", e.at(P_ATOM)),
                None if whole => e.s,
                None => e.at(P_ATOM),
            };
            out.push_str(&text);
            rest = after;
        }
        out.push_str(rest);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_table_patterns() {
        assert_eq!(
            parse("Beta(m * kappa, (1 - m) * kappa)"),
            Pat::Call(
                "Beta".into(),
                vec![
                    Pat::Call(
                        "mul".into(),
                        vec![Pat::Var("m".into()), Pat::Var("kappa".into())]
                    ),
                    Pat::Call(
                        "mul".into(),
                        vec![
                            Pat::Call("sub".into(), vec![Pat::Num(1.0), Pat::Var("m".into())]),
                            Pat::Var("kappa".into())
                        ]
                    ),
                ]
            )
        );
        for row in crate::tables::laws() {
            parse(&row.pattern);
        }
    }
}
