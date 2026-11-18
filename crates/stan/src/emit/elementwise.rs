//! Values Stan has no whole-array expression for (outer broadcasts through
//! `addaxes`, elementwise `ifelse` and comparisons, `rowstack` of an array of
//! vectors), filled elementwise by nested loops.

use super::{Emitter, Ex, P_ADD, P_ATOM, P_CMP, P_MUL, P_POW, P_TERNARY, P_UNARY};
use crate::refuse::{Refusal, Result};
use crate::stan_type::StanType;
use flatppl_core::{Dim, Node, NodeId, Scalar, Type};

const P_OR: u8 = 1;
const P_AND: u8 = 2;

/// Static extents of a value, outermost first; `None` for a dynamic one.
fn dims_of(ty: &Type) -> Vec<Option<u32>> {
    let d = |d: &Dim| match d {
        Dim::Static(n) => Some(*n),
        Dim::Dynamic => None,
    };
    match ty {
        Type::Array { shape, elem } => {
            let mut out: Vec<_> = shape.iter().map(d).collect();
            out.extend(dims_of(elem));
            out
        }
        Type::TVector { len, elem } => {
            let mut out = vec![d(len)];
            out.extend(dims_of(elem));
            out
        }
        _ => vec![],
    }
}

impl<'m> Emitter<'m> {
    /// `true` when `node` decomposes into per-element scalar expressions.
    pub(super) fn elementwise(&self, node: NodeId) -> bool {
        matches!(
            self.head(node),
            Some("broadcast" | "addaxes" | "rowstack" | "colstack" | "indicesof")
        )
    }

    /// `name` declared as `t` and filled by loops over its extents.
    pub(super) fn fill_loops(
        &mut self,
        name: &str,
        t: &StanType,
        node: NodeId,
    ) -> Result<Vec<String>> {
        let extents: Vec<String> = t.array.iter().chain(&t.dims).cloned().collect();
        let mut vars = Vec::new();
        for _ in &extents {
            vars.push(self.loop_var());
        }
        let body = self.elem(node, &vars);
        for v in &vars {
            self.used.remove(v);
        }
        let body = body?;
        let mut text = format!("{name}[{}] = {};", vars.join(", "), body.s);
        for (v, n) in vars.iter().zip(&extents).rev() {
            let inner: String = text.lines().map(|l| format!("  {l}\n")).collect();
            text = format!("for ({v} in 1:{n}) {{\n{inner}}}");
        }
        Ok(vec![format!("{};", t.render(name)), text])
    }

    pub(super) fn loop_var(&mut self) -> String {
        for c in ["i", "j", "k", "l", "m", "n"] {
            if !self.used.contains(c) && !self.reserved.contains(c) {
                self.used.insert(c.to_string());
                return c.to_string();
            }
        }
        self.fresh_local("idx")
    }

    /// The element of `node` at the loop indices `idx` (one per axis).
    pub(super) fn elem(&mut self, node: NodeId, idx: &[String]) -> Result<Ex> {
        let rank = self.shape(node).map(|s| s.rank).unwrap_or(0);
        if rank == 0 {
            return self.expr(node);
        }
        if self
            .fctx
            .as_ref()
            .is_some_and(|ctx| ctx.plan_arrays.is_some() && ctx.lifted.contains_key(&node))
        {
            return self.elem_indexed(node, node, idx);
        }
        let c = self.call(node);
        match self.head(node) {
            Some("broadcast") => {
                let c = c.unwrap();
                let Node::Const(f) = self.m.node(c.args[0]) else {
                    if self.user_callee(c.args[0]).is_some() {
                        return self.elem_indexed(node, node, idx);
                    }
                    return Err(Refusal::not_yet(Some(node), "broadcast of a user function"));
                };
                let f = self.name(*f);
                let args = c.args[1..].to_vec();
                let mut parts = Vec::new();
                for &a in &args {
                    parts.push(self.elem(a, idx)?);
                }
                self.apply(node, f, &args, parts)
            }
            Some("addaxes") => {
                let c = c.unwrap();
                let lit = |me: &Self, id: NodeId| match me.m.node(id) {
                    Node::Lit(Scalar::Int(n)) => usize::try_from(*n).ok(),
                    _ => None,
                };
                let (Some(lead), Some(trail)) = (lit(self, c.args[1]), lit(self, c.args[2])) else {
                    return Err(Refusal::not_yet(Some(node), "addaxes with computed counts"));
                };
                let inner = &idx[lead..idx.len() - trail];
                self.elem(c.args[0], inner)
            }
            Some("indicesof") if idx.len() == 1 => Ok(Ex::atom(idx[0].clone())),
            Some("rowstack") => {
                let x = c.unwrap().args[0];
                self.elem_indexed(node, x, idx)
            }
            Some("colstack") => {
                let x = c.unwrap().args[0];
                let swapped = [idx[1].clone(), idx[0].clone()];
                self.elem_indexed(node, x, &swapped)
            }
            _ => self.elem_indexed(node, node, idx),
        }
    }

    /// `x[idx]` for a value `x` that prints as a name; size-one axes index 1.
    fn elem_indexed(&mut self, at: NodeId, x: NodeId, idx: &[String]) -> Result<Ex> {
        if let [i] = idx
            && let Some(s) = self.gather_at(x, i)?
        {
            return Ok(Ex::atom(s));
        }
        let e = self.expr(x)?;
        // A short literal indexes in place: `([a, b]')[i]`.
        if self.head(x) == Some("vector") {
            return Ok(Ex::atom(format!("({})[{}]", e.s, idx.join(", "))));
        }
        // Any other value is computed once, then indexed.
        let e = if e.prec < P_ATOM || !e.s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            if self.shape(x).is_none_or(|s| s.rank == 0) {
                return Err(Refusal::not_yet(
                    Some(at),
                    "an elementwise use of this expression",
                ));
            }
            let base = match &self.owner {
                Some((name, _)) => format!("{name}_part"),
                None => "part".to_string(),
            };
            Ex::atom(self.place(&base, x, Ok(e))?)
        } else {
            e
        };
        let dims = self.ty(x).as_ref().map(dims_of).unwrap_or_default();
        let parts: Vec<String> = idx
            .iter()
            .enumerate()
            .map(|(a, i)| {
                if dims.get(a) == Some(&Some(1)) {
                    "1".to_string()
                } else {
                    i.clone()
                }
            })
            .collect();
        Ok(Ex::atom(format!("{}[{}]", e.s, parts.join(", "))))
    }

    /// A scalar function applied to per-element arguments.
    pub(super) fn apply(
        &mut self,
        node: NodeId,
        f: &str,
        args: &[NodeId],
        parts: Vec<Ex>,
    ) -> Result<Ex> {
        let int = |k: usize| self.shape(args[k]).is_some_and(|s| s.int);
        let bin = |op: &str, prec: u8| Ex {
            s: format!("{} {op} {}", parts[0].at(prec), parts[1].at(prec + 1)),
            prec,
        };
        Ok(match (f, parts.len()) {
            ("add", 2) => bin("+", P_ADD),
            ("sub", 2) => bin("-", P_ADD),
            ("mul", 2) => bin("*", P_MUL),
            ("divide", 2) if int(0) && int(1) => Ex {
                s: format!("1.0 * {} / {}", parts[0].at(P_MUL), parts[1].at(P_MUL + 1)),
                prec: P_MUL,
            },
            ("divide", 2) => bin("/", P_MUL),
            ("pow", 2) => Ex {
                s: format!("{} ^ {}", parts[0].at(P_POW + 1), parts[1].at(P_POW)),
                prec: P_POW,
            },
            ("neg", 1) => Ex {
                s: format!("-{}", parts[0].at(P_UNARY)),
                prec: P_UNARY,
            },
            ("lnot", 1) => Ex {
                s: format!("!{}", parts[0].at(P_UNARY)),
                prec: P_UNARY,
            },
            ("equal", 2) => cmp(&parts, "=="),
            ("unequal", 2) => cmp(&parts, "!="),
            ("lt", 2) => cmp(&parts, "<"),
            ("le", 2) => cmp(&parts, "<="),
            ("gt", 2) => cmp(&parts, ">"),
            ("ge", 2) => cmp(&parts, ">="),
            ("land", 2) => bin("&&", P_AND),
            ("lor", 2) => bin("||", P_OR),
            ("ifelse", 3) => Ex {
                s: format!(
                    "{} ? {} : {}",
                    parts[0].at(P_CMP),
                    parts[1].at(P_TERNARY + 1),
                    parts[2].at(P_TERNARY)
                ),
                prec: P_TERNARY,
            },
            (name, _) => return self.table_fn(node, name, None, parts),
        })
    }
}

fn cmp(parts: &[Ex], op: &str) -> Ex {
    Ex {
        s: format!("{} {op} {}", parts[0].at(P_CMP + 1), parts[1].at(P_CMP + 1)),
        prec: P_CMP,
    }
}
