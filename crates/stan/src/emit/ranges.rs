//! Affine integer ranges: `indicesof(v)` and its integer shifts and scalings.
//! Element `i` is `start + step * (i - 1)`. A unit-step range indexes as a
//! slice `a:b`; any other prints as `linspaced_int_array`, never a loop.

use super::{Emitter, Ex, P_ADD, P_MUL, P_UNARY};
use flatppl_core::{Node, NodeId, Scalar};

/// `konst + extra`, an integer expression.
#[derive(Clone, Debug)]
struct Offset {
    konst: i64,
    extra: Option<Ex>,
}

impl Offset {
    fn add(&self, c: i64, e: Option<&Ex>) -> Offset {
        let extra = match (&self.extra, e) {
            (None, None) => None,
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            (Some(a), Some(b)) => Some(Ex {
                s: format!("{} + {}", a.at(P_ADD), b.at(P_ADD + 1)),
                prec: P_ADD,
            }),
        };
        Offset {
            konst: self.konst + c,
            extra,
        }
    }

    fn neg(&self) -> Offset {
        Offset {
            konst: -self.konst,
            extra: self.extra.as_ref().map(|e| Ex {
                s: format!("-{}", e.at(P_UNARY)),
                prec: P_UNARY,
            }),
        }
    }

    fn scale(&self, k: i64) -> Offset {
        Offset {
            konst: self.konst * k,
            extra: self.extra.as_ref().map(|e| Ex {
                s: format!("{k} * {}", e.at(P_MUL + 1)),
                prec: P_MUL,
            }),
        }
    }

    fn print(&self) -> String {
        match &self.extra {
            None => self.konst.to_string(),
            Some(e) if self.konst == 0 => e.s.clone(),
            Some(e) if self.konst > 0 => format!("{} + {}", e.at(P_ADD), self.konst),
            Some(e) => format!("{} - {}", e.at(P_ADD), -self.konst),
        }
    }
}

/// An affine range of `n` integers.
pub(super) struct Range {
    n: u32,
    /// The length as the model writes it, for a plain `indicesof`.
    written: Option<Ex>,
    start: Offset,
    step: i64,
}

impl Range {
    fn last(&self) -> Offset {
        self.start.add(self.step * (i64::from(self.n) - 1), None)
    }

    /// The range as an index: a slice when the step is one.
    pub(super) fn index(&self) -> String {
        if let Some(n) = &self.written {
            return format!("1:{}", n.at(super::P_ATOM));
        }
        if self.step == 1 {
            format!("{}:{}", self.start.print(), self.last().print())
        } else {
            self.array()
        }
    }

    /// The range as an integer array.
    pub(super) fn array(&self) -> String {
        if let Some(n) = &self.written {
            return format!("linspaced_int_array({0}, 1, {0})", n.s);
        }
        let n = self.n;
        if self.step >= 0 || n == 1 {
            format!(
                "linspaced_int_array({n}, {}, {})",
                self.start.print(),
                self.last().print()
            )
        } else {
            // linspaced_int_array increases; a decreasing range reverses it.
            format!(
                "reverse(linspaced_int_array({n}, {}, {}))",
                self.last().print(),
                self.start.print()
            )
        }
    }
}

impl<'m> Emitter<'m> {
    /// `id` as an affine integer range, when it is one.
    pub(super) fn range(&mut self, id: NodeId) -> Option<Range> {
        let sh = self.shape(id)?;
        if !(sh.int && sh.rank == 1) {
            return None;
        }
        let n = match self.ty(id)? {
            flatppl_core::Type::Array { shape, .. } => match shape.as_ref() {
                [flatppl_core::Dim::Static(n)] => *n,
                _ => return None,
            },
            _ => return None,
        };
        let r = self.resolve(id);
        let c = self.call(r)?;
        let (f, args): (&str, Vec<NodeId>) = match self.head(r)? {
            "indicesof" => {
                let v = self.resolve(c.args[0]);
                let written = match (self.head(v), self.call(v).map(|c| c.args.to_vec())) {
                    (Some("ones" | "zeros"), Some(a)) if a.len() == 1 => self.expr(a[0]).ok(),
                    (Some("fill"), Some(a)) if a.len() == 2 => self.expr(a[1]).ok(),
                    _ => None,
                };
                return Some(Range {
                    n,
                    written,
                    start: Offset {
                        konst: 1,
                        extra: None,
                    },
                    step: 1,
                });
            }
            "broadcast" => match self.m.node(c.args[0]) {
                Node::Const(f) => (self.name(*f), c.args[1..].to_vec()),
                _ => return None,
            },
            h @ ("add" | "sub" | "neg") => (h, c.args.to_vec()),
            _ => return None,
        };
        let scalar = |me: &Self, a: NodeId| me.shape(a).is_some_and(|s| s.rank == 0 && s.int);
        let lit = |me: &Self, a: NodeId| match me.m.node(me.resolve(a)) {
            Node::Lit(Scalar::Int(k)) => Some(*k),
            _ => None,
        };
        // A scalar operand: a literal folds, anything else stays symbolic.
        let operand = |me: &mut Self, a: NodeId| -> Option<(i64, Option<Ex>)> {
            match lit(me, a) {
                Some(k) => Some((k, None)),
                None => me.expr(a).ok().map(|e| (0, Some(e))),
            }
        };
        match (f, args.as_slice()) {
            ("add", [a, b]) | ("add", [b, a]) if scalar(self, *b) && !scalar(self, *a) => {
                let x = self.range(*a)?;
                let (k, e) = operand(self, *b)?;
                Some(Range {
                    start: x.start.add(k, e.as_ref()),
                    written: None,
                    ..x
                })
            }
            ("sub", [a, b]) if scalar(self, *b) => {
                let x = self.range(*a)?;
                let (k, e) = operand(self, *b)?;
                let neg = e.map(|e| Ex {
                    s: format!("-{}", e.at(P_UNARY)),
                    prec: P_UNARY,
                });
                Some(Range {
                    start: x.start.add(-k, neg.as_ref()),
                    written: None,
                    ..x
                })
            }
            ("sub", [b, a]) if scalar(self, *b) => {
                let x = self.range(*a)?;
                let (k, e) = operand(self, *b)?;
                Some(Range {
                    n: x.n,
                    written: None,
                    start: x.start.neg().add(k, e.as_ref()),
                    step: -x.step,
                })
            }
            ("mul", [a, b]) | ("mul", [b, a]) if lit(self, *b).is_some() => {
                let x = self.range(*a)?;
                let k = lit(self, *b)?;
                Some(Range {
                    n: x.n,
                    written: None,
                    start: x.start.scale(k),
                    step: x.step * k,
                })
            }
            ("neg", [a]) => {
                let x = self.range(*a)?;
                Some(Range {
                    n: x.n,
                    written: None,
                    start: x.start.neg(),
                    step: -x.step,
                })
            }
            _ => None,
        }
    }
}
