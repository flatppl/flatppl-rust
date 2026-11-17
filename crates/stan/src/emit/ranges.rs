//! Affine integer ranges: `indicesof(v)` and its integer shifts and scalings.
//! Element `i` is `start + step * (i - 1)`. A unit-step range indexes as a
//! slice `a:b`; any other prints as `linspaced_int_array`, never a loop. The
//! length and ends are symbolic, so the range follows the data's shape.

use super::Emitter;
use super::extents::Lin;
use flatppl_core::{Dim, Node, NodeId, Scalar, Type};

/// An affine range of `n` integers.
pub(super) struct Range {
    n: Lin,
    start: Lin,
    step: i64,
}

/// A bound of a slice: parenthesized unless it is one atom.
fn bound(l: &Lin) -> String {
    if l.atomic() {
        l.print()
    } else {
        format!("({})", l.print())
    }
}

impl Range {
    fn last(&self) -> Lin {
        self.start.plus(&self.n.scale(self.step)).add(-self.step)
    }

    /// The range as an index: a slice when the step is one.
    pub(super) fn index(&self) -> String {
        if self.step == 1 {
            format!("{}:{}", bound(&self.start), bound(&self.last()))
        } else {
            self.array()
        }
    }

    /// The range as an integer array.
    pub(super) fn array(&self) -> String {
        let n = self.n.print();
        if self.step >= 0 || self.n.as_const() == Some(1) {
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
        let r = self.resolve(id);
        let c = self.call(r)?;
        let (f, args): (&str, Vec<NodeId>) = match self.head(r)? {
            "indicesof" => {
                let n = match self.sym_extent(r, 0, 0) {
                    Some(n) => n,
                    None => match self.ty(id)? {
                        Type::Array { shape, .. } => match shape.as_ref() {
                            [Dim::Static(n)] => Lin::konst(i64::from(*n)),
                            _ => return None,
                        },
                        _ => return None,
                    },
                };
                return Some(Range {
                    n,
                    start: Lin::konst(1),
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
        match (f, args.as_slice()) {
            ("add", [a, b]) | ("add", [b, a]) if scalar(self, *b) && !scalar(self, *a) => {
                let x = self.range(*a)?;
                let k = self.lin_int(*b)?;
                Some(Range {
                    start: x.start.plus(&k),
                    ..x
                })
            }
            ("sub", [a, b]) if scalar(self, *b) => {
                let x = self.range(*a)?;
                let k = self.lin_int(*b)?;
                Some(Range {
                    start: x.start.plus(&k.scale(-1)),
                    ..x
                })
            }
            ("sub", [b, a]) if scalar(self, *b) => {
                let x = self.range(*a)?;
                let k = self.lin_int(*b)?;
                Some(Range {
                    n: x.n,
                    start: x.start.scale(-1).plus(&k),
                    step: -x.step,
                })
            }
            ("mul", [a, b]) | ("mul", [b, a]) if lit(self, *b).is_some() => {
                let x = self.range(*a)?;
                let k = lit(self, *b)?;
                Some(Range {
                    n: x.n,
                    start: x.start.scale(k),
                    step: x.step * k,
                })
            }
            ("neg", [a]) => {
                let x = self.range(*a)?;
                Some(Range {
                    n: x.n,
                    start: x.start.scale(-1),
                    step: -x.step,
                })
            }
            _ => None,
        }
    }
}
