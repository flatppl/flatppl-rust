//! `pushfwd(f, M)` through an affine bijection f(x) = a·x + b (spec §06
//! "Engine contract for `pushfwd` density evaluation"):
//! log density(y) = log density_M(f⁻¹(y)) − log|a|. The variate's
//! statement is `(y - b) / a ~ M;` plus `target += -log(|a|);`, and a
//! parameter is declared on M's support mapped through f.

use super::structure::Law;
use super::{Emitter, Ex, P_ADD, P_ATOM, P_MUL, P_UNARY};
use crate::refuse::{Refusal, RefusalKind, Result};
use crate::stan_type::{bounds_of, number};
use flatppl_core::{Node, NodeId, Phase, Ref, Scalar};

/// A coefficient of the map: its number when it folds, its Stan text, and
/// whether that text is an integer in Stan. FlatPPL's `/` is real division,
/// so a quotient of two integer texts prints `1.0 * x / y`. `+`, `-` and
/// `*` of integers are exact (overflow aside).
#[derive(Clone)]
struct Coef {
    num: Option<f64>,
    ex: Ex,
    int: bool,
}

fn lit(x: f64) -> Coef {
    let ex = if x < 0.0 {
        Ex {
            s: format!("-{}", number(-x)),
            prec: P_UNARY,
        }
    } else {
        Ex::atom(number(x))
    };
    // `number` prints a whole number without a decimal point.
    let int = x.fract() == 0.0;
    Coef {
        num: Some(x),
        ex,
        int,
    }
}

fn op(x: &Coef, y: &Coef, sym: &str, prec: u8, fold: fn(f64, f64) -> f64) -> Coef {
    match (x.num, y.num) {
        (Some(a), Some(b)) => lit(fold(a, b)),
        _ if sym == "/" => {
            let l = if x.int && y.int {
                format!("1.0 * {}", x.ex.at(P_MUL))
            } else {
                x.ex.at(prec)
            };
            Coef {
                num: None,
                ex: Ex {
                    s: format!("{l} / {}", y.ex.at(prec + 1)),
                    prec,
                },
                int: false,
            }
        }
        _ => Coef {
            num: None,
            ex: Ex {
                s: format!("{} {sym} {}", x.ex.at(prec), y.ex.at(prec + 1)),
                prec,
            },
            int: x.int && y.int,
        },
    }
}

fn add(x: &Coef, y: &Coef) -> Coef {
    match (x.num, y.num) {
        (Some(0.0), _) => y.clone(),
        (_, Some(0.0)) => x.clone(),
        _ => op(x, y, "+", P_ADD, |a, b| a + b),
    }
}

fn sub(x: &Coef, y: &Coef) -> Coef {
    match (x.num, y.num) {
        (_, Some(0.0)) => x.clone(),
        (Some(0.0), None) => Coef {
            num: None,
            ex: Ex {
                s: format!("-{}", y.ex.at(P_UNARY)),
                prec: P_UNARY,
            },
            int: y.int,
        },
        _ => op(x, y, "-", P_ADD, |a, b| a - b),
    }
}

fn mul(x: &Coef, y: &Coef) -> Coef {
    match (x.num, y.num) {
        (Some(0.0), _) | (_, Some(0.0)) => lit(0.0),
        (Some(1.0), _) => y.clone(),
        (_, Some(1.0)) => x.clone(),
        _ => op(x, y, "*", P_MUL, |a, b| a * b),
    }
}

fn div(x: &Coef, y: &Coef) -> Coef {
    match (x.num, y.num) {
        (Some(0.0), _) => lit(0.0),
        (_, Some(1.0)) => x.clone(),
        _ => op(x, y, "/", P_MUL, |a, b| a / b),
    }
}

/// f(x) = a·x + b.
pub(super) struct Affine {
    a: Coef,
    b: Coef,
}

impl Affine {
    /// f⁻¹(y) = (y − b) / a, folded for literal coefficients. A negative
    /// literal scale prints as (b − y) / |a|.
    fn inverse(&self, y: &str) -> String {
        let flip = self.a.num.is_some_and(|a| a < 0.0);
        let shifted = match (flip, self.b.num) {
            (false, Some(0.0)) => Ex::atom(y.to_string()),
            (false, Some(b)) if b < 0.0 => Ex {
                s: format!("{y} + {}", number(-b)),
                prec: P_ADD,
            },
            (false, _) => Ex {
                s: format!("{y} - {}", self.b.ex.at(P_ADD + 1)),
                prec: P_ADD,
            },
            (true, Some(0.0)) => Ex {
                s: format!("-{y}"),
                prec: P_UNARY,
            },
            (true, _) => Ex {
                s: format!("{} - {y}", self.b.ex.at(P_ADD)),
                prec: P_ADD,
            },
        };
        match self.a.num {
            Some(a) if a.abs() == 1.0 => shifted.s,
            Some(a) => format!("{} / {}", shifted.at(P_MUL), number(a.abs())),
            None => format!("{} / {}", shifted.at(P_MUL), self.a.ex.at(P_MUL + 1)),
        }
    }

    /// `target += -log|a|;`, none where |a| = 1.
    fn jacobian(&self) -> Option<String> {
        match self.a.num {
            Some(a) if a.abs() == 1.0 => None,
            Some(a) => Some(format!("target += -log({});", number(a.abs()))),
            None => Some(format!("target += -log(abs({}));", self.a.ex.s)),
        }
    }

    /// f(e) for a support end `e`.
    fn image(&self, e: &str) -> String {
        let end = match e.parse::<f64>() {
            Ok(x) => lit(x),
            Err(_) => {
                let atom = e.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
                // An end in integer text stays exact: `image` only adds and
                // multiplies it.
                Coef {
                    num: None,
                    ex: Ex {
                        s: e.to_string(),
                        prec: if atom { P_ATOM } else { P_ADD },
                    },
                    int: true,
                }
            }
        };
        let out = add(&mul(&self.a, &end), &self.b);
        out.num.map(number).unwrap_or(out.ex.s)
    }
}

impl<'m> Emitter<'m> {
    /// `pushfwd` through a field projection is no bijection.
    pub(super) fn pushfwd_projection(&self, m: NodeId) -> Result<()> {
        let f = self.resolve(self.call(m).unwrap().args[0]);
        let projection = self.head(f) == Some("functionof")
            && matches!(
                self.head(self.resolve(self.call(f).unwrap().args[0])),
                Some("get" | "record")
            );
        if projection {
            return Err(Refusal::new(
                RefusalKind::NonBijectivePushfwd,
                Some(m),
                "pushfwd",
                "`pushfwd` through a non-bijective map (a field projection) has no density \
                 Stan can express",
            ));
        }
        Ok(())
    }

    /// The affine map of `pushfwd(f, M)` and M.
    fn affine_pushfwd(&mut self, m: NodeId) -> Result<(Affine, NodeId)> {
        self.pushfwd_projection(m)?;
        let c = self.call(m).unwrap();
        if c.args.len() != 2 || !c.named.is_empty() {
            return Err(Refusal::not_yet(Some(m), "pushfwd in this form"));
        }
        let (f, base) = (c.args[0], c.args[1]);
        let fun = self
            .user_callee(f)
            .ok_or_else(|| Refusal::not_yet(Some(m), "pushfwd through this map"))?;
        let (body, entries) = self.callable_parts(fun)?;
        let [(_, x)] = entries[..] else {
            return Err(Refusal::not_yet(Some(m), "pushfwd through this map"));
        };
        let Some((a, b)) = self.affine(body, x, m)? else {
            return Err(Refusal::new(
                RefusalKind::NonBijectivePushfwd,
                Some(m),
                "pushfwd",
                "`pushfwd` through a constant map is no bijection; its law has no density",
            ));
        };
        if a.num == Some(0.0) {
            return Err(Refusal::new(
                RefusalKind::NonBijectivePushfwd,
                Some(m),
                "pushfwd",
                "`pushfwd` through an affine map with scale 0 is no bijection; its law has \
                 no density",
            ));
        }
        Ok((Affine { a, b }, base))
    }

    /// `(a, b)` of `id` as `a·x + b`, `None` when `id` does not read `x`.
    fn affine(&mut self, id: NodeId, x: Ref, at: NodeId) -> Result<Option<(Coef, Coef)>> {
        if matches!(self.m.node(id), Node::Ref(r) if *r == x) {
            return Ok(Some((lit(1.0), lit(0.0))));
        }
        if !self.reads(id, x) {
            return Ok(None);
        }
        let not_affine = || Refusal::not_yet(Some(at), "pushfwd through a map that is not affine");
        let args = self.call(id).map(|c| c.args.to_vec()).unwrap_or_default();
        let head = self.head(id).ok_or_else(not_affine)?;
        let part = |me: &mut Self, a: NodeId| -> Result<Part> {
            match me.affine(a, x, at)? {
                Some(ab) => Ok(Part::Var(ab)),
                None => me.coef(a, at).map(Part::Const),
            }
        };
        let out = match (head, args.as_slice()) {
            ("neg", [u]) => match part(self, *u)? {
                Part::Var((a, b)) => (sub(&lit(0.0), &a), sub(&lit(0.0), &b)),
                Part::Const(_) => return Err(not_affine()),
            },
            ("add" | "sub" | "mul" | "divide", [u, v]) => {
                match (head, part(self, *u)?, part(self, *v)?) {
                    ("add", Part::Var((a, b)), Part::Const(c))
                    | ("add", Part::Const(c), Part::Var((a, b))) => (a, add(&b, &c)),
                    ("sub", Part::Var((a, b)), Part::Const(c)) => (a, sub(&b, &c)),
                    ("sub", Part::Const(c), Part::Var((a, b))) => (sub(&lit(0.0), &a), sub(&c, &b)),
                    ("mul", Part::Var((a, b)), Part::Const(c))
                    | ("mul", Part::Const(c), Part::Var((a, b))) => (mul(&a, &c), mul(&b, &c)),
                    ("divide", Part::Var((a, b)), Part::Const(c)) => (div(&a, &c), div(&b, &c)),
                    _ => return Err(not_affine()),
                }
            }
            _ => return Err(not_affine()),
        };
        Ok(Some(out))
    }

    /// A scalar coefficient that does not depend on parameters.
    fn coef(&mut self, c: NodeId, at: NodeId) -> Result<Coef> {
        if self.m.phase_of(c) != Some(Phase::Fixed) {
            return Err(Refusal::not_yet(
                Some(at),
                "pushfwd through a map whose coefficients depend on parameters",
            ));
        }
        if self.shape(c).is_none_or(|s| s.rank != 0) {
            return Err(Refusal::not_yet(
                Some(at),
                "pushfwd through a map with an array coefficient",
            ));
        }
        match self.fold(c) {
            Some(x) => Ok(lit(x)),
            None => {
                let ex = self.at_module(|me| me.expr(c))?;
                let int = self.shape(c).is_some_and(|s| s.int);
                Ok(Coef { num: None, ex, int })
            }
        }
    }

    /// The number of a literal arithmetic expression (`-3`, `1 / 2`).
    fn fold(&self, c: NodeId) -> Option<f64> {
        let c = self.resolve(c);
        match self.m.node(c) {
            Node::Lit(Scalar::Int(k)) => return Some(*k as f64),
            Node::Lit(Scalar::Real(r)) => return Some(*r),
            _ => {}
        }
        let args = self.call(c)?.args.to_vec();
        match (self.head(c)?, args.as_slice()) {
            ("neg", [u]) => Some(-self.fold(*u)?),
            ("add", [u, v]) => Some(self.fold(*u)? + self.fold(*v)?),
            ("sub", [u, v]) => Some(self.fold(*u)? - self.fold(*v)?),
            ("mul", [u, v]) => Some(self.fold(*u)? * self.fold(*v)?),
            ("divide", [u, v]) => Some(self.fold(*u)? / self.fold(*v)?),
            _ => None,
        }
    }

    /// `true` when `id` reads the input `x`.
    fn reads(&self, id: NodeId, x: Ref) -> bool {
        let mut stack = vec![id];
        while let Some(n) = stack.pop() {
            if matches!(self.m.node(n), Node::Ref(r) if *r == x) {
                return true;
            }
            self.m.for_each_child(n, |k| stack.push(k));
        }
        false
    }

    /// The statements of a scalar variate `lhs ~ pushfwd(f, M)`.
    pub(super) fn pushfwd_statement(
        &mut self,
        lhs: &str,
        m: NodeId,
        lhs_rank: usize,
    ) -> Result<Vec<String>> {
        let (map, base) = self.affine_pushfwd(m)?;
        if lhs_rank != 0 {
            return Err(Refusal::not_yet(Some(m), "pushfwd of an array variate"));
        }
        let law: Law = self.law_any(base)?;
        // `T[L, U]` truncates M's own variate, the inverse, as it should.
        // An `extra` line (only an ordered `iid`, a vector law) is not
        // written for the inverse, so it refuses.
        if law.discrete
            || law.multivariate
            || law.declared
            || law.support.is_some()
            || !law.extra.is_empty()
        {
            return Err(Refusal::not_yet(Some(m), "pushfwd of this law"));
        }
        let inverse = map.inverse(lhs);
        let y = if law.lhs("{y}") == "{y}" {
            inverse
        } else {
            law.lhs(&format!("({inverse})"))
        };
        // A data scale that is 0 makes the map no bijection: name it at load.
        if map.a.num.is_none() {
            let a = &map.a.ex.s;
            self.tdata.push(format!(
                "if ({a} == 0) {{\n  reject(\"pushfwd of {lhs}: the scale {a} is 0, so the map is no bijection\");\n}}"
            ));
        }
        // An unnormalized base prints `target += …_lpdf(…)` (`Law::statement`).
        let mut out = vec![law.statement(m, &y, &law.call_text(None))?];
        out.extend(map.jacobian());
        // An `unchecked_support` base (von Mises): its preimage is tested.
        Ok(self.unchecked_guard(&y, false, &law, out))
    }

    /// M's support mapped through f: the bounds of a `pushfwd` parameter.
    pub(super) fn pushfwd_support(
        &mut self,
        m: NodeId,
    ) -> Result<Option<(Option<String>, Option<String>)>> {
        let (map, base) = self.affine_pushfwd(m)?;
        let (lo, hi) = match self.support(base)? {
            Some(ends) => ends,
            None => {
                let b = self
                    .m
                    .valueset_of(self.resolve(base))
                    .map(bounds_of)
                    .unwrap_or_default();
                (b.lower, b.upper)
            }
        };
        if lo.is_none() && hi.is_none() {
            return Ok(None);
        }
        let (lo, hi) = (lo.map(|e| map.image(&e)), hi.map(|e| map.image(&e)));
        match map.a.num {
            Some(a) if a > 0.0 => Ok(Some((lo, hi))),
            Some(_) => Ok(Some((hi, lo))),
            None => Err(Refusal::not_yet(
                Some(m),
                "pushfwd of a bounded law through a scale of unknown sign",
            )),
        }
    }
}

enum Part {
    Var((Coef, Coef)),
    Const(Coef),
}
