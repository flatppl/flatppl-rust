//! `pushfwd(f, M)` through a §06 bijection f (spec §06 "Engine contract
//! for `pushfwd` density evaluation", case 1): log density(y) =
//! log density_M(f⁻¹(y)) + log|d f⁻¹/dy|. f is a chain of steps from the
//! registry `STEPS`, plus the two parameterized steps, affine and
//! `pow(_, k)`. The variate's statement is `f⁻¹(y) ~ M;` plus one
//! `target +=` per step, and a parameter is declared on M's support mapped
//! through f. A derived prior field `v = f(x)` of one draw x takes the same
//! path as `pushfwd(f, law(x))`.

use super::structure::Law;
use super::{Emitter, Ex, P_ADD, P_ATOM, P_MUL, P_UNARY};
use crate::refuse::{Refusal, RefusalKind, Result};
use crate::stan_type::{bounds_of, number};
use flatppl_core::{Node, NodeId, Phase, Ref, RefNs, Scalar};

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
    fn inverse(&self, y: &str) -> Ex {
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
            Some(a) if a.abs() == 1.0 => shifted,
            Some(a) => Ex {
                s: format!("{} / {}", shifted.at(P_MUL), number(a.abs())),
                prec: P_MUL,
            },
            None => Ex {
                s: format!("{} / {}", shifted.at(P_MUL), self.a.ex.at(P_MUL + 1)),
                prec: P_MUL,
            },
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

/// One row of the §06 bijection registry (case 1): a monotone map that
/// Stan 2.39 inverts in closed form. A row lands only with its check: a
/// BridgeStan score against an oracle that uses the forward function alone.
struct StepRow {
    /// The FlatPPL function.
    name: &'static str,
    /// f⁻¹(y) in Stan: `{y}` is y in argument position.
    inverse: &'static str,
    /// log|d f⁻¹/dy| at y: `{y}` in argument position, `{Y}` as a factor.
    log_jacobian: &'static str,
    /// The forward domain's ends; `None` is unbounded.
    domain: (Option<f64>, Option<f64>),
    /// f at −∞ and at +∞, as Stan text and its number; `None` is an
    /// unbounded image end. Such an end is a limit, never reached: open.
    limits: (Limit, Limit),
    /// f of a support end `{e}` in Stan.
    forward: &'static str,
    /// f of a literal end, when it folds to a number.
    fold: fn(f64) -> Option<f64>,
    /// Every row is increasing; the decreasing steps are the parameterized
    /// ones (affine with a < 0, `pow` with k < 0).
    increasing: bool,
}

const UNBOUNDED: (Option<f64>, Option<f64>) = (None, None);

/// A map's limit at −∞ or +∞: Stan text and number, `None` unbounded.
type Limit = Option<(&'static str, f64)>;

static STEPS: &[StepRow] = &[
    StepRow {
        name: "exp",
        inverse: "log({y})",
        log_jacobian: "-log({y})",
        domain: UNBOUNDED,
        limits: (Some(("0", 0.0)), None),
        forward: "exp({e})",
        fold: |x| Some(x.exp()),
        increasing: true,
    },
    StepRow {
        name: "log",
        inverse: "exp({y})",
        log_jacobian: "{y}",
        domain: (Some(0.0), None),
        limits: (None, None),
        forward: "log({e})",
        fold: |x| Some(x.ln()),
        increasing: true,
    },
    StepRow {
        name: "log10",
        inverse: "pow(10, {y})",
        log_jacobian: "{Y} * log(10) + log(log(10))",
        domain: (Some(0.0), None),
        limits: (None, None),
        forward: "log10({e})",
        fold: |x| Some(x.log10()),
        increasing: true,
    },
    StepRow {
        name: "log1p",
        inverse: "expm1({y})",
        log_jacobian: "{y}",
        domain: (Some(-1.0), None),
        limits: (None, None),
        forward: "log1p({e})",
        fold: |x| Some(x.ln_1p()),
        increasing: true,
    },
    StepRow {
        name: "expm1",
        inverse: "log1p({y})",
        log_jacobian: "-log1p({y})",
        domain: UNBOUNDED,
        limits: (Some(("-1", -1.0)), None),
        forward: "expm1({e})",
        fold: |x| Some(x.exp_m1()),
        increasing: true,
    },
    StepRow {
        name: "logit",
        inverse: "inv_logit({y})",
        log_jacobian: "log_inv_logit({y}) + log1m_inv_logit({y})",
        domain: (Some(0.0), Some(1.0)),
        limits: (None, None),
        forward: "logit({e})",
        fold: |x| Some((x / (1.0 - x)).ln()),
        increasing: true,
    },
    StepRow {
        name: "invlogit",
        inverse: "logit({y})",
        log_jacobian: "-log({y}) - log1m({y})",
        domain: UNBOUNDED,
        limits: (Some(("0", 0.0)), Some(("1", 1.0))),
        forward: "inv_logit({e})",
        fold: |x| Some(1.0 / (1.0 + (-x).exp())),
        increasing: true,
    },
    StepRow {
        name: "probit",
        inverse: "Phi({y})",
        log_jacobian: "std_normal_lpdf({y})",
        domain: (Some(0.0), Some(1.0)),
        limits: (None, None),
        forward: "inv_Phi({e})",
        fold: |x| match x {
            0.0 => Some(f64::NEG_INFINITY),
            0.5 => Some(0.0),
            1.0 => Some(f64::INFINITY),
            _ => None,
        },
        increasing: true,
    },
    StepRow {
        name: "invprobit",
        inverse: "inv_Phi({y})",
        log_jacobian: "-std_normal_lpdf(inv_Phi({y}))",
        domain: UNBOUNDED,
        limits: (Some(("0", 0.0)), Some(("1", 1.0))),
        forward: "Phi({e})",
        fold: |x| (x == 0.0).then_some(0.5),
        increasing: true,
    },
    StepRow {
        name: "atan",
        inverse: "tan({y})",
        log_jacobian: "-2 * log(cos({y}))",
        domain: UNBOUNDED,
        limits: (
            Some(("-pi() / 2", -std::f64::consts::FRAC_PI_2)),
            Some(("pi() / 2", std::f64::consts::FRAC_PI_2)),
        ),
        forward: "atan({e})",
        fold: |x| Some(x.atan()),
        increasing: true,
    },
    StepRow {
        name: "sinh",
        inverse: "asinh({y})",
        log_jacobian: "-0.5 * log1p(square({y}))",
        domain: UNBOUNDED,
        limits: (None, None),
        forward: "sinh({e})",
        fold: |x| Some(x.sinh()),
        increasing: true,
    },
    StepRow {
        name: "asinh",
        inverse: "sinh({y})",
        // log(cosh(y)) without the overflow of cosh for |y| > 710.
        log_jacobian: "abs({y}) + log1p_exp(-2 * abs({y})) - log(2)",
        domain: UNBOUNDED,
        limits: (None, None),
        forward: "asinh({e})",
        fold: |x| Some(x.asinh()),
        increasing: true,
    },
    StepRow {
        name: "tanh",
        inverse: "atanh({y})",
        log_jacobian: "-log1m(square({y}))",
        domain: UNBOUNDED,
        limits: (Some(("-1", -1.0)), Some(("1", 1.0))),
        forward: "tanh({e})",
        fold: |x| Some(x.tanh()),
        increasing: true,
    },
];

/// One step of a map, innermost first in a chain.
enum Step {
    Affine(Affine),
    /// `pow(_, k)` with a finite nonzero literal k (`sqrt` is k = 1/2).
    Pow(f64),
    Row(&'static StepRow),
}

/// An odd integer k: `pow(_, k)` is then a bijection of the reals.
fn odd(k: f64) -> bool {
    k.fract() == 0.0 && (k as i64) % 2 != 0
}

/// A support or image end: its Stan text, its number where it folds, and
/// whether it is open (a limit of the map at ±∞, never reached). An end
/// that is the image of a finite base end is closed (the closed-end
/// reading of the base support).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct End {
    pub text: String,
    pub num: Option<f64>,
    pub open: bool,
}

impl End {
    fn of(text: String) -> End {
        let num = text.parse::<f64>().ok();
        End {
            text,
            num,
            open: false,
        }
    }

    /// A folded value; an infinite one is an unbounded end.
    fn value(x: f64, open: bool) -> Option<End> {
        x.is_finite().then(|| End {
            text: number(x),
            num: Some(x),
            open,
        })
    }
}

/// `c * log(y)` for a literal c, with `log(y)` and `-log(y)` for ±1.
fn times_log(c: f64, log: &str) -> Option<String> {
    match c {
        0.0 => None,
        1.0 => Some(log.to_string()),
        -1.0 => Some(format!("-{log}")),
        _ => Some(format!("{} * {log}", number(c))),
    }
}

/// Terms joined into one sum, a leading minus kept as `- `.
fn sum(terms: Vec<String>) -> Option<String> {
    let mut out = String::new();
    for t in terms {
        if out.is_empty() {
            out = t;
        } else if let Some(rest) = t.strip_prefix('-') {
            out = format!("{out} - {rest}");
        } else {
            out = format!("{out} + {t}");
        }
    }
    (!out.is_empty()).then_some(out)
}

impl Step {
    /// The name a refusal uses.
    fn name(&self) -> &'static str {
        match self {
            Step::Affine(_) => "an affine map",
            Step::Pow(k) if *k == 0.5 => "sqrt",
            Step::Pow(_) => "pow",
            Step::Row(r) => r.name,
        }
    }

    /// f⁻¹(y).
    fn inverse(&self, y: &Ex) -> Ex {
        match self {
            Step::Affine(a) => a.inverse(&y.at(P_UNARY)),
            Step::Pow(k) => {
                let r = 1.0 / k;
                let call = |f: &str| Ex::atom(format!("{f}({})", y.s));
                if odd(*k) {
                    match *k {
                        1.0 => y.clone(),
                        3.0 => call("cbrt"),
                        -1.0 => call("inv"),
                        _ => Ex {
                            s: format!(
                                "{} < 0 ? -pow(-{}, {r}) : pow({}, {r})",
                                y.at(P_ADD),
                                y.at(P_UNARY),
                                y.s,
                                r = number(r)
                            ),
                            prec: super::P_TERNARY,
                        },
                    }
                } else if r == 2.0 {
                    call("square")
                } else if r == 0.5 {
                    call("sqrt")
                } else if r == -1.0 {
                    call("inv")
                } else {
                    Ex::atom(format!("pow({}, {})", y.s, number(r)))
                }
            }
            Step::Row(row) => Ex::atom(row.inverse.replace("{y}", &y.s)),
        }
    }

    /// The `target +=` line of log|d f⁻¹/dy| at y, none where it is 0.
    fn jacobian(&self, y: &Ex) -> Option<String> {
        let term = match self {
            Step::Affine(a) => return a.jacobian(),
            Step::Pow(k) => {
                let r = 1.0 / k;
                let log = if odd(*k) {
                    format!("log(abs({}))", y.s)
                } else {
                    format!("log({})", y.s)
                };
                // log|r| = −log|k|: the shorter literal prints (`log(2)` for
                // sqrt, `-log(3)` for k = 3).
                let scale = (r.abs() != 1.0).then(|| {
                    let (rk, kk) = (number(r.abs()), number(k.abs()));
                    if kk.len() < rk.len() {
                        format!("-log({kk})")
                    } else {
                        format!("log({rk})")
                    }
                });
                sum(scale.into_iter().chain(times_log(r - 1.0, &log)).collect())?
            }
            Step::Row(row) => row
                .log_jacobian
                .replace("{y}", &y.s)
                .replace("{Y}", &y.at(P_MUL)),
        };
        Some(format!("target += {term};"))
    }

    /// The forward domain.
    fn domain(&self) -> (Option<f64>, Option<f64>) {
        match self {
            Step::Affine(_) => UNBOUNDED,
            Step::Pow(k) if odd(*k) && *k > 0.0 => UNBOUNDED,
            // §06: any other exponent needs the support within the
            // nonnegative reals; a negative one also keeps it away from 0.
            Step::Pow(_) => (Some(0.0), None),
            Step::Row(r) => r.domain,
        }
    }

    /// The image of the support ends `(lo, hi)`: each end mapped, swapped
    /// for a decreasing step. `None` when an affine scale has no sign.
    #[allow(clippy::type_complexity)]
    fn image(&self, lo: Option<End>, hi: Option<End>) -> Option<(Option<End>, Option<End>)> {
        match self {
            Step::Affine(a) => {
                if lo.is_none() && hi.is_none() {
                    return Some((None, None));
                }
                let at = |e: Option<End>| {
                    e.map(|e| match (e.num, a.a.num, a.b.num) {
                        (Some(x), Some(s), Some(t)) => End {
                            text: number(s * x + t),
                            num: Some(s * x + t),
                            open: e.open,
                        },
                        _ => End {
                            text: a.image(&e.text),
                            num: None,
                            open: e.open,
                        },
                    })
                };
                let (lo, hi) = (at(lo), at(hi));
                match a.a.num {
                    Some(s) if s > 0.0 => Some((lo, hi)),
                    Some(_) => Some((hi, lo)),
                    None => None,
                }
            }
            Step::Pow(k) => {
                let at = |e: Option<End>, inf: f64| -> Option<End> {
                    match e {
                        None => End::value(inf.powf(*k), true).filter(|_| *k < 0.0),
                        Some(e) => match e.num {
                            Some(x) => End::value(x.powf(*k), e.open),
                            None => Some(End {
                                text: format!("pow({}, {})", e.text, number(*k)),
                                num: None,
                                open: e.open,
                            }),
                        },
                    }
                };
                let (l, h) = (at(lo, f64::NEG_INFINITY), at(hi, f64::INFINITY));
                Some(if *k > 0.0 { (l, h) } else { (h, l) })
            }
            Step::Row(row) => {
                let at = |e: Option<End>, limit: Option<(&str, f64)>| -> Option<End> {
                    match e {
                        None => limit.map(|(t, v)| End {
                            text: t.to_string(),
                            num: Some(v),
                            open: true,
                        }),
                        Some(e) => match e.num.and_then(row.fold) {
                            Some(v) => End::value(v, e.open),
                            None => Some(End {
                                text: row.forward.replace("{e}", &e.text),
                                num: None,
                                open: e.open,
                            }),
                        },
                    }
                };
                let (l, h) = (at(lo, row.limits.0), at(hi, row.limits.1));
                Some(if row.increasing { (l, h) } else { (h, l) })
            }
        }
    }
}

/// The steps of a map, innermost first.
pub(super) struct Bijection {
    steps: Vec<Step>,
}

impl Bijection {
    /// f⁻¹(y): the steps undone outermost first.
    fn inverse(&self, y: &str) -> Ex {
        let mut v = Ex::atom(y.to_string());
        for s in self.steps.iter().rev() {
            v = s.inverse(&v);
        }
        v
    }

    /// One `target +=` per step, outermost first, each at its own output.
    fn jacobians(&self, y: &str) -> Vec<String> {
        let mut v = Ex::atom(y.to_string());
        let mut out = Vec::new();
        for s in self.steps.iter().rev() {
            out.extend(s.jacobian(&v));
            v = s.inverse(&v);
        }
        out
    }

    fn affine_only(&self) -> bool {
        self.steps.iter().all(|s| matches!(s, Step::Affine(_)))
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

    /// The map of `pushfwd(f, M)`, or of a derived prior field, as its
    /// body, its input and M.
    /// A built-in name as the map (`pushfwd(log, M)`) is not supported yet:
    /// inference leaves its variate untyped.
    pub(super) fn bare_pushfwd_map(&self, m: NodeId) -> Result<()> {
        let m = self.resolve(m);
        if self.head(m) == Some("pushfwd")
            && let Some(&f) = self.call(m).unwrap().args.first()
            && let Node::Const(name) = self.m.node(self.resolve(f))
        {
            let name = self.name(*name);
            return Err(Refusal::not_yet(
                Some(m),
                format!("pushfwd through the bare name `{name}`; write `fn({name}(_))`"),
            ));
        }
        Ok(())
    }

    fn pushfwd_parts(&mut self, m: NodeId) -> Result<(NodeId, Ref, NodeId)> {
        if let Some(&parts) = self.derived.get(&m) {
            return Ok(parts);
        }
        self.bare_pushfwd_map(m)?;
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
        Ok((body, x, base))
    }

    /// The bijection of `pushfwd(f, M)` and M.
    fn bijection(&mut self, m: NodeId) -> Result<(Bijection, NodeId)> {
        let (body, x, base) = self.pushfwd_parts(m)?;
        let Some(steps) = self.steps(body, x, m)? else {
            return Err(Refusal::new(
                RefusalKind::NonBijectivePushfwd,
                Some(m),
                "pushfwd",
                "`pushfwd` through a constant map is no bijection; its law has no density",
            ));
        };
        for s in &steps {
            if let Step::Affine(a) = s
                && a.a.num == Some(0.0)
            {
                return Err(Refusal::new(
                    RefusalKind::NonBijectivePushfwd,
                    Some(m),
                    "pushfwd",
                    "`pushfwd` through an affine map with scale 0 is no bijection; its law has \
                     no density",
                ));
            }
        }
        Ok((Bijection { steps }, base))
    }

    /// The steps of `id` as a map of `x`, innermost first; `None` when `id`
    /// does not read `x`. Consecutive affine operations fold into one step.
    fn steps(&mut self, id: NodeId, x: Ref, at: NodeId) -> Result<Option<Vec<Step>>> {
        let id = self.resolve(id);
        if matches!(self.m.node(id), Node::Ref(r) if *r == x) {
            return Ok(Some(vec![]));
        }
        if !self.reads(id, x) {
            return Ok(None);
        }
        let args = self.call(id).map(|c| c.args.to_vec()).unwrap_or_default();
        let Some(head) = self.head(id) else {
            return Err(Refusal::not_yet(Some(at), "pushfwd through this map"));
        };
        let unregistered = || {
            Refusal::not_yet(
                Some(at),
                format!("pushfwd through `{head}`, which is not in the §06 bijection registry"),
            )
        };
        let more_than_once = || {
            Refusal::not_yet(
                Some(at),
                "pushfwd through a map that reads its input more than once",
            )
        };
        // One argument reads x: the inner map, then this step.
        let inner = |me: &mut Self, u: NodeId| -> Result<Vec<Step>> {
            Ok(me.steps(u, x, at)?.expect("the argument reads x"))
        };
        match (head, args.as_slice()) {
            ("sqrt", [u]) => {
                let mut s = inner(self, *u)?;
                s.push(Step::Pow(0.5));
                return Ok(Some(s));
            }
            ("pow", [u, k]) => {
                if self.reads(*k, x) {
                    return Err(more_than_once());
                }
                let k = self.fold(*k).filter(|k| k.is_finite() && *k != 0.0).ok_or_else(|| {
                    Refusal::not_yet(
                        Some(at),
                        "pushfwd through `pow` with an exponent that is not a finite nonzero literal",
                    )
                })?;
                let mut s = inner(self, *u)?;
                s.push(Step::Pow(k));
                return Ok(Some(s));
            }
            ("neg" | "add" | "sub" | "mul" | "divide", _) => {}
            (h, [u]) => {
                let row = STEPS
                    .iter()
                    .find(|r| r.name == h)
                    .ok_or_else(unregistered)?;
                let mut s = inner(self, *u)?;
                s.push(Step::Row(row));
                return Ok(Some(s));
            }
            _ => return Err(unregistered()),
        }
        // An affine operation on the one argument that reads x.
        let reading: Vec<usize> = (0..args.len())
            .filter(|&k| self.reads(args[k], x))
            .collect();
        let &[k] = reading.as_slice() else {
            return Err(more_than_once());
        };
        let mut s = inner(self, args[k])?;
        let (a, b) = match s.last() {
            Some(Step::Affine(Affine { a, b })) => (a.clone(), b.clone()),
            _ => (lit(1.0), lit(0.0)),
        };
        if matches!(s.last(), Some(Step::Affine(_))) {
            s.pop();
        }
        let other = |me: &mut Self| me.coef(args[1 - k], at);
        let out = match (head, k) {
            ("neg", 0) => (sub(&lit(0.0), &a), sub(&lit(0.0), &b)),
            ("add", _) => {
                let c = other(self)?;
                (a, add(&b, &c))
            }
            ("sub", 0) => {
                let c = other(self)?;
                (a, sub(&b, &c))
            }
            ("sub", 1) => {
                let c = other(self)?;
                (sub(&lit(0.0), &a), sub(&c, &b))
            }
            ("mul", _) => {
                let c = other(self)?;
                (mul(&a, &c), mul(&b, &c))
            }
            ("divide", 0) => {
                let c = other(self)?;
                (div(&a, &c), div(&b, &c))
            }
            // c / g(x) is `pow(_, -1)` of g(x), then the scale c.
            ("divide", 1) => {
                let c = other(self)?;
                if a.num != Some(1.0) || b.num != Some(0.0) {
                    s.push(Step::Affine(Affine { a, b }));
                }
                s.push(Step::Pow(-1.0));
                (c, lit(0.0))
            }
            _ => return Err(unregistered()),
        };
        s.push(Step::Affine(Affine { a: out.0, b: out.1 }));
        Ok(Some(s))
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

    /// `true` when `id` reads the input `x`, through module bindings that
    /// are no draws.
    fn reads(&self, id: NodeId, x: Ref) -> bool {
        let mut stack = vec![id];
        while let Some(n) = stack.pop() {
            let n = self.resolve(n);
            if matches!(self.m.node(n), Node::Ref(r) if *r == x) {
                return true;
            }
            self.m.for_each_child(n, |k| stack.push(k));
        }
        false
    }

    /// M's support ends, from its law or its value set.
    fn base_support(&mut self, base: NodeId) -> Result<(Option<String>, Option<String>)> {
        Ok(match self.support(base)? {
            Some(ends) => ends,
            None => {
                let b = self
                    .m
                    .valueset_of(self.resolve(base))
                    .map(bounds_of)
                    .unwrap_or_default();
                (b.lower, b.upper)
            }
        })
    }

    /// M's support through the steps, innermost first: each step's §06
    /// domain must hold the support it receives. `None` when an affine
    /// scale of unknown sign leaves the ends unknown.
    #[allow(clippy::type_complexity)]
    fn image_of_support(
        &mut self,
        m: NodeId,
        map: &Bijection,
        base: NodeId,
    ) -> Result<Option<(Option<End>, Option<End>)>> {
        let (lo, hi) = self.base_support(base)?;
        let (mut lo, mut hi) = (lo.map(End::of), hi.map(End::of));
        let mut known = true;
        for s in &map.steps {
            let (dlo, dhi) = s.domain();
            if dlo.is_some() || dhi.is_some() {
                // An end of the base support on the boundary of the domain
                // passes (`log` of an Exponential at 0): a pushforward of a
                // law with no atom is defined up to a null set, and one end
                // point is null. A base with an atom there cannot reach this
                // path: a discrete base refuses in `pushfwd_statement`, and a
                // mixture with a `Dirac` is no `law_any` (tests
                // `pushfwd_of_a_*_base_refuses`).
                let symbolic = |e: &Option<End>| e.clone().filter(|e| e.num.is_none());
                let end = symbolic(&lo)
                    .filter(|_| dlo.is_some())
                    .or(symbolic(&hi).filter(|_| dhi.is_some()));
                if let Some(e) = end.filter(|_| known) {
                    return Err(Refusal::not_yet(
                        Some(m),
                        format!(
                            "pushfwd through `{}` of a law whose support end `{}` is not a literal, so its domain cannot be checked",
                            s.name(),
                            e.text
                        ),
                    ));
                }
                let num = |e: &Option<End>| e.as_ref().and_then(|e| e.num);
                let within = known
                    && dlo.is_none_or(|d| num(&lo).is_some_and(|v| v >= d))
                    && dhi.is_none_or(|d| num(&hi).is_some_and(|v| v <= d));
                if !within {
                    let text = match s {
                        // §06 exempts odd integer exponents from the domain
                        // rule; a negative power is still not monotone on a
                        // support that holds 0.
                        Step::Pow(k) if *k < 0.0 => "pushfwd through `pow` with a negative exponent of a law whose support contains 0, where it is not monotone".to_string(),
                        _ => format!(
                            "pushfwd through `{}` of a law whose support is not within its domain (§06)",
                            s.name()
                        ),
                    };
                    return Err(Refusal::not_yet(Some(m), text));
                }
            }
            if !known {
                continue;
            }
            match s.image(lo, hi) {
                Some((l, h)) => (lo, hi) = (l, h),
                None => {
                    known = false;
                    (lo, hi) = (None, None);
                }
            }
        }
        Ok(known.then_some((lo, hi)))
    }

    /// The statements of a scalar variate `lhs ~ pushfwd(f, M)`.
    pub(super) fn pushfwd_statement(
        &mut self,
        lhs: &str,
        m: NodeId,
        lhs_rank: usize,
    ) -> Result<Vec<String>> {
        let (map, base) = self.bijection(m)?;
        if lhs_rank != 0 {
            return Err(Refusal::not_yet(Some(m), "pushfwd of an array variate"));
        }
        // §06: a domain-restricted step needs M's support within its domain.
        let image = if map.affine_only() {
            None
        } else {
            match self.image_of_support(m, &map, base)? {
                Some(ends) => Some(ends),
                None => {
                    return Err(Refusal::not_yet(
                        Some(m),
                        "pushfwd of a bounded law through a scale of unknown sign",
                    ));
                }
            }
        };
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
        if law
            .support_type
            .is_some_and(crate::stan_type::square_support)
        {
            return Err(Refusal::not_yet(
                Some(m),
                "pushfwd of a law on a constrained matrix set",
            ));
        }
        // A conditional inverse (an odd `pow`) is parenthesized before `~`.
        let inverse = map.inverse(lhs).at(super::P_CMP);
        let y = if law.lhs("{y}") == "{y}" {
            inverse
        } else {
            law.lhs(&format!("({inverse})"))
        };
        // A data scale that is 0 makes the map no bijection: name it at load.
        for s in &map.steps {
            if let Step::Affine(a) = s
                && a.a.num.is_none()
            {
                let a = &a.a.ex.s;
                self.tdata.push(format!(
                    "if ({a} == 0) {{\n  reject(\"pushfwd of {lhs}: the scale {a} is 0, so the map is no bijection\");\n}}"
                ));
            }
        }
        // An unnormalized base prints `target += …_lpdf(…)` (`Law::statement`).
        let mut out = vec![law.statement(m, &y, &law.call_text(None))?];
        out.extend(map.jacobians(lhs));
        // Outside the image the inverse leaves its domain or the line gives
        // a finite value (`log(-0.1)`, `square(-0.1)`): wherever the variate's
        // declaration does not state an image end (data, a query input, a
        // point), that end is tested and gives -inf. A declared parameter
        // states it. An affine map has no test of its own: its base law
        // checks the preimage.
        if let Some((lo, hi)) = image {
            let decl = self
                .decls
                .get(lhs)
                .map(|t| t.bounds.clone())
                .unwrap_or_default();
            // An open end (a limit of the map) is outside the image itself.
            let mut parts = Vec::new();
            if let Some(e) = lo.filter(|e| decl.lower.as_ref() != Some(&e.text)) {
                parts.push(format!(
                    "{lhs} {} {}",
                    if e.open { "<=" } else { "<" },
                    e.text
                ));
            }
            if let Some(e) = hi.filter(|e| decl.upper.as_ref() != Some(&e.text)) {
                parts.push(format!(
                    "{lhs} {} {}",
                    if e.open { ">=" } else { ">" },
                    e.text
                ));
            }
            if let Some(test) = (!parts.is_empty()).then(|| parts.join(" || ")) {
                let body: String = out
                    .iter()
                    .flat_map(|l| l.lines())
                    .map(|l| format!("  {l}\n"))
                    .collect();
                out = vec![format!(
                    "if ({test}) {{\n  target += negative_infinity();\n}} else {{\n{body}}}"
                )];
            }
        }
        // An `unchecked_support` base (von Mises): its preimage is tested.
        self.unchecked_guard(&y, false, &law, out)
    }

    /// M's support mapped through f: the bounds of a `pushfwd` parameter.
    pub(super) fn pushfwd_support(
        &mut self,
        m: NodeId,
    ) -> Result<Option<(Option<String>, Option<String>)>> {
        let (map, base) = self.bijection(m)?;
        match self.image_of_support(m, &map, base)? {
            Some((None, None)) => Ok(None),
            Some((lo, hi)) => Ok(Some((lo.map(|e| e.text), hi.map(|e| e.text)))),
            None => Err(Refusal::not_yet(
                Some(m),
                "pushfwd of a bounded law through a scale of unknown sign",
            )),
        }
    }

    /// f⁻¹ of a derived prior field's Stan name: the value its draw reads.
    pub(super) fn derived_draw_value(&mut self, law: NodeId, field: &str) -> Result<String> {
        let (map, _) = self.bijection(law)?;
        Ok(map.inverse(field).at(P_ATOM))
    }

    /// A prior field `v = f(x)` of one draw `x` (§06: its law is
    /// `pushfwd(f, law(x))`): `value` is f(x). Returns x's binding and law.
    pub(super) fn derived_field(
        &mut self,
        name: &str,
        value: NodeId,
        at: NodeId,
    ) -> Result<(flatppl_core::Symbol, NodeId)> {
        let value = self.resolve(value);
        let mut draws: Vec<Ref> = Vec::new();
        let mut stack = vec![value];
        while let Some(n) = stack.pop() {
            let n = self.resolve(n);
            if let Node::Ref(r) = self.m.node(n)
                && r.ns == RefNs::SelfMod
                && let Some(b) = self.m.binding_by_name(r.name)
                && self.head(self.m.binding(b).rhs) == Some("draw")
            {
                if !draws.contains(r) {
                    draws.push(*r);
                }
                continue;
            }
            self.m.for_each_child(n, |k| stack.push(k));
        }
        let x = match draws.as_slice() {
            [x] => *x,
            [] => {
                return Err(Refusal::not_yet(
                    Some(at),
                    format!("the derived prior field `{name}`"),
                ));
            }
            _ => {
                return Err(Refusal::not_yet(
                    Some(at),
                    "a derived prior field of more than one draw",
                ));
            }
        };
        let draw = self.m.binding(self.m.binding_by_name(x.name).unwrap()).rhs;
        let base = self.call(draw).unwrap().args[0];
        self.derived.insert(value, (value, x, base));
        // The map must be a registry chain: refuse here, not at the statement.
        self.bijection(value)?;
        Ok((x.name, base))
    }
}

#[cfg(test)]
mod tests {
    /// Every registry row, `pow` (odd and other k, `sqrt`) and the affine
    /// step have a golden that pushes a law through them.
    #[test]
    fn every_step_has_a_golden() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden");
        let mut sources = String::new();
        for e in std::fs::read_dir(dir).unwrap() {
            let path = e.unwrap().path();
            if path.extension().is_some_and(|x| x == "flatppl") {
                sources.push_str(&std::fs::read_to_string(&path).unwrap());
            }
        }
        let mut needles: Vec<String> = super::STEPS
            .iter()
            .map(|r| format!("pushfwd(fn({}(_)", r.name))
            .collect();
        needles.extend(
            [
                "pushfwd(fn(sqrt(_)",
                "pushfwd(fn(pow(_, 3)",
                "pushfwd(fn(pow(_, -1)",
                "pushfwd(fn(pow(_, 1.5)",
                "pushfwd(fn(2 * _ - 1)",
            ]
            .map(String::from),
        );
        for n in needles {
            assert!(sources.contains(&n), "no golden for `{n}`");
        }
    }
}
