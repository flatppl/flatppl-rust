//! §06 Poisson processes. `BinnedPoissonProcess(bins, intensity)` is
//! independent Poisson counts with the intensity's bin masses (§06
//! `broadcast(Poisson, m)`). `PoissonProcess(intensity)` scores
//! `-totalmass(intensity) + Σ_i log λ(t_i)`: the variate is the set of
//! points, so there is no k! term. Both read the intensity through its log
//! mass on an interval and its log density at a point, over the closed
//! shapes: a normalized law with `_lcdf`, `weighted` and `logweighted` by a
//! value, `superpose`, `truncate` to an interval, `normalize`, `Lebesgue`.
//! Values are folded (`fold_add`, …) and carry their precedence.

use super::structure::{Folded, Law, fold_add, fold_div, fold_mul, fold_neg, fold_sub, number_ex};
use super::{Emitter, Ex, P_CMP, P_TERNARY};
use crate::refuse::{Refusal, RefusalKind, Result};
use flatppl_core::{Node, NodeId, Phase, Scalar, Type};

type F = Folded;

/// An interval end: `None` is open (±inf).
type End = Option<F>;

/// `||` and `&&` sit between the ternary and the comparisons.
const P_OR: u8 = 1;
const P_AND: u8 = 2;

const NEG_INF: &str = "negative_infinity()";

/// The counts of a binned process over log bin masses, as the function
/// `name_lpmf`. Stan's `poisson_log_lpmf(0 | negative_infinity())` is NaN, so
/// a bin of zero mass is scored here: 0 without a count, -inf with one.
fn binned_text(name: &str) -> String {
    format!(
        "real {name}_lpmf(array[] int n, vector lm) {{
  real lp = 0;
  for (k in 1:num_elements(n)) {{
    if (lm[k] == negative_infinity()) {{
      lp += n[k] == 0 ? 0.0 : negative_infinity();
    }} else {{
      lp += poisson_log_lpmf(n[k] | lm[k]);
    }}
  }}
  return lp;
}}"
    )
}

/// A bin edge as written.
enum Edge {
    NegInf,
    PosInf,
    Value { v: F, fixed: bool },
}

/// `-Λ + Σ log λ(t_i)`: one expression where every part vectorizes, else the
/// total mass and the per-event term of a loop.
enum Score {
    Whole(Ex),
    Loop { total: F, each: Ex },
}

fn call(f: &str, args: &[&Ex]) -> Ex {
    let a: Vec<&str> = args.iter().map(|x| x.s.as_str()).collect();
    Ex::atom(format!("{f}({})", a.join(", ")))
}

/// `a op b` at precedence `prec`, left-associative.
fn binary(a: &Ex, op: &str, b: &Ex, prec: u8) -> Ex {
    Ex {
        s: format!("{} {op} {}", a.at(prec), b.at(prec + 1)),
        prec,
    }
}

fn ternary(test: &Ex, a: &Ex, b: &Ex) -> Ex {
    Ex {
        s: format!(
            "{} ? {} : {}",
            test.at(P_TERNARY + 1),
            a.at(P_TERNARY + 1),
            b.at(P_TERNARY)
        ),
        prec: P_TERNARY,
    }
}

/// `log(x)`, folded for 1.
fn fold_log(x: &F) -> F {
    match x.0 {
        Some(1.0) => number_ex(0.0),
        _ => (None, call("log", &[&x.1]), false),
    }
}

/// `exp(x)`, folded for 0.
fn fold_exp(x: &F) -> F {
    match x.0 {
        Some(0.0) => number_ex(1.0),
        _ => (None, call("exp", &[&x.1]), false),
    }
}

fn log_sum_exp(parts: &[F]) -> F {
    match parts {
        [one] => one.clone(),
        [a, b] => (None, call("log_sum_exp", &[&a.1, &b.1]), false),
        _ => {
            let all: Vec<&str> = parts.iter().map(|p| p.1.s.as_str()).collect();
            (
                None,
                Ex::atom(format!("log_sum_exp([{}]')", all.join(", "))),
                false,
            )
        }
    }
}

/// The larger (`max`) or smaller end; an open end gives way.
fn pick(x: &End, y: &End, max: bool) -> End {
    match (x, y) {
        (None, e) | (e, None) => e.clone(),
        (Some(p), Some(q)) => Some(match (p.0, q.0) {
            (Some(u), Some(v)) => {
                if (u >= v) == max {
                    p.clone()
                } else {
                    q.clone()
                }
            }
            _ => (
                None,
                call(if max { "fmax" } else { "fmin" }, &[&p.1, &q.1]),
                false,
            ),
        }),
    }
}

/// The overlap `[lo, hi] ∩ [a, b]`: its ends, and the test that it is not
/// empty where that is known only at run time.
type Overlap = (End, End, Option<Ex>);

/// `[lo, hi] ∩ [a, b]`; `None` when a numeric end on each side makes it
/// empty for every value. An overlap that may be empty at run time carries
/// its test: its mass is taken only when the test holds, because the partials
/// of `log_diff_exp(x, x)` are NaN, and Stan's reverse pass multiplies every
/// partial, used or not.
fn intersect(lo: &End, hi: &End, a: &End, b: &End) -> Option<Overlap> {
    let numeric = |e: &End| e.as_ref().and_then(|v| v.0);
    let low = [lo, a].into_iter().filter_map(numeric).reduce(f64::max);
    let high = [hi, b].into_iter().filter_map(numeric).reduce(f64::min);
    if let (Some(p), Some(q)) = (low, high)
        && q <= p
    {
        return None;
    }
    let (l, h) = (pick(lo, a, true), pick(hi, b, false));
    let test = match (&l, &h) {
        (Some(x), Some(y)) if x.0.is_none() || y.0.is_none() => {
            Some(binary(&x.1, "<", &y.1, P_CMP))
        }
        _ => None,
    };
    Some((l, h, test))
}

/// `value` where the overlap `test` holds, else `otherwise`.
fn when_nonempty(test: Option<Ex>, value: F, otherwise: Ex) -> F {
    match test {
        Some(t) => (None, ternary(&t, &value.1, &otherwise), false),
        None => value,
    }
}

/// `inner`, and -inf where `y` (each element of an array `y`) lies outside
/// `[a, b]`.
fn guarded(y: &Ex, array: bool, a: &End, b: &End, inner: F) -> F {
    let (small, large) = if array {
        (call("min", &[y]), call("max", &[y]))
    } else {
        (y.clone(), y.clone())
    };
    let mut parts: Vec<Ex> = Vec::new();
    if let Some(a) = a {
        parts.push(binary(&small, "<", &a.1, P_CMP));
    }
    if let Some(b) = b {
        parts.push(binary(&large, ">", &b.1, P_CMP));
    }
    let Some(first) = parts.first().cloned() else {
        return inner;
    };
    let out = parts[1..]
        .iter()
        .fold(first, |acc, p| binary(&acc, "||", p, P_OR));
    (None, ternary(&out, &Ex::atom(NEG_INF), &inner.1), false)
}

fn infinite(at: NodeId) -> Refusal {
    Refusal::new(
        RefusalKind::IllFormedModule,
        Some(at),
        "an intensity of infinite mass",
        "spec §06 takes a Poisson process intensity of finite mass, and `Lebesgue` over an \
         unbounded set has none",
    )
}

/// `normalize` of a measure of zero mass: §06 divides by a total mass that
/// must be finite and nonzero.
pub(super) fn zero_mass(at: Option<NodeId>, what: &str) -> Refusal {
    Refusal::new(
        RefusalKind::IllFormedModule,
        at,
        "`normalize` of a measure of zero mass",
        format!(
            "spec §06 takes `normalize` of a measure whose total mass is finite and nonzero, \
             and {what} has zero mass"
        ),
    )
}

impl<'m> Emitter<'m> {
    /// `true` for a `PoissonProcess` or `BinnedPoissonProcess` law.
    pub(super) fn is_process(&self, m: NodeId) -> bool {
        matches!(
            self.head(self.resolve(m)),
            Some("PoissonProcess" | "BinnedPoissonProcess")
        )
    }

    /// The statements for the observed variate `lhs ~ m`.
    pub(super) fn process_statements(&mut self, lhs: &str, m: NodeId) -> Result<Vec<String>> {
        let r = self.resolve(m);
        if !self.is_data(lhs) {
            return Err(Refusal::not_yet(
                Some(r),
                "a Poisson process variate that is not observed",
            ));
        }
        if self.head(r) == Some("BinnedPoissonProcess") {
            let lm = self.bin_lmasses(r)?;
            let f = self.binned_function();
            return Ok(vec![format!("{lhs} ~ {f}({lm});")]);
        }
        Ok(match self.events_score(r, &Ex::atom(lhs))? {
            Score::Whole(s) => vec![format!("target += {};", s.s)],
            Score::Loop { total, each } => {
                let i = self.fresh_index();
                let lines = vec![
                    format!("target += {};", fold_neg(&total).1.s),
                    format!(
                        "for ({i} in 1:num_elements({lhs})) {{\n  target += {};\n}}",
                        each.s
                    )
                    .replace('\u{2}', &i),
                ];
                self.used.remove(&i);
                lines
            }
        })
    }

    /// `logdensityof(m, x)` as an expression.
    pub(super) fn process_density(&mut self, id: NodeId, m: NodeId, x: NodeId) -> Result<String> {
        let r = self.resolve(m);
        let fixed = self.m.phase_of(id) == Some(Phase::Fixed);
        if self.head(r) == Some("BinnedPoissonProcess") {
            let lm = self.bin_lmasses(r)?;
            let n = self.expr(x)?;
            let f = self.binned_function();
            return Ok(format!("{f}_lpmf({} | {lm})", n.s));
        }
        // The intensity decides first: record-valued points refuse before
        // their table is printed.
        if let Some(mu) = self.process_arg(r, 0, "intensity") {
            self.check_intensity(r, mu)?;
        }
        let e = self.real(x)?;
        let t = if super::structure::is_identifier(&e.s) {
            e.s
        } else {
            let name = self.fresh("events");
            let n = format!("num_elements({})", e.s);
            self.put(fixed, vec![format!("vector[{n}] {name} = {};", e.s)]);
            name
        };
        Ok(match self.events_score(r, &Ex::atom(&t))? {
            // The caller reads the result as one atom.
            Score::Whole(s) => s.at(super::P_ATOM),
            Score::Loop { total, each } => {
                let name = self.fresh("process_lp");
                let i = self.fresh_index();
                let each = each.s.replace('\u{2}', &i);
                self.put(
                    fixed,
                    vec![
                        format!("real {name} = {};", fold_neg(&total).1.s),
                        format!("for ({i} in 1:num_elements({t})) {{\n  {name} += {each};\n}}"),
                    ],
                );
                self.used.remove(&i);
                name
            }
        })
    }

    /// The argument at `k`, or the named one.
    fn process_arg(&self, r: NodeId, k: usize, name: &str) -> Option<NodeId> {
        let c = self.call(r)?;
        c.args.get(k).copied().or_else(|| {
            c.named
                .iter()
                .find(|a| self.name(a.name) == name)
                .map(|a| a.value)
        })
    }

    /// A kernel intensity and record-valued points have no form here.
    fn check_intensity(&self, at: NodeId, mu: NodeId) -> Result<()> {
        match self.ty(mu) {
            Some(Type::Kernel { .. }) => Err(Refusal::not_yet(Some(at), "a kernel intensity")),
            Some(Type::Measure { domain, .. }) if matches!(*domain, Type::Record(_)) => Err(
                Refusal::not_yet(Some(at), "a Poisson process over record-valued points"),
            ),
            _ => Ok(()),
        }
    }

    /// The PoissonProcess score over the events `t`. The loop index of a
    /// per-event term is `\u{2}`, filled by the caller.
    fn events_score(&mut self, r: NodeId, t: &Ex) -> Result<Score> {
        let mu = self
            .process_arg(r, 0, "intensity")
            .ok_or_else(|| Refusal::not_yet(Some(r), "this `PoissonProcess` call"))?;
        self.check_intensity(r, mu)?;
        let total = self.mass_on(mu, &None, &None)?;
        let n = (None, call("num_elements", &[t]), true);
        if let Some(sum) = self.events_lpdf(mu, t, &n)? {
            return Ok(Score::Whole(fold_sub(&sum, &total).1));
        }
        let each = self.point_lpdf(mu, &Ex::atom(format!("{}[\u{2}]", t.s)))?;
        Ok(Score::Loop {
            total,
            each: each.1,
        })
    }

    /// The log bin masses of a `BinnedPoissonProcess`, as one vector.
    fn bin_lmasses(&mut self, r: NodeId) -> Result<String> {
        let (Some(bins), Some(mu)) = (
            self.process_arg(r, 0, "bins"),
            self.process_arg(r, 1, "intensity"),
        ) else {
            return Err(Refusal::not_yet(
                Some(r),
                "this `BinnedPoissonProcess` call",
            ));
        };
        let edges = self.bin_edges(r, bins)?;
        self.check_intensity(r, mu)?;
        // §06 takes a finite-mass intensity here too.
        self.mass_on(mu, &None, &None)?;
        let mut parts = Vec::with_capacity(edges.len());
        for w in edges.windows(2) {
            parts.push(match self.lmass(mu, &w[0], &w[1])? {
                Some(m) => m.1.s,
                None => NEG_INF.to_string(),
            });
        }
        Ok(format!("[{}]'", parts.join(", ")))
    }

    /// The binned counts function, emitted once under a name that no module
    /// binding, record field or user function takes, with its `_lpmf` suffix.
    fn binned_function(&mut self) -> String {
        if let Some(name) = &self.binned_fn {
            return name.clone();
        }
        let base = "flatppl_binned_poisson_log";
        let taken = |me: &Self, n: &str| me.used.contains(n) || me.reserved.contains(n);
        let mut name = base.to_string();
        let mut k = 2;
        while taken(self, &name) || taken(self, &format!("{name}_lpmf")) {
            name = format!("{base}_{k}");
            k += 1;
        }
        self.used.insert(name.clone());
        self.used.insert(format!("{name}_lpmf"));
        self.functions.push(binned_text(&name));
        self.binned_fn = Some(name.clone());
        name
    }

    /// The bin edges, a static count of them: open at a literal ±inf end.
    /// Inference refuses written edges that do not increase (§07
    /// `bincounts`); edges read from inputs, data or parameters are tested
    /// where they live.
    fn bin_edges(&mut self, at: NodeId, bins: NodeId) -> Result<Vec<End>> {
        let r = self.resolve(bins);
        if matches!(self.ty(bins).or(self.ty(r)), Some(Type::Record(_))) {
            return Err(Refusal::not_yet(
                Some(at),
                "a `BinnedPoissonProcess` with a record of edge vectors",
            ));
        }
        let dynamic = || Refusal::not_yet(Some(at), "a bin-edge vector of non-static length");
        let args = self.call(r).map(|c| c.args.to_vec()).unwrap_or_default();
        let mut edges = Vec::new();
        match self.head(r) {
            Some("vector") => {
                for a in args {
                    edges.push(self.edge(a)?);
                }
            }
            Some(h @ ("linspace" | "extlinspace")) if args.len() == 3 => {
                let n = match self.m.node(self.resolve(args[2])) {
                    Node::Lit(Scalar::Int(n)) if *n >= 2 => *n,
                    _ => return Err(dynamic()),
                };
                let fixed = [args[0], args[1]]
                    .iter()
                    .all(|&e| self.m.phase_of(e) == Some(Phase::Fixed));
                let (a, b) = (self.folded(args[0])?, self.folded(args[1])?);
                if h == "extlinspace" {
                    edges.push(Edge::NegInf);
                }
                let step = fold_div(&fold_sub(&b, &a), &number_ex((n - 1) as f64));
                for k in 0..n {
                    let v = if k == n - 1 {
                        b.clone()
                    } else {
                        fold_add(&a, &fold_mul(&number_ex(k as f64), &step))
                    };
                    edges.push(Edge::Value { v, fixed });
                }
                if h == "extlinspace" {
                    edges.push(Edge::PosInf);
                }
            }
            _ => {
                let len = match self.ty(bins) {
                    Some(Type::Array { shape, .. }) => match shape.as_ref() {
                        [flatppl_core::Dim::Static(n)] => *n as usize,
                        _ => return Err(dynamic()),
                    },
                    _ => return Err(dynamic()),
                };
                let e = self.expr(bins)?;
                if !super::structure::is_identifier(&e.s) {
                    return Err(dynamic());
                }
                let fixed = self.m.phase_of(bins) == Some(Phase::Fixed);
                let int = self.shape(bins).is_none_or(|s| s.int);
                for k in 1..=len {
                    edges.push(Edge::Value {
                        v: (None, Ex::atom(format!("{}[{k}]", e.s)), int),
                        fixed,
                    });
                }
            }
        }
        let mut tests: Option<Ex> = None;
        let mut fixed_tests = true;
        for pair in edges.windows(2) {
            if let [
                Edge::Value { v: a, fixed: fa },
                Edge::Value { v: b, fixed: fb },
            ] = pair
                && (a.0.is_none() || b.0.is_none())
            {
                let t = binary(&a.1, "<", &b.1, P_CMP);
                tests = Some(match tests {
                    Some(acc) => binary(&acc, "&&", &t, P_AND),
                    None => t,
                });
                fixed_tests &= *fa && *fb;
            }
        }
        if let Some(t) = tests {
            self.put(
                fixed_tests,
                vec![format!(
                    "if (!({})) {{\n  reject(\"BinnedPoissonProcess: bin edges must be strictly increasing \
                     (bincounts)\");\n}}",
                    t.s
                )],
            );
        }
        Ok(edges
            .into_iter()
            .map(|e| match e {
                Edge::Value { v, .. } => Some(v),
                _ => None,
            })
            .collect())
    }

    /// `true` for the literal `inf`.
    fn is_inf(&self, x: NodeId) -> bool {
        matches!(self.m.node(self.resolve(x)), Node::Const(c) if self.name(*c) == "inf")
    }

    /// An end of a written interval: open at a literal ±inf.
    fn written_end(&mut self, a: NodeId) -> Result<End> {
        let r = self.resolve(a);
        if self.is_inf(r)
            || (self.head(r) == Some("neg") && self.is_inf(self.call(r).unwrap().args[0]))
        {
            return Ok(None);
        }
        Ok(Some(self.folded(a)?))
    }

    /// One bin edge of a literal edge vector.
    fn edge(&mut self, a: NodeId) -> Result<Edge> {
        let r = self.resolve(a);
        if self.is_inf(r) {
            return Ok(Edge::PosInf);
        }
        if self.head(r) == Some("neg") && self.is_inf(self.call(r).unwrap().args[0]) {
            return Ok(Edge::NegInf);
        }
        Ok(Edge::Value {
            v: self.folded(a)?,
            fixed: self.m.phase_of(a) == Some(Phase::Fixed),
        })
    }

    /// The ends of a support set (`interval(a, b)`, `reals`, `posreals`, …).
    fn set_ends(&mut self, s: NodeId) -> Result<(End, End)> {
        let r = self.resolve(s);
        if self.head(r) == Some("interval") {
            let args = self.call(r).unwrap().args.to_vec();
            if let [a, b] = args[..] {
                return Ok((self.written_end(a)?, self.written_end(b)?));
            }
        }
        if let Node::Const(c) = self.m.node(r) {
            match self.name(*c) {
                "reals" => return Ok((None, None)),
                "posreals" | "nonnegreals" => return Ok((Some(number_ex(0.0)), None)),
                "unitinterval" => return Ok((Some(number_ex(0.0)), Some(number_ex(1.0)))),
                _ => {}
            }
        }
        Err(Refusal::not_yet(
            Some(r),
            "an intensity support of this set",
        ))
    }

    /// The value a `weighted` or `logweighted` intensity carries. A weight
    /// that is a function of the variate has no closed mass here; a negative
    /// literal weight is ill-formed (§06).
    fn intensity_weight(&mut self, r: NodeId, args: &[NodeId]) -> Result<F> {
        let w = args[0];
        if args.len() != 2
            || matches!(
                self.ty(w),
                Some(Type::Function { .. } | Type::Kernel { .. })
            )
        {
            return Err(Refusal::not_yet(
                Some(r),
                "an intensity weighted by a function of the variate",
            ));
        }
        if self.head(r) == Some("weighted") {
            self.nonnegative_weight(w)?;
        }
        self.folded(w)
    }

    /// A leaf intensity: a normalized law of one continuous variate.
    fn intensity_leaf(&mut self, r: NodeId) -> Result<Law> {
        if self.head(r) == Some("Dirac") {
            return Err(Refusal::not_yet(
                Some(r),
                "an intensity with an atom (`Dirac`)",
            ));
        }
        let law = self.law(r, false)?;
        if law.discrete {
            return Err(Refusal::not_yet(
                Some(r),
                format!("a discrete intensity (`{}`)", law.flatppl),
            ));
        }
        if law.multivariate
            || law.trunc.is_some()
            || law.support.is_some()
            || law.declared
            || law.unnormalized
        {
            return Err(Refusal::not_yet(
                Some(r),
                format!("the intensity law `{}` in this form", law.flatppl),
            ));
        }
        Ok(law)
    }

    fn intensity_args(&self, r: NodeId) -> Vec<NodeId> {
        self.call(r).map(|c| c.args.to_vec()).unwrap_or_default()
    }

    /// The ends of a `Lebesgue` intensity's support.
    fn lebesgue_ends(&mut self, r: NodeId) -> Result<(End, End)> {
        let s = self
            .process_arg(r, 0, "support")
            .ok_or_else(|| Refusal::not_yet(Some(r), "this `Lebesgue` intensity"))?;
        self.set_ends(s)
    }

    /// The whole log mass of a normalized measure `m`, held in one local when
    /// it is not a literal, so every bin and every event reads it once.
    fn whole_lmass(&mut self, m: NodeId) -> Result<F> {
        let r = self.resolve(m);
        if let Some(name) = self.whole_lmasses.get(&r) {
            return Ok((None, Ex::atom(name.clone()), false));
        }
        let whole = self
            .lmass(m, &None, &None)?
            .ok_or_else(|| zero_mass(Some(r), "a measure empty on its whole support"))?;
        if whole.0.is_some() {
            return Ok(whole);
        }
        let name = self.fresh_local("lmass");
        let fixed = self.m.phase_of(r) == Some(Phase::Fixed);
        // `normalize` of an intensity whose mass is zero at run time rejects.
        let value = self.nonzero_normalizer(whole.1.s);
        self.put(fixed, vec![format!("real {name} = {value};")]);
        self.whole_lmasses.insert(r, name.clone());
        Ok((None, Ex::atom(name), false))
    }

    /// The log mass of the intensity on `[lo, hi]`; `None` when it is empty
    /// for every value.
    fn lmass(&mut self, mu: NodeId, lo: &End, hi: &End) -> Result<Option<F>> {
        let r = self.resolve(mu);
        let args = self.intensity_args(r);
        Ok(match self.head(r) {
            Some("superpose") => {
                let mut parts = Vec::new();
                for a in args {
                    parts.extend(self.lmass(a, lo, hi)?);
                }
                (!parts.is_empty()).then(|| log_sum_exp(&parts))
            }
            Some(h @ ("weighted" | "logweighted")) => {
                let w = self.intensity_weight(r, &args)?;
                let w = if h == "weighted" { fold_log(&w) } else { w };
                self.lmass(args[1], lo, hi)?
                    .map(|inner| fold_add(&w, &inner))
            }
            Some("normalize") => match self.lmass(args[0], lo, hi)? {
                Some(inner) => Some(fold_sub(&inner, &self.whole_lmass(args[0])?)),
                None => None,
            },
            Some("truncate") => {
                let (a, b) = self.set_ends(args[1])?;
                match intersect(lo, hi, &a, &b) {
                    None => None,
                    Some((l, h, test)) => self
                        .lmass(args[0], &l, &h)?
                        .map(|inner| when_nonempty(test, inner, Ex::atom(NEG_INF))),
                }
            }
            Some("Lebesgue") => {
                let (a, b) = self.lebesgue_ends(r)?;
                match intersect(lo, hi, &a, &b) {
                    None => None,
                    Some((Some(l), Some(h), test)) => Some(when_nonempty(
                        test,
                        fold_log(&fold_sub(&h, &l)),
                        Ex::atom(NEG_INF),
                    )),
                    Some(_) => return Err(infinite(r)),
                }
            }
            _ => {
                let law = self.intensity_leaf(r)?;
                if !law.lcdf {
                    return Err(Refusal::not_yet(
                        Some(r),
                        format!(
                            "the intensity law `{}`, which has no `_lcdf` in Stan",
                            law.flatppl
                        ),
                    ));
                }
                let end = |e: &End| e.as_ref().map(|v| v.1.s.clone());
                let (l, h) = (end(lo), end(hi));
                let text = self.interval_lmass(&law, l.as_deref(), h.as_deref());
                match text.as_str() {
                    NEG_INF => None,
                    "0" => Some(number_ex(0.0)),
                    // The interval mass prints one call or a parenthesized test.
                    _ => Some((None, Ex::atom(text), false)),
                }
            }
        })
    }

    /// The mass of the intensity on `[lo, hi]`, linear where it is a closed
    /// linear form (a `Lebesgue` width, a weight, a sum); a law's part is
    /// `exp` of its log mass.
    fn mass_on(&mut self, mu: NodeId, lo: &End, hi: &End) -> Result<F> {
        let r = self.resolve(mu);
        let args = self.intensity_args(r);
        let zero = number_ex(0.0);
        Ok(match self.head(r) {
            Some("superpose") => {
                let mut total = zero;
                for a in args {
                    total = fold_add(&total, &self.mass_on(a, lo, hi)?);
                }
                total
            }
            Some(h @ ("weighted" | "logweighted")) => {
                let w = self.intensity_weight(r, &args)?;
                let w = if h == "weighted" { w } else { fold_exp(&w) };
                fold_mul(&w, &self.mass_on(args[1], lo, hi)?)
            }
            Some("truncate") => {
                let (a, b) = self.set_ends(args[1])?;
                match intersect(lo, hi, &a, &b) {
                    None => zero,
                    Some((l, h, test)) => {
                        let inner = self.mass_on(args[0], &l, &h)?;
                        when_nonempty(test, inner, zero.1)
                    }
                }
            }
            Some("Lebesgue") => {
                let (a, b) = self.lebesgue_ends(r)?;
                match intersect(lo, hi, &a, &b) {
                    None => zero,
                    Some((Some(l), Some(h), test)) => when_nonempty(test, fold_sub(&h, &l), zero.1),
                    Some(_) => return Err(infinite(r)),
                }
            }
            _ if lo.is_none() && hi.is_none() => {
                // A normalized measure: its mass is 1 once its own is finite.
                match self.head(r) {
                    Some("normalize") => {
                        self.whole_lmass(args[0])?;
                    }
                    _ => {
                        self.intensity_leaf(r)?;
                    }
                }
                number_ex(1.0)
            }
            _ => match self.lmass(r, lo, hi)? {
                Some(l) => fold_exp(&l),
                None => zero,
            },
        })
    }

    /// The log density of the intensity at one point `y`.
    fn point_lpdf(&mut self, mu: NodeId, y: &Ex) -> Result<F> {
        let r = self.resolve(mu);
        let args = self.intensity_args(r);
        Ok(match self.head(r) {
            Some("superpose") => {
                let mut parts = Vec::new();
                for a in args {
                    parts.push(self.point_lpdf(a, y)?);
                }
                log_sum_exp(&parts)
            }
            Some(h @ ("weighted" | "logweighted")) => {
                let w = self.intensity_weight(r, &args)?;
                let w = if h == "weighted" { fold_log(&w) } else { w };
                fold_add(&w, &self.point_lpdf(args[1], y)?)
            }
            Some("normalize") => {
                let inner = self.point_lpdf(args[0], y)?;
                fold_sub(&inner, &self.whole_lmass(args[0])?)
            }
            Some("truncate") => {
                let (a, b) = self.set_ends(args[1])?;
                let inner = self.point_lpdf(args[0], y)?;
                guarded(y, false, &a, &b, inner)
            }
            Some("Lebesgue") => {
                let (a, b) = self.lebesgue_ends(r)?;
                guarded(y, false, &a, &b, number_ex(0.0))
            }
            _ => {
                let law = self.intensity_leaf(r)?;
                (
                    None,
                    Ex::atom(self.law_density(r, &law, &y.s, None)?),
                    false,
                )
            }
        })
    }

    /// The sum of the intensity's log density over the events `t` (`n` of
    /// them) as one expression; `None` where a part does not vectorize (a
    /// `superpose`: Stan has no elementwise `log_sum_exp` of two vectors).
    fn events_lpdf(&mut self, mu: NodeId, t: &Ex, n: &F) -> Result<Option<F>> {
        let r = self.resolve(mu);
        let args = self.intensity_args(r);
        Ok(Some(match self.head(r) {
            Some("superpose") => return Ok(None),
            Some(h @ ("weighted" | "logweighted")) => {
                let w = self.intensity_weight(r, &args)?;
                let Some(inner) = self.events_lpdf(args[1], t, n)? else {
                    return Ok(None);
                };
                let w = if h == "weighted" { fold_log(&w) } else { w };
                fold_add(&fold_mul(n, &w), &inner)
            }
            Some("normalize") => {
                let Some(inner) = self.events_lpdf(args[0], t, n)? else {
                    return Ok(None);
                };
                let whole = self.whole_lmass(args[0])?;
                fold_sub(&inner, &fold_mul(n, &whole))
            }
            Some("truncate") => {
                let (a, b) = self.set_ends(args[1])?;
                let Some(inner) = self.events_lpdf(args[0], t, n)? else {
                    return Ok(None);
                };
                guarded(t, true, &a, &b, inner)
            }
            Some("Lebesgue") => {
                let (a, b) = self.lebesgue_ends(r)?;
                guarded(t, true, &a, &b, number_ex(0.0))
            }
            _ => {
                let law = self.intensity_leaf(r)?;
                if !law.vectorized {
                    return Ok(None);
                }
                (
                    None,
                    Ex::atom(self.law_density(r, &law, &t.s, None)?),
                    false,
                )
            }
        }))
    }
}
