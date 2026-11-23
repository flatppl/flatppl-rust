//! `T[lo, hi]` with both ends finite, as the law's statement minus its log
//! interval mass from a helper (ruling of 2026-11-18). Stan's `T[lo, hi]`
//! takes log(F(hi) − F(lo)) from the `lcdf` pair, which rounds to 0 − 0 far
//! in the upper tail. The helper switches to the survival pair there
//! (`tail: lccdf` rows). The normal law (`tail: normal_lphi`) takes its
//! interval mass, one-sided too, from its own log Φ: Stan's `normal_lccdf`
//! loses digits from 4.5 sd and is -inf from 8.3 sd, and the derivative of
//! `normal_lcdf` is an approximation (BridgeStan against mpmath, Stan 2.39).
//! A normal statement keeps `T[lo, hi]` only where Stan's value and gradient
//! were measured close to exact, decided at emit time: a constant mass (every
//! end the mean, or a literal lower end with z <= 4 or upper end with z >= -4
//! under literal mu and sigma; value within 3.1e-14 relative, no partial), or
//! `T[L, ]` with a literal L below a literal mu for any sigma (value and
//! gradient within 4e-16 of mpmath).
//! The flag is set only for laws whose far-tail functions were checked in
//! BridgeStan. A law whose support has an end keeps Stan's `T[lo, hi]` only
//! where each written end lies strictly inside the support at emit time;
//! otherwise it takes the support-end rule of [`Emitter::bounded_lmass`].
//! Every other law keeps `T[lo, hi]`.

use super::mixture::split_args;
use super::structure::{Folded, LOOP, Law, fold_add, fold_div, fold_sub, number_ex};
use super::{Emitter, Ex};
use crate::refuse::Result;

impl<'m> Emitter<'m> {
    /// The statements for a truncated `lhs ~ law`: the statement minus the
    /// log interval mass, for a two-sided truncation of a `tail` row, for a
    /// law whose support has an end that a written end is not decided, at emit
    /// time, to lie strictly inside, and for a normal statement the keep rule
    /// does not cover (below). `None` keeps Stan's `T[lo, hi]`. A value
    /// outside the window is -inf: data always, a parameter at every end its
    /// declaration does not carry.
    pub(super) fn interval_mass(
        &mut self,
        lhs: &str,
        law: &Law,
        lhs_rank: usize,
        per_element: bool,
    ) -> Result<Option<Vec<String>>> {
        let Some((lo, hi)) = &law.trunc else {
            return Ok(None);
        };
        // The mass is shared by every element only with shared arguments.
        // A discrete law's ends are integer-typed here (`normalize` refuses others).
        if law.multivariate
            || per_element
            || lhs_rank > 1
            || law.lhs("{y}") != "{y}"
            || law.call.contains(LOOP)
        {
            return Ok(None);
        }
        let ends = law.support_ends();
        let inside = |x: &Option<String>| {
            x.as_deref().is_none_or(|x| {
                (ends.0.is_none() || decided(Some(x), &ends.0, |a, b| a > b, false))
                    && (ends.1.is_none() || decided(Some(x), &ends.1, |a, b| a < b, false))
            })
        };
        let keep = if law.tail == "normal_lphi" {
            // Stan's normal CDFs lose the far tail beyond an end (and their
            // derivative is an approximation): `T[]` stays for an end decided
            // at emit time to be the mean, a lower end with z <= 4, or an
            // upper end with z >= -4.
            let (_, args) = stan_call(law);
            let (mu, sigma) = match &args[..] {
                [m, s] => (m.clone(), s.clone()),
                _ => ("0".to_string(), "1".to_string()),
            };
            // `T[]` stays where Stan's value and gradient were measured close to
            // exact, decided at emit time. A constant mass: literal mu, sigma and
            // end with the end on the mean's side of the far tail, or an end
            // equal to mu (value within 3.1e-14 relative, no partial). A mass
            // with a partial in sigma: only `T[L, ]` with a literal L below a
            // literal mu, which takes `normal_lccdf` alone; its derivative
            // matched mpmath to 4e-16, while `normal_lcdf`'s was off by up to
            // 1.3e-6 in the same shape (BridgeStan, Stan 2.39).
            let one_sided_lower = hi.is_none();
            let near = |x: &Option<String>, lower: bool| {
                x.as_deref().is_none_or(|x| {
                    let sided = |z: f64, bound: f64| if lower { z <= bound } else { z >= -bound };
                    x == mu
                        || match (number(x), number(&mu), number(&sigma)) {
                            (Some(e), Some(m), Some(s)) => sided((e - m) / s, 4.0),
                            (Some(e), Some(m), None) => {
                                e == m || (lower && one_sided_lower && e < m)
                            }
                            _ => false,
                        }
                })
            };
            near(lo, true) && near(hi, false)
        } else {
            let two_sided_tail = !law.tail.is_empty() && lo.is_some() && hi.is_some();
            !two_sided_tail && inside(lo) && inside(hi)
        };
        if keep {
            return Ok(None);
        }
        let mass = self.interval_lmass(law, lo.as_deref(), hi.as_deref());
        let count = if lhs_rank == 0 {
            String::new()
        } else {
            format!("{} * ", self.extent_text_of(lhs))
        };
        let statement = format!("{lhs} ~ {};", law.call);
        let lines = match mass.as_str() {
            // The whole support: the truncation changes nothing.
            "0" => vec![statement],
            "negative_infinity()" => {
                return Err(super::intensity::zero_mass(
                    None,
                    "a truncation to an interval outside the support",
                ));
            }
            _ => {
                let mass = self.nonzero_normalizer(mass);
                vec![statement, format!("target += -{count}{mass};")]
            }
        };
        // A value outside [lo, hi] and the support has zero density; inside,
        // the mass is finite. An open end tests the support's own end there.
        let (tlo, thi) = law.within(lo.as_deref(), hi.as_deref());
        let (mut tlo, mut thi) = (tlo.or(ends.0), thi.or(ends.1));
        if !self.is_data(lhs) {
            // A parameter's declared bounds carry the support and literal ends;
            // every other end is tested here. The caller learns that the form
            // relies on the declaration (`relies_on_bounds`).
            self.relies_on_bounds = true;
            if self.point_statement {
                return Ok(Some(lines));
            }
            let bounds = self.decls.get(lhs).map(|t| t.bounds.clone());
            // A declared bound at or inside the end makes its test redundant.
            let declared =
                |end: &Option<String>, bound: Option<&String>, lower: bool| match (end, bound) {
                    (Some(e), Some(b)) => {
                        e == b
                            || match (number(e), number(b)) {
                                (Some(x), Some(y)) => (lower && y >= x) || (!lower && y <= x),
                                _ => false,
                            }
                    }
                    _ => false,
                };
            if declared(&tlo, bounds.as_ref().and_then(|b| b.lower.as_ref()), true) {
                tlo = None;
            }
            if declared(&thi, bounds.as_ref().and_then(|b| b.upper.as_ref()), false) {
                thi = None;
            }
            if tlo.is_none() && thi.is_none() {
                return Ok(Some(lines));
            }
        }
        Ok(self
            .guarded(lhs, tlo.as_deref(), thi.as_deref(), lines)
            .map(|g| vec![g]))
    }

    /// A normalizer's log mass, through the zero-mass check where it can be
    /// -inf at run time: `normalize` of a measure of zero mass is ill-formed
    /// (§06), so it rejects there, never giving +inf. That is a helper call
    /// with an end that is not a literal, or a run-time empty-window test.
    pub(super) fn nonzero_normalizer(&mut self, mass: String) -> String {
        if !self.can_vanish(&mass) {
            return mass;
        }
        let f = self.nonzero_function();
        format!("{f}({mass})")
    }

    fn can_vanish(&self, mass: &str) -> bool {
        if mass.contains("negative_infinity()") {
            return true;
        }
        let mut names = vec!["_interval_lmass(".to_string()];
        if let Some(n) = &self.normal_lmass_fn {
            names.push(format!("{n}("));
        }
        names.iter().any(|n| {
            mass.match_indices(n.as_str()).any(|(i, _)| {
                let rest = &mass[i + n.len()..];
                let mut depth = 1;
                let mut end = rest.len();
                for (j, c) in rest.char_indices() {
                    match c {
                        '(' | '[' => depth += 1,
                        ')' | ']' => {
                            depth -= 1;
                            if depth == 0 {
                                end = j;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                split_args(&rest[..end])
                    .iter()
                    .take(2)
                    .any(|a| number(a).is_none())
            })
        })
    }

    /// The function that passes a normalizer's log mass through and rejects
    /// one of zero mass, emitted on first use under a name that no module
    /// binding, record field or user function takes.
    fn nonzero_function(&mut self) -> String {
        if let Some(name) = &self.nonzero_fn {
            return name.clone();
        }
        let name = self.fresh_local("flatppl_nonzero_lmass");
        self.functions.push(format!(
            "real {name}(real lmass) {{\n  \
             if (lmass == negative_infinity()) {{\n    \
             reject(\"normalize of a measure of zero mass: spec section 06 takes normalize of a measure whose total mass is finite and nonzero\");\n  \
             }}\n  \
             return lmass;\n}}"
        ));
        self.nonzero_fn = Some(name.clone());
        name
    }

    /// The log mass of `[lo, hi]` under `law` (its `call`, [`LOOP`] kept),
    /// either end open: the helper for a two-sided mass of a `tail` row, the
    /// `lccdf` for a lower end alone, the `lcdf` for an upper end alone, the
    /// `lcdf` pair otherwise. A discrete law's mass counts `lo`, so its
    /// lower end is `lo - 1`, as in Stan's own `T[lo, hi]`. A law whose
    /// support has an end takes [`Self::bounded_lmass`].
    pub(super) fn interval_lmass(
        &mut self,
        law: &Law,
        lo: Option<&str>,
        hi: Option<&str>,
    ) -> String {
        let (stan, args) = stan_call(law);
        if law.tail == "normal_lphi" {
            return self.normal_lmass(&args, lo, hi);
        }
        let ends = law.support_ends();
        if (ends.0.is_some() || ends.1.is_some()) && law.lhs("{y}") == "{y}" {
            return self.bounded_lmass(law, &stan, &args, lo, hi, ends);
        }
        let at = |x: &str| {
            let y = law.lhs(x);
            if args.is_empty() {
                y
            } else {
                format!("{y} | {}", args.join(", "))
            }
        };
        let below = |lo: &str| match (law.discrete, lo.parse::<i64>()) {
            (false, _) => lo.to_string(),
            (true, Ok(v)) => (v - 1).to_string(),
            (true, Err(_)) => format!("{lo} - 1"),
        };
        let (lo_c, hi_c) = (lo.map(|x| law.clamp(x)), hi.map(|x| law.clamp(x)));
        match (lo, hi) {
            (Some(lo), Some(hi)) if !law.tail.is_empty() && law.lhs("{y}") == "{y}" => {
                let helper = self.interval_helper(&stan, law, args.len());
                let mut all = vec![lo.to_string(), hi.to_string()];
                all.extend(args.iter().cloned());
                format!("{helper}({})", all.join(", "))
            }
            (Some(_), Some(_)) => format!(
                "log_diff_exp({stan}_lcdf({}), {stan}_lcdf({}))",
                at(hi_c.as_deref().unwrap()),
                at(&below(lo_c.as_deref().unwrap()))
            ),
            (Some(_), None) => format!("{stan}_lccdf({})", at(&below(lo_c.as_deref().unwrap()))),
            (None, Some(_)) => format!("{stan}_lcdf({})", at(hi_c.as_deref().unwrap())),
            (None, None) => "0".to_string(),
        }
    }

    /// The log mass of `[lo, hi]`, either end open, under a law whose support
    /// has an end (`floor`, `ceil`). No CDF is evaluated at or beyond a
    /// support end: there Stan's `_lcdf` and `_lccdf` throw, or return a
    /// value whose partials are not finite (`exponential_lcdf(0)`,
    /// `beta_lcdf(1)`, `weibull_lccdf(0)`; `uniform_lcdf` is -inf above its
    /// support; BridgeStan, Stan 2.39). An end that is at or beyond the
    /// support's for every value is open, an interval empty for every value
    /// is -inf, and every other case goes through the row's helper, whose
    /// early returns take the same decisions at run time.
    fn bounded_lmass(
        &mut self,
        law: &Law,
        stan: &str,
        args: &[String],
        lo: Option<&str>,
        hi: Option<&str>,
        (floor, ceil): (Option<String>, Option<String>),
    ) -> String {
        let cmp = decided;
        let le = |a: f64, b: f64| a <= b;
        let ge = |a: f64, b: f64| a >= b;
        let lt = |a: f64, b: f64| a < b;
        let gt = |a: f64, b: f64| a > b;
        // Empty for every value. A discrete law has an atom at each end.
        let empty = if law.discrete {
            cmp(hi, &floor, lt, false) || cmp(lo, &ceil, gt, false)
        } else {
            cmp(hi, &floor, le, true) || cmp(lo, &ceil, ge, true)
        };
        if empty {
            return "negative_infinity()".to_string();
        }
        // An end at or beyond the support's is open.
        let lo = lo.filter(|_| !cmp(lo, &floor, le, true));
        let hi = hi.filter(|_| !cmp(hi, &ceil, ge, true));
        // A lone CDF only for an end decided at emit time to lie strictly
        // inside the support; any other end goes to the helper.
        let inside = |x: Option<&str>| {
            x.is_some()
                && (floor.is_none() || cmp(x, &floor, gt, false))
                && (ceil.is_none() || cmp(x, &ceil, lt, false))
        };
        let (lo_in, hi_in) = (inside(lo), inside(hi));
        let at = |x: &str| {
            if args.is_empty() {
                x.to_string()
            } else {
                format!("{x} | {}", args.join(", "))
            }
        };
        let below = |lo: &str| match (law.discrete, lo.parse::<i64>()) {
            (false, _) => lo.to_string(),
            (true, Ok(v)) => (v - 1).to_string(),
            (true, Err(_)) => format!("{} - 1", operand(lo)),
        };
        let lcdf = |x: &str| format!("{stan}_lcdf({})", at(x));
        let lccdf = |x: &str| format!("{stan}_lccdf({})", at(x));
        match (lo, hi) {
            (None, None) => "0".to_string(),
            (Some(l), None) if lo_in => lccdf(&below(l)),
            (None, Some(h)) if hi_in => lcdf(h),
            (Some(l), Some(h)) if lo_in && hi_in && law.tail.is_empty() => {
                format!("log_diff_exp({}, {})", lcdf(h), lcdf(&below(l)))
            }
            // An upper end alone, or a lower one alone, with no support end on
            // its own side: one test, not the helper.
            (Some(l), None) if ceil.is_none() => format!(
                "({} <= {} ? 0 : {})",
                operand(l),
                operand(floor.as_deref().unwrap_or_default()),
                lccdf(&below(l))
            ),
            (None, Some(h)) if floor.is_none() => format!(
                "({} >= {} ? 0 : {})",
                operand(h),
                operand(ceil.as_deref().unwrap_or_default()),
                lcdf(h)
            ),
            (l, h) => {
                let helper = self.interval_helper(stan, law, args.len());
                let mut all = vec![
                    l.or(floor.as_deref()).unwrap_or_default().to_string(),
                    h.or(ceil.as_deref()).unwrap_or_default().to_string(),
                ];
                all.extend(args.iter().cloned());
                format!("{helper}({})", all.join(", "))
            }
        }
    }

    /// `y < lo || y > hi` (`min(y)`/`max(y)` for an array), either end open.
    pub(super) fn outside_test(
        y: &str,
        array: bool,
        lo: Option<&str>,
        hi: Option<&str>,
    ) -> Option<String> {
        let (small, large) = if array {
            (format!("min({y})"), format!("max({y})"))
        } else {
            (y.to_string(), y.to_string())
        };
        let mut parts = Vec::new();
        if let Some(lo) = lo {
            parts.push(format!("{small} < {lo}"));
        }
        if let Some(hi) = hi {
            parts.push(format!("{large} > {hi}"));
        }
        (!parts.is_empty()).then(|| parts.join(" || "))
    }

    /// The name of the helper for the Stan law `stan`, emitted once.
    fn interval_helper(&mut self, stan: &str, law: &Law, nargs: usize) -> String {
        let name = format!("{stan}_interval_lmass");
        if self
            .functions
            .iter()
            .any(|f| f.contains(&format!(" {name}(")))
        {
            return name;
        }
        let ty = if law.discrete { "int" } else { "real" };
        let params: Vec<String> = (1..=nargs).map(|k| format!("p{k}")).collect();
        let decl: String = params.iter().map(|p| format!(", real {p}")).collect();
        let p = params.join(", ");
        let at = |x: &str| {
            if p.is_empty() {
                x.to_string()
            } else {
                format!("{x} | {p}")
            }
        };
        // The support's ends inside the helper: a literal, or the parameter
        // its row names (`floor`, `ceil`).
        let (floor, ceil) = law.support_ends_in(|k| format!("p{}", k + 1));
        let disc = law.discrete;
        let mut body = String::new();
        let ret =
            |cond: String, value: &str| format!("  if ({cond}) {{\n    return {value};\n  }}\n");
        // A window whose ends cross at run time has no mass: `log_diff_exp`
        // of an equal or crossed pair is NaN. A discrete [k, k] keeps its atom.
        body += &ret(
            format!("lo {} hi", if disc { ">" } else { ">=" }),
            "negative_infinity()",
        );
        // Empty: a discrete law has an atom at each end.
        if let Some(f) = &floor {
            body += &ret(
                format!("hi {} {f}", if disc { "<" } else { "<=" }),
                "negative_infinity()",
            );
        }
        if let Some(c) = &ceil {
            body += &ret(
                format!("lo {} {c}", if disc { ">" } else { ">=" }),
                "negative_infinity()",
            );
        }
        // An end at or beyond the support's is open: no CDF is evaluated there.
        if let (Some(f), Some(c)) = (&floor, &ceil) {
            body += &ret(format!("lo <= {f} && hi >= {c}"), "0");
        }
        let low = if disc { "lo - 1" } else { "lo" };
        if let Some(f) = &floor {
            body += &ret(format!("lo <= {f}"), &format!("{stan}_lcdf({})", at("hi")));
        }
        if let Some(c) = &ceil {
            body += &ret(format!("hi >= {c}"), &format!("{stan}_lccdf({})", at(low)));
        }
        let pair = if law.tail.is_empty() {
            format!(
                "  return log_diff_exp({stan}_lcdf({}), {stan}_lcdf({}));\n",
                at("hi"),
                at(low)
            )
        } else {
            format!(
                "  // Above the median the survival pair keeps the precision that the lcdf pair loses.\n  \
                 real la = {stan}_lccdf({});\n  \
                 if (la < log(0.5)) {{\n    \
                 return log_diff_exp(la, {stan}_lccdf({}));\n  \
                 }}\n  \
                 return log_diff_exp({stan}_lcdf({}), {stan}_lcdf({}));\n",
                at(low),
                at("hi"),
                at("hi"),
                at(low),
            )
        };
        let text = format!(
            "real {name}({ty} lo, {ty} hi{decl}) {{\n  \
             // No CDF is evaluated at or beyond a support end, where Stan's value or partial is not finite.\n\
             {body}{pair}}}"
        );
        self.functions.push(text);
        name
    }

    /// The log mass of `[lo, hi]`, either end open, under `normal(mu, sigma)`
    /// (`args`), or `std_normal` (no `args`): every form through one log Φ.
    fn normal_lmass(&mut self, args: &[String], lo: Option<&str>, hi: Option<&str>) -> String {
        let (mu, sigma) = match args {
            [m, s] => (self.printed(m), self.printed(s)),
            _ => (number_ex(0.0), number_ex(1.0)),
        };
        match (lo, hi) {
            (Some(lo), Some(hi)) => {
                let lmass = self.normal_lmass_function();
                format!("{lmass}({lo}, {hi}, {}, {})", mu.1.s, sigma.1.s)
            }
            // [lo, inf) has mass Phi((mu - lo) / sigma): the lower tail of the
            // mirror image.
            (Some(lo), None) => {
                let lphi = self.lphi_function();
                let z = fold_div(&fold_sub(&mu, &self.printed(lo)), &sigma);
                format!("{lphi}({})", z.1.s)
            }
            (None, Some(hi)) => {
                let lphi = self.lphi_function();
                let z = fold_div(&fold_sub(&self.printed(hi), &mu), &sigma);
                format!("{lphi}({})", z.1.s)
            }
            (None, None) => "0".to_string(),
        }
    }

    /// The normal log Φ, emitted on first use under a name that no module
    /// binding, record field or user function takes.
    fn lphi_function(&mut self) -> String {
        if let Some(name) = &self.lphi_fn {
            return name.clone();
        }
        let lphi = self.fresh_local("flatppl_lphi");
        // Below z = -37 erfc is subnormal: `std_normal_lcdf` keeps the value
        // exact (2e-14) but its derivative is an approximation, 2.5e-4 relative
        // at worst against mpmath; no model has mass there (a Mills-ratio series
        // would take it to 8e-12, see TODO-flatppl-rust.md). Above 0 the
        // complement keeps the relative digits of log Phi near 0.
        self.functions.push(format!(
            "real {lphi}(real z) {{\n  \
             // log Phi(z) from erfc for z >= -37: Stan's normal_lcdf derivative is an approximation.\n  \
             if (z > 0) {{\n    \
             return log1m(0.5 * erfc(z / sqrt2()));\n  \
             }}\n  \
             if (z >= -37) {{\n    \
             return log(erfc(-z / sqrt2())) - log2();\n  \
             }}\n  \
             return std_normal_lcdf(z);\n}}"
        ));
        self.lphi_fn = Some(lphi.clone());
        lphi
    }

    /// The normal interval log mass, emitted on first use (with the log Φ it
    /// calls) under a name that no module binding, record field or user
    /// function takes.
    fn normal_lmass_function(&mut self) -> String {
        if let Some(name) = &self.normal_lmass_fn {
            return name.clone();
        }
        let lphi = self.lphi_function();
        let lmass = self.fresh_local("flatppl_normal_lmass");
        self.functions.push(format!(
            "real {lmass}(real lo, real hi, real mu, real sigma) {{\n  \
             // A window whose ends cross at run time has no mass (`log_diff_exp` of an equal\n  \
             // or crossed pair is NaN). An interval above the median is taken as its mirror\n  \
             // image below it.\n  \
             if (lo >= hi) {{\n    \
             return negative_infinity();\n  \
             }}\n  \
             real a = (lo - mu) / sigma;\n  \
             real b = (hi - mu) / sigma;\n  \
             if (a > 0) {{\n    \
             return log_diff_exp({lphi}(-a), {lphi}(-b));\n  \
             }}\n  \
             return log_diff_exp({lphi}(b), {lphi}(a));\n}}"
        ));
        self.normal_lmass_fn = Some(lmass.clone());
        lmass
    }

    /// A printed value as a folding operand: a numeric literal is its number,
    /// other text an atom or a lowest-precedence expression. This is the one
    /// place the module reads printed text: a row's arguments and a
    /// truncation's ends arrive printed.
    pub(super) fn printed(&self, s: &str) -> Folded {
        if let Ok(v) = s.parse::<f64>() {
            return number_ex(v);
        }
        let ex = if operand(s) == s {
            Ex::atom(s)
        } else {
            Ex {
                s: s.to_string(),
                prec: super::P_TERNARY,
            }
        };
        (None, ex, self.may_be_int(s))
    }

    /// Whether printed text can be integer-typed in Stan: a declared name,
    /// or an element of one, reads its declaration; any other text can.
    fn may_be_int(&self, s: &str) -> bool {
        let name = &s[..s.find('[').unwrap_or(s.len())];
        if operand(s) != s || !super::structure::is_identifier(name) {
            return true;
        }
        self.decls.get(name).is_none_or(|t| t.base == "int")
    }

    /// The declared outer extent of a Stan variable, as `extent_of` prints it.
    fn extent_text_of(&self, name: &str) -> String {
        self.decls
            .get(name)
            .and_then(|t| t.array.first().or(t.dims.first()).cloned())
            .unwrap_or_else(|| format!("num_elements({name})"))
    }
}

/// Whether `x` and the support end `e` satisfy `f` at emit time: the same
/// text (when `same`), or two numeric literals. Von Mises's ends are `±pi()`.
fn decided(x: Option<&str>, e: &Option<String>, f: fn(f64, f64) -> bool, same: bool) -> bool {
    match (x, e) {
        (Some(x), Some(e)) => {
            (same && x == e) || matches!((number(x), number(e)), (Some(a), Some(b)) if f(a, b))
        }
        _ => false,
    }
}

/// A printed numeric literal, or von Mises's `±pi()`.
fn number(s: &str) -> Option<f64> {
    match s {
        "pi()" => Some(std::f64::consts::PI),
        "-pi()" => Some(-std::f64::consts::PI),
        _ => s.parse::<f64>().ok(),
    }
}

/// A law's Stan function name and its printed arguments.
fn stan_call(law: &Law) -> (String, Vec<String>) {
    let open = law.call.find('(').unwrap_or(law.call.len());
    let args = law.call[open..]
        .strip_prefix('(')
        .and_then(|x| x.strip_suffix(')'))
        .map(split_args)
        .unwrap_or_default();
    (law.call[..open].to_string(), args)
}

/// `+ 0` folded away from printed `x + y` (a log weight and a log mass).
pub(super) fn add_printed(me: &Emitter, x: &str, y: &str) -> String {
    match (x.parse::<f64>(), y.parse::<f64>()) {
        (Err(_), Err(_)) => format!("{x} + {y}"),
        _ => fold_add(&me.printed(x), &me.printed(y)).1.s,
    }
}

/// `s` as an operand of a sign, `-` or `/`: parenthesized unless it is a
/// name, an unsigned number or one call.
fn operand(s: &str) -> String {
    let mut depth = 0i32;
    let atomic = !s.starts_with('-')
        && s.chars().all(|c| {
            match c {
                '(' | '[' => depth += 1,
                ')' | ']' => depth -= 1,
                _ => {}
            }
            depth > 0 || c.is_ascii_alphanumeric() || "_.()[]".contains(c)
        });
    if atomic {
        s.to_string()
    } else {
        format!("({s})")
    }
}
