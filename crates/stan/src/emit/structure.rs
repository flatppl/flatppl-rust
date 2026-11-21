//! Target decomposition and the `~` statements.

use super::pattern;
use super::{Emitter, Ex, Flat, P_ATOM, PriorField, RowSize};
use crate::ManifestParam;
use crate::refuse::{Refusal, RefusalKind, Result};
use crate::stan_type::StanType;
use crate::tables;
use flatppl_core::{Dim, Inputs, NamedKind, Node, NodeId, RefNs, Scalar, Symbol, Type, ValueSet};
use std::collections::{HashMap, HashSet};

/// The loop index a per-element argument of a broadcast law is rendered with.
pub(super) const LOOP: &str = "\u{1}";

/// A law as a Stan sampling statement right-hand side.
pub(super) struct Law {
    /// `stan(args)` for one statement over the whole variate.
    pub call: String,
    /// `call` with rank-2 arguments flattened, for a flattened variate.
    flat_call: Option<String>,
    /// `stan(args)` with [`LOOP`] where a broadcast loop indexes.
    pub indexed: String,
    /// A broadcast varies a vector parameter per element, which Stan does not
    /// vectorize (`categorical` with one simplex per observation).
    vector_per_element: bool,
    /// The `distributions.ron` variate template (`{y}`, `{y} + 1`).
    lhs: String,
    /// `T[lower, upper]`, either side open.
    pub trunc: Option<(Option<String>, Option<String>)>,
    pub vectorized: bool,
    pub multivariate: bool,
    /// Stan defines `_lcdf` and `_lccdf`, so `T[L, U]` is well formed.
    pub lcdf: bool,
    /// An integer variate.
    pub discrete: bool,
    /// The support an unnormalized `truncate` restricts its variate to.
    pub support: Option<(Option<String>, Option<String>)>,
    /// The `distributions.ron` guard with its arguments filled: `{y}` stays.
    /// The second form indexes per element ([`LOOP`]).
    pub guard: Option<(String, String)>,
    /// The `distributions.ron` variate bounds as written (`0`, `{n} - 1`):
    /// with `guard`, the support a Dirac atom beside this law is tested
    /// against.
    pub bounds: (Option<String>, Option<String>),
    /// The row's `tail` flag (see `interval_mass`).
    pub tail: String,
    /// The position among the Stan arguments of the row's `floor` argument.
    pub floor_arg: Option<usize>,
    /// Statements the law adds after its `~` line (`target += lgamma(K + 1);`).
    pub extra: Vec<String>,
    /// The restriction lives in the variate's declaration (`ordered[K]`):
    /// the law is exact only on a parameter declared from its sets.
    pub declared: bool,
    /// A row with `normalized: false` (`ContinuedPoisson`): its statement is
    /// `target += D_lpdf(y | …);`, not `~`.
    pub unnormalized: bool,
    /// The row's Stan name (`flatppl_continued_poisson`): `call` is
    /// `name(args)`.
    pub stan_name: String,
    /// The FlatPPL law the row maps (`Poisson`), for messages.
    pub flatppl: String,
    /// A user-defined `_lpdf` row (`kind: custom_lpdf`): its overloads take
    /// `vector` variates, not integer arrays.
    pub custom: bool,
    /// The row's Stan function does not enforce `bounds` (von Mises): a
    /// variate whose declaration does not state them is tested against them.
    pub unchecked_support: bool,
    /// The row's Stan function returns -inf outside `bounds` (`total_support`).
    pub total_support: bool,
    /// The constrained Stan type of the row's `variate` (`cholesky_factor_corr`).
    pub support_type: Option<&'static str>,
}

impl Law {
    pub(super) fn rhs(&self, index: Option<&str>) -> String {
        self.truncate(self.call_text(index))
    }

    /// `stan(args)` with no truncation, at `index` for a per-element law.
    pub(super) fn call_text(&self, index: Option<&str>) -> String {
        match index {
            Some(i) => self.indexed.replace(LOOP, i),
            None => self.call.clone(),
        }
    }

    fn truncate(&self, mut s: String) -> String {
        if let Some((lo, hi)) = &self.trunc {
            s.push_str(&format!(
                " T[{}, {}]",
                lo.as_deref().map(|x| self.clamp(x)).unwrap_or_default(),
                hi.as_deref().map(|x| self.clamp(x)).unwrap_or_default()
            ));
        }
        s
    }

    /// The literal ends of the law's support: its variate bounds, or the
    /// argument its row names as `floor`.
    fn support_ends(&self) -> (Option<String>, Option<String>) {
        // A literal end: a number, or von Mises's `±pi()`.
        let num = |b: &Option<String>| {
            b.clone()
                .filter(|v| v.parse::<f64>().is_ok() || matches!(v.as_str(), "pi()" | "-pi()"))
        };
        let floor = match self.floor_arg {
            Some(k) => {
                let open = self.call.find('(').unwrap_or(self.call.len());
                self.call[open..]
                    .strip_prefix('(')
                    .and_then(|v| v.strip_suffix(')'))
                    .map(super::mixture::split_args)
                    .and_then(|a| a.get(k).cloned())
            }
            None => num(&self.bounds.0),
        };
        (floor, num(&self.bounds.1))
    }

    fn pick(&self, x: &str, b: &str, larger: bool) -> String {
        pick(x, b, larger, self.discrete)
    }

    /// A truncation end clamped into the law's support, for a CDF argument:
    /// Stan's CDFs raise outside it.
    pub(super) fn clamp(&self, x: &str) -> String {
        let (floor, ceil) = self.support_ends();
        let mut x = x.to_string();
        if let Some(f) = &floor {
            x = self.pick(&x, f, true);
        }
        if let Some(c) = &ceil {
            x = self.pick(&x, c, false);
        }
        x
    }

    /// `[lo, hi]` intersected with the law's support, for an outside test:
    /// the lower end raised to the support's, the upper one lowered.
    pub(super) fn within(
        &self,
        lo: Option<&str>,
        hi: Option<&str>,
    ) -> (Option<String>, Option<String>) {
        let (floor, ceil) = self.support_ends();
        let lo = lo.map(|x| match &floor {
            Some(f) => self.pick(x, f, true),
            None => x.to_string(),
        });
        let hi = hi.map(|x| match &ceil {
            Some(c) => self.pick(x, c, false),
            None => x.to_string(),
        });
        (lo, hi)
    }

    /// The variate side for `y`.
    pub(super) fn lhs(&self, y: &str) -> String {
        self.lhs.replace("{y}", y)
    }

    /// `y ~ call T[…];`, or `target += D_lpdf(y | args);` for an
    /// unnormalized row. `call` is the bare `stan(args)` (`call_text`); its
    /// name and arguments come from the row, not from a printed suffix.
    pub(super) fn statement(&self, at: NodeId, y: &str, call: &str) -> Result<String> {
        if !self.unnormalized {
            return Ok(format!("{y} ~ {};", self.truncate(call.to_string())));
        }
        // An unnormalized row has no CDF, so it is never truncated here.
        let args = call
            .strip_prefix(self.stan_name.as_str())
            .and_then(|a| a.strip_prefix('('))
            .and_then(|a| a.strip_suffix(')'))
            .filter(|_| self.trunc.is_none());
        let Some(args) = args else {
            return Err(Refusal::not_yet(
                Some(at),
                format!("the unnormalized law `{}` in this form", self.flatppl),
            ));
        };
        let name = &self.stan_name;
        Ok(if args.is_empty() {
            format!("target += {name}_lpdf({y});")
        } else {
            format!("target += {name}_lpdf({y} | {args});")
        })
    }
}

/// The kernel of a likelihood: its inputs and its outputs paired with the
/// observed values.
struct Observation {
    /// The draw binding the output names, when it is a binding.
    draw: Option<Symbol>,
    /// The law of the output variate.
    law: NodeId,
    /// The observed value.
    value: NodeId,
    /// Name for a data variable the observation needs.
    name: String,
}

/// A likelihood's input signature (§06 "Likelihood construction"), over
/// every part of a `joint_likelihood`.
struct Signature {
    /// Each input name with every binding it sets: a kernel input's value,
    /// a measure likelihood's free `elementof` of that name.
    inputs: Vec<(Symbol, Vec<Symbol>)>,
    /// The `elementof` the parts read outside a kernel boundary and no input
    /// sets: the module inputs the likelihood closes over.
    free: Vec<Symbol>,
}

impl<'m> Emitter<'m> {
    /// Emit the target binding `target`.
    pub(crate) fn emit_target(&mut self, target: Symbol) -> Result<()> {
        // The ABI target of a declared signature (§13): its `elementof`
        // leaves must all be listed in `inputs`.
        self.abi_target = self.name(target) == "outputs" && !self.abi_inputs().is_empty();
        let bid = self.m.alias_target(target).ok_or_else(|| {
            Refusal::new(
                RefusalKind::NoTarget,
                None,
                self.name(target).to_string(),
                "no such binding",
            )
        })?;
        let rhs = self.m.binding(bid).rhs;
        match self.head(rhs) {
            Some("bayesupdate") => {
                let c = self.call(rhs).unwrap();
                let (lik, prior) = (c.args[0], c.args[1]);
                let fields = self.prior_fields(prior)?;
                self.declare_params(rhs, &fields)?;
                for (name, sym) in std::mem::take(&mut self.links) {
                    if let Some(stan) = self.fields.get(&name).cloned() {
                        self.env.insert(sym, stan);
                    }
                }
                self.bind_derived_draws(&fields)?;
                let obs = self.likelihood(lik, true)?;
                self.prior_statements(&fields)?;
                self.observe(obs)
            }
            // `logweighted(lw, prior)` as the target: a posterior (§06
            // "Posterior construction") with no separate likelihood.
            Some("logweighted") => {
                let fields = self.prior_fields(rhs)?;
                self.declare_params(rhs, &fields)?;
                for (name, sym) in std::mem::take(&mut self.links) {
                    if let Some(stan) = self.fields.get(&name).cloned() {
                        self.env.insert(sym, stan);
                    }
                }
                self.bind_derived_draws(&fields)?;
                self.prior_statements(&fields)
            }
            Some("likelihoodof" | "joint_likelihood") => {
                let obs = self.likelihood(rhs, false)?;
                self.observe(obs)
            }
            // `logdensityof(M, x)`: the likelihood of `M` at the data `x`. The
            // free inputs `M` reads are its parameters.
            Some("logdensityof") => {
                let c = self.call(rhs).unwrap();
                let [m, x] = c.args[..] else {
                    return Err(Refusal::not_yet(Some(rhs), "logdensityof with this arity"));
                };
                let mr = self.resolve(m);
                if self.head(mr) == Some("restrict") {
                    return Err(Self::restrict_not_yet(mr));
                }
                if matches!(self.head(mr), Some("bayesupdate" | "logweighted")) {
                    return self.posterior_at_point(mr, x);
                }
                if matches!(self.head(mr), Some("likelihoodof" | "joint_likelihood")) {
                    return self.likelihood_at_point(mr, x);
                }
                self.density_term(m, x)
            }
            // A sum of log densities is one program: each term adds its
            // statements to `target` (§06: independent terms add).
            Some("add" | "sum") if self.reads_log_density(rhs) => {
                let mut terms = Vec::new();
                self.density_terms(rhs, &mut terms)?;
                for &(m, _) in &terms {
                    let mr = self.resolve(m);
                    if matches!(self.head(mr), Some("bayesupdate" | "logweighted")) {
                        return Err(Refusal::not_yet(Some(m), "a sum with a posterior term"));
                    }
                }
                // All terms' inputs first, so term order does not decide a block.
                let measures: Vec<NodeId> = terms.iter().map(|t| t.0).collect();
                let points: Vec<NodeId> = terms.iter().map(|t| t.1).collect();
                self.declare_query_inputs(&measures, &points)?;
                for (m, x) in terms {
                    let obs = self.measure_observations(m, x)?;
                    self.observe(obs)?;
                }
                Ok(())
            }
            Some("sub") if self.reads_log_density(rhs) => {
                Err(Refusal::not_yet(Some(rhs), "a difference of log densities"))
            }
            // A restricted measure has a density (the joint at the observed
            // fields); its Stan form is not written yet.
            Some("restrict") => Err(Self::restrict_not_yet(rhs)),
            Some("tuple") => Err(Refusal::new(
                RefusalKind::NotADensity,
                Some(rhs),
                format!("`{}` of several values", self.name(target)),
                format!(
                    "`{}` holds several values; a Stan program has one output, its log density",
                    self.name(target)
                ),
            )),
            Some("log")
                if self
                    .call(rhs)
                    .and_then(|c| c.args.first())
                    .is_some_and(|&a| self.head(self.resolve(a)) == Some("densityof")) =>
            {
                Err(Refusal::not_yet(
                    Some(rhs),
                    "`log(densityof(…))`; write `logdensityof(…)`",
                ))
            }
            // `densityof` and `exp(logdensityof(…))` are densities, not log
            // densities. Another expression over a density may be a log
            // density (`log(densityof(a, x)) + …`): not yet supported.
            Some(h)
                if !self.reads_any(rhs, &["logdensityof", "densityof"])
                    || matches!(h, "densityof" | "exp") =>
            {
                let h = self.written_head(rhs, h);
                Err(Refusal::new(
                    RefusalKind::NotADensity,
                    Some(rhs),
                    format!("a `{h}` target"),
                    format!(
                        "`{}` is a `{h}`, not a log density; a Stan program's one output is its log density",
                        self.name(target)
                    ),
                ))
            }
            Some(h) => Err(Refusal::not_yet(
                Some(rhs),
                format!("a `{h}` target (`{}`)", self.name(target)),
            )),
            None => Err(Refusal::not_yet(
                Some(rhs),
                format!("the target `{}`", self.name(target)),
            )),
        }
    }

    /// The head of `id` as the user wrote it: a user call by its callee's
    /// name (`kernel(…)`), not the internal `%call`.
    fn written_head(&self, id: NodeId, head: &str) -> String {
        if !head.starts_with('%') {
            return head.to_string();
        }
        match self.call(id).map(|c| c.head) {
            Some(flatppl_core::CallHead::User(callee)) => match self.m.node(callee) {
                Node::Ref(r) => format!("{}(…)", self.name(r.name)),
                _ => "call".to_string(),
            },
            _ => "call".to_string(),
        }
    }

    fn restrict_not_yet(at: NodeId) -> Refusal {
        Refusal::not_yet(
            Some(at),
            "`restrict` of a joint law to observed fields (the posterior density)",
        )
    }

    /// `logdensityof(M, x)` of a measure: the likelihood of M at the point x.
    /// The free inputs M and the point read are the program's inputs.
    fn density_term(&mut self, m: NodeId, x: NodeId) -> Result<()> {
        self.declare_query_inputs(&[m], &[x])?;
        let obs = self.measure_observations(m, x)?;
        self.observe(obs)
    }

    /// The `logdensityof(M, x)` terms of a sum: `a + b`, `sum([a, b, …])`
    /// of a literal list, through bindings.
    fn density_terms(&mut self, e: NodeId, out: &mut Vec<(NodeId, NodeId)>) -> Result<()> {
        let r = self.resolve(e);
        let c = self.call(r);
        match (self.head(r), c.map(|c| c.args.to_vec())) {
            (Some("add"), Some(args)) if args.len() == 2 => {
                for a in args {
                    self.density_terms(a, out)?;
                }
                Ok(())
            }
            (Some("sum"), Some(args)) if args.len() == 1 => {
                let list = self.resolve(args[0]);
                match (self.head(list), self.call(list)) {
                    (Some("vector"), Some(lc)) => {
                        for &a in lc.args.iter() {
                            self.density_terms(a, out)?;
                        }
                        Ok(())
                    }
                    _ => Err(Refusal::not_yet(
                        Some(r),
                        "a `sum` of a computed list of log densities",
                    )),
                }
            }
            (Some("logdensityof"), Some(args)) if args.len() == 2 => {
                out.push((args[0], args[1]));
                Ok(())
            }
            (h, _) => Err(Refusal::not_yet(
                Some(r),
                match h {
                    Some(h) => format!(
                        "a term that is not a `logdensityof` (`{h}`) in a sum of log densities"
                    ),
                    None => "a term that is not a `logdensityof` in a sum of log densities".into(),
                },
            )),
        }
    }

    /// `true` when `e` reads a `logdensityof` (through bindings).
    fn reads_log_density(&self, e: NodeId) -> bool {
        self.reads_any(e, &["logdensityof"])
    }

    /// `true` when `e` reads a call to one of `heads` (through bindings).
    fn reads_any(&self, e: NodeId, heads: &[&str]) -> bool {
        let mut stack = vec![e];
        let mut seen = HashSet::new();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            if self.head(id).is_some_and(|h| heads.contains(&h)) {
                return true;
            }
            if let Node::Ref(r) = self.m.node(id)
                && r.ns == RefNs::SelfMod
                && let Some(b) = self
                    .m
                    .binding_by_name(*self.m.alias_chain(r.name).last().unwrap())
            {
                stack.push(self.m.binding(b).rhs);
            }
            self.m.for_each_child(id, |k| stack.push(k));
        }
        false
    }

    /// The bounds of the field's sets (`declare_from_sets`) that the Stan
    /// variable `stan` does not carry in its declaration, for the support
    /// test; `None` when its declaration has another base (`ordered` against
    /// none). A `positive_ordered` field is base `ordered` with lower bound 0.
    #[allow(clippy::type_complexity)]
    fn field_set_residual(
        &mut self,
        stan: &str,
        f: &PriorField,
    ) -> Result<Option<(Option<String>, Option<String>)>> {
        let Some(t) = self.decls.get(stan).cloned() else {
            return Ok(None);
        };
        let (mut ordered, mut positive) = (false, false);
        let (mut lowers, mut uppers) = (Vec::new(), Vec::new());
        for &s in &f.sets {
            let mut e = self.resolve(s);
            while self.head(e) == Some("cartpow") {
                e = self.resolve(self.call(e).unwrap().args[0]);
            }
            match (self.head(e), self.m.node(e)) {
                (Some("ordered"), _) => ordered = true,
                (Some("interval"), _) => {
                    let (lo, hi) = self.interval(e)?;
                    lowers.extend(lo);
                    uppers.extend(hi);
                }
                (_, Node::Const(c)) if matches!(self.name(*c), "posreals" | "nonnegreals") => {
                    positive = true;
                }
                (_, Node::Const(c)) if self.name(*c) == "reals" => {}
                _ => return Ok(None),
            }
        }
        lowers.dedup();
        uppers.dedup();
        if positive {
            lowers.push("0".to_string());
            lowers.dedup();
        }
        if lowers.len() > 1 || uppers.len() > 1 {
            return Ok(None);
        }
        let input_ordered = matches!(t.base, "ordered" | "positive_ordered");
        if ordered != input_ordered {
            return Ok(None);
        }
        // A bound the declaration states (or `positive_ordered`'s 0) is carried.
        let num = |b: Option<&str>| b.and_then(|v| v.parse::<f64>().ok());
        let carried_lower = if t.base == "positive_ordered" {
            Some("0")
        } else {
            t.bounds.lower.as_deref()
        };
        let residual = |want: Option<&String>, have: Option<&str>| match want {
            Some(w) if num(Some(w)).is_none() || num(Some(w)) != num(have) => Some(w.clone()),
            _ => None,
        };
        Ok(Some((
            residual(lowers.first(), carried_lower),
            residual(uppers.first(), t.bounds.upper.as_deref()),
        )))
    }

    /// `logdensityof(L, record(k = e, …))` of a likelihood: the likelihood's
    /// value at the point, each kernel input set from the point (§06: a
    /// likelihood is a function of the kernel inputs).
    fn likelihood_at_point(&mut self, lik: NodeId, x: NodeId) -> Result<()> {
        let point = self.resolve(x);
        if self.head(point) != Some("record") {
            return Err(Refusal::not_yet(
                Some(x),
                "a likelihood at a point that is not a record",
            ));
        }
        // §06: one point field per input of the likelihood, each binding
        // every denotation of its name. The `elementof` the parts read
        // outside a kernel boundary, the point's and the observed values'
        // are the program's inputs, in one pass.
        let sig = self.likelihood_signature(lik);
        self.check_point_keys(point, &sig.inputs)?;
        let bound = self.point_binding(point, &sig.inputs)?;
        let mut points = vec![x];
        self.likelihood_parts(lik, &mut Vec::new(), &mut points);
        self.declare_inputs(&sig.free, &points)?;
        let named: Vec<(Symbol, NodeId)> = self
            .call(point)
            .unwrap()
            .named
            .iter()
            .map(|na| (na.name, na.value))
            .collect();
        for (field, e) in named {
            let stan = self.point_value(field, e)?;
            for &(_, sym) in bound.iter().filter(|b| b.0 == field) {
                self.env.insert(sym, stan.clone());
            }
            self.fields.insert(field, stan);
        }
        let obs = self.likelihood(lik, false)?;
        self.observe(obs)
    }

    /// The Stan text of a point's field value: an input's name, or a local.
    fn point_value(&mut self, field: Symbol, e: NodeId) -> Result<String> {
        Ok(match self.self_ref(e) {
            Some(s) => self.bind(*self.m.alias_chain(s).last().unwrap())?,
            None => {
                let ex = self.expr(e)?;
                if is_identifier(&ex.s) {
                    ex.s
                } else {
                    let base = self.name(field).to_string();
                    self.place(&base, e, Ok(ex))?
                }
            }
        })
    }

    /// `logdensityof(post, record(f = e, …))` of a posterior (`bayesupdate`
    /// or `logweighted`): each prior field takes the point's value, and the
    /// inputs the point reads are the program's parameters (§06: the density
    /// of the posterior measure at the point).
    fn posterior_at_point(&mut self, post: NodeId, x: NodeId) -> Result<()> {
        let (lik, prior) = match self.head(post) {
            Some("bayesupdate") => {
                let c = self.call(post).unwrap();
                (Some(c.args[0]), c.args[1])
            }
            _ => (None, post),
        };
        let point = self.resolve(x);
        if self.head(point) != Some("record") {
            return Err(Refusal::not_yet(
                Some(x),
                "a posterior at a point that is not a record",
            ));
        }
        let fields = self.prior_fields(prior)?;
        // The prior's inputs, the likelihood's outside its kernel boundaries,
        // the point's and the observed values': in one pass. A prior field
        // declared by `elementof` takes the point's value, not a declaration.
        let denotes: Vec<(Symbol, Vec<Symbol>)> = fields
            .iter()
            .filter_map(|f| {
                let b = f.binding?;
                let rhs = self.m.binding(self.m.binding_by_name(b)?).rhs;
                (self.head(rhs) == Some("elementof")).then(|| (f.field, vec![b]))
            })
            .collect();
        let bound = self.point_binding(point, &denotes)?;
        let mut params = self.free_input_syms(prior);
        let mut points = vec![x];
        if let Some(l) = lik {
            params.extend(self.likelihood_signature(l).free);
            self.likelihood_parts(l, &mut Vec::new(), &mut points);
        }
        params.retain(|s| !bound.iter().any(|b| b.1 == *s));
        self.declare_inputs(&params, &points)?;
        for f in &fields {
            let e = self.obs_field(x, f.field)?;
            let stan = self.point_value(f.field, e)?;
            for &(_, sym) in bound.iter().filter(|b| b.0 == f.field) {
                self.env.insert(sym, stan.clone());
            }
            // A declared set (`ordered`, an interval) restricts a parameter
            // through its declaration: a point carries it as an input declared
            // on the same base, and the support test checks any bound that
            // declaration does not state.
            if !f.sets.is_empty() {
                let Some(residual) = self.field_set_residual(&stan, f)? else {
                    return Err(Refusal::not_yet(
                        Some(f.node),
                        format!(
                            "the prior field `{}` with a declared set at a point that is not an input on that set",
                            self.name(f.field)
                        ),
                    ));
                };
                if residual.0.is_some() || residual.1.is_some() {
                    self.point_guards.insert(f.field, residual);
                }
                self.set_declared.insert(stan.clone());
            }
            if let Some(b) = f.binding {
                self.env.insert(b, stan.clone());
            }
            self.fields.insert(f.field, stan);
        }
        for (name, sym) in std::mem::take(&mut self.links) {
            if let Some(stan) = self.fields.get(&name).cloned() {
                self.env.insert(sym, stan);
            }
        }
        self.bind_derived_draws(&fields)?;
        let obs = match lik {
            Some(l) => Some(self.likelihood(l, true)?),
            None => None,
        };
        self.point_fields = fields.iter().map(|f| f.field).collect();
        self.prior_statements(&fields)?;
        match obs {
            Some(o) => self.observe(o),
            None => Ok(()),
        }
    }

    /// Each derived prior field's draw x reads f⁻¹ of the field (§06: x is
    /// no parameter of its own).
    fn bind_derived_draws(&mut self, fields: &[PriorField]) -> Result<()> {
        for f in fields {
            let Some(law) = f.law else { continue };
            let Some(&(_, x, _)) = self.derived.get(&law) else {
                continue;
            };
            let stan = self.fields[&f.field].clone();
            let value = self.derived_draw_value(law, &stan)?;
            self.env.insert(x.name, value);
        }
        Ok(())
    }

    // ---------------------------------------------------------------- prior

    /// The prior's `~` lines, then one `target +=` per log weight.
    fn prior_statements(&mut self, fields: &[PriorField]) -> Result<()> {
        for f in fields {
            let lhs = self.fields[&f.field].clone();
            self.relies_on_bounds = false;
            let mut lines = match f.law {
                Some(law) => self.tilde(&lhs, law)?,
                None => vec![],
            };
            let relies_on_bounds = std::mem::take(&mut self.relies_on_bounds);
            // A field set from a point is tested against its support only where
            // the line alone gives a finite value outside it: a declared set's
            // bounds its input does not state, a truncation printed without
            // `T[]` (the render reports it), or a field with no statement
            // (`tilde` itself guards an `unchecked_support` law).
            if self.point_fields.contains(&f.field) {
                let bounds = match (self.point_guards.get(&f.field).cloned(), f.law) {
                    (Some(b), _) => Some(b),
                    (None, Some(law)) if lines.is_empty() || relies_on_bounds => {
                        self.support(law)?
                    }
                    (None, Some(_)) => None,
                    (None, None) => self.interval(self.resolve(f.node)).ok(),
                };
                if let Some((lo, hi)) = bounds
                    && (lo.is_some() || hi.is_some())
                {
                    let Some(g) = self.guarded(&lhs, lo.as_deref(), hi.as_deref(), lines.clone())
                    else {
                        return Err(Refusal::not_yet(
                            Some(f.node),
                            format!(
                                "the restricted prior field `{}` at a point",
                                self.name(f.field)
                            ),
                        ));
                    };
                    lines = vec![g];
                }
            }
            self.prior.extend(lines);
        }
        self.emit_weights()
    }

    fn prior_fields(&mut self, prior: NodeId) -> Result<Vec<PriorField>> {
        let p = self.resolve(prior);
        let mut out = Vec::new();
        match self.head(p) {
            Some("lawof") => {
                let rec = self.resolve(self.call(p).unwrap().args[0]);
                if self.head(rec) != Some("record") {
                    return Err(Refusal::not_yet(Some(p), "a prior `lawof` of a non-record"));
                }
                let named: Vec<(Symbol, NodeId)> = self
                    .call(rec)
                    .unwrap()
                    .named
                    .iter()
                    .map(|na| (na.name, na.value))
                    .collect();
                // A field `v = f(x)` of one draw x has law `pushfwd(f, law(x))`
                // (§06); x is then no parameter of its own.
                let mut behind: Vec<(Symbol, Symbol)> = Vec::new();
                for (field, value) in named {
                    let sym = self
                        .self_ref(value)
                        .map(|s| *self.m.alias_chain(s).last().unwrap());
                    let rhs = sym.map(|s| self.m.binding(self.m.binding_by_name(s).unwrap()).rhs);
                    let (binding, law) = match (sym, rhs) {
                        (Some(s), Some(rhs)) if self.head(rhs) == Some("draw") => {
                            (Some(s), self.call(rhs).unwrap().args[0])
                        }
                        _ => {
                            let name = self.name(field).to_string();
                            let (x, _) = self.derived_field(&name, value, value)?;
                            behind.push((x, field));
                            (sym, self.resolve(value))
                        }
                    };
                    out.push(PriorField {
                        field,
                        binding,
                        law: Some(law),
                        node: value,
                        sets: self.law_sets(law),
                    });
                }
                for (k, &(x, field)) in behind.iter().enumerate() {
                    let node = out.iter().find(|f| f.field == field).unwrap().node;
                    if out.iter().any(|f| f.binding == Some(x)) {
                        return Err(Refusal::not_yet(
                            Some(node),
                            format!(
                                "a derived prior field of `{}`, which is itself a prior field",
                                self.name(x)
                            ),
                        ));
                    }
                    if behind[..k].iter().any(|b| b.0 == x) {
                        return Err(Refusal::not_yet(
                            Some(node),
                            format!(
                                "two derived prior fields of the one draw `{}`",
                                self.name(x)
                            ),
                        ));
                    }
                }
            }
            Some("joint") => {
                let c = self.call(p).unwrap();
                for &a in c.args.iter() {
                    out.extend(self.prior_fields(a)?);
                }
                for na in c.named.iter() {
                    out.push(PriorField {
                        field: na.name,
                        binding: None,
                        law: Some(na.value),
                        node: na.value,
                        sets: self.law_sets(na.value),
                    });
                }
            }
            Some("jointchain") => {
                // jointchain(M, K…): M's fields, then each kernel's outputs.
                // A kernel input binds to the earlier field of the same name.
                let c = self.call(p).unwrap();
                if !c.named.is_empty() {
                    return Err(Refusal::not_yet(Some(p), "a keyword jointchain prior"));
                }
                let args = c.args.to_vec();
                out.extend(self.prior_fields(args[0])?);
                for &k in &args[1..] {
                    let k = self.resolve(k);
                    let kc = self.call(k).filter(|_| self.head(k) == Some("kernelof"));
                    let Some(kc) = kc else {
                        return Err(Refusal::not_yet(
                            Some(k),
                            "a jointchain step that is not a kernelof",
                        ));
                    };
                    if let Some(Inputs::Spec(list)) = &kc.inputs {
                        for (name, r) in list.iter() {
                            if r.ns == RefNs::SelfMod {
                                self.links
                                    .push((*name, *self.m.alias_chain(r.name).last().unwrap()));
                            }
                        }
                    }
                    let rec = self.resolve(kc.args[0]);
                    if self.head(rec) != Some("record") {
                        return Err(Refusal::not_yet(
                            Some(k),
                            "a jointchain step over a non-record",
                        ));
                    }
                    for na in self.call(rec).unwrap().named.iter() {
                        let (draw, law) = self.output_law(na.value)?;
                        out.push(PriorField {
                            field: na.name,
                            binding: draw,
                            law: Some(law),
                            node: na.value,
                            sets: vec![],
                        });
                    }
                }
            }
            Some("Lebesgue") => {
                let c = self.call(p).unwrap();
                let support = c
                    .args
                    .first()
                    .copied()
                    .or_else(|| c.named.first().map(|n| n.value))
                    .map(|s| self.resolve(s));
                let Some(support) = support.filter(|&s| self.head(s) == Some("cartprod")) else {
                    return Err(Refusal::not_yet(Some(p), "an unnamed Lebesgue prior"));
                };
                for na in self.call(support).unwrap().named.iter() {
                    out.push(PriorField {
                        field: na.name,
                        binding: None,
                        law: None,
                        node: na.value,
                        sets: vec![],
                    });
                }
            }
            Some("logweighted") => {
                let c = self.call(p).unwrap();
                let [lw, base] = c.args[..] else {
                    // `logweighted(lw, M, p)`: no model uses the callback form.
                    return Err(Refusal::not_yet(
                        Some(p),
                        "a `logweighted` prior with a callback parameter",
                    ));
                };
                if !c.named.is_empty() {
                    return Err(Refusal::not_yet(
                        Some(p),
                        "a `logweighted` prior with keyword arguments",
                    ));
                }
                out.extend(self.prior_fields(base)?);
                self.weights.push((lw, base));
            }
            // A record-level restriction of Lebesgue measure: each field keeps
            // its own and its restricting set, which declare it.
            Some("truncate") => {
                let c = self.call(p).unwrap();
                let (inner, set) = (self.resolve(c.args[0]), self.resolve(c.args[1]));
                if self.head(inner) != Some("Lebesgue") || self.head(set) != Some("cartprod") {
                    return Err(Refusal::not_yet(Some(p), "a `truncate` prior of this form"));
                }
                let support = self.call(inner).unwrap();
                let support = support
                    .args
                    .first()
                    .copied()
                    .or_else(|| support.named.first().map(|n| n.value))
                    .map(|s| self.resolve(s));
                let mut fs = self.prior_fields(inner)?;
                for f in &mut fs {
                    for s in [support, Some(set)].into_iter().flatten() {
                        if let Some(na) = self
                            .call(s)
                            .unwrap()
                            .named
                            .iter()
                            .find(|na| na.name == f.field)
                        {
                            f.sets.push(na.value);
                        }
                    }
                }
                out.extend(fs);
            }
            Some(h) => return Err(Refusal::not_yet(Some(p), format!("a `{h}` prior"))),
            None => return Err(Refusal::not_yet(Some(p), "this prior")),
        }
        Ok(out)
    }

    /// Declare one parameter per prior field, typed by the posterior's domain.
    fn declare_params(&mut self, post: NodeId, fields: &[PriorField]) -> Result<()> {
        let domain = match self.m.type_of(post) {
            Some(Type::Measure { domain, .. }) => domain.as_ref().clone(),
            _ => return Err(Refusal::not_yet(Some(post), "an untyped posterior")),
        };
        let Type::Record(types) = domain else {
            return Err(Refusal::not_yet(
                Some(post),
                "a posterior over a non-record",
            ));
        };
        let sets = match self.m.valueset_of(post) {
            Some(ValueSet::RecordSet(s)) => s.to_vec(),
            _ => vec![],
        };
        for f in fields {
            let ty = types
                .iter()
                .find(|(n, _)| *n == f.field)
                .map(|(_, t)| t.clone())
                .ok_or_else(|| {
                    Refusal::not_yet(Some(f.node), "a prior field outside the domain")
                })?;
            let vs = sets
                .iter()
                .find(|(n, _)| *n == f.field)
                .map(|(_, v)| v.clone());
            // The bare-name refusal comes before the untyped variate's.
            if let Some(law) = f.law {
                self.bare_pushfwd_map(law)?;
            }
            if ty_is_int(&ty) {
                return Err(Refusal::new(
                    RefusalKind::DiscreteParameter,
                    Some(f.node),
                    format!("prior field `{}`", self.name(f.field)),
                    format!(
                        "prior field `{}` has a discrete law; Stan has no discrete parameters",
                        self.name(f.field)
                    ),
                ));
            }
            let stan = self.fresh(self.name(f.field));
            let hints = match f.law {
                Some(law) => self.law_hints(law),
                None => self.set_hints(f.node),
            };
            // A law whose support is a constrained set (§08: correlation,
            // covariance matrices, the simplex) declares its variate on it.
            let support = match f.law {
                Some(law) => self.law_support(law)?,
                None => None,
            };
            let ty = match support {
                Some((_, axes, Some(n))) => with_extent(&ty, axes, n),
                _ => ty,
            };
            let mut t = if f.sets.is_empty() {
                self.declare_ty(&ty, vs.as_ref(), true, f.node, &stan, &hints)?
            } else {
                let hints = self.set_hints(f.sets[0]);
                self.set_declared.insert(stan.clone());
                self.declare_from_sets(&ty, f, &stan, &hints)?
            };
            // A support the value set cannot state (a computed bound) comes from
            // the law itself.
            if let Some(law) = f.law
                && let Some((lo, hi)) = self.support(law)?
            {
                // A map's image of its base support is exact; the value set
                // can be wider (`exp(sqrt(t))` is in posreals, its image
                // starts at 1), and the declaration must not admit points
                // the map never reaches.
                let m = self.resolve(law);
                let image = self.head(m) == Some("pushfwd") || self.derived.contains_key(&m);
                if t.bounds.lower.is_none() || (image && lo.is_some()) {
                    t.bounds.lower = lo;
                }
                if t.bounds.upper.is_none() || (image && hi.is_some()) {
                    t.bounds.upper = hi;
                }
            }
            if let Some((s, ..)) = support
                && t.base != s
            {
                if !matches!(t.base, "vector" | "matrix") {
                    return Err(Refusal::not_yet(
                        Some(f.node),
                        format!("a `{s}` variate declared as `{}`", t.base),
                    ));
                }
                t.support = Some(s);
                t.bounds = Default::default();
            }
            self.params.push(format!("{};", t.render(&stan)));
            self.record_decl(&stan, t);
            if let Some(b) = f.binding {
                self.env.insert(b, stan.clone());
            }
            self.fields.insert(f.field, stan.clone());
            self.manifest_bindings.push(f.binding);
            self.manifest.push(ManifestParam {
                stan,
                field: self.name(f.field).to_string(),
                input: None,
                index: None,
            });
        }
        Ok(())
    }

    // ----------------------------------------------------------- likelihood

    fn likelihood(&mut self, l: NodeId, has_prior: bool) -> Result<Vec<Observation>> {
        let l = self.resolve(l);
        match self.head(l) {
            Some("joint_likelihood") => {
                let args = self.call(l).unwrap().args.to_vec();
                let mut out = Vec::new();
                for a in args {
                    out.extend(self.likelihood(a, has_prior)?);
                }
                Ok(out)
            }
            Some("likelihoodof") => {
                let c = self.call(l).unwrap();
                let (k, obs) = (self.resolve(c.args[0]), c.args[1]);
                self.kernel(k, obs, has_prior)
            }
            Some(h) => Err(Refusal::not_yet(Some(l), format!("a `{h}` likelihood"))),
            None => Err(Refusal::not_yet(Some(l), "this likelihood")),
        }
    }

    fn kernel(&mut self, k: NodeId, obs: NodeId, has_prior: bool) -> Result<Vec<Observation>> {
        // `likelihoodof(M, obs)` of a measure: M's free inputs are its kernel
        // inputs (as for `logdensityof(M, x)`).
        if !has_prior && matches!(self.m.type_of(k), Some(Type::Measure { .. })) {
            self.free_inputs(k)?;
            return self.measure_observations(k, obs);
        }
        let c = self
            .call(k)
            .ok_or_else(|| Refusal::not_yet(Some(k), "this kernel"))?;
        let inputs: Vec<(Symbol, Symbol)> = match (&c.inputs, self.head(k)) {
            (Some(Inputs::Spec(list)), _) => list
                .iter()
                .filter(|(_, r)| r.ns == RefNs::SelfMod)
                .map(|(n, r)| (*n, r.name))
                .collect(),
            (Some(Inputs::Auto), _) => self
                .m
                .auto_inputs_of(k)
                .unwrap_or(&[])
                .iter()
                .filter(|(_, r)| r.ns == RefNs::SelfMod)
                .map(|(n, r)| (*n, r.name))
                .collect(),
            _ => {
                return Err(Refusal::not_yet(
                    Some(k),
                    "a kernel that is not a reification",
                ));
            }
        };
        for (name, sym) in inputs {
            let sym = *self.m.alias_chain(sym).last().unwrap();
            if let Some(stan) = self.fields.get(&name).cloned() {
                self.env.insert(sym, stan);
                continue;
            }
            if has_prior {
                return Err(Refusal::not_yet(
                    Some(k),
                    format!("kernel input `{}` with no prior field", self.name(name)),
                ));
            }
            // A likelihood target: the input is a flat parameter.
            let rhs = self.m.binding(self.m.binding_by_name(sym).unwrap()).rhs;
            let stan = self.fresh(self.name(name));
            let vs = self.m.valueset_of(rhs).cloned();
            let hints = match self.call(rhs) {
                Some(c) if self.head(rhs) == Some("elementof") => self.set_hints(c.args[0]),
                _ => vec![],
            };
            // An integer or boolean kernel input is data, as a free input is.
            if self.m.type_of(rhs).is_some_and(ty_is_int) {
                let t = self.declare_data(rhs, vs.as_ref(), true, &stan, &hints)?;
                self.data.push(format!("{};", t.render(&stan)));
                self.record_decl(&stan, t);
                self.env.insert(sym, stan);
                continue;
            }
            let t = self.declare(rhs, vs.as_ref(), true, &stan, &hints)?;
            self.params.push(format!("{};", t.render(&stan)));
            self.record_decl(&stan, t);
            self.env.insert(sym, stan.clone());
            self.fields.insert(name, stan.clone());
            self.manifest_bindings.push(Some(sym));
            self.manifest.push(ManifestParam {
                stan,
                field: self.name(name).to_string(),
                input: None,
                index: None,
            });
        }
        let body = c.args[0];
        match self.head(k) {
            Some("kernelof") => {
                let out = self.resolve(body);
                let pairs: Vec<(Option<Symbol>, NodeId)> = if self.head(out) == Some("record") {
                    self.call(out)
                        .unwrap()
                        .named
                        .iter()
                        .map(|na| (Some(na.name), na.value))
                        .collect()
                } else {
                    vec![(None, body)]
                };
                let mut result = Vec::new();
                for (field, value) in pairs {
                    let (draw, law) = self.output_law(value)?;
                    let obs_value = match field {
                        Some(f) => self.obs_field(obs, f)?,
                        None => obs,
                    };
                    let name = match field {
                        Some(f) => self.name(f).to_string(),
                        None => "y".to_string(),
                    };
                    result.push(Observation {
                        draw,
                        law,
                        value: obs_value,
                        name,
                    });
                }
                Ok(result)
            }
            // A keyword `joint` body has a record variate: one observation
            // per field, as for a `kernelof` record.
            Some("functionof")
                if self.head(self.resolve(body)) == Some("joint")
                    && self
                        .call(self.resolve(body))
                        .is_some_and(|j| j.args.is_empty() && !j.named.is_empty()) =>
            {
                let parts: Vec<(Symbol, NodeId)> = self
                    .call(self.resolve(body))
                    .unwrap()
                    .named
                    .iter()
                    .map(|na| (na.name, na.value))
                    .collect();
                let mut result = Vec::new();
                for (f, law) in parts {
                    let lr = self.resolve(law);
                    if self.head(lr) == Some("joint")
                        && self.call(lr).is_some_and(|j| !j.named.is_empty())
                    {
                        return Err(Refusal::not_yet(
                            Some(law),
                            format!(
                                "a nested record observation (the keyword `joint` field `{}` inside a keyword `joint`)",
                                self.name(f)
                            ),
                        ));
                    }
                    result.push(Observation {
                        draw: None,
                        law,
                        value: self.obs_field(obs, f)?,
                        name: self.name(f).to_string(),
                    });
                }
                Ok(result)
            }
            Some("functionof") => Ok(vec![Observation {
                draw: None,
                law: body,
                value: obs,
                name: "y".to_string(),
            }]),
            _ => Err(Refusal::not_yet(Some(k), "this kernel")),
        }
    }

    /// The `elementof` bindings `m` reads, in module order.
    fn free_input_syms(&self, m: NodeId) -> Vec<Symbol> {
        self.free_input_syms_cut(m, &HashSet::new())
    }

    /// The `elementof` bindings `m` reads other than through a binding in
    /// `cut` (a kernel input's value), in module order.
    fn free_input_syms_cut(&self, m: NodeId, cut: &HashSet<Symbol>) -> Vec<Symbol> {
        let found = self.reached(m, cut);
        self.m
            .bindings()
            .map(|(_, b)| b)
            .filter(|b| found.contains(&b.name) && self.head(b.rhs) == Some("elementof"))
            .map(|b| b.name)
            .collect()
    }

    /// The bindings (alias ends) `m` reads, through bindings, up to an
    /// `elementof` and not through a binding in `cut`. A reification
    /// (`functionof`, `kernelof`, their `fn` and named-function forms) is a
    /// boundary: its body does not read its own inputs, a call sets them.
    fn reached(&self, m: NodeId, cut: &HashSet<Symbol>) -> HashSet<Symbol> {
        let mut found = HashSet::new();
        // Each node is walked once per boundary context.
        let mut cuts = vec![cut.clone()];
        let mut stack = vec![(m, 0usize)];
        let mut seen = HashSet::new();
        while let Some((id, at)) = stack.pop() {
            if !seen.insert((id, at)) {
                continue;
            }
            if let Node::Ref(r) = self.m.node(id)
                && r.ns == RefNs::SelfMod
                && let Some(b) = self
                    .m
                    .binding_by_name(*self.m.alias_chain(r.name).last().unwrap())
            {
                let name = self.m.binding(b).name;
                if cuts[at].contains(&name) {
                    continue;
                }
                found.insert(name);
                let rhs = self.m.binding(b).rhs;
                if self.head(rhs) == Some("elementof") {
                    continue;
                }
                stack.push((rhs, at));
            }
            let inputs = match self.m.node(id) {
                Node::Call(c) => match &c.inputs {
                    Some(Inputs::Spec(list)) => Some(&list[..]),
                    Some(Inputs::Auto) => self.m.auto_inputs_of(id),
                    None => None,
                },
                _ => None,
            };
            let mut inner = at;
            if let Some(list) = inputs {
                let mut c = cuts[at].clone();
                c.extend(
                    list.iter()
                        .filter(|(_, r)| r.ns == RefNs::SelfMod)
                        .map(|(_, r)| *self.m.alias_chain(r.name).last().unwrap()),
                );
                inner = match cuts.iter().position(|k| *k == c) {
                    Some(i) => i,
                    None => {
                        cuts.push(c);
                        cuts.len() - 1
                    }
                };
            }
            self.m.for_each_child(id, |k| stack.push((k, inner)));
        }
        found
    }

    /// Declare each `elementof` binding `m` reads as a flat parameter, in
    /// module order.
    fn free_inputs(&mut self, m: NodeId) -> Result<()> {
        for sym in self.free_input_syms(m) {
            self.declare_free_input(sym, false)?;
        }
        Ok(())
    }

    /// Declare the `elementof` binding `sym` once: data when it is integer or
    /// boolean (Stan has no integer parameters) or when `as_data`, else a
    /// flat parameter, a scoring input as a kernel input is.
    fn declare_free_input(&mut self, sym: Symbol, as_data: bool) -> Result<()> {
        if self.env.contains_key(&sym) {
            return Ok(());
        }
        let rhs = self.m.binding(self.m.binding_by_name(sym).unwrap()).rhs;
        // §13 "Signature": `inputs` lists every `elementof` an output reaches.
        if self.abi_target && !self.abi_syms().contains(&sym) {
            let name = self.name(sym).to_string();
            return Err(Refusal::new(
                RefusalKind::IllFormedModule,
                Some(rhs),
                format!("`{name}` outside `inputs`"),
                format!(
                    "`outputs` reads the `elementof` `{name}`, which `inputs` does not list; \
                     the module's signature is ill-formed (§13)"
                ),
            ));
        }
        let stan = self.fresh(self.name(sym));
        let vs = self.m.valueset_of(rhs).cloned();
        let hints = self.set_hints(self.call(rhs).unwrap().args[0]);
        if as_data || self.m.type_of(rhs).is_some_and(ty_is_int) {
            let t = self.declare_data(rhs, vs.as_ref(), true, &stan, &hints)?;
            self.data.push(format!("{};", t.render(&stan)));
            self.record_decl(&stan, t);
            self.env.insert(sym, stan);
            return Ok(());
        }
        let t = self.declare(rhs, vs.as_ref(), true, &stan, &hints)?;
        self.params.push(format!("{};", t.render(&stan)));
        self.record_decl(&stan, t);
        self.env.insert(sym, stan.clone());
        self.fields.insert(sym, stan.clone());
        self.manifest_bindings.push(Some(sym));
        self.manifest.push(ManifestParam {
            stan,
            field: self.name(sym).to_string(),
            input: None,
            index: None,
        });
        Ok(())
    }

    /// The alias ends of the module inputs (`inputs = (…)`).
    fn abi_syms(&self) -> HashSet<Symbol> {
        self.abi_inputs()
            .into_iter()
            .map(|w| *self.m.alias_chain(w).last().unwrap())
            .collect()
    }

    /// The inputs of a density query, declared in one order-free pass: an
    /// `elementof` any measure reads is a parameter; one only points read is
    /// data, or a parameter when the query declares `inputs`.
    fn declare_query_inputs(&mut self, measures: &[NodeId], points: &[NodeId]) -> Result<()> {
        let params: Vec<Symbol> = measures
            .iter()
            .flat_map(|&m| self.free_input_syms(m))
            .collect();
        self.declare_inputs(&params, points)
    }

    /// Declare `params` as parameters, then each `elementof` the points read
    /// (data without `inputs`). A point that restates an input it sets
    /// (`record(mu = mu)`) declares it.
    fn declare_inputs(&mut self, params: &[Symbol], points: &[NodeId]) -> Result<()> {
        let as_data = self.abi_inputs().is_empty();
        for &sym in params {
            self.declare_free_input(sym, false)?;
        }
        for &x in points {
            for sym in self.free_input_syms(x) {
                self.declare_free_input(sym, as_data)?;
            }
        }
        Ok(())
    }

    /// The measures and the observed values of a likelihood.
    fn likelihood_parts(&self, l: NodeId, measures: &mut Vec<NodeId>, obs: &mut Vec<NodeId>) {
        let l = self.resolve(l);
        let c = self.call(l);
        match (self.head(l), c) {
            (Some("joint_likelihood"), Some(c)) => {
                for &a in c.args.iter() {
                    self.likelihood_parts(a, measures, obs);
                }
            }
            (Some("likelihoodof"), Some(c)) if c.args.len() == 2 => {
                measures.push(c.args[0]);
                obs.push(c.args[1]);
            }
            _ => measures.push(l),
        }
    }

    /// The input signature of a likelihood, over every part of a
    /// `joint_likelihood`: a kernel's inputs denote their value bindings and
    /// cut what lies behind them; a measure's free `elementof` are its inputs.
    fn likelihood_signature(&self, lik: NodeId) -> Signature {
        let mut measures = Vec::new();
        self.likelihood_parts(lik, &mut measures, &mut Vec::new());
        let mut inputs: Vec<(Symbol, Vec<Symbol>)> = Vec::new();
        let mut add = |name: Symbol, sym: Option<Symbol>| {
            let at = match inputs.iter().position(|(n, _)| *n == name) {
                Some(i) => i,
                None => {
                    inputs.push((name, vec![]));
                    inputs.len() - 1
                }
            };
            if let Some(s) = sym
                && !inputs[at].1.contains(&s)
            {
                inputs[at].1.push(s);
            }
        };
        let mut free: Vec<Symbol> = Vec::new();
        for m in measures {
            let r = self.resolve(m);
            let Some(c) = self.call(r) else { continue };
            let list = match &c.inputs {
                Some(Inputs::Spec(list)) => &list[..],
                Some(Inputs::Auto) => self.m.auto_inputs_of(r).unwrap_or(&[]),
                None => {
                    if matches!(self.m.type_of(r), Some(Type::Measure { .. })) {
                        for s in self.free_input_syms(r) {
                            add(s, Some(s));
                        }
                    }
                    continue;
                }
            };
            let mut cut = HashSet::new();
            for (name, v) in list.iter() {
                let sym =
                    (v.ns == RefNs::SelfMod).then(|| *self.m.alias_chain(v.name).last().unwrap());
                cut.extend(sym);
                add(*name, sym);
            }
            for &a in c.args.iter() {
                for s in self.free_input_syms_cut(a, &cut) {
                    if !free.contains(&s) {
                        free.push(s);
                    }
                }
            }
        }
        free.retain(|s| !inputs.iter().any(|(_, d)| d.contains(s)));
        Signature { inputs, free }
    }

    /// §06: a likelihood's point has one field per input, by name.
    fn check_point_keys(&self, point: NodeId, inputs: &[(Symbol, Vec<Symbol>)]) -> Result<()> {
        let keys: Vec<Symbol> = self
            .call(point)
            .unwrap()
            .named
            .iter()
            .map(|na| na.name)
            .collect();
        let names = inputs
            .iter()
            .map(|(n, _)| format!("`{}`", self.name(*n)))
            .collect::<Vec<_>>()
            .join(", ");
        let ill = |short: String, long: String| {
            Err(Refusal::new(
                RefusalKind::IllFormedModule,
                Some(point),
                short,
                long,
            ))
        };
        if let Some(&k) = keys.iter().find(|k| !inputs.iter().any(|(n, _)| n == *k)) {
            let k = self.name(k);
            return ill(
                format!("point field `{k}` is not a likelihood input"),
                format!(
                    "the point sets `{k}`, which is not an input of the likelihood (inputs: {names}); \
                     §06 gives a likelihood point one field per input"
                ),
            );
        }
        if let Some((n, _)) = inputs.iter().find(|(n, _)| !keys.contains(n)) {
            let n = self.name(*n);
            return ill(
                format!("point lacks the likelihood input `{n}`"),
                format!(
                    "the point does not set the likelihood input `{n}` (inputs: {names}); \
                     §06 gives a likelihood point one field per input"
                ),
            );
        }
        Ok(())
    }

    /// The bindings a record point sets, each key binding every denotation of
    /// its name. A point value may read a binding the point sets only where
    /// it restates it (`record(mu = mu)`): otherwise one Stan name would
    /// carry two values.
    fn point_binding(
        &self,
        point: NodeId,
        denotes: &[(Symbol, Vec<Symbol>)],
    ) -> Result<Vec<(Symbol, Symbol)>> {
        let named: Vec<(Symbol, NodeId)> = self
            .call(point)
            .unwrap()
            .named
            .iter()
            .map(|na| (na.name, na.value))
            .collect();
        let mut out: Vec<(Symbol, Symbol)> = Vec::new();
        for &(key, _) in &named {
            for (_, ds) in denotes.iter().filter(|(n, _)| *n == key) {
                for &d in ds {
                    if let Some(&(other, _)) = out.iter().find(|b| b.1 == d) {
                        return Err(Refusal::not_yet(
                            Some(point),
                            format!(
                                "`{}` set by both `{}` and `{}`",
                                self.name(d),
                                self.name(other),
                                self.name(key)
                            ),
                        ));
                    }
                    out.push((key, d));
                }
            }
        }
        for &(key, e) in &named {
            let reads = self.reached(e, &HashSet::new());
            let restated = self
                .self_ref(e)
                .map(|s| *self.m.alias_chain(s).last().unwrap());
            for &(k, d) in &out {
                if !reads.contains(&d) || (k == key && restated == Some(d)) {
                    continue;
                }
                let whose = if k == key {
                    "the input it sets".to_string()
                } else {
                    format!("the input `{}` sets", self.name(k))
                };
                return Err(Refusal::not_yet(
                    Some(e),
                    format!(
                        "the point value of `{}` reads `{}`, {whose}",
                        self.name(key),
                        self.name(d)
                    ),
                ));
            }
        }
        Ok(out)
    }

    /// The observations of data `x` under the measure `m`: one per field of a
    /// `lawof(record(…))`, else one.
    fn measure_observations(&mut self, m: NodeId, x: NodeId) -> Result<Vec<Observation>> {
        let r = self.resolve(m);
        if self.head(r) == Some("lawof") {
            let inner = self.resolve(self.call(r).unwrap().args[0]);
            if self.head(inner) == Some("record") {
                let mut out: Vec<Observation> = Vec::new();
                for na in self.call(inner).unwrap().named.iter().copied() {
                    let (draw, law) = self.output_law(na.value)?;
                    if let Some(d) = draw
                        && out.iter().any(|o| o.draw == draw)
                    {
                        let name = self.name(d).to_string();
                        return Err(Refusal::new(
                            RefusalKind::SingularJoint,
                            Some(m),
                            format!("the joint law of `{name}`"),
                            format!(
                                "the joint law holds the draw `{name}` in two fields; it is \
                                 concentrated on a diagonal and has no density"
                            ),
                        ));
                    }
                    out.push(Observation {
                        draw,
                        law,
                        value: self.obs_field(x, na.name)?,
                        name: self.name(na.name).to_string(),
                    });
                }
                return Ok(out);
            }
            let (draw, law) = self.output_law(inner)?;
            return Ok(vec![Observation {
                draw,
                law,
                value: x,
                name: "y".to_string(),
            }]);
        }
        let name = match self.self_ref(x) {
            Some(s) => self.name(s).to_string(),
            None => "y".to_string(),
        };
        Ok(vec![Observation {
            draw: None,
            law: m,
            value: x,
            name,
        }])
    }

    /// The draw binding and law of a kernel output.
    fn output_law(&self, value: NodeId) -> Result<(Option<Symbol>, NodeId)> {
        if let Some(sym) = self.self_ref(value) {
            let sym = *self.m.alias_chain(sym).last().unwrap();
            let rhs = self.m.binding(self.m.binding_by_name(sym).unwrap()).rhs;
            if self.head(rhs) == Some("draw") {
                return Ok((Some(sym), self.call(rhs).unwrap().args[0]));
            }
        } else if self.head(value) == Some("draw") {
            return Ok((None, self.call(value).unwrap().args[0]));
        }
        Err(Refusal::not_yet(
            Some(value),
            "a kernel output that is not a draw",
        ))
    }

    fn obs_field(&self, obs: NodeId, field: Symbol) -> Result<NodeId> {
        let r = self.resolve(obs);
        if self.head(r) == Some("record")
            && let Some(na) = self
                .call(r)
                .unwrap()
                .named
                .iter()
                .find(|na| na.name == field && na.kind == NamedKind::Field)
        {
            return Ok(na.value);
        }
        Err(Refusal::not_yet(
            Some(obs),
            format!("observed data with no field `{}`", self.name(field)),
        ))
    }

    /// Emit the likelihood statements: each observed value becomes a data
    /// variable, and the output variate is `~` its law.
    fn observe(&mut self, obs: Vec<Observation>) -> Result<()> {
        let mut conditioned = Vec::new();
        let mut factors = Vec::new();
        for o in obs {
            let lhs = match self.observed_value(&o) {
                Ok(lhs) => lhs,
                // A law Stan cannot express outranks data this emitter cannot
                // write yet.
                Err(e) if !e.kind.is_never() => {
                    let probe = self.law(o.law, true);
                    return Err(match probe {
                        Err(never) if never.kind.is_never() => never,
                        _ => e,
                    });
                }
                Err(e) => return Err(e),
            };
            if let Some(d) = o.draw {
                self.env.insert(d, lhs.clone());
            }
            let support = self.truncation_support(o.law)?;
            let mut lines = self.tilde(&lhs, o.law)?;
            if let Some((lo, hi)) = support
                && let Some(g) = self.guarded(&lhs, lo.as_deref(), hi.as_deref(), lines.clone())
            {
                lines = vec![g];
            }
            let site = self.normal_site(&lhs, o.value, o.law, conditioned.len());
            factors.push((lines, site));
            if let Some(draw) = o.draw
                && (self.json.contains_key(&lhs) || self.deferred.iter().any(|d| d.stan == lhs))
            {
                conditioned.push(draw);
            }
        }
        // Allocate ordinary data and model names before adding any summaries.
        for (ordinary, site) in factors {
            if let Some(score) = site.and_then(|s| self.observed_normal(s, &conditioned)) {
                self.lik.push(score);
            } else {
                self.lik.extend(ordinary);
            }
        }
        Ok(())
    }

    /// The support an unnormalized `truncate` restricts observed data to:
    /// outside it the density is zero (spec §06 `truncate`). A data bound
    /// would make Stan reject the data instead (ruling of 2026-11-18).
    #[allow(clippy::type_complexity)]
    fn truncation_support(
        &mut self,
        law: NodeId,
    ) -> Result<Option<(Option<String>, Option<String>)>> {
        // Through every `iid` level: `iid(iid(D, m), k)` holds D's variates.
        let mut r = self.resolve(law);
        while self.head(r) == Some("iid") {
            r = self.resolve(self.call(r).unwrap().args[0]);
        }
        if self.head(r) != Some("truncate") {
            return Ok(None);
        }
        let l = self.law(r, false)?;
        Ok(l.support
            .as_ref()
            .map(|(lo, hi)| l.within(lo.as_deref(), hi.as_deref())))
    }

    /// `body` for data inside [lo, hi], `target += negative_infinity();`
    /// outside: the statements never see a value outside their law's
    /// support. One whole-value test; an array of vectors (Stan 2.39 cannot
    /// flatten it) is tested row by row into one flag.
    pub(super) fn guarded(
        &mut self,
        data: &str,
        lo: Option<&str>,
        hi: Option<&str>,
        body: Vec<String>,
    ) -> Option<String> {
        if lo.is_none() && hi.is_none() {
            return None;
        }
        let t = self.decls.get(data)?.clone();
        // A vector bound (one end per component) is compared elementwise.
        let vector = |b: Option<&str>| b.is_some_and(|b| b.starts_with('['));
        if (vector(lo) || vector(hi)) && t.array.len() + t.dims.len() != 1 {
            return None;
        }
        let test = |v: &str, whole: bool| -> String {
            let (small, big) = if whole {
                (format!("min({v})"), format!("max({v})"))
            } else {
                (v.to_string(), v.to_string())
            };
            let mut parts = Vec::new();
            if let Some(l) = lo {
                parts.push(if vector(lo) {
                    format!("min({v} - {l}) < 0")
                } else {
                    format!("{small} < {l}")
                });
            }
            if let Some(h) = hi {
                parts.push(if vector(hi) {
                    format!("max({v} - {h}) > 0")
                } else {
                    format!("{big} > {h}")
                });
            }
            parts.join(" || ")
        };
        let indent = |lines: &[String]| -> String {
            lines
                .iter()
                .flat_map(|l| l.lines())
                .map(|l| format!("  {l}\n"))
                .collect()
        };
        let branch = |cond: &str| -> String {
            let mut out = format!("if ({cond}) {{\n  target += negative_infinity();\n}}");
            if !body.is_empty() {
                out.push_str(&format!(" else {{\n{}}}", indent(&body)));
            }
            out
        };
        Some(match (t.array.len(), t.dims.len()) {
            (0, 0) => branch(&test(data, false)),
            (0, _) | (1, 0) => branch(&test(data, true)),
            (_, 0) => branch(&test(&format!("to_array_1d({data})"), true)),
            _ => {
                let i = self.fresh_index();
                let flag = self.fresh_local("outside");
                let n = t.array[0].clone();
                let rows = format!(
                    "int {flag} = 0;\nfor ({i} in 1:{n}) {{\n  {flag} = {flag} || {};\n}}",
                    test(&format!("{data}[{i}]"), true)
                );
                let inner = branch(&flag);
                self.used.remove(&i);
                self.used.remove(&flag);
                format!("{{\n{}}}", indent(&[rows, inner]))
            }
        })
    }

    fn observed_value(&mut self, o: &Observation) -> Result<String> {
        // Stan rejects a non-finite variate, where FlatPPL scores zero density.
        let v = self.resolve(o.value);
        let inf =
            |me: &Self, x: NodeId| matches!(me.m.node(x), Node::Const(c) if me.name(*c) == "inf");
        if inf(self, v) || (self.head(v) == Some("neg") && inf(self, self.call(v).unwrap().args[0]))
        {
            return Err(Refusal::not_yet(
                Some(o.value),
                "an observed value at infinity (Stan rejects a non-finite variate)",
            ));
        }
        // A data path or fixed binding prints as its own data name.
        let outer = std::mem::replace(&mut self.rows_ok, true);
        let e = self.expr(o.value);
        self.rows_ok = outer;
        let e: Ex = e?;
        if e.prec == P_ATOM && is_identifier(&e.s) {
            return Ok(e.s);
        }
        if let Some(v) = crate::data::const_json(self.m, o.value) {
            let stan = self.fresh(&o.name);
            let t = self.declare_data(o.value, None, false, &stan, &[])?;
            self.data.push(format!("{};", t.render(&stan)));
            self.record_decl(&stan, t);
            self.json.insert(stan.clone(), v);
            return Ok(stan);
        }
        let stan = self.fresh(&o.name);
        let t = self.declare(o.value, None, false, &stan, &[])?;
        self.tdata.push(format!("{} = {};", t.render(&stan), e.s));
        self.record_decl(&stan, t);
        Ok(stan)
    }

    // ---------------------------------------------------------------- laws

    /// Sampling statements for `lhs ~ measure`.
    fn tilde(&mut self, lhs: &str, measure: NodeId) -> Result<Vec<String>> {
        self.constrain_variate(lhs, measure);
        let lhs_rank = self.lhs_rank(lhs);
        let m = self.resolve(measure);
        // A mixture is `target +=` of its log density, per element of an iid
        // variate: Stan has no observation-level vectorized mixture.
        if let Some(lines) = self.mixture_statements(lhs, m)? {
            return Ok(lines);
        }
        // A chain over each observed row (spec §06 "Dependent composition").
        if self.head(m) == Some("iid")
            && let inner = self.resolve(self.call(m).unwrap().args[0])
            && self.head(inner) == Some("markovchain")
        {
            return self.markovchain_statement(lhs, inner);
        }
        // One trajectory: the step law over the lag, one statement. The two
        // chain rules dispatch on the shape alone.
        if matches!(self.head(m), Some("markovchain" | "kscan")) {
            return self.chain_statement(lhs, m);
        }
        if self.head(m) == Some("iid")
            && self.head(self.resolve(self.call(m).unwrap().args[0])) == Some("kscan")
        {
            return Err(Refusal::not_yet(
                Some(m),
                "an iid of a `kscan` (only an iid of a `markovchain` over data rows)",
            ));
        }
        if self.head(m) == Some("pushfwd") || self.derived.contains_key(&m) {
            return self.pushfwd_statement(lhs, m, lhs_rank);
        }
        // `iid(Lebesgue(S), n)` is `Lebesgue(S^n)` (§06 "Fundamental
        // measures"): density 1, no statement; the declaration carries S.
        if self.lebesgue_product(m) {
            // Observed data outside S has zero density.
            let data = self.is_data(lhs);
            if data && let Some((lo, hi)) = self.support(m)? {
                return Ok(self
                    .guarded(lhs, lo.as_deref(), hi.as_deref(), vec![])
                    .into_iter()
                    .collect());
            }
            return Ok(vec![]);
        }
        let flat = self.flat_order(lhs);
        let (law, per_element) = match self.head(m) {
            Some("Lebesgue") => return Ok(vec![]),
            Some("iid")
                if self.head(self.resolve(self.call(m).unwrap().args[0])) == Some("iid") =>
            {
                return self.nested_iid(lhs, m);
            }
            // An iid product of one law: the law's arguments are shared.
            Some("iid") => (self.law(self.call(m).unwrap().args[0], false)?, false),
            Some("broadcast") => {
                self.flat = flat;
                let law = self.law(m, true);
                self.flat = None;
                (law?, true)
            }
            // A joint of different laws: one statement per component.
            Some("joint") if self.call(m).unwrap().named.is_empty() => match self.law(m, false) {
                Ok(law) => (law, false),
                Err(e) if e.kind.is_never() => return Err(e),
                Err(_) => {
                    let mut out = Vec::new();
                    for (k, a) in self.call(m).unwrap().args.iter().copied().enumerate() {
                        let law = self.law(a, false)?;
                        if law.multivariate {
                            return Err(Refusal::not_yet(Some(m), "a joint of vector laws"));
                        }
                        self.declared_variate(lhs, &law, m)?;
                        let y = format!("{lhs}[{}]", k + 1);
                        let mut lines =
                            vec![law.statement(m, &law.lhs(&y), &law.call_text(None))?];
                        lines.extend(law.extra.iter().cloned());
                        out.extend(self.unchecked_guard(&y, false, &law, lines)?);
                    }
                    return Ok(out);
                }
            },
            _ => (self.law(m, false)?, false),
        };
        self.declared_variate(lhs, &law, m)?;
        self.integer_data(m, &law, lhs)?;
        if let Some(lines) = self.interval_mass(lhs, &law, lhs_rank, per_element)? {
            return Ok(lines);
        }
        let elementwise = per_element || self.head(m) == Some("iid");
        let shifted = law.lhs("{y}") != "{y}";
        // A loop where Stan has no vectorized form: a law that does not
        // vectorize, a truncated law, a vector law over an array of variates,
        // or a shifted variate (`{y} + 1`) of an integer array.
        let one_variate = !elementwise && (law.multivariate || lhs_rank == 0);
        // Stan vectorizes `T[L, U]` where the law has `_lcdf` and `_lccdf`,
        // which `truncated` requires, so truncation alone never loops.
        let loops = !one_variate
            && (!law.vectorized
                || law.multivariate
                || law.vector_per_element
                || (shifted && lhs_rank > 0));
        if loops {
            let i = self.fresh_index();
            let n = self.extent_of(lhs);
            let index = per_element.then_some(i.as_str());
            let stmt = format!(
                "for ({i} in 1:{n}) {{\n  {}\n}}",
                law.statement(m, &law.lhs(&format!("{lhs}[{i}]")), &law.call_text(index))?
            );
            self.used.remove(&i);
            let mut out = vec![stmt];
            out.extend(law.extra.iter().cloned());
            let out = self.unchecked_guard(lhs, lhs_rank > 0, &law, out)?;
            return Ok(self.support_guarded(lhs, &law, out));
        }
        let (target, rhs) = match (flat, &law.flat_call) {
            (Some(Flat::Row), Some(call)) if !law.multivariate => {
                (format!("to_array_1d({lhs})"), call.clone())
            }
            (Some(Flat::Col), Some(call)) if !law.multivariate => {
                (format!("to_vector({lhs})"), call.clone())
            }
            // A rank-2 variate flattened: a matrix by `to_vector`; an array
            // (`array[2, 3] int`) by `to_array_1d`, which a discrete law takes
            // as it is and a continuous one as a vector.
            _ if lhs_rank >= 2 && !law.multivariate => {
                // Stan flattens a matrix or a rank-2 array, not a rank-3
                // value or an array of vectors.
                let t = self.decls.get(lhs);
                if lhs_rank >= 3 {
                    return Err(Refusal::not_yet(Some(m), "a law over a rank-3 variate"));
                }
                if t.is_some_and(|t| t.array.len() == 1 && t.dims.len() == 1) {
                    return Err(Refusal::not_yet(
                        Some(m),
                        "a law over an array of vectors as one statement",
                    ));
                }
                let array = self.decls.get(lhs).is_some_and(|t| t.array.len() >= 2);
                let target = match (array, law.discrete) {
                    (true, true) => format!("to_array_1d({lhs})"),
                    (true, false) => format!("to_vector(to_array_1d({lhs}))"),
                    (false, _) => format!("to_vector({lhs})"),
                };
                (target, law.call_text(None))
            }
            // A user-defined `_lpdf` takes real variates: integer observations
            // of a custom row (ContinuedPoisson counts) as a vector. Stan's
            // built-in densities take the integer array itself.
            _ if law.custom
                && !law.discrete
                && self
                    .decls
                    .get(lhs)
                    .is_some_and(|t| t.is_int() && t.rank() == 1) =>
            {
                (format!("to_vector({lhs})"), law.call_text(None))
            }
            _ => (lhs.to_string(), law.call_text(None)),
        };
        let mut out = vec![law.statement(m, &law.lhs(&target), &rhs)?];
        out.extend(law.extra.iter().cloned());
        let out = self.unchecked_guard(lhs, lhs_rank > 0, &law, out)?;
        Ok(self.support_guarded(lhs, &law, out))
    }

    /// `lines` behind a test of `expr` against the law's support, for a row
    /// whose Stan function does not enforce it (`unchecked_support`: von
    /// Mises, `lkj_corr_cholesky`): outside it the density is zero. A
    /// variate whose declaration states the support needs no test.
    pub(super) fn unchecked_guard(
        &self,
        expr: &str,
        array: bool,
        law: &Law,
        lines: Vec<String>,
    ) -> Result<Vec<String>> {
        // A two-sided `T[L, U]`, its ends clamped into the support, already
        // gives -inf outside it.
        if !law.unchecked_support || matches!(law.trunc, Some((Some(_), Some(_)))) {
            return Ok(lines);
        }
        let (lo, hi) = &law.bounds;
        if self.decls.get(expr).is_some_and(|t| {
            &t.bounds.lower == lo && &t.bounds.upper == hi && t.support == law.support_type
        }) {
            return Ok(lines);
        }
        let Some(test) = self.unchecked_test(expr, array, law)? else {
            return Ok(lines);
        };
        let body: String = lines
            .iter()
            .flat_map(|l| l.lines())
            .map(|l| format!("  {l}\n"))
            .collect();
        Ok(vec![format!(
            "if ({test}) {{\n  target += negative_infinity();\n}} else {{\n{body}}}"
        )])
    }

    /// The test that `y` lies outside an `unchecked_support` law's support:
    /// its `variate` bounds, or for `cholesky_factor_corr` rows that are not
    /// unit length (`lkj_corr_cholesky_lpdf` checks the triangle and the
    /// diagonal itself, not the row norms).
    pub(super) fn unchecked_test(&self, y: &str, array: bool, law: &Law) -> Result<Option<String>> {
        if law.support_type == Some("cholesky_factor_corr") {
            // The variate is itself a matrix: an array is one of rank > 2.
            if self.decls.get(y).is_some_and(|t| t.rank() > 2) {
                return Err(Refusal::not_yet(
                    None,
                    "an array of Cholesky correlation factors not declared as such",
                ));
            }
            // 1e-8 is Stan's own constraint tolerance (`CONSTRAINT_TOLERANCE`),
            // with which it checks a declared `cholesky_factor_corr`.
            return Ok(Some(format!("max(abs(rows_dot_self({y}) - 1)) > 1e-8")));
        }
        Ok(Self::outside_test(
            y,
            array,
            law.bounds.0.as_deref(),
            law.bounds.1.as_deref(),
        ))
    }

    /// `T[lo, hi]` on data reaching outside the law's support: Stan's `_lpdf`
    /// raises on such data before `T` tests it, where the density is zero.
    fn support_guarded(&mut self, lhs: &str, law: &Law, lines: Vec<String>) -> Vec<String> {
        let Some((lo, hi)) = &law.trunc else {
            return lines;
        };
        let (wl, wh) = law.within(lo.as_deref(), hi.as_deref());
        if (&wl, &wh) == (lo, hi) || !self.is_data(lhs) {
            return lines;
        }
        match self.guarded(lhs, wl.as_deref(), wh.as_deref(), lines.clone()) {
            Some(g) => vec![g],
            None => lines,
        }
    }

    /// `iid(iid(D, m), k)`, D^⊗(k·m) (spec §06 "Independent composition"),
    /// is `iid(D, [k, m])` over the declared nesting. An `array[k, m]` or a
    /// `matrix` variate is one flattened statement. Stan 2.39 cannot flatten
    /// `array[k] vector[m]`, so that variate is one statement per row.
    fn nested_iid(&mut self, lhs: &str, m: NodeId) -> Result<Vec<String>> {
        let inner = self.resolve(self.call(m).unwrap().args[0]);
        let d = self.call(inner).unwrap().args[0];
        if self.head(self.resolve(d)) == Some("iid") {
            return Err(Refusal::not_yet(Some(m), "an iid nested more than twice"));
        }
        let law = self.law(d, false)?;
        self.declared_variate(lhs, &law, m)?;
        self.integer_data(m, &law, lhs)?;
        if law.multivariate
            || !law.vectorized
            || law.vector_per_element
            || law.lhs("{y}") != "{y}"
            || !law.extra.is_empty()
            || law.unnormalized
        {
            return Err(Refusal::not_yet(Some(m), "a nested iid of this law"));
        }
        let rhs = law.rhs(None);
        let Some(t) = self.decls.get(lhs).cloned() else {
            return Err(Refusal::not_yet(Some(m), "a nested iid of this variate"));
        };
        match (self.flat_order(lhs), t.array.len(), t.base, t.dims.len()) {
            (Some(Flat::Row), ..) => Ok(vec![format!("to_array_1d({lhs}) ~ {rhs};")]),
            (Some(Flat::Col), ..) => Ok(vec![format!("to_vector({lhs}) ~ {rhs};")]),
            (None, 1, "vector", 1) => {
                let i = self.fresh_index();
                let n = t.array[0].clone();
                self.used.remove(&i);
                Ok(vec![format!(
                    "for ({i} in 1:{n}) {{\n  {lhs}[{i}] ~ {rhs};\n}}"
                )])
            }
            _ => Err(Refusal::not_yet(Some(m), "a nested iid of this variate")),
        }
    }

    /// A law whose restriction the declaration carries is exact only on a
    /// parameter declared from its sets: elsewhere the restriction is lost.
    /// Observed data under a discrete law must be integer-typed: Stan's
    /// `_lpmf` takes `int` arguments only, and a real value is not cast
    /// (integers by type, not value).
    pub(super) fn integer_data(&self, at: NodeId, law: &Law, y: &str) -> Result<()> {
        // `to_vector(y)` or `to_array_1d(y)`: the wrapper first, then the index.
        let mut base = y;
        for w in ["to_vector(", "to_array_1d("] {
            if let Some(inner) = base.strip_prefix(w) {
                base = inner.strip_suffix(')').unwrap_or(inner);
            }
        }
        let base = base.split('[').next().unwrap_or(base);
        if law.discrete && self.decls.get(base).is_some_and(|t| !t.is_int()) {
            // A real scoring input is a parameter, and Stan has no integer
            // parameters to cast it to.
            let what = if self.is_data(base) { "data" } else { "input" };
            return Err(Refusal::not_yet(
                Some(at),
                format!("real-typed {what} under the discrete law `{}`", law.flatppl),
            ));
        }
        Ok(())
    }

    fn declared_variate(&self, lhs: &str, law: &Law, at: NodeId) -> Result<()> {
        if law.declared && !self.set_declared.contains(lhs) {
            return Err(Refusal::not_yet(
                Some(at),
                "an ordered restriction of a variate that is not a parameter",
            ));
        }
        Ok(())
    }

    /// The law of a `normalize(…)`: an unnormalized row inside it has no
    /// mass the emitter can divide by (ContinuedPoisson's Z(λ) has no closed
    /// form), so its density is not the normalized one.
    pub(super) fn normalized(at: NodeId, law: Law) -> Result<Law> {
        if law.unnormalized {
            return Err(Refusal::not_yet(
                Some(at),
                format!(
                    "`normalize` of a measure holding the unnormalized law `{}`",
                    law.flatppl
                ),
            ));
        }
        Ok(law)
    }

    /// `y ~ iid(markovchain(…), M)` at observed rows: one statement over
    /// the flattened rows. A chain over parameters has no observed lag.
    fn markovchain_statement(&mut self, lhs: &str, mc: NodeId) -> Result<Vec<String>> {
        let data = self.is_data(lhs);
        if !data {
            return Err(Refusal::not_yet(Some(mc), "a markovchain as a prior"));
        }
        let rl = self.markovchain_row_law(lhs, mc, None)?;
        let n = self
            .decls
            .get(lhs)
            .and_then(|t| t.array.first().cloned())
            .unwrap_or_else(|| format!("size({lhs})"));
        Ok(vec![rl.statement(lhs, &n)])
    }

    fn fresh_index(&mut self) -> String {
        for c in ["i", "j", "k", "n"] {
            if !self.used.contains(c) && !self.reserved.contains(c) {
                self.used.insert(c.to_string());
                return c.to_string();
            }
        }
        self.fresh_local("idx")
    }

    /// The declared outer extent of a Stan variable.
    fn extent_of(&self, name: &str) -> String {
        self.decls
            .get(name)
            .and_then(|t: &StanType| t.array.first().or(t.dims.first()).cloned())
            .unwrap_or_else(|| format!("size({name})"))
    }

    /// `D_lpdf(x | args)` (`_lpmf` for a discrete law) from the law's
    /// `distributions.ron` row, the same text `~` uses.
    /// The law of a measure, outside a broadcast.
    pub(super) fn law_any(&mut self, m: NodeId) -> Result<Law> {
        let broadcast = self.head(self.resolve(m)) == Some("broadcast");
        self.law(m, broadcast)
    }

    /// A table law whose array arguments vary per element ([`LOOP`]).
    pub(super) fn table_law_per(&mut self, at: NodeId, name: &str, args: &[NodeId]) -> Result<Law> {
        self.table_law(at, name, args, true)
    }

    /// `D_lpdf(y | args)` (`_lpmf` for a discrete law) for a variate already
    /// printed, per element at `index` when given.
    pub(super) fn law_density(
        &mut self,
        at: NodeId,
        law: &Law,
        y: &str,
        index: Option<&str>,
    ) -> Result<String> {
        let call = match index {
            Some(i) => law.indexed.replace(LOOP, i),
            None => law.call.clone(),
        };
        let open = call
            .find('(')
            .ok_or_else(|| Refusal::not_yet(Some(at), "the log density of this law"))?;
        let (name, args) = (&call[..open], &call[open + 1..call.len() - 1]);
        let suffix = if law.discrete { "_lpmf" } else { "_lpdf" };
        let y = law.lhs(y);
        Ok(if args.is_empty() {
            format!("{name}{suffix}({y})")
        } else {
            format!("{name}{suffix}({y} | {args})")
        })
    }

    pub(super) fn density_call(&mut self, id: NodeId, m: NodeId, x: NodeId) -> Result<String> {
        let mut law = self.law(m, false)?;
        let restricted = law
            .trunc
            .take()
            .map(|t| (t, true))
            .or(law.support.take().map(|t| (t, false)));
        if restricted.is_some() && self.shape(x).is_none_or(|s| s.rank != 0) {
            return Err(Refusal::not_yet(
                Some(id),
                "logdensityof of a truncated law over an array",
            ));
        }
        if law.declared {
            return Err(Refusal::not_yet(
                Some(id),
                "logdensityof of an ordered restriction",
            ));
        }
        // Over an array, `D_lpdf(x | …)` is the sum only where Stan
        // vectorizes the law.
        if self.shape(x).is_some_and(|s| s.rank >= 1) && !law.vectorized && !law.multivariate {
            return Err(Refusal::not_yet(
                Some(id),
                "logdensityof over an array under a law Stan does not vectorize",
            ));
        }
        let x = if law.discrete {
            self.expr(x)?
        } else {
            self.real(x)?
        };
        // A shifted variate (`{y} + 1`, Categorical0) is exact on the
        // integers: a bijection under counting measure, with no Jacobian.
        let y = if law.lhs("{y}") == "{y}" {
            x.s.clone()
        } else {
            x.at(P_ATOM)
        };
        let lp = self.law_density(id, &law, &y, None)?;
        // A truncation is zero outside `[lo, hi]`; a normalized one divides
        // by its mass there (§06 "Normalization and mass").
        let Some(((lo, hi), normalized)) = restricted else {
            return Ok(lp);
        };
        let lp = if normalized {
            let mass = self.interval_lmass(&law, lo.as_deref(), hi.as_deref());
            format!("{lp} - {mass}")
        } else {
            lp
        };
        let point = x.at(P_ATOM);
        Ok(
            match Self::outside_test(
                &point,
                false,
                law.within(lo.as_deref(), hi.as_deref()).0.as_deref(),
                law.within(lo.as_deref(), hi.as_deref()).1.as_deref(),
            ) {
                Some(out) => format!("({out} ? negative_infinity() : {lp})"),
                None => lp,
            },
        )
    }

    /// A parameter declared by the sets a record-level `truncate` gives it:
    /// `ordered(K)` → `ordered[K]`, with a positive set → `positive_ordered[K]`,
    /// interval sets → bounds. They take precedence over the inferred value
    /// set, which cannot state a positive ordered set.
    fn declare_from_sets(
        &mut self,
        ty: &Type,
        f: &PriorField,
        stan: &str,
        hints: &[Option<NodeId>],
    ) -> Result<StanType> {
        let mut t = self.declare_ty(ty, None, true, f.node, stan, hints)?;
        let (mut ordered, mut positive) = (false, false);
        let mut lowers: Vec<String> = Vec::new();
        let mut uppers: Vec<String> = Vec::new();
        for &s in &f.sets {
            let mut e = self.resolve(s);
            while self.head(e) == Some("cartpow") {
                e = self.resolve(self.call(e).unwrap().args[0]);
            }
            match (self.head(e), self.m.node(e)) {
                (Some("ordered"), _) => ordered = true,
                (Some("interval"), _) => {
                    let (lo, hi) = self.interval(e)?;
                    lowers.extend(lo);
                    uppers.extend(hi);
                }
                (_, Node::Const(c)) if matches!(self.name(*c), "posreals" | "nonnegreals") => {
                    positive = true;
                }
                (_, Node::Const(c)) if self.name(*c) == "reals" => {}
                _ => {
                    return Err(Refusal::not_yet(
                        Some(s),
                        "a truncation of a Lebesgue prior to this set",
                    ));
                }
            }
        }
        lowers.dedup();
        uppers.dedup();
        if ordered {
            // Stan has no bounded ordered type.
            if !lowers.is_empty() || !uppers.is_empty() {
                return Err(Refusal::not_yet(
                    Some(f.node),
                    "an ordered parameter with interval bounds",
                ));
            }
            t.base = if positive {
                "positive_ordered"
            } else {
                "ordered"
            };
            t.bounds = Default::default();
            return Ok(t);
        }
        if lowers.len() > 1 || uppers.len() > 1 {
            return Err(Refusal::not_yet(
                Some(f.node),
                "the intersection of different bounds",
            ));
        }
        // The lower bound would be max(0, lo), which this emitter does not write.
        if positive && !lowers.is_empty() {
            return Err(Refusal::not_yet(
                Some(f.node),
                "a positive set with an interval lower bound",
            ));
        }
        if positive {
            lowers.push("0".into());
        }
        t.bounds.lower = lowers.pop();
        t.bounds.upper = uppers.pop();
        Ok(t)
    }

    /// The flattening a rank-2 variate takes in one vectorized statement.
    fn flat_order(&self, name: &str) -> Option<Flat> {
        let t = self.decls.get(name)?;
        match (t.array.len(), t.base, t.dims.len()) {
            (2, _, 0) => Some(Flat::Row),
            (0, "matrix", 2) => Some(Flat::Col),
            _ => None,
        }
    }

    fn lhs_rank(&self, name: &str) -> usize {
        self.decls.get(name).map(StanType::rank).unwrap_or(0)
    }

    /// The Stan law of a FlatPPL measure. `broadcast` maps a law over arrays.
    pub(super) fn law(&mut self, m: NodeId, broadcast: bool) -> Result<Law> {
        let m = self.resolve(m);
        let head = self
            .head(m)
            .ok_or_else(|| Refusal::not_yet(Some(m), "this measure"))?;
        match head {
            "normalize" => {
                let inner = self.resolve(self.call(m).unwrap().args[0]);
                if let Some(dims) = self.weighted_lebesgue_dims(inner)
                    && dims > 1
                {
                    return Err(Refusal::new(
                        RefusalKind::NumericNormalizer,
                        Some(m),
                        "normalize",
                        format!(
                            "`normalize` of a weighted Lebesgue measure on a {dims}-dimensional \
                             support has no closed-form normalizer"
                        ),
                    ));
                }
                if self.head(inner) == Some("weighted") {
                    let law = self.constant_weight_law(m, inner, broadcast)?;
                    return Self::normalized(m, law);
                }
                if self.head(inner) != Some("truncate") {
                    return Err(Refusal::not_yet(Some(m), "normalize of this measure"));
                }
                let c = self.call(inner).unwrap();
                if let Some(law) = self.ordered_iid(m, c.args[0], c.args[1])? {
                    return Self::normalized(m, law);
                }
                // The mass of the ordered cone is 1/K! only for exchangeable
                // components; P(X1 < … < XK) has no closed form otherwise.
                if self.head(self.resolve(c.args[1])) == Some("ordered") {
                    return Err(Refusal::new(
                        RefusalKind::NumericNormalizer,
                        Some(m),
                        "normalize",
                        "`normalize` of an ordered restriction of non-identical components has no \
                         closed-form normalizer",
                    ));
                }
                let (base, set) = (c.args[0], c.args[1]);
                let integers = matches!(self.m.node(self.resolve(set)), Node::Const(k)
                    if matches!(self.name(*k), "posintegers" | "nonnegintegers"));
                if integers && !self.law_any(base)?.discrete {
                    return Err(Refusal::new(
                        RefusalKind::ZeroMass,
                        Some(inner),
                        "normalize",
                        "`normalize` of a continuous law restricted to a set of integers: the \
                         restriction has total mass 0, so the normalized measure is undefined",
                    ));
                }
                let law = self.truncated(inner)?;
                // Stan's `T[L, U]` of a discrete law takes integer ends only.
                if law.discrete && !self.integer_ends(set) {
                    return Err(Refusal::not_yet(
                        Some(inner),
                        "a discrete law truncated at a non-integer end",
                    ));
                }
                Self::normalized(m, law)
            }
            // `tilt(iid(D, K), Lebesgue(ordered(K)))`: D's density on the ordered
            // cone, normalized (§06 "Normalization and mass").
            "tilt" => {
                let c = self.call(m).unwrap();
                let base = self.resolve(c.args[1]);
                let set = (self.head(base) == Some("Lebesgue"))
                    .then(|| self.call(base).unwrap())
                    .and_then(|b| {
                        b.args
                            .first()
                            .copied()
                            .or_else(|| b.named.first().map(|n| n.value))
                    });
                match set {
                    Some(set) => match self.ordered_iid(m, c.args[0], set)? {
                        Some(law) => Ok(law),
                        None => Err(Refusal::not_yet(Some(m), "the law `tilt` in this form")),
                    },
                    None => Err(Refusal::not_yet(Some(m), "the law `tilt` in this form")),
                }
            }
            // An unnormalized restriction keeps the density; the declaration
            // carries the support. A chain with an `ordered` set declares the
            // parameter's type (`ordered`, `positive_ordered`, see `law_sets`).
            "truncate" => {
                if !self.law_sets(m).is_empty() {
                    let mut inner = m;
                    while self.head(inner) == Some("truncate")
                        && self.declarable_set(self.call(inner).unwrap().args[1])
                    {
                        inner = self.resolve(self.call(inner).unwrap().args[0]);
                    }
                    let mut law = self.law(inner, false)?;
                    law.declared = true;
                    return Ok(law);
                }
                let mut law = self.truncated(m)?;
                law.support = law.trunc.take();
                Ok(law)
            }
            "joint" => self.joint_law(m),
            "broadcast" if broadcast => {
                let c = self.call(m).unwrap();
                // A standard-module law broadcast over arrays (`hep.ContinuedPoisson.(rate)`):
                // its table row, as for a built-in law.
                if let (Node::Ref(r), Some(module)) =
                    (self.m.node(c.args[0]), self.standard_module(c.args[0]))
                {
                    if !c.named.is_empty() {
                        return Err(Refusal::not_yet(
                            Some(m),
                            "a module law broadcast with keyword arguments",
                        ));
                    }
                    let member = self.name(r.name).to_string();
                    let args = c.args[1..].to_vec();
                    return self.table_law_in(m, &member, Some(&module), &args, true);
                }
                let Node::Const(f) = self.m.node(c.args[0]) else {
                    // A user law: one vectorized statement when it can be,
                    // else its body per element, in a loop.
                    if let Some((body, ctx)) = self.vectorized_user_law(m)? {
                        let outer = self.fctx.replace(ctx);
                        let law = self.law(body, false);
                        self.fctx = outer;
                        if let Ok(law) = law
                            && law.vectorized
                            && !law.multivariate
                        {
                            return Ok(law);
                        }
                    }
                    let (body, ctx) = self.user_law_body(m, LOOP)?;
                    let outer = self.fctx.replace(ctx);
                    let law = self.law(body, false);
                    let ctx = std::mem::replace(&mut self.fctx, outer);
                    if ctx.is_some_and(|c| !c.lines.is_empty()) {
                        return Err(Refusal::not_yet(
                            Some(m),
                            "a user law whose arguments need statements",
                        ));
                    }
                    let mut law = law?;
                    law.vectorized = false;
                    return Ok(law);
                };
                let name = self.name(*f);
                if !c.named.is_empty() {
                    return Err(Refusal::not_yet(
                        Some(m),
                        format!("{name}.(…) with keyword arguments"),
                    ));
                }
                let args = c.args[1..].to_vec();
                self.table_law(m, name, &args, true)
            }
            "kchain" => {
                let first = self.resolve(self.call(m).unwrap().args[0]);
                if self.finite_discrete(first) {
                    return Err(Refusal::not_yet(
                        Some(m),
                        "kchain over a finite discrete variate",
                    ));
                }
                Err(Refusal::new(
                    RefusalKind::KchainMarginal,
                    Some(m),
                    "kchain",
                    "`kchain` marginalizes its continuous intermediate variate; Stan has no \
                     marginalization integral",
                ))
            }
            // A bijective `pushfwd` is one scalar variate's statement (see
            // `pushfwd_statement`); as a law inside another it would lose its
            // Jacobian.
            "pushfwd" => {
                self.pushfwd_projection(m)?;
                Err(Refusal::not_yet(Some(m), "pushfwd"))
            }
            "%call" => {
                let c = self.call(m).unwrap();
                let flatppl_core::CallHead::User(callee) = c.head else {
                    unreachable!()
                };
                let (Node::Ref(r), Some(module)) =
                    (self.m.node(callee), self.standard_module(callee))
                else {
                    return Err(Refusal::not_yet(Some(m), "a law from a user callable"));
                };
                let member = self.name(r.name);
                let args = self.law_args(member, m)?;
                self.table_law_in(m, member, Some(&module), &args, false)
            }
            // `locscale(normalize(truncate(D, I)), m, s)` with s > 0 is
            // `normalize(truncate(locscale(D, m, s), m + s·I))`: D's
            // `locscale` row truncated to the mapped interval.
            "locscale" if let Some((args, lo, hi)) = self.locscale_truncation(m)? => {
                let mut law = self.locscale_law(m, args)?;
                if law.multivariate || !law.lcdf {
                    return Err(Refusal::not_yet(
                        Some(m),
                        "locscale of a truncated law with no Stan CDF",
                    ));
                }
                law.trunc = Some((lo, hi));
                Ok(law)
            }
            "locscale" => {
                let c = self.call(m).unwrap();
                match c.args[..] {
                    [d, loc, scale] if c.named.is_empty() => self.locscale_law(m, [d, loc, scale]),
                    _ => Err(Refusal::not_yet(Some(m), "locscale in this form")),
                }
            }
            name => {
                let args = if name.starts_with(char::is_uppercase) {
                    self.law_args(name, m)?
                } else {
                    let c = self.call(m).unwrap();
                    if !c.named.is_empty() {
                        return Err(Refusal::not_yet(
                            Some(m),
                            format!("{name} with keyword arguments"),
                        ));
                    }
                    c.args.to_vec()
                };
                self.table_law(m, name, &args, false)
            }
        }
    }

    /// The support dimension of `weighted(f, Lebesgue(S))` or `logweighted(…)`.
    fn weighted_lebesgue_dims(&self, m: NodeId) -> Option<usize> {
        if !matches!(self.head(m), Some("weighted" | "logweighted")) {
            return None;
        }
        let c = self.call(m)?;
        let base = self.resolve(*c.args.get(1)?);
        if self.head(base) != Some("Lebesgue") {
            return None;
        }
        let bc = self.call(base)?;
        let support = bc
            .args
            .first()
            .copied()
            .or_else(|| bc.named.first().map(|n| n.value))?;
        let support = self.resolve(support);
        Some(match self.head(support) {
            Some("cartprod") => {
                let s = self.call(support)?;
                s.args.len() + s.named.len()
            }
            _ => 1,
        })
    }

    /// `true` for a law with finite discrete support (`Bernoulli`, `Categorical`).
    fn finite_discrete(&self, m: NodeId) -> bool {
        let m = match self.head(m) {
            Some("lawof") => self
                .call(m)
                .and_then(|c| self.self_ref(c.args[0]))
                .and_then(|s| {
                    let rhs = self.m.binding(self.m.binding_by_name(s)?).rhs;
                    (self.head(rhs) == Some("draw"))
                        .then(|| self.resolve(self.call(rhs).unwrap().args[0]))
                }),
            _ => Some(m),
        };
        m.is_some_and(|m| {
            matches!(
                self.head(m),
                Some("Bernoulli" | "Categorical" | "Categorical0")
            )
        })
    }

    /// The interval a prior law restricts its variate to, when it names one.
    pub(super) fn support(
        &mut self,
        law: NodeId,
    ) -> Result<Option<(Option<String>, Option<String>)>> {
        let m = self.resolve(law);
        if self.derived.contains_key(&m) {
            return self.pushfwd_support(m);
        }
        match self.head(m) {
            // A positional joint: vector bounds, one end per component.
            Some("joint") if self.call(m).unwrap().named.is_empty() => {
                let mut los = Vec::new();
                let mut his = Vec::new();
                for a in self.call(m).unwrap().args.iter().copied() {
                    match self.support(a)? {
                        Some((lo, hi)) => {
                            los.push(lo);
                            his.push(hi);
                        }
                        None => return Ok(None),
                    }
                }
                let side = |v: Vec<Option<String>>| -> Option<String> {
                    let v: Option<Vec<String>> = v.into_iter().collect();
                    v.map(|v| format!("[{}]'", v.join(", ")))
                };
                Ok(Some((side(los), side(his))))
            }
            // A declared set is the declaration's own (see `declare_from_sets`).
            Some("truncate") if !self.law_sets(m).is_empty() => Ok(None),
            Some("iid" | "normalize") => self.support(self.call(m).unwrap().args[0]),
            // The interval intersected with the base law's support: a bound
            // outside it would admit values with zero density.
            Some("truncate") => {
                let (base, set) = {
                    let c = self.call(m).unwrap();
                    (c.args[0], self.resolve(c.args[1]))
                };
                let (mut lo, mut hi) = self.interval(set)?;
                // The base law's literal support from its table rows, where
                // they all agree: rendering the law here would declare its
                // data early.
                let rows = self
                    .head(self.resolve(base))
                    .map(|h| tables::law_rows_by_head(h, None))
                    .unwrap_or_default();
                let ends: Vec<_> = rows
                    .iter()
                    .map(|r| {
                        (
                            variate_bounds(&r.variate),
                            r.variate.starts_with("int"),
                            r.floor.is_empty(),
                        )
                    })
                    .collect();
                if let Some(first) = ends.first()
                    && first.2
                    && ends.iter().all(|e| e == first)
                {
                    let ((floor, ceil), discrete, _) = first;
                    let num = |b: &Option<String>| b.clone().filter(|v| v.parse::<f64>().is_ok());
                    if let (Some(x), Some(f)) = (&lo, num(floor)) {
                        lo = Some(pick(x, &f, true, *discrete));
                    }
                    if let (Some(x), Some(c)) = (&hi, num(ceil)) {
                        hi = Some(pick(x, &c, false, *discrete));
                    }
                }
                Ok(Some((lo, hi)))
            }
            Some("Uniform") => {
                let s = self.resolve(self.law_args("Uniform", m)?[0]);
                self.interval(s).map(Some)
            }
            Some("pushfwd") => self.pushfwd_support(m),
            // Each trajectory element ranges over the step law's support.
            Some("markovchain" | "kscan") => self.chain_support(m),
            Some("%call") => self.module_law_support(m),
            Some("locscale") => Ok(self.locscale_truncation(m)?.map(|(_, lo, hi)| (lo, hi))),
            // `Lebesgue(S)`: its variate ranges over S.
            Some("Lebesgue") => {
                let c = self.call(m).unwrap();
                match c
                    .args
                    .first()
                    .copied()
                    .or_else(|| c.named.first().map(|n| n.value))
                {
                    // A set with no interval form (`cartpow`, `ordered`) is
                    // the declaration's own, from the value set.
                    Some(set) => {
                        let set = self.resolve(set);
                        Ok(self.interval(set).ok())
                    }
                    None => Ok(None),
                }
            }
            _ => Ok(self.unchecked_bounds(m)),
        }
    }

    /// The constrained Stan type a prior law's support is (its rows'
    /// `variate`), the variate axes it spans, and the static size a
    /// `sizeof(a)[i]` size gives where inference leaves the extent open.
    fn law_support(&self, law: NodeId) -> Result<Option<(&'static str, usize, Option<u32>)>> {
        let mut m = self.resolve(law);
        while self.head(m) == Some("iid") {
            m = self.resolve(self.call(m).unwrap().args[0]);
        }
        let Some(head) = self.head(m) else {
            return Ok(None);
        };
        if head == "broadcast" {
            let constrained = match self.m.node(self.call(m).unwrap().args[0]) {
                Node::Const(f) => tables::law_rows_by_head(self.name(*f), None)
                    .iter()
                    .any(|r| {
                        r.variate
                            .split_once('[')
                            .is_some_and(|(b, _)| crate::stan_type::support_type(b).is_some())
                    }),
                _ => false,
            };
            if constrained {
                return Err(Refusal::not_yet(
                    Some(m),
                    "a broadcast of a law on a constrained set",
                ));
            }
            return Ok(None);
        }
        let Some(v) = self.row_variate(head, m) else {
            return Ok(None);
        };
        let Some(s) = v.support else {
            return Ok(None);
        };
        let n = match v.size {
            RowSize::Extent(a, i) => match self.ty(a) {
                Some(Type::Array { shape, .. }) if i >= 1 => match shape.get(i - 1) {
                    Some(Dim::Static(n)) => Some(*n),
                    _ => None,
                },
                _ => None,
            }
            .map(Some)
            .ok_or_else(|| {
                Refusal::not_yet(
                    Some(m),
                    format!("a `{head}` variate whose size `{}` is not static", v.expr),
                )
            })?,
            _ => None,
        };
        Ok(Some((s, v.axes, n)))
    }

    /// The variate bounds of a law whose Stan function does not enforce them
    /// (`unchecked_support` rows: von Mises), for `D(…)` and `D.(…)`.
    fn unchecked_bounds(&self, m: NodeId) -> Option<(Option<String>, Option<String>)> {
        let head = match self.head(m)? {
            "iid" => return self.unchecked_bounds(self.resolve(self.call(m)?.args[0])),
            "broadcast" => match self.m.node(self.call(m)?.args[0]) {
                Node::Const(f) => self.name(*f).to_string(),
                _ => return None,
            },
            h => h.to_string(),
        };
        let rows = tables::law_rows_by_head(&head, None);
        if rows.is_empty() || !rows.iter().all(|r| r.unchecked_support) {
            return None;
        }
        Some(variate_bounds(&rows[0].variate))
    }

    /// The variate bounds of a standard-module law's row (`Argus`:
    /// `lower=0, upper={resonance}`), its parameters filled in: a parameter
    /// drawn from it is declared on its support.
    fn module_law_support(
        &mut self,
        m: NodeId,
    ) -> Result<Option<(Option<String>, Option<String>)>> {
        let c = self.call(m).unwrap();
        let flatppl_core::CallHead::User(callee) = c.head else {
            return Ok(None);
        };
        let (Node::Ref(r), Some(module)) = (self.m.node(callee), self.standard_module(callee))
        else {
            return Ok(None);
        };
        let member = self.name(r.name).to_string();
        let Some(row) = tables::law_rows_by_head(&member, Some(&module))
            .into_iter()
            .next()
        else {
            return Ok(None);
        };
        let (lo, hi) = variate_bounds(&row.variate);
        if lo.is_none() && hi.is_none() {
            return Ok(None);
        }
        let args = self.law_args(&member, m)?;
        let fill = |me: &mut Self, b: Option<String>| -> Result<Option<String>> {
            let Some(mut b) = b else {
                return Ok(None);
            };
            for (k, p) in row.params.iter().enumerate() {
                let key = format!("{{{p}}}");
                if b.contains(&key) {
                    let a = args.get(k).copied().ok_or_else(|| {
                        Refusal::not_yet(Some(m), format!("{member} with a missing argument"))
                    })?;
                    // A declaration bound reads data or an earlier
                    // parameter, not a model-block value.
                    if !me.declarable_bound(a) {
                        return Err(Refusal::not_yet(
                            Some(m),
                            format!(
                                "a `{member}` parameter whose support bound `{p}` is a derived value"
                            ),
                        ));
                    }
                    b = b.replace(&key, &me.real(a)?.s);
                }
            }
            Ok(Some(b))
        };
        let lo = fill(self, lo)?;
        let hi = fill(self, hi)?;
        Ok(Some((lo, hi)))
    }

    /// A value a parameter declaration may bound by: a literal, fixed data,
    /// or a drawn parameter (declared in field order before its readers).
    fn declarable_bound(&self, a: NodeId) -> bool {
        if self.literal_number(a).is_some() {
            return true;
        }
        let r = self.resolve(a);
        if self.m.phase_of(r) == Some(flatppl_core::Phase::Fixed) {
            return true;
        }
        // A module binding drawn by `~`: a parameter of the prior.
        self.self_ref(a)
            .and_then(|s| self.m.binding_by_name(s))
            .is_some_and(|b| self.head(self.m.binding(b).rhs) == Some("draw"))
    }

    /// `iid(D, K)` restricted to `ordered(K)` and normalized: D's own
    /// statement on an `ordered[K]` parameter plus `target += lgamma(K + 1);`.
    /// K iid continuous draws are exchangeable and tie with probability 0, so
    /// the ordered cone has mass 1/K! and the density gains log K!.
    fn ordered_iid(&mut self, at: NodeId, measure: NodeId, set: NodeId) -> Result<Option<Law>> {
        let (iid, set) = (self.resolve(measure), self.resolve(set));
        if self.head(iid) != Some("iid") || self.head(set) != Some("ordered") {
            return Ok(None);
        }
        let ic = self.call(iid).unwrap();
        let (d, k) = (ic.args[0], ic.args[1]);
        let k_set = self.call(set).unwrap().args[0];
        let (Some(kl), Some(ks)) = (self.lin_int(k), self.lin_int(k_set)) else {
            return Ok(None);
        };
        if kl != ks {
            return Ok(None);
        }
        let mut law = self.law(d, false)?;
        if law.multivariate
            || law.discrete
            || law.trunc.is_some()
            || law.support.is_some()
            || law.declared
        {
            return Err(Refusal::not_yet(
                Some(at),
                "an ordered restriction of this law",
            ));
        }
        law.extra
            .push(format!("target += lgamma({});", kl.add(1).print()));
        law.declared = true;
        Ok(Some(law))
    }

    /// `ordered(K)`, `posreals`, `nonnegreals` or `cartpow` of a positive set:
    /// a set the parameter's declared type carries (see `declare_from_sets`).
    fn declarable_set(&self, s: NodeId) -> bool {
        let mut e = self.resolve(s);
        while self.head(e) == Some("cartpow") {
            e = self.resolve(self.call(e).unwrap().args[0]);
        }
        self.head(e) == Some("ordered")
            || matches!(self.m.node(e), Node::Const(c) if matches!(self.name(*c), "posreals" | "nonnegreals"))
    }

    /// The sets a prior law's truncation chain restricts its variate to, when
    /// one is `ordered`: they declare the parameter.
    pub(super) fn law_sets(&self, law: NodeId) -> Vec<NodeId> {
        let mut sets = Vec::new();
        let mut m = self.resolve(law);
        loop {
            match self.head(m) {
                Some("truncate") if self.declarable_set(self.call(m).unwrap().args[1]) => {
                    sets.push(self.call(m).unwrap().args[1]);
                    m = self.resolve(self.call(m).unwrap().args[0]);
                }
                Some("normalize") => m = self.resolve(self.call(m).unwrap().args[0]),
                Some("tilt") => {
                    let base = self.resolve(self.call(m).unwrap().args[1]);
                    if self.head(base) == Some("Lebesgue")
                        && let Some(s) = self.call(base).unwrap().args.first()
                    {
                        sets.push(*s);
                    }
                    m = self.resolve(self.call(m).unwrap().args[0]);
                }
                _ => break,
            }
        }
        let ordered = sets
            .iter()
            .any(|&s| self.head(self.resolve(s)) == Some("ordered"));
        if ordered { sets } else { vec![] }
    }

    /// A positional `joint` of scalar laws over a vector variate (§06
    /// "Independent composition"): one vectorized statement when every
    /// component is the same Stan law, with the arguments that differ as
    /// vectors.
    fn joint_law(&mut self, m: NodeId) -> Result<Law> {
        let c = self.call(m).unwrap();
        if !c.named.is_empty() || c.args.is_empty() {
            return Err(Refusal::not_yet(Some(m), "the law `joint` in this form"));
        }
        let mut laws = Vec::new();
        for a in c.args.iter().copied() {
            laws.push(self.law(a, false)?);
        }
        // Components that resolve to different rows of one law
        // (`std_normal()` beside `normal(3, 1)`) share its general row.
        let stan = |l: &Law| l.call.split('(').next().unwrap_or("").to_string();
        if laws.iter().any(|l| stan(l) != stan(&laws[0])) {
            let outer = std::mem::replace(&mut self.general_rows, true);
            let mut general = Vec::new();
            let mut ok = true;
            for a in c.args.iter().copied() {
                match self.law(a, false) {
                    Ok(l) => general.push(l),
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            self.general_rows = outer;
            if ok {
                laws = general;
            }
        }
        let split = |call: &str| -> Option<(String, Vec<String>)> {
            let open = call.find('(')?;
            let inner = &call[open + 1..call.len() - 1];
            let mut parts = Vec::new();
            let (mut depth, mut start) = (0i32, 0);
            for (i, ch) in inner.char_indices() {
                match ch {
                    '(' | '[' | '{' => depth += 1,
                    ')' | ']' | '}' => depth -= 1,
                    ',' if depth == 0 => {
                        parts.push(inner[start..i].trim().to_string());
                        start = i + 1;
                    }
                    _ => {}
                }
            }
            if !inner.trim().is_empty() {
                parts.push(inner[start..].trim().to_string());
            }
            Some((call[..open].to_string(), parts))
        };
        let parts: Vec<_> = laws
            .iter()
            .map(|l| split(&l.call))
            .collect::<Option<_>>()
            .ok_or_else(|| Refusal::not_yet(Some(m), "the law `joint` in this form"))?;
        let same = parts
            .iter()
            .all(|(n, a)| *n == parts[0].0 && a.len() == parts[0].1.len())
            && laws.iter().all(|l| {
                l.vectorized
                    && !l.multivariate
                    && l.trunc.is_none()
                    && l.support.is_none()
                    && l.extra.is_empty()
                    && !l.declared
                    && l.lhs == "{y}"
            });
        if !same {
            return Err(Refusal::not_yet(Some(m), "a joint of different laws"));
        }
        let args: Vec<String> = (0..parts[0].1.len())
            .map(|j| {
                let col: Vec<&String> = parts.iter().map(|(_, a)| &a[j]).collect();
                if col.iter().all(|x| *x == col[0]) {
                    col[0].clone()
                } else {
                    let items: Vec<&str> = col.iter().map(|x| x.as_str()).collect();
                    format!("[{}]'", items.join(", "))
                }
            })
            .collect();
        let mut law = laws.into_iter().next().unwrap();
        law.call = format!("{}({})", parts[0].0, args.join(", "));
        law.indexed = law.call.clone();
        law.flat_call = None;
        Ok(law)
    }

    fn truncated(&mut self, t: NodeId) -> Result<Law> {
        let c = self.call(t).unwrap();
        let (inner, set) = (c.args[0], self.resolve(c.args[1]));
        let mut law = self.law(inner, false)?;
        if law.multivariate {
            return Err(Refusal::not_yet(Some(t), "truncate of a multivariate law"));
        }
        if !law.lcdf {
            return Err(Refusal::not_yet(
                Some(t),
                "truncate of a law with no Stan CDF",
            ));
        }
        let (mut lo, hi) = match self.m.node(set) {
            // A set of integers: integer ends, the upper one open.
            Node::Const(c) if matches!(self.name(*c), "posintegers" | "nonnegintegers") => {
                if !law.discrete {
                    return Err(Refusal::not_yet(
                        Some(t),
                        "a continuous law restricted to a set of integers (the zero measure)",
                    ));
                }
                let lo = if self.name(*c) == "posintegers" {
                    "1"
                } else {
                    "0"
                };
                (Some(lo.to_string()), None)
            }
            _ => self.interval(set)?,
        };
        // `posreals` excludes 0: an integer variate starts at 1. Stan's
        // discrete `T[L, U]` keeps `L`.
        if law.discrete && matches!(self.m.node(set), Node::Const(c) if self.name(*c) == "posreals")
        {
            lo = Some("1".into());
        }
        law.trunc = Some((lo, hi));
        Ok(law)
    }

    /// The law of `locscale(D, m, s)`: D's `locscale` row where it has one
    /// (`StudentT`), else, for a location-scale family `D(a, b)`, the law
    /// `D(m + s·a, s·b)` (the pushforward of x ↦ m + s·x), with s > 0 provable.
    fn locscale_law(&mut self, m: NodeId, [d, loc, scale]: [NodeId; 3]) -> Result<Law> {
        let row = self.table_law(m, "locscale", &[d, loc, scale], false);
        let dr = self.resolve(d);
        let family = self
            .head(dr)
            .filter(|h| matches!(*h, "Normal" | "Cauchy" | "Logistic" | "Laplace"));
        let (Err(no_row), Some(family)) = (&row, family) else {
            return row;
        };
        if no_row.kind.is_never() {
            return row;
        }
        if !self.positive_value(scale) {
            return Err(Refusal::not_yet(
                Some(m),
                "locscale with a scale of unknown sign",
            ));
        }
        let scalar = |me: &Self, x: NodeId| me.shape(x).is_some_and(|s| s.rank == 0);
        if !scalar(self, loc) || !scalar(self, scale) {
            return Err(Refusal::not_yet(
                Some(m),
                "locscale with an array location or scale",
            ));
        }
        let params = self.law_args(family, dr)?;
        let [a, b] = params[..] else {
            return Err(Refusal::not_yet(Some(m), "locscale of this law"));
        };
        let (mv, sv, av, bv) = (
            self.folded(loc)?,
            self.folded(scale)?,
            self.folded(a)?,
            self.folded(b)?,
        );
        let location = fold_add(&mv, &fold_mul(&sv, &av));
        let spread = fold_mul(&sv, &bv);
        let names = flatppl_infer::distribution_param_names(family).unwrap_or_default();
        let general = format!("{family}({})", names.join(", "));
        let Some(row) = tables::law_rows_by_head(family, None)
            .into_iter()
            .find(|r| r.pattern == general)
        else {
            return Err(Refusal::not_yet(Some(m), "locscale of this law"));
        };
        let pat = pattern::parse(&row.pattern);
        let mut binds = HashMap::new();
        for (n, v) in names.iter().zip([location, spread]) {
            binds.insert(n.to_string(), pattern::Bound::Expr(v.1));
        }
        self.law_from_row(m, family, row, pat, binds, false)
    }

    /// A positive literal, or a value of a positive set.
    /// `normalize(weighted(c, M))` with `c = ifelse(s > 0.0 && isfinite(s),
    /// 1.0, 0.0)` (ruling R1). With c > 0 and constant in the variate,
    /// `normalize` gives M back (§06 `normalize`): M·c / (c·Z_M). It holds for
    /// a positive parameter `s`, and for data `s` that `transformed data`
    /// checks once (a validation, not a density term). Inside M, an `ifelse`
    /// on the same condition takes its first branch.
    fn constant_weight_law(&mut self, m: NodeId, inner: NodeId, broadcast: bool) -> Result<Law> {
        let refuse =
            || Refusal::not_yet(Some(m), "normalize of a weighted measure with this weight");
        let (c, base) = {
            let wc = self.call(inner).unwrap();
            (wc.args[0], wc.args[1])
        };
        let (cond, s) = self.positive_finite_indicator(c).ok_or_else(refuse)?;
        // A local of a broadcast user law stands for its argument.
        let arg = match self.m.node(self.resolve(s)) {
            Node::Ref(r) if r.ns == flatppl_core::RefNs::Local => self
                .fctx
                .as_ref()
                .and_then(|x| x.args.get(&r.name).copied())
                .ok_or_else(refuse)?,
            _ => s,
        };
        let text = self.at_module(|me| me.expr(arg))?.s;
        if self.positive_parameter(arg)? {
            // s > 0 and finite by its declaration or value set.
        } else if let Some(checks) = self.root_of_squares(arg)? {
            for check in checks {
                if !self.tdata.contains(&check) {
                    self.tdata.push(check);
                }
            }
        } else if self.is_data(&text) {
            let check = match self.shape(arg).map(|sh| sh.rank) {
                Some(0) => format!(
                    "if (!({text} > 0) || is_inf({text})) {{\n  reject(\"normalize(weighted(c, M)) has weight 0: {text} is \", {text});\n}}"
                ),
                Some(1) => {
                    let i = self.fresh_index();
                    let n = self.extent_of(&text);
                    self.used.remove(&i);
                    format!(
                        "for ({i} in 1:{n}) {{\n  if (!({text}[{i}] > 0) || is_inf({text}[{i}])) {{\n    reject(\"normalize(weighted(c, M)) has weight 0 at {text}[\", {i}, \"]\");\n  }}\n}}"
                    )
                }
                _ => return Err(refuse()),
            };
            if !self.tdata.contains(&check) {
                self.tdata.push(check);
            }
        } else {
            return Err(refuse());
        }
        self.true_conds.push(cond);
        let out = self.law(base, broadcast);
        self.true_conds.pop();
        out
    }

    /// A parameter declared `lower=0`, or a `posreals` value: positive and
    /// finite.
    fn positive_parameter(&mut self, x: NodeId) -> Result<bool> {
        if self.positive_value(x) {
            return Ok(true);
        }
        let text = self.at_module(|me| me.expr(x))?.s;
        Ok(!self.is_data(&text)
            && self.decls.get(&text).is_some_and(|t| {
                t.bounds
                    .lower
                    .as_deref()
                    .and_then(|l| l.parse::<f64>().ok())
                    .is_some_and(|l| l >= 0.0)
            }))
    }

    /// `s = sqrt(a .* a .+ p .* p)` (either summand first, elementwise or
    /// scalar) with p a positive parameter and a data. Then p > 0 gives
    /// p² > 0, so a² + p² > 0 and s > 0; s is finite where a and p are, and
    /// p is a declared real. The `transformed data` checks returned test
    /// that a is finite (not inf, not NaN). `None`: not this form.
    fn root_of_squares(&mut self, s: NodeId) -> Result<Option<Vec<String>>> {
        let op = |me: &Self, x: NodeId, name: &str| -> Option<Vec<NodeId>> {
            let x = me.resolve(x);
            let c = me.call(x)?;
            if me.head(x) == Some(name) {
                return Some(c.args.to_vec());
            }
            if me.head(x) == Some("broadcast")
                && matches!(me.m.node(c.args[0]), Node::Const(f) if me.name(*f) == name)
            {
                return Some(c.args[1..].to_vec());
            }
            None
        };
        let square = |me: &Self, x: NodeId| -> Option<NodeId> {
            match op(me, x, "mul")?[..] {
                [u, v] if me.m.structural_eq(u, v) => Some(u),
                _ => None,
            }
        };
        let Some(&[inner]) = op(self, s, "sqrt").as_deref() else {
            return Ok(None);
        };
        let Some(&[u, v]) = op(self, inner, "add").as_deref() else {
            return Ok(None);
        };
        let (Some(a), Some(b)) = (square(self, u), square(self, v)) else {
            return Ok(None);
        };
        let data = if self.positive_parameter(b)? {
            a
        } else if self.positive_parameter(a)? {
            b
        } else {
            return Ok(None);
        };
        let text = self.at_module(|me| me.expr(data))?.s;
        if !self.is_data(&text) {
            return Ok(None);
        }
        let check = match self.shape(data).map(|sh| sh.rank) {
            Some(0) => format!(
                "if (is_inf({text}) || is_nan({text})) {{\n  reject(\"{text} is not finite: \", {text});\n}}"
            ),
            Some(1) => {
                let i = self.fresh_index();
                let n = self.extent_of(&text);
                self.used.remove(&i);
                format!(
                    "for ({i} in 1:{n}) {{\n  if (is_inf({text}[{i}]) || is_nan({text}[{i}])) {{\n    reject(\"{text}[\", {i}, \"] is not finite\");\n  }}\n}}"
                )
            }
            _ => return Ok(None),
        };
        Ok(Some(vec![check]))
    }

    /// `ifelse(s > 0.0 && isfinite(s), 1.0, 0.0)`, either order of the two
    /// tests: the condition and `s`.
    pub(super) fn positive_finite_indicator(&self, c: NodeId) -> Option<(NodeId, NodeId)> {
        let c = self.resolve(c);
        if self.head(c) != Some("ifelse") {
            return None;
        }
        let [cond, one, zero] = self.call(c)?.args[..] else {
            return None;
        };
        if self.literal_number(one) != Some(1.0) || self.literal_number(zero) != Some(0.0) {
            return None;
        }
        let cr = self.resolve(cond);
        if self.head(cr) != Some("land") {
            return None;
        }
        let [p, q] = self.call(cr)?.args[..] else {
            return None;
        };
        let gt = |x: NodeId| {
            let x = self.resolve(x);
            let a = &self.call(x)?.args;
            (self.head(x) == Some("gt") && a.len() == 2 && self.literal_number(a[1]) == Some(0.0))
                .then(|| a[0])
        };
        let finite = |x: NodeId| {
            let x = self.resolve(x);
            let a = &self.call(x)?.args;
            (self.head(x) == Some("isfinite") && a.len() == 1).then(|| a[0])
        };
        let (a, b) = match (gt(p), finite(q), finite(p), gt(q)) {
            (Some(a), Some(b), _, _) | (_, _, Some(b), Some(a)) => (a, b),
            _ => return None,
        };
        self.m.structural_eq(a, b).then_some((cond, a))
    }

    fn positive_value(&self, x: NodeId) -> bool {
        match self.literal_number(x) {
            Some(v) => v > 0.0,
            None => matches!(
                self.m.valueset_of(self.resolve(x)),
                Some(ValueSet::PosReals)
            ),
        }
    }

    /// A scalar's number when it is a literal, and its Stan text.
    fn folded(&mut self, x: NodeId) -> Result<(Option<f64>, Ex)> {
        match self.literal_number(x) {
            Some(n) => Ok(number_ex(n)),
            None => Ok((None, self.real(x)?)),
        }
    }

    /// The number of a literal arithmetic expression (`-1`, `1 / 2`).
    fn literal_number(&self, x: NodeId) -> Option<f64> {
        let r = self.resolve(x);
        match self.m.node(r) {
            Node::Lit(Scalar::Int(k)) => return Some(*k as f64),
            Node::Lit(Scalar::Real(v)) => return Some(*v),
            _ => {}
        }
        let args = self.call(r)?.args.to_vec();
        match (self.head(r)?, args.as_slice()) {
            ("neg", [u]) => Some(-self.literal_number(*u)?),
            ("add", [u, v]) => Some(self.literal_number(*u)? + self.literal_number(*v)?),
            ("sub", [u, v]) => Some(self.literal_number(*u)? - self.literal_number(*v)?),
            ("mul", [u, v]) => Some(self.literal_number(*u)? * self.literal_number(*v)?),
            ("divide", [u, v]) => Some(self.literal_number(*u)? / self.literal_number(*v)?),
            _ => None,
        }
    }

    /// `Lebesgue(S)`, or an `iid` product of it.
    fn lebesgue_product(&self, m: NodeId) -> bool {
        let r = self.resolve(m);
        match self.head(r) {
            Some("Lebesgue") => true,
            Some("iid") => self.lebesgue_product(self.call(r).unwrap().args[0]),
            _ => false,
        }
    }

    /// `locscale(normalize(truncate(D, I)), m, s)`: the arguments `[D, m, s]`
    /// of D's `locscale` row and the ends of `m + s·I`. `None` for another
    /// form. The scale must be positive: a positive literal or a value of a
    /// positive set.
    #[allow(clippy::type_complexity)]
    fn locscale_truncation(
        &mut self,
        m: NodeId,
    ) -> Result<Option<([NodeId; 3], Option<String>, Option<String>)>> {
        let c = self.call(m).unwrap();
        let [base, loc, scale] = c.args[..] else {
            return Ok(None);
        };
        if !c.named.is_empty() {
            return Ok(None);
        }
        let norm = self.resolve(base);
        if self.head(norm) != Some("normalize") {
            return Ok(None);
        }
        let t = self.resolve(self.call(norm).unwrap().args[0]);
        if self.head(t) != Some("truncate") {
            return Ok(None);
        }
        let tc = self.call(t).unwrap();
        let (d, set) = (tc.args[0], self.resolve(tc.args[1]));
        let num = |me: &Self, x: NodeId| match me.m.node(me.resolve(x)) {
            Node::Lit(Scalar::Int(k)) => Some(*k as f64),
            Node::Lit(Scalar::Real(r)) => Some(*r),
            _ => None,
        };
        if !self.positive_value(scale) {
            return Err(Refusal::not_yet(
                Some(m),
                "locscale of a truncated law with a scale of unknown sign",
            ));
        }
        let (lo, hi) = self.interval(set)?;
        let (lt, st) = (self.real(loc)?, self.real(scale)?);
        let (ln, sn) = (num(self, loc), num(self, scale));
        let image = |e: String| -> String {
            match (ln, sn, e.parse::<f64>()) {
                (Some(a), Some(b), Ok(x)) => crate::stan_type::number(a + b * x),
                _ => format!(
                    "{} + {} * {}",
                    lt.at(super::P_ADD),
                    st.at(super::P_MUL),
                    if is_identifier(&e) || e.parse::<f64>().is_ok() {
                        e
                    } else {
                        format!("({e})")
                    }
                ),
            }
        };
        // An end written as `(c - m) / s`, with m and s the location and
        // scale, maps back to c exactly.
        let ends: Vec<Option<NodeId>> = match (self.head(set), self.call(set)) {
            (Some("interval"), Some(c)) => c.args.iter().map(|&e| Some(e)).collect(),
            _ => vec![None, None],
        };
        let mut mapped = [lo.map(image), hi.map(image)];
        for (k, end) in ends.into_iter().enumerate().take(2) {
            if let Some(e) = end
                && mapped[k].is_some()
                && let Some(c) = self.inverse_end(e, loc, scale)
            {
                mapped[k] = Some(self.real(c)?.s);
            }
        }
        let [lo, hi] = mapped;
        Ok(Some(([d, loc, scale], lo, hi)))
    }

    /// `c` when `e` is `(c - m) / s`, with `m` and `s` the same nodes (or
    /// equal literals) as `loc` and `scale`.
    fn inverse_end(&self, e: NodeId, loc: NodeId, scale: NodeId) -> Option<NodeId> {
        let same = |me: &Self, a: NodeId, b: NodeId| {
            me.m.structural_eq(me.resolve(a), me.resolve(b))
                || matches!((me.literal_number(a), me.literal_number(b)), (Some(x), Some(y)) if x == y)
        };
        let d = self.resolve(e);
        let [num, s] = self
            .call(d)
            .filter(|_| self.head(d) == Some("divide"))?
            .args[..]
        else {
            return None;
        };
        let n = self.resolve(num);
        let [c, m] = self.call(n).filter(|_| self.head(n) == Some("sub"))?.args[..] else {
            return None;
        };
        (same(self, m, loc) && same(self, s, scale)).then_some(c)
    }

    /// The finite ends of an interval set; `None` for an open side.
    /// `true` unless `s` is an `interval` with a finite end that is not
    /// integer-typed (integers by type, not value).
    pub(super) fn integer_ends(&self, s: NodeId) -> bool {
        let s = self.resolve(s);
        if self.head(s) != Some("interval") {
            return true;
        }
        self.call(s).unwrap().args.iter().all(|&a| {
            let r = self.resolve(a);
            let inf =
                |n: NodeId| matches!(self.m.node(n), Node::Const(c) if self.name(*c) == "inf");
            inf(r)
                || (self.head(r) == Some("neg") && inf(self.call(r).unwrap().args[0]))
                || self
                    .m
                    .type_of(a)
                    .or_else(|| self.m.type_of(r))
                    .is_some_and(ty_is_int)
        })
    }

    fn interval(&mut self, s: NodeId) -> Result<(Option<String>, Option<String>)> {
        match self.m.node(s) {
            Node::Const(c) => match self.name(*c) {
                "unitinterval" => Ok((Some("0".into()), Some("1".into()))),
                "posreals" | "nonnegreals" => Ok((Some("0".into()), None)),
                "reals" => Ok((None, None)),
                other => Err(Refusal::not_yet(Some(s), format!("the set `{other}`"))),
            },
            _ if self.head(s) == Some("interval") => {
                let c = self.call(s).unwrap();
                let mut ends = Vec::new();
                for &a in c.args.iter() {
                    let r = self.resolve(a);
                    let open = matches!(self.m.node(r), Node::Const(c) if self.name(*c) == "inf")
                        || (self.head(r) == Some("neg")
                            && matches!(self.m.node(self.call(r).unwrap().args[0]), Node::Const(c) if self.name(*c) == "inf"));
                    ends.push(if open { None } else { Some(self.real(a)?.s) });
                }
                Ok((ends[0].clone(), ends[1].clone()))
            }
            _ => Err(Refusal::not_yet(Some(s), "a truncation to this set")),
        }
    }

    /// A law from `distributions.ron`: the most specific row whose pattern
    /// matches `name(args…)`. With `broadcast`, an argument whose rank exceeds
    /// the law parameter's is indexed per element.
    fn table_law(
        &mut self,
        m: NodeId,
        name: &str,
        args: &[NodeId],
        broadcast: bool,
    ) -> Result<Law> {
        self.table_law_in(m, name, None, args, broadcast)
    }

    fn table_law_in(
        &mut self,
        m: NodeId,
        name: &str,
        module: Option<&str>,
        args: &[NodeId],
        broadcast: bool,
    ) -> Result<Law> {
        let params = flatppl_infer::distribution_param_names(name).unwrap_or_default();
        // §04 "Auto-splatting": a callable with two or more required inputs
        // called with a single positional record or table "is decomposed into
        // keyword arguments (record field names and values or table column
        // names and columns)". A table reaches a law only through a broadcast
        // (`D.(T)`, a `ksuperpose` component).
        if let [a] = args
            && params.len() >= 2
            && matches!(
                self.ty(*a),
                Some(Type::Record(_) | Type::Table { .. } | Type::Any) | None
            )
        {
            return self.splat_law(m, name, module, *a, &params, broadcast);
        }
        let mut found = None;
        let general = format!("{name}({})", params.join(", "));
        for row in tables::law_rows_by_head(name, module) {
            // A joint's components share one row: the general one.
            if self.general_rows && row.pattern != general {
                continue;
            }
            let pat = pattern::parse(&row.pattern);
            let mut binds = HashMap::new();
            if self.match_call(&pat, args, &mut binds) {
                found = Some((row, pat, binds));
                break;
            }
        }
        let Some((row, pat, binds)) = found else {
            return Err(Refusal::not_yet(
                Some(m),
                format!("the law `{name}` in this form"),
            ));
        };
        self.law_from_row(m, name, row, pat, binds, broadcast)
    }

    /// A law called with one record argument, its fields as the keyword
    /// arguments: a record literal's field values match the rows as written
    /// arguments; a computed record's field texts bind the variables of a row
    /// whose arguments are all variables.
    /// The keyword arguments a lone positional argument at a splat site gives
    /// (§04 "Auto-splatting"): a record or table literal's field or column
    /// names and values. `None` for a computed record, which
    /// [`Self::splat_law`] reads by its field texts; any other argument (a
    /// loaded table, an untyped value) refuses.
    pub(super) fn splat_fields(&self, arg: NodeId) -> Result<Option<Vec<(String, NodeId)>>> {
        let r = self.resolve(arg);
        match self.head(r) {
            Some("record" | "table") => Ok(Some(
                self.call(r)
                    .unwrap()
                    .named
                    .iter()
                    .map(|na| (self.name(na.name).to_string(), na.value))
                    .collect(),
            )),
            _ if matches!(self.ty(arg), Some(Type::Record(_))) => Ok(None),
            _ => Err(Refusal::not_yet(
                Some(arg),
                "a splat of an argument that is not a record or a `table(…)` literal",
            )),
        }
    }

    fn splat_law(
        &mut self,
        m: NodeId,
        name: &str,
        module: Option<&str>,
        rec: NodeId,
        params: &[String],
        broadcast: bool,
    ) -> Result<Law> {
        let unknown = |mut fields: Vec<String>| -> Result<()> {
            fields.sort();
            let mut want = params.to_vec();
            want.sort();
            if fields == want {
                return Ok(());
            }
            Err(Refusal::not_yet(
                Some(m),
                format!(
                    "`{name}` of a record or table with the fields ({}), not its inputs ({})",
                    fields.join(", "),
                    params.join(", ")
                ),
            ))
        };
        if let Some(named) = self.splat_fields(rec)? {
            unknown(named.iter().map(|f| f.0.clone()).collect())?;
            let ordered: Vec<NodeId> = params
                .iter()
                .map(|p| named.iter().find(|f| &f.0 == p).unwrap().1)
                .collect();
            return self.table_law_in(m, name, module, &ordered, broadcast);
        }
        let fields = self.record_map(rec)?;
        unknown(fields.keys().cloned().collect())?;
        for row in tables::law_rows_by_head(name, module) {
            let pat = pattern::parse(&row.pattern);
            let pattern::Pat::Call(_, pargs) = &pat else {
                continue;
            };
            if pargs.len() != params.len() {
                continue;
            }
            let mut binds = HashMap::new();
            let all_vars = pargs.iter().zip(params).all(|(p, name)| match p {
                pattern::Pat::Var(v) => {
                    binds.insert(
                        v.clone(),
                        pattern::Bound::Expr(Ex::atom(fields[name].clone())),
                    );
                    true
                }
                _ => false,
            });
            if all_vars {
                return self.law_from_row(m, name, row, pat, binds, broadcast);
            }
        }
        Err(Refusal::not_yet(
            Some(m),
            format!("the law `{name}` of a computed record in this form"),
        ))
    }

    /// The law of a matched `distributions.ron` row with its pattern
    /// variables bound.
    fn law_from_row(
        &mut self,
        m: NodeId,
        name: &str,
        row: &'static tables::LawRow,
        pat: pattern::Pat,
        binds: HashMap<String, pattern::Bound>,
        broadcast: bool,
    ) -> Result<Law> {
        match row.kind.as_str() {
            "todo" => {
                return Err(Refusal::new(
                    RefusalKind::NotYetSupported,
                    Some(m),
                    name.to_string(),
                    format!(
                        "`{name}` is not supported by the Stan emitter yet ({})",
                        row.note
                    ),
                ));
            }
            "none" => {
                return Err(Refusal::new(
                    RefusalKind::NoStanFunction,
                    Some(m),
                    name.to_string(),
                    format!("`{name}` has no Stan density: {}", row.note),
                ));
            }
            // A user-defined `_lpdf`: its text goes into `functions` once.
            "custom_lpdf" => self.add_functions(&row.functions),
            _ => {}
        }
        if row.status.starts_with("unverified") {
            return Err(Refusal::not_yet(
                Some(m),
                format!("the law `{name}` (its Stan mapping is unverified)"),
            ));
        }
        if !row.normalized && row.kind != "custom_lpdf" {
            return Err(Refusal::not_yet(
                Some(m),
                format!("the unnormalized law `{name}`"),
            ));
        }
        // Expected rank of each variable: the law parameter's rank when the
        // variable is a direct argument, a scalar inside a nested call.
        let ranks = flatppl_infer::distribution_param_ranks(name).unwrap_or_default();
        let mut expected: HashMap<String, usize> = HashMap::new();
        if let pattern::Pat::Call(_, pargs) = &pat {
            for (k, p) in pargs.iter().enumerate() {
                if let pattern::Pat::Var(v) = p {
                    expected.insert(v.clone(), ranks.get(k).copied().unwrap_or(0));
                }
            }
        }
        let mut per: HashMap<String, bool> = HashMap::new();
        for (v, b) in &binds {
            if let pattern::Bound::Node(n) = b {
                let rank = self.shape(*n).map(|s| s.rank).unwrap_or(0);
                per.insert(
                    v.clone(),
                    broadcast && rank > expected.get(v).copied().unwrap_or(0),
                );
            }
        }
        let is_per = |v: &str| per.get(v).copied().unwrap_or(false);
        let scalar_variate = row.variate.starts_with("real") || row.variate.starts_with("int");
        // `{y}` is the variate, not a template variable: keep it through `fill`.
        let guard = if row.guard.is_empty() {
            None
        } else {
            let t = row.guard.replace("{y}", "\u{2}");
            let whole = self
                .fill(&t, &binds, &is_per, None)?
                .replace('\u{2}', "{y}");
            let each = self
                .fill(&t, &binds, &is_per, Some(LOOP))?
                .replace('\u{2}', "{y}");
            Some((whole, each))
        };
        // A template's elementwise operators become scalar ones when every
        // operand they can see is a scalar (`1, p ./ (1 - p)` for one `p`).
        // The rule reads each operand's rank in the printed text: a
        // whole-array lift has an element shape but a vector text. Only the
        // template changes: a bound operand prints its own operators
        // (`sum(vs .* vs)` held in a loop).
        let rank_of = |me: &Self, b: &pattern::Bound| match b {
            pattern::Bound::Node(n)
                if me.fctx.as_ref().is_some_and(|c| c.lifted_full.contains(n)) =>
            {
                1
            }
            pattern::Bound::Node(n) => me.shape(*n).map(|s| s.rank).unwrap_or(0),
            _ => 0,
        };
        let whole_scalar = binds.values().all(|b| rank_of(self, b) == 0);
        let each_scalar = binds
            .iter()
            .all(|(v, b)| rank_of(self, b) == 0 || (is_per(v) && rank_of(self, b) == 1));
        let template = |scalar: bool| {
            if scalar {
                scalar_operators(&row.args)
            } else {
                row.args.clone()
            }
        };
        let flat = self.flat.take().filter(|_| scalar_variate);
        let mut flat_args = None;
        if flat.is_some() {
            self.flat = flat;
            let filled = self.fill(&template(whole_scalar), &binds, &is_per, None);
            self.flat = None;
            flat_args = Some(filled?);
        }
        let args = self.fill(&template(whole_scalar), &binds, &is_per, None)?;
        let indexed = self.fill(&template(each_scalar), &binds, &is_per, Some(LOOP))?;
        let vector_per_element = per
            .iter()
            .any(|(v, &p)| p && expected.get(v).copied().unwrap_or(0) > 0);
        // The normalized custom rows have overloads for scalar parameters
        // only (a vector variate, scalar parameters).
        if row.kind == "custom_lpdf"
            && row.normalized
            && binds.values().any(|b| rank_of(self, b) > 0)
        {
            return Err(Refusal::not_yet(
                Some(m),
                format!("the law `{name}` with array parameters"),
            ));
        }
        Ok(Law {
            call: format!("{}({args})", row.stan),
            flat_call: flat_args.map(|a| format!("{}({a})", row.stan)),
            indexed: format!("{}({indexed})", row.stan),
            vector_per_element,
            lhs: row.lhs.clone(),
            trunc: None,
            vectorized: row.vectorized,
            multivariate: !scalar_variate,
            lcdf: row.lcdf,
            discrete: row.variate.starts_with("int"),
            support: None,
            guard,
            bounds: variate_bounds(&row.variate),
            tail: row.tail.clone(),
            floor_arg: (!row.floor.is_empty())
                .then(|| row.args.split(", ").position(|a| a == row.floor))
                .flatten(),
            extra: vec![],
            declared: false,
            unnormalized: !row.normalized,
            stan_name: row.stan.clone(),
            flatppl: row.flatppl.clone(),
            custom: row.kind == "custom_lpdf",
            unchecked_support: row.unchecked_support,
            total_support: row.total_support,
            support_type: row
                .variate
                .split_once('[')
                .and_then(|(b, _)| crate::stan_type::support_type(b)),
        })
    }

    /// Fill in the query input names: `outputs = logdensityof(T, record(f = x))`.
    pub(crate) fn query_inputs(&mut self, target: Symbol) {
        let Some(out) = self
            .m
            .bindings()
            .find(|(_, b)| self.m.resolve(b.name) == "outputs")
            .map(|(_, b)| b.rhs)
        else {
            return;
        };
        let out = self.resolve(out);
        if self.head(out) != Some("logdensityof") {
            return;
        }
        let c = self.call(out).unwrap();
        let Some(t) = self.self_ref(c.args[0]) else {
            return;
        };
        if self.m.alias_chain(t).last() != self.m.alias_chain(target).last() {
            return;
        }
        let point = self.resolve(c.args[1]);
        if self.head(point) != Some("record") {
            return;
        }
        for na in self.call(point).unwrap().named.iter() {
            let field = self.name(na.name);
            let (input, index) = match self.self_ref(na.value) {
                Some(input) => (Some(input), None),
                None => match self.point_index(na.value) {
                    Some((input, index)) => (Some(input), Some(index)),
                    None => (None, None),
                },
            };
            let input = input.map(|i| self.abi_name(i));
            if let Some(input) = input
                && let Some(p) = self.manifest.iter_mut().find(|p| p.field == field)
            {
                p.input = Some(input);
                p.index = index;
            }
        }
    }

    /// `theta[k]` or `theta[[k, …]]` with literal indices: the input and the
    /// index.
    fn point_index(&self, v: NodeId) -> Option<(Symbol, crate::ManifestIndex)> {
        let g = self.resolve(v);
        if self.head(g) != Some("get") {
            return None;
        }
        let c = self.call(g)?;
        let [x, sel] = c.args[..] else {
            return None;
        };
        let input = self.self_ref(x)?;
        let int = |me: &Self, n: NodeId| match me.m.node(n) {
            Node::Lit(Scalar::Int(k)) => Some(*k),
            _ => None,
        };
        if let Some(k) = int(self, sel) {
            return Some((input, crate::ManifestIndex::Scalar(k)));
        }
        let list = self.resolve(sel);
        if self.head(list) != Some("vector") {
            return None;
        }
        let ks: Option<Vec<i64>> = self
            .call(list)?
            .args
            .iter()
            .map(|&a| int(self, a))
            .collect();
        Some((input, crate::ManifestIndex::Slice(ks?)))
    }

    /// A parameter whose own binding is a module input (`inputs = (…)`,
    /// matched by alias end) is bound by that ABI name. A query point that
    /// binds the same field to another input is a contradiction: refuse.
    pub(crate) fn abi_parameters(&mut self) -> Result<()> {
        let abi: Vec<(Symbol, Symbol)> = self
            .abi_inputs()
            .into_iter()
            .map(|a| (a, *self.m.alias_chain(a).last().unwrap()))
            .collect();
        for (p, b) in self.manifest.iter_mut().zip(&self.manifest_bindings) {
            let Some(b) = b else { continue };
            let end = *self.m.alias_chain(*b).last().unwrap();
            let Some((written, _)) = abi.iter().find(|(_, e)| *e == end) else {
                continue;
            };
            let name = self.m.resolve(*written).to_string();
            match &p.input {
                None => p.input = Some(name),
                Some(other) if *other != name || p.index.is_some() => {
                    return Err(Refusal::not_yet(
                        None,
                        format!(
                            "a query point that binds `{}` to `{other}` while it is the input `{name}`",
                            p.field
                        ),
                    ));
                }
                Some(_) => {}
            }
        }
        Ok(())
    }

    /// The names written in the module's `inputs = (…)`, the ABI names.
    fn abi_inputs(&self) -> Vec<Symbol> {
        let Some(rhs) = self
            .m
            .bindings()
            .find(|(_, b)| self.m.resolve(b.name) == "inputs")
            .map(|(_, b)| b.rhs)
        else {
            return vec![];
        };
        let items: Vec<NodeId> = match self.call(rhs) {
            Some(c) if self.self_ref(rhs).is_none() => c.args.to_vec(),
            _ => vec![rhs],
        };
        items
            .into_iter()
            .filter_map(|it| self.self_ref(it))
            .collect()
    }

    /// The ABI name of an input the query point reads: the `inputs` entry
    /// with the same alias end, else the name as written.
    fn abi_name(&self, written: Symbol) -> String {
        let end = self.m.alias_chain(written).last().copied();
        let abi = self
            .abi_inputs()
            .into_iter()
            .find(|&a| self.m.alias_chain(a).last().copied() == end)
            .unwrap_or(written);
        self.m.resolve(abi).to_string()
    }

    /// The module inputs (`inputs = (…)`) the program declares as data.
    pub(crate) fn data_inputs(&self) -> Vec<crate::ManifestData> {
        let mut out = Vec::new();
        for written in self.abi_inputs() {
            // The ABI name is the one written in `inputs`; its alias end is
            // the binding the program declared.
            let sym = *self.m.alias_chain(written).last().unwrap();
            let Some(stan) = self.env.get(&sym) else {
                continue;
            };
            let Some(line) = self.data.iter().find(|l| {
                l.strip_suffix(';')
                    .and_then(|l| l.rsplit_once(' '))
                    .is_some_and(|(_, n)| n == stan)
            }) else {
                continue;
            };
            let ty = line.strip_suffix(';').unwrap().rsplit_once(' ').unwrap().0;
            out.push(crate::ManifestData {
                input: self.name(written).to_string(),
                stan: stan.clone(),
                ty: ty.to_string(),
            });
        }
        out
    }
}

fn scalar_operators(s: &str) -> String {
    s.replace(" .* ", " * ")
        .replace(" ./ ", " / ")
        .replace(" .^ ", " ^ ")
}

fn ty_is_int(ty: &Type) -> bool {
    super::shape_of(ty).is_some_and(|s| s.int)
}

pub(super) fn is_identifier(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && s.starts_with(|c: char| c.is_ascii_alphabetic())
}

/// `lower` and `upper` of a `distributions.ron` variate type
/// (`int<lower=0, upper=1>`).
/// The larger (or smaller) of two ends, folded when both are literals.
fn pick(x: &str, b: &str, larger: bool, discrete: bool) -> String {
    // A literal value: a number, or von Mises's `±pi()`.
    let lit = |s: &str| -> std::result::Result<f64, ()> {
        match s {
            "pi()" => Ok(std::f64::consts::PI),
            "-pi()" => Ok(-std::f64::consts::PI),
            _ => s.parse::<f64>().map_err(|_| ()),
        }
    };
    match (lit(x), lit(b)) {
        // A tie keeps the end as written, or the integer literal for a
        // discrete law (`0`, not `0.0`, reaches an integer CDF argument).
        (Ok(xv), Ok(bv)) if xv == bv => {
            if discrete && x.parse::<i64>().is_err() && b.parse::<i64>().is_ok() {
                b.to_string()
            } else {
                x.to_string()
            }
        }
        (Ok(xv), Ok(bv)) if (xv >= bv) == larger => x.to_string(),
        (Ok(_), Ok(_)) => b.to_string(),
        _ => {
            let f = match (discrete, larger) {
                (true, true) => "max",
                (true, false) => "min",
                (false, true) => "fmax",
                (false, false) => "fmin",
            };
            format!("{f}({x}, {b})")
        }
    }
}

/// `ty` with its last `axes` extents set to `n`: a variate size inference
/// leaves open, from the law's `sizes`.
fn with_extent(ty: &Type, axes: usize, n: u32) -> Type {
    match ty {
        Type::Array { shape, elem } if matches!(elem.as_ref(), Type::Array { .. }) => Type::Array {
            shape: shape.clone(),
            elem: Box::new(with_extent(elem, axes, n)),
        },
        Type::Array { shape, elem } => {
            let mut s = shape.to_vec();
            let k = s.len().saturating_sub(axes);
            for d in &mut s[k..] {
                *d = Dim::Static(n);
            }
            Type::Array {
                shape: s.into(),
                elem: elem.clone(),
            }
        }
        t => t.clone(),
    }
}

fn variate_bounds(variate: &str) -> (Option<String>, Option<String>) {
    let inner = variate
        .split_once('<')
        .and_then(|(_, r)| r.rsplit_once('>'))
        .map(|(b, _)| b)
        .unwrap_or("");
    let mut out = (None, None);
    for part in inner.split(", ") {
        match part.split_once('=') {
            Some(("lower", v)) => out.0 = Some(v.trim().to_string()),
            Some(("upper", v)) => out.1 = Some(v.trim().to_string()),
            _ => {}
        }
    }
    out
}

fn number_ex(x: f64) -> (Option<f64>, Ex) {
    let s = crate::stan_type::number(x);
    let ex = if x < 0.0 {
        Ex {
            s,
            prec: super::P_UNARY,
        }
    } else {
        Ex::atom(s)
    };
    (Some(x), ex)
}

/// `x + y`, folded for literals and 0.
fn fold_add(x: &(Option<f64>, Ex), y: &(Option<f64>, Ex)) -> (Option<f64>, Ex) {
    match (x.0, y.0) {
        (Some(a), Some(b)) => number_ex(a + b),
        (Some(0.0), _) => y.clone(),
        (_, Some(0.0)) => x.clone(),
        _ => (
            None,
            Ex {
                s: format!("{} + {}", x.1.at(super::P_ADD), y.1.at(super::P_ADD + 1)),
                prec: super::P_ADD,
            },
        ),
    }
}

/// `x * y`, folded for literals, 0 and 1.
fn fold_mul(x: &(Option<f64>, Ex), y: &(Option<f64>, Ex)) -> (Option<f64>, Ex) {
    match (x.0, y.0) {
        (Some(a), Some(b)) => number_ex(a * b),
        (Some(0.0), _) | (_, Some(0.0)) => number_ex(0.0),
        (Some(1.0), _) => y.clone(),
        (_, Some(1.0)) => x.clone(),
        _ => (
            None,
            Ex {
                s: format!("{} * {}", x.1.at(super::P_MUL), y.1.at(super::P_MUL + 1)),
                prec: super::P_MUL,
            },
        ),
    }
}
