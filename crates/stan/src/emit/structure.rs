//! Target decomposition and the `~` statements.

use super::pattern;
use super::{Emitter, Ex, Flat, P_ATOM, PriorField};
use crate::ManifestParam;
use crate::refuse::{Refusal, RefusalKind, Result};
use crate::stan_type::StanType;
use crate::tables;
use flatppl_core::{Inputs, NamedKind, Node, NodeId, RefNs, Symbol, Type, ValueSet};
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
    lcdf: bool,
    /// An integer variate.
    pub discrete: bool,
    /// The support an unnormalized `truncate` restricts its variate to.
    pub support: Option<(Option<String>, Option<String>)>,
    /// The `distributions.ron` guard with its arguments filled: `{y}` stays.
    /// The second form indexes per element ([`LOOP`]).
    pub guard: Option<(String, String)>,
    /// Statements the law adds after its `~` line (`target += lgamma(K + 1);`).
    pub extra: Vec<String>,
    /// The restriction lives in the variate's declaration (`ordered[K]`):
    /// the law is exact only on a parameter declared from its sets.
    pub declared: bool,
}

impl Law {
    fn rhs(&self, index: Option<&str>) -> String {
        let s = match index {
            Some(i) => self.indexed.replace(LOOP, i),
            None => self.call.clone(),
        };
        self.truncate(s)
    }

    fn truncate(&self, mut s: String) -> String {
        if let Some((lo, hi)) = &self.trunc {
            s.push_str(&format!(
                " T[{}, {}]",
                lo.as_deref().unwrap_or(""),
                hi.as_deref().unwrap_or("")
            ));
        }
        s
    }

    /// The variate side for `y`.
    fn lhs(&self, y: &str) -> String {
        self.lhs.replace("{y}", y)
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

impl<'m> Emitter<'m> {
    /// Emit the target binding `target`.
    pub(crate) fn emit_target(&mut self, target: Symbol) -> Result<()> {
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
                self.free_inputs(m)?;
                let obs = self.measure_observations(m, x)?;
                self.observe(obs)
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

    // ---------------------------------------------------------------- prior

    /// The prior's `~` lines, then one `target +=` per log weight.
    fn prior_statements(&mut self, fields: &[PriorField]) -> Result<()> {
        for f in fields {
            if let Some(law) = f.law {
                let lhs = self.fields[&f.field].clone();
                let lines = self.tilde(&lhs, law)?;
                self.prior.extend(lines);
            }
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
                for na in self.call(rec).unwrap().named.iter() {
                    let Some(sym) = self.self_ref(na.value) else {
                        return Err(Refusal::not_yet(
                            Some(na.value),
                            format!("the derived prior field `{}`", self.name(na.name)),
                        ));
                    };
                    let sym = *self.m.alias_chain(sym).last().unwrap();
                    let rhs = self.m.binding(self.m.binding_by_name(sym).unwrap()).rhs;
                    if self.head(rhs) != Some("draw") {
                        return Err(Refusal::not_yet(
                            Some(na.value),
                            format!("the derived prior field `{}`", self.name(na.name)),
                        ));
                    }
                    let law = self.call(rhs).unwrap().args[0];
                    out.push(PriorField {
                        field: na.name,
                        binding: Some(sym),
                        law: Some(law),
                        node: na.value,
                        sets: self.law_sets(law),
                    });
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
                if t.bounds.lower.is_none() {
                    t.bounds.lower = lo;
                }
                if t.bounds.upper.is_none() {
                    t.bounds.upper = hi;
                }
            }
            self.params.push(format!("{};", t.render(&stan)));
            self.record_decl(&stan, t);
            if let Some(b) = f.binding {
                self.env.insert(b, stan.clone());
            }
            self.fields.insert(f.field, stan.clone());
            self.manifest.push(ManifestParam {
                stan,
                field: self.name(f.field).to_string(),
                input: None,
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
            let t = self.declare(rhs, vs.as_ref(), true, &stan, &hints)?;
            self.params.push(format!("{};", t.render(&stan)));
            self.record_decl(&stan, t);
            self.env.insert(sym, stan.clone());
            self.fields.insert(name, stan.clone());
            self.manifest.push(ManifestParam {
                stan,
                field: self.name(name).to_string(),
                input: None,
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
            Some("functionof") => Ok(vec![Observation {
                draw: None,
                law: body,
                value: obs,
                name: "y".to_string(),
            }]),
            _ => Err(Refusal::not_yet(Some(k), "this kernel")),
        }
    }

    /// Declare each `elementof` binding `m` reads as a flat parameter, in
    /// module order.
    fn free_inputs(&mut self, m: NodeId) -> Result<()> {
        let mut found = HashSet::new();
        let mut stack = vec![m];
        let mut seen = HashSet::new();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            if let Node::Ref(r) = self.m.node(id)
                && r.ns == RefNs::SelfMod
                && let Some(b) = self
                    .m
                    .binding_by_name(*self.m.alias_chain(r.name).last().unwrap())
            {
                let rhs = self.m.binding(b).rhs;
                if self.head(rhs) == Some("elementof") {
                    found.insert(self.m.binding(b).name);
                    continue;
                }
                stack.push(rhs);
            }
            self.m.for_each_child(id, |k| stack.push(k));
        }
        let order: Vec<Symbol> = self
            .m
            .bindings()
            .map(|(_, b)| b.name)
            .filter(|n| found.contains(n))
            .collect();
        for sym in order {
            let rhs = self.m.binding(self.m.binding_by_name(sym).unwrap()).rhs;
            let stan = self.fresh(self.name(sym));
            let vs = self.m.valueset_of(rhs).cloned();
            let hints = self.set_hints(self.call(rhs).unwrap().args[0]);
            let t = self.declare(rhs, vs.as_ref(), true, &stan, &hints)?;
            self.params.push(format!("{};", t.render(&stan)));
            self.record_decl(&stan, t);
            self.env.insert(sym, stan.clone());
            // A free input is a scoring input, as a kernel input is.
            self.fields.insert(sym, stan.clone());
            self.manifest.push(ManifestParam {
                stan,
                field: self.name(sym).to_string(),
                input: None,
            });
        }
        Ok(())
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
            self.bound_by_support(&lhs, o.law)?;
            let lines = self.tilde(&lhs, o.law)?;
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

    /// Data observed under an unnormalized `truncate` outside its support
    /// has zero density: its `data` declaration takes the support as bounds,
    /// so Stan rejects such data instead of scoring it.
    fn bound_by_support(&mut self, data: &str, law: NodeId) -> Result<()> {
        let r = self.resolve(law);
        let r = match self.head(r) {
            Some("iid") => self.resolve(self.call(r).unwrap().args[0]),
            _ => r,
        };
        if self.head(r) != Some("truncate") {
            return Ok(());
        }
        let Some((lo, hi)) = self.law(r, false)?.support else {
            return Ok(());
        };
        let Some(t) = self.decls.get(data).cloned() else {
            return Ok(());
        };
        let old = format!("{};", t.render(data));
        let Some(pos) = self.data.iter().position(|l| *l == old) else {
            return Ok(());
        };
        let mut t = t;
        if t.bounds.lower.is_none() {
            t.bounds.lower = lo;
        }
        if t.bounds.upper.is_none() {
            t.bounds.upper = hi;
        }
        self.data[pos] = format!("{};", t.render(data));
        self.record_decl(data, t);
        Ok(())
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
        let e: Ex = self.expr(o.value)?;
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
        let flat = self.flat_order(lhs);
        let (law, per_element) = match self.head(m) {
            Some("Lebesgue") => return Ok(vec![]),
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
                        out.push(format!(
                            "{} ~ {};",
                            law.lhs(&format!("{lhs}[{}]", k + 1)),
                            law.rhs(None)
                        ));
                        out.extend(law.extra);
                    }
                    return Ok(out);
                }
            },
            _ => (self.law(m, false)?, false),
        };
        self.declared_variate(lhs, &law, m)?;
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
                "for ({i} in 1:{n}) {{\n  {} ~ {};\n}}",
                law.lhs(&format!("{lhs}[{i}]")),
                law.rhs(index)
            );
            self.used.remove(&i);
            let mut out = vec![stmt];
            out.extend(law.extra);
            return Ok(out);
        }
        let (target, rhs) = match (flat, &law.flat_call) {
            (Some(Flat::Row), Some(call)) if !law.multivariate => {
                (format!("to_array_1d({lhs})"), law.truncate(call.clone()))
            }
            (Some(Flat::Col), Some(call)) if !law.multivariate => {
                (format!("to_vector({lhs})"), law.truncate(call.clone()))
            }
            _ if lhs_rank >= 2 && !law.multivariate => (format!("to_vector({lhs})"), law.rhs(None)),
            _ => (lhs.to_string(), law.rhs(None)),
        };
        let mut out = vec![format!("{} ~ {rhs};", law.lhs(&target))];
        out.extend(law.extra);
        Ok(out)
    }

    /// A law whose restriction the declaration carries is exact only on a
    /// parameter declared from its sets: elsewhere the restriction is lost.
    fn declared_variate(&self, lhs: &str, law: &Law, at: NodeId) -> Result<()> {
        if law.declared && !self.set_declared.contains(lhs) {
            return Err(Refusal::not_yet(
                Some(at),
                "an ordered restriction of a variate that is not a parameter",
            ));
        }
        Ok(())
    }

    fn fresh_index(&mut self) -> String {
        for c in ["i", "j", "k", "n"] {
            if !self.used.contains(c) {
                self.used.insert(c.to_string());
                return c.to_string();
            }
        }
        self.fresh("idx")
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
        let law = self.law(m, false)?;
        if law.trunc.is_some() || law.support.is_some() {
            return Err(Refusal::not_yet(
                Some(id),
                "logdensityof of a truncated law",
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
        self.law_density(id, &law, &y, None)
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
    fn law(&mut self, m: NodeId, broadcast: bool) -> Result<Law> {
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
                if self.head(inner) != Some("truncate") {
                    return Err(Refusal::not_yet(Some(m), "normalize of this measure"));
                }
                let c = self.call(inner).unwrap();
                if let Some(law) = self.ordered_iid(m, c.args[0], c.args[1])? {
                    return Ok(law);
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
                self.truncated(inner)
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
                let Node::Const(f) = self.m.node(c.args[0]) else {
                    // A user law: one vectorized statement when it can be,
                    // else its body per element, in a loop.
                    if let Some((body, ctx, nfull)) = self.vectorized_user_law(m)? {
                        let outer = self.fctx.replace(ctx);
                        let law = self.law(body, false);
                        self.fctx = outer;
                        // A template operator between two whole arrays would
                        // not be elementwise.
                        if let Ok(law) = law
                            && law.vectorized
                            && !law.multivariate
                            && (nfull < 2 || !law.call.contains(['*', '/', '^']))
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
            "pushfwd" => {
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
    fn support(&mut self, law: NodeId) -> Result<Option<(Option<String>, Option<String>)>> {
        let m = self.resolve(law);
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
            Some("truncate") => {
                let set = self.resolve(self.call(m).unwrap().args[1]);
                self.interval(set).map(Some)
            }
            Some("Uniform") => {
                let s = self.resolve(self.law_args("Uniform", m)?[0]);
                self.interval(s).map(Some)
            }
            _ => Ok(None),
        }
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
        let (mut lo, hi) = self.interval(set)?;
        // `posreals` excludes 0: an integer variate starts at 1. Stan's
        // discrete `T[L, U]` keeps `L`.
        if law.discrete && matches!(self.m.node(set), Node::Const(c) if self.name(*c) == "posreals")
        {
            lo = Some("1".into());
        }
        law.trunc = Some((lo, hi));
        Ok(law)
    }

    /// The finite ends of an interval set; `None` for an open side.
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
        let mut found = None;
        let general = format!(
            "{name}({})",
            flatppl_infer::distribution_param_names(name)
                .unwrap_or_default()
                .join(", ")
        );
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
            "custom_lpdf" => {
                return Err(Refusal::not_yet(
                    Some(m),
                    format!("the law `{name}` (a user-defined Stan density)"),
                ));
            }
            _ => {}
        }
        if row.status.starts_with("unverified") {
            return Err(Refusal::not_yet(
                Some(m),
                format!("the law `{name}` (its Stan mapping is unverified)"),
            ));
        }
        if !row.normalized {
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
        let flat = self.flat.take().filter(|_| scalar_variate);
        let mut flat_args = None;
        if flat.is_some() {
            self.flat = flat;
            let filled = self.fill(&row.args, &binds, &is_per, None);
            self.flat = None;
            flat_args = Some(filled?);
        }
        let mut args = self.fill(&row.args, &binds, &is_per, None)?;
        let mut indexed = self.fill(&row.args, &binds, &is_per, Some(LOOP))?;
        // A template's elementwise operators become scalar ones when every
        // operand they can see is a scalar (`1, p ./ (1 - p)` for one `p`).
        let rank_of = |me: &Self, b: &pattern::Bound| match b {
            pattern::Bound::Node(n) => me.shape(*n).map(|s| s.rank).unwrap_or(0),
            _ => 0,
        };
        if binds.values().all(|b| rank_of(self, b) == 0) {
            args = scalar_operators(&args);
            flat_args = flat_args.map(|a| scalar_operators(&a));
        }
        if binds
            .iter()
            .all(|(v, b)| rank_of(self, b) == 0 || (is_per(v) && rank_of(self, b) == 1))
        {
            indexed = scalar_operators(&indexed);
        }
        let vector_per_element = per
            .iter()
            .any(|(v, &p)| p && expected.get(v).copied().unwrap_or(0) > 0);
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
            extra: vec![],
            declared: false,
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
            if let Some(input) = self.self_ref(na.value)
                && let Some(p) = self.manifest.iter_mut().find(|p| p.field == field)
            {
                p.input = Some(self.m.resolve(input).to_string());
            }
        }
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
