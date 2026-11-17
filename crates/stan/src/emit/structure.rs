//! Target decomposition and the `~` statements.

use super::pattern;
use super::{Emitter, Ex, P_ATOM, PriorField};
use crate::ManifestParam;
use crate::refuse::{Refusal, RefusalKind, Result};
use crate::stan_type::StanType;
use crate::tables;
use flatppl_core::{Inputs, NamedKind, Node, NodeId, RefNs, Symbol, Type, ValueSet};
use std::collections::HashMap;

/// The loop index a per-element argument of a broadcast law is rendered with.
const LOOP: &str = "\u{1}";

/// A law as a Stan sampling statement right-hand side.
struct Law {
    /// `stan(args)` for one statement over the whole variate.
    call: String,
    /// `stan(args)` with [`LOOP`] where a broadcast loop indexes.
    indexed: String,
    /// A broadcast varies a vector parameter per element, which Stan does not
    /// vectorize (`categorical` with one simplex per observation).
    vector_per_element: bool,
    /// The `distributions.ron` variate template (`{y}`, `{y} + 1`).
    lhs: String,
    /// `T[lower, upper]`, either side open.
    trunc: Option<(Option<String>, Option<String>)>,
    vectorized: bool,
    multivariate: bool,
    /// Stan defines `_lcdf` and `_lccdf`, so `T[L, U]` is well formed.
    lcdf: bool,
}

impl Law {
    fn rhs(&self, index: Option<&str>) -> String {
        let mut s = match index {
            Some(i) => self.indexed.replace(LOOP, i),
            None => self.call.clone(),
        };
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
                for f in &fields {
                    if let Some(law) = f.law {
                        let lhs = self.fields[&f.field].clone();
                        let lines = self.tilde(&lhs, law)?;
                        self.prior.extend(lines);
                    }
                }
                self.observe(obs)
            }
            Some("likelihoodof" | "joint_likelihood") => {
                let obs = self.likelihood(rhs, false)?;
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
                    out.push(PriorField {
                        field: na.name,
                        binding: Some(sym),
                        law: Some(self.call(rhs).unwrap().args[0]),
                        node: na.value,
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
                    });
                }
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
            let mut t = self.declare_ty(&ty, vs.as_ref(), true, f.node, &stan, &hints)?;
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
            let lines = self.tilde(&lhs, o.law)?;
            self.lik.extend(lines);
        }
        Ok(())
    }

    fn observed_value(&mut self, o: &Observation) -> Result<String> {
        // A data path or fixed binding prints as its own data name.
        let e: Ex = self.expr(o.value)?;
        if e.prec == P_ATOM && is_identifier(&e.s) {
            return Ok(e.s);
        }
        if let Some(v) = crate::data::const_json(self.m, o.value) {
            let stan = self.fresh(&o.name);
            let t = self.declare(o.value, None, false, &stan, &[])?;
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
        let lhs_rank = self.lhs_rank(lhs);
        let m = self.resolve(measure);
        let (law, per_element) = match self.head(m) {
            Some("Lebesgue") => return Ok(vec![]),
            // An iid product of one law: the law's arguments are shared.
            Some("iid") => (self.law(self.call(m).unwrap().args[0], false)?, false),
            Some("broadcast") => (self.law(m, true)?, true),
            _ => (self.law(m, false)?, false),
        };
        let elementwise = per_element || self.head(m) == Some("iid");
        let shifted = law.lhs("{y}") != "{y}";
        // A loop where Stan has no vectorized form: a law that does not
        // vectorize, a truncated law, a vector law over an array of variates,
        // or a shifted variate (`{y} + 1`) of an integer array.
        let one_variate = !elementwise && (law.multivariate || lhs_rank == 0);
        let loops = !one_variate
            && (!law.vectorized
                || law.trunc.is_some()
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
            return Ok(vec![stmt]);
        }
        let target = if lhs_rank >= 2 && !law.multivariate {
            format!("to_vector({lhs})")
        } else {
            lhs.to_string()
        };
        Ok(vec![format!("{} ~ {};", law.lhs(&target), law.rhs(None))])
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
                self.truncated(inner)
            }
            // An unnormalized restriction keeps the density; the declaration bound
            // carries the support.
            "truncate" => {
                let mut law = self.truncated(m)?;
                law.trunc = None;
                Ok(law)
            }
            "broadcast" if broadcast => {
                let c = self.call(m).unwrap();
                let Node::Const(f) = self.m.node(c.args[0]) else {
                    return Err(Refusal::not_yet(Some(m), "a broadcast of a user law"));
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
        let (lo, hi) = self.interval(set)?;
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
        for row in tables::law_rows_by_head(name, module) {
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
        let scalar_variate = row.variate.starts_with("real") || row.variate.starts_with("int");
        Ok(Law {
            call: format!("{}({args})", row.stan),
            indexed: format!("{}({indexed})", row.stan),
            vector_per_element,
            lhs: row.lhs.clone(),
            trunc: None,
            vectorized: row.vectorized,
            multivariate: !scalar_variate,
            lcdf: row.lcdf,
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

fn is_identifier(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && s.starts_with(|c: char| c.is_ascii_alphabetic())
}
