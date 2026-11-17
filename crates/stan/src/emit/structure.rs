//! Target decomposition and the `~` statements.

use super::{Emitter, Ex, P_ATOM, P_MUL, PriorField};
use crate::refuse::{Refusal, RefusalKind, Result};
use crate::stan_type::StanType;
use crate::tables::{self, ArgMap};
use crate::{ManifestParam, Truncation};
use flatppl_core::{Inputs, NamedKind, Node, NodeId, RefNs, Symbol, Type, ValueSet};

/// A law as a Stan sampling statement right-hand side.
struct Law {
    stan: String,
    args: Vec<Ex>,
    /// `T[lower, upper]`, either side open.
    trunc: Option<(Option<String>, Option<String>)>,
    vectorized: bool,
    multivariate: bool,
    /// The argument expressions that vary per element of a broadcast.
    per_element: Vec<bool>,
}

impl Law {
    fn rhs(&self, index: Option<&str>) -> String {
        let args: Vec<String> = self
            .args
            .iter()
            .zip(&self.per_element)
            .map(|(a, &per)| match (index, per) {
                (Some(i), true) => format!("{}[{i}]", a.at(P_ATOM)),
                _ => a.s.clone(),
            })
            .collect();
        let mut s = format!("{}({})", self.stan, args.join(", "));
        if let Some((lo, hi)) = &self.trunc {
            s.push_str(&format!(
                " T[{}, {}]",
                lo.as_deref().unwrap_or(""),
                hi.as_deref().unwrap_or("")
            ));
        }
        s
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
            let mut t = self.declare_ty(&ty, vs.as_ref(), true, f.node)?;
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
            let t = self.declare(rhs, vs.as_ref(), true)?;
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
            let lhs = self.observed_value(&o)?;
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
            let t = self.declare(o.value, None, false)?;
            self.data.push(format!("{};", t.render(&stan)));
            self.record_decl(&stan, t);
            self.json.insert(stan.clone(), v);
            return Ok(stan);
        }
        let stan = self.fresh(&o.name);
        let t = self.declare(o.value, None, false)?;
        self.tdata.push(format!("{} = {};", t.render(&stan), e.s));
        self.record_decl(&stan, t);
        Ok(stan)
    }

    // ---------------------------------------------------------------- laws

    /// Sampling statements for `lhs ~ measure`.
    fn tilde(&mut self, lhs: &str, measure: NodeId) -> Result<Vec<String>> {
        let lhs_rank = self.lhs_rank(lhs);
        let m = self.resolve(measure);
        match self.head(m) {
            Some("Lebesgue") => Ok(vec![]),
            Some("iid") => {
                let c = self.call(m).unwrap();
                let inner = c.args[0];
                let law = self.law(inner, false)?;
                if law.multivariate || law.trunc.is_some() || !law.vectorized {
                    let i = self.fresh_index();
                    let n = self.extent_of(lhs);
                    let stmt = format!(
                        "for ({i} in 1:{n}) {{\n  {lhs}[{i}] ~ {};\n}}",
                        law.rhs(None)
                    );
                    self.used.remove(&i);
                    return Ok(vec![stmt]);
                }
                let target = if lhs_rank >= 2 {
                    format!("to_vector({lhs})")
                } else {
                    lhs.to_string()
                };
                Ok(vec![format!("{target} ~ {};", law.rhs(None))])
            }
            Some("broadcast") => {
                let law = self.law(m, true)?;
                let loops = !law.vectorized || law.trunc.is_some() || law.multivariate;
                if loops {
                    let i = self.fresh_index();
                    let n = self.extent_of(lhs);
                    let stmt = format!(
                        "for ({i} in 1:{n}) {{\n  {lhs}[{i}] ~ {};\n}}",
                        law.rhs(Some(&i))
                    );
                    self.used.remove(&i);
                    return Ok(vec![stmt]);
                }
                let target = if lhs_rank >= 2 {
                    format!("to_vector({lhs})")
                } else {
                    lhs.to_string()
                };
                Ok(vec![format!("{target} ~ {};", law.rhs(None))])
            }
            _ => {
                let law = self.law(m, false)?;
                if law.trunc.is_some() && lhs_rank > 0 {
                    let i = self.fresh_index();
                    let n = self.extent_of(lhs);
                    let stmt = format!(
                        "for ({i} in 1:{n}) {{\n  {lhs}[{i}] ~ {};\n}}",
                        law.rhs(None)
                    );
                    self.used.remove(&i);
                    return Ok(vec![stmt]);
                }
                let target = if lhs_rank >= 2 && !law.multivariate {
                    format!("to_vector({lhs})")
                } else {
                    lhs.to_string()
                };
                Ok(vec![format!("{target} ~ {};", law.rhs(None))])
            }
        }
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
                if self.head(inner) != Some("truncate") {
                    return Err(Refusal::not_yet(Some(m), "normalize of this measure"));
                }
                let mut law = self.truncated(inner)?;
                if self.opts.truncation == Truncation::BoundOnly {
                    law.trunc = None;
                }
                Ok(law)
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
                let args = c.args[1..].to_vec();
                if !c.named.is_empty() {
                    return Err(Refusal::not_yet(
                        Some(m),
                        format!("{name}.(…) with keyword arguments"),
                    ));
                }
                let base_ranks = flatppl_infer::distribution_param_ranks(name).unwrap_or_default();
                let per: Vec<bool> = args
                    .iter()
                    .enumerate()
                    .map(|(k, &a)| {
                        let r = self.shape(a).map(|s| s.rank).unwrap_or(0);
                        r > base_ranks.get(k).copied().unwrap_or(0)
                    })
                    .collect();
                self.dist(m, name, &args, &per)
            }
            "locscale" => {
                let c = self.call(m).unwrap();
                let base = self.resolve(c.args[0]);
                let (shift, scale) = (c.args[1], c.args[2]);
                match self.head(base) {
                    Some("StudentT") => {
                        let nu = self.law_args("StudentT", base)?[0];
                        let nu = self.real(nu)?;
                        let shift = self.real(shift)?;
                        let scale = self.real(scale)?;
                        Ok(Law {
                            stan: "student_t".into(),
                            args: vec![nu, shift, scale],
                            trunc: None,
                            vectorized: true,
                            multivariate: false,
                            per_element: vec![false; 3],
                        })
                    }
                    _ => Err(Refusal::not_yet(Some(m), "locscale of this law")),
                }
            }
            "Uniform" => {
                let s = self.resolve(self.law_args("Uniform", m)?[0]);
                let (lo, hi) = self.interval(s)?;
                match (lo, hi) {
                    (Some(lo), Some(hi)) => Ok(Law {
                        stan: "uniform".into(),
                        args: vec![Ex::atom(lo), Ex::atom(hi)],
                        trunc: None,
                        vectorized: true,
                        multivariate: false,
                        per_element: vec![false; 2],
                    }),
                    _ => Err(Refusal::not_yet(Some(m), "a Uniform on an unbounded set")),
                }
            }
            name => {
                let args = self.law_args(name, m)?;
                let per = vec![false; args.len()];
                self.dist(m, name, &args, &per)
            }
        }
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

    fn dist(&mut self, m: NodeId, name: &str, args: &[NodeId], per: &[bool]) -> Result<Law> {
        let Some(map) = tables::law(name) else {
            return Err(Refusal::not_yet(Some(m), name.to_string()));
        };
        let mut out = Vec::new();
        let mut per_out = Vec::new();
        for a in map.args {
            match a {
                ArgMap::Arg(k) => {
                    let id = *args
                        .get(*k)
                        .ok_or_else(|| Refusal::not_yet(Some(m), format!("{name} arguments")))?;
                    out.push(self.expr(id)?);
                    per_out.push(per.get(*k).copied().unwrap_or(false));
                }
                ArgMap::Lit(s) => {
                    out.push(Ex::atom(*s));
                    per_out.push(false);
                }
                ArgMap::Odds(k) => {
                    let p = self.expr(args[*k])?;
                    out.push(Ex {
                        s: format!("{} / (1 - {})", p.at(P_MUL), p.at(P_MUL + 1)),
                        prec: P_MUL,
                    });
                    per_out.push(per.get(*k).copied().unwrap_or(false));
                }
            }
        }
        Ok(Law {
            stan: map.stan.to_string(),
            args: out,
            trunc: None,
            vectorized: map.vectorized,
            multivariate: map.multivariate,
            per_element: per_out,
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

fn ty_is_int(ty: &Type) -> bool {
    super::shape_of(ty).is_some_and(|s| s.int)
}

fn is_identifier(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && s.starts_with(|c: char| c.is_ascii_alphabetic())
}
