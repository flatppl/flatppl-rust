//! Exact posterior laws for independent conjugate fields (§06 posterior construction).

use std::collections::HashMap;

use flatppl_core::calling::reified_inputs;
use flatppl_core::graph::{
    build_call as op, declared_binding, draw_argument, expect_builtin_call, resolve_ref,
};
use flatppl_core::{
    CallHead, Dim, Module, NamedArg, NamedKind, Node, NodeId, RefNs, Scalar, ScalarType, Symbol,
    Type,
};
use flatppl_infer::constructors::{find_parameter, split_kernel_constructor};

use crate::conjugacy::match_scalar_row;
use crate::record_marginals::match_dirichlet_observation;
use crate::{ConjugateRowId, measure_stochastic_ancestors, references_latent};

mod evidence;
pub use evidence::{beta_log_evidence, conjugate_log_evidence};

/// The posterior record contains one field with this native probability law.
/// The posterior has a probability law only when `valid` holds.
pub struct ConjugatePosterior {
    pub update: NodeId,
    pub field: Symbol,
    pub distribution: NodeId,
    pub valid: NodeId,
}

/// Recognize `normalize(bayesupdate(L, prior))` in an inferred graph. Prior
/// and likelihood share a single named parameter. Every likelihood factor
/// must have a conjugate update before the result is usable.
pub fn normalized_conjugate_posterior(
    m: &mut Module,
    measure: NodeId,
) -> Option<ConjugatePosterior> {
    let [posterior] = operands(m, measure, "normalize")? else {
        return None;
    };
    conjugate_update(m, *posterior, None)
}

pub struct ConjugateProduct {
    pub update: NodeId,
    /// Each field and its native law, in prior order.
    pub fields: Vec<ConjugateField>,
    pub valid: NodeId,
}

pub struct ConjugateField {
    pub field: Symbol,
    pub distribution: NodeId,
    pub valid: NodeId,
    pub likelihoods: Vec<Symbol>,
}

pub fn normalized_conjugate_product(m: &mut Module, measure: NodeId) -> Option<ConjugateProduct> {
    let [posterior] = operands(m, measure, "normalize")? else {
        return None;
    };
    conjugate_product(m, *posterior)
}

/// Match the whole posterior before producing independent native laws.
pub fn conjugate_product(m: &mut Module, posterior: NodeId) -> Option<ConjugateProduct> {
    let (update, matched) = match_updates(m, posterior)?;
    let mut fields = Vec::with_capacity(matched.len());
    for matched in matched {
        fields.push(apply_field(m, matched)?);
    }
    let valid = fields
        .iter()
        .map(|field| field.valid)
        .reduce(|valid, next| op(m, "land", &[valid, next]))?;
    Some(ConjugateProduct {
        update,
        fields,
        valid,
    })
}

/// Inspect the same update that sampling lowers. A failed match leaves the
/// mathematical question open. Factor names are collected only for inspection.
pub fn conjugate_update(
    m: &mut Module,
    posterior: NodeId,
    likelihoods: Option<&mut Vec<Symbol>>,
) -> Option<ConjugatePosterior> {
    let (update, matched) = match_single_update(m, posterior)?;
    let result = apply_field(m, matched)?;
    if let Some(names) = likelihoods {
        names.extend(result.likelihoods);
    }
    Some(ConjugatePosterior {
        update,
        field: result.field,
        distribution: result.distribution,
        valid: result.valid,
    })
}

fn apply_field(m: &mut Module, matched: MatchedUpdate) -> Option<ConjugateField> {
    let applied = evidence::apply(m, &matched, false)?;
    let law = applied.law;
    let valid = op(m, "land", &[matched.model_valid, applied.supported]);
    let updated = law.valid(m);
    let valid = op(m, "land", &[valid, updated]);
    let distribution = law.distribution(m, valid, matched.model_valid);
    Some(ConjugateField {
        field: matched.field,
        distribution,
        valid,
        likelihoods: matched.factors.iter().map(|factor| factor.family).collect(),
    })
}

struct MatchedUpdate {
    field: Symbol,
    prior: UpdatedLaw,
    model_valid: NodeId,
    factors: Vec<MatchedFactor>,
}

struct MatchedFactor {
    family: Symbol,
    row: Update,
    data: Data,
}

fn match_single_update(m: &mut Module, posterior: NodeId) -> Option<(NodeId, MatchedUpdate)> {
    let (update, matched) = match_updates(m, posterior)?;
    let [matched]: [MatchedUpdate; 1] = matched.try_into().ok()?;
    Some((update, matched))
}

/// Check the complete update before a consumer constructs any summaries.
fn match_updates(m: &mut Module, posterior: NodeId) -> Option<(NodeId, Vec<MatchedUpdate>)> {
    let update = resolve_ref(m, posterior).0;
    let [likelihood, prior] = operands(m, update, "bayesupdate")? else {
        return None;
    };
    let (likelihood, prior) = (*likelihood, *prior);
    let priors = if let Some([value]) = operands(m, prior, "lawof") {
        let (field, value) = single_field(m, *value, "record")?;
        vec![(field, draw_argument(m, declared_binding(m, value).0)?)]
    } else {
        named_fields(m, prior, "joint")?
            .iter()
            .map(|f| (f.name, f.value))
            .collect()
    };
    let mut factors = Vec::new();
    likelihood_factors(m, likelihood, &mut Vec::new(), &mut factors)?;
    let multi = priors.len() > 1;
    let mut observations: HashMap<Symbol, Vec<Observation>> = HashMap::new();
    for (kernel, data) in factors {
        if multi {
            product_observations(m, kernel, data, &priors, &mut observations)?;
        } else {
            let field = priors.first()?.0;
            let observed = observation(m, kernel, data, field)?;
            observations.entry(field).or_default().push(observed);
        }
    }
    let mut matched = Vec::with_capacity(priors.len());
    for (field, prior) in priors {
        matched.push(match_field(m, field, prior, observations.remove(&field)?)?);
    }
    observations.is_empty().then_some((update, matched))
}

fn match_field(
    m: &mut Module,
    field: Symbol,
    prior: NodeId,
    observations: Vec<Observation>,
) -> Option<MatchedUpdate> {
    if !measure_stochastic_ancestors(m, prior, &[]).is_empty() {
        return None;
    }
    let (family, args) = split_kernel_constructor(m, resolve_ref(m, prior).0)?;
    let law = UpdatedLaw::from_prior(m, family, &args)?;
    let mut model_valid = law.valid(m);
    let mut matched = Vec::with_capacity(observations.len());
    for observation in observations {
        let Observation {
            latent,
            body,
            data,
            batch,
        } = observation;
        let (observations, options) = split_kernel_constructor(m, resolve_ref(m, body).0)?;
        let row = if matches!(law, UpdatedLaw::Dirichlet(_)) {
            let (_, dimensions, component) =
                match_dirichlet_observation(m, latent, &args, (observations, &options), &[])?;
            Update::Categories {
                dimensions,
                trials: component.trials,
                one_based: m.resolve(observations) == "Categorical",
            }
        } else {
            let row = match_scalar_row(m, (family, &args), (observations, &options), latent, &[])?
                .ok()?;
            let arg = |p| find_parameter(m, &options, p);
            match row {
                ConjugateRowId::BetaBernoulli => Update::Binomial(None),
                ConjugateRowId::BetaBinomial => Update::Binomial(Some(arg("n")?)),
                ConjugateRowId::GammaPoissonRate => Update::Poisson,
                ConjugateRowId::NormalMean => Update::NormalMean(arg("sigma")?),
                ConjugateRowId::InverseGammaVariance => Update::NormalVariance(arg("mu")?),
                _ => return None,
            }
        };
        // A reified kernel integrates any other unobserved draw. Conditioning
        // on such a nuisance would change the posterior.
        if row
            .nuisance()
            .is_some_and(|p| !measure_stochastic_ancestors(m, p, &[]).is_empty())
        {
            return None;
        }
        let ty = m.type_of(data).cloned().or_else(|| {
            flatppl_infer::infer_expression(m, data, &[])
                .0
                .remove(&data)
        })?;
        if let Some((n, size)) = batch {
            let n = m.alloc(Node::Lit(Scalar::Int(n.into())));
            let delta = op(m, "sub", &[size, n]);
            let agrees = op(m, "iszero", &[delta]);
            model_valid = op(m, "land", &[model_valid, agrees]);
        }
        let data = Data::new(m, data, batch.map(|(n, _)| n), &row, &ty)?;
        if let Some(check) = row.parameter_domain(m) {
            model_valid = op(m, "land", &[model_valid, check]);
        }
        matched.push(MatchedFactor {
            family: observations,
            row,
            data,
        });
    }
    Some(MatchedUpdate {
        field,
        prior: law,
        model_valid,
        factors: matched,
    })
}

struct Observation {
    latent: Symbol,
    body: NodeId,
    data: NodeId,
    batch: Option<(u32, NodeId)>,
}

/// Split a named joint likelihood only when each component reads precisely one
/// kernel input. External values shared by components are conditioned inputs.
fn product_observations(
    m: &mut Module,
    kernel: NodeId,
    data: NodeId,
    priors: &[(Symbol, NodeId)],
    out: &mut HashMap<Symbol, Vec<Observation>>,
) -> Option<()> {
    let kernel = resolve_ref(m, kernel).0;
    let Node::Call(call) = m.node(kernel) else {
        return None;
    };
    let inputs = reified_inputs(m, kernel, call.inputs.as_ref()?)?;
    if let [(field, _)] = inputs {
        if !priors.iter().any(|(name, _)| name == field) {
            return None;
        }
        let field = *field;
        let observed = observation(m, kernel, data, field)?;
        out.entry(field).or_default().push(observed);
        return Some(());
    }
    if inputs.is_empty()
        || inputs.iter().any(|(field, target)| {
            target.ns != RefNs::SelfMod || !priors.iter().any(|(name, _)| name == field)
        })
    {
        return None;
    }
    let inputs = inputs.to_vec();
    let call = expect_builtin_call(m, kernel, "functionof")?;
    let [body] = *call.args else { return None };
    let factors = named_fields(m, declared_binding(m, body).0, "joint")?.to_vec();
    let labels = flatppl_core::calling::splat_fields(m, data)?;
    if labels.len() != factors.len() || factors.iter().any(|f| !labels.contains(&f.name)) {
        return None;
    }
    for factor in factors {
        let mut owners = inputs
            .iter()
            .filter(|(_, target)| references_latent(m, factor.value, target.name, &[]));
        let &(field, target) = owners.next()?;
        if owners.next().is_some() {
            return None;
        }
        let value = flatppl_core::calling::project_field(m, data, factor.name);
        let body = declared_binding(m, factor.value).0;
        let observed = batched_observation(m, body, value, target.name)?;
        out.entry(field).or_default().push(observed);
    }
    Some(())
}

/// Open a kernel at its declared boundary and pair its event with the data.
fn observation(
    m: &mut Module,
    kernel: NodeId,
    observed: NodeId,
    field: Symbol,
) -> Option<Observation> {
    let kernel = resolve_ref(m, kernel).0;
    let Node::Call(call) = m.node(kernel) else {
        return None;
    };
    let CallHead::Builtin(head) = call.head else {
        return None;
    };
    let reification = m.resolve(head);
    if !matches!(reification, "kernelof" | "functionof") {
        return None;
    }
    let [(label, target)] = reified_inputs(m, kernel, call.inputs.as_ref()?)? else {
        return None;
    };
    if *label != field || target.ns != RefNs::SelfMod {
        return None;
    }
    let latent = target.name;
    let [body] = *call.args else { return None };
    let mut body = declared_binding(m, body).0;
    let mut data = observed;
    if reification == "kernelof" {
        if let Some((label, value)) = single_field(m, body, "record") {
            if flatppl_core::calling::splat_fields(m, data)?.as_slice() != [label] {
                return None;
            }
            // An input record can replace a literal record bound in the source.
            data = flatppl_core::calling::project_field(m, data, label);
            body = declared_binding(m, value).0;
        }
        body = draw_argument(m, body)?;
    }
    batched_observation(m, body, data, latent)
}

fn batched_observation(
    m: &mut Module,
    body: NodeId,
    data: NodeId,
    latent: Symbol,
) -> Option<Observation> {
    let (body, batch) = if let Some([base, size]) = operands(m, body, "iid") {
        let (base, size) = (*base, *size);
        if references_latent(m, size, latent, &[]) {
            return None;
        }
        (
            base,
            Some((
                u32::try_from(flatppl_infer::fixed_int_value(m, size)?).ok()?,
                size,
            )),
        )
    } else {
        (body, None)
    };
    Some(Observation {
        latent,
        body,
        data,
        batch,
    })
}

fn operands<'a>(m: &'a Module, node: NodeId, name: &str) -> Option<&'a [NodeId]> {
    let c = expect_builtin_call(m, resolve_ref(m, node).0, name)?;
    c.named.is_empty().then_some(&c.args)
}

fn single_field(m: &Module, id: NodeId, head: &str) -> Option<(Symbol, NodeId)> {
    let [field] = named_fields(m, id, head)? else {
        return None;
    };
    Some((field.name, field.value))
}

fn named_fields<'m>(m: &'m Module, id: NodeId, head: &str) -> Option<&'m [NamedArg]> {
    let call = expect_builtin_call(m, resolve_ref(m, id).0, head)?;
    (call.args.is_empty()
        && !call.named.is_empty()
        && call
            .named
            .iter()
            .all(|field| field.kind == NamedKind::Field))
    .then_some(&call.named)
}

fn likelihood_factors(
    m: &Module,
    id: NodeId,
    path: &mut Vec<NodeId>,
    out: &mut Vec<(NodeId, NodeId)>,
) -> Option<()> {
    let id = resolve_ref(m, id).0;
    if let Some([kernel, data]) = operands(m, id, "likelihoodof") {
        out.push((*kernel, *data));
        return Some(());
    }
    let factors = operands(m, id, "joint_likelihood")?;
    if factors.is_empty() || path.contains(&id) {
        return None;
    }
    path.push(id);
    for &factor in factors {
        likelihood_factors(m, factor, path, out)?;
    }
    path.pop();
    Some(())
}

#[derive(Clone, Copy)]
enum UpdatedLaw {
    Beta(NodeId, NodeId),
    Gamma(NodeId, NodeId),
    Normal(NodeId, NodeId),
    InverseGamma(NodeId, NodeId),
    Dirichlet(NodeId),
}

#[derive(Clone, Copy)]
enum Update {
    Binomial(Option<NodeId>),
    Poisson,
    NormalMean(NodeId),
    NormalVariance(NodeId),
    Categories {
        dimensions: u32,
        trials: Option<NodeId>,
        one_based: bool,
    },
}

impl Update {
    fn nuisance(&self) -> Option<NodeId> {
        match *self {
            Self::Binomial(n) | Self::Categories { trials: n, .. } => n,
            Self::NormalMean(sigma) | Self::NormalVariance(sigma) => Some(sigma),
            Self::Poisson => None,
        }
    }

    fn parameter_domain(&self, m: &mut Module) -> Option<NodeId> {
        Some(match *self {
            Self::Binomial(Some(n))
            | Self::Categories {
                trials: Some(n), ..
            } => valid_trials(m, n),
            Self::NormalMean(sigma) => positive(m, sigma),
            Self::NormalVariance(mean) => op(m, "isfinite", &[mean]),
            _ => return None,
        })
    }
}

#[derive(Clone, Copy)]
struct Data {
    value: NodeId,
    count: Option<u32>,
    integer: bool,
}

impl Data {
    fn new(
        m: &mut Module,
        value: NodeId,
        count: Option<u32>,
        row: &Update,
        ty: &Type,
    ) -> Option<Self> {
        let cell = if let Some(n) = count {
            let Type::Array { shape, elem } = ty else {
                return None;
            };
            // A fixed size that evaluates to zero has a dynamic axis in FlatPIR.
            let empty = n == 0 && shape.as_ref() == [Dim::Dynamic];
            if !empty && shape.as_ref() != [Dim::Static(n)] {
                return None;
            }
            elem.as_ref()
        } else {
            ty
        };
        let matches = match row {
            Update::Categories {
                dimensions,
                trials: Some(_),
                ..
            } => {
                matches!(cell, Type::Array { shape, elem } if shape.as_ref() == [Dim::Static(*dimensions)]
                    && matches!(elem.as_ref(), Type::Scalar(ScalarType::Integer | ScalarType::Real)))
            }
            Update::Binomial(None) => matches!(
                cell,
                Type::Scalar(ScalarType::Boolean | ScalarType::Integer | ScalarType::Real)
            ),
            _ => matches!(cell, Type::Scalar(ScalarType::Integer | ScalarType::Real)),
        };
        let boolean = *cell == Type::Scalar(ScalarType::Boolean);
        let leaf = match cell {
            Type::Array { elem, .. } => elem.as_ref(),
            scalar => scalar,
        };
        let integer = boolean || *leaf == Type::Scalar(ScalarType::Integer);
        if !matches {
            return None;
        }
        let mut data = Self {
            value,
            count,
            integer,
        };
        if boolean {
            let one = m.alloc(Node::Lit(Scalar::Int(1)));
            let zero = m.alloc(Node::Lit(Scalar::Int(0)));
            data.value = data.apply(m, "ifelse", &[value, one, zero]);
        }
        Some(data)
    }

    fn apply(self, m: &mut Module, name: &str, args: &[NodeId]) -> NodeId {
        if self.count.is_some() {
            map(m, name, args)
        } else {
            op(m, name, args)
        }
    }

    fn reduce(self, m: &mut Module, name: &str, values: NodeId) -> NodeId {
        if self.count.is_some() {
            op(m, name, &[values])
        } else {
            values
        }
    }

    fn sum(self, m: &mut Module, values: NodeId) -> NodeId {
        self.reduce(m, "sum", values)
    }

    fn count(self, m: &mut Module) -> NodeId {
        real(m, f64::from(self.count.unwrap_or(1)))
    }
}

fn real(m: &mut Module, n: f64) -> NodeId {
    m.alloc(Node::Lit(Scalar::Real(n)))
}

fn map(m: &mut Module, name: &str, args: &[NodeId]) -> NodeId {
    let head = m.intern(name);
    let head = m.alloc(Node::Const(head));
    let args: Vec<_> = std::iter::once(head).chain(args.iter().copied()).collect();
    op(m, "broadcast", &args)
}

fn positive(m: &mut Module, p: NodeId) -> NodeId {
    let finite = op(m, "isfinite", &[p]);
    let zero = real(m, 0.0);
    let pos = op(m, "gt", &[p, zero]);
    op(m, "land", &[finite, pos])
}

impl UpdatedLaw {
    fn from_prior(m: &Module, family: Symbol, args: &[(Symbol, NodeId)]) -> Option<Self> {
        let p = |name| find_parameter(m, args, name);
        match m.resolve(family) {
            "Beta" => Some(Self::Beta(p("alpha")?, p("beta")?)),
            "Gamma" => Some(Self::Gamma(p("shape")?, p("rate")?)),
            "Normal" => Some(Self::Normal(p("mu")?, p("sigma")?)),
            "InverseGamma" => Some(Self::InverseGamma(p("shape")?, p("scale")?)),
            "Dirichlet" => Some(Self::Dirichlet(p("alpha")?)),
            _ => None,
        }
    }

    fn valid(&self, m: &mut Module) -> NodeId {
        match *self {
            Self::Beta(a, b) | Self::Gamma(a, b) | Self::InverseGamma(a, b) => {
                let a = positive(m, a);
                let b = positive(m, b);
                op(m, "land", &[a, b])
            }
            Self::Normal(mean, sigma) => {
                let mean = op(m, "isfinite", &[mean]);
                let sigma = positive(m, sigma);
                op(m, "land", &[mean, sigma])
            }
            Self::Dirichlet(alpha) => {
                let zero = real(m, 0.0);
                let positive = map(m, "gt", &[alpha, zero]);
                let finite = map(m, "isfinite", &[alpha]);
                let valid = map(m, "land", &[positive, finite]);
                op(m, "lall", &[valid])
            }
        }
    }

    fn distribution(self, m: &mut Module, valid: NodeId, model_valid: NodeId) -> NodeId {
        let one = real(m, 1.0);
        let zero = real(m, 0.0);
        // Zero evidence has an undefined draw. An invalid source parameter
        // must also trip the native constructor's positive-parameter check.
        let fallback = op(m, "ifelse", &[model_valid, one, zero]);
        let (name, a, b) = match self {
            Self::Beta(a, b) => ("Beta", a, b),
            Self::Normal(a, b) => ("Normal", a, b),
            Self::Gamma(a, b) => ("Gamma", a, b),
            Self::InverseGamma(a, b) => ("InverseGamma", a, b),
            Self::Dirichlet(a) => {
                let a = map(m, "ifelse", &[valid, a, fallback]);
                return op(m, "Dirichlet", &[a]);
            }
        };
        let a = op(m, "ifelse", &[valid, a, one]);
        let b = op(m, "ifelse", &[valid, b, fallback]);
        op(m, name, &[a, b])
    }
}

fn valid_trials(m: &mut Module, n: NodeId) -> NodeId {
    let positive = positive(m, n);
    if matches!(m.type_of(n), Some(Type::Scalar(ScalarType::Integer)))
        || matches!(m.node(n), Node::Lit(Scalar::Int(_)))
    {
        return positive;
    }
    let real_n = op(m, "real", &[n]);
    let floor = op(m, "floor", &[real_n]);
    let integer = op(m, "le", &[real_n, floor]);
    op(m, "land", &[positive, integer])
}

/// Check count support before summing, so a negative or fractional cell
/// cannot be hidden by another observation. Bounds compare the source values.
fn supported_counts(
    m: &mut Module,
    value: NodeId,
    integer: bool,
    upper: Option<NodeId>,
    array: bool,
) -> NodeId {
    let apply = if array { map } else { op };
    let zero = m.alloc(Node::Lit(Scalar::Int(0)));
    let mut valid = apply(m, "ge", &[value, zero]);
    if !integer {
        let finite = apply(m, "isfinite", &[value]);
        let floor = apply(m, "floor", &[value]);
        let whole = apply(m, "le", &[value, floor]);
        let whole = apply(m, "land", &[finite, whole]);
        valid = apply(m, "land", &[valid, whole]);
    }
    if let Some(max) = upper {
        let in_range = apply(m, "le", &[value, max]);
        valid = apply(m, "land", &[valid, in_range]);
    }
    if array {
        op(m, "lall", &[valid])
    } else {
        valid
    }
}

struct MultinomialData {
    cells: NodeId,
    counts: NodeId,
    totals: NodeId,
    real_totals: NodeId,
}

fn multinomial_data(m: &mut Module, data: Data, size: NodeId) -> MultinomialData {
    let cells = if data.count.is_some() {
        op(m, "rowstack", &[data.value])
    } else {
        data.value
    };
    let totals = data.apply(m, "sum", &[data.value]);
    let reals = map(m, "real", &[cells]);
    let (counts, real_totals) = if let Some(rows) = data.count {
        let rows = m.alloc(Node::Lit(Scalar::Int(rows.into())));
        let one = real(m, 1.0);
        let across_rows = op(m, "fill", &[one, rows]);
        let across_categories = op(m, "fill", &[one, size]);
        let transposed = op(m, "transpose", &[reals]);
        (
            op(m, "mul", &[transposed, across_rows]),
            op(m, "mul", &[reals, across_categories]),
        )
    } else {
        (reals, op(m, "sum", &[reals]))
    };
    MultinomialData {
        cells,
        counts,
        totals,
        real_totals,
    }
}

impl MultinomialData {
    fn supported(&self, m: &mut Module, data: Data, trials: NodeId) -> NodeId {
        let valid_cells = supported_counts(m, self.cells, data.integer, Some(trials), true);
        let diff = data.apply(m, "sub", &[self.totals, trials]);
        let matches = data.apply(m, "iszero", &[diff]);
        // Nonnegative integer totals that wrap back to n exceed 2n in real arithmetic.
        let two = real(m, 2.0);
        let limit = op(m, "mul", &[trials, two]);
        let no_wrap = data.apply(m, "le", &[self.real_totals, limit]);
        let rows_valid = data.apply(m, "land", &[matches, no_wrap]);
        let rows_valid = data.reduce(m, "lall", rows_valid);
        op(m, "land", &[valid_cells, rows_valid])
    }
}

fn category_counts(m: &mut Module, data: Data, d: u32, one_based: bool) -> (NodeId, NodeId) {
    if data.count.is_some() {
        let boundary = flatppl_core::query::promoted_input_names(m);
        let (counts, valid) = crate::sufficient_statistics::categorical::label_counts(
            m,
            data.value,
            data.integer,
            d,
            one_based,
            &boundary,
        );
        return (map(m, "real", &[counts]), valid);
    }
    let size = m.alloc(Node::Lit(Scalar::Int(d.into())));
    let offset = i64::from(one_based);
    let last = offset + i64::from(d) - 1;
    let upper = m.alloc(Node::Lit(Scalar::Int(last)));
    let mut valid = supported_counts(
        m,
        data.value,
        data.integer,
        Some(upper),
        data.count.is_some(),
    );
    if one_based {
        let zero = m.alloc(Node::Lit(Scalar::Int(0)));
        let positive = data.apply(m, "gt", &[data.value, zero]);
        let positive = data.reduce(m, "lall", positive);
        valid = op(m, "land", &[valid, positive]);
    }
    if d == 1 {
        let n = data.count(m);
        return (op(m, "vector", &[n]), valid);
    }
    let first = real(m, offset as f64);
    let last = real(m, last as f64);
    let labels = op(m, "linspace", &[first, last, size]);
    let delta = map(m, "sub", &[labels, data.value]);
    let same = map(m, "iszero", &[delta]);
    let counts = map(m, "real", &[same]);
    (counts, valid)
}
