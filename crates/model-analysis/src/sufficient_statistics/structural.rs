//! Plans for an observed factor before measure lowering.

use super::{Factor, Reducer, call, resolve};
use flatppl_core::{
    Binding, BindingId, Call, Dim, Module, NamedKind, NodeId, ScalarType, Symbol, Type,
};
use std::collections::HashSet;

/// Data summaries and a complete scalar score for one observed IID Normal factor.
/// Expression IDs belong to the owned module. Evaluate summaries in the order
/// returned and keep the score at the original factor's scope.
pub struct IidNormalPlan {
    module: Module,
    summaries: Vec<BindingId>,
    score: NodeId,
    all_finite: NodeId,
    count: u32,
}

impl IidNormalPlan {
    /// The source snapshot with the factor's deterministic expressions.
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Data-only bindings in dependency order.
    pub fn summaries(&self) -> impl ExactSizeIterator<Item = &Binding> {
        self.summaries.iter().map(|&id| self.module.binding(id))
    }

    /// The fully normalized score expression in this plan's module.
    pub fn score(&self) -> NodeId {
        self.score
    }

    /// A backend with native parameter or data checks can retain its original
    /// density when an observation is non-finite.
    pub fn observations_are_finite(&self) -> NodeId {
        self.all_finite
    }

    /// The static plate size used in the formula. A backend that allows data to
    /// change this extent must retain ordinary scoring for other extents.
    pub fn observation_count(&self) -> u32 {
        self.count
    }
}

/// Plan `iid(Normal(mu, sigma), n)` at an observation site in an inferred module.
/// Pass the operand nodes from that site and every input bound by the structural
/// target. The query's reserved `inputs` binding is independent of this boundary.
/// Returns `None` when the plate, data-only observations or shared parameters
/// cannot be proved. The supplied module is immutable.
pub fn plan_iid_normal(
    m: &Module,
    law: NodeId,
    observed: NodeId,
    scoring_inputs: &[Symbol],
    data_inputs: &[Symbol],
) -> Option<IidNormalPlan> {
    let aliases = |inputs: &[Symbol]| -> Option<HashSet<Symbol>> {
        let mut names = HashSet::new();
        for &name in inputs {
            m.binding_by_name(name)?;
            names.extend(m.alias_chain(name));
        }
        Some(names)
    };
    for &name in data_inputs {
        let input = m.binding(m.binding_by_name(name)?).rhs;
        if !m.type_of(input)?.is_concrete_value() {
            return None;
        }
    }
    let allowed = aliases(data_inputs)?;
    let mut boundary = aliases(scoring_inputs)?;
    if !boundary.is_disjoint(&allowed) {
        return None;
    }
    boundary.extend(&allowed);

    let law = resolve(m, law, &boundary);
    let iid = call(m, law, "iid")?;
    let [base, size] = *iid.args else { return None };
    if iid.inputs.is_some() || !iid.named.is_empty() {
        return None;
    }
    let base = resolve(m, base, &boundary);
    let normal = call(m, base, "Normal")?;
    let parameters = normal_parameters(m, normal)?;
    let Type::Array { shape, elem } = m.type_of(observed)? else {
        return None;
    };
    let [Dim::Static(count)] = shape.as_ref() else {
        return None;
    };
    let count = *count;
    if count < 2 || elem.as_ref() != &Type::Scalar(ScalarType::Real) {
        return None;
    }
    let mut reducer = Reducer::new(boundary, allowed);
    if !reducer.data_only(m, observed) || !reducer.fixed(m, size) {
        return None;
    }

    let mut module = m.clone();
    if flatppl_infer::fixed_int_value(&mut module, size)? != i64::from(count) {
        return None;
    }
    let factor = Factor {
        constructor: module.intern("Normal"),
        parameters: parameters
            .into_iter()
            .map(|(name, value)| (module.intern(name), value))
            .collect(),
        observed,
    };
    let score = reducer.reduce_moment_factor(&mut module, &factor)?;
    let key = reducer.data_key(&module, observed);
    let all_finite = *reducer.cache.get(&(vec![key], super::NORMAL_FINITE))?;
    let summaries = reducer
        .summaries
        .iter()
        .map(|&name| module.binding_by_name(name))
        .collect::<Option<Vec<_>>>()?;
    for &id in &summaries {
        let rhs = module.binding(id).rhs;
        flatppl_infer::annotate_rebuilt(&mut module, rhs);
    }
    flatppl_infer::annotate_rebuilt(&mut module, score);
    Some(IidNormalPlan {
        module,
        summaries,
        score,
        all_finite,
        count,
    })
}

fn normal_parameters(m: &Module, call: &Call) -> Option<Vec<(&'static str, NodeId)>> {
    if call.inputs.is_some() || call.args.len() + call.named.len() != 2 {
        return None;
    }
    let mut parameters: Vec<_> = ["mu", "sigma"]
        .into_iter()
        .zip(call.args.iter().copied())
        .collect();
    for arg in &call.named {
        let name = match (arg.kind, m.resolve(arg.name)) {
            (NamedKind::Kwarg, "mu") => "mu",
            (NamedKind::Kwarg, "sigma") => "sigma",
            _ => return None,
        };
        parameters.push((name, arg.value));
    }
    if parameters[0].0 == parameters[1].0
        || !parameters.iter().all(|&(_, id)| {
            matches!(
                m.type_of(id),
                Some(Type::Scalar(ScalarType::Real | ScalarType::Integer))
            )
        })
    {
        return None;
    }
    Some(parameters)
}
