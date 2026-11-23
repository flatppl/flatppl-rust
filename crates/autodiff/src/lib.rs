//! Forward and reverse differentiation of an inferred, deterministic FlatPPL query.
//!
//! Use [`determinize_query`] to prepare the inferred model for [`gradient`].
//! The returned module is ordinary FlatPDL.

mod affine_scan;
mod aggregate;
mod callbacks;
mod contract;
mod fixed_scan;
mod integral;
mod math;
mod metric;
mod probability_scan;
mod record;
mod recurrence;
mod rules;
mod source;
mod tensor;
mod transforms;
mod vector_scan;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

use flatppl_core::graph::{build_call, expect_builtin_call};
use flatppl_core::{
    Binding, BindingId, CallHead, DEFAULT_MAX_DEPTH, Dim, Module, Node, NodeId, Ref, RefNs, Scalar,
    ScalarType, Symbol, Type,
};

use flatppl_determinizer::LoweringOptions;

const REAL: Type = Type::Scalar(ScalarType::Real);

/// A query or operation outside the differentiation contract.
#[derive(Debug)]
pub struct Error {
    /// A related node in the input module, when available.
    pub node: Option<NodeId>,
    message: String,
}

impl Error {
    fn new(node: Option<NodeId>, message: impl Into<String>) -> Self {
        Self {
            node,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

/// Lower an inferred model's query to FlatPDL before optimization.
///
/// Use the same dependency bundle supplied to inference. A host can load fixed
/// data into the lowered module before passing it to [`gradient`].
pub fn determinize_query(
    module: &Module,
    bundle: &flatppl_infer::ModuleBundle,
) -> Result<Module, flatppl_determinizer::RefuseError> {
    determinize_query_with_options(module, bundle, &LoweringOptions::default())
}

/// Lower an inferred query with the caller's deterministic extensions enabled.
///
/// The AD pass always receives the result before post-determinization optimization.
pub fn determinize_query_with_options(
    module: &Module,
    bundle: &flatppl_infer::ModuleBundle,
    options: &LoweringOptions<'_>,
) -> Result<Module, flatppl_determinizer::RefuseError> {
    let roots: Vec<_> = module
        .bindings()
        .filter(|(_, binding)| {
            matches!(
                module.resolve(binding.name),
                "inputs" | "outputs" | "flatppl_compat"
            )
        })
        .map(|(_, binding)| binding.name)
        .collect();
    let lowering = LoweringOptions {
        disable_optimizations: true,
        ..*options
    };
    flatppl_determinizer::determinize_with_options(module, bundle, Some(&roots), &lowering)
}

/// Prepare a derivative module for compact, reparsable FlatPPL source output.
///
/// This preserves shared expressions inside large callbacks as explicitly
/// bound helper calls. Re-determinize the printed source before execution.
pub fn prepare_source(module: &mut Module) {
    flatppl_determinizer::prepare_serialization(module, None);
    source::factor_callables(module);
}

/// Lower a source AD query with portable numerical integration enabled.
pub const NUMERICAL_SOURCE_OPTIONS: LoweringOptions<'static> = LoweringOptions {
    disable_optimizations: false,
    disable_sufficient_statistics: false,
    share_normal_scale: false,
    retain_standard_functions: &[("special-functions", "log_ndtr")],
    compact_log_cdf_difference: false,
    preserve_rand_tuple: false,
    numerical_integrals: true,
    vector_sort: false,
    restricted_sampling: false,
    supports_cdf: Some(|_| false),
};

/// Prepare a derivative module for compact, reparsable FlatPPL source output.
///
/// Pass the options used to differentiate the module. Numerical integration
/// is printed as a weighted Lebesgue mass, which requires the same numerical
/// opt-in when determinizing the printed source.
pub fn prepare_source_with_options(
    module: &mut Module,
    options: &LoweringOptions<'_>,
) -> Result<(), flatppl_determinizer::RefuseError> {
    flatppl_determinizer::prepare_serialization(module, None);
    if options.numerical_integrals {
        flatppl_determinizer::reify_numerical_integrals(module)?;
    }
    source::factor_callables(module);
    Ok(())
}

/// Append one derivative result per requested input, in request order.
///
/// The module must declare a single real scalar `outputs` value. A query with
/// no arguments may omit `inputs`. An empty selection returns just the primal.
/// Each requested name must occur in `inputs` and denote a real scalar or a
/// fixed-shape real array, including a fixed nonempty vector of real vectors,
/// or a nonempty record of such values. Each derivative has the input's shape. All
/// declared inputs are independent, including promoted fixed or derived
/// bindings. The returned module preserves the input signature and replaces
/// `outputs` with `(value, derivative, ...)`.
///
/// Derivatives are defined on the smooth, finite domain of supported operations.
/// Active unsupported operations return an error. Other inputs and inactive
/// expressions are held fixed.
pub fn gradient(module: &Module, wrt: &[&str]) -> Result<Module, Error> {
    gradient_with_options(module, wrt, &LoweringOptions::default())
}

/// Apply [`gradient`] to a module that uses the requested deterministic extensions.
///
/// Numerical integrals differentiate the integrand and its moving boundaries.
/// Reverse mode supports real scalar, fixed vector, and fixed matrix captures.
pub fn gradient_with_options(
    module: &Module,
    wrt: &[&str],
    options: &LoweringOptions<'_>,
) -> Result<Module, Error> {
    transform(module, wrt, Mode::Gradient, options)
}

/// Append one vector–Jacobian product per selected input.
///
/// Append one cotangent input per original output. Results contain the original
/// outputs followed by the input cotangents in request order. Selected inputs
/// can be real scalars, fixed-shape arrays, or nonempty records of these values.
/// Outputs can use the same real-valued scalars, arrays and records as inputs.
/// An empty selection appends the output cotangents and returns just the primals.
pub fn vjp(module: &Module, wrt: &[&str]) -> Result<Module, Error> {
    vjp_with_options(module, wrt, &LoweringOptions::default())
}

/// Apply [`vjp`] to a module that uses the requested deterministic extensions.
///
/// Numerical integrals differentiate the integrand and its moving boundaries.
/// Reverse mode supports real scalar, fixed vector, and fixed matrix captures.
pub fn vjp_with_options(
    module: &Module,
    wrt: &[&str],
    options: &LoweringOptions<'_>,
) -> Result<Module, Error> {
    transform(module, wrt, Mode::Vjp, options)
}

/// Append one Jacobian–vector product per original output.
///
/// Append one tangent input per selected input. Results contain the original
/// outputs followed by their tangents. Selected inputs can be real scalars,
/// fixed-shape arrays, or nonempty records of these values.
/// Outputs can use the same real-valued scalars, arrays and records as inputs.
/// An empty selection takes no tangents and returns one zero tangent per primal.
pub fn jvp(module: &Module, wrt: &[&str]) -> Result<Module, Error> {
    jvp_with_options(module, wrt, &LoweringOptions::default())
}

/// Apply [`jvp`] to a module that uses the requested deterministic extensions.
///
/// Numerical integrals integrate the directional derivative of the integrand
/// and add terms for its moving boundaries.
pub fn jvp_with_options(
    module: &Module,
    wrt: &[&str],
    options: &LoweringOptions<'_>,
) -> Result<Module, Error> {
    transform(module, wrt, Mode::Jvp, options)
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Mode {
    Gradient,
    Vjp,
    Jvp,
}

fn transform(
    module: &Module,
    wrt: &[&str],
    mode: Mode,
    options: &LoweringOptions<'_>,
) -> Result<Module, Error> {
    check_flatpdl(module, options)?;
    let query = Query::read(module, wrt, mode)?;
    let legalized = if wrt.is_empty() {
        None
    } else {
        callbacks::legalize(module, &query)
    };
    let (module, query) = match legalized.as_ref() {
        Some(legalized) => (legalized, Query::read(legalized, wrt, mode)?),
        None => (module, query),
    };
    let roots: Vec<_> = query
        .results
        .iter()
        .map(|&node| {
            if wrt.is_empty() {
                Ok(Key::Node(node))
            } else {
                query.key(module, node)
            }
        })
        .collect::<Result<_, _>>()?;
    let mut plan = Plan::new(module, &query);
    if !wrt.is_empty() {
        for &root in &roots {
            plan.visit(root, 0)?;
        }
    }
    let mut requests = BTreeSet::new();
    plan.collect_rules(mode, &mut requests);
    let rules = rules::Rules::load(requests)?;

    let mut emitter = Emitter::new(module, &query);
    let seeds = match mode {
        Mode::Gradient => vec![real(&mut emitter.module, 1.0)],
        Mode::Vjp => query
            .results
            .iter()
            .enumerate()
            .map(|(i, &output)| tensor::seed(&mut emitter, &format!("cotangent_{}", i + 1), output))
            .collect::<Result<Vec<_>, _>>()?,
        Mode::Jvp => query
            .wrt
            .iter()
            .map(|input| {
                tensor::seed(
                    &mut emitter,
                    &format!("tangent_{}", input.name),
                    input.reference,
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    if mode != Mode::Gradient && !seeds.is_empty() {
        let mut inputs = query
            .input
            .map(|binding| abi_terms(module, binding).to_vec())
            .unwrap_or_default();
        inputs.extend_from_slice(&seeds);
        let signature = build_abi(&mut emitter.module, &inputs);
        if let Some(binding) = query.input {
            emitter.module.set_binding_rhs(binding, signature);
        } else {
            emitter.define("inputs", signature);
        }
    }
    let derived = if mode == Mode::Jvp {
        forward(&mut emitter, &plan, &rules, &roots, &seeds)?
    } else {
        reverse(&mut emitter, &plan, &rules, &roots, &seeds, mode)?
    };
    let mut values: Vec<_> = query
        .results
        .iter()
        .map(|&node| emitter.primal(node))
        .collect();
    let mut expected: Vec<_> = query
        .results
        .iter()
        .map(|&node| module.type_of(node).unwrap().clone())
        .collect();
    for (name, reference, value) in derived {
        values.push(emitter.define(&name, value));
        expected.push(module.type_of(reference).unwrap().clone());
    }
    let result = build_abi(&mut emitter.module, &values);
    emitter.module.set_binding_rhs(query.output, result);
    if let Some(diagnostic) = flatppl_infer::infer(&mut emitter.module)
        .into_iter()
        .find(|d| d.severity == flatppl_infer::Severity::Error)
    {
        return Err(Error::new(None, diagnostic.message));
    }
    let expected = if expected.len() == 1 {
        expected.pop().unwrap()
    } else {
        Type::Tuple(expected.into())
    };
    if emitter.module.type_of(result) != Some(&expected) {
        let actual = emitter.module.type_of(result);
        return Err(Error::new(
            None,
            format!("expected derivative result type {expected:?}, got {actual:?}"),
        ));
    }
    check_flatpdl(&emitter.module, options)?;
    Ok(emitter.module)
}

type Derived = Vec<(String, NodeId, NodeId)>;
type Contributions = HashMap<Key, Vec<(NodeId, bool)>>;

fn reverse(
    emitter: &mut Emitter,
    plan: &Plan<'_>,
    rules: &rules::Rules<'_>,
    roots: &[Key],
    seeds: &[NodeId],
    mode: Mode,
) -> Result<Derived, Error> {
    let mut contributions = reverse_values(emitter, plan, rules, roots, seeds, mode)?;
    plan.query
        .wrt
        .iter()
        .map(|input| {
            let derivative = match contributions.remove(&Key::Input(input.binding)) {
                Some(terms) => combine_contributions(emitter, input.reference, terms),
                None => emitter.zero_like(input.reference)?,
            };
            Ok((format!("d_{}", input.name), input.reference, derivative))
        })
        .collect()
}

fn reverse_values(
    emitter: &mut Emitter,
    plan: &Plan<'_>,
    rules: &rules::Rules<'_>,
    roots: &[Key],
    seeds: &[NodeId],
    mode: Mode,
) -> Result<Contributions, Error> {
    reverse_values_with_mask(emitter, plan, rules, roots, seeds, mode, false)
}

fn reverse_values_with_mask(
    emitter: &mut Emitter,
    plan: &Plan<'_>,
    rules: &rules::Rules<'_>,
    roots: &[Key],
    seeds: &[NodeId],
    mode: Mode,
    seed_masked: bool,
) -> Result<Contributions, Error> {
    let mut contributions = Contributions::new();
    for (&root, &seed) in roots.iter().zip(seeds) {
        contributions
            .entry(root)
            .or_default()
            .push((seed, seed_masked));
    }
    for step in plan.steps.iter().rev() {
        let terms = contributions.remove(&Key::Node(step.node)).ok_or_else(|| {
            Error::new(
                Some(step.node),
                "missing cotangent for an active expression",
            )
        })?;
        let masked = terms.iter().any(|&(_, masked)| masked);
        let terms: Vec<_> = terms.into_iter().map(|(value, _)| value).collect();
        let bar = emitter.named_adjoint("_ad_bar", step.node, &terms);
        if step.op == fixed_scan::OP {
            let projection =
                fixed_scan::Projection::read(plan.module, &plan.query.inputs, step.node)
                    .ok_or_else(|| Error::new(Some(step.node), "missing projected scan"))?;
            let terms = projection.pullback(emitter, bar, masked)?;
            for (slot, term) in [0].into_iter().chain(2..step.targets.len()).zip(terms) {
                if let Some(target) = step.targets[slot].filter(|key| plan.activity[key]) {
                    contributions
                        .entry(target)
                        .or_default()
                        .push((term, masked));
                }
            }
            continue;
        }
        let scope = plan.integrals.get(&step.node);
        let aggregate = plan.aggregates.get(&step.node);
        if scope.is_none() && aggregate.is_none() {
            step.check_types(plan.module)?;
        }
        if masked && math::requires_unmasked(step.op) {
            return Err(Error::new(
                Some(step.node),
                format!("`{}` needs a mask-aware pullback", step.op),
            ));
        }
        let integral = scope
            .map(|scope| Ok::<_, Error>((scope, scope.integrand_adjoints(emitter, rules, mode)?)))
            .transpose()?;
        for (slot, target) in step.targets.iter().enumerate() {
            let Some(target) = target.filter(|key| plan.activity[key]) else {
                continue;
            };
            let mut contribution = if let Some((scope, adjoints)) = &integral {
                scope.pullback(emitter, step, slot, bar, adjoints)?
            } else if let Some(scope) = aggregate {
                scope.pullback(emitter, rules, slot, bar, masked)?
            } else {
                let template = step.rule(rules, mode, slot)?;
                template.instantiate(emitter, step, bar, masked)?
            };
            if math::complex_rule_name(step.op)
                && tensor::real_shape(plan.module, step.args[slot]).is_some()
            {
                contribution = build_call(&mut emitter.module, "real", &[contribution]);
            }
            if masked && math::complex_rule_name(step.op) {
                contribution = math::zero_when_masked_complex(
                    &mut emitter.module,
                    bar,
                    contribution,
                    math::scalar_complex(plan.module, step.node),
                    math::scalar_complex(plan.module, step.args[slot]),
                );
            } else if masked && math::scalar_reduction(step.op) {
                contribution =
                    tensor::zero_when_masked(&mut emitter.module, bar, contribution, true);
            } else if masked && (!structural(step.op) || multivariate_density(step.op)) {
                let pointwise =
                    step.pointwise(plan.module) || multivariate_density(step.op) || scope.is_some();
                contribution =
                    tensor::zero_when_masked(&mut emitter.module, bar, contribution, pointwise);
            }
            let contribution = if step.pointwise(plan.module) {
                tensor::sum_to_shape(emitter, contribution, step.node, step.args[slot])?
            } else {
                contribution
            };
            contributions
                .entry(target)
                .or_default()
                .push((contribution, masked || step.op == "ifelse"));
        }
    }
    Ok(contributions)
}

fn combine_contributions(
    emitter: &mut Emitter,
    primal: NodeId,
    terms: Vec<(NodeId, bool)>,
) -> NodeId {
    let terms: Vec<_> = terms.into_iter().map(|(value, _)| value).collect();
    tensor::add_terms(emitter, primal, &terms)
}

fn forward(
    emitter: &mut Emitter,
    plan: &Plan<'_>,
    rules: &rules::Rules<'_>,
    roots: &[Key],
    seeds: &[NodeId],
) -> Result<Derived, Error> {
    let tangents = plan
        .query
        .wrt
        .iter()
        .zip(seeds)
        .map(|(input, &seed)| (Key::Input(input.binding), seed))
        .collect();
    let tangents = forward_values(emitter, plan, rules, tangents)?;
    roots
        .iter()
        .zip(&plan.query.results)
        .enumerate()
        .map(|(i, (key, &node))| {
            let tangent = match tangents.get(key) {
                Some(&tangent) => tangent,
                None => emitter.zero_like(node)?,
            };
            Ok((format!("d_output_{}", i + 1), node, tangent))
        })
        .collect()
}

fn forward_values(
    emitter: &mut Emitter,
    plan: &Plan<'_>,
    rules: &rules::Rules<'_>,
    mut tangents: HashMap<Key, NodeId>,
) -> Result<HashMap<Key, NodeId>, Error> {
    for step in &plan.steps {
        if !step
            .targets
            .iter()
            .flatten()
            .any(|key| tangents.contains_key(key))
        {
            continue;
        }
        if let Some(scope) = plan.integrals.get(&step.node) {
            let tangent = scope.pushforward(emitter, rules, step, &tangents)?;
            tangents.insert(Key::Node(step.node), tangent);
            continue;
        }
        if step.op == fixed_scan::OP {
            let projection =
                fixed_scan::Projection::read(plan.module, &plan.query.inputs, step.node)
                    .ok_or_else(|| Error::new(Some(step.node), "missing projected scan"))?;
            let tangent = projection.pushforward(emitter, step, &tangents)?;
            tangents.insert(Key::Node(step.node), tangent);
            continue;
        }
        if let Some(scope) = plan.aggregates.get(&step.node) {
            let tangent = scope.pushforward(emitter, rules, step, &tangents)?;
            tangents.insert(Key::Node(step.node), tangent);
            continue;
        }
        step.check_types(plan.module)?;
        let terms: Vec<_> = step
            .targets
            .iter()
            .enumerate()
            .filter_map(|(slot, target)| {
                target
                    .and_then(|key| tangents.get(&key))
                    .map(|&tangent| (slot, tangent))
            })
            .map(|(slot, tangent)| {
                let template = step.rule(rules, Mode::Jvp, slot)?;
                let term = template.instantiate(emitter, step, tangent, false)?;
                if step.pointwise(plan.module) {
                    tensor::expand_like(emitter, term, step.node)
                } else {
                    Ok(term)
                }
            })
            .collect::<Result<_, Error>>()?;
        let tangent = emitter.named_adjoint("_ad_tangent", step.node, &terms);
        tangents.insert(Key::Node(step.node), tangent);
    }
    Ok(tangents)
}

fn check_flatpdl(module: &Module, options: &LoweringOptions<'_>) -> Result<(), Error> {
    flatppl_determinizer::is_flatpdl_with_options(module, options)
        .map_err(|issues| Error::new(Some(issues[0].node), issues[0].reason.clone()))
}

// A promoted binding is a separate runtime value. Its original RHS may also
// occur in the model as a different value, so its NodeId cannot identify it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Key {
    Input(BindingId),
    Node(NodeId),
}

struct Query<'a> {
    inputs: HashSet<BindingId>,
    input: Option<BindingId>,
    output: BindingId,
    results: Vec<NodeId>,
    wrt: Vec<Input<'a>>,
}

struct Input<'a> {
    name: &'a str,
    binding: BindingId,
    reference: NodeId,
}

impl<'a> Query<'a> {
    fn read(module: &Module, wrt: &[&'a str], mode: Mode) -> Result<Self, Error> {
        let abi = |name| {
            module
                .bindings()
                .find(|(_, b)| module.resolve(b.name) == name)
        };
        let input = abi("inputs").map(|(binding, _)| binding);
        if input.is_none() && !wrt.is_empty() {
            return Err(Error::new(
                None,
                "declare `inputs` for the derivative query",
            ));
        }
        let (output, result) = abi("outputs")
            .ok_or_else(|| Error::new(None, "declare `outputs` for the derivative query"))?;
        let results = abi_terms(module, output).to_vec();
        if mode == Mode::Gradient
            && (results.len() != 1 || module.type_of(results[0]) != Some(&REAL))
        {
            return Err(Error::new(
                Some(result.rhs),
                "`outputs` must be one real scalar",
            ));
        }
        if results.is_empty()
            || results
                .iter()
                .any(|&node| !module.type_of(node).is_some_and(record::is_differentiable))
        {
            return Err(Error::new(
                Some(result.rhs),
                "`outputs` must contain real scalars, fixed-shape real arrays, or nonempty records of those",
            ));
        }
        let items = input.map_or(&[][..], |binding| abi_terms(module, binding));
        let mut names = HashMap::new();
        let mut targets = HashSet::new();
        for &item in items {
            let Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) = module.node(item)
            else {
                return Err(Error::new(
                    Some(item),
                    "each `inputs` element must name a binding",
                ));
            };
            let target = binding_target(module, item, *name)?;
            if !targets.insert(target) {
                return Err(Error::new(Some(item), "duplicate binding in `inputs`"));
            }
            names.insert(module.resolve(*name), (target, item));
        }
        let mut selected = HashSet::new();
        let mut requested = Vec::with_capacity(wrt.len());
        for &name in wrt {
            let &(target, item) = names.get(name).ok_or_else(|| {
                Error::new(None, format!("`{name}` is not a declared query input"))
            })?;
            if !selected.insert(target) {
                return Err(Error::new(
                    Some(item),
                    format!("`{name}` is selected more than once"),
                ));
            }
            if !module.type_of(item).is_some_and(record::is_differentiable) {
                return Err(Error::new(
                    Some(item),
                    format!(
                        "`{name}` must be a real scalar, fixed-shape flat real array, or fixed nonempty vector of real vectors"
                    ),
                ));
            }
            requested.push(Input {
                name,
                binding: target,
                reference: item,
            });
        }
        Ok(Self {
            inputs: targets,
            input,
            output,
            results,
            wrt: requested,
        })
    }

    fn key(&self, module: &Module, id: NodeId) -> Result<Key, Error> {
        self.key_at(module, id, 0)
    }

    fn key_at(&self, module: &Module, id: NodeId, depth: usize) -> Result<Key, Error> {
        if depth > DEFAULT_MAX_DEPTH {
            return Err(Error::new(
                Some(id),
                "projection exceeds the expression depth limit",
            ));
        }
        let node = if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = module.node(id)
        {
            let target = binding_target(module, id, *name)?;
            if self.inputs.contains(&target) {
                return Ok(Key::Input(target));
            }
            module.binding(target).rhs
        } else {
            id
        };
        if let Some(child) = self.project_constructor(module, node, depth)? {
            self.key_at(module, child, depth + 1)
        } else {
            Ok(Key::Node(node))
        }
    }

    // A type-preserving structural projection exposes only the chosen child
    // to the derivative planner. Keep the primal graph and ABI intact.
    fn project_constructor(
        &self,
        module: &Module,
        node: NodeId,
        depth: usize,
    ) -> Result<Option<NodeId>, Error> {
        let (base, call) = if let Some(call) = expect_builtin_call(module, node, "get") {
            (1, call)
        } else if let Some(call) = expect_builtin_call(module, node, "get0") {
            (0, call)
        } else {
            return Ok(None);
        };
        let Some((&container, selectors)) = call.args.split_first() else {
            return Ok(None);
        };
        if selectors.is_empty() {
            return Ok(None);
        }
        if !matches!(
            module.type_of(container),
            Some(Type::Record(_) | Type::Tuple(_))
        ) {
            return Ok(None);
        }
        let mut child = container;
        for &selector in selectors {
            let Key::Node(container) = self.key_at(module, child, depth + 1)? else {
                return Ok(None);
            };
            let replacement = match module.node(selector) {
                Node::Lit(Scalar::Str(name)) => expect_builtin_call(module, container, "record")
                    .and_then(|record| {
                        record
                            .named
                            .iter()
                            .find(|field| module.resolve(field.name) == &**name)
                    })
                    .map(|field| field.value),
                Node::Lit(Scalar::Int(index)) => index
                    .checked_sub(base)
                    .and_then(|index| usize::try_from(index).ok())
                    .and_then(|index| {
                        expect_builtin_call(module, container, "tuple")?
                            .args
                            .get(index)
                    })
                    .copied(),
                _ => None,
            };
            let Some(replacement) = replacement else {
                return Ok(None);
            };
            child = replacement;
        }
        Ok(module
            .type_of(node)
            .filter(|&ty| Some(ty) == module.type_of(child))
            .map(|_| child))
    }
}

fn build_abi(module: &mut Module, values: &[NodeId]) -> NodeId {
    match values {
        [value] => *value,
        _ => build_call(module, "tuple", values),
    }
}

fn abi_terms(module: &Module, binding: BindingId) -> &[NodeId] {
    let rhs = &module.binding(binding).rhs;
    expect_builtin_call(module, *rhs, "tuple").map_or(std::slice::from_ref(rhs), |call| &call.args)
}

fn binding_target(module: &Module, id: NodeId, name: Symbol) -> Result<BindingId, Error> {
    let target = module.alias_target(name).ok_or_else(|| {
        Error::new(
            Some(id),
            format!("unbound reference `{}`", module.resolve(name)),
        )
    })?;
    if matches!(
        module.node(module.binding(target).rhs),
        Node::Ref(Ref {
            ns: RefNs::SelfMod,
            ..
        })
    ) {
        return Err(Error::new(Some(id), "cyclic binding reference"));
    }
    Ok(target)
}

struct Plan<'a> {
    module: &'a Module,
    query: &'a Query<'a>,
    activity: HashMap<Key, bool>,
    steps: Vec<Step<'a>>,
    integrals: HashMap<NodeId, integral::Scope<'a>>,
    aggregates: HashMap<NodeId, aggregate::Scope<'a>>,
}

impl<'a> Plan<'a> {
    fn new(module: &'a Module, query: &'a Query<'a>) -> Self {
        Self {
            module,
            query,
            activity: HashMap::new(),
            steps: Vec::new(),
            integrals: HashMap::new(),
            aggregates: HashMap::new(),
        }
    }

    fn collect_rules(&self, mode: Mode, requests: &mut BTreeSet<rules::RuleKey<'a>>) {
        for step in &self.steps {
            if step.op == fixed_scan::OP {
                continue;
            }
            if let Some(scope) = self.integrals.get(&step.node) {
                scope.plan.collect_rules(mode, requests);
                continue;
            }
            if let Some(scope) = self.aggregates.get(&step.node) {
                scope.plan.collect_rules(Mode::Jvp, requests);
                continue;
            }
            requests.extend(
                step.targets
                    .iter()
                    .enumerate()
                    .filter(|(_, key)| key.is_some_and(|key| self.activity[&key]))
                    .map(|(slot, _)| step.rule_key(mode, slot)),
            );
        }
    }

    fn visit(&mut self, root: Key, depth: usize) -> Result<bool, Error> {
        if depth > DEFAULT_MAX_DEPTH {
            return Err(Error::new(
                None,
                "derivative scopes exceed the expression depth limit",
            ));
        }
        // Named states can form long dependency chains even though every
        // authored expression and callable body has a bounded depth.
        let mut pending = HashSet::new();
        let mut work = vec![Dependency::Enter(root)];
        while let Some(item) = work.pop() {
            match item {
                Dependency::Enter(key) => {
                    if self.activity.contains_key(&key) {
                        continue;
                    }
                    let prepared = match key {
                        Key::Input(binding) => Prepared::Known(
                            self.query.wrt.iter().any(|input| input.binding == binding),
                        ),
                        Key::Node(node) => {
                            if pending.contains(&key) {
                                return Err(Error::new(Some(node), "cyclic derivative dependency"));
                            }
                            self.prepare_node(node, depth)?
                        }
                    };
                    match prepared {
                        Prepared::Known(active) => {
                            self.activity.insert(key, active);
                        }
                        Prepared::Step(step) => {
                            pending.insert(key);
                            let continuation = work.len();
                            work.extend(
                                step.targets
                                    .iter()
                                    .rev()
                                    .flatten()
                                    .map(|&target| Dependency::Enter(target)),
                            );
                            // The postorder entry must run after all incoming operands.
                            work.insert(continuation, Dependency::Finish(step));
                        }
                    }
                }
                Dependency::Finish(step) => {
                    if step.op == fixed_scan::OP
                        && step.targets[1].is_some_and(|grid| self.activity[&grid])
                    {
                        return Err(Error::new(
                            Some(step.node),
                            "a fixed scan grid cannot be an active input",
                        ));
                    }
                    let key = Key::Node(step.node);
                    let active = step
                        .targets
                        .iter()
                        .flatten()
                        .any(|target| self.activity[target]);
                    if active {
                        self.steps.push(step);
                    }
                    pending.remove(&key);
                    self.activity.insert(key, active);
                }
            }
        }
        Ok(self.activity[&root])
    }

    fn prepare_node(&mut self, node: NodeId, depth: usize) -> Result<Prepared<'a>, Error> {
        let Node::Call(call) = self.module.node(node) else {
            return Ok(Prepared::Known(false));
        };
        let op = match call.head {
            CallHead::Builtin(symbol) => self.module.resolve(symbol),
            CallHead::User(callee) => special_op(self.module, callee).unwrap_or("user call"),
        };
        if matches!(op, flatppl_core::LOG_INTEGRAL | flatppl_core::MASS_INTEGRAL) {
            return integral::visit(self, node, op, &call.args, depth).map(Prepared::Known);
        }
        if op == "aggregate"
            && call.named.is_empty()
            && contract::SumProduct::read(self.module, &call.args).is_none()
            && let Some(active) = aggregate::visit(self, node, &call.args, depth)?
        {
            return Ok(Prepared::Known(active));
        }
        if matches!(op, "elementof" | "external" | "load_data") {
            return Err(Error::new(
                Some(node),
                "a runtime value must be declared in `inputs`",
            ));
        }
        // The content of a fixed-shape input has no influence on its shape.
        if matches!(op, "lengthof" | "sizeof" | "indicesof" | "indicesof0")
            && let [input] = call.args.as_ref()
            && (tensor::numeric_shape(self.module, *input).is_some()
                || tensor::nested_vector(self.module, *input).is_some())
        {
            return Ok(Prepared::Known(false));
        }
        let positional_broadcast = if op == "broadcast" && call.named.is_empty() {
            call.args.split_first().and_then(|(&head, args)| {
                let op = match self.module.node(head) {
                    Node::Const(head) => self.module.resolve(*head),
                    _ => special_op(self.module, head)?,
                };
                Some((op, args))
            })
        } else {
            None
        };
        let (op, args, broadcast) = if let Some((rule, args)) =
            density_args(self.module, self.query, node)
                .or_else(|| transforms::args(self.module, self.query, call, op))
        {
            (rule, args, op == "broadcast")
        } else if op == "get"
            && let Some(projection) =
                fixed_scan::Projection::read(self.module, &self.query.inputs, node)
        {
            (fixed_scan::OP, projection.args(), false)
        } else if op == "aggregate"
            && call.named.is_empty()
            && let Some(matrix) = aggregate::row_maximum(self.module, &call.args)
        {
            ("row maximum", vec![matrix], false)
        } else if op == "aggregate"
            && call.named.is_empty()
            && let Some(contraction) = contract::SumProduct::read(self.module, &call.args)
        {
            ("indexed sum product", contraction.args().to_vec(), false)
        } else if op == "get"
            && let Some(args) = vector_scan::operands(self.module, &self.query.inputs, node)
        {
            (vector_scan::OP, args.to_vec(), false)
        } else if op == "scan"
            && let Some(args) = probability_scan::args(self.module, &self.query.inputs, &call.args)
        {
            ("probability scan", args.to_vec(), false)
        } else if op == "scan"
            && let Some(args) = affine_scan::args(self.module, &self.query.inputs, &call.args)
        {
            ("affine scan", args.to_vec(), false)
        } else if let Some(matrix) = callbacks::packed_rows(self.module, node) {
            ("pack matrix rows", vec![matrix], false)
        } else if let Some(args) = callbacks::packed_vectors(self.module, node) {
            ("pack vector cells", args.to_vec(), false)
        } else if let Some((head, args)) =
            positional_broadcast.filter(|(head, _)| !structural(head))
        {
            (head, args.to_vec(), true)
        } else if matches!(call.head, CallHead::User(_)) && op != "user call" {
            (op, call.args.to_vec(), false)
        } else if matches!(op, "rowstack" | "colstack")
            && let [container] = call.args.as_ref()
            && let Key::Node(container) = self.query.key(self.module, *container)?
            && let Some(vector) = expect_builtin_call(self.module, container, "vector")
            && vector.named.is_empty()
        {
            let rule = if op == "rowstack" {
                "stack rows"
            } else {
                "stack columns"
            };
            (rule, vector.args.to_vec(), false)
        } else if op == "mul" && tensor::matmul(self.module, &call.args) {
            ("matrix product", call.args.to_vec(), false)
        } else {
            (op, self.module.node(node).children(), false)
        };
        let op = math::complex_rule(self.module, node, &args, op).unwrap_or(op);
        let mut targets = Vec::with_capacity(args.len());
        for (slot, &arg) in args.iter().enumerate() {
            if op == "ifelse" && slot == 0 {
                targets.push(None);
                continue;
            }
            let target = self.query.key(self.module, arg)?;
            targets.push(Some(target));
        }
        Ok(Prepared::Step(Step {
            node,
            op,
            args,
            targets,
            broadcast,
        }))
    }
}

enum Dependency<'a> {
    Enter(Key),
    Finish(Step<'a>),
}

enum Prepared<'a> {
    Known(bool),
    Step(Step<'a>),
}

fn special_op(module: &Module, callee: NodeId) -> Option<&'static str> {
    let (module, member) = flatppl_determinizer::standard_function(module, callee)?;
    if module != "special-functions" {
        return None;
    }
    match member.as_str() {
        "erf" => Some("erf"),
        "erfc" => Some("erfc"),
        "erfcx" => Some("erfcx"),
        "log_ndtr" => Some("log_ndtr"),
        "gammainc" => Some("gammainc"),
        "gammaincc" => Some("gammaincc"),
        "log_gammainc" => Some("log_gammainc"),
        "log_gammaincc" => Some("log_gammaincc"),
        "betainc" => Some("betainc"),
        "betaincc" => Some("betaincc"),
        "log_betainc" => Some("log_betainc"),
        "log_betaincc" => Some("log_betaincc"),
        "beta" => Some("beta"),
        "logbeta" => Some("logbeta"),
        "digamma" => Some("digamma"),
        "polygamma" => Some("polygamma"),
        _ => None,
    }
}

fn density_args(
    module: &Module,
    query: &Query<'_>,
    node: NodeId,
) -> Option<(&'static str, Vec<NodeId>)> {
    let named =
        |id, name| matches!(module.node(id), Node::Const(sym) if module.resolve(*sym) == name);
    let mapped = callbacks::density_operands(module, node);
    let (tag, params, observation, batched) = if let Some(operands) = mapped.as_ref() {
        (operands.tag, None, operands.observation, true)
    } else if let Some(call) = expect_builtin_call(module, node, "builtin_logdensityof") {
        let &[tag, params, observation] = call.args.as_ref() else {
            return None;
        };
        (tag, Some(params), observation, false)
    } else {
        let call = expect_builtin_call(module, node, "broadcast")?;
        let &[head, tag, params, observation] = call.args.as_ref() else {
            return None;
        };
        if !call.named.is_empty() || !named(head, "builtin_logdensityof") {
            return None;
        }
        (tag, Some(params), observation, true)
    };
    let Node::Const(tag) = module.node(tag) else {
        return None;
    };
    let (rule, fields): (&str, &[&str]) = match module.resolve(*tag) {
        "Normal" => ("Normal log density", &["mu", "sigma"]),
        "NormalGLM" if !batched => ("Normal GLM log density", &["x", "alpha", "beta", "sigma"]),
        "CategoricalLogitGLM" if !batched => {
            ("Categorical GLM log density", &["x", "alpha", "beta"])
        }
        "OrderedLogitGLM" if !batched => ("Ordered GLM log density", &["x", "beta", "cutpoints"]),
        "BernoulliLogitGLM" if !batched => ("Bernoulli GLM log density", &["x", "alpha", "beta"]),
        "BinomialLogitGLM" if !batched => {
            ("Binomial GLM log density", &["x", "alpha", "beta", "n"])
        }
        "PoissonLogGLM" if !batched => ("Poisson GLM log density", &["x", "alpha", "beta"]),
        "NegativeBinomial2LogGLM" if !batched => (
            "NegativeBinomial2 GLM log density",
            &["x", "alpha", "beta", "psi"],
        ),
        "StudentTGLM" if !batched => (
            "StudentT GLM log density",
            &["x", "alpha", "beta", "sigma", "nu"],
        ),
        "GammaLogGLM" if !batched => ("Gamma GLM log density", &["x", "alpha", "beta", "shape"]),
        "BetaLogitGLM" if !batched => {
            ("Beta GLM log density", &["x", "alpha", "beta", "precision"])
        }
        "GeneralizedNormal" => ("GeneralizedNormal log density", &["mean", "alpha", "beta"]),
        "VonMises" => ("VonMises log density", &["mu", "kappa"]),
        "ChiSquared" => ("ChiSquared log density", &["k"]),
        "Pareto" => ("Pareto log density", &["shape", "scale"]),
        "Rayleigh" => ("Rayleigh log density", &["scale"]),
        "Weibull" => ("Weibull log density", &["shape", "scale"]),
        "Geometric" => ("Geometric log density", &["p"]),
        "ContinuedPoisson" | "hep-physics::ContinuedPoisson" => {
            ("ContinuedPoisson log density", &["rate"])
        }
        "Categorical" if batched => ("iid Categorical log density", &["p"]),
        "Categorical0" if batched => ("iid Categorical0 log density", &["p"]),
        "Categorical" => ("Categorical log density", &["p"]),
        "Categorical0" => ("Categorical0 log density", &["p"]),
        "Cauchy" => ("Cauchy log density", &["location", "scale"]),
        "LogNormal" => ("LogNormal log density", &["mu", "sigma"]),
        "Exponential" => ("Exponential log density", &["rate"]),
        "Beta" => ("Beta log density", &["alpha", "beta"]),
        "StudentT" => ("StudentT log density", &["nu"]),
        "Logistic" => ("Logistic log density", &["mu", "s"]),
        "Laplace" => ("Laplace log density", &["location", "scale"]),
        "Gamma" => ("Gamma log density", &["shape", "rate"]),
        "InverseGamma" => ("InverseGamma log density", &["shape", "scale"]),
        "Bernoulli" => ("Bernoulli log density", &["p"]),
        "Binomial" => ("Binomial log density", &["n", "p"]),
        "NegativeBinomial" => ("NegativeBinomial log density", &["alpha", "beta"]),
        "NegativeBinomial2" => ("NegativeBinomial2 log density", &["mu", "psi"]),
        "Poisson" => ("Poisson log density", &["rate"]),
        "Dirichlet" if batched => ("iid Dirichlet log density", &["alpha"]),
        "Dirichlet" => ("Dirichlet log density", &["alpha"]),
        "MvNormal" if !batched => ("MvNormal log density", &["mu", "cov"]),
        "Wishart" if !batched => ("Wishart log density", &["nu", "scale"]),
        "LKJ" if !batched => ("LKJ log density", &["n", "eta"]),
        "LKJCholesky" if !batched => ("LKJCholesky log density", &["n", "eta"]),
        "InverseWishart" if !batched => ("InverseWishart log density", &["nu", "scale"]),
        "Multinomial" if !batched => ("Multinomial log density", &["n", "p"]),
        "Uniform" => ("Uniform log density", &["support"]),
        _ => return None,
    };
    if let Some(operands) = mapped {
        let mut args = fields
            .iter()
            .map(|&name| {
                operands
                    .fields
                    .iter()
                    .find_map(|&(label, value)| (module.resolve(label) == name).then_some(value))
            })
            .collect::<Option<Vec<_>>>()?;
        if let Some(stable) = bernoulli_logit_args(module, query, rule, &args, observation) {
            return Some(stable);
        }
        if rule == "iid Dirichlet log density" {
            return iid_dirichlet_args(module, query, node, args[0], observation);
        }
        args.push(observation);
        return Some((rule, args));
    }
    let Key::Node(params) = query.key(module, params?).ok()? else {
        return None;
    };
    let record = if batched {
        let call = expect_builtin_call(module, params, "broadcast")?;
        let [head] = call.args.as_ref() else {
            return None;
        };
        if !named(*head, "record") {
            return None;
        }
        call
    } else {
        expect_builtin_call(module, params, "record")?
    };
    let field = |name| {
        record
            .named
            .iter()
            .find(|arg| module.resolve(arg.name) == name)
            .map(|arg| arg.value)
    };
    let mut args = fields
        .iter()
        .map(|&name| field(name))
        .collect::<Option<Vec<_>>>()?;
    if let Some(stable) = bernoulli_logit_args(module, query, rule, &args, observation) {
        return Some(stable);
    }
    if rule == "iid Dirichlet log density" {
        return iid_dirichlet_args(module, query, node, args[0], observation);
    }
    if rule == "Uniform log density"
        && let Some([lo, hi]) = uniform_interval(module, query, args[0], batched)
    {
        return Some(("Uniform interval log density", vec![lo, hi, observation]));
    }
    // A singleton wrapper around a probability vector shares it across all observations.
    // The inverse pass scatters the observation cotangents into that one vector.
    if matches!(
        rule,
        "iid Categorical log density" | "iid Categorical0 log density"
    ) {
        if !matches!(tensor::numeric_shape(module, observation), Some([_])) {
            return None;
        }
        if let Ok(Key::Node(wrapped)) = query.key(module, args[0])
            && let Some(call) = expect_builtin_call(module, wrapped, "vector")
            && let &[probabilities] = call.args.as_ref()
            && matches!(tensor::real_shape(module, probabilities), Some([_]))
        {
            args[0] = probabilities;
        } else if matches!(tensor::real_shape(module, args[0]), Some([_, _]))
            || tensor::nested_vector(module, args[0]).is_some()
        {
            let rule = if rule == "iid Categorical log density" {
                "rowwise Categorical log density"
            } else {
                "rowwise Categorical0 log density"
            };
            args.push(observation);
            return Some((rule, args));
        } else {
            return None;
        }
    }
    args.push(observation);
    Some((rule, args))
}

fn bernoulli_logit_args(
    module: &Module,
    query: &Query<'_>,
    rule: &str,
    args: &[NodeId],
    observation: NodeId,
) -> Option<(&'static str, Vec<NodeId>)> {
    let ("Bernoulli log density", &[probability]) = (rule, args) else {
        return None;
    };
    // Preserve a declared probability input as an independent variable, even
    // if its source binding happens to be authored as invlogit(logit).
    let Key::Node(probability) = query.key(module, probability).ok()? else {
        return None;
    };
    let Node::Call(call) = module.node(probability) else {
        return None;
    };
    if !call.named.is_empty() || call.inputs.is_some() {
        return None;
    }
    let CallHead::Builtin(op) = call.head else {
        return None;
    };
    let logit = match (module.resolve(op), call.args.as_ref()) {
        ("invlogit", [logit]) => *logit,
        ("broadcast", [head, logit]) if matches!(module.node(*head), Node::Const(op) if module.resolve(*op) == "invlogit") => {
            *logit
        }
        _ => return None,
    };
    Some(("Bernoulli logit log density", vec![logit, observation]))
}

fn iid_dirichlet_args(
    module: &Module,
    query: &Query<'_>,
    score: NodeId,
    boxed_alpha: NodeId,
    observation: NodeId,
) -> Option<(&'static str, Vec<NodeId>)> {
    let Key::Node(wrapper) = query.key(module, boxed_alpha).ok()? else {
        return None;
    };
    let wrapper = expect_builtin_call(module, wrapper, "vector")?;
    let &[alpha] = wrapper.args.as_ref() else {
        return None;
    };
    if !wrapper.named.is_empty() {
        return None;
    }
    let [Dim::Static(cols)] = *tensor::real_shape(module, alpha)? else {
        return None;
    };
    let (rows, event_cols) = match tensor::real_shape(module, observation) {
        Some([Dim::Static(rows), Dim::Static(event_cols)]) => (*rows, *event_cols),
        _ => tensor::nested_vector(module, observation)?,
    };
    (cols == event_cols && tensor::real_shape(module, score) == Some(&[Dim::Static(rows)]))
        .then_some(("iid Dirichlet log density", vec![alpha, observation]))
}

fn uniform_interval(
    module: &Module,
    query: &Query<'_>,
    set: NodeId,
    batched: bool,
) -> Option<[NodeId; 2]> {
    let Key::Node(set) = query.key(module, set).ok()? else {
        return None;
    };
    let set = if batched {
        let &[set] = expect_builtin_call(module, set, "vector")?.args.as_ref() else {
            return None;
        };
        let Key::Node(set) = query.key(module, set).ok()? else {
            return None;
        };
        set
    } else {
        set
    };
    expect_builtin_call(module, set, "interval")?
        .args
        .as_ref()
        .try_into()
        .ok()
}

struct Step<'a> {
    node: NodeId,
    op: &'a str,
    args: Vec<NodeId>,
    targets: Vec<Option<Key>>,
    broadcast: bool,
}

fn multivariate_density(op: &str) -> bool {
    matches!(
        op,
        "Dirichlet log density"
            | "MvNormal log density"
            | "Wishart log density"
            | "LKJ log density"
            | "LKJCholesky log density"
            | "InverseWishart log density"
            | "Multinomial log density"
            | "Categorical log density"
            | "Categorical0 log density"
    ) || op.ends_with(" GLM log density")
}

fn structural(op: &str) -> bool {
    matches!(
        op,
        "sum"
            | "cumsum"
            | "cummax"
            | "cummin"
            | "reverse"
            | "mean"
            | "logsumexp"
            | "vector"
            | "array"
            | "fill"
            | "cat"
            | "record"
            | "addaxes"
            | "get"
            | "get0"
            | "stack rows"
            | "stack columns"
            | "pack vector cells"
            | "pack matrix rows"
            | "rowstack"
            | "colstack"
            | "matrix product"
            | "scoped aggregate"
            | "affine scan"
            | "probability scan"
            | "row maximum"
            | vector_scan::OP
            | "transpose"
            | "iid Dirichlet log density"
            | "iid Categorical log density"
            | "iid Categorical0 log density"
            | "rowwise Categorical log density"
            | "rowwise Categorical0 log density"
    ) || math::structural(op)
        || multivariate_density(op)
}

impl<'a> Step<'a> {
    fn rule_key(&self, mode: Mode, slot: usize) -> rules::RuleKey<'a> {
        let catalog =
            if mode == Mode::Jvp && (structural(self.op) || math::complex_pushforward(self.op)) {
                "Pushforward"
            } else {
                "Pullback"
            };
        (catalog, self.op, self.args.len(), slot)
    }

    fn rule<'r>(
        &self,
        rules: &'r rules::Rules<'a>,
        mode: Mode,
        slot: usize,
    ) -> Result<&'r rules::Template, Error> {
        rules.get(self.rule_key(mode, slot)).ok_or_else(|| {
            if self.op == "Poisson log density" && slot == 1 {
                return Error::new(
                    Some(self.node),
                    "Poisson log density has no real derivative with respect to its discrete observation",
                );
            }
            let name = if mode == Mode::Jvp {
                "JVP rule"
            } else {
                "pullback"
            };
            Error::new(
                Some(self.node),
                format!("no {name} for `{}` operand {}", self.op, slot + 1),
            )
        })
    }

    fn pointwise(&self, module: &Module) -> bool {
        !structural(self.op)
            && tensor::real_shape(module, self.node).is_some_and(|shape| !shape.is_empty())
    }

    fn check_types(&self, module: &Module) -> Result<(), Error> {
        if self.op == "record"
            && !self.broadcast
            && module
                .type_of(self.node)
                .is_some_and(record::is_differentiable)
        {
            return Ok(());
        }
        let nested_vector = |node| tensor::nested_vector(module, node).is_some();
        let record_access =
            matches!(self.op, "get" | "get0") && record::is_access(module, &self.args);
        if math::complex_rule_name(self.op) {
            if math::scalar_arithmetic(module, self.node)
                && self
                    .args
                    .iter()
                    .all(|&arg| math::scalar_arithmetic(module, arg))
            {
                return Ok(());
            }
            return Err(Error::new(
                Some(self.node),
                "complex pullbacks currently require scalar intermediates",
            ));
        }
        let have_shapes = (tensor::real_shape(module, self.node).is_some()
            || matches!(
                self.op,
                "vector" | "cat" | "get" | "get0" | "pack vector cells" | "pack matrix rows"
            ) && nested_vector(self.node)
            || record_access
                && module
                    .type_of(self.node)
                    .is_some_and(record::is_differentiable))
            && self.args.iter().enumerate().all(|(slot, &arg)| {
                if record_access {
                    return true;
                }
                if self.op == "ifelse" && slot == 0 {
                    tensor::boolean_shape(module, arg)
                } else if matches!(self.op, "get" | "get0") && slot > 0 {
                    let axes = tensor::index_axes(module, arg);
                    if nested_vector(self.args[0]) {
                        self.args.len() == 2 && axes.is_some_and(|rank| rank <= 1)
                    } else {
                        axes.is_some()
                    }
                } else {
                    tensor::numeric_shape(module, arg).is_some()
                        || matches!(
                            self.op,
                            "rowstack"
                                | "colstack"
                                | "get"
                                | "get0"
                                | "rowwise Categorical log density"
                                | "rowwise Categorical0 log density"
                        ) && slot == 0
                            && nested_vector(arg)
                        || self.op == "iid Dirichlet log density" && slot == 1 && nested_vector(arg)
                        || self.op == "cat" && nested_vector(arg)
                        || self.op == "Uniform log density" && slot == 0
                        || matches!(
                            (self.op, slot),
                            ("Bernoulli log density" | "Bernoulli logit log density", 1)
                                | ("Bernoulli GLM log density", 3)
                        ) && tensor::boolean_shape(module, arg)
                }
            });
        if !have_shapes {
            return Err(Error::new(
                Some(self.node),
                format!(
                    "`{}` pullback requires real scalars or fixed-shape arrays",
                    self.op
                ),
            ));
        }
        if matches!(self.op, "stack rows" | "stack columns")
            && (!matches!(tensor::real_shape(module, self.node), Some([_, _]))
                || self
                    .args
                    .iter()
                    .any(|&arg| !matches!(tensor::numeric_shape(module, arg), Some([_]))))
        {
            return Err(Error::new(
                Some(self.node),
                "stack pullback requires equal-length vectors",
            ));
        }
        if matches!(self.op, "cumsum" | "cummax" | "cummin" | "reverse")
            && !matches!(tensor::real_shape(module, self.node), Some([_]))
        {
            return Err(Error::new(
                Some(self.node),
                "cumulative and reverse pullbacks require a fixed-length real vector",
            ));
        }
        if self.op == "prod" && !matches!(tensor::real_shape(module, self.args[0]), Some([_])) {
            return Err(Error::new(
                Some(self.node),
                "product pullback requires a real vector",
            ));
        }
        if matches!(self.op, "var" | "std") && math::sample_degrees(module, self.args[0]).is_none()
        {
            return Err(Error::new(
                Some(self.node),
                "sample moment pullback requires at least two real cells",
            ));
        }
        if self.op == "diag" && math::square_size(module, self.args[0]).is_none() {
            return Err(Error::new(
                Some(self.node),
                "diagonal pullback requires a nonempty real square matrix",
            ));
        }
        if self.op == "det" && !math::cofactor_shape(module, self.args[0]) {
            return Err(Error::new(
                Some(self.node),
                format!(
                    "determinant pullback requires a nonempty real square matrix of size at most {}",
                    math::MAX_COFACTOR_SIZE
                ),
            ));
        }
        if self.op == "transpose" && !matches!(tensor::real_shape(module, self.node), Some([_, _]))
        {
            return Err(Error::new(
                Some(self.node),
                "transpose pullback requires a matrix",
            ));
        }
        if self.op == "mul"
            && !self.broadcast
            && self
                .args
                .iter()
                .all(|&arg| tensor::numeric_shape(module, arg).is_some_and(|s| !s.is_empty()))
        {
            return Err(Error::new(
                Some(self.node),
                "matrix products need a contraction pullback",
            ));
        }
        Ok(())
    }
}

struct Emitter {
    module: Module,
    names: HashMap<NodeId, Symbol>,
    suffix: usize,
    inline: bool,
    vector_scans: HashMap<NodeId, vector_scan::Prepared>,
    vector_scan_bars: HashMap<NodeId, NodeId>,
}

impl Emitter {
    fn new(module: &Module, query: &Query<'_>) -> Self {
        let mut names = HashMap::new();
        for (id, binding) in module.bindings() {
            if id != query.output && !query.inputs.contains(&id) {
                names.entry(binding.rhs).or_insert(binding.name);
            }
        }
        Self {
            module: module.clone(),
            names,
            suffix: 0,
            inline: false,
            vector_scans: HashMap::new(),
            vector_scan_bars: HashMap::new(),
        }
    }

    fn primal(&mut self, id: NodeId) -> NodeId {
        if !matches!(self.module.node(id), Node::Call(_)) {
            return id;
        }
        if let Some(&name) = self.names.get(&id) {
            return self.reference(name);
        }
        if self.inline {
            return id;
        }
        let id_ref = self.define("_ad_value", id);
        let Node::Ref(reference) = self.module.node(id_ref) else {
            unreachable!()
        };
        self.names.insert(id, reference.name);
        id_ref
    }

    fn named_adjoint(&mut self, name: &str, primal: NodeId, terms: &[NodeId]) -> NodeId {
        let sum = tensor::add_terms(self, primal, terms);
        if !self.inline && matches!(self.module.node(sum), Node::Call(_)) {
            self.define(name, sum)
        } else {
            sum
        }
    }

    fn zero_like(&mut self, primal: NodeId) -> Result<NodeId, Error> {
        let zero = real(&mut self.module, 0.0);
        tensor::fill_like(self, zero, primal)
            .ok_or_else(|| Error::new(Some(primal), "cannot construct a shape-matched zero"))
    }

    fn define(&mut self, base: &str, rhs: NodeId) -> NodeId {
        let mut name = self.module.intern(base);
        while self.module.binding_by_name(name).is_some() {
            self.suffix += 1;
            name = self.module.intern(&format!("{base}_{}", self.suffix));
        }
        self.module.add_binding(Binding {
            name,
            rhs,
            doc: None,
            public: !base.starts_with('_'),
            synthetic: false,
        });
        self.reference(name)
    }

    fn reference(&mut self, name: Symbol) -> NodeId {
        self.module.alloc(Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }))
    }
}

fn real(module: &mut Module, value: f64) -> NodeId {
    module.alloc(Node::Lit(Scalar::Real(value)))
}
