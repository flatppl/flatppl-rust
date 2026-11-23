//! Forward and reverse differentiation of an inferred, deterministic FlatPPL query.
//!
//! Use [`determinize_query`] to prepare the inferred model for [`gradient`].
//! The returned module is ordinary FlatPDL.

mod callbacks;
mod contract;
mod integral;
mod math;
mod record;
mod rules;
mod tensor;
mod transforms;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

use flatppl_core::graph::{build_call, expect_builtin_call};
use flatppl_core::{
    Binding, BindingId, CallHead, DEFAULT_MAX_DEPTH, Module, Node, NodeId, Ref, RefNs, Scalar,
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

/// Append one derivative result per requested input, in request order.
///
/// The module must declare `inputs` and a single real scalar `outputs` value.
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
/// This mode requires scalar captures when differentiating a numerical integral.
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
/// Outputs are real scalars or fixed-shape flat real arrays.
pub fn vjp(module: &Module, wrt: &[&str]) -> Result<Module, Error> {
    vjp_with_options(module, wrt, &LoweringOptions::default())
}

/// Apply [`vjp`] to a module that uses the requested deterministic extensions.
///
/// Numerical integrals differentiate the integrand and its moving boundaries.
/// This mode requires scalar captures when differentiating a numerical integral.
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
/// Outputs are real scalars or fixed-shape flat real arrays.
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
    let legalized = callbacks::legalize(module, &query);
    let (module, query) = match legalized.as_ref() {
        Some(legalized) => (legalized, Query::read(legalized, wrt, mode)?),
        None => (module, query),
    };
    let roots: Vec<_> = query
        .results
        .iter()
        .map(|&node| query.key(module, node))
        .collect::<Result<_, _>>()?;
    let mut plan = Plan::new(module, &query);
    for &root in &roots {
        plan.visit(root, 0)?;
    }
    let mut requests = BTreeSet::new();
    plan.collect_rules(mode, &mut requests);
    let rules = rules::Rules::load(requests)?;

    let mut emitter = Emitter::new(module, &query);
    let mut inputs = abi_terms(module, query.input).to_vec();
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
    if mode != Mode::Gradient {
        inputs.extend_from_slice(&seeds);
        let signature = build_call(&mut emitter.module, "tuple", &inputs);
        emitter.module.set_binding_rhs(query.input, signature);
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
    let result = build_call(&mut emitter.module, "tuple", &values);
    emitter.module.set_binding_rhs(query.output, result);
    if let Some(diagnostic) = flatppl_infer::infer(&mut emitter.module)
        .into_iter()
        .find(|d| d.severity == flatppl_infer::Severity::Error)
    {
        return Err(Error::new(None, diagnostic.message));
    }
    let expected = Type::Tuple(expected.into());
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

fn reverse(
    emitter: &mut Emitter,
    plan: &Plan<'_>,
    rules: &rules::Rules<'_>,
    roots: &[Key],
    seeds: &[NodeId],
    mode: Mode,
) -> Result<Derived, Error> {
    let mut contributions: HashMap<Key, Vec<(NodeId, bool)>> = HashMap::new();
    for (&root, &seed) in roots.iter().zip(seeds) {
        contributions.entry(root).or_default().push((seed, false));
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
        let scope = plan.integrals.get(&step.node);
        if scope.is_none() {
            step.check_types(plan.module)?;
        }
        if masked && step.op == "matrix product" {
            return Err(Error::new(
                Some(step.node),
                "a masked matrix product needs a mask-aware contraction pullback",
            ));
        }
        if masked && math::requires_unmasked(step.op) {
            return Err(Error::new(
                Some(step.node),
                format!("`{}` needs a mask-aware pullback", step.op),
            ));
        }
        for (slot, target) in step.targets.iter().enumerate() {
            let Some(target) = target.filter(|key| plan.activity[key]) else {
                continue;
            };
            let mut contribution = if let Some(scope) = scope {
                scope.pullback(emitter, rules, step, slot, bar)?
            } else {
                let template = step.rule(rules, mode, slot)?;
                template.instantiate(emitter, step, bar)?
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
                let pointwise = step.pointwise(plan.module) || multivariate_density(step.op);
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
    plan.query
        .wrt
        .iter()
        .map(|input| {
            let derivative = match contributions.remove(&Key::Input(input.binding)) {
                Some(terms) => {
                    let terms: Vec<_> = terms.into_iter().map(|(value, _)| value).collect();
                    tensor::add_terms(emitter, input.reference, &terms)
                }
                None => emitter.zero_like(input.reference)?,
            };
            Ok((format!("d_{}", input.name), input.reference, derivative))
        })
        .collect()
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
                let term = template.instantiate(emitter, step, tangent)?;
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
    input: BindingId,
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
        if wrt.is_empty() {
            return Err(Error::new(
                None,
                "select at least one input to differentiate",
            ));
        }
        let abi = |name| {
            module
                .bindings()
                .find(|(_, b)| module.resolve(b.name) == name)
                .ok_or_else(|| {
                    Error::new(None, format!("declare `{name}` for the derivative query"))
                })
        };
        let (input, _) = abi("inputs")?;
        let (output, result) = abi("outputs")?;
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
                .any(|&node| tensor::real_shape(module, node).is_none())
        {
            return Err(Error::new(
                Some(result.rhs),
                "`outputs` must contain real scalars or fixed-shape real arrays",
            ));
        }
        let items = abi_terms(module, input);
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
    in_integrand: bool,
}

impl<'a> Plan<'a> {
    fn new(module: &'a Module, query: &'a Query<'a>) -> Self {
        Self {
            module,
            query,
            activity: HashMap::new(),
            steps: Vec::new(),
            integrals: HashMap::new(),
            in_integrand: false,
        }
    }

    fn collect_rules(&self, mode: Mode, requests: &mut BTreeSet<rules::RuleKey<'a>>) {
        for step in &self.steps {
            if let Some(scope) = self.integrals.get(&step.node) {
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

    fn visit(&mut self, key: Key, depth: usize) -> Result<bool, Error> {
        if let Some(&active) = self.activity.get(&key) {
            return Ok(active);
        }
        if depth > DEFAULT_MAX_DEPTH {
            return Err(Error::new(
                None,
                "derivative DAG exceeds the expression depth limit",
            ));
        }
        let active = match key {
            Key::Input(binding) => self.query.wrt.iter().any(|input| input.binding == binding),
            Key::Node(node) => self.visit_node(node, depth)?,
        };
        self.activity.insert(key, active);
        Ok(active)
    }

    fn visit_node(&mut self, node: NodeId, depth: usize) -> Result<bool, Error> {
        let Node::Call(call) = self.module.node(node) else {
            return Ok(false);
        };
        let op = match call.head {
            CallHead::Builtin(symbol) => self.module.resolve(symbol),
            CallHead::User(callee) => special_op(self.module, callee).unwrap_or("user call"),
        };
        if matches!(op, flatppl_core::LOG_INTEGRAL | flatppl_core::MASS_INTEGRAL) {
            return integral::visit(self, node, op, &call.args, depth);
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
            return Ok(false);
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
        } else if op == "aggregate"
            && call.named.is_empty()
            && let Some(contraction) = contract::SumProduct::read(self.module, &call.args)
        {
            ("indexed sum product", contraction.args().to_vec(), false)
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
        let mut active = false;
        for (slot, &arg) in args.iter().enumerate() {
            if op == "ifelse" && slot == 0 {
                targets.push(None);
                continue;
            }
            let target = self.query.key(self.module, arg)?;
            active |= self.visit(target, depth + 1)?;
            targets.push(Some(target));
        }
        if active {
            self.steps.push(Step {
                node,
                op,
                args,
                targets,
                broadcast,
            });
        }
        Ok(active)
    }
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
        "Dirichlet" if !batched => ("Dirichlet log density", &["alpha"]),
        "MvNormal" if !batched => ("MvNormal log density", &["mu", "cov"]),
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
    // A singleton wrapper around a probability vector shares it across all observations.
    // The inverse pass scatters the observation cotangents into that one vector.
    if matches!(
        rule,
        "iid Categorical log density" | "iid Categorical0 log density"
    ) {
        let Key::Node(wrapped) = query.key(module, args[0]).ok()? else {
            return None;
        };
        let call = expect_builtin_call(module, wrapped, "vector")?;
        let &[probabilities] = call.args.as_ref() else {
            return None;
        };
        if !matches!(tensor::real_shape(module, probabilities), Some([_]))
            || !matches!(tensor::numeric_shape(module, observation), Some([_]))
        {
            return None;
        }
        args[0] = probabilities;
    }
    args.push(observation);
    Some((rule, args))
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
            | "Categorical log density"
            | "Categorical0 log density"
    )
}

fn structural(op: &str) -> bool {
    matches!(
        op,
        "sum"
            | "mean"
            | "logsumexp"
            | "vector"
            | "addaxes"
            | "get"
            | "get0"
            | "stack rows"
            | "stack columns"
            | "rowstack"
            | "colstack"
            | "matrix product"
            | "transpose"
            | "iid Categorical log density"
            | "iid Categorical0 log density"
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
            || self.op == "vector" && nested_vector(self.node)
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
                        self.args.len() == 2 && axes == Some(0)
                    } else {
                        axes.is_some()
                    }
                } else {
                    tensor::numeric_shape(module, arg).is_some()
                        || matches!(self.op, "rowstack" | "colstack" | "get" | "get0")
                            && slot == 0
                            && nested_vector(arg)
                        || self.op == "Uniform log density" && slot == 0
                        || self.op == "Bernoulli log density"
                            && slot == 1
                            && tensor::boolean_shape(module, arg)
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
        }
    }

    fn primal(&mut self, id: NodeId) -> NodeId {
        if self.inline || !matches!(self.module.node(id), Node::Call(_)) {
            return id;
        }
        if let Some(&name) = self.names.get(&id) {
            return self.reference(name);
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
