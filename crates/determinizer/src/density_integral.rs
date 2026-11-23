//! Positive integrals in the compiler's internal IR.
//! Exact rules run first. Ordinary FlatPDL conformance rejects this extension.

use super::*;

pub(crate) fn stable_interval(m: &Module, measure: NodeId) -> Option<[NodeId; 3]> {
    let (measure, _) = resolve_ref(m, measure);
    let truncate = expect_builtin_call(m, measure, "truncate")?;
    let [base, set] = *truncate.args else {
        return None;
    };
    let (base, _) = resolve_ref(m, base);
    if !matches!(
        builtin_name(m, base),
        Some("Normal" | "Exponential" | "Weibull")
    ) {
        return None;
    }
    let (set, _) = resolve_ref(m, set);
    let [lo, hi] = *expect_builtin_call(m, set, "interval")?.args else {
        return None;
    };
    Some([base, lo, hi])
}

pub(crate) fn log_stable_interval_query(m: &Module, node: NodeId) -> Option<[NodeId; 3]> {
    let [mass] = *expect_builtin_call(m, node, "log")?.args else {
        return None;
    };
    let [measure] = *expect_builtin_call(m, mass, "totalmass")?.args else {
        return None;
    };
    stable_interval(m, measure)
}

fn point(m: &mut Module) -> NodeId {
    let token = m.alloc(Node::Lit(Scalar::Int(m.node_count() as i64)));
    let point = build_call(m, flatppl_core::INTEGRATION_POINT, &[token]);
    m.set_type(point, Type::Scalar(ScalarType::Real));
    m.set_phase(point, flatppl_core::Phase::Fixed);
    point
}

/// Sum or integrate one named scalar latent shared by a conditional measure.
pub(super) fn marginal(
    m: &mut Module,
    measure: NodeId,
    value: NodeId,
) -> Option<Result<NodeId, RefuseError>> {
    let ancestors = measure_stochastic_ancestors(m, measure, &[]);
    marginal_with_ancestors(m, measure, value, &ancestors)
}

/// Score all fields conditionally before integrating their shared ancestor.
pub(super) fn record_marginal(
    m: &mut Module,
    record: NodeId,
    value: NodeId,
) -> Option<Result<NodeId, RefuseError>> {
    let fields = expect_builtin_call(m, record, "record")?
        .named
        .iter()
        .map(|field| resolve_component_draw(m, field.value))
        .collect::<Option<Vec<_>>>()?;
    let siblings = fields.iter().map(|field| field.3).collect::<Vec<_>>();
    let ancestors = fields
        .iter()
        .flat_map(|&(measure, _, transform, _)| {
            std::iter::once(measure)
                .chain(transform)
                .flat_map(|node| measure_stochastic_ancestors(m, node, &siblings))
        })
        .collect::<Vec<_>>();
    marginal_with_ancestors(m, record, value, &ancestors)
}

fn marginal_with_ancestors(
    m: &mut Module,
    measure: NodeId,
    value: NodeId,
    ancestors: &[StochasticAncestor],
) -> Option<Result<NodeId, RefuseError>> {
    let (name, prior) = sole_named_ancestor(ancestors)?;
    expectation(m, prior, &mut |m, coordinate| {
        let bound = [(
            Ref {
                ns: RefNs::SelfMod,
                name,
            },
            coordinate,
        )];
        refuse_boundary_named_query_point(m, value, &bound)?;
        m.push_conditioned_inputs(&[name]);
        let conditional = lower_measure_density(m, measure, value);
        m.pop_conditioned_inputs(1);
        Ok(crate::kernel::substitute_admitted(
            m,
            conditional?,
            &bound,
            crate::kernel::Substitute::All,
            true,
        ))
    })
}

/// The explicit single-step form uses the same scalar expectation as lawof.
pub(crate) fn kernel_marginal(
    m: &mut Module,
    prior: NodeId,
    kernel_node: NodeId,
    kernel: &crate::kernel::Kernel,
    value: NodeId,
) -> Option<Result<NodeId, RefuseError>> {
    expectation(m, prior, &mut |m, coordinate| {
        kernel_conditional(m, kernel_node, kernel, coordinate, value)
    })
}

/// Apply one latent value without changing query points or captured draws.
pub(crate) fn kernel_conditional(
    m: &mut Module,
    kernel_node: NodeId,
    kernel: &crate::kernel::Kernel,
    coordinate: NodeId,
    value: NodeId,
) -> Result<NodeId, RefuseError> {
    let absorbed = crate::kernel::resolve_kernel(m, kernel_node).is_some();
    let bound = [(kernel.inputs[0].1, coordinate)];
    refuse_boundary_named_query_point(m, value, &bound)?;
    let mut body = crate::kernel::substitute_admitted(
        m,
        kernel.body,
        &bound,
        crate::kernel::Substitute::LocalOnly,
        absorbed,
    );
    // kernelof includes lawof. functionof may capture an ambient draw instead.
    if absorbed {
        body = build_call(m, "lawof", &[body]);
    }
    let names: Vec<_> = bound
        .iter()
        .filter(|(r, _)| r.ns == RefNs::SelfMod)
        .map(|(r, _)| r.name)
        .collect();
    m.push_conditioned_inputs(&names);
    let conditional = lower_measure_density(m, body, value);
    m.pop_conditioned_inputs(names.len());
    Ok(crate::kernel::substitute_admitted(
        m,
        conditional?,
        &bound,
        crate::kernel::Substitute::All,
        absorbed,
    ))
}

fn expectation(
    m: &mut Module,
    prior: NodeId,
    conditional: &mut dyn FnMut(&mut Module, NodeId) -> Result<NodeId, RefuseError>,
) -> Option<Result<NodeId, RefuseError>> {
    if measure_reaches_draw(m, prior, &[]) {
        return None;
    }
    let (resolved, _) = resolve_ref_chain(m, prior);
    if let Some(call) = expect_builtin_call(m, resolved, "superpose") {
        let components = call.args.to_vec();
        let mut terms = Vec::with_capacity(components.len());
        for component in components {
            match expectation(m, component, conditional)? {
                Ok(term) => terms.push(term),
                Err(error) => return Some(Err(error)),
            }
        }
        let terms = build_call(m, "vector", &terms);
        return Some(Ok(build_call(m, "logsumexp", &[terms])));
    }
    if let Some(call) = expect_builtin_call(m, resolved, "normalize") {
        let base = *call.args.first()?;
        let normalizer = mass(m, base)?;
        return Some(normalizer.and_then(|log_z| {
            let score = expectation(m, base, conditional)
                .ok_or_else(|| refuse(prior, m, "unsupported normalized scalar prior"))??;
            Ok(normalized_logdensity(m, score, log_z))
        }));
    }
    if matches!(builtin_name(m, resolved), Some("weighted" | "logweighted")) {
        let logarithmic = builtin_name(m, resolved) == Some("logweighted");
        let Node::Call(call) = m.node(resolved) else {
            return None;
        };
        if call.args.len() < 2 {
            return None;
        }
        let (weight, base) = (call.args[0], call.args[1]);
        let parameter = crate::extras::bind_extras(m, resolved, weight, 2, 1).ok()?;
        if parameter.is_empty() && !weight_is_variate_dependent(m, weight) {
            let score = expectation(m, base, conditional)?;
            return Some(score.map(|score| {
                let weight = if logarithmic {
                    weight
                } else {
                    build_call(m, "log", &[weight])
                };
                build_call(m, "add", &[weight, score])
            }));
        }
        return expectation(m, base, &mut |m, point| {
            let score = conditional(m, point)?;
            let weight = build_callback_call(m, weight, point, &parameter)?;
            let weight = crate::canon::inline_expression(m, weight);
            let weight = if logarithmic {
                weight
            } else {
                build_call(m, "log", &[weight])
            };
            Ok(build_call(m, "add", &[weight, score]))
        });
    }
    if let Some(expectation) = discrete_truncate_expectation(m, resolved, conditional) {
        return Some(expectation);
    }
    if builtin_name(m, resolved) == Some("Dirac") {
        if !matches!(m.type_of(resolved), Some(Type::Measure { domain, .. })
            if matches!(domain.as_ref(), Type::Scalar(_)))
        {
            return None;
        }
        let (_, parameters) = split_kernel_constructor(m, resolved)?;
        let atom = find_kwarg(m, &parameters, "value")?;
        return Some(conditional(m, atom));
    }
    if let Some(atoms) = crate::marginal::classify_atoms(m, resolved) {
        return Some((|| {
            let mut terms = Vec::with_capacity(atoms.len());
            for atom in atoms {
                let point = m.alloc(Node::Lit(Scalar::Int(atom)));
                let score = conditional(m, point)?;
                let mass = lower_measure_density(m, prior, point)?;
                terms.push(build_call(m, "add", &[score, mass]));
            }
            let terms = build_call(m, "vector", &terms);
            Ok(build_call(m, "logsumexp", &[terms]))
        })());
    }
    let (lo, hi) = bounds(m, prior)?;
    let coordinate = point(m);
    Some(conditional(m, coordinate).and_then(|conditional| {
        // A constant unit integrand needs only the prior's known total mass.
        if matches!(m.node(conditional), Node::Lit(Scalar::Real(0.0)))
            && let Some(mass) = closed_form_totalmass(m, prior)
        {
            return Ok(build_call(m, "log", &[mass]));
        }
        let prior = lower_measure_density(m, prior, coordinate)?;
        let integrand = build_call(m, "add", &[conditional, prior]);
        Ok(build_call(
            m,
            flatppl_core::LOG_INTEGRAL,
            &[integrand, coordinate, lo, hi],
        ))
    }))
}

pub(super) fn mass(m: &mut Module, measure: NodeId) -> Option<Result<NodeId, RefuseError>> {
    if let Some(integral) = lebesgue_mass(m, measure, false) {
        return Some(integral.map(|(mass, logarithmic)| {
            if logarithmic {
                mass
            } else {
                log_nonnegative_mass(m, mass)
            }
        }));
    }
    expectation(m, measure, &mut |m, _| {
        Ok(m.alloc(Node::Lit(Scalar::Real(0.0))))
    })
}

pub(crate) fn lebesgue_totalmass(
    m: &mut Module,
    measure: NodeId,
    allow_scalar: bool,
) -> Option<Result<NodeId, RefuseError>> {
    Some(
        lebesgue_mass(m, measure, allow_scalar)?.map(|(mass, logarithmic)| {
            if logarithmic {
                build_call(m, "exp", &[mass])
            } else {
                mass
            }
        }),
    )
}

// Direct Lebesgue reweighting gives unit reference density on the support.
// Integrate the callback alone, preserving raw derivatives at zero.
fn lebesgue_mass(
    m: &mut Module,
    measure: NodeId,
    allow_scalar: bool,
) -> Option<Result<(NodeId, bool), RefuseError>> {
    let (node, _) = resolve_ref_chain(m, measure);
    let logarithmic = match builtin_name(m, node)? {
        "weighted" => false,
        "logweighted" => true,
        _ => return None,
    };
    let Node::Call(call) = m.node(node) else {
        return None;
    };
    let [weight, base, ..] = *call.args else {
        return None;
    };
    let rectangle = rectangular_bounds(m, base);
    let interval = if rectangle.is_none() && allow_scalar {
        let (base, _) = resolve_ref_chain(m, base);
        let lebesgue = expect_builtin_call(m, base, "Lebesgue")?;
        let support = named_or_positional(m, lebesgue, "support")?;
        let (support, _) = resolve_ref_chain(m, support);
        let call = expect_builtin_call(m, support, "interval")?;
        let [lo, hi] = *call.args else {
            return None;
        };
        Some([lo, hi])
    } else {
        None
    };
    if (rectangle.is_none() && interval.is_none()) || measure_reaches_draw(m, measure, &[]) {
        return None;
    }
    Some((|| {
        let parameters = crate::extras::bind_extras(m, node, weight, 2, 1)?;
        let x = point(m);
        let mut axes = Vec::with_capacity(6);
        let coordinates = if let Some([lo, hi]) = interval {
            axes.extend([x, lo, hi]);
            x
        } else {
            let [xlo, xhi, ylo, yhi] = rectangle.expect("a rectangle or interval was verified");
            let y = point(m);
            axes.extend([x, xlo, xhi, y, ylo, yhi]);
            build_call(m, "vector", &[x, y])
        };
        let body = if !parameters.is_empty() || weight_is_variate_dependent(m, weight) {
            let body = build_callback_call(m, weight, coordinates, &parameters)?;
            crate::canon::inline_expression(m, body)
        } else {
            weight
        };
        let op = if logarithmic {
            flatppl_core::LOG_INTEGRAL
        } else {
            flatppl_core::MASS_INTEGRAL
        };
        let args: Vec<_> = std::iter::once(body).chain(axes).collect();
        let mass = build_call(m, op, &args);
        m.set_type(mass, Type::Scalar(ScalarType::Real));
        Ok((mass, logarithmic))
    })())
}

/// Preserve live bounds so quadrature can differentiate through their limits.
pub(super) fn rectangular_bounds(m: &Module, measure: NodeId) -> Option<[NodeId; 4]> {
    let (measure, _) = resolve_ref_chain(m, measure);
    let base = expect_builtin_call(m, measure, "Lebesgue")?;
    let support = named_or_positional(m, base, "support")?;
    rectangular_support_bounds(m, support)
}

pub(super) fn rectangular_support_bounds(m: &Module, support: NodeId) -> Option<[NodeId; 4]> {
    let (support, _) = resolve_ref_chain(m, support);
    let product = expect_builtin_call(m, support, "cartprod")?;
    let (&[x, y], true) = (product.args.as_ref(), product.named.is_empty()) else {
        return None;
    };
    let interval = |node| {
        let (node, _) = resolve_ref_chain(m, node);
        let bounds = expect_builtin_call(m, node, "interval")?;
        let (&[lo, hi], true) = (bounds.args.as_ref(), bounds.named.is_empty()) else {
            return None;
        };
        [lo, hi]
            .into_iter()
            .all(|bound| static_real(m, bound).is_none_or(f64::is_finite))
            .then_some([lo, hi])
    };
    let [xlo, xhi] = interval(x)?;
    let [ylo, yhi] = interval(y)?;
    Some([xlo, xhi, ylo, yhi])
}

/// Integrate the raw callback of a weighted scalar reference. Unlike a log
/// normalizer this stays differentiable when a smooth nonnegative weight is zero.
pub(super) fn raw_weighted_mass(m: &mut Module, measure: NodeId) -> Option<NodeId> {
    if measure_reaches_draw(m, measure, &[]) {
        return None;
    }
    let (node, _) = resolve_ref_chain(m, measure);
    let call = expect_builtin_call(m, node, "weighted")?;
    let (&weight, &base) = (call.args.first()?, call.args.get(1)?);
    let parameters = crate::extras::bind_extras(m, node, weight, 2, 1).ok()?;
    let (lo, hi) = bounds(m, base)?;
    let coordinate = point(m);
    let body = build_callback_call(m, weight, coordinate, &parameters).ok()?;
    let body = crate::canon::inline_expression(m, body);
    let result = build_call(m, flatppl_core::MASS_INTEGRAL, &[body, coordinate, lo, hi]);
    m.set_type(result, Type::Scalar(ScalarType::Real));
    Some(result)
}

/// Restricting a nonnegative count law to [1, inf] retains mass 1 - P(0).
pub(super) fn positive_count_log_mass(
    m: &mut Module,
    base: NodeId,
    lo: NodeId,
    hi: NodeId,
) -> Option<NodeId> {
    if static_real(m, lo) != Some(1.0) || static_real(m, hi) != Some(f64::INFINITY) {
        return None;
    }
    let (base, _) = resolve_ref_chain(m, base);
    let (constructor, parameters) = split_kernel_constructor(m, base)?;
    let log_p0 = match m.resolve(constructor) {
        "Poisson" => {
            let rate = find_kwarg(m, &parameters, "rate")?;
            build_call(m, "neg", &[rate])
        }
        "NegativeBinomial" => {
            let alpha = find_kwarg(m, &parameters, "alpha")?;
            let beta = find_kwarg(m, &parameters, "beta")?;
            let one = m.alloc(Node::Lit(Scalar::Real(1.0)));
            let reciprocal = build_call(m, "divide", &[one, beta]);
            let log_ratio = build_call(m, "log1p", &[reciprocal]);
            let magnitude = build_call(m, "mul", &[alpha, log_ratio]);
            build_call(m, "neg", &[magnitude])
        }
        _ => return None,
    };
    // expm1 avoids cancellation when almost all probability is at zero.
    let difference = build_call(m, "expm1", &[log_p0]);
    let mass = build_call(m, "neg", &[difference]);
    let log_mass = build_call(m, "log", &[mass]);
    let zero = m.alloc(Node::Lit(Scalar::Real(0.0)));
    let mut valid = m.alloc(Node::Lit(Scalar::Bool(true)));
    for (_, parameter) in parameters {
        let positive = build_call(m, "gt", &[parameter, zero]);
        let finite = build_call(m, "isfinite", &[parameter]);
        let admissible = build_call(m, "land", &[positive, finite]);
        valid = build_call(m, "land", &[valid, admissible]);
    }
    Some(exponential_tilt::guard(m, log_mass, valid))
}

/// The atoms a discrete law keeps under `interval(lo, hi)`: the closed interval
/// holds both ends (§06 truncate, §03 interval).
enum KeptAtoms {
    /// These atoms, listed.
    Listed(Vec<i64>),
    /// Every atom of a count law from `.0` on: the set's upper end is unbounded.
    From(i64),
}

/// The atoms of a discrete `base` in the closed range [lo, hi], or None when a
/// listed set or the atoms below an unbounded end would exceed
/// [`crate::marginal::MAX_ATOMS`].
fn kept_atoms(m: &mut Module, base: NodeId, lo: f64, hi: f64) -> Option<KeptAtoms> {
    if lo.is_nan() || hi.is_nan() {
        return None;
    }
    let inside = |k: i64| lo <= k as f64 && k as f64 <= hi;
    if let Some(atoms) = crate::marginal::classify_atoms(m, base) {
        return Some(KeptAtoms::Listed(
            atoms.into_iter().filter(|&k| inside(k)).collect(),
        ));
    }
    let (constructor, _) = split_kernel_constructor(m, base)?;
    if !matches!(
        m.resolve(constructor),
        "Poisson" | "Geometric" | "NegativeBinomial" | "NegativeBinomial2"
    ) {
        return None;
    }
    // Count laws live on the nonnegative integers.
    let first = lo.ceil().max(0.0);
    if hi == f64::INFINITY {
        return (first <= crate::marginal::MAX_ATOMS as f64)
            .then_some(KeptAtoms::From(first as i64));
    }
    let last = hi.floor();
    (first <= last && last - first < crate::marginal::MAX_ATOMS as f64)
        .then(|| KeptAtoms::Listed((first as i64..=last as i64).collect()))
}

/// The closed range of integer atoms a truncation set holds: a static
/// `interval(lo, hi)`, or a §03 named set. On integer atoms `posreals` (0, inf]
/// holds the same atoms as [1, inf].
fn integer_set_range(m: &Module, set: NodeId) -> Option<(f64, f64)> {
    let (set, _) = resolve_ref(m, set);
    if let Some(interval) = expect_builtin_call(m, set, "interval").filter(|c| c.args.len() == 2) {
        return Some((
            static_real(m, interval.args[0])?,
            static_real(m, interval.args[1])?,
        ));
    }
    let Node::Const(name) = m.node(set) else {
        return None;
    };
    match m.resolve(*name) {
        "reals" | "integers" => Some((f64::NEG_INFINITY, f64::INFINITY)),
        "posreals" | "posintegers" => Some((1.0, f64::INFINITY)),
        "nonnegreals" | "nonnegintegers" => Some((0.0, f64::INFINITY)),
        "unitinterval" => Some((0.0, 1.0)),
        _ => None,
    }
}

/// The base and the closed atom range of `truncate(<discrete law>, S)`.
pub(super) fn discrete_truncate(m: &Module, node: NodeId) -> Option<(NodeId, f64, f64)> {
    let call = expect_builtin_call(m, node, "truncate").filter(|c| c.args.len() == 2)?;
    let (base, set) = (call.args[0], call.args[1]);
    let (lo, hi) = integer_set_range(m, set)?;
    let (base, _) = resolve_ref_chain(m, base);
    matches!(m.type_of(base), Some(Type::Measure { domain, .. })
        if matches!(**domain, Type::Scalar(ScalarType::Integer)))
    .then_some((base, lo, hi))
}

/// log M([lo, hi]) for a discrete law M. An unbounded upper end keeps
/// 1 - P(X < lo), so only the atoms below lo are summed.
pub(super) fn discrete_log_mass(m: &mut Module, base: NodeId, lo: f64, hi: f64) -> Option<NodeId> {
    let (base, _) = resolve_ref_chain(m, base);
    match kept_atoms(m, base, lo, hi)? {
        KeptAtoms::Listed(atoms) => log_pmf_sum(m, base, atoms),
        KeptAtoms::From(0) => Some(m.alloc(Node::Lit(Scalar::Real(0.0)))),
        KeptAtoms::From(first) => {
            let excluded = log_pmf_sum(m, base, (0..first).collect())?;
            // expm1 keeps the mass when nearly all of it lies below lo.
            let difference = build_call(m, "expm1", &[excluded]);
            let mass = build_call(m, "neg", &[difference]);
            Some(build_call(m, "log", &[mass]))
        }
    }
}

/// log Σ_k pmf(k) over the given atoms; None when there are none.
fn log_pmf_sum(m: &mut Module, base: NodeId, atoms: Vec<i64>) -> Option<NodeId> {
    if atoms.is_empty() {
        return None;
    }
    let mut terms = Vec::with_capacity(atoms.len());
    for atom in atoms {
        let point = m.alloc(Node::Lit(Scalar::Int(atom)));
        terms.push(lower_measure_density(m, base, point).ok()?);
    }
    let terms = build_call(m, "vector", &terms);
    Some(build_call(m, "logsumexp", &[terms]))
}

/// The expectation over `truncate(<discrete law>, interval)`: a sum over the
/// kept atoms. An unbounded set is summable only for the unit integrand.
fn discrete_truncate_expectation(
    m: &mut Module,
    node: NodeId,
    conditional: &mut dyn FnMut(&mut Module, NodeId) -> Result<NodeId, RefuseError>,
) -> Option<Result<NodeId, RefuseError>> {
    let (base, lo, hi) = discrete_truncate(m, node)?;
    match kept_atoms(m, base, lo, hi)? {
        KeptAtoms::Listed(atoms) => Some((|| {
            let mut terms = Vec::with_capacity(atoms.len());
            for atom in atoms {
                let point = m.alloc(Node::Lit(Scalar::Int(atom)));
                let score = conditional(m, point)?;
                let mass = lower_measure_density(m, base, point)?;
                terms.push(build_call(m, "add", &[score, mass]));
            }
            if terms.is_empty() {
                return Err(refuse(
                    node,
                    m,
                    "the truncation set holds no atom of the law",
                ));
            }
            let terms = build_call(m, "vector", &terms);
            Ok(build_call(m, "logsumexp", &[terms]))
        })()),
        KeptAtoms::From(_) => {
            let coordinate = point(m);
            match conditional(m, coordinate) {
                Ok(unit) if matches!(m.node(unit), Node::Lit(Scalar::Real(0.0))) => {
                    discrete_log_mass(m, base, lo, hi).map(Ok)
                }
                Ok(_) => None,
                Err(error) => Some(Err(error)),
            }
        }
    }
}

/// Keep one interval error budget when the target cannot emit the exact CDF.
pub(crate) fn lower_interval_mass(
    m: &mut Module,
    node: NodeId,
    options: &crate::LoweringOptions<'_>,
) -> Result<NodeId, RefuseError> {
    let call = expect_builtin_call(m, node, flatppl_core::LOG_INTERVAL_MASS)
        .ok_or_else(|| refuse(node, m, "expected an interval mass"))?;
    let [base, lo, hi] = call.args.as_ref() else {
        return Err(refuse(
            node,
            m,
            "interval mass expects a constructor and two bounds",
        ));
    };
    let (base, lo, hi) = (*base, *lo, *hi);
    // lower_normalize admits only normalized scalar continuous bases here.
    if let Some((support_lo, support_hi)) = bounds(m, base)
        && let (Some(lo), Some(hi), Some(support_lo), Some(support_hi)) = (
            static_real(m, lo),
            static_real(m, hi),
            static_real(m, support_lo),
            static_real(m, support_hi),
        )
        && lo <= support_lo
        && hi >= support_hi
    {
        return Ok(m.alloc(Node::Lit(Scalar::Real(0.0))));
    }
    let (kernel, input) = kernel_and_input(m, base)?;
    let Node::Const(ctor) = m.node(kernel) else {
        unreachable!("kernel_and_input returns a constructor tag");
    };
    let ctor = *ctor;
    if m.resolve(ctor) == "Normal" && options.retains("special-functions", "log_ndtr") {
        let (mu, sigma) = (find_field(m, input, "mu"), find_field(m, input, "sigma"));
        if let (Some(mu), Some(sigma)) = (mu, sigma) {
            // Keep the distribution's parameter checks before replacing its
            // CDF difference with a log-space calculation.
            let witness = build_touniform(m, kernel, input, mu);
            let valid = build_call(m, "isfinite", &[witness]);
            return Ok(normal_log_mass(
                m,
                mu,
                sigma,
                lo,
                hi,
                valid,
                options.compact_log_cdf_difference,
            ));
        }
    }
    if let Some(log_mass) = survival_log_mass(m, ctor, input, lo, hi) {
        return Ok(log_mass);
    }
    if m.resolve(ctor) == "Gamma"
        && static_real(m, hi) == Some(f64::INFINITY)
        && let Some(log_mass) = erlang_log_survival(m, input, lo)
    {
        return Ok(log_mass);
    }
    if m.resolve(ctor) == "StudentT"
        && let Some(log_mass) = student_t_log_mass(m, input, lo, hi)
    {
        return Ok(log_mass);
    }
    if options.numerical_integrals
        && options
            .supports_cdf
            .is_some_and(|supports| !supports(m.resolve(ctor)))
        && !(student_t_endpoint_cdf(m, kernel, input, lo).is_some()
            && student_t_endpoint_cdf(m, kernel, input, hi).is_some())
    {
        let (base_lo, base_hi) = bounds(m, base).ok_or_else(|| {
            refuse(
                node,
                m,
                "numerical interval mass needs a scalar continuous constructor",
            )
        })?;
        let lo = build_call(m, "max", &[lo, base_lo]);
        let hi = build_call(m, "min", &[hi, base_hi]);
        let hi = build_call(m, "max", &[lo, hi]);
        let coordinate = point(m);
        let integrand = lower_measure_density(m, base, coordinate)?;
        let log_z = build_call(
            m,
            flatppl_core::LOG_INTEGRAL,
            &[integrand, coordinate, lo, hi],
        );
        let inf = m.alloc(Node::Lit(Scalar::Real(f64::INFINITY)));
        let valid = build_call(m, "lt", &[log_z, inf]);
        let nan = m.alloc(Node::Lit(Scalar::Real(f64::NAN)));
        return Ok(build_call(m, "ifelse", &[valid, log_z, nan]));
    }
    // Infinite endpoints have constant CDF values. Emitting their location/scale
    // arithmetic would introduce infinity * zero in otherwise finite gradients.
    // A select differentiates both branches, so an endpoint that may be
    // infinite at run time feeds the CDF a finite interior point instead.
    let mut interior = None;
    let mut cdf = |m: &mut Module, x| match static_real(m, x) {
        Some(x) if x.is_infinite() => {
            m.alloc(Node::Lit(Scalar::Real(if x < 0.0 { 0.0 } else { 1.0 })))
        }
        Some(_) => build_touniform(m, kernel, input, x),
        None => {
            let interior = *interior.get_or_insert_with(|| interior_point(m, base));
            let finite = build_call(m, "isfinite", &[x]);
            let safe = build_call(m, "ifelse", &[finite, x, interior]);
            let cdf = build_touniform(m, kernel, input, safe);
            let zero = m.alloc(Node::Lit(Scalar::Real(0.0)));
            let one = m.alloc(Node::Lit(Scalar::Real(1.0)));
            let negative = build_call(m, "lt", &[x, zero]);
            let positive = build_call(m, "gt", &[x, zero]);
            // A NaN endpoint has neither sign. Propagate it.
            let limit = build_call(m, "ifelse", &[positive, one, x]);
            let limit = build_call(m, "ifelse", &[negative, zero, limit]);
            build_call(m, "ifelse", &[finite, cdf, limit])
        }
    };
    let cdf_hi = cdf(m, hi);
    let cdf_lo = cdf(m, lo);
    let z = match (static_real(m, cdf_hi), static_real(m, cdf_lo)) {
        (Some(hi), Some(lo)) => m.alloc(Node::Lit(Scalar::Real(hi - lo))),
        _ => build_call(m, "sub", &[cdf_hi, cdf_lo]),
    };
    // An empty intersection has exactly zero mass. Keep log's argument valid
    // before AD so its unused derivative cannot introduce 0/0 into a zero bin.
    Ok(log_nonnegative_mass(m, z))
}

/// For cumulative hazard H, the interval mass is exp(-H(lo)) times
/// 1 - exp(-(H(hi) - H(lo))). Compute both differences in log space.
fn survival_log_mass(
    m: &mut Module,
    ctor: Symbol,
    input: NodeId,
    lo: NodeId,
    hi: NodeId,
) -> Option<NodeId> {
    let (rate, weibull) = match m.resolve(ctor) {
        "Exponential" => (Some(find_field(m, input, "rate")?), None),
        "Weibull" => (
            None,
            Some([
                find_field(m, input, "shape")?,
                find_field(m, input, "scale")?,
            ]),
        ),
        _ => return None,
    };
    let real = |m: &mut Module, value| m.alloc(Node::Lit(Scalar::Real(value)));
    let zero = real(m, 0.0);
    let half = real(m, 0.5);
    let one = real(m, 1.0);
    let mut valid = m.alloc(Node::Lit(Scalar::Bool(true)));
    let mut parameter = |m: &mut Module, x| {
        let positive = build_call(m, "gt", &[x, zero]);
        let finite = build_call(m, "isfinite", &[x]);
        let allowed = build_call(m, "land", &[positive, finite]);
        valid = build_call(m, "land", &[valid, allowed]);
        build_call(m, "ifelse", &[allowed, x, one])
    };
    let rate = rate.map(|rate| parameter(m, rate));
    let weibull = weibull.map(|params| params.map(|x| parameter(m, x)));
    let nan_lo = build_call(m, "isnan", &[lo]);
    let nan_hi = build_call(m, "isnan", &[hi]);
    let nan = build_call(m, "lor", &[nan_lo, nan_hi]);
    let bounds_valid = build_call(m, "lnot", &[nan]);
    let valid = build_call(m, "land", &[valid, bounds_valid]);
    let lower = build_call(m, "max", &[lo, zero]);
    let nonempty = build_call(m, "lt", &[lower, hi]);
    let upper_finite = build_call(m, "isfinite", &[hi]);
    let bounded = build_call(m, "land", &[nonempty, upper_finite]);
    // Safe placeholders keep inactive endpoints out of log and power arithmetic.
    let lower_positive = build_call(m, "gt", &[lower, zero]);
    let positive = build_call(m, "land", &[nonempty, lower_positive]);
    let hazard_at = build_call(m, "ifelse", &[positive, lower, one]);
    let l = build_call(m, "ifelse", &[bounded, lower, half]);
    let h = build_call(m, "ifelse", &[bounded, hi, one]);
    let width = build_call(m, "sub", &[h, l]);
    let (hazard, log_gap) = if let Some(rate) = rate {
        let hazard = build_call(m, "mul", &[rate, hazard_at]);
        let log_rate = build_call(m, "log", &[rate]);
        let log_width = build_call(m, "log", &[width]);
        (hazard, build_call(m, "add", &[log_rate, log_width]))
    } else {
        let [shape, scale] = weibull?;
        let log_scale = build_call(m, "log", &[scale]);
        let log_hazard = |m: &mut Module, x| {
            let log_x = build_call(m, "log", &[x]);
            let log_standardized = build_call(m, "sub", &[log_x, log_scale]);
            build_call(m, "mul", &[shape, log_standardized])
        };
        let log_hazard_at = log_hazard(m, hazard_at);
        let hazard = build_call(m, "exp", &[log_hazard_at]);
        let log_h = log_hazard(m, h);
        // The close-endpoint path preserves a small relative width before log1p.
        let h_half = build_call(m, "mul", &[h, half]);
        let near = build_call(m, "gt", &[l, h_half]);
        let near_width = build_call(m, "ifelse", &[near, width, half]);
        let near_upper = build_call(m, "ifelse", &[near, h, one]);
        let relative = build_call(m, "divide", &[near_width, near_upper]);
        let negative = build_call(m, "neg", &[relative]);
        let near_ratio = build_call(m, "log1p", &[negative]);
        let l_positive = build_call(m, "gt", &[l, zero]);
        let not_near = build_call(m, "lnot", &[near]);
        let far = build_call(m, "land", &[l_positive, not_near]);
        let safe_l = build_call(m, "ifelse", &[far, l, half]);
        let safe_h = build_call(m, "ifelse", &[far, h, one]);
        let log_l = build_call(m, "log", &[safe_l]);
        let log_h_input = build_call(m, "log", &[safe_h]);
        let far_ratio = build_call(m, "sub", &[log_l, log_h_input]);
        let ratio = build_call(m, "ifelse", &[near, near_ratio, far_ratio]);
        let negative_ratio = build_call(m, "neg", &[ratio]);
        let log_ratio = build_call(m, "log", &[negative_ratio]);
        let log_shape = build_call(m, "log", &[shape]);
        let log_hazard_ratio = build_call(m, "add", &[log_shape, log_ratio]);
        let fraction = log_survival_gap(m, log_hazard_ratio);
        let fraction = build_call(m, "ifelse", &[l_positive, fraction, zero]);
        (hazard, build_call(m, "add", &[log_h, fraction]))
    };
    let hazard = build_call(m, "ifelse", &[positive, hazard, zero]);
    let gap = log_survival_gap(m, log_gap);
    let gap = build_call(m, "ifelse", &[upper_finite, gap, zero]);
    let log_mass = build_call(m, "sub", &[gap, hazard]);
    let neg_inf = real(m, f64::NEG_INFINITY);
    let log_mass = build_call(m, "ifelse", &[nonempty, log_mass, neg_inf]);
    Some(exponential_tilt::guard(m, log_mass, valid))
}

/// log(1 - exp(-exp(x))). Outside the cutoffs the limits are accurate to
/// floating-point precision and avoid underflow or an overflowing derivative.
fn log_survival_gap(m: &mut Module, x: NodeId) -> NodeId {
    let real = |m: &mut Module, value| m.alloc(Node::Lit(Scalar::Real(value)));
    let low = real(m, -36.0);
    let high = real(m, 7.0);
    let zero = real(m, 0.0);
    let small = build_call(m, "lt", &[x, low]);
    let large = build_call(m, "gt", &[x, high]);
    let outside = build_call(m, "lor", &[small, large]);
    let safe = build_call(m, "ifelse", &[outside, zero, x]);
    let magnitude = build_call(m, "exp", &[safe]);
    let negative = build_call(m, "neg", &[magnitude]);
    let log_mass = exponential_tilt::log1mexp(m, negative);
    let log_mass = build_call(m, "ifelse", &[small, x, log_mass]);
    build_call(m, "ifelse", &[large, zero, log_mass])
}

fn normal_log_mass(
    m: &mut Module,
    mu: NodeId,
    sigma: NodeId,
    lo: NodeId,
    hi: NodeId,
    valid: NodeId,
    compact_difference: bool,
) -> NodeId {
    let real = |m: &mut Module, value| m.alloc(Node::Lit(Scalar::Real(value)));
    let zero = real(m, 0.0);
    let one = real(m, 1.0);
    let positive_scale = build_call(m, "gt", &[sigma, zero]);
    let finite_scale = build_call(m, "isfinite", &[sigma]);
    let positive_scale = build_call(m, "land", &[positive_scale, finite_scale]);
    let valid = build_call(m, "land", &[valid, positive_scale]);
    let mu = build_call(m, "ifelse", &[valid, mu, zero]);
    let sigma = build_call(m, "ifelse", &[valid, sigma, one]);
    let log_ndtr = normal_log_cdf_member(m);
    let apply = |m: &mut Module, x| {
        let nan = build_call(m, "isnan", &[x]);
        let safe = build_call(m, "ifelse", &[nan, zero, x]);
        m.alloc(Node::Call(Call {
            head: CallHead::User(log_ndtr),
            args: Box::new([safe]),
            named: Box::default(),
            inputs: None,
        }))
    };
    let (bound_lo, bound_hi) = (lo, hi);
    let lo = standardized_normal_bound(m, mu, sigma, lo);
    let hi = standardized_normal_bound(m, mu, sigma, hi);
    let log_mass = if static_real(m, bound_lo) == Some(f64::NEG_INFINITY) {
        apply(m, hi)
    } else if static_real(m, bound_hi) == Some(f64::INFINITY) {
        let reflected = build_call(m, "neg", &[lo]);
        apply(m, reflected)
    } else {
        normal_log_mass_difference(m, lo, hi, bound_hi, compact_difference, apply)
    };
    let nan_lo = build_call(m, "isnan", &[bound_lo]);
    let nan_hi = build_call(m, "isnan", &[bound_hi]);
    let nan_bound = build_call(m, "lor", &[nan_lo, nan_hi]);
    let valid_bounds = build_call(m, "lnot", &[nan_bound]);
    let valid = build_call(m, "land", &[valid, valid_bounds]);
    let nan = real(m, f64::NAN);
    build_call(m, "ifelse", &[valid, log_mass, nan])
}

fn standardized_normal_bound(m: &mut Module, mu: NodeId, sigma: NodeId, bound: NodeId) -> NodeId {
    if static_real(m, bound).is_some_and(f64::is_infinite) {
        return bound;
    }
    // Keep infinite endpoints out of location and scale arithmetic. A NaN
    // endpoint is masked here and restored by the caller after the CDF call.
    let finite = build_call(m, "isfinite", &[bound]);
    let safe = build_call(m, "ifelse", &[finite, bound, mu]);
    let difference = build_call(m, "sub", &[safe, mu]);
    let finite_z = build_call(m, "divide", &[difference, sigma]);
    let nan = build_call(m, "isnan", &[bound]);
    let zero = m.alloc(Node::Lit(Scalar::Real(0.0)));
    let endpoint = build_call(m, "ifelse", &[nan, zero, bound]);
    build_call(m, "ifelse", &[finite, finite_z, endpoint])
}

fn normal_log_mass_difference(
    m: &mut Module,
    lo: NodeId,
    hi: NodeId,
    bound_hi: NodeId,
    compact_difference: bool,
    log_cdf: impl Fn(&mut Module, NodeId) -> NodeId,
) -> NodeId {
    let zero = m.alloc(Node::Lit(Scalar::Real(0.0)));
    let reflected = build_call(m, "gt", &[lo, zero]);
    let neg_lo = build_call(m, "neg", &[lo]);
    let neg_hi = build_call(m, "neg", &[hi]);
    // Reflect positive intervals before subtraction to keep two small tails
    // instead of two probabilities rounded to one.
    let x = build_call(m, "ifelse", &[reflected, neg_lo, hi]);
    let y = build_call(m, "ifelse", &[reflected, neg_hi, lo]);
    // With an unbounded upper endpoint, the mass is a single reflected tail
    // regardless of the sign of the lower endpoint.
    let lower_inf = m.alloc(Node::Lit(Scalar::Real(f64::NEG_INFINITY)));
    let positive = build_call(m, "gt", &[bound_hi, zero]);
    let infinite = build_call(m, "isinf", &[bound_hi]);
    let no = m.alloc(Node::Lit(Scalar::Bool(false)));
    let right = build_call(m, "ifelse", &[infinite, positive, no]);
    let x = build_call(m, "ifelse", &[right, neg_lo, x]);
    let y = build_call(m, "ifelse", &[right, lower_inf, y]);
    let log_x = log_cdf(m, x);
    let log_y = log_cdf(m, y);
    if compact_difference {
        return build_call(m, flatppl_core::LOG_CDF_DIFFERENCE, &[log_x, log_y]);
    }
    let difference = build_call(m, "sub", &[log_y, log_x]);
    let nonempty = build_call(m, "lt", &[difference, zero]);
    let minus_one = m.alloc(Node::Lit(Scalar::Real(-1.0)));
    let safe = build_call(m, "ifelse", &[nonempty, difference, minus_one]);
    let log_ratio_complement = exponential_tilt::log1mexp(m, safe);
    let log_mass = build_call(m, "add", &[log_x, log_ratio_complement]);
    let neg_inf = m.alloc(Node::Lit(Scalar::Real(f64::NEG_INFINITY)));
    build_call(m, "ifelse", &[nonempty, log_mass, neg_inf])
}

fn normal_log_cdf_member(m: &mut Module) -> NodeId {
    let is_special_functions = |id| {
        expect_builtin_call(m, id, "standard_module")
            .and_then(|call| call.args.first())
            .is_some_and(|&arg| matches!(m.node(arg), Node::Lit(Scalar::Str(name)) if name.as_ref() == "special-functions"))
    };
    let alias = m
        .bindings()
        .find_map(|(_, binding)| is_special_functions(binding.rhs).then_some(binding.name));
    let alias = alias.unwrap_or_else(|| {
        let alias = (0..)
            .find_map(|index| {
                let name = m.intern(&format!("_normal_special_{index}"));
                m.binding_by_name(name).is_none().then_some(name)
            })
            .unwrap();
        let module = m.alloc(Node::Lit(Scalar::Str("special-functions".into())));
        let version = m.alloc(Node::Lit(Scalar::Str("0.1".into())));
        let rhs = build_call(m, "standard_module", &[module, version]);
        m.add_binding(flatppl_core::Binding {
            name: alias,
            rhs,
            doc: None,
            public: false,
            synthetic: true,
        });
        alias
    });
    let name = m.intern("log_ndtr");
    m.alloc(Node::Ref(Ref {
        ns: RefNs::Module(alias),
        name,
    }))
}

/// A finite point inside the support of `base`. A CDF's derivatives can be
/// infinite on a support boundary, so the placeholder avoids the boundary too.
fn interior_point(m: &mut Module, base: NodeId) -> NodeId {
    let Some((lo, hi)) = bounds(m, base) else {
        return m.alloc(Node::Lit(Scalar::Real(0.0)));
    };
    let one = m.alloc(Node::Lit(Scalar::Real(1.0)));
    match (static_real(m, lo), static_real(m, hi)) {
        (Some(lo), Some(hi)) => {
            let point = match (lo.is_finite(), hi.is_finite()) {
                (true, true) => 0.5 * (lo + hi),
                (true, false) => lo + 1.0,
                (false, true) => hi - 1.0,
                (false, false) => 0.0,
            };
            m.alloc(Node::Lit(Scalar::Real(point)))
        }
        (_, Some(hi)) if hi == f64::INFINITY => build_call(m, "add", &[lo, one]),
        (Some(lo), _) if lo == f64::NEG_INFINITY => build_call(m, "sub", &[hi, one]),
        _ => {
            let sum = build_call(m, "add", &[lo, hi]);
            let half = m.alloc(Node::Lit(Scalar::Real(0.5)));
            build_call(m, "mul", &[half, sum])
        }
    }
}

/// For nu=2m, the smaller Student-t tail is I_p(m,m), where
/// p=(1-|x|/sqrt(nu+x*x))/2. Integer beta tails are finite binomial sums.
fn student_t_log_mass(m: &mut Module, input: NodeId, lo: NodeId, hi: NodeId) -> Option<NodeId> {
    let nu = static_real(m, find_field(m, input, "nu")?)?;
    // Bound source-driven expansion, as for the integer Gamma rule below.
    if !(2.0..=128.0).contains(&nu) || nu % 2.0 != 0.0 {
        return None;
    }
    let zero = m.alloc(Node::Lit(Scalar::Real(0.0)));
    // Reflect positive intervals so small masses do not subtract CDFs near one.
    let positive = build_call(m, "ge", &[lo, zero]);
    let neg_hi = build_call(m, "neg", &[hi]);
    let neg_lo = build_call(m, "neg", &[lo]);
    let left = build_call(m, "ifelse", &[positive, neg_hi, lo]);
    let right = build_call(m, "ifelse", &[positive, neg_lo, hi]);
    let left = even_student_t_cdf(m, nu as usize, left);
    let right = even_student_t_cdf(m, nu as usize, right);
    let mass = build_call(m, "sub", &[right, left]);
    Some(log_nonnegative_mass(m, mass))
}

fn even_student_t_cdf(m: &mut Module, nu: usize, x: NodeId) -> NodeId {
    if let Some(x) = static_real(m, x)
        && (x == 0.0 || x.is_infinite())
    {
        let cdf = if x < 0.0 {
            0.0
        } else if x > 0.0 {
            1.0
        } else {
            0.5
        };
        return m.alloc(Node::Lit(Scalar::Real(cdf)));
    }
    let zero = m.alloc(Node::Lit(Scalar::Real(0.0)));
    let one = m.alloc(Node::Lit(Scalar::Real(1.0)));
    let half = m.alloc(Node::Lit(Scalar::Real(0.5)));
    let root_nu = m.alloc(Node::Lit(Scalar::Real((nu as f64).sqrt())));
    let negative = build_call(m, "lt", &[x, zero]);
    let neg_x = build_call(m, "neg", &[x]);
    // Select a side at zero: abs'(0)=0 would erase the CDF's nonzero derivative.
    let magnitude = build_call(m, "ifelse", &[negative, neg_x, x]);
    let finite = build_call(m, "isfinite", &[x]);
    let magnitude = build_call(m, "ifelse", &[finite, magnitude, zero]);
    let scaled = build_call(m, "divide", &[magnitude, root_nu]);
    let squared = build_call(m, "mul", &[scaled, scaled]);
    let denominator = build_call(m, "add", &[one, squared]);
    let cosine_squared = build_call(m, "divide", &[one, denominator]);
    let cosine = build_call(m, "sqrt", &[cosine_squared]);
    let sine = build_call(m, "mul", &[scaled, cosine]);
    let denominator = build_call(m, "add", &[one, sine]);
    let numerator = build_call(m, "mul", &[half, cosine_squared]);
    // Equivalent to (1-sine)/2 without cancellation in the tail.
    let p = build_call(m, "divide", &[numerator, denominator]);
    let q = build_call(m, "sub", &[one, p]);
    let order = nu / 2;
    let mut term = one;
    for j in 1..=order {
        let coefficient = m.alloc(Node::Lit(Scalar::Real((order - 1 + j) as f64 / j as f64)));
        let factor = build_call(m, "mul", &[coefficient, p]);
        term = build_call(m, "mul", &[term, factor]);
    }
    for _ in 1..order {
        term = build_call(m, "mul", &[term, q]);
    }
    let mut tail = term;
    let ratio = build_call(m, "divide", &[p, q]);
    for j in order..nu - 1 {
        let coefficient = m.alloc(Node::Lit(Scalar::Real(
            (nu - 1 - j) as f64 / (j + 1) as f64,
        )));
        let factor = build_call(m, "mul", &[coefficient, ratio]);
        term = build_call(m, "mul", &[term, factor]);
        tail = build_call(m, "add", &[tail, term]);
    }
    let complement = build_call(m, "sub", &[one, tail]);
    let cdf = build_call(m, "ifelse", &[negative, tail, complement]);
    let positive = build_call(m, "gt", &[x, zero]);
    // A nonfinite input with neither sign is already NaN; propagate it.
    let nonfinite = build_call(m, "ifelse", &[positive, one, x]);
    let nonfinite = build_call(m, "ifelse", &[negative, zero, nonfinite]);
    build_call(m, "ifelse", &[finite, cdf, nonfinite])
}

/// Integer-shape Gamma survival: exp(-z) * sum(z^j/j!), z = rate*x.
/// DLMF 8.4.10–11. Keep the mass in log space, including far-tail truncation.
fn erlang_log_survival(m: &mut Module, input: NodeId, x: NodeId) -> Option<NodeId> {
    let shape = static_real(m, find_field(m, input, "shape")?)?;
    // Bound source-driven graph expansion. Other shapes retain the CDF path.
    if !(1.0..=128.0).contains(&shape) || shape.fract() != 0.0 {
        return None;
    }
    let rate = find_field(m, input, "rate")?;
    let zero = m.alloc(Node::Lit(Scalar::Real(0.0)));
    let one = m.alloc(Node::Lit(Scalar::Real(1.0)));
    let positive = build_call(m, "gt", &[x, zero]);
    let finite = build_call(m, "isfinite", &[x]);
    let interior = build_call(m, "land", &[positive, finite]);
    let safe_x = build_call(m, "ifelse", &[interior, x, one]);
    let z = build_call(m, "mul", &[rate, safe_x]);
    let log_polynomial = if shape == 1.0 {
        zero
    } else if shape == 2.0 {
        build_call(m, "log1p", &[z])
    } else {
        let log_z = build_call(m, "log", &[z]);
        let mut terms = vec![zero];
        let mut log_factorial = 0.0;
        for j in 1..shape as usize {
            log_factorial += (j as f64).ln();
            let power = m.alloc(Node::Lit(Scalar::Real(j as f64)));
            let constant = m.alloc(Node::Lit(Scalar::Real(log_factorial)));
            let power = build_call(m, "mul", &[power, log_z]);
            terms.push(build_call(m, "sub", &[power, constant]));
        }
        let terms = build_call(m, "vector", &terms);
        build_call(m, "logsumexp", &[terms])
    };
    let log_mass = build_call(m, "sub", &[log_polynomial, z]);
    let neg_inf = m.alloc(Node::Lit(Scalar::Real(f64::NEG_INFINITY)));
    let positive_mass = build_call(m, "ifelse", &[finite, log_mass, neg_inf]);
    Some(build_call(m, "ifelse", &[positive, positive_mass, zero]))
}

/// A conservative scalar support interval. Preserve dynamic interval endpoints.
fn bounds(m: &mut Module, measure: NodeId) -> Option<(NodeId, NodeId)> {
    let (node, _) = resolve_ref_chain(m, measure);
    if !continuous(m, node)
        || !matches!(m.type_of(node), Some(Type::Measure { domain, .. })
        if matches!(domain.as_ref(), Type::Scalar(ScalarType::Real)))
    {
        return None;
    }
    if let Node::Call(c) = m.node(node).clone()
        && let CallHead::Builtin(head) = c.head
    {
        match m.resolve(head) {
            "Pareto" => {
                let (_, parameters) = split_kernel_constructor(m, node)?;
                let lo = find_kwarg(m, &parameters, "scale")?;
                return Some((lo, m.alloc(Node::Lit(Scalar::Real(f64::INFINITY)))));
            }
            "VonMises" => {
                return Some((
                    m.alloc(Node::Lit(Scalar::Real(-std::f64::consts::PI))),
                    m.alloc(Node::Lit(Scalar::Real(std::f64::consts::PI))),
                ));
            }
            _ => (),
        }
        let support = match m.resolve(head) {
            "truncate" => c.args.get(1).copied(),
            "Lebesgue" | "Uniform" => named_or_positional(m, &c, "support"),
            "weighted" | "logweighted" if c.args.len() >= 2 => {
                return bounds(m, c.args[1]);
            }
            "normalize" if c.args.len() == 1 => return bounds(m, c.args[0]),
            _ => None,
        };
        if let Some(set) = support {
            let (set, _) = resolve_ref_chain(m, set);
            if let Some(interval) = expect_builtin_call(m, set, "interval")
                && interval.args.len() == 2
            {
                let (mut lo, mut hi) = (interval.args[0], interval.args[1]);
                if m.resolve(head) == "truncate" {
                    let (base_lo, base_hi) = bounds(m, c.args[0])?;
                    lo = build_call(m, "max", &[lo, base_lo]);
                    hi = build_call(m, "min", &[hi, base_hi]);
                    hi = build_call(m, "max", &[lo, hi]);
                }
                return Some((lo, hi));
            }
        }
    }
    let (lo, hi) = match m.valueset_of(node)? {
        ValueSet::Reals => (f64::NEG_INFINITY, f64::INFINITY),
        ValueSet::PosReals | ValueSet::NonNegReals => (0.0, f64::INFINITY),
        ValueSet::UnitInterval => (0.0, 1.0),
        ValueSet::Interval(lo, hi) => (*lo, *hi),
        _ => return None,
    };
    Some((
        m.alloc(Node::Lit(Scalar::Real(lo))),
        m.alloc(Node::Lit(Scalar::Real(hi))),
    ))
}

/// A real support does not prove absolute continuity: Dirac and mixtures can
/// have atoms there. Admit only constructors with a scalar Lebesgue density.
fn continuous(m: &Module, measure: NodeId) -> bool {
    let (node, _) = resolve_ref_chain(m, measure);
    let Node::Call(c) = m.node(node) else {
        return false;
    };
    let CallHead::Builtin(head) = c.head else {
        return false;
    };
    match m.resolve(head) {
        "Lebesgue" | "Uniform" | "Normal" | "Exponential" | "Gamma" | "InverseGamma" | "Beta"
        | "StudentT" | "Cauchy" | "Laplace" | "Logistic" | "LogNormal" | "Weibull" | "Rayleigh"
        | "ChiSquared" | "GeneralizedNormal" | "Pareto" | "VonMises" => true,
        "weighted" | "logweighted" if c.args.len() >= 2 => continuous(m, c.args[1]),
        "truncate" | "normalize" => c.args.first().is_some_and(|&base| continuous(m, base)),
        _ => false,
    }
}
