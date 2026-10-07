//! Positive one-dimensional integrals in the compiler's internal IR.
//! Exact rules run first. Ordinary FlatPDL conformance rejects this extension.

use super::*;

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
    ancestors: &[Ancestor],
) -> Option<Result<NodeId, RefuseError>> {
    let Ancestor::Named { name, prior } = *ancestors.first()? else {
        return None;
    };
    if ancestors
        .iter()
        .any(|a| !matches!(a, Ancestor::Named { name: other, .. } if *other == name))
    {
        return None;
    }
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
        let [weight, base] = call.args.as_ref() else {
            return None;
        };
        let (weight, base) = (*weight, *base);
        if !weight_is_variate_dependent(m, weight) {
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
            let weight = build_weight_call(m, weight, point)?;
            let weight = if logarithmic {
                weight
            } else {
                build_call(m, "log", &[weight])
            };
            Ok(build_call(m, "add", &[weight, score]))
        });
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
    expectation(m, measure, &mut |m, _| {
        Ok(m.alloc(Node::Lit(Scalar::Real(0.0))))
    })
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
    let (kernel, input) = kernel_and_input(m, base)?;
    let Node::Const(ctor) = m.node(kernel) else {
        unreachable!("kernel_and_input returns a constructor tag");
    };
    if options.numerical_integrals
        && options
            .supports_cdf
            .is_some_and(|supports| !supports(m.resolve(*ctor)))
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
        let finite = build_call(m, "isfinite", &[log_z]);
        let nan = m.alloc(Node::Lit(Scalar::Real(f64::NAN)));
        return Ok(build_call(m, "ifelse", &[finite, log_z, nan]));
    }
    let cdf_hi = build_touniform(m, kernel, input, hi);
    let cdf_lo = build_touniform(m, kernel, input, lo);
    let z = build_call(m, "sub", &[cdf_hi, cdf_lo]);
    Ok(build_call(m, "log", &[z]))
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
            "weighted" | "logweighted" if c.args.len() == 2 => return bounds(m, c.args[1]),
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
        | "StudentT" | "Cauchy" | "Laplace" | "Logistic" | "LogNormal" | "Weibull"
        | "ChiSquared" | "GeneralizedNormal" | "Pareto" | "VonMises" => true,
        "weighted" | "logweighted" if c.args.len() == 2 => continuous(m, c.args[1]),
        "truncate" | "normalize" => c.args.first().is_some_and(|&base| continuous(m, base)),
        _ => false,
    }
}
