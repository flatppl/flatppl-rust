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
        if !matches!(call.args.len(), 2 | 3) {
            return None;
        }
        let (weight, base, parameter) = (call.args[0], call.args[1], call.args.get(2).copied());
        if parameter.is_none() && !weight_is_variate_dependent(m, weight) {
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
            let weight = build_weight_call(m, weight, point, parameter)?;
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
    Some(build_call(m, "log", &[mass]))
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
            "weighted" | "logweighted" if matches!(c.args.len(), 2 | 3) => {
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
        | "StudentT" | "Cauchy" | "Laplace" | "Logistic" | "LogNormal" | "Weibull"
        | "ChiSquared" | "GeneralizedNormal" | "Pareto" | "VonMises" => true,
        "weighted" | "logweighted" if matches!(c.args.len(), 2 | 3) => continuous(m, c.args[1]),
        "truncate" | "normalize" => c.args.first().is_some_and(|&base| continuous(m, base)),
        _ => false,
    }
}
