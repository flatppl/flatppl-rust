//! Closed normalizers on ordered and zero-sum sets. Other bases retain the
//! ordinary normalize/logweighted path.

use super::*;

#[derive(Clone, Copy)]
pub(crate) enum NormalSet {
    Ordered,
    ZeroSum,
}

pub(crate) struct NormalOnSet {
    pub kind: NormalSet,
    pub n: usize,
    pub mu: NodeId,
    pub sigma: NodeId,
    support: NodeId,
}

pub(crate) enum SetLaw {
    IidNormal(NormalOnSet),
    OrderedIid {
        source: NodeId,
        kernel: NodeId,
        n: usize,
        support: NodeId,
    },
    ZeroSumNormals(NormalOnSet),
}

pub(crate) fn on_set(m: &mut Module, node: NodeId) -> Result<Option<SetLaw>, RefuseError> {
    let Some(call) = expect_builtin_call(m, node, "tilt") else {
        return Ok(None);
    };
    let [source, base] = *call.args else {
        return Ok(None);
    };
    let (base, _) = resolve_ref_chain(m, base);
    let Some(lebesgue) = expect_builtin_call(m, base, "Lebesgue") else {
        return Ok(None);
    };
    let Some(support) = named_or_positional(m, lebesgue, "support") else {
        return Ok(None);
    };
    let (set, _) = resolve_ref_chain(m, support);
    let kind = match builtin_name(m, set) {
        Some("ordered") => NormalSet::Ordered,
        Some("zerosum") => NormalSet::ZeroSum,
        _ => return Ok(None),
    };
    let n = iid_static_size(m, base).filter(|&n| n > 0).ok_or_else(|| {
        refuse(
            node,
            m,
            "tilt on ordered/zerosum requires a positive static size",
        )
    })?;
    let (mut source, _) = resolve_ref_chain(m, source);
    if let Some(law) = expect_builtin_call(m, source, "lawof") {
        let Some(&value) = law.args.first() else {
            return Ok(None);
        };
        source = resolve_ref_chain(m, value).0;
        if let Some(measure) = draw_argument(m, source) {
            source = resolve_ref_chain(m, measure).0;
        }
        if measure_reaches_draw(m, source, &[]) {
            return Ok(None);
        }
    }
    if iid_static_size(m, source) != Some(n) {
        return Err(refuse(
            node,
            m,
            "tilt source must have the same static vector size as its base",
        ));
    }
    let inner = expect_builtin_call(m, source, "iid")
        .and_then(|c| c.args.first())
        .copied();
    if let Some(inner) = inner {
        if let Some((_, parameters)) = normal_ctor(m, inner)
            && let (Some(mu), Some(sigma)) = (
                find_kwarg(m, &parameters, "mu"),
                find_kwarg(m, &parameters, "sigma"),
            )
        {
            return Ok(Some(SetLaw::IidNormal(NormalOnSet {
                kind,
                n,
                mu,
                sigma,
                support,
            })));
        }
        let (inner, _) = resolve_ref_chain(m, inner);
        if matches!(kind, NormalSet::Ordered)
            && matches!(m.type_of(inner), Some(Type::Measure { domain, mass: Mass::Normalized })
                if matches!(domain.as_ref(), Type::Scalar(ScalarType::Real)))
            && superposition_reference(m, inner, &mut Vec::new()) == Some(Reference::Lebesgue)
        {
            return Ok(Some(SetLaw::OrderedIid {
                source,
                kernel: inner,
                n,
                support,
            }));
        }
    }
    if matches!(kind, NormalSet::ZeroSum)
        && let Some((mu, sigma)) = broadcast_normal(m, source)
    {
        return Ok(Some(SetLaw::ZeroSumNormals(NormalOnSet {
            kind,
            n,
            mu,
            sigma,
            support,
        })));
    }
    let law = inner
        .map(|inner| resolve_ref_chain(m, inner).0)
        .unwrap_or(source);
    let law = builtin_name(m, law).unwrap_or("source measure");
    let reason = match kind {
        NormalSet::Ordered => {
            format!("tilt on ordered requires iid of a continuous scalar law; {law} is unsupported")
        }
        NormalSet::ZeroSum => format!(
            "tilt on zerosum has no implemented closed-form sum law for {law}; independent Normals are supported"
        ),
    };
    Err(refuse(node, m, &reason))
}

fn broadcast_normal(m: &mut Module, source: NodeId) -> Option<(NodeId, NodeId)> {
    let call = expect_builtin_call(m, source, "broadcast")?.clone();
    let (&head, parameters) = call.args.split_first()?;
    let head = resolve_ref_chain(m, head).0;
    if !is_normal_const(m, head) {
        return None;
    }
    let names = [m.intern("mu"), m.intern("sigma")];
    let parameters = crate::kernel::bind_call_args(m, &names, parameters, &call.named, false)?;
    Some((parameters[0], parameters[1]))
}

pub(super) fn lower_density(
    m: &mut Module,
    node: NodeId,
    point: NodeId,
) -> Result<NodeId, RefuseError> {
    if let Some(law) = exponential_tilt::recognize(m, node) {
        return exponential_tilt::density(m, law, point);
    }
    if let Some(law) = on_set(m, node)? {
        return match law {
            SetLaw::IidNormal(normal) => Ok(normal_density(m, normal, point)),
            SetLaw::ZeroSumNormals(normal) => Ok(zerosum_density(m, normal, point)),
            SetLaw::OrderedIid {
                source, n, support, ..
            } => {
                let density = lower_measure_density(m, source, point)?;
                let n = m.alloc(Node::Lit(Scalar::Real(n as f64 + 1.0)));
                let constant = build_call(m, "loggamma", &[n]);
                let density = build_call(m, "add", &[density, constant]);
                let support = build_call(m, "in", &[point, support]);
                Ok(gate_density(m, support, density))
            }
        };
    }
    let call =
        expect_builtin_call(m, node, "tilt").ok_or_else(|| refuse(node, m, "expected tilt"))?;
    let [source, base] = *call.args else {
        return Err(refuse(node, m, "tilt expects two positional measures"));
    };
    let (base_node, _) = resolve_ref_chain(m, base);
    let Some(Type::Measure { domain, .. }) = m.type_of(base_node).cloned() else {
        return Err(refuse(
            node,
            m,
            "tilt requires closed measures at the query point",
        ));
    };
    let (label, reference, value) = broadcast::placeholder(m, *domain.clone());
    let density = build_call(m, "logdensityof", &[source, value]);
    let head = CallHead::Builtin(m.intern("functionof"));
    let weight = m.alloc(Node::Call(Call {
        head,
        args: vec![density].into(),
        named: vec![].into(),
        inputs: Some(Inputs::Spec(vec![(label, reference)].into())),
    }));
    let weighted = build_call(m, "logweighted", &[weight, base]);
    m.set_type(
        weighted,
        Type::Measure {
            domain,
            mass: Mass::Unknown,
        },
    );
    let normalized = build_call(m, "normalize", &[weighted]);
    lower_normalize(m, normalized, point)
}

fn normal_density(m: &mut Module, normal: NormalOnSet, point: NodeId) -> NodeId {
    let half = m.alloc(Node::Lit(Scalar::Real(0.5)));
    let log_tau = m.alloc(Node::Lit(Scalar::Real(std::f64::consts::TAU.ln() / 2.0)));
    let log_sigma = build_call(m, "log", &[normal.sigma]);
    let scale_constant = build_call(m, "add", &[log_sigma, log_tau]);
    let centered = match normal.kind {
        NormalSet::Ordered => elementwise(m, "sub", &[point, normal.mu]),
        NormalSet::ZeroSum => point,
    };
    let standard = elementwise(m, "divide", &[centered, normal.sigma]);
    let square = elementwise(m, "mul", &[standard, standard]);
    let sum = build_call(m, "sum", &[square]);
    let penalty = build_call(m, "mul", &[half, sum]);
    let (dimension, constant) = match normal.kind {
        NormalSet::Ordered => {
            let arg = m.alloc(Node::Lit(Scalar::Real(normal.n as f64 + 1.0)));
            (normal.n, build_call(m, "loggamma", &[arg]))
        }
        NormalSet::ZeroSum => {
            let constant = m.alloc(Node::Lit(Scalar::Real((normal.n as f64).ln() / 2.0)));
            (normal.n - 1, constant)
        }
    };
    let dimension = m.alloc(Node::Lit(Scalar::Real(dimension as f64)));
    let scaled = build_call(m, "mul", &[dimension, scale_constant]);
    let density = build_call(m, "sub", &[constant, scaled]);
    let density = build_call(m, "sub", &[density, penalty]);
    let support = build_call(m, "in", &[point, normal.support]);
    gate_density(m, support, density)
}

/// Complete the square about the conditional mean. This avoids cancellation
/// between the source log density and the log density of its sum at zero.
fn zerosum_density(m: &mut Module, normal: NormalOnSet, point: NodeId) -> NodeId {
    if normal.n == 1 {
        let zero = m.alloc(Node::Lit(Scalar::Real(0.0)));
        let support = build_call(m, "in", &[point, normal.support]);
        return gate_density(m, support, zero);
    }
    let (mu, sigma, variance, total) = normal_parameters(m, &normal);
    let center = project_sum(m, mu, variance, total);
    let delta = elementwise(m, "sub", &[point, center]);
    let standard = elementwise(m, "divide", &[delta, sigma]);
    let square = elementwise(m, "mul", &[standard, standard]);
    let sum = build_call(m, "sum", &[square]);
    let log_sigma = elementwise(m, "log", &[sigma]);
    let log_sigma = build_call(m, "sum", &[log_sigma]);
    let log_total = build_call(m, "log", &[total]);
    let half = m.alloc(Node::Lit(Scalar::Real(0.5)));
    let half_log_total = build_call(m, "mul", &[half, log_total]);
    let scale = build_call(m, "maximum", &[sigma]);
    let log_scale = build_call(m, "log", &[scale]);
    let normalizer = build_call(m, "add", &[log_scale, half_log_total]);
    let constant = m.alloc(Node::Lit(Scalar::Real(
        (normal.n as f64 - 1.0) * std::f64::consts::TAU.ln() / 2.0,
    )));
    let density = build_call(m, "sub", &[normalizer, log_sigma]);
    let density = build_call(m, "sub", &[density, constant]);
    let penalty = build_call(m, "mul", &[half, sum]);
    let density = build_call(m, "sub", &[density, penalty]);
    let support = build_call(m, "in", &[point, normal.support]);
    gate_density(m, support, density)
}

pub(crate) fn normal_parameters(
    m: &mut Module,
    normal: &NormalOnSet,
) -> (NodeId, NodeId, NodeId, NodeId) {
    let zero = m.alloc(Node::Lit(Scalar::Real(0.0)));
    let n = m.alloc(Node::Lit(Scalar::Int(normal.n as i64)));
    let zeros = build_call(m, "fill", &[zero, n]);
    let mu = elementwise(m, "add", &[normal.mu, zeros]);
    let sigma = elementwise(m, "add", &[normal.sigma, zeros]);
    // Only variance ratios enter the projection. Scale before squaring so
    // finite standard deviations need not have representable variances.
    let scale = build_call(m, "maximum", &[sigma]);
    let relative = elementwise(m, "divide", &[sigma, scale]);
    let variance = elementwise(m, "mul", &[relative, relative]);
    let total = build_call(m, "sum", &[variance]);
    (mu, sigma, variance, total)
}

pub(crate) fn project_sum(
    m: &mut Module,
    value: NodeId,
    variance: NodeId,
    total: NodeId,
) -> NodeId {
    let sum = build_call(m, "sum", &[value]);
    let ratio = build_call(m, "divide", &[sum, total]);
    let correction = elementwise(m, "mul", &[variance, ratio]);
    elementwise(m, "sub", &[value, correction])
}

pub(crate) fn elementwise(m: &mut Module, name: &str, args: &[NodeId]) -> NodeId {
    let head = m.intern(name);
    let head = m.alloc(Node::Const(head));
    let args: Vec<_> = std::iter::once(head).chain(args.iter().copied()).collect();
    build_call(m, "broadcast", &args)
}

/// Apply both kernels by name before normalizing. A closed operand ignores the
/// input; each reification retains its own boundary and captured ancestry.
pub(super) fn lower_application(
    m: &mut Module,
    node: NodeId,
    point: NodeId,
) -> Option<Result<NodeId, RefuseError>> {
    let Node::Call(application) = m.node(node).clone() else {
        return None;
    };
    let CallHead::User(callee) = application.head else {
        return None;
    };
    let (callee, _) = resolve_ref_chain(m, callee);
    let tilt = expect_builtin_call(m, callee, "tilt")?.clone();
    Some((|| {
        let [source, base] = *tilt.args else {
            return Err(refuse(callee, m, "tilt expects two operands"));
        };
        let Some(Type::Kernel { inputs, .. }) = m.type_of(callee).cloned() else {
            return Err(refuse(callee, m, "tilt has no resolved kernel interface"));
        };
        let keyword_only = [source, base]
            .into_iter()
            .find_map(|operand| {
                crate::kernel::resolve_reified(m, operand).map(|kernel| kernel.auto)
            })
            .unwrap_or(false);
        let values = crate::kernel::bind_call_args(
            m,
            &inputs,
            &application.args,
            &application.named,
            keyword_only,
        )
        .ok_or_else(|| refuse(node, m, if keyword_only {
            "tilt application must bind each input once by name; its automatic interface is keyword-only"
        } else {
            "tilt application must bind each input once"
        }))?;
        let supplied: Vec<_> = inputs.iter().copied().zip(values).collect();
        let source = apply_operand(m, source, &supplied, point)?;
        let base = apply_operand(m, base, &supplied, point)?;
        let closed = build_call(m, "tilt", &[source, base]);
        lower_density(m, closed, point)
    })())
}

fn apply_operand(
    m: &mut Module,
    operand: NodeId,
    supplied: &[(Symbol, NodeId)],
    point: NodeId,
) -> Result<NodeId, RefuseError> {
    let (resolved, _) = resolve_ref_chain(m, operand);
    if matches!(m.type_of(resolved), Some(Type::Measure { .. })) {
        return Ok(operand);
    }
    let kernel = crate::kernel::resolve_reified(m, operand).ok_or_else(|| {
        refuse(
            operand,
            m,
            "tilt kernel operand is not a supported reification",
        )
    })?;
    let bound = kernel
        .inputs
        .iter()
        .map(|(name, target)| {
            supplied
                .iter()
                .find(|(input, _)| input == name)
                .map(|(_, value)| (*target, *value))
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            refuse(
                operand,
                m,
                "tilt kernels must share their named input interface",
            )
        })?;
    refuse_boundary_named_query_point(m, point, &bound)?;
    let absorbed = crate::kernel::resolve_kernel(m, operand).is_some();
    let body = if absorbed {
        let body = crate::kernel::declared_body(m, kernel.body);
        draw_argument(m, body).unwrap_or(body)
    } else {
        kernel.body
    };
    let result = crate::kernel::substitute_admitted(
        m,
        body,
        &bound,
        crate::kernel::Substitute::All,
        absorbed,
    );
    let names: Vec<_> = bound
        .iter()
        .map(|(reference, value)| (reference.name, *value))
        .collect();
    carry_types(m, body, result, &names);
    if absorbed
        && (!matches!(m.type_of(result), Some(Type::Measure { .. }))
            || measure_reaches_draw(m, result, &[]))
    {
        let law = build_call(m, "lawof", &[result]);
        if let Some(ty) = m.type_of(result).cloned() {
            m.set_type(law, ty);
        }
        Ok(law)
    } else {
        Ok(result)
    }
}
