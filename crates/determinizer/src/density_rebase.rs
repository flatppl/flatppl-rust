//! Normalization on an explicit base. Exact vector Normal rules precede the
//! ordinary normalize/logweighted path; unsupported masses retain its refusal.

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

pub(crate) fn normal_on_set(m: &mut Module, node: NodeId) -> Option<NormalOnSet> {
    let call = expect_builtin_call(m, node, "rebase")?;
    let [source, base] = *call.args else {
        return None;
    };
    let (mut source, _) = resolve_ref_chain(m, source);
    if let Some(law) = expect_builtin_call(m, source, "lawof") {
        source = resolve_ref_chain(m, *law.args.first()?).0;
        if let Some(measure) = draw_argument(m, source) {
            source = resolve_ref_chain(m, measure).0;
        }
        if measure_reaches_draw(m, source, &[]) {
            return None;
        }
    }
    let iid = expect_builtin_call(m, source, "iid")?;
    let inner = *iid.args.first()?;
    let n = iid_static_size(m, source)?;
    if n == 0 {
        return None;
    }
    let (base, _) = resolve_ref_chain(m, base);
    let lebesgue = expect_builtin_call(m, base, "Lebesgue")?;
    let support = named_or_positional(m, lebesgue, "support")?;
    let (set, _) = resolve_ref_chain(m, support);
    let kind = match builtin_name(m, set)? {
        "ordered" => NormalSet::Ordered,
        "zerosum" => NormalSet::ZeroSum,
        _ => return None,
    };
    let Type::Measure { domain, .. } = m.type_of(base)? else {
        return None;
    };
    if !matches!(domain.as_ref(), Type::Array { shape, .. }
        if shape.as_ref() == [Dim::Static(n as u32)])
    {
        return None;
    }
    let (_, parameters) = normal_ctor(m, inner)?;
    Some(NormalOnSet {
        kind,
        n,
        mu: find_kwarg(m, &parameters, "mu")?,
        sigma: find_kwarg(m, &parameters, "sigma")?,
        support,
    })
}

pub(super) fn lower_density(
    m: &mut Module,
    node: NodeId,
    point: NodeId,
) -> Result<NodeId, RefuseError> {
    if let Some(normal) = normal_on_set(m, node) {
        return Ok(normal_density(m, normal, point));
    }
    let call =
        expect_builtin_call(m, node, "rebase").ok_or_else(|| refuse(node, m, "expected rebase"))?;
    let [source, base] = *call.args else {
        return Err(refuse(node, m, "rebase expects two positional measures"));
    };
    let (base_node, _) = resolve_ref_chain(m, base);
    let Some(Type::Measure { domain, .. }) = m.type_of(base_node).cloned() else {
        return Err(refuse(
            node,
            m,
            "rebase requires closed measures at the query point",
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
    let rebase = expect_builtin_call(m, callee, "rebase")?.clone();
    Some((|| {
        let [source, base] = *rebase.args else {
            return Err(refuse(callee, m, "rebase expects two operands"));
        };
        let Some(Type::Kernel { inputs, .. }) = m.type_of(callee).cloned() else {
            return Err(refuse(callee, m, "rebase has no resolved kernel interface"));
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
            "rebase application must bind each input once by name; its automatic interface is keyword-only"
        } else {
            "rebase application must bind each input once"
        }))?;
        let supplied: Vec<_> = inputs.iter().copied().zip(values).collect();
        let source = apply_operand(m, source, &supplied, point)?;
        let base = apply_operand(m, base, &supplied, point)?;
        let closed = build_call(m, "rebase", &[source, base]);
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
            "rebase kernel operand is not a supported reification",
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
                "rebase kernels must share their named input interface",
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
