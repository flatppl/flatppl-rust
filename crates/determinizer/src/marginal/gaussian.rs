//! Exact affine Gaussian integration with the original covariance checks.

use super::{
    LogDensity, MarginalForm, RefuseError, build_constructor, find_kwarg, is_input_ref, real,
};
use crate::density::{
    build_call, build_record, draw_argument, measure_stochastic_ancestors, split_kernel_constructor,
};
use flatppl_core::{
    CallHead, Dim, Module, Node, NodeId, Ref, RefNs, Scalar, ScalarType, Symbol, Type,
};
use std::collections::HashSet;

fn shape(m: &Module, node: NodeId) -> Option<Vec<u32>> {
    let Type::Array { shape, elem } = m.type_of(node)? else {
        return None;
    };
    if !matches!(
        elem.as_ref(),
        Type::Scalar(ScalarType::Real | ScalarType::Integer)
    ) {
        return None;
    }
    shape
        .iter()
        .map(|dim| match dim {
            Dim::Static(n) if *n > 0 => Some(*n),
            _ => None,
        })
        .collect()
}

/// Resolve aliases without crossing the current query's given inputs.
fn expression(m: &Module, mut node: NodeId, latent: Symbol) -> Option<NodeId> {
    let mut seen = HashSet::new();
    while let Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }) = m.node(node)
    {
        if *name == latent || m.is_conditioned_input(*name) {
            break;
        }
        if !seen.insert(node) {
            return None;
        }
        let Some(binding) = m.binding_by_name(*name) else {
            break;
        };
        node = m.declared_binding_rhs(binding);
    }
    Some(node)
}

/// Follow binding aliases too. Unknown callable scope cannot prove independence.
fn independent(m: &Module, node: NodeId, latent: Symbol) -> bool {
    let mut stack = vec![node];
    let mut seen = HashSet::new();
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        match m.node(node) {
            Node::Ref(Ref { name, .. }) if *name == latent => return false,
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) => {
                if !m.is_conditioned_input(*name) {
                    let Some(binding) = m.binding_by_name(*name) else {
                        return false;
                    };
                    stack.push(m.declared_binding_rhs(binding));
                }
            }
            Node::Ref(_) | Node::Hole => return false,
            Node::Call(call) if call.inputs.is_some() || matches!(call.head, CallHead::User(_)) => {
                return false;
            }
            _ => stack.extend(m.node(node).children()),
        }
    }
    true
}

fn product(m: &Module, node: NodeId, latent: Symbol) -> Option<NodeId> {
    let node = expression(m, node, latent)?;
    let Node::Call(call) = m.node(node) else {
        return None;
    };
    let CallHead::Builtin(head) = call.head else {
        return None;
    };
    if m.resolve(head) != "mul"
        || call.args.len() != 2
        || !call.named.is_empty()
        || call.inputs.is_some()
    {
        return None;
    }
    let rhs = expression(m, call.args[1], latent)?;
    is_input_ref(m, rhs, latent).then_some(call.args[0])
}

fn affine(m: &Module, node: NodeId, latent: Symbol) -> Option<(NodeId, Option<NodeId>)> {
    if let Some(matrix) = product(m, node, latent) {
        return Some((matrix, None));
    }
    let node = expression(m, node, latent)?;
    let Node::Call(call) = m.node(node) else {
        return None;
    };
    let CallHead::Builtin(head) = call.head else {
        return None;
    };
    if !call.named.is_empty() || call.inputs.is_some() {
        return None;
    }
    let args = match m.resolve(head) {
        "add" if call.args.len() == 2 => &call.args[..],
        "broadcast"
            if call.args.len() == 3
                && matches!(m.node(call.args[0]), Node::Const(name) if m.resolve(*name) == "add") =>
        {
            &call.args[1..]
        }
        _ => return None,
    };
    for (linear, offset) in [(args[0], args[1]), (args[1], args[0])] {
        if let Some(matrix) = product(m, linear, latent) {
            return Some((matrix, Some(offset)));
        }
    }
    None
}

pub(super) fn marginal(
    m: &mut Module,
    latent: Symbol,
    prior: &[(Symbol, NodeId)],
    likelihood: &[(Symbol, NodeId)],
) -> Option<Result<MarginalForm, RefuseError>> {
    let params = parameters(m, latent, prior, likelihood)?;
    Some(Ok(MarginalForm::LogDensity(LogDensity {
        build: logpdf,
        params: params.to_vec(),
    })))
}

fn parameters(
    m: &mut Module,
    latent: Symbol,
    prior: &[(Symbol, NodeId)],
    likelihood: &[(Symbol, NodeId)],
) -> Option<[NodeId; 5]> {
    let mu = find_kwarg(m, prior, "mu")?;
    let covariance = find_kwarg(m, prior, "cov")?;
    let noise = find_kwarg(m, likelihood, "cov")?;
    let likelihood_mean = find_kwarg(m, likelihood, "mu")?;
    let (matrix, offset) = affine(m, likelihood_mean, latent)?;
    let [n]: [u32; 1] = shape(m, mu)?.try_into().ok()?;
    let [rows, cols]: [u32; 2] = shape(m, matrix)?.try_into().ok()?;
    if cols != n
        || shape(m, likelihood_mean)? != [rows]
        || shape(m, covariance)? != [n, n]
        || shape(m, noise)? != [rows, rows]
        || offset.is_some_and(|offset| shape(m, offset) != Some(vec![rows]))
        || [mu, covariance, matrix, noise]
            .into_iter()
            .chain(offset)
            .any(|node| !independent(m, node, latent))
    {
        return None;
    }
    let offset = offset.unwrap_or_else(|| {
        let zero = real(m, 0.0);
        let size = m.alloc(Node::Lit(Scalar::Int(i64::from(rows))));
        build_call(m, "fill", &[zero, size])
    });
    Some([mu, covariance, matrix, offset, noise])
}

fn native_score(m: &mut Module, mean: NodeId, cov: NodeId, value: NodeId) -> NodeId {
    let mu = m.intern("mu");
    let covariance = m.intern("cov");
    let params = build_record(m, &[(mu, mean), (covariance, cov)]);
    let kernel = m.intern("MvNormal");
    let kernel = m.alloc(Node::Const(kernel));
    build_call(m, "builtin_logdensityof", &[kernel, params, value])
}

fn real_array(m: &mut Module, value: NodeId) -> NodeId {
    let head = m.intern("real");
    let head = m.alloc(Node::Const(head));
    build_call(m, "broadcast", &[head, value])
}

fn component_checks(
    m: &mut Module,
    mu: NodeId,
    covariance: NodeId,
    offset: NodeId,
    noise: NodeId,
) -> (NodeId, NodeId) {
    let prior_check = native_score(m, mu, covariance, mu);
    let noise_check = native_score(m, offset, noise, offset);
    let valid_prior = build_call(m, "isfinite", &[prior_check]);
    let valid_noise = build_call(m, "isfinite", &[noise_check]);
    let valid = build_call(m, "land", &[valid_prior, valid_noise]);
    let invalid = build_call(m, "add", &[prior_check, noise_check]);
    (valid, invalid)
}

fn logpdf(m: &mut Module, params: &[NodeId], variates: &[NodeId]) -> NodeId {
    let [mu, covariance, matrix, offset, noise] =
        [params[0], params[1], params[2], params[3], params[4]];
    // The original A*z is real because z is Gaussian. Preserve that arithmetic
    // even when the authored means, matrices, and covariances are integers.
    let [mu, covariance, matrix, offset, noise] =
        [mu, covariance, matrix, offset, noise].map(|value| real_array(m, value));
    let mean = build_call(m, "mul", &[matrix, mu]);
    let mean = build_call(m, "add", &[mean, offset]);
    let ac = build_call(m, "mul", &[matrix, covariance]);
    let transpose = build_call(m, "transpose", &[matrix]);
    let propagated = build_call(m, "mul", &[ac, transpose]);
    let marginal_cov = build_call(m, "add", &[propagated, noise]);
    let score = native_score(m, mean, marginal_cov, super::sole_variate(variates));
    // Valid marginal covariance must not mask an invalid component covariance.
    let (valid, invalid) = component_checks(m, mu, covariance, offset, noise);
    build_call(m, "ifelse", &[valid, score, invalid])
}

fn declared_draw(m: &Module, mut value: NodeId) -> Option<(Symbol, NodeId)> {
    let mut seen = HashSet::new();
    loop {
        let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = m.node(value)
        else {
            return None;
        };
        if m.is_conditioned_input(*name) || !seen.insert(value) {
            return None;
        }
        let binding = m.binding_by_name(*name)?;
        value = m.declared_binding_rhs(binding);
        if let Some(prior) = draw_argument(m, value) {
            return Some((*name, prior));
        }
    }
}

/// Exact reverse disintegration of one affine Gaussian edge. The caller keeps
/// the selected record label and exposes the observed field as a kernel input.
pub(crate) fn conditional_draw(
    m: &mut Module,
    selected: &[(Symbol, NodeId)],
    complement: &[(Symbol, NodeId)],
) -> Option<NodeId> {
    let [(_, latent_value)] = selected else {
        return None;
    };
    let [(_, observation)] = complement else {
        return None;
    };
    let (latent, prior) = declared_draw(m, *latent_value)?;
    let (_, likelihood) = declared_draw(m, *observation)?;
    let (ancestor, ancestor_prior) = super::sole_named_ancestor(m, likelihood, &[])?;
    if ancestor != latent
        || ancestor_prior != prior
        || !measure_stochastic_ancestors(m, prior, &[]).is_empty()
    {
        return None;
    }
    let (prior_family, prior_kwargs) = split_kernel_constructor(m, prior)?;
    let (lik_family, lik_kwargs) = split_kernel_constructor(m, likelihood)?;
    if m.resolve(prior_family) != "MvNormal" || m.resolve(lik_family) != "MvNormal" {
        return None;
    }
    let params = parameters(m, latent, &prior_kwargs, &lik_kwargs)?;
    let [n]: [u32; 1] = shape(m, params[0])?.try_into().ok()?;
    // One vector solve per latent axis, with O(n^2) literal basis entries.
    // Keep the generated graph bounded until matrix RHS solves are portable.
    if n > 64 {
        return None;
    }
    let [mu, covariance, matrix, offset, noise] = params.map(|value| real_array(m, value));
    let observed = real_array(m, *observation);
    let mean = build_call(m, "mul", &[matrix, mu]);
    let mean = build_call(m, "add", &[mean, offset]);
    let ac = build_call(m, "mul", &[matrix, covariance]);
    let at = build_call(m, "transpose", &[matrix]);
    let propagated = build_call(m, "mul", &[ac, at]);
    let marginal_cov = build_call(m, "add", &[propagated, noise]);
    let zero = real(m, 0.0);
    let one = real(m, 1.0);
    let mut basis = Vec::new();
    let mut gain_rows = Vec::new();
    for i in 0..n {
        let entries: Vec<_> = (0..n).map(|j| if i == j { one } else { zero }).collect();
        let axis = build_call(m, "vector", &entries);
        basis.push(axis);
        let column = build_call(m, "mul", &[ac, axis]);
        gain_rows.push(build_call(m, "linsolve", &[marginal_cov, column]));
    }
    let rows = build_call(m, "vector", &gain_rows);
    let gain = build_call(m, "rowstack", &[rows]);
    let residual = build_call(m, "sub", &[observed, mean]);
    let adjustment = build_call(m, "mul", &[gain, residual]);
    let posterior_mu = build_call(m, "add", &[mu, adjustment]);
    let basis = build_call(m, "vector", &basis);
    let identity = build_call(m, "rowstack", &[basis]);
    let ka = build_call(m, "mul", &[gain, matrix]);
    let b = build_call(m, "sub", &[identity, ka]);
    let bc = build_call(m, "mul", &[b, covariance]);
    let bt = build_call(m, "transpose", &[b]);
    let propagated = build_call(m, "mul", &[bc, bt]);
    let kr = build_call(m, "mul", &[gain, noise]);
    let kt = build_call(m, "transpose", &[gain]);
    let measurement = build_call(m, "mul", &[kr, kt]);
    let posterior_cov = build_call(m, "add", &[propagated, measurement]);
    let (valid, _) = component_checks(m, mu, covariance, offset, noise);
    let invalid = build_call(m, "neg", &[identity]);
    let posterior_cov = build_call(m, "ifelse", &[valid, posterior_cov, invalid]);
    let measure = build_constructor(
        m,
        "MvNormal",
        &[("mu", posterior_mu), ("cov", posterior_cov)],
    );
    Some(build_call(m, "draw", &[measure]))
}
