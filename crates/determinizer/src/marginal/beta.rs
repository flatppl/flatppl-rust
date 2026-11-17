//! Shared Beta probability integrals, retaining ordered observation coefficients.

use super::{
    LogDensity, MarginalForm, SharedLatentRecordLaw, build_constructor, find_kwarg, is_input_ref,
    real, references_latent, split_kernel_constructor,
};
use crate::density::{build_call, build_record};
use flatppl_core::{Module, Node, NodeId, Scalar, Symbol};

fn positive_finite(m: &mut Module, x: NodeId) -> NodeId {
    let zero = real(m, 0.0);
    let positive = build_call(m, "gt", &[x, zero]);
    let finite = build_call(m, "isfinite", &[x]);
    build_call(m, "land", &[positive, finite])
}

fn valid_shapes(m: &mut Module, alpha: NodeId, beta: NodeId) -> NodeId {
    let a = positive_finite(m, alpha);
    let b = positive_finite(m, beta);
    build_call(m, "land", &[a, b])
}

fn safe_shape(m: &mut Module, valid: NodeId, shape: NodeId) -> NodeId {
    let one = real(m, 1.0);
    build_call(m, "ifelse", &[valid, shape, one])
}

pub(super) fn bernoulli_marginal(
    m: &mut Module,
    prior: &[(Symbol, NodeId)],
    _: &[(Symbol, NodeId)],
) -> Option<MarginalForm> {
    let alpha = find_kwarg(m, prior, "alpha")?;
    let beta = find_kwarg(m, prior, "beta")?;
    let valid = valid_shapes(m, alpha, beta);
    let alpha = safe_shape(m, valid, alpha);
    let beta = safe_shape(m, valid, beta);
    // Scaling preserves the mean when alpha + beta would overflow.
    let scale = build_call(m, "max", &[alpha, beta]);
    let a = build_call(m, "divide", &[alpha, scale]);
    let b = build_call(m, "divide", &[beta, scale]);
    let sum = build_call(m, "add", &[a, b]);
    let mean = build_call(m, "divide", &[a, sum]);
    // Invalid prior shapes must remain a native parameter-domain failure.
    let invalid = real(m, -1.0);
    let p = build_call(m, "ifelse", &[valid, mean, invalid]);
    Some(MarginalForm::Measure(build_constructor(
        m,
        "Bernoulli",
        &[("p", p)],
    )))
}

pub(super) fn shared_law(
    m: &mut Module,
    latent: Symbol,
    prior: &[(Symbol, NodeId)],
    measures: &[NodeId],
    exempt: &[NodeId],
) -> Option<SharedLatentRecordLaw> {
    let mut params = vec![
        find_kwarg(m, prior, "alpha")?,
        find_kwarg(m, prior, "beta")?,
    ];
    for &measure in measures {
        let (family, kwargs) = split_kernel_constructor(m, measure)?;
        let n = match m.resolve(family) {
            "Bernoulli" => m.alloc(Node::Lit(Scalar::Int(1))),
            "Binomial" => {
                let n = find_kwarg(m, &kwargs, "n")?;
                if references_latent(m, n, latent, exempt) {
                    return None;
                }
                n
            }
            _ => return None,
        };
        if !is_input_ref(m, find_kwarg(m, &kwargs, "p")?, latent) {
            return None;
        }
        params.push(m.alloc(Node::Const(family)));
        params.push(n);
    }
    Some(SharedLatentRecordLaw {
        latent,
        field_count: measures.len(),
        form: LogDensity {
            build: shared_logpmf,
            params,
        },
    })
}

/// Each native half-probability anchor retains its own support and coefficient.
/// The one beta ratio integrates the shared probability jointly, not per field.
fn shared_logpmf(m: &mut Module, params: &[NodeId], variates: &[NodeId]) -> NodeId {
    debug_assert_eq!(params.len(), 2 + 2 * variates.len());
    let [alpha, beta] = [params[0], params[1]];
    let valid = valid_shapes(m, alpha, beta);
    let half = real(m, 0.5);
    let invalid = real(m, -1.0);
    let p = build_call(m, "ifelse", &[valid, half, invalid]);
    let alpha = safe_shape(m, valid, alpha);
    let beta = safe_shape(m, valid, beta);
    let zero = real(m, 0.0);
    let mut anchors = Vec::new();
    let mut successes = Vec::new();
    let mut trials = Vec::new();
    let p_name = m.intern("p");
    let n_name = m.intern("n");
    for (&[kernel, n], &value) in params[2..].as_chunks::<2>().0.iter().zip(variates) {
        let bernoulli =
            matches!(m.node(kernel), Node::Const(family) if m.resolve(*family) == "Bernoulli");
        let record = if bernoulli {
            build_record(m, &[(p_name, p)])
        } else {
            build_record(m, &[(n_name, n), (p_name, p)])
        };
        anchors.push(build_call(
            m,
            "builtin_logdensityof",
            &[kernel, record, value],
        ));
        let k = build_call(m, "real", &[value]);
        let n = build_call(m, "real", &[n]);
        let finite_k = build_call(m, "isfinite", &[k]);
        let finite_n = build_call(m, "isfinite", &[n]);
        let k = build_call(m, "ifelse", &[finite_k, k, zero]);
        let n = build_call(m, "ifelse", &[finite_n, n, zero]);
        let nonnegative = build_call(m, "ge", &[k, zero]);
        let within = build_call(m, "le", &[k, n]);
        let mut count_valid = build_call(m, "land", &[finite_k, finite_n]);
        for condition in [nonnegative, within] {
            count_valid = build_call(m, "land", &[count_valid, condition]);
        }
        successes.push(build_call(m, "ifelse", &[count_valid, k, zero]));
        trials.push(build_call(m, "ifelse", &[count_valid, n, zero]));
    }
    let k = crate::density::fold_add(m, &successes);
    let n = crate::density::fold_add(m, &trials);
    let failures = build_call(m, "sub", &[n, k]);
    let post_a = build_call(m, "add", &[alpha, k]);
    let post_b = build_call(m, "add", &[beta, failures]);
    // At one half, log f_prior - log f_posterior is the beta ratio
    // plus N*log(2). Native Beta evaluation reuses stable log-beta and
    // retains the original prior's parameter checks.
    let alpha_name = m.intern("alpha");
    let beta_name = m.intern("beta");
    let prior = build_record(m, &[(alpha_name, params[0]), (beta_name, params[1])]);
    let posterior = build_record(m, &[(alpha_name, post_a), (beta_name, post_b)]);
    let beta_name = m.intern("Beta");
    let kernel = m.alloc(Node::Const(beta_name));
    let prior = build_call(m, "builtin_logdensityof", &[kernel, prior, half]);
    let posterior = build_call(m, "builtin_logdensityof", &[kernel, posterior, half]);
    let ratio = build_call(m, "sub", &[prior, posterior]);
    let anchors = crate::density::fold_add(m, &anchors);
    build_call(m, "add", &[anchors, ratio])
}
