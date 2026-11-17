//! Exact shared simplex-probability integrals with native support checks.

use super::{
    LogDensity, MarginalForm, RefuseError, SharedLatentRecordLaw, build_constructor, find_kwarg,
    is_input_ref, real, references_latent, split_kernel_constructor,
};
use crate::density::{build_call, build_record};
use flatppl_core::{Dim, Module, Node, NodeId, Scalar, ScalarType, Symbol, Type};

// Category labels must stay exact in both supported real precisions.
const MAX_CATEGORIES: u32 = 1 << 24;

fn dimension(m: &Module, alpha: NodeId) -> Option<u32> {
    let Type::Array { shape, elem } = m.type_of(alpha)? else {
        return None;
    };
    let [Dim::Static(n)] = shape.as_ref() else {
        return None;
    };
    (matches!(
        elem.as_ref(),
        Type::Scalar(ScalarType::Real | ScalarType::Integer)
    ) && *n > 0
        && *n <= MAX_CATEGORIES)
        .then_some(*n)
}

fn map(m: &mut Module, name: &str, args: &[NodeId]) -> NodeId {
    let head = m.intern(name);
    let head = m.alloc(Node::Const(head));
    let mut all = vec![head];
    all.extend_from_slice(args);
    build_call(m, "broadcast", &all)
}

fn filled(m: &mut Module, value: f64, size: NodeId) -> NodeId {
    let value = real(m, value);
    build_call(m, "fill", &[value, size])
}

fn valid_shapes(m: &mut Module, alpha: NodeId) -> NodeId {
    let zero = real(m, 0.0);
    let positive = map(m, "gt", &[alpha, zero]);
    let finite = map(m, "isfinite", &[alpha]);
    let valid = map(m, "land", &[positive, finite]);
    build_call(m, "lall", &[valid])
}

pub(super) fn marginal(
    m: &mut Module,
    latent: Symbol,
    prior: &[(Symbol, NodeId)],
    likelihood: &[(Symbol, NodeId)],
    family: &str,
    exempt: &[NodeId],
) -> Option<Result<MarginalForm, RefuseError>> {
    if !matches!(family, "Categorical" | "Categorical0" | "Multinomial")
        || !is_input_ref(m, find_kwarg(m, likelihood, "p")?, latent)
        || likelihood.iter().any(|(name, value)| {
            m.resolve(*name) != "p" && references_latent(m, *value, latent, exempt)
        })
    {
        return None;
    }
    let alpha = find_kwarg(m, prior, "alpha")?;
    let d = dimension(m, alpha)?;
    let size = m.alloc(Node::Lit(Scalar::Int(i64::from(d))));
    if family == "Multinomial" {
        let n = find_kwarg(m, likelihood, "n")?;
        if !matches!(
            m.type_of(n),
            Some(Type::Scalar(ScalarType::Integer | ScalarType::Real))
        ) {
            return None;
        }
        let head = m.intern(family);
        let head = m.alloc(Node::Const(head));
        return Some(Ok(MarginalForm::LogDensity(LogDensity {
            build: shared_logpmf,
            params: vec![alpha, size, head, n],
        })));
    }
    let valid = valid_shapes(m, alpha);
    let ones = filled(m, 1.0, size);
    let safe = build_call(m, "ifelse", &[valid, alpha, ones]);
    let scale = build_call(m, "maximum", &[safe]);
    let scaled = map(m, "divide", &[safe, scale]);
    let total = build_call(m, "sum", &[scaled]);
    let probability = map(m, "divide", &[scaled, total]);
    let invalid = filled(m, -1.0, size);
    let probability = build_call(m, "ifelse", &[valid, probability, invalid]);
    Some(Ok(MarginalForm::Measure(build_constructor(
        m,
        family,
        &[("p", probability)],
    ))))
}

pub(super) fn shared_law(
    m: &mut Module,
    latent: Symbol,
    prior: &[(Symbol, NodeId)],
    measures: &[NodeId],
    exempt: &[NodeId],
) -> Option<SharedLatentRecordLaw> {
    let alpha = find_kwarg(m, prior, "alpha")?;
    let d = dimension(m, alpha)?;
    let size = m.alloc(Node::Lit(Scalar::Int(i64::from(d))));
    let mut params = vec![alpha, size];
    for &measure in measures {
        let (family, kwargs) = split_kernel_constructor(m, measure)?;
        let n = match m.resolve(family) {
            "Categorical" | "Categorical0" => m.alloc(Node::Lit(Scalar::Int(1))),
            "Multinomial" => {
                let n = find_kwarg(m, &kwargs, "n")?;
                if references_latent(m, n, latent, exempt)
                    || !matches!(
                        m.type_of(n),
                        Some(Type::Scalar(ScalarType::Integer | ScalarType::Real))
                    )
                {
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

fn shared_logpmf(m: &mut Module, params: &[NodeId], variates: &[NodeId]) -> NodeId {
    debug_assert_eq!(params.len(), 2 + 2 * variates.len());
    let [alpha, size] = [params[0], params[1]];
    let Node::Lit(Scalar::Int(d)) = m.node(size) else {
        unreachable!()
    };
    let d = *d;
    let uniform = filled(m, 1.0 / d as f64, size);
    let zero = real(m, 0.0);
    let zeros = filled(m, 0.0, size);
    let ones = filled(m, 1.0, size);
    let valid = valid_shapes(m, alpha);
    let safe_alpha = build_call(m, "ifelse", &[valid, alpha, ones]);
    let invalid = filled(m, -1.0, size);
    let probability = build_call(m, "ifelse", &[valid, uniform, invalid]);
    let p_name = m.intern("p");
    let n_name = m.intern("n");
    let mut anchors = Vec::new();
    let mut counts = Vec::new();
    for (&[kernel, n], &value) in params[2..].as_chunks::<2>().0.iter().zip(variates) {
        let Node::Const(name) = m.node(kernel) else {
            unreachable!()
        };
        let family = m.resolve(*name).to_owned();
        let record = if family == "Multinomial" {
            build_record(m, &[(n_name, n), (p_name, probability)])
        } else {
            build_record(m, &[(p_name, probability)])
        };
        anchors.push(build_call(
            m,
            "builtin_logdensityof",
            &[kernel, record, value],
        ));
        let count = if family == "Multinomial" {
            let raw = map(m, "real", &[value]);
            let finite = map(m, "isfinite", &[raw]);
            let finite = build_call(m, "lall", &[finite]);
            let positive = map(m, "ge", &[raw, zero]);
            let positive = build_call(m, "lall", &[positive]);
            // The native anchor owns exact integer totals. A real sum can
            // round differently from real(n), even for supported counts.
            let supported = build_call(m, "land", &[finite, positive]);
            build_call(m, "ifelse", &[supported, raw, zeros])
        } else {
            let offset = i64::from(family == "Categorical");
            let first = real(m, offset as f64);
            let last = real(m, (d - 1 + offset) as f64);
            let labels = build_call(m, "linspace", &[first, last, size]);
            let point = build_call(m, "real", &[value]);
            let difference = map(m, "sub", &[labels, point]);
            let same = map(m, "iszero", &[difference]);
            map(m, "real", &[same])
        };
        counts.push(count);
    }
    let count = crate::density::fold_add(m, &counts);
    let post = build_call(m, "add", &[safe_alpha, count]);
    let alpha_name = m.intern("alpha");
    let prior = build_record(m, &[(alpha_name, alpha)]);
    let posterior = build_record(m, &[(alpha_name, post)]);
    let head = m.intern("Dirichlet");
    let kernel = m.alloc(Node::Const(head));
    let prior = build_call(m, "builtin_logdensityof", &[kernel, prior, uniform]);
    let posterior = build_call(m, "builtin_logdensityof", &[kernel, posterior, uniform]);
    let ratio = build_call(m, "sub", &[prior, posterior]);
    let anchors = crate::density::fold_add(m, &anchors);
    build_call(m, "add", &[anchors, ratio])
}
