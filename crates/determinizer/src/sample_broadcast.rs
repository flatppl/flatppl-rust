//! Sample one independent product of scalar row laws, retaining the row axes.

use super::*;
use crate::density::broadcast::{cell_input, map_body};
use flatppl_core::{Call, Dim};

pub(super) fn lower(
    m: &mut Module,
    node: NodeId,
    rng: NodeId,
    sample_name: Option<Symbol>,
) -> Result<(NodeId, NodeId), RefuseError> {
    let call = expect_builtin_call(m, node, "broadcast").unwrap().clone();
    let Some((&head, args)) = call.args.split_first() else {
        return Err(refuse(node, m, "broadcast sample requires a kernel"));
    };
    let mut inputs = Vec::new();
    let mut actuals = Vec::new();
    let mut rank = None;
    let mut bind = |m: &mut Module, value| {
        let resolved = crate::density::resolve_ref_chain(m, value).0;
        let ty = m
            .type_of(value)
            .or_else(|| m.type_of(resolved))
            .cloned()
            .or_else(|| {
                flatppl_infer::infer_expression(m, value, &[])
                    .0
                    .remove(&value)
            });
        if let Some(ty) = &ty {
            m.set_type(value, ty.clone());
        }
        match ty {
            Some(Type::Array { shape, elem }) => {
                if !matches!(*elem, Type::Scalar(_))
                    || shape.iter().any(|d| !matches!(d, Dim::Static(_)))
                    || rank.is_some_and(|r| r != shape.len())
                {
                    return Err(refuse(
                        value,
                        m,
                        "broadcast sampling requires statically shaped arrays of scalar cells",
                    ));
                }
                rank = Some(shape.len());
                cell_input(m, value, &mut inputs, &mut actuals)
            }
            Some(Type::Record(_) | Type::Table { .. } | Type::Tuple(_)) => {
                Err(refuse(value, m, "broadcast sampling requires scalar cells"))
            }
            _ => Ok(value),
        }
    };
    let args = args
        .iter()
        .map(|&arg| bind(m, arg))
        .collect::<Result<Vec<_>, _>>()?;
    let mut named = call.named.to_vec();
    for arg in &mut named {
        arg.value = bind(m, arg.value)?;
    }
    let head = match m.node(head) {
        Node::Const(name) => CallHead::Builtin(*name),
        _ => CallHead::User(head),
    };
    let measure = m.alloc(Node::Call(Call {
        head,
        args: args.into(),
        named: named.into(),
        inputs: None,
    }));
    if let Some(rank) = rank {
        Scope {
            inputs,
            actuals,
            rank,
        }
        .sample(m, measure, rng, sample_name)
    } else {
        let draw = build_call(m, "draw", &[measure]);
        lower_draw(m, draw, rng, sample_name)
    }
}

struct Scope {
    inputs: Vec<(Symbol, Ref)>,
    actuals: Vec<NodeId>,
    rank: usize,
}

impl Scope {
    fn map(&self, m: &mut Module, body: NodeId) -> NodeId {
        map_body(
            m,
            body,
            self.inputs.clone(),
            self.actuals.clone(),
            self.rank,
        )
    }

    fn sample(
        &self,
        m: &mut Module,
        measure: NodeId,
        rng: NodeId,
        sample_name: Option<Symbol>,
    ) -> Result<(NodeId, NodeId), RefuseError> {
        let measure = crate::density::resolve_ref_chain(m, measure).0;
        if let Some(body) = crate::kernel::reduce_kernel_application(m, measure) {
            return self.sample(m, body, rng, sample_name);
        }
        if m.type_of(measure).is_none()
            && let Some(ty) = flatppl_infer::infer_expression(m, measure, &[])
                .0
                .remove(&measure)
        {
            m.set_type(measure, ty);
        }
        if !matches!(m.type_of(measure), Some(Type::Measure { domain, .. })
            if matches!(domain.as_ref(), Type::Scalar(_)))
        {
            return Err(refuse(
                measure,
                m,
                "broadcast sampling requires a scalar row law",
            ));
        }
        match builtin_name(m, measure) {
            Some("locscale") => {
                let (base, shift, scale) = scalar_locscale(m, measure)?;
                return self.sample_map(m, base, rng, sample_name, |m, point| {
                    let scaled = build_call(m, "mul", &[scale, point]);
                    build_call(m, "add", &[scaled, shift])
                });
            }
            Some("normalize") => {
                let (base, valid) = normalized_sample_base(m, measure)?;
                return self.sample_map(m, base, rng, sample_name, |m, point| {
                    crate::density::exponential_tilt::guard(m, point, valid)
                });
            }
            _ => {}
        }
        let (ctor, parameters) = split_kernel_constructor(m, measure).ok_or_else(|| {
            refuse(
                measure,
                m,
                "broadcast sample requires a supported scalar constructor",
            )
        })?;
        let parameters = parameters
            .into_iter()
            .map(|(name, value)| (name, self.map(m, value)))
            .collect::<Vec<_>>();
        let input = build_record(m, &parameters);
        let ctor = m.alloc(Node::Const(ctor));
        let kernel = build_call(m, "broadcasted", &[ctor]);
        Ok(build_sample_term(m, kernel, input, rng, sample_name))
    }

    fn sample_map(
        &self,
        m: &mut Module,
        base: NodeId,
        rng: NodeId,
        sample_name: Option<Symbol>,
        transform: impl FnOnce(&mut Module, NodeId) -> NodeId,
    ) -> Result<(NodeId, NodeId), RefuseError> {
        let (value, next) = self.sample(m, base, rng, sample_name)?;
        let base = crate::density::resolve_ref_chain(m, base).0;
        let ty = m.type_of(base).cloned().or_else(|| {
            flatppl_infer::infer_expression(m, base, &[])
                .0
                .remove(&base)
        });
        let Some(Type::Measure { domain, .. }) = ty else {
            return Err(refuse(
                base,
                m,
                "broadcast sample has no scalar base domain",
            ));
        };
        let (label, reference, point) = crate::density::broadcast::placeholder(m, *domain);
        let mut inputs = self.inputs.clone();
        let mut actuals = self.actuals.clone();
        inputs.push((label, reference));
        actuals.push(value);
        let body = transform(m, point);
        Ok((map_body(m, body, inputs, actuals, self.rank), next))
    }
}
