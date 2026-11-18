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
    let resolved_head = crate::density::resolve_ref_chain(m, head).0;
    let vector_law = crate::crossmodule::std_module_member(m, resolved_head)
        .is_some_and(|(module, _)| module == "regression");
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
                if (!vector_law && !matches!(*elem, Type::Scalar(_)))
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
            Some(Type::Table {
                nrows: Dim::Static(_),
                ..
            }) if rank.is_none_or(|r| r == 1) => {
                // §03 tables iterate one row axis. cell_input projects the
                // columns and builds the row record, retaining vector fields.
                rank = Some(1);
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
    if vector_law
        && let Some(Type::Measure { domain, mass }) = m.type_of(node).cloned()
        && let Type::Array { elem, .. } = *domain
    {
        m.set_type(measure, Type::Measure { domain: elem, mass });
    }
    if let Some(rank) = rank {
        Scope {
            inputs,
            actuals,
            rank,
            vector_law,
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
    vector_law: bool,
}

/// The lanes of equal-length columns: one cell placeholder per column, standing
/// for that column's value in the lane. A scalar law built over the placeholders
/// samples once per lane, independently, in one batched draw (§04 broadcasting).
pub(crate) struct Lanes {
    scope: Scope,
    /// The placeholders, in column order. A table column gives a record point.
    pub(crate) points: Vec<NodeId>,
}

impl Lanes {
    pub(crate) fn over(m: &mut Module, columns: &[NodeId]) -> Result<Self, RefuseError> {
        let mut inputs = Vec::new();
        let mut actuals = Vec::new();
        let points = columns
            .iter()
            .map(|&column| cell_input(m, column, &mut inputs, &mut actuals))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            scope: Scope {
                inputs,
                actuals,
                rank: 1,
                vector_law: false,
            },
            points,
        })
    }

    /// Draw `law`, a scalar law over [`Self::points`], once per lane.
    pub(crate) fn sample(
        &self,
        m: &mut Module,
        law: NodeId,
        rng: NodeId,
    ) -> Result<(NodeId, NodeId), RefuseError> {
        self.scope.sample(m, law, rng, None)
    }
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
            if matches!(domain.as_ref(), Type::Scalar(_))
                || self.vector_law && matches!(domain.as_ref(), Type::Array { shape, elem }
                    if shape.len() == 1 && matches!(elem.as_ref(), Type::Scalar(_))))
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
                if let Some((tag, parameters)) = restricted::parameters(m, measure) {
                    return Ok(self.sample_constructor(m, tag, parameters, rng, sample_name));
                }
                if let Some(mixture) = zero_atom_mixture(m, measure) {
                    return self.sample_zero_atom_mixture(m, mixture, rng, sample_name);
                }
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
        Ok(self.sample_constructor(m, ctor, parameters, rng, sample_name))
    }

    fn sample_constructor(
        &self,
        m: &mut Module,
        ctor: Symbol,
        parameters: Vec<(Symbol, NodeId)>,
        rng: NodeId,
        sample_name: Option<Symbol>,
    ) -> (NodeId, NodeId) {
        let parameters = parameters
            .into_iter()
            .map(|(name, value)| (name, self.map(m, value)))
            .collect::<Vec<_>>();
        let input = build_record(m, &parameters);
        let ctor = m.alloc(Node::Const(ctor));
        let kernel = build_call(m, "broadcasted", &[ctor]);
        build_sample_term(m, kernel, input, rng, sample_name)
    }

    fn sample_zero_atom_mixture(
        &self,
        m: &mut Module,
        mixture: ZeroAtomMixture,
        rng: NodeId,
        sample_name: Option<Symbol>,
    ) -> Result<(NodeId, NodeId), RefuseError> {
        let ZeroAtomMixture {
            weight,
            atom,
            base,
            domain,
        } = mixture;
        let bernoulli = m.intern("Bernoulli");
        let p = m.intern("p");
        let (gate, next) = self.sample_constructor(m, bernoulli, vec![(p, weight)], rng, None);
        let (variate, next) = self.sample(m, base, next, sample_name)?;

        let (gate_name, gate_ref, gate_point) =
            crate::density::broadcast::placeholder(m, Type::Scalar(ScalarType::Integer));
        let (base_name, base_ref, base_point) = crate::density::broadcast::placeholder(m, domain);
        let one = m.alloc(Node::Lit(Scalar::Int(1)));
        let is_atom = build_call(m, "equal", &[gate_point, one]);
        let selected = build_call(m, "ifelse", &[is_atom, atom, base_point]);
        let value = map_body(
            m,
            selected,
            vec![(gate_name, gate_ref), (base_name, base_ref)],
            vec![gate, variate],
            self.rank,
        );
        Ok((value, next))
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

struct ZeroAtomMixture {
    weight: NodeId,
    atom: NodeId,
    base: NodeId,
    domain: Type,
}

/// Generated formula mixtures weigh a zero atom by p and a probability kernel
/// by 1 - p. The Bernoulli gate validates p and an independent base draw is
/// taken in every lane so key advancement is well defined at either endpoint.
fn zero_atom_mixture(m: &mut Module, measure: NodeId) -> Option<ZeroAtomMixture> {
    let [inner] = *expect_builtin_call(m, measure, "normalize")?.args else {
        return None;
    };
    let [first, second] = *expect_builtin_call(m, resolve_ref(m, inner).0, "superpose")?.args
    else {
        return None;
    };
    let component = |node| {
        let [weight, law] = *expect_builtin_call(m, resolve_ref(m, node).0, "weighted")?.args
        else {
            return None;
        };
        Some((weight, resolve_ref(m, law).0))
    };
    let (weight, dirac) = component(first)?;
    if crate::density::weight_is_variate_dependent(m, weight) {
        return None;
    }
    let (complement, base) = component(second)?;
    let [atom] = *expect_builtin_call(m, dirac, "Dirac")?.args else {
        return None;
    };
    let atom = resolve_ref(m, atom).0;
    if !matches!(m.node(atom), Node::Lit(Scalar::Int(0) | Scalar::Real(0.0))) {
        return None;
    }
    let [one, probability] = *expect_builtin_call(m, resolve_ref(m, complement).0, "sub")?.args
    else {
        return None;
    };
    let resolved_weight = resolve_ref(m, weight).0;
    let probability = resolve_ref(m, probability).0;
    // A reference can be duplicated by inlining without duplicating its value.
    // Separate but identical stochastic expressions need not be the same draw.
    let same_weight = resolved_weight == probability
        || matches!(m.node(resolved_weight), Node::Ref(_) | Node::Lit(_))
            && m.structural_eq(resolved_weight, probability);
    if crate::density::static_real(m, one) != Some(1.0) || !same_weight {
        return None;
    }
    flatppl_infer::distribution_param_ranks(builtin_name(m, base)?)?;
    let Type::Measure {
        domain,
        mass: flatppl_core::Mass::Normalized,
    } = crate::density::known_type(m, base)?
    else {
        return None;
    };
    matches!(domain.as_ref(), Type::Scalar(_)).then_some(ZeroAtomMixture {
        weight,
        atom,
        base,
        domain: *domain,
    })
}
