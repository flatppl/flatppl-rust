//! Black-box members (flatppl-bridge) lower to the protocol's primitive queries.
//!
//! A box member is a `(%ref alias member)` whose alias the host bound to a box
//! ([`Module::box_binding`]). The member stays a module reference in FlatPDL:
//!
//! - a measure member `M` scores as `builtin_logdensityof(B.M, x)` and samples
//!   as `builtin_sample(rng, B.M, sizes...)`, because a measure is a nullary
//!   kernel and FlatPPL has no empty-record input;
//! - a kernel application `B.K(args)` scores as
//!   `builtin_logdensityof(B.K, input, x)` and samples as
//!   `builtin_sample(rng, B.K, input, sizes...)`;
//! - a function application `B.f(args)` stays an application.
//!
//! The box never sees measure algebra. The density and sample rules reduce the
//! forms around a member to these leaves, and refuse the rest.

use std::sync::Arc;

use flatppl_core::{
    BoxBinding, BoxKind, BoxMember, CallHead, Module, NamedKind, Node, NodeId, Ref, RefNs, Scalar,
    Symbol, Type,
};

use crate::density::{build_call, build_record, refuse, resolve_ref};
use crate::refuse::RefuseError;

/// A box member reference.
pub(crate) struct BoxRef {
    /// The `(%ref alias member)` node.
    pub node: NodeId,
    pub binding: Arc<BoxBinding>,
    pub name: String,
}

impl BoxRef {
    pub fn member(&self) -> &BoxMember {
        &self.binding.members[self.name.as_str()]
    }

    fn spelled(&self, m: &Module) -> String {
        match m.node(self.node) {
            Node::Ref(Ref {
                ns: RefNs::Module(alias),
                ..
            }) => format!("{}.{}", m.resolve(*alias), self.name),
            _ => self.name.clone(),
        }
    }
}

/// The box member that `node` names, directly or through bare-name aliases.
pub(crate) fn box_ref(m: &Module, node: NodeId) -> Option<BoxRef> {
    let (node, _) = resolve_ref(m, node);
    let Node::Ref(Ref {
        ns: RefNs::Module(alias),
        name,
    }) = *m.node(node)
    else {
        return None;
    };
    let binding = m.box_binding(alias)?.clone();
    let name = m.resolve(name).to_string();
    binding
        .members
        .contains_key(name.as_str())
        .then_some(BoxRef {
            node,
            binding,
            name,
        })
}

/// The box and member name that `node` names, through bare-name aliases.
pub fn box_member(m: &Module, node: NodeId) -> Option<(Arc<BoxBinding>, String)> {
    box_ref(m, node).map(|target| (target.binding, target.name))
}

/// `true` if `node` names a box member.
pub(crate) fn is_box_ref(m: &Module, node: NodeId) -> bool {
    box_ref(m, node).is_some()
}

/// A box application `B.f(args)` or `B.K(args)`: the member and its input
/// value. A record input collects keyword arguments in the declared field
/// order. Positional arguments bind the fields in order, and one positional
/// record value is the whole input.
pub(crate) fn box_application(
    m: &mut Module,
    node: NodeId,
) -> Option<Result<(BoxRef, NodeId), RefuseError>> {
    let (node, _) = resolve_ref(m, node);
    let Node::Call(call) = m.node(node) else {
        return None;
    };
    let CallHead::User(callee) = call.head else {
        return None;
    };
    let target = box_ref(m, callee)?;
    let (args, named) = (call.args.to_vec(), call.named.to_vec());
    Some(application_input(m, node, &target, &args, &named).map(|input| (target, input)))
}

fn application_input(
    m: &mut Module,
    node: NodeId,
    target: &BoxRef,
    args: &[NodeId],
    named: &[flatppl_core::NamedArg],
) -> Result<NodeId, RefuseError> {
    let spelled = target.spelled(m);
    let member = target.member();
    if member.kind == BoxKind::Measure {
        return Err(refuse(
            node,
            m,
            &format!("box member `{spelled}` is a measure and takes no input"),
        ));
    }
    if named.iter().any(|n| n.kind != NamedKind::Kwarg) {
        return Err(refuse(node, m, "a box application takes plain arguments"));
    }
    let fields: Option<Vec<Symbol>> = match member.input.as_ref().map(|set| &set.ty) {
        Some(Type::Record(fields)) => Some(fields.iter().map(|(name, _)| *name).collect()),
        _ => None,
    };
    match (fields, args, named) {
        (Some(fields), [], named) if !named.is_empty() => {
            let mut values = Vec::with_capacity(fields.len());
            for field in &fields {
                let value = named
                    .iter()
                    .find(|n| m.resolve(n.name) == m.resolve(*field))
                    .map(|n| n.value)
                    .ok_or_else(|| {
                        refuse(
                            node,
                            m,
                            &format!(
                                "box member `{spelled}` needs the argument `{}`",
                                m.resolve(*field)
                            ),
                        )
                    })?;
                values.push((*field, value));
            }
            if named.len() != fields.len() {
                return Err(refuse(
                    node,
                    m,
                    &format!("box member `{spelled}` takes the arguments of its input record only"),
                ));
            }
            Ok(build_record(m, &values))
        }
        (Some(fields), args, []) if args.len() == fields.len() && fields.len() > 1 => {
            let values: Vec<_> = fields.iter().copied().zip(args.iter().copied()).collect();
            Ok(build_record(m, &values))
        }
        (_, [whole], []) => Ok(*whole),
        (None, [], [one]) if m.resolve(one.name) == flatppl_infer::BOX_INPUT => Ok(one.value),
        _ => Err(refuse(
            node,
            m,
            &format!("box member `{spelled}` takes one input, as positional or keyword arguments"),
        )),
    }
}

/// A box leaf in measure position: the member and, for a kernel, its input.
/// `None` when `node` is not a box measure or box kernel application.
pub(crate) fn measure_leaf(
    m: &mut Module,
    node: NodeId,
) -> Option<Result<(BoxRef, Option<NodeId>), RefuseError>> {
    if let Some(target) = box_ref(m, node) {
        let spelled = target.spelled(m);
        return Some(match target.member().kind {
            BoxKind::Measure => Ok((target, None)),
            BoxKind::Kernel => Err(refuse(
                node,
                m,
                &format!("box kernel `{spelled}` is not a measure: apply it to an input"),
            )),
            BoxKind::Function => Err(refuse(
                node,
                m,
                &format!("box function `{spelled}` is not a measure"),
            )),
        });
    }
    let (target, input) = match box_application(m, node)? {
        Ok(found) => found,
        Err(error) => return Some(Err(error)),
    };
    if target.member().kind == BoxKind::Function {
        let spelled = target.spelled(m);
        return Some(Err(refuse(
            node,
            m,
            &format!("box function `{spelled}` returns a value, not a measure"),
        )));
    }
    Some(Ok((target, Some(input))))
}

fn require_query(
    m: &Module,
    node: NodeId,
    target: &BoxRef,
    query: &str,
) -> Result<(), RefuseError> {
    if target.member().supports(query) {
        return Ok(());
    }
    Err(refuse(
        node,
        m,
        &format!(
            "box member `{}` does not answer `{query}` (box `{}` describes its queries as {:?})",
            target.spelled(m),
            target.binding.name,
            target.member().queries
        ),
    ))
}

/// `logdensityof(leaf, v)` for a box leaf.
pub(crate) fn density_leaf(
    m: &mut Module,
    node: NodeId,
    v: NodeId,
) -> Option<Result<NodeId, RefuseError>> {
    Some(measure_leaf(m, node)?.and_then(|(target, input)| {
        require_query(m, node, &target, "builtin_logdensityof")?;
        let mut args = vec![target.node];
        args.extend(input);
        args.push(v);
        Ok(build_call(m, "builtin_logdensityof", &args))
    }))
}

/// `sum(broadcast(builtin_logdensityof, leaf..., v))` for `iid(leaf, shape)`.
/// A scalar variate broadcasts over every cell of `v`. A non-scalar variate
/// needs a one-axis product, whose cells are whole variates.
pub(crate) fn iid_density(
    m: &mut Module,
    inner: NodeId,
    v: NodeId,
    shape: &[usize],
) -> Option<Result<NodeId, RefuseError>> {
    Some(measure_leaf(m, inner)?.and_then(|(target, input)| {
        require_query(m, inner, &target, "builtin_logdensityof")?;
        let scalar = matches!(
            target.member().variate.as_ref().map(|set| &set.ty),
            Some(Type::Scalar(_))
        );
        if !scalar && shape.len() != 1 {
            return Err(refuse(
                inner,
                m,
                "iid of a box member with a non-scalar variate needs a one-axis size",
            ));
        }
        let head = m.intern("builtin_logdensityof");
        let callee = m.alloc(Node::Const(head));
        let mut args = vec![callee, target.node];
        // A record is no broadcast operand (§04). A size-one table of the input
        // holds it constant across the cells instead.
        if let Some(input) = input {
            args.push(singleton(m, input, shape.len()));
        }
        args.push(v);
        let cells = build_call(m, "broadcast", &args);
        Ok(build_call(m, "sum", &[cells]))
    }))
}

/// A size-one array of `input` with `depth` axes: nested `vector`s, or for a
/// record the broadcast record of size-one field arrays.
fn singleton(m: &mut Module, input: NodeId, depth: usize) -> NodeId {
    let record = crate::density::materialize_record_value(m, input);
    let Some(fields) = crate::density::expect_builtin_call(m, record, "record").map(|c| {
        c.named
            .iter()
            .map(|n| (n.name, n.value))
            .collect::<Vec<_>>()
    }) else {
        let mut wrapped = input;
        for _ in 0..depth {
            wrapped = build_call(m, "vector", &[wrapped]);
        }
        return wrapped;
    };
    let named = fields
        .into_iter()
        .map(|(name, value)| {
            let mut wrapped = value;
            for _ in 0..depth {
                wrapped = build_call(m, "vector", &[wrapped]);
            }
            flatppl_core::NamedArg {
                kind: NamedKind::Kwarg,
                name,
                value: wrapped,
            }
        })
        .collect::<Vec<_>>();
    let head = m.intern("record");
    let head = m.alloc(Node::Const(head));
    let broadcast = m.intern("broadcast");
    m.alloc(Node::Call(flatppl_core::Call {
        head: CallHead::Builtin(broadcast),
        args: vec![head].into(),
        named: named.into(),
        inputs: None,
    }))
}

/// `builtin_sample(rng, leaf..., shape...)` for a box leaf. Only a normalized
/// member samples: sampling exists for probability measures only.
pub(crate) fn sample_leaf(
    m: &mut Module,
    node: NodeId,
    rng: NodeId,
    shape: &[usize],
) -> Option<Result<NodeId, RefuseError>> {
    Some(measure_leaf(m, node)?.and_then(|(target, input)| {
        require_normalized(m, node, &target)?;
        require_query(m, node, &target, "builtin_sample")?;
        let mut args = vec![rng, target.node];
        args.extend(input);
        args.extend(
            shape
                .iter()
                .map(|&n| m.alloc(Node::Lit(Scalar::Int(n as i64)))),
        );
        Ok(build_call(m, "builtin_sample", &args))
    }))
}

fn require_normalized(m: &Module, node: NodeId, target: &BoxRef) -> Result<(), RefuseError> {
    if target.member().normalized {
        return Ok(());
    }
    Err(refuse(
        node,
        m,
        &format!(
            "box member `{}` is not normalized: `~` and `rand` need a probability measure",
            target.spelled(m)
        ),
    ))
}

/// Refuse `~` and `rand` on an unnormalized box member before any lowering.
/// Sampling exists only for probability measures, so even a density query
/// over such a draw has no generative meaning.
pub(crate) fn refuse_unnormalized_draws(m: &Module) -> Result<(), RefuseError> {
    for id in (0..m.node_count()).map(NodeId::from_usize) {
        let measure = match m.node(id) {
            Node::Call(c) => match c.head {
                CallHead::Builtin(h) if m.resolve(h) == "draw" => c.args.first().copied(),
                CallHead::Builtin(h) if m.resolve(h) == "rand" => c.args.first().copied(),
                _ => None,
            },
            _ => None,
        };
        let Some(measure) = measure else { continue };
        let measure = strip_iid(m, measure);
        if let Some(target) = leaf_member(m, measure)
            && target.member().kind != BoxKind::Function
        {
            require_normalized(m, measure, &target)?;
        }
    }
    Ok(())
}

/// The member of a box measure or box application, without building its input.
fn leaf_member(m: &Module, node: NodeId) -> Option<BoxRef> {
    let (node, _) = resolve_ref(m, node);
    match m.node(node) {
        Node::Call(c) => match c.head {
            CallHead::User(callee) => box_ref(m, callee),
            CallHead::Builtin(_) => None,
        },
        _ => box_ref(m, node),
    }
}

fn strip_iid(m: &Module, node: NodeId) -> NodeId {
    let (node, _) = resolve_ref(m, node);
    match crate::density::expect_builtin_call(m, node, "iid") {
        Some(c) if !c.args.is_empty() => strip_iid(m, c.args[0]),
        _ => node,
    }
}

use flatppl_core::Idx;
