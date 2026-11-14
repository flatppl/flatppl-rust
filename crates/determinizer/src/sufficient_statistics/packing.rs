//! Scalar summary fields share typed vector inputs across all emitters.

use super::{PREFIX, build_call, local_ref};
use flatppl_core::{Binding, Dim, Module, Node, NodeId, Scalar, ScalarType, Symbol, Type};
use std::collections::HashMap;

pub(super) struct PackedSummaries {
    pub summaries: Vec<Symbol>,
    pub projections: Vec<(Symbol, NodeId)>,
}

/// Add preparation buffers and return score projections for their scalar fields.
/// Call before cloning preparation; apply projections only to the score clone.
pub(super) fn pack(m: &mut Module, summaries: &[Symbol]) -> Option<PackedSummaries> {
    let mut groups = Vec::<(ScalarType, Vec<Symbol>)>::new();
    for &name in summaries {
        let rhs = m.binding(m.binding_by_name(name)?).rhs;
        if let Some(Type::Scalar(
            kind @ (ScalarType::Real | ScalarType::Integer | ScalarType::Boolean),
        )) = m.type_of(rhs)
        {
            if let Some((_, fields)) = groups.iter_mut().find(|(key, _)| key == kind) {
                fields.push(name);
            } else {
                groups.push((*kind, vec![name]));
            }
        }
    }
    let mut buffers = HashMap::new();
    let mut projections = Vec::new();
    let mut serial = 0;
    for (kind, fields) in groups {
        if fields.len() < 2 {
            continue;
        }
        let size = u32::try_from(fields.len()).ok()?;
        let name = loop {
            let name = m.intern(&format!("{PREFIX}pack_{serial}"));
            serial += 1;
            if m.binding_by_name(name).is_none() {
                break name;
            }
        };
        let values = fields
            .iter()
            .map(|&field| local_ref(m, field))
            .collect::<Vec<_>>();
        let rhs = build_call(m, "vector", &values);
        m.set_type(
            rhs,
            Type::Array {
                shape: vec![Dim::Static(size)].into(),
                elem: Box::new(Type::Scalar(kind)),
            },
        );
        m.add_binding(Binding {
            name,
            rhs,
            doc: None,
            public: false,
            synthetic: true,
        });
        for (index, &field) in fields.iter().enumerate() {
            let buffer = local_ref(m, name);
            let selector = m.alloc(Node::Lit(Scalar::Int(index as i64 + 1)));
            let projection = build_call(m, "get", &[buffer, selector]);
            m.set_type(projection, Type::Scalar(kind));
            projections.push((field, projection));
            buffers.insert(field, (name, index == 0));
        }
    }
    let mut outputs = Vec::new();
    for &name in summaries {
        if let Some((buffer, first)) = buffers.get(&name) {
            if *first {
                outputs.push(*buffer)
            }
        } else {
            outputs.push(name);
        }
    }
    Some(PackedSummaries {
        summaries: outputs,
        projections,
    })
}
