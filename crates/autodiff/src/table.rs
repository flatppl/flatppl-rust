//! Column-wise tangents and cotangents of fixed-row real tables.

use flatppl_core::graph::build_call;
use flatppl_core::{Dim, NodeId, Symbol, Type};

use crate::{Emitter, record, tensor};

pub(super) fn column_type(cell: &Type, nrows: Dim) -> Type {
    if let Type::Record(columns) = cell {
        Type::Table {
            columns: columns.clone(),
            nrows,
        }
    } else {
        Type::Array {
            shape: Box::new([nrows]),
            elem: Box::new(cell.clone()),
        }
    }
}

// The table type stores row cells. Its tangent coordinates are the columns.
pub(super) fn columns(ty: &Type) -> Option<Vec<(Symbol, Type)>> {
    let Type::Table { columns, nrows } = ty else {
        return None;
    };
    Some(
        columns
            .iter()
            .map(|(name, cell)| (*name, column_type(cell, *nrows)))
            .collect(),
    )
}

pub(super) fn seed_set(emitter: &mut Emitter, primal: NodeId, ty: &Type) -> Option<NodeId> {
    let Type::Table {
        columns,
        nrows: Dim::Static(nrows),
    } = ty
    else {
        return None;
    };
    let m = &mut emitter.module;
    let index = tensor::int(m, 1);
    let row = build_call(m, "get", &[primal, index]);
    let row_set = record::seed_set(emitter, row, columns)?;
    let nrows = tensor::int(&mut emitter.module, i64::from(*nrows));
    Some(build_call(
        &mut emitter.module,
        "cartpow",
        &[row_set, nrows],
    ))
}

pub(super) fn scatter(
    emitter: &mut Emitter,
    bar: NodeId,
    primal: NodeId,
    ty: &Type,
    selectors: &[NodeId],
    base: i64,
) -> Option<NodeId> {
    let (&selector, rest) = selectors.split_first()?;
    let columns = columns(ty)?;
    if record::selected_field(&emitter.module, &columns, selector).is_some() {
        return record::scatter(emitter, bar, primal, &columns, selectors, base, "table");
    }
    if tensor::index_axes(&emitter.module, selector) != Some(0) {
        return None;
    }
    let Type::Table { columns: cells, .. } = ty else {
        return None;
    };
    let row_bar = if rest.is_empty() {
        bar
    } else {
        let op = if base == 0 { "get0" } else { "get" };
        let row = build_call(&mut emitter.module, op, &[primal, selector]);
        record::scatter(emitter, bar, row, cells, rest, base, "record")?
    };
    let adjoints = record::active_fields(&columns)
        .map(|&(name, ref ty)| {
            let column = record::project(&mut emitter.module, primal, name);
            let bar = record::project(&mut emitter.module, row_bar, name);
            Some((
                name,
                tensor::scatter_typed(emitter, bar, column, &[selector], base, ty)?,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(record::build_named(&mut emitter.module, "table", &adjoints))
}
