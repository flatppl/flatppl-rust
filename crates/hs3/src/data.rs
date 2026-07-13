//! HS3 `data` entries → embedded FlatPPL tables and domains (spec §03).
//!
//! Each HS3 dataset (`data` block) is lowered to top-level bindings:
//!   - `<name> = table(axis = [...], ...)` — the observed sample, one column per
//!     observable axis (spec §03 "Tables": an unbinned multivariate event sample
//!     is the table data-carrier). A binned dataset becomes a single `counts`
//!     column.
//!   - `<name>_domain = cartprod(axis = interval(min, max), ...)` — the
//!     observable region, emitted when the dataset declares axes (spec §03
//!     "Cartesian product" / preset domains).
use crate::builder::Builder;
use crate::error::{Error, Result};
use crate::model::{Datum, Document};
use crate::presets::cartprod_of_axes;
use flatppl_core::Module;
use flatppl_core::NodeId;
use std::collections::BTreeMap;

/// The columnar shape of a lowered dataset: the table's column names (observable
/// axis names, in order) and its row count. Lets the likelihood emitter reference
/// the embedded table (`get(<name>, "<axis>")`, or the table itself) without
/// re-materializing the values.
pub(crate) struct DataShape {
    pub columns: Vec<String>,
    pub n_rows: usize,
}

/// Step 3b: emit every `data` entry as `<name> = table(...)` plus, when it
/// carries observable axes, a companion `<name>_domain = cartprod(...)`.
pub(crate) fn emit_data(m: &mut Module, doc: &Document) -> Result<()> {
    let mut b = Builder::new(m);
    for d in &doc.data {
        if let Some(table) = build_table(&mut b, d)? {
            b.bind(&d.name, table);
        }
        if !d.axes.is_empty() {
            let cp = cartprod_of_axes(&mut b, &d.axes);
            b.bind(&format!("{}_domain", d.name), cp);
        }
    }
    Ok(())
}

/// Column names for a dataset's table: the declared axis names when their count
/// matches the data arity, else positional `c1, c2, …`. Errors on a ragged
/// unbinned sample or an axis-count/arity mismatch (either would silently drop
/// or mislabel columns).
pub(crate) fn datum_columns(d: &Datum) -> Result<Vec<String>> {
    if d.weights.is_some() {
        return Err(Error::Unimplemented(format!(
            "datum `{}` carries per-event weights",
            d.name
        )));
    }
    if d.entries_uncertainties.is_some() {
        return Err(Error::Unimplemented(format!(
            "datum `{}` carries entries_uncertainties",
            d.name
        )));
    }
    if d.uncertainty.is_some() {
        return Err(Error::Unimplemented(format!(
            "datum `{}` carries an uncertainty block",
            d.name
        )));
    }
    if !d.entries.is_empty() {
        let arity = d.entries[0].len();
        for (i, e) in d.entries.iter().enumerate() {
            if e.len() != arity {
                return Err(Error::Unsupported(format!(
                    "unbinned datum `{}` is ragged: entry {i} has {} coordinate(s), expected {arity}",
                    d.name,
                    e.len()
                )));
            }
        }
        if d.axes.is_empty() {
            Ok((1..=arity).map(|i| format!("c{i}")).collect())
        } else if d.axes.len() == arity {
            Ok(d.axes.iter().map(|a| a.name.clone()).collect())
        } else {
            Err(Error::Unsupported(format!(
                "unbinned datum `{}` declares {} axes but has {arity}-coordinate entries",
                d.name,
                d.axes.len()
            )))
        }
    } else if d.contents.is_some() {
        // Binned counts: a single column of per-bin observed counts.
        Ok(vec!["counts".to_string()])
    } else {
        Ok(Vec::new())
    }
}

/// Build the `table(...)` node for a dataset, or `None` when it carries no
/// observations (neither `entries` nor `contents`).
fn build_table(b: &mut Builder, d: &Datum) -> Result<Option<NodeId>> {
    let columns = datum_columns(d)?;
    if columns.is_empty() {
        return Ok(None);
    }
    let col_nodes: Vec<NodeId> = if d.entries.is_empty() {
        // Binned: a single `counts` column.
        let contents = d.contents.as_deref().unwrap_or(&[]);
        let elems: Vec<NodeId> = contents.iter().map(|&v| b.lit_real(v)).collect();
        vec![b.array(&elems)]
    } else {
        // Unbinned: one column per coordinate, materialized in entry order.
        (0..columns.len())
            .map(|j| {
                let elems: Vec<NodeId> = d.entries.iter().map(|e| b.lit_real(e[j])).collect();
                b.array(&elems)
            })
            .collect()
    };
    let fields: Vec<(&str, NodeId)> = columns.iter().map(|s| s.as_str()).zip(col_nodes).collect();
    Ok(Some(b.call_fields("table", &fields)))
}

/// Per-datum columnar shapes for the likelihood emitter, over the by-name
/// observation source (datasets with `entries`; binned data is consumed by the
/// channel-assembly path). Mirrors [`emit_data`]'s column choice so a
/// `get(<name>, "<col>")` reference always names a real column.
pub(crate) fn data_shapes(doc: &Document) -> Result<BTreeMap<String, DataShape>> {
    let mut out = BTreeMap::new();
    for d in doc.data.iter().filter(|d| !d.entries.is_empty()) {
        out.insert(
            d.name.clone(),
            DataShape {
                columns: datum_columns(d)?,
                n_rows: d.entries.len(),
            },
        );
    }
    Ok(out)
}
