//! Labelled data appendix grids, shared by document exporters.

use crate::ast::{Fence, Math, Rel};

pub(crate) struct Grid {
    pub(crate) headers: Vec<Math>,
    pub(crate) rows: Vec<Vec<Math>>,
}

/// Preserve numeric strings, column names and row order. Values arrive from
/// full_value, never from the elided membership/placeholder shown in the body.
pub(crate) fn grid(value: &Math) -> Option<Grid> {
    if let Math::Fenced {
        open: Fence::Paren,
        items,
        ..
    } = value
    {
        return Some(Grid {
            headers: vec![Math::text("index"), Math::text("value")],
            rows: items
                .iter()
                .enumerate()
                .map(|(i, value)| vec![Math::int((i + 1) as i64), value.clone()])
                .collect(),
        });
    }
    let Math::Apply { head, args } = value else {
        return None;
    };
    if !matches!(head.as_ref(), Math::Text(name) if name == "table") || args.is_empty() {
        return None;
    }
    let mut headers = Vec::new();
    let mut columns = Vec::new();
    for arg in args {
        let Math::Relation {
            lhs,
            rel: Rel::Eq,
            rhs,
        } = arg
        else {
            return None;
        };
        let Math::Fenced {
            open: Fence::Paren,
            items,
            ..
        } = rhs.as_ref()
        else {
            return None;
        };
        headers.push(lhs.as_ref().clone());
        columns.push(items);
    }
    let nrows = columns[0].len();
    if columns.iter().any(|c| c.len() != nrows) {
        return None;
    }
    Some(Grid {
        headers,
        rows: (0..nrows)
            .map(|i| columns.iter().map(|c| c[i].clone()).collect())
            .collect(),
    })
}
