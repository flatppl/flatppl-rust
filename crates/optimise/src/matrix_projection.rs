//! Composition of fixed row and column selectors before matrix lowering.

use flatppl_core::{Idx, NodeId};

use crate::{RulePolicy, engine::integer_plan_rows};

pub enum MatrixProjectionFact {
    Product(NodeId, NodeId, NodeId),
    Multiply(NodeId, NodeId, NodeId),
    OneHotRows(NodeId, i64, i64),
    RealMatrix(NodeId, i64, i64),
    SumTree(NodeId, i64, i64),
    Ones(NodeId, i64),
}

/// The original matrix and its two fixed selectors.
#[derive(Debug, PartialEq, Eq)]
pub struct MatrixProjection {
    pub matrix: NodeId,
    pub row_selector: NodeId,
    pub column_selector: NodeId,
}

pub fn plan_matrix_projection(
    output: NodeId,
    facts: &[MatrixProjectionFact],
) -> Option<MatrixProjection> {
    plan_matrix_projection_with_rules(output, facts, &RulePolicy::default())
}

/// Compose matrix selectors using the caller's egglog rules.
pub fn plan_matrix_projection_with_rules(
    output: NodeId,
    facts: &[MatrixProjectionFact],
    rules: &RulePolicy,
) -> Option<MatrixProjection> {
    let id = |id: NodeId| id.index() as i64;
    let mut facts = facts
        .iter()
        .map(|fact| match *fact {
            MatrixProjectionFact::Product(out, left, right) => {
                ("product", vec![id(out), id(left), id(right)])
            }
            MatrixProjectionFact::Multiply(out, left, right) => {
                ("multiply", vec![id(out), id(left), id(right)])
            }
            MatrixProjectionFact::OneHotRows(matrix, rows, cols) => {
                ("one-hot-rows", vec![id(matrix), rows, cols])
            }
            MatrixProjectionFact::RealMatrix(matrix, rows, cols) => {
                ("real-matrix", vec![id(matrix), rows, cols])
            }
            MatrixProjectionFact::SumTree(matrix, padded, levels) => {
                ("sum-tree", vec![id(matrix), padded, levels])
            }
            MatrixProjectionFact::Ones(vector, size) => ("ones", vec![id(vector), size]),
        })
        .collect::<Vec<_>>();
    facts.push(("result", vec![id(output)]));
    let facts = facts
        .iter()
        .map(|(name, operands)| (*name, operands.as_slice()))
        .collect::<Vec<_>>();
    let plans = integer_plan_rows(
        "matrix_projection.egg",
        include_str!("rules/matrix_projection.egg"),
        "matrix-projection",
        &facts,
        "paired-gather",
        rules,
    )?;
    let [plan] = plans.as_slice() else {
        return None;
    };
    let [matrix, row, column] = plan.as_slice() else {
        return None;
    };
    let node = |id: i64| Some(NodeId::from_usize(u32::try_from(id).ok()? as usize));
    Some(MatrixProjection {
        matrix: node(*matrix)?,
        row_selector: node(*row)?,
        column_selector: node(*column)?,
    })
}
