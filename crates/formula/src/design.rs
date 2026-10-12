pub use crate::contrasts::Contrast;
use crate::{ColumnExpression, Error, Formula, GroupTerm, Result, Table, Term};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Design {
    pub predictors: Vec<Predictor>,
    pub columns: Vec<DesignColumn>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Predictor {
    pub column: String,
    pub factor: Option<FactorEncoding>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FactorEncoding {
    pub levels: Vec<String>,
    pub ordered: bool,
    pub contrast: Contrast,
    /// Rows follow training levels; columns follow `column_names`.
    /// Empty when the factor uses only indicator components.
    pub matrix: Vec<Vec<f64>>,
    pub column_names: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DesignColumn {
    pub name: String,
    /// One-based formula term index, matching R's model.matrix assignment.
    pub term: usize,
    pub components: Vec<DesignComponent>,
}

/// All recipe indices are zero-based; prepared factor input codes are one-based.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DesignComponent {
    Numeric { predictor: usize },
    Contrast { predictor: usize, column: usize },
    Indicator { predictor: usize, level: usize },
}

pub(crate) struct TermView<'a> {
    pub intercept: bool,
    pub variables: &'a [String],
    pub fixed: &'a [Term],
    pub expressions: &'a BTreeMap<String, ColumnExpression>,
}

impl<'a> From<&'a Formula> for TermView<'a> {
    fn from(formula: &'a Formula) -> Self {
        Self {
            intercept: formula.intercept,
            variables: &formula.variables,
            fixed: &formula.fixed,
            expressions: &formula.expressions,
        }
    }
}

impl<'a> From<&'a GroupTerm> for TermView<'a> {
    fn from(group: &'a GroupTerm) -> Self {
        Self {
            intercept: group.intercept,
            variables: &group.variables,
            fixed: &group.fixed,
            expressions: &group.expressions,
        }
    }
}

pub(crate) fn build(
    formula: TermView<'_>,
    table: &Table,
    rows: &[usize],
    contrasts: &BTreeMap<String, Contrast>,
) -> Result<(Design, Vec<f64>)> {
    let mut predictors = formula
        .variables
        .iter()
        .map(|name| {
            if formula.expressions.contains_key(name) {
                if contrasts.contains_key(name) {
                    return Err(Error(format!(
                        "contrast supplied for numeric predictor `{name}`"
                    )));
                }
                return Ok(Predictor {
                    column: name.clone(),
                    factor: None,
                });
            }
            let column = table.column(name)?;
            let factor = column
                .factor_levels(rows)
                .map(|(levels, ordered)| -> Result<_> {
                    if levels.len() < 2 {
                        return Err(Error(format!(
                            "predictor `{name}`: contrasts require at least two observed levels"
                        )));
                    }
                    let contrast = contrasts.get(name).copied().unwrap_or(if ordered {
                        Contrast::Polynomial
                    } else {
                        Contrast::Treatment
                    });
                    Ok(FactorEncoding {
                        levels,
                        ordered,
                        contrast,
                        matrix: Vec::new(),
                        column_names: Vec::new(),
                    })
                });
            if factor.is_none() && contrasts.contains_key(name) {
                return Err(Error(format!(
                    "contrast supplied for numeric predictor `{name}`"
                )));
            }
            Ok(Predictor {
                column: name.clone(),
                factor: factor.transpose()?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut columns = Vec::new();
    let mut first_factor = !formula.intercept;
    for (term_index, term) in formula.fixed.iter().enumerate() {
        let mut term_columns = vec![DesignColumn {
            name: String::new(),
            term: term_index + 1,
            components: Vec::new(),
        }];
        for (predictor, input) in predictors.iter_mut().enumerate() {
            if !term.factors.contains(&input.column) {
                continue;
            }
            let choices = if let Some(factor) = &mut input.factor {
                // R terms() chooses contrasts only when the reduced term is
                // contained in an earlier term (the empty term also qualifies).
                let reduced: Vec<_> = term
                    .factors
                    .iter()
                    .filter(|name| *name != &input.column)
                    .collect();
                let use_contrast = reduced.is_empty()
                    || formula.fixed[..term_index]
                        .iter()
                        .any(|earlier| reduced.iter().all(|name| earlier.factors.contains(name)));
                let dummy = first_factor || !use_contrast;
                first_factor = false;
                if dummy {
                    factor
                        .levels
                        .iter()
                        .enumerate()
                        .map(|(level, name)| {
                            (
                                format!("{}{name}", input.column),
                                DesignComponent::Indicator { predictor, level },
                            )
                        })
                        .collect::<Vec<_>>()
                } else {
                    if factor.matrix.is_empty() {
                        (factor.column_names, factor.matrix) =
                            crate::contrasts::matrix(factor.contrast, &factor.levels).map_err(
                                |error| Error(format!("predictor `{}`: {error}", input.column)),
                            )?;
                    }
                    factor
                        .column_names
                        .iter()
                        .enumerate()
                        .map(|(column, name)| {
                            (
                                format!("{}{name}", input.column),
                                DesignComponent::Contrast { predictor, column },
                            )
                        })
                        .collect()
                }
            } else {
                vec![(input.column.clone(), DesignComponent::Numeric { predictor })]
            };
            let mut expanded = Vec::new();
            // The first variable varies fastest within each interaction.
            for (name, component) in choices {
                for previous in &term_columns {
                    let mut next = previous.clone();
                    if !next.name.is_empty() {
                        next.name.push(':');
                    }
                    next.name.push_str(&name);
                    next.components.push(component.clone());
                    expanded.push(next);
                }
            }
            term_columns = expanded;
        }
        columns.extend(term_columns);
    }
    let design = Design {
        predictors,
        columns,
    };
    let means = means(&design, &formula, table, rows)?;
    Ok((design, means))
}

fn means(
    design: &Design,
    formula: &TermView<'_>,
    table: &Table,
    rows: &[usize],
) -> Result<Vec<f64>> {
    let inputs = design
        .predictors
        .iter()
        .map(|predictor| {
            if let Some(expression) = formula.expressions.get(&predictor.column) {
                return expression.values(table, rows);
            }
            let column = table.column(&predictor.column)?;
            if let Some(factor) = &predictor.factor {
                column
                    .factor_codes(&factor.levels, rows)
                    .map(|codes| codes.into_iter().map(|code| (code - 1) as f64).collect())
            } else {
                column.numeric(rows)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    design
        .columns
        .iter()
        .map(|column| {
            let mean = (0..rows.len())
                .map(|row| {
                    column
                        .components
                        .iter()
                        .map(|component| match *component {
                            DesignComponent::Numeric { predictor } => inputs[predictor][row],
                            DesignComponent::Contrast { predictor, column } => {
                                let factor = design.predictors[predictor].factor.as_ref().unwrap();
                                factor.matrix[inputs[predictor][row] as usize][column]
                            }
                            DesignComponent::Indicator { predictor, level } => {
                                if inputs[predictor][row] as usize == level {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                        })
                        .product::<f64>()
                })
                .sum::<f64>()
                / rows.len() as f64;
            if !mean.is_finite() {
                return Err(Error(format!(
                    "design column `{}` has an unrepresentable mean",
                    column.name
                )));
            }
            Ok(mean)
        })
        .collect()
}
