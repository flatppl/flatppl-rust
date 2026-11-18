use crate::{Error, Result, data::ColumnModel};
use flatppl_data::{Column, ColumnValues};
use serde::{Deserialize, Serialize};

// Each step adds a simplex coordinate and a reported parameter.
const MAX_STEPS: i64 = 4_096;

/// The level order or integer range fitted for a monotonic predictor.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Monotonic {
    Ordered { levels: Vec<String> },
    Integer { start: i64, steps: i64 },
}

impl Monotonic {
    pub(crate) fn fit(column: &Column, rows: &[usize]) -> Result<Self> {
        let model = match &column.values {
            ColumnValues::Factor { ordered: true, .. } => {
                let (levels, _) = column.factor_levels(rows).unwrap();
                Self::Ordered { levels }
            }
            _ => {
                let values = integer_scores(column, rows)?;
                let start = *values.iter().min().unwrap();
                let end = *values.iter().max().unwrap();
                let steps = end.checked_sub(start).ok_or_else(|| {
                    Error(format!(
                        "monotonic predictor `{}` has too wide an integer range",
                        column.name
                    ))
                })?;
                Self::Integer { start, steps }
            }
        };
        match model.steps() {
            1..=MAX_STEPS => Ok(model),
            0 => Err(Error(format!(
                "monotonic predictor `{}` needs at least two levels",
                column.name
            ))),
            _ => Err(Error(format!(
                "monotonic predictor `{}` exceeds the limit of {MAX_STEPS} steps",
                column.name
            ))),
        }
    }

    pub(crate) fn steps(&self) -> i64 {
        match self {
            Self::Ordered { levels } => levels.len() as i64 - 1,
            Self::Integer { steps, .. } => *steps,
        }
    }

    pub(crate) fn codes(&self, column: &Column, rows: &[usize]) -> Result<Vec<i64>> {
        match self {
            Self::Ordered { levels } if matches!(&column.values, ColumnValues::Factor { .. }) => {
                column.factor_codes(levels, rows)
            }
            Self::Ordered { .. } => Err(Error(format!(
                "monotonic predictor `{}` requires its training factor labels",
                column.name
            ))),
            Self::Integer { start, steps } => integer_scores(column, rows)?
                .into_iter()
                .map(|value| {
                    value
                        .checked_sub(*start)
                        .filter(|step| (0..=*steps).contains(step))
                        .map(|step| step + 1)
                        .ok_or_else(|| Error(format!(
                            "integer score {value} is outside the training range for monotonic predictor `{}`",
                            column.name
                        )))
                })
                .collect(),
        }
    }
}

fn integer_scores(column: &Column, rows: &[usize]) -> Result<Vec<i64>> {
    let bad_score = || {
        Error(format!(
            "monotonic predictor `{}` requires integer scores or an ordered factor",
            column.name
        ))
    };
    if column.like_factor() {
        return Err(bad_score());
    }
    if let ColumnValues::Int(scores) = &column.values {
        return Ok(rows.iter().map(|&row| scores[row]).collect());
    }
    column
        .numeric(rows)?
        .into_iter()
        .map(|value| {
            if value.fract() == 0.0 && (i64::MIN as f64..-(i64::MIN as f64)).contains(&value) {
                Ok(value as i64)
            } else {
                Err(bad_score())
            }
        })
        .collect()
}

pub(crate) fn coefficient(column: &str) -> String {
    format!("mo{}", crate::emit::coefficient_name(column))
}
