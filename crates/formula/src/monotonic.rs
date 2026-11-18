use crate::{
    ColumnExpression, Contrast, DataValue, Design, DesignComponent, Error, Result, Table, Term,
    data::ColumnModel,
    design::{self, TermView},
    emit::{self, predictor_name},
};
use flatppl_data::{Column, ColumnValues};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct MonotonicCall {
    pub column: String,
    pub id: Option<String>,
}

impl MonotonicCall {
    pub(crate) fn label(&self) -> String {
        match &self.id {
            None => format!("mo({})", self.column),
            Some(id) => format!("mo({}, id = \"{id}\")", self.column),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub(crate) struct MonotonicTerms {
    pub variables: Vec<String>,
    pub terms: Vec<Term>,
    pub calls: BTreeMap<String, MonotonicCall>,
    pub expressions: BTreeMap<String, ColumnExpression>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct MonotonicDesign {
    pub design: Design,
    pub columns: Vec<(String, Monotonic)>,
    pub effects: Vec<MonotonicEffect>,
    pub simplexes: Vec<Simplex>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct MonotonicEffect {
    pub name: String,
    pub factors: Vec<(usize, usize)>,
    pub has_covariate: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Simplex {
    pub name: String,
    pub steps: i64,
}

impl MonotonicTerms {
    fn placeholder_expressions(&self) -> BTreeMap<String, ColumnExpression> {
        let mut expressions = self.expressions.clone();
        expressions.extend(
            self.calls
                .keys()
                .map(|call| (call.clone(), ColumnExpression::Number(1.0))),
        );
        expressions
    }
}

impl MonotonicDesign {
    pub(crate) fn fit(
        terms: &MonotonicTerms,
        table: &Table,
        rows: &[usize],
        contrasts: &BTreeMap<String, Contrast>,
    ) -> Result<Self> {
        let (mut design, _) = design::build(
            TermView {
                intercept: true,
                variables: &terms.variables,
                fixed: &terms.terms,
                expressions: &terms.placeholder_expressions(),
            },
            table,
            rows,
            contrasts,
        )?;
        let mut columns = Vec::new();
        let mut by_column = HashMap::new();
        let mut by_id = HashMap::new();
        let mut simplexes: Vec<Simplex> = Vec::new();
        let mut effects = Vec::new();
        for column in &mut design.columns {
            let name = emit::coefficient_name(&column.name);
            let mut factors = Vec::new();
            for component in &column.components {
                let Some(call) = special_call(component, &design.predictors, terms) else {
                    continue;
                };
                let column_index = match by_column.get(&call.column) {
                    Some(&i) => i,
                    None => {
                        let i = columns.len();
                        columns.push((
                            call.column.clone(),
                            Monotonic::fit(table.column(&call.column)?, rows)?,
                        ));
                        by_column.insert(call.column.clone(), i);
                        i
                    }
                };
                let steps = columns[column_index].1.steps();
                let simplex_index = simplexes.len();
                let simplex = match &call.id {
                    Some(id) => *by_id.entry(id).or_insert(simplex_index),
                    None => simplex_index,
                };
                if simplex == simplex_index {
                    simplexes.push(Simplex {
                        name: call
                            .id
                            .clone()
                            .unwrap_or_else(|| format!("{name}{}", factors.len() + 1)),
                        steps,
                    });
                } else if simplexes[simplex].steps != steps {
                    return Err(Error(format!(
                        "monotonic simplex id `{}` requires the same number of steps",
                        simplexes[simplex].name
                    )));
                }
                factors.push((column_index, simplex));
            }
            column
                .components
                .retain(|component| special_call(component, &design.predictors, terms).is_none());
            effects.push(MonotonicEffect {
                name,
                factors,
                has_covariate: !column.components.is_empty(),
            });
        }
        let mut names = HashSet::new();
        for simplex in &simplexes {
            if !names.insert(&simplex.name) {
                return Err(Error(format!(
                    "monotonic simplex name `{}` is ambiguous",
                    simplex.name
                )));
            }
        }
        Ok(Self {
            design,
            columns,
            effects,
            simplexes,
        })
    }

    pub(crate) fn prepare(
        &self,
        terms: &MonotonicTerms,
        dpar: &str,
        table: &Table,
        rows: &[usize],
        constants: &mut BTreeMap<String, DataValue>,
    ) -> Result<()> {
        for (i, (column, coding)) in self.columns.iter().enumerate() {
            let values = coding.codes(table.column(column)?, rows)?;
            constants.insert(
                predictor_name(dpar, &format!("mo_{}", i + 1)),
                DataValue::Integers(values),
            );
        }
        let inputs =
            design::numeric_inputs(&self.design, &terms.placeholder_expressions(), table, rows)?;
        for (i, (effect, column)) in self.effects.iter().zip(&self.design.columns).enumerate() {
            if effect.has_covariate {
                let values: Vec<_> = (0..rows.len())
                    .map(|row| design::column_value(&self.design, column, &inputs, row))
                    .collect();
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(Error(format!(
                        "monotonic covariate `{}` is not finite",
                        effect.name
                    )));
                }
                constants.insert(
                    predictor_name(dpar, &format!("mo_cov_{}", i + 1)),
                    DataValue::Real64(values),
                );
            }
        }
        Ok(())
    }
}

fn special_call<'a>(
    component: &DesignComponent,
    predictors: &[design::Predictor],
    terms: &'a MonotonicTerms,
) -> Option<&'a MonotonicCall> {
    match *component {
        DesignComponent::Numeric { predictor } => terms.calls.get(&predictors[predictor].column),
        _ => None,
    }
}
