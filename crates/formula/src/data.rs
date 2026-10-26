use crate::{Error, Family, Formula, GroupLevel, Recipe, Result, Trials};
use flatppl_data::{Column, ColumnValues, Table};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataMode {
    Training,
    Prediction,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct DataDocument {
    pub constants: BTreeMap<String, DataValue>,
    /// Zero-based row indices in the supplied table, after missing-row omission.
    pub rows: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum DataValue {
    Integer(i64),
    Real32(Vec<f32>),
    Real64(Vec<f64>),
    Integers(Vec<i64>),
}

impl DataDocument {
    /// Prepared numeric columns for a JSON sidecar; the row count stays structural.
    pub fn columns_json(&self) -> Result<serde_json::Value> {
        let mut columns = serde_json::Map::new();
        for (name, value) in &self.constants {
            let value = match value {
                DataValue::Integer(_) if name == "N" => continue,
                DataValue::Integer(_) => {
                    return Err(Error(format!("unexpected scalar data binding `{name}`")));
                }
                DataValue::Real32(values) => {
                    // JSON has no dtype; preserve the exact stored f32 values.
                    serde_json::to_value(values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>())?
                }
                DataValue::Real64(values) => serde_json::to_value(values)?,
                DataValue::Integers(values) => serde_json::to_value(values)?,
            };
            columns.insert(name.clone(), value);
        }
        Ok(serde_json::Value::Object(columns))
    }

    /// Bind prepared columns through a file source in a copy of an external template.
    pub fn with_data_source(
        &self,
        template: flatppl_core::Module,
        source: &str,
    ) -> Result<flatppl_core::Module> {
        crate::emit::with_data_source(template, self, source)
    }
}

/// Model-frame reads of a data column.
pub(crate) trait ColumnModel {
    fn like_factor(&self) -> bool;
    fn present(&self, row: usize) -> bool;
    fn numeric(&self, rows: &[usize]) -> Result<Vec<f64>>;
    fn counts(&self, rows: &[usize]) -> Result<Vec<i64>>;
    fn data(&self, rows: &[usize]) -> Result<DataValue>;
    fn factor_levels(&self, rows: &[usize]) -> Option<(Vec<String>, bool)>;
    fn factor_codes(&self, levels: &[String], rows: &[usize]) -> Result<Vec<i64>>;
    fn level(&self, row: usize) -> Result<GroupLevel>;
    fn levels(&self, rows: &[usize]) -> Result<Vec<GroupLevel>>;
}

impl ColumnModel for Column {
    fn like_factor(&self) -> bool {
        matches!(
            self.values,
            ColumnValues::Factor { .. } | ColumnValues::Bool(_)
        )
    }

    fn present(&self, row: usize) -> bool {
        self.valid.as_ref().is_none_or(|v| v[row])
            && match &self.values {
                ColumnValues::Real32(v) => !v[row].is_nan(),
                ColumnValues::Real64(v) => !v[row].is_nan(),
                _ => true,
            }
    }

    fn numeric(&self, rows: &[usize]) -> Result<Vec<f64>> {
        let values: Vec<_> = match &self.values {
            ColumnValues::Real32(v) => rows.iter().map(|&i| f64::from(v[i])).collect(),
            ColumnValues::Real64(v) => rows.iter().map(|&i| v[i]).collect(),
            ColumnValues::Int(v) => rows.iter().map(|&i| v[i] as f64).collect(),
            ColumnValues::Factor { .. } | ColumnValues::Bool(_) => {
                return Err(Error(format!("column `{}` must be numeric", self.name)));
            }
        };
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error(format!(
                "column `{}` contains infinite values",
                self.name
            )));
        }
        Ok(values)
    }

    fn counts(&self, rows: &[usize]) -> Result<Vec<i64>> {
        let values = if let ColumnValues::Int(values) = &self.values {
            rows.iter().map(|&i| values[i]).collect()
        } else {
            self.numeric(rows)?
                .into_iter()
                .map(|value| {
                    if value < 0.0 || value.fract() != 0.0 || value >= -(i64::MIN as f64) {
                        return Err(Error(format!(
                            "column `{}` must contain nonnegative integers",
                            self.name
                        )));
                    }
                    Ok(value as i64)
                })
                .collect::<Result<Vec<_>>>()?
        };
        if values.iter().any(|&value| value < 0) {
            return Err(Error(format!(
                "column `{}` must contain nonnegative integers",
                self.name
            )));
        }
        Ok(values)
    }

    fn data(&self, rows: &[usize]) -> Result<DataValue> {
        let values = match &self.values {
            ColumnValues::Real32(v) => DataValue::Real32(rows.iter().map(|&i| v[i]).collect()),
            ColumnValues::Real64(v) => DataValue::Real64(rows.iter().map(|&i| v[i]).collect()),
            ColumnValues::Int(v) => DataValue::Integers(rows.iter().map(|&i| v[i]).collect()),
            ColumnValues::Factor { .. } | ColumnValues::Bool(_) => {
                return Err(Error(format!("column `{}` must be numeric", self.name)));
            }
        };
        let finite = match &values {
            DataValue::Real32(v) => v.iter().all(|v| v.is_finite()),
            DataValue::Real64(v) => v.iter().all(|v| v.is_finite()),
            _ => true,
        };
        if !finite {
            return Err(Error(format!(
                "column `{}` contains infinite values",
                self.name
            )));
        }
        Ok(values)
    }

    fn factor_levels(&self, rows: &[usize]) -> Option<(Vec<String>, bool)> {
        match &self.values {
            ColumnValues::Factor {
                levels,
                codes,
                ordered,
            } => {
                let observed: HashSet<_> = rows.iter().map(|&row| codes[row] as usize).collect();
                Some((
                    levels
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| observed.contains(i))
                        .map(|(_, level)| level.clone())
                        .collect(),
                    *ordered,
                ))
            }
            // R logical contrasts retain both possible values even when only
            // one occurs in the training data.
            ColumnValues::Bool(_) => Some((vec!["FALSE".into(), "TRUE".into()], false)),
            _ => None,
        }
    }

    fn factor_codes(&self, levels: &[String], rows: &[usize]) -> Result<Vec<i64>> {
        let indices: HashMap<_, _> = levels
            .iter()
            .enumerate()
            .map(|(index, level)| (level.as_str(), index as i64 + 1))
            .collect();
        rows.iter()
            .map(|&row| {
                let level = match &self.values {
                    ColumnValues::Factor { levels, codes, .. } => {
                        levels[codes[row] as usize].as_str()
                    }
                    ColumnValues::Bool(values) => {
                        if values[row] {
                            "TRUE"
                        } else {
                            "FALSE"
                        }
                    }
                    _ => {
                        return Err(Error(format!(
                            "predictor `{}` must be a factor or logical column",
                            self.name
                        )));
                    }
                };
                indices.get(level).copied().ok_or_else(|| {
                    Error(format!(
                        "unseen level `{level}` in predictor `{}`",
                        self.name
                    ))
                })
            })
            .collect()
    }

    fn level(&self, row: usize) -> Result<GroupLevel> {
        Ok(match &self.values {
            ColumnValues::Real32(v) if v[row].is_finite() => GroupLevel::Real(f64::from(v[row])),
            ColumnValues::Real64(v) if v[row].is_finite() => GroupLevel::Real(v[row]),
            ColumnValues::Int(v) => GroupLevel::Integer(v[row]),
            ColumnValues::Bool(v) => GroupLevel::Text(if v[row] { "TRUE" } else { "FALSE" }.into()),
            ColumnValues::Factor { levels, codes, .. } => {
                GroupLevel::Text(levels[codes[row] as usize].clone())
            }
            _ => {
                return Err(Error(format!(
                    "group `{}` contains infinite values",
                    self.name
                )));
            }
        })
    }

    fn levels(&self, rows: &[usize]) -> Result<Vec<GroupLevel>> {
        if let ColumnValues::Factor { levels, codes, .. } = &self.values {
            let observed: HashSet<_> = rows.iter().map(|&row| codes[row] as usize).collect();
            return Ok(levels
                .iter()
                .enumerate()
                .filter(|(i, _)| observed.contains(i))
                .map(|(_, label)| GroupLevel::Text(label.clone()))
                .collect());
        }
        let mut levels = rows
            .iter()
            .map(|&i| self.level(i))
            .collect::<Result<Vec<_>>>()?;
        match &self.values {
            ColumnValues::Int(_) => levels.sort_by_key(|value| {
                let GroupLevel::Integer(value) = value else {
                    unreachable!()
                };
                *value
            }),
            ColumnValues::Bool(_) => levels.sort_by(|a, b| {
                let (GroupLevel::Text(a), GroupLevel::Text(b)) = (a, b) else {
                    unreachable!()
                };
                a.cmp(b)
            }),
            _ => levels.sort_by(|a, b| {
                let (GroupLevel::Real(a), GroupLevel::Real(b)) = (a, b) else {
                    unreachable!()
                };
                a.total_cmp(b)
            }),
        }
        levels.dedup();
        Ok(levels)
    }
}

pub(crate) fn model_rows(
    formula: &Formula,
    predictors: &[&Formula],
    table: &Table,
    mode: DataMode,
) -> Result<Vec<usize>> {
    table.validate()?;
    if predictors
        .iter()
        .any(|formula| formula.uses_intercept_column())
        && let Some(column) = table
            .columns
            .iter()
            .find(|column| column.name == "Intercept")
    {
        let rows: Vec<_> = (0..table.nrows).collect();
        if rows.iter().any(|&row| !column.present(row))
            || column.numeric(&rows)?.iter().any(|&value| value != 1.0)
        {
            return Err(Error(
                "reserved data column `Intercept` must contain only ones".into(),
            ));
        }
    }
    let mut se_columns = Vec::new();
    if let Some(se) = &formula.se {
        se.value.columns(&mut se_columns);
    }
    let names = predictors
        .iter()
        .flat_map(|formula| {
            formula.predictor_columns().into_iter().chain(
                formula
                    .groups
                    .iter()
                    .flat_map(|group| group.columns.iter().map(String::as_str)),
            )
        })
        .chain(match &formula.trials {
            Some(Trials::Column(name)) => Some(name.as_str()),
            _ => None,
        })
        .chain(
            (mode == DataMode::Training)
                .then_some(formula.weights.as_deref())
                .flatten(),
        )
        .chain(se_columns)
        .chain((mode == DataMode::Training).then_some(formula.response.as_str()));
    let columns = names
        .map(|name| table.column(name).map_err(Error::from))
        .collect::<Result<Vec<_>>>()?;
    let mut rows: Vec<_> = (0..table.nrows)
        .filter(|&row| columns.iter().all(|column| column.present(row)))
        .collect();
    for expression in predictors
        .iter()
        .flat_map(|formula| {
            formula.expressions.values().chain(&formula.offsets).chain(
                formula
                    .groups
                    .iter()
                    .flat_map(|group| group.expressions.values()),
            )
        })
        .chain(formula.se.as_ref().map(|se| &se.value))
    {
        let values = expression.values(table, &rows)?;
        rows = rows
            .into_iter()
            .zip(values)
            .filter_map(|(row, value)| (!value.is_nan()).then_some(row))
            .collect();
    }
    if rows.is_empty() {
        return Err(Error("no complete model rows remain".into()));
    }
    Ok(rows)
}

/// Prepare a fresh snapshot using training means, levels and prior defaults.
/// Prediction needs neither responses nor weights and never refits the recipe.
pub fn prepare_data(recipe: &Recipe, table: &Table, mode: DataMode) -> Result<DataDocument> {
    let predictors: Vec<_> = recipe.predictors.values().map(|p| &p.formula).collect();
    let rows = model_rows(&recipe.formula, &predictors, table, mode)?;
    let mut constants = BTreeMap::new();
    constants.insert("N".into(), DataValue::Integer(rows.len() as i64));
    for (dpar, predictor) in recipe.predictors() {
        prepare_predictor(predictor, dpar, table, &rows, &mut constants)?;
    }
    if mode == DataMode::Training {
        constants.insert(
            "y_data".into(),
            response(
                recipe.family,
                recipe.response_levels.as_deref(),
                &recipe.formula,
                table,
                &rows,
            )?,
        );
        if let Some(name) = &recipe.formula.weights {
            let weights = table.column(name)?;
            if weights.numeric(&rows)?.iter().any(|&value| value < 0.0) {
                return Err(Error("response weights must be nonnegative".into()));
            }
            constants.insert("weights_data".into(), weights.data(&rows)?);
        }
    }
    if let Some(Trials::Column(name)) = &recipe.formula.trials {
        constants.insert(
            "trials_data".into(),
            DataValue::Integers(table.column(name)?.counts(&rows)?),
        );
    }
    if let Some(se) = &recipe.formula.se {
        let values = se.value.data(table, &rows)?;
        let valid = |value: f64| value >= 0.0 && (se.sigma || value > 0.0);
        let valid = match &values {
            DataValue::Real32(values) => values.iter().all(|&value| valid(f64::from(value))),
            DataValue::Real64(values) => values.iter().all(|&value| valid(value)),
            _ => unreachable!("numeric expressions produce real vectors"),
        };
        if !valid {
            return Err(Error(
                "se values must be nonnegative, and positive when sigma=FALSE".into(),
            ));
        }
        constants.insert("se_data".into(), values);
    }
    Ok(DataDocument { constants, rows })
}

fn prepare_predictor(
    recipe: &crate::PredictorRecipe,
    dpar: &str,
    table: &Table,
    rows: &[usize],
    constants: &mut BTreeMap<String, DataValue>,
) -> Result<()> {
    for (smooth, term) in recipe.smooths.iter().zip(&recipe.formula.smooths) {
        let column = table.column(&term.column)?;
        let values = smooth.basis.evaluate(&column.numeric(rows)?)?;
        for (j, name) in smooth.inputs.iter().enumerate() {
            let values = if matches!(&column.values, ColumnValues::Real32(_)) {
                DataValue::Real32(values.iter().map(|row| row[j] as f32).collect())
            } else {
                DataValue::Real64(values.iter().map(|row| row[j]).collect())
            };
            constants.insert(name.clone(), values);
        }
    }
    for (i, predictor) in recipe.design.predictors.iter().enumerate() {
        if let Some(expression) = recipe.formula.expressions.get(&predictor.column) {
            constants.insert(
                crate::emit::predictor_name(dpar, &format!("x_{}", i + 1)),
                expression.data(table, rows)?,
            );
            continue;
        }
        let column = table.column(&predictor.column)?;
        let values = match &predictor.factor {
            Some(factor) => DataValue::Integers(column.factor_codes(&factor.levels, rows)?),
            None => column.data(rows)?,
        };
        constants.insert(
            crate::emit::predictor_name(dpar, &format!("x_{}", i + 1)),
            values,
        );
    }
    if let Some(offset) = recipe.formula.offset() {
        constants.insert(
            crate::emit::predictor_name(dpar, "offset"),
            offset.data(table, rows)?,
        );
    }
    for (group, term) in recipe.groups.iter().zip(&recipe.formula.groups) {
        for (predictor, name) in group.design.predictors.iter().zip(&group.inputs) {
            if constants.contains_key(name) {
                continue;
            }
            let values = if let Some(expression) = term.expressions.get(&predictor.column) {
                expression.data(table, rows)?
            } else {
                let column = table.column(&predictor.column)?;
                match &predictor.factor {
                    Some(factor) => DataValue::Integers(column.factor_codes(&factor.levels, rows)?),
                    None => column.data(rows)?,
                }
            };
            constants.insert(name.clone(), values);
        }
        let levels: HashMap<_, _> = group
            .levels
            .iter()
            .enumerate()
            .map(|(index, level)| (level_key(level), index as i64 + 1))
            .collect();
        let indices = rows
            .iter()
            .map(|&row| {
                let level = group_level(&term.columns, table, row)?;
                levels.get(&level_key(&level)).copied().ok_or_else(|| {
                    Error(format!(
                        "unseen level {level:?} in group `{}`",
                        group.column
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        constants.insert(group.index.clone(), DataValue::Integers(indices));
    }
    Ok(())
}

pub(crate) fn response_levels(
    family: Family,
    column: &Column,
    rows: &[usize],
) -> Result<Option<Vec<GroupLevel>>> {
    if family == Family::Categorical {
        let levels = column.levels(rows)?;
        if levels.len() < 2 {
            return Err(Error(
                "categorical responses require at least two categories".into(),
            ));
        }
        return Ok(Some(levels));
    }
    if family == Family::Cumulative {
        let levels = match &column.values {
            ColumnValues::Factor { ordered: true, .. } => column.levels(rows)?,
            ColumnValues::Factor { .. } | ColumnValues::Bool(_) => {
                return Err(Error(
                    "cumulative responses require ordered factors or positive integer codes".into(),
                ));
            }
            _ => {
                let values = column.counts(rows)?;
                if values.contains(&0) {
                    return Err(Error("cumulative response codes must be positive".into()));
                }
                // Integer category labels retain gaps, unlike dropped factor levels.
                (1..=*values.iter().max().unwrap())
                    .map(GroupLevel::Integer)
                    .collect()
            }
        };
        if levels.len() < 2 {
            return Err(Error(
                "cumulative responses require at least two categories".into(),
            ));
        }
        return Ok(Some(levels));
    }
    if family != Family::Bernoulli {
        return Ok(None);
    }
    let mut levels = column.levels(rows)?;
    if !column.like_factor() && levels.len() == 1 {
        if level_key(&levels[0]) == LevelKey::Integer(0) {
            levels.push(GroupLevel::Integer(1));
        } else {
            levels.insert(0, GroupLevel::Integer(0));
        }
    }
    if levels.len() > 2 {
        return Err(Error(
            "Bernoulli responses must contain at most two distinct levels".into(),
        ));
    }
    Ok(Some(levels))
}

pub(crate) fn response(
    family: Family,
    levels: Option<&[GroupLevel]>,
    formula: &Formula,
    table: &Table,
    rows: &[usize],
) -> Result<DataValue> {
    let column = table.column(&formula.response)?;
    if matches!(
        family,
        Family::Bernoulli | Family::Cumulative | Family::Categorical
    ) {
        let first = i64::from(family != Family::Bernoulli);
        let indices: HashMap<_, _> = levels
            .expect("response levels were saved")
            .iter()
            .enumerate()
            .map(|(i, level)| (level_key(level), i as i64 + first))
            .collect();
        return Ok(DataValue::Integers(
            rows.iter()
                .map(|&row| {
                    let value = column.level(row)?;
                    indices
                        .get(&level_key(&value))
                        .copied()
                        .ok_or_else(|| Error(format!("unseen response level {value:?}")))
                })
                .collect::<Result<_>>()?,
        ));
    }
    if family.discrete() {
        let values = column.counts(rows)?;
        if family.uses_trials() {
            let trials = match formula
                .trials
                .as_ref()
                .expect("binomial trials were validated")
            {
                Trials::Constant(n) => vec![*n; rows.len()],
                Trials::Column(name) => table.column(name)?.counts(rows)?,
            };
            if values.iter().zip(trials).any(|(&y, n)| y > n) {
                return Err(Error(
                    "binomial responses must not exceed the number of trials".into(),
                ));
            }
        }
        return Ok(DataValue::Integers(values));
    }
    if matches!(
        family,
        Family::Gamma | Family::Exponential | Family::Lognormal | Family::Weibull
    ) && column.numeric(rows)?.iter().any(|&value| value <= 0.0)
    {
        return Err(Error(
            "Gamma, exponential, lognormal and Weibull responses must be strictly positive".into(),
        ));
    }
    if matches!(family, Family::HurdleGamma | Family::HurdleLognormal)
        && column.numeric(rows)?.iter().any(|&value| value < 0.0)
    {
        return Err(Error("hurdle responses must be nonnegative".into()));
    }
    if matches!(family, Family::Beta | Family::ZeroInflatedBeta)
        && column
            .numeric(rows)?
            .iter()
            .any(|&value| !(0.0..1.0).contains(&value) || (value == 0.0 && family == Family::Beta))
    {
        return Err(Error(
            "beta responses must lie between 0 and 1, with zero allowed only for zero inflation"
                .into(),
        ));
    }
    if family == Family::VonMises
        && column
            .numeric(rows)?
            .iter()
            .any(|value| !(-std::f64::consts::PI..=std::f64::consts::PI).contains(value))
    {
        return Err(Error(
            "von Mises responses must lie between -pi and pi".into(),
        ));
    }
    column.data(rows)
}

#[derive(PartialEq, Eq, Hash)]
enum LevelKey<'a> {
    Integer(i64),
    Real(u64),
    Text(&'a str),
    Interaction(Vec<LevelKey<'a>>),
}

fn level_key(level: &GroupLevel) -> LevelKey<'_> {
    match level {
        GroupLevel::Integer(value) => LevelKey::Integer(*value),
        GroupLevel::Real(value)
            if value.fract() == 0.0 && *value >= i64::MIN as f64 && *value < -(i64::MIN as f64) =>
        {
            LevelKey::Integer(*value as i64)
        }
        GroupLevel::Real(value) => LevelKey::Real(value.to_bits()),
        GroupLevel::Text(value) => LevelKey::Text(value),
        GroupLevel::Interaction(values) => {
            LevelKey::Interaction(values.iter().map(level_key).collect())
        }
    }
}

fn group_level(columns: &[String], table: &Table, row: usize) -> Result<GroupLevel> {
    if columns.len() == 1 {
        table.column(&columns[0])?.level(row)
    } else {
        columns
            .iter()
            .map(|name| table.column(name)?.level(row))
            .collect::<Result<Vec<_>>>()
            .map(GroupLevel::Interaction)
    }
}

pub(crate) fn group_levels(
    columns: &[String],
    table: &Table,
    rows: &[usize],
) -> Result<Vec<GroupLevel>> {
    if columns.len() == 1 {
        return table.column(&columns[0])?.levels(rows);
    }
    let mut levels = rows
        .iter()
        .map(|&row| group_level(columns, table, row))
        .collect::<Result<Vec<_>>>()?;
    // brms combines observed grouping values into labels before factorizing;
    // compound labels use C collation, not the input factors' level order.
    levels.sort_by_key(ToString::to_string);
    levels.dedup();
    if levels
        .windows(2)
        .any(|pair| pair[0].to_string() == pair[1].to_string())
    {
        return Err(Error(format!(
            "group `{}` has ambiguous combined labels",
            columns.join(":")
        )));
    }
    Ok(levels)
}
