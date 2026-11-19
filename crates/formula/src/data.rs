use crate::{
    Error, Family, Formula, GroupInfo, GroupLevel, Recipe, Result, Trials, arma, autoregression,
    formula::ThresholdGroup,
};
use flatppl_data::{Column, ColumnValues, Table};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

pub(crate) const THRESHOLD_GROUP_INDEX: &str = "threshold_group_index_data";

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

/// Shapes of standard normal innovations to append on the last axis of each
/// affected group parameter. The fitted labels remain the leading entries.
pub type NewGroupShapes = BTreeMap<String, Vec<usize>>;

/// A prediction template and its prepared future rows and history anchors.
pub struct PreparedForecast {
    pub module: flatppl_core::Module,
    pub data: DataDocument,
    pub new_group_shapes: NewGroupShapes,
}

impl PreparedForecast {
    /// Attach the fitted model's parameter names and training metadata.
    pub fn attach_model_info(&mut self, info: &crate::ModelInfo) -> Result<()> {
        crate::metadata::emit(&mut self.module, info)
    }
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
                DataValue::Integer(_) if name == "N" || name == autoregression::DESIGN_N => {
                    continue;
                }
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
    extra_columns: &[String],
    table: &Table,
    mode: DataMode,
) -> Result<Vec<usize>> {
    observation_rows(formula, predictors, extra_columns, table, mode, false)
}

fn observation_rows(
    formula: &Formula,
    predictors: &[&Formula],
    extra_columns: &[String],
    table: &Table,
    mode: DataMode,
    history: bool,
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
            formula
                .threshold_group
                .as_ref()
                .map(|group| group.column.as_str()),
        )
        .chain(
            (mode == DataMode::Training)
                .then_some(formula.weights.as_deref())
                .flatten(),
        )
        .chain(se_columns)
        .chain(
            formula
                .truncation
                .iter()
                .flat_map(crate::Truncation::columns),
        )
        .chain(formula.censoring.iter().filter_map(|censoring| {
            (mode == DataMode::Training).then_some(censoring.code.as_str())
        }))
        .chain(
            (mode == DataMode::Training)
                .then_some(formula.decision.as_deref())
                .flatten(),
        )
        .chain(extra_columns.iter().map(String::as_str))
        .chain((mode == DataMode::Training || history).then_some(formula.response.as_str()));
    let columns = names
        .map(|name| table.column(name).map_err(Error::from))
        .collect::<Result<Vec<_>>>()?;
    let mut rows: Vec<_> = (0..table.nrows)
        .filter(|&row| columns.iter().all(|column| column.present(row)))
        .collect();
    for expression in predictors
        .iter()
        .flat_map(|formula| {
            formula
                .expressions
                .values()
                .chain(
                    formula
                        .monotonic_terms
                        .iter()
                        .flat_map(|terms| terms.expressions.values()),
                )
                .chain(&formula.offsets)
                .chain(
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
        if history {
            return Ok(rows);
        }
        return Err(Error("no complete model rows remain".into()));
    }
    if let Some(ar) = &formula.ar {
        let group = table.column(&ar.group)?;
        let time = table.column(&ar.time)?;
        if rows
            .iter()
            .any(|&row| !group.present(row) || !time.present(row))
        {
            return Err(Error("ar() group and time cannot be missing".into()));
        }
        let levels = group.levels(&rows)?;
        let indices: HashMap<_, _> = levels
            .iter()
            .enumerate()
            .map(|(index, level)| (level_key(level), index))
            .collect();
        let mut ordered = rows
            .into_iter()
            .map(|row| {
                let level = group.level(row)?;
                Ok((indices[&level_key(&level)], ar_time(time, row)?, row))
            })
            .collect::<Result<Vec<_>>>()?;
        ordered.sort_unstable();
        rows = ordered.into_iter().map(|(_, _, row)| row).collect();
    }
    Ok(rows)
}

fn ar_time(column: &Column, row: usize) -> Result<i64> {
    match column.level(row)? {
        GroupLevel::Integer(value) => Ok(value),
        GroupLevel::Real(value)
            if value.fract() == 0.0 && value >= i64::MIN as f64 && value < -(i64::MIN as f64) =>
        {
            Ok(value as i64)
        }
        _ => Err(Error("ar() time must contain integer-valued times".into())),
    }
}

fn prepare_ar(
    ar: &crate::Ar1,
    conditional: bool,
    table: &Table,
    rows: &[usize],
    constants: &mut BTreeMap<String, DataValue>,
) -> Result<()> {
    let groups = table.column(&ar.group)?;
    let times = table.column(&ar.time)?;
    let mut gaps = vec![0];
    let mut predecessors = vec![1];
    let mut previous_group = groups.level(rows[0])?;
    let mut previous_time = ar_time(times, rows[0])?;
    for (i, &row) in rows.iter().enumerate().skip(1) {
        let group = groups.level(row)?;
        let time = ar_time(times, row)?;
        if group == previous_group {
            let gap = ar_gap(time, previous_time)?;
            if gap == 0 {
                return Err(Error("ar() times must be unique within a group".into()));
            }
            if conditional && gap != 1 {
                return Err(Error(
                    "conditional ARMA requires consecutive times within each group".into(),
                ));
            }
            gaps.push(gap);
            predecessors.push(i as i64);
        } else {
            gaps.push(0);
            predecessors.push(i as i64 + 1);
        }
        previous_group = group;
        previous_time = time;
    }
    constants.insert(crate::autoregression::GAP.into(), DataValue::Integers(gaps));
    constants.insert(
        crate::autoregression::PREVIOUS.into(),
        DataValue::Integers(predecessors),
    );
    Ok(())
}

fn ar_gap(time: i64, previous: i64) -> Result<i64> {
    time.checked_sub(previous)
        .filter(|&gap| gap <= i32::MAX as i64)
        .ok_or_else(|| Error("ar() time gap exceeds the supported integer range".into()))
}

/// Prepare a fresh snapshot using training means, levels and prior defaults.
/// Prediction needs neither responses nor weights and never refits the recipe.
pub fn prepare_data(recipe: &Recipe, table: &Table, mode: DataMode) -> Result<DataDocument> {
    prepare_data_inner(recipe, table, mode, None)
}

/// Prepare prediction data, assigning unseen labels indices after the fitted
/// levels in each group and reporting the corresponding innovation shapes.
pub fn prepare_data_with_new_groups(
    recipe: &Recipe,
    table: &Table,
) -> Result<(DataDocument, NewGroupShapes)> {
    let mut shapes = NewGroupShapes::new();
    let data = prepare_data_inner(recipe, table, DataMode::Prediction, Some(&mut shapes))?;
    Ok((data, shapes))
}

/// Prepare strictly future AR(1) rows conditional on the supplied history.
/// The history is encoded with the fitted recipe. Groups absent from the
/// history start from stationary AR(1) residuals or zero conditional ARMA state.
pub fn prepare_forecast(
    recipe: &Recipe,
    future: &Table,
    history: &Table,
    allow_new_levels: bool,
) -> Result<PreparedForecast> {
    let mut recipe = recipe.clone();
    if recipe.formula.ar.is_none() {
        return Err(Error("history requires a Gaussian AR or MA formula".into()));
    }
    let shared = recipe.shared_groups.clone();
    let mut module = crate::emit::model_with_forecast(&mut recipe, &shared, true)?;
    let mut shapes = NewGroupShapes::new();
    let groups = allow_new_levels.then_some(&mut shapes);
    let mut data = prepare_data_inner(&recipe, future, DataMode::Prediction, groups)?;
    add_history(&recipe, future, history, &mut data)?;
    if !shapes.is_empty() {
        module = recipe.extend_group_draws(module, &shapes)?;
    }
    Ok(PreparedForecast {
        module,
        data,
        new_group_shapes: shapes,
    })
}

fn add_history(
    recipe: &Recipe,
    future: &Table,
    history: &Table,
    data: &mut DataDocument,
) -> Result<()> {
    let ar = recipe.formula.ar.as_ref().expect("validated AR formula");
    let predictors: Vec<_> = recipe.predictors.values().map(|p| &p.formula).collect();
    let nonlinear = recipe
        .nonlinear
        .as_ref()
        .map_or(&[][..], |p| p.columns.as_slice());
    let history_rows = observation_rows(
        &recipe.formula,
        &predictors,
        nonlinear,
        history,
        DataMode::Prediction,
        true,
    )?;
    if !recipe.formula.arma_order.stationary_ar1() {
        return add_arma_history(recipe, future, history, &history_rows, nonlinear, data);
    }
    let history_groups = history.column(&ar.group)?;
    let history_times = history.column(&ar.time)?;
    let response = history.column(&recipe.formula.response)?;
    let levels = history_rows
        .iter()
        .map(|&row| history_groups.level(row))
        .collect::<Result<Vec<_>>>()?;
    let mut last = HashMap::new();
    for (&row, level) in history_rows.iter().zip(&levels) {
        let time = ar_time(history_times, row)?;
        if let Some((previous, _)) = last.insert(level_key(level), (time, row))
            && time == previous
        {
            return Err(Error(
                "ar() history times must be unique within a group".into(),
            ));
        }
    }

    let groups = future.column(&ar.group)?;
    let times = future.column(&ar.time)?;
    let mut anchors = Vec::with_capacity(data.rows.len());
    let mut elapsed = Vec::with_capacity(data.rows.len());
    let mut observed = Vec::with_capacity(data.rows.len());
    let mut anchor_rows = Vec::new();
    let DataValue::Integers(mut previous) =
        data.constants.remove(autoregression::PREVIOUS).unwrap()
    else {
        unreachable!("AR preparation produces integer predecessors");
    };
    let DataValue::Integers(gaps) = data.constants.get_mut(autoregression::GAP).unwrap() else {
        unreachable!("AR preparation produces integer gaps");
    };
    let n = data.rows.len() as i64;
    for (i, &row) in data.rows.iter().enumerate() {
        let group = groups.level(row)?;
        let time = ar_time(times, row)?;
        if let Some(&(last_time, anchor_row)) = last.get(&level_key(&group)) {
            if time <= last_time {
                return Err(Error(format!(
                    "ar() forecast time {time} must follow the latest history time {last_time} in group `{group}`"
                )));
            }
            let gap = ar_gap(time, last_time)?;
            let (index, value) = if gaps[i] == 0 {
                let index = n + anchor_rows.len() as i64 + 1;
                let value = response.numeric(&[anchor_row])?[0];
                anchor_rows.push(anchor_row);
                gaps[i] = gap;
                previous[i] = index;
                (index, value)
            } else {
                (anchors[i - 1], observed[i - 1])
            };
            anchors.push(index);
            elapsed.push(gap);
            observed.push(value);
        } else {
            anchors.push(i as i64 + 1);
            elapsed.push(0);
            observed.push(0.0);
        }
    }
    append_history_design(recipe, history, &anchor_rows, nonlinear, data)?;
    data.constants.extend([
        (
            autoregression::DESIGN_N.into(),
            DataValue::Integer(n + anchor_rows.len() as i64),
        ),
        (
            autoregression::PREVIOUS.into(),
            DataValue::Integers(previous),
        ),
        (autoregression::ANCHOR.into(), DataValue::Integers(anchors)),
        (autoregression::ELAPSED.into(), DataValue::Integers(elapsed)),
        (autoregression::OBSERVED.into(), DataValue::Real64(observed)),
    ]);
    Ok(())
}

fn add_arma_history(
    recipe: &Recipe,
    future: &Table,
    history: &Table,
    history_rows: &[usize],
    nonlinear: &[String],
    data: &mut DataDocument,
) -> Result<()> {
    let ar = recipe.formula.ar.as_ref().unwrap();
    let history_groups = history.column(&ar.group)?;
    let history_times = history.column(&ar.time)?;
    let response = history.column(&recipe.formula.response)?;
    let levels = history_rows
        .iter()
        .map(|&row| history_groups.level(row))
        .collect::<Result<Vec<_>>>()?;
    let mut grouped: HashMap<_, Vec<usize>> = HashMap::new();
    for (&row, level) in history_rows.iter().zip(&levels) {
        grouped.entry(level_key(level)).or_default().push(row);
    }
    let groups = future.column(&ar.group)?;
    let times = future.column(&ar.time)?;
    let DataValue::Integers(gaps) = &data.constants[autoregression::GAP] else {
        unreachable!("AR preparation produces integer gaps");
    };
    let n = data.rows.len() as i64;
    let mut prepared_history = Vec::new();
    let mut order = Vec::new();
    let mut future_indices = Vec::with_capacity(data.rows.len());
    let mut observed = Vec::new();
    let mut steps = Vec::new();
    for (i, (&row, &gap)) in data.rows.iter().zip(gaps).enumerate() {
        let mut step = gap;
        if gap == 0 {
            let group = groups.level(row)?;
            if let Some(rows) = grouped.get(&level_key(&group)) {
                let mut previous = None;
                for (&history_row, value) in rows.iter().zip(response.numeric(rows)?) {
                    let time = ar_time(history_times, history_row)?;
                    if previous.is_some_and(|previous| time.checked_sub(previous) != Some(1)) {
                        return Err(Error(
                            "conditional ARMA requires consecutive history times within each group"
                                .into(),
                        ));
                    }
                    steps.push(i64::from(previous.is_some()));
                    previous = Some(time);
                    prepared_history.push(history_row);
                    order.push(n + prepared_history.len() as i64);
                    observed.push(value);
                }
                let time = ar_time(times, row)?;
                if previous.and_then(|last| time.checked_sub(last)) != Some(1) {
                    return Err(Error("conditional ARMA forecasts must start one time step after history within each group".into()));
                }
                step = 1;
            }
        }
        order.push(i as i64 + 1);
        future_indices.push(order.len() as i64);
        observed.push(0.0);
        steps.push(step);
    }
    append_history_design(recipe, history, &prepared_history, nonlinear, data)?;
    data.constants.extend([
        (
            autoregression::DESIGN_N.into(),
            DataValue::Integer(order.len() as i64),
        ),
        (arma::ORDER.into(), DataValue::Integers(order)),
        (arma::FUTURE.into(), DataValue::Integers(future_indices)),
        (arma::HISTORY.into(), DataValue::Real64(observed)),
        (arma::STEP.into(), DataValue::Integers(steps)),
    ]);
    Ok(())
}

fn append_history_design(
    recipe: &Recipe,
    history: &Table,
    rows: &[usize],
    nonlinear: &[String],
    data: &mut DataDocument,
) -> Result<()> {
    if !rows.is_empty() {
        let mut design = BTreeMap::new();
        // History uses the fitted group coordinates before any new levels.
        prepare_design(recipe, history, rows, nonlinear, &mut design, None)?;
        for (name, history) in design {
            append_history(data.constants.get_mut(&name).unwrap(), history);
        }
    }
    Ok(())
}

fn append_history(future: &mut DataValue, history: DataValue) {
    match (future, history) {
        (DataValue::Real32(a), DataValue::Real32(b)) => a.extend(b),
        (DataValue::Real64(a), DataValue::Real64(b)) => a.extend(b),
        (DataValue::Integers(a), DataValue::Integers(b)) => a.extend(b),
        (future, history) => {
            let reals = |data| match data {
                DataValue::Real32(v) => v.into_iter().map(f64::from).collect(),
                DataValue::Integers(v) => v.into_iter().map(|value| value as f64).collect(),
                DataValue::Real64(v) => v,
                DataValue::Integer(_) => unreachable!("predictor inputs are vectors"),
            };
            let mut values = reals(std::mem::replace(future, DataValue::Real64(Vec::new())));
            values.extend(reals(history));
            *future = DataValue::Real64(values);
        }
    }
}

fn prepare_data_inner(
    recipe: &Recipe,
    table: &Table,
    mode: DataMode,
    new_groups: Option<&mut NewGroupShapes>,
) -> Result<DataDocument> {
    let predictors: Vec<_> = recipe.predictors.values().map(|p| &p.formula).collect();
    let nonlinear_columns = recipe
        .nonlinear
        .as_ref()
        .map_or(&[][..], |p| p.columns.as_slice());
    let required_columns = if mode == DataMode::Training && recipe.paired_response.is_some() {
        recipe.paired_response.as_slice()
    } else {
        nonlinear_columns
    };
    let rows = model_rows(&recipe.formula, &predictors, required_columns, table, mode)?;
    let mut constants = BTreeMap::new();
    constants.insert("N".into(), DataValue::Integer(rows.len() as i64));
    prepare_design(
        recipe,
        table,
        &rows,
        nonlinear_columns,
        &mut constants,
        new_groups,
    )?;
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
        if recipe.paired_response.is_some() {
            let second = &recipe.predictors[crate::paired::SECOND_MEAN].formula;
            constants.insert(
                crate::paired::SECOND_DATA.into(),
                response(
                    recipe.second_family(),
                    recipe.paired_response_levels.as_deref(),
                    second,
                    table,
                    &rows,
                )?,
            );
        }
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
    if let Some(group) = &recipe.threshold_groups {
        let column = table.column(&group.name)?;
        let indices = threshold_group_indices(column, &group.levels, &rows)?;
        constants.insert(THRESHOLD_GROUP_INDEX.into(), DataValue::Integers(indices));
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
    crate::response::prepare(
        &recipe.formula,
        recipe.family,
        table,
        &rows,
        mode,
        &mut constants,
    )?;
    if let Some(ar) = &recipe.formula.ar {
        let conditional = !recipe.formula.arma_order.stationary_ar1();
        prepare_ar(ar, conditional, table, &rows, &mut constants)?;
    }
    Ok(DataDocument { constants, rows })
}

fn prepare_design(
    recipe: &Recipe,
    table: &Table,
    rows: &[usize],
    nonlinear_columns: &[String],
    constants: &mut BTreeMap<String, DataValue>,
    mut new_groups: Option<&mut NewGroupShapes>,
) -> Result<()> {
    for (i, name) in nonlinear_columns.iter().enumerate() {
        let values = crate::ColumnExpression::Column(name.clone()).data(table, rows)?;
        constants.insert(crate::nonlinear::data_name(i), values);
    }
    for (i, name) in recipe.formula.category_specific.iter().enumerate() {
        let values = crate::ColumnExpression::Column(name.clone()).data(table, rows)?;
        constants.insert(crate::adjacent::data_name(i), values);
    }
    for (dpar, predictor) in recipe.predictors() {
        prepare_predictor(
            predictor,
            &recipe.predictor_suffix(dpar),
            table,
            rows,
            constants,
            new_groups.as_deref_mut(),
        )?;
    }
    Ok(())
}

fn prepare_predictor(
    recipe: &crate::PredictorRecipe,
    dpar: &str,
    table: &Table,
    rows: &[usize],
    constants: &mut BTreeMap<String, DataValue>,
    mut new_groups: Option<&mut NewGroupShapes>,
) -> Result<()> {
    if let Some(design) = &recipe.monotonic_design {
        let terms = recipe
            .formula
            .monotonic_terms
            .as_ref()
            .ok_or_else(|| Error("invalid retained monotonic design".into()))?;
        design.prepare(terms, dpar, table, rows, constants)?;
    }
    for (i, (coding, column)) in recipe
        .monotonic
        .iter()
        .zip(&recipe.formula.monotonic)
        .enumerate()
    {
        let codes = coding.codes(table.column(column)?, rows)?;
        let name = crate::emit::predictor_name(dpar, &format!("mo_{}", i + 1));
        constants.insert(name, DataValue::Integers(codes));
    }
    for (smooth, term) in recipe.smooths.iter().zip(&recipe.formula.smooths) {
        if let Some(by) = &smooth.by {
            let name = term
                .by
                .as_deref()
                .ok_or_else(|| Error("invalid retained smooth by term".into()))?;
            let column = table.column(name)?;
            if !matches!(&column.values, ColumnValues::Factor { .. }) {
                return Err(Error(format!("smooth by column `{name}` must be a factor")));
            }
            let codes = column.factor_codes(&by.levels, rows)?;
            constants.insert(by.index.clone(), DataValue::Integers(codes));
        }
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
    for (smooth, term) in recipe
        .tensor_smooths
        .iter()
        .zip(&recipe.formula.tensor_smooths)
    {
        let values = smooth.values(term, table, rows)?;
        let inputs = smooth
            .fixed
            .iter()
            .chain(smooth.blocks.iter().flat_map(|block| &block.inputs));
        for (j, name) in inputs.enumerate() {
            constants.insert(
                name.clone(),
                DataValue::Real64(values.iter().map(|row| row[j]).collect()),
            );
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
        if constants.contains_key(&group.index) {
            continue;
        }
        let mut levels: HashMap<_, _> = group
            .levels
            .iter()
            .enumerate()
            .map(|(index, level)| (level_key(level), index as i64 + 1))
            .collect();
        let indices = if let Some(shapes) = new_groups.as_deref_mut() {
            let row_levels = rows
                .iter()
                .map(|&row| group_level(&term.columns, table, row))
                .collect::<Result<Vec<_>>>()?;
            let fitted_count = levels.len();
            let indices = row_levels
                .iter()
                .map(|level| {
                    let index = levels.len() as i64 + 1;
                    *levels.entry(level_key(level)).or_insert(index)
                })
                .collect();
            let new_count = levels.len() - fitted_count;
            if new_count > 0 {
                let coefficients = group.innovation_width();
                let shape = if coefficients == 1 {
                    vec![new_count]
                } else {
                    vec![coefficients, new_count]
                };
                shapes.insert(group.z.clone(), shape);
            }
            indices
        } else {
            rows.iter()
                .map(|&row| {
                    let level = group_level(&term.columns, table, row)?;
                    levels.get(&level_key(&level)).copied().ok_or_else(|| {
                        Error(format!(
                            "unseen level {level:?} in group `{}`",
                            group.column
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?
        };
        constants.insert(group.index.clone(), DataValue::Integers(indices));
    }
    Ok(())
}

pub(crate) fn fit_threshold_groups(
    group: &ThresholdGroup,
    column: &Column,
    response: &Column,
    rows: &[usize],
    common_levels: &[GroupLevel],
) -> Result<GroupInfo> {
    let levels = column.levels(rows)?;
    let count = common_levels.len() - 1;
    if let Some(requested) = group.count {
        if requested != count {
            return Err(Error(format!(
                "thres() requested {requested} thresholds, but the retained response has {count}"
            )));
        }
    } else {
        let mut group_rows = vec![Vec::new(); levels.len()];
        for (&row, index) in rows
            .iter()
            .zip(threshold_group_indices(column, &levels, rows)?)
        {
            group_rows[index as usize - 1].push(row);
        }
        for (level, rows) in levels.iter().zip(group_rows) {
            if !matches!(response_levels(Family::Cumulative, response, &rows), Ok(Some(ref inferred)) if inferred == common_levels)
            {
                return Err(Error(format!(
                    "threshold group `{level}` in `{}` does not share the fitted response levels; use thres({count}, gr={}) to use the common response scale",
                    column.name, column.name
                )));
            }
        }
    }
    Ok(GroupInfo {
        name: group.column.clone(),
        levels,
    })
}

fn threshold_group_indices(
    column: &Column,
    levels: &[GroupLevel],
    rows: &[usize],
) -> Result<Vec<i64>> {
    let indices: HashMap<_, _> = levels
        .iter()
        .enumerate()
        .map(|(i, level)| (level_key(level), i as i64 + 1))
        .collect();
    rows.iter()
        .map(|&row| {
            let level = column.level(row)?;
            indices.get(&level_key(&level)).copied().ok_or_else(|| {
                Error(format!(
                    "unseen threshold group `{level}` in `{}`",
                    column.name
                ))
            })
        })
        .collect()
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
    if family.ordinal() {
        let name = if family == Family::Cumulative {
            "cumulative"
        } else {
            "acat"
        };
        let levels = match &column.values {
            ColumnValues::Factor { ordered: true, .. } => column.levels(rows)?,
            ColumnValues::Factor { .. } | ColumnValues::Bool(_) => {
                return Err(Error(format!(
                    "{name} responses require ordered factors or positive integer codes"
                )));
            }
            _ => {
                let values = column.counts(rows)?;
                if values.contains(&0) {
                    return Err(Error(format!("{name} response codes must be positive")));
                }
                // Integer category labels retain gaps, unlike dropped factor levels.
                (1..=*values.iter().max().unwrap())
                    .map(GroupLevel::Integer)
                    .collect()
            }
        };
        if levels.len() < 2 {
            return Err(Error(format!(
                "{name} responses require at least two categories"
            )));
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
        Family::Bernoulli | Family::Cumulative | Family::Acat | Family::Categorical
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
    if family == Family::Wiener && column.numeric(rows)?.iter().any(|&time| time <= 0.0) {
        return Err(Error(
            "Wiener response times must be strictly positive".into(),
        ));
    }
    if matches!(family, Family::Exponential | Family::Weibull) && formula.censoring.is_some() {
        if column.numeric(rows)?.iter().any(|&value| value < 0.0) {
            return Err(Error("survival times must be nonnegative".into()));
        }
    } else if matches!(
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
