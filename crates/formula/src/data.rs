use crate::{Error, Formula, GroupLevel, Recipe, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Table {
    pub columns: Vec<Column>,
    pub nrows: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Column {
    pub name: String,
    pub values: ColumnValues,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid: Option<Vec<bool>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", content = "values", rename_all = "snake_case")]
pub enum ColumnValues {
    Real32(Vec<f32>),
    Real64(Vec<f64>),
    Int(Vec<i64>),
    /// Zero-based codes. Explicit level order is preserved.
    Factor {
        levels: Vec<String>,
        codes: Vec<u32>,
    },
}

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

impl Table {
    pub(crate) fn column(&self, name: &str) -> Result<&Column> {
        self.columns
            .iter()
            .find(|column| column.name == name)
            .ok_or_else(|| Error(format!("missing data column `{name}`")))
    }

    fn validate(&self) -> Result<()> {
        let mut names = HashSet::new();
        for column in &self.columns {
            if !names.insert(&column.name) {
                return Err(Error(format!("duplicate data column `{}`", column.name)));
            }
            let len = match &column.values {
                ColumnValues::Real32(v) => v.len(),
                ColumnValues::Real64(v) => v.len(),
                ColumnValues::Int(v) => v.len(),
                ColumnValues::Factor { levels, codes } => {
                    if levels.iter().collect::<HashSet<_>>().len() != levels.len() {
                        return Err(Error(format!("duplicate levels in `{}`", column.name)));
                    }
                    if codes.iter().enumerate().any(|(i, &code)| {
                        column
                            .valid
                            .as_ref()
                            .is_none_or(|mask| mask.get(i) == Some(&true))
                            && code as usize >= levels.len()
                    }) {
                        return Err(Error(format!("invalid factor code in `{}`", column.name)));
                    }
                    codes.len()
                }
            };
            if len != self.nrows || column.valid.as_ref().is_some_and(|v| v.len() != len) {
                return Err(Error(format!(
                    "column `{}` has inconsistent row or validity lengths",
                    column.name
                )));
            }
        }
        Ok(())
    }
}

impl Column {
    fn present(&self, row: usize) -> bool {
        self.valid.as_ref().is_none_or(|v| v[row])
            && match &self.values {
                ColumnValues::Real32(v) => !v[row].is_nan(),
                ColumnValues::Real64(v) => !v[row].is_nan(),
                _ => true,
            }
    }

    pub(crate) fn numeric(&self, rows: &[usize]) -> Result<Vec<f64>> {
        let values: Vec<_> = match &self.values {
            ColumnValues::Real32(v) => rows.iter().map(|&i| f64::from(v[i])).collect(),
            ColumnValues::Real64(v) => rows.iter().map(|&i| v[i]).collect(),
            ColumnValues::Int(v) => rows.iter().map(|&i| v[i] as f64).collect(),
            ColumnValues::Factor { .. } => {
                return Err(Error(format!(
                    "factor predictor or response `{}` is unsupported in the Gaussian numeric subset",
                    self.name
                )));
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

    pub(crate) fn mean(&self, rows: &[usize]) -> Result<f64> {
        let values = self.numeric(rows)?;
        let mean = values.iter().sum::<f64>() / rows.len() as f64;
        if !mean.is_finite() {
            return Err(Error(format!(
                "column `{}` has an unrepresentable mean",
                self.name
            )));
        }
        Ok(mean)
    }

    fn data(&self, rows: &[usize]) -> Result<DataValue> {
        let values = match &self.values {
            ColumnValues::Real32(v) => DataValue::Real32(rows.iter().map(|&i| v[i]).collect()),
            ColumnValues::Real64(v) => DataValue::Real64(rows.iter().map(|&i| v[i]).collect()),
            ColumnValues::Int(v) => DataValue::Integers(rows.iter().map(|&i| v[i]).collect()),
            ColumnValues::Factor { .. } => {
                return Err(Error(format!(
                    "factor predictor or response `{}` is unsupported in the Gaussian numeric subset",
                    self.name
                )));
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

    fn level(&self, row: usize) -> Result<GroupLevel> {
        Ok(match &self.values {
            ColumnValues::Real32(v) if v[row].is_finite() => GroupLevel::Real(f64::from(v[row])),
            ColumnValues::Real64(v) if v[row].is_finite() => GroupLevel::Real(v[row]),
            ColumnValues::Int(v) => GroupLevel::Integer(v[row]),
            ColumnValues::Factor { levels, codes } => {
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

    pub(crate) fn levels(&self, rows: &[usize]) -> Result<Vec<GroupLevel>> {
        if let ColumnValues::Factor { levels, codes } = &self.values {
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

pub(crate) fn model_rows(formula: &Formula, table: &Table, mode: DataMode) -> Result<Vec<usize>> {
    table.validate()?;
    let names = formula
        .fixed
        .iter()
        .chain(&formula.groups)
        .chain((mode == DataMode::Training).then_some(&formula.response));
    let columns = names
        .map(|name| table.column(name))
        .collect::<Result<Vec<_>>>()?;
    let rows: Vec<_> = (0..table.nrows)
        .filter(|&row| columns.iter().all(|column| column.present(row)))
        .collect();
    if rows.is_empty() {
        return Err(Error("no complete model rows remain".into()));
    }
    Ok(rows)
}

/// Prepare a fresh snapshot using training means, levels and prior defaults.
/// Prediction does not require a response column and never refits the recipe.
pub fn prepare_data(recipe: &Recipe, table: &Table, mode: DataMode) -> Result<DataDocument> {
    let rows = model_rows(&recipe.formula, table, mode)?;
    let mut constants = BTreeMap::new();
    constants.insert("N".into(), DataValue::Integer(rows.len() as i64));
    for (i, name) in recipe.formula.fixed.iter().enumerate() {
        constants.insert(format!("x_{}", i + 1), table.column(name)?.data(&rows)?);
    }
    if mode == DataMode::Training {
        constants.insert(
            "y_data".into(),
            table.column(&recipe.formula.response)?.data(&rows)?,
        );
    }
    for group in &recipe.groups {
        let column = table.column(&group.column)?;
        let levels: HashMap<_, _> = group
            .levels
            .iter()
            .enumerate()
            .map(|(index, level)| (level_key(level), index as i64 + 1))
            .collect();
        let indices = rows
            .iter()
            .map(|&row| {
                let level = column.level(row)?;
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
    Ok(DataDocument { constants, rows })
}

#[derive(PartialEq, Eq, Hash)]
enum LevelKey<'a> {
    Integer(i64),
    Real(u64),
    Text(&'a str),
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
    }
}
