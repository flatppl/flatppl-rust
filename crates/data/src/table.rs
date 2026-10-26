//! Typed columnar tables: what an Arrow source decodes to, and what the formula
//! frontend builds from delimited or JSON text.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

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
    Bool(Vec<bool>),
    /// Zero-based codes. Explicit level order is preserved.
    Factor {
        levels: Vec<String>,
        codes: Vec<u32>,
        #[serde(default)]
        ordered: bool,
    },
}

impl Table {
    /// The column named `name`.
    pub fn column(&self, name: &str) -> Result<&Column> {
        self.columns
            .iter()
            .find(|column| column.name == name)
            .ok_or_else(|| Error(format!("missing data column `{name}`")))
    }

    /// Check distinct names, consistent lengths and valid factor codes.
    pub fn validate(&self) -> Result<()> {
        let mut names = HashSet::new();
        for column in &self.columns {
            if !names.insert(&column.name) {
                return Err(Error(format!("duplicate data column `{}`", column.name)));
            }
            let len = match &column.values {
                ColumnValues::Real32(v) => v.len(),
                ColumnValues::Real64(v) => v.len(),
                ColumnValues::Int(v) => v.len(),
                ColumnValues::Bool(v) => v.len(),
                ColumnValues::Factor { levels, codes, .. } => {
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
