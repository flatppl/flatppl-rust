use crate::{ColumnValues, DataValue, Error, Result, Table};
use serde::{Deserialize, Serialize};
use std::fmt;

/// An elementwise numeric expression retained for new-data preparation.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum ColumnExpression {
    Column(String),
    Number(f64),
    Identity(Box<Self>),
    Log(Box<Self>),
    Negate(Box<Self>),
    Parentheses(Box<Self>),
    Binary(Arithmetic, Box<Self>, Box<Self>),
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Arithmetic {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
}

impl ColumnExpression {
    pub(crate) fn columns<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Self::Column(name) => {
                if !out.contains(&name.as_str()) {
                    out.push(name);
                }
            }
            Self::Number(_) => {}
            Self::Identity(x) | Self::Log(x) | Self::Negate(x) | Self::Parentheses(x) => {
                x.columns(out);
            }
            Self::Binary(_, x, y) => {
                x.columns(out);
                y.columns(out);
            }
        }
    }

    pub(crate) fn values(&self, table: &Table, rows: &[usize]) -> Result<Vec<f64>> {
        let values = match self {
            Self::Column(name) => table.column(name)?.numeric(rows)?,
            Self::Number(value) => vec![*value; rows.len()],
            Self::Identity(x) | Self::Parentheses(x) => x.values(table, rows)?,
            Self::Log(x) => x.values(table, rows)?.into_iter().map(f64::ln).collect(),
            Self::Negate(x) => x.values(table, rows)?.into_iter().map(|v| -v).collect(),
            Self::Binary(op, x, y) => x
                .values(table, rows)?
                .into_iter()
                .zip(y.values(table, rows)?)
                .map(|(x, y)| match op {
                    Arithmetic::Add => x + y,
                    Arithmetic::Subtract => x - y,
                    Arithmetic::Multiply => x * y,
                    Arithmetic::Divide => x / y,
                    Arithmetic::Power => x.powf(y),
                })
                .collect(),
        };
        Ok(values)
    }

    pub(crate) fn data(&self, table: &Table, rows: &[usize]) -> Result<DataValue> {
        let values = self.values(table, rows)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error(format!(
                "expression `{self}` contains non-finite values"
            )));
        }
        let mut names = Vec::new();
        self.columns(&mut names);
        let single = !names.is_empty()
            && names.iter().all(|name| {
                matches!(
                    table.column(name).map(|c| &c.values),
                    Ok(ColumnValues::Real32(_))
                )
            });
        if single {
            let values: Vec<_> = values.into_iter().map(|v| v as f32).collect();
            if values.iter().any(|v| !v.is_finite()) {
                return Err(Error(format!("expression `{self}` exceeds float32 range")));
            }
            Ok(DataValue::Real32(values))
        } else {
            Ok(DataValue::Real64(values))
        }
    }
}

impl fmt::Display for ColumnExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Column(name) => f.write_str(name),
            Self::Number(value) => write!(f, "{value}"),
            Self::Identity(x) => write!(f, "I({x})"),
            Self::Log(x) => write!(f, "log({x})"),
            Self::Negate(x) => write!(f, "-{x}"),
            Self::Parentheses(x) => write!(f, "({x})"),
            Self::Binary(op, x, y) => {
                let op = match op {
                    Arithmetic::Add => " + ",
                    Arithmetic::Subtract => " - ",
                    Arithmetic::Multiply => " * ",
                    Arithmetic::Divide => "/",
                    Arithmetic::Power => "^",
                };
                write!(f, "{x}{op}{y}")
            }
        }
    }
}
