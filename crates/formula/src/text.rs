use crate::{Column, ColumnValues, Error, Result, Table};
use serde_json::{Number, Value};
use std::{collections::BTreeSet, io::Read};

impl Table {
    /// Read a typed table, an array of row objects, or an object of column arrays.
    /// Missing record fields and JSON nulls remain missing until model-frame
    /// construction. Strings become unordered factors with sorted levels.
    pub fn from_json(source: &str) -> Result<Self> {
        let value: Value =
            serde_json::from_str(source).map_err(|e| Error(format!("invalid table JSON: {e}")))?;
        let table = match value {
            Value::Object(ref object) if object.get("nrows").is_some_and(|v| !v.is_array()) => {
                serde_json::from_value(value)
                    .map_err(|e| Error(format!("invalid typed table JSON: {e}")))?
            }
            Value::Object(object) => {
                let nrows = object
                    .values()
                    .next()
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len);
                let mut columns = Vec::with_capacity(object.len());
                for (name, values) in object {
                    let Value::Array(values) = values else {
                        return Err(Error(format!("JSON column `{name}` must be an array")));
                    };
                    columns.push(column(name, values)?);
                }
                Self { columns, nrows }
            }
            Value::Array(rows) => {
                let rows = rows
                    .into_iter()
                    .map(|row| match row {
                        Value::Object(row) => Ok(row),
                        _ => Err(Error("JSON table rows must be objects".into())),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let names: BTreeSet<_> = rows.iter().flat_map(|r| r.keys()).collect();
                let columns = names
                    .into_iter()
                    .map(|name| {
                        column(
                            name.clone(),
                            rows.iter()
                                .map(|row| row.get(name).cloned().unwrap_or(Value::Null))
                                .collect(),
                        )
                    })
                    .collect::<Result<_>>()?;
                Self {
                    columns,
                    nrows: rows.len(),
                }
            }
            _ => return Err(Error("JSON table must contain records or columns".into())),
        };
        table.validate()?;
        Ok(table)
    }

    /// Read a headered CSV table. Empty fields and `NA` are null, including when
    /// quoted. Other fields are not trimmed. Columns infer booleans, integers,
    /// finite real numbers, or unordered string factors, in that order.
    pub fn from_csv(reader: impl Read) -> Result<Self> {
        let mut reader = csv::Reader::from_reader(reader);
        let names = reader
            .headers()
            .map_err(csv_error)?
            .iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let mut values = vec![Vec::new(); names.len()];
        let mut nrows = 0;
        for record in reader.records() {
            let record = record.map_err(csv_error)?;
            for (column, value) in values.iter_mut().zip(&record) {
                column.push(value.to_owned());
            }
            nrows += 1;
        }
        let columns = names
            .into_iter()
            .zip(values)
            .map(|(name, values)| csv_column(name, values))
            .collect::<Result<_>>()?;
        let table = Self { columns, nrows };
        table.validate()?;
        Ok(table)
    }
}

fn csv_error(error: csv::Error) -> Error {
    Error(format!("invalid table CSV: {error}"))
}

fn csv_column(name: String, values: Vec<String>) -> Result<Column> {
    let present = values.iter().filter(|v| !matches!(v.as_str(), "" | "NA"));
    let boolean = present.clone().next().is_some()
        && present
            .clone()
            .all(|v| matches!(v.as_str(), "true" | "false" | "TRUE" | "FALSE"));
    let numeric = present.clone().all(|v| v.parse::<f64>().is_ok());
    let values = values
        .into_iter()
        .map(|value| match value.as_str() {
            "" | "NA" => Ok(Value::Null),
            _ if boolean => Ok(Value::Bool(matches!(value.as_str(), "true" | "TRUE"))),
            _ if numeric && integer_literal(&value) => value
                .parse::<i64>()
                .map(Value::from)
                .map_err(|_| Error(format!("integer in column `{name}` exceeds int64"))),
            _ if numeric => {
                let number = value.parse::<f64>().expect("numeric column");
                Number::from_f64(number)
                    .map(Value::Number)
                    .ok_or_else(|| Error(format!("non-finite number in column `{name}`")))
            }
            _ => Ok(Value::String(value)),
        })
        .collect::<Result<_>>()?;
    column(name, values)
}

fn integer_literal(text: &str) -> bool {
    let digits = text.strip_prefix(['-', '+']).unwrap_or(text);
    !digits.is_empty() && digits.bytes().all(|c| c.is_ascii_digit())
}

fn column(name: String, values: Vec<Value>) -> Result<Column> {
    let valid: Vec<_> = values.iter().map(|v| !v.is_null()).collect();
    let first = values.iter().find(|v| !v.is_null());
    let mixed = || Error(format!("mixed or nested JSON types in column `{name}`"));
    let data = match first {
        None => ColumnValues::Real64(vec![0.0; values.len()]),
        Some(Value::Bool(_)) => ColumnValues::Bool(
            values
                .iter()
                .map(|v| match v {
                    Value::Null => Ok(false),
                    Value::Bool(v) => Ok(*v),
                    _ => Err(mixed()),
                })
                .collect::<Result<_>>()?,
        ),
        Some(Value::String(_)) => {
            let levels: Vec<_> = values
                .iter()
                .filter_map(Value::as_str)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(str::to_owned)
                .collect();
            let codes = values
                .iter()
                .map(|v| match v {
                    Value::Null => Ok(0),
                    Value::String(v) => u32::try_from(levels.binary_search(v).unwrap())
                        .map_err(|_| Error(format!("too many factor levels in `{name}`"))),
                    _ => Err(mixed()),
                })
                .collect::<Result<_>>()?;
            ColumnValues::Factor {
                levels,
                codes,
                ordered: false,
            }
        }
        Some(Value::Number(_)) => {
            let numbers = values
                .iter()
                .map(|v| match v {
                    Value::Null => Ok(None),
                    Value::Number(v) => Ok(Some(v)),
                    _ => Err(mixed()),
                })
                .collect::<Result<Vec<_>>>()?;
            let real = numbers
                .iter()
                .flatten()
                .any(|n| !integer_literal(&n.to_string()));
            if real {
                ColumnValues::Real64(
                    numbers
                        .into_iter()
                        .map(|n| match n {
                            None => Ok(0.0),
                            Some(n) if integer_literal(&n.to_string()) => {
                                let integer = integer(n, &name)?;
                                let real = integer as f64;
                                if real as i128 != i128::from(integer) {
                                    return Err(Error(format!(
                                        "integer in column `{name}` loses precision as float64"
                                    )));
                                }
                                Ok(real)
                            }
                            Some(n) => n.as_f64().filter(|n| n.is_finite()).ok_or_else(|| {
                                Error(format!("non-finite number in column `{name}`"))
                            }),
                        })
                        .collect::<Result<_>>()?,
                )
            } else {
                ColumnValues::Int(
                    numbers
                        .into_iter()
                        .map(|n| n.map_or(Ok(0), |n| integer(n, &name)))
                        .collect::<Result<_>>()?,
                )
            }
        }
        Some(_) => return Err(mixed()),
    };
    Ok(Column {
        name,
        values: data,
        valid: valid.iter().any(|&v| !v).then_some(valid),
    })
}

fn integer(number: &Number, name: &str) -> Result<i64> {
    number
        .as_i64()
        .ok_or_else(|| Error(format!("integer in column `{name}` exceeds int64")))
}
