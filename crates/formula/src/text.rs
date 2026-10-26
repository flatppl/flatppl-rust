use crate::{Column, ColumnValues, Error, Result, Table};
use serde_json::{Number, Value, value::RawValue};
use std::collections::{BTreeMap, BTreeSet};

type JsonRow<'a> = BTreeMap<String, &'a RawValue>;

/// Read a typed table, an array of row objects, or an object of column arrays.
/// Missing record fields and JSON nulls remain missing until model-frame
/// construction. Strings become unordered factors with sorted levels.
pub fn read_json_table(source: &str) -> Result<Table> {
    let table = match source.trim_start().as_bytes().first() {
        Some(b'{') => {
            let object: JsonRow<'_> = read_json(source)?;
            if object
                .get("nrows")
                .is_some_and(|v| !v.get().starts_with('['))
            {
                let table: Table = read_json(source)?;
                table.validate()?;
                return Ok(table);
            }
            let mut nrows = 0;
            let mut columns = Vec::with_capacity(object.len());
            for (name, values) in object {
                let values: Vec<&RawValue> = read_json(values.get())
                    .map_err(|_| Error(format!("JSON column `{name}` must be an array")))?;
                if columns.is_empty() {
                    nrows = values.len();
                }
                let values = values
                    .into_iter()
                    .map(|value| json_cell(value, &name))
                    .collect::<Result<_>>()?;
                columns.push(column(name, values)?);
            }
            Table { columns, nrows }
        }
        Some(b'[') => {
            let rows: Vec<JsonRow<'_>> = read_json(source)?;
            let names: BTreeSet<_> = rows.iter().flat_map(|r| r.keys()).collect();
            let columns = names
                .into_iter()
                .map(|name| {
                    column(
                        name.clone(),
                        rows.iter()
                            .map(|row| {
                                row.get(name)
                                    .map_or(Ok(Value::Null), |value| json_cell(value, name))
                            })
                            .collect::<Result<_>>()?,
                    )
                })
                .collect::<Result<_>>()?;
            Table {
                columns,
                nrows: rows.len(),
            }
        }
        _ => return Err(Error("JSON table must contain records or columns".into())),
    };
    table.validate()?;
    Ok(table)
}

/// Read a headered CSV table through [`flatppl_data::read_csv`], which trims
/// fields and skips `#` comment lines. Empty fields and `NA` are null. Columns
/// infer booleans, integers, finite real numbers, or unordered string factors, in
/// that order.
pub fn read_csv_table(bytes: &[u8]) -> Result<Table> {
    let text = flatppl_data::read_csv(bytes)?;
    let columns = text
        .names
        .iter()
        .enumerate()
        .map(|(i, name)| csv_column(name.clone(), text.column(i).map(str::to_owned).collect()))
        .collect::<Result<_>>()?;
    let table = Table {
        columns,
        nrows: text.rows.len(),
    };
    table.validate()?;
    Ok(table)
}

fn read_json<'a, T: serde::Deserialize<'a>>(source: &'a str) -> Result<T> {
    serde_json::from_str(source).map_err(|e| Error(format!("invalid table JSON: {e}")))
}

fn json_cell(value: &RawValue, name: &str) -> Result<Value> {
    let source = value.get();
    if integer_literal(source) {
        source
            .parse::<i64>()
            .map(Value::from)
            .map_err(|_| Error(format!("integer in column `{name}` exceeds int64")))
    } else {
        read_json(source)
    }
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
