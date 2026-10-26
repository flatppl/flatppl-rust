//! Decode a `load_data` source by its extension (spec §07 `load_data`): JSON, CSV
//! and WSV, Arrow IPC, and Safetensors, each through its one reader in
//! `flatppl-data`. Decoding infers no FlatPPL shape. The declared value set
//! selects and checks the value ([`crate::data`]).

use flatppl_data::safetensors::{self, TensorValues};
use flatppl_data::{ColumnValues, Table};
use serde_json::Value;

/// A decoded source, before the declared value set selects from it.
#[derive(Clone, Debug)]
pub(crate) enum Datum {
    /// A missing cell: an empty delimited field, an Arrow null, a JSON null.
    Null,
    Bool(bool),
    Int(i64),
    Real(f64),
    /// A JSON number written with a fraction or an exponent. JSON has one
    /// number type, so its value, not its spelling, decides its set.
    Number(f64),
    /// A binary32 (or narrower) value, kept at its source precision.
    Real32(f32),
    /// A delimited cell, parsed only once its scalar set is known.
    Text(String),
    List(Vec<Datum>),
    /// A JSON object or a Safetensors record, fields in source order.
    Map(Vec<(String, Datum)>),
    /// Named columns of a delimited or Arrow table.
    Columns(Vec<(String, Datum)>),
    /// A Safetensors tensor: row-major leaves of a fixed shape.
    Tensor {
        shape: Vec<usize>,
        data: Vec<Datum>,
    },
    /// A column no value set can select, with the reason. It refuses only when
    /// selected.
    Unreadable(String),
}

impl Datum {
    /// The named entry of a record-like datum.
    pub(crate) fn field(&self, name: &str) -> Option<&Datum> {
        match self {
            Datum::Map(fields) | Datum::Columns(fields) => {
                fields.iter().find(|(n, _)| n == name).map(|(_, v)| v)
            }
            _ => None,
        }
    }
}

/// The formats §07 requires, by extension.
pub(crate) enum Format {
    Json,
    Csv,
    Wsv,
    Arrow,
    Safetensors,
}

/// The format of `source` from its extension, ignoring a URL query or fragment.
pub(crate) fn format(source: &str) -> Result<Format, String> {
    let path = source.split(['?', '#']).next().unwrap_or(source);
    let name = path.rsplit('/').next().unwrap_or(path);
    let extension = name
        .rsplit_once('.')
        .map(|(_, ext)| ext.to_ascii_lowercase())
        .unwrap_or_default();
    match extension.as_str() {
        "json" => Ok(Format::Json),
        "csv" => Ok(Format::Csv),
        "wsv" => Ok(Format::Wsv),
        "arrow" | "arrows" => Ok(Format::Arrow),
        "safetensors" => Ok(Format::Safetensors),
        _ => Err(format!(
            "unsupported file extension {}; §07 `load_data` reads `.json`, `.csv`, `.wsv`, \
             `.arrow`, `.arrows` and `.safetensors`",
            if extension.is_empty() {
                "(none)".to_owned()
            } else {
                format!("`.{extension}`")
            }
        )),
    }
}

/// Decode `bytes` in `format`.
pub(crate) fn decode(format: Format, bytes: &[u8]) -> Result<Datum, String> {
    match format {
        Format::Json => {
            let json: Value = serde_json::from_slice(bytes).map_err(|e| format!("JSON: {e}"))?;
            json_datum(&json)
        }
        Format::Csv => flatppl_data::read_csv(bytes)
            .map(text_columns)
            .map_err(|e| e.0),
        Format::Wsv => flatppl_data::read_wsv(bytes)
            .map(text_columns)
            .map_err(|e| e.0),
        Format::Arrow => Table::from_arrow_ipc_values(bytes)
            .map(arrow_columns)
            .map_err(|e| e.0),
        Format::Safetensors => safetensors::read(bytes)
            .map(safetensors_entry)
            .map_err(|e| e.0),
    }
}

fn json_datum(json: &Value) -> Result<Datum, String> {
    Ok(match json {
        Value::Null => Datum::Null,
        Value::Bool(b) => Datum::Bool(*b),
        Value::Number(n) => match n.as_i64() {
            Some(i) => Datum::Int(i),
            None if n.is_u64() => Datum::Unreadable(format!("integer {n} exceeds int64")),
            None => Datum::Number(n.as_f64().ok_or("number is not representable")?),
        },
        // JSON types its values, and a string is not a number.
        Value::String(s) => {
            Datum::Unreadable(format!("JSON string {s:?} is not a number or a boolean"))
        }
        Value::Array(items) => Datum::List(items.iter().map(json_datum).collect::<Result<_, _>>()?),
        Value::Object(fields) => Datum::Map(
            fields
                .iter()
                .map(|(k, v)| Ok((k.clone(), json_datum(v)?)))
                .collect::<Result<_, String>>()?,
        ),
    })
}

/// Columns of text cells from a delimited source. An empty cell is missing.
fn text_columns(table: flatppl_data::TextTable) -> Datum {
    Datum::Columns(
        table
            .names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let cells = table
                    .column(i)
                    .map(|cell| {
                        if cell.is_empty() {
                            Datum::Null
                        } else {
                            Datum::Text(cell.to_owned())
                        }
                    })
                    .collect();
                (name.clone(), Datum::List(cells))
            })
            .collect(),
    )
}

fn arrow_columns(table: Table) -> Datum {
    let mut columns: Vec<(String, Datum)> = table
        .columns
        .into_iter()
        .map(|column| {
            let valid = |i: usize| column.valid.as_ref().is_none_or(|v| v[i]);
            let cells: Vec<Datum> = match &column.values {
                ColumnValues::Real32(v) => v.iter().map(|&x| Datum::Real32(x)).collect(),
                ColumnValues::Real64(v) => v.iter().map(|&x| Datum::Real(x)).collect(),
                ColumnValues::Int(v) => v.iter().map(|&x| Datum::Int(x)).collect(),
                ColumnValues::Bool(v) => v.iter().map(|&x| Datum::Bool(x)).collect(),
                // Arrow is typed, and a string is not a number.
                ColumnValues::Factor { .. } => {
                    let reason = format!(
                        "column `{}` holds Arrow strings, and `load_data` reads numbers and \
                         booleans",
                        column.name
                    );
                    return (column.name, Datum::Unreadable(reason));
                }
            };
            let cells = cells
                .into_iter()
                .enumerate()
                .map(|(i, cell)| if valid(i) { cell } else { Datum::Null })
                .collect();
            (column.name, Datum::List(cells))
        })
        .collect();
    columns.extend(
        table
            .unreadable
            .into_iter()
            .map(|(name, reason)| (name, Datum::Unreadable(reason))),
    );
    Datum::Columns(columns)
}

/// A Safetensors record tree, each tensor's leaves at their source kind.
fn safetensors_entry(entry: safetensors::Entry) -> Datum {
    match entry {
        safetensors::Entry::Record(fields) => Datum::Map(
            fields
                .into_iter()
                .map(|(name, entry)| (name, safetensors_entry(entry)))
                .collect(),
        ),
        safetensors::Entry::Tensor(tensor) => {
            let data = match tensor.values {
                TensorValues::Real32(v) => v.into_iter().map(Datum::Real32).collect(),
                TensorValues::Real64(v) => v.into_iter().map(Datum::Real).collect(),
                TensorValues::Int(v) => v.into_iter().map(Datum::Int).collect(),
                TensorValues::Bool(v) => v.into_iter().map(Datum::Bool).collect(),
            };
            Datum::Tensor {
                shape: tensor.shape,
                data,
            }
        }
    }
}
