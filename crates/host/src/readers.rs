//! Decode a `load_data` source by its extension (spec §07 `load_data`): JSON, CSV
//! and WSV, Arrow IPC, and Safetensors. Decoding infers no FlatPPL shape. The
//! declared value set selects and checks the value ([`crate::data`]).

use std::collections::{BTreeMap, HashSet};

use flatppl_formula::{ColumnValues, Table};
use serde_json::Value;

/// A decoded source, before the declared value set selects from it.
#[derive(Clone, Debug)]
pub(crate) enum Datum {
    /// A missing cell: an empty delimited field, an Arrow null, a JSON null.
    Null,
    Bool(bool),
    Int(i64),
    Real(f64),
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
        Format::Csv => csv_columns(bytes),
        Format::Wsv => wsv_columns(bytes),
        Format::Arrow => Table::from_arrow_ipc(bytes)
            .map_err(|e| e.to_string())
            .map(arrow_columns),
        Format::Safetensors => safetensors(bytes),
    }
}

fn json_datum(json: &Value) -> Result<Datum, String> {
    Ok(match json {
        Value::Null => Datum::Null,
        Value::Bool(b) => Datum::Bool(*b),
        Value::Number(n) => match n.as_i64() {
            Some(i) => Datum::Int(i),
            None if n.is_u64() => return Err(format!("integer {n} exceeds int64")),
            None => Datum::Real(n.as_f64().ok_or("number is not representable")?),
        },
        Value::String(s) => Datum::Text(s.clone()),
        Value::Array(items) => Datum::List(items.iter().map(json_datum).collect::<Result<_, _>>()?),
        Value::Object(fields) => Datum::Map(
            fields
                .iter()
                .map(|(k, v)| Ok((k.clone(), json_datum(v)?)))
                .collect::<Result<_, String>>()?,
        ),
    })
}

/// Columns from a header row and data rows of trimmed text cells.
fn columns(header: Vec<String>, rows: Vec<Vec<String>>) -> Result<Datum, String> {
    let mut seen = HashSet::new();
    if header
        .iter()
        .any(|name| name.is_empty() || !seen.insert(name.as_str()))
    {
        return Err("column names in the first row must be nonempty and distinct".into());
    }
    let mut cells = vec![Vec::with_capacity(rows.len()); header.len()];
    for (i, row) in rows.into_iter().enumerate() {
        if row.len() != header.len() {
            return Err(format!(
                "row {} has {} fields, but the first row names {} columns",
                i + 1,
                row.len(),
                header.len()
            ));
        }
        for (column, cell) in cells.iter_mut().zip(row) {
            column.push(if cell.is_empty() {
                Datum::Null
            } else {
                Datum::Text(cell)
            });
        }
    }
    Ok(Datum::Columns(
        header
            .into_iter()
            .zip(cells)
            .map(|(name, cells)| (name, Datum::List(cells)))
            .collect(),
    ))
}

/// Comma-separated values with column names in the first row (§07). Fields are
/// trimmed, and a line whose first character is `#` is a comment.
fn csv_columns(bytes: &[u8]) -> Result<Datum, String> {
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .comment(Some(b'#'))
        .from_reader(bytes);
    let csv_error = |e: csv::Error| format!("CSV: {e}");
    let header = reader
        .headers()
        .map_err(csv_error)?
        .iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let rows = reader
        .records()
        .map(|record| record.map(|r| r.iter().map(str::to_owned).collect()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(csv_error)?;
    columns(header, rows)
}

/// Whitespace-separated values with column names in the first row (§07). Blank
/// lines and lines starting with `#` are skipped.
fn wsv_columns(bytes: &[u8]) -> Result<Datum, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("WSV: {e}"))?;
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            line.split_whitespace()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        });
    let header = lines.next().ok_or("WSV: missing column names")?;
    columns(header, lines.collect())
}

fn arrow_columns(table: Table) -> Datum {
    Datum::Columns(
        table
            .columns
            .into_iter()
            .map(|column| {
                let valid = |i: usize| column.valid.as_ref().is_none_or(|v| v[i]);
                let cells: Vec<Datum> = match &column.values {
                    ColumnValues::Real32(v) => v.iter().map(|&x| Datum::Real32(x)).collect(),
                    ColumnValues::Real64(v) => v.iter().map(|&x| Datum::Real(x)).collect(),
                    ColumnValues::Int(v) => v.iter().map(|&x| Datum::Int(x)).collect(),
                    ColumnValues::Bool(v) => v.iter().map(|&x| Datum::Bool(x)).collect(),
                    ColumnValues::Factor { levels, codes, .. } => codes
                        .iter()
                        .map(|&c| Datum::Text(levels[c as usize].clone()))
                        .collect(),
                };
                let cells = cells
                    .into_iter()
                    .enumerate()
                    .map(|(i, cell)| if valid(i) { cell } else { Datum::Null })
                    .collect();
                (column.name, Datum::List(cells))
            })
            .collect(),
    )
}

/// One Safetensors header entry.
#[derive(serde::Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// A Safetensors file as a nested record (§07): a key's dot-separated segments
/// form a record path, after leading and trailing dots are trimmed.
fn safetensors(bytes: &[u8]) -> Result<Datum, String> {
    let fail = |message: String| format!("Safetensors: {message}");
    let length = bytes
        .get(..8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("eight bytes")))
        .ok_or_else(|| fail("missing the header length".into()))?;
    let header_end = usize::try_from(length)
        .ok()
        .and_then(|n| n.checked_add(8))
        .filter(|&end| end <= bytes.len())
        .ok_or_else(|| fail("header length exceeds the file".into()))?;
    let header: BTreeMap<String, Value> =
        serde_json::from_slice(&bytes[8..header_end]).map_err(|e| fail(format!("header: {e}")))?;
    let data = &bytes[header_end..];
    let mut root = Node::default();
    let mut paths = HashSet::new();
    for (key, entry) in header {
        if key == "__metadata__" {
            continue;
        }
        let path = key.trim_matches('.');
        if path.is_empty() {
            return Err(fail(format!(
                "key `{key}` is empty after trimming its dots"
            )));
        }
        let segments: Vec<&str> = path.split('.').collect();
        if segments.iter().any(|s| s.is_empty()) {
            return Err(fail(format!("key `{key}` has an empty path segment")));
        }
        if !paths.insert(path.to_owned()) {
            return Err(fail(format!(
                "keys trimming to `{path}` name the same record path"
            )));
        }
        let info: TensorInfo =
            serde_json::from_value(entry).map_err(|e| fail(format!("tensor `{key}`: {e}")))?;
        let tensor = tensor(&info, data).map_err(|e| fail(format!("tensor `{key}`: {e}")))?;
        root.insert(&segments, tensor).map_err(|prefix| {
            fail(format!(
                "`{prefix}` is both a tensor and a record path prefix"
            ))
        })?;
    }
    Ok(root.into_datum())
}

/// A record path tree under construction, children in key order.
#[derive(Default)]
struct Node {
    leaf: Option<Datum>,
    children: Vec<(String, Node)>,
}

impl Node {
    /// Insert a leaf, refusing a path that is both a leaf and a prefix (§07).
    fn insert(&mut self, segments: &[&str], leaf: Datum) -> Result<(), String> {
        self.insert_at(segments, 0, leaf)
    }

    fn insert_at(&mut self, segments: &[&str], depth: usize, leaf: Datum) -> Result<(), String> {
        let prefix = || segments[..depth].join(".");
        if self.leaf.is_some() {
            return Err(prefix());
        }
        let Some(segment) = segments.get(depth) else {
            if !self.children.is_empty() {
                return Err(prefix());
            }
            self.leaf = Some(leaf);
            return Ok(());
        };
        let index = match self.children.iter().position(|(n, _)| n == segment) {
            Some(i) => i,
            None => {
                self.children.push(((*segment).to_owned(), Node::default()));
                self.children.len() - 1
            }
        };
        self.children[index].1.insert_at(segments, depth + 1, leaf)
    }

    fn into_datum(self) -> Datum {
        match self.leaf {
            Some(leaf) => leaf,
            None => Datum::Map(
                self.children
                    .into_iter()
                    .map(|(name, node)| (name, node.into_datum()))
                    .collect(),
            ),
        }
    }
}

/// Decode one tensor's little-endian bytes. Floats keep binary32 precision or
/// wider, integers widen to int64, and `BOOL` is one byte per value.
fn tensor(info: &TensorInfo, data: &[u8]) -> Result<Datum, String> {
    let [begin, end] = info.data_offsets;
    let bytes = data
        .get(begin..end)
        .ok_or("data offsets lie outside the data buffer")?;
    let count = info
        .shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or("element count overflows")?;
    macro_rules! decode {
        ($ty:ty, $size:literal, $wrap:expr) => {{
            if bytes.len() != count * $size {
                return Err(format!(
                    "{} bytes hold {} {} values, but the shape {:?} needs {count}",
                    bytes.len(),
                    bytes.len() / $size,
                    info.dtype,
                    info.shape
                ));
            }
            bytes
                .chunks_exact($size)
                .map(|b| <$ty>::from_le_bytes(b.try_into().expect("chunk size")))
                .map($wrap)
                .collect::<Result<Vec<Datum>, String>>()?
        }};
    }
    let leaves = match info.dtype.as_str() {
        "F64" => decode!(f64, 8, |x| Ok(Datum::Real(x))),
        "F32" => decode!(f32, 4, |x| Ok(Datum::Real32(x))),
        "F16" => decode!(u16, 2, |x| Ok(Datum::Real32(
            half::f16::from_bits(x).to_f32()
        ))),
        "BF16" => decode!(u16, 2, |x| Ok(Datum::Real32(
            half::bf16::from_bits(x).to_f32()
        ))),
        "I64" => decode!(i64, 8, |x| Ok(Datum::Int(x))),
        "I32" => decode!(i32, 4, |x| Ok(Datum::Int(x.into()))),
        "I16" => decode!(i16, 2, |x| Ok(Datum::Int(x.into()))),
        "I8" => decode!(i8, 1, |x| Ok(Datum::Int(x.into()))),
        "U64" => decode!(u64, 8, |x| i64::try_from(x)
            .map(Datum::Int)
            .map_err(|_| format!("U64 value {x} exceeds int64"))),
        "U32" => decode!(u32, 4, |x| Ok(Datum::Int(x.into()))),
        "U16" => decode!(u16, 2, |x| Ok(Datum::Int(x.into()))),
        "U8" => decode!(u8, 1, |x| Ok(Datum::Int(x.into()))),
        "BOOL" => decode!(u8, 1, |x| match x {
            0 => Ok(Datum::Bool(false)),
            1 => Ok(Datum::Bool(true)),
            other => Err(format!("BOOL byte {other} is neither 0 nor 1")),
        }),
        other => {
            return Err(format!(
                "dtype {other} has no FlatPPL value set (§07 maps float, integer and bool \
                 dtypes)"
            ));
        }
    };
    Ok(Datum::Tensor {
        shape: info.shape.clone(),
        data: leaves,
    })
}
