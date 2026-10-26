use crate::{Column, ColumnValues, Error, Result, Table};
use arrow_array::{
    Array, BooleanArray, DictionaryArray, Float16Array, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, LargeStringArray, RecordBatchReader, StringArray,
    StringViewArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array, new_empty_array, types,
};
use arrow_ipc::reader::{FileReader, StreamReader};
use arrow_schema::{DataType, Field};
use std::collections::{BTreeSet, HashMap};
use std::io::Cursor;

impl Table {
    /// Decode an Arrow IPC file or stream, preserving source precision and nulls.
    /// Dictionary levels retain their declared order; plain strings use sorted
    /// levels. Model-frame construction, not ingestion, selects complete rows.
    pub fn from_arrow_ipc(bytes: &[u8]) -> Result<Self> {
        let cursor = Cursor::new(bytes);
        if bytes.starts_with(b"ARROW1") {
            read_batches(FileReader::try_new(cursor, None).map_err(ipc_error)?)
        } else {
            read_batches(StreamReader::try_new(cursor, None).map_err(ipc_error)?)
        }
    }
}

fn ipc_error(error: arrow_schema::ArrowError) -> Error {
    Error(format!("Arrow IPC: {error}"))
}

fn read_batches(reader: impl RecordBatchReader) -> Result<Table> {
    let schema = reader.schema();
    // A column this reader cannot decode is kept as a reason, so that a table
    // selecting other columns still reads (§07: a value set may omit columns).
    let mut unreadable = Vec::new();
    let mut readable = Vec::new();
    let mut columns = Vec::new();
    for (index, field) in schema.fields().iter().enumerate() {
        match column(field, new_empty_array(field.data_type()).as_ref()) {
            Ok(column) => {
                readable.push(index);
                columns.push(column);
            }
            Err(error) => unreadable.push((field.name().clone(), error.0)),
        }
    }
    let fields: Vec<_> = readable.iter().map(|&i| schema.field(i)).collect();
    let mut orders = vec![Vec::new(); columns.len()];
    let mut nrows = 0;
    for batch in reader {
        let batch = batch.map_err(ipc_error)?;
        let arrays: Vec<_> = readable.iter().map(|&i| batch.column(i)).collect();
        for (((out, field), array), order) in
            columns.iter_mut().zip(&fields).zip(arrays).zip(&mut orders)
        {
            append(
                out,
                column(field, array.as_ref())?,
                nrows,
                batch.num_rows(),
                order,
            )?;
        }
        nrows += batch.num_rows();
    }
    for ((column, field), order) in columns.iter_mut().zip(&fields).zip(orders) {
        let ColumnValues::Factor {
            levels,
            codes,
            ordered,
        } = &mut column.values
        else {
            continue;
        };
        let indices = if *ordered {
            ordered_indices(levels.len(), &order).ok_or_else(|| {
                Error(format!(
                    "ordered categories in `{}` change order between Arrow batches",
                    column.name
                ))
            })?
        } else if matches!(
            field.data_type(),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        ) {
            let mut indices: Vec<_> = (0..levels.len()).collect();
            indices.sort_by_key(|&i| &levels[i]);
            indices
        } else {
            continue;
        };
        let mut mapping = vec![0; levels.len()];
        for (new, &old) in indices.iter().enumerate() {
            mapping[old] = new as u32;
        }
        for (row, code) in codes.iter_mut().enumerate() {
            if column.valid.as_ref().is_none_or(|valid| valid[row]) {
                *code = mapping[*code as usize];
            }
        }
        *levels = indices.into_iter().map(|i| levels[i].clone()).collect();
    }
    Ok(Table {
        columns,
        nrows,
        unreadable,
    })
}

// Each ordered dictionary constrains only its own labels. Defer the total
// order until all batches arrive, so later dictionaries may insert levels.
fn ordered_indices(count: usize, edges: &[(u32, u32)]) -> Option<Vec<usize>> {
    let mut outgoing = vec![Vec::new(); count];
    let mut incoming = vec![0; count];
    for &(from, to) in edges {
        outgoing[from as usize].push(to as usize);
        incoming[to as usize] += 1;
    }
    let mut ready: BTreeSet<_> = (0..count).filter(|&i| incoming[i] == 0).collect();
    let mut indices = Vec::with_capacity(count);
    while let Some(i) = ready.pop_first() {
        indices.push(i);
        for &next in &outgoing[i] {
            incoming[next] -= 1;
            if incoming[next] == 0 {
                ready.insert(next);
            }
        }
    }
    (indices.len() == count).then_some(indices)
}

fn array<T: 'static>(values: &dyn Array) -> &T {
    values
        .as_any()
        .downcast_ref()
        .expect("Arrow IPC validates array datatypes")
}

fn column(field: &Field, values: &dyn Array) -> Result<Column> {
    macro_rules! integers {
        ($ty:ty) => {
            ColumnValues::Int(
                array::<$ty>(values)
                    .iter()
                    .map(|value| {
                        i64::try_from(value.unwrap_or_default()).map_err(|_| {
                            Error(format!(
                                "column `{}` contains an integer outside int64",
                                field.name()
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        };
    }
    let data = match values.data_type() {
        DataType::Float16 => ColumnValues::Real32(
            array::<Float16Array>(values)
                .iter()
                .map(|value| value.unwrap_or_default().to_f32())
                .collect(),
        ),
        DataType::Float32 => ColumnValues::Real32(
            array::<Float32Array>(values)
                .iter()
                .map(Option::unwrap_or_default)
                .collect(),
        ),
        DataType::Float64 => ColumnValues::Real64(
            array::<Float64Array>(values)
                .iter()
                .map(Option::unwrap_or_default)
                .collect(),
        ),
        DataType::Int8 => integers!(Int8Array),
        DataType::Int16 => integers!(Int16Array),
        DataType::Int32 => integers!(Int32Array),
        DataType::Int64 => integers!(Int64Array),
        DataType::UInt8 => integers!(UInt8Array),
        DataType::UInt16 => integers!(UInt16Array),
        DataType::UInt32 => integers!(UInt32Array),
        DataType::UInt64 => integers!(UInt64Array),
        DataType::Boolean => ColumnValues::Bool(
            array::<BooleanArray>(values)
                .iter()
                .map(Option::unwrap_or_default)
                .collect(),
        ),
        DataType::Null => ColumnValues::Real64(vec![0.0; values.len()]),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            return factor(field, labels(values)?, (0..values.len()).map(Some));
        }
        DataType::Dictionary(key, _) => {
            return match key.as_ref() {
                DataType::Int8 => dictionary::<types::Int8Type>(field, values),
                DataType::Int16 => dictionary::<types::Int16Type>(field, values),
                DataType::Int32 => dictionary::<types::Int32Type>(field, values),
                DataType::Int64 => dictionary::<types::Int64Type>(field, values),
                DataType::UInt8 => dictionary::<types::UInt8Type>(field, values),
                DataType::UInt16 => dictionary::<types::UInt16Type>(field, values),
                DataType::UInt32 => dictionary::<types::UInt32Type>(field, values),
                DataType::UInt64 => dictionary::<types::UInt64Type>(field, values),
                _ => Err(Error(format!("unsupported Arrow dictionary key {key}"))),
            };
        }
        other => {
            return Err(Error(format!(
                "column `{}` has unsupported Arrow type {other}",
                field.name()
            )));
        }
    };
    Ok(Column {
        name: field.name().clone(),
        values: data,
        // A `Null`-typed array has no validity buffer, so only its logical
        // nulls say that every entry is missing.
        valid: values
            .logical_nulls()
            .filter(|nulls| nulls.null_count() != 0)
            .map(|nulls| (0..values.len()).map(|i| nulls.is_valid(i)).collect()),
    })
}

fn labels(values: &dyn Array) -> Result<Vec<Option<String>>> {
    macro_rules! strings {
        ($ty:ty) => {
            array::<$ty>(values)
                .iter()
                .map(|value| value.map(|x| x.to_string()))
                .collect()
        };
    }
    Ok(match values.data_type() {
        DataType::Utf8 => strings!(StringArray),
        DataType::LargeUtf8 => strings!(LargeStringArray),
        DataType::Utf8View => strings!(StringViewArray),
        DataType::Int8 => strings!(Int8Array),
        DataType::Int16 => strings!(Int16Array),
        DataType::Int32 => strings!(Int32Array),
        DataType::Int64 => strings!(Int64Array),
        DataType::UInt8 => strings!(UInt8Array),
        DataType::UInt16 => strings!(UInt16Array),
        DataType::UInt32 => strings!(UInt32Array),
        DataType::UInt64 => strings!(UInt64Array),
        DataType::Float32 => strings!(Float32Array),
        DataType::Float64 => strings!(Float64Array),
        DataType::Boolean => array::<BooleanArray>(values)
            .iter()
            .map(|value| value.map(|x| if x { "TRUE" } else { "FALSE" }.to_string()))
            .collect(),
        other => return Err(Error(format!("unsupported Arrow category type {other}"))),
    })
}

fn dictionary<K: types::ArrowDictionaryKeyType>(
    field: &Field,
    values: &dyn Array,
) -> Result<Column> {
    let dictionary = array::<DictionaryArray<K>>(values);
    factor(
        field,
        labels(dictionary.values().as_ref())?,
        dictionary.keys_iter(),
    )
}

fn factor(
    field: &Field,
    labels: Vec<Option<String>>,
    keys: impl Iterator<Item = Option<usize>>,
) -> Result<Column> {
    let mut indices = HashMap::new();
    let mut levels = Vec::new();
    let mapping: Vec<_> = labels
        .into_iter()
        .map(|label| {
            label.map(|label| {
                *indices.entry(label.clone()).or_insert_with(|| {
                    let index = levels.len() as u32;
                    levels.push(label);
                    index
                })
            })
        })
        .collect();
    let mut codes = Vec::new();
    let mut valid = Vec::new();
    for key in keys {
        let code = key.and_then(|key| mapping[key]);
        codes.push(code.unwrap_or_default());
        valid.push(code.is_some());
    }
    Ok(Column {
        name: field.name().clone(),
        values: ColumnValues::Factor {
            levels,
            codes,
            ordered: field.dict_is_ordered().unwrap_or(false),
        },
        valid: valid.iter().any(|x| !x).then_some(valid),
    })
}

fn append(
    out: &mut Column,
    next: Column,
    before: usize,
    count: usize,
    order: &mut Vec<(u32, u32)>,
) -> Result<()> {
    if out.valid.is_some() || next.valid.is_some() {
        out.valid
            .get_or_insert_with(|| vec![true; before])
            .extend(next.valid.unwrap_or_else(|| vec![true; count]));
    }
    match (&mut out.values, next.values) {
        (ColumnValues::Real32(a), ColumnValues::Real32(b)) => a.extend(b),
        (ColumnValues::Real64(a), ColumnValues::Real64(b)) => a.extend(b),
        (ColumnValues::Int(a), ColumnValues::Int(b)) => a.extend(b),
        (ColumnValues::Bool(a), ColumnValues::Bool(b)) => a.extend(b),
        (
            ColumnValues::Factor {
                levels,
                codes,
                ordered,
            },
            ColumnValues::Factor {
                levels: added,
                codes: incoming,
                ..
            },
        ) => {
            let mut indices: HashMap<_, _> = levels
                .iter()
                .cloned()
                .enumerate()
                .map(|(i, s)| (s, i as u32))
                .collect();
            let mut mapping = Vec::with_capacity(added.len());
            for label in added {
                let index = *indices.entry(label.clone()).or_insert_with(|| {
                    let index = levels.len() as u32;
                    levels.push(label);
                    index
                });
                mapping.push(index);
            }
            if *ordered {
                order.extend(mapping.windows(2).map(|pair| (pair[0], pair[1])));
            }
            codes.extend(incoming.into_iter().enumerate().map(|(row, code)| {
                if out.valid.as_ref().is_none_or(|valid| valid[before + row]) {
                    mapping[code as usize]
                } else {
                    0
                }
            }));
        }
        _ => {
            return Err(Error(format!(
                "column `{}` changes type between Arrow batches",
                out.name
            )));
        }
    }
    Ok(())
}
