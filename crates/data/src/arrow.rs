use crate::{Column, ColumnValues, Error, Result, Table};
use arrow_array::{
    Array, BooleanArray, DictionaryArray, Float16Array, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, LargeStringArray, RecordBatchReader, StringArray,
    StringViewArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array, new_empty_array, types,
};
use arrow_ipc::reader::{FileReader, StreamReader};
use arrow_schema::{ArrowError, DataType, Field};
use std::collections::{BTreeSet, HashMap};
use std::io::Cursor;

impl Table {
    /// Decode an Arrow IPC file or stream, preserving source precision and nulls.
    /// Dictionary levels retain their declared order; plain strings use sorted
    /// levels. Model-frame construction, not ingestion, selects complete rows.
    pub fn from_arrow_ipc(bytes: &[u8]) -> Result<Self> {
        read(bytes, false)
    }

    /// Decode an Arrow IPC file or stream as `load_data` values (§07). A
    /// dictionary over numbers or booleans reads as its values, because §07
    /// types a column by its values, not by their encoding.
    pub fn from_arrow_ipc_values(bytes: &[u8]) -> Result<Self> {
        read(bytes, true)
    }
}

fn read(bytes: &[u8], decode: bool) -> Result<Table> {
    check_buffers(bytes)?;
    let cursor = Cursor::new(bytes);
    if bytes.starts_with(b"ARROW1") {
        read_batches(
            FileReader::try_new(cursor, None).map_err(ipc_error)?,
            decode,
        )
    } else {
        read_batches(
            StreamReader::try_new(cursor, None).map_err(ipc_error)?,
            decode,
        )
    }
}

/// arrow-ipc trusts the lengths a file states: a buffer at a negative offset, or a
/// footer block shorter than its message, panics, and a compressed buffer's
/// length prefix sizes an allocation. Check them first. Framing that does not
/// parse is left for arrow-ipc to refuse.
fn check_buffers(bytes: &[u8]) -> Result<()> {
    let malformed = |what: &str| Error(format!("Arrow IPC: malformed file ({what})"));
    let word = |at: usize| {
        let word = bytes.get(at..at.checked_add(4)?)?;
        Some(i32::from_le_bytes(word.try_into().ok()?))
    };
    // Where the message at `at` ends its metadata, and its body length, once its
    // buffers check; `None` if it does not parse.
    let check = |at: usize| -> Result<Option<(usize, usize)>> {
        let (mut at, mut length) = (at + 4, word(at));
        if length == Some(-1) {
            (at, length) = (at + 4, word(at));
        }
        let Some(length) = length
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0)
        else {
            return Ok(None);
        };
        let Some(message) = bytes
            .get(at..at.saturating_add(length))
            .and_then(|m| arrow_ipc::root_as_message(m).ok())
        else {
            return Ok(None);
        };
        let Ok(body) = usize::try_from(message.bodyLength()) else {
            return Ok(None);
        };
        let start = at + length;
        let Some(batch) = message
            .header_as_record_batch()
            .or_else(|| message.header_as_dictionary_batch().and_then(|d| d.data()))
        else {
            return Ok(Some((start, body)));
        };
        // A compressed buffer opens with its decompressed length, and no codec
        // expands a byte further than this.
        let codec = batch.compression().map(|c| c.codec());
        for buffer in batch.buffers().iter().flatten() {
            let (offset, size) = (buffer.offset(), buffer.length());
            if offset < 0
                || size < 0
                || offset
                    .checked_add(size)
                    .is_none_or(|end| end as usize > body)
            {
                return Err(malformed("a buffer leaves its message body"));
            }
            let (Some(codec), true) = (codec, size > 0) else {
                continue;
            };
            let from = start + offset as usize;
            let Some(prefix) = bytes.get(from..from + 8).filter(|_| size >= 8) else {
                return Err(malformed("a compressed buffer has no length prefix"));
            };
            let stated = i64::from_le_bytes(prefix.try_into().expect("eight bytes"));
            if stated == -1 {
                continue;
            }
            let frames = bytes
                .get(from + 8..from + size as usize)
                .unwrap_or_default();
            let Some(most) = frame_capacity(codec, frames) else {
                return Err(malformed(
                    "a compressed buffer does not hold its codec's frames",
                ));
            };
            if stated < 0 || stated as u64 > most {
                return Err(malformed(&format!(
                    "a compressed buffer of {size} bytes states {stated} decompressed bytes, \
                     and its frames hold at most {most}"
                )));
            }
        }
        Ok(Some((start, body)))
    };
    if !bytes.starts_with(b"ARROW1") {
        let mut at = Some(0);
        while let Some(next) = at {
            at = check(next)?.and_then(|(start, body)| start.checked_add(body));
        }
        return Ok(());
    }
    let footer = bytes.len().checked_sub(10).and_then(|end| {
        let length = usize::try_from(word(end)?).ok()?;
        arrow_ipc::root_as_footer(bytes.get(end.checked_sub(length)?..end)?).ok()
    });
    let Some(footer) = footer else {
        return Ok(());
    };
    // FileReader slices a message by its block's lengths, not by the message's own.
    let blocks = footer.dictionaries().into_iter().flatten();
    for block in blocks.chain(footer.recordBatches().into_iter().flatten()) {
        let Ok(offset) = usize::try_from(block.offset()) else {
            continue;
        };
        if let Some((start, body)) = check(offset)?
            && (usize::try_from(block.metaDataLength()).ok() != Some(start - offset)
                || usize::try_from(block.bodyLength()).ok() != Some(body))
        {
            return Err(malformed("a footer block disagrees with its message"));
        }
    }
    Ok(())
}

/// The most bytes `data`, a run of zstd or LZ4 frames, can decompress to, read
/// from the block headers without decoding: a raw or uncompressed block holds its
/// own size, an RLE block states its size, and a compressed block yields at most
/// the format's block maximum. `None` when the frames do not parse.
fn frame_capacity(codec: arrow_ipc::CompressionType, data: &[u8]) -> Option<u64> {
    let byte = |at: usize| data.get(at).copied();
    let word = |at: usize| Some(u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?));
    let zstd = codec == arrow_ipc::CompressionType::ZSTD;
    let (mut at, mut total) = (0usize, 0u64);
    while at < data.len() {
        let magic = word(at)?;
        // Skippable frames (both formats) carry no output.
        if magic & 0xFFFF_FFF0 == 0x184D_2A50 {
            at = at.checked_add(8 + word(at + 4)? as usize)?;
            if at > data.len() {
                return None;
            }
            continue;
        }
        if zstd {
            if magic != 0xFD2F_B528 {
                return None;
            }
            let descriptor = byte(at + 4)?;
            let single = descriptor & 0x20 != 0;
            let content = [usize::from(single), 2, 4, 8][usize::from(descriptor >> 6)];
            let dictionary = [0, 1, 2, 4][usize::from(descriptor & 3)];
            at += 5 + usize::from(!single) + dictionary + content;
            loop {
                let header = u32::from(byte(at)?)
                    | u32::from(byte(at + 1)?) << 8
                    | u32::from(byte(at + 2)?) << 16;
                let size = (header >> 3) as usize;
                if size > 1 << 17 {
                    return None;
                }
                let (payload, output) = match (header >> 1) & 3 {
                    0 => (size, size as u64),
                    1 => (1, size as u64),
                    2 => (size, 1 << 17),
                    _ => return None,
                };
                at = at.checked_add(3 + payload)?;
                total = total.saturating_add(output);
                if header & 1 == 1 {
                    break;
                }
            }
            at += if descriptor & 0x04 != 0 { 4 } else { 0 };
        } else {
            if magic != 0x184D_2204 {
                return None;
            }
            let flags = byte(at + 4)?;
            let block_max: u64 = match (byte(at + 5)? >> 4) & 7 {
                4 => 64 << 10,
                5 => 256 << 10,
                6 => 1 << 20,
                7 => 4 << 20,
                _ => return None,
            };
            at += 7 + if flags & 0x08 != 0 { 8 } else { 0 } + if flags & 0x01 != 0 { 4 } else { 0 };
            loop {
                let block = word(at)?;
                at += 4;
                if block == 0 {
                    break;
                }
                let size = (block & 0x7FFF_FFFF) as usize;
                if size as u64 > block_max {
                    return None;
                }
                let output = if block & 0x8000_0000 != 0 {
                    size as u64
                } else {
                    block_max
                };
                at = at.checked_add(size + if flags & 0x10 != 0 { 4 } else { 0 })?;
                total = total.saturating_add(output);
            }
            at += if flags & 0x04 != 0 { 4 } else { 0 };
        }
        if at > data.len() {
            return None;
        }
    }
    Some(total)
}

fn ipc_error(error: ArrowError) -> Error {
    // Without a codec, arrow-ipc names a cargo feature, which a user cannot act on.
    if let ArrowError::InvalidArgumentError(message) = &error
        && let Some((codec, _)) = message.split_once(" IPC decompression requires")
    {
        return Error(format!(
            "{codec}-compressed Arrow IPC is not supported in this build"
        ));
    }
    Error(format!("Arrow IPC: {error}"))
}

fn read_batches(reader: impl RecordBatchReader, decode: bool) -> Result<Table> {
    let schema = reader.schema();
    // A column this reader cannot decode is kept as a reason, so that a table
    // selecting other columns still reads (§07: a value set may omit columns).
    let mut unreadable = Vec::new();
    let mut readable = Vec::new();
    let mut columns = Vec::new();
    for (index, field) in schema.fields().iter().enumerate() {
        match column(field, new_empty_array(field.data_type()).as_ref(), decode) {
            Ok(column) => {
                readable.push(index);
                columns.push(Ok(column));
            }
            Err(error) => unreadable.push((field.name().clone(), error.0)),
        }
    }
    let mut orders = vec![Vec::new(); columns.len()];
    let mut nrows = 0;
    for batch in reader {
        let batch = batch.map_err(ipc_error)?;
        // A value that fails in a later batch also spoils only its own column.
        for ((slot, &index), order) in columns.iter_mut().zip(&readable).zip(&mut orders) {
            let Ok(out) = slot else { continue };
            let next = column(schema.field(index), batch.column(index).as_ref(), decode);
            if let Err(error) =
                next.and_then(|next| append(out, next, nrows, batch.num_rows(), order))
            {
                *slot = Err(error.0);
            }
        }
        nrows += batch.num_rows();
    }
    let mut kept = Vec::with_capacity(columns.len());
    for ((slot, index), order) in columns.into_iter().zip(readable).zip(orders) {
        let field = schema.field(index);
        match slot.and_then(|column| finish_levels(column, field, &order)) {
            Ok(column) => kept.push(column),
            Err(reason) => unreadable.push((field.name().clone(), reason)),
        }
    }
    Ok(Table {
        columns: kept,
        nrows,
        unreadable,
    })
}

/// Settle a categorical column's level order once every batch has arrived.
fn finish_levels(
    mut column: Column,
    field: &Field,
    order: &[(u32, u32)],
) -> std::result::Result<Column, String> {
    let ColumnValues::Factor {
        levels,
        codes,
        ordered,
    } = &mut column.values
    else {
        return Ok(column);
    };
    let indices = if *ordered {
        ordered_indices(levels.len(), order).ok_or_else(|| {
            format!(
                "ordered categories in `{}` change order between Arrow batches",
                column.name
            )
        })?
    } else if matches!(
        field.data_type(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    ) {
        let mut indices: Vec<_> = (0..levels.len()).collect();
        indices.sort_by_key(|&i| &levels[i]);
        indices
    } else {
        return Ok(column);
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
    Ok(column)
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

fn column(field: &Field, values: &dyn Array, decode: bool) -> Result<Column> {
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
        // No buffer bounds a Null array's length, and every value is missing, so
        // selecting it refuses anyway. It is never materialised.
        DataType::Null => {
            return Err(Error(format!(
                "column `{}` has the Arrow type Null, so every value is missing",
                field.name()
            )));
        }
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            return factor(field, labels(values)?, (0..values.len()).map(Some));
        }
        DataType::Dictionary(key, value) => {
            let decode = decode
                && !matches!(
                    value.as_ref(),
                    DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
                );
            return match key.as_ref() {
                DataType::Int8 => dictionary::<types::Int8Type>(field, values, decode),
                DataType::Int16 => dictionary::<types::Int16Type>(field, values, decode),
                DataType::Int32 => dictionary::<types::Int32Type>(field, values, decode),
                DataType::Int64 => dictionary::<types::Int64Type>(field, values, decode),
                DataType::UInt8 => dictionary::<types::UInt8Type>(field, values, decode),
                DataType::UInt16 => dictionary::<types::UInt16Type>(field, values, decode),
                DataType::UInt32 => dictionary::<types::UInt32Type>(field, values, decode),
                DataType::UInt64 => dictionary::<types::UInt64Type>(field, values, decode),
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
    decode: bool,
) -> Result<Column> {
    let dictionary = array::<DictionaryArray<K>>(values);
    if decode {
        let inner = column(field, dictionary.values().as_ref(), true)?;
        return Ok(gather(inner, dictionary.keys_iter().collect()));
    }
    factor(
        field,
        labels(dictionary.values().as_ref())?,
        dictionary.keys_iter(),
    )
}

/// Expand dictionary values by their keys. A null key, or a key at a null
/// value, is missing.
fn gather(inner: Column, keys: Vec<Option<usize>>) -> Column {
    fn take<T: Copy + Default>(values: &[T], keys: &[Option<usize>]) -> Vec<T> {
        keys.iter()
            .map(|key| key.map_or_else(T::default, |key| values[key]))
            .collect()
    }
    let values = match &inner.values {
        ColumnValues::Real32(v) => ColumnValues::Real32(take(v, &keys)),
        ColumnValues::Real64(v) => ColumnValues::Real64(take(v, &keys)),
        ColumnValues::Int(v) => ColumnValues::Int(take(v, &keys)),
        ColumnValues::Bool(v) => ColumnValues::Bool(take(v, &keys)),
        ColumnValues::Factor {
            levels,
            codes,
            ordered,
        } => ColumnValues::Factor {
            levels: levels.clone(),
            codes: take(codes, &keys),
            ordered: *ordered,
        },
    };
    let valid: Vec<bool> = keys
        .iter()
        .map(|key| key.is_some_and(|key| inner.valid.as_ref().is_none_or(|valid| valid[key])))
        .collect();
    Column {
        name: inner.name,
        values,
        valid: valid.iter().any(|x| !x).then_some(valid),
    }
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
