//! Safetensors (§07): a nested record whose leaves are the file's tensors. A
//! key's dot-separated segments form a record path, after leading and trailing
//! dots are trimmed. Keys must be nonempty, have no empty segments and name
//! distinct paths, and no path may be both a prefix and a leaf.

use crate::{Error, Result};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

/// A record path tree: a tensor, or named children in key order.
#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    Tensor(Tensor),
    Record(Vec<(String, Entry)>),
}

/// One tensor, row-major, at its source precision. Narrower floats widen to
/// binary32 and integers to int64.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub values: TensorValues,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TensorValues {
    Real32(Vec<f32>),
    Real64(Vec<f64>),
    Int(Vec<i64>),
    Bool(Vec<bool>),
}

/// One header entry.
#[derive(serde::Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// Parse a Safetensors file into its record tree.
pub fn read(bytes: &[u8]) -> Result<Entry> {
    let fail = |message: String| Error(format!("Safetensors: {message}"));
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
        root.insert(&segments, 0, tensor).map_err(|prefix| {
            fail(format!(
                "`{prefix}` is both a tensor and a record path prefix"
            ))
        })?;
    }
    Ok(root.into_entry())
}

/// A record path tree under construction, children in key order.
#[derive(Default)]
struct Node {
    leaf: Option<Tensor>,
    children: Vec<(String, Node)>,
}

impl Node {
    /// Insert a leaf, refusing a path that is both a leaf and a prefix (§07).
    /// The error is the conflicting path.
    fn insert(
        &mut self,
        segments: &[&str],
        depth: usize,
        leaf: Tensor,
    ) -> std::result::Result<(), String> {
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
        self.children[index].1.insert(segments, depth + 1, leaf)
    }

    fn into_entry(self) -> Entry {
        match self.leaf {
            Some(leaf) => Entry::Tensor(leaf),
            None => Entry::Record(
                self.children
                    .into_iter()
                    .map(|(name, node)| (name, node.into_entry()))
                    .collect(),
            ),
        }
    }
}

/// Decode one tensor's little-endian bytes.
fn tensor(info: &TensorInfo, data: &[u8]) -> std::result::Result<Tensor, String> {
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
        ($ty:ty, $size:literal, $map:expr) => {{
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
                .as_chunks::<$size>()
                .0
                .iter()
                .map(|b| <$ty>::from_le_bytes(*b))
                .map($map)
                .collect::<std::result::Result<Vec<_>, String>>()?
        }};
    }
    let values = match info.dtype.as_str() {
        "F64" => TensorValues::Real64(decode!(f64, 8, Ok)),
        "F32" => TensorValues::Real32(decode!(f32, 4, Ok)),
        "F16" => TensorValues::Real32(decode!(u16, 2, |x| Ok(half::f16::from_bits(x).to_f32()))),
        "BF16" => TensorValues::Real32(decode!(u16, 2, |x| Ok(half::bf16::from_bits(x).to_f32()))),
        "I64" => TensorValues::Int(decode!(i64, 8, Ok)),
        "I32" => TensorValues::Int(decode!(i32, 4, |x| Ok(i64::from(x)))),
        "I16" => TensorValues::Int(decode!(i16, 2, |x| Ok(i64::from(x)))),
        "I8" => TensorValues::Int(decode!(i8, 1, |x| Ok(i64::from(x)))),
        "U64" => TensorValues::Int(decode!(u64, 8, |x| i64::try_from(x)
            .map_err(|_| format!("U64 value {x} exceeds int64")))),
        "U32" => TensorValues::Int(decode!(u32, 4, |x| Ok(i64::from(x)))),
        "U16" => TensorValues::Int(decode!(u16, 2, |x| Ok(i64::from(x)))),
        "U8" => TensorValues::Int(decode!(u8, 1, |x| Ok(i64::from(x)))),
        "BOOL" => TensorValues::Bool(decode!(u8, 1, |x| match x {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(format!("BOOL byte {other} is neither 0 nor 1")),
        })),
        other => {
            return Err(format!(
                "dtype {other} has no FlatPPL value set (§07 maps float, integer and bool \
                 dtypes)"
            ));
        }
    };
    Ok(Tensor {
        shape: info.shape.clone(),
        values,
    })
}
