//! §07 "Array and table generation" (`onehot`), "Array and table operations"
//! (`reverse`, `tile`, `splitblocks`, `joinblocks`, `partition`, `blockdiagmat`,
//! `bandedmat`), "Membership, filtering, and bin selection" (`selectbins`,
//! `filter`) and "Binning" (`bincounts`).
//!
//! StableHLO shapes are static, so every size these heads read must be written
//! as a literal or be a static operand extent. An array of arrays (§03 "Vectors
//! of vectors are not interpreted as matrices") lowers to the same tensor as a
//! matrix, so its nesting is read from the inferred type. None of these heads is in the host-batching set,
//! so their operands arrive unbatched; a batched one is refused.

use flatppl_core::{CallHead, Node, NodeId, Scalar, Type};

use crate::emitter::{Axes, Emitter};
use crate::mlir::{ElemKind, MlirTy, Value};
use crate::ops::{args_exact, get_literal_value, linspace_points};
use crate::refuse::EmitError;

fn unbatched(e: &Emitter, id: NodeId, v: &Value, head: &str) -> Result<(), EmitError> {
    if e.batch_rank(v) == 0 {
        return Ok(());
    }
    Err(EmitError::at(
        id,
        format!("`{head}` has no batched lowering here"),
    ))
}

/// Static cell dimensions of an unbatched value.
fn dims(e: &Emitter, id: NodeId, v: &Value, head: &str) -> Result<Vec<u64>, EmitError> {
    unbatched(e, id, v, head)?;
    match e.cell_ty(v) {
        MlirTy::Scalar => Ok(vec![]),
        MlirTy::Ranked(d) => d.into_iter().collect::<Option<Vec<_>>>().ok_or_else(|| {
            EmitError::at(id, format!("`{head}` needs a statically shaped operand"))
        }),
        _ => Err(EmitError::at(
            id,
            format!("`{head}` needs an array operand"),
        )),
    }
}

/// A positive size written as an integer literal or a vector of them.
fn written_sizes(e: &Emitter, id: NodeId, node: NodeId, head: &str) -> Result<Vec<u64>, EmitError> {
    let one = |n: NodeId| match e.node(n) {
        Node::Lit(Scalar::Int(k)) if *k > 0 => u64::try_from(*k).ok(),
        _ => None,
    };
    let sizes = match e.node(node) {
        Node::Call(c) if matches!(c.head, CallHead::Builtin(h) if e.resolve(h) == "vector") => {
            c.args.iter().map(|&a| one(a)).collect::<Option<Vec<_>>>()
        }
        _ => one(node).map(|k| vec![k]),
    };
    sizes.ok_or_else(|| {
        EmitError::at(
            id,
            format!("`{head}` needs its size written as positive integer literals (StableHLO shapes are static)"),
        )
    })
}

/// The outer and inner ranks of an operand's inferred type: `(rank, 0)` for a plain
/// array, `(outer, inner)` for an array of arrays. A rowstacked matrix and a vector
/// of vectors lower to the same tensor, so only the type tells them apart.
fn nest_ranks(e: &Emitter, node: NodeId) -> Option<(usize, usize)> {
    let Type::Array { shape, elem } = e.type_of(node)? else {
        return None;
    };
    match elem.as_ref() {
        Type::Array { shape: inner, elem } if !matches!(elem.as_ref(), Type::Array { .. }) => {
            Some((shape.len(), inner.len()))
        }
        Type::Array { .. } => None,
        _ => Some((shape.len(), 0)),
    }
}

fn plain(e: &Emitter, id: NodeId, node: NodeId, head: &str) -> Result<(), EmitError> {
    match nest_ranks(e, node) {
        Some((_, 0)) => Ok(()),
        _ => Err(EmitError::at(
            id,
            format!("`{head}` lowers a plain array here, not an array of arrays"),
        )),
    }
}

fn tensor(dims: &[u64]) -> MlirTy {
    if dims.is_empty() {
        MlirTy::Scalar
    } else {
        MlirTy::Ranked(dims.iter().map(|&d| Some(d)).collect())
    }
}

fn layered(e: &mut Emitter, v: &Value, layers: Vec<usize>) -> Value {
    e.axes_view(v, Axes { batch: 0, layers })
}

/// `onehot(i, n)`: 1.0 at the 1-based position `i` of a length-`n` vector. A
/// fixed `i` outside 1..n refuses, since §07 defines no value there.
pub(crate) fn lower_onehot(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [i, n] = args_exact(id, args)?;
    let [n] = written_sizes(e, id, n, "onehot")?[..] else {
        return Err(EmitError::at(id, "`onehot` takes a scalar length"));
    };
    let i = e.lower_node(i)?;
    unbatched(e, id, &i, "onehot")?;
    if e.cell_ty(&i) != MlirTy::Scalar {
        return Err(EmitError::at(
            id,
            "`onehot` under a broadcast has no lowering here: each cell is a vector",
        ));
    }
    if let Some(k) = e.constant_int(&i)
        && !(1..=n as i64).contains(&k)
    {
        return Err(EmitError::at(
            id,
            format!(
                "onehot: index {k} is outside 1..{n}; spec §07 defines `onehot(i, n)` for i ∈ {{1, …, n}}"
            ),
        ));
    }
    let i = e.checked_index(id, &i, n, 1)?;
    let idx = e.index_vector(n, false);
    let hit = e.compare("EQ", &idx, &i);
    Ok(e.convert(&hit, ElemKind::Real))
}

/// `reverse(xs)` of a vector.
pub(crate) fn lower_reverse(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [xs] = args_exact(id, args)?;
    let v = e.lower_node(xs)?;
    dims(e, id, &v, "reverse")?;
    // A vector of vectors is a vector too: its outer order reverses.
    if !matches!(nest_ranks(e, xs), Some((1, _))) {
        return Err(EmitError::at(id, "`reverse` lowers a vector here"));
    }
    Ok(e.reverse(&v, &[0]))
}

/// `tile(A, size)`: broadcast `A` to `[n₀, d₀, n₁, d₁, …]`, then merge each pair.
pub(crate) fn lower_tile(e: &mut Emitter, id: NodeId, args: &[NodeId]) -> Result<Value, EmitError> {
    let [a, size] = args_exact(id, args)?;
    let v = e.lower_node(a)?;
    let d = dims(e, id, &v, "tile")?;
    let n = written_sizes(e, id, size, "tile")?;
    plain(e, id, a, "tile")?;
    if d.is_empty() || n.len() != d.len() {
        return Err(EmitError::at(
            id,
            "`tile` needs one tile count per axis of an array (spec §07)",
        ));
    }
    let wide: Vec<u64> = n.iter().zip(&d).flat_map(|(&k, &m)| [k, m]).collect();
    let map: Vec<u64> = (0..d.len() as u64).map(|k| 2 * k + 1).collect();
    let b = e.broadcast_in_dim(&v, &map, tensor(&wide));
    let out: Vec<u64> = n.iter().zip(&d).map(|(k, m)| k * m).collect();
    Ok(e.reshape(&b, tensor(&out)))
}

/// `splitblocks(A, b)`: reshape to `[q₀, b₀, q₁, b₁, …]` and move the block axes
/// inward, an array of `q` arrays of shape `b`.
pub(crate) fn lower_splitblocks(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [a, blocks] = args_exact(id, args)?;
    let v = e.lower_node(a)?;
    let d = dims(e, id, &v, "splitblocks")?;
    let b = written_sizes(e, id, blocks, "splitblocks")?;
    plain(e, id, a, "splitblocks")?;
    if d.is_empty() || b.len() != d.len() {
        return Err(EmitError::at(
            id,
            "`splitblocks` needs one block size per axis of an array",
        ));
    }
    if d.iter().zip(&b).any(|(m, k)| m % k != 0) {
        return Err(EmitError::at(
            id,
            "splitblocks: each axis must be divisible by its block size (spec §07)",
        ));
    }
    let r = d.len();
    let pairs: Vec<u64> = d.iter().zip(&b).flat_map(|(m, k)| [m / k, *k]).collect();
    let v = e.reshape(&v, tensor(&pairs));
    let perm: Vec<u64> = (0..r as u64)
        .map(|k| 2 * k)
        .chain((0..r as u64).map(|k| 2 * k + 1))
        .collect();
    let t = e.transpose(&v, &perm);
    Ok(layered(e, &t, vec![r, r]))
}

/// `joinblocks(A)`: the inverse of `splitblocks`.
pub(crate) fn lower_joinblocks(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [a] = args_exact(id, args)?;
    let v = e.lower_node(a)?;
    let d = dims(e, id, &v, "joinblocks")?;
    // The type knows the nesting; a `splitblocks` result the type leaves open
    // carries it in its axes.
    let nesting = nest_ranks(e, a)
        .filter(|&(_, inner)| inner > 0)
        .or_else(|| match e.axes_of(&v).layers[..] {
            [outer, inner] => Some((outer, inner)),
            _ => None,
        });
    let Some((outer, inner)) = nesting else {
        return Err(EmitError::at(
            id,
            "`joinblocks` takes an array of arrays (spec §07)",
        ));
    };
    if outer != inner {
        return Err(EmitError::at(
            id,
            "joinblocks: the outer and inner arrays must have the same number of dimensions (spec §07)",
        ));
    }
    let r = outer as u64;
    let perm: Vec<u64> = (0..r).flat_map(|k| [k, r + k]).collect();
    let t = e.transpose(&v, &perm);
    let out: Vec<u64> = (0..outer).map(|k| d[k] * d[outer + k]).collect();
    Ok(e.reshape(&t, tensor(&out)))
}

/// `partition(xs, spec)` into equal groups, an array of arrays. Unequal groups
/// make a ragged array, which has no tensor form.
pub(crate) fn lower_partition(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [xs, spec] = args_exact(id, args)?;
    let v = e.lower_node(xs)?;
    let d = dims(e, id, &v, "partition")?;
    let [len] = d[..] else {
        return Err(EmitError::at(id, "`partition` splits a vector"));
    };
    plain(e, id, xs, "partition")?;
    let sizes = written_sizes(e, id, spec, "partition")?;
    let n = sizes[0];
    let k = if sizes.len() == 1 {
        if len % n != 0 {
            return Err(EmitError::at(
                id,
                format!(
                    "partition: a vector of length {len} does not split into groups of {n}; spec §07 requires the group size to divide the length"
                ),
            ));
        }
        len / n
    } else {
        let total: u64 = sizes.iter().sum();
        if total != len {
            return Err(EmitError::at(
                id,
                format!(
                    "partition: the group sizes sum to {total}, but the vector has length {len}; spec §07 requires them to sum to the length"
                ),
            ));
        }
        if sizes.iter().any(|&k| k != n) {
            return Err(EmitError::at(
                id,
                "partition: groups of unequal size make a ragged array, which has no tensor form here; equal groups lower",
            ));
        }
        sizes.len() as u64
    };
    let r = e.reshape(&v, tensor(&[k, n]));
    Ok(layered(e, &r, vec![1, 1]))
}

/// `blockdiagmat(mats)`: each matrix padded to its diagonal position, summed.
/// A written `[A, B]` keeps its blocks apart, so they may differ in shape.
pub(crate) fn lower_blockdiagmat(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [mats] = args_exact(id, args)?;
    let blocks = match e.node(mats).clone() {
        Node::Call(c) if matches!(c.head, CallHead::Builtin(h) if e.resolve(h) == "vector") => c
            .args
            .iter()
            .map(|&m| e.lower_node(m))
            .collect::<Result<Vec<_>, _>>()?,
        _ => {
            let v = e.lower_node(mats)?;
            let d = dims(e, id, &v, "blockdiagmat")?;
            if nest_ranks(e, mats) != Some((1, 2)) {
                return Err(EmitError::at(
                    id,
                    "`blockdiagmat` takes a vector of matrices (spec §07)",
                ));
            }
            (0..d[0])
                .map(|i| get_literal_value(e, id, v.clone(), i))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    let shapes = blocks
        .iter()
        .map(|b| match dims(e, id, b, "blockdiagmat")?[..] {
            [r, c] => Ok((r, c)),
            _ => Err(EmitError::at(
                id,
                "blockdiagmat: each block must be a matrix (spec §07)",
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (rows, cols) = shapes.iter().fold((0, 0), |(r, c), (a, b)| (r + a, c + b));
    if blocks.is_empty() {
        return Err(EmitError::at(
            id,
            "blockdiagmat: an empty vector of matrices has no tensor form here",
        ));
    }
    // The element kind inference gives the result: integer blocks stay integer.
    let kind = e.node_ty(id).map_or(ElemKind::Real, |(_, kind)| kind);
    let mut acc: Option<Value> = None;
    let (mut r0, mut c0) = (0, 0);
    for (b, (r, c)) in blocks.iter().zip(&shapes) {
        let b = e.convert(b, kind);
        let p = e.pad_zeros(&b, &[r0, c0], &[rows - r0 - r, cols - c0 - c]);
        acc = Some(match acc {
            None => p,
            Some(a) => e.add(&a, &p),
        });
        r0 += r;
        c0 += c;
    }
    Ok(acc.expect("nonempty"))
}

/// `bandedmat(v, rows)`: row `i` holds `v` from column `i`, in `v`'s element kind.
pub(crate) fn lower_bandedmat(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [v, rows] = args_exact(id, args)?;
    let v = e.lower_node(v)?;
    let [k] = dims(e, id, &v, "bandedmat")?[..] else {
        return Err(EmitError::at(id, "`bandedmat` takes a vector (spec §07)"));
    };
    let [m] = written_sizes(e, id, rows, "bandedmat")?[..] else {
        return Err(EmitError::at(id, "`bandedmat` takes a scalar row count"));
    };
    let row = e.reshape(&v, tensor(&[1, k]));
    let mut acc: Option<Value> = None;
    for i in 0..m {
        let p = e.pad_zeros(&row, &[i, i], &[m - 1 - i, m - 1 - i]);
        acc = Some(match acc {
            None => p,
            Some(a) => e.add(&a, &p),
        });
    }
    Ok(acc.expect("a positive row count"))
}

/// `bincounts(bins, data)` over edge vector `bins`: per bin, the number of data
/// points in `[eᵢ, eᵢ₊₁)`, the last bin closed on the right; points outside are
/// ignored. Counts are integers.
pub(crate) fn lower_bincounts(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [bins, data] = args_exact(id, args)?;
    let edges = e.lower_node(bins)?;
    let [ne] = dims(e, id, &edges, "bincounts")?[..] else {
        return Err(EmitError::at(
            id,
            "bincounts: multi-dimensional binning (a record of edge vectors) has no lowering here",
        ));
    };
    if ne < 2 {
        return Err(EmitError::at(
            id,
            "bincounts: at least two edges are needed for a bin",
        ));
    }
    let x = e.lower_node(data)?;
    let [m] = dims(e, id, &x, "bincounts")?[..] else {
        return Err(EmitError::at(id, "bincounts: the data must be a vector"));
    };
    let n = ne - 1;
    let edges = e.convert(&edges, ElemKind::Real);
    let x = e.convert(&x, ElemKind::Real);
    let lo = e.slice(&edges, &[0], &[n], &[1]);
    let hi = e.slice(&edges, &[1], &[ne], &[1]);
    let grid = tensor(&[n, m]);
    let xs = e.broadcast_in_dim(&x, &[1], grid.clone());
    let los = e.broadcast_in_dim(&lo, &[0], grid.clone());
    let his = e.broadcast_in_dim(&hi, &[0], grid.clone());
    let above = e.compare("GE", &xs, &los);
    let below_open = e.compare("LT", &xs, &his);
    let below_closed = e.compare("LE", &xs, &his);
    let row = e.index_vector(n, true);
    let last_int = e.int_value_const(n as i64 - 1);
    let is_last = e.compare("EQ", &row, &last_int);
    let is_last = e.broadcast_in_dim(&is_last, &[0], grid);
    let below = e.select(&is_last, &below_closed, &below_open);
    let inside = e.and(&above, &below);
    let ones = e.convert(&inside, ElemKind::Real);
    let counts = e.reduce_sum_last_axis(&ones);
    Ok(e.convert(&counts, ElemKind::Int))
}

/// The values of a fixed real scalar or vector, ±inf included. The emitter's
/// constant table holds finite values only, so `inf`, `-inf`, a written vector
/// holding them and `extlinspace` are read from the expression.
fn fixed_reals(e: &mut Emitter, node: NodeId) -> Option<Vec<f64>> {
    let node = e.resolve_ref_one(node);
    let scalar = |e: &mut Emitter, n: NodeId| match fixed_reals(e, n)?[..] {
        [x] => Some(x),
        _ => None,
    };
    let head = match e.node(node).clone() {
        Node::Const(s) => Some((e.resolve(s).to_owned(), Vec::new())),
        Node::Call(c) => match c.head {
            CallHead::Builtin(h) => Some((e.resolve(h).to_owned(), c.args.to_vec())),
            CallHead::User(_) => None,
        },
        _ => None,
    };
    match head.as_ref().map(|(h, a)| (h.as_str(), &a[..])) {
        Some(("inf", [])) => return Some(vec![f64::INFINITY]),
        Some(("neg", &[x])) => return scalar(e, x).map(|x| vec![-x]),
        Some(("vector", args)) => return args.iter().map(|&a| scalar(e, a)).collect(),
        Some(("extlinspace", args)) => {
            let points = linspace_points(e, node, args, true).ok()?;
            let mut edges = vec![f64::NEG_INFINITY];
            edges.extend(e.constant_reals(&points)?);
            edges.push(f64::INFINITY);
            return Some(edges);
        }
        _ => {}
    }
    let v = e.lower_node(node).ok()?;
    let v = if e.cell_ty(&v) == MlirTy::Scalar {
        e.reshape(&v, MlirTy::Ranked(vec![Some(1)]))
    } else {
        v
    };
    e.constant_reals(&v)
}

/// The closed or half-open bounds of a `selectbins` region: `(lo, lo_open, hi)`.
fn region_bounds(
    e: &mut Emitter,
    id: NodeId,
    region: NodeId,
) -> Result<(f64, bool, f64), EmitError> {
    let fixed = |e: &mut Emitter, n: NodeId| match fixed_reals(e, n)?[..] {
        [x] => Some(x),
        _ => None,
    };
    let resolved = e.resolve_ref_one(region);
    match e.node(resolved).clone() {
        Node::Const(s) => match e.resolve(s) {
            "reals" => Ok((f64::NEG_INFINITY, false, f64::INFINITY)),
            "posreals" => Ok((0.0, true, f64::INFINITY)),
            "nonnegreals" => Ok((0.0, false, f64::INFINITY)),
            "unitinterval" => Ok((0.0, false, 1.0)),
            other => Err(EmitError::at(
                id,
                format!("selectbins: region `{other}` is not an interval of reals"),
            )),
        },
        Node::Call(c)
            if matches!(c.head, CallHead::Builtin(h) if e.resolve(h) == "interval")
                && c.args.len() == 2 =>
        {
            match (fixed(e, c.args[0]), fixed(e, c.args[1])) {
                (Some(lo), Some(hi)) => Ok((lo, false, hi)),
                _ => Err(EmitError::at(
                    id,
                    "selectbins: the region bounds must be fixed, since they decide the result length",
                )),
            }
        }
        _ => Err(EmitError::at(
            id,
            "selectbins: the region must be an interval or a predefined interval set",
        )),
    }
}

/// `selectbins(edges, region, counts)`: the counts of the bins that intersect
/// `region`. Bins are `[eᵢ, eᵢ₊₁)`, the last closed, as in `bincounts`. The
/// selection fixes the result length, so edges and region must be fixed.
pub(crate) fn lower_selectbins(
    e: &mut Emitter,
    id: NodeId,
    args: &[NodeId],
) -> Result<Value, EmitError> {
    let [edges, region, counts] = args_exact(id, args)?;
    let ev = e.lower_node(edges)?;
    dims(e, id, &ev, "selectbins")?;
    let Some(edges) = fixed_reals(e, edges) else {
        return Err(EmitError::at(
            id,
            "selectbins: the edges must be fixed, since they decide the result length",
        ));
    };
    let (lo, lo_open, hi) = region_bounds(e, id, region)?;
    let cv = e.lower_node(counts)?;
    let [n] = dims(e, id, &cv, "selectbins")?[..] else {
        return Err(EmitError::at(id, "selectbins: the counts must be a vector"));
    };
    if edges.len() as u64 != n + 1 {
        return Err(EmitError::at(
            id,
            "selectbins: `edges` must be one longer than `counts` (spec §07)",
        ));
    }
    let last = edges.len() - 2;
    let picked = (0..=last)
        .filter(|&i| {
            let (a, b) = (edges[i], edges[i + 1]);
            let starts = a <= hi;
            let ends = if i == last { b >= lo } else { b > lo };
            lo <= hi && starts && ends && !(lo_open && b <= lo)
        })
        .map(|i| get_literal_value(e, id, cv.clone(), i as u64))
        .collect::<Result<Vec<_>, _>>()?;
    if picked.is_empty() {
        return Err(EmitError::at(
            id,
            "selectbins: no bin meets the region, and an empty result has no tensor form here",
        ));
    }
    Ok(e.vector(&picked))
}

/// `filter`'s result length depends on the data, which StableHLO's static
/// shapes cannot carry.
pub(crate) fn refuse_filter(id: NodeId) -> EmitError {
    EmitError::at(
        id,
        "`filter` keeps a data-dependent number of elements, and StableHLO shapes are static; \
         weight or mask the elements instead (for example with `ifelse`), or filter fixed data \
         before the query",
    )
}
