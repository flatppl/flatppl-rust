//! Pack independent sums of static selections from one tensor. Unequal lengths
//! use an appended zero row, never a masked live value (which may be NaN/inf).
//! This terminal pass reads typed provenance and edits only exclusive view chains.

use super::*;

struct Segment {
    result: Value,
    source: Value,
    axis: usize,
    indices: Vec<u64>,
    retained: Vec<u64>,
    views: Vec<usize>,
    line: usize,
    init: String,
}

fn segment(
    out: &Emitter<'_>,
    producer: &pointwise::Producer,
    definitions: &HashMap<&str, usize>,
    uses: &HashMap<&str, usize>,
) -> Option<Segment> {
    let Pointwise::Reduce(input, axis, op, init) = &producer.op else {
        return None;
    };
    if op != "stablehlo.add"
        || !matches!(init.as_str(), "0" | "0.000000e+00")
        || producer.value.elem == ElemKind::Bool
    {
        return None;
    }
    let mut value = input;
    let mut order = (0..shape(&value.ty).len() as u64).collect::<Vec<_>>();
    let mut views = Vec::new();
    loop {
        if uses.get(value.ssa.as_str()) != Some(&1) {
            return None;
        }
        views.push(*definitions.get(value.ssa.as_str())?);
        match &out.pointwise.get(&value.ssa)?.op {
            Pointwise::Reshape(input) if input.ty == value.ty => value = input,
            Pointwise::Transpose(input, perm) => {
                order = order.iter().map(|&d| perm[d as usize]).collect();
                value = input;
            }
            Pointwise::Gather(source, selected, indices) => {
                if order.remove(*axis) != *selected as u64 || indices.is_empty() {
                    return None;
                }
                let dims = shape(&source.ty);
                let expected = order.iter().map(|&d| dims[d as usize]).collect::<Vec<_>>();
                if expected != shape(&producer.value.ty)
                    || dims.iter().any(Option::is_none)
                    || source.elem != producer.value.elem
                {
                    return None;
                }
                return Some(Segment {
                    result: producer.value.clone(),
                    source: source.clone(),
                    axis: *selected,
                    indices: indices.clone(),
                    retained: order,
                    views,
                    line: *definitions.get(producer.value.ssa.as_str())?,
                    init: init.clone(),
                });
            }
            _ => return None,
        }
    }
}

/// Return the final live body. No emitter cache is used after any rewrite.
pub(super) fn finish(out: &Emitter<'_>, rets: &[&Value]) -> String {
    let lines = out.live_lines(rets);
    if out.cur_key.is_some() {
        return lines.join("\n");
    }
    let mut definitions = HashMap::new();
    let mut uses = HashMap::new();
    for value in rets {
        *uses.entry(value.ssa.as_str()).or_insert(0) += 1;
    }
    for (i, line) in lines.iter().enumerate() {
        let Some((ssa, rhs)) = line.split_once(" = ") else {
            return lines.join("\n");
        };
        // Region and multi-result forms retain their original scoped lowering.
        if !ssa
            .strip_prefix('%')
            .is_some_and(|name| !name.is_empty() && name.bytes().all(|c| c.is_ascii_digit()))
        {
            return lines.join("\n");
        }
        definitions.insert(ssa, i);
        for (_, input) in packing::ssa_uses(rhs) {
            *uses.entry(input).or_insert(0) += 1;
        }
    }
    let mut groups: Vec<Vec<Segment>> = Vec::new();
    let mut by_source = HashMap::new();
    for line in &lines {
        let (ssa, _) = line.split_once(" = ").unwrap();
        let Some(producer) = out.pointwise.get(ssa) else {
            continue;
        };
        let Some(segment) = segment(out, producer, &definitions, &uses) else {
            continue;
        };
        let key = (
            segment.source.ssa.clone(),
            segment.axis,
            segment.retained.clone(),
        );
        let group = *by_source.entry(key).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[group].push(segment);
    }
    let mut edits = HashMap::new();
    let mut removed = HashSet::new();
    let mut next = out.next;
    for group in groups.into_iter().filter(|g| g.len() > 1) {
        let maximum = group.iter().map(|s| s.indices.len()).max().unwrap();
        let rows = group.iter().map(|s| s.indices.len()).sum::<usize>();
        let Some(padded) = maximum.checked_mul(group.len()) else {
            continue;
        };
        // Bound extra reduction work to the number of live rows.
        if padded - rows > rows {
            continue;
        }
        let first = &group[0];
        let width = shape(&first.source.ty)[first.axis].unwrap();
        if width >= i64::MAX as u64
            || (matches!(out.dtype, Dtype::F32) && i32::try_from(width).is_err())
        {
            continue;
        }
        // Fresh constants must dominate every replacement. The original cache
        // can hold an identical index tensor defined after the first segment.
        let mut scratch = Emitter::new(out.m, out.dtype);
        scratch.next = next;
        let sums = emit_sums(&mut scratch, &group, maximum);
        let shared = scratch.body.clone();
        scratch.body.clear();
        for (lane, segment) in group.iter().enumerate() {
            let mut starts = vec![0; shape(&sums.ty).len()];
            let mut limits = shape(&sums.ty)
                .iter()
                .map(|d| d.unwrap())
                .collect::<Vec<_>>();
            starts[0] = lane as u64;
            limits[0] = lane as u64 + 1;
            let view = scratch.slice(&sums, &starts, &limits, &vec![1; starts.len()]);
            scratch.push(&format!(
                "{} = stablehlo.reshape {} : ({}) -> {}",
                segment.result.ssa,
                view.ssa,
                view.ty.render(out.dtype, view.elem),
                segment.result.ty.render(out.dtype, segment.result.elem),
            ));
            let mut replacement = std::mem::take(&mut scratch.body);
            if lane == 0 {
                replacement.insert_str(0, &shared);
            }
            edits.insert(segment.line, replacement);
            removed.extend(segment.views.iter().copied());
        }
        next = scratch.next;
    }
    let mut body = String::new();
    for (i, line) in lines.iter().enumerate() {
        if let Some(replacement) = edits.get(&i) {
            body.push_str(replacement);
        } else if !removed.contains(&i) {
            body.push_str(line);
            body.push('\n');
        }
    }
    body
}

fn emit_sums(out: &mut Emitter<'_>, segments: &[Segment], maximum: usize) -> Value {
    let first = &segments[0];
    let source = &first.source;
    let axis = first.axis;
    let dims = shape(&source.ty);
    let scalar_ty = MlirTy::Scalar.render(out.dtype, source.elem);
    let zero = Value {
        ssa: out.pure(format!(
            "stablehlo.constant dense<{}> : {scalar_ty}",
            first.init
        )),
        ty: MlirTy::Scalar,
        elem: source.elem,
    };
    let gathered = padded_gather(out, segments, maximum, &zero);
    let sums = out.reduce_axis_lit("stablehlo.add", &first.init, &gathered, axis + 1);
    let mut perm = vec![0];
    perm.extend(
        first
            .retained
            .iter()
            .map(|&d| 1 + d - u64::from(d > axis as u64)),
    );
    if perm.iter().copied().eq(0..dims.len() as u64) {
        sums
    } else {
        out.transpose(&sums, &perm)
    }
}

fn padded_gather(
    out: &mut Emitter<'_>,
    segments: &[Segment],
    maximum: usize,
    zero: &Value,
) -> Value {
    let first = &segments[0];
    let source = &first.source;
    let axis = first.axis;
    let dims = shape(&source.ty);
    let width = dims[axis].unwrap();
    let mut row_dims = dims.to_vec();
    row_dims[axis] = Some(1);
    let row = out.broadcast_in_dim(zero, &[], MlirTy::Ranked(row_dims.clone()));
    let sizes = row_dims
        .iter()
        .map(|d| d.unwrap().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    row_dims[axis] = Some(width + 1);
    let extended_ty = MlirTy::Ranked(row_dims);
    let extended = out.pure(format!(
        "stablehlo.concatenate {}, {}, dim = {axis} : ({}, {}) -> {}",
        source.ssa,
        row.ssa,
        source.ty.render(out.dtype, source.elem),
        row.ty.render(out.dtype, row.elem),
        extended_ty.render(out.dtype, source.elem),
    ));
    let count = segments.len() as u64;
    let index = out
        .folded_constant(
            segments
                .iter()
                .flat_map(|s| {
                    (0..maximum).map(move |i| {
                        Scalar::Int(s.indices.get(i).copied().unwrap_or(width) as i64)
                    })
                })
                .collect(),
            MlirTy::Ranked(vec![Some(count), Some(maximum as u64), Some(1)]),
            Axes::default(),
        )
        .expect("validated in-bounds indices fit the target integer type");
    // Keep the source axis order inside each segment and preserve its ordered
    // indices. Only the new leading axis separates independent sums.
    let mut gathered_dims = dims.to_vec();
    gathered_dims[axis] = Some(maximum as u64);
    gathered_dims.insert(0, Some(count));
    let ty = MlirTy::Ranked(gathered_dims);
    let offsets = (0..dims.len())
        .filter(|&d| d != axis)
        .map(|d| (d + 1).to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Value {
        ssa: out.pure(format!(
            "\"stablehlo.gather\"({extended}, {}) <{{dimension_numbers = #stablehlo.gather<offset_dims = [{offsets}], collapsed_slice_dims = [{axis}], start_index_map = [{axis}], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: {sizes}>}}> : ({}, {}) -> {}",
            index.ssa, extended_ty.render(out.dtype, source.elem),
            index.ty.render(out.dtype, ElemKind::Int), ty.render(out.dtype, source.elem),
        )),
        ty,
        elem: source.elem,
    }
}
