//! Pack independent reductions of static selections from the same tensors. Unequal
//! lengths use an identity row, never a masked live value (which may be NaN/inf).
//! This terminal pass reads typed provenance and edits only exclusive view chains.

use super::*;

struct Segment {
    result: Value,
    sources: Vec<Value>,
    axis: usize,
    indices: Vec<u64>,
    retained: Vec<u64>,
    views: Vec<usize>,
    line: usize,
    op: String,
    init: String,
}

/// Recover a static selection through singleton-only reshapes. Other reshapes
/// can mix retained axes and are not views of columns along the reduction axis.
fn selection<'a>(
    out: &'a Emitter<'_>,
    value: &'a Value,
    axis: usize,
) -> Option<(&'a Value, Vec<u64>)> {
    let mut current = value;
    while let Some(pointwise::Producer {
        op: Pointwise::Reshape(input),
        ..
    }) = out.pointwise.get(&current.ssa)
    {
        if shape(&current.ty)
            .iter()
            .filter(|&&d| d != Some(1))
            .ne(shape(&input.ty).iter().filter(|&&d| d != Some(1)))
        {
            return None;
        }
        current = input;
    }
    let (mut source, indices) = match &out.pointwise.get(&current.ssa)?.op {
        Pointwise::Gather(source, selected, indices) if *selected == axis => {
            (source, indices.clone())
        }
        Pointwise::Slice(source, starts, limits, strides) => {
            let dims = shape(&source.ty);
            let index = *starts.get(axis)?;
            if limits.get(axis).copied() != index.checked_add(1)
                || strides.iter().any(|&stride| stride != 1)
                || dims.iter().enumerate().any(|(d, &size)| {
                    size.is_none() || (d != axis && (starts[d] != 0 || Some(limits[d]) != size))
                })
            {
                return None;
            }
            (source, vec![index])
        }
        _ => return None,
    };
    let mut selected_shape = shape(&source.ty).to_vec();
    let width = selected_shape.get(axis).copied().flatten()?;
    if indices.is_empty() || indices.iter().any(|&index| index >= width) {
        return None;
    }
    selected_shape[axis] = Some(indices.len() as u64);
    if selected_shape != shape(&value.ty) || source.elem != value.elem {
        return None;
    }
    while let Some(pointwise::Producer {
        op: Pointwise::Reshape(input),
        ..
    }) = out.pointwise.get(&source.ssa)
    {
        if input.ty != source.ty {
            break;
        }
        source = input;
    }
    Some((source, indices))
}

fn concat_selection(
    out: &Emitter<'_>,
    parts: &[Value],
    axis: usize,
) -> Option<(Vec<Value>, Vec<u64>)> {
    let selections = parts
        .iter()
        .map(|part| selection(out, part, axis))
        .collect::<Option<Vec<_>>>()?;
    let mut sources = selections
        .iter()
        .map(|(source, _)| *source)
        .collect::<Vec<_>>();
    sources.sort_unstable_by_key(|source| &source.ssa);
    sources.dedup_by_key(|source| &source.ssa);
    let first = sources.first()?;
    let mut width = 0_u64;
    let mut offsets = HashMap::new();
    for source in &sources {
        let dims = shape(&source.ty);
        if source.elem != first.elem
            || dims.len() != shape(&first.ty).len()
            || dims
                .iter()
                .enumerate()
                .any(|(d, size)| size.is_none() || (d != axis && size != &shape(&first.ty)[d]))
        {
            return None;
        }
        offsets.insert(source.ssa.as_str(), width);
        width = width.checked_add(dims[axis]?)?;
    }
    let indices = selections
        .iter()
        .flat_map(|(source, selected)| {
            let offset = offsets[source.ssa.as_str()];
            selected.iter().map(move |&index| offset + index)
        })
        .collect();
    Some((sources.into_iter().cloned().collect(), indices))
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
    let identity = match op.as_str() {
        "stablehlo.add" => matches!(init.as_str(), "0" | "0.000000e+00"),
        "stablehlo.multiply" => matches!(init.as_str(), "1" | "1.000000e+00"),
        _ => false,
    };
    if !identity || producer.value.elem == ElemKind::Bool {
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
        let (sources, selected, indices) = match &out.pointwise.get(&value.ssa)?.op {
            Pointwise::Reshape(input) if input.ty == value.ty => {
                value = input;
                continue;
            }
            Pointwise::Transpose(input, perm) => {
                order = order.iter().map(|&d| perm[d as usize]).collect();
                value = input;
                continue;
            }
            Pointwise::Gather(source, selected, indices) => {
                (vec![source.clone()], *selected, indices.clone())
            }
            Pointwise::Concat(parts, selected) => {
                let (sources, indices) = concat_selection(out, parts, *selected)?;
                (sources, *selected, indices)
            }
            _ => return None,
        };
        if order.remove(*axis) != selected as u64 || indices.is_empty() {
            return None;
        }
        let dims = shape(&sources[0].ty);
        let expected = order.iter().map(|&d| dims[d as usize]).collect::<Vec<_>>();
        if expected != shape(&producer.value.ty)
            || dims.iter().any(Option::is_none)
            || sources[0].elem != producer.value.elem
        {
            return None;
        }
        return Some(Segment {
            result: producer.value.clone(),
            sources,
            axis: selected,
            indices,
            retained: order,
            views,
            line: *definitions.get(producer.value.ssa.as_str())?,
            op: op.clone(),
            init: init.clone(),
        });
    }
}

/// Absorb whole exact-source groups into the smallest admissible superset.
/// Never move a source before its definition or discard a rejected exact group.
fn absorb_subsets(
    out: &Emitter<'_>,
    args: &[(String, MlirTy, ElemKind)],
    definitions: &HashMap<&str, usize>,
    groups: &mut [Vec<Segment>],
) {
    let mut order = (0..groups.len()).collect::<Vec<_>>();
    order.sort_by_key(|&i| (groups[i][0].sources.len(), groups[i][0].line));
    let mut rank = vec![0; groups.len()];
    let mut by_first_source: HashMap<String, Vec<usize>> = HashMap::new();
    for (position, &i) in order.iter().enumerate() {
        rank[i] = position;
        by_first_source
            .entry(groups[i][0].sources[0].ssa.clone())
            .or_default()
            .push(i);
    }
    for &target in &order {
        let Some(first) = groups[target].first() else {
            continue;
        };
        // A subset's first source must occur in the superset. Avoid visiting
        // every unrelated group while retaining the original candidate order.
        let mut candidates = first
            .sources
            .iter()
            .filter_map(|s| by_first_source.get(&s.ssa))
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        candidates.sort_by_key(|&i| rank[i]);
        for other in candidates {
            let (Some(superset), Some(subset)) = (groups[target].first(), groups[other].first())
            else {
                continue;
            };
            if subset.sources.len() >= superset.sources.len()
                || subset.axis != superset.axis
                || subset.retained != superset.retained
                || subset.op != superset.op
                || subset.init != superset.init
                || subset.result.ty != superset.result.ty
                || subset.result.elem != superset.result.elem
                || !subset.sources.iter().all(|s| superset.sources.contains(s))
            {
                continue;
            }
            let first_line = superset.line.min(subset.line);
            if superset.sources.iter().any(|source| {
                !definitions
                    .get(source.ssa.as_str())
                    .is_some_and(|&line| line < first_line)
                    && !args.iter().any(|(name, ty, elem)| {
                        name == &source.ssa && ty == &source.ty && elem == &source.elem
                    })
            }) {
                continue;
            }
            let segments = groups[target].iter().chain(&groups[other]);
            let maximum = segments.clone().map(|s| s.indices.len()).max().unwrap();
            let Some(rows) = segments
                .clone()
                .try_fold(0_usize, |n, s| n.checked_add(s.indices.len()))
            else {
                continue;
            };
            let Some(padded) = maximum.checked_mul(segments.count()) else {
                continue;
            };
            if padded - rows > rows {
                continue;
            }
            let mut offsets = HashMap::new();
            let Some(width) = superset.sources.iter().try_fold(0_u64, |offset, source| {
                offsets.insert(source.ssa.as_str(), offset);
                offset.checked_add(shape(&source.ty)[superset.axis]?)
            }) else {
                continue;
            };
            if width >= i64::MAX as u64
                || (matches!(out.dtype, Dtype::F32) && i32::try_from(width).is_err())
            {
                continue;
            }
            let mut start = 0;
            let spans = subset
                .sources
                .iter()
                .map(|source| {
                    let end = start + shape(&source.ty)[subset.axis].unwrap();
                    let span = (start..end, offsets[source.ssa.as_str()]);
                    start = end;
                    span
                })
                .collect::<Vec<_>>();
            let remapped = groups[other]
                .iter()
                .map(|segment| {
                    segment
                        .indices
                        .iter()
                        .map(|index| {
                            spans.iter().find_map(|(range, offset)| {
                                range
                                    .contains(index)
                                    .then(|| offset + (index - range.start))
                            })
                        })
                        .collect::<Option<Vec<_>>>()
                })
                .collect::<Option<Vec<_>>>();
            let Some(remapped) = remapped else {
                continue;
            };
            let sources = superset.sources.clone();
            let mut moved = std::mem::take(&mut groups[other]);
            for (segment, indices) in moved.iter_mut().zip(remapped) {
                segment.sources = sources.clone();
                segment.indices = indices;
            }
            groups[target].append(&mut moved);
            groups[target].sort_by_key(|s| s.line);
        }
    }
}

/// Return the final live body. No emitter cache is used after any rewrite.
pub(super) fn finish(
    out: &Emitter<'_>,
    args: &[(String, MlirTy, ElemKind)],
    rets: &[&Value],
) -> String {
    let lines = out.live_lines(&out.body, rets);
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
            segment
                .sources
                .iter()
                .map(|source| source.ssa.clone())
                .collect::<Vec<_>>(),
            segment.axis,
            segment.retained.clone(),
            segment.op.clone(),
            segment.init.clone(),
        );
        let group = *by_source.entry(key).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[group].push(segment);
    }
    absorb_subsets(out, args, &definitions, &mut groups);
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
        let Some(width) = first.sources.iter().try_fold(0_u64, |width, source| {
            width.checked_add(shape(&source.ty)[first.axis]?)
        }) else {
            continue;
        };
        if width >= i64::MAX as u64
            || (matches!(out.dtype, Dtype::F32) && i32::try_from(width).is_err())
        {
            continue;
        }
        // Fresh constants must dominate every replacement. The original cache
        // can hold an identical index tensor defined after the first segment.
        let mut scratch = out.scratch_emitter();
        scratch.next = next;
        let reduced = emit_reductions(&mut scratch, &group, maximum);
        let shared = scratch.body.clone();
        scratch.body.clear();
        for (lane, segment) in group.iter().enumerate() {
            let mut starts = vec![0; shape(&reduced.ty).len()];
            let mut limits = shape(&reduced.ty)
                .iter()
                .map(|d| d.unwrap())
                .collect::<Vec<_>>();
            starts[0] = lane as u64;
            limits[0] = lane as u64 + 1;
            let view = scratch.slice(&reduced, &starts, &limits, &vec![1; starts.len()]);
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
    // Removed selections can orphan their scalar slice/reshape children.
    out.live_lines(&body, rets).join("\n")
}

fn emit_reductions(out: &mut Emitter<'_>, segments: &[Segment], maximum: usize) -> Value {
    let first = &segments[0];
    let source = &first.sources[0];
    let axis = first.axis;
    let dims = shape(&source.ty);
    let scalar_ty = MlirTy::Scalar.render(out.dtype, source.elem);
    let identity = Value {
        ssa: out.pure(format!(
            "stablehlo.constant dense<{}> : {scalar_ty}",
            first.init
        )),
        ty: MlirTy::Scalar,
        elem: source.elem,
    };
    let gathered = padded_gather(out, segments, maximum, &identity);
    let reduced = out.reduce_axis_lit(&first.op, &first.init, &gathered, axis + 1);
    let mut perm = vec![0];
    perm.extend(
        first
            .retained
            .iter()
            .map(|&d| 1 + d - u64::from(d > axis as u64)),
    );
    if perm.iter().copied().eq(0..dims.len() as u64) {
        reduced
    } else {
        out.transpose(&reduced, &perm)
    }
}

fn padded_gather(
    out: &mut Emitter<'_>,
    segments: &[Segment],
    maximum: usize,
    identity: &Value,
) -> Value {
    let first = &segments[0];
    let source = &first.sources[0];
    let axis = first.axis;
    let dims = shape(&source.ty);
    let width = first
        .sources
        .iter()
        .map(|source| shape(&source.ty)[axis].unwrap())
        .sum::<u64>();
    let mut row_dims = dims.to_vec();
    row_dims[axis] = Some(1);
    let row = out.broadcast_in_dim(identity, &[], MlirTy::Ranked(row_dims.clone()));
    let sizes = row_dims
        .iter()
        .map(|d| d.unwrap().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    row_dims[axis] = Some(width + 1);
    let extended_ty = MlirTy::Ranked(row_dims);
    let names = first
        .sources
        .iter()
        .chain([&row])
        .map(|source| source.ssa.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let types = first
        .sources
        .iter()
        .chain([&row])
        .map(|source| source.ty.render(out.dtype, source.elem))
        .collect::<Vec<_>>()
        .join(", ");
    let extended = out.pure(format!(
        "stablehlo.concatenate {names}, dim = {axis} : ({types}) -> {}",
        extended_ty.render(out.dtype, source.elem),
    ));
    let count = segments.len() as u64;
    // A complete sequential table is a view of the padded source. Preserve
    // identity positions and the reduction axis, including the last short row.
    if let Some(rows) = count.checked_mul(maximum as u64)
        && rows <= width + 1
        && segments
            .iter()
            .flat_map(|s| (0..maximum).map(move |i| s.indices.get(i).copied().unwrap_or(width)))
            .eq(0..rows)
    {
        let input = Value {
            ssa: extended,
            ty: extended_ty,
            elem: source.elem,
        };
        let mut limits = dims.iter().map(|d| d.unwrap()).collect::<Vec<_>>();
        limits[axis] = rows;
        let selected = out.slice(&input, &vec![0; dims.len()], &limits, &vec![1; dims.len()]);
        let mut split = dims.to_vec();
        split[axis] = Some(maximum as u64);
        split.insert(axis, Some(count));
        let view = out.reshape(&selected, MlirTy::Ranked(split));
        let mut perm = vec![axis as u64];
        perm.extend((0..dims.len() as u64 + 1).filter(|&d| d != axis as u64));
        return out.transpose(&view, &perm);
    }
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
    // indices. Only the new leading axis separates independent reductions.
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
