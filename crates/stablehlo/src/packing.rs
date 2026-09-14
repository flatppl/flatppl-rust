//! Pack independent pointwise operations into an extra tensor axis. Recipes
//! supply types and semantics; opaque instructions only have their SSA uses
//! renamed. Each value has one canonical packet, with ordered views at uses.

use super::*;

const PACKET_AXIS: usize = 0;

struct Instruction<'a> {
    rhs: &'a str,
    inputs: Vec<&'a str>,
    depth: usize,
    pure: bool,
    class: usize,
}

type Signature = (
    usize,
    usize,
    (String, Vec<u64>),
    MlirTy,
    ElemKind,
    Axes,
    Vec<(MlirTy, ElemKind, Axes, usize)>,
);

struct Group {
    values: Vec<Value>,
    packet: Option<Value>,
}

struct Packer<'a, 'm> {
    source: &'a Emitter<'m>,
    out: Emitter<'m>,
    instructions: HashMap<&'a str, Instruction<'a>>,
    groups: Vec<Group>,
    members: HashMap<String, (usize, usize)>,
    originals: HashMap<String, String>,
    packets: HashMap<Vec<String>, Value>,
}

fn dimensions(ty: &MlirTy) -> Option<Vec<u64>> {
    match ty {
        MlirTy::Scalar => Some(vec![]),
        MlirTy::Ranked(dims) => dims.iter().copied().collect(),
        _ => None,
    }
}

pub(super) fn ssa_uses(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.match_indices('%').map(|(start, _)| {
        let end = text[start + 1..]
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .map_or(text.len(), |n| start + 1 + n);
        (start, &text[start..end])
    })
}

pub(super) fn pack(
    source: &Emitter<'_>,
    args: &[(String, MlirTy, ElemKind)],
    rets: &[&Value],
) -> Option<(String, Vec<Value>)> {
    // RNG state and region-local recipes never enter the straight-line pass.
    if source.cur_key.is_some() {
        return None;
    }
    let lines = source.live_lines(rets);
    let mut instructions: HashMap<&str, Instruction<'_>> = HashMap::new();
    let mut groups: Vec<Group> = Vec::new();
    let mut classes: HashMap<Signature, usize> = HashMap::new();
    let arguments = args
        .iter()
        .map(|(name, ..)| name.as_str())
        .collect::<HashSet<_>>();
    let pure = source
        .pure_ops
        .values()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let mut schedule = Vec::new();
    let mut segment = 0;
    for line in &lines {
        let (ssa, rhs) = line.trim().split_once(" = ")?;
        if !ssa.strip_prefix('%')?.bytes().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let inputs = ssa_uses(rhs).map(|(_, name)| name).collect::<Vec<_>>();
        let mut depth = 0;
        for input in &inputs {
            if let Some(op) = instructions.get(input) {
                depth = depth.max(op.depth);
            } else if !arguments.contains(input) {
                // Block arguments, multi-result projections and other region
                // forms retain the original lowering rather than escaping scope.
                return None;
            }
        }
        depth += 1;
        let is_pure = pure.contains(ssa);
        schedule.push((segment, depth, ssa));
        instructions.insert(
            ssa,
            Instruction {
                rhs,
                inputs,
                depth,
                pure: is_pure,
                class: usize::from(source.constants.contains_key(ssa)),
            },
        );
        if !is_pure {
            segment += 1;
            continue;
        }
        let Some(producer) = source.pointwise.get(ssa) else {
            continue;
        };
        let Some(signature) = producer.op.signature() else {
            continue;
        };
        let operands = producer.op.inputs();
        if dimensions(&producer.value.ty).is_none()
            || operands.iter().any(|v| dimensions(&v.ty).is_none())
        {
            continue;
        }
        let key = (
            segment,
            depth,
            signature,
            producer.value.ty.clone(),
            producer.value.elem,
            source.axes_of(&producer.value),
            operands
                .iter()
                .map(|v| {
                    (
                        v.ty.clone(),
                        v.elem,
                        source.axes_of(v),
                        instructions.get(v.ssa.as_str()).map_or(0, |op| op.class),
                    )
                })
                .collect(),
        );
        let group = *classes.entry(key).or_insert_with(|| {
            groups.push(Group {
                values: vec![],
                packet: None,
            });
            groups.len() - 1
        });
        groups[group].values.push(producer.value.clone());
        // Intern the ordered producer-tree shape once. Matching only opcode
        // and depth mixes unrelated chains and adds costly packet permutations.
        instructions.get_mut(ssa).unwrap().class = group + 2;
    }
    groups.retain(|g| g.values.len() > 1);
    if groups.is_empty() {
        return None;
    }
    let members = groups
        .iter()
        .enumerate()
        .flat_map(|(group, members)| {
            members
                .values
                .iter()
                .enumerate()
                .map(move |(lane, v)| (v.ssa.clone(), (group, lane)))
        })
        .collect();
    let mut out = Emitter::new(source.m, source.dtype);
    out.next = source.next;
    let mut packer = Packer {
        source,
        out,
        instructions,
        groups,
        members,
        originals: args
            .iter()
            .map(|(name, ..)| (name.clone(), name.clone()))
            .collect(),
        packets: HashMap::new(),
    };
    // Build dependencies first so a long expression chain does not consume
    // the call stack. Segment order also preserves every opaque barrier.
    schedule.sort_by_key(|&(segment, depth, _)| (segment, depth));
    for (_, _, name) in schedule {
        if let Some(&(group, _)) = packer.members.get(name) {
            packer.group_packet(group);
            if matches!(source.pointwise[name].op, Pointwise::Broadcast(..)) {
                packer.original(name);
            }
        } else {
            packer.original(name);
        }
    }
    let returns = rets
        .iter()
        .map(|v| Value {
            ssa: packer.original(&v.ssa),
            ..(*v).clone()
        })
        .collect::<Vec<_>>();
    let refs = returns.iter().collect::<Vec<_>>();
    let packed = packer.out.live_lines(&refs);
    // Small cones can cost more views than they save in arithmetic. Preserve
    // their original text as well as avoiding extra work for the backend.
    (packed.len() < lines.len()).then(|| (packed.join("\n"), returns))
}

impl Packer<'_, '_> {
    fn stacked(&self, value: &Value, count: usize) -> MlirTy {
        let mut dims = shape(&value.ty).to_vec();
        dims.insert(PACKET_AXIS, Some(count as u64));
        MlirTy::Ranked(dims)
    }

    fn original(&mut self, name: &str) -> String {
        if let Some(value) = self.originals.get(name) {
            return value.clone();
        }
        let out = if let Some(&(group, lane)) = self.members.get(name) {
            let value = self.groups[group].values[lane].clone();
            if let Pointwise::Broadcast(input, dims) = self.source.pointwise[name].op.clone() {
                let input = Value {
                    ssa: self.original(&input.ssa),
                    ..input
                };
                self.out.broadcast_in_dim(&input, &dims, value.ty).ssa
            } else {
                let packet = self.group_packet(group);
                let selected = self.select(&packet, PACKET_AXIS, &[lane]);
                self.squeeze(&selected, PACKET_AXIS, value.ty).ssa
            }
        } else {
            let op = &self.instructions[name];
            let rhs = op.rhs;
            let pure = op.pure;
            let inputs = op.inputs.clone();
            let replacements = inputs
                .iter()
                .map(|name| self.original(name))
                .collect::<Vec<_>>();
            let mut rewritten = String::with_capacity(rhs.len());
            let mut start = 0;
            for ((offset, name), replacement) in ssa_uses(rhs).zip(replacements) {
                rewritten.push_str(&rhs[start..offset]);
                rewritten.push_str(&replacement);
                start = offset + name.len();
            }
            rewritten.push_str(&rhs[start..]);
            if pure {
                self.out.pure(rewritten)
            } else {
                let ssa = self.out.fresh();
                self.out.push(&format!("{ssa} = {rewritten}"));
                ssa
            }
        };
        self.originals.insert(name.to_owned(), out.clone());
        out
    }

    fn group_packet(&mut self, group: usize) -> Value {
        if let Some(value) = &self.groups[group].packet {
            return value.clone();
        }
        let values = self.groups[group].values.clone();
        let out = self.packet(&values);
        self.groups[group].packet = Some(out.clone());
        out
    }

    fn packet(&mut self, values: &[Value]) -> Value {
        let key = values.iter().map(|v| v.ssa.clone()).collect::<Vec<_>>();
        if let Some(value) = self.packets.get(&key) {
            return value.clone();
        }
        let out = self.build_packet(values);
        self.packets.insert(key, out.clone());
        out
    }

    fn build_packet(&mut self, values: &[Value]) -> Value {
        let first = &values[0];
        let axis = PACKET_AXIS;
        let ty = self.stacked(first, values.len());
        if let Some(value) = self.constants(values) {
            return value;
        }
        if values.iter().all(|v| v.ssa == first.ssa) {
            let value = Value {
                ssa: self.original(&first.ssa),
                ..first.clone()
            };
            let dims = (0..shape(&first.ty).len())
                .map(|d| (d + usize::from(d >= axis)) as u64)
                .collect::<Vec<_>>();
            return self.out.broadcast_in_dim(&value, &dims, ty);
        }
        if let Some(value) = self.input_slices(values) {
            return value;
        }
        let group = self.members.get(&first.ssa).map(|&(group, _)| group);
        if let Some(group) = group
            && values
                .iter()
                .all(|v| self.members.get(&v.ssa).is_some_and(|&(g, _)| g == group))
        {
            let op = self.source.pointwise[&first.ssa].op.clone();
            // Select the source lanes before adding broadcast dimensions.
            // A canonical expanded packet can otherwise materialize a large
            // temporary merely to feed disjoint slices to its consumers.
            if !matches!(op, Pointwise::Broadcast(..))
                && values
                    .iter()
                    .map(|v| &v.ssa)
                    .ne(self.groups[group].values.iter().map(|v| &v.ssa))
            {
                let packet = self.group_packet(group);
                let indices = values
                    .iter()
                    .map(|v| self.members[&v.ssa].1)
                    .collect::<Vec<_>>();
                return self.select(&packet, axis, &indices);
            }
            let arity = op.inputs().len();
            let mut inputs = Vec::with_capacity(arity);
            for column in 0..arity {
                let lane_inputs = values
                    .iter()
                    .map(|v| self.source.pointwise[&v.ssa].op.inputs()[column].clone())
                    .collect::<Vec<_>>();
                inputs.push(self.packet(&lane_inputs));
            }
            let hoisted = (!matches!(op, Pointwise::Broadcast(..)))
                .then(|| self.hoist_inputs(&inputs, None))
                .flatten();
            if let Some((ref values, _)) = hoisted {
                inputs = values.clone();
            }
            let result = match op {
                Pointwise::Unary(op, _) => self.out.unary(&op, &inputs[0]),
                Pointwise::Binary(op, ..) => self.out.emit_binary(&op, &inputs[0], &inputs[1]),
                Pointwise::Compare(dir, ..) => self.out.compare(&dir, &inputs[0], &inputs[1]),
                Pointwise::Select(..) => self.out.select(&inputs[0], &inputs[1], &inputs[2]),
                Pointwise::Convert(_, elem) => self.out.convert(&inputs[0], elem),
                Pointwise::Broadcast(_, dims) => {
                    let mut dims = dims
                        .iter()
                        .map(|&d| d + u64::from(d >= axis as u64))
                        .collect::<Vec<_>>();
                    dims.insert(PACKET_AXIS, axis as u64);
                    self.out.broadcast_in_dim(&inputs[0], &dims, ty.clone())
                }
                Pointwise::Reshape(_) | Pointwise::Slice(..) => unreachable!(),
            };
            return match hoisted {
                Some((_, dims)) => self.out.broadcast_in_dim(&result, &dims, ty),
                None => result,
            };
        }
        // Keep already-packed sources together, then restore the exact order
        // requested by this consumer. No lane is recomputed for a permutation.
        let mut buckets: Vec<Vec<(usize, Value)>> = Vec::new();
        let mut positions = HashMap::new();
        for (position, value) in values.iter().enumerate() {
            let group = self.members.get(&value.ssa).map(|&(g, _)| g);
            let bucket = *positions.entry(group).or_insert_with(|| {
                buckets.push(vec![]);
                buckets.len() - 1
            });
            buckets[bucket].push((position, value.clone()));
        }
        if buckets.len() > 1 && buckets.iter().any(|rows| rows.len() > 1) {
            let mut parts = Vec::new();
            let mut inverse = vec![0; values.len()];
            let mut next = 0;
            for rows in buckets {
                for (position, _) in &rows {
                    inverse[*position] = next;
                    next += 1;
                }
                parts.push(self.packet(&rows.into_iter().map(|(_, v)| v).collect::<Vec<_>>()));
            }
            let joined = self.concatenate(&parts, axis);
            return self.select(&joined, axis, &inverse);
        }
        let single = self.stacked(first, 1);
        let parts = values
            .iter()
            .map(|value| {
                let value = Value {
                    ssa: self.original(&value.ssa),
                    ..value.clone()
                };
                self.out.reshape(&value, single.clone())
            })
            .collect::<Vec<_>>();
        self.concatenate(&parts, axis)
    }

    /// Compute pointwise packets only over axes used by their operands.
    /// This reverses consumer expansion without changing the fallback emitter.
    fn hoist_inputs(
        &mut self,
        values: &[Value],
        keep: Option<usize>,
    ) -> Option<(Vec<Value>, Vec<u64>)> {
        let rank = shape(&values[0].ty).len();
        let inputs = values
            .iter()
            .map(
                |value| match self.out.pointwise.get(&value.ssa).map(|p| &p.op) {
                    Some(Pointwise::Broadcast(base, dims)) => (base.clone(), dims.clone()),
                    _ => (value.clone(), (0..rank as u64).collect()),
                },
            )
            .collect::<Vec<_>>();
        let mut used = vec![false; rank];
        if let Some(axis) = keep {
            used[axis] = true;
        }
        for (_, dims) in &inputs {
            for &dim in dims {
                used[dim as usize] = true;
            }
        }
        let dims = used
            .iter()
            .enumerate()
            .filter_map(|(d, &used)| used.then_some(d as u64))
            .collect::<Vec<_>>();
        if dims.len() == rank {
            return None;
        }
        let values = values
            .iter()
            .zip(inputs)
            .map(|(original, (value, map))| {
                let smaller = MlirTy::Ranked(
                    dims.iter()
                        .map(|&d| shape(&original.ty)[d as usize])
                        .collect(),
                );
                let map = map
                    .iter()
                    .map(|d| dims.binary_search(d).unwrap() as u64)
                    .collect::<Vec<_>>();
                self.out.broadcast_in_dim(&value, &map, smaller)
            })
            .collect();
        Some((values, dims))
    }

    fn squeeze(&mut self, value: &Value, axis: usize, ty: MlirTy) -> Value {
        if let Some(Pointwise::Broadcast(mut base, mut map)) =
            self.out.pointwise.get(&value.ssa).map(|p| p.op.clone())
        {
            if let Some(input_axis) = map.iter().position(|&d| d == axis as u64) {
                let mut dims = shape(&base.ty).to_vec();
                let extent = dims.remove(input_axis);
                debug_assert_eq!(extent, Some(1));
                base = self.out.reshape(&base, MlirTy::Ranked(dims));
                map.remove(input_axis);
            }
            for dim in &mut map {
                *dim -= u64::from(*dim > axis as u64);
            }
            self.out.broadcast_in_dim(&base, &map, ty)
        } else {
            self.out.reshape(value, ty)
        }
    }

    fn constants(&mut self, values: &[Value]) -> Option<Value> {
        let data = values
            .iter()
            .map(|v| self.source.constants.get(&v.ssa))
            .collect::<Option<Vec<_>>>()?;
        let ty = self.stacked(&values[0], values.len());
        if data.iter().all(|v| v.len() == 1) {
            if values.iter().all(|v| v.ssa == values[0].ssa) {
                let scalar = self.out.folded_constant(
                    vec![data[0][0].clone()],
                    MlirTy::Scalar,
                    Axes::default(),
                )?;
                return Some(self.out.broadcast_in_dim(&scalar, &[], ty));
            }
            let scalars = data.iter().map(|v| v[0].clone()).collect();
            let vector = self.out.folded_constant(
                scalars,
                MlirTy::Ranked(vec![Some(values.len() as u64)]),
                Axes::default(),
            )?;
            return Some(
                self.out
                    .broadcast_in_dim(&vector, &[PACKET_AXIS as u64], ty),
            );
        }
        // A shared dense tensor stays one constant plus a broadcast, rather
        // than serializing a copy for every packet lane.
        if values.iter().all(|v| v.ssa == values[0].ssa) {
            return None;
        }
        let count = dimensions(&values[0].ty)?
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))?;
        if data.iter().any(|v| v.len() != count) {
            return None;
        }
        self.out.folded_constant(
            data.into_iter().flat_map(|v| v.iter().cloned()).collect(),
            ty,
            Axes::default(),
        )
    }

    fn input_slices(&mut self, values: &[Value]) -> Option<Value> {
        let mut source: Option<Value> = None;
        let mut selected_axis = 0;
        let mut indices = Vec::new();
        for value in values {
            let Pointwise::Reshape(part) = &self.source.pointwise.get(&value.ssa)?.op else {
                return None;
            };
            let Pointwise::Slice(base, starts, limits, strides) =
                &self.source.pointwise.get(&part.ssa)?.op
            else {
                return None;
            };
            let dims = dimensions(&base.ty)?;
            let changed = (0..dims.len())
                .filter(|&d| starts[d] != 0 || limits[d] != dims[d])
                .collect::<Vec<_>>();
            let [axis] = changed.as_slice() else {
                return None;
            };
            let axis = *axis;
            let mut cell = dims.clone();
            cell.remove(axis);
            if limits[axis] != starts[axis] + 1
                || strides.iter().any(|&s| s != 1)
                || cell != dimensions(&value.ty)?
                || source
                    .as_ref()
                    .is_some_and(|s| s.ssa != base.ssa || selected_axis != axis)
            {
                return None;
            }
            source = Some(base.clone());
            selected_axis = axis;
            indices.push(usize::try_from(starts[axis]).ok()?);
        }
        let mut source = source?;
        source.ssa = self.original(&source.ssa);
        let selected = self.select(&source, selected_axis, &indices);
        let mut perm = (0..shape(&selected.ty).len() as u64).collect::<Vec<_>>();
        let moved = perm.remove(selected_axis);
        perm.insert(PACKET_AXIS, moved);
        Some(self.out.transpose(&selected, &perm))
    }

    fn select(&mut self, source: &Value, axis: usize, indices: &[usize]) -> Value {
        let mut dims = dimensions(&source.ty).expect("packet dimensions are static");
        if indices.len() as u64 == dims[axis] && indices.iter().copied().eq(0..indices.len()) {
            return source.clone();
        }
        if let Some(Pointwise::Broadcast(mut base, mut map)) =
            self.out.pointwise.get(&source.ssa).map(|p| p.op.clone())
        {
            while let Some(Pointwise::Broadcast(next, dims)) =
                self.out.pointwise.get(&base.ssa).map(|p| p.op.clone())
            {
                map = dims.iter().map(|&d| map[d as usize]).collect();
                base = next;
            }
            let selected = match map.iter().position(|&d| d == axis as u64) {
                Some(axis) if shape(&base.ty)[axis] != Some(1) => self.select(&base, axis, indices),
                _ => base,
            };
            dims[axis] = indices.len() as u64;
            return self.out.broadcast_in_dim(
                &selected,
                &map,
                MlirTy::Ranked(dims.into_iter().map(Some).collect()),
            );
        }
        let repeats = indices.iter().take_while(|&&i| i == indices[0]).count();
        if repeats > 1
            && indices.len().is_multiple_of(repeats)
            && indices
                .chunks_exact(repeats)
                .all(|part| part.iter().all(|&i| i == part[0]))
        {
            let unique = indices.iter().step_by(repeats).copied().collect::<Vec<_>>();
            let selected = self.select(source, axis, &unique);
            let mut repeated = dimensions(&selected.ty).unwrap();
            repeated.insert(axis + 1, repeats as u64);
            let map = (0..dims.len())
                .map(|d| (d + usize::from(d > axis)) as u64)
                .collect::<Vec<_>>();
            let value = self.out.broadcast_in_dim(
                &selected,
                &map,
                MlirTy::Ranked(repeated.into_iter().map(Some).collect()),
            );
            dims[axis] = indices.len() as u64;
            return self
                .out
                .reshape(&value, MlirTy::Ranked(dims.into_iter().map(Some).collect()));
        }
        let stride = indices
            .get(1)
            .and_then(|&n| n.checked_sub(indices[0]))
            .unwrap_or(1);
        if stride > 0
            && indices
                .windows(2)
                .all(|w| w[1].checked_sub(w[0]) == Some(stride))
        {
            let mut starts = vec![0; dims.len()];
            let mut strides = vec![1; dims.len()];
            starts[axis] = indices[0] as u64;
            dims[axis] = indices[indices.len() - 1] as u64 + 1;
            strides[axis] = stride as u64;
            return self.out.slice(source, &starts, &dims, &strides);
        }
        // A few contiguous runs need no runtime index lookup. Bound the view
        // count so irregular permutations remain one tensor gather.
        let breaks = indices
            .windows(2)
            .enumerate()
            .filter_map(|(i, pair)| (pair[1].checked_sub(pair[0]) != Some(1)).then_some(i + 1))
            .take(4)
            .collect::<Vec<_>>();
        if breaks.len() < 4 {
            let mut start = 0;
            let mut parts = Vec::new();
            for end in breaks.into_iter().chain(std::iter::once(indices.len())) {
                parts.push(self.select(source, axis, &indices[start..end]));
                start = end;
            }
            return self.concatenate(&parts, axis);
        }
        if indices.iter().any(|&i| {
            i64::try_from(i).is_err()
                || (matches!(self.out.dtype, Dtype::F32) && i32::try_from(i).is_err())
        }) {
            // Static slice bounds need not fit the runtime index element type.
            let parts = indices
                .iter()
                .map(|&i| self.select(source, axis, &[i]))
                .collect::<Vec<_>>();
            return self.concatenate(&parts, axis);
        }
        let index_ty = MlirTy::Ranked(vec![Some(indices.len() as u64), Some(1)]);
        let index = self
            .out
            .folded_constant(
                indices.iter().map(|&i| Scalar::Int(i as i64)).collect(),
                index_ty,
                Axes::default(),
            )
            .expect("indices fit the target integer type");
        let mut result_dims = dims.clone();
        result_dims[axis] = indices.len() as u64;
        let ty = MlirTy::Ranked(result_dims.into_iter().map(Some).collect());
        dims[axis] = 1;
        let offsets = (0..dims.len())
            .filter(|&d| d != axis)
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let sizes = dims
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let from = source.ty.render(self.out.dtype, source.elem);
        let index_ty = index.ty.render(self.out.dtype, ElemKind::Int);
        let to = ty.render(self.out.dtype, source.elem);
        let ssa = self.out.pure(format!(
            "\"stablehlo.gather\"({}, {}) <{{dimension_numbers = #stablehlo.gather<offset_dims = [{offsets}], collapsed_slice_dims = [{axis}], start_index_map = [{axis}], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: {sizes}>}}> : ({from}, {index_ty}) -> {to}",
            source.ssa, index.ssa,
        ));
        Value {
            ssa,
            ty,
            elem: source.elem,
        }
    }

    fn concatenate(&mut self, values: &[Value], axis: usize) -> Value {
        if values.len() == 1 {
            return values[0].clone();
        }
        let elem = values[0].elem;
        let mut dims = shape(&values[0].ty).to_vec();
        dims[axis] = Some(values.iter().map(|v| shape(&v.ty)[axis].unwrap()).sum());
        let ty = MlirTy::Ranked(dims);
        if let Some((values, map)) = self.hoist_inputs(values, Some(axis)) {
            let selected_axis = map.binary_search(&(axis as u64)).unwrap();
            let small = self.concatenate(&values, selected_axis);
            return self.out.broadcast_in_dim(&small, &map, ty);
        }
        let operands = values
            .iter()
            .map(|v| v.ssa.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let types = values
            .iter()
            .map(|v| v.ty.render(self.out.dtype, elem))
            .collect::<Vec<_>>()
            .join(", ");
        let result = ty.render(self.out.dtype, elem);
        let ssa = self.out.pure(format!(
            "stablehlo.concatenate {operands}, dim = {axis} : ({types}) -> {result}"
        ));
        Value { ssa, ty, elem }
    }
}
