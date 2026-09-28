//! Share irregular lane selections that feed singleton-only broadcast views.
//! Gather requests retain their typed source, indices and exact definition span.
//! Finalization changes only those definitions, after packing has finished.

use std::ops::Range;

use super::*;

struct Request {
    source: Value,
    axis: usize,
    indices: Vec<usize>,
    value: Value,
    definition: Range<usize>,
}

#[derive(Default)]
pub(super) struct Requests(Vec<Request>);

impl Requests {
    pub(super) fn emit(
        &mut self,
        out: &mut Emitter<'_>,
        source: &Value,
        axis: usize,
        indices: &[usize],
    ) -> Value {
        let (value, definition) = gather(out, source, axis, indices);
        self.remember(source, axis, indices, &value, definition);
        value
    }

    pub(super) fn remember(
        &mut self,
        source: &Value,
        axis: usize,
        indices: &[usize],
        value: &Value,
        definition: Range<usize>,
    ) {
        // CSE may reuse a prior definition. Record each definition only once.
        if !definition.is_empty() {
            self.0.push(Request {
                source: source.clone(),
                axis,
                indices: indices.to_vec(),
                value: value.clone(),
                definition,
            });
        }
    }

    /// Terminal body rewrite: no further operations may use the emitter caches.
    pub(super) fn finish(self, out: &mut Emitter<'_>, rets: &[&Value]) {
        if self.0.is_empty() {
            return;
        }
        let mut users: HashMap<&str, Option<&str>> =
            rets.iter().map(|v| (v.ssa.as_str(), None)).collect();
        for line in out.live_lines(&out.body, rets) {
            let Some((ssa, rhs)) = line.split_once(" = ") else {
                continue;
            };
            for (_, input) in packing::ssa_uses(rhs) {
                users
                    .entry(input)
                    .and_modify(|user| *user = None)
                    .or_insert(Some(ssa));
            }
        }
        let mut groups: Vec<Vec<&Request>> = Vec::new();
        let mut by_source = HashMap::new();
        for request in &self.0 {
            let Some(Some(user)) = users.get(request.value.ssa.as_str()) else {
                continue;
            };
            let Some(producer) = out.pointwise.get(*user) else {
                continue;
            };
            let Pointwise::Broadcast(input, dims) = &producer.op else {
                continue;
            };
            let target = shape(&producer.value.ty);
            if target.len() <= shape(&input.ty).len()
                || !dims
                    .iter()
                    .zip(shape(&input.ty))
                    .all(|(&d, n)| target.get(d as usize) == Some(n))
                || !target
                    .iter()
                    .enumerate()
                    .all(|(d, &n)| dims.contains(&(d as u64)) || n == Some(1))
            {
                continue;
            }
            let key = (request.source.ssa.as_str(), request.axis);
            let group = *by_source.entry(key).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[group].push(request);
        }

        let mut edits = Vec::new();
        for requests in groups.into_iter().filter(|requests| requests.len() > 1) {
            let first = requests[0];
            let indices = requests
                .iter()
                .flat_map(|r| r.indices.iter().copied())
                .collect::<Vec<_>>();
            // A fresh cache prevents reuse of an index constant defined after
            // the first request, where the shared gather must be inserted.
            let mut scratch = Emitter::new(out.m, out.dtype);
            scratch.next = out.next;
            let (shared, _) = gather(&mut scratch, &first.source, first.axis, &indices);
            out.next = scratch.next;
            let mut offset = 0;
            for (number, request) in requests.into_iter().enumerate() {
                let end = offset + request.indices.len();
                let bounds = shape(&shared.ty)
                    .iter()
                    .enumerate()
                    .map(|(axis, dim)| {
                        if axis == first.axis {
                            format!("{offset}:{end}")
                        } else {
                            format!("0:{}", dim.unwrap())
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let mut replacement = if number == 0 {
                    scratch.body.clone()
                } else {
                    String::new()
                };
                replacement.push_str(&format!(
                    "{} = stablehlo.slice {} [{bounds}] : ({}) -> {}\n",
                    request.value.ssa,
                    shared.ssa,
                    shared.ty.render(out.dtype, shared.elem),
                    request.value.ty.render(out.dtype, request.value.elem),
                ));
                edits.push((request.definition.clone(), replacement));
                offset = end;
            }
        }
        if edits.is_empty() {
            return;
        }
        edits.sort_by_key(|(span, _)| span.start);
        let mut body = String::with_capacity(out.body.len());
        let mut start = 0;
        for (span, replacement) in edits {
            body.push_str(&out.body[start..span.start]);
            body.push_str(&replacement);
            start = span.end;
        }
        body.push_str(&out.body[start..]);
        out.body = body;
    }
}

fn gather(
    out: &mut Emitter<'_>,
    source: &Value,
    axis: usize,
    indices: &[usize],
) -> (Value, Range<usize>) {
    let index_ty = MlirTy::Ranked(vec![Some(indices.len() as u64), Some(1)]);
    let index = out
        .folded_constant(
            indices.iter().map(|&i| Scalar::Int(i as i64)).collect(),
            index_ty,
            Axes::default(),
        )
        .expect("indices fit the target integer type");
    let mut dims = shape(&source.ty).to_vec();
    dims[axis] = Some(indices.len() as u64);
    let ty = MlirTy::Ranked(dims.clone());
    dims[axis] = Some(1);
    let offsets = (0..dims.len())
        .filter(|&d| d != axis)
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let sizes = dims
        .iter()
        .map(|d| d.unwrap().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let from = source.ty.render(out.dtype, source.elem);
    let index_ty = index.ty.render(out.dtype, ElemKind::Int);
    let to = ty.render(out.dtype, source.elem);
    let start = out.body.len();
    let ssa = out.pure(format!(
        "\"stablehlo.gather\"({}, {}) <{{dimension_numbers = #stablehlo.gather<offset_dims = [{offsets}], collapsed_slice_dims = [{axis}], start_index_map = [{axis}], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: {sizes}>}}> : ({from}, {index_ty}) -> {to}",
        source.ssa, index.ssa,
    ));
    (
        Value {
            ssa,
            ty,
            elem: source.elem,
        },
        start..out.body.len(),
    )
}
