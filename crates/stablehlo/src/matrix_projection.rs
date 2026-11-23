//! Demand just the paired cells of a matrix selected by two fixed one-hot tables.

use flatppl_core::Idx;
use flatppl_optimise::{MatrixProjectionFact, plan_matrix_projection};

use super::*;

// The fixed-data evaluator reads through views without materializing them.
const MAX_SELECTOR_CELLS: usize = 65_536;

impl Emitter<'_> {
    /// Find the product and Hadamard nodes before emitting the dense contractions.
    /// Prove the selectors on the lowered, typed recipes, where ABI arguments
    /// and actual target precision already determine which cells are fixed.
    pub(crate) fn lower_matrix_projection(
        &mut self,
        output: NodeId,
        masked: NodeId,
        weights: NodeId,
    ) -> Option<Value> {
        use MatrixProjectionFact::{Multiply, OneHotRows, Ones, Product, RealMatrix, SumTree};

        if self.target != crate::Target::Cpu
            || self.region_depth != 0
            || !self.broadcast_frame.is_empty()
            || self.host_batch
        {
            return None;
        }
        let (masked, [op, left, right]) = self.projection_call(masked, "broadcast")?;
        if !matches!(self.node(op), Node::Const(op) if self.resolve(*op) == "mul") {
            return None;
        }
        let (selected, [row, matrix], col, selected_first) =
            if let Some((selected, operands)) = self.projection_call(left, "mul") {
                (selected, operands, right, true)
            } else {
                let (selected, operands) = self.projection_call(right, "mul")?;
                (selected, operands, left, false)
            };

        if !matches!(self.type_of(row), Some(Type::Array { .. }))
            || !matches!(self.type_of(matrix), Some(Type::Array { .. }))
        {
            return None;
        }
        let mut trial = self.clone();
        let early_col = if selected_first {
            None
        } else {
            Some(trial.lower_node(col).ok()?)
        };
        let row_mask = trial.lower_node(row).ok()?;
        let (observations, rows, row_indices) = trial.one_hot_projection_rows(&row_mask)?;
        let source = trial.lower_node(matrix).ok()?;
        let [Some(height), Some(columns)] = shape(&source.ty) else {
            return None;
        };
        if source.elem != ElemKind::Real || trial.batch_rank(&source) != 0 || *height != rows {
            return None;
        }
        let col_mask = match early_col {
            Some(value) => value,
            None => trial.lower_node(col).ok()?,
        };
        let (col_observations, col_width, col_indices) =
            trial.one_hot_projection_rows(&col_mask)?;
        let ones = trial.lower_node(weights).ok()?;
        let [Some(weight_count)] = shape(&ones.ty) else {
            return None;
        };
        if trial.batch_rank(&ones) != 0
            || *weight_count != *columns
            || *weight_count > constants::MAX_FOLDED_ELEMENTS as u64
            || !(0..*weight_count)
                .all(|i| trial.fixed_projection_bit(&ones, i as usize) == Some(true))
        {
            return None;
        }

        let cells = height.checked_mul(*columns)?;
        let padded = cells.checked_next_power_of_two()?;
        if padded > MAX_SELECTOR_CELLS as u64 {
            return None;
        }
        let levels = padded.trailing_zeros();
        let facts = [
            Product(output, masked, weights),
            Multiply(masked, left, right),
            Product(selected, row, matrix),
            OneHotRows(row, observations as i64, rows as i64),
            OneHotRows(col, col_observations as i64, col_width as i64),
            RealMatrix(matrix, *height as i64, *columns as i64),
            SumTree(matrix, padded as i64, i64::from(levels)),
            Ones(weights, *weight_count as i64),
        ];
        let plan = plan_matrix_projection(output, &facts)?;
        if plan.matrix != matrix
            || plan.row_selector != row
            || plan.column_selector != col
            || self.projection_has_indexed_consumer(output)
        {
            return None;
        }
        let indices = row_indices
            .into_iter()
            .zip(col_indices)
            .map(|(r, c)| r * columns + c)
            .collect::<Vec<_>>();
        let result = trial.emit_matrix_projection(
            output,
            &source,
            [&row_mask, &col_mask, &ones],
            &indices,
            padded,
        )?;
        *self = trial;
        Some(result)
    }

    fn projection_call<const N: usize>(
        &self,
        mut id: NodeId,
        builtin: &str,
    ) -> Option<(NodeId, [NodeId; N])> {
        for _ in 0..16 {
            // A promoted input (even if its authored RHS is a product) keeps
            // its bound SSA value. Preserve any shared value already emitted.
            if self.memo.contains_key(&id) {
                return None;
            }
            let rhs = self.resolve_ref_one(id);
            if rhs == id {
                let Node::Call(call) = self.node(id) else {
                    return None;
                };
                if !matches!(call.head, CallHead::Builtin(head) if self.resolve(head) == builtin)
                    || !call.named.is_empty()
                    || call.inputs.is_some()
                {
                    return None;
                }
                return Some((id, call.args.as_ref().try_into().ok()?));
            }
            id = rhs;
        }
        None
    }

    fn projection_has_indexed_consumer(&self, output: NodeId) -> bool {
        // Specialized scalar indexing can fold away all but one cotangent
        // in a dense dot transpose, retaining the sign of the remaining
        // zero products. A scatter initialized at +0 would erase that sign.
        if self.m.node_count() > MAX_SELECTOR_CELLS {
            return true;
        }
        let mut pending = Vec::new();
        for index in 0..self.m.node_count() {
            let Node::Call(call) = self.node(NodeId::from_usize(index)) else {
                continue;
            };
            let CallHead::Builtin(op) = call.head else {
                continue;
            };
            let container = match (self.resolve(op), call.args.as_ref()) {
                ("get" | "get0", [container, ..]) => container,
                ("broadcast", [head, container, ..]) if matches!(self.node(*head), Node::Const(op) if matches!(self.resolve(*op), "get" | "get0")) => {
                    container
                }
                _ => continue,
            };
            pending.push(*container);
        }
        let mut seen = HashSet::new();
        while let Some(node) = pending.pop() {
            if !seen.insert(node) {
                continue;
            }
            if node == output {
                return true;
            }
            let rhs = self.resolve_ref_one(node);
            if rhs != node {
                pending.push(rhs);
            }
            self.m.for_each_child(node, |child| pending.push(child));
        }
        false
    }

    fn fixed_projection_bit(&self, value: &Value, index: usize) -> Option<bool> {
        match self.fixed_scalar_at(value, index)? {
            Scalar::Int(0) | Scalar::Bool(false) => Some(false),
            Scalar::Int(1) | Scalar::Bool(true) => Some(true),
            Scalar::Real(x) if x.to_bits() == 0.0f64.to_bits() => Some(false),
            Scalar::Real(1.0) => Some(true),
            _ => None,
        }
    }

    fn one_hot_projection_rows(&self, value: &Value) -> Option<(u64, u64, Vec<u64>)> {
        let [Some(rows), Some(columns)] = shape(&value.ty) else {
            return None;
        };
        if self.batch_rank(value) != 0
            || *rows > constants::MAX_FOLDED_ELEMENTS as u64
            || *columns == 0
            || rows.checked_mul(*columns)? > MAX_SELECTOR_CELLS as u64
        {
            return None;
        }
        let mut indices = Vec::with_capacity(*rows as usize);
        for i in 0..*rows {
            let mut index = None;
            for j in 0..*columns {
                if self.fixed_projection_bit(value, (i * columns + j) as usize)?
                    && index.replace(j).is_some()
                {
                    return None;
                }
            }
            indices.push(index?);
        }
        Some((*rows, *columns, indices))
    }

    fn emit_matrix_projection(
        &mut self,
        output: NodeId,
        source: &Value,
        [rows, columns, weights]: [&Value; 3],
        indices: &[u64],
        padded: u64,
    ) -> Option<Value> {
        let result_ty = MlirTy::Ranked(vec![Some(indices.len() as u64)]);
        let max = match self.dtype {
            Dtype::F32 => f64::from(f32::MAX),
            Dtype::F64 => f64::MAX,
        };
        // Every tree sum has at most padded leaves. Use the original dots
        // when any source entry could make a tree sum overflow, including
        // non-finite cells that would participate through a zero weight.
        let bound = self.scalar(max * 0.5 / padded as f64);
        let magnitude = self.abs(source);
        let safe = self.compare("LE", &magnitude, &bound);
        let safe = self.reduce_all(&safe);

        let outer_body = std::mem::take(&mut self.body);
        let outer_pure = std::mem::take(&mut self.pure_ops);
        let outer_expanded = std::mem::take(&mut self.expanded);
        let selected = self.gather_with_zero_adjoint_edges(output, source, indices, padded)?;
        let fast_body = std::mem::take(&mut self.body);
        self.pure_ops.clear();
        self.expanded.clear();
        let dense = self.matmat(rows, source);
        let masked = self.mul(&dense, columns);
        let dense = self.matvec(&masked, weights);
        let dense_body = std::mem::replace(&mut self.body, outer_body);
        self.pure_ops = outer_pure;
        self.expanded = outer_expanded;

        let ssa = self.fresh();
        let ty = result_ty.render(self.dtype, ElemKind::Real);
        self.push(&format!(
            "{ssa} = \"stablehlo.if\"({}) ({{\n{fast_body}stablehlo.return {} : {ty}\n}}, {{\n{dense_body}stablehlo.return {} : {ty}\n}}) : (tensor<i1>) -> {ty}",
            safe.ssa, selected.ssa, dense.ssa
        ));
        self.remember_axes(&ssa, self.axes_of(&selected));
        Some(Value {
            ssa,
            ty: result_ty,
            elem: ElemKind::Real,
        })
    }

    fn gather_with_zero_adjoint_edges(
        &mut self,
        output: NodeId,
        source: &Value,
        indices: &[u64],
        padded: u64,
    ) -> Option<Value> {
        let vector = |n| MlirTy::Ranked(vec![Some(n)]);
        let index = |e: &mut Self, cells: Vec<Scalar>| {
            let ty = vector(cells.len() as u64);
            e.folded_constant(cells, ty, Axes::default())
        };
        let count = constants::element_count(&source.ty)? as u64;
        let flat = self.reshape(source, vector(count));
        let selected = index(
            self,
            indices.iter().map(|&i| Scalar::Int(i as i64)).collect(),
        )?;
        let selected = self.gather_axis(&flat, &selected, 0, 0);

        // The siblings on a leaf's path partition every other source cell.
        // Multiplying their sum by zero keeps the original zero-weight edges
        // in reverse mode. Reverse applies the zero to the cotangent before
        // accumulation, so an infinite or NaN cotangent still reaches the
        // unselected cells as NaN while the selected cell gets the cotangent.
        let mut level = self.pad_zeros(&flat, &[0], &[padded - count]);
        let depth = padded.trailing_zeros();
        let mut levels = Vec::with_capacity(depth as usize);
        let mut offsets = Vec::with_capacity(depth as usize);
        let mut width = padded;
        let mut offset = 0;
        for d in 0..depth {
            offsets.push(offset);
            offset += width;
            levels.push(level.clone());
            if d + 1 < depth {
                width /= 2;
                let pairs = self.reshape(&level, MlirTy::Ranked(vec![Some(width), Some(2)]));
                level = self.reduce_sum_last_axis(&pairs);
            }
        }
        let tree = self.concatenate(output, &levels).ok()?;
        let paths = indices
            .iter()
            .flat_map(|&leaf| {
                offsets
                    .iter()
                    .enumerate()
                    .map(move |(bit, offset)| Scalar::Int((offset + ((leaf >> bit) ^ 1)) as i64))
            })
            .collect();
        let siblings = index(self, paths)?;
        let siblings = self.gather_axis(&tree, &siblings, 0, 0);
        let sibling_ty = MlirTy::Ranked(vec![Some(indices.len() as u64), Some(u64::from(depth))]);
        let siblings = self.reshape(&siblings, sibling_ty);
        let complement = self.reduce_sum_last_axis(&siblings);
        let complement = self.abs(&complement);
        let zero = self.scalar(0.0);
        let complement = self.mul(&complement, &zero);
        Some(self.add(&selected, &complement))
    }
}
