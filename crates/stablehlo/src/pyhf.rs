//! StableHLO fast paths for retained `pyhf_helpers` standard-module calls.
//! Other targets expand the portable reference during determinization.

use super::*;

impl Emitter<'_> {
    pub(super) fn lower_pyhf_call(
        &mut self,
        id: NodeId,
        callee: NodeId,
        args: &[NodeId],
        broadcast: bool,
    ) -> Option<Result<Value, EmitError>> {
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        if module != "pyhf_helpers" {
            return None;
        }
        if matches!(self.m.node(id), Node::Call(call) if !call.named.is_empty()) {
            return Some(Err(EmitError::at(
                id,
                "pyhf helper arguments must be bound by determinize",
            )));
        }
        Some(self.emit_pyhf_call(id, &member, args, broadcast))
    }

    fn emit_pyhf_call(
        &mut self,
        id: NodeId,
        member: &str,
        args: &[NodeId],
        broadcast: bool,
    ) -> Result<Value, EmitError> {
        if !broadcast && let Some(YieldRows { blocks, rows }) = self.yield_rows(member, args) {
            return self.lower_yield_rows(member, args, &blocks, &rows);
        }
        let mut values = args
            .iter()
            .map(|&arg| {
                let value = self.lower_node(arg)?;
                Ok(self.typed_axes(arg, value))
            })
            .collect::<Result<Vec<_>, EmitError>>()?;
        let parent = if broadcast {
            Some(self.enter_broadcast(id, &mut values)?)
        } else {
            None
        };
        let result = match (member, values.as_slice()) {
            // Converted normsys lanes carry one alpha per lane, so no product
            // broadcasts an operand and native products add no contraction.
            ("normsys_factor", [lo, hi, alpha]) => {
                Ok(self.native_products(|e| e.normsys_factor(lo, hi, alpha)))
            }
            ("histosys_shift", [lo, nominal, hi, alpha]) => {
                Ok(self.histosys_shift(lo, nominal, hi, alpha))
            }
            ("sample_yields", [nominal, shifts, factors]) => {
                let sums = self.pyhf_reduce(shifts, 1, false);
                let products = self.pyhf_reduce(factors, 1, true);
                let shifted = self.add(nominal, &sums);
                Ok(self.mul(&shifted, &products))
            }
            ("expected_counts", [samples]) => Ok(self.pyhf_reduce(samples, 0, false)),
            _ => Err(EmitError::at(
                id,
                format!("unsupported pyhf helper call `{member}`"),
            )),
        };
        match parent {
            Some(parent) => {
                let frame = self.broadcast_frame.clone();
                let result =
                    result.map(|value| self.finish_broadcast(&value, &frame, parent.len()));
                self.broadcast_frame = parent;
                result
            }
            None => result,
        }
    }

    /// Read converter-form `sample_yields` arguments: nominal rows, one shift
    /// per sample, and each sample's factors in multiplication order, where
    /// `fill(x, bins)` carries a scalar and a unit fill pads. `None` for any
    /// other spelling, which takes the dense lowering.
    fn yield_block(&self, nominal: NodeId, shifts: NodeId, factors: NodeId) -> Option<YieldBlock> {
        let nominals = self.call_args(nominal, "rowstack")?;
        let [rows] = nominals else { return None };
        let nominals = self.call_args(*rows, "vector")?.to_vec();
        let samples = nominals.len();
        let shifts = match self.uniform_fill(shifts) {
            Some(0.0) => vec![Vec::new(); samples],
            Some(_) => return None,
            None => {
                let (cells, width) = self.rank3_cells(shifts, samples)?;
                cells
                    .chunks(width)
                    .map(|row| {
                        row.iter()
                            .copied()
                            .filter(|&cell| self.scalar_fill(cell) != Some(Some(0.0)))
                            .collect()
                    })
                    .collect()
            }
        };
        let factors = match self.uniform_fill(factors) {
            Some(1.0) => vec![Vec::new(); samples],
            Some(_) => return None,
            None => {
                let (cells, width) = self.rank3_cells(factors, samples)?;
                cells
                    .chunks(width)
                    .map(|row| {
                        row.iter()
                            .filter_map(|&cell| match self.scalar_fill(cell) {
                                Some(Some(1.0)) => None,
                                Some(_) => Some(YieldFactor::Scalar(
                                    self.call_args(cell, "fill").unwrap()[0],
                                )),
                                None => Some(YieldFactor::PerBin(cell)),
                            })
                            .collect()
                    })
                    .collect()
            }
        };
        let [_, bins] = self.shape_of(nominal)?[..] else {
            return None;
        };
        let scalars = factors
            .iter()
            .map(|row| {
                row.iter()
                    .map(|factor| match *factor {
                        YieldFactor::Scalar(node) => self.factor_leaves(node).len(),
                        YieldFactor::PerBin(_) => 0,
                    })
                    .sum()
            })
            .collect();
        Some(YieldBlock {
            nominals,
            shifts,
            factors,
            bins,
            scalars,
        })
    }

    /// The converter-form blocks behind `sample_yields(…)`,
    /// `expected_counts(sample_yields(…))` or `expected_counts` of a stack of
    /// such blocks' rows, with the (block, row) of each summed sample in order.
    fn yield_rows(&self, member: &str, args: &[NodeId]) -> Option<YieldRows> {
        let whole = |block: YieldBlock| {
            let rows = (0..block.nominals.len()).map(|row| (0, row)).collect();
            YieldRows {
                blocks: vec![block],
                rows,
            }
        };
        match (member, args) {
            ("sample_yields", &[nominal, shifts, factors]) => {
                self.yield_block(nominal, shifts, factors).map(whole)
            }
            ("expected_counts", &[samples]) => {
                if let Some(block) = self.yields_call(samples) {
                    return Some(whole(block));
                }
                let [rows] = self.call_args(samples, "rowstack")? else {
                    return None;
                };
                let mut sources: Vec<NodeId> = Vec::new();
                let mut blocks = Vec::new();
                let mut cells = Vec::new();
                for &cell in self.call_args(*rows, "vector")? {
                    let [yields, row, all] = self.call_args(cell, "get")? else {
                        return None;
                    };
                    if !matches!(self.node(*all), Node::Const(sym) if self.resolve(*sym) == "all") {
                        return None;
                    }
                    let yields = self.resolve_ref_one(*yields);
                    let block = match sources.iter().position(|&source| source == yields) {
                        Some(block) => block,
                        None => {
                            blocks.push(self.yields_call(yields)?);
                            sources.push(yields);
                            blocks.len() - 1
                        }
                    };
                    let row = self.literal(*row)? as usize;
                    if row == 0 || row > blocks[block].nominals.len() {
                        return None;
                    }
                    cells.push((block, row - 1));
                }
                Some(YieldRows {
                    blocks,
                    rows: cells,
                })
            }
            _ => None,
        }
    }

    /// The converter-form block of a `pyhf_helpers.sample_yields(…)` call.
    fn yields_call(&self, id: NodeId) -> Option<YieldBlock> {
        let Node::Call(call) = self.node(self.resolve_ref_one(id)) else {
            return None;
        };
        let CallHead::User(callee) = call.head else {
            return None;
        };
        let (module, member) = flatppl_determinizer::standard_function(self.m, callee)?;
        match (module.as_str(), member.as_str(), &call.args[..]) {
            ("pyhf_helpers", "sample_yields", &[nominal, shifts, factors])
                if call.named.is_empty() =>
            {
                self.yield_block(nominal, shifts, factors)
            }
            _ => None,
        }
    }

    /// Lower converter-form yields. The CPU target multiplies each row as the
    /// per-sample chain in the order the rows are summed. The GPU target takes
    /// the summed rows as one block, whatever class they came from, and scales
    /// the stacked templates by its scalar column where that pays.
    fn lower_yield_rows(
        &mut self,
        member: &str,
        args: &[NodeId],
        blocks: &[YieldBlock],
        rows: &[(usize, usize)],
    ) -> Result<Value, EmitError> {
        let gpu = self.target == crate::Target::Gpu;
        let merged = gpu.then(|| YieldBlock::merge(blocks, rows));
        let samples = match merged {
            Some(block) if block.takes_column() => self.scaled_stack(&block)?,
            _ => {
                let mut values = Vec::with_capacity(rows.len());
                for &(block, row) in rows {
                    values.push(self.chained_row(&blocks[block], row)?);
                }
                // A one-sample sum is that sample's row, as in the chain.
                if member == "expected_counts" && values.len() == 1 {
                    return Ok(values.remove(0));
                }
                self.stack_rows(&values)
            }
        };
        Ok(if member == "expected_counts" {
            let samples = self.typed_axes(args[0], samples);
            self.pyhf_reduce(&samples, 0, false)
        } else {
            samples
        })
    }

    /// A sample's nominal plus its shift rows.
    fn shifted_template(&mut self, block: &YieldBlock, sample: usize) -> Result<Value, EmitError> {
        let mut template = self.lower_node(block.nominals[sample])?;
        for &row in &block.shifts[sample] {
            let row = self.lower_node(row)?;
            template = self.add(&template, &row);
        }
        Ok(template)
    }

    /// Multiply a sample's template by its factors. The CPU target keeps the
    /// chain's order, factor by factor. The GPU target applies the per-bin
    /// factors, then the product of all scalar factors once, so the reverse
    /// pass contracts the bins once per sample.
    fn chained_row(&mut self, block: &YieldBlock, sample: usize) -> Result<Value, EmitError> {
        let mut row = self.shifted_template(block, sample)?;
        let fold = self.target == crate::Target::Gpu;
        let mut scale: Option<Value> = None;
        for factor in &block.factors[sample] {
            match *factor {
                YieldFactor::PerBin(node) => {
                    let factor = self.lower_node(node)?;
                    row = self.mul(&row, &factor);
                }
                YieldFactor::Scalar(node) if fold => {
                    let factor = self.lower_node(node)?;
                    scale = Some(match scale {
                        Some(scale) => self.mul(&scale, &factor),
                        None => factor,
                    });
                }
                YieldFactor::Scalar(node) => {
                    for node in self.factor_leaves(node) {
                        let factor = self.lower_node(node)?;
                        row = self.mul(&row, &factor);
                    }
                }
            }
        }
        if let Some(scale) = scale {
            row = self.mul(&row, &scale);
        }
        Ok(self.convert(&row, ElemKind::Real))
    }

    /// Apply each sample's shift and per-bin factors, then multiply the stacked
    /// templates once by the column of per-sample scalar products. Products
    /// commute, so this equals the chain. The reverse pass then contracts the
    /// bins once per channel instead of once per scalar factor.
    fn scaled_stack(&mut self, block: &YieldBlock) -> Result<Value, EmitError> {
        let mut templates = Vec::with_capacity(block.nominals.len());
        let mut scales = Vec::with_capacity(block.nominals.len());
        for (sample, factors) in block.factors.iter().enumerate() {
            let mut template = self.shifted_template(block, sample)?;
            let mut scale: Option<Value> = None;
            for factor in factors {
                match *factor {
                    YieldFactor::PerBin(node) => {
                        let factor = self.lower_node(node)?;
                        template = self.mul(&template, &factor);
                    }
                    YieldFactor::Scalar(node) => {
                        let factor = self.lower_node(node)?;
                        scale = Some(match scale {
                            Some(scale) => self.mul(&scale, &factor),
                            None => factor,
                        });
                    }
                }
            }
            let scale = scale.unwrap_or_else(|| self.scalar(1.0));
            let scale = self.convert(&scale, ElemKind::Real);
            scales.push(self.vector(&[scale]));
            templates.push(self.convert(&template, ElemKind::Real));
        }
        let templates = self.stack_rows(&templates);
        let column = self.vector(&scales);
        Ok(self.mul(&templates, &column))
    }

    fn stack_rows(&mut self, rows: &[Value]) -> Value {
        match self.contiguous_rows(rows) {
            Some(value) => value,
            None => self.vector(rows),
        }
    }

    /// The factors of a left-folded scalar `mul` tree, in order.
    fn factor_leaves(&self, id: NodeId) -> Vec<NodeId> {
        match self.call_args(id, "mul") {
            Some(&[left, right]) => {
                let mut leaves = self.factor_leaves(left);
                leaves.push(right);
                leaves
            }
            _ => vec![id],
        }
    }

    /// The arguments of `id`, followed through bindings, when it calls `head`.
    fn call_args(&self, id: NodeId, head: &str) -> Option<&[NodeId]> {
        match self.node(self.resolve_ref_one(id)) {
            Node::Call(call) if matches!(call.head, CallHead::Builtin(sym) if self.resolve(sym) == head) => {
                Some(&call.args)
            }
            _ => None,
        }
    }

    fn literal(&self, id: NodeId) -> Option<f64> {
        match self.node(self.resolve_ref_one(id)) {
            Node::Lit(Scalar::Real(value)) => Some(*value),
            Node::Lit(Scalar::Int(value)) => Some(*value as f64),
            _ => None,
        }
    }

    fn literals(&self, id: NodeId) -> Option<Vec<f64>> {
        self.call_args(id, "vector")?
            .iter()
            .map(|&arg| self.literal(arg))
            .collect()
    }

    /// The value of `fill(value, size)` with a literal value and a size list.
    fn uniform_fill(&self, id: NodeId) -> Option<f64> {
        let [value, size] = self.call_args(id, "fill")? else {
            return None;
        };
        self.literals(*size)?;
        self.literal(*value)
    }

    /// `Some(literal)` for `fill(x, bins)`, with `literal` set when `x` is one.
    fn scalar_fill(&self, id: NodeId) -> Option<Option<f64>> {
        let [value, _] = self.call_args(id, "fill")? else {
            return None;
        };
        Some(self.literal(*value))
    }

    /// The cells of `array(cat(cells…), [samples, width, bins], [1, 2, 3])`,
    /// each a bin vector or a scalar filling the bins.
    fn rank3_cells(&self, id: NodeId, samples: usize) -> Option<(Vec<NodeId>, usize)> {
        let [data, size, order] = self.call_args(id, "array")? else {
            return None;
        };
        let [s, width, bins] = self.literals(*size)?[..] else {
            return None;
        };
        if self.literals(*order)? != [1.0, 2.0, 3.0] || s as usize != samples || width < 1.0 {
            return None;
        }
        let cells = self.call_args(*data, "cat")?.to_vec();
        let bins = bins as u64;
        let fits = |cell: NodeId| match self.call_args(cell, "fill") {
            Some(&[value, size]) => {
                self.literal(size) == Some(bins as f64) && self.shape_of(value) == Some(vec![])
            }
            _ => self.shape_of(cell) == Some(vec![bins]),
        };
        (cells.len() == samples * width as usize && cells.iter().all(|&cell| fits(cell)))
            .then_some((cells, width as usize))
    }

    /// The static tensor shape of a node's inferred type.
    fn shape_of(&self, id: NodeId) -> Option<Vec<u64>> {
        let ty = self.type_of(id)?;
        match crate::types::mlir_type_of_ty(id, ty, self.dtype).ok()?.0 {
            MlirTy::Scalar => Some(vec![]),
            MlirTy::Ranked(dims) => dims.into_iter().collect(),
            _ => None,
        }
    }

    /// Reduce a model axis without transposing it past bin or batch axes.
    fn pyhf_reduce(&mut self, value: &Value, axis: usize, product: bool) -> Value {
        let value = self.convert(value, ElemKind::Real);
        let identity = if product {
            "1.000000e+00"
        } else {
            "0.000000e+00"
        };
        let op = if product {
            "stablehlo.multiply"
        } else {
            "stablehlo.add"
        };
        self.reduce_axis_lit(op, identity, &value, self.batch_rank(&value) + axis)
    }

    /// HistFactory code4p returns the additive shift, not the shifted template.
    /// Keep modifier/bin axes intact and avoid cancellation against the nominal.
    fn histosys_shift(&mut self, lo: &Value, nominal: &Value, hi: &Value, alpha: &Value) -> Value {
        let up = self.sub(hi, nominal);
        let up = self.convert(&up, ElemKind::Real);
        let left = self.sub(lo, nominal);
        let zero = self.scalar(0.0);
        let down = self.sub(&zero, &left);
        // §09 requires finite alpha. Fixed positive-zero shifts reduce to
        // alpha * 0 in both regions, including the sign of a zero result.
        // Runtime anchors must retain their derivatives even when equal.
        if [&up, &down].iter().all(|value| {
            self.constants.get(&value.ssa).is_some_and(|data| {
                data.iter()
                    .all(|x| matches!(x, Scalar::Real(v) if v.to_bits() == 0))
            })
        }) {
            let (shape, _) = self.broadcast_pair(&up, &down);
            let (shape, _) = self.broadcast_pair(&shape, alpha);
            let zero = self.constant_like(0.0, &shape);
            return self.mul(alpha, &zero);
        }
        let sum = self.add(&up, &down);
        let half = self.scalar(0.5);
        let symmetric = self.mul(&sum, &half);
        let difference = self.sub(&up, &down);
        let sixteenth = self.scalar(0.0625);
        let asymmetric = self.mul(&difference, &sixteenth);
        let alpha = self.convert(alpha, ElemKind::Real);
        let one = self.scalar(1.0);
        let minus_one = self.scalar(-1.0);
        let x = self.max(&alpha, &minus_one);
        let x = self.min(&x, &one);
        // Share the alpha-only scale across bins. Keep the outer x so tiny
        // shifts survive large anchors and negative zero keeps its sign.
        // Native products here never broadcast over bins. The bin products
        // below keep the dot form, whose alpha adjoint is one contraction per
        // channel rather than one per selected sample run.
        let scale = self.native_products(|e| {
            let x2 = e.mul(&x, &x);
            let three = e.scalar(3.0);
            let minus_ten = e.scalar(-10.0);
            let fifteen = e.scalar(15.0);
            let term = e.mul(&x2, &three);
            let term = e.add(&minus_ten, &term);
            let term = e.mul(&x2, &term);
            let term = e.add(&fifteen, &term);
            e.mul(&x, &term)
        });
        let term = self.mul(&asymmetric, &scale);
        let term = self.add(&symmetric, &term);
        let polynomial = self.mul(&x, &term);
        let positive = self.compare("GT", &alpha, &zero);
        let slope = self.select(&positive, &up, &down);
        let tail = self.mul(&alpha, &slope);
        let abs_alpha = self.abs(&alpha);
        let interior = self.compare("LT", &abs_alpha, &one);
        self.select(&interior, &polynomial, &tail)
    }

    /// Unit-centered §09 interpolation. Coefficient operations fold at target
    /// precision when anchors are fixed, without expanding a callable body.
    fn normsys_factor(&mut self, lo: &Value, hi: &Value, alpha: &Value) -> Value {
        // The reference divides both anchors by its Real unit center first.
        let lo = &self.convert(lo, ElemKind::Real);
        let hi = &self.convert(hi, ElemKind::Real);
        let alpha = &self.convert(alpha, ElemKind::Real);
        let log_hi = self.log(hi);
        let log_lo = self.log(lo);
        let up1 = self.mul(hi, &log_hi);
        let down1 = self.mul(lo, &log_lo);
        let up2 = self.mul(&up1, &log_hi);
        let down2 = self.mul(&down1, &log_lo);
        let half = self.scalar(0.5);
        let s0 = self.add(hi, lo);
        let s0 = self.mul(&s0, &half);
        let a0 = self.sub(hi, lo);
        let a0 = self.mul(&a0, &half);
        let s1 = self.sub(&up1, &down1);
        let s1 = self.mul(&s1, &half);
        let a1 = self.add(&up1, &down1);
        let a1 = self.mul(&a1, &half);
        let s2 = self.add(&up2, &down2);
        let s2 = self.mul(&s2, &half);
        let a2 = self.sub(&up2, &down2);
        let a2 = self.mul(&a2, &half);

        let coefficients = [
            self.interpolation_coefficient([&a0, &s1, &a2], [15.0, -7.0, 1.0], 0.0, 0.125),
            self.interpolation_coefficient([&s0, &a1, &s2], [24.0, -9.0, 1.0], -24.0, 0.125),
            self.interpolation_coefficient([&a0, &s1, &a2], [-5.0, 5.0, -1.0], 0.0, 0.25),
            self.interpolation_coefficient([&s0, &a1, &s2], [-12.0, 7.0, -1.0], 12.0, 0.25),
            self.interpolation_coefficient([&a0, &s1, &a2], [3.0, -3.0, 1.0], 0.0, 0.125),
            self.interpolation_coefficient([&s0, &a1, &s2], [8.0, -5.0, 1.0], -8.0, 0.125),
        ];
        let zero = self.scalar(0.0);
        let one = self.scalar(1.0);
        let abs_alpha = self.abs(alpha);
        let interior = self.compare("LT", &abs_alpha, &one);
        // Bound inactive powers without making x depend on the output predicate.
        // The tails own the knots, so clamp's boundary derivative is never used.
        let minus_one = self.scalar(-1.0);
        let x = self.max(alpha, &minus_one);
        let x = self.min(&x, &one);
        let pairs = [0, 2, 4].map(|i| {
            let term = self.mul(&x, &coefficients[i + 1]);
            self.add(&coefficients[i], &term)
        });
        let x2 = self.mul(&x, &x);
        let term = self.mul(&x2, &pairs[2]);
        let term = self.add(&pairs[1], &term);
        let term = self.mul(&x2, &term);
        let term = self.add(&pairs[0], &term);
        let term = self.mul(&x, &term);
        let polynomial = self.add(&one, &term);

        // Choose by sign even inside the polynomial region. Then the inactive
        // exponential stays between one and its finite positive anchor.
        let positive = self.compare("GT", alpha, &zero);
        let minus_log_lo = self.neg(&log_lo);
        let slope = self.select(&positive, &log_hi, &minus_log_lo);
        let exponent = self.mul(alpha, &slope);
        let tail = self.exp(&exponent);
        self.select(&interior, &polynomial, &tail)
    }

    fn interpolation_coefficient(
        &mut self,
        terms: [&Value; 3],
        weights: [f64; 3],
        offset: f64,
        scale: f64,
    ) -> Value {
        let mut result = self.scalar(offset);
        for (term, weight) in terms.into_iter().zip(weights) {
            let weight = self.scalar(weight);
            let term = self.mul(term, &weight);
            result = self.add(&result, &term);
        }
        let scale = self.scalar(scale);
        self.mul(&result, &scale)
    }
}

/// Converter-form blocks and the (block, row) of each summed sample in order.
struct YieldRows {
    blocks: Vec<YieldBlock>,
    rows: Vec<(usize, usize)>,
}

/// One channel's `sample_yields` operands in multiplication order.
struct YieldBlock {
    nominals: Vec<NodeId>,
    /// Each sample's non-zero shift rows.
    shifts: Vec<Vec<NodeId>>,
    factors: Vec<Vec<YieldFactor>>,
    bins: u64,
    /// Each sample's count of scalar factors.
    scalars: Vec<usize>,
}

impl YieldBlock {
    /// The given rows of several blocks as one block, in that order.
    fn merge(blocks: &[YieldBlock], rows: &[(usize, usize)]) -> YieldBlock {
        let pick = |&(block, row): &(usize, usize)| (&blocks[block], row);
        YieldBlock {
            nominals: rows.iter().map(pick).map(|(b, r)| b.nominals[r]).collect(),
            shifts: rows
                .iter()
                .map(pick)
                .map(|(b, r)| b.shifts[r].clone())
                .collect(),
            factors: rows
                .iter()
                .map(pick)
                .map(|(b, r)| b.factors[r].clone())
                .collect(),
            bins: blocks[0].bins,
            scalars: rows.iter().map(pick).map(|(b, r)| b.scalars[r]).collect(),
        }
    }

    /// A scalar column pays on GPU once samples average two scalar factors.
    /// A one-bin stack times a column never finishes Enzyme's pass pipeline.
    fn takes_column(&self) -> bool {
        let samples = self.nominals.len();
        samples > 1 && self.bins > 1 && self.scalars.iter().sum::<usize>() >= 2 * samples
    }
}

#[derive(Clone, Copy)]
enum YieldFactor {
    Scalar(NodeId),
    PerBin(NodeId),
}
