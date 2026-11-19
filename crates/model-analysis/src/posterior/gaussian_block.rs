//! Exact fixed-design Gaussian regression in standardized prior coordinates.

use super::{likelihood_factors, map, named_fields, op, operands, positive, real};
use flatppl_core::calling::{project_field, reified_inputs, splat_fields};
use flatppl_core::graph::{
    expect_builtin_call, fold_add, is_input_ref, resolve_ref, resolve_ref_one,
};
use flatppl_core::{Module, Node, NodeId, RefNs, Scalar, ScalarType, Symbol, Type};
use flatppl_infer::constructors::{find_parameter, split_kernel_constructor};

use crate::{measure_stochastic_ancestors, references_latent};

const LOG_2PI: f64 = 1.8378770664093453;

// Portable QR can use a square workspace on GPU. Bound the first rule to
// explicitly authored small designs.
const MAX_QR_ROWS: usize = 64;

pub struct GaussianBlock {
    pub update: NodeId,
    pub fields: Vec<Symbol>,
    pub observations: usize,
    mean: NodeId,
    prior_scale: NodeId,
    factor: NodeId,
    projected: NodeId,
    residual: NodeId,
    log_noise_scale: NodeId,
    model_valid: NodeId,
    data_valid: NodeId,
    system_valid: NodeId,
}

struct Prior {
    field: Symbol,
    mean: NodeId,
    scale: NodeId,
}

struct Row {
    design: Vec<NodeId>,
    offset: NodeId,
    scale: NodeId,
    data: NodeId,
}

pub fn normalized(m: &mut Module, node: NodeId) -> Option<GaussianBlock> {
    let [posterior] = operands(m, node, "normalize")? else {
        return None;
    };
    update(m, *posterior)
}

pub fn update(m: &mut Module, posterior: NodeId) -> Option<GaussianBlock> {
    let update = resolve_ref(m, posterior).0;
    let [likelihood, prior] = *operands(m, update, "bayesupdate")? else {
        return None;
    };
    let fields = named_fields(m, prior, "joint")?.to_vec();
    if fields.len() < 2 || fields.len() >= MAX_QR_ROWS {
        return None;
    }
    let mut priors = Vec::with_capacity(fields.len());
    for field in fields {
        if !measure_stochastic_ancestors(m, field.value, &[]).is_empty() {
            return None;
        }
        let [mean, scale] = normal(m, field.value)?;
        if !scalar(m, mean) || !scalar(m, scale) {
            return None;
        }
        priors.push(Prior {
            field: field.name,
            mean,
            scale,
        });
    }
    let mut factors = Vec::new();
    likelihood_factors(m, likelihood, &mut Vec::new(), &mut factors)?;
    let mut rows = Vec::new();
    for (kernel, data) in factors {
        collect_rows(m, &priors, kernel, data, &mut rows)?;
    }
    if rows.is_empty() {
        return None;
    }
    Some(factor(m, update, priors, rows))
}

fn normal(m: &mut Module, measure: NodeId) -> Option<[NodeId; 2]> {
    let (name, args) = split_kernel_constructor(m, resolve_ref(m, measure).0)?;
    if m.resolve(name) != "Normal" {
        return None;
    }
    Some([
        find_parameter(m, &args, "mu")?,
        find_parameter(m, &args, "sigma")?,
    ])
}

fn scalar(m: &mut Module, node: NodeId) -> bool {
    let kind = m.type_of(node).cloned().or_else(|| {
        flatppl_infer::infer_expression(m, node, &[])
            .0
            .remove(&node)
    });
    matches!(
        kind,
        Some(Type::Scalar(ScalarType::Real | ScalarType::Integer))
    )
}

fn collect_rows(
    m: &mut Module,
    priors: &[Prior],
    kernel: NodeId,
    data: NodeId,
    rows: &mut Vec<Row>,
) -> Option<()> {
    let kernel = resolve_ref(m, kernel).0;
    let call = expect_builtin_call(m, kernel, "functionof")?;
    let [body] = *call.args else { return None };
    let mut targets = vec![None; priors.len()];
    for (label, target) in reified_inputs(m, kernel, call.inputs.as_ref()?)? {
        let index = priors.iter().position(|prior| prior.field == *label)?;
        if target.ns != RefNs::SelfMod || targets.contains(&Some(target.name)) {
            return None;
        }
        targets[index] = Some(target.name);
    }
    if let Some(fields) = named_fields(m, body, "joint") {
        let fields = fields.to_vec();
        let labels = splat_fields(m, data)?;
        if fields.len() != labels.len() || fields.iter().any(|field| !labels.contains(&field.name))
        {
            return None;
        }
        for field in fields {
            let observed = project_field(m, data, field.name);
            push_row(m, &targets, field.value, observed, rows)?;
        }
    } else {
        push_row(m, &targets, body, data, rows)?;
    }
    Some(())
}

fn push_row(
    m: &mut Module,
    targets: &[Option<Symbol>],
    measure: NodeId,
    data: NodeId,
    rows: &mut Vec<Row>,
) -> Option<()> {
    if targets.len() + rows.len() >= MAX_QR_ROWS {
        return None;
    }
    let [mean, scale] = normal(m, measure)?;
    if !scalar(m, data) || !scalar(m, scale) || !independent(m, scale, targets) {
        return None;
    }
    let mut design = affine(m, mean, targets, 0)?;
    let offset = design.pop()?;
    rows.push(Row {
        design,
        offset,
        scale,
        data,
    });
    Some(())
}

fn independent(m: &Module, value: NodeId, targets: &[Option<Symbol>]) -> bool {
    !targets
        .iter()
        .flatten()
        .any(|&name| references_latent(m, value, name, &[]))
        && measure_stochastic_ancestors(m, value, &[]).is_empty()
}

/// Coefficients followed by the offset. Stop at reified targets and promoted
/// inputs before following aliases. Refuse nonlinear uses of the coefficients.
fn affine(
    m: &mut Module,
    value: NodeId,
    targets: &[Option<Symbol>],
    depth: usize,
) -> Option<Vec<NodeId>> {
    if depth >= 32 {
        return None;
    }
    let zero = real(m, 0.0);
    if let Some(index) = targets
        .iter()
        .position(|name| name.is_some_and(|name| is_input_ref(m, value, name)))
    {
        let mut result = vec![zero; targets.len() + 1];
        result[index] = real(m, 1.0);
        return Some(result);
    }
    if independent(m, value, targets) && scalar(m, value) {
        let mut result = vec![zero; targets.len()];
        result.push(op(m, "real", &[value]));
        return Some(result);
    }
    if let Node::Ref(reference) = m.node(value) {
        if m.is_conditioned_input(reference.name) {
            return None;
        }
        let (next, binding) = resolve_ref_one(m, value);
        binding?;
        return affine(m, next, targets, depth + 1);
    }
    let node = value;
    if !scalar(m, node) {
        return None;
    }
    for head in ["add", "sub"] {
        if let Some([left, right]) = operands(m, node, head) {
            let (left, right) = (*left, *right);
            let left = affine(m, left, targets, depth + 1)?;
            let right = affine(m, right, targets, depth + 1)?;
            return Some(
                left.into_iter()
                    .zip(right)
                    .map(|(a, b)| op(m, head, &[a, b]))
                    .collect(),
            );
        }
    }
    if let Some([inner]) = operands(m, node, "neg") {
        return Some(
            affine(m, *inner, targets, depth + 1)?
                .into_iter()
                .map(|a| op(m, "neg", &[a]))
                .collect(),
        );
    }
    for head in ["mul", "divide"] {
        if let Some([left, right]) = operands(m, node, head) {
            let (left, right) = (*left, *right);
            let (inner, scale) = if independent(m, right, targets) && scalar(m, right) {
                (left, right)
            } else if head == "mul" && independent(m, left, targets) && scalar(m, left) {
                (right, left)
            } else {
                return None;
            };
            let scale = op(m, "real", &[scale]);
            return Some(
                affine(m, inner, targets, depth + 1)?
                    .into_iter()
                    .map(|a| op(m, head, &[a, scale]))
                    .collect(),
            );
        }
    }
    None
}

fn factor(m: &mut Module, update: NodeId, priors: Vec<Prior>, rows: Vec<Row>) -> GaussianBlock {
    let mut valid = Vec::new();
    for prior in &priors {
        valid.push(op(m, "isfinite", &[prior.mean]));
        valid.push(positive(m, prior.scale));
    }
    for row in &rows {
        valid.push(positive(m, row.scale));
        valid.push(op(m, "isfinite", &[row.offset]));
        valid.extend(row.design.iter().map(|&value| op(m, "isfinite", &[value])));
    }
    let model_valid = conjunction(m, &valid);
    let observed: Vec<_> = rows
        .iter()
        .map(|row| op(m, "isfinite", &[row.data]))
        .collect();
    let data_valid = conjunction(m, &observed);
    let count = priors.len();
    let n = m.alloc(Node::Lit(Scalar::Int(count as i64)));
    let zero = real(m, 0.0);
    let one = real(m, 1.0);
    let mut matrix = Vec::new();
    for j in 0..count {
        let mut row = vec![zero; count];
        row[j] = one;
        matrix.push(op(m, "vector", &row));
    }
    let mut response = vec![zero; count];
    let mut noise = Vec::new();
    let means: Vec<_> = priors
        .iter()
        .map(|p| safe_real(m, model_valid, p.mean, 0.0))
        .collect();
    let scales: Vec<_> = priors
        .iter()
        .map(|p| safe_real(m, model_valid, p.scale, 1.0))
        .collect();
    let mean = op(m, "vector", &means);
    let prior_scale = op(m, "vector", &scales);
    let mut system = Vec::new();
    for row in &rows {
        let scale = safe_real(m, model_valid, row.scale, 1.0);
        noise.push(op(m, "log", &[scale]));
        let offset = safe_real(m, model_valid, row.offset, 0.0);
        let data = safe_real(m, data_valid, row.data, 0.0);
        let mut prediction = vec![offset];
        let mut design = Vec::new();
        for ((&x, &mu), &s) in row.design.iter().zip(&means).zip(&scales) {
            let x = safe_real(m, model_valid, x, 0.0);
            prediction.push(op(m, "mul", &[x, mu]));
            let coefficient = standardized_coefficient(m, x, s, scale);
            design.push(finite_or_zero(m, coefficient, &mut system));
        }
        let prediction = fold_add(m, &prediction);
        let difference = op(m, "sub", &[data, prediction]);
        let weighted = op(m, "divide", &[difference, scale]);
        response.push(finite_or_zero(m, weighted, &mut system));
        matrix.push(op(m, "vector", &design));
    }
    let matrix = op(m, "vector", &matrix);
    let matrix = op(m, "rowstack", &[matrix]);
    let qr = op(m, "qr", &[matrix]);
    let q = qr_field(m, qr, "Q");
    let r = qr_field(m, qr, "R");
    let qt = op(m, "transpose", &[q]);
    let response = op(m, "vector", &response);
    let projected = op(m, "mul", &[qt, response]);
    let reconstructed = op(m, "mul", &[q, projected]);
    let residual = map(m, "sub", &[response, reconstructed]);
    let diag = op(m, "diag", &[r]);
    let finite = map(m, "isfinite", &[diag]);
    let magnitude = map(m, "abs", &[diag]);
    let nz = map(m, "gt", &[magnitude, zero]);
    let pivots = map(m, "land", &[finite, nz]);
    system.push(op(m, "lall", &[pivots]));
    let system_valid = conjunction(m, &system);
    let identity = op(m, "eye", &[n]);
    let factor = op(m, "ifelse", &[system_valid, r, identity]);
    GaussianBlock {
        update,
        fields: priors.into_iter().map(|prior| prior.field).collect(),
        observations: rows.len(),
        mean,
        prior_scale,
        factor,
        projected,
        residual,
        log_noise_scale: fold_add(m, &noise),
        model_valid,
        data_valid,
        system_valid,
    }
}

fn qr_field(m: &mut Module, qr: NodeId, name: &str) -> NodeId {
    let name = m.alloc(Node::Lit(Scalar::Str(name.into())));
    op(m, "get", &[qr, name])
}

fn conjunction(m: &mut Module, flags: &[NodeId]) -> NodeId {
    match flags {
        [] => m.alloc(Node::Lit(Scalar::Bool(true))),
        [only] => *only,
        _ => {
            let (left, right) = flags.split_at(flags.len() / 2);
            let left = conjunction(m, left);
            let right = conjunction(m, right);
            op(m, "land", &[left, right])
        }
    }
}

/// Whiten x * prior / noise without overflowing the unscaled product.
fn standardized_coefficient(m: &mut Module, x: NodeId, prior: NodeId, noise: NodeId) -> NodeId {
    let one = real(m, 1.0);
    let magnitude = op(m, "abs", &[x]);
    let larger_x = op(m, "ge", &[magnitude, prior]);
    let larger = op(m, "ifelse", &[larger_x, x, prior]);
    let smaller = op(m, "ifelse", &[larger_x, prior, x]);
    let shrink = op(m, "ge", &[noise, one]);
    let first = op(m, "ifelse", &[shrink, larger, smaller]);
    let second = op(m, "ifelse", &[shrink, smaller, larger]);
    let ratio = op(m, "divide", &[first, noise]);
    let early = op(m, "isfinite", &[ratio]);
    let first_divisor = op(m, "ifelse", &[early, noise, one]);
    let last_divisor = op(m, "ifelse", &[early, one, noise]);
    let first = op(m, "divide", &[first, first_divisor]);
    let product = op(m, "mul", &[first, second]);
    op(m, "divide", &[product, last_divisor])
}

fn safe_real(m: &mut Module, valid: NodeId, value: NodeId, fallback: f64) -> NodeId {
    let value = op(m, "real", &[value]);
    let fallback = real(m, fallback);
    op(m, "ifelse", &[valid, value, fallback])
}

fn finite_or_zero(m: &mut Module, value: NodeId, checks: &mut Vec<NodeId>) -> NodeId {
    let finite = op(m, "isfinite", &[value]);
    checks.push(finite);
    let zero = real(m, 0.0);
    op(m, "ifelse", &[finite, value, zero])
}

impl GaussianBlock {
    pub fn model_valid(&self) -> NodeId {
        self.model_valid
    }

    pub fn valid(&self, m: &mut Module) -> NodeId {
        conjunction(m, &[self.model_valid, self.data_valid, self.system_valid])
    }

    fn log_determinant(&self, m: &mut Module) -> NodeId {
        let diag = op(m, "diag", &[self.factor]);
        let abs = map(m, "abs", &[diag]);
        let log = map(m, "log", &[abs]);
        op(m, "sum", &[log])
    }

    pub fn log_density(&self, m: &mut Module, coordinates: NodeId) -> NodeId {
        let centered = map(m, "sub", &[coordinates, self.mean]);
        let z = map(m, "divide", &[centered, self.prior_scale]);
        let fitted = op(m, "mul", &[self.factor, z]);
        let residual = map(m, "sub", &[fitted, self.projected]);
        let square = map(m, "mul", &[residual, residual]);
        let norm = op(m, "sum", &[square]);
        let half = real(m, 0.5);
        let energy = op(m, "mul", &[half, norm]);
        let log_scale = map(m, "log", &[self.prior_scale]);
        let log_scale = op(m, "sum", &[log_scale]);
        let constant = real(m, 0.5 * self.fields.len() as f64 * LOG_2PI);
        let determinant = self.log_determinant(m);
        let cost = fold_add(m, &[constant, log_scale, energy]);
        let score = op(m, "sub", &[determinant, cost]);
        let valid = self.valid(m);
        let nan = real(m, f64::NAN);
        let score = op(m, "ifelse", &[valid, score, nan]);
        let anchor = super::evidence::parameter_anchor(m, self.model_valid);
        op(m, "add", &[score, anchor])
    }

    pub fn log_evidence(&self, m: &mut Module) -> NodeId {
        let square = map(m, "mul", &[self.residual, self.residual]);
        let norm = op(m, "sum", &[square]);
        let half = real(m, 0.5);
        let energy = op(m, "mul", &[half, norm]);
        let constant = real(m, 0.5 * self.observations as f64 * LOG_2PI);
        let determinant = self.log_determinant(m);
        let cost = fold_add(m, &[constant, self.log_noise_scale, determinant, energy]);
        let score = op(m, "neg", &[cost]);
        let nan = real(m, f64::NAN);
        let score = op(m, "ifelse", &[self.system_valid, score, nan]);
        let minus_inf = real(m, f64::NEG_INFINITY);
        let score = op(m, "ifelse", &[self.data_valid, score, minus_inf]);
        let anchor = super::evidence::parameter_anchor(m, self.model_valid);
        op(m, "add", &[score, anchor])
    }

    pub fn draw_from_standard(&self, m: &mut Module, standard: NodeId) -> NodeId {
        let response = map(m, "add", &[self.projected, standard]);
        let solved = op(m, "linsolve", &[self.factor, response]);
        let scaled = map(m, "mul", &[self.prior_scale, solved]);
        map(m, "add", &[self.mean, scaled])
    }
}
