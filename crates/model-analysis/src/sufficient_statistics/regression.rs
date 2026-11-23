//! Fixed-design Gaussian residuals, expressed with portable array operations.

use super::multivariate::{
    cells, dot, entry, finite_data, get, held_cells, real_shape, residual_factor, vector,
};
use super::{
    Expr, Reducer, data_only, local_ref, map, materialize, num, op, operation,
    per_cell_with_invariants, resolve,
};
use flatppl_core::{Binding, Dim, Module, Node, NodeId, Scalar, ScalarType, Type, ValueSet};

fn data_shape(ty: &Type) -> Option<Vec<usize>> {
    let Type::Array { shape, elem } = ty else {
        return None;
    };
    if !matches!(
        elem.as_ref(),
        Type::Scalar(ScalarType::Real | ScalarType::Integer)
    ) {
        return None;
    }
    shape
        .iter()
        .map(|dim| match dim {
            Dim::Static(n) if *n > 0 => Some(*n as usize),
            _ => None,
        })
        .collect()
}

struct Term {
    column: NodeId,
    coefficient: Expr,
}

#[derive(Default)]
struct Design {
    terms: Vec<Term>,
    offsets: Vec<(NodeId, f64)>,
}

fn scalar(m: &Module, id: NodeId) -> bool {
    matches!(
        m.type_of(id),
        Some(Type::Scalar(ScalarType::Real | ScalarType::Integer))
    )
}

/// Generated design-column projections need not have set annotations yet.
pub(super) fn bounds(m: &Module, id: NodeId, reducer: &Reducer) -> Option<(f64, f64)> {
    let id = resolve(m, id, &reducer.boundary);
    match m.node(id) {
        Node::Lit(Scalar::Real(x)) if x.is_finite() => return Some((*x, *x)),
        Node::Lit(Scalar::Int(x)) => return Some((*x as f64, *x as f64)),
        _ => {}
    }
    let mut set = m.valueset_of(id);
    while let Some(ValueSet::CartPow(cell, _)) = set {
        set = Some(cell);
    }
    match set {
        Some(ValueSet::Interval(lo, hi)) if lo.is_finite() && hi.is_finite() => {
            return Some((*lo, *hi));
        }
        Some(ValueSet::Integers | ValueSet::NonNegIntegers | ValueSet::PosIntegers) => {
            return Some((i64::MIN as f64, i64::MAX as f64));
        }
        _ => {}
    }
    let (name, args) = operation(m, id)?;
    match (name, args) {
        ("get" | "get0" | "fill", [array, ..]) => bounds(m, *array, reducer),
        ("neg", [value]) => bounds(m, *value, reducer).map(|(lo, hi)| (-hi, -lo)),
        ("vector" | "rowstack" | "colstack" | "cat", values) => values
            .iter()
            .map(|&value| bounds(m, value, reducer))
            .try_fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), bound| {
                let (a, b) = bound?;
                Some((lo.min(a), hi.max(b)))
            }),
        _ => None,
    }
}

fn magnitude(m: &Module, id: NodeId, reducer: &Reducer) -> Option<f64> {
    bounds(m, id, reducer).map(|(lo, hi)| lo.abs().max(hi.abs()))
}

#[derive(Clone, Copy)]
struct RowScales {
    values: NodeId,
    minimum_relative: f64,
    minimum_sigma: f64,
}

/// Separate known row scales from a shared scalar only when the original
/// products cannot underflow or overflow in either supported precision.
fn row_scales(
    reducer: &Reducer,
    m: &mut Module,
    sigma: NodeId,
    count: usize,
) -> Option<(NodeId, Option<RowScales>)> {
    if let Some(sigma) = reducer.scalar(m, sigma) {
        return Some((sigma, None));
    }
    let id = resolve(m, sigma, &reducer.boundary);
    let (shared, scales) = if reducer.data_only(m, id) {
        (None, id)
    } else {
        let ("mul", [a, b]) = operation(m, id)? else {
            return None;
        };
        if let Some(shared) = reducer.scalar(m, *a) {
            (Some(shared), *b)
        } else {
            (Some(reducer.scalar(m, *b)?), *a)
        }
    };
    if !reducer.data_only(m, scales)
        || m.type_of(scales).and_then(real_shape).as_deref() != Some(&[count])
    {
        return None;
    }
    let (lo, hi) = bounds(m, scales, reducer)?;
    let (a, b) = shared.map_or(Some((1.0, 1.0)), |s| bounds(m, s, reducer))?;
    // A finite primitive quadratic bounds its standardized residual by
    // sqrt(MAX). Bound both possible expansion factors before cancellation.
    if (lo as f32) < f32::MIN_POSITIVE
        || (a as f32) < f32::MIN_POSITIVE
        || hi as f32 > f32::MAX.sqrt()
        || b as f32 > f32::MAX.sqrt()
    {
        return None;
    }
    let minimum = (lo as f32) * (a as f32);
    let maximum = (hi as f32) * (b as f32);
    let relative_minimum = (lo as f32) / (hi as f32);
    if minimum < f32::MIN_POSITIVE
        || !minimum.is_finite()
        || !maximum.is_finite()
        || relative_minimum < f32::MIN_POSITIVE
    {
        return None;
    }
    let shared = shared.unwrap_or_else(|| materialize(m, num(1.0)));
    Some((
        shared,
        Some(RowScales {
            values: scales,
            minimum_relative: lo / hi,
            minimum_sigma: f64::from(minimum),
        }),
    ))
}

/// Collect fixed columns and arbitrary scalar coefficient expressions. A live
/// design matrix or a nonlinear operation on a whole mean vector stays live.
fn collect(
    reducer: &Reducer,
    m: &mut Module,
    id: NodeId,
    count: usize,
    sign: f64,
    design: &mut Design,
) -> Option<()> {
    let id = resolve(m, id, &reducer.boundary);
    if data_only(m, id, &reducer.allowed, &reducer.boundary) {
        if !finite_data(m, id, reducer) {
            return None;
        }
        let value = if scalar(m, id) {
            materialize(m, op("fill", [Expr::Node(id), Expr::Integer(count as i64)]))
        } else {
            if data_shape(m.type_of(id)?)? != [count] {
                return None;
            }
            id
        };
        design.offsets.push((value, sign));
        return Some(());
    }
    if scalar(m, id) {
        let ones = materialize(m, op("fill", [num(1.0), Expr::Integer(count as i64)]));
        design.terms.push(Term {
            column: ones,
            coefficient: op("mul", [num(sign), Expr::Node(id)]),
        });
        return Some(());
    }
    // Integer array arithmetic can wrap before later real promotion.
    if m.type_of(id).and_then(real_shape).as_deref() != Some(&[count]) {
        return None;
    }
    let (name, args) = operation(m, id)?;
    let name = name.to_owned();
    let args = args.to_vec();
    match (name.as_str(), args.as_slice()) {
        ("add" | "sub", [a, b]) => {
            collect(reducer, m, *a, count, sign, design)?;
            collect(
                reducer,
                m,
                *b,
                count,
                if name == "sub" { -sign } else { sign },
                design,
            )
        }
        ("neg", [x]) => collect(reducer, m, *x, count, -sign, design),
        ("mul", [a, b]) => {
            let a = resolve(m, *a, &reducer.boundary);
            let b = resolve(m, *b, &reducer.boundary);
            if let Some(shape) = m.type_of(a).and_then(data_shape)
                && let [rows, columns] = shape.as_slice()
                && *rows == count
                && m.type_of(b).and_then(real_shape).as_deref() == Some(&[*columns])
                && data_only(m, a, &reducer.allowed, &reducer.boundary)
                && finite_data(m, a, reducer)
            {
                for j in 0..*columns {
                    let column = materialize(
                        m,
                        op(
                            "get",
                            [
                                Expr::Node(a),
                                Expr::Constant("all"),
                                Expr::Integer(j as i64 + 1),
                            ],
                        ),
                    );
                    design.terms.push(Term {
                        column,
                        coefficient: op("mul", [num(sign), get(Expr::Node(b), j)]),
                    });
                }
                return Some(());
            }
            let (column, coefficient) = if scalar(m, a) {
                (b, a)
            } else if scalar(m, b) {
                (a, b)
            } else {
                return None;
            };
            if m.type_of(column).and_then(data_shape).as_deref() != Some(&[count])
                || !data_only(m, column, &reducer.allowed, &reducer.boundary)
                || !finite_data(m, column, reducer)
            {
                return None;
            }
            design.terms.push(Term {
                column,
                coefficient: op("mul", [num(sign), Expr::Node(coefficient)]),
            });
            Some(())
        }
        _ => None,
    }
}

/// Name preparation intermediates without making them extra scoring inputs.
pub(super) fn setup(m: &mut Module, expression: Expr) -> Expr {
    let rhs = materialize(m, expression);
    let mut serial = 0;
    let name = loop {
        let name = m.intern(&format!("__likelihood_setup_{serial}"));
        if m.binding_by_name(name).is_none() {
            break name;
        }
        serial += 1;
    };
    m.add_binding(Binding {
        name,
        rhs,
        doc: None,
        public: false,
        synthetic: true,
    });
    Expr::Node(local_ref(m, name))
}

fn column_scales(columns: &[Expr]) -> Expr {
    vector(
        columns
            .iter()
            .map(|column| op("maximum", [map("abs", [column.clone()])])),
    )
}

fn safe_scales(scales: Expr) -> Expr {
    map(
        "ifelse",
        [map("iszero", [scales.clone()]), num(1.0), scales],
    )
}

/// Orthogonal transformations preserve the norm of the scaled residual column.
/// Its norm sets the scale derivative near a fitted reference, where two large
/// terms cancel. Restore it from the centered rows before discarding them.
fn match_residual_norm(
    m: &mut Module,
    factor: Expr,
    residual: Expr,
    scales: Expr,
    count: usize,
    p: usize,
) -> Expr {
    let factor = setup(m, factor);
    let last = op(
        "get",
        [
            factor.clone(),
            Expr::Constant("all"),
            Expr::Integer(p as i64 + 1),
        ],
    );
    let squared = setup(m, dot(last.clone(), last));
    let denominator = op(
        "ifelse",
        [op("iszero", [squared.clone()]), num(1.0), squared],
    );
    let normalized = setup(m, map("divide", [residual, get(scales, p)]));
    let target = op(
        "divide",
        [dot(normalized.clone(), normalized), num(count as f64)],
    );
    let gain = op("sqrt", [op("divide", [target, denominator])]);
    let weights = vector((0..p).map(|_| num(1.0)).chain([gain]));
    map("mul", [factor, op("rowstack", [vector([weights])])])
}

fn factor(columns: &[Expr], scales: Expr, observed: NodeId, count: usize, preparing: bool) -> Expr {
    let dimension = columns.len();
    let normalized = op(
        "colstack",
        [vector(columns.iter().enumerate().map(|(j, column)| {
            map("divide", [column.clone(), get(scales.clone(), j)])
        }))],
    );
    let row = vector((0..dimension).map(|j| {
        op(
            "get",
            [Expr::Held(0), Expr::Local, Expr::Integer(j as i64 + 1)],
        )
    }));
    let rows = held_cells(
        map("divide", [row, op("sqrt", [num(count as f64)])]),
        op("indicesof", [Expr::Node(observed)]),
        [normalized],
    );
    if count <= dimension {
        return op("rowstack", [rows]);
    }
    if preparing
        && let Some(blocks) = count.div_ceil(dimension).checked_next_power_of_two()
        && let Some(padded) = blocks.checked_mul(dimension)
        && let Ok(padding) = i64::try_from(padded - count)
    {
        // Merge pairs of blocks to limit Householder rounding depth to log2(blocks).
        // A linear scan of fixed rows can lose precision in the residual norm.
        let rows = if padding == 0 {
            rows
        } else {
            let zero = op("fill", [num(0.0), Expr::Integer(dimension as i64)]);
            op("cat", [rows, op("fill", [zero, Expr::Integer(padding)])])
        };
        let shape = vector([
            Expr::Integer(2 * dimension as i64),
            Expr::Integer(dimension as i64),
        ]);
        let mut matrix = op("rowstack", [rows]);
        for _ in 1..blocks.ilog2() {
            let pairs = op("splitblocks", [matrix, shape.clone()]);
            let factors = cells(op("get", [op("qr", [Expr::Local]), Expr::Text("R")]), pairs);
            matrix = op("joinblocks", [factors]);
        }
        return op("get", [op("qr", [matrix]), Expr::Text("R")]);
    }
    residual_factor(rows, count, dimension)
}

pub(super) fn reduce(reducer: &mut Reducer, m: &mut Module, id: NodeId) -> Option<Expr> {
    let f = per_cell_with_invariants(m, id, &reducer.boundary)?;
    if m.resolve(f.constructor) != "Normal"
        || !data_only(m, f.observed, &reducer.allowed, &reducer.boundary)
        || !finite_data(m, f.observed, reducer)
    {
        return None;
    }
    let shape = data_shape(m.type_of(f.observed)?)?;
    let [count] = shape.as_slice() else {
        return None;
    };
    let count = *count;
    if count < 2 {
        return None;
    }
    let mu = f
        .parameters
        .iter()
        .find(|(name, _)| m.resolve(*name) == "mu")?
        .1;
    let sigma = f
        .parameters
        .iter()
        .find(|(name, _)| m.resolve(*name) == "sigma")?
        .1;
    let (sigma, row_scales) = row_scales(reducer, m, sigma, count)?;
    if m.type_of(mu).and_then(real_shape).as_deref() != Some(&[count]) {
        return None;
    }
    let mut design = Design::default();
    collect(reducer, m, mu, count, 1.0, &mut design)?;
    if design.terms.is_empty() {
        return None;
    }
    let p = design.terms.len();
    let mut key = vec![f.observed];
    key.extend(design.terms.iter().map(|t| t.column));
    key.extend(design.offsets.iter().map(|(id, _)| *id));
    if let Some(scales) = row_scales {
        let response = design
            .offsets
            .iter()
            .try_fold(magnitude(m, f.observed, reducer)?, |bound, (offset, _)| {
                Some(bound + magnitude(m, *offset, reducer)?)
            })?;
        let largest = design.terms.iter().try_fold(response, |bound, term| {
            Some(bound.max(magnitude(m, term.column, reducer)?))
        })?;
        if !(2.0 * (largest as f32) / scales.minimum_relative as f32).is_finite() {
            return None;
        }
        // A rounded reference prediction must stay small relative to the
        // narrowest noise scale. Otherwise F32 centering can change the score.
        if largest * (design.offsets.len() + 1) as f64 * f64::from(f32::EPSILON)
            > scales.minimum_sigma * f64::from(f32::EPSILON.sqrt())
        {
            return None;
        }
        key.push(scales.values);
    }
    // Offset signs are part of the data graph, hence part of the cache key.
    let mut response = map("real", [Expr::Node(f.observed)]);
    for (offset, sign) in design.offsets {
        let signed = materialize(m, map("mul", [num(sign), Expr::Node(offset)]));
        key.push(signed);
        response = map("sub", [response, Expr::Node(signed)]);
    }
    // QR setup costs more than one raw matrix-vector score. Runtime designs
    // need the explicit preparation boundary to amortize that cost.
    if !reducer.preparing
        && key
            .iter()
            .any(|&id| !data_only(m, id, &Default::default(), &reducer.boundary))
    {
        return None;
    }
    let (inverse_center, relative_scales) = if let Some(scales) = row_scales {
        let center = setup(m, op("maximum", [Expr::Node(scales.values)]));
        let inverse = op("divide", [num(1.0), center.clone()]);
        let relative = setup(
            m,
            map("divide", [Expr::Node(scales.values), center.clone()]),
        );
        (Some(inverse), Some(relative))
    } else {
        (None, None)
    };
    let response = setup(m, response);
    let columns = design
        .terms
        .iter()
        .map(|t| map("real", [Expr::Node(t.column)]))
        .collect::<Vec<_>>();
    let matrix = setup(m, op("colstack", [vector(columns.clone())]));
    let reference = if count <= p + 1 {
        op("fill", [num(0.0), Expr::Integer(p as i64)])
    } else {
        let mut initial_columns = columns.clone();
        initial_columns.push(response.clone());
        let initial_scales = setup(m, safe_scales(column_scales(&initial_columns)));
        let initial = setup(
            m,
            factor(
                &initial_columns,
                initial_scales.clone(),
                f.observed,
                count,
                reducer.preparing,
            ),
        );
        // Small pivots affect only the reference choice, never the model rank.
        // Use a conservative float32 rounding bound in both precisions.
        let threshold = num(8.0 * (p + 1) as f64 * f64::from(f32::EPSILON));
        let active = (0..p)
            .map(|j| {
                op(
                    "gt",
                    [op("abs", [entry(initial.clone(), j, j)]), threshold.clone()],
                )
            })
            .collect::<Vec<_>>();
        let triangular = op(
            "rowstack",
            [vector((0..p).map(|j| {
                vector((0..p).map(|k| {
                    op(
                        "ifelse",
                        [
                            active[j].clone(),
                            entry(initial.clone(), j, k),
                            num(if j == k { 1.0 } else { 0.0 }),
                        ],
                    )
                }))
            }))],
        );
        let rhs = vector((0..p).map(|j| {
            op(
                "ifelse",
                [active[j].clone(), entry(initial.clone(), j, p), num(0.0)],
            )
        }));
        let solved = setup(m, op("linsolve", [triangular, rhs]));
        vector((0..p).map(|j| {
            op(
                "divide",
                [
                    op(
                        "mul",
                        [get(solved.clone(), j), get(initial_scales.clone(), p)],
                    ),
                    get(initial_scales.clone(), j),
                ],
            )
        }))
    };
    let proposed = setup(m, reference);
    let finite = op("lall", [map("isfinite", [proposed.clone()])]);
    let zeros = op("fill", [num(0.0), Expr::Integer(p as i64)]);
    let safe = setup(m, op("ifelse", [finite.clone(), proposed, zeros.clone()]));
    let mut trial_residual = op(
        "sub",
        [response.clone(), op("mul", [matrix.clone(), safe.clone()])],
    );
    if let Some(scales) = &relative_scales {
        trial_residual = map("divide", [trial_residual, scales.clone()]);
    }
    let mut valid = op(
        "land",
        [finite, op("lall", [map("isfinite", [trial_residual])])],
    );
    if let Some(scales) = row_scales {
        // Large cancelling coefficients can make a finite fitted reference
        // inaccurate even for small data. This changes only the reference.
        let prediction_magnitude = op(
            "maximum",
            [op(
                "mul",
                [map("abs", [matrix.clone()]), map("abs", [safe.clone()])],
            )],
        );
        let resolution = op(
            "mul",
            [
                num((2 * p + 1) as f64 * f64::from(f32::EPSILON)),
                prediction_magnitude,
            ],
        );
        let accurate = op(
            "le",
            [
                resolution,
                num(scales.minimum_sigma * f64::from(f32::EPSILON.sqrt())),
            ],
        );
        valid = op("land", [valid, accurate]);
    }
    // A finite coefficient vector can still overflow its prediction. Reject
    // the whole reference before constructing the compressed residual.
    let reference = op("ifelse", [valid, safe, zeros]);
    let kind = if row_scales.is_some() {
        "weighted_regression_reference"
    } else {
        "regression_reference"
    };
    let reference = reducer.summary_for(m, &key, kind, reference);
    let residual = setup(
        m,
        op("sub", [response, op("mul", [matrix, reference.clone()])]),
    );
    let mut final_columns = columns;
    final_columns.push(residual);
    if let Some(scales) = &relative_scales {
        final_columns = final_columns
            .into_iter()
            .map(|column| map("divide", [column, scales.clone()]))
            .collect();
    }
    let mut scale_fields = safe_scales(column_scales(&final_columns));
    if let Some(inverse) = inverse_center {
        let correction = op(
            "sum",
            [map("log", [Expr::Node(row_scales.unwrap().values)])],
        );
        scale_fields = op("cat", [scale_fields, vector([inverse, correction])]);
    }
    let kind = if row_scales.is_some() {
        "weighted_regression_scales"
    } else {
        "regression_scales"
    };
    let buffer = reducer.summary_for(m, &key, kind, scale_fields);
    let (scales, inverse_center, correction) = if row_scales.is_some() {
        (
            vector((0..=p).map(|j| get(buffer.clone(), j))),
            Some(get(buffer.clone(), p + 1)),
            Some(get(buffer, p + 2)),
        )
    } else {
        (buffer, None, None)
    };
    let kind = if row_scales.is_some() {
        "weighted_regression_factor"
    } else {
        "regression_factor"
    };
    let r = factor(
        &final_columns,
        scales.clone(),
        f.observed,
        count,
        reducer.preparing,
    );
    let r = if reducer.preparing && count > p + 1 {
        match_residual_norm(m, r, final_columns[p].clone(), scales.clone(), count, p)
    } else {
        r
    };
    let r = reducer.summary_for(m, &key, kind, r);
    let coefficients = vector(
        design
            .terms
            .into_iter()
            .enumerate()
            .map(|(j, term)| op("sub", [term.coefficient, get(reference.clone(), j)]))
            .chain([num(-1.0)]),
    );
    let residual = op("mul", [r, map("mul", [scales, coefficients])]);
    let standardized = if let Some(inverse) = inverse_center {
        // A prepared reciprocal avoids reverse-mode division by a tiny data
        // scale. Apply an expanding live division before a shrinking data
        // scale so a small residual cannot disappear before standardization.
        let small = op("lt", [Expr::Node(sigma), num(1.0)]);
        let first = op("ifelse", [small.clone(), Expr::Node(sigma), num(1.0)]);
        let last = op("ifelse", [small, num(1.0), Expr::Node(sigma)]);
        let residual = map("divide", [residual, first]);
        map("divide", [map("mul", [residual, inverse]), last])
    } else {
        map("divide", [residual, Expr::Node(sigma)])
    };
    let quadratic = op("sum", [map("mul", [standardized.clone(), standardized])]);
    let zero = materialize(m, num(0.0));
    let parameters = f
        .parameters
        .iter()
        .map(|&(name, _)| (name, if m.resolve(name) == "mu" { zero } else { sigma }))
        .collect::<Vec<_>>();
    let normalizer = reducer.anchor(m, &f, &parameters, num(0.0));
    let score = op(
        "mul",
        [
            num(count as f64),
            op("sub", [normalizer, op("mul", [num(0.5), quadratic])]),
        ],
    );
    Some(if let Some(correction) = correction {
        op("sub", [score, correction])
    } else {
        score
    })
}
