//! Fixed-design Gaussian residuals, expressed with portable array operations.

use super::multivariate::{
    entry, finite_data, get, held_cells, real_shape, residual_factor, vector,
};
use super::{
    Expr, Reducer, data_only, local_ref, map, materialize, num, op, per_cell_with_invariants,
    resolve,
};
use flatppl_core::{Binding, CallHead, Dim, Module, Node, NodeId, ScalarType, Type};

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

fn arithmetic(m: &Module, id: NodeId) -> Option<(&str, &[NodeId])> {
    let Node::Call(c) = m.node(id) else {
        return None;
    };
    let CallHead::Builtin(head) = c.head else {
        return None;
    };
    let name = m.resolve(head);
    if name == "broadcast" {
        let (head, args) = c.args.split_first()?;
        let Node::Const(head) = m.node(*head) else {
            return None;
        };
        Some((m.resolve(*head), args))
    } else {
        Some((name, &c.args))
    }
}

fn scalar(m: &Module, id: NodeId) -> bool {
    matches!(
        m.type_of(id),
        Some(Type::Scalar(ScalarType::Real | ScalarType::Integer))
    )
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
    let (name, args) = arithmetic(m, id)?;
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
fn setup(m: &mut Module, expression: Expr) -> Expr {
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

fn factor(columns: &[Expr], scales: Expr, observed: NodeId, count: usize) -> Expr {
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
    let sigma = reducer.scalar(m, sigma)?;
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
            factor(&initial_columns, initial_scales.clone(), f.observed, count),
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
    let trial_residual = op(
        "sub",
        [response.clone(), op("mul", [matrix.clone(), safe.clone()])],
    );
    let valid = op(
        "land",
        [finite, op("lall", [map("isfinite", [trial_residual])])],
    );
    // A finite coefficient vector can still overflow its prediction. Reject
    // the whole reference before constructing the compressed residual.
    let reference = op("ifelse", [valid, safe, zeros]);
    let reference = reducer.summary_for(m, &key, "regression_reference", reference);
    let residual = setup(
        m,
        op("sub", [response, op("mul", [matrix, reference.clone()])]),
    );
    let mut final_columns = columns;
    final_columns.push(residual);
    let scales = reducer.summary_for(
        m,
        &key,
        "regression_scales",
        safe_scales(column_scales(&final_columns)),
    );
    let r = reducer.summary_for(
        m,
        &key,
        "regression_factor",
        factor(&final_columns, scales.clone(), f.observed, count),
    );
    let coefficients = vector(
        design
            .terms
            .into_iter()
            .enumerate()
            .map(|(j, term)| op("sub", [term.coefficient, get(reference.clone(), j)]))
            .chain([num(-1.0)]),
    );
    let residual = op("mul", [r, map("mul", [scales, coefficients])]);
    let standardized = map("divide", [residual, Expr::Node(sigma)]);
    let quadratic = op("sum", [map("mul", [standardized.clone(), standardized])]);
    let zero = materialize(m, num(0.0));
    let parameters = f
        .parameters
        .iter()
        .map(|&(name, _)| (name, if m.resolve(name) == "mu" { zero } else { sigma }))
        .collect::<Vec<_>>();
    let normalizer = reducer.anchor(m, &f, &parameters, num(0.0));
    Some(op(
        "mul",
        [
            num(count as f64),
            op("sub", [normalizer, op("mul", [num(0.5), quadratic])]),
        ],
    ))
}
