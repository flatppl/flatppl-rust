//! Local angular statistics avoid subtracting nearly equal cosines.

use super::{
    Expr, Reducer, call, data_only, map, materialize, num, op, per_cell_with_invariants, resolve,
};
use flatppl_core::{Dim, Module, NodeId, ScalarType, Type, ValueSet};

fn bounded(m: &Module, id: NodeId, r: &Reducer, lo: f64, hi: f64) -> bool {
    let root = resolve(m, id, &r.boundary);
    if let flatppl_core::Node::Ref(v) = m.node(root)
        && r.boundary.contains(&v.name)
    {
        let Some(bid) = m.binding_by_name(v.name) else {
            return false;
        };
        let rhs = m.binding(bid).rhs;
        if !["elementof", "external", "load_data"]
            .iter()
            .any(|head| call(m, rhs, head).is_some())
        {
            return false;
        }
    }
    let set = match m.valueset_of(id) {
        Some(ValueSet::CartPow(set, _)) => set.as_ref(),
        Some(set) => set,
        None => return false,
    };
    matches!(set, ValueSet::Interval(a,b) if *a >= lo && *b <= hi)
}

pub(super) fn reduce(r: &mut Reducer, m: &mut Module, id: NodeId) -> Option<Expr> {
    let f = per_cell_with_invariants(m, id, &r.boundary)?;
    if m.resolve(f.constructor) != "VonMises"
        || !data_only(m, f.observed, &r.allowed, &r.boundary)
        || (!r.preparing && !data_only(m, f.observed, &Default::default(), &r.boundary))
    {
        return None;
    }
    let Type::Array { shape, elem } = m.type_of(f.observed)? else {
        return None;
    };
    let [Dim::Static(n)] = shape.as_ref() else {
        return None;
    };
    if *n < 2 || !matches!(elem.as_ref(), Type::Scalar(ScalarType::Real)) {
        return None;
    }
    let n = num(*n as f64);
    let parameters = f
        .parameters
        .iter()
        .map(|&(key, value)| Some((key, r.scalar(m, value)?)))
        .collect::<Option<Vec<_>>>()?;
    let parameter = |name| {
        parameters
            .iter()
            .find(|(s, _)| m.resolve(*s) == name)
            .map(|(_, v)| *v)
    };
    let mu = parameter("mu")?;
    let kappa = parameter("kappa")?;
    let pi = std::f64::consts::PI;
    // Trig addition changes float32 argument rounding for huge locations, and
    // concentrated primitives lose low bits when subtracting their normalizer.
    if !bounded(m, mu, r, -pi, pi) || !bounded(m, kappa, r, 0.0, 64.0) {
        return None;
    }
    let observed = f.observed;
    if !super::multivariate::finite_data(m, observed, r) {
        return None;
    }
    let x = Expr::Node(observed);
    let valid = map(
        "land",
        [
            map("ge", [x.clone(), num(-pi)]),
            map("le", [x.clone(), num(pi)]),
        ],
    );
    let all_valid = r.summary(m, observed, "von_mises_valid", op("lall", [valid.clone()]));
    let safe = Expr::Node(materialize(m, map("ifelse", [valid, x, num(0.0)])));
    let center = r.summary(
        m,
        observed,
        "von_mises_center",
        op("get", [safe.clone(), Expr::Integer(1)]),
    );
    let delta = map("sub", [safe, center.clone()]);
    let half_sine = map("sin", [map("divide", [delta.clone(), num(2.0)])]);
    let cosine_residual = r.summary(
        m,
        observed,
        "von_mises_cosine_residual",
        op(
            "sum",
            [map(
                "mul",
                [num(-2.0), map("mul", [half_sine.clone(), half_sine])],
            )],
        ),
    );
    let sine = r.summary(
        m,
        observed,
        "von_mises_sine",
        op("sum", [map("sin", [delta])]),
    );
    let angle = op("sub", [center.clone(), Expr::Node(mu)]);
    let anchor = r.anchor(m, &f, &parameters, center);
    let correction = op(
        "mul",
        [
            Expr::Node(kappa),
            op(
                "sub",
                [
                    op("mul", [op("cos", [angle.clone()]), cosine_residual]),
                    op("mul", [op("sin", [angle]), sine]),
                ],
            ),
        ],
    );
    let score = op("add", [op("mul", [n, anchor]), correction]);
    // The reference primitive preserves domain checks even off the circle.
    let outside = r.anchor(m, &f, &parameters, num(2.0 * pi));
    Some(op("ifelse", [all_valid, score, outside]))
}
