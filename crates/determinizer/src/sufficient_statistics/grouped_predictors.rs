//! Lift pointwise predictors over one runtime partition before family reduction.

use std::collections::HashMap;

use super::grouped::{factor, gather_parts};
use super::multivariate::held_cells;
use super::regression::{bounds, setup};
use super::runtime_partition::{Partition, size};
use super::{Expr, Factor, Reducer, call, map, materialize, num, op, resolve};
use flatppl_core::{Dim, Module, Node, NodeId, ScalarType, Type, ValueSet};

enum Plan {
    Scalar(NodeId),
    Gather(NodeId),
    Apply(&'static str, Vec<usize>),
}

#[derive(Default)]
struct Predictor {
    plans: Vec<Plan>,
    ranges: Vec<Option<(f64, f64)>>,
    pullbacks: Vec<Option<f64>>,
    memo: HashMap<NodeId, usize>,
    gather: Option<NodeId>,
    partition: Option<(NodeId, i64, u32)>,
    links: usize,
}

fn operation(m: &Module, id: NodeId) -> Option<(&'static str, Vec<NodeId>)> {
    let c = call(m, id, "broadcast")?;
    let (head, args) = c.args.split_first()?;
    let Node::Const(head) = m.node(*head) else {
        return None;
    };
    let name = match (m.resolve(*head), args.len()) {
        ("add", 2) => "add",
        ("sub", 2) => "sub",
        ("mul", 2) => "mul",
        ("divide", 2) => "divide",
        ("neg", 1) => "neg",
        ("exp", 1) => "exp",
        ("invlogit", 1) => "invlogit",
        _ => return None,
    };
    Some((name, args.to_vec()))
}

/// Enclose endpoint arithmetic in both precisions, including a small rounding
/// margin for the native exp/invlogit implementations. Unknown ranges fall back
/// in families that require a numerical proof.
fn interval(name: &str, ranges: &[Option<(f64, f64)>]) -> Option<(f64, f64)> {
    let values = ranges.iter().copied().collect::<Option<Vec<_>>>()?;
    let (a, b) = values[0];
    let mut points = vec![vec![a], vec![b]];
    if let Some(&(c, d)) = values.get(1) {
        points = vec![vec![a, c], vec![a, d], vec![b, c], vec![b, d]];
        if name == "divide" && c <= 0.0 && d >= 0.0 {
            return None;
        }
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for point in points {
        let x = point[0];
        let y = point.get(1).copied().unwrap_or(0.0);
        let value = match name {
            "add" => x + y,
            "sub" => x - y,
            "mul" => x * y,
            "divide" => x / y,
            "neg" => -x,
            "exp" => x.exp(),
            "invlogit" => 1.0 / (1.0 + (-x).exp()),
            _ => return None,
        };
        let x = x as f32;
        let y = y as f32;
        if [x, y]
            .iter()
            .any(|value| *value != 0.0 && value.abs() < f32::MIN_POSITIVE)
        {
            return None;
        }
        let single = match name {
            "add" => x + y,
            "sub" => x - y,
            "mul" => x * y,
            "divide" => x / y,
            "neg" => -x,
            "exp" => x.exp(),
            "invlogit" => 1.0 / (1.0 + (-x).exp()),
            _ => return None,
        };
        if !value.is_finite()
            || !single.is_finite()
            || (single != 0.0 && single.abs() < f32::MIN_POSITIVE)
            || (value != 0.0 && value.abs() < f64::from(f32::MIN_POSITIVE))
        {
            return None;
        }
        let mut lower = single;
        let mut upper = single;
        for _ in 0..8 {
            lower = lower.next_down();
            upper = upper.next_up();
        }
        lo = lo.min(value.next_down()).min(f64::from(lower));
        hi = hi.max(value.next_up()).max(f64::from(upper));
    }
    if name == "invlogit" {
        lo = lo.max(0.0);
        hi = hi.min(1.0);
    }
    Some((lo, hi))
}

impl Predictor {
    fn parse(&mut self, m: &Module, id: NodeId, r: &Reducer, depth: usize) -> Option<usize> {
        if depth >= 64 {
            return None;
        }
        let id = resolve(m, id, &r.boundary);
        if let Some(&plan) = self.memo.get(&id) {
            return Some(plan);
        }
        if self.plans.len() >= 128 {
            return None;
        }
        let (plan, range) = if let Some(scalar) = r.scalar(m, id) {
            (Plan::Scalar(scalar), bounds(m, scalar, r))
        } else if let Some((array, indices, origin)) = gather_parts(m, id, r) {
            let groups = size(m, array, ScalarType::Real)?;
            let indices = resolve(m, indices, &r.boundary);
            if self.partition.is_some_and(|(previous, base, extent)| {
                base != origin
                    || extent != groups
                    || r.data_key(m, previous) != r.data_key(m, indices)
            }) {
                return None;
            }
            self.partition = Some((indices, origin, groups));
            self.gather = Some(id);
            (Plan::Gather(array), bounds(m, array, r))
        } else {
            size(m, id, ScalarType::Real)?;
            let (name, args) = operation(m, id)?;
            let children = args
                .iter()
                .map(|&arg| self.parse(m, arg, r, depth + 1))
                .collect::<Option<Vec<_>>>()?;
            let ranges = children
                .iter()
                .map(|&child| self.ranges[child])
                .collect::<Vec<_>>();
            self.links += 1;
            (Plan::Apply(name, children), interval(name, &ranges))
        };
        let index = self.plans.len();
        if index >= 128 {
            return None;
        }
        let pullback = self.pullback_bound(&plan);
        self.plans.push(plan);
        self.ranges.push(range);
        self.pullbacks.push(pullback);
        self.memo.insert(id, index);
        Some(index)
    }

    // Bound every intermediate reverse path, not only the final derivative.
    // Small link derivatives cannot repair an adjoint that already overflowed.
    fn pullback_bound(&self, plan: &Plan) -> Option<f64> {
        let Plan::Apply(name, children) = plan else {
            return Some(1.0);
        };
        let magnitude = |i: usize| self.ranges[i].map(|(lo, hi)| lo.abs().max(hi.abs()));
        let minimum = |i: usize| {
            self.ranges[i]
                .and_then(|(lo, hi)| (lo > 0.0 || hi < 0.0).then_some(lo.abs().min(hi.abs())))
        };
        let left = children[0];
        let right = children.get(1).copied();
        let jacobians = match *name {
            "mul" => vec![magnitude(right?)?, magnitude(left)?],
            "divide" => {
                let denominator = minimum(right?)?;
                let maximum = magnitude(right?)?;
                if (denominator as f32).powi(2) < f32::MIN_POSITIVE * 128.0
                    || (maximum as f32).powi(2) > f32::MAX / 128.0
                {
                    return None;
                }
                // Quotient reverse mode can multiply adjoint*numerator before
                // dividing by the squared denominator. Bound that product too.
                let numerator = magnitude(left)?;
                vec![
                    1.0 / denominator,
                    numerator.max(numerator / denominator.powi(2)),
                ]
            }
            "exp" => vec![self.ranges[left]?.1.exp()],
            "invlogit" => vec![0.25],
            _ => vec![1.0; children.len()],
        };
        let bound = children
            .iter()
            .zip(jacobians)
            .try_fold(0.0, |sum, (&child, jacobian)| {
                Some(sum + jacobian.max(1.0) * self.pullbacks[child]?)
            })?;
        bound.is_finite().then_some(bound.max(1.0))
    }

    fn finite_adjoints(
        &self,
        m: &Module,
        f: &Factor,
        roots: &[usize],
        r: &Reducer,
        count: u32,
    ) -> Option<()> {
        let parameter = |name: &str| {
            f.parameters
                .iter()
                .zip(roots)
                .find(|((key, _), _)| m.resolve(*key) == name)
                .and_then(|(_, &root)| self.ranges[root])
        };
        let data = bounds(m, f.observed, r);
        let family = m.resolve(f.constructor);
        let mut total = 0.0;
        for ((name, _), &root) in f.parameters.iter().zip(roots) {
            let name = m.resolve(*name);
            if name == "n" {
                continue;
            }
            let (lo, hi) = self.ranges[root]?;
            let derivative = match family {
                "Bernoulli" | "Binomial" | "Geometric" => {
                    if lo <= 0.0 || hi >= 1.0 {
                        return None;
                    }
                    if family == "Geometric" {
                        1.0 / lo + i64::MAX as f64 / (1.0 - hi)
                    } else {
                        let trials = if family == "Binomial" {
                            parameter("n")?.1
                        } else {
                            1.0
                        };
                        trials * (1.0 / lo + 1.0 / (1.0 - hi))
                    }
                }
                "Poisson" if lo > 0.0 => i64::MAX as f64 / lo + 1.0,
                "Exponential" if lo > 0.0 => 1.0 / lo + data?.1.abs(),
                "Gamma" if lo > 0.0 => {
                    if name == "rate" {
                        parameter("shape")?.1 / lo + data?.1.abs()
                    } else {
                        let (a, b) = parameter("rate")?;
                        let (x, y) = data?;
                        1.0 / lo
                            + hi.ln().abs()
                            + a.ln().abs()
                            + b.ln().abs()
                            + x.ln().abs()
                            + y.ln().abs()
                            + 2.0
                    }
                }
                "Beta" if lo > 0.0 => {
                    let (a, b) = parameter(if name == "alpha" { "beta" } else { "alpha" })?;
                    let (x, y) = data?;
                    let log_data = if name == "alpha" {
                        x.ln().abs().max(y.ln().abs())
                    } else {
                        (-x).ln_1p().abs().max((-y).ln_1p().abs())
                    };
                    log_data + 1.0 / lo + hi.ln().abs() + 1.0 / (lo + a) + (hi + b).ln().abs() + 4.0
                }
                "Normal" | "LogNormal" => {
                    let (a, b) = parameter("mu")?;
                    let (s, _) = parameter("sigma")?;
                    if s <= 0.0 {
                        return None;
                    }
                    let (x, y) = data?;
                    let (x, y) = if family == "LogNormal" {
                        (x.ln(), y.ln())
                    } else {
                        (x, y)
                    };
                    let difference = (x - b).abs().max((y - a).abs());
                    if name == "mu" {
                        difference / s.powi(2)
                    } else {
                        1.0 / s + difference.powi(2) / s.powi(3)
                    }
                }
                _ => return None,
            };
            total += derivative * self.pullbacks[root]?;
        }
        (total.is_finite() && f64::from(count) * total < f64::from(f32::MAX) / 128.0).then_some(())
    }

    fn rebuild(
        &self,
        m: &mut Module,
        index: usize,
        active: &Expr,
        cache: &mut [Option<Expr>],
    ) -> Expr {
        if let Some(value) = &cache[index] {
            return value.clone();
        }
        let value = match &self.plans[index] {
            Plan::Scalar(value) | Plan::Gather(value) => Expr::Node(*value),
            Plan::Apply(name, children) => {
                let operands = children.iter().enumerate().map(|(position, &child)| {
                    let value = self.rebuild(m, child, active, cache);
                    let fallback = if *name == "divide" && position == 1 { 1 } else { 0 };
                    let integer = matches!(&self.plans[child], Plan::Scalar(id) if matches!(m.type_of(*id), Some(Type::Scalar(ScalarType::Integer))));
                    let fallback = if integer { Expr::Integer(fallback) } else { num(fallback as f64) };
                    // Mask every operand before its operation. Masking only the
                    // result leaves inactive exp/quotient adjoints nonfinite.
                    map("ifelse", [active.clone(), value, fallback])
                }).collect::<Vec<_>>();
                setup(m, map(name, operands))
            }
        };
        cache[index] = Some(value.clone());
        value
    }
}

pub(super) fn reduce(r: &mut Reducer, m: &mut Module, id: NodeId) -> Option<NodeId> {
    if !r.preparing {
        return None;
    }
    let f = factor(m, id, r)?;
    let family = m.resolve(f.constructor).to_owned();
    let kind = match family.as_str() {
        "Bernoulli" => ScalarType::Boolean,
        "Poisson" | "Geometric" | "Binomial" => ScalarType::Integer,
        "Normal" | "LogNormal" | "Gamma" | "Beta" | "Exponential" => ScalarType::Real,
        _ => return None,
    };
    let mut predictor = Predictor::default();
    let roots = f
        .parameters
        .iter()
        .map(|&(_, value)| predictor.parse(m, value, r, 0))
        .collect::<Option<Vec<_>>>()?;
    if predictor.links == 0 {
        return None;
    }
    let fake = Factor {
        constructor: f.constructor,
        parameters: vec![(f.parameters[0].0, predictor.gather?)],
        observed: f.observed,
    };
    let partition = Partition::new(m, &fake, r, kind)?;
    predictor.finite_adjoints(m, &f, &roots, r, partition.count)?;
    let mut candidate = m.clone();
    let mut reducer = r.clone();
    let occupancy = match family.as_str() {
        "Bernoulli" => {
            let counts = super::grouped::runtime_bernoulli_counts(
                &mut reducer,
                &mut candidate,
                f.observed,
                partition.indices,
                partition.origin,
                partition.groups,
            );
            let end = op("mul", [Expr::Local, Expr::Integer(2)]);
            held_cells(
                op(
                    "add",
                    [
                        op(
                            "get",
                            [Expr::Held(0), op("sub", [end.clone(), Expr::Integer(1)])],
                        ),
                        op("get", [Expr::Held(0), end]),
                    ],
                ),
                partition.axes(),
                [counts],
            )
        }
        "Binomial" => {
            let (counts, outcomes, _) = super::grouped_counts::binomial_counts(
                &mut reducer,
                &mut candidate,
                &f,
                &partition,
            )?;
            let offset = op(
                "mul",
                [
                    op("sub", [Expr::Local, Expr::Integer(1)]),
                    Expr::Integer(i64::from(outcomes)),
                ],
            );
            let indices = map(
                "add",
                [
                    op(
                        "indicesof",
                        [op(
                            "fill",
                            [Expr::Integer(0), Expr::Integer(i64::from(outcomes))],
                        )],
                    ),
                    offset,
                ],
            );
            held_cells(
                op("sum", [op("get", [Expr::Held(0), indices])]),
                partition.axes(),
                [counts],
            )
        }
        _ => {
            let checked = partition.checked(&mut candidate);
            partition.counts(&mut reducer, &mut candidate, checked)
        }
    };
    let active = setup(&mut candidate, map("gt", [occupancy, Expr::Integer(0)]));
    let mut cache = vec![None; predictor.plans.len()];
    let mut parameters = Vec::new();
    for (&(name, old), &root) in f.parameters.iter().zip(&roots) {
        let mut values = predictor.rebuild(&mut candidate, root, &active, &mut cache);
        if matches!(predictor.plans[root], Plan::Scalar(_)) {
            parameters.push((name, materialize(&mut candidate, values)));
            continue;
        }
        if matches!(predictor.plans[root], Plan::Apply(..)) {
            let fallback = if candidate.resolve(name) == "p" {
                0.5
            } else if candidate.resolve(name) == "mu"
                && matches!(family.as_str(), "Normal" | "LogNormal")
            {
                0.0
            } else {
                1.0
            };
            values = setup(
                &mut candidate,
                map("ifelse", [active.clone(), values, num(fallback)]),
            );
            let node = materialize(&mut candidate, values.clone());
            let rhs = resolve(&candidate, node, &reducer.boundary);
            let shape = Dim::Static(partition.groups);
            candidate.set_type(
                node,
                Type::Array {
                    shape: Box::new([shape]),
                    elem: Box::new(Type::Scalar(ScalarType::Real)),
                },
            );
            if let Some((lo, hi)) = predictor.ranges[root] {
                let set = ValueSet::CartPow(
                    Box::new(ValueSet::Interval(lo.min(fallback), hi.max(fallback))),
                    shape,
                );
                candidate.set_valueset(node, set.clone());
                candidate.set_valueset(rhs, set);
            }
        }
        let value = materialize(
            &mut candidate,
            op(
                if partition.origin == 0 { "get0" } else { "get" },
                [values, Expr::Node(partition.indices)],
            ),
        );
        candidate.set_type(value, m.type_of(old)?.clone());
        parameters.push((name, value));
    }
    let rebuilt = Factor {
        constructor: f.constructor,
        parameters,
        observed: f.observed,
    };
    let result = super::grouped::reduce_factor(&mut reducer, &mut candidate, &rebuilt)
        .or_else(|| super::grouped_moments::reduce_factor(&mut reducer, &mut candidate, &rebuilt))
        .or_else(|| super::grouped_positive::reduce_factor(&mut reducer, &mut candidate, &rebuilt))
        .or_else(|| super::grouped_counts::reduce_factor(&mut reducer, &mut candidate, &rebuilt))
        .or_else(|| {
            super::grouped_poisson::reduce_factor(&mut reducer, &mut candidate, &rebuilt)
        })?;
    reducer.rule = Some(("runtime grouped predictors", None));
    *m = candidate;
    *r = reducer;
    Some(result)
}
