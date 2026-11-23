//! Differentiate the mathematical integral, independently of the quadrature mesh.

use flatppl_core::graph::{build_call, expect_builtin_call, rebuild_with_children};
use flatppl_core::{CallHead, Dim, INTEGRATION_POINT, LOG_INTEGRAL, MASS_INTEGRAL, Scalar};

use crate::{Emitter, Error, HashMap, HashSet, Key, Mode, Module, Node, NodeId, Plan, Query, Step};
use crate::{combine_contributions, forward_values, real, reverse_values, rules::Rules, tensor};

pub(super) struct Scope<'a> {
    pub(super) plan: Plan<'a>,
    root: Key,
    body: NodeId,
    axes: &'a [[NodeId; 3]],
    breakpoints: Breakpoints,
    active_cuts: Vec<usize>,
    logarithmic: bool,
}

#[derive(Default)]
struct Breakpoints {
    cuts: Vec<Cut>,
    intervals: Vec<(NodeId, NodeId, [NodeId; 2])>,
}

struct Cut {
    value: NodeId,
    comparisons: Vec<(NodeId, bool)>,
    intervals: Vec<(NodeId, NodeId, bool)>,
}

pub(super) fn visit<'a>(
    outer: &mut Plan<'a>,
    node: NodeId,
    op: &'a str,
    args: &'a [NodeId],
    depth: usize,
) -> Result<bool, Error> {
    let [body, coordinates @ ..] = args else {
        return Err(Error::new(Some(node), "missing numerical integrand"));
    };
    let (axes, extra) = coordinates.as_chunks::<3>();
    if !extra.is_empty() || !matches!(axes.len(), 1 | 2) {
        return Err(Error::new(
            Some(node),
            "expected one or two integration axes",
        ));
    }
    let root = outer.query.key(outer.module, *body)?;
    let mut plan = Plan::new(outer.module, outer.query);
    let interior_active = plan.visit(root, depth + 1)?;
    let breakpoints = find_cuts(outer.module, outer.query, *body, axes)?;
    let mut active_cuts = Vec::new();
    let mut args = Vec::new();
    let mut targets = Vec::new();
    let mut active = interior_active;
    for &[_, lo, hi] in axes {
        for bound in [lo, hi] {
            let key = outer.query.key(outer.module, bound)?;
            active |= outer.visit(key, depth + 1)?;
            targets.push(Some(key));
            args.push(bound);
        }
    }
    for (i, cut) in breakpoints.cuts.iter().enumerate() {
        let key = outer.query.key(outer.module, cut.value)?;
        if outer.visit(key, depth + 1)? {
            active = true;
            active_cuts.push(i);
            targets.push(Some(key));
            args.push(cut.value);
        }
    }
    for input in &outer.query.wrt {
        let key = Key::Input(input.binding);
        if plan.activity.get(&key) == Some(&true) {
            outer.visit(key, depth + 1)?;
            targets.push(Some(key));
            args.push(input.reference);
        }
    }
    if active {
        outer.integrals.insert(
            node,
            Scope {
                plan,
                root,
                body: *body,
                axes,
                breakpoints,
                active_cuts,
                logarithmic: op == LOG_INTEGRAL,
            },
        );
        outer.steps.push(Step {
            node,
            op,
            args,
            targets,
            broadcast: false,
        });
    }
    Ok(active)
}

impl<'a> Scope<'a> {
    pub(super) fn integrand_adjoints(
        &self,
        emitter: &mut Emitter,
        rules: &Rules<'a>,
        mode: Mode,
    ) -> Result<HashMap<Key, NodeId>, Error> {
        if !self.plan.activity[&self.root] {
            return Ok(HashMap::new());
        }
        let one = real(&mut emitter.module, 1.0);
        let inline = std::mem::replace(&mut emitter.inline, true);
        let contributions = reverse_values(emitter, &self.plan, rules, &[self.root], &[one], mode);
        emitter.inline = inline;
        let mut contributions = contributions?;
        Ok(self
            .plan
            .query
            .wrt
            .iter()
            .filter_map(|input| {
                let key = Key::Input(input.binding);
                let terms = contributions.remove(&key)?;
                Some((key, combine_contributions(emitter, input.reference, terms)))
            })
            .collect())
    }

    pub(super) fn pullback(
        &self,
        emitter: &mut Emitter,
        step: &Step<'_>,
        slot: usize,
        cotangent: NodeId,
        adjoints: &HashMap<Key, NodeId>,
    ) -> Result<NodeId, Error> {
        let fixed = self.axes.len() * 2 + self.active_cuts.len();
        let factor = if slot < fixed {
            self.boundary(emitter, step.node, slot)
        } else {
            let key = step.targets[slot].unwrap();
            let &adjoint = adjoints.get(&key).ok_or_else(|| {
                Error::new(Some(step.node), "missing active integrand derivative")
            })?;
            self.capture_pullback(emitter, step.node, step.args[slot], adjoint)?
        };
        Ok(build_call(&mut emitter.module, "mul", &[cotangent, factor]))
    }

    fn capture_pullback(
        &self,
        emitter: &mut Emitter,
        primal: NodeId,
        arg: NodeId,
        adjoint: NodeId,
    ) -> Result<NodeId, Error> {
        let integrate = |emitter: &mut Emitter, indices: &[u32]| {
            let mut args = vec![adjoint];
            args.extend(
                indices
                    .iter()
                    .map(|&index| tensor::int(&mut emitter.module, i64::from(index))),
            );
            let component = build_call(&mut emitter.module, "get", &args);
            self.integrate(emitter, primal, component)
        };
        // Quadrature emits a scalar. Project each component of the shared
        // adjoint and restore the capture's shape outside the binder.
        match tensor::real_shape(self.plan.module, arg) {
            Some([]) => Ok(self.integrate(emitter, primal, adjoint)),
            Some(shape) if shape.contains(&Dim::Static(0)) => emitter.zero_like(arg),
            Some([Dim::Static(n)]) => {
                let terms = (1..=*n)
                    .map(|i| integrate(emitter, &[i]))
                    .collect::<Vec<_>>();
                Ok(build_call(&mut emitter.module, "vector", &terms))
            }
            Some([Dim::Static(rows), Dim::Static(cols)]) => {
                let mut vectors = Vec::new();
                for i in 1..=*rows {
                    let terms = (1..=*cols)
                        .map(|j| integrate(emitter, &[i, j]))
                        .collect::<Vec<_>>();
                    vectors.push(build_call(&mut emitter.module, "vector", &terms));
                }
                let vectors = build_call(&mut emitter.module, "vector", &vectors);
                Ok(build_call(&mut emitter.module, "rowstack", &[vectors]))
            }
            _ => Err(Error::new(
                Some(arg),
                "a numerical integral pullback needs a real scalar, fixed vector, or fixed matrix capture",
            )),
        }
    }

    pub(super) fn pushforward(
        &self,
        emitter: &mut Emitter,
        rules: &Rules<'a>,
        step: &Step<'_>,
        tangents: &HashMap<Key, NodeId>,
    ) -> Result<NodeId, Error> {
        let bounds = self.axes.len() * 2 + self.active_cuts.len();
        let mut terms = Vec::new();
        for (slot, key) in step.targets[..bounds].iter().enumerate() {
            if let Some(&tangent) = key.and_then(|key| tangents.get(&key)) {
                let factor = self.boundary(emitter, step.node, slot);
                let term = build_call(&mut emitter.module, "mul", &[tangent, factor]);
                terms.push(term);
            }
        }
        let captures = step.targets[bounds..]
            .iter()
            .flatten()
            .filter_map(|&key| tangents.get(&key).map(|&tangent| (key, tangent)))
            .collect::<HashMap<_, _>>();
        if !captures.is_empty()
            && let Some(term) = self.interior(emitter, rules, step.node, captures)?
        {
            terms.push(term);
        }
        Ok(emitter.named_adjoint("_ad_integral_tangent", step.node, &terms))
    }

    fn interior(
        &self,
        emitter: &mut Emitter,
        rules: &Rules<'a>,
        primal: NodeId,
        captures: HashMap<Key, NodeId>,
    ) -> Result<Option<NodeId>, Error> {
        // Keep expressions involving the integration coordinates inside the binder.
        let inline = std::mem::replace(&mut emitter.inline, true);
        let tangents = forward_values(emitter, &self.plan, rules, captures);
        emitter.inline = inline;
        let Some(&derivative) = tangents?.get(&self.root) else {
            return Ok(None);
        };
        Ok(Some(self.integrate(emitter, primal, derivative)))
    }

    fn integrate(&self, emitter: &mut Emitter, primal: NodeId, derivative: NodeId) -> NodeId {
        let (derivative, center) = if self.logarithmic {
            // Center the score before integration. A nearly constant score in a
            // tail interval would otherwise amplify rounding of the mass ratio.
            let mut points = HashMap::new();
            for &[point, lo, hi] in self.axes {
                points.insert(point, interior_point(&mut emitter.module, lo, hi));
            }
            let center = substitute(&mut emitter.module, derivative, &mut points);
            let center = emitter.primal(center);
            let finite = build_call(&mut emitter.module, "isfinite", &[center]);
            let zero = real(&mut emitter.module, 0.0);
            let center = build_call(&mut emitter.module, "ifelse", &[finite, center, zero]);
            let residual = build_call(&mut emitter.module, "sub", &[derivative, center]);
            (residual, Some(center))
        } else {
            (derivative, None)
        };
        let negative = build_call(&mut emitter.module, "neg", &[derivative]);
        let positive = self.integral_part(emitter, primal, derivative);
        let negative = self.integral_part(emitter, primal, negative);
        let signed = build_call(&mut emitter.module, "sub", &[positive, negative]);
        match center {
            Some(center) => build_call(&mut emitter.module, "add", &[signed, center]),
            None => signed,
        }
    }

    fn integral_part(&self, emitter: &mut Emitter, primal: NodeId, derivative: NodeId) -> NodeId {
        let module = &mut emitter.module;
        let zero = real(module, 0.0);
        let part = build_call(module, "max", &[derivative, zero]);
        if !self.logarithmic {
            return integral(emitter, MASS_INTEGRAL, part, self.axes.iter());
        }
        let part = build_call(module, "log", &[part]);
        let weighted = build_call(module, "add", &[self.body, part]);
        let neg_inf = real(module, f64::NEG_INFINITY);
        let outside = build_call(module, "isinf", &[self.body]);
        let weighted = build_call(module, "ifelse", &[outside, neg_inf, weighted]);
        normalized_log_integral(emitter, primal, weighted, self.axes.iter())
    }

    fn boundary(&self, emitter: &mut Emitter, primal: NodeId, slot: usize) -> NodeId {
        if slot >= self.axes.len() * 2 {
            let cut = &self.breakpoints.cuts[self.active_cuts[slot - self.axes.len() * 2]];
            return self.jump(emitter, primal, cut);
        }

        let axis = slot / 2;
        let sign = slot % 2;
        let point = self.axes[axis][0];
        let bound = self.axes[axis][sign + 1];
        let mut values = HashMap::from([(point, bound)]);
        for &(predicate, coordinate, [lo, hi]) in &self.breakpoints.intervals {
            if coordinate != point {
                continue;
            }
            let lower_op = if sign == 0 { "le" } else { "lt" };
            let upper_op = if sign == 0 { "lt" } else { "le" };
            let above = build_call(&mut emitter.module, lower_op, &[lo, bound]);
            let below = build_call(&mut emitter.module, upper_op, &[bound, hi]);
            values.insert(
                predicate,
                build_call(&mut emitter.module, "land", &[above, below]),
            );
        }
        for cut in &self.breakpoints.cuts {
            for &(predicate, true_on_left) in &cut.comparisons {
                let op = match (true_on_left, sign == 1) {
                    (true, true) => "le",
                    (true, false) => "lt",
                    (false, true) => "gt",
                    (false, false) => "ge",
                };
                let tied = build_call(&mut emitter.module, op, &[bound, cut.value]);
                values.insert(predicate, tied);
            }
        }
        let body = substitute(&mut emitter.module, self.body, &mut values);
        let value = match (self.axes.len(), self.logarithmic) {
            (1, false) => body,
            (1, true) => log_ratio(emitter, primal, body),
            (_, false) => integral(emitter, MASS_INTEGRAL, body, [&self.axes[1 - axis]]),
            (_, true) => normalized_log_integral(emitter, primal, body, [&self.axes[1 - axis]]),
        };
        if sign == 0 {
            build_call(&mut emitter.module, "neg", &[value])
        } else {
            value
        }
    }

    fn jump(&self, emitter: &mut Emitter, primal: NodeId, cut: &Cut) -> NodeId {
        let [point, lo, hi] = self.axes[0];
        let mut side = |left| {
            let mut values = HashMap::from([(point, cut.value)]);
            for &(predicate, true_on_left) in &cut.comparisons {
                let test = emitter
                    .module
                    .alloc(Node::Lit(Scalar::Bool(left == true_on_left)));
                values.insert(predicate, test);
            }
            for &(predicate, other, lower) in &cut.intervals {
                let test = if left == lower {
                    emitter.module.alloc(Node::Lit(Scalar::Bool(false)))
                } else {
                    let limits = if lower {
                        [cut.value, other]
                    } else {
                        [other, cut.value]
                    };
                    build_call(&mut emitter.module, "lt", &limits)
                };
                values.insert(predicate, test);
            }
            let body = substitute(&mut emitter.module, self.body, &mut values);
            if self.logarithmic {
                log_ratio(emitter, primal, body)
            } else {
                body
            }
        };
        let left = side(true);
        let right = side(false);
        let module = &mut emitter.module;
        let delta = build_call(module, "sub", &[left, right]);
        let inside = build_call(module, "lt", &[lo, cut.value]);
        let zero = real(module, 0.0);
        let delta = build_call(module, "ifelse", &[inside, delta, zero]);
        let inside = build_call(module, "lt", &[cut.value, hi]);
        build_call(module, "ifelse", &[inside, delta, zero])
    }
}

fn interior_point(module: &mut Module, lo: NodeId, hi: NodeId) -> NodeId {
    let lo_finite = build_call(module, "isfinite", &[lo]);
    let hi_finite = build_call(module, "isfinite", &[hi]);
    let half = real(module, 0.5);
    let lo_half = build_call(module, "mul", &[lo, half]);
    let hi_half = build_call(module, "mul", &[hi, half]);
    let middle = build_call(module, "add", &[lo_half, hi_half]);
    let one = real(module, 1.0);
    let above = build_call(module, "add", &[lo, one]);
    let below = build_call(module, "sub", &[hi, one]);
    let zero = real(module, 0.0);
    let from_lo = build_call(module, "ifelse", &[hi_finite, middle, above]);
    let from_hi = build_call(module, "ifelse", &[hi_finite, below, zero]);
    build_call(module, "ifelse", &[lo_finite, from_lo, from_hi])
}

fn integral<'a>(
    emitter: &mut Emitter,
    op: &str,
    body: NodeId,
    axes: impl IntoIterator<Item = &'a [NodeId; 3]>,
) -> NodeId {
    let module = &mut emitter.module;
    let mut args = vec![body];
    let mut coordinates = HashMap::new();
    for &[coordinate, lo, hi] in axes {
        // A named inner integral can reuse the primal's own coordinate.
        // Give this binder distinct IDs so quadrature rebinding keeps it fixed.
        let token = module.node_count() as i64;
        let token = tensor::int(module, token);
        let point = build_call(module, INTEGRATION_POINT, &[token]);
        coordinates.insert(coordinate, point);
        args.extend([point, lo, hi]);
    }
    args[0] = substitute(module, body, &mut coordinates);
    let integral = build_call(module, op, &args);
    // Primal conformance permits inner integrals only through closed
    // bindings. Keep derivative integrals at the same lexical boundary.
    if emitter.inline {
        emitter.define("_ad_integral", integral)
    } else {
        integral
    }
}

fn normalized_log_integral<'a>(
    emitter: &mut Emitter,
    primal: NodeId,
    body: NodeId,
    axes: impl IntoIterator<Item = &'a [NodeId; 3]>,
) -> NodeId {
    // Apply the normalization before quadrature. Otherwise a small primal
    // mass magnifies the numerator's absolute quadrature error.
    let denominator = emitter.primal(primal);
    let body = build_call(&mut emitter.module, "sub", &[body, denominator]);
    let value = integral(emitter, LOG_INTEGRAL, body, axes);
    build_call(&mut emitter.module, "exp", &[value])
}

fn log_ratio(emitter: &mut Emitter, primal: NodeId, numerator: NodeId) -> NodeId {
    let denominator = emitter.primal(primal);
    let ratio = build_call(&mut emitter.module, "sub", &[numerator, denominator]);
    build_call(&mut emitter.module, "exp", &[ratio])
}

fn substitute(module: &mut Module, id: NodeId, values: &mut HashMap<NodeId, NodeId>) -> NodeId {
    if let Some(&node) = values.get(&id) {
        return node;
    }
    // The verifier forbids a coordinate in a top-level binding.
    let children = module.node(id).children();
    let replaced = children
        .iter()
        .map(|&child| substitute(module, child, values))
        .collect::<Vec<_>>();
    let node = if children == replaced {
        id
    } else {
        rebuild_with_children(module, id, &replaced)
    };
    values.insert(id, node);
    node
}

fn find_cuts(
    module: &Module,
    query: &Query<'_>,
    body: NodeId,
    axes: &[[NodeId; 3]],
) -> Result<Breakpoints, Error> {
    let coordinates = axes.iter().map(|axis| axis[0]).collect::<HashSet<_>>();
    let mut seen = HashSet::new();
    let mut pending = vec![body];
    let mut indices = HashMap::new();
    let mut breakpoints = Breakpoints::default();
    while let Some(node) = pending.pop() {
        if !seen.insert(node) {
            continue;
        }
        let children = module.node(node).children();
        if let Node::Call(call) = module.node(node)
            && let CallHead::Builtin(head) = call.head
            && let [lhs, rhs] = call.args.as_ref()
        {
            let op = module.resolve(head);
            if matches!(op, "lt" | "le" | "gt" | "ge" | "in") {
                let lhs_is_point = coordinates.contains(lhs);
                let rhs_is_point = coordinates.contains(rhs);
                let (point, active) = cutoff_dependencies(module, query, &coordinates, &children)?;
                if op == "in"
                    && lhs_is_point
                    && let Key::Node(set) = query.key(module, *rhs)?
                    && let Some(interval) = expect_builtin_call(module, set, "interval")
                    && let &[lo, hi] = interval.args.as_ref()
                    && !cutoff_dependencies(module, query, &coordinates, &[lo, hi])?.0
                    && let Some(&[_, bound_lo, bound_hi]) = axes.iter().find(|axis| axis[0] == *lhs)
                {
                    let identical = query.key(module, lo)? == query.key(module, bound_lo)?
                        && query.key(module, hi)? == query.key(module, bound_hi)?;
                    if identical || axes.len() == 1 {
                        breakpoints.intervals.push((node, *lhs, [lo, hi]));
                        if !identical {
                            for (value, other, lower) in [(lo, hi, true), (hi, lo, false)] {
                                let cut =
                                    get_cut(module, query, &mut indices, &mut breakpoints, value)?;
                                cut.intervals.push((node, other, lower));
                            }
                        }
                        pending.extend(children);
                        continue;
                    }
                }
                if point && axes.len() == 1 && op != "in" && lhs_is_point != rhs_is_point {
                    let value = if lhs_is_point { *rhs } else { *lhs };
                    let (depends_on_point, _) =
                        cutoff_dependencies(module, query, &coordinates, &[value])?;
                    if !depends_on_point {
                        let true_on_left = matches!(op, "lt" | "le") == lhs_is_point;
                        let cut = get_cut(module, query, &mut indices, &mut breakpoints, value)?;
                        cut.comparisons.push((node, true_on_left));
                        pending.extend(children);
                        continue;
                    }
                }
                if point && active {
                    return Err(Error::new(
                        Some(node),
                        "a moving cutoff needs a direct scalar coordinate comparison",
                    ));
                }
            }
        }
        pending.extend(children);
    }
    Ok(breakpoints)
}

fn get_cut<'a>(
    module: &Module,
    query: &Query<'_>,
    indices: &mut HashMap<Key, usize>,
    breakpoints: &'a mut Breakpoints,
    value: NodeId,
) -> Result<&'a mut Cut, Error> {
    let key = query.key(module, value)?;
    let i = *indices.entry(key).or_insert_with(|| {
        let i = breakpoints.cuts.len();
        breakpoints.cuts.push(Cut {
            value,
            comparisons: Vec::new(),
            intervals: Vec::new(),
        });
        i
    });
    Ok(&mut breakpoints.cuts[i])
}

fn cutoff_dependencies(
    module: &Module,
    query: &Query<'_>,
    coordinates: &HashSet<NodeId>,
    terms: &[NodeId],
) -> Result<(bool, bool), Error> {
    let mut point = false;
    let mut active = false;
    let mut terms = terms.to_vec();
    let mut seen = HashSet::new();
    while let Some(term) = terms.pop() {
        if !seen.insert(term) {
            continue;
        }
        match query.key(module, term)? {
            Key::Input(binding) => {
                active |= query.wrt.iter().any(|input| input.binding == binding);
            }
            Key::Node(node) => {
                point |= coordinates.contains(&node);
                module.for_each_child(node, |child| terms.push(child));
            }
        }
    }
    Ok((point, active))
}
