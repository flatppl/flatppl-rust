//! Local scalar differentiation inside a finite sum or mean aggregate.

use flatppl_core::graph::{build_call, expect_builtin_call};

use crate::contract::{aggregate, axis_name, checked_axes, cotangent_product, indexed};
use crate::{
    Emitter, Error, HashMap, HashSet, Key, Module, Node, NodeId, Plan, Step, Symbol,
    forward_values, real, rules::Rules, tensor,
};

pub(super) struct Scope<'a> {
    pub(super) plan: Plan<'a>,
    root: Key,
    reduction: NodeId,
    count: u64,
    output_axes: Vec<NodeId>,
    sites: Vec<Site>,
}

struct Site {
    local: NodeId,
    primal: NodeId,
    axes: Vec<NodeId>,
}

pub(super) fn visit<'a>(
    outer: &mut Plan<'a>,
    node: NodeId,
    args: &[NodeId],
    depth: usize,
) -> Result<Option<bool>, Error> {
    let Some(Aggregate {
        reduction,
        axes: output_axes,
        body,
        sites,
        count,
    }) = read(outer.module, args)
    else {
        return Ok(None);
    };
    let mut plan = Plan::new(outer.module, outer.query);
    plan.in_integrand = outer.in_integrand;
    let root = outer.query.key(outer.module, body)?;
    let mut inputs = Vec::with_capacity(sites.len());
    let mut targets = Vec::with_capacity(sites.len());
    let mut active = false;
    let mut captures = Vec::with_capacity(sites.len());
    for site in sites {
        let local = outer.query.key(outer.module, site.local)?;
        if plan.activity.contains_key(&local) {
            continue;
        }
        let target = outer.query.key(outer.module, site.primal)?;
        let depends = outer.visit(target, depth + 1)?;
        plan.activity.insert(local, depends);
        active |= depends;
        targets.push(depends.then_some(target));
        inputs.push(site.primal);
        captures.push(site);
    }
    if active {
        plan.visit(root, depth + 1)?;
        outer.aggregates.insert(
            node,
            Scope {
                plan,
                root,
                reduction,
                count,
                output_axes,
                sites: captures,
            },
        );
        outer.steps.push(Step {
            node,
            op: "scoped aggregate",
            args: inputs,
            targets,
            broadcast: false,
        });
    }
    Ok(Some(active))
}

impl<'a> Scope<'a> {
    pub(super) fn pullback(
        &self,
        emitter: &mut Emitter,
        rules: &Rules<'a>,
        slot: usize,
        bar: NodeId,
        masked: bool,
    ) -> Result<NodeId, Error> {
        let site = &self.sites[slot];
        let local = self.plan.query.key(self.plan.module, site.local)?;
        let one = real(&mut emitter.module, 1.0);
        let seed = HashMap::from([(local, one)]);
        let Some(partial) = self.differentiate(emitter, rules, seed)? else {
            return emitter.zero_like(site.primal);
        };
        let m = &mut emitter.module;
        let bar = indexed(m, bar, &self.output_axes);
        let derivative = cotangent_product(m, bar, partial, masked);
        let anchor = self.anchor_axes(emitter, derivative)?;
        let m = &mut emitter.module;
        let sum = m.intern("sum");
        let sum = m.alloc(Node::Const(sum));
        let numerator = aggregate(m, sum, &site.axes, anchor);
        if self.count == 1 {
            Ok(numerator)
        } else {
            let count = real(m, self.count as f64);
            Ok(tensor::broadcast(m, "divide", &[numerator, count]))
        }
    }

    pub(super) fn pushforward(
        &self,
        emitter: &mut Emitter,
        rules: &Rules<'a>,
        step: &Step<'_>,
        tangents: &HashMap<Key, NodeId>,
    ) -> Result<NodeId, Error> {
        let mut seed = HashMap::new();
        for (site, &arg) in self.sites.iter().zip(&step.targets) {
            if let Some(&tangent) = arg.and_then(|key| tangents.get(&key)) {
                let local = self.plan.query.key(self.plan.module, site.local)?;
                let value = indexed(&mut emitter.module, tangent, &site.axes);
                seed.insert(local, value);
            }
        }
        let Some(tangent) = self.differentiate(emitter, rules, seed)? else {
            return emitter.zero_like(step.node);
        };
        let tangent = self.anchor_axes(emitter, tangent)?;
        Ok(aggregate(
            &mut emitter.module,
            self.reduction,
            &self.output_axes,
            tangent,
        ))
    }

    fn differentiate(
        &self,
        emitter: &mut Emitter,
        rules: &Rules<'a>,
        seed: HashMap<Key, NodeId>,
    ) -> Result<Option<NodeId>, Error> {
        // A binding containing a scoped index cannot be hoisted.
        let inline = std::mem::replace(&mut emitter.inline, true);
        let tangents = forward_values(emitter, &self.plan, rules, seed);
        emitter.inline = inline;
        Ok(tangents?.remove(&self.root))
    }

    fn anchor_axes(&self, emitter: &mut Emitter, mut body: NodeId) -> Result<NodeId, Error> {
        let mut used = axis_names(&emitter.module, body);
        for site in &self.sites {
            let mut added = false;
            for &axis in &site.axes {
                let name = axis_name(&emitter.module, axis).expect("validated axis");
                added |= used.insert(name);
            }
            if added {
                // The local derivative can remove an axis present in the primal.
                // Carry its extent so the same finite sum counts every cell.
                let one = real(&mut emitter.module, 1.0);
                let ones = tensor::fill_like(emitter, one, site.primal).ok_or_else(|| {
                    Error::new(Some(site.primal), "aggregate needs a fixed extent")
                })?;
                let m = &mut emitter.module;
                let cell = indexed(m, ones, &site.axes);
                body = build_call(m, "mul", &[body, cell]);
            }
        }
        Ok(body)
    }
}

struct Aggregate {
    reduction: NodeId,
    axes: Vec<NodeId>,
    body: NodeId,
    sites: Vec<Site>,
    count: u64,
}

fn read(module: &Module, args: &[NodeId]) -> Option<Aggregate> {
    let [reduction, axes, body] = *args else {
        return None;
    };
    let Node::Const(reducer) = module.node(reduction) else {
        return None;
    };
    let mean = match module.resolve(*reducer) {
        "mean" => true,
        "sum" => false,
        _ => return None,
    };
    let axes = expect_builtin_call(module, axes, "vector")?.args.to_vec();
    checked_axes(module, &axes)?;
    let mut extents = HashMap::new();
    let mut sites = Vec::new();
    let mut pending = vec![body];
    let mut seen = HashSet::new();
    while let Some(node) = pending.pop() {
        if !seen.insert(node) {
            continue;
        }
        let get = expect_builtin_call(module, node, "get")
            .or_else(|| expect_builtin_call(module, node, "get0"));
        if let Some(get) = get
            && let [primal, selectors @ ..] = get.args.as_ref()
            && selectors
                .iter()
                .any(|&s| matches!(module.node(s), Node::Axis(_)))
        {
            checked_axes(module, selectors)?;
            let shape = tensor::real_shape(module, *primal)?;
            if shape.len() != selectors.len() {
                return None;
            }
            for (&axis, &dim) in selectors.iter().zip(shape) {
                let name = axis_name(module, axis)?;
                if extents.insert(name, dim).is_some_and(|old| old != dim) {
                    return None;
                }
            }
            sites.push(Site {
                local: node,
                primal: *primal,
                axes: selectors.to_vec(),
            });
        } else if matches!(module.node(node), Node::Ref(_))
            && tensor::real_shape(module, node) == Some(&[])
        {
            sites.push(Site {
                local: node,
                primal: node,
                axes: Vec::new(),
            });
        } else {
            if ["aggregate", "metricsum"]
                .iter()
                .any(|head| expect_builtin_call(module, node, head).is_some())
            {
                return None;
            }
            pending.extend(module.node(node).children());
        }
    }
    if extents.is_empty()
        || !axes
            .iter()
            .all(|&axis| axis_name(module, axis).is_some_and(|name| extents.contains_key(&name)))
    {
        return None;
    }
    let count = if mean {
        for &axis in &axes {
            extents.remove(&axis_name(module, axis)?);
        }
        extents.values().try_fold(1u64, |count, dim| match dim {
            flatppl_core::Dim::Static(size) => count.checked_mul(u64::from(*size)),
            flatppl_core::Dim::Dynamic => None,
        })?
    } else {
        1
    };
    if count == 0 {
        return None;
    }
    Some(Aggregate {
        reduction,
        axes,
        body,
        sites,
        count,
    })
}

fn axis_names(module: &Module, body: NodeId) -> HashSet<Symbol> {
    let mut axes = HashSet::new();
    let mut visited = HashSet::new();
    let mut pending = vec![body];
    while let Some(node) = pending.pop() {
        if !visited.insert(node) {
            continue;
        }
        if let Some(axis) = axis_name(module, node) {
            axes.insert(axis);
        } else {
            pending.extend(module.node(node).children());
        }
    }
    axes
}
