//! Lower real metric axis transformations to ordinary deterministic tensor algebra.

use flatppl_core::graph::{build_call, expect_builtin_call, rebuild_with_children};
use flatppl_core::{Axis, DEFAULT_MAX_DEPTH, Dim, Module, Node, NodeId, RefNs, Symbol, Variance};
use std::collections::{HashMap, HashSet};

use crate::contract::{aggregate, axis, indexed};
use crate::tensor;

pub(super) fn lower(module: &mut Module, node: NodeId) -> Option<NodeId> {
    let call = expect_builtin_call(module, node, "metricsum")?;
    let [metric, axes, body] = *call.args.as_ref() else {
        return None;
    };
    let [Dim::Static(rows), Dim::Static(cols)] = tensor::real_shape(module, metric)? else {
        return None;
    };
    if rows != cols || *rows == 0 {
        return None;
    }
    let extent = *rows;
    let output = output_axes(module, axes)?;
    let sites = sites(module, body, extent)?;
    if !valid_variances(&output, &sites) {
        return None;
    }

    let sum = module.intern("sum");
    let sum = module.alloc(Node::Const(sum));
    let inverse = build_call(module, "inv", &[metric]);
    let mut substitutions = HashMap::new();
    for site in sites {
        let mut array = site.array;
        let rank = site.axes.len();
        for (position, selector) in site.axes.iter().enumerate() {
            if selector.variance == Some(Variance::Lower) {
                array = transform_axis(module, sum, array, inverse, rank, position);
            }
        }
        let selectors = if rank == 2 && site.axes[0].name == site.axes[1].name {
            array = build_call(module, "diag", &[array]);
            vec![plain_axis(module, site.axes[0])]
        } else {
            site.axes
                .into_iter()
                .map(|ax| plain_axis(module, ax))
                .collect()
        };
        let cell = indexed(module, array, &selectors);
        substitutions.insert(site.node, cell);
    }
    let body = substitute(module, body, &mut substitutions, 0)?;
    let axes: Vec<_> = output.iter().map(|&ax| plain_axis(module, ax)).collect();
    let mut result = aggregate(module, sum, &axes, body);
    for (position, selector) in output.iter().enumerate() {
        if selector.variance == Some(Variance::Lower) {
            result = transform_axis(module, sum, result, metric, axes.len(), position);
        }
    }
    let expected = module.type_of(node)?.clone();
    let (types, diagnostics) = flatppl_infer::infer_expression(module, result, &[]);
    if types.get(&result) != Some(&expected)
        || diagnostics
            .iter()
            .any(|d| d.severity == flatppl_infer::Severity::Error)
    {
        return None;
    }
    for (node, ty) in types {
        if module.type_of(node).is_none() {
            module.set_type(node, ty);
        }
    }
    if has_local_reference(module, result) {
        Some(result)
    } else {
        crate::recurrence::bind_value(module, result, "_ad_metric")
    }
}

struct Site {
    node: NodeId,
    array: NodeId,
    axes: Vec<Axis>,
}

fn output_axes(module: &Module, node: NodeId) -> Option<Vec<Axis>> {
    let call = expect_builtin_call(module, node, "vector")?;
    let mut names = HashSet::new();
    call.args
        .iter()
        .map(|&id| {
            let axis = read_axis(module, id)?;
            names.insert(axis.name).then_some(axis)
        })
        .collect()
}

fn sites(module: &Module, body: NodeId, extent: u32) -> Option<Vec<Site>> {
    let mut sites = Vec::new();
    let mut visited = HashSet::new();
    let mut pending = vec![body];
    while let Some(node) = pending.pop() {
        if !visited.insert(node) {
            continue;
        }
        if let Some(call) = expect_builtin_call(module, node, "get")
            .or_else(|| expect_builtin_call(module, node, "get0"))
            && let [array, selectors @ ..] = call.args.as_ref()
            && selectors
                .iter()
                .any(|&id| matches!(module.node(id), Node::Axis(_)))
        {
            let shape = tensor::real_shape(module, *array)?;
            if shape.len() != selectors.len() || shape.iter().any(|d| *d != Dim::Static(extent)) {
                return None;
            }
            let axes = selectors
                .iter()
                .map(|&id| read_axis(module, id))
                .collect::<Option<Vec<_>>>()?;
            if axes.len() > 2
                && axes
                    .iter()
                    .map(|axis| axis.name)
                    .collect::<HashSet<_>>()
                    .len()
                    != axes.len()
            {
                // A diagonal over two positions of a higher-rank tensor
                // requires a scatter-compatible view.
                return None;
            }
            sites.push(Site {
                node,
                array: *array,
                axes,
            });
        } else {
            if matches!(module.node(node), Node::Axis(_))
                || ["aggregate", "metricsum"]
                    .iter()
                    .any(|head| expect_builtin_call(module, node, head).is_some())
            {
                return None;
            }
            pending.extend(module.node(node).children());
        }
    }
    Some(sites)
}

fn valid_variances(output: &[Axis], sites: &[Site]) -> bool {
    let mut occurrences: HashMap<Symbol, Vec<_>> = HashMap::new();
    for selector in sites.iter().flat_map(|site| &site.axes) {
        occurrences
            .entry(selector.name)
            .or_default()
            .push(selector.variance);
    }
    output
        .iter()
        .all(|axis| occurrences.contains_key(&axis.name))
        && occurrences.into_iter().all(|(name, variances)| {
            if let Some(axis) = output.iter().find(|axis| axis.name == name) {
                variances.iter().all(|v| *v == axis.variance)
            } else {
                matches!(variances.as_slice(), [_])
                    || matches!(variances.as_slice(), [first, second] if first != second)
            }
        })
}

fn read_axis(module: &Module, node: NodeId) -> Option<Axis> {
    match module.node(node) {
        Node::Axis(axis) if axis.variance.is_some() => Some(*axis),
        _ => None,
    }
}

fn plain_axis(module: &mut Module, original: Axis) -> NodeId {
    module.alloc(Node::Axis(Axis {
        variance: None,
        ..original
    }))
}

fn transform_axis(
    module: &mut Module,
    sum: NodeId,
    value: NodeId,
    matrix: NodeId,
    rank: usize,
    position: usize,
) -> NodeId {
    match (rank, position) {
        (1 | 2, 0) => build_call(module, "mul", &[matrix, value]),
        (2, 1) => {
            let matrix = build_call(module, "transpose", &[matrix]);
            build_call(module, "mul", &[value, matrix])
        }
        _ => {
            let axis_ids: Vec<_> = (0..rank)
                .map(|i| axis(module, &format!("admetric{i}")))
                .collect();
            let contracted = axis(module, "admetriccontracted");
            let weight = indexed(module, matrix, &[axis_ids[position], contracted]);
            let mut selectors = axis_ids.clone();
            selectors[position] = contracted;
            let cell = indexed(module, value, &selectors);
            let body = build_call(module, "mul", &[weight, cell]);
            aggregate(module, sum, &axis_ids, body)
        }
    }
}

fn substitute(
    module: &mut Module,
    node: NodeId,
    replacements: &mut HashMap<NodeId, NodeId>,
    depth: usize,
) -> Option<NodeId> {
    if depth > DEFAULT_MAX_DEPTH {
        return None;
    }
    if let Some(&replacement) = replacements.get(&node) {
        return Some(replacement);
    }
    let children = module.node(node).children();
    let rewritten: Vec<_> = children
        .iter()
        .map(|&child| substitute(module, child, replacements, depth + 1))
        .collect::<Option<_>>()?;
    let result = if children == rewritten {
        node
    } else {
        rebuild_with_children(module, node, &rewritten)
    };
    replacements.insert(node, result);
    Some(result)
}

fn has_local_reference(module: &Module, root: NodeId) -> bool {
    let mut visited = HashSet::new();
    let mut pending = vec![root];
    while let Some(id) = pending.pop() {
        if !visited.insert(id) {
            continue;
        }
        if matches!(module.node(id), Node::Ref(reference) if reference.ns == RefNs::Local) {
            return true;
        }
        pending.extend(module.node(id).children());
    }
    false
}
