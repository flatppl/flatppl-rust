//! Lift derivative templates over outer broadcast frames.

use flatppl_core::graph::{build_call, expect_builtin_call, fold_add};
use flatppl_core::{Dim, Module, Node, NodeId, Type};

use crate::{Emitter, Error, Key, Mode, Query, Step, callbacks, contract, real, rules, tensor};

pub(super) fn rule(name: &str) -> Option<(&'static str, &'static [&'static str])> {
    Some(match name {
        "NormalGLM" => ("Normal GLM log density", &["x", "alpha", "beta", "sigma"]),
        "CategoricalLogitGLM" => ("Categorical GLM log density", &["x", "alpha", "beta"]),
        "OrderedLogitGLM" => ("Ordered GLM log density", &["x", "beta", "cutpoints"]),
        "BernoulliLogitGLM" => ("Bernoulli GLM log density", &["x", "alpha", "beta"]),
        "BinomialLogitGLM" => ("Binomial GLM log density", &["x", "alpha", "beta", "n"]),
        "PoissonLogGLM" => ("Poisson GLM log density", &["x", "alpha", "beta"]),
        "NegativeBinomial2LogGLM" => (
            "NegativeBinomial2 GLM log density",
            &["x", "alpha", "beta", "psi"],
        ),
        "StudentTGLM" => (
            "StudentT GLM log density",
            &["x", "alpha", "beta", "sigma", "nu"],
        ),
        "GammaLogGLM" => ("Gamma GLM log density", &["x", "alpha", "beta", "shape"]),
        "BetaLogitGLM" => ("Beta GLM log density", &["x", "alpha", "beta", "precision"]),
        _ => return None,
    })
}

fn is_const(module: &Module, id: NodeId, name: &str) -> bool {
    matches!(module.node(id), Node::Const(symbol) if module.resolve(*symbol) == name)
}

pub(super) fn operands(
    module: &Module,
    query: &Query<'_>,
    node: NodeId,
) -> Option<(&'static str, Vec<NodeId>)> {
    if let Some(operands) = structural_operands(module, query, node) {
        return Some(operands);
    }
    let (tag, fields, observation) = if let Some(mapped) = callbacks::density_operands(module, node)
    {
        (mapped.tag, mapped.fields, mapped.observation)
    } else {
        let call = expect_builtin_call(module, node, "broadcast")?;
        let [head, tag, params, observation] = *call.args else {
            return None;
        };
        if !call.named.is_empty() || !is_const(module, head, "builtin_logdensityof") {
            return None;
        }
        let Key::Node(params) = query.key(module, params).ok()? else {
            return None;
        };
        let params = expect_builtin_call(module, params, "broadcast")?;
        let [record] = *params.args else {
            return None;
        };
        if !is_const(module, record, "record") {
            return None;
        }
        (
            tag,
            params
                .named
                .iter()
                .map(|field| (field.name, field.value))
                .collect(),
            observation,
        )
    };
    let Node::Const(family) = module.node(tag) else {
        return None;
    };
    let (op, names) = rule(module.resolve(*family))?;
    let mut values = names
        .iter()
        .map(|name| {
            fields
                .iter()
                .find_map(|&(key, value)| (module.resolve(key) == *name).then_some(value))
        })
        .collect::<Option<Vec<_>>>()?;
    values.push(observation);
    Some((op, values))
}

fn structural_operands(
    module: &Module,
    query: &Query<'_>,
    node: NodeId,
) -> Option<(&'static str, Vec<NodeId>)> {
    let call = expect_builtin_call(module, node, "broadcast")?;
    let (&head, positional) = call.args.split_first()?;
    let Key::Node(head) = query.key(module, head).ok()? else {
        return None;
    };
    let Node::Const(name) = module.node(head) else {
        return None;
    };
    let (op, names) = match module.resolve(*name) {
        "conv" => ("conv", &["v", "kernel"][..]),
        "crosscorr" => ("crosscorr", &["v", "kernel"][..]),
        "cross" => ("cross", &["a", "b"][..]),
        "row_gram" => ("row_gram", &["A"][..]),
        "col_gram" => ("col_gram", &["A"][..]),
        "self_outer" => ("self_outer", &["x"][..]),
        "quadform" => ("quadform", &["A", "x"][..]),
        "softmax" => ("softmax", &["x"][..]),
        "logsoftmax" => ("logsoftmax", &["x"][..]),
        "l1unit" => ("l1unit", &["x"][..]),
        "l2unit" => ("l2unit", &["x"][..]),
        "sum" => ("sum", &["x"][..]),
        "l2norm" => ("l2norm", &["x"][..]),
        _ => return None,
    };
    if positional.len() + call.named.len() != names.len() {
        return None;
    }
    let Type::Array { shape, elem } = module.type_of(node)? else {
        return None;
    };
    if !shape
        .iter()
        .all(|dim| matches!(dim, Dim::Static(n) if *n > 0))
        || tensor::real_shape_of(elem).is_none()
    {
        return None;
    }
    let args = names
        .iter()
        .enumerate()
        .map(|(slot, name)| {
            positional.get(slot).copied().or_else(|| {
                call.named
                    .iter()
                    .find_map(|arg| (module.resolve(arg.name) == *name).then_some(arg.value))
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some((op, args))
}

fn frame(ty: &Type) -> Option<&[Dim]> {
    match ty {
        Type::Array { shape, .. } => Some(shape),
        _ => None,
    }
}

fn cell_type(ty: &Type) -> &Type {
    match ty {
        Type::Array { elem, .. } => elem,
        _ => ty,
    }
}

pub(super) fn pushforward<'a>(
    emitter: &mut Emitter,
    rules: &rules::Rules<'a>,
    step: &Step<'a>,
    tangents: &std::collections::HashMap<Key, NodeId>,
) -> Result<NodeId, Error> {
    let seeds: Vec<_> = step
        .targets
        .iter()
        .enumerate()
        .filter_map(|(slot, target)| {
            target
                .and_then(|target| tangents.get(&target))
                .map(|&seed| (slot, seed))
        })
        .collect();
    mapped(emitter, rules, step, Mode::Jvp, &seeds, false)
}

pub(super) fn pullback<'a>(
    emitter: &mut Emitter,
    rules: &rules::Rules<'a>,
    step: &Step<'a>,
    slot: usize,
    bar: NodeId,
    masked: bool,
) -> Result<NodeId, Error> {
    let answer = mapped(emitter, rules, step, Mode::Vjp, &[(slot, bar)], masked)?;
    let target = step.args[slot];
    let fail = || {
        Error::new(
            Some(step.node),
            "cannot sum the repeated broadcast cotangent cells",
        )
    };
    let ty = emitter.module.type_of(target).ok_or_else(fail)?.clone();
    let frame = emitter
        .module
        .type_of(step.node)
        .and_then(frame)
        .ok_or_else(fail)?
        .to_vec();
    let Type::Array { shape, elem } = &ty else {
        return Ok(build_call(&mut emitter.module, "sum", &[answer]));
    };
    if **shape == frame {
        return Ok(answer);
    }
    if **elem == crate::REAL {
        return tensor::sum_to_shape(emitter, answer, step.node, target);
    }
    let cell = tensor::real_shape_of(elem).ok_or_else(fail)?;
    reduce_array_cells(&mut emitter.module, answer, &frame, shape, cell.len()).ok_or_else(fail)
}

// Retain the cell axes during reduction, then restore the outer collection
// layer, including singleton axes repeated by broadcast.
fn reduce_array_cells(
    module: &mut Module,
    answer: NodeId,
    frame: &[Dim],
    shape: &[Dim],
    cell_rank: usize,
) -> Option<NodeId> {
    let (kept, dims) = kept_axes(frame, shape)?;
    let axes: Vec<_> = (0..frame.len() + cell_rank)
        .map(|axis| contract::axis(module, &format!("cell_ad_{axis}")))
        .collect();
    let value = contract::indexed(module, answer, &axes);
    let output_axes: Vec<_> = kept
        .iter()
        .map(|&i| axes[i])
        .chain(axes[frame.len()..].iter().copied())
        .collect();
    let sum = module.intern("sum");
    let sum = module.alloc(Node::Const(sum));
    let reduced = contract::aggregate(module, sum, &output_axes, value);
    let rank = dims.len();
    let one = tensor::int(module, 1);
    if kept.is_empty() {
        let size = if rank == 1 {
            one
        } else {
            build_call(module, "vector", &vec![one; rank])
        };
        return Some(build_call(module, "fill", &[reduced, size]));
    }

    let total = dims
        .iter()
        .try_fold(1_i64, |total, &dim| total.checked_mul(dim))?;
    let total = tensor::int(module, total);
    let flat = build_call(module, "zeros", &[total]);
    let flat = build_call(module, "indicesof0", &[flat]);
    let size: Vec<_> = dims.iter().map(|&dim| tensor::int(module, dim)).collect();
    let size = build_call(module, "vector", &size);
    let order: Vec<_> = (1..=rank).map(|i| tensor::int(module, i as i64)).collect();
    let order = build_call(module, "vector", &order);
    let mut captures = Vec::new();
    let values = callbacks::callback_input(module, "reduced", &mut captures);
    let mut selectors = Vec::with_capacity(kept.len());
    let mut coordinates = Vec::with_capacity(kept.len());
    for i in kept {
        let stride: i64 = dims[i + 1..].iter().product();
        let stride = tensor::int(module, stride);
        let dim = tensor::int(module, dims[i]);
        let positions = tensor::broadcast(module, "div", &[flat, stride]);
        let positions = tensor::broadcast(module, "mod", &[positions, dim]);
        let positions = tensor::broadcast(module, "add", &[positions, one]);
        let positions = build_call(module, "array", &[positions, size, order]);
        coordinates.push(positions);
        selectors.push(callbacks::callback_input(
            module,
            &format!("axis_{i}"),
            &mut captures,
        ));
    }
    let cell = contract::indexed(module, values, &selectors);
    let function = callbacks::reify(module, cell, captures);
    let singleton = build_call(module, "vector", &vec![one; rank]);
    let held = build_call(module, "fill", &[reduced, singleton]);
    let args: Vec<_> = [function, held].into_iter().chain(coordinates).collect();
    Some(build_call(module, "broadcast", &args))
}

fn kept_axes(frame: &[Dim], shape: &[Dim]) -> Option<(Vec<usize>, Vec<i64>)> {
    if frame.len() != shape.len() {
        return None;
    }
    let mut kept = Vec::new();
    let mut dims = Vec::with_capacity(shape.len());
    for (i, (&full, &dim)) in frame.iter().zip(shape).enumerate() {
        let (Dim::Static(full), Dim::Static(dim)) = (full, dim) else {
            return None;
        };
        if full == dim && dim != 1 {
            kept.push(i);
        } else if dim != 1 {
            return None;
        }
        dims.push(i64::from(dim));
    }
    Some((kept, dims))
}

fn mapped<'a>(
    emitter: &mut Emitter,
    rules: &rules::Rules<'a>,
    step: &Step<'a>,
    mode: Mode,
    seeds: &[(usize, NodeId)],
    masked: bool,
) -> Result<NodeId, Error> {
    let type_error = || Error::new(Some(step.node), "unknown broadcast operand or result type");
    // GLM rules use only their parameters and observation. Other lifted
    // rules also receive the primal result so Out has the actual cell value.
    let use_result = !step.op.ends_with(" GLM log density");
    let primal_result = use_result.then(|| emitter.primal(step.node));
    let m = &mut emitter.module;
    let result_type = cell_type(m.type_of(step.node).ok_or_else(type_error)?).clone();
    let mut boundary = Vec::new();
    let mut args = Vec::with_capacity(step.args.len());
    for (i, &value) in step.args.iter().enumerate() {
        let ty = cell_type(m.type_of(value).ok_or_else(type_error)?).clone();
        let reference = callbacks::callback_input(m, &format!("cell_arg_{i}"), &mut boundary);
        m.set_type(reference, ty);
        args.push(reference);
    }
    let cell_result = if use_result {
        callbacks::callback_input(m, "cell_output", &mut boundary)
    } else {
        real(m, 0.0)
    };
    m.set_type(cell_result, result_type.clone());
    let cell = Step {
        node: cell_result,
        op: step.op,
        args,
        targets: Vec::new(),
        broadcast: false,
    };
    cell.check_types(m)?;
    let mut refs = Vec::with_capacity(seeds.len());
    for &(slot, _) in seeds {
        let seed_type = if mode == Mode::Jvp {
            m.type_of(cell.args[slot]).ok_or_else(type_error)?.clone()
        } else {
            result_type.clone()
        };
        let seed = callbacks::callback_input(m, &format!("cell_seed_{slot}"), &mut boundary);
        m.set_type(seed, seed_type);
        refs.push((slot, seed));
    }
    let inline = std::mem::replace(&mut emitter.inline, true);
    let terms: Result<Vec<_>, Error> = refs
        .into_iter()
        .map(|(slot, seed)| {
            let result = cell
                .rule(rules, mode, slot)?
                .instantiate(emitter, &cell, seed, masked)?;
            if masked {
                Ok(mask_cell(&mut emitter.module, seed, result, &result_type))
            } else {
                Ok(result)
            }
        })
        .collect();
    emitter.inline = inline;
    let m = &mut emitter.module;
    let body = fold_add(m, &terms?);
    let function = callbacks::reify(m, body, boundary);
    let operands: Vec<_> = std::iter::once(function)
        .chain(step.args.iter().copied())
        .chain(primal_result)
        .chain(seeds.iter().map(|&(_, seed)| seed))
        .collect();
    Ok(build_call(m, "broadcast", &operands))
}

fn mask_cell(module: &mut Module, seed: NodeId, result: NodeId, result_type: &Type) -> NodeId {
    let count = if *result_type == crate::REAL {
        seed
    } else {
        let zeros = tensor::broadcast(module, "iszero", &[seed]);
        let nonzeros = tensor::broadcast(module, "lnot", &[zeros]);
        let nonzeros = tensor::broadcast(module, "real", &[nonzeros]);
        build_call(module, "sum", &[nonzeros])
    };
    let inactive = build_call(module, "iszero", &[count]);
    let zero = real(module, 0.0);
    tensor::broadcast(module, "ifelse", &[inactive, zero, result])
}
