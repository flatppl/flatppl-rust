//! Expose the data flow of pure scalar broadcast callbacks to the AD graph.

use std::collections::{HashMap, HashSet};

use flatppl_core::calling::{bind_call_args, reified_inputs};
use flatppl_core::graph::{
    build_call, expect_builtin_call, rebuild_with_children, resolve_ref, resolve_ref_one,
};
use flatppl_core::{
    BindingId, Call, CallHead, DEFAULT_MAX_DEPTH, Dim, Idx, Inputs, Module, NamedArg, NamedKind,
    Node, NodeId, Ref, RefNs, Scalar, ScalarType, Symbol, Type,
};

use crate::{Query, tensor};

pub(super) fn legalize(module: &Module, query: &Query<'_>) -> Option<Module> {
    // Avoid cloning the arena for queries with no callable broadcasts.
    if !(0..module.node_count()).any(|index| {
        expect_builtin_call(module, NodeId::from_usize(index), "broadcast")
            .and_then(|call| call.args.first())
            .is_some_and(|&head| {
                let head = resolve_ref(module, head).0;
                expect_builtin_call(module, head, "functionof").is_some()
                    || matches!(module.node(head), Node::Const(op) if cumulative(module.resolve(*op)))
            })
    }) {
        return None;
    }
    let mut module = module.clone();
    let mut walker = Walker {
        module: &mut module,
        visited: HashSet::new(),
        rewritten: HashMap::new(),
        inputs: &query.inputs,
        changed: false,
    };
    walker.binding(query.output, 0);
    walker.changed.then_some(module)
}

struct Walker<'a> {
    module: &'a mut Module,
    visited: HashSet<BindingId>,
    rewritten: HashMap<NodeId, NodeId>,
    inputs: &'a HashSet<BindingId>,
    changed: bool,
}

impl Walker<'_> {
    fn binding(&mut self, binding: BindingId, depth: usize) {
        if !self.inputs.contains(&binding) && self.visited.insert(binding) {
            let root = self.module.binding(binding).rhs;
            let replacement = self.expression(root, depth);
            if replacement != root {
                self.module.set_binding_rhs(binding, replacement);
                self.changed = true;
            }
        }
    }

    fn expression(&mut self, node: NodeId, depth: usize) -> NodeId {
        if depth > DEFAULT_MAX_DEPTH {
            return node;
        }
        if let Some(&replacement) = self.rewritten.get(&node) {
            return replacement;
        }
        let replacement = match self.module.node(node) {
            Node::Ref(_) => {
                if let (_, Some(binding)) = resolve_ref_one(self.module, node) {
                    self.binding(binding, depth + 1);
                }
                node
            }
            // Callable bodies are in a different scope and can be invoked from
            // several sites. Only specialize a body at its application.
            Node::Call(call) if call.inputs.is_none() => {
                let children = self.module.node(node).children();
                let rewritten: Vec<_> = children
                    .iter()
                    .map(|&id| self.expression(id, depth + 1))
                    .collect();
                let current = if rewritten == children {
                    node
                } else {
                    let rebuilt = rebuild_with_children(self.module, node, &rewritten);
                    if let Some(ty) = self.module.type_of(node).cloned() {
                        self.module.set_type(rebuilt, ty);
                    }
                    rebuilt
                };
                lift_broadcast(self.module, self.inputs, current)
                    .or_else(|| expand_cumulative(self.module, self.inputs, current))
                    .unwrap_or(current)
            }
            _ => node,
        };
        self.rewritten.insert(node, replacement);
        replacement
    }
}

fn lift_broadcast(
    module: &mut Module,
    inputs: &HashSet<BindingId>,
    node: NodeId,
) -> Option<NodeId> {
    let call = expect_builtin_call(module, node, "broadcast")?.clone();
    let (&head, positional) = call.args.split_first()?;
    let head = resolve_unpromoted(module, inputs, head)?;
    let function = expect_builtin_call(module, head, "functionof")?;
    let [body] = *function.args else { return None };
    let boundary = function.inputs.as_ref()?;
    let entries = reified_inputs(module, head, boundary)?.to_vec();
    let keyword_only = matches!(boundary, Inputs::Auto);
    let expected = module.type_of(node)?.clone();
    let Type::Array { shape, elem } = &expected else {
        return None;
    };
    if !scalar(elem) || shape.iter().any(|dim| matches!(dim, Dim::Static(0))) {
        return None;
    }
    let names: Vec<_> = entries.iter().map(|&(label, _)| label).collect();
    let operands = bind_call_args(module, &names, positional, &call.named, keyword_only)?;
    let mut bound = Vec::with_capacity(entries.len());
    for ((_, reference), value) in entries.into_iter().zip(operands) {
        let ty = module.type_of(value)?;
        let shape = match ty {
            Type::Array { shape, elem } if scalar(elem) => shape.to_vec(),
            ty if scalar(ty) => Vec::new(),
            _ => return None,
        };
        bound.push((reference, Cell { value, shape }));
    }
    let mut scope = Scope {
        module,
        inputs,
        bound,
        dependencies: HashMap::new(),
        lifted: HashMap::new(),
    };
    scope.depends(body, 0)?;
    let result = scope.expression(body)?;
    let result = expand(scope.module, result, shape, elem)?;
    annotate_equivalent(scope.module, result.value, &expected)
}

fn annotate_equivalent(module: &mut Module, result: NodeId, expected: &Type) -> Option<NodeId> {
    let (types, diagnostics) = flatppl_infer::infer_expression(module, result, &[]);
    if types.get(&result) != Some(expected)
        || diagnostics
            .iter()
            .any(|diag| diag.severity == flatppl_infer::Severity::Error)
    {
        return None;
    }
    for (node, ty) in types {
        if module.type_of(node).is_none() {
            module.set_type(node, ty);
        }
    }
    Some(result)
}

fn resolve_unpromoted(
    module: &Module,
    inputs: &HashSet<BindingId>,
    mut node: NodeId,
) -> Option<NodeId> {
    for _ in 0..=DEFAULT_MAX_DEPTH {
        let (rhs, Some(binding)) = resolve_ref_one(module, node) else {
            return Some(node);
        };
        if inputs.contains(&binding) {
            return None;
        }
        node = rhs;
    }
    None
}

fn scalar(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Scalar(ScalarType::Real | ScalarType::Integer | ScalarType::Boolean)
    )
}

#[derive(Clone)]
struct Cell {
    value: NodeId,
    shape: Vec<Dim>,
}

impl Cell {
    fn scalar(value: NodeId) -> Self {
        Self {
            value,
            shape: Vec::new(),
        }
    }
}

struct Scope<'a> {
    module: &'a mut Module,
    inputs: &'a HashSet<BindingId>,
    bound: Vec<(Ref, Cell)>,
    dependencies: HashMap<NodeId, Option<bool>>,
    lifted: HashMap<NodeId, Cell>,
}

impl Scope<'_> {
    fn target(&self, reference: Ref) -> Option<Cell> {
        self.bound
            .iter()
            .find_map(|(target, value)| (*target == reference).then(|| value.clone()))
    }

    fn rhs(&self, reference: Ref) -> Option<NodeId> {
        if reference.ns != RefNs::SelfMod {
            return None;
        }
        let binding = self.module.binding_by_name(reference.name)?;
        (!self.inputs.contains(&binding)).then(|| self.module.binding(binding).rhs)
    }

    // Cut before following each reference. In particular, a runtime input's
    // original definition is not part of its callback value.
    fn depends(&mut self, node: NodeId, depth: usize) -> Option<bool> {
        if let Some(&depends) = self.dependencies.get(&node) {
            return depends;
        }
        if depth > DEFAULT_MAX_DEPTH {
            return None;
        }
        // An unfinished entry also detects cycles through same-module bindings.
        self.dependencies.insert(node, None);
        let depends = match self.module.node(node) {
            Node::Ref(reference) if self.target(*reference).is_some() => true,
            Node::Ref(reference) if reference.ns != RefNs::Local => self
                .rhs(*reference)
                .map_or(Some(false), |rhs| self.depends(rhs, depth + 1))?,
            Node::Ref(_) => return None,
            Node::Call(call)
                if call.inputs.is_some()
                    || matches!(call.head, CallHead::Builtin(op) if self.module.resolve(op) == "builtin_sample") =>
            {
                return None;
            }
            node => {
                let children = node.children();
                let mut depends = false;
                for child in children {
                    depends |= self.depends(child, depth + 1)?;
                }
                depends
            }
        };
        self.dependencies.insert(node, Some(depends));
        Some(depends)
    }

    fn expression(&mut self, node: NodeId) -> Option<Cell> {
        if let Some(cell) = self.lifted.get(&node) {
            return Some(cell.clone());
        }
        let cell = if !self.depends(node, 0)? && self.module.type_of(node).is_some_and(scalar) {
            Cell::scalar(node)
        } else {
            match self.module.node(node) {
                Node::Ref(reference) => {
                    let reference = *reference;
                    self.target(reference)
                        .or_else(|| self.expression(self.rhs(reference)?))?
                }
                Node::Call(call) => self.call(call.clone())?,
                _ => return None,
            }
        };
        self.lifted.insert(node, cell.clone());
        Some(cell)
    }

    fn instantiate(&mut self, node: NodeId) -> Option<NodeId> {
        if !self.depends(node, 0)? {
            return Some(node);
        }
        if let Some(cell) = self.lifted.get(&node) {
            return Some(cell.value);
        }
        let value = match self.module.node(node) {
            Node::Ref(reference) => {
                let reference = *reference;
                if let Some(target) = self.target(reference) {
                    target.value
                } else {
                    self.instantiate(self.rhs(reference)?)?
                }
            }
            Node::Call(_) => {
                let children = self
                    .module
                    .node(node)
                    .children()
                    .into_iter()
                    .map(|child| self.instantiate(child))
                    .collect::<Option<Vec<_>>>()?;
                rebuild_with_children(self.module, node, &children)
            }
            _ => return None,
        };
        self.lifted.insert(node, Cell::scalar(value));
        Some(value)
    }

    fn call(&mut self, call: Call) -> Option<Cell> {
        let CallHead::Builtin(op) = call.head else {
            return None;
        };
        let name = self.module.resolve(op).to_owned();
        if name == "builtin_logdensityof" && call.named.is_empty() {
            return self.density(&call.args);
        }
        if !call.named.is_empty() {
            return None;
        }
        match name.as_str() {
            "get" => return self.projection(&call.args),
            "in" => return self.membership(&call.args),
            "logsumexp" => return self.logsumexp(&call.args),
            _ => {}
        }
        if !pointwise(&name) {
            return None;
        }
        let cells: Vec<_> = call
            .args
            .iter()
            .map(|&arg| self.expression(arg))
            .collect::<Option<_>>()?;
        let shape = joined_shape(&cells)?;
        let operands: Vec<_> = cells.into_iter().map(|cell| cell.value).collect();
        let value = if shape.is_empty() {
            build_call(self.module, &name, &operands)
        } else {
            tensor::broadcast(self.module, &name, &operands)
        };
        Some(Cell { value, shape })
    }

    fn resolve_structure(&self, mut node: NodeId) -> NodeId {
        for _ in 0..=DEFAULT_MAX_DEPTH {
            let Node::Ref(reference) = self.module.node(node) else {
                break;
            };
            if self.target(*reference).is_some() {
                break;
            }
            let Some(rhs) = self.rhs(*reference) else {
                break;
            };
            node = rhs;
        }
        node
    }

    fn membership(&mut self, args: &[NodeId]) -> Option<Cell> {
        let [point, set] = *args else {
            return None;
        };
        if self.depends(set, 0)? {
            return None;
        }
        let point = self.expression(point)?;
        let m = &mut *self.module;
        let value = if point.shape.is_empty() {
            build_call(m, "in", &[point.value, set])
        } else {
            // Keep the set in the callable scope. It has no numeric
            // representation and is not itself a mapped argument.
            let mut entries = Vec::new();
            let argument = callback_input(m, "point", &mut entries);
            let body = build_call(m, "in", &[argument, set]);
            let function = reify(m, body, entries);
            build_call(m, "broadcast", &[function, point.value])
        };
        Some(Cell {
            value,
            shape: point.shape,
        })
    }

    fn projection(&mut self, args: &[NodeId]) -> Option<Cell> {
        let [structure, key] = *args else {
            return None;
        };
        let structure = self.resolve_structure(structure);
        let key = self.resolve_structure(key);
        let value = match self.module.node(key) {
            Node::Lit(Scalar::Str(field)) => {
                expect_builtin_call(self.module, structure, "record")?
                    .named
                    .iter()
                    .find(|arg| self.module.resolve(arg.name) == field.as_ref())?
                    .value
            }
            Node::Lit(Scalar::Int(index)) => {
                let index = usize::try_from(*index).ok()?.checked_sub(1)?;
                let items = expect_builtin_call(self.module, structure, "tuple")
                    .or_else(|| expect_builtin_call(self.module, structure, "vector"))?;
                *items.args.get(index)?
            }
            _ => return None,
        };
        self.expression(value)
    }

    fn logsumexp(&mut self, args: &[NodeId]) -> Option<Cell> {
        let [values] = *args else {
            return None;
        };
        let values = self.resolve_structure(values);
        let terms = expect_builtin_call(self.module, values, "vector")?
            .args
            .clone();
        let (&first, rest) = terms.split_first()?;
        let mut result = self.expression(first)?;
        for &term in rest {
            let next = self.expression(term)?;
            let shape = join(&result.shape, &next.shape)?;
            let call = if shape.is_empty() {
                build_call
            } else {
                tensor::broadcast
            };
            let m = &mut *self.module;
            let a = result.value;
            let b = next.value;
            let left = call(m, "gt", &[a, b]);
            let high = call(m, "ifelse", &[left, a, b]);
            let low = call(m, "ifelse", &[left, b, a]);
            // An infinite maximum already determines the result unless the
            // other term is NaN. Finite ties keep both derivative paths.
            let infinite = call(m, "isinf", &[high]);
            let invalid = call(m, "isnan", &[low]);
            let valid = call(m, "lnot", &[invalid]);
            let saturated = call(m, "land", &[infinite, valid]);
            let zero = m.alloc(Node::Lit(Scalar::Real(0.0)));
            let gap = call(m, "sub", &[low, high]);
            let gap = call(m, "ifelse", &[saturated, zero, gap]);
            let relative = call(m, "exp", &[gap]);
            let correction = call(m, "log1p", &[relative]);
            let value = call(m, "add", &[high, correction]);
            result = Cell { value, shape };
        }
        Some(result)
    }

    fn density(&mut self, args: &[NodeId]) -> Option<Cell> {
        let [tag, params, observation] = *args else {
            return None;
        };
        let Node::Const(_) = self.module.node(tag) else {
            return None;
        };
        let params = self.resolve_structure(params);
        let mut record = expect_builtin_call(self.module, params, "record")?.clone();
        if !record.args.is_empty()
            || record
                .named
                .iter()
                .any(|field| field.kind != NamedKind::Field)
        {
            return None;
        }
        let observation = self.expression(observation)?;
        let cells: Vec<_> = record
            .named
            .iter()
            .map(|f| self.expression(f.value))
            .collect::<Option<_>>()?;
        let shape = join(&joined_shape(&cells)?, &observation.shape)?;
        let value = if shape.is_empty() {
            for (field, cell) in record.named.iter_mut().zip(cells) {
                field.value = cell.value;
            }
            let params = self.module.alloc(Node::Call(record));
            build_call(
                self.module,
                "builtin_logdensityof",
                &[tag, params, observation.value],
            )
        } else {
            let operands = record
                .named
                .iter()
                .zip(cells)
                .map(|(field, cell)| (field.name, cell.value))
                .collect();
            broadcast_density(self.module, tag, operands, observation.value)
        };
        Some(Cell { value, shape })
    }
}

/// A scalar density can map over more than one axis. FlatPPL tables have one
/// row axis, so keep the per-cell parameter record inside the reified body.
fn broadcast_density(
    module: &mut Module,
    tag: NodeId,
    fields: Vec<(Symbol, NodeId)>,
    observation: NodeId,
) -> NodeId {
    let mut arguments = Vec::with_capacity(fields.len() + 2);
    let mut entries = Vec::with_capacity(fields.len() + 1);
    let mut parameters = Vec::with_capacity(fields.len());
    for (name, argument) in fields {
        let label = format!("parameter_{}", module.resolve(name));
        let reference = callback_input(module, &label, &mut entries);
        parameters.push(NamedArg {
            kind: NamedKind::Field,
            name,
            value: reference,
        });
        arguments.push(argument);
    }
    let record = module.intern("record");
    let params = module.alloc(Node::Call(Call {
        head: CallHead::Builtin(record),
        args: Box::new([]),
        named: parameters.into(),
        inputs: None,
    }));
    let observation_ref = callback_input(module, "observation", &mut entries);
    let body = build_call(
        module,
        "builtin_logdensityof",
        &[tag, params, observation_ref],
    );
    let function = reify(module, body, entries);
    arguments.insert(0, function);
    arguments.push(observation);
    build_call(module, "broadcast", &arguments)
}

fn reify(module: &mut Module, body: NodeId, entries: Vec<(Symbol, Ref)>) -> NodeId {
    let functionof = module.intern("functionof");
    module.alloc(Node::Call(Call {
        head: CallHead::Builtin(functionof),
        args: Box::new([body]),
        named: Box::new([]),
        inputs: Some(Inputs::Spec(entries.into())),
    }))
}

fn callback_input(module: &mut Module, label: &str, entries: &mut Vec<(Symbol, Ref)>) -> NodeId {
    let target = Ref {
        ns: RefNs::Local,
        name: module.intern(&format!("_{label}_")),
    };
    entries.push((module.intern(label), target));
    module.alloc(Node::Ref(target))
}

pub(super) struct DensityOperands {
    pub tag: NodeId,
    pub fields: Vec<(Symbol, NodeId)>,
    pub observation: NodeId,
}

/// Match a scalar density body which uses only its explicit input leaves.
/// Fields and the observation are given as the actual outer broadcast operands.
pub(super) fn density_operands(module: &Module, node: NodeId) -> Option<DensityOperands> {
    let call = expect_builtin_call(module, node, "broadcast")?;
    if !call.named.is_empty() {
        return None;
    }
    let (&head, args) = call.args.split_first()?;
    let function = expect_builtin_call(module, head, "functionof")?;
    let Some(Inputs::Spec(entries)) = function.inputs.as_ref() else {
        return None;
    };
    if args.len() != entries.len() {
        return None;
    }
    let [body] = *function.args else {
        return None;
    };
    let density = expect_builtin_call(module, body, "builtin_logdensityof")?;
    let [tag, params, observation] = *density.args else {
        return None;
    };
    let record = expect_builtin_call(module, params, "record")?;
    if !density.named.is_empty()
        || !record.args.is_empty()
        || record
            .named
            .iter()
            .any(|field| field.kind != NamedKind::Field)
    {
        return None;
    }
    let operand = |target| {
        let Node::Ref(reference) = module.node(target) else {
            return None;
        };
        if reference.ns != RefNs::Local {
            return None;
        }
        entries
            .iter()
            .zip(args)
            .find_map(|((_, boundary), &arg)| (boundary == reference).then_some(arg))
    };
    let fields = record
        .named
        .iter()
        .map(|field| Some((field.name, operand(field.value)?)))
        .collect::<Option<_>>()?;
    Some(DensityOperands {
        tag,
        fields,
        observation: operand(observation)?,
    })
}

fn cumulative(name: &str) -> bool {
    matches!(name, "cumsum" | "cummax" | "cummin")
}

/// Specialize small fixed-length vector maps without crossing a reification
/// boundary or assigning one cell's placeholders to another cell.
fn expand_cumulative(
    module: &mut Module,
    inputs: &HashSet<BindingId>,
    node: NodeId,
) -> Option<NodeId> {
    // Explicit cells make one copy of the body per lane. Keep this fallback
    // bounded until the AD graph can describe a vectorized structured map.
    const MAX_EXPLICIT_CELLS: u32 = 32;
    let expected = module.type_of(node)?.clone();
    let (length, _) = tensor::nested_vector_type(&expected)?;
    if length > MAX_EXPLICIT_CELLS {
        return None;
    }
    let call = expect_builtin_call(module, node, "broadcast")?.clone();
    let (&head, args) = call.args.split_first()?;
    let head = resolve_unpromoted(module, inputs, head)?;
    let mut cells = Vec::with_capacity(length as usize);
    if let Node::Const(op) = module.node(head) {
        let op = module.resolve(*op).to_owned();
        let [operand] = *args else {
            return None;
        };
        if !cumulative(&op) || !call.named.is_empty() {
            return None;
        }
        for index in 0..length {
            let cell = pick_cell(module, inputs, operand, index, length)?;
            cells.push(build_call(module, &op, &[cell]));
        }
    } else {
        let function = expect_builtin_call(module, head, "functionof")?;
        let [body] = *function.args else {
            return None;
        };
        let body_op = match module.node(resolve_unpromoted(module, inputs, body)?) {
            Node::Call(call) => match call.head {
                CallHead::Builtin(op) => module.resolve(op),
                CallHead::User(_) => return None,
            },
            _ => return None,
        };
        if !cumulative(body_op) {
            return None;
        }
        let boundary = function.inputs.as_ref()?;
        let entries = reified_inputs(module, head, boundary)?.to_vec();
        let keyword_only = matches!(boundary, Inputs::Auto);
        let names: Vec<_> = entries.iter().map(|&(label, _)| label).collect();
        let operands = bind_call_args(module, &names, args, &call.named, keyword_only)?;
        for index in 0..length {
            let mut bound = Vec::with_capacity(entries.len());
            for ((_, reference), &value) in entries.iter().zip(&operands) {
                let value = pick_cell(module, inputs, value, index, length)?;
                bound.push((*reference, Cell::scalar(value)));
            }
            let mut scope = Scope {
                module,
                inputs,
                bound,
                dependencies: HashMap::new(),
                lifted: HashMap::new(),
            };
            scope.depends(body, 0)?;
            cells.push(scope.instantiate(body)?);
        }
    }
    let result = build_call(module, "vector", &cells);
    annotate_equivalent(module, result, &expected)
}

fn pick_cell(
    module: &mut Module,
    inputs: &HashSet<BindingId>,
    value: NodeId,
    index: u32,
    length: u32,
) -> Option<NodeId> {
    let extent = match module.type_of(value)? {
        Type::Array { shape, .. } => match shape.as_ref() {
            [Dim::Static(extent)] if *extent == length || *extent == 1 => *extent,
            _ => return None,
        },
        ty if scalar(ty) => return Some(value),
        _ => return None,
    };
    let index = if extent == 1 { 0 } else { index };
    if let Some(resolved) = resolve_unpromoted(module, inputs, value)
        && let Some(vector) = expect_builtin_call(module, resolved, "vector")
    {
        return vector.args.get(index as usize).copied();
    }
    let position = tensor::int(module, i64::from(index) + 1);
    Some(build_call(module, "get", &[value, position]))
}

/// Scalar kernels whose operation has the same meaning in base broadcast.
fn pointwise(name: &str) -> bool {
    matches!(
        name,
        "abs"
            | "add"
            | "divide"
            | "gt"
            | "ge"
            | "lt"
            | "le"
            | "unequal"
            | "inv"
            | "isnan"
            | "isinf"
            | "atan"
            | "ceil"
            | "cos"
            | "div"
            | "equal"
            | "exp"
            | "expm1"
            | "floor"
            | "ifelse"
            | "invlogit"
            | "isfinite"
            | "iszero"
            | "land"
            | "lnot"
            | "log"
            | "log1p"
            | "logit"
            | "lor"
            | "max"
            | "min"
            | "mul"
            | "neg"
            | "pow"
            | "reciprocal"
            | "sin"
            | "sqrt"
            | "sub"
            | "tan"
    )
}

fn joined_shape(cells: &[Cell]) -> Option<Vec<Dim>> {
    cells
        .iter()
        .try_fold(Vec::new(), |shape, cell| join(&shape, &cell.shape))
}

fn join(left: &[Dim], right: &[Dim]) -> Option<Vec<Dim>> {
    if left.is_empty() || right.is_empty() {
        return Some(if left.is_empty() { right } else { left }.to_vec());
    }
    if left.len() != right.len() {
        return None;
    }
    left.iter()
        .zip(right)
        .map(|(&left, &right)| match (left, right) {
            (left, right) if left == right => Some(left),
            (Dim::Static(1), dim) | (dim, Dim::Static(1)) => Some(dim),
            (Dim::Dynamic, _) | (_, Dim::Dynamic) => Some(Dim::Dynamic),
            _ => None,
        })
        .collect()
}

fn expand(module: &mut Module, cell: Cell, shape: &[Dim], elem: &Type) -> Option<Cell> {
    if cell.shape == shape {
        return Some(cell);
    }
    if join(&cell.shape, shape)? != shape {
        return None;
    }
    let dims: Vec<_> = shape
        .iter()
        .map(|dim| match dim {
            Dim::Static(n) => Some(tensor::int(module, i64::from(*n))),
            Dim::Dynamic => None,
        })
        .collect::<Option<_>>()?;
    let size = build_call(module, "vector", &dims);
    let zero = match elem {
        Type::Scalar(ScalarType::Real) => Scalar::Real(0.0),
        Type::Scalar(ScalarType::Integer) => Scalar::Int(0),
        Type::Scalar(ScalarType::Boolean) => Scalar::Bool(false),
        _ => return None,
    };
    let zero = module.alloc(Node::Lit(zero));
    let frame = build_call(module, "fill", &[zero, size]);
    let yes = module.alloc(Node::Lit(Scalar::Bool(true)));
    let value = tensor::broadcast(module, "ifelse", &[yes, cell.value, frame]);
    Some(Cell {
        value,
        shape: shape.to_vec(),
    })
}
