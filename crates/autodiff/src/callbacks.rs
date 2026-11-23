//! Expose the data flow of pure scalar broadcast callbacks to the AD graph.

use std::collections::{HashMap, HashSet};

use flatppl_core::calling::{bind_call_args, reified_inputs};
use flatppl_core::graph::{
    build_call, expect_builtin_call, rebuild_with_children, resolve_ref, resolve_ref_one,
};
use flatppl_core::{
    Binding, BindingId, Call, CallHead, DEFAULT_MAX_DEPTH, Dim, Idx, Inputs, Module, NamedArg,
    NamedKind, Node, NodeId, Ref, RefNs, Scalar, ScalarType, Symbol, Type,
};

use crate::{Query, metric, recurrence, tensor};

pub(super) fn legalize(module: &Module, query: &Query<'_>) -> Option<Module> {
    // Avoid cloning the arena when no supported expansion occurs.
    if !(0..module.node_count()).any(|index| {
        let id = NodeId::from_usize(index);
        if ["metricsum", "reduce", "scan"]
            .iter()
            .any(|name| expect_builtin_call(module, id, name).is_some())
        {
            return true;
        }
        expect_builtin_call(module, id, "broadcast")
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
                    .or_else(|| expand_static_cells(self.module, self.inputs, current))
                    .or_else(|| metric::lower(self.module, current))
                    .or_else(|| recurrence::lower(self.module, self.inputs, current))
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
    let mut scope = Scope::new(module, inputs);
    for ((_, reference), value) in entries.into_iter().zip(operands) {
        let ty = scope.module.type_of(value)?.clone();
        let mapped = match &ty {
            Type::Array { shape, elem } if scalar(elem) => Operand::Scalar(Cell {
                value,
                shape: shape.to_vec(),
            }),
            Type::Array { shape, elem } => {
                let (layout, width) = event_layout(elem)?;
                let cells = scope.nested_components(value, shape, layout, width)?;
                Operand::Vector(Event { layout, cells })
            }
            ty if scalar(ty) => Operand::Scalar(Cell::scalar(value)),
            _ => return None,
        };
        scope.bound.push((reference, mapped));
    }
    scope.depends(body, 0)?;
    let result = scope.expression(body)?;
    let result = expand(scope.module, result, shape, elem)?;
    let result = annotate_equivalent(scope.module, result.value, &expected)?;
    Some(bind_shared_terms(scope.module, result, &scope.shared))
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

// Expanding an event vector grows the symbolic graph once per event
// component, independent of the number of outer broadcast cells.
const MAX_EXPLICIT_EVENT_WIDTH: u32 = 32;

#[derive(Clone, Copy, PartialEq, Eq)]
enum EventLayout {
    Vector,
    Row,
}

fn event_layout(ty: &Type) -> Option<(EventLayout, u32)> {
    let Type::Array { shape, elem } = ty else {
        return None;
    };
    let (layout, width) = match shape.as_ref() {
        [Dim::Static(width)] => (EventLayout::Vector, *width),
        [Dim::Static(1), Dim::Static(width)] => (EventLayout::Row, *width),
        _ => return None,
    };
    (scalar(elem) && (1..=MAX_EXPLICIT_EVENT_WIDTH).contains(&width)).then_some((layout, width))
}

#[derive(Clone)]
struct Event {
    layout: EventLayout,
    cells: Vec<Cell>,
}

impl Event {
    fn vector(cells: Vec<Cell>) -> Self {
        Self {
            layout: EventLayout::Vector,
            cells,
        }
    }

    fn into_vector(self) -> Option<Vec<Cell>> {
        (self.layout == EventLayout::Vector).then_some(self.cells)
    }
}

#[derive(Clone)]
enum Operand {
    Scalar(Cell),
    Vector(Event),
}

impl Operand {
    fn item(&self, index: usize) -> Option<Cell> {
        match self {
            Self::Scalar(cell) => Some(cell.clone()),
            Self::Vector(event) => event
                .cells
                .get(if event.cells.len() == 1 { 0 } else { index })
                .cloned(),
        }
    }
}

struct Scope<'a> {
    module: &'a mut Module,
    inputs: &'a HashSet<BindingId>,
    bound: Vec<(Ref, Operand)>,
    dependencies: HashMap<NodeId, Option<bool>>,
    lifted: HashMap<NodeId, Cell>,
    vectors: HashMap<NodeId, Event>,
    shared: HashSet<NodeId>,
}

impl Scope<'_> {
    fn new<'a>(module: &'a mut Module, inputs: &'a HashSet<BindingId>) -> Scope<'a> {
        Scope {
            module,
            inputs,
            bound: Vec::new(),
            dependencies: HashMap::new(),
            lifted: HashMap::new(),
            vectors: HashMap::new(),
            shared: HashSet::new(),
        }
    }

    fn target(&self, reference: Ref) -> Option<&Operand> {
        self.bound
            .iter()
            .find_map(|(target, value)| (*target == reference).then_some(value))
    }

    fn nested_components(
        &mut self,
        value: NodeId,
        shape: &[Dim],
        layout: EventLayout,
        width: u32,
    ) -> Option<Vec<Cell>> {
        let m = &mut *self.module;
        if shape == [Dim::Static(1)]
            && let Some(root) = resolve_unpromoted(m, self.inputs, value)
            && let Some(vector) = expect_builtin_call(m, root, "vector")
            && let [whole] = *vector.args
        {
            return Some(event_components(m, whole, layout, width));
        }
        let source = match (layout, shape) {
            (EventLayout::Vector, [Dim::Static(_)]) => build_call(m, "rowstack", &[value]),
            (EventLayout::Row, [Dim::Static(rows), Dim::Static(1)]) => {
                let root = resolve_unpromoted(m, self.inputs, value)?;
                let split = expect_builtin_call(m, root, "splitblocks")?;
                let [source, _] = *split.args else {
                    return None;
                };
                let Type::Array { shape, elem } = m.type_of(source)? else {
                    return None;
                };
                if shape.as_ref() != [Dim::Static(*rows), Dim::Static(width)] || !scalar(elem) {
                    return None;
                }
                source
            }
            _ => return None,
        };
        let all = m.intern("all");
        let all = m.alloc(Node::Const(all));
        let zero = tensor::int(m, 0);
        let one = tensor::int(m, 1);
        Some(
            (1..=width)
                .map(|index| {
                    let index = tensor::int(m, i64::from(index));
                    let column = build_call(m, "get", &[source, all, index]);
                    let value = if layout == EventLayout::Row {
                        build_call(m, "addaxes", &[column, zero, one])
                    } else {
                        column
                    };
                    Cell {
                        value,
                        shape: shape.to_vec(),
                    }
                })
                .collect(),
        )
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
            Node::Call(call) if call.inputs.is_some() => {
                self.check_nested_scope(node, depth + 1)?;
                false
            }
            Node::Call(call) if matches!(call.head, CallHead::Builtin(op) if self.module.resolve(op) == "builtin_sample") =>
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

    fn check_nested_scope(&mut self, node: NodeId, depth: usize) -> Option<()> {
        let callable = expect_builtin_call(self.module, node, "functionof")?;
        let Some(Inputs::Spec(entries)) = callable.inputs.as_ref() else {
            return None;
        };
        let [body] = *callable.args else {
            return None;
        };
        let locals: HashSet<_> = entries.iter().map(|&(_, target)| target.name).collect();
        if entries.iter().any(|&(_, r)| r.ns != RefNs::Local) || locals.len() != entries.len() {
            return None;
        }
        let mut seen = HashSet::new();
        let mut pending = vec![body];
        while let Some(node) = pending.pop() {
            if !seen.insert(node) {
                continue;
            }
            match self.module.node(node) {
                Node::Ref(reference) if reference.ns == RefNs::Local => {
                    if !locals.contains(&reference.name) {
                        return None;
                    }
                }
                Node::Ref(_) => {
                    if self.depends(node, depth)? {
                        return None;
                    }
                }
                Node::Call(call)
                    if call.inputs.is_some()
                        || matches!(call.head, CallHead::Builtin(op) if self.module.resolve(op) == "builtin_sample") =>
                {
                    return None;
                }
                node => node.for_each_child(|child| pending.push(child)),
            }
        }
        Some(())
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
                    match self.target(reference) {
                        Some(Operand::Scalar(cell)) => cell.clone(),
                        Some(Operand::Vector(_)) => return None,
                        None => self.expression(self.rhs(reference)?)?,
                    }
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
                match self.target(reference) {
                    Some(Operand::Scalar(cell)) => cell.value,
                    Some(Operand::Vector(_)) => return None,
                    None => self.instantiate(self.rhs(reference)?)?,
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
            "get" | "get0" => {
                return self.projection(&call.args, i64::from(name == "get"));
            }
            "in" => return self.membership(&call.args),
            "logsumexp" => return self.logsumexp(&call.args),
            "sum" => return self.sum(&call.args),
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
        self.pointwise_call(&name, cells)
    }

    fn pointwise_call(&mut self, name: &str, operands: Vec<Cell>) -> Option<Cell> {
        let shape = joined_shape(&operands)?;
        let args: Vec<_> = operands.into_iter().map(|cell| cell.value).collect();
        let call = if shape.is_empty() {
            build_call
        } else {
            tensor::broadcast
        };
        Some(Cell {
            value: call(self.module, name, &args),
            shape,
        })
    }

    fn operand(&mut self, node: NodeId) -> Option<Operand> {
        self.vector(node)
            .map(Operand::Vector)
            .or_else(|| self.expression(node).map(Operand::Scalar))
    }

    fn vector(&mut self, node: NodeId) -> Option<Event> {
        if let Some(terms) = self.vectors.get(&node) {
            return Some(terms.clone());
        }
        let terms = if !self.depends(node, 0)?
            && let Some((layout, width)) = self.module.type_of(node).and_then(event_layout)
        {
            Event {
                layout,
                cells: event_components(self.module, node, layout, width),
            }
        } else {
            match self.module.node(node) {
                Node::Ref(reference) => {
                    let reference = *reference;
                    match self.target(reference) {
                        Some(Operand::Vector(terms)) => terms.clone(),
                        Some(Operand::Scalar(_)) => return None,
                        None => self.vector(self.rhs(reference)?)?,
                    }
                }
                Node::Call(call) => {
                    let call = call.clone();
                    let CallHead::Builtin(head) = call.head else {
                        return None;
                    };
                    if !call.named.is_empty() {
                        return None;
                    }
                    match self.module.resolve(head) {
                        "vector"
                            if !call.args.is_empty()
                                && call.args.len() <= MAX_EXPLICIT_EVENT_WIDTH as usize =>
                        {
                            let cells = call
                                .args
                                .iter()
                                .map(|&id| self.expression(id))
                                .collect::<Option<_>>()?;
                            Event::vector(cells)
                        }
                        "broadcast" => self.vector_broadcast(&call.args)?,
                        "softmax" => Event::vector(self.softmax(&call.args)?),
                        "get" | "get0" => {
                            let base = i64::from(self.module.resolve(head) == "get");
                            Event::vector(self.vector_get(&call.args, base)?)
                        }
                        "cat" if call.args.len() >= 2 => {
                            let mut cells = Vec::new();
                            for operand in call.args {
                                cells.extend(self.vector(operand)?.into_vector()?);
                                if cells.len() > MAX_EXPLICIT_EVENT_WIDTH as usize {
                                    return None;
                                }
                            }
                            Event::vector(cells)
                        }
                        _ => return None,
                    }
                }
                _ => return None,
            }
        };
        self.vectors.insert(node, terms.clone());
        Some(terms)
    }

    fn softmax(&mut self, args: &[NodeId]) -> Option<Vec<Cell>> {
        let [input] = *args else {
            return None;
        };
        let logits = self.vector(input)?.into_vector()?;
        let logits = logits
            .into_iter()
            .map(|term| self.pointwise_call("real", vec![term]))
            .collect::<Option<Vec<_>>>()?;
        let (first, rest) = logits.split_first()?;
        let mut highest = first.clone();
        for term in rest {
            let better = self.pointwise_call("gt", vec![term.clone(), highest.clone()])?;
            let nan = self.pointwise_call("isnan", vec![term.clone()])?;
            let select = self.pointwise_call("lor", vec![better, nan])?;
            highest = self.pointwise_call("ifelse", vec![select, term.clone(), highest])?;
            self.shared.insert(highest.value);
        }
        let mut weights = Vec::with_capacity(logits.len());
        for term in logits {
            self.shared.insert(term.value);
            let shift = self.pointwise_call("sub", vec![term, highest.clone()])?;
            let weight = self.pointwise_call("exp", vec![shift])?;
            self.shared.insert(weight.value);
            weights.push(weight);
        }
        let (first, rest) = weights.split_first()?;
        let total = rest.iter().try_fold(first.clone(), |sum, term| {
            self.pointwise_call("add", vec![sum, term.clone()])
        })?;
        self.shared.insert(total.value);
        weights
            .into_iter()
            .map(|term| self.pointwise_call("divide", vec![term, total.clone()]))
            .collect()
    }

    fn vector_broadcast(&mut self, args: &[NodeId]) -> Option<Event> {
        let (&head, args) = args.split_first()?;
        let head = self.resolve_structure(head);
        let Node::Const(op) = self.module.node(head) else {
            return self.apply_vector_callable(head, args);
        };
        let name = self.module.resolve(*op).to_owned();
        if name == "builtin_logdensityof" {
            let [tag, params, observation] = *args else {
                return None;
            };
            let params = self.resolve_structure(params);
            let record = expect_builtin_call(self.module, params, "broadcast")?;
            let [record_head] = *record.args else {
                return None;
            };
            if !matches!(self.module.node(record_head), Node::Const(op) if self.module.resolve(*op) == "record")
            {
                return None;
            }
            let fields: Vec<_> = record
                .named
                .iter()
                .map(|arg| (arg.name, arg.value))
                .collect();
            let mut operands: Vec<_> = fields
                .iter()
                .map(|&(_, node)| self.operand(node))
                .collect::<Option<_>>()?;
            operands.push(self.operand(observation)?);
            let (layout, width) = operand_shape(&operands)?;
            let cells = (0..width)
                .map(|index| {
                    let observation = operands.last()?.item(index)?;
                    let fields = fields
                        .iter()
                        .zip(&operands)
                        .map(|(&(name, _), operand)| Some((name, operand.item(index)?)))
                        .collect::<Option<_>>()?;
                    self.score(tag, fields, observation)
                })
                .collect::<Option<_>>()?;
            return Some(Event { layout, cells });
        }
        if !pointwise(&name) {
            return None;
        }
        let operands = args
            .iter()
            .map(|&arg| self.operand(arg))
            .collect::<Option<Vec<_>>>()?;
        let (layout, width) = operand_shape(&operands)?;
        let cells = (0..width)
            .map(|index| {
                let args = operands
                    .iter()
                    .map(|operand| operand.item(index))
                    .collect::<Option<_>>()?;
                self.pointwise_call(&name, args)
            })
            .collect::<Option<_>>()?;
        Some(Event { layout, cells })
    }

    fn apply_vector_callable(&mut self, function: NodeId, args: &[NodeId]) -> Option<Event> {
        self.check_nested_scope(function, 0)?;
        let callable = expect_builtin_call(self.module, function, "functionof")?;
        let Some(Inputs::Spec(entries)) = callable.inputs.as_ref() else {
            return None;
        };
        let body = callable.args[0];
        let entries = entries.clone();
        let names: Vec<_> = entries.iter().map(|&(name, _)| name).collect();
        let args = bind_call_args(self.module, &names, args, &[], false)?;
        let operands: Vec<_> = args
            .into_iter()
            .map(|arg| self.operand(arg))
            .collect::<Option<_>>()?;
        let (layout, width) = operand_shape(&operands)?;
        let mut result = Vec::with_capacity(width);
        for index in 0..width {
            let mut nested = Scope::new(self.module, self.inputs);
            for ((_, target), operand) in entries.iter().zip(&operands) {
                nested
                    .bound
                    .push((*target, Operand::Scalar(operand.item(index)?)));
            }
            result.push(nested.expression(body)?);
            self.shared.extend(nested.shared);
        }
        Some(Event {
            layout,
            cells: result,
        })
    }

    fn vector_get(&mut self, args: &[NodeId], base: i64) -> Option<Vec<Cell>> {
        let [source, selector] = *args else {
            return None;
        };
        let selector = self.resolve_structure(selector);
        let indices = if let Some(vector) = expect_builtin_call(self.module, selector, "vector") {
            vector
                .args
                .iter()
                .map(|&arg| match self.module.node(self.resolve_structure(arg)) {
                    Node::Lit(Scalar::Int(i)) => Some(*i),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?
        } else {
            let [Dim::Static(len)] = tensor::numeric_shape(self.module, selector)? else {
                return None;
            };
            if *len > MAX_EXPLICIT_EVENT_WIDTH {
                return None;
            }
            let first = if expect_builtin_call(self.module, selector, "indicesof").is_some() {
                1
            } else if expect_builtin_call(self.module, selector, "indicesof0").is_some() {
                0
            } else {
                return None;
            };
            (first..first + i64::from(*len)).collect()
        };
        let input = self.vector(source)?.into_vector()?;
        indices
            .into_iter()
            .map(|index| {
                input
                    .get(usize::try_from(index.checked_sub(base)?).ok()?)
                    .cloned()
            })
            .collect()
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
        let point = self.expression(point)?;
        if self.depends(set, 0)? {
            return self.interval_membership(point, set);
        }
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

    fn interval_membership(&mut self, point: Cell, set: NodeId) -> Option<Cell> {
        let set = self.resolve_structure(set);
        let interval = expect_builtin_call(self.module, set, "interval")?;
        let [lower, upper] = *interval.args else {
            return None;
        };
        if !interval.named.is_empty() {
            return None;
        }
        let lower = self.expression(lower)?;
        let upper = self.expression(upper)?;
        let operands = [point, lower, upper];
        let shape = joined_shape(&operands)?;
        let values = operands.map(|cell| cell.value);
        let m = &mut *self.module;
        let value = if shape.is_empty() {
            let interval = build_call(m, "interval", &values[1..]);
            build_call(m, "in", &[values[0], interval])
        } else {
            let mut entries = Vec::new();
            let point = callback_input(m, "point", &mut entries);
            let lower = callback_input(m, "lower", &mut entries);
            let upper = callback_input(m, "upper", &mut entries);
            let interval = build_call(m, "interval", &[lower, upper]);
            let body = build_call(m, "in", &[point, interval]);
            let function = reify(m, body, entries);
            let [point, lower, upper] = values;
            build_call(m, "broadcast", &[function, point, lower, upper])
        };
        Some(Cell { value, shape })
    }

    fn projection(&mut self, args: &[NodeId], base: i64) -> Option<Cell> {
        let [structure, key] = *args else {
            return None;
        };
        let key = self.resolve_structure(key);
        if let Node::Lit(Scalar::Int(index)) = self.module.node(key) {
            let index = usize::try_from(index.checked_sub(base)?).ok()?;
            if let Some(terms) = self.vector(structure) {
                return terms.into_vector()?.get(index).cloned();
            }
        }
        let structure = self.resolve_structure(structure);
        let value = match self.module.node(key) {
            Node::Lit(Scalar::Str(field)) => {
                expect_builtin_call(self.module, structure, "record")?
                    .named
                    .iter()
                    .find(|arg| self.module.resolve(arg.name) == field.as_ref())?
                    .value
            }
            Node::Lit(Scalar::Int(index)) => {
                let index = usize::try_from(index.checked_sub(base)?).ok()?;
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
        let terms = self.vector(values)?.into_vector()?;
        let (first, rest) = terms.split_first()?;
        let mut result = first.clone();
        for next in rest {
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
            self.shared.extend([a, b, left, high, low]);
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

    fn sum(&mut self, args: &[NodeId]) -> Option<Cell> {
        let [values] = *args else {
            return None;
        };
        let mut terms = self.vector(values)?.cells.into_iter();
        let first = terms.next()?;
        terms.try_fold(first, |a, b| self.pointwise_call("add", vec![a, b]))
    }

    fn density(&mut self, args: &[NodeId]) -> Option<Cell> {
        let [tag, params, observation] = *args else {
            return None;
        };
        let params = self.resolve_structure(params);
        let record = expect_builtin_call(self.module, params, "record")?;
        if !record.args.is_empty() || record.named.iter().any(|arg| arg.kind != NamedKind::Field) {
            return None;
        }
        let fields: Vec<_> = record
            .named
            .iter()
            .map(|arg| (arg.name, arg.value))
            .collect();
        let tag_name = match self.module.node(tag) {
            Node::Const(symbol) => self.module.resolve(*symbol),
            _ => return None,
        };
        if matches!(tag_name, "Categorical" | "Categorical0") {
            let base = i64::from(tag_name == "Categorical");
            let [(field, probabilities)] = *fields.as_slice() else {
                return None;
            };
            if self.module.resolve(field) != "p" {
                return None;
            }
            let probabilities = self.vector(probabilities)?.into_vector()?;
            let observation = self.expression(observation)?;
            return self.categorical(observation, probabilities, base);
        }
        if matches!(self.module.node(tag), Node::Const(symbol) if self.module.resolve(*symbol) == "Dirac")
            && let Some(observation) = self.vector(observation)
        {
            let [(field, atom)] = *fields.as_slice() else {
                return None;
            };
            if self.module.resolve(field) != "value" {
                return None;
            }
            let atoms = self.vector(atom)?;
            return self.discrete_dirac(observation, atoms);
        }
        let observation = self.expression(observation)?;
        let fields = fields
            .into_iter()
            .map(|(name, arg)| Some((name, self.expression(arg)?)))
            .collect::<Option<_>>()?;
        self.score(tag, fields, observation)
    }

    fn categorical(
        &mut self,
        observation: Cell,
        probabilities: Vec<Cell>,
        base: i64,
    ) -> Option<Cell> {
        flatppl_infer::annotate_rebuilt(self.module, observation.value);
        let ty = self.module.type_of(observation.value)?;
        let ty = match ty {
            Type::Array { elem, .. } => elem.as_ref(),
            ty => ty,
        };
        if ty != &Type::Scalar(ScalarType::Integer) {
            return None;
        }
        let upper = base + i64::try_from(probabilities.len()).ok()?;
        let lower = Cell::scalar(tensor::int(self.module, base));
        let upper = Cell::scalar(tensor::int(self.module, upper));
        let ge = self.pointwise_call("ge", vec![observation.clone(), lower])?;
        let lt = self.pointwise_call("lt", vec![observation.clone(), upper])?;
        let valid = self.pointwise_call("land", vec![ge, lt])?;
        let one = self.module.alloc(Node::Lit(Scalar::Real(1.0)));
        let mut chosen = Cell::scalar(one);
        for (index, probability) in probabilities.into_iter().enumerate() {
            let index = Cell::scalar(tensor::int(self.module, base + index as i64));
            let matches = self.pointwise_call("equal", vec![observation.clone(), index])?;
            chosen = self.pointwise_call("ifelse", vec![matches, probability, chosen])?;
        }
        let score = self.pointwise_call("log", vec![chosen])?;
        let outside = Cell::scalar(negative_infinity(self.module));
        self.pointwise_call("ifelse", vec![valid, score, outside])
    }

    fn discrete_dirac(&mut self, observation: Event, atoms: Event) -> Option<Cell> {
        if observation.layout != atoms.layout || observation.cells.len() != atoms.cells.len() {
            return None;
        }
        let mut members = Vec::with_capacity(observation.cells.len());
        for (observation, atom) in observation.cells.into_iter().zip(atoms.cells) {
            let mut integers = Vec::with_capacity(2);
            for term in [observation, atom] {
                flatppl_infer::annotate_rebuilt(self.module, term.value);
                let ty = self.module.type_of(term.value)?;
                let elem = match ty {
                    Type::Array { elem, .. } => elem.as_ref(),
                    ty => ty,
                };
                match elem {
                    Type::Scalar(ScalarType::Integer) => integers.push(term),
                    Type::Scalar(ScalarType::Boolean) => {
                        integers.push(self.pointwise_call("integer", vec![term])?);
                    }
                    _ => return None,
                }
            }
            members.push(self.pointwise_call("equal", integers)?);
        }
        let mut members = members.into_iter();
        let first = members.next()?;
        let at_atom = members.try_fold(first, |a, b| self.pointwise_call("land", vec![a, b]))?;
        let m = &mut *self.module;
        let zero = m.alloc(Node::Lit(Scalar::Real(0.0)));
        let neg_inf = negative_infinity(m);
        self.pointwise_call(
            "ifelse",
            vec![at_atom, Cell::scalar(zero), Cell::scalar(neg_inf)],
        )
    }

    fn score(
        &mut self,
        tag: NodeId,
        fields: Vec<(Symbol, Cell)>,
        observation: Cell,
    ) -> Option<Cell> {
        if !matches!(self.module.node(tag), Node::Const(_)) {
            return None;
        }
        let shape = fields
            .iter()
            .try_fold(observation.shape.clone(), |shape, (_, cell)| {
                join(&shape, &cell.shape)
            })?;
        let m = &mut *self.module;
        let fields: Vec<_> = fields
            .into_iter()
            .map(|(name, cell)| (name, cell.value))
            .collect();
        let value = if shape.is_empty() {
            let record = m.intern("record");
            let named = fields
                .into_iter()
                .map(|(name, value)| NamedArg {
                    kind: NamedKind::Field,
                    name,
                    value,
                })
                .collect();
            let params = m.alloc(Node::Call(Call {
                head: CallHead::Builtin(record),
                args: Box::new([]),
                named,
                inputs: None,
            }));
            build_call(m, "builtin_logdensityof", &[tag, params, observation.value])
        } else {
            broadcast_density(m, tag, fields, observation.value)
        };
        Some(Cell { value, shape })
    }
}

// A FlatPPL source expression is a tree. Give the repeated generated
// logsumexp terms names so nested combinations do not reprint their operands.
fn bind_shared_terms(module: &mut Module, root: NodeId, shared: &HashSet<NodeId>) -> NodeId {
    fn walk(
        module: &mut Module,
        shared: &HashSet<NodeId>,
        mapped: &mut HashMap<NodeId, NodeId>,
        original: NodeId,
    ) -> NodeId {
        if let Some(&node) = mapped.get(&original) {
            return node;
        }
        let node = match module.node(original) {
            Node::Call(call) if call.inputs.is_none() => {
                let children = module.node(original).children();
                let replacements: Vec<_> = children
                    .iter()
                    .map(|&child| walk(module, shared, mapped, child))
                    .collect();
                let ty = module.type_of(original).cloned();
                let body = if children == replacements {
                    original
                } else {
                    let rebuilt = rebuild_with_children(module, original, &replacements);
                    if let Some(ty) = ty.as_ref() {
                        module.set_type(rebuilt, ty.clone());
                    }
                    rebuilt
                };
                if shared.contains(&original) {
                    let mut serial = original.index();
                    let name = loop {
                        let name = module.intern(&format!("_ad_callback_value_{serial}"));
                        if module.binding_by_name(name).is_none() {
                            break name;
                        }
                        serial += 1;
                    };
                    module.add_binding(Binding {
                        name,
                        rhs: body,
                        doc: None,
                        public: false,
                        synthetic: false,
                    });
                    let reference = module.alloc(Node::Ref(Ref {
                        ns: RefNs::SelfMod,
                        name,
                    }));
                    if let Some(ty) = ty {
                        module.set_type(reference, ty);
                    }
                    reference
                } else {
                    body
                }
            }
            _ => original,
        };
        mapped.insert(original, node);
        node
    }

    if shared.is_empty() {
        root
    } else {
        walk(module, shared, &mut HashMap::new(), root)
    }
}

fn negative_infinity(module: &mut Module) -> NodeId {
    let inf = module.intern("inf");
    let inf = module.alloc(Node::Const(inf));
    build_call(module, "neg", &[inf])
}

fn operand_shape(operands: &[Operand]) -> Option<(EventLayout, usize)> {
    let mut shape = None;
    for item in operands {
        let Operand::Vector(event) = item else {
            continue;
        };
        let width = event.cells.len();
        shape = Some(match shape {
            None => (event.layout, width),
            Some((layout, other))
                if layout == event.layout && (width == 1 || other == 1 || width == other) =>
            {
                (layout, width.max(other))
            }
            _ => return None,
        });
    }
    shape
}

fn event_components(
    module: &mut Module,
    value: NodeId,
    layout: EventLayout,
    width: u32,
) -> Vec<Cell> {
    if let Some(fill) = expect_builtin_call(module, value, "fill")
        && let [value, _] = *fill.args
        && fill.named.is_empty()
    {
        return vec![Cell::scalar(value); width as usize];
    }
    let row = tensor::int(module, 1);
    (1..=width)
        .map(|index| {
            let index = tensor::int(module, i64::from(index));
            let args = if layout == EventLayout::Row {
                vec![value, row, index]
            } else {
                vec![value, index]
            };
            Cell::scalar(build_call(module, "get", &args))
        })
        .collect()
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
fn expand_static_cells(
    module: &mut Module,
    inputs: &HashSet<BindingId>,
    node: NodeId,
) -> Option<NodeId> {
    // Explicit cells make one copy of the body per lane. Keep this fallback
    // bounded until the AD graph can describe a vectorized structured map.
    const MAX_EXPLICIT_CELLS: u32 = 32;
    let expected = module.type_of(node)?.clone();
    let length = match tensor::real_shape_of(&expected) {
        Some([Dim::Static(length)]) => *length,
        _ => tensor::nested_vector_type(&expected)?.0,
    };
    if !(1..=MAX_EXPLICIT_CELLS).contains(&length) {
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
        let boundary = function.inputs.as_ref()?;
        let entries = reified_inputs(module, head, boundary)?.to_vec();
        let keyword_only = matches!(boundary, Inputs::Auto);
        let names: Vec<_> = entries.iter().map(|&(label, _)| label).collect();
        let operands = bind_call_args(module, &names, args, &call.named, keyword_only)?;
        for index in 0..length {
            let mut scope = Scope::new(module, inputs);
            for ((_, reference), &value) in entries.iter().zip(&operands) {
                let value = pick_cell(scope.module, inputs, value, index, length)?;
                scope
                    .bound
                    .push((*reference, Operand::Scalar(Cell::scalar(value))));
            }
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
