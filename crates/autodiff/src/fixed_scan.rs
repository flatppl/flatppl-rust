//! Loop-preserving differentiation of a fixed-grid record scan projection.

use std::collections::{BTreeSet, HashMap, HashSet};

use flatppl_core::graph::{build_call, build_record, expect_builtin_call};
use flatppl_core::{
    BindingId, Call, CallHead, Dim, Inputs, Module, Node, NodeId, Ref, RefNs, Scalar, Symbol, Type,
};

use crate::{
    Emitter, Error, Key, Mode, Plan, Query, REAL, Step, callbacks, combine_contributions,
    forward_values, recurrence, reverse_values_with_mask, rules, tensor,
};

pub(super) const OP: &str = "fixed record scan projection";

pub(super) struct Projection {
    node: NodeId,
    scan: NodeId,
    body: NodeId,
    formals: Vec<Ref>,
    initial: NodeId,
    sequence: NodeId,
    parameters: Vec<NodeId>,
    tuple_parameters: bool,
    fields: Vec<Symbol>,
    projected: Symbol,
    length: u32,
}

impl Projection {
    pub(super) fn read(module: &Module, inputs: &HashSet<BindingId>, node: NodeId) -> Option<Self> {
        let projection = expect_builtin_call(module, node, "get")?;
        let [scan, key] = *projection.args.as_ref() else {
            return None;
        };
        let Node::Lit(Scalar::Str(name)) = module.node(key) else {
            return None;
        };
        let scan = callbacks::resolve_unpromoted(module, inputs, scan)?;
        let call = expect_builtin_call(module, scan, "scan")?;
        if !projection.named.is_empty() || !call.named.is_empty() {
            return None;
        }
        let [callback, initial, sequence, params] = *call.args.as_ref() else {
            return None;
        };
        let Some(Type::Table {
            columns,
            nrows: Dim::Static(length),
        }) = module.type_of(scan)
        else {
            return None;
        };
        if *length == 0
            || tensor::real_shape(module, sequence) != Some(&[Dim::Static(*length)])
            || module.type_of(initial) != Some(&Type::Record(columns.clone()))
            || columns
                .iter()
                .any(|(_, ty)| !matches!(tensor::real_shape_of(ty), Some([] | [Dim::Static(1..)])))
        {
            return None;
        }
        let projected = columns
            .iter()
            .find(|(symbol, _)| module.resolve(*symbol) == name.as_ref())?
            .0;
        let (body, formals) = recurrence::callback(module, inputs, callback)?;
        if formals.len() != 3 || formals.iter().any(|arg| arg.ns != RefNs::Local) {
            return None;
        }
        let tuple = callbacks::resolve_unpromoted(module, inputs, params)
            .and_then(|node| expect_builtin_call(module, node, "tuple"));
        let parameters = tuple.map_or_else(|| vec![params], |call| call.args.to_vec());
        if parameters.is_empty()
            || parameters
                .iter()
                .any(|&arg| module.type_of(arg) != Some(&REAL))
        {
            return None;
        }
        Some(Self {
            node,
            scan,
            body,
            formals,
            initial,
            sequence,
            parameters,
            tuple_parameters: tuple.is_some(),
            fields: columns.iter().map(|(name, _)| *name).collect(),
            projected,
            length: *length,
        })
    }

    pub(super) fn args(&self) -> Vec<NodeId> {
        [self.initial, self.sequence]
            .into_iter()
            .chain(self.parameters.iter().copied())
            .collect()
    }

    pub(super) fn pullback(
        &self,
        emitter: &mut Emitter,
        bar: NodeId,
        masked: bool,
    ) -> Result<Vec<NodeId>, Error> {
        let state_zero = emitter.zero_like(self.initial)?;
        let primal_scan = emitter.primal(self.scan);
        let m = &mut emitter.module;
        let symbols = Fields::new(m, self.parameters.len());
        let carry = local(m, "_ad_scan_reverse_carry");
        let tick = local(m, "_ad_scan_reverse_index");
        let context = local(m, "_ad_scan_reverse_context");

        let context_fields = [
            (
                symbols.params,
                named_parameters(m, &symbols.parameters, &self.parameters),
            ),
            (symbols.tape, self.columns(m, primal_scan)),
            (symbols.initial, self.initial),
            (symbols.times, self.sequence),
            (symbols.bar, bar),
        ];
        let context_value = build_record(m, &context_fields);
        let context_tape = field(m, context, symbols.tape);
        let context_initial = field(m, context, symbols.initial);
        let context_times = field(m, context, symbols.times);
        let context_bar = field(m, context, symbols.bar);
        let context_params = field(m, context, symbols.params);
        let carry_state = field(m, carry, symbols.state);
        let carry_params = field(m, carry, symbols.params);
        let one = tensor::int(m, 1);
        let previous = build_call(m, "sub", &[tick, one]);
        let safe = build_call(m, "max", &[previous, one]);
        let first = build_call(m, "equal", &[tick, one]);

        let mut previous_fields = Vec::new();
        let mut incoming_fields = Vec::new();
        for &name in &self.fields {
            let series = field(m, context_tape, name);
            let old = at(m, series, safe);
            let initial = field(m, context_initial, name);
            let value = build_call(m, "ifelse", &[first, initial, old]);
            previous_fields.push((name, value));
            let accumulated = field(m, carry_state, name);
            let incoming = if name == self.projected {
                let incoming = at(m, context_bar, tick);
                build_call(m, "add", &[accumulated, incoming])
            } else {
                accumulated
            };
            incoming_fields.push((name, incoming));
        }
        let state = build_record(m, &previous_fields);
        let time = at(m, context_times, tick);
        let args = parameter_fields(m, context_params, &symbols.parameters);
        let incoming = build_record(m, &incoming_fields);
        let (_, gradients) = self.local_products(
            emitter,
            &[state, time],
            &args,
            &[carry, tick, context],
            Product::Reverse {
                seed: incoming,
                masked,
            },
        )?;
        let m = &mut emitter.module;
        let mut params = Vec::new();
        for (&name, &adjoint) in symbols.parameters.iter().zip(&gradients[1..]) {
            let accumulated = field(m, carry_params, name);
            params.push((name, build_call(m, "add", &[accumulated, adjoint])));
        }
        let params = build_record(m, &params);
        let body = build_record(
            m,
            &[(symbols.state, gradients[0]), (symbols.params, params)],
        );
        let step = function(m, body, &[carry, tick, context]);
        let zero = crate::real(m, 0.0);
        let zeros: Vec<_> = symbols
            .parameters
            .iter()
            .map(|&name| (name, zero))
            .collect();
        let params_zero = build_record(m, &zeros);
        let start = build_record(
            m,
            &[(symbols.state, state_zero), (symbols.params, params_zero)],
        );
        let indices = build_call(m, "indicesof", &[self.sequence]);
        let reversed = build_call(m, "reverse", &[indices]);
        let scan = build_call(m, "scan", &[step, start, reversed, context_value]);
        let scan = emitter.primal(scan);
        let m = &mut emitter.module;
        let last = tensor::int(m, i64::from(self.length));
        let state_series = field(m, scan, symbols.state);
        let param_series = field(m, scan, symbols.params);
        let state = at(m, state_series, last);
        let params = at(m, param_series, last);
        let mut result = vec![state];
        result.extend(parameter_fields(m, params, &symbols.parameters));
        Ok(result)
    }

    pub(super) fn pushforward(
        &self,
        emitter: &mut Emitter,
        step: &Step<'_>,
        tangents: &HashMap<Key, NodeId>,
    ) -> Result<NodeId, Error> {
        let tangent = |slot: usize| {
            step.targets[slot]
                .and_then(|key| tangents.get(&key))
                .copied()
        };
        let starting_tangent = match tangent(0) {
            Some(node) => node,
            None => emitter.zero_like(self.initial)?,
        };
        let m = &mut emitter.module;
        let symbols = Fields::new(m, self.parameters.len());
        let carry = local(m, "_ad_scan_forward_carry");
        let time = local(m, "_ad_scan_forward_time");
        let context = local(m, "_ad_scan_forward_context");
        let zero = crate::real(m, 0.0);
        let dots: Vec<_> = (2..2 + self.parameters.len())
            .map(|i| tangent(i).unwrap_or(zero))
            .collect();
        let params = named_parameters(m, &symbols.parameters, &self.parameters);
        let param_tangents = named_parameters(m, &symbols.parameters, &dots);
        let context_value = build_record(
            m,
            &[(symbols.params, params), (symbols.tangent, param_tangents)],
        );
        let state = field(m, carry, symbols.state);
        let state_d = field(m, carry, symbols.tangent);
        let params = field(m, context, symbols.params);
        let param_tangents = field(m, context, symbols.tangent);
        let args = parameter_fields(m, params, &symbols.parameters);
        let mut seeds = vec![state_d];
        seeds.extend(parameter_fields(m, param_tangents, &symbols.parameters));
        let (primal, result) = self.local_products(
            emitter,
            &[state, time],
            &args,
            &[carry, time, context],
            Product::Forward(&seeds),
        )?;
        let m = &mut emitter.module;
        let body = build_record(m, &[(symbols.state, primal), (symbols.tangent, result[0])]);
        let function = function(m, body, &[carry, time, context]);
        let start = build_record(
            m,
            &[
                (symbols.state, self.initial),
                (symbols.tangent, starting_tangent),
            ],
        );
        let scan = build_call(m, "scan", &[function, start, self.sequence, context_value]);
        let tangents = field(m, scan, symbols.tangent);
        Ok(field(m, tangents, self.projected))
    }

    fn columns(&self, module: &mut Module, scan: NodeId) -> NodeId {
        let columns: Vec<_> = self
            .fields
            .iter()
            .map(|&name| (name, field(module, scan, name)))
            .collect();
        build_record(module, &columns)
    }

    fn local_products(
        &self,
        emitter: &mut Emitter,
        state_time: &[NodeId; 2],
        parameters: &[NodeId],
        scope: &[NodeId],
        product: Product<'_>,
    ) -> Result<(NodeId, Vec<NodeId>), Error> {
        let refused = || {
            Error::new(
                Some(self.node),
                "cannot construct the local scan derivative",
            )
        };
        let mut module = emitter.module.clone();
        let originals: HashSet<_> = module.bindings().map(|(id, _)| id).collect();
        let site = |name| {
            module
                .bindings()
                .find(|(_, binding)| module.resolve(binding.name) == name)
                .map(|(id, binding)| (id, binding.rhs))
        };
        let (inputs, old_input) = site("inputs").ok_or_else(refused)?;
        let (outputs, old_output) = site("outputs").ok_or_else(refused)?;
        let first = tensor::int(&mut module, 1);
        let time = at(&mut module, self.sequence, first);
        let base: Vec<_> = [self.initial, time]
            .into_iter()
            .chain(self.parameters.iter().copied())
            .collect();
        let mut local = Vec::new();
        let mut names = Vec::new();
        let state_type = module.type_of(self.initial).cloned().ok_or_else(refused)?;
        for (i, arg) in base.into_iter().enumerate() {
            let boundary = build_call(&mut module, "identity", &[arg]);
            module.set_type(boundary, if i == 0 { state_type.clone() } else { REAL });
            let value = recurrence::bind_value(&mut module, boundary, "_ad_scan_argument")
                .ok_or_else(refused)?;
            let Node::Ref(reference) = module.node(value) else {
                return Err(refused());
            };
            names.push(module.resolve(reference.name).to_owned());
            local.push(value);
        }
        let capture = if self.tuple_parameters {
            build_call(&mut module, "tuple", &local[2..])
        } else {
            local[2]
        };
        let actuals = [local[0], local[1], capture];
        let substitutions: Vec<_> = self.formals.iter().copied().zip(actuals).collect();
        let body = recurrence::substitute(
            &mut module,
            &originals,
            self.body,
            &substitutions,
            &mut HashMap::new(),
            0,
        )
        .ok_or_else(refused)?;
        let signature = build_call(&mut module, "tuple", &local);
        module.set_binding_rhs(inputs, signature);
        module.set_binding_rhs(outputs, body);
        if let Some(diag) = flatppl_infer::infer(&mut module)
            .into_iter()
            .find(|d| d.severity == flatppl_infer::Severity::Error)
        {
            return Err(Error::new(Some(self.node), diag.message));
        }
        let names: Vec<_> = names
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 1)
            .map(|(_, x)| x.as_str())
            .collect();
        let mode = match product {
            Product::Forward(_) => Mode::Jvp,
            Product::Reverse { .. } => Mode::Vjp,
        };
        let mut module = std::borrow::Cow::Owned(module);
        let query = Query::read(&mut module, &names, mode)?;
        let root = query.key(&module, body)?;
        let mut plan = Plan::new(&module, &query);
        plan.visit(root, 0)?;
        let mut requests = BTreeSet::new();
        plan.collect_rules(mode, &mut requests);
        let rules = rules::Rules::load(requests)?;
        let mut inner = Emitter::new(&module, &query);
        inner.inline = true;
        let derived = match product {
            Product::Forward(seeds) => {
                let tangents = query
                    .wrt
                    .iter()
                    .zip(seeds)
                    .map(|(input, &seed)| (Key::Input(input.binding), seed))
                    .collect();
                let result = forward_values(&mut inner, &plan, &rules, tangents)?;
                vec![match result.get(&root) {
                    Some(&result) => result,
                    None => inner.zero_like(body)?,
                }]
            }
            Product::Reverse { seed, masked } => {
                let mut result = reverse_values_with_mask(
                    &mut inner,
                    &plan,
                    &rules,
                    &[root],
                    &[seed],
                    mode,
                    masked,
                )?;
                query
                    .wrt
                    .iter()
                    .map(|input| match result.remove(&Key::Input(input.binding)) {
                        Some(terms) => combine_contributions(&mut inner, input.reference, terms),
                        None => inner.zero_like(input.reference),
                    })
                    .collect::<Result<Vec<_>, Error>>()?
            }
        };
        let actual: Vec<_> = state_time
            .iter()
            .copied()
            .chain(parameters.iter().copied())
            .collect();
        let mut replacements: Vec<_> = local
            .iter()
            .zip(actual)
            .map(|(&arg, replacement)| {
                let Node::Ref(reference) = inner.module.node(arg) else {
                    unreachable!()
                };
                (*reference, replacement)
            })
            .collect();
        for &arg in scope {
            let Node::Ref(reference) = inner.module.node(arg) else {
                return Err(refused());
            };
            replacements.push((*reference, arg));
        }
        let mut shared = HashMap::new();
        let mut rewrite = |node| {
            recurrence::substitute(
                &mut inner.module,
                &originals,
                node,
                &replacements,
                &mut shared,
                0,
            )
            .ok_or_else(refused)
        };
        let primal = rewrite(body)?;
        let derivative = derived
            .into_iter()
            .map(rewrite)
            .collect::<Result<Vec<_>, _>>()?;
        inner.module.set_binding_rhs(inputs, old_input);
        inner.module.set_binding_rhs(outputs, old_output);
        let temporary: HashSet<_> = local
            .iter()
            .filter_map(|&arg| match module.node(arg) {
                Node::Ref(reference) => module.binding_by_name(reference.name),
                _ => None,
            })
            .collect();
        let retained = inner
            .module
            .bindings()
            .filter_map(|(id, _)| (!temporary.contains(&id)).then_some(id))
            .collect();
        inner.module.retain_bindings(&retained);
        emitter.module = inner.module;
        Ok((primal, derivative))
    }
}

enum Product<'a> {
    Forward(&'a [NodeId]),
    Reverse { seed: NodeId, masked: bool },
}

struct Fields {
    state: Symbol,
    params: Symbol,
    tape: Symbol,
    initial: Symbol,
    times: Symbol,
    bar: Symbol,
    tangent: Symbol,
    parameters: Vec<Symbol>,
}

impl Fields {
    fn new(module: &mut Module, arity: usize) -> Self {
        Self {
            state: module.intern("state"),
            params: module.intern("params"),
            tape: module.intern("tape"),
            initial: module.intern("initial"),
            times: module.intern("times"),
            bar: module.intern("bar"),
            tangent: module.intern("tangent"),
            parameters: (0..arity)
                .map(|i| module.intern(&format!("arg{i}")))
                .collect(),
        }
    }
}

fn named_parameters(module: &mut Module, names: &[Symbol], parameters: &[NodeId]) -> NodeId {
    let fields: Vec<_> = names
        .iter()
        .copied()
        .zip(parameters.iter().copied())
        .collect();
    build_record(module, &fields)
}

fn parameter_fields(module: &mut Module, record: NodeId, names: &[Symbol]) -> Vec<NodeId> {
    names
        .iter()
        .map(|&name| field(module, record, name))
        .collect()
}

fn field(module: &mut Module, record: NodeId, name: Symbol) -> NodeId {
    let key = module.alloc(Node::Lit(Scalar::Str(module.resolve(name).into())));
    at(module, record, key)
}

fn at(module: &mut Module, container: NodeId, selector: NodeId) -> NodeId {
    build_call(module, "get", &[container, selector])
}

fn local(module: &mut Module, name: &str) -> NodeId {
    let name = module.intern(&format!("{name}_"));
    module.alloc(Node::Ref(Ref {
        ns: RefNs::Local,
        name,
    }))
}

fn function(module: &mut Module, body: NodeId, parameters: &[NodeId]) -> NodeId {
    let parameters: Vec<_> = parameters
        .iter()
        .map(|&node| {
            let Node::Ref(reference) = module.node(node) else {
                unreachable!()
            };
            (reference.name, *reference)
        })
        .collect();
    let name = module.intern("functionof");
    module.alloc(Node::Call(Call {
        head: CallHead::Builtin(name),
        args: Box::new([body]),
        named: Box::new([]),
        inputs: Some(Inputs::Spec(parameters.into())),
    }))
}
