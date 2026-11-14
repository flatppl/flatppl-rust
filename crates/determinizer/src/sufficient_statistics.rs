//! Portable likelihood reductions and an explicit data-preparation boundary.
//!
//! Summary bindings are ordinary FlatPDL expressions. Backends may bake them,
//! place them in a setup block, or compile the two modules returned here.

use std::collections::{HashMap, HashSet};

use flatppl_core::{
    Binding, Call, CallHead, Dim, Idx, Inputs, Module, Node, NodeId, Ref, RefNs, Scalar,
    ScalarType, Symbol, Type, ValueSet,
};

use crate::RefuseError;

mod circular;
mod correlation;
mod grouped;
mod histogram;
mod multivariate;
mod poisson;
mod regression;
mod scalar;
use crate::density::{build_call, build_record};

const PREFIX: &str = "__likelihood_summary_";

/// Two ordinary deterministic queries. Run `preparation` once per data binding,
/// then pass its results to the appended `score` inputs in `summaries` order.
/// Raw data used outside reduced factors remains an input of `score`.
#[derive(Debug)]
pub struct PreparedLikelihood {
    pub preparation: Module,
    pub score: Module,
    pub summaries: Vec<String>,
}

#[derive(Clone)]
enum Expr {
    Node(NodeId),
    Number(f64),
    Integer(i64),
    Constant(&'static str),
    Call(&'static str, Vec<Expr>),
    Map(&'static str, Vec<Expr>),
    Local,
    Next,
    Held(usize),
    Text(&'static str),
    Cells(Box<Expr>, Box<Expr>, Vec<Expr>),
    Scan(Box<Expr>, Box<Expr>, Box<Expr>),
}

fn op(name: &'static str, args: impl IntoIterator<Item = Expr>) -> Expr {
    Expr::Call(name, args.into_iter().collect())
}

fn map(name: &'static str, args: impl IntoIterator<Item = Expr>) -> Expr {
    Expr::Map(name, args.into_iter().collect())
}

fn num(x: f64) -> Expr {
    Expr::Number(x)
}

fn positive_in_both_precisions(x: f64) -> bool {
    x > 0.0 && x.is_finite() && (x as f32) > 0.0 && (x as f32).is_finite()
}

fn interior_in_both_precisions(x: f64) -> bool {
    positive_in_both_precisions(x) && x < 1.0 && (x as f32) < 1.0
}

fn relative_log(x: Expr, center: Expr) -> Expr {
    let relative = map(
        "divide",
        [map("sub", [x.clone(), center.clone()]), center.clone()],
    );
    let near = map("lt", [map("abs", [relative.clone()]), num(0.5)]);
    let safe = map("ifelse", [near.clone(), relative, num(0.0)]);
    map(
        "ifelse",
        [
            near,
            map("log1p", [safe]),
            map("sub", [map("log", [x]), op("log", [center])]),
        ],
    )
}

fn complement_relative_log(x: Expr, center: Expr) -> Expr {
    let relative = map(
        "divide",
        [
            map("sub", [center.clone(), x.clone()]),
            op("sub", [num(1.0), center.clone()]),
        ],
    );
    let near = map("lt", [map("abs", [relative.clone()]), num(0.5)]);
    let safe = map("ifelse", [near.clone(), relative, num(0.0)]);
    map(
        "ifelse",
        [
            near,
            map("log1p", [safe]),
            map(
                "sub",
                [
                    map("log1p", [map("neg", [x])]),
                    op("log1p", [op("neg", [center])]),
                ],
            ),
        ],
    )
}

fn materialize(m: &mut Module, expr: Expr) -> NodeId {
    match expr {
        Expr::Node(id) => id,
        Expr::Number(x) => m.alloc(Node::Lit(Scalar::Real(x))),
        Expr::Integer(x) => m.alloc(Node::Lit(Scalar::Int(x))),
        Expr::Constant(name) => {
            let name = m.intern(name);
            m.alloc(Node::Const(name))
        }
        Expr::Map(name, args) => {
            let head = m.intern(name);
            let head = m.alloc(Node::Const(head));
            let mut operands = vec![head];
            operands.extend(args.into_iter().map(|x| materialize(m, x)));
            build_call(m, "broadcast", &operands)
        }
        Expr::Call(name, args) => {
            let args: Vec<_> = args.into_iter().map(|x| materialize(m, x)).collect();
            build_call(m, name, &args)
        }
        Expr::Local => {
            let name = m.intern("_likelihood_cell_");
            m.alloc(Node::Ref(Ref {
                ns: RefNs::Local,
                name,
            }))
        }
        Expr::Next => {
            let name = m.intern("_likelihood_next_");
            m.alloc(Node::Ref(Ref {
                ns: RefNs::Local,
                name,
            }))
        }
        Expr::Held(index) => {
            let name = m.intern(&format!("_likelihood_held_{index}_"));
            m.alloc(Node::Ref(Ref {
                ns: RefNs::Local,
                name,
            }))
        }
        Expr::Text(text) => m.alloc(Node::Lit(Scalar::Str(text.into()))),
        Expr::Scan(body, initial, data) => {
            let body = materialize(m, *body);
            let inputs = [
                ("likelihood_cell", "_likelihood_cell_"),
                ("likelihood_next", "_likelihood_next_"),
            ]
            .into_iter()
            .map(|(name, local)| {
                let name = m.intern(name);
                let local = m.intern(local);
                (
                    name,
                    Ref {
                        ns: RefNs::Local,
                        name: local,
                    },
                )
            })
            .collect::<Vec<_>>();
            let head = m.intern("functionof");
            let function = m.alloc(Node::Call(Call {
                head: CallHead::Builtin(head),
                args: vec![body].into(),
                named: vec![].into(),
                inputs: Some(Inputs::Spec(inputs.into())),
            }));
            let initial = materialize(m, *initial);
            let data = materialize(m, *data);
            build_call(m, "scan", &[function, initial, data])
        }
        Expr::Cells(body, data, held) => {
            let body = materialize(m, *body);
            let name = m.intern("likelihood_cell");
            let local = m.intern("_likelihood_cell_");
            let mut inputs = vec![(
                name,
                Ref {
                    ns: RefNs::Local,
                    name: local,
                },
            )];
            for index in 0..held.len() {
                let name = m.intern(&format!("likelihood_held_{index}"));
                let local = m.intern(&format!("_likelihood_held_{index}_"));
                inputs.push((
                    name,
                    Ref {
                        ns: RefNs::Local,
                        name: local,
                    },
                ));
            }
            let head = m.intern("functionof");
            let function = m.alloc(Node::Call(Call {
                head: CallHead::Builtin(head),
                args: vec![body].into(),
                named: vec![].into(),
                inputs: Some(Inputs::Spec(inputs.into())),
            }));
            let data = materialize(m, *data);
            let mut operands = vec![function, data];
            for value in held {
                let value = materialize(m, value);
                operands.push(build_call(m, "vector", &[value]));
            }
            build_call(m, "broadcast", &operands)
        }
    }
}

fn local_ref(m: &mut Module, name: Symbol) -> NodeId {
    m.alloc(Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name,
    }))
}

fn binding(m: &Module, name: &str) -> Option<NodeId> {
    m.bindings()
        .find(|(_, b)| m.resolve(b.name) == name)
        .map(|(_, b)| b.rhs)
}

fn call<'a>(m: &'a Module, id: NodeId, name: &str) -> Option<&'a Call> {
    match m.node(id) {
        Node::Call(c) if matches!(c.head, CallHead::Builtin(h) if m.resolve(h) == name) => Some(c),
        _ => None,
    }
}

fn resolve(m: &Module, mut id: NodeId, boundary: &HashSet<Symbol>) -> NodeId {
    let mut seen = HashSet::new();
    while seen.insert(id) {
        let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name,
        }) = m.node(id)
        else {
            break;
        };
        if boundary.contains(name) {
            break;
        }
        let Some(bid) = m.binding_by_name(*name) else {
            break;
        };
        id = m.binding(bid).rhs;
    }
    id
}

fn signature(m: &Module, name: &str) -> Vec<NodeId> {
    binding(m, name)
        .map(|id| call(m, id, "tuple").map_or_else(|| vec![id], |c| c.args.to_vec()))
        .unwrap_or_default()
}

fn set_signature(m: &mut Module, name: &str, values: &[NodeId]) {
    let rhs = match values {
        [only] => *only,
        _ => build_call(m, "tuple", values),
    };
    let symbol = m.intern(name);
    if let Some(bid) = m.binding_by_name(symbol) {
        m.set_binding_rhs(bid, rhs);
    } else {
        m.add_binding(Binding {
            name: symbol,
            rhs,
            doc: None,
            public: true,
            synthetic: false,
        });
    }
}

/// Does this graph depend only on the allowed ABI inputs and fixed values?
/// ABI boundaries override the source RHS, even for promoted fixed bindings.
fn data_only(
    m: &Module,
    root: NodeId,
    allowed: &HashSet<Symbol>,
    boundary: &HashSet<Symbol>,
) -> bool {
    let mut pending = vec![(root, Vec::<Symbol>::new())];
    let mut seen = HashSet::new();
    while let Some((id, locals)) = pending.pop() {
        if !seen.insert((id, locals.clone())) {
            continue;
        }
        match m.node(id) {
            Node::Ref(Ref {
                ns: RefNs::SelfMod,
                name,
            }) => {
                if boundary.contains(name) {
                    if !allowed.contains(name) {
                        return false;
                    }
                } else if let Some(bid) = m.binding_by_name(*name) {
                    pending.push((m.binding(bid).rhs, locals));
                } else {
                    return false;
                }
            }
            Node::Ref(Ref {
                ns: RefNs::Local,
                name,
            }) if locals.contains(name) => {}
            Node::Ref(_) | Node::Axis(_) | Node::Hole => return false,
            Node::Call(c) if c.inputs.is_some() => {
                let (CallHead::Builtin(head), Some(Inputs::Spec(inputs))) = (&c.head, &c.inputs)
                else {
                    return false;
                };
                if m.resolve(*head) != "functionof" {
                    return false;
                }
                let mut bound = locals;
                bound.extend(inputs.iter().map(|(_, r)| r.name));
                pending.extend(c.args.iter().map(|&child| (child, bound.clone())));
            }
            Node::Call(c) if matches!(c.head, CallHead::Builtin(h) if matches!(m.resolve(h), "elementof" | "draw" | "builtin_sample")) =>
            {
                return false;
            }
            node => pending.extend(
                node.children()
                    .into_iter()
                    .map(|child| (child, locals.clone())),
            ),
        }
    }
    true
}

struct Factor {
    constructor: Symbol,
    parameters: Vec<(Symbol, NodeId)>,
    observed: NodeId,
}

fn per_cell(m: &Module, id: NodeId, boundary: &HashSet<Symbol>) -> Option<Factor> {
    per_cell_inner(m, id, boundary, false)
}

fn per_cell_with_invariants(m: &Module, id: NodeId, boundary: &HashSet<Symbol>) -> Option<Factor> {
    per_cell_inner(m, id, boundary, true)
}

fn per_cell_inner(
    m: &Module,
    id: NodeId,
    boundary: &HashSet<Symbol>,
    scalar_captures: bool,
) -> Option<Factor> {
    let sum = call(m, id, "sum")?;
    let [arg] = *sum.args else { return None };
    let broadcast = call(m, resolve(m, arg, boundary), "broadcast")?;
    let args = broadcast.args.as_ref();
    if let [head, kernel, input, observed] = args
        && matches!(m.node(*head), Node::Const(h) if m.resolve(*h) == "builtin_logdensityof")
        && let Node::Const(constructor) = m.node(*kernel)
        && let Some(record) = call(m, resolve(m, *input, boundary), "broadcast")
        && matches!(record.args.as_ref(), [head] if matches!(m.node(*head), Node::Const(h) if m.resolve(*h) == "record"))
    {
        return Some(Factor {
            constructor: *constructor,
            parameters: record.named.iter().map(|a| (a.name, a.value)).collect(),
            observed: *observed,
        });
    }
    let (head, operands) = args.split_first()?;
    let function = call(m, resolve(m, *head, boundary), "functionof")?;
    let (Some(Inputs::Spec(inputs)), [body]) = (&function.inputs, function.args.as_ref()) else {
        return None;
    };
    if inputs.len() != operands.len() {
        return None;
    }
    let body = call(m, *body, "builtin_logdensityof")?;
    let [kernel, record, point] = *body.args else {
        return None;
    };
    let Node::Const(constructor) = m.node(kernel) else {
        return None;
    };
    let operand = |id| {
        let Node::Ref(r) = m.node(id) else {
            return None;
        };
        inputs
            .iter()
            .position(|(_, input)| input == r)
            .map(|at| operands[at])
    };
    let record = call(m, record, "record")?;
    let parameters = record
        .named
        .iter()
        .map(|a| {
            let value = operand(a.value).or_else(|| {
                (scalar_captures
                    && matches!(m.type_of(a.value), Some(Type::Scalar(_)))
                    && data_only(m, a.value, boundary, boundary))
                .then_some(a.value)
            })?;
            Some((a.name, value))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Factor {
        constructor: *constructor,
        parameters,
        observed: operand(point)?,
    })
}

#[derive(Clone)]
struct Reducer {
    preparing: bool,
    boundary: HashSet<Symbol>,
    allowed: HashSet<Symbol>,
    summaries: Vec<Symbol>,
    cache: HashMap<(Vec<DataKey>, &'static str), NodeId>,
    count: usize,
    next_serial: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum DataKey {
    Input(Symbol),
    Node(NodeId),
}

impl Reducer {
    fn summary(&mut self, m: &mut Module, data: NodeId, kind: &'static str, expr: Expr) -> Expr {
        self.summary_for(m, &[data], kind, expr)
    }

    fn summary_for(
        &mut self,
        m: &mut Module,
        data: &[NodeId],
        kind: &'static str,
        expr: Expr,
    ) -> Expr {
        let key = data
            .iter()
            .map(|&data| {
                let data = resolve(m, data, &self.boundary);
                match m.node(data) {
                    Node::Ref(Ref {
                        ns: RefNs::SelfMod,
                        name,
                    }) if self.boundary.contains(name) => DataKey::Input(*name),
                    _ => DataKey::Node(data),
                }
            })
            .collect::<Vec<_>>();
        if let Some(&id) = self.cache.get(&(key.clone(), kind)) {
            return Expr::Node(id);
        }
        let rhs = materialize(m, expr);
        let name = loop {
            let name = m.intern(&format!("{PREFIX}{}", self.next_serial));
            self.next_serial += 1;
            if m.binding_by_name(name).is_none() {
                break name;
            }
        };
        m.add_binding(Binding {
            name,
            rhs,
            doc: None,
            public: false,
            synthetic: true,
        });
        self.summaries.push(name);
        let id = local_ref(m, name);
        self.cache.insert((key, kind), id);
        Expr::Node(id)
    }

    fn scalar(&self, m: &Module, id: NodeId) -> Option<NodeId> {
        let resolved = resolve(m, id, &self.boundary);
        if let Some(vector) = call(m, resolved, "vector")
            && let [only] = *vector.args
        {
            return Some(only);
        }
        matches!(
            m.type_of(id).or_else(|| m.type_of(resolved)),
            Some(Type::Scalar(_))
        )
        .then_some(id)
    }

    fn anchor(
        &self,
        m: &mut Module,
        f: &Factor,
        parameters: &[(Symbol, NodeId)],
        point: Expr,
    ) -> Expr {
        let kernel = m.alloc(Node::Const(f.constructor));
        let record = build_record(m, parameters);
        op(
            "builtin_logdensityof",
            [Expr::Node(kernel), Expr::Node(record), point],
        )
    }

    fn reduce(&mut self, m: &mut Module, id: NodeId) -> Option<NodeId> {
        if let Some(result) = grouped::reduce(self, m, id) {
            return Some(result);
        }
        if let Some(result) = correlation::reduce(self, m, id) {
            self.count += 1;
            return Some(materialize(m, result));
        }
        if let Some(result) = circular::reduce(self, m, id) {
            self.count += 1;
            return Some(materialize(m, result));
        }
        if let Some(result) = scalar::reduce(self, m, id) {
            self.count += 1;
            return Some(materialize(m, result));
        }
        if let Some(result) = regression::reduce(self, m, id) {
            self.count += 1;
            return Some(materialize(m, result));
        }
        if let Some(result) = multivariate::reduce(self, m, id) {
            self.count += 1;
            return Some(materialize(m, result));
        }
        if let Some(result) = self.reduce_moments(m, id) {
            return Some(result);
        }
        let factor = per_cell(m, id, &self.boundary)?;
        if !data_only(m, factor.observed, &self.allowed, &self.boundary) {
            return None;
        }
        if let Some(result) = histogram::reduce(self, m, &factor) {
            self.count += 1;
            return Some(materialize(m, result));
        }
        poisson::reduce(self, m, &factor)
    }

    fn reduce_moments(&mut self, m: &mut Module, id: NodeId) -> Option<NodeId> {
        let f = per_cell(m, id, &self.boundary)?;
        self.reduce_moment_factor(m, &f)
    }

    fn reduce_moment_factor(&mut self, m: &mut Module, f: &Factor) -> Option<NodeId> {
        if !data_only(m, f.observed, &self.allowed, &self.boundary) {
            return None;
        }
        let Type::Array { shape, elem } = m.type_of(f.observed)?.clone() else {
            return None;
        };
        let [Dim::Static(size)] = shape.as_ref() else {
            return None;
        };
        if *size < 2 || !matches!(*elem, Type::Scalar(_)) {
            return None;
        }
        let name = m.resolve(f.constructor).to_string();
        let parameters = f
            .parameters
            .iter()
            .map(|&(name, id)| {
                Some((
                    name,
                    if m.resolve(name) == "support" {
                        match call(m, resolve(m, id, &self.boundary), "vector") {
                            Some(c) if c.args.len() == 1 => c.args[0],
                            _ => id,
                        }
                    } else {
                        self.scalar(m, id)?
                    },
                ))
            })
            .collect::<Option<Vec<_>>>()?;
        let param = |key: &str| {
            parameters
                .iter()
                .find(|(s, _)| m.resolve(*s) == key)
                .map(|(_, id)| Expr::Node(*id))
        };
        // Resolve all parameter names before mutating the module.
        let mu = param("mu");
        let sigma = param("sigma");
        let rate = param("rate");
        let alpha = param("alpha");
        let beta = param("beta");
        let shape_parameter = param("shape");
        let p = param("p");
        let trials = param("n");
        let support = parameters
            .iter()
            .find(|(s, _)| m.resolve(*s) == "support")
            .map(|(_, id)| *id);
        let observed = f.observed;
        let x = Expr::Node(observed);
        let n = num(*size as f64);
        let real = matches!(*elem, Type::Scalar(ScalarType::Real));
        let scalar_support = match m.valueset_of(observed) {
            Some(ValueSet::CartPow(set, _)) => Some(set.as_ref()),
            _ => None,
        };
        let fixed = fixed_values(m, observed, &self.boundary);
        let positive = fixed
            .as_ref()
            .is_some_and(|v| v.iter().all(|x| positive_in_both_precisions(*x)))
            || matches!(scalar_support, Some(ValueSet::Interval(lo, hi)) if positive_in_both_precisions(*lo) && positive_in_both_precisions(*hi));
        let interior = fixed
            .as_ref()
            .is_some_and(|v| v.iter().all(|x| interior_in_both_precisions(*x)))
            || matches!(scalar_support, Some(ValueSet::Interval(lo, hi)) if interior_in_both_precisions(*lo) && interior_in_both_precisions(*hi));
        let nonnegative = fixed
            .as_ref()
            .is_some_and(|v| v.iter().all(|x| x.is_finite() && *x >= 0.0))
            || matches!(
                scalar_support,
                Some(ValueSet::NonNegIntegers | ValueSet::PosIntegers)
            )
            || matches!(scalar_support, Some(ValueSet::Interval(lo, hi)) if *lo >= 0.0 && hi.is_finite());
        let result = match name.as_str() {
            "Normal" | "LogNormal" if real => {
                let mu = mu?;
                let sigma = sigma?;
                if fixed
                    .as_ref()
                    .is_some_and(|v| v.iter().any(|x| !x.is_finite()))
                {
                    return None;
                }
                let valid = if name == "LogNormal" {
                    map(
                        "land",
                        [
                            map("isfinite", [x.clone()]),
                            map("gt", [x.clone(), num(0.0)]),
                        ],
                    )
                } else {
                    map("isfinite", [x.clone()])
                };
                let all_finite = self.summary(
                    m,
                    observed,
                    if name == "Normal" {
                        "all_finite"
                    } else {
                        "all_positive_finite"
                    },
                    op("lall", [valid.clone()]),
                );
                let raw_minimum = if name == "Normal" {
                    Some(self.summary(m, observed, "minimum", op("minimum", [x.clone()])))
                } else {
                    None
                };
                let safe = map(
                    "ifelse",
                    [
                        valid,
                        x.clone(),
                        num(if name == "Normal" { 0.0 } else { 1.0 }),
                    ],
                );
                let y = if name == "LogNormal" {
                    map("log", [safe])
                } else {
                    safe
                };
                let log_values = y.clone();
                let first = op("get", [y.clone(), Expr::Integer(1)]);
                let mixed_sign = op(
                    "land",
                    [
                        op("lt", [op("minimum", [y.clone()]), num(0.0)]),
                        op("gt", [op("maximum", [y.clone()]), num(0.0)]),
                    ],
                );
                let first = op("ifelse", [mixed_sign, num(0.0), first]);
                let center_expr = op(
                    "add",
                    [
                        first.clone(),
                        op(
                            "sum",
                            [map("divide", [map("sub", [y.clone(), first]), n.clone()])],
                        ),
                    ],
                );
                let center = self.summary(
                    m,
                    observed,
                    if name == "Normal" {
                        "center"
                    } else {
                        "log_center"
                    },
                    center_expr,
                );
                let residuals = map("sub", [y, center.clone()]);
                let residual_sum = self.summary(
                    m,
                    observed,
                    if name == "Normal" {
                        "residual_sum"
                    } else {
                        "log_residual_sum"
                    },
                    op("sum", [map("divide", [residuals.clone(), n.clone()])]),
                );
                let scale = self.summary(
                    m,
                    observed,
                    if name == "Normal" {
                        "residual_scale"
                    } else {
                        "log_residual_scale"
                    },
                    op("maximum", [map("abs", [residuals.clone()])]),
                );
                let safe_scale = op(
                    "ifelse",
                    [op("iszero", [scale.clone()]), num(1.0), scale.clone()],
                );
                let scaled = map("divide", [residuals, safe_scale]);
                let spread = self.summary(
                    m,
                    observed,
                    if name == "Normal" {
                        "spread"
                    } else {
                        "log_spread"
                    },
                    op("sqrt", [op("sum", [map("mul", [scaled.clone(), scaled])])]),
                );
                let z = op("divide", [op("sub", [center.clone(), mu]), sigma.clone()]);
                let cross = op(
                    "mul",
                    [
                        op("mul", [num(2.0), n.clone()]),
                        op(
                            "mul",
                            [z, op("divide", [residual_sum.clone(), sigma.clone()])],
                        ),
                    ],
                );
                let cross = op("ifelse", [op("iszero", [residual_sum]), num(0.0), cross]);
                let spread = op("mul", [op("divide", [scale, sigma]), spread]);
                let correction = op(
                    "mul",
                    [
                        num(0.5),
                        op("add", [cross, op("mul", [spread.clone(), spread])]),
                    ],
                );
                let score = if name == "Normal" {
                    op(
                        "sub",
                        [
                            op("mul", [n, self.anchor(m, f, &parameters, center)]),
                            correction,
                        ],
                    )
                } else {
                    // Use Normal in log space and retain the full Jacobian.
                    let normal = m.intern("Normal");
                    let normal = Factor {
                        constructor: normal,
                        parameters: parameters.clone(),
                        observed,
                    };
                    let logs = self.summary(m, observed, "safe_log_sum", op("sum", [log_values]));
                    op(
                        "sub",
                        [
                            op(
                                "sub",
                                [
                                    op("mul", [n, self.anchor(m, &normal, &parameters, center)]),
                                    correction,
                                ],
                            ),
                            logs,
                        ],
                    )
                };
                let score = op(
                    "ifelse",
                    [all_finite, score, op("neg", [Expr::Constant("inf")])],
                );
                if let Some(raw_minimum) = raw_minimum {
                    op(
                        "ifelse",
                        [op("isnan", [raw_minimum.clone()]), raw_minimum, score],
                    )
                } else {
                    // LogNormal's primitive gates on x > 0, including NaN.
                    score
                }
            }
            "Gamma" if real && positive => {
                let shape = shape_parameter?;
                let rate = rate?;
                let first = op("get", [x.clone(), Expr::Integer(1)]);
                let center = self.summary(
                    m,
                    observed,
                    "positive_center",
                    op(
                        "add",
                        [
                            first.clone(),
                            op(
                                "sum",
                                [map("divide", [map("sub", [x.clone(), first]), n.clone()])],
                            ),
                        ],
                    ),
                );
                let residual = self.summary(
                    m,
                    observed,
                    "positive_residual_sum",
                    op("sum", [map("sub", [x.clone(), center.clone()])]),
                );
                let logs = self.summary(
                    m,
                    observed,
                    "positive_relative_log_sum",
                    op("sum", [relative_log(x, center.clone())]),
                );
                op(
                    "sub",
                    [
                        op(
                            "add",
                            [
                                op("mul", [n, self.anchor(m, f, &parameters, center)]),
                                op("mul", [op("sub", [shape, num(1.0)]), logs]),
                            ],
                        ),
                        op("mul", [rate, residual]),
                    ],
                )
            }
            "Beta" if real && interior => {
                let alpha = alpha?;
                let beta = beta?;
                let center = self.summary(
                    m,
                    observed,
                    "unit_center",
                    op("sum", [map("divide", [x.clone(), n.clone()])]),
                );
                let logs = self.summary(
                    m,
                    observed,
                    "unit_relative_log_sum",
                    op("sum", [relative_log(x.clone(), center.clone())]),
                );
                let complements = self.summary(
                    m,
                    observed,
                    "unit_complement_relative_log_sum",
                    op("sum", [complement_relative_log(x, center.clone())]),
                );
                let normalizer = self.anchor(m, f, &parameters, center);
                op(
                    "add",
                    [
                        op("mul", [n, normalizer]),
                        op(
                            "add",
                            [
                                op("mul", [op("sub", [alpha, num(1.0)]), logs]),
                                op("mul", [op("sub", [beta, num(1.0)]), complements]),
                            ],
                        ),
                    ],
                )
            }
            "Exponential" if real => {
                let minimum = self.summary(m, observed, "minimum", op("minimum", [x.clone()]));
                let valid = op("ge", [minimum.clone(), num(0.0)]);
                let total = self.summary(
                    m,
                    observed,
                    "nonnegative_sum",
                    op("ifelse", [valid.clone(), op("sum", [x]), num(0.0)]),
                );
                let score = op(
                    "sub",
                    [
                        op("mul", [n, self.anchor(m, f, &parameters, num(0.0))]),
                        op("mul", [rate?, total]),
                    ],
                );
                op(
                    "ifelse",
                    [
                        op("isnan", [minimum.clone()]),
                        minimum,
                        op("ifelse", [valid, score, op("neg", [Expr::Constant("inf")])]),
                    ],
                )
            }
            "Poisson" | "ContinuedPoisson"
                if nonnegative
                    && (fixed.as_ref().is_some_and(|v| v.iter().all(|x| *x < 64.0))
                        || matches!(scalar_support, Some(ValueSet::Interval(_, hi)) if *hi < 64.0))
                    && (name != "Poisson"
                        || matches!(*elem, Type::Scalar(ScalarType::Integer))) =>
            {
                let rate = rate?;
                let total = self.summary(m, observed, "sum", op("sum", [x.clone()]));
                let factorials = self.summary(
                    m,
                    observed,
                    "factorials",
                    op("sum", [map("loggamma", [map("add", [x, num(1.0)])])]),
                );
                let safe_rate = op(
                    "ifelse",
                    [
                        op(
                            "lor",
                            [op("iszero", [total.clone()]), op("iszero", [rate.clone()])],
                        ),
                        num(1.0),
                        rate.clone(),
                    ],
                );
                let score = op(
                    "sub",
                    [
                        op(
                            "add",
                            [
                                op("mul", [n, self.anchor(m, f, &parameters, num(0.0))]),
                                op("mul", [total.clone(), op("log", [safe_rate])]),
                            ],
                        ),
                        factorials,
                    ],
                );
                op(
                    "ifelse",
                    [
                        op("land", [op("iszero", [rate]), op("gt", [total, num(0.0)])]),
                        op("neg", [Expr::Constant("inf")]),
                        score,
                    ],
                )
            }
            "Bernoulli"
                if matches!(*elem, Type::Scalar(ScalarType::Boolean))
                    || fixed
                        .as_ref()
                        .is_some_and(|v| v.iter().all(|x| *x == 0.0 || *x == 1.0)) =>
            {
                let successes = self.summary(m, observed, "successes", op("sum", [x]));
                let failures = op("sub", [n, successes.clone()]);
                let p = p?;
                let p_name = parameters[0].0;
                let yes = weighted_bernoulli(m, p_name, p.clone(), successes, true);
                let no = weighted_bernoulli(m, p_name, p, failures, false);
                op("add", [yes, no])
            }
            "Geometric" if nonnegative && matches!(*elem, Type::Scalar(ScalarType::Integer)) => {
                // Widen each count before summing: i32 accumulation can wrap
                // even when each authored count fits the target integer type.
                let total = self.summary(m, observed, "real_sum", op("sum", [map("real", [x])]));
                let p = p?;
                let p_id = materialize(m, p.clone());
                let bernoulli = m.intern("Bernoulli");
                let bernoulli = Factor {
                    constructor: bernoulli,
                    parameters: parameters.clone(),
                    observed,
                };
                let p_name = m.intern("p");
                let success = m.alloc(Node::Lit(Scalar::Bool(true)));
                let anchor = self.anchor(m, &bernoulli, &[(p_name, p_id)], Expr::Node(success));
                let safe_p = op("ifelse", [op("iszero", [total.clone()]), num(0.5), p]);
                op(
                    "add",
                    [
                        op("mul", [n, anchor]),
                        op("mul", [total, op("log1p", [op("neg", [safe_p])])]),
                    ],
                )
            }
            "Binomial" => {
                let Expr::Node(trials) = trials? else {
                    return None;
                };
                let trials_id = resolve(m, trials, &self.boundary);
                let trials = match m.node(trials_id) {
                    Node::Lit(Scalar::Int(n)) if *n > 0 && *n < 64 => *n as f64,
                    _ => return None,
                };
                if !fixed.as_ref().is_some_and(|v| {
                    v.iter()
                        .all(|x| *x >= 0.0 && *x <= trials && x.fract() == 0.0)
                }) {
                    return None;
                }
                let total = self.summary(m, observed, "sum", op("sum", [x.clone()]));
                let coefficients = self.summary(
                    m,
                    observed,
                    "binomial_coefficients",
                    op(
                        "sum",
                        [map(
                            "sub",
                            [
                                map(
                                    "sub",
                                    [
                                        op("loggamma", [num(trials + 1.0)]),
                                        map("loggamma", [map("add", [x.clone(), num(1.0)])]),
                                    ],
                                ),
                                map("loggamma", [map("sub", [num(trials + 1.0), x])]),
                            ],
                        )],
                    ),
                );
                let failures = op("sub", [op("mul", [n, num(trials)]), total.clone()]);
                let bernoulli_p = m.intern("p");
                let p = p?;
                let yes = weighted_bernoulli(m, bernoulli_p, p.clone(), total, true);
                let no = weighted_bernoulli(m, bernoulli_p, p, failures, false);
                op("add", [coefficients, op("add", [yes, no])])
            }
            "Uniform" => {
                let support = resolve(m, support?, &self.boundary);
                call(m, support, "interval")?;
                let min = self.summary(m, observed, "minimum", op("minimum", [x.clone()]));
                let max = self.summary(m, observed, "maximum", op("maximum", [x]));
                op(
                    "mul",
                    [
                        op("divide", [n, num(2.0)]),
                        op(
                            "add",
                            [
                                self.anchor(m, f, &parameters, min),
                                self.anchor(m, f, &parameters, max),
                            ],
                        ),
                    ],
                )
            }
            _ => return None,
        };
        self.count += 1;
        Some(materialize(m, result))
    }

    fn walk(
        &mut self,
        m: &mut Module,
        root: NodeId,
        mapped: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        if let Some(&id) = mapped.get(&root) {
            return id;
        }
        if matches!(m.node(root), Node::Call(c) if c.inputs.is_some()) {
            return root;
        }
        let children = m.node(root).children();
        let replacements: Vec<_> = children
            .iter()
            .map(|&child| self.walk(m, child, mapped))
            .collect();
        let id = if children == replacements {
            root
        } else {
            crate::driver::rebuild_with_children(m, root, &replacements)
        };
        let result = self.reduce(m, id).unwrap_or(id);
        mapped.insert(root, result);
        result
    }
}

fn weighted_bernoulli(m: &mut Module, p_name: Symbol, p: Expr, count: Expr, success: bool) -> Expr {
    let empty = op("iszero", [count.clone()]);
    let safe_p = materialize(m, op("ifelse", [empty.clone(), num(0.5), p]));
    let kernel = m.intern("Bernoulli");
    let kernel = m.alloc(Node::Const(kernel));
    let record = build_record(m, &[(p_name, safe_p)]);
    let point = m.alloc(Node::Lit(Scalar::Bool(success)));
    let density = op(
        "builtin_logdensityof",
        [Expr::Node(kernel), Expr::Node(record), Expr::Node(point)],
    );
    op("ifelse", [empty, num(0.0), op("mul", [count, density])])
}

fn fixed_values(m: &Module, root: NodeId, boundary: &HashSet<Symbol>) -> Option<Vec<f64>> {
    let id = resolve(m, root, boundary);
    let c = call(m, id, "vector")?;
    c.args
        .iter()
        .map(|&id| match m.node(resolve(m, id, boundary)) {
            Node::Lit(Scalar::Real(x)) => Some(*x),
            Node::Lit(Scalar::Int(x)) => Some(*x as f64),
            Node::Lit(Scalar::Bool(x)) => Some(if *x { 1.0 } else { 0.0 }),
            _ => None,
        })
        .collect()
}

fn reduce(m: &mut Module, allowed: HashSet<Symbol>, preparing: bool) -> (usize, Vec<Symbol>) {
    let boundary = crate::canon::promoted_input_names(m);
    let mut reducer = Reducer {
        preparing,
        boundary,
        allowed,
        summaries: Vec::new(),
        cache: HashMap::new(),
        count: 0,
        next_serial: 0,
    };
    let bindings: Vec<_> = m
        .bindings()
        .filter(|(bid, b)| {
            !crate::canon::is_reserved_abi_binding(m, *bid)
                && !m.resolve(b.name).starts_with(PREFIX)
        })
        .map(|(bid, b)| (bid, b.rhs))
        .collect();
    let mut mapped = HashMap::new();
    for (bid, rhs) in bindings {
        let rhs = reducer.walk(m, rhs, &mut mapped);
        m.set_binding_rhs(bid, rhs);
    }
    if reducer.count > 0 {
        let _ = flatppl_infer::infer(m);
    }
    (reducer.count, reducer.summaries)
}

/// Reduce shared-parameter independent likelihood factors to portable summary
/// expressions. This does not change the query ABI or cache mutable caller data.
/// Returns the number of reduced factors. Unsupported factors remain unchanged.
pub fn sufficient_statistics(m: &mut Module) -> usize {
    let allowed = crate::canon::promoted_input_names(m)
        .into_iter()
        .filter(|name| {
            m.binding_by_name(*name)
                .is_some_and(|bid| m.binding_phase(bid) == Some(flatppl_core::Phase::Fixed))
        })
        .collect();
    reduce(m, allowed, false).0
}

/// Split a FlatPDL query into preparation and scoring without changing ordinary
/// emission. Explicit data inputs may be fixed or parametric in the source.
pub fn prepare_likelihood(
    m: &Module,
    data_inputs: &[String],
) -> Result<PreparedLikelihood, RefuseError> {
    let error_node = binding(m, "outputs").unwrap_or_else(|| NodeId::from_usize(0));
    let refuse = |reason: &str| RefuseError {
        node: error_node,
        construct: "prepare_likelihood".into(),
        reason: reason.into(),
    };
    if binding(m, "outputs").is_none() {
        return Err(refuse("declare outputs before preparing a likelihood"));
    }
    let input_nodes = signature(m, "inputs");
    let input_names: HashSet<_> = input_nodes
        .iter()
        .filter_map(|id| match m.node(*id) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => Some(m.resolve(r.name).to_string()),
            _ => None,
        })
        .collect();
    if data_inputs.iter().any(|name| !input_names.contains(name))
        || data_inputs.iter().collect::<HashSet<_>>().len() != data_inputs.len()
    {
        return Err(refuse(
            "data inputs must name distinct declared query inputs",
        ));
    }
    let mut score = m.clone();
    let roots = [score.intern("inputs"), score.intern("outputs")];
    crate::canon::retain_reachable(&mut score, &roots);
    let allowed: HashSet<_> = data_inputs.iter().map(|name| score.intern(name)).collect();
    let (_, added) = reduce(&mut score, allowed.clone(), true);
    let boundary = crate::canon::promoted_input_names(&score);
    let mut summaries: Vec<_> = score
        .bindings()
        .filter(|(_, b)| {
            b.synthetic
                && score.resolve(b.name).starts_with(PREFIX)
                && data_only(&score, b.rhs, &allowed, &boundary)
        })
        .map(|(_, b)| b.name)
        .collect();
    for name in added {
        if !summaries.contains(&name) {
            summaries.push(name);
        }
    }
    if summaries.is_empty() {
        return Err(refuse(
            "no supported sufficient-statistic likelihood factor",
        ));
    }
    for &name in &summaries {
        let bid = score.binding_by_name(name).unwrap();
        if !data_only(&score, score.binding(bid).rhs, &allowed, &boundary) {
            return Err(refuse("a summary depends on a scoring input"));
        }
    }
    let mut preparation = score.clone();
    let outputs: Vec<_> = summaries
        .iter()
        .map(|&name| local_ref(&mut preparation, name))
        .collect();
    let inputs: Vec<_> = input_nodes
        .iter()
        .copied()
        .filter(|id| matches!(preparation.node(*id), Node::Ref(r) if allowed.contains(&r.name)))
        .collect();
    if inputs.is_empty() {
        let keep = preparation
            .bindings()
            .filter(|(_, b)| preparation.resolve(b.name) != "inputs")
            .map(|(bid, _)| bid)
            .collect();
        preparation.retain_bindings(&keep);
    } else {
        set_signature(&mut preparation, "inputs", &inputs);
    }
    set_signature(&mut preparation, "outputs", &outputs);
    for &name in &summaries {
        let bid = score.binding_by_name(name).unwrap();
        let ty = score
            .type_of(score.binding(bid).rhs)
            .cloned()
            .ok_or_else(|| refuse("a summary has no inferred type"))?;
        let set = summary_set(&mut score, &ty).ok_or_else(|| {
            refuse(&format!(
                "summary {} has unsupported type {ty:?}",
                score.resolve(name)
            ))
        })?;
        let rhs = build_call(&mut score, "external", &[set]);
        score.set_binding_rhs(bid, rhs);
    }
    let output = binding(&score, "outputs").unwrap();
    let needed: HashSet<_> = allowed
        .iter()
        .copied()
        .filter(|name| depends_on(&score, output, *name))
        .collect();
    let mut inputs: Vec<_> = input_nodes.into_iter().filter(|id| !matches!(score.node(*id), Node::Ref(r) if allowed.contains(&r.name) && !needed.contains(&r.name))).collect();
    inputs.extend(summaries.iter().map(|&name| local_ref(&mut score, name)));
    set_signature(&mut score, "inputs", &inputs);
    for module in [&mut preparation, &mut score] {
        let roots = [module.intern("inputs"), module.intern("outputs")];
        crate::canon::retain_reachable(module, &roots);
        let diagnostics = flatppl_infer::infer(module);
        if diagnostics
            .iter()
            .any(|d| matches!(d.severity, flatppl_infer::Severity::Error))
        {
            return Err(refuse(&format!(
                "summary inference failed: {diagnostics:?}"
            )));
        }
    }
    let summaries = summaries
        .iter()
        .map(|name| score.resolve(*name).to_string())
        .collect();
    Ok(PreparedLikelihood {
        preparation,
        score,
        summaries,
    })
}

fn summary_set(m: &mut Module, ty: &Type) -> Option<NodeId> {
    match ty {
        Type::Scalar(kind) => {
            let name = match kind {
                ScalarType::Integer => "integers",
                ScalarType::Boolean => "booleans",
                ScalarType::Real => "reals",
                _ => return None,
            };
            let name = m.intern(name);
            Some(m.alloc(Node::Const(name)))
        }
        Type::Array { shape, elem } => {
            let set = summary_set(m, elem)?;
            let dims = shape
                .iter()
                .map(|dim| match dim {
                    Dim::Static(n) if *n > 0 => {
                        Some(m.alloc(Node::Lit(Scalar::Int(i64::from(*n)))))
                    }
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            let size = if let [only] = dims.as_slice() {
                *only
            } else {
                build_call(m, "vector", &dims)
            };
            Some(build_call(m, "cartpow", &[set, size]))
        }
        _ => None,
    }
}

fn depends_on(m: &Module, root: NodeId, name: Symbol) -> bool {
    let mut stack = vec![root];
    let mut seen = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let Node::Ref(Ref {
            ns: RefNs::SelfMod,
            name: found,
        }) = m.node(id)
        {
            if *found == name {
                return true;
            }
            if let Some(bid) = m.binding_by_name(*found) {
                stack.push(m.binding(bid).rhs);
            }
        } else {
            stack.extend(m.node(id).children());
        }
    }
    false
}
