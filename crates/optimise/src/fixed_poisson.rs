//! Partition a fixed indexed Poisson score into compact site scores.

use flatppl_core::graph::{build_call, expect_builtin_call};
use flatppl_core::{
    Call, CallHead, Dim, Module, NamedArg, NamedKind, Node, NodeId, Phase, Scalar, ScalarType, Type,
};
use std::collections::HashMap;

use crate::engine::{Fact, FactArg};

// Match the native Poisson direct branch. Every row at a site with a larger
// count keeps its native score and its original within-site pullback.
const COUNT_CUTOFF: usize = 64;
const MAX_ROWS: usize = 8192;
const MAX_COUNT: i64 = 255;

#[derive(Default)]
struct Group {
    size: usize,
    total: usize,
    log_factorial: f64,
    native: bool,
}

pub(super) struct Partition {
    pub source: NodeId,
    pub readers: [NodeId; 4],
    original: NodeId,
    rows: usize,
    guard_sites: Option<Vec<i64>>,
    summaries: Vec<(i64, Group)>,
    native: Vec<(i64, i64)>,
}

fn vector_width(m: &Module, id: NodeId, kind: ScalarType) -> Option<usize> {
    let Type::Array { shape, elem } = m.type_of(id)? else {
        return None;
    };
    let [Dim::Static(width)] = **shape else {
        return None;
    };
    (**elem == Type::Scalar(kind)).then_some(width as usize)
}

fn call<'a>(m: &'a Module, id: NodeId, head: &str) -> Option<&'a Call> {
    if m.is_sample_result(id) || m.sample_origin(id).is_some() {
        return None;
    }
    expect_builtin_call(m, id, head).filter(|call| call.inputs.is_none())
}

fn args<'a>(m: &'a Module, id: NodeId, head: &str) -> Option<&'a [NodeId]> {
    let call = call(m, id, head)?;
    call.named.is_empty().then_some(&call.args)
}

fn constant(m: &Module, id: NodeId, name: &str) -> bool {
    matches!(m.node(id), Node::Const(value) if m.resolve(*value) == name)
}

fn fixed_vector(
    m: &Module,
    mut id: NodeId,
    resolve: &impl Fn(NodeId) -> Option<NodeId>,
) -> Option<Vec<i64>> {
    let width = vector_width(m, id, ScalarType::Integer)?;
    if width > MAX_ROWS {
        return None;
    }
    for _ in 0..flatppl_core::DEFAULT_MAX_DEPTH {
        id = resolve(id)?;
        if m.phase_of(id) != Some(Phase::Fixed) {
            return None;
        }
        if let Some(elements) = args(m, id, "vector") {
            if elements.len() != width {
                return None;
            }
            return elements
                .iter()
                .map(|&element| {
                    let element = resolve(element)?;
                    match m.node(element) {
                        Node::Lit(Scalar::Int(value))
                            if !m.is_sample_result(element)
                                && m.sample_origin(element).is_none() =>
                        {
                            Some(i64::from(i32::try_from(*value).ok()?))
                        }
                        _ => None,
                    }
                })
                .collect();
        }
        let &[record, key] = args(m, id, "get")? else {
            return None;
        };
        let Node::Lit(Scalar::Str(field)) = m.node(resolve(key)?) else {
            return None;
        };
        let record = call(m, resolve(record)?, "record")?;
        if !record.args.is_empty() {
            return None;
        }
        id = record
            .named
            .iter()
            .find(|arg| arg.kind == NamedKind::Field && m.resolve(arg.name) == field.as_ref())?
            .value;
    }
    None
}

impl Partition {
    pub(super) fn prove(
        m: &Module,
        sum: NodeId,
        resolve: impl Fn(NodeId) -> Option<NodeId>,
    ) -> Option<Self> {
        let &[lanes] = args(m, sum, "sum")? else {
            return None;
        };
        let lanes = resolve(lanes)?;
        let &[head, kernel, record, observed] = args(m, lanes, "broadcast")? else {
            return None;
        };
        if !constant(m, resolve(head)?, "builtin_logdensityof")
            || !constant(m, resolve(kernel)?, "Poisson")
        {
            return None;
        }
        let records = resolve(record)?;
        let record = call(m, records, "broadcast")?;
        let ([record_head], [rate]) = (record.args.as_ref(), record.named.as_ref()) else {
            return None;
        };
        if !constant(m, resolve(*record_head)?, "record")
            || rate.kind != NamedKind::Kwarg
            || m.resolve(rate.name) != "rate"
        {
            return None;
        }
        let rate = resolve(rate.value)?;
        let &[exp, selected] = args(m, rate, "broadcast")? else {
            return None;
        };
        if !constant(m, resolve(exp)?, "exp") {
            return None;
        }
        let selected = resolve(selected)?;
        let (selection, origin) = args(m, selected, "get")
            .map(|args| (args, 1))
            .or_else(|| args(m, selected, "get0").map(|args| (args, 0)))?;
        let &[source, selector] = selection else {
            return None;
        };
        let width = vector_width(m, source, ScalarType::Real)?;
        if width == 0 || width > i32::MAX as usize {
            return None;
        }
        let y = fixed_vector(m, observed, &resolve)?;
        let indices = fixed_vector(m, selector, &resolve)?;
        if y.len() < 64
            || y.len() != indices.len()
            || y.iter().any(|&count| !(0..=MAX_COUNT).contains(&count))
        {
            return None;
        }
        let rows = y.len();
        let mut lookup = HashMap::new();
        let mut summaries = Vec::new();
        let mut table = Vec::with_capacity(rows);
        let mut log_factorial = [0.0; COUNT_CUTOFF];
        for n in 2..COUNT_CUTOFF {
            log_factorial[n] = log_factorial[n - 1] + (n as f64).ln();
        }
        for (index, count) in indices.into_iter().zip(y) {
            let site = index.checked_add(1 - origin)?;
            if !(1..=width as i64).contains(&site) {
                return None;
            }
            let at = *lookup.entry(site).or_insert_with(|| {
                summaries.push((site, Group::default()));
                summaries.len() - 1
            });
            let group = &mut summaries[at].1;
            if let Some(&log_factorial) = log_factorial.get(count as usize) {
                group.size += 1;
                group.total += count as usize;
                group.log_factorial += log_factorial;
            } else {
                group.native = true;
            }
            table.push((site, count));
        }
        let native: Vec<_> = table
            .into_iter()
            .filter(|(site, _)| summaries[lookup[site]].1.native)
            .collect();
        summaries.retain(|(_, group)| !group.native);
        let retained = summaries.len() + native.len();
        if retained * 4 > rows {
            return None;
        }
        let guard_sites = (lookup.len() != width).then(|| {
            let mut sites: Vec<_> = lookup.into_keys().collect();
            sites.sort_unstable();
            sites
        });
        Some(Self {
            source,
            readers: [lanes, records, rate, selected],
            original: sum,
            rows,
            guard_sites,
            summaries,
            native,
        })
    }

    pub(super) fn fact(&self, cells: usize, source: usize, key: usize) -> Fact {
        use FactArg::{Call, Expr, Int};
        Fact {
            name: "fixed-poisson-partition",
            args: vec![
                Expr(cells),
                Expr(source),
                Call("Leaf", vec![Int(key as i64)]),
                Int(self.rows as i64),
                Int((self.summaries.len() + self.native.len()) as i64),
            ],
        }
    }

    pub(super) fn rebuild(&self, m: &mut Module, source: NodeId) -> NodeId {
        // This interval keeps rates, their sums, and the log-rate pullback
        // for ≤8192 rows of ≤255 counts normal even at f32. Use the same
        // range at either precision so the rewritten module can be re-emitted.
        let bound = m.alloc(Node::Lit(Scalar::Real(60.0)));
        let used = if let Some(sites) = &self.guard_sites {
            let sites = vector(m, sites.iter().copied().map(Scalar::Int));
            build_call(m, "get", &[source, sites])
        } else {
            source
        };
        let magnitude = map(m, "abs", &[used]);
        let bounded = map(m, "lt", &[magnitude, bound]);
        let safe = build_call(m, "lall", &[bounded]);
        let compact = self.compact_score(m, source);
        let guard = build_call(m, "ifelse", &[safe, compact, self.original]);
        m.mark_guarded_likelihood(guard);
        guard
    }

    fn compact_score(&self, m: &mut Module, source: NodeId) -> NodeId {
        let sites = vector(m, self.summaries.iter().map(|(site, _)| Scalar::Int(*site)));
        let totals = vector(
            m,
            self.summaries
                .iter()
                .map(|(_, group)| Scalar::Real(group.total as f64)),
        );
        let counts = vector(
            m,
            self.summaries
                .iter()
                .map(|(_, group)| Scalar::Real(group.size as f64)),
        );
        let factorials = vector(
            m,
            self.summaries
                .iter()
                .map(|(_, group)| Scalar::Real(group.log_factorial)),
        );
        let rate = rates(m, source, sites);
        let log_rate = map(m, "log", &[rate]);
        let linear = map(m, "mul", &[totals, log_rate]);
        let exposure = map(m, "mul", &[counts, rate]);
        let unnormalised = map(m, "sub", &[linear, exposure]);
        let scores = map(m, "sub", &[unnormalised, factorials]);
        let score = build_call(m, "sum", &[scores]);
        if self.native.is_empty() {
            return score;
        }
        let sites = vector(m, self.native.iter().map(|&(site, _)| Scalar::Int(site)));
        let observed = vector(m, self.native.iter().map(|&(_, count)| Scalar::Int(count)));
        let rate = rates(m, source, sites);
        let record = head(m, "record");
        let rate_name = m.intern("rate");
        let broadcast = m.intern("broadcast");
        let parameters = m.alloc(Node::Call(Call {
            head: CallHead::Builtin(broadcast),
            args: Box::new([record]),
            named: Box::new([NamedArg {
                kind: NamedKind::Kwarg,
                name: rate_name,
                value: rate,
            }]),
            inputs: None,
        }));
        let poisson = head(m, "Poisson");
        let native = map(m, "builtin_logdensityof", &[poisson, parameters, observed]);
        let native = build_call(m, "sum", &[native]);
        build_call(m, "add", &[score, native])
    }
}

fn head(m: &mut Module, name: &str) -> NodeId {
    let head = m.intern(name);
    m.alloc(Node::Const(head))
}

fn vector(m: &mut Module, values: impl Iterator<Item = Scalar>) -> NodeId {
    let values: Vec<_> = values.map(|value| m.alloc(Node::Lit(value))).collect();
    build_call(m, "vector", &values)
}

fn rates(m: &mut Module, source: NodeId, sites: NodeId) -> NodeId {
    let selected = build_call(m, "get", &[source, sites]);
    map(m, "exp", &[selected])
}

fn map(m: &mut Module, op: &str, values: &[NodeId]) -> NodeId {
    let op = head(m, op);
    let args: Vec<_> = std::iter::once(op).chain(values.iter().copied()).collect();
    build_call(m, "broadcast", &args)
}
