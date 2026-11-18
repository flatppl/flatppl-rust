//! Predictive count laws and the probability of the observed allocation given the total.

use super::{Data, Evidence, UpdatedLaw, op, real, safe};
use crate::posterior::{category_counts, map, multinomial_data, supported_counts};
use flatppl_core::graph::build_record;
use flatppl_core::{Call, CallHead, Inputs, Module, Node, NodeId, Ref, RefNs, Scalar};

/// Apply a native scalar density to the supplied columns. Pass every argument
/// explicitly so its reified function has no captures.
fn density(
    m: &mut Module,
    name: &str,
    parameters: &[(&str, NodeId)],
    y: NodeId,
    broadcast: bool,
) -> NodeId {
    let mut operands = Vec::new();
    let mut inputs = Vec::new();
    let mut argument = |m: &mut Module, name: &str, value| {
        if !broadcast {
            return value;
        }
        operands.push(value);
        let local = Ref {
            ns: RefNs::Local,
            name: m.intern(&format!("_evidence_{name}_")),
        };
        let name = m.intern(name);
        inputs.push((name, local));
        m.alloc(Node::Ref(local))
    };
    let mut fields = Vec::new();
    for &(param, value) in parameters {
        let value = argument(m, param, value);
        fields.push((m.intern(param), value));
    }
    let y = argument(m, "count", y);
    let record = build_record(m, &fields);
    let family = m.intern(name);
    let family = m.alloc(Node::Const(family));
    let score = op(m, "builtin_logdensityof", &[family, record, y]);
    if !broadcast {
        return score;
    }
    let head = m.intern("functionof");
    let function = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Box::new([score]),
        named: Box::new([]),
        inputs: Some(Inputs::Spec(inputs.into())),
    }));
    operands.insert(0, function);
    op(m, "broadcast", &operands)
}

/// Each zero-rate cell here has count zero and contributes log(1).
fn allocation_poisson(m: &mut Module, rate: NodeId, y: NodeId, array: bool) -> NodeId {
    let apply = if array { map } else { op };
    let zero = real(m, 0.);
    let one = real(m, 1.);
    let positive = apply(m, "gt", &[rate, zero]);
    let safe_rate = apply(m, "ifelse", &[positive, rate, one]);
    let score = density(m, "Poisson", &[("rate", safe_rate)], y, array);
    apply(m, "ifelse", &[positive, score, zero])
}

pub(super) fn poisson(
    m: &mut Module,
    plan: &mut Evidence,
    law: &mut UpdatedLaw,
    prior: [NodeId; 2],
    data: Data,
    valid: NodeId,
) -> (NodeId, NodeId) {
    let observed = supported_counts(m, data.value, data.integer, None, data.count.is_some());
    let observed = plan.summary(m, observed);
    let zero = real(m, 0.);
    let y = data.apply(m, "real", &[data.value]);
    let y = data.apply(m, "ifelse", &[observed, y, zero]);
    let total = data.sum(m, y);
    let total = plan.summary(m, total);
    let n = data.count(m);
    let a = safe(m, valid, prior[0], 1.);
    let b = safe(m, valid, prior[1], 1.);
    let rate = op(m, "divide", &[b, n]);
    let pooled = density(
        m,
        "NegativeBinomial",
        &[("alpha", a), ("beta", rate)],
        total,
        false,
    );
    let score = if data.count.unwrap_or(1) == 1 {
        pooled
    } else {
        // Independent Poisson counts conditional on their sum are multinomial.
        let mean = op(m, "divide", &[total, n]);
        let cells = allocation_poisson(m, mean, y, true);
        let sum = op(m, "sum", &[cells]);
        let pooled_at_mean = allocation_poisson(m, total, total, false);
        let correction = op(m, "sub", &[sum, pooled_at_mean]);
        let correction = plan.summary(m, correction);
        op(m, "add", &[pooled, correction])
    };
    *law = UpdatedLaw::Gamma(op(m, "add", &[a, total]), op(m, "add", &[b, n]));
    (score, observed)
}

fn category_allocation(m: &mut Module, counts: NodeId, n: NodeId) -> NodeId {
    let zero = real(m, 0.);
    let one = real(m, 1.);
    let share = map(m, "divide", &[counts, n]);
    let positive = map(m, "gt", &[counts, zero]);
    let share = map(m, "ifelse", &[positive, share, one]);
    let logs = map(m, "log", &[share]);
    let ordered = map(m, "mul", &[counts, logs]);
    let ordered = op(m, "sum", &[ordered]);
    let by_cell = allocation_poisson(m, counts, counts, true);
    let by_cell = op(m, "sum", &[by_cell]);
    let pooled = allocation_poisson(m, n, n, false);
    let coefficient = op(m, "sub", &[by_cell, pooled]);
    op(m, "sub", &[ordered, coefficient])
}

fn multinomial(
    m: &mut Module,
    plan: &mut Evidence,
    data: Data,
    d: u32,
    trials: NodeId,
) -> (NodeId, NodeId, NodeId) {
    let size = m.alloc(Node::Lit(Scalar::Int(d.into())));
    let raw = multinomial_data(m, data, size);
    // Check each row once against the first row, retaining its exact integer
    // total for the online comparison with the model's trial count.
    let row_size = if data.count.is_some() {
        let first = m.alloc(Node::Lit(Scalar::Int(1)));
        op(m, "get", &[raw.totals, first])
    } else {
        raw.totals
    };
    let row_size = plan.summary(m, row_size);
    let supported = raw.supported(m, data, row_size);
    let supported = plan.summary(m, supported);
    let difference = op(m, "sub", &[row_size, trials]);
    let matches = op(m, "iszero", &[difference]);
    let observed = op(m, "land", &[supported, matches]);
    let zero = real(m, 0.);
    let counts = map(m, "ifelse", &[supported, raw.counts, zero]);
    let counts = plan.summary(m, counts);
    if data.count.unwrap_or(1) == 1 || d == 1 {
        return (counts, observed, zero);
    }

    // Conditional on the margins, each allocation has the probability given by
    // independent Poisson cells divided by the column and row probabilities.
    let mrows = data.count(m);
    let rates = map(m, "divide", &[counts, mrows]);
    let one_axis = m.alloc(Node::Lit(Scalar::Int(1)));
    let no_axis = m.alloc(Node::Lit(Scalar::Int(0)));
    let rates = op(m, "addaxes", &[rates, one_axis, no_axis]);
    let y = map(m, "real", &[raw.cells]);
    let y = map(m, "ifelse", &[supported, y, zero]);
    let cells = allocation_poisson(m, rates, y, true);
    let cells = op(m, "sum", &[cells]);
    let columns = allocation_poisson(m, counts, counts, true);
    let columns = op(m, "sum", &[columns]);
    let total = op(m, "sum", &[counts]);
    let row_size = op(m, "real", &[row_size]);
    let row_size = safe(m, supported, row_size, 0.);
    let row = allocation_poisson(m, row_size, row_size, false);
    let rows = op(m, "mul", &[mrows, row]);
    let total_score = allocation_poisson(m, total, total, false);
    let log_allocation = op(m, "sub", &[cells, columns]);
    let margins = op(m, "sub", &[rows, total_score]);
    let correction = op(m, "sub", &[log_allocation, margins]);
    (counts, observed, plan.summary(m, correction))
}

pub(super) fn categories(
    m: &mut Module,
    plan: &mut Evidence,
    law: &mut UpdatedLaw,
    alpha: NodeId,
    data: Data,
    shape: (u32, Option<NodeId>, bool),
    valid: NodeId,
) -> (NodeId, NodeId) {
    let (d, trials, one_based) = shape;
    let zero = real(m, 0.);
    let (counts, observed, correction) = if let Some(n) = trials {
        multinomial(m, plan, data, d, n)
    } else {
        let (counts, observed) = category_counts(m, data, d, None, one_based);
        let counts = plan.summary(m, counts);
        let observed = plan.summary(m, observed);
        let correction = if data.count.unwrap_or(1) == 1 || d == 1 {
            zero
        } else {
            let n = data.count(m);
            let value = category_allocation(m, counts, n);
            plan.summary(m, value)
        };
        (counts, observed, correction)
    };
    let one = real(m, 1.);
    let alpha = map(m, "ifelse", &[valid, alpha, one]);
    *law = UpdatedLaw::Dirichlet(op(m, "add", &[alpha, counts]));
    if d == 1 {
        return (zero, observed);
    }
    if trials.is_none() && data.count.unwrap_or(1) == 1 {
        // One label has probability alpha[i]/sum(alpha). Scale by the largest
        // shape so even finite concentrations with an overflowing sum work.
        let largest = op(m, "maximum", &[alpha]);
        let scaled = map(m, "divide", &[alpha, largest]);
        let total = op(m, "sum", &[scaled]);
        let selected = map(m, "mul", &[counts, alpha]);
        let selected = op(m, "sum", &[selected]);
        let selected = safe(m, observed, selected, 1.);
        let share = op(m, "divide", &[selected, largest]);
        let positive = op(m, "gt", &[share, zero]);
        let safe_share = safe(m, positive, share, 1.);
        let log_share = op(m, "log", &[safe_share]);
        let ls = op(m, "log", &[selected]);
        let lm = op(m, "log", &[largest]);
        let difference = op(m, "sub", &[ls, lm]);
        let log_share = op(m, "ifelse", &[positive, log_share, difference]);
        let normalizer = op(m, "log", &[total]);
        return (op(m, "sub", &[log_share, normalizer]), observed);
    }
    let total = op(m, "sum", &[counts]);
    let shape = op(m, "sum", &[alpha]);
    let n = op(m, "max", &[total, one]);
    // Independent negative binomials with a shared rate, conditional on their
    // sum, have the Dirichlet-multinomial law. Center the total at its mean.
    let rate = op(m, "divide", &[shape, n]);
    let by_category = density(
        m,
        "NegativeBinomial",
        &[("alpha", alpha), ("beta", rate)],
        counts,
        true,
    );
    let by_category = op(m, "sum", &[by_category]);
    let pooled = density(
        m,
        "NegativeBinomial",
        &[("alpha", shape), ("beta", rate)],
        total,
        false,
    );
    let score = op(m, "sub", &[by_category, pooled]);
    (op(m, "add", &[score, correction]), observed)
}
