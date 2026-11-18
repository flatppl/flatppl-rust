use crate::{
    autoregression::{ArmaOrder, DESIGN_MEAN, DESIGN_N, GAP},
    emit::{broadcast, external_vector, external_vector_sized},
};
use flatppl_convert::Builder;
use flatppl_core::{Call, CallHead, Inputs, Node, NodeId, Ref, RefNs};

#[derive(Clone, Copy)]
enum Direction {
    Responses,
    Innovations,
}

pub(crate) const ORDER: &str = "arma_order_data";
pub(crate) const FUTURE: &str = "arma_future_data";
pub(crate) const HISTORY: &str = "arma_history_data";
pub(crate) const STEP: &str = "arma_step_data";

pub(crate) fn declare_forecast(b: &mut Builder<'_>) {
    let size = b.call_head("posintegers");
    let external = b.call("external", &[size]);
    b.bind(DESIGN_N, external);
    external_vector(b, FUTURE, "posintegers");
    external_vector_sized(b, ORDER, "posintegers", DESIGN_N);
    external_vector_sized(b, HISTORY, "reals", DESIGN_N);
    external_vector_sized(b, STEP, "nonnegintegers", DESIGN_N);
}

pub(crate) fn draw(b: &mut Builder<'_>, ar: ArmaOrder, mu: NodeId, forecast: bool) -> NodeId {
    let zero = b.lit_real(0.0);
    let sigma = b.self_ref("sigma");
    let normal = b.call("Normal", &[zero, sigma]);
    let n = b.self_ref("N");
    let noise = b.call("iid", &[normal, n]);
    let values = b.call("draw", &[noise]);
    b.bind("arma_innovations", values);
    let values = b.self_ref("arma_innovations");
    recur(b, ar, Direction::Responses, values, mu, forecast)
}

pub(crate) fn forecast_mean(b: &mut Builder<'_>, ar: ArmaOrder, mu: NodeId) -> NodeId {
    let n = b.self_ref("N");
    let noise = b.call("zeros", &[n]);
    recur(b, ar, Direction::Responses, noise, mu, true)
}

pub(crate) fn pointwise(b: &mut Builder<'_>, ar: ArmaOrder, mu: NodeId, forecast: bool) -> NodeId {
    let y = b.self_ref("y_data");
    let errors = recur(b, ar, Direction::Innovations, y, mu, forecast);
    let error = local(b, "_arma_error_");
    let sigma = local(b, "_arma_sigma_");
    let value = b.m.alloc(Node::Ref(error));
    let sd = b.m.alloc(Node::Ref(sigma));
    let zero = b.lit_real(0.0);
    let law = b.call("Normal", &[zero, sd]);
    let score = b.call("logdensityof", &[law, value]);
    let score = callback(b, &[error, sigma], score);
    let sd = b.self_ref("sigma");
    b.call("broadcast", &[score, errors, sd])
}

fn recur(
    b: &mut Builder<'_>,
    ar: ArmaOrder,
    direction: Direction,
    mut values: NodeId,
    mut mu: NodeId,
    forecast: bool,
) -> NodeId {
    let mut fields = Vec::new();
    let gap = b.self_ref(if forecast { STEP } else { GAP });
    if forecast {
        let order = b.self_ref(ORDER);
        let means = b.self_ref(DESIGN_MEAN);
        mu = b.call("get", &[means, order]);
        let n = b.self_ref("N");
        let known = broadcast(b, "gt", &[order, n]);
        let index = broadcast(b, "min", &[order, n]);
        let observed = b.self_ref(HISTORY);
        values = b.call("get", &[values, index]);
        match direction {
            Direction::Responses => {
                let residual = broadcast(b, "sub", &[observed, mu]);
                fields.extend([("known", known), ("residual", residual)]);
            }
            Direction::Innovations => {
                values = broadcast(b, "ifelse", &[known, observed, values]);
            }
        }
    }
    let zero = b.lit_int(0);
    let reset = broadcast(b, "equal", &[gap, zero]);
    let values = match direction {
        Direction::Responses => values,
        Direction::Innovations => broadcast(b, "sub", &[values, mu]),
    };
    fields.extend([("reset", reset), ("value", values)]);
    let rows = b.call_fields("table", &fields);
    let state = local(b, "_arma_state_");
    let row = local(b, "_arma_row_");
    let coefficients = local(b, "_arma_coefficients_");
    let s = b.m.alloc(Node::Ref(state));
    let x = b.m.alloc(Node::Ref(row));
    let c = b.m.alloc(Node::Ref(coefficients));
    let reset = field(b, x, "reset");
    let zero = b.lit_real(0.0);
    let mut center = zero;
    let mut shifted = Vec::new();
    for (kind, order) in ar.coefficients() {
        let vector = field(b, c, kind);
        for lag in 1..=order {
            let name = format!("{kind}_{lag}");
            let lagged = field(b, s, &name);
            let lagged = b.call("ifelse", &[reset, zero, lagged]);
            let index = b.lit_int(lag as i64);
            let coefficient = b.call("get", &[vector, index]);
            let term = b.call("mul", &[coefficient, lagged]);
            center = b.call("add", &[center, term]);
            if lag < order {
                shifted.push((format!("{kind}_{}", lag + 1), lagged));
            }
        }
    }
    let value = field(b, x, "value");
    let (residual, innovation) = match direction {
        Direction::Responses => {
            let residual = b.call("add", &[center, value]);
            if forecast {
                let known = field(b, x, "known");
                let observed = field(b, x, "residual");
                let error = b.call("sub", &[observed, center]);
                (
                    b.call("ifelse", &[known, observed, residual]),
                    b.call("ifelse", &[known, error, value]),
                )
            } else {
                (residual, value)
            }
        }
        Direction::Innovations => (value, b.call("sub", &[value, center])),
    };
    let mut fields = vec![("ar_1".into(), residual), ("ma_1".into(), innovation)];
    fields.extend(shifted);
    let fields: Vec<_> = fields
        .iter()
        .map(|(name, value)| (name.as_str(), *value))
        .collect();
    let next = b.call_fields("record", &fields);
    let initial: Vec<_> = fields.iter().map(|(name, _)| (*name, zero)).collect();
    let initial = b.call_fields("record", &initial);
    let step = callback(b, &[state, row, coefficients], next);
    let parameters: Vec<_> = ar
        .coefficients()
        .map(|(kind, _)| (kind, b.self_ref(kind)))
        .collect();
    let parameters = b.call_fields("record", &parameters);
    let history = b.call("scan", &[step, initial, rows, parameters]);
    let result = match direction {
        Direction::Responses => {
            let residuals = field(b, history, "ar_1");
            broadcast(b, "add", &[mu, residuals])
        }
        Direction::Innovations => field(b, history, "ma_1"),
    };
    if forecast {
        let rows = b.self_ref(FUTURE);
        b.call("get", &[result, rows])
    } else {
        result
    }
}

fn field(b: &mut Builder<'_>, record: NodeId, name: &str) -> NodeId {
    let name = b.str_lit(name);
    b.call("get", &[record, name])
}

fn local(b: &mut Builder<'_>, name: &str) -> Ref {
    Ref {
        ns: RefNs::Local,
        name: b.sym(name),
    }
}

fn callback(b: &mut Builder<'_>, arguments: &[Ref], result: NodeId) -> NodeId {
    let head = b.sym("functionof");
    b.m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Box::new([result]),
        named: Box::default(),
        inputs: Some(Inputs::Spec(
            arguments.iter().map(|&r| (r.name, r)).collect(),
        )),
    }))
}
