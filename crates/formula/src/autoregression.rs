use crate::{Error, Family, Formula, Result, emit::broadcast};
use flatppl_convert::Builder;
use flatppl_core::{Call, CallHead, Inputs, Node, NodeId, Ref, RefNs};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Ar1 {
    pub time: String,
    pub group: String,
}

impl Ar1 {
    pub(crate) fn parse(text: &str) -> Result<Self> {
        let invalid = || Error("ar() expects `time = column, group = column`".into());
        let mut time = None;
        let mut group = None;
        for argument in text.split(',') {
            let (keyword, value) = argument.split_once('=').ok_or_else(invalid)?;
            let column = crate::formula::name(value.trim()).map_err(|_| invalid())?;
            match keyword.trim() {
                "time" if time.is_none() => time = Some(column),
                "group" if group.is_none() => group = Some(column),
                _ => return Err(invalid()),
            }
        }
        Ok(Self {
            time: time.ok_or_else(invalid)?,
            group: group.ok_or_else(invalid)?,
        })
    }

    pub(crate) fn validate(
        &self,
        formula: &Formula,
        family: Family,
        sigma_predictor: bool,
    ) -> Result<()> {
        if family != Family::Gaussian
            || sigma_predictor
            || formula.se.is_some()
            || formula.censoring.is_some()
            || formula.truncation.is_some()
        {
            return Err(Error(
                "ar() requires a Gaussian response with scalar sigma and without se(), cens() or trunc()".into(),
            ));
        }
        if self.time == self.group
            || self.time == formula.response
            || self.group == formula.response
        {
            return Err(Error(
                "ar() response, time and group columns must be distinct".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) const GAP: &str = "ar_gap_data";
pub(crate) const PREVIOUS: &str = "ar_previous_data";
pub(crate) const DESIGN_N: &str = "ar_design_N";
pub(crate) const DESIGN_MEAN: &str = "ar_design_mu";
pub(crate) const ANCHOR: &str = "ar_anchor_data";
pub(crate) const ELAPSED: &str = "ar_elapsed_data";
pub(crate) const OBSERVED: &str = "ar_observed_data";

pub(crate) fn declare(b: &mut Builder<'_>, forecast: bool) {
    crate::emit::external_vector(b, GAP, "nonnegintegers");
    crate::emit::external_vector(b, PREVIOUS, "posintegers");
    if forecast {
        let size_set = b.call_head("posintegers");
        let external = b.call("external", &[size_set]);
        b.bind(DESIGN_N, external);
        crate::emit::external_vector(b, ANCHOR, "posintegers");
        crate::emit::external_vector(b, ELAPSED, "nonnegintegers");
        crate::emit::external_vector(b, OBSERVED, "reals");
    }
}

pub(crate) fn law(b: &mut Builder<'_>, mu: NodeId, forecast: bool) -> NodeId {
    let previous = b.self_ref(PREVIOUS);
    let previous_mu = previous_mean(b, mu, previous, forecast);
    let (phi, sd) = transitions(b);
    let mut predictors = vec![
        ("mu", mu),
        ("previous_mu", previous_mu),
        ("phi", phi),
        ("sd", sd),
    ];
    if forecast {
        let n = b.self_ref("N");
        let reset = broadcast(b, "gt", &[previous, n]);
        predictors.extend([("reset", reset), ("observed", b.self_ref(OBSERVED))]);
    }
    let predictors = b.call_fields("table", &predictors);
    let state = Ref {
        ns: RefNs::Local,
        name: b.sym("_previous_y_"),
    };
    let row = Ref {
        ns: RefNs::Local,
        name: b.sym("_ar_row_"),
    };
    let y = b.m.alloc(Node::Ref(state));
    let row_value = b.m.alloc(Node::Ref(row));
    let mut field = |name| {
        let name = b.str_lit(name);
        b.call("get", &[row_value, name])
    };
    let [mu, previous_mu, phi, sd] = ["mu", "previous_mu", "phi", "sd"].map(&mut field);
    let y = if forecast {
        let reset = field("reset");
        let observed = field("observed");
        b.call("ifelse", &[reset, observed, y])
    } else {
        y
    };
    let center = conditional_mean(b, mu, previous_mu, phi, y);
    let normal = b.call("Normal", &[center, sd]);
    let head = b.sym("functionof");
    let state_argument = b.sym("previous_y");
    let row_argument = b.sym("row");
    let step = b.m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Box::new([normal]),
        named: Box::default(),
        inputs: Some(Inputs::Spec(Box::new([
            (state_argument, state),
            (row_argument, row),
        ]))),
    }));
    let initial = b.lit_real(0.0);
    b.call("kscan", &[step, initial, predictors])
}

pub(crate) fn pointwise(b: &mut Builder<'_>, mu: NodeId, forecast: bool) -> NodeId {
    let y = b.self_ref("y_data");
    let previous = b.self_ref(PREVIOUS);
    let previous_y = if forecast {
        let n = b.self_ref("N");
        let reset = broadcast(b, "gt", &[previous, n]);
        let one = b.lit_int(1);
        let index = broadcast(b, "ifelse", &[reset, one, previous]);
        let value = b.call("get", &[y, index]);
        let observed = b.self_ref(OBSERVED);
        broadcast(b, "ifelse", &[reset, observed, value])
    } else {
        b.call("get", &[y, previous])
    };
    let previous_mu = previous_mean(b, mu, previous, forecast);
    let (phi, sd) = transitions(b);
    let center = conditional_mean(b, mu, previous_mu, phi, previous_y);
    let value_argument = b.sym("value");
    let center_argument = b.sym("center");
    let sd_argument = b.sym("sd");
    let [value, center_ref, sd_ref] = ["_y_", "_center_", "_sd_"].map(|name| Ref {
        ns: RefNs::Local,
        name: b.sym(name),
    });
    let [observed, loc, scale] = [value, center_ref, sd_ref].map(|r| b.m.alloc(Node::Ref(r)));
    let normal = b.call("Normal", &[loc, scale]);
    let score = b.call("logdensityof", &[normal, observed]);
    let head = b.sym("functionof");
    let score = b.m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Box::new([score]),
        named: Box::default(),
        inputs: Some(Inputs::Spec(Box::new([
            (value_argument, value),
            (center_argument, center_ref),
            (sd_argument, sd_ref),
        ]))),
    }));
    b.call("broadcast", &[score, y, center, sd])
}

pub(crate) fn forecast_mean(b: &mut Builder<'_>, mu: NodeId) -> NodeId {
    let elapsed = b.self_ref(ELAPSED);
    let rho = b.self_ref("ar");
    let (log_rho, zero_rho) = log_magnitude(b, rho);
    let phi = correlation(b, elapsed, rho, log_rho, zero_rho);
    let observed = b.self_ref(OBSERVED);
    let anchor = b.self_ref(ANCHOR);
    let anchor_mu = previous_mean(b, mu, anchor, true);
    conditional_mean(b, mu, anchor_mu, phi, observed)
}

fn previous_mean(b: &mut Builder<'_>, mu: NodeId, previous: NodeId, forecast: bool) -> NodeId {
    let all_mu = if forecast {
        b.self_ref(DESIGN_MEAN)
    } else {
        mu
    };
    b.call("get", &[all_mu, previous])
}

fn conditional_mean(
    b: &mut Builder<'_>,
    mu: NodeId,
    previous_mu: NodeId,
    phi: NodeId,
    y: NodeId,
) -> NodeId {
    let residual = broadcast(b, "sub", &[y, previous_mu]);
    let correction = broadcast(b, "mul", &[phi, residual]);
    broadcast(b, "add", &[mu, correction])
}

fn log_magnitude(b: &mut Builder<'_>, rho: NodeId) -> (NodeId, NodeId) {
    let half = b.lit_real(0.5);
    let zero_rho = b.call("iszero", &[rho]);
    let magnitude = b.call("abs", &[rho]);
    let safe = b.call("ifelse", &[zero_rho, half, magnitude]);
    (b.call("log", &[safe]), zero_rho)
}

fn correlation(
    b: &mut Builder<'_>,
    gap: NodeId,
    rho: NodeId,
    log_rho: NodeId,
    zero_rho: NodeId,
) -> NodeId {
    let zero = b.lit_int(0);
    let one = b.lit_int(1);
    let two = b.lit_int(2);
    let neg_one = b.lit_real(-1.0);
    let start = broadcast(b, "equal", &[gap, zero]);
    let single = broadcast(b, "equal", &[gap, one]);
    let exponent = broadcast(b, "mul", &[gap, log_rho]);
    let magnitude_gap = broadcast(b, "exp", &[exponent]);
    let parity = broadcast(b, "mod", &[gap, two]);
    let odd = broadcast(b, "equal", &[parity, one]);
    let negative_rho = b.call("lt", &[rho, zero]);
    let negative = broadcast(b, "land", &[negative_rho, odd]);
    let sign = broadcast(b, "ifelse", &[negative, neg_one, one]);
    let phi = broadcast(b, "mul", &[sign, magnitude_gap]);
    let phi = broadcast(b, "ifelse", &[zero_rho, zero, phi]);
    // At lag one the derivative of phi with respect to rho is one even at zero.
    let phi = broadcast(b, "ifelse", &[single, rho, phi]);
    broadcast(b, "ifelse", &[start, zero, phi])
}

fn transitions(b: &mut Builder<'_>) -> (NodeId, NodeId) {
    let gap = b.self_ref(GAP);
    let rho = b.self_ref("ar");
    let sigma = b.self_ref("sigma");
    let zero = b.lit_int(0);
    let one = b.lit_int(1);
    let two = b.lit_int(2);
    let neg_one = b.lit_real(-1.0);
    let start = broadcast(b, "equal", &[gap, zero]);
    let (log_rho, zero_rho) = log_magnitude(b, rho);
    let phi = correlation(b, gap, rho, log_rho, zero_rho);
    // With unit innovation variance, the lag-d transition variance is the
    // geometric sum of d terms and the initial variance is its infinite limit.
    let stable_gap = broadcast(b, "ifelse", &[start, one, gap]);
    let log_rho_sq = b.call("mul", &[two, log_rho]);
    let exponent_sq = broadcast(b, "mul", &[stable_gap, log_rho_sq]);
    let numerator = broadcast(b, "expm1", &[exponent_sq]);
    let denominator = b.call("expm1", &[log_rho_sq]);
    let transition = broadcast(b, "divide", &[numerator, denominator]);
    let transition = broadcast(b, "ifelse", &[zero_rho, one, transition]);
    let stationary = b.call("divide", &[neg_one, denominator]);
    let stationary = b.call("ifelse", &[zero_rho, one, stationary]);
    let variance = broadcast(b, "ifelse", &[start, stationary, transition]);
    let multiplier = broadcast(b, "sqrt", &[variance]);
    let sd = broadcast(b, "mul", &[sigma, multiplier]);
    (phi, sd)
}
