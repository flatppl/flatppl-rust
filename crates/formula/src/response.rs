use crate::{
    DataMode, DataValue, Error, Family, Formula, GroupLevel, Result, Table, data::ColumnModel,
};
use flatppl_convert::Builder;
use flatppl_core::{Call, CallHead, Inputs, Node, NodeId, Ref, RefNs};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Censoring {
    pub code: String,
    pub upper: Option<String>,
}

impl Censoring {
    pub(crate) fn parse(text: &str) -> Result<Self> {
        let name = |text: &str| crate::formula::name(text.trim());
        let (code, upper) = match text.split_once(',') {
            Some((code, upper)) => {
                let upper = upper.trim();
                let upper = upper
                    .strip_prefix("y2")
                    .and_then(|text| text.trim_start().strip_prefix('='))
                    .unwrap_or(upper);
                (name(code)?, Some(name(upper)?))
            }
            None => (name(text)?, None),
        };
        Ok(Self { code, upper })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Truncation {
    pub lower: Option<TruncationBound>,
    pub upper: Option<TruncationBound>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum TruncationBound {
    Constant(f64),
    Column(String),
}

impl Truncation {
    pub(crate) fn parse(text: &str) -> Result<Self> {
        let invalid =
            || Error("trunc() expects finite numbers or columns as `lb` and `ub` bounds".into());
        let mut bounds = [None, None];
        let mut assigned = [false; 2];
        for argument in text.split(',') {
            let (side, value) = match argument.split_once('=') {
                Some((key, value)) => match key.trim() {
                    "lb" => (0, value),
                    "ub" => (1, value),
                    _ => return Err(invalid()),
                },
                None => (
                    assigned.iter().position(|used| !used).ok_or_else(invalid)?,
                    argument,
                ),
            };
            if assigned[side] {
                return Err(invalid());
            }
            assigned[side] = true;
            let value = value.trim();
            bounds[side] = match value.parse::<f64>() {
                Ok(value) if value.is_finite() => Some(TruncationBound::Constant(value)),
                Ok(value)
                    if value
                        == if side == 0 {
                            f64::NEG_INFINITY
                        } else {
                            f64::INFINITY
                        } =>
                {
                    None
                }
                Ok(_) => return Err(invalid()),
                Err(_) => Some(TruncationBound::Column(
                    crate::formula::name(value).map_err(|_| invalid())?,
                )),
            };
        }
        let [lower, upper] = bounds;
        if lower.is_none() && upper.is_none() {
            return Err(invalid());
        }
        Ok(Self { lower, upper })
    }

    pub(crate) fn columns(&self) -> impl Iterator<Item = &str> {
        [&self.lower, &self.upper]
            .into_iter()
            .filter_map(|bound| match bound {
                Some(TruncationBound::Column(name)) => Some(name.as_str()),
                _ => None,
            })
    }

    fn named_bounds(&self) -> [(&str, &Option<TruncationBound>); 2] {
        [(LOWER, &self.lower), (UPPER, &self.upper)]
    }

    fn emit_bounds(
        &self,
        b: &mut Builder<'_>,
        mut column: impl FnMut(&mut Builder<'_>, &str) -> NodeId,
    ) -> [NodeId; 2] {
        self.named_bounds().map(|(name, bound)| match bound {
            Some(TruncationBound::Constant(value)) => b.lit_real(*value),
            Some(TruncationBound::Column(_)) => column(b, name),
            None => b.lit_real(if name == LOWER {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }),
        })
    }
}

const CENSOR_CODE: &str = "cens_data";
const CENSOR_UPPER: &str = "cens_upper_data";
const LOWER: &str = "trunc_lower_data";
const UPPER: &str = "trunc_upper_data";

pub(crate) fn prepare(
    formula: &Formula,
    family: Family,
    table: &Table,
    rows: &[usize],
    mode: DataMode,
    constants: &mut BTreeMap<String, DataValue>,
) -> Result<()> {
    let truncation = formula
        .truncation
        .as_ref()
        .map(|truncation| {
            let lower = bound_data(
                &truncation.lower,
                table,
                rows,
                LOWER,
                f64::NEG_INFINITY,
                constants,
            )?;
            let upper = bound_data(
                &truncation.upper,
                table,
                rows,
                UPPER,
                f64::INFINITY,
                constants,
            )?;
            if lower.iter().zip(&upper).any(|(lo, hi)| lo >= hi) {
                return Err(Error(
                    "truncation lower bounds must be smaller than upper bounds".into(),
                ));
            }
            Ok((lower, upper))
        })
        .transpose()?;
    if mode != DataMode::Training || (formula.censoring.is_none() && truncation.is_none()) {
        return Ok(());
    }

    let y = table.column(&formula.response)?.numeric(rows)?;
    if let Some(censoring) = &formula.censoring {
        let status = table.column(&censoring.code)?;
        let upper = censoring
            .upper
            .as_deref()
            .map(|name| table.column(name))
            .transpose()?;
        let mut codes = Vec::with_capacity(rows.len());
        let mut upper_values = Vec::with_capacity(rows.len());
        for (i, &row) in rows.iter().enumerate() {
            let code = censor_code(status.level(row)?)?;
            if code == 0 && y[i] == 0.0 && matches!(family, Family::Exponential | Family::Weibull) {
                return Err(Error(
                    "exact survival times must be strictly positive".into(),
                ));
            }
            let end = match code {
                2 => {
                    let upper = upper.filter(|col| col.present(row)).ok_or_else(|| {
                        Error(
                            "interval censoring requires a finite `y2` on each interval row".into(),
                        )
                    })?;
                    let end = upper.numeric(&[row])?[0];
                    if end <= y[i] {
                        return Err(Error(
                            "interval censoring requires `y2` greater than the response".into(),
                        ));
                    }
                    end
                }
                _ => y[i],
            };
            if let Some((lower, upper)) = &truncation {
                let (lo, hi) = (lower[i], upper[i]);
                let valid = match code {
                    -1 => y[i] > lo,
                    0 => lo <= y[i] && y[i] <= hi,
                    1 => y[i] < hi,
                    2 => lo.max(y[i]) < hi.min(end),
                    _ => unreachable!("censor status was validated"),
                };
                if !valid {
                    return Err(Error(
                        "a censored or exact observation has no overlap with its truncation range"
                            .into(),
                    ));
                }
            }
            codes.push(code);
            upper_values.push(end);
        }
        constants.insert(CENSOR_CODE.into(), DataValue::Integers(codes));
        if upper.is_some() {
            constants.insert(CENSOR_UPPER.into(), DataValue::Real64(upper_values));
        }
    } else if let Some((lower, upper)) = truncation
        && y.iter()
            .zip(lower.iter().zip(&upper))
            .any(|(y, (lo, hi))| y < lo || y > hi)
    {
        return Err(Error("a response is outside its truncation range".into()));
    }
    Ok(())
}

fn bound_data(
    bound: &Option<TruncationBound>,
    table: &Table,
    rows: &[usize],
    input: &str,
    infinite: f64,
    constants: &mut BTreeMap<String, DataValue>,
) -> Result<Vec<f64>> {
    Ok(match bound {
        Some(TruncationBound::Constant(value)) => vec![*value; rows.len()],
        Some(TruncationBound::Column(name)) => {
            let column = table.column(name)?;
            let values = column.numeric(rows)?;
            constants.insert(input.into(), column.data(rows)?);
            values
        }
        None => vec![infinite; rows.len()],
    })
}

fn censor_code(level: GroupLevel) -> Result<i64> {
    Ok(match level {
        GroupLevel::Text(name) => match name.as_str() {
            "left" => -1,
            "none" => 0,
            "right" => 1,
            "interval" => 2,
            _ => return Err(invalid_censor_code()),
        },
        GroupLevel::Integer(value) if (-1..=2).contains(&value) => value,
        GroupLevel::Real(value) if value.fract() == 0.0 && (-1.0..=2.0).contains(&value) => {
            value as i64
        }
        _ => return Err(invalid_censor_code()),
    })
}

fn invalid_censor_code() -> Error {
    Error("censor status must be left, none, right or interval (-1, 0, 1 or 2)".into())
}

pub(crate) fn score_columns(formula: &Formula) -> impl Iterator<Item = (&str, &str)> {
    let censor = formula.censoring.iter().flat_map(|censoring| {
        [(CENSOR_CODE, "integers"), (CENSOR_UPPER, "reals")]
            .into_iter()
            .filter(|(name, _)| *name == CENSOR_CODE || censoring.upper.is_some())
    });
    let truncation = formula.truncation.iter().flat_map(|truncation| {
        truncation
            .named_bounds()
            .into_iter()
            .filter_map(|(name, bound)| {
                matches!(bound, Some(TruncationBound::Column(_))).then_some((name, "reals"))
            })
    });
    censor.chain(truncation)
}

pub(crate) fn declare(b: &mut Builder<'_>, formula: &Formula) {
    for (name, set) in score_columns(formula) {
        crate::emit::external_vector(b, name, set);
    }
}

/// The forward response law is conditional on truncation. Censoring belongs
/// to the observed score and does not alter the underlying response law.
pub(crate) fn truncated_normal(
    b: &mut Builder<'_>,
    mean: NodeId,
    scale: NodeId,
    truncation: &Truncation,
) -> NodeId {
    truncated_normal_rows(
        b,
        mean,
        scale,
        truncation,
        "truncated_response",
        |b, mean, scale, bounds| {
            let normal = b.call("Normal", &[mean, scale]);
            let restricted = truncate(b, normal, bounds);
            b.call("normalize", &[restricted])
        },
    )
}

pub(crate) fn truncated_normal_rows(
    b: &mut Builder<'_>,
    mean: NodeId,
    scale: NodeId,
    truncation: &Truncation,
    name: &str,
    row: impl FnOnce(&mut Builder<'_>, NodeId, NodeId, [NodeId; 2]) -> NodeId,
) -> NodeId {
    let mut args = Vec::new();
    let mut inputs = Vec::new();
    let mut argument = |b: &mut Builder<'_>, name: &str, value| {
        let reference = Ref {
            ns: RefNs::Local,
            name: b.sym(&format!("_{name}_")),
        };
        args.push(value);
        inputs.push((b.sym(name), reference));
        b.m.alloc(Node::Ref(reference))
    };
    let mean = argument(b, "mu", mean);
    let scale = argument(b, "sigma", scale);
    let bounds = truncation.emit_bounds(b, |b, name| {
        let value = b.self_ref(name);
        argument(b, name, value)
    });
    let result = row(b, mean, scale, bounds);
    let head = CallHead::Builtin(b.sym("functionof"));
    let callback = b.m.alloc(Node::Call(Call {
        head,
        args: Box::new([result]),
        named: Box::default(),
        inputs: Some(Inputs::Spec(inputs.into())),
    }));
    let name = b.alloc_name(name);
    b.bind(&name, callback);
    let callback = b.self_ref(&name);
    args.insert(0, callback);
    b.call("broadcast", &args)
}

pub(crate) fn truncate(b: &mut Builder<'_>, law: NodeId, bounds: [NodeId; 2]) -> NodeId {
    let interval = b.call("interval", &bounds);
    b.call("truncate", &[law, interval])
}

pub(crate) fn log_mass(b: &mut Builder<'_>, restricted: NodeId) -> NodeId {
    let mass = b.call("totalmass", &[restricted]);
    b.call("log", &[mass])
}

pub(crate) fn score(
    b: &mut Builder<'_>,
    formula: &Formula,
    law: NodeId,
    value: NodeId,
    inputs: &[(&str, NodeId)],
) -> NodeId {
    let input = |name: &str| inputs.iter().find(|(key, _)| *key == name).unwrap().1;
    let limits = formula
        .truncation
        .as_ref()
        .map(|truncation| truncation.emit_bounds(b, |_, name| input(name)));
    let supported = limits.map_or(law, |bounds| truncate(b, law, bounds));
    let exact = b.call("logdensityof", &[supported, value]);
    let score = if let Some(censoring) = &formula.censoring {
        let code = input(CENSOR_CODE);
        let zero = b.lit_int(0);
        let two = b.lit_int(2);
        let negative = b.lit_real(f64::NEG_INFINITY);
        let positive = b.lit_real(f64::INFINITY);
        let right_or_interval = b.call("gt", &[code, zero]);
        let left = b.call("lt", &[code, zero]);
        let uncensored = b.call("equal", &[code, zero]);
        let lower = b.call("ifelse", &[right_or_interval, value, negative]);
        let upper = if censoring.upper.is_some() {
            let interval = b.call("equal", &[code, two]);
            b.call("ifelse", &[interval, input(CENSOR_UPPER), positive])
        } else {
            positive
        };
        let upper = b.call("ifelse", &[left, value, upper]);
        let event_bounds = if let Some([lo, hi]) = limits {
            [b.call("max", &[lo, lower]), b.call("min", &[hi, upper])]
        } else {
            [lower, upper]
        };
        let event = truncate(b, law, event_bounds);
        let event_score = log_mass(b, event);
        b.call("ifelse", &[uncensored, exact, event_score])
    } else {
        exact
    };
    if limits.is_some() {
        let log_z = log_mass(b, supported);
        b.call("sub", &[score, log_z])
    } else {
        score
    }
}

pub(crate) fn observed_likelihood(
    b: &mut Builder<'_>,
    rows: NodeId,
    parameters: &[String],
) -> NodeId {
    // A single fixed atom carries the observed event log likelihood without
    // turning censor intervals into point observations or altering K_model.
    let score = b.call("sum", &[rows]);
    let observed = b.lit_int(0);
    let atom = b.call("Dirac", &[observed]);
    let law = b.call("logweighted", &[score, atom]);
    let kernel = crate::emit::parameter_function(b, "functionof", law, parameters);
    b.call("likelihoodof", &[kernel, observed])
}
