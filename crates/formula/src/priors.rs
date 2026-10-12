use crate::{Error, Formula, Result, Spec};
use flatppl_core::{CallHead, Module, Node, NodeId, Scalar};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Prior {
    pub class: String,
    pub prior: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct NormalPrior {
    pub mean: f64,
    pub sd: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Priors {
    pub intercept_location: f64,
    pub scale: f64,
    pub b: Option<NormalPrior>,
    pub intercept: Option<NormalPrior>,
    pub sigma: Option<NormalPrior>,
    pub sd: Vec<Option<NormalPrior>>,
}

pub(crate) fn resolve(spec: &Spec, formula: &Formula, response: &[f64]) -> Result<Priors> {
    let location = median(response.to_vec());
    let mad = 1.4826 * median(response.iter().map(|y| (y - location).abs()).collect());
    let mut out = Priors {
        intercept_location: round_decimal(location),
        scale: round_decimal(mad).max(2.5),
        b: None,
        intercept: None,
        sigma: None,
        sd: vec![None; formula.groups.len()],
    };
    let mut selected = std::collections::HashSet::new();
    for prior in &spec.priors {
        if !selected.insert((&prior.class, &prior.group)) {
            return Err(Error(format!(
                "duplicate prior for class `{}` and group {:?}",
                prior.class, prior.group
            )));
        }
        let normal = normal(&prior.prior)?;
        match (prior.class.as_str(), prior.group.as_deref()) {
            ("b", None) if !formula.fixed.is_empty() => out.b = Some(normal),
            ("Intercept", None) if formula.intercept => out.intercept = Some(normal),
            ("sigma", None) => out.sigma = Some(normal),
            ("sd", None) if !formula.groups.is_empty() => {}
            ("sd", Some(group)) if formula.groups.iter().any(|v| v == group) => {}
            _ => {
                return Err(Error(format!(
                    "prior selector class `{}` group {:?} matches no supported parameter",
                    prior.class, prior.group
                )));
            }
        }
    }
    // Group-specific priors override the class prior, independent of input order.
    for (index, group) in formula.groups.iter().enumerate() {
        let prior = spec
            .priors
            .iter()
            .find(|p| p.class == "sd" && p.group.as_deref() == Some(group))
            .or_else(|| {
                spec.priors
                    .iter()
                    .find(|p| p.class == "sd" && p.group.is_none())
            });
        out.sd[index] = prior.map(|p| normal(&p.prior)).transpose()?;
    }
    if !out.scale.is_finite() || !out.intercept_location.is_finite() {
        return Err(Error(
            "response statistics exceed the finite Gaussian prior range".into(),
        ));
    }
    Ok(out)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        values[middle - 1] / 2.0 + values[middle] / 2.0
    } else {
        values[middle]
    }
}

// Compare distances in the input scale rather than rounding x*10, which can
// erase the binary rounding difference relevant to R's round(x, digits=1).
fn round_decimal(value: f64) -> f64 {
    let x = value.abs();
    let index = (x * 10.0).floor();
    let lower = index / 10.0;
    let upper = (index + 1.0) / 10.0;
    let distance = x - lower;
    let rounded = if distance > upper - x || (distance == upper - x && index % 2.0 == 1.0) {
        upper
    } else {
        lower
    };
    rounded.copysign(value)
}

fn normal(text: &str) -> Result<NormalPrior> {
    let text = text.trim();
    let text = text
        .strip_prefix("normal(")
        .map(|rest| format!("Normal({rest}"))
        .unwrap_or_else(|| text.into());
    let parsed = flatppl_syntax::parse(&format!("prior = {text}"))
        .map_err(|e| Error(format!("invalid prior `{text}`: {e}")))?;
    let fail = || {
        Error(format!(
            "unsupported prior `{text}`; this slice accepts Normal(mean, sd)"
        ))
    };
    let bindings: Vec<_> = parsed.public_bindings().collect();
    if bindings.len() != 1 {
        return Err(fail());
    }
    let Node::Call(call) = parsed.node(bindings[0].1.rhs) else {
        return Err(fail());
    };
    if !matches!(call.head, CallHead::Builtin(s) if parsed.resolve(s) == "Normal")
        || call.args.len() != 2
        || !call.named.is_empty()
    {
        return Err(fail());
    }
    let mean = scalar(&parsed, call.args[0]).ok_or_else(fail)?;
    let sd = scalar(&parsed, call.args[1]).ok_or_else(fail)?;
    if !mean.is_finite() || !sd.is_finite() || sd <= 0.0 {
        return Err(Error(format!(
            "prior `{text}` requires finite mean and positive finite scale"
        )));
    }
    Ok(NormalPrior { mean, sd })
}

fn scalar(module: &Module, id: NodeId) -> Option<f64> {
    match module.node(id) {
        Node::Lit(Scalar::Real(v)) => Some(*v),
        Node::Lit(Scalar::Int(v)) => Some(*v as f64),
        Node::Call(call)
            if matches!(call.head, CallHead::Builtin(s) if module.resolve(s) == "neg")
                && call.args.len() == 1 =>
        {
            scalar(module, call.args[0]).map(|v| -v)
        }
        _ => None,
    }
}
