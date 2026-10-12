use crate::{Design, Error, Formula, Result, Spec};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

pub(crate) use crate::prior_expr::{PriorNode, emit};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Prior {
    pub class: String,
    pub prior: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coef: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dpar: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lb: Option<PriorBound>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ub: Option<PriorBound>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum PriorBound {
    Number(f64),
    Expression(String),
}

/// One parameter's prior after class and coefficient selectors have resolved.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ResolvedPrior {
    pub prior: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lb: Option<PriorBound>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ub: Option<PriorBound>,
}

impl ResolvedPrior {
    pub fn new(prior: impl Into<String>) -> Self {
        Self {
            prior: prior.into(),
            lb: None,
            ub: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Priors {
    pub intercept_location: f64,
    pub scale: f64,
    pub sigma_scale: f64,
    pub intercept: ResolvedPrior,
    pub b: Vec<ResolvedPrior>,
    pub sd: Vec<ResolvedPrior>,
    pub aux: BTreeMap<String, ResolvedPrior>,
}

pub(crate) struct PriorTarget<'a> {
    pub class: &'a str,
    pub coef: Option<&'a str>,
    pub group: Option<&'a str>,
    pub dpar: Option<&'a str>,
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.filter(|value| !value.is_empty())
}

impl Prior {
    fn matches(&self, target: &PriorTarget<'_>) -> bool {
        self.class == target.class
            && [
                (self.coef.as_deref(), target.coef),
                (self.group.as_deref(), target.group),
                (self.dpar.as_deref(), target.dpar),
            ]
            .iter()
            .all(|&(selector, value)| nonempty(selector).is_none_or(|s| Some(s) == value))
    }

    fn specificity(&self) -> (bool, bool, bool) {
        (
            nonempty(self.coef.as_deref()).is_some(),
            nonempty(self.group.as_deref()).is_some(),
            nonempty(self.dpar.as_deref()).is_some(),
        )
    }
}

pub(crate) fn select(
    priors: &[Prior],
    targets: &[PriorTarget<'_>],
) -> Result<Vec<Option<ResolvedPrior>>> {
    let mut seen = HashSet::new();
    for prior in priors {
        let selector = (
            &prior.class,
            nonempty(prior.coef.as_deref()),
            nonempty(prior.group.as_deref()),
            nonempty(prior.dpar.as_deref()),
        );
        if !seen.insert(selector) {
            return Err(Error(format!(
                "duplicate prior selector for class `{}`",
                prior.class
            )));
        }
        if nonempty(prior.coef.as_deref()).is_some() && (prior.lb.is_some() || prior.ub.is_some()) {
            return Err(Error(
                "coefficient-specific priors inherit bounds from their class prior".into(),
            ));
        }
        if prior.class == "sd"
            && nonempty(prior.coef.as_deref()).is_some()
            && nonempty(prior.group.as_deref()).is_none()
        {
            return Err(Error(
                "a coefficient-specific sd prior requires a group".into(),
            ));
        }
        if !targets.iter().any(|target| prior.matches(target)) {
            return Err(Error(format!(
                "prior selector class `{}` coef {:?} group {:?} dpar {:?} matches no parameter",
                prior.class, prior.coef, prior.group, prior.dpar
            )));
        }
    }
    Ok(targets
        .iter()
        .map(|target| {
            let matched: Vec<_> = priors
                .iter()
                .filter(|prior| prior.matches(target))
                .collect();
            let selected = matched
                .iter()
                .copied()
                .max_by_key(|prior| prior.specificity())?;
            // brms takes both bounds from the most specific base row, independently
            // of a coefficient-specific distribution override.
            let bounds = matched
                .iter()
                .copied()
                .filter(|prior| {
                    nonempty(prior.coef.as_deref()).is_none()
                        && (prior.lb.is_some() || prior.ub.is_some())
                })
                .max_by_key(|prior| prior.specificity());
            Some(ResolvedPrior {
                prior: selected.prior.clone(),
                lb: bounds.and_then(|prior| prior.lb.clone()),
                ub: bounds.and_then(|prior| prior.ub.clone()),
            })
        })
        .collect())
}

pub(crate) fn resolve(
    spec: &Spec,
    formula: &Formula,
    design: &Design,
    response: &[f64],
    offset_mean: f64,
) -> Result<Priors> {
    let location = median(response.to_vec());
    let mad = 1.4826 * median(response.iter().map(|y| (y - location).abs()).collect());
    let location = round_decimal(location) - offset_mean;
    let scale = round_decimal(mad).max(2.5);
    if !scale.is_finite() || !location.is_finite() {
        return Err(Error(
            "response statistics exceed the finite Gaussian prior range".into(),
        ));
    }
    let names: Vec<_> = design
        .columns
        .iter()
        .map(|c| crate::emit::coefficient_name(&c.name))
        .collect();
    let mut targets = Vec::new();
    if formula.intercept {
        targets.push(PriorTarget {
            class: "Intercept",
            coef: None,
            group: None,
            dpar: None,
        });
    }
    targets.extend(names.iter().map(|name| PriorTarget {
        class: "b",
        coef: Some(name),
        group: None,
        dpar: None,
    }));
    let group_names: Vec<_> = formula.groups.iter().map(|g| g.name()).collect();
    targets.extend(group_names.iter().map(|name| PriorTarget {
        class: "sd",
        coef: Some("Intercept"),
        group: Some(name),
        dpar: None,
    }));
    targets.push(PriorTarget {
        class: "sigma",
        coef: None,
        group: None,
        dpar: None,
    });
    let mut selected = select(&spec.priors, &targets)?.into_iter();
    let student = |location| ResolvedPrior::new(format!("student_t(3, {location}, {scale})"));
    let intercept = if formula.intercept {
        selected
            .next()
            .unwrap()
            .unwrap_or_else(|| student(location))
    } else {
        student(location)
    };
    let b = names
        .iter()
        .map(|_| {
            selected
                .next()
                .unwrap()
                .unwrap_or_else(|| ResolvedPrior::new("Flat()"))
        })
        .collect();
    let sd = formula
        .groups
        .iter()
        .map(|_| selected.next().unwrap().unwrap_or_else(|| student(0.0)))
        .collect();
    let sigma = selected.next().unwrap().unwrap_or_else(|| student(0.0));
    Ok(Priors {
        intercept_location: location,
        scale,
        sigma_scale: scale,
        intercept,
        b,
        sd,
        aux: BTreeMap::from([("sigma".into(), sigma)]),
    })
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

// R rounds at the input scale: multiplying first can erase a rounding tie.
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
