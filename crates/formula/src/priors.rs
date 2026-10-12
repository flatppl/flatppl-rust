use crate::{Design, Error, Family, Formula, Group, Link, Result, Spec, family::default_scale};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

pub(crate) use crate::prior_expr::{PriorNode, emit, emit_correlation};

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
    pub sd: Vec<Vec<ResolvedPrior>>,
    pub cor: Vec<Option<ResolvedPrior>>,
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

fn canonical_class(class: &str) -> &str {
    if class == "L" { "cor" } else { class }
}

impl Prior {
    fn matches(&self, target: &PriorTarget<'_>) -> bool {
        canonical_class(&self.class) == target.class
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
            canonical_class(&prior.class),
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
    groups: &[Group],
    (family, link, thresholds): (Family, Link, usize),
    response: Option<&[f64]>,
    offset_mean: f64,
) -> Result<Priors> {
    let scale_link = family.prior_scale_link(link);
    let (mut location, scale) = default_scale(scale_link, response);
    if response.is_some() && scale_link.uses_response_scale() {
        location -= offset_mean;
    }
    let (_, sigma_scale) = default_scale(family.prior_scale_link(Link::Identity), response);
    let names: Vec<_> = design
        .columns
        .iter()
        .map(|c| crate::emit::coefficient_name(&c.name))
        .collect();
    let mut targets = Vec::new();
    let auxiliaries: Vec<_> = family
        .auxiliaries()
        .iter()
        .copied()
        .filter(|&name| name != "sigma" || formula.se.as_ref().is_none_or(|se| se.sigma))
        .collect();
    let threshold_names: Vec<_> = (1..=thresholds).map(|i| i.to_string()).collect();
    if thresholds > 0 {
        targets.extend(threshold_names.iter().map(|name| PriorTarget {
            class: "Intercept",
            coef: Some(name),
            group: None,
            dpar: None,
        }));
    } else if formula.intercept {
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
    targets.extend(group_targets(groups, formula));
    targets.extend(auxiliaries.iter().map(|name| PriorTarget {
        class: name,
        coef: None,
        group: None,
        dpar: None,
    }));
    let mut selected = select(&spec.priors, &targets)?.into_iter();
    let student = |location| ResolvedPrior::new(format!("student_t(3, {location}, {scale})"));
    let default_intercept = || {
        if family == Family::VonMises && link == Link::TanHalf {
            ResolvedPrior::new("student_t(1, 0, 1)")
        } else {
            student(location)
        }
    };
    let intercept = if thresholds > 0 {
        let mut priors =
            (0..thresholds).map(|_| selected.next().unwrap().unwrap_or_else(|| student(0.0)));
        let first = priors.next().unwrap();
        for prior in priors {
            if !crate::prior_expr::equivalent(&first, &prior)? {
                return Err(Error("cumulative thresholds require one shared continuous prior; unequal threshold priors are not supported".into()));
            }
        }
        first
    } else if formula.intercept {
        selected.next().unwrap().unwrap_or_else(default_intercept)
    } else {
        default_intercept()
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
    let mut sd = Vec::new();
    let mut cor = Vec::new();
    for (group, term) in groups.iter().zip(&formula.groups) {
        sd.push(
            group
                .coefficients
                .iter()
                .map(|_| selected.next().unwrap().unwrap_or_else(|| student(0.0)))
                .collect(),
        );
        cor.push((term.correlated && group.coefficients.len() > 1).then(|| {
            selected
                .next()
                .unwrap()
                .unwrap_or_else(|| ResolvedPrior::new("lkj(1)"))
        }));
    }
    let aux = auxiliaries
        .iter()
        .map(|&name| {
            let prior = selected
                .next()
                .unwrap()
                .unwrap_or_else(|| ResolvedPrior::new(family.default_auxiliary(name, sigma_scale)));
            (name.into(), prior)
        })
        .collect();
    Ok(Priors {
        intercept_location: location,
        scale,
        sigma_scale,
        intercept,
        b,
        sd,
        cor,
        aux,
    })
}

fn group_targets<'a>(groups: &'a [Group], formula: &Formula) -> Vec<PriorTarget<'a>> {
    let mut targets = Vec::new();
    for (group, term) in groups.iter().zip(&formula.groups) {
        targets.extend(group.coefficients.iter().map(|coefficient| PriorTarget {
            class: "sd",
            coef: Some(coefficient),
            group: Some(&group.column),
            dpar: None,
        }));
        if term.correlated && group.coefficients.len() > 1 {
            targets.push(PriorTarget {
                class: "cor",
                coef: None,
                group: Some(&group.column),
                dpar: None,
            });
        }
    }
    targets
}
