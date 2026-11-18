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
    pub nlpar: Option<String>,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) bcs: Vec<ResolvedPrior>,
    #[serde(default)]
    pub bs: Vec<ResolvedPrior>,
    #[serde(default)]
    pub sds: Vec<ResolvedPrior>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bmo: Vec<ResolvedPrior>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub simo: Vec<ResolvedPrior>,
    pub sd: Vec<Vec<ResolvedPrior>>,
    pub cor: Vec<Option<ResolvedPrior>>,
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

pub(crate) fn dpar(value: Option<&str>) -> Option<&str> {
    nonempty(value).filter(|&name| name != "mu")
}

impl Prior {
    fn predictor(&self) -> Option<&str> {
        self.nlpar.as_deref().or_else(|| dpar(self.dpar.as_deref()))
    }

    fn matches(&self, target: &PriorTarget<'_>) -> bool {
        canonical_class(&self.class) == target.class
            && self.predictor() == target.dpar
            && [
                (self.coef.as_deref(), target.coef),
                (self.group.as_deref(), target.group),
            ]
            .iter()
            .all(|&(selector, value)| nonempty(selector).is_none_or(|s| Some(s) == value))
    }

    fn specificity(&self) -> (bool, bool, bool) {
        (
            nonempty(self.coef.as_deref()).is_some(),
            nonempty(self.group.as_deref()).is_some(),
            self.predictor().is_some(),
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
            prior.predictor(),
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
                "prior selector class `{}` coef {:?} group {:?} predictor {:?} matches no parameter",
                prior.class,
                prior.coef,
                prior.group,
                prior.predictor()
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

pub(crate) struct PriorContext<'a> {
    pub family: Family,
    pub link: Link,
    pub thresholds: usize,
    pub dpar: Option<&'a str>,
    pub nonlinear: bool,
    pub response: Option<&'a [f64]>,
    pub offset_mean: f64,
    pub group_scale: f64,
    pub global_cor: bool,
    pub paired_sigma_scale: Option<f64>,
}

pub(crate) fn resolve(
    spec: &Spec,
    formula: &Formula,
    design: &Design,
    groups: &[Group],
    smooths: &[crate::Smooth],
    monotonic_design: Option<&crate::monotonic::MonotonicDesign>,
    context: PriorContext<'_>,
) -> Result<(Priors, BTreeMap<String, ResolvedPrior>)> {
    let PriorContext {
        family,
        link,
        thresholds,
        dpar,
        nonlinear,
        response,
        offset_mean,
        group_scale,
        global_cor,
        paired_sigma_scale,
    } = context;
    let second_mean = spec.paired_formula.is_some() && dpar == Some(crate::paired::SECOND_MEAN);
    let scale_link = family.prior_scale_link(link);
    let (mut location, scale) = default_scale(scale_link, response);
    if dpar.is_some() && !second_mean {
        location = 0.0;
    } else if response.is_some() && scale_link.uses_response_scale() {
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
        .filter(|name| dpar.is_none() && !spec.dpars.contains_key(*name))
        .filter(|&name| name != "disc")
        .filter(|&name| name != "sigma" || formula.se.as_ref().is_none_or(|se| se.sigma))
        .chain((dpar.is_none() && formula.ar.is_some()).then_some("ar"))
        .chain(
            paired_sigma_scale
                .iter()
                .flat_map(|_| crate::paired::AUXILIARIES),
        )
        .collect();
    let threshold_names: Vec<_> = (1..=thresholds).map(|i| i.to_string()).collect();
    if thresholds > 0 {
        targets.extend(threshold_names.iter().map(|name| PriorTarget {
            class: "Intercept",
            coef: Some(name),
            group: None,
            dpar,
        }));
    } else if formula.intercept {
        targets.push(PriorTarget {
            class: if nonlinear { "b" } else { "Intercept" },
            coef: nonlinear.then_some("Intercept"),
            group: None,
            dpar,
        });
    }
    targets.extend(names.iter().map(|name| PriorTarget {
        class: "b",
        coef: Some(name),
        group: None,
        dpar,
    }));
    targets.extend(formula.category_specific.iter().map(|name| PriorTarget {
        class: "b",
        coef: Some(name),
        group: None,
        dpar,
    }));
    let smooth_terms = formula
        .smooths
        .iter()
        .zip(smooths)
        .flat_map(|(term, smooth)| smooth.curves().map(move |(level, _, _)| (term, level)));
    let (mut smooth_names, mut smooth_labels): (Vec<_>, Vec<_>) = smooth_terms
        .map(|(term, level)| (term.coefficient(level), term.curve_label(level)))
        .unzip();
    for term in &formula.tensor_smooths {
        let name = term.name();
        smooth_names.extend((1..=crate::tensor::FIXED_COLS).map(|i| format!("{name}_{i}")));
        let label = term.label();
        smooth_labels.extend(crate::tensor::BLOCKS.iter().map(|_| label.clone()));
    }
    targets.extend(smooth_names.iter().map(|name| PriorTarget {
        class: "b",
        coef: Some(name),
        group: None,
        dpar,
    }));
    targets.extend(smooth_labels.iter().map(|label| PriorTarget {
        class: "sds",
        coef: Some(label),
        group: None,
        dpar,
    }));
    let (monotonic_names, simplex_names): (Vec<_>, Vec<_>) = match monotonic_design {
        Some(design) => (
            design
                .effects
                .iter()
                .map(|effect| effect.name.clone())
                .collect(),
            design
                .simplexes
                .iter()
                .map(|simplex| simplex.name.clone())
                .collect(),
        ),
        None => {
            let names: Vec<_> = formula
                .monotonic
                .iter()
                .map(|column| crate::monotonic::coefficient(column))
                .collect();
            let simplexes = names.iter().map(|name| format!("{name}1")).collect();
            (names, simplexes)
        }
    };
    targets.extend(monotonic_names.iter().map(|name| PriorTarget {
        class: "b",
        coef: Some(name),
        group: None,
        dpar,
    }));
    targets.extend(simplex_names.iter().map(|name| PriorTarget {
        class: "simo",
        coef: Some(name),
        group: None,
        dpar,
    }));
    targets.extend(group_targets(groups, formula, dpar));
    targets.extend(auxiliaries.iter().map(|name| PriorTarget {
        class: name,
        coef: None,
        group: None,
        dpar: None,
    }));
    let priors: Vec<_> = spec
        .priors
        .iter()
        .filter(|prior| prior.predictor() == dpar)
        .filter(|prior| !(global_cor && dpar.is_none() && canonical_class(&prior.class) == "cor"))
        .cloned()
        .collect();
    let selected = select(&priors, &targets)?;
    if nonlinear {
        for (target, prior) in targets.iter().zip(&selected) {
            if target.class == "b" && prior.is_none() {
                return Err(Error(format!(
                    "nonlinear parameter `{}` coefficient `{}` requires an explicit prior",
                    dpar.unwrap(),
                    target.coef.unwrap()
                )));
            }
        }
    }
    let mut selected = selected.into_iter();
    let student = |location| ResolvedPrior::new(format!("student_t(3, {location}, {scale})"));
    let default_intercept = || {
        if nonlinear {
            // No intercept is emitted when the nonlinear design omits it.
            student(0.0)
        } else if let Some(name) = dpar.filter(|_| !second_mean) {
            ResolvedPrior::new(family.default_predictor_prior(name, link, scale))
        } else if family == Family::VonMises && link == Link::TanHalf {
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
                return Err(Error("ordinal thresholds require one shared continuous prior; unequal threshold priors are not supported".into()));
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
                .unwrap_or_else(|| ResolvedPrior::new("Lebesgue(reals)"))
        })
        .collect();
    let mut sd = Vec::new();
    let bcs = formula
        .category_specific
        .iter()
        .map(|_| {
            selected
                .next()
                .unwrap()
                .unwrap_or_else(|| ResolvedPrior::new("Lebesgue(reals)"))
        })
        .collect();
    let bs = smooth_names
        .iter()
        .map(|_| {
            selected
                .next()
                .unwrap()
                .unwrap_or_else(|| ResolvedPrior::new("Lebesgue(reals)"))
        })
        .collect();
    let sds = smooth_labels
        .iter()
        .map(|_| {
            selected
                .next()
                .unwrap()
                .unwrap_or_else(|| ResolvedPrior::new(format!("student_t(3, 0, {group_scale})")))
        })
        .collect();
    let bmo = monotonic_names
        .iter()
        .map(|_| {
            selected
                .next()
                .unwrap()
                .unwrap_or_else(|| ResolvedPrior::new("Lebesgue(reals)"))
        })
        .collect();
    let simo = simplex_names
        .iter()
        .map(|_| {
            selected
                .next()
                .unwrap()
                .unwrap_or_else(|| ResolvedPrior::new("dirichlet(1)"))
        })
        .collect();
    let mut cor = Vec::new();
    for (group, term) in groups.iter().zip(&formula.groups) {
        sd.push(
            group
                .coefficients
                .iter()
                .map(|_| {
                    selected.next().unwrap().unwrap_or_else(|| {
                        ResolvedPrior::new(format!("student_t(3, 0, {group_scale})"))
                    })
                })
                .collect(),
        );
        cor.push(
            (term.id.is_none() && term.correlated && group.coefficients.len() > 1).then(|| {
                selected
                    .next()
                    .unwrap()
                    .unwrap_or_else(|| ResolvedPrior::new("lkj(1)"))
            }),
        );
    }
    let aux = auxiliaries
        .iter()
        .map(|&name| {
            let prior = selected.next().unwrap().unwrap_or_else(|| match name {
                "ar" | "rescor" => ResolvedPrior::new("uniform(-1, 1)"),
                "sigma2" => ResolvedPrior::new(
                    family.default_auxiliary("sigma", paired_sigma_scale.unwrap()),
                ),
                _ => ResolvedPrior::new(family.default_auxiliary(name, sigma_scale)),
            });
            (name.into(), prior)
        })
        .collect();
    Ok((
        Priors {
            intercept_location: location,
            scale,
            sigma_scale,
            intercept,
            b,
            bcs,
            bs,
            sds,
            bmo,
            simo,
            sd,
            cor,
        },
        aux,
    ))
}

fn group_targets<'a>(
    groups: &'a [Group],
    formula: &Formula,
    dpar: Option<&'a str>,
) -> Vec<PriorTarget<'a>> {
    let mut targets = Vec::new();
    for (group, term) in groups.iter().zip(&formula.groups) {
        targets.extend(group.coefficients.iter().map(|coefficient| PriorTarget {
            class: "sd",
            coef: Some(coefficient),
            group: Some(&group.column),
            dpar,
        }));
        if term.id.is_none() && term.correlated && group.coefficients.len() > 1 {
            targets.push(PriorTarget {
                class: "cor",
                coef: None,
                group: Some(&group.column),
                dpar,
            });
        }
    }
    targets
}

pub(crate) fn global_correlations(
    priors: &[Prior],
    recipe: &mut crate::Recipe,
    shared: &mut [crate::shared_groups::Block],
) -> Result<()> {
    let priors: Vec<_> = priors
        .iter()
        .filter(|prior| prior.predictor().is_none() && canonical_class(&prior.class) == "cor")
        .cloned()
        .collect();
    // Categorical priors historically select a grouping factor across all
    // category predictors. For other families keep the mean's legacy scope.
    let local: Vec<_> = recipe
        .predictors()
        .filter(|(name, _)| *name == "mu" || recipe.family == Family::Categorical)
        .flat_map(|(name, predictor)| {
            predictor
                .groups
                .iter()
                .zip(&predictor.priors.cor)
                .enumerate()
                .filter(|(_, (_, prior))| prior.is_some())
                .map(move |(i, (group, _))| (name.to_owned(), i, group.column.as_str()))
        })
        .collect();
    let targets: Vec<_> = local
        .iter()
        .map(|(_, _, group)| *group)
        .chain(shared.iter().map(|block| block.group.as_str()))
        .map(|group| PriorTarget {
            class: "cor",
            coef: None,
            group: Some(group),
            dpar: None,
        })
        .collect();
    let mut selected = select(&priors, &targets)?.into_iter();
    let local: Vec<_> = local.into_iter().map(|(name, i, _)| (name, i)).collect();
    for (name, i) in local {
        if let Some(prior) = selected.next().unwrap() {
            recipe.predictors.get_mut(&name).unwrap().priors.cor[i] = Some(prior);
        }
    }
    for (block, selected) in shared.iter_mut().zip(selected) {
        if let Some(selected) = selected {
            block.prior = selected;
        }
    }
    Ok(())
}
