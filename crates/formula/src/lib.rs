//! Pure Rust conversion of a bounded brms formula subset into FlatPPL.
//!
//! Scalar response families, numeric and factor predictors, interactions,
//! numeric expressions, offsets, group effects and thin-plate smooths are supported.
//! It does not fit a model.

mod adjacent;
mod arma;
mod autoregression;
mod beta_binomial;
mod categorical;
mod contrasts;
mod data;
mod design;
mod emit;
mod exgaussian;
mod expectation;
mod expression;
mod family;
mod formula;
mod metadata;
mod monotonic;
mod nonlinear;
mod ordinal;
mod paired;
mod prior_expr;
mod prior_model;
mod priors;
mod response;
mod shared_groups;
mod splines;
mod tensor;
mod text;
mod wiener;
mod wiener_sample;

use flatppl_core::Module;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fmt};

pub use autoregression::Ar1;
use data::ColumnModel;
pub use data::{
    DataDocument, DataMode, DataValue, NewGroupShapes, PreparedForecast, prepare_data,
    prepare_data_with_new_groups, prepare_forecast,
};
pub use design::{Contrast, Design, DesignColumn, DesignComponent, FactorEncoding, Predictor};
pub use expression::{Arithmetic, ColumnExpression};
pub use family::{Family, Link};
pub use flatppl_data::{Column, ColumnValues, Table};
pub use formula::{Formula, GroupTerm, SmoothTerm, StandardError, Term, Trials, parse_formula};
pub use monotonic::Monotonic;
pub use nonlinear::NonlinearPredictor;
pub use priors::{Prior, PriorBound, Priors, ResolvedPrior};
pub use response::{Censoring, Truncation, TruncationBound};
pub use splines::{Smooth, ThinPlate};
pub use text::{read_csv_table, read_json_table};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<flatppl_data::Error> for Error {
    fn from(error: flatppl_data::Error) -> Self {
        Error(error.0)
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self(format!("invalid formula JSON: {value}"))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub formula: String,
    /// A second response formula.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paired_formula: Option<String>,
    /// The second response family. Omit this for a paired Gaussian model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paired_family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paired_link: Option<String>,
    #[serde(default = "gaussian")]
    pub family: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub dpars: BTreeMap<String, DistributionalFormula>,
    /// Named formulas for parameters used in a nonlinear response expression.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub nlpars: BTreeMap<String, String>,
    #[serde(default)]
    pub priors: Vec<Prior>,
    #[serde(default)]
    pub options: Options,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Table>,
}

/// A predictor for a family parameter, for example `sigma ~ g`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionalFormula {
    pub formula: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<String>,
}

fn gaussian() -> String {
    "gaussian".into()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(from = "OptionsInput")]
pub struct Options {
    pub center: bool,
    pub contrasts: std::collections::BTreeMap<String, Contrast>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            center: default_center(),
            contrasts: std::collections::BTreeMap::new(),
        }
    }
}

fn default_center() -> bool {
    true
}

// Read older specifications without restoring their data-embedding behavior.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OptionsInput {
    #[serde(default = "default_center")]
    center: bool,
    #[serde(default)]
    contrasts: std::collections::BTreeMap<String, Contrast>,
    #[serde(default, rename = "data_placement")]
    _legacy_placement: LegacyDataPlacement,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum LegacyDataPlacement {
    #[default]
    External,
    Inline,
}

impl From<OptionsInput> for Options {
    fn from(input: OptionsInput) -> Self {
        Self {
            center: input.center,
            contrasts: input.contrasts,
        }
    }
}

pub struct Conversion {
    pub module: Module,
    pub data: DataDocument,
    pub recipe: Recipe,
    pub info: ModelInfo,
    complete_template: Option<Module>,
}

impl Conversion {
    /// When training used partial outcomes, return the unmasked template for new data.
    pub fn complete_template(&self) -> Option<&Module> {
        self.complete_template.as_ref()
    }

    /// Export a file-backed module and its prepared numeric JSON sidecar.
    /// `source` is the sidecar path as resolved from the generated module.
    /// The caller writes both files. The original module remains external for
    /// in-memory data binding; only the structural row count is literal here.
    pub fn with_data_source(&self, source: &str) -> Result<(Module, serde_json::Value)> {
        Ok((
            self.data.with_data_source(self.module.clone(), source)?,
            self.data.columns_json()?,
        ))
    }
}

/// Training preprocessing state. A new fit builds a new recipe.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Recipe {
    pub formula: Formula,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paired_response: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) paired_family: Option<Family>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) paired_response_levels: Option<Vec<GroupLevel>>,
    pub family: Family,
    pub link: Link,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_levels: Option<Vec<GroupLevel>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) threshold_groups: Option<GroupInfo>,
    /// Predictor names in nonreference response-category order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub category_predictors: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nonlinear: Option<NonlinearPredictor>,
    pub predictors: BTreeMap<String, PredictorRecipe>,
    pub auxiliary_priors: BTreeMap<String, ResolvedPrior>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) shared_groups: Vec<shared_groups::Block>,
}

/// Fitted coding, centring and priors for one distributional predictor.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PredictorRecipe {
    pub formula: Formula,
    #[serde(default)]
    pub link: Link,
    pub design: Design,
    pub center: bool,
    pub means: Vec<f64>,
    pub groups: Vec<Group>,
    #[serde(default)]
    pub smooths: Vec<Smooth>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) tensor_smooths: Vec<tensor::TensorSmooth>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub monotonic: Vec<Monotonic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) monotonic_design: Option<monotonic::MonotonicDesign>,
    pub priors: Priors,
}

impl Recipe {
    /// Adapt the complete-outcome template to a snapshot with partial observations.
    pub fn partial_outcome_template(
        &self,
        template: &Module,
        data: &DataDocument,
    ) -> Result<Option<Module>> {
        if !data.constants.contains_key(paired::OBSERVED) {
            return Ok(None);
        }
        let second = self
            .predictors
            .get(paired::SECOND_MEAN)
            .ok_or_else(|| Error("partial outcomes require a paired formula".into()))?;
        let mut module = template.clone();
        paired::use_partial_outcomes(&mut module, self.second_family(), [self.link, second.link])?;
        Ok(Some(module))
    }

    /// Expand noncentered group draws in this recipe's prediction template.
    /// `shapes` comes from [`prepare_data_with_new_groups`].
    pub fn extend_group_draws(&self, template: Module, shapes: &NewGroupShapes) -> Result<Module> {
        emit::extend_group_draws(self, template, shapes)
    }

    pub(crate) fn is_nonlinear_parameter(&self, name: &str) -> bool {
        self.nonlinear
            .as_ref()
            .is_some_and(|p| p.parameters.iter().any(|parameter| parameter == name))
    }

    pub(crate) fn predictor_suffix<'a>(&self, name: &'a str) -> std::borrow::Cow<'a, str> {
        if self.is_nonlinear_parameter(name) {
            crate::nonlinear::parameter_suffix(name).into()
        } else {
            name.into()
        }
    }

    pub(crate) fn predictors(&self) -> impl Iterator<Item = (&str, &PredictorRecipe)> {
        std::iter::once("mu")
            .chain(self.paired_response.as_ref().map(|_| paired::SECOND_MEAN))
            .chain(self.family.auxiliaries().iter().copied())
            .chain(self.category_predictors.iter().map(String::as_str))
            .chain(
                self.nonlinear
                    .iter()
                    .flat_map(|p| p.parameters.iter().map(String::as_str)),
            )
            .filter_map(|name| self.predictors.get(name).map(|predictor| (name, predictor)))
    }

    pub(crate) fn auxiliary_names(&self) -> impl Iterator<Item = &'static str> {
        self.family
            .auxiliaries()
            .iter()
            .copied()
            .filter(|&name| name != "disc" || self.predictors.contains_key(name))
            .chain(
                self.paired_response
                    .iter()
                    .filter(|_| self.second_family() == Family::Gaussian)
                    .flat_map(|_| paired::AUXILIARIES),
            )
    }

    pub(crate) fn second_family(&self) -> Family {
        self.paired_family.unwrap_or(Family::Gaussian)
    }

    pub(crate) fn thresholds(&self) -> usize {
        if self.family.ordinal() {
            self.response_levels.as_ref().unwrap().len() - 1
        } else {
            0
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Group {
    pub column: String,
    pub levels: Vec<GroupLevel>,
    pub design: Design,
    pub coefficients: Vec<String>,
    pub inputs: Vec<String>,
    pub correlation: Option<String>,
    pub correlation_matrix: Option<String>,
    pub sd: String,
    pub z: String,
    pub r: String,
    pub index: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) shared: Option<shared_groups::Coefficients>,
}

impl Group {
    pub(crate) fn innovation_width(&self) -> usize {
        self.shared
            .map_or(self.coefficients.len(), |block| block.width)
    }

    pub(crate) fn coefficient_offset(&self) -> usize {
        self.shared.map_or(0, |block| block.offset)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum GroupLevel {
    Integer(i64),
    Real(f64),
    Text(String),
    Interaction(Vec<GroupLevel>),
}

impl fmt::Display for GroupLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Integer(value) => write!(f, "{value}"),
            Self::Real(value) => write!(f, "{value}"),
            Self::Text(value) => f.write_str(value),
            Self::Interaction(levels) => {
                for (i, level) in levels.iter().enumerate() {
                    if i != 0 {
                        f.write_str("_")?;
                    }
                    write!(f, "{level}")?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelInfo {
    pub formula: String,
    /// Column names in paired prediction output order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_columns: Option<[String; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paired_family: Option<Family>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paired_response_levels: Option<Vec<GroupLevel>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_column: Option<String>,
    pub family: String,
    pub link: String,
    pub group_collation: String,
    pub default_prior_rule: String,
    pub design_columns: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub factor_levels: BTreeMap<String, FactorInfo>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub group_levels: BTreeMap<String, GroupInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_levels: Option<Vec<GroupLevel>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_groups: Option<GroupInfo>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub dpars: BTreeMap<String, PredictorInfo>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub nlpars: BTreeMap<String, PredictorInfo>,
    pub parameters: Vec<Parameter>,
    pub has_improper_prior: bool,
    pub prior_factors: BTreeMap<String, PriorFactor>,
    pub prior_sampling: PriorSampling,
    pub training_rows: Vec<usize>,
    /// Additive difference from the target using normalized Stan density functions.
    pub score_offset: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PredictorInfo {
    pub formula: String,
    pub link: String,
    pub design_columns: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub factor_levels: BTreeMap<String, FactorInfo>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub group_levels: BTreeMap<String, GroupInfo>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FactorInfo {
    pub name: String,
    pub levels: Vec<String>,
    pub ordered: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GroupInfo {
    pub name: String,
    pub levels: Vec<GroupLevel>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ParameterRole {
    Sampled,
    Fixed,
    Derived,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PriorSampling {
    /// A prior factor is not a probability measure.
    Unavailable,
    /// Conversion does not test an engine's sampler implementation.
    NotChecked,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PriorFactor {
    /// The inferred FlatPIR total-mass class, not a sampler capability.
    pub mass: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Parameter {
    pub name: String,
    pub binding: String,
    /// One-based array index, matching FlatPPL and brms names.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    /// One-based matrix coordinates, when the binding is matrix-valued.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indices: Option<Vec<usize>>,
    pub role: ParameterRole,
    /// Legacy flag: true for both fixed and derived values.
    pub derived: bool,
}

/// Read a conversion specification. Paths and filesystem access stay with hosts.
pub fn read(json: &str) -> Result<Spec> {
    Ok(serde_json::from_str(json)?)
}

/// Build a model and an immutable training recipe without I/O.
pub fn build(spec: &Spec, table: &Table) -> Result<Conversion> {
    let (family, link) = Family::resolve(&spec.family, spec.link.as_deref())?;
    let (formula, nonlinear) = nonlinear::parse(spec, family)?;
    if let Some(group) = &formula.threshold_group {
        if family != Family::Cumulative {
            return Err(Error("thres(gr=...) requires a cumulative response".into()));
        }
        if group.column == formula.response {
            return Err(Error("thres() requires a separate grouping column".into()));
        }
    }
    let second_formula = paired::parse(spec, family, &formula)?;
    let (second_family, second_link) = match spec.paired_family.as_deref() {
        Some(name) => Family::resolve(name, spec.paired_link.as_deref())?,
        None => (Family::Gaussian, link),
    };
    let paired_response = second_formula
        .as_ref()
        .map(|formula| formula.response.clone());
    if family.ordinal() && !formula.intercept {
        return Err(Error(format!(
            "{} models require threshold intercepts",
            spec.family
        )));
    }
    if (family == Family::Wiener) != formula.decision.is_some() {
        return Err(Error(
            "`dec(...)` is required for Wiener responses and is unavailable for other families"
                .into(),
        ));
    }
    if formula.se.is_some() && !matches!(family, Family::Gaussian | Family::Student) {
        return Err(Error(
            "`se(...)` is available for Gaussian and Student families".into(),
        ));
    }
    if formula.censoring.is_some()
        && !matches!(
            family,
            Family::Gaussian | Family::Weibull | Family::Exponential
        )
    {
        return Err(Error(
            "`cens(...)` is available for Gaussian, Weibull and exponential responses".into(),
        ));
    }
    if formula.truncation.is_some() && family != Family::Gaussian {
        return Err(Error(
            "`trunc(...)` is available for Gaussian responses".into(),
        ));
    }
    if let Some(ar) = &formula.ar {
        ar.validate(&formula, family, spec.dpars.contains_key("sigma"))?;
    }
    if family.uses_trials() != formula.trials.is_some() {
        return Err(Error(
            "`trials(...)` is required for binomial families and is unavailable for other families"
                .into(),
        ));
    }
    let initial_levels = if family == Family::Categorical {
        let mut response_only = formula.clone();
        response_only.weights = None;
        let rows = data::model_rows(&response_only, &[], &[], table, DataMode::Training, None)?;
        data::response_levels(family, table.column(&formula.response)?, &rows)?
    } else {
        None
    };
    let category_predictors = initial_levels
        .as_deref()
        .map(categorical::predictor_names)
        .transpose()?
        .unwrap_or_default();
    let mut formulas = if family == Family::Categorical {
        category_predictors
            .iter()
            .map(|name| (name.as_str(), formula.clone(), link))
            .collect()
    } else {
        vec![("mu", formula.clone(), link)]
    };
    if let Some(second_formula) = second_formula {
        formulas.push((paired::SECOND_MEAN, second_formula, second_link));
    }
    for (name, spec) in &spec.dpars {
        let formula = parse_formula(&spec.formula)?;
        if formula.response != *name || formula.has_response_additions() {
            return Err(Error(format!(
                "distributional formula must be `{name} ~ terms` without response additions"
            )));
        }
        if family == Family::Categorical {
            let predictor = formulas
                .iter_mut()
                .find(|(dpar, _, _)| *dpar == name)
                .ok_or_else(|| {
                    Error(format!(
                        "categorical predictor `{name}` is not one of {}",
                        category_predictors.join(", ")
                    ))
                })?;
            let (_, link) = Family::resolve("categorical", spec.link.as_deref())?;
            *predictor = (name, formula, link);
        } else {
            let link = family.auxiliary_link(name, spec.link.as_deref())?;
            formulas.push((name, formula, link));
        }
    }
    if spec.dpars.contains_key("sigma") && formula.se.as_ref().is_some_and(|se| !se.sigma) {
        return Err(Error(
            "sigma cannot be predicted with se(sigma=FALSE)".into(),
        ));
    }
    for (name, text) in &spec.nlpars {
        let formula = parse_formula(text)?;
        if formula.response != *name || formula.has_response_additions() {
            return Err(Error(format!(
                "nonlinear parameter formula must be `{name} ~ terms` without response additions"
            )));
        }
        if formula
            .predictor_columns()
            .iter()
            .any(|column| spec.nlpars.contains_key(*column))
        {
            return Err(Error(format!(
                "nonlinear parameter formula `{name}` must use data columns"
            )));
        }
        formulas.push((name, formula, Link::Identity));
    }
    for prior in &spec.priors {
        let dpar = priors::dpar(prior.dpar.as_deref());
        if let Some(name) = &prior.nlpar {
            if prior.dpar.is_some() || !spec.nlpars.contains_key(name) {
                return Err(Error(format!("invalid nonlinear prior selector `{name}`")));
            }
        } else if dpar.is_some_and(|name| spec.nlpars.contains_key(name)) {
            return Err(Error("use `nlpar` to select a nonlinear parameter".into()));
        }
        let correlation = matches!(prior.class.as_str(), "cor" | "L");
        if family == Family::Categorical && dpar.is_none() && !correlation {
            return Err(Error(
                "categorical priors must select a category predictor with `dpar`".into(),
            ));
        }
        if family == Family::Categorical && dpar.is_some() && correlation {
            return Err(Error(
                "categorical correlation priors select `group`, without `dpar`".into(),
            ));
        }
        if let Some(name) = dpar
            && !formulas.iter().any(|(dpar, _, _)| *dpar == name)
        {
            return Err(Error(format!(
                "prior selects unpredicted distributional parameter `{name}`"
            )));
        }
    }
    for (name, predictor, _) in &formulas {
        if !predictor.category_specific.is_empty() && (family != Family::Acat || *name != "mu") {
            return Err(Error(
                "`cs(...)` is available in the `acat` response predictor".into(),
            ));
        }
        if predictor
            .predictor_columns()
            .iter()
            .any(|&name| formula.response == name || paired_response.as_deref() == Some(name))
        {
            return Err(Error(
                "the response cannot also be a predictor in this subset".into(),
            ));
        }
        if predictor.intercept
            && !spec.options.center
            && !spec.nlpars.contains_key(*name)
            && (*name != "mu" || nonlinear.is_none())
        {
            return Err(Error("center=false with an intercept is not yet supported; use the default centred intercept or an explicit no-intercept formula".into()));
        }
    }
    let predictor_formulas: Vec<_> = formulas.iter().map(|(_, formula, _)| formula).collect();
    let extra_columns = nonlinear.as_ref().map_or(&[][..], |p| p.columns.as_slice());
    let rows = data::model_rows(
        &formula,
        &predictor_formulas,
        extra_columns,
        table,
        DataMode::Training,
        paired_response.as_deref(),
    )?;
    for name in spec.options.contrasts.keys() {
        if !formulas.iter().any(|(_, formula, _)| {
            formula.variables.contains(name)
                || formula.groups.iter().any(|g| g.variables.contains(name))
                || formula
                    .monotonic_terms
                    .as_ref()
                    .is_some_and(|terms| terms.variables.contains(name))
        }) {
            return Err(Error(format!(
                "contrast supplied for unused predictor `{name}`"
            )));
        }
    }
    let response_column = table.column(&formula.response)?;
    let first_observed = paired_response
        .as_ref()
        .map(|_| data::observed_rows(response_column, &rows));
    let first_rows = first_observed.as_deref().unwrap_or(&rows);
    let response_levels = data::response_levels(family, response_column, first_rows)?;
    let threshold_groups = formula
        .threshold_group
        .as_ref()
        .map(|group| {
            data::fit_threshold_groups(
                group,
                table.column(&group.column)?,
                response_column,
                &rows,
                response_levels
                    .as_deref()
                    .expect("validated cumulative family"),
            )
        })
        .transpose()?;
    if family == Family::Categorical && initial_levels != response_levels {
        return Err(Error("missing-row omission removed an entire categorical response level; filter rows and refit the formula explicitly".into()));
    }
    let thresholds = if family.ordinal() {
        response_levels.as_ref().unwrap().len() - 1
    } else {
        0
    };
    data::response(
        family,
        response_levels.as_deref(),
        &formula,
        table,
        first_rows,
    )?;
    let response = if response_column.like_factor() || first_rows.is_empty() {
        None
    } else {
        Some(response_column.numeric(first_rows)?)
    };
    let (_, group_scale) =
        family::default_scale(family.prior_scale_link(link), response.as_deref());
    let second_column = paired_response
        .as_deref()
        .map(|name| table.column(name))
        .transpose()?;
    let second_observed = second_column.map(|column| data::observed_rows(column, &rows));
    let second_rows = second_observed.as_deref().unwrap_or(&[]);
    let paired_response_levels = second_column
        .map(|column| data::response_levels(second_family, column, second_rows))
        .transpose()?
        .flatten();
    let second_response = second_column
        .filter(|column| {
            !second_rows.is_empty() && (second_family == Family::Gaussian || !column.like_factor())
        })
        .map(|column| column.numeric(second_rows))
        .transpose()?;
    let second_sigma_scale = (paired_response.is_some() && second_family == Family::Gaussian)
        .then(|| family::default_scale(Link::Identity, second_response.as_deref()).1);
    let second_group_scale = family::default_scale(
        second_family.prior_scale_link(second_link),
        second_response.as_deref(),
    )
    .1;
    let response_formula = formula;
    let global_cor = family == Family::Categorical
        || formulas
            .iter()
            .any(|(_, formula, _)| formula.groups.iter().any(|term| term.id.is_some()));
    let mut predictors = BTreeMap::new();
    let mut auxiliary_priors = BTreeMap::new();
    for (name, formula, link) in formulas {
        let predictor_family = if name == paired::SECOND_MEAN {
            second_family
        } else {
            family
        };
        let is_nonlinear = spec.nlpars.contains_key(name);
        let (design, means) =
            design::build((&formula).into(), table, &rows, &spec.options.contrasts)?;
        let smooths = formula
            .smooths
            .iter()
            .map(|term| Smooth::fit(term, table, &rows))
            .collect::<Result<Vec<_>>>()?;
        let tensor_smooths = formula
            .tensor_smooths
            .iter()
            .map(|term| tensor::TensorSmooth::fit(term, table, &rows))
            .collect::<Result<Vec<_>>>()?;
        let offset_mean = formula
            .offset()
            .map(|offset| offset.values(table, &rows))
            .transpose()?
            .map_or(0.0, |values| values.iter().sum::<f64>() / rows.len() as f64);
        let monotonic = formula
            .monotonic
            .iter()
            .map(|column| Monotonic::fit(table.column(column)?, &rows))
            .collect::<Result<Vec<_>>>()?;
        let monotonic_design = formula
            .monotonic_terms
            .as_ref()
            .map(|terms| {
                monotonic::MonotonicDesign::fit(terms, table, &rows, &spec.options.contrasts)
            })
            .transpose()?;
        let groups = formula
            .groups
            .iter()
            .map(|term| {
                let (design, _) =
                    design::build(term.into(), table, &rows, &spec.options.contrasts)?;
                let coefficients: Vec<_> = term
                    .intercept
                    .then_some("Intercept".into())
                    .into_iter()
                    .chain(
                        design
                            .columns
                            .iter()
                            .map(|column| emit::coefficient_name(&column.name)),
                    )
                    .collect();
                if coefficients.is_empty() {
                    return Err(Error(format!(
                        "group `{}` has no group effects",
                        term.name()
                    )));
                }
                Ok(Group {
                    column: term.name(),
                    levels: data::group_levels(&term.columns, table, &rows)?,
                    design,
                    coefficients,
                    inputs: Vec::new(),
                    correlation: None,
                    correlation_matrix: None,
                    sd: String::new(),
                    z: String::new(),
                    r: String::new(),
                    index: String::new(),
                    shared: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut group_coefficients = std::collections::HashSet::new();
        for group in &groups {
            for coefficient in &group.coefficients {
                if !group_coefficients.insert((&group.column, coefficient)) {
                    return Err(Error(format!(
                        "duplicated group effect `{coefficient}` for `{}`",
                        group.column
                    )));
                }
            }
        }
        let (priors, aux) = priors::resolve(
            spec,
            &formula,
            &design,
            &groups,
            &smooths,
            monotonic_design.as_ref(),
            priors::PriorContext {
                family: predictor_family,
                link,
                thresholds: if name == "mu" { thresholds } else { 0 },
                dpar: (name != "mu").then_some(name),
                nonlinear: is_nonlinear,
                response: if is_nonlinear {
                    None
                } else if name == paired::SECOND_MEAN && paired_response.is_some() {
                    second_response.as_deref()
                } else {
                    response.as_deref()
                },
                offset_mean,
                global_cor,
                group_scale: if name == paired::SECOND_MEAN {
                    second_group_scale
                } else {
                    group_scale
                },
                paired_sigma_scale: if name == "mu" {
                    second_sigma_scale
                } else {
                    None
                },
            },
        )?;
        auxiliary_priors.extend(aux);
        if name == "mu" && nonlinear.is_some() && thresholds == 0 {
            continue;
        }
        predictors.insert(
            name.to_owned(),
            PredictorRecipe {
                center: !is_nonlinear && spec.options.center && formula.intercept,
                formula,
                link,
                design,
                means,
                groups,
                smooths,
                tensor_smooths,
                monotonic,
                monotonic_design,
                priors,
            },
        );
    }
    let mut recipe = Recipe {
        formula: response_formula,
        paired_response,
        paired_family: spec.paired_family.as_ref().map(|_| second_family),
        paired_response_levels,
        family,
        link,
        response_levels,
        threshold_groups,
        category_predictors,
        nonlinear,
        predictors,
        auxiliary_priors,
        shared_groups: Vec::new(),
    };
    let mut shared = shared_groups::plan(&recipe)?;
    if global_cor {
        priors::global_correlations(&spec.priors, &mut recipe, &mut shared)?;
    }
    let module = emit::model(&mut recipe, &shared)?;
    let data = prepare_data(&recipe, table, DataMode::Training)?;
    let text = flatppl_syntax::print(&module);
    let module = flatppl_syntax::parse(&text)
        .map_err(|e| Error(format!("generated module failed to reparse: {e}")))?;
    let partial = recipe.partial_outcome_template(&module, &data)?;
    let mut validation = emit::inline(partial.as_ref().unwrap_or(&module).clone(), &data)?;
    let diagnostics = flatppl_infer::infer(&mut validation);
    if let Some(diagnostic) = diagnostics.first() {
        return Err(Error(format!(
            "generated formula model: {}",
            diagnostic.message
        )));
    }
    let info = emit::info(spec, &recipe, &data, &validation, &shared)?;
    let annotate = |mut module: Module| -> Result<Module> {
        metadata::emit(&mut module, &info)?;
        flatppl_syntax::parse(&flatppl_syntax::print(&module))
            .map_err(|e| Error(format!("generated formula metadata failed to reparse: {e}")))
    };
    let (module, complete_template) = if let Some(partial) = partial {
        (annotate(partial)?, Some(annotate(module)?))
    } else {
        (annotate(module)?, None)
    };
    recipe.shared_groups = shared;
    Ok(Conversion {
        module,
        data,
        recipe,
        info,
        complete_template,
    })
}
