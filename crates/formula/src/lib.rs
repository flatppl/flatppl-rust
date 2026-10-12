//! Pure Rust conversion of a bounded brms formula subset into FlatPPL.
//!
//! Scalar response families, numeric and factor predictors, interactions,
//! numeric expressions, offsets and group effects are supported.
//! It does not fit a model.

#[cfg(feature = "arrow")]
mod arrow;
mod contrasts;
mod data;
mod design;
mod emit;
mod expression;
mod family;
mod formula;
mod ordinal;
mod prior_expr;
mod priors;
mod text;

use flatppl_core::Module;
use serde::{Deserialize, Serialize};
use std::fmt;

pub use data::{Column, ColumnValues, DataDocument, DataMode, DataValue, Table, prepare_data};
pub use design::{Contrast, Design, DesignColumn, DesignComponent, FactorEncoding, Predictor};
pub use expression::{Arithmetic, ColumnExpression};
pub use family::{Family, Link};
pub use formula::{Formula, GroupTerm, StandardError, Term, Trials, parse_formula};
pub use priors::{Prior, PriorBound, Priors, ResolvedPrior};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self(format!("invalid formula JSON: {value}"))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub formula: String,
    #[serde(default = "gaussian")]
    pub family: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<String>,
    #[serde(default)]
    pub priors: Vec<Prior>,
    #[serde(default)]
    pub options: Options,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Table>,
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
}

impl Conversion {
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
    #[serde(default)]
    pub family: Family,
    #[serde(default)]
    pub link: Link,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_levels: Option<Vec<GroupLevel>>,
    pub design: Design,
    pub center: bool,
    pub means: Vec<f64>,
    pub groups: Vec<Group>,
    pub priors: Priors,
}

impl Recipe {
    pub(crate) fn thresholds(&self) -> usize {
        if self.family == Family::Cumulative {
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
    pub family: String,
    pub link: String,
    pub group_collation: String,
    pub default_prior_rule: String,
    pub design_columns: Vec<String>,
    pub parameters: Vec<Parameter>,
    pub has_improper_prior: bool,
    pub training_rows: Vec<usize>,
    /// Additive difference from the target using normalized Stan density functions.
    pub score_offset: f64,
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
    pub derived: bool,
}

/// Read a conversion specification. Paths and filesystem access stay with hosts.
pub fn read(json: &str) -> Result<Spec> {
    Ok(serde_json::from_str(json)?)
}

/// Build a model and an immutable training recipe without I/O.
pub fn build(spec: &Spec, table: &Table) -> Result<Conversion> {
    let (family, link) = Family::resolve(&spec.family, spec.link.as_deref())?;
    let formula = parse_formula(&spec.formula)?;
    if family == Family::Cumulative && !formula.intercept {
        return Err(Error(
            "cumulative models require threshold intercepts".into(),
        ));
    }
    if formula.se.is_some() && !matches!(family, Family::Gaussian | Family::Student) {
        return Err(Error(
            "`se(...)` is available for Gaussian and Student families".into(),
        ));
    }
    if (family == Family::Binomial) != formula.trials.is_some() {
        return Err(Error(
            "`trials(...)` is required for binomial and is unavailable for other families".into(),
        ));
    }
    if formula
        .predictor_columns()
        .contains(&formula.response.as_str())
    {
        return Err(Error(
            "the response cannot also be a predictor in this subset".into(),
        ));
    }
    if formula.intercept && !spec.options.center {
        return Err(Error("center=false with an intercept is not yet supported; use the default centred intercept or an explicit no-intercept formula".into()));
    }
    let rows = data::model_rows(&formula, table, DataMode::Training)?;
    for name in spec.options.contrasts.keys() {
        if !formula.variables.contains(name)
            && !formula.groups.iter().any(|g| g.variables.contains(name))
        {
            return Err(Error(format!(
                "contrast supplied for unused predictor `{name}`"
            )));
        }
    }
    let (design, means) = design::build((&formula).into(), table, &rows, &spec.options.contrasts)?;
    let response_column = table.column(&formula.response)?;
    let response_levels = data::response_levels(family, response_column, &rows)?;
    let thresholds = if family == Family::Cumulative {
        response_levels.as_ref().unwrap().len() - 1
    } else {
        0
    };
    data::response(family, response_levels.as_deref(), &formula, table, &rows)?;
    let response = if response_column.like_factor() {
        None
    } else {
        Some(response_column.numeric(&rows)?)
    };
    let offset_mean = formula
        .offset()
        .map(|offset| offset.values(table, &rows))
        .transpose()?
        .map_or(0.0, |values| values.iter().sum::<f64>() / rows.len() as f64);
    let groups = formula
        .groups
        .iter()
        .map(|term| {
            let (design, _) = design::build(term.into(), table, &rows, &spec.options.contrasts)?;
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
    let priors = priors::resolve(
        spec,
        &formula,
        &design,
        &groups,
        (family, link, thresholds),
        response.as_deref(),
        offset_mean,
    )?;
    let mut recipe = Recipe {
        center: spec.options.center && formula.intercept,
        formula,
        family,
        link,
        response_levels,
        design,
        means,
        groups,
        priors,
    };
    let module = emit::model(spec, &mut recipe)?;
    let data = prepare_data(&recipe, table, DataMode::Training)?;
    let text = flatppl_syntax::print(&module);
    let mut module = flatppl_syntax::parse(&text)
        .map_err(|e| Error(format!("generated module failed to reparse: {e}")))?;
    let mut validation = emit::inline(module.clone(), &data)?;
    let diagnostics = flatppl_infer::infer(&mut validation);
    if let Some(diagnostic) = diagnostics.first() {
        return Err(Error(format!(
            "generated formula model: {}",
            diagnostic.message
        )));
    }
    let info = emit::info(spec, &recipe, &data, &validation)?;
    emit::mark_improper(&mut module, info.has_improper_prior);
    Ok(Conversion {
        module,
        data,
        recipe,
        info,
    })
}
