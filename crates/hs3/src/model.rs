//! Serde structs mirroring HS3 v0.2.9 JSON. Fields are a superset-tolerant subset:
//! unknown keys are ignored; `parameter`/`name` and `entries`/`parameters` aliases
//! cover the spec-vs-paper-vs-ROOT naming differences.
use serde::Deserialize;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// HS3 `functions` block
// ---------------------------------------------------------------------------

/// A single HS3 `functions` entry (product, sum, or generic_function).
#[derive(Debug, Deserialize)]
pub struct Function {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: String,
    /// All remaining fields (type-specific); e.g. `factors`, `summands`, `expression`.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Document {
    #[serde(default)]
    pub distributions: Vec<Distribution>,
    #[serde(default)]
    pub likelihoods: Vec<Likelihood>,
    #[serde(default)]
    pub domains: Vec<Domain>,
    #[serde(default)]
    pub parameter_points: Vec<ParameterPoint>,
    /// HS3 unbinned/binned data sets (paper § A.1/A.2).
    #[serde(default)]
    pub data: Vec<Datum>,
    /// HS3 `functions` block: deterministic derived quantities.
    #[serde(default)]
    pub functions: Vec<Function>,
    /// HS3 `analyses` block (paper § A.5): named analysis configurations binding
    /// a likelihood to a POI / domain. This is inference configuration, out of
    /// scope for a model-only importer — it is intentionally not lowered. Parsed
    /// for schema fidelity (and so its presence is not mistaken for an unknown key).
    #[allow(dead_code)]
    #[serde(default)]
    pub analyses: Vec<serde_json::Value>,
}

/// An HS3 data set (unbinned or binned). `unbinned` data uses `entries`; the
/// `binned` histfactory observation uses `contents`. Other types are accepted
/// without error.
#[derive(Debug, Deserialize)]
pub struct Datum {
    pub name: String,
    #[serde(rename = "type", default)]
    pub kind: String,
    /// Unbinned entries: each inner vec is one event's coordinates.
    // NOTE: deserialized in full, allocating proportional to the input document
    // size with no entry-count cap. Fine for the current CLI-on-local-file threat
    // model; if untrusted HS3 documents are ever ingested non-interactively, add a
    // bound on the number of entries (e.g. a custom deserializer with a limit).
    #[serde(default)]
    pub entries: Vec<Vec<f64>>,
    /// Binned observation bin contents (histfactory channel observed counts).
    #[serde(default)]
    pub contents: Option<Vec<f64>>,
    /// Observable axes. Only each axis's `name` is read — it identifies an
    /// observable variable (used to infer the observable of a generic_dist /
    /// generic_function expression). `DomainAxis` parses `{name, min?, max?}`
    /// and ignores any extra keys (e.g. `nbins`).
    #[serde(default)]
    pub axes: Vec<DomainAxis>,
}

#[derive(Debug, Deserialize)]
pub struct Distribution {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ParameterPoint {
    pub name: String,
    #[serde(default, alias = "parameters")]
    pub entries: Vec<ParamValue>,
}

#[derive(Debug, Deserialize)]
pub struct ParamValue {
    pub name: String,
    pub value: f64,
    #[serde(default)]
    pub r#const: bool,
}

#[derive(Debug, Deserialize)]
pub struct Domain {
    pub name: String,
    #[serde(default)]
    pub axes: Vec<DomainAxis>,
}

#[derive(Debug, Deserialize)]
pub struct DomainAxis {
    pub name: String,
    /// Optional. The HS3 spec text lists `min`/`max` as required, but RooFit's
    /// HS3 export omits a bound for an unbounded parameter range (e.g. a global
    /// observable on `[0, ∞)`). We tolerate the omission (treated as ±∞).
    pub min: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct Likelihood {
    pub name: String,
    #[serde(default)]
    pub distributions: Vec<String>,
    #[serde(default)]
    pub data: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SampleData {
    Struct {
        contents: Vec<f64>,
        #[serde(default)]
        errors: Vec<f64>,
    },
    Flat(Vec<f64>),
}

impl SampleData {
    /// Bin contents (nominal yields), regardless of variant.
    pub fn contents(&self) -> &[f64] {
        match self {
            SampleData::Struct { contents, .. } => contents,
            SampleData::Flat(v) => v,
        }
    }

    /// Per-bin errors (empty for the flat variant or when omitted).
    pub fn errors(&self) -> &[f64] {
        match self {
            SampleData::Struct { errors, .. } => errors,
            SampleData::Flat(_) => &[],
        }
    }
}

// ---- Native HS3 histfactory_dist ----

/// A native HS3 `histfactory_dist` distribution body (paper § A.3), deserialized
/// from the `Distribution.extra` map.
#[derive(Debug, Deserialize)]
pub struct HistFactory {
    // Parsed for schema fidelity only — the bin count is taken from sample
    // contents, never from the axes. `#[serde(default)]` because we never read it:
    // requiring a field we don't use would hard-fail a document for no reason if a
    // producer omits it. (ROOT writes `axes` unconditionally, so this is defensive
    // robustness against other HS3 producers, not a ROOT-omission workaround.)
    #[allow(dead_code)]
    #[serde(default)]
    pub axes: Vec<HfAxis>,
    pub samples: Vec<Sample>,
}

/// A binning axis. Parsed for schema fidelity; the converter derives bin count
/// from sample contents rather than the axis metadata.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct HfAxis {
    pub name: String,
    pub nbins: Option<usize>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub edges: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
pub struct Sample {
    pub name: String,
    pub data: SampleData,
    #[serde(default)]
    pub modifiers: Vec<Modifier>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Modifier {
    #[serde(rename = "type")]
    pub kind: String,
    /// The controlling parameter (modern HS3 single-parameter modifiers:
    /// normfactor/normsys/histosys). Kept SEPARATE from `name` — modern HS3
    /// carries BOTH (`name` = modifier instance, `parameter` = nuisance), and
    /// aliasing them would reject such documents as a duplicate field.
    #[serde(default)]
    pub parameter: Option<String>,
    /// Modifier instance name. The pyhf / paper-appendix form names the
    /// controlling parameter HERE (no separate `parameter`); see
    /// [`Modifier::effective_param`].
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub data: Option<serde_json::Value>,
    /// Per-bin parameter names of a multi-bin modifier (modern HS3
    /// staterror/shapefactor). Collapsed to a single vector-binding name by
    /// [`Modifier::effective_param`].
    #[serde(default)]
    pub parameters: Vec<String>,
    /// Constraint *type* string (`Gauss`/`Gaussian`/`Poisson`/…). Consulted for
    /// staterror (ROOT default Poisson; `Gauss`/`Gaussian` ⇒ Normal);
    /// normsys/histosys constraints are fixed by their interpolation.
    ///
    /// `constraint_type` is the modern-HS3 / ROOT spelling of this same type
    /// string — ROOT's HS3 writer emits `constraint_type`, while the pyhf / paper
    /// dialect emits `constraint` — so we accept both via the alias. (Without this
    /// a ROOT-exported Gaussian-constrained staterror was silently mislowered to
    /// the Poisson default.) ROOT's *other* form, `constraint_name` — a reference
    /// to a named constraint pdf, used for custom constraints — is NOT resolved
    /// here: such a modifier falls back to the default (Poisson). Resolving named
    /// constraints needs a distributions-block lookup; deferred until a real file
    /// needs it.
    #[serde(default, alias = "constraint_type")]
    pub constraint: Option<String>,
    #[serde(default)]
    pub interpolation: Option<String>,
}

impl Modifier {
    /// The single FlatPPL binding name this modifier's free parameter(s) map
    /// to, resolved across the HS3 dialects: an explicit `parameter` (modern),
    /// else `name` (pyhf / paper appendix), else the collapsed `parameters`
    /// array (modern multi-bin staterror/shapefactor). `None` only when a
    /// modifier carries no parameter identity at all.
    pub fn effective_param(&self) -> Option<String> {
        if let Some(p) = &self.parameter {
            return Some(p.clone());
        }
        if let Some(n) = &self.name {
            return Some(n.clone());
        }
        if !self.parameters.is_empty() {
            return Some(derive_vector_param_name(&self.parameters));
        }
        None
    }
}

/// Collapse a modern-HS3 per-bin `parameters` array (e.g.
/// `["gamma_stat_0", "gamma_stat_1"]`) to a single FlatPPL vector-binding
/// name: the longest common prefix with trailing separators/digits trimmed
/// (`gamma_stat_0`/`gamma_stat_1` → `gamma_stat`), falling back to the first
/// entry. Identical arrays yield the same name, preserving correlation.
pub fn derive_vector_param_name(parameters: &[String]) -> String {
    let first = &parameters[0];
    let mut prefix_len = first.len();
    for p in &parameters[1..] {
        let common = first
            .bytes()
            .zip(p.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        prefix_len = prefix_len.min(common);
    }
    let trimmed = first[..prefix_len].trim_end_matches(|c: char| c == '_' || c.is_ascii_digit());
    if trimmed.is_empty() {
        first.clone()
    } else {
        trimmed.to_string()
    }
}

// ---- pyhf top-level document ----

/// Top-level pyhf JSON document.
///
/// Supports two schemas:
/// - **New (workspace):** `observations: [{name, data:[...]}]` and top-level `measurements`.
/// - **Old (HistFactory dump):** `data: {"<channel>": [...]}` map and `toplvl.measurements`.
#[derive(Debug, Deserialize)]
pub struct PyhfDocument {
    pub channels: Vec<PyhfChannel>,
    /// New-format per-channel observations (list of `{name, data}`).
    #[serde(default)]
    pub observations: Vec<PyhfObservation>,
    /// New-format top-level measurements.
    #[serde(default)]
    pub measurements: Vec<PyhfMeasurement>,
    /// Old-format observations: a map from channel name → data vector.
    #[serde(default)]
    pub data: Option<BTreeMap<String, Vec<f64>>>,
    /// Old-format measurements nested under `toplvl`.
    #[serde(default)]
    pub toplvl: Option<PyhfToplvl>,
}

/// Old-format `toplvl` wrapper holding measurements.
#[derive(Debug, Deserialize, Default)]
pub struct PyhfToplvl {
    #[serde(default)]
    pub measurements: Vec<PyhfMeasurement>,
}

#[derive(Debug, Deserialize)]
pub struct PyhfChannel {
    pub name: String,
    #[serde(default)]
    pub samples: Vec<PyhfSample>,
}

#[derive(Debug, Deserialize)]
pub struct PyhfSample {
    // used in iteration 2 (diagnostic/logging per sample)
    #[allow(dead_code)]
    pub name: String,
    pub data: Vec<f64>,
    #[serde(default)]
    pub modifiers: Vec<Modifier>,
}

#[derive(Debug, Deserialize)]
pub struct PyhfObservation {
    pub name: String,
    pub data: Vec<f64>,
}

/// A parameter entry in a measurement config (for lumi and other constrained params).
#[derive(Debug, Deserialize)]
pub struct PyhfParam {
    pub name: String,
    // Lumi observed nominal: `auxdata.first()` is used (default 1.0 when absent).
    #[serde(default)]
    pub auxdata: Vec<f64>,
    #[serde(default)]
    pub sigmas: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct PyhfMeasurement {
    #[allow(dead_code)]
    pub name: String,
    #[serde(default)]
    pub config: PyhfMeasurementConfig,
}

#[derive(Debug, Deserialize, Default)]
pub struct PyhfMeasurementConfig {
    #[allow(dead_code)]
    #[serde(default)]
    pub poi: Option<String>,
    #[serde(default)]
    pub parameters: Vec<PyhfParam>,
}

#[cfg(test)]
mod tests {
    use super::*;
    const MINIMAL: &str = r#"{
      "distributions": [
        {"name": "mass", "type": "gaussian_dist",
         "mean": "mu_param", "sigma": "sigma_param", "x": "mass_obs"}
      ],
      "parameter_points": [
        {"name": "default", "entries": [
          {"name": "mu_param", "value": 5.28},
          {"name": "sigma_param", "value": 0.003}
        ]}
      ]
    }"#;
    #[test]
    fn parse_minimal_gaussian() {
        let doc: Document = serde_json::from_str(MINIMAL).unwrap();
        assert_eq!(doc.distributions.len(), 1);
        assert_eq!(doc.distributions[0].name, "mass");
        assert_eq!(doc.distributions[0].kind, "gaussian_dist");
        let pp = &doc.parameter_points[0];
        assert_eq!(pp.entries.len(), 2);
        assert_eq!(pp.entries[0].name, "mu_param");
        assert_eq!(pp.entries[0].value, 5.28);
    }

    #[test]
    fn modifier_accepts_root_constraint_type_alias() {
        // ROOT's native-HS3 writer emits the constraint *type* string under
        // `constraint_type`; the pyhf / paper dialect uses `constraint`. Both must
        // land in the same field, else a Gaussian-constrained staterror is
        // silently mislowered to the Poisson default.
        let root: Modifier = serde_json::from_str(
            r#"{"type":"staterror","parameter":"gamma","constraint_type":"Gaussian"}"#,
        )
        .unwrap();
        assert_eq!(root.constraint.as_deref(), Some("Gaussian"));

        let pyhf: Modifier = serde_json::from_str(
            r#"{"type":"staterror","parameter":"gamma","constraint":"Poisson"}"#,
        )
        .unwrap();
        assert_eq!(pyhf.constraint.as_deref(), Some("Poisson"));
    }

    #[test]
    fn sample_data_struct_variant() {
        let json = r#"{"contents":[1.0,2.0],"errors":[]}"#;
        let sd: SampleData = serde_json::from_str(json).unwrap();
        match sd {
            SampleData::Struct { contents, errors } => {
                assert_eq!(contents, vec![1.0, 2.0]);
                assert_eq!(errors, vec![] as Vec<f64>);
            }
            SampleData::Flat(_) => panic!("expected Struct variant"),
        }
    }

    #[test]
    fn sample_data_flat_variant() {
        let json = r#"[1.0,2.0]"#;
        let sd: SampleData = serde_json::from_str(json).unwrap();
        match sd {
            SampleData::Flat(v) => assert_eq!(v, vec![1.0, 2.0]),
            SampleData::Struct { .. } => panic!("expected Flat variant"),
        }
    }
}
