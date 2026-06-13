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
    #[serde(default)]
    pub entries: Vec<Vec<f64>>,
    /// Binned observation bin contents (histfactory channel observed counts).
    #[serde(default)]
    pub contents: Option<Vec<f64>>,
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
    pub min: f64,
    pub max: f64,
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
    // Axes are parsed for schema fidelity; bin count is taken from sample contents.
    #[allow(dead_code)]
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
    #[serde(default, alias = "name")]
    pub parameter: Option<String>,
    #[serde(default)]
    pub data: Option<serde_json::Value>,
    // Multi-parameter modifier names (currently parsed but not used in logic)
    #[allow(dead_code)]
    #[serde(default)]
    pub parameters: Vec<String>,
    #[serde(default)]
    pub constraint: Option<String>,
    #[serde(default)]
    pub interpolation: Option<String>,
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
    // Parsed for schema fidelity; lumi observed nominal is taken as 1.0.
    #[allow(dead_code)]
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
