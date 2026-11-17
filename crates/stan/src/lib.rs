//! `flatppl-stan` — emits a Stan program from a FlatPPL model.
//!
//! The emitter is structural: it reads the type-inferred surface module (not
//! determinized FlatPDL) and writes the program a person would write. One
//! target binding becomes one Stan program:
//!
//! - `bayesupdate(L, prior)`: each prior record field is a parameter, typed and
//!   bounded by the posterior's inferred domain, with `field ~ law;`. An
//!   improper `Lebesgue` part is a parameter with no `~`.
//! - `likelihoodof(kernelof(record(o = y), inputs…), obs)`: each observed
//!   value is a data variable and `o ~ law(...)` its likelihood term. With no
//!   prior, every kernel input is a flat parameter.
//! - Fixed literals and loaded data go to `data`, fixed computed values to
//!   `transformed data`, and values that depend on parameters to `model`
//!   locals (see [`Placement`]).
//!
//! Observed IID Normal factors can use data summaries and a normalized
//! `target +=` expression. Stan keeps every constant of a `~` term when
//! the density is evaluated with `propto=false`, so the program's log density
//! at `propto=false, jacobian=false` equals FlatPPL's, including the
//! normalizer of a truncated law, written `T[L, U]`.
//!
//! A construct with no sensible Stan form refuses with a never-kind
//! [`RefusalKind`]; one this emitter does not write yet refuses with
//! [`RefusalKind::NotYetSupported`].

mod data;
mod emit;
mod refuse;
mod stan_type;
mod tables;

pub use data::DataSource;
pub use refuse::{Refusal, RefusalKind};

use flatppl_core::Module;
use serde_json::{Map, Value};

/// Where values that depend on parameters go.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Placement {
    /// `model`-block locals: not written to the output.
    #[default]
    ModelLocals,
    /// `transformed parameters`: written to the output on every draw.
    TransformedParameters,
}

#[derive(Clone, Debug)]
pub struct EmitOptions {
    /// The target binding; `None` picks the module's single `bayesupdate`.
    pub target: Option<String>,
    /// Where values that depend on parameters go (`--derived`).
    pub derived: Placement,
    /// Prepare shared data summaries for supported observed IID factors (default: true).
    pub sufficient_statistics: bool,
}

impl Default for EmitOptions {
    fn default() -> Self {
        Self {
            target: None,
            derived: Placement::default(),
            sufficient_statistics: true,
        }
    }
}

/// One Stan parameter and the FlatPPL field it stands for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestParam {
    pub stan: String,
    /// The prior record field or kernel input name.
    pub field: String,
    /// The query input bound to this field by `outputs = logdensityof(target,
    /// record(field = input, …))`, when the module declares one.
    pub input: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Manifest {
    pub target: String,
    /// Parameters in declaration order.
    pub parameters: Vec<ManifestParam>,
}

impl Placement {
    /// The `--derived` spelling.
    pub fn parse(s: &str) -> Option<Placement> {
        match s {
            "model" => Some(Placement::ModelLocals),
            "transformed-parameters" => Some(Placement::TransformedParameters),
            _ => None,
        }
    }
}

impl Manifest {
    pub fn to_json(&self) -> Value {
        let params = self
            .parameters
            .iter()
            .map(|p| {
                let mut o = Map::new();
                o.insert("stan".into(), Value::from(p.stan.clone()));
                o.insert("field".into(), Value::from(p.field.clone()));
                if let Some(i) = &p.input {
                    o.insert("input".into(), Value::from(i.clone()));
                }
                Value::Object(o)
            })
            .collect();
        let mut o = Map::new();
        o.insert("target".into(), Value::from(self.target.clone()));
        o.insert("parameters".into(), Value::Array(params));
        Value::Object(o)
    }
}

/// An emitted program.
#[derive(Clone, Debug)]
pub struct StanProgram {
    pub code: String,
    /// Stan JSON data: literal values now, `load_data` values after
    /// [`complete_data`].
    pub data: Map<String, Value>,
    /// Data variables still to be read from `load_data` sources.
    pub deferred_data: Vec<DataSource>,
    pub manifest: Manifest,
}

/// Emit the Stan program for one target of an inferred module.
pub fn emit(module: &Module, opts: &EmitOptions) -> Result<StanProgram, Refusal> {
    emit::program(module, opts)
}

/// Fill the deferred data values from `loaded`: the same module after the host
/// replaced its `load_data` calls by literals.
pub fn complete_data(program: &mut StanProgram, loaded: &Module) -> Result<(), Refusal> {
    data::complete(&mut program.data, &program.deferred_data, loaded)?;
    program.deferred_data.clear();
    Ok(())
}
