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
//! - `restrict(J, record(o = y))` and the parts of `disintegrate(["o"], J)`,
//!   for `J = lawof(record(…))` with observed leaf fields: the same program as
//!   the explicit `lawof` prior and `kernelof` kernel.
//! - Fixed literals and loaded data go to `data`, fixed computed values to
//!   `transformed data`, and values that depend on parameters to `model`
//!   locals (see [`Placement`]).
//!
//! With `Reduce::sufficient`, observed IID Normal factors use data summaries
//! and a normalized `target +=` expression. Stan keeps every constant of a `~` term when
//! the density is evaluated with `propto=false`, so the program's log density
//! at `propto=false, jacobian=false` equals FlatPPL's, including the
//! normalizer of a truncated law, written `T[L, U]`.
//!
//! A construct with no sensible Stan form refuses with a never-kind
//! [`RefusalKind`]; one this emitter does not write yet refuses with
//! [`RefusalKind::NotYetSupported`].
//!
//! To keep an explicit row Gram product in teaching output, pass a rule policy
//! to [`emit_with_rules`]. [`emit`] uses the standard Stan profile with every
//! eligible rule enabled.
//!
//! ```
//! use flatppl_core::Module;
//! use flatppl_stan::{EmitOptions, RulePolicy, StanProgram, emit_with_rules};
//!
//! fn teaching_stan(module: &Module) -> Result<StanProgram, Box<dyn std::error::Error>> {
//!     let rules = RulePolicy::new(&["stan.row-gram"], &[])?;
//!     Ok(emit_with_rules(module, &EmitOptions::default(), &rules)?)
//! }
//! ```

mod data;
mod emit;
mod refuse;
mod stan_type;
mod tables;

pub use data::DataSource;
pub use flatppl_optimise::{RulePolicy, RuleSelector};
pub use refuse::{Refusal, RefusalKind};

use std::collections::{HashMap, HashSet};

use flatppl_core::{Idx, Module, Symbol};
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

#[derive(Clone, Debug, Default)]
pub struct EmitOptions {
    /// The target binding; `None` picks the module's single posterior (a
    /// `bayesupdate`, or a `restrict` to observed fields).
    pub target: Option<String>,
    /// Where values that depend on parameters go (`--derived`).
    pub derived: Placement,
    /// The opt-in reductions (`--reduce`). All are off by default.
    pub reduce: Reduce,
}

/// Opt-in reductions of the emitted program, one flag per family.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Reduce {
    /// Shared data summaries for observed IID Normal factors.
    pub sufficient: bool,
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
    /// The index of `input` the query binds this field to (FlatPPL
    /// 1-based): `theta[k]` or `theta[[k, …]]`.
    pub index: Option<ManifestIndex>,
}

/// A query point's index into a query input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ManifestIndex {
    /// `theta[k]`: one element, a scalar field.
    Scalar(i64),
    /// `theta[[k, …]]`: these elements, a vector field.
    Slice(Vec<i64>),
}

/// A module input (`inputs = (…)`) that the program declares as data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestData {
    /// The FlatPPL binding name.
    pub input: String,
    /// The Stan data name.
    pub stan: String,
    /// The Stan declaration without the name (`vector[N]`, `real<lower=0>`).
    pub ty: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Manifest {
    pub target: String,
    /// Parameters in declaration order.
    pub parameters: Vec<ManifestParam>,
    /// Module inputs declared as data, in `inputs` order.
    pub data: Vec<ManifestData>,
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
                match &p.index {
                    Some(ManifestIndex::Scalar(k)) => {
                        o.insert("index".into(), Value::from(*k));
                    }
                    Some(ManifestIndex::Slice(ks)) => {
                        o.insert("index".into(), Value::from(ks.clone()));
                    }
                    None => {}
                }
                Value::Object(o)
            })
            .collect();
        let mut o = Map::new();
        o.insert("target".into(), Value::from(self.target.clone()));
        o.insert("parameters".into(), Value::Array(params));
        if !self.data.is_empty() {
            let data = self
                .data
                .iter()
                .map(|d| {
                    let mut o = Map::new();
                    o.insert("input".into(), Value::from(d.input.clone()));
                    o.insert("stan".into(), Value::from(d.stan.clone()));
                    o.insert("type".into(), Value::from(d.ty.clone()));
                    Value::Object(o)
                })
                .collect();
            o.insert("data".into(), Value::Array(data));
        }
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
    emit_with_rules(module, opts, &RulePolicy::default())
}

/// Emit Stan with explicit control of the egglog rules applied to its target.
pub fn emit_with_rules(
    module: &Module,
    opts: &EmitOptions,
    rules: &RulePolicy,
) -> Result<StanProgram, Refusal> {
    emit::program(module, opts, rules)
}

/// [`emit`] for a module that reads `load_module` members: the members are
/// imported into the module's own graph first (§11: tooling may flatten
/// modules internally), and the result is inferred again with `bundle`. A
/// refusal inside an imported member is located at the host node that reads
/// it. A module with no `load_module` binding is emitted as it is.
pub fn emit_with_bundle(
    module: &Module,
    bundle: &flatppl_infer::ModuleBundle,
    opts: &EmitOptions,
) -> Result<StanProgram, Refusal> {
    emit_with_bundle_and_rules(module, bundle, opts, &RulePolicy::default())
}

/// Emit a module and its imports with explicit egglog rule controls.
pub fn emit_with_bundle_and_rules(
    module: &Module,
    bundle: &flatppl_infer::ModuleBundle,
    opts: &EmitOptions,
    rules: &RulePolicy,
) -> Result<StanProgram, Refusal> {
    if !module
        .bindings()
        .any(|(_, b)| is_load_module(module, b.rhs))
    {
        return emit_with_rules(module, opts, rules);
    }
    let grafted = flatppl_determinizer::graft_imports(module, bundle)
        .map_err(|e| Refusal::not_yet(Some(e.node), format!("{}: {}", e.construct, e.reason)))?;
    let mut work = grafted.module.clone();
    let diagnostics = flatppl_infer::infer_module(&mut work, bundle, flatppl_infer::Level::Shape);
    // A static error the first inference could not see, because a size or a
    // value lay in another module (data of 2 values against an imported
    // count of 3).
    if let Some(d) = diagnostics
        .iter()
        .find(|d| d.severity == flatppl_infer::Severity::Error)
    {
        return Err(Refusal::new(
            RefusalKind::IllFormedModule,
            d.node.map(|n| emit::host_node(module, &work, &grafted, n)),
            "the module with its imported members",
            d.message.clone(),
        ));
    }
    let (target, optimised) = emit::prepare(&work, opts, bundle, rules).map_err(|mut r| {
        r.node = r.node.map(|n| emit::host_node(module, &work, &grafted, n));
        r
    })?;
    if let Some(optimised) = optimised {
        work = optimised;
    }
    // A public member that is a positive integer sizes data as in the host.
    let public: HashSet<Symbol> = grafted
        .imports
        .iter()
        .filter(|i| !i.member.starts_with('_'))
        .map(|i| work.binding(i.binding).name)
        .collect();
    // A first emission records the bindings the program names and the names
    // it makes. The graft renames a member that clashes with any host
    // binding, also one the program never prints (an alias
    // `f_b = common.f_b`), so the graft's names do not decide. A display name
    // must know every printed import of its member name, so it cannot be
    // decided at the first `name()` call: two emissions.
    let first = emit::ImportNames {
        display: HashMap::new(),
        public: public.clone(),
    };
    let import_syms: HashSet<Symbol> = grafted
        .imports
        .iter()
        .map(|i| work.binding(i.binding).name)
        .collect();
    // A name the emitter made from an import's own name (its function name)
    // is that import's, not something else's.
    let import_names: HashSet<String> = import_syms
        .iter()
        .map(|&s| emit::sanitize(work.resolve(s)))
        .collect();
    let (printed, mut taken): (Vec<bool>, HashSet<String>) =
        match emit::program_printed(&work, opts, &first, target) {
            (Ok(_), p) => (
                grafted
                    .imports
                    .iter()
                    .map(|i| p.named.contains(&work.binding(i.binding).name))
                    .collect(),
                p.named
                    .iter()
                    .filter(|s| !import_syms.contains(s))
                    .map(|&s| emit::sanitize(work.resolve(s)))
                    .chain(p.used.into_iter().filter(|u| !import_names.contains(u)))
                    .collect(),
            ),
            // The emission below refuses the same way; its text names every
            // import as if printed.
            (Err(_), p) => (
                vec![true; grafted.imports.len()],
                work.bindings()
                    .filter(|(b, _)| b.index() < grafted.host_bindings)
                    .map(|(_, b)| emit::sanitize(work.resolve(b.name)))
                    .chain(p.used.into_iter().filter(|u| !import_names.contains(u)))
                    .collect(),
            ),
        };
    // A printed import keeps its member name. One whose name the program
    // prints for something else, or whose member name comes from more than
    // one printed instance, prints as `{alias}_{member}` (`m_theta`, `m1_c`
    // and `m2_c`), so the text does not depend on graft order.
    let shown: Vec<&flatppl_determinizer::Import> = grafted
        .imports
        .iter()
        .zip(&printed)
        .filter_map(|(i, &p)| p.then_some(i))
        .collect();
    let mut count: HashMap<&str, usize> = HashMap::new();
    for import in &shown {
        *count.entry(import.member.as_str()).or_default() += 1;
    }
    let mut names = emit::ImportNames {
        display: HashMap::new(),
        public,
    };
    for import in shown {
        let name = work.binding(import.binding).name;
        let member = emit::sanitize(&import.member);
        let display = if count[import.member.as_str()] == 1 && !taken.contains(&member) {
            member
        } else {
            // Two members of one name reached through one alias (a module
            // and a module it loads) take the emitter's `_2` suffix.
            let base = emit::sanitize(&format!("{}_{}", import.alias, import.member));
            let mut display = base.clone();
            let mut k = 2;
            while taken.contains(&display) {
                display = format!("{base}_{k}");
                k += 1;
            }
            display
        };
        taken.insert(display.clone());
        if display != work.resolve(name) {
            names.display.insert(name, work.intern(&display));
        }
    }
    emit::program_printed(&work, opts, &names, target)
        .0
        .map_err(|mut r| {
            r.node = r.node.map(|n| emit::host_node(module, &work, &grafted, n));
            r
        })
}

fn is_load_module(m: &Module, id: flatppl_core::NodeId) -> bool {
    matches!(m.node(id), flatppl_core::Node::Call(c)
        if matches!(c.head, flatppl_core::CallHead::Builtin(s) if m.resolve(s) == "load_module"))
}

/// Fill the deferred data values from `loaded`: the same module after the host
/// replaced its `load_data` calls by literals.
pub fn complete_data(program: &mut StanProgram, loaded: &Module) -> Result<(), Refusal> {
    data::complete(&mut program.data, &program.deferred_data, loaded)?;
    program.deferred_data.clear();
    Ok(())
}
