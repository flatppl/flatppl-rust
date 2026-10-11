//! pyhf measurement metadata: one literal record per measurement.
//!
//! The record carries what a fit needs and pyhf derives: parameters in pyhf's
//! registration order, with suggested initial values, bounds and fixed flags,
//! constraint widths and nominal auxiliary data, plus the channel layout. It is
//! dead code for every query, so the emitted StableHLO does not change.
use crate::builder::{Builder, sanitize_ident};
use crate::error::{Error, Result};
use crate::histfactory::{shapesys_tau, staterror_delta};
use crate::model::{Modifier, PyhfDocument, PyhfParam};
use flatppl_core::id::NodeId;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// pyhf 0.7.6 builds modifiers in this order (`pyhf.modifiers.histfactory_set`).
const TYPE_ORDER: [&str; 7] = [
    "histosys",
    "lumi",
    "normfactor",
    "normsys",
    "shapefactor",
    "shapesys",
    "staterror",
];

/// The order in which pyhf registers parameters, by pyhf name.
///
/// pyhf scans channels, samples and modifier names in sorted order and lets
/// each modifier type register a parameter at the first sample it affects. The
/// types follow [`TYPE_ORDER`]. Staterror registers after the scan over keys
/// that exist for every channel, so its parameters sort by name. A name shared
/// by two types (normsys and histosys) is one parameter, at its first position.
/// (`pyhf.pdf._nominal_and_modifiers_from_spec`, `pyhf.mixins`.)
pub(crate) fn registration_order(doc: &PyhfDocument) -> Vec<String> {
    let channels: BTreeSet<&str> = doc.channels().iter().map(|c| c.name.as_str()).collect();
    let samples: BTreeSet<&str> = doc
        .channels()
        .iter()
        .flat_map(|c| &c.samples)
        .map(|s| s.name.as_str())
        .collect();
    let mut first: BTreeMap<&str, BTreeMap<String, (usize, usize)>> = BTreeMap::new();
    for (ci, channel) in channels.iter().enumerate() {
        let Some(c) = doc.channels().iter().find(|c| c.name == *channel) else {
            continue;
        };
        for (si, sample) in samples.iter().enumerate() {
            let Some(s) = c.samples.iter().find(|s| s.name == *sample) else {
                continue;
            };
            for m in &s.modifiers {
                if let Some(name) = m.effective_param() {
                    first
                        .entry(m.kind.as_str())
                        .or_default()
                        .entry(name)
                        .or_insert((ci, si));
                }
            }
        }
    }
    let mut order: Vec<String> = Vec::new();
    for kind in TYPE_ORDER {
        let Some(names) = first.get(kind) else {
            continue;
        };
        let mut sorted: Vec<_> = names.iter().collect();
        if kind != "staterror" {
            sorted.sort_by(|(a, pa), (b, pb)| (pa, a).cmp(&(pb, b)));
        }
        for (name, _) in sorted {
            if !order.contains(name) {
                order.push(name.clone());
            }
        }
    }
    order
}

/// One channel's layout, for the record's `channels` field.
#[derive(Debug, Clone)]
pub struct ChannelRecord {
    pub name: String,
    pub bins: usize,
    pub observed: String,
    pub expected: String,
}

/// What the model alone determines about a parameter's paramset.
struct Derived {
    kind: &'static str,
    scalar: bool,
    n: usize,
    init: Option<f64>,
    bounds: Option<(f64, f64)>,
    fixed: Vec<bool>,
    constraint: Option<Constraint>,
}

enum Constraint {
    /// `constrained_by_normal`: `auxdata` and Gaussian widths.
    Normal { auxdata: Vec<f64>, sigmas: Vec<f64> },
    /// `constrained_by_poisson`: `auxdata` and Poisson rate factors.
    Poisson {
        auxdata: Vec<f64>,
        factors: Vec<f64>,
    },
}

/// Inputs the emitted model already computed.
pub(crate) struct Emitted<'a> {
    pub channels: &'a [ChannelRecord],
    /// Channel-summed nominal and squared error per staterror parameter.
    pub staterror: &'a BTreeMap<String, (Vec<f64>, Vec<f64>)>,
}

fn derive(doc: &PyhfDocument, param: &str, emitted: &Emitted<'_>) -> Result<Derived> {
    let occurrences: Vec<(&Modifier, &[f64])> = doc
        .channels()
        .iter()
        .flat_map(|c| &c.samples)
        .flat_map(|s| s.modifiers.iter().map(move |m| (m, s.data.as_slice())))
        .filter(|(m, _)| m.effective_param().as_deref() == Some(param))
        .collect();
    let kind = TYPE_ORDER
        .into_iter()
        .find(|kind| occurrences.iter().any(|(m, _)| m.kind == *kind))
        .ok_or_else(|| Error::Unsupported(format!("parameter `{param}` has no modifier")))?;
    let (modifier, nominal) = occurrences
        .iter()
        .find(|(m, _)| m.kind == kind)
        .copied()
        .expect("kind came from an occurrence");
    let scalar = |init, lo, hi, constraint| Derived {
        kind,
        scalar: true,
        n: 1,
        init: Some(init),
        bounds: Some((lo, hi)),
        fixed: vec![false],
        constraint,
    };
    Ok(match kind {
        "normfactor" => scalar(1.0, 0.0, 10.0, None),
        "normsys" | "histosys" => scalar(
            0.0,
            -5.0,
            5.0,
            Some(Constraint::Normal {
                auxdata: vec![0.0],
                sigmas: vec![1.0],
            }),
        ),
        "lumi" => Derived {
            kind,
            scalar: true,
            n: 1,
            init: None,
            bounds: None,
            fixed: vec![false],
            constraint: Some(Constraint::Normal {
                auxdata: Vec::new(),
                sigmas: Vec::new(),
            }),
        },
        "shapefactor" => Derived {
            kind,
            scalar: false,
            n: nominal.len(),
            init: Some(1.0),
            bounds: Some((0.0, 10.0)),
            fixed: vec![false; nominal.len()],
            constraint: None,
        },
        "shapesys" => {
            let sigma = numbers(modifier)?;
            let factors = shapesys_tau(nominal, &sigma);
            let fixed = nominal
                .iter()
                .zip(&sigma)
                .map(|(n, s)| !(*n > 0.0 && *s > 0.0))
                .collect();
            Derived {
                kind,
                scalar: false,
                n: nominal.len(),
                init: Some(1.0),
                bounds: Some((1e-10, 10.0)),
                fixed,
                constraint: Some(Constraint::Poisson {
                    auxdata: factors.clone(),
                    factors,
                }),
            }
        }
        "staterror" => {
            let (sum_nom, sum_sq) = emitted.staterror.get(param).ok_or_else(|| {
                Error::Unsupported(format!("staterror `{param}` has no accumulated yields"))
            })?;
            // pyhf fixes a component whose relative error is zero and gives it a unit width.
            let fixed = sum_nom
                .iter()
                .zip(sum_sq)
                .map(|(n, sq)| !(*n > 0.0 && sq.sqrt() / n > 0.0))
                .collect();
            Derived {
                kind,
                scalar: false,
                n: sum_nom.len(),
                init: Some(1.0),
                bounds: Some((1e-10, 10.0)),
                fixed,
                constraint: Some(Constraint::Normal {
                    auxdata: vec![1.0; sum_nom.len()],
                    sigmas: staterror_delta(sum_nom, sum_sq),
                }),
            }
        }
        _ => unreachable!("TYPE_ORDER lists only supported kinds"),
    })
}

/// Bind `name = external(cartprod(<param> = <set>, ...))`, one field per
/// constrained parameter in `order`, which is pyhf's `auxdata_order`. Gaussian
/// centres are reals. Poisson centres are counts, possibly fractional.
pub(crate) fn emit_auxdata(
    b: &mut Builder,
    doc: &PyhfDocument,
    emitted: &Emitted<'_>,
    order: &[String],
    name: &str,
) -> Result<()> {
    let mut fields = Vec::new();
    for param in order {
        let derived = derive(doc, param, emitted)?;
        let (set, poisson) = match &derived.constraint {
            None => continue,
            Some(Constraint::Normal { .. }) => ("reals", false),
            Some(Constraint::Poisson { .. }) => ("nonnegreals", true),
        };
        let set = b.call_head(set);
        let domain = if derived.scalar && !poisson {
            set
        } else {
            let n = b.lit_int(derived.n as i64);
            b.call("cartpow", &[set, n])
        };
        fields.push((param.as_str(), domain));
    }
    if fields.is_empty() {
        return Ok(());
    }
    if b.alloc_name(name) != name {
        return Err(Error::Unsupported(format!(
            "the auxiliary-data binding `{name}` collides with a generated binding"
        )));
    }
    let set = b.call_fields("cartprod", &fields);
    let external = b.call("external", &[set]);
    b.bind_doc(
        name,
        external,
        &["Auxiliary-measurement centres, one per constrained parameter in pyhf's `auxdata_order`."],
    );
    Ok(())
}

fn numbers(modifier: &Modifier) -> Result<Vec<f64>> {
    modifier
        .data
        .as_ref()
        .and_then(serde_json::Value::as_array)
        .map(|items| items.iter().filter_map(serde_json::Value::as_f64).collect())
        .ok_or_else(|| {
            Error::Unsupported(format!("`{}` modifier data is not a list", modifier.kind))
        })
}

/// One parameter's values for one measurement: defaults with the measurement
/// configuration's overrides on top (`pyhf.parameters.reduce_paramsets_requirements`).
struct Values {
    init: Option<Vec<f64>>,
    bounds: Option<Vec<(f64, f64)>>,
    fixed: Vec<bool>,
    width: Option<Vec<f64>>,
    auxdata: Option<Vec<f64>>,
}

fn resolve(param: &str, derived: &Derived, config: Option<&PyhfParam>) -> Result<Values> {
    let n = derived.n;
    let sized = |field: &str, len: usize| -> Result<()> {
        if len == n {
            return Ok(());
        }
        // pyhf: InvalidModel.
        Err(Error::Unsupported(format!(
            "configuration sets {len} `{field}` value(s) on parameter `{param}`, which has {n}"
        )))
    };
    let mut init = derived.init.map(|v| vec![v; n]);
    let mut bounds = derived.bounds.map(|v| vec![v; n]);
    let mut fixed = derived.fixed.clone();
    let (mut auxdata, mut width) = match &derived.constraint {
        None => (None, None),
        Some(Constraint::Normal { auxdata, sigmas }) => (
            Some(auxdata.clone()),
            Some(if sigmas.is_empty() && derived.kind != "lumi" {
                vec![1.0; n]
            } else {
                sigmas.clone()
            }),
        ),
        Some(Constraint::Poisson { auxdata, factors }) => {
            (Some(auxdata.clone()), Some(poisson_width(factors)))
        }
    };
    if let Some(config) = config {
        if !config.inits.is_empty() {
            sized("inits", config.inits.len())?;
            init = Some(config.inits.clone());
        }
        if !config.bounds.is_empty() {
            sized("bounds", config.bounds.len())?;
            bounds = Some(config.bounds.iter().map(|[lo, hi]| (*lo, *hi)).collect());
        }
        if let Some(value) = config.fixed {
            fixed = vec![value; n];
        }
        if !config.auxdata.is_empty() && auxdata.is_some() {
            auxdata = Some(config.auxdata.clone());
        }
        match &derived.constraint {
            Some(Constraint::Normal { .. }) if !config.sigmas.is_empty() => {
                width = Some(config.sigmas.clone());
            }
            Some(Constraint::Poisson { .. }) if !config.factors.is_empty() => {
                width = Some(poisson_width(&config.factors));
            }
            _ => {}
        }
    }
    let present = |values: Option<Vec<f64>>| values.filter(|v| !v.is_empty());
    Ok(Values {
        init,
        bounds,
        fixed,
        width: present(width),
        auxdata: present(auxdata),
    })
}

fn poisson_width(factors: &[f64]) -> Vec<f64> {
    factors.iter().map(|f| (1.0 / f).sqrt()).collect()
}

/// Emit one `pyhf_measurement_<name>` record per measurement, or one
/// `pyhf_measurement` record for a model without measurements.
///
/// `order` holds emitted parameter names in pyhf's registration order, and
/// `original` maps an emitted name back to its pyhf name.
pub(crate) fn emit_records(
    b: &mut Builder,
    doc: &PyhfDocument,
    emitted: &Emitted<'_>,
    order: &[String],
    original: &HashMap<String, String>,
) -> Result<()> {
    let derived: Vec<(&String, Derived)> = order
        .iter()
        .map(|p| derive(doc, p, emitted).map(|d| (p, d)))
        .collect::<Result<_>>()?;
    let measurements: Vec<_> = doc.measurements().collect();
    if measurements.is_empty() {
        let record = record(
            b,
            None,
            true,
            None,
            doc.parameters(),
            &derived,
            emitted,
            original,
        )?;
        b.bind_unique_doc(
            "pyhf_measurement",
            record,
            "pyhf fit settings and channel layout.",
        );
        return Ok(());
    }
    let selected = doc.selected_measurement();
    for (index, measurement) in measurements.iter().enumerate() {
        let poi = measurement.config.poi.as_deref().filter(|p| !p.is_empty());
        let node = record(
            b,
            Some(&measurement.name),
            index == selected,
            poi,
            &measurement.config.parameters,
            &derived,
            emitted,
            original,
        )?;
        b.bind_unique_doc(
            &format!("pyhf_measurement_{}", sanitize_ident(&measurement.name)),
            node,
            &format!(
                "pyhf measurement {:?}: fit settings and channel layout.",
                measurement.name
            ),
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record(
    b: &mut Builder,
    name: Option<&str>,
    selected: bool,
    poi: Option<&str>,
    config: &[PyhfParam],
    derived: &[(&String, Derived)],
    emitted: &Emitted<'_>,
    original: &HashMap<String, String>,
) -> Result<NodeId> {
    let mut parameters = Vec::new();
    for (param, d) in derived {
        let values = resolve(param, d, config.iter().find(|p| &p.name == *param))?;
        let reals = |b: &mut Builder, v: &[f64]| -> NodeId {
            if d.scalar {
                b.lit_real(v[0])
            } else {
                let items: Vec<_> = v.iter().map(|x| b.lit_real(*x)).collect();
                b.array(&items)
            }
        };
        let mut fields: Vec<(String, NodeId)> = Vec::new();
        let pyhf_name = original.get(*param).map_or(param.as_str(), String::as_str);
        fields.push(("pyhf".into(), b.str_lit(pyhf_name)));
        if let Some(init) = &values.init {
            fields.push(("init".into(), reals(b, init)));
        }
        if let Some(bounds) = &values.bounds {
            let lo: Vec<f64> = bounds.iter().map(|(lo, _)| *lo).collect();
            let hi: Vec<f64> = bounds.iter().map(|(_, hi)| *hi).collect();
            fields.push(("lo".into(), reals(b, &lo)));
            fields.push(("hi".into(), reals(b, &hi)));
        }
        let fixed = if d.scalar {
            b.lit_bool(values.fixed[0])
        } else {
            let items: Vec<_> = values.fixed.iter().map(|x| b.lit_bool(*x)).collect();
            b.array(&items)
        };
        fields.push(("fixed".into(), fixed));
        if let Some(width) = &values.width {
            fields.push(("width".into(), reals(b, width)));
        }
        if let Some(auxdata) = &values.auxdata {
            fields.push(("auxdata".into(), reals(b, auxdata)));
        }
        let fields: Vec<(&str, NodeId)> = fields.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        parameters.push(((*param).clone(), b.call_fields("record", &fields)));
    }
    let parameters: Vec<(&str, NodeId)> =
        parameters.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let parameters = b.call_fields("record", &parameters);

    let mut used = BTreeSet::new();
    let mut channels = Vec::new();
    for channel in emitted.channels {
        let base = sanitize_ident(&channel.name);
        let mut key = base.clone();
        let mut suffix = 2;
        while !used.insert(key.clone()) {
            key = format!("{base}_{suffix}");
            suffix += 1;
        }
        let name = b.str_lit(&channel.name);
        let bins = b.lit_int(channel.bins as i64);
        let observed = b.str_lit(&channel.observed);
        let expected = b.str_lit(&channel.expected);
        let fields = [
            ("name", name),
            ("bins", bins),
            ("observed", observed),
            ("expected", expected),
        ];
        channels.push((key, b.call_fields("record", &fields)));
    }
    let channels: Vec<(&str, NodeId)> = channels.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let channels = b.call_fields("record", &channels);

    let mut fields: Vec<(&str, NodeId)> = Vec::new();
    if let Some(name) = name {
        fields.push(("name", b.str_lit(name)));
    }
    fields.push(("selected", b.lit_bool(selected)));
    if let Some(poi) = poi {
        fields.push(("poi", b.str_lit(poi)));
    }
    fields.push(("parameters", parameters));
    fields.push(("channels", channels));
    Ok(b.call_fields("record", &fields))
}
