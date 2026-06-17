//! pyhf workspace JSON → FlatPPL module assembly.
//!
//! Implements the lift described in §12 "pyhf uncorrelated_background" of the
//! profiles doc: each channel → `broadcast(Poisson, expected)` obs model, with
//! modifier effects generating auxiliary likelihood terms as needed.
use crate::builder::Builder;
use crate::error::{Error, Result};
use crate::histfactory::{
    Effect, ParamDomain, mod_spec, modifier_effect, require_param, require_spec, sample_nominal,
    staterror_aux,
};
use crate::model::{PyhfDocument, PyhfParam, SampleData};
use flatppl_core::Module;
use flatppl_core::id::NodeId;
use flatppl_core::node::{Call, CallHead, Node};
use std::collections::HashSet;

/// Convert a pyhf workspace document into a FlatPPL [`Module`].
pub fn pyhf_to_module(doc: &PyhfDocument) -> Result<Module> {
    let mut m = Module::new();
    {
        let mut b = Builder::new(&mut m);
        emit_pyhf(&mut b, doc)?;
    }
    Ok(m)
}

fn emit_pyhf(b: &mut Builder, doc: &PyhfDocument) -> Result<()> {
    // `hepphys = standard_module("particle-physics", "0.1")`
    emit_standard_module(b);

    // Per channel: declare free params, build expected, obs model, likelihoods.
    for channel in &doc.channels {
        emit_channel(b, doc, channel)?;
    }

    // Measurement parameter-of-interest → record binding (so `config.poi`
    // survives the lift; FlatPPL has no dedicated POI construct).
    emit_poi(b, doc);

    Ok(())
}

/// Emit each measurement's parameter-of-interest as `<measurement> = record(poi = <param>)`.
///
/// pyhf's `config.poi` names a free parameter already declared by a modifier; a
/// record preserves the association without inventing a language construct.
/// Measurements from either schema (top-level `measurements` or old-format
/// `toplvl.measurements`) are covered.
fn emit_poi(b: &mut Builder, doc: &PyhfDocument) {
    let toplvl = doc.toplvl.iter().flat_map(|t| t.measurements.iter());
    for meas in doc.measurements.iter().chain(toplvl) {
        match &meas.config.poi {
            // An empty `poi` string means "no POI declared" — skip it (emitting
            // `record(poi = )` would be syntactically invalid).
            Some(poi) if !poi.is_empty() => {
                let poi_ref = b.self_ref(poi);
                let rec = b.call_kw("record", &[("poi", poi_ref)]);
                b.bind(&meas.name, rec);
            }
            _ => {}
        }
    }
}

/// Emit `hepphys = standard_module("particle-physics", "0.1")`.
pub(crate) fn emit_standard_module(b: &mut Builder) {
    let name_arg = b.str_lit("particle-physics");
    let ver_arg = b.str_lit("0.1");
    let head = b.sym("standard_module");
    let node = b.m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: vec![name_arg, ver_arg].into(),
        named: Vec::new().into(),
        inputs: None,
    }));
    b.bind("hepphys", node);
}

/// The lumi constraint's resolved config: the Normal `sigma` and the observed
/// point `nom` (pyhf `auxdata`, §12:208 "observed at lumi_nom").
#[derive(Debug, Clone, Copy)]
pub struct LumiConfig {
    /// Normal constraint width (lumi `sigmas[0]`).
    pub sigma: f64,
    /// Observed value the constraint is evaluated at (lumi `auxdata[0]`, default 1.0).
    pub nom: f64,
}

/// Find the lumi parameter config entry across both new- and old-format measurements.
fn find_lumi_param(doc: &PyhfDocument) -> Option<&PyhfParam> {
    // New format: top-level measurements
    for m in &doc.measurements {
        for p in &m.config.parameters {
            if p.name == "lumi" {
                return Some(p);
            }
        }
    }
    // Old format: toplvl.measurements
    if let Some(toplvl) = &doc.toplvl {
        for m in &toplvl.measurements {
            for p in &m.config.parameters {
                if p.name == "lumi" {
                    return Some(p);
                }
            }
        }
    }
    None
}

fn emit_channel(
    b: &mut Builder,
    doc: &PyhfDocument,
    channel: &crate::model::PyhfChannel,
) -> Result<()> {
    let channel_name = &channel.name;

    // Build the generic sample view: (name, nominal, modifiers).
    let samples: Vec<(&str, &[f64], &[crate::model::Modifier])> = channel
        .samples
        .iter()
        .map(|s| (s.name.as_str(), s.data.as_slice(), s.modifiers.as_slice()))
        .collect();

    // Resolve observation for this channel into an array node (and its bin count).
    let (observed, n_observed) = find_obs(b, doc, channel_name)?;

    // Resolve lumi config (sigma) if any sample carries a lumi modifier.
    let has_lumi = channel.samples.iter().any(|s| {
        s.modifiers
            .iter()
            .any(|m| mod_spec(&m.kind).is_some_and(|spec| spec.channel_lumi))
    });
    let lumi = if has_lumi {
        let lumi_cfg = find_lumi_param(doc).ok_or_else(|| {
            Error::Unsupported(
                "lumi modifier present but no `lumi` parameter entry found in measurement config \
                 (need `sigmas` and `auxdata`)"
                    .into(),
            )
        })?;
        let sigma = *lumi_cfg
            .sigmas
            .first()
            .ok_or_else(|| Error::Unsupported("lumi config `sigmas` array is empty".into()))?;
        // The constraint is observed at `lumi_nom` (§12:208); pyhf carries this in
        // the lumi parameter's `auxdata`. Default to 1.0 when the array is absent
        // (the pyhf convention), matching prior hardcoded behaviour.
        let nom = lumi_cfg.auxdata.first().copied().unwrap_or(1.0);
        Some(LumiConfig { sigma, nom })
    } else {
        None
    };

    assemble_channel(b, channel_name, &samples, observed, n_observed, lumi)
}

/// Assemble one channel's observation model and auxiliary likelihood terms,
/// shared by the pyhf and native HS3 `histfactory_dist` paths.
///
/// `samples` is a list of `(name, nominal_bins, modifiers)`. Staterror errors
/// are read from each staterror modifier's `data` array (an array of per-bin
/// errors); the native path injects the sample's `errors` there before calling.
///
/// Steps: declare free params, fold modifiers via `modifier_effect`/`Effect`
/// (histosys replaces nominal), aggregate staterror across staterror-carrying
/// samples into one Gauss aux per bin, emit one lumi aux (if `lumi` is `Some`),
/// per-sample normsys aux, sum samples → `broadcast(Poisson, expected)`, then
/// `joint_likelihood(L_obs, aux...)`.
pub fn assemble_channel(
    b: &mut Builder,
    channel_name: &str,
    samples: &[(&str, &[f64], &[crate::model::Modifier])],
    observed: NodeId,
    n_observed: usize,
    lumi: Option<LumiConfig>,
) -> Result<()> {
    // ---- Pass 0: validate channel shape ----
    //
    // Every sample must have the same number of bins, and that count must match
    // the observed-data length. A degenerate (empty / ragged) channel would emit a
    // `broadcast(Poisson, [])` or length-mismatched model that is silently wrong.
    let n_bins = samples
        .first()
        .map(|(_, nominal, _)| nominal.len())
        .ok_or_else(|| Error::Unsupported(format!("channel `{channel_name}` has no samples")))?;
    if n_bins == 0 {
        return Err(Error::Unsupported(format!(
            "channel `{channel_name}`: samples have zero bins"
        )));
    }
    for (name, nominal, _) in samples {
        if nominal.len() != n_bins {
            return Err(Error::Unsupported(format!(
                "channel `{channel_name}`: sample `{name}` has {} bins but expected {n_bins}",
                nominal.len()
            )));
        }
    }
    if n_observed != n_bins {
        return Err(Error::Unsupported(format!(
            "channel `{channel_name}`: observed data has {n_observed} bins but samples have {n_bins}"
        )));
    }

    // ---- Pass 1: aggregate staterror nominals and errors across samples ----
    //
    // Staterror is channel-shared: one gamma per named staterror param, one aux
    // per bin.  Only samples that carry a staterror modifier contribute to the
    // sum (nominals and squared errors) for that parameter.
    //
    // Key: staterror parameter name
    // Value: (sum_nom[b], sum_sq_err[b]) accumulated over staterror samples
    let mut staterror_acc: std::collections::BTreeMap<String, (Vec<f64>, Vec<f64>)> =
        std::collections::BTreeMap::new();
    // Per-staterror-param constraint type (None ⇒ ROOT default Poisson).
    let mut staterror_constraint: std::collections::BTreeMap<String, Option<String>> =
        std::collections::BTreeMap::new();

    for (name, nominal, modifiers) in samples {
        for modifier in *modifiers {
            if mod_spec(&modifier.kind).is_some_and(|spec| spec.channel_staterror) {
                let param_name = modifier.effective_param().ok_or_else(|| {
                    Error::Unsupported(format!(
                        "channel `{channel_name}`: staterror modifier on sample `{name}` is \
                         missing its `parameter`"
                    ))
                })?;
                let param_name: &str = param_name.as_str();
                // Per-bin errors are required and must match the channel bin count;
                // a ragged array used to be silently truncated/zero-padded.
                let arr = modifier
                    .data
                    .as_ref()
                    .and_then(|d| d.as_array())
                    .ok_or_else(|| {
                        Error::Unsupported(format!(
                            "channel `{channel_name}`: staterror `{param_name}` on sample \
                             `{name}` is missing its per-bin error array"
                        ))
                    })?;
                if arr.len() != n_bins {
                    return Err(Error::Unsupported(format!(
                        "channel `{channel_name}`: staterror `{param_name}` on sample `{name}` \
                         has {} bins but the channel has {n_bins}",
                        arr.len()
                    )));
                }
                let mut errors: Vec<f64> = Vec::with_capacity(n_bins);
                for (i, v) in arr.iter().enumerate() {
                    let e = v.as_f64().ok_or_else(|| {
                        Error::Unsupported(format!(
                            "channel `{channel_name}`: staterror `{param_name}` error {i} on \
                             sample `{name}` is not a number"
                        ))
                    })?;
                    errors.push(e);
                }

                staterror_constraint
                    .entry(param_name.to_string())
                    .or_insert_with(|| modifier.constraint.clone());
                let entry = staterror_acc
                    .entry(param_name.to_string())
                    .or_insert_with(|| (vec![0.0; n_bins], vec![0.0; n_bins]));
                for i in 0..n_bins {
                    entry.0[i] += nominal[i]; // sum of nominals
                    entry.1[i] += errors[i] * errors[i]; // sum of squared errors
                }
            }
        }
    }

    // ---- Pass 2: declare free params (idempotent by name) ----
    let mut declared: HashSet<String> = HashSet::new();

    for (_name, _nominal, modifiers) in samples {
        for modifier in *modifiers {
            // `declare_modifier_param` validates the modifier kind and requires a
            // `parameter`; only dedupe once we know the name is present.
            let param_name = declare_modifier_param(b, modifier, n_bins, &declared)?;
            declared.insert(param_name);
        }
    }

    // ---- Pass 3: lumi aux once (if present) ----
    let mut aux_terms: Vec<NodeId> = Vec::new();

    if let Some(LumiConfig { sigma, nom }) = lumi {
        let lam = b.self_ref("lumi");
        let sigma_node = b.lit_real(sigma);
        let nom_node = b.lit_real(nom);
        let lumi_normal = b.call_kw("Normal", &[("mu", lam), ("sigma", sigma_node)]);
        let lumi_aux = b.call("likelihoodof", &[lumi_normal, nom_node]);
        aux_terms.push(lumi_aux);
    }

    // ---- Pass 4: emit staterror per-bin aux terms ----
    // Channel-summed Barlow–Beeston: per bin `sum_nom` (Σ sample nominals) and
    // `sum_sq` (Σ squared errors). staterror_aux derives δ (Gaussian) or
    // τ = sum_nom²/sum_sq (Poisson) from these directly — exact, no √-then-square
    // round-trip.
    for (param_name, (sum_nom, sum_sq)) in &staterror_acc {
        let gamma = b.self_ref(param_name);
        // ROOT default is Poisson; `constraint: "Gauss"`/`"Gaussian"` selects Normal.
        let gaussian = matches!(
            staterror_constraint
                .get(param_name)
                .and_then(|c| c.as_deref()),
            Some("Gauss") | Some("Gaussian")
        );
        let aux = staterror_aux(b, gamma, sum_nom, sum_sq, gaussian);
        aux_terms.push(aux);
    }

    // ---- Pass 5: build per-sample expected vectors ----
    let mut sample_expected: Vec<NodeId> = Vec::new();

    for (_name, nominal, modifiers) in samples {
        let data = SampleData::Flat(nominal.to_vec());
        let mut nom = sample_nominal(b, &data);

        // First pass: apply nominal-replacing modifiers (histosys).
        for modifier in *modifiers {
            if mod_spec(&modifier.kind).is_some_and(|spec| spec.replaces_nominal) {
                let effect = modifier_effect(b, modifier, nom, n_bins)?;
                match effect {
                    Effect::HistoSys { new_nom, aux } => {
                        nom = new_nom;
                        aux_terms.push(aux);
                    }
                    _ => unreachable!("replaces_nominal modifier must produce HistoSys effect"),
                }
            }
        }

        // Second pass: apply all multiplicative modifiers. Every non-replacing
        // effect reduces to `acc = broadcast(mul, acc, factor)` plus an optional
        // aux term, so extract `(factor, Option<aux>)` and emit one broadcast.
        let mut acc = nom;
        for modifier in *modifiers {
            if mod_spec(&modifier.kind).is_some_and(|spec| spec.replaces_nominal) {
                continue; // already handled above
            }
            let effect = modifier_effect(b, modifier, nom, n_bins)?;
            let (factor, aux) = match effect {
                Effect::MulParam(param) => (param, None),
                Effect::MulGammaWithAux { gamma, aux } => (gamma, Some(aux)),
                Effect::NormSys { factor, aux } => (factor, Some(aux)),
                Effect::MulLumi(lam) => (lam, None),
                Effect::MulStaterrGamma(gamma) => (gamma, None),
                Effect::MulShapefactorGamma(gamma) => (gamma, None),
                Effect::HistoSys { .. } => unreachable!("histosys handled in first pass"),
            };
            let mul = b.call_head("mul");
            acc = b.call("broadcast", &[mul, acc, factor]);
            if let Some(aux) = aux {
                aux_terms.push(aux);
            }
        }
        sample_expected.push(acc);
    }

    // Fold samples: `expected_total = broadcast(add, s0, s1, ...)` (pairwise fold).
    let expected = if sample_expected.is_empty() {
        b.array(&[])
    } else {
        let mut acc = sample_expected[0];
        for &next in &sample_expected[1..] {
            let add = b.call_head("add");
            acc = b.call("broadcast", &[add, acc, next]);
        }
        acc
    };

    // `obs_model = broadcast(Poisson, expected)`
    let poisson = b.call_head("Poisson");
    let obs_model = b.call("broadcast", &[poisson, expected]);
    let obs_model_name = format!("obs_model_{channel_name}");
    let obs_doc = format!(
        "HS3 histfactory channel '{}': samples × modifiers → broadcast(Poisson, expected)",
        channel_name
    );
    b.bind_doc(&obs_model_name, obs_model, &[obs_doc.as_str()]);

    // `L_obs = likelihoodof(obs_model, observed)`
    let obs_model_ref = b.self_ref(&obs_model_name);
    let l_obs = b.call("likelihoodof", &[obs_model_ref, observed]);

    // Combine into final likelihood.
    let l_name = format!("L_{channel_name}");
    let l_combined = if aux_terms.is_empty() {
        l_obs
    } else {
        let mut all_terms = vec![l_obs];
        all_terms.extend_from_slice(&aux_terms);
        b.call("joint_likelihood", &all_terms)
    };
    let l_doc = "HS3 histfactory likelihood: main Poisson term + auxiliary constraint terms (joint_likelihood)";
    b.bind_doc(&l_name, l_combined, &[l_doc]);

    Ok(())
}

/// Validate a modifier and declare its free-parameter binding, returning the
/// parameter name. A modifier of a known kind that is missing its `parameter`, or
/// of an unknown kind, is rejected here rather than emitting a malformed binding.
/// If the parameter is already in `declared`, the binding is skipped (idempotent)
/// but the name is still returned and validated.
fn declare_modifier_param(
    b: &mut Builder,
    modifier: &crate::model::Modifier,
    n_bins: usize,
    declared: &HashSet<String>,
) -> Result<String> {
    // Kind validation + the `parameter` requirement are driven by the shared
    // MOD_SPECS table (histfactory.rs), so this path cannot disagree with
    // `modifier_effect` about which kinds are supported or need a parameter.
    let spec = require_spec(modifier)?;
    let param = require_param(modifier, spec)?;
    let param: &str = param.as_str();

    if !declared.contains(param) {
        let set = param_domain_set(b, spec.param_domain, n_bins);
        b.bind_set(param, set);
    }
    Ok(param.to_string())
}

/// Emit the FlatPPL set node a [`ParamDomain`] corresponds to.
fn param_domain_set(b: &mut Builder, domain: ParamDomain, n_bins: usize) -> NodeId {
    match domain {
        ParamDomain::Reals => b.call_head("reals"),
        ParamDomain::PosReals => b.call_head("posreals"),
        ParamDomain::PosRealsPow => {
            let posreals = b.call_head("posreals");
            let n_node = b.lit_int(n_bins as i64);
            b.call("cartpow", &[posreals, n_node])
        }
    }
}

/// Resolve the observed-data vector for `channel_name` from either schema.
///
/// New format: `doc.observations` list keyed by name.
/// Old format: `doc.data` map keyed by channel name.
fn find_obs(b: &mut Builder, doc: &PyhfDocument, channel_name: &str) -> Result<(NodeId, usize)> {
    if let Some(obs) = doc.observations.iter().find(|o| o.name == channel_name) {
        let elems: Vec<NodeId> = obs.data.iter().map(|x| b.lit_real(*x)).collect();
        return Ok((b.array(&elems), obs.data.len()));
    }
    if let Some(map) = &doc.data {
        if let Some(data) = map.get(channel_name) {
            let elems: Vec<NodeId> = data.iter().map(|x| b.lit_real(*x)).collect();
            return Ok((b.array(&elems), data.len()));
        }
    }
    Err(Error::NoObservation(channel_name.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Modifier;
    use crate::model::{
        PyhfChannel, PyhfMeasurement, PyhfMeasurementConfig, PyhfObservation, PyhfParam, PyhfSample,
    };
    use flatppl_syntax::{Syntax, print_with};

    fn make_uncorrelated_doc() -> PyhfDocument {
        PyhfDocument {
            channels: vec![PyhfChannel {
                name: "singlechannel".into(),
                samples: vec![
                    PyhfSample {
                        name: "signal".into(),
                        data: vec![12.0, 11.0],
                        modifiers: vec![Modifier {
                            kind: "normfactor".into(),
                            parameter: Some("mu".into()),
                            name: None,
                            parameters: vec![],
                            data: None,
                            constraint: None,
                            interpolation: None,
                        }],
                    },
                    PyhfSample {
                        name: "background".into(),
                        data: vec![50.0, 52.0],
                        modifiers: vec![Modifier {
                            kind: "shapesys".into(),
                            parameter: Some("uncorr_bkguncrt".into()),
                            name: None,
                            parameters: vec![],
                            data: Some(serde_json::json!([3.0, 7.0])),
                            constraint: None,
                            interpolation: None,
                        }],
                    },
                ],
            }],
            observations: vec![PyhfObservation {
                name: "singlechannel".into(),
                data: vec![51.0, 48.0],
            }],
            measurements: vec![PyhfMeasurement {
                name: "m".into(),
                config: PyhfMeasurementConfig {
                    poi: Some("mu".into()),
                    parameters: vec![],
                },
            }],
            data: None,
            toplvl: None,
        }
    }

    #[test]
    fn pyhf_module_contains_required_constructs() {
        let doc = make_uncorrelated_doc();
        let m = pyhf_to_module(&doc).unwrap();
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("broadcast(Poisson"), "got:\n{text}");
        assert!(text.contains("ContinuedPoisson"), "got:\n{text}");
        assert!(text.contains("joint_likelihood("), "got:\n{text}");
        assert!(text.contains("likelihoodof("), "got:\n{text}");
        assert!(!text.contains("fn("), "must be point-free, got:\n{text}");
        assert!(
            text.contains("cartpow(posreals, 2)"),
            "expected integer size in cartpow, got:\n{text}"
        );
    }

    #[test]
    fn hepphys_standard_module_binding_present() {
        let doc = make_uncorrelated_doc();
        let m = pyhf_to_module(&doc).unwrap();
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("standard_module"), "got:\n{text}");
        assert!(text.contains("particle-physics"), "got:\n{text}");
    }

    #[test]
    fn normsys_emits_normal_aux_and_interp() {
        // Single-channel doc with one normsys modifier
        let doc = PyhfDocument {
            channels: vec![PyhfChannel {
                name: "ch".into(),
                samples: vec![PyhfSample {
                    name: "sig".into(),
                    data: vec![10.0, 20.0],
                    modifiers: vec![Modifier {
                        kind: "normsys".into(),
                        parameter: Some("alpha1".into()),
                        name: None,
                        parameters: vec![],
                        data: Some(serde_json::json!({"hi": 1.1, "lo": 0.9})),
                        constraint: None,
                        interpolation: None,
                    }],
                }],
            }],
            observations: vec![PyhfObservation {
                name: "ch".into(),
                data: vec![12.0, 18.0],
            }],
            measurements: vec![],
            data: None,
            toplvl: None,
        };
        let m = pyhf_to_module(&doc).unwrap();
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("Normal"), "missing Normal aux, got:\n{text}");
        assert!(
            text.contains("interp_poly6"),
            "missing interp fn, got:\n{text}"
        );
        assert!(!text.contains("fn("), "must be point-free, got:\n{text}");
    }

    #[test]
    fn lumi_requires_config_param() {
        // lumi modifier with no measurement config → should return Err
        let doc = PyhfDocument {
            channels: vec![PyhfChannel {
                name: "ch".into(),
                samples: vec![PyhfSample {
                    name: "bg".into(),
                    data: vec![100.0],
                    modifiers: vec![Modifier {
                        kind: "lumi".into(),
                        parameter: Some("lumi".into()),
                        name: None,
                        parameters: vec![],
                        data: None,
                        constraint: None,
                        interpolation: None,
                    }],
                }],
            }],
            observations: vec![PyhfObservation {
                name: "ch".into(),
                data: vec![99.0],
            }],
            measurements: vec![],
            data: None,
            toplvl: None,
        };
        assert!(
            pyhf_to_module(&doc).is_err(),
            "should fail without lumi config"
        );
    }

    #[test]
    fn lumi_with_config_converts() {
        // lumi modifier with proper config → converts OK, emits Normal aux
        let doc = PyhfDocument {
            channels: vec![PyhfChannel {
                name: "ch".into(),
                samples: vec![PyhfSample {
                    name: "bg".into(),
                    data: vec![100.0],
                    modifiers: vec![Modifier {
                        kind: "lumi".into(),
                        parameter: Some("lumi".into()),
                        name: None,
                        parameters: vec![],
                        data: None,
                        constraint: None,
                        interpolation: None,
                    }],
                }],
            }],
            observations: vec![PyhfObservation {
                name: "ch".into(),
                data: vec![99.0],
            }],
            measurements: vec![PyhfMeasurement {
                name: "m".into(),
                config: PyhfMeasurementConfig {
                    poi: None,
                    parameters: vec![PyhfParam {
                        name: "lumi".into(),
                        auxdata: vec![1.0],
                        sigmas: vec![0.1],
                    }],
                },
            }],
            data: None,
            toplvl: None,
        };
        let m = pyhf_to_module(&doc).unwrap();
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("Normal"), "missing lumi Normal, got:\n{text}");
        assert!(!text.contains("fn("), "must be point-free, got:\n{text}");
    }
}
