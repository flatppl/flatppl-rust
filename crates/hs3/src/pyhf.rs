//! pyhf workspace JSON → FlatPPL module assembly.
//!
//! Implements the lift described in §12 "pyhf uncorrelated_background" of the
//! profiles doc: each channel → `broadcast(Poisson, expected)` obs model, with
//! modifier effects generating auxiliary likelihood terms as needed.
use crate::builder::Builder;
use crate::error::{Error, Result};
use crate::histfactory::{
    Effect, ParamDomain, PendingConstraint, emit_lumi_constraint, emit_normal01_constraint,
    emit_shapesys_constraint, emit_staterror_constraint, mod_spec, modifier_effect, require_param,
    require_spec, sample_nominal,
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
        // `flatppl_compat` leads the generated module (spec §11).
        b.stamp_compat();
        emit_pyhf(&mut b, doc)?;
    }
    Ok(m)
}

fn emit_pyhf(b: &mut Builder, doc: &PyhfDocument) -> Result<()> {
    // `hepphys = standard_module("particle-physics", "0.1")`
    emit_standard_module(b);

    // Per channel: declare free params, build expected, obs model, and accumulate
    // observation + constraint likelihood terms.
    let mut terms = Terms::default();
    for channel in &doc.channels {
        emit_channel(b, doc, channel, &mut terms)?;
    }

    // Measurement parameter-of-interest → record binding (so `config.poi`
    // survives the lift; FlatPPL has no dedicated POI construct).
    emit_poi(b, doc);

    // The flat top-level `likelihood` = joint of every channel's observation term
    // and all constraint terms.
    bind_likelihood(b, &terms);

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

/// Emit `alias = standard_module("<module>", "<version>")`.
pub(crate) fn bind_standard_module(b: &mut Builder, alias: &str, module: &str, version: &str) {
    let name_arg = b.str_lit(module);
    let ver_arg = b.str_lit(version);
    let head = b.sym("standard_module");
    let node = b.m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: vec![name_arg, ver_arg].into(),
        named: Vec::new().into(),
        inputs: None,
    }));
    b.bind(alias, node);
}

/// Emit `hepphys = standard_module("particle-physics", "0.1")`.
pub(crate) fn emit_standard_module(b: &mut Builder) {
    bind_standard_module(b, "hepphys", "particle-physics", "0.1");
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
    terms: &mut Terms,
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

    assemble_channel(b, channel_name, &samples, observed, n_observed, lumi, terms)
}

/// Accumulated likelihood terms across a workspace's channels: the per-channel
/// observation terms and the constraint (auxiliary-measurement) terms. The caller
/// binds the flat top-level `likelihood` from these via [`bind_likelihood`].
#[derive(Default)]
pub struct Terms {
    /// One observation likelihood term per channel.
    observation: Vec<NodeId>,
    /// Constraint (auxiliary-measurement) likelihood terms.
    constraints: Vec<NodeId>,
    /// Parameters whose constraint has already been emitted — a constraint
    /// belongs to its (possibly shared) parameter, so it is emitted exactly once
    /// (pyhf: one auxiliary measurement per constrained parameter).
    emitted_params: HashSet<String>,
}

/// Bind the flat top-level `likelihood` = `joint_likelihood(observation terms…,
/// constraint terms…)`. A single term is aliased directly; nothing is emitted
/// when there are no terms.
pub fn bind_likelihood(b: &mut Builder, terms: &Terms) {
    let mut all = terms.observation.clone();
    all.extend(terms.constraints.iter().copied());
    let node = match all.len() {
        0 => return,
        1 => all[0],
        _ => b.call("joint_likelihood", &all),
    };
    b.bind_unique_doc(
        "likelihood",
        node,
        "Full likelihood: observation and constraint terms.",
    );
}

/// Assemble one channel into named, channel-scoped FlatPPL bindings, pushing its
/// observation term and constraint terms into `terms`. Shared by the pyhf and
/// native-HS3 `histfactory_dist` paths.
///
/// `samples` is `(name, nominal_bins, modifiers)`. Staterror errors are read from
/// each staterror modifier's `data` array; the native path injects the sample's
/// `errors` there before calling.
///
/// Emits `<channel>_observed`, `<channel>_<sample>_{nominal,expected}`,
/// `<channel>_expected`, `<channel>_model` (`functionof(broadcast(Poisson, …))` —
/// a kernel, as `likelihoodof` requires, spec §06), `<channel>_likelihood` (the
/// observation term), plus the auxiliary constraint terms (lumi, staterror,
/// per-sample shapesys/normsys/histosys).
pub fn assemble_channel(
    b: &mut Builder,
    channel_name: &str,
    samples: &[(&str, &[f64], &[crate::model::Modifier])],
    observed: NodeId,
    n_observed: usize,
    lumi: Option<LumiConfig>,
    terms: &mut Terms,
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

    // ---- Observed counts for this channel ----
    let observed_name = b.bind_unique_doc(
        &format!("{channel_name}_observed"),
        observed,
        &format!("Observed event counts for channel \"{channel_name}\"."),
    );

    // ---- Per-sample expected yields: nominal template, then modifiers ----
    // Per-sample constraints are collected here and emitted *after* the
    // observation term, so the file reads observation-first then constraints
    // (each carries the sample nominal its `tau` needs).
    let mut sample_expected: Vec<NodeId> = Vec::new();
    let mut pending: Vec<(String, PendingConstraint, NodeId)> = Vec::new();
    for (sname, nominal, modifiers) in samples {
        let data = SampleData::Flat(nominal.to_vec());
        let nom_arr = sample_nominal(b, &data);
        let nom_name = b.bind_unique_doc(
            &format!("{channel_name}_{sname}_nominal"),
            nom_arr,
            &format!("Nominal yields for sample \"{sname}\"."),
        );
        let mut nom = b.self_ref(&nom_name);

        // First: nominal-replacing modifiers (histosys).
        for modifier in *modifiers {
            if mod_spec(&modifier.kind).is_some_and(|spec| spec.replaces_nominal) {
                let (effect, constraint) = modifier_effect(b, modifier, nom, n_bins)?;
                if let Effect::ReplaceNominal(new_nom) = effect {
                    nom = new_nom;
                }
                if let Some((param, pc)) = constraint {
                    pending.push((param, pc, nom));
                }
            }
        }

        // Then: multiplicative modifiers (`acc = broadcast(mul, acc, factor)`).
        let mut acc = nom;
        for modifier in *modifiers {
            if mod_spec(&modifier.kind).is_some_and(|spec| spec.replaces_nominal) {
                continue;
            }
            let (effect, constraint) = modifier_effect(b, modifier, nom, n_bins)?;
            if let Effect::Multiply(factor) = effect {
                let mul = b.call_head("mul");
                acc = b.call("broadcast", &[mul, acc, factor]);
            }
            // shapesys uses the sample's nominal for its tau; other constraints ignore it.
            if let Some((param, pc)) = constraint {
                pending.push((param, pc, nom));
            }
        }

        let exp_name = b.bind_unique_doc(
            &format!("{channel_name}_{sname}_expected"),
            acc,
            &format!("Expected yields for sample \"{sname}\" (nominal x modifiers)."),
        );
        sample_expected.push(b.self_ref(&exp_name));
    }

    // ---- Total expected per bin (sum over samples) ----
    let total = if sample_expected.is_empty() {
        b.array(&[])
    } else {
        let mut acc = sample_expected[0];
        for &next in &sample_expected[1..] {
            let add = b.call_head("add");
            acc = b.call("broadcast", &[add, acc, next]);
        }
        acc
    };
    let expected_name = b.bind_unique_doc(
        &format!("{channel_name}_expected"),
        total,
        "Total expected counts per bin (sum over samples).",
    );

    // ---- Observation model + likelihood term ----
    // `functionof` reifies the parameter-dependent measure into a kernel, as
    // `likelihoodof` requires (spec §06).
    let poisson = b.call_head("Poisson");
    let expected_ref = b.self_ref(&expected_name);
    let obs_measure = b.call("broadcast", &[poisson, expected_ref]);
    let obs_kernel = b.functionof(obs_measure);
    let model_name = b.bind_unique_doc(
        &format!("{channel_name}_model"),
        obs_kernel,
        &format!("Per-bin Poisson observation model for channel \"{channel_name}\"."),
    );
    let model_ref = b.self_ref(&model_name);
    let observed_ref = b.self_ref(&observed_name);
    let obs_term = b.call("likelihoodof", &[model_ref, observed_ref]);
    let obs_term_name = b.bind_unique_doc(
        &format!("{channel_name}_likelihood"),
        obs_term,
        &format!("Observation likelihood term for channel \"{channel_name}\"."),
    );
    terms.observation.push(b.self_ref(&obs_term_name));

    // ---- Per-sample constraint terms (parameter-keyed, once per parameter) ----
    for (param, pending_c, nominal) in pending {
        emit_pending_constraint(b, terms, param, pending_c, nominal);
    }

    // ---- lumi constraint (global; one Normal aux, deduped across channels) ----
    if let Some(LumiConfig { sigma, nom }) = lumi
        && terms.emitted_params.insert("lumi".to_string())
    {
        let term = emit_lumi_constraint(b, "lumi", sigma, nom);
        terms.constraints.push(term);
    }

    // ---- staterror constraints (channel-summed Barlow-Beeston, one per param) ----
    for (param_name, (sum_nom, sum_sq)) in &staterror_acc {
        if terms.emitted_params.insert(param_name.clone()) {
            // ROOT default is Poisson; `constraint: "Gauss"`/`"Gaussian"` selects Normal.
            let gaussian = matches!(
                staterror_constraint
                    .get(param_name)
                    .and_then(|c| c.as_deref()),
                Some("Gauss") | Some("Gaussian")
            );
            let term = emit_staterror_constraint(b, param_name, sum_nom, sum_sq, gaussian);
            terms.constraints.push(term);
        }
    }

    Ok(())
}

/// Emit a per-parameter constraint once (deduped via `terms.emitted_params`) and
/// push its likelihood term. `nominal` is the sample's nominal-yield ref, used by
/// the shapesys constraint's `tau`; other constraints ignore it.
fn emit_pending_constraint(
    b: &mut Builder,
    terms: &mut Terms,
    param: String,
    pending: PendingConstraint,
    nominal: NodeId,
) {
    if !terms.emitted_params.insert(param.clone()) {
        return; // already emitted (shared parameter)
    }
    let term = match pending {
        PendingConstraint::Shapesys { sigma } => {
            emit_shapesys_constraint(b, &param, nominal, &sigma)
        }
        PendingConstraint::Normal01 => emit_normal01_constraint(b, &param),
    };
    terms.constraints.push(term);
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
