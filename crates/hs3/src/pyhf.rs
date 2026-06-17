//! pyhf workspace JSON → FlatPPL module assembly.
//!
//! Implements the lift described in §12 "pyhf uncorrelated_background" of the
//! profiles doc: each channel → `broadcast(Poisson, expected)` obs model, with
//! modifier effects generating auxiliary likelihood terms as needed.
use crate::builder::Builder;
use crate::error::{Error, Result};
use crate::histfactory::{Effect, modifier_effect, sample_nominal, staterror_aux};
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

    Ok(())
}

/// Emit `hepphys = standard_module("particle-physics", "0.1")`.
fn emit_standard_module(b: &mut Builder) {
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

    // Resolve observation for this channel into an array node.
    let observed = find_obs(b, doc, channel_name)?;

    // Resolve lumi config (sigma) if any sample carries a lumi modifier.
    let has_lumi = channel
        .samples
        .iter()
        .any(|s| s.modifiers.iter().any(|m| m.kind == "lumi"));
    let lumi = if has_lumi {
        let lumi_cfg = find_lumi_param(doc).ok_or_else(|| {
            Error::Unsupported(
                "lumi modifier present but no `lumi` parameter entry found in measurement config \
                 (need `sigmas` and `auxdata`)"
                    .into(),
            )
        })?;
        let sigma_lumi = *lumi_cfg
            .sigmas
            .first()
            .ok_or_else(|| Error::Unsupported("lumi config `sigmas` array is empty".into()))?;
        Some(sigma_lumi)
    } else {
        None
    };

    assemble_channel(b, channel_name, &samples, observed, lumi)
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
    lumi: Option<f64>,
) -> Result<()> {
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

    for (_name, nominal, modifiers) in samples {
        for modifier in *modifiers {
            if modifier.kind == "staterror" {
                let param_name = modifier.parameter.as_deref().unwrap_or("staterror");
                let n = nominal.len();
                let errors: Vec<f64> = modifier
                    .data
                    .as_ref()
                    .and_then(|d| d.as_array())
                    .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
                    .unwrap_or_else(|| vec![0.0; n]);

                let entry = staterror_acc
                    .entry(param_name.to_string())
                    .or_insert_with(|| (vec![0.0; n], vec![0.0; n]));
                for i in 0..n.min(entry.0.len()) {
                    entry.0[i] += nominal[i]; // sum of nominals
                    let e = errors.get(i).copied().unwrap_or(0.0);
                    entry.1[i] += e * e; // sum of squared errors
                }
            }
        }
    }

    // Compute delta arrays: delta_b = sqrt(sum_sq_err_b) / sum_nom_b
    let staterror_delta: std::collections::BTreeMap<String, Vec<f64>> = staterror_acc
        .iter()
        .map(|(name, (sum_nom, sum_sq))| {
            let delta: Vec<f64> = sum_nom
                .iter()
                .zip(sum_sq.iter())
                .map(|(nom, sq)| if *nom > 0.0 { sq.sqrt() / nom } else { 0.0 })
                .collect();
            (name.clone(), delta)
        })
        .collect();

    // ---- Pass 2: declare free params (idempotent by name) ----
    let mut declared: HashSet<String> = HashSet::new();

    for (_name, nominal, modifiers) in samples {
        let n_bins = nominal.len();
        for modifier in *modifiers {
            let param_name = match &modifier.parameter {
                Some(p) => p.clone(),
                None => continue,
            };
            if declared.contains(&param_name) {
                continue;
            }
            declare_modifier_param(b, modifier, n_bins)?;
            declared.insert(param_name);
        }
    }

    // ---- Pass 3: lumi aux once (if present) ----
    let mut aux_terms: Vec<NodeId> = Vec::new();

    if let Some(sigma_lumi) = lumi {
        let lam = b.self_ref("lumi");
        let sigma_node = b.lit_real(sigma_lumi);
        let nom_node = b.lit_real(1.0);
        let lumi_normal = b.call_kw("Normal", &[("mu", lam), ("sigma", sigma_node)]);
        let lumi_aux = b.call("likelihoodof", &[lumi_normal, nom_node]);
        aux_terms.push(lumi_aux);
    }

    // ---- Pass 4: emit staterror per-bin aux terms ----
    for (param_name, delta_vals) in &staterror_delta {
        let gamma = b.self_ref(param_name);
        let aux = staterror_aux(b, gamma, delta_vals);
        aux_terms.push(aux);
    }

    // ---- Pass 5: build per-sample expected vectors ----
    let mut sample_expected: Vec<NodeId> = Vec::new();

    for (_name, nominal, modifiers) in samples {
        let data = SampleData::Flat(nominal.to_vec());
        let mut nom = sample_nominal(b, &data);

        // First pass: apply histosys modifiers (they replace nominal).
        for modifier in *modifiers {
            if modifier.kind == "histosys" {
                let effect = modifier_effect(b, modifier, nom)?;
                match effect {
                    Effect::HistoSys { new_nom, aux } => {
                        nom = new_nom;
                        aux_terms.push(aux);
                    }
                    _ => unreachable!("histosys must produce HistoSys effect"),
                }
            }
        }

        // Second pass: apply all multiplicative modifiers.
        let mut acc = nom;
        for modifier in *modifiers {
            if modifier.kind == "histosys" {
                continue; // already handled above
            }
            let effect = modifier_effect(b, modifier, nom)?;
            match effect {
                Effect::MulParam(param) => {
                    let mul = b.call_head("mul");
                    acc = b.call("broadcast", &[mul, acc, param]);
                }
                Effect::MulGammaWithAux { gamma, aux } => {
                    let mul = b.call_head("mul");
                    acc = b.call("broadcast", &[mul, acc, gamma]);
                    aux_terms.push(aux);
                }
                Effect::NormSys { factor, aux } => {
                    let mul = b.call_head("mul");
                    acc = b.call("broadcast", &[mul, acc, factor]);
                    aux_terms.push(aux);
                }
                Effect::HistoSys { .. } => {
                    unreachable!("histosys handled in first pass")
                }
                Effect::MulLumi(lam) => {
                    let mul = b.call_head("mul");
                    acc = b.call("broadcast", &[mul, acc, lam]);
                }
                Effect::MulStaterrGamma(gamma) => {
                    let mul = b.call_head("mul");
                    acc = b.call("broadcast", &[mul, acc, gamma]);
                }
                Effect::MulShapefactorGamma(gamma) => {
                    let mul = b.call_head("mul");
                    acc = b.call("broadcast", &[mul, acc, gamma]);
                }
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

/// Declare the free-parameter binding for a modifier (deduplication is handled by
/// the caller tracking `declared` names — this function always writes the binding).
fn declare_modifier_param(
    b: &mut Builder,
    modifier: &crate::model::Modifier,
    n_bins: usize,
) -> Result<()> {
    let param = match &modifier.parameter {
        Some(p) => p.clone(),
        None => return Ok(()),
    };

    match modifier.kind.as_str() {
        "normfactor" => {
            let set = b.call_head("nonnegreals");
            b.bind_set(&param, set);
        }
        "shapesys" => {
            let posreals = b.call_head("posreals");
            let n_node = b.lit_int(n_bins as i64);
            let set = b.call("cartpow", &[posreals, n_node]);
            b.bind_set(&param, set);
        }
        "normsys" => {
            // alpha: real-valued nuisance
            let set = b.call_head("reals");
            b.bind_set(&param, set);
        }
        "histosys" => {
            // alpha: real-valued nuisance
            let set = b.call_head("reals");
            b.bind_set(&param, set);
        }
        "lumi" => {
            // lam: positive real
            let set = b.call_head("posreals");
            b.bind_set(&param, set);
        }
        "staterror" => {
            // gamma: cartpow(posreals, n_bins)
            let posreals = b.call_head("posreals");
            let n_node = b.lit_int(n_bins as i64);
            let set = b.call("cartpow", &[posreals, n_node]);
            b.bind_set(&param, set);
        }
        "shapefactor" => {
            // gamma: cartpow(posreals, n_bins)
            let posreals = b.call_head("posreals");
            let n_node = b.lit_int(n_bins as i64);
            let set = b.call("cartpow", &[posreals, n_node]);
            b.bind_set(&param, set);
        }
        _ => {
            // Unknown modifier — modifier_effect will error; skip declaration.
        }
    }
    Ok(())
}

/// Resolve the observed-data vector for `channel_name` from either schema.
///
/// New format: `doc.observations` list keyed by name.
/// Old format: `doc.data` map keyed by channel name.
fn find_obs(b: &mut Builder, doc: &PyhfDocument, channel_name: &str) -> Result<NodeId> {
    if let Some(obs) = doc.observations.iter().find(|o| o.name == channel_name) {
        let elems: Vec<NodeId> = obs.data.iter().map(|x| b.lit_real(*x)).collect();
        return Ok(b.array(&elems));
    }
    if let Some(map) = &doc.data {
        if let Some(data) = map.get(channel_name) {
            let elems: Vec<NodeId> = data.iter().map(|x| b.lit_real(*x)).collect();
            return Ok(b.array(&elems));
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
