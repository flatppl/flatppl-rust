//! histfactory_dist -> broadcast/arithmetic effects + per-modifier auxiliary
//! likelihood terms (12-profiles.md; pyhf/ROOT-verified). Point-free (no `fn`).
use crate::builder::Builder;
use crate::error::{Error, Result};
use crate::model::{Modifier, SampleData};
use flatppl_core::id::NodeId;

/// The sample's nominal data as a FlatPPL array node.
pub fn sample_nominal(b: &mut Builder, data: &SampleData) -> NodeId {
    let contents: &[f64] = match data {
        SampleData::Struct { contents, .. } => contents,
        SampleData::Flat(v) => v,
    };
    let elems: Vec<NodeId> = contents.iter().map(|x| b.lit_real(*x)).collect();
    b.array(&elems)
}

/// A JSON array (modifier.data) -> FlatPPL array node.
fn json_array(b: &mut Builder, v: &serde_json::Value) -> NodeId {
    let elems: Vec<NodeId> = v
        .as_array()
        .map(|a| {
            a.iter()
                .map(|x| b.lit_real(x.as_f64().unwrap_or(0.0)))
                .collect()
        })
        .unwrap_or_default();
    b.array(&elems)
}

/// tau = broadcast(pow, broadcast(divide, nom, sigma), 2)  [point-free].
fn tau(b: &mut Builder, nom: NodeId, sigma: NodeId) -> NodeId {
    let divide = b.call_head("divide");
    let div = b.call("broadcast", &[divide, nom, sigma]);
    let pow = b.call_head("pow");
    let two = b.lit_int(2);
    b.call("broadcast", &[pow, div, two])
}

/// shapesys auxiliary likelihood term (point-free).
///
/// `likelihoodof(broadcast(hepphys.ContinuedPoisson, bcmul(gamma, tau)), tau)`
pub fn shapesys_aux(b: &mut Builder, gamma: NodeId, nom: NodeId, sigma: NodeId) -> NodeId {
    let t = tau(b, nom, sigma);
    let prod = b.call("bcmul", &[gamma, t]);
    let cp = b.module_call("hepphys", "ContinuedPoisson");
    let aux_model = b.call("broadcast", &[cp, prod]);
    b.call("likelihoodof", &[aux_model, t])
}

/// Map `modifier.interpolation` field (or None) to the hepphys interp function name.
///
/// normsys default: `interp_poly6_exp`
/// histosys default: `interp_poly6_lin`
pub fn interp_fn(code: Option<&str>, default: &str) -> &'static str {
    match code.unwrap_or(default) {
        "lin" => "interp_pwlin",
        "log" => "interp_pwexp",
        "parabolic" => "interp_poly2_lin",
        "poly6" => "interp_poly6_lin",
        _ => {
            // Treat the default string as the fallback case identifier.
            // normsys passes "interp_poly6_exp", histosys passes "interp_poly6_lin".
            if default == "interp_poly6_exp" {
                "interp_poly6_exp"
            } else {
                "interp_poly6_lin"
            }
        }
    }
}

/// The deterministic effect a modifier applies to a sample, plus optional aux term.
///
/// Variants are consumed by `emit_channel` in `pyhf.rs`.
pub enum Effect {
    /// Multiply `expected` by a free scalar parameter: `broadcast(mul, expected, p)`.
    MulParam(NodeId),
    /// Multiply `expected` element-wise by gamma, with a ContinuedPoisson aux term.
    MulGammaWithAux { gamma: NodeId, aux: NodeId },
    /// Multiply `expected` by a normsys interpolation factor (scalar), with a
    /// Normal(alpha, 1.0) aux term.
    NormSys { factor: NodeId, aux: NodeId },
    /// Replace the sample nominal with an interpolated array (histosys). No direct
    /// multiplication applied here; the caller splices `new_nom` in.  Carries a
    /// Normal(alpha, 1.0) aux term.
    HistoSys { new_nom: NodeId, aux: NodeId },
    /// Multiply by a shared lumi parameter; the caller emits the aux once.
    MulLumi(NodeId),
    /// Multiply element-wise by a shared per-bin staterror gamma; aux emitted once
    /// per channel by the caller after aggregating nominals and errors.
    MulStaterrGamma(NodeId),
    /// Multiply by a free per-bin shapefactor gamma (no aux).
    MulShapefactorGamma(NodeId),
}

/// Map one `Modifier` to its `Effect`.
///
/// * `nom` is the sample's nominal data node (needed for histosys interpolation and
///   shapesys tau).
/// * `lumi_declared` signals that `declare_modifier_param` has already been called for
///   the lumi parameter — the caller passes `true` on the second sample.
///
/// Aux terms for **lumi** and **staterror** are NOT generated here; they are
/// assembled channel-wide by `emit_channel` in `pyhf.rs`.
pub fn modifier_effect(b: &mut Builder, m: &Modifier, nom: NodeId) -> Result<Effect> {
    let param = m.parameter.as_deref().unwrap_or("");
    match m.kind.as_str() {
        "normfactor" => Ok(Effect::MulParam(b.self_ref(param))),

        "shapesys" => {
            let gamma = b.self_ref(param);
            let sigma = m
                .data
                .as_ref()
                .map(|d| json_array(b, d))
                .unwrap_or_else(|| b.array(&[]));
            let aux = shapesys_aux(b, gamma, nom, sigma);
            Ok(Effect::MulGammaWithAux { gamma, aux })
        }

        "normsys" => {
            // data = {hi: <f64>, lo: <f64>}
            let (lo_val, hi_val) = parse_normsys_data(m)?;
            let lo = b.lit_real(lo_val);
            let one = b.lit_real(1.0);
            let hi = b.lit_real(hi_val);
            let alpha = b.self_ref(param);
            let fn_name = interp_fn(m.interpolation.as_deref(), "interp_poly6_exp");
            let factor = b.module_user_call("hepphys", fn_name, &[lo, one, hi, alpha]);
            // aux: likelihoodof(Normal(mu=alpha, sigma=1.0), 0.0)
            let aux = normsys_aux(b, alpha);
            Ok(Effect::NormSys { factor, aux })
        }

        "histosys" => {
            // data = {hi: {contents:[...]}, lo: {contents:[...]}}
            let (lo_arr, hi_arr) = parse_histosys_data(b, m)?;
            let alpha = b.self_ref(param);
            let fn_name = interp_fn(m.interpolation.as_deref(), "interp_poly6_lin");
            let new_nom = b.module_user_call("hepphys", fn_name, &[lo_arr, nom, hi_arr, alpha]);
            let aux = normsys_aux(b, alpha); // same Normal(alpha,1) form
            Ok(Effect::HistoSys { new_nom, aux })
        }

        "lumi" => {
            let lam = b.self_ref(param);
            Ok(Effect::MulLumi(lam))
        }

        "staterror" => {
            if m.constraint.as_deref() == Some("Poisson") {
                return Err(Error::Unsupported(
                    "staterror with Poisson constraint is not yet supported (only Gaussian BB-lite)"
                        .into(),
                ));
            }
            let gamma = b.self_ref(param);
            Ok(Effect::MulStaterrGamma(gamma))
        }

        "shapefactor" => {
            let gamma = b.self_ref(param);
            Ok(Effect::MulShapefactorGamma(gamma))
        }

        other => Err(Error::UnknownModifier(other.to_string())),
    }
}

/// Parse normsys modifier data `{hi: f64, lo: f64}`.
fn parse_normsys_data(m: &Modifier) -> Result<(f64, f64)> {
    let data = m.data.as_ref().ok_or_else(|| {
        Error::Unsupported(format!(
            "normsys modifier `{}` missing data",
            m.parameter.as_deref().unwrap_or("?")
        ))
    })?;
    let lo = data["lo"].as_f64().ok_or_else(|| {
        Error::Unsupported(format!(
            "normsys `{}`: lo is not a number",
            m.parameter.as_deref().unwrap_or("?")
        ))
    })?;
    let hi = data["hi"].as_f64().ok_or_else(|| {
        Error::Unsupported(format!(
            "normsys `{}`: hi is not a number",
            m.parameter.as_deref().unwrap_or("?")
        ))
    })?;
    Ok((lo, hi))
}

/// Parse histosys modifier data `{hi: {contents:[...]}, lo: {contents:[...]}}`.
fn parse_histosys_data(b: &mut Builder, m: &Modifier) -> Result<(NodeId, NodeId)> {
    let data = m.data.as_ref().ok_or_else(|| {
        Error::Unsupported(format!(
            "histosys modifier `{}` missing data",
            m.parameter.as_deref().unwrap_or("?")
        ))
    })?;
    let lo_arr = json_array(b, &data["lo"]["contents"]);
    let hi_arr = json_array(b, &data["hi"]["contents"]);
    Ok((lo_arr, hi_arr))
}

/// `likelihoodof(Normal(mu=alpha, sigma=1.0), 0.0)` — shared by normsys and histosys.
pub fn normsys_aux(b: &mut Builder, alpha: NodeId) -> NodeId {
    let sigma_one = b.lit_real(1.0);
    let normal = b.call_kw("Normal", &[("mu", alpha), ("sigma", sigma_one)]);
    let obs_zero = b.lit_real(0.0);
    b.call("likelihoodof", &[normal, obs_zero])
}

/// Staterror BB-lite aux: `likelihoodof(broadcast(Normal, gamma, delta), 1.0)`
///
/// `delta` is precomputed as a Rust Vec<f64>:
///   `delta_b = sqrt(sum_s err_{s,b}^2) / sum_s nom_{s,b}`
pub fn staterror_aux(b: &mut Builder, gamma: NodeId, delta_vals: &[f64]) -> NodeId {
    let delta_elems: Vec<NodeId> = delta_vals.iter().map(|x| b.lit_real(*x)).collect();
    let delta = b.array(&delta_elems);
    let normal_head = b.call_head("Normal");
    let broadcast_model = b.call("broadcast", &[normal_head, gamma, delta]);
    let obs_one = b.lit_real(1.0);
    b.call("likelihoodof", &[broadcast_model, obs_one])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::Builder;
    use flatppl_syntax::{Syntax, print_with};

    #[test]
    fn parse_pyhf_channel_sample() {
        let json = serde_json::json!({
          "name": "singlechannel",
          "samples": [
            {"name":"signal","data":[12.0,11.0],
             "modifiers":[{"name":"mu","type":"normfactor"}]},
            {"name":"background","data":[50.0,52.0],
             "modifiers":[{"name":"uncorr_bkguncrt","type":"shapesys","data":[3.0,7.0]}]}
          ]
        });
        let ch: crate::model::PyhfChannel = serde_json::from_value(json).unwrap();
        assert_eq!(ch.samples.len(), 2);
        assert_eq!(ch.samples[1].modifiers[0].kind, "shapesys");
        assert_eq!(
            ch.samples[1].modifiers[0].parameter.as_deref(),
            Some("uncorr_bkguncrt")
        );
    }

    #[test]
    fn shapesys_aux_is_point_free_continued_poisson() {
        let mut m = flatppl_core::Module::new();
        let aux = {
            let mut b = Builder::new(&mut m);
            let nom = {
                let a = b.lit_real(50.0);
                let c = b.lit_real(52.0);
                b.array(&[a, c])
            };
            let sigma = {
                let a = b.lit_real(3.0);
                let c = b.lit_real(7.0);
                b.array(&[a, c])
            };
            let gamma = b.self_ref("gamma");
            shapesys_aux(&mut b, gamma, nom, sigma)
        };
        {
            let mut b = Builder::new(&mut m);
            b.bind("L_aux", aux);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("ContinuedPoisson"), "got:\n{text}");
        assert!(text.contains("likelihoodof"), "got:\n{text}");
        assert!(!text.contains("fn("), "MUST be point-free, got:\n{text}");
    }

    #[test]
    fn normsys_aux_emits_normal() {
        let mut m = flatppl_core::Module::new();
        {
            let mut b = Builder::new(&mut m);
            let alpha = b.self_ref("alpha");
            let aux = normsys_aux(&mut b, alpha);
            b.bind("L_aux", aux);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("Normal"), "got:\n{text}");
        assert!(text.contains("likelihoodof"), "got:\n{text}");
        assert!(!text.contains("fn("), "must be point-free, got:\n{text}");
    }

    #[test]
    fn staterror_aux_emits_broadcast_normal() {
        let mut m = flatppl_core::Module::new();
        {
            let mut b = Builder::new(&mut m);
            let gamma = b.self_ref("gamma");
            let aux = staterror_aux(&mut b, gamma, &[0.05, 0.1]);
            b.bind("L_aux", aux);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("Normal"), "got:\n{text}");
        assert!(text.contains("broadcast"), "got:\n{text}");
        assert!(text.contains("0.05"), "got:\n{text}");
        assert!(text.contains("0.1"), "got:\n{text}");
        assert!(!text.contains("fn("), "must be point-free, got:\n{text}");
    }
}
