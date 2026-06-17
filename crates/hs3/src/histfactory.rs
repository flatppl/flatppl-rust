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
///
/// `what` names the field for the error message. Returns `Err(Unsupported)` if
/// `v` is not a JSON array or if any element is not a number — silently coercing
/// a non-numeric entry to `0.0` would emit a wrong-but-plausible model.
fn json_array(b: &mut Builder, v: &serde_json::Value, what: &str) -> Result<NodeId> {
    let arr = v
        .as_array()
        .ok_or_else(|| Error::Unsupported(format!("{what}: expected a JSON array of numbers")))?;
    let mut elems: Vec<NodeId> = Vec::with_capacity(arr.len());
    for (i, x) in arr.iter().enumerate() {
        let val = x
            .as_f64()
            .ok_or_else(|| Error::Unsupported(format!("{what}: element {i} is not a number")))?;
        elems.push(b.lit_real(val));
    }
    Ok(b.array(&elems))
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
/// `likelihoodof(broadcast(hepphys.ContinuedPoisson, broadcast(mul, gamma, tau)), tau)`
pub fn shapesys_aux(b: &mut Builder, gamma: NodeId, nom: NodeId, sigma: NodeId) -> NodeId {
    let t = tau(b, nom, sigma);
    let mul = b.call_head("mul");
    let prod = b.call("broadcast", &[mul, gamma, t]);
    let cp = b.module_call("hepphys", "ContinuedPoisson");
    let aux_model = b.call("broadcast", &[cp, prod]);
    b.call("likelihoodof", &[aux_model, t])
}

// pyhf interpolation codes (the `interpolation` field on a modifier).
const INTERP_CODE_LIN: &str = "lin";
const INTERP_CODE_LOG: &str = "log";
const INTERP_CODE_PARABOLIC: &str = "parabolic";
const INTERP_CODE_POLY6: &str = "poly6";

// hepphys interpolation function names these codes map to.
const INTERP_PWLIN: &str = "interp_pwlin";
const INTERP_PWEXP: &str = "interp_pwexp";
const INTERP_POLY2_LIN: &str = "interp_poly2_lin";
const INTERP_POLY6_LIN: &str = "interp_poly6_lin";

/// normsys default interpolation function.
pub const INTERP_NORMSYS_DEFAULT: &str = "interp_poly6_exp";
/// histosys default interpolation function.
pub const INTERP_HISTOSYS_DEFAULT: &str = "interp_poly6_lin";

/// Map `modifier.interpolation` field (or None) to the hepphys interp function name.
///
/// `default` is the interp function used when the field is absent or unrecognised —
/// `INTERP_NORMSYS_DEFAULT` for normsys, `INTERP_HISTOSYS_DEFAULT` for histosys.
pub fn interp_fn(code: Option<&str>, default: &'static str) -> &'static str {
    match code {
        Some(INTERP_CODE_LIN) => INTERP_PWLIN,
        Some(INTERP_CODE_LOG) => INTERP_PWEXP,
        Some(INTERP_CODE_PARABOLIC) => INTERP_POLY2_LIN,
        Some(INTERP_CODE_POLY6) => INTERP_POLY6_LIN,
        // Absent or unrecognised: fall back to the caller's default.
        _ => default,
    }
}

/// The set a modifier's free parameter ranges over.
///
/// `PosRealsPow` is `cartpow(posreals, n_bins)` — one positive real per bin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamDomain {
    /// `reals` — normfactor scale + normsys/histosys alpha nuisance (spec §12:206).
    Reals,
    /// `posreals` — scalar lumi.
    PosReals,
    /// `cartpow(posreals, n_bins)` — per-bin gammas (shapesys/staterror/shapefactor).
    PosRealsPow,
}

/// Static description of one histfactory modifier kind. The single source of
/// truth for kind-dependent knowledge that `modifier_effect` (histfactory.rs) and
/// `declare_modifier_param` (pyhf.rs) both consume — adding a modifier is a
/// one-row edit to [`MOD_SPECS`].
#[derive(Debug, Clone, Copy)]
pub struct ModSpec {
    /// The pyhf modifier `type` string.
    pub kind: &'static str,
    /// The set the modifier's free parameter ranges over.
    pub param_domain: ParamDomain,
    /// Whether a `parameter` field is mandatory (all current kinds require one).
    pub requires_param: bool,
    /// histosys: applied in a first pass that replaces the sample nominal rather
    /// than multiplying into the running product.
    pub replaces_nominal: bool,
    /// lumi: needs the channel's `lumi` measurement-config entry and emits one
    /// channel-wide Normal aux term.
    pub channel_lumi: bool,
    /// staterror: nominals and squared errors are aggregated across samples into
    /// one Gaussian (BB-lite) aux term per bin, emitted channel-wide.
    pub channel_staterror: bool,
}

/// The modifier-kind table. One row per supported histfactory modifier.
pub const MOD_SPECS: &[ModSpec] = &[
    // normfactor: `factor = elementof(reals)` per spec §12:206 (not constrained nonneg).
    ModSpec {
        kind: "normfactor",
        param_domain: ParamDomain::Reals,
        requires_param: true,
        replaces_nominal: false,
        channel_lumi: false,
        channel_staterror: false,
    },
    ModSpec {
        kind: "shapesys",
        param_domain: ParamDomain::PosRealsPow,
        requires_param: true,
        replaces_nominal: false,
        channel_lumi: false,
        channel_staterror: false,
    },
    ModSpec {
        kind: "normsys",
        param_domain: ParamDomain::Reals,
        requires_param: true,
        replaces_nominal: false,
        channel_lumi: false,
        channel_staterror: false,
    },
    ModSpec {
        kind: "histosys",
        param_domain: ParamDomain::Reals,
        requires_param: true,
        replaces_nominal: true,
        channel_lumi: false,
        channel_staterror: false,
    },
    ModSpec {
        kind: "lumi",
        param_domain: ParamDomain::PosReals,
        requires_param: true,
        replaces_nominal: false,
        channel_lumi: true,
        channel_staterror: false,
    },
    ModSpec {
        kind: "staterror",
        param_domain: ParamDomain::PosRealsPow,
        requires_param: true,
        replaces_nominal: false,
        channel_lumi: false,
        channel_staterror: true,
    },
    ModSpec {
        kind: "shapefactor",
        param_domain: ParamDomain::PosRealsPow,
        requires_param: true,
        replaces_nominal: false,
        channel_lumi: false,
        channel_staterror: false,
    },
];

/// Look up the [`ModSpec`] for a modifier `kind`, or `None` if unsupported.
pub fn mod_spec(kind: &str) -> Option<&'static ModSpec> {
    MOD_SPECS.iter().find(|s| s.kind == kind)
}

/// The [`ModSpec`] for a modifier, or `Err(UnknownModifier)` if the kind is unsupported.
pub fn require_spec(m: &Modifier) -> Result<&'static ModSpec> {
    mod_spec(&m.kind).ok_or_else(|| Error::UnknownModifier(m.kind.clone()))
}

/// The parameter name a modifier binds, validated against its [`ModSpec`].
///
/// Returns `Err(UnknownModifier)` for an unsupported kind, or `Err(Unsupported)`
/// if the kind requires a `parameter` but none is present — a missing `parameter`
/// would otherwise emit a `self.""` ref or a broadcast with a missing operand.
/// Shared by `modifier_effect` (histfactory.rs) and `declare_modifier_param`
/// (pyhf.rs) so the two paths cannot disagree.
pub fn require_param(m: &Modifier, spec: &ModSpec) -> Result<String> {
    match m.effective_param() {
        Some(p) => Ok(p),
        None if spec.requires_param => Err(Error::Unsupported(format!(
            "{} modifier is missing its `parameter` (name of the free parameter it binds)",
            m.kind
        ))),
        // A kind that does not require a parameter but has none: caller decides.
        None => Ok(String::new()),
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
///
/// `nom_len` is the number of bins in the sample nominal — used to validate that
/// histosys `lo`/`hi` content arrays have a matching length.
pub fn modifier_effect(
    b: &mut Builder,
    m: &Modifier,
    nom: NodeId,
    nom_len: usize,
) -> Result<Effect> {
    let spec = require_spec(m)?;
    let param = require_param(m, spec)?;
    let param: &str = param.as_str();
    // The Effect *shape* still varies per kind, but kind validation and the
    // `parameter` requirement are now driven by `spec`/[`MOD_SPECS`].
    match spec.kind {
        "normfactor" => Ok(Effect::MulParam(b.self_ref(param))),

        "shapesys" => {
            let gamma = b.self_ref(param);
            // sigma is the per-bin uncertainty array; it is required for the tau term.
            let data = m.data.as_ref().ok_or_else(|| {
                Error::Unsupported(format!("shapesys `{param}` missing data (per-bin errors)"))
            })?;
            let sigma = json_array(b, data, &format!("shapesys `{param}` data"))?;
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
            let fn_name = interp_fn(m.interpolation.as_deref(), INTERP_NORMSYS_DEFAULT);
            let factor = b.module_user_call("hepphys", fn_name, &[lo, one, hi, alpha]);
            // aux: likelihoodof(Normal(mu=alpha, sigma=1.0), 0.0)
            let aux = normsys_aux(b, alpha);
            Ok(Effect::NormSys { factor, aux })
        }

        "histosys" => {
            // data = {hi: {contents:[...]}, lo: {contents:[...]}}
            let (lo_arr, hi_arr) = parse_histosys_data(b, m, nom_len)?;
            let alpha = b.self_ref(param);
            let fn_name = interp_fn(m.interpolation.as_deref(), INTERP_HISTOSYS_DEFAULT);
            let new_nom = b.module_user_call("hepphys", fn_name, &[lo_arr, nom, hi_arr, alpha]);
            let aux = normsys_aux(b, alpha); // same Normal(alpha,1) form
            Ok(Effect::HistoSys { new_nom, aux })
        }

        "lumi" => Ok(Effect::MulLumi(b.self_ref(param))),

        // The aux constraint (built in pyhf.rs from the channel-summed
        // uncertainties) is a per-bin Poisson term, matching ROOT HistFactory
        // (which always Poisson-constrains staterror γ, regardless of the HS3
        // `constraint` field — pyhf's Gaussian staterror is an approximation we
        // intentionally do not follow).
        "staterror" => Ok(Effect::MulStaterrGamma(b.self_ref(param))),

        "shapefactor" => Ok(Effect::MulShapefactorGamma(b.self_ref(param))),

        // Unreachable: `require_spec` already rejected unknown kinds, and every
        // row in MOD_SPECS is handled above.
        other => unreachable!("MOD_SPECS row `{other}` has no Effect mapping"),
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
///
/// Validates that both `lo.contents` and `hi.contents` exist, are numeric arrays,
/// and have `nom_len` bins — a ragged or missing array would otherwise feed a
/// length-mismatched (or empty) array into the interpolation function.
fn parse_histosys_data(b: &mut Builder, m: &Modifier, nom_len: usize) -> Result<(NodeId, NodeId)> {
    let param = m.parameter.as_deref().unwrap_or("?");
    let data = m
        .data
        .as_ref()
        .ok_or_else(|| Error::Unsupported(format!("histosys modifier `{param}` missing data")))?;
    let lo_arr = histosys_contents(b, data, "lo", param, nom_len)?;
    let hi_arr = histosys_contents(b, data, "hi", param, nom_len)?;
    Ok((lo_arr, hi_arr))
}

/// Parse and validate one of the `lo`/`hi` `contents` arrays of a histosys modifier.
fn histosys_contents(
    b: &mut Builder,
    data: &serde_json::Value,
    side: &str,
    param: &str,
    nom_len: usize,
) -> Result<NodeId> {
    // Two source shapes: pyhf workspaces use a flat `{side}_data` array
    // (`hi_data` / `lo_data`); native HS³ uses a binned `{side}: {contents}`.
    let pyhf_key = format!("{side}_data");
    let contents = if !data[&pyhf_key].is_null() {
        &data[&pyhf_key]
    } else {
        &data[side]["contents"]
    };
    if contents.is_null() {
        return Err(Error::Unsupported(format!(
            "histosys `{param}`: neither `{side}_data` (pyhf) nor `{side}.contents` (HS3) is present"
        )));
    }
    let what = format!("histosys `{param}` {side}.contents");
    let arr = contents
        .as_array()
        .ok_or_else(|| Error::Unsupported(format!("{what}: expected a JSON array of numbers")))?;
    if arr.len() != nom_len {
        return Err(Error::Unsupported(format!(
            "{what}: has {} bins but the sample nominal has {nom_len}",
            arr.len()
        )));
    }
    json_array(b, contents, &what)
}

/// `likelihoodof(Normal(mu=alpha, sigma=1.0), 0.0)` — shared by normsys and histosys.
pub fn normsys_aux(b: &mut Builder, alpha: NodeId) -> NodeId {
    let sigma_one = b.lit_real(1.0);
    let normal = b.call_kw("Normal", &[("mu", alpha), ("sigma", sigma_one)]);
    let obs_zero = b.lit_real(0.0);
    b.call("likelihoodof", &[normal, obs_zero])
}

/// Staterror Barlow–Beeston aux. The per-bin scale `gamma_b` is constrained by
/// the channel-summed relative uncertainty
/// `delta_b = sqrt(sum_s err_{s,b}^2) / sum_s nom_{s,b}`.
///
/// The constraint TYPE follows ROOT HistFactory's `StatErrorConfig`: **Poisson
/// by default** (and for `constraint: "Poisson"`), **Gaussian** for
/// `constraint: "Gauss"`/`"Gaussian"`. (ROOT honours this; its HS3 *reader*
/// currently forces Poisson, but native HistFactory and the spec allow both.)
///
/// - Poisson: `likelihoodof(broadcast(ContinuedPoisson, broadcast(mul, gamma, tau)), tau)`,
///   effective count `tau_b = 1/delta_b^2` (= ROOT's `gamma·tau` poisMean, observed at `tau`).
/// - Gaussian: `likelihoodof(broadcast(Normal, gamma, delta), [1.0, …])`.
pub fn staterror_aux(
    b: &mut Builder,
    gamma: NodeId,
    sum_nom: &[f64],
    sum_sq: &[f64],
    gaussian: bool,
) -> NodeId {
    if gaussian {
        // delta_b = sqrt(sum_sq_b) / sum_nom_b (relative uncertainty).
        let delta_elems: Vec<NodeId> = sum_nom
            .iter()
            .zip(sum_sq.iter())
            .map(|(n, sq)| b.lit_real(if *n > 0.0 { sq.sqrt() / n } else { 0.0 }))
            .collect();
        let delta = b.array(&delta_elems);
        let normal = b.call_head("Normal");
        let model = b.call("broadcast", &[normal, gamma, delta]);
        let ones: Vec<NodeId> = sum_nom.iter().map(|_| b.lit_real(1.0)).collect();
        let obs = b.array(&ones);
        return b.call("likelihoodof", &[model, obs]);
    }
    // tau_b = sum_nom_b^2 / sum_sq_b = 1/delta_b^2 (effective counts), computed
    // directly from the sums to match ROOT exactly (e.g. 100^2/25 = 400, not
    // 399.999… via a sqrt-then-square round trip).
    let tau_elems: Vec<NodeId> = sum_nom
        .iter()
        .zip(sum_sq.iter())
        .map(|(n, sq)| b.lit_real(if *sq > 0.0 { n * n / sq } else { 0.0 }))
        .collect();
    let tau = b.array(&tau_elems);
    let mul = b.call_head("mul");
    let prod = b.call("broadcast", &[mul, gamma, tau]);
    let cp = b.module_call("hepphys", "ContinuedPoisson");
    let aux_model = b.call("broadcast", &[cp, prod]);
    b.call("likelihoodof", &[aux_model, tau])
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
        // pyhf form names the parameter via `name`; the effective param resolves it.
        assert_eq!(
            ch.samples[1].modifiers[0].effective_param().as_deref(),
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
    fn staterror_aux_poisson_default() {
        // ROOT default: Poisson constraint via ContinuedPoisson, effective count
        // tau = sum_nom^2/sum_sq (exact): 100^2/25 = 400, 100^2/100 = 100.
        let mut m = flatppl_core::Module::new();
        {
            let mut b = Builder::new(&mut m);
            let gamma = b.self_ref("gamma");
            let aux = staterror_aux(&mut b, gamma, &[100.0, 100.0], &[25.0, 100.0], false);
            b.bind("L_aux", aux);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("ContinuedPoisson"), "got:\n{text}");
        assert!(
            !text.contains("Normal"),
            "Poisson default must not emit Normal, got:\n{text}"
        );
        assert!(
            text.contains("[400.0, 100.0]"),
            "exact tau = [400, 100], got:\n{text}"
        );
        assert!(!text.contains("fn("), "must be point-free, got:\n{text}");
    }

    #[test]
    fn staterror_aux_gaussian_option() {
        // `constraint: "Gauss"` → Normal(gamma, delta), delta = sqrt(sum_sq)/sum_nom
        // = [0.05, 0.1], observed at per-bin 1.0.
        let mut m = flatppl_core::Module::new();
        {
            let mut b = Builder::new(&mut m);
            let gamma = b.self_ref("gamma");
            let aux = staterror_aux(&mut b, gamma, &[100.0, 100.0], &[25.0, 100.0], true);
            b.bind("L_aux", aux);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("Normal"), "got:\n{text}");
        assert!(
            text.contains("[0.05, 0.1]"),
            "delta = [0.05, 0.1], got:\n{text}"
        );
        assert!(!text.contains("fn("), "must be point-free, got:\n{text}");
    }
}
