//! Fundamental HS3 distribution `type` -> FlatPPL distribution call.
use crate::builder::Builder;
use crate::error::{Error, Result};
use crate::expr;
use crate::model::Distribution;
use flatppl_core::id::NodeId;
use flatppl_core::node::{Call, CallHead, Inputs, NamedArg, NamedKind, Node, Ref, RefNs};

/// Build a FlatPPL array node from a JSON array of scalars (numbers/strings).
fn array_of_values(b: &mut Builder, arr: &[serde_json::Value]) -> NodeId {
    let elems: Vec<NodeId> = arr.iter().map(|v| field_node(b, v)).collect();
    b.array(&elems)
}

/// Build a FlatPPL 2-D array (vector of vectors) from a JSON 2-D array.
fn array2d_of_values(b: &mut Builder, arr: &[serde_json::Value]) -> NodeId {
    let rows: Vec<NodeId> = arr
        .iter()
        .map(|row| {
            if let Some(inner) = row.as_array() {
                array_of_values(b, inner)
            } else {
                field_node(b, row)
            }
        })
        .collect();
    b.array(&rows)
}

fn field_node(b: &mut Builder, v: &serde_json::Value) -> NodeId {
    match v {
        serde_json::Value::Number(n) => b.lit_real(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => b.self_ref(s),
        _ => b.lit_real(0.0),
    }
}

/// Bare distribution call (no relabel; caller wraps with the variate).
pub fn emit_distribution(b: &mut Builder, d: &Distribution) -> Result<NodeId> {
    match d.kind.as_str() {
        "gaussian_dist" | "normal_dist" => {
            let mut kws: Vec<(&str, NodeId)> = Vec::new();
            if let Some(v) = d.extra.get("mean") {
                kws.push(("mu", field_node(b, v)));
            }
            if let Some(v) = d.extra.get("sigma") {
                kws.push(("sigma", field_node(b, v)));
            }
            Ok(b.call_kw("Normal", &kws))
        }
        "poisson_dist" => {
            let mut kws: Vec<(&str, NodeId)> = Vec::new();
            if let Some(v) = d.extra.get("mean") {
                kws.push(("rate", field_node(b, v)));
            }
            Ok(b.call_kw("Poisson", &kws))
        }
        "exponential_dist" => {
            let mut kws: Vec<(&str, NodeId)> = Vec::new();
            if let Some(v) = d.extra.get("c") {
                kws.push(("rate", field_node(b, v)));
            }
            Ok(b.call_kw("Exponential", &kws))
        }
        "lognormal_dist" => {
            let mut kws: Vec<(&str, NodeId)> = Vec::new();
            if let Some(v) = d.extra.get("mu") {
                kws.push(("mu", field_node(b, v)));
            }
            if let Some(v) = d.extra.get("sigma") {
                kws.push(("sigma", field_node(b, v)));
            }
            Ok(b.call_kw("LogNormal", &kws))
        }
        "uniform_dist" => Ok(b.call_kw("Uniform", &[])),
        // §12: product_dist maps to joint over its factor sub-distributions.
        // joint uses NamedKind::Field (like `record`/`cartprod`), so named entries
        // must be built with Field kind.  Factor names are the field labels.
        // NOTE: both factors here share the same variate ("x") — this models a
        // density-product, not a joint independent distribution over distinct
        // variates.  The spec §12 mapping is `product_dist → joint`; whether
        // same-variate joint is semantically correct is a spec-fidelity question.
        // ponytail: §12 maps product_dist→joint; same-variate product may need review
        "product_dist" => {
            let factors: Vec<String> = d
                .extra
                .get("factors")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            if factors.is_empty() {
                return Err(Error::Unsupported(
                    "product_dist with no factors".into(),
                ));
            }
            let named: Vec<NamedArg> = factors
                .iter()
                .map(|f| {
                    let name = b.m.intern(f);
                    let value = b.self_ref(f);
                    NamedArg { kind: NamedKind::Field, name, value }
                })
                .collect();
            let head = b.m.intern("joint");
            Ok(b.m.alloc(Node::Call(Call {
                head: CallHead::Builtin(head),
                args: Vec::new().into(),
                named: named.into(),
                inputs: None,
            })))
        }
        "generalized_normal_dist" => {
            let mut kws: Vec<(&str, NodeId)> = Vec::new();
            if let Some(v) = d.extra.get("mean") {
                kws.push(("mean", field_node(b, v)));
            }
            if let Some(v) = d.extra.get("alpha") {
                kws.push(("alpha", field_node(b, v)));
            }
            if let Some(v) = d.extra.get("beta") {
                kws.push(("beta", field_node(b, v)));
            }
            Ok(b.call_kw("GeneralizedNormal", &kws))
        }
        "multivariate_normal_dist" => {
            let mut kws: Vec<(&str, NodeId)> = Vec::new();
            if let Some(arr) = d.extra.get("mean").and_then(|v| v.as_array()) {
                kws.push(("mu", array_of_values(b, arr)));
            }
            if let Some(arr) = d.extra.get("covariances").and_then(|v| v.as_array()) {
                kws.push(("cov", array2d_of_values(b, arr)));
            }
            Ok(b.call_kw("MvNormal", &kws))
        }
        "crystalball_dist" => {
            // Double-sided when any _L/_R variant fields are present.
            let double_sided = d.extra.contains_key("sigma_L")
                || d.extra.contains_key("sigma_R")
                || d.extra.contains_key("alpha_L")
                || d.extra.contains_key("n_L");
            if double_sided {
                let m0 = d.extra.get("m0").map(|v| field_node(b, v)).unwrap_or_else(|| b.lit_real(0.0));
                let sigma_l = d.extra.get("sigma_L").map(|v| field_node(b, v)).unwrap_or_else(|| b.lit_real(1.0));
                let sigma_r = d.extra.get("sigma_R").map(|v| field_node(b, v)).unwrap_or_else(|| b.lit_real(1.0));
                let alpha_l = d.extra.get("alpha_L").map(|v| field_node(b, v)).unwrap_or_else(|| b.lit_real(1.0));
                let n_l = d.extra.get("n_L").map(|v| field_node(b, v)).unwrap_or_else(|| b.lit_real(1.0));
                let alpha_r = d.extra.get("alpha_R").map(|v| field_node(b, v)).unwrap_or_else(|| b.lit_real(1.0));
                let n_r = d.extra.get("n_R").map(|v| field_node(b, v)).unwrap_or_else(|| b.lit_real(1.0));
                Ok(b.module_user_call("hepphys", "DoubleSidedCrystalBall", &[m0, sigma_l, sigma_r, alpha_l, n_l, alpha_r, n_r]))
            } else {
                let m0 = d.extra.get("m0").map(|v| field_node(b, v)).unwrap_or_else(|| b.lit_real(0.0));
                let sigma = d.extra.get("sigma").map(|v| field_node(b, v)).unwrap_or_else(|| b.lit_real(1.0));
                let alpha = d.extra.get("alpha").map(|v| field_node(b, v)).unwrap_or_else(|| b.lit_real(1.0));
                let n = d.extra.get("n").map(|v| field_node(b, v)).unwrap_or_else(|| b.lit_real(1.0));
                Ok(b.module_user_call("hepphys", "CrystalBall", &[m0, sigma, alpha, n]))
            }
        }
        "argus_dist" => {
            let resonance = d.extra.get("resonance").map(|v| field_node(b, v)).unwrap_or_else(|| b.lit_real(0.0));
            let slope = d.extra.get("slope").map(|v| field_node(b, v)).unwrap_or_else(|| b.lit_real(-1.0));
            let power = d.extra.get("power").map(|v| field_node(b, v)).unwrap_or_else(|| b.lit_real(0.5));
            Ok(b.module_user_call("hepphys", "Argus", &[resonance, slope, power]))
        }
        // §12: mixture_dist maps to normalize(superpose(weighted(c1, s1), weighted(c2, s2), ...))
        // Summand names are self-refs (other distributions in the document).
        // If extended==true there are N coefficients for N summands; if extended==false (or absent)
        // there are N-1 explicit coefficients and the Nth is implicit = 1 - sum(given).
        // No variate of its own — the summand dists carry their variates (like product_dist).
        "mixture_dist" => {
            let summands: Vec<String> = d
                .extra
                .get("summands")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            if summands.is_empty() {
                return Err(Error::Unsupported("mixture_dist with no summands".into()));
            }
            let coeff_vals: Vec<&serde_json::Value> = d
                .extra
                .get("coefficients")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().collect())
                .unwrap_or_default();
            let extended = d
                .extra
                .get("extended")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            // Build coefficient NodeIds, computing the implicit last for non-extended.
            let n = summands.len();
            let mut coeff_nodes: Vec<NodeId> = coeff_vals.iter().map(|v| field_node(b, v)).collect();

            if !extended {
                // Non-extended: N-1 explicit coefficients; implicit Nth = 1 - sum(given).
                let all_numeric = coeff_vals.iter().all(|v| v.is_number());
                if all_numeric {
                    let given_sum: f64 = coeff_vals
                        .iter()
                        .filter_map(|v| v.as_f64())
                        .sum();
                    coeff_nodes.push(b.lit_real(1.0 - given_sum));
                } else {
                    // At least one symbolic coefficient: build `1 - (c1 + c2 + ...)`
                    // using scalar sub/add builtins.
                    let sum_node: NodeId = if coeff_nodes.is_empty() {
                        b.lit_real(0.0)
                    } else if coeff_nodes.len() == 1 {
                        coeff_nodes[0]
                    } else {
                        let mut acc = coeff_nodes[0];
                        for &c in &coeff_nodes[1..] {
                            acc = b.call("add", &[acc, c]);
                        }
                        acc
                    };
                    let one = b.lit_real(1.0);
                    let implicit = b.call("sub", &[one, sum_node]);
                    coeff_nodes.push(implicit);
                }
            }

            if coeff_nodes.len() != n {
                return Err(Error::Unsupported(format!(
                    "mixture_dist: expected {} coefficients for {} summands, got {}",
                    if extended { n } else { n - 1 },
                    n,
                    coeff_nodes.len() - (if extended { 0 } else { 1 }),
                )));
            }

            // Build weighted(ci, si) pairs.
            let weighted_nodes: Vec<NodeId> = summands
                .iter()
                .zip(coeff_nodes.iter())
                .map(|(s, &c)| {
                    let sref = b.self_ref(s);
                    b.call("weighted", &[c, sref])
                })
                .collect();

            // superpose(weighted(c1,s1), weighted(c2,s2), ...)
            let superpose_node = b.call("superpose", &weighted_nodes);
            // normalize(superpose(...))
            Ok(b.call("normalize", &[superpose_node]))
        }

        // §12: generic_dist → normalize(weighted(<expr_fn>, Lebesgue(reals)))
        // `expression` is a C-like formula string referencing the observable `x`.
        "generic_dist" => {
            let expression = d
                .extra
                .get("expression")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Unsupported("generic_dist missing `expression` field".into()))?;
            let weight_fn = expr::parse_expr_as_fn(b, expression, "x")?;
            let lebesgue = build_lebesgue_reals(b);
            let weighted = b.call("weighted", &[weight_fn, lebesgue]);
            Ok(b.call("normalize", &[weighted]))
        }

        // §12: density_function_dist → normalize(weighted(<named_fn>, Lebesgue(reals)))
        // `function` names a binding (from the `functions` block) that is already
        // a callable accepting the observable.
        "density_function_dist" => {
            let fname = d
                .extra
                .get("function")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Unsupported("density_function_dist missing `function` field".into()))?;
            let fn_ref = b.self_ref(fname);
            let lebesgue = build_lebesgue_reals(b);
            let weighted = b.call("weighted", &[fn_ref, lebesgue]);
            Ok(b.call("normalize", &[weighted]))
        }

        // §12: log_density_function_dist → normalize(logweighted(<named_fn>, Lebesgue(reals)))
        "log_density_function_dist" => {
            let fname = d
                .extra
                .get("function")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Unsupported("log_density_function_dist missing `function` field".into()))?;
            let fn_ref = b.self_ref(fname);
            let lebesgue = build_lebesgue_reals(b);
            let logweighted = b.call("logweighted", &[fn_ref, lebesgue]);
            Ok(b.call("normalize", &[logweighted]))
        }

        // §08: rate_extended_dist → PoissonProcess(weighted(<rate>, self_ref(<distribution>)))
        // No own variate — the inner distribution carries it.
        "rate_extended_dist" => {
            let rate = d
                .extra
                .get("rate")
                .ok_or_else(|| Error::Unsupported("rate_extended_dist missing `rate` field".into()))?;
            let dist_name = d
                .extra
                .get("distribution")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Unsupported("rate_extended_dist missing `distribution` field".into()))?;
            let rate_node = field_node(b, rate);
            let shape = b.self_ref(dist_name);
            let weighted = b.call("weighted", &[rate_node, shape]);
            Ok(b.call("PoissonProcess", &[weighted]))
        }

        // §08: rate_density_dist → PoissonProcess(weighted(self_ref(<function>), Lebesgue(reals)))
        // No own variate.
        "rate_density_dist" => {
            let fname = d
                .extra
                .get("function")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Unsupported("rate_density_dist missing `function` field".into()))?;
            let fn_ref = b.self_ref(fname);
            let lebesgue = build_lebesgue_reals(b);
            let weighted = b.call("weighted", &[fn_ref, lebesgue]);
            Ok(b.call("PoissonProcess", &[weighted]))
        }

        // §08: bincounts_extended_dist → BinnedPoissonProcess(<bins>, weighted(<rate>, self_ref(<distribution>)))
        "bincounts_extended_dist" => {
            let rate = d
                .extra
                .get("rate")
                .ok_or_else(|| Error::Unsupported("bincounts_extended_dist missing `rate` field".into()))?;
            let dist_name = d
                .extra
                .get("distribution")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Unsupported("bincounts_extended_dist missing `distribution` field".into()))?;
            let axes = d
                .extra
                .get("axes")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Unsupported("bincounts_extended_dist missing `axes` field".into()))?;
            let bins = build_bins(b, axes)?;
            let rate_node = field_node(b, rate);
            let shape = b.self_ref(dist_name);
            let weighted = b.call("weighted", &[rate_node, shape]);
            Ok(b.call("BinnedPoissonProcess", &[bins, weighted]))
        }

        // §08: bincounts_density_dist → BinnedPoissonProcess(<bins>, weighted(self_ref(<function>), Lebesgue(reals)))
        "bincounts_density_dist" => {
            let fname = d
                .extra
                .get("function")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Unsupported("bincounts_density_dist missing `function` field".into()))?;
            let axes = d
                .extra
                .get("axes")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Unsupported("bincounts_density_dist missing `axes` field".into()))?;
            let bins = build_bins(b, axes)?;
            let fn_ref = b.self_ref(fname);
            let lebesgue = build_lebesgue_reals(b);
            let weighted = b.call("weighted", &[fn_ref, lebesgue]);
            Ok(b.call("BinnedPoissonProcess", &[bins, weighted]))
        }

        // §08: polynomial_dist → normalize(weighted(functionof(polynomial([c...], _x_), x = _x_), Lebesgue(reals)))
        // Variate is the `x` field.
        "polynomial_dist" => {
            let coeff_arr = d
                .extra
                .get("coefficients")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Unsupported("polynomial_dist missing `coefficients` field".into()))?
                .clone();
            // Build coefficient vector.
            let coeff_elems: Vec<NodeId> = coeff_arr.iter().map(|v| field_node(b, v)).collect();
            let coeff_vec = b.array(&coeff_elems);
            // Build functionof(polynomial([...], _x_), x = _x_) with obs_name from the `x` field.
            let obs_name = d
                .extra
                .get("x")
                .and_then(|v| v.as_str())
                .unwrap_or("x");
            let weight_fn = build_polynomial_fn(b, coeff_vec, obs_name);
            let lebesgue = build_lebesgue_reals(b);
            let weighted = b.call("weighted", &[weight_fn, lebesgue]);
            Ok(b.call("normalize", &[weighted]))
        }

        // §08: barlow_beeston_lite_poisson_constraint_dist →
        //   relabel(broadcast(Poisson, [<expected...>]), [<x names...>])
        "barlow_beeston_lite_poisson_constraint_dist" => {
            let x_names: Vec<String> = d
                .extra
                .get("x")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Unsupported(
                    "barlow_beeston_lite_poisson_constraint_dist missing `x` field".into(),
                ))?
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            let expected_arr = d
                .extra
                .get("expected")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Unsupported(
                    "barlow_beeston_lite_poisson_constraint_dist missing `expected` field".into(),
                ))?
                .clone();
            // Build expected vector.
            let exp_elems: Vec<NodeId> = expected_arr.iter().map(|v| field_node(b, v)).collect();
            let exp_vec = b.array(&exp_elems);
            // broadcast(Poisson, [expected...])
            let poisson_head = b.call_head("Poisson");
            let broadcasted = b.call("broadcast", &[poisson_head, exp_vec]);
            // relabel(..., ["x1", "x2", ...])
            let label_nodes: Vec<NodeId> = x_names.iter().map(|n| b.str_lit(n)).collect();
            let labels = b.array(&label_nodes);
            Ok(b.call("relabel", &[broadcasted, labels]))
        }

        // HS3 RBW uses a multi-channel parameterization; no 1:1 FlatPPL map yet
        "relativistic_breit_wigner_dist" => Err(Error::Unsupported(
            "relativistic_breit_wigner_dist: HS3 uses a multi-channel parameterization with no 1:1 FlatPPL map".into(),
        )),
        "histfactory_dist" => Err(Error::Unsupported(
            "histfactory_dist handled by histfactory.rs".into(),
        )),
        other => Err(Error::UnknownDistType(other.to_string())),
    }
}

/// Build an edge-vector node from a single HS3 axis object.
///
/// Axis forms:
///   `{nbins: N, min: lo, max: hi}` → linear edge vector with N+1 edges.
///   `{edges: [...]}` → literal edge vector.
/// Multi-axis (>1 element) is unsupported and returns `Err(Error::Unsupported)`.
fn build_bins(b: &mut Builder, axes: &[serde_json::Value]) -> Result<NodeId> {
    if axes.len() != 1 {
        return Err(Error::Unsupported(
            "multi-axis bincounts not yet supported".into(),
        ));
    }
    let axis = &axes[0];
    // {edges: [...]} form
    if let Some(edges) = axis.get("edges").and_then(|v| v.as_array()) {
        let nodes: Vec<NodeId> = edges.iter().map(|v| field_node(b, v)).collect();
        return Ok(b.array(&nodes));
    }
    // {nbins, min, max} form
    let nbins = axis
        .get("nbins")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| Error::Unsupported("axis missing both `edges` and `nbins`".into()))?;
    let lo = axis
        .get("min")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| Error::Unsupported("axis missing `min`".into()))?;
    let hi = axis
        .get("max")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| Error::Unsupported("axis missing `max`".into()))?;
    let step = (hi - lo) / nbins as f64;
    let edges: Vec<NodeId> = (0..=nbins)
        .map(|i| b.lit_real(lo + step * i as f64))
        .collect();
    Ok(b.array(&edges))
}

/// Build `functionof(polynomial(coeff_vec, _x_), x = _x_)`.
///
/// This produces a callable weight function over the observable `obs_name`:
/// `obs_name -> polynomial([c1, c2, ...], obs_name)`.
///
/// The placeholder name follows the same convention as `expr::wrap_lambda`:
/// `obs_name` → `_<obs_name>_`.
fn build_polynomial_fn(b: &mut Builder, coeff_vec: NodeId, obs_name: &str) -> NodeId {
    let name_sym = b.m.intern(obs_name);
    let ph_name = format!("_{obs_name}_");
    let ph_sym = b.m.intern(&ph_name);
    // Build Local ref for the placeholder — this is the `x` argument to polynomial.
    let local_ref = b.m.alloc(Node::Ref(Ref {
        ns: RefNs::Local,
        name: ph_sym,
    }));
    // polynomial([...], _x_)
    let body = b.call("polynomial", &[coeff_vec, local_ref]);
    // functionof(body, inputs: Spec[(obs_name -> Local(_ph_))])
    let head = CallHead::Builtin(b.m.intern("functionof"));
    let entry = (
        name_sym,
        Ref {
            ns: RefNs::Local,
            name: ph_sym,
        },
    );
    b.m.alloc(Node::Call(Call {
        head,
        args: Box::new([body]),
        named: Box::new([]),
        inputs: Some(Inputs::Spec(Box::new([entry]))),
    }))
}

/// Build `Lebesgue(reals)` — the continuous reference measure on ℝ.
fn build_lebesgue_reals(b: &mut Builder) -> NodeId {
    let reals_sym = b.m.intern("reals");
    let reals = b.m.alloc(flatppl_core::node::Node::Const(reals_sym));
    b.call("Lebesgue", &[reals])
}

/// The scalar variate field name for the given distribution kind.
/// Returns the field key whose value is the observed-variable name.
fn variate_field(kind: &str) -> &'static str {
    match kind {
        "crystalball_dist" => "m",
        "argus_dist" => "mass",
        _ => "x",
    }
}

/// Returns true if this distribution kind carries no variate of its own
/// (variate comes from summand/factor distributions, or the expression embeds it).
fn has_no_own_variate(kind: &str) -> bool {
    matches!(
        kind,
        "mixture_dist"
            | "product_dist"
            | "generic_dist"
            | "density_function_dist"
            | "log_density_function_dist"
            // Poisson-process types: variate comes from inner distribution / is the count space.
            | "rate_extended_dist"
            | "rate_density_dist"
            | "bincounts_extended_dist"
            | "bincounts_density_dist"
    )
}

/// The variate name(s) for a distribution, if any.
///
/// Returns `VariateName::Single` for scalar variates and
/// `VariateName::Multiple` for `multivariate_normal_dist` whose `x` field is
/// an array of observed-variable names.
#[derive(Debug, PartialEq)]
pub enum VariateName {
    Single(String),
    Multiple(Vec<String>),
}

/// Extract the variate from a distribution.  Returns `None` if the variate
/// field is absent.
pub fn variate_name(d: &Distribution) -> Option<VariateName> {
    // Composite distributions carry no variate of their own.
    if has_no_own_variate(&d.kind) {
        return None;
    }
    // Distributions with an array-valued `x` field: multivariate_normal_dist and
    // barlow_beeston_lite_poisson_constraint_dist both use x as a list of obs names.
    // barlow_beeston_lite emits relabel(..., [x_names]) itself — return None so the
    // convert.rs caller does NOT wrap with an additional relabel.
    if d.kind == "barlow_beeston_lite_poisson_constraint_dist" {
        return None;
    }
    if d.kind == "multivariate_normal_dist" {
        // x is a JSON array of variable-name strings
        if let Some(arr) = d.extra.get("x").and_then(|v| v.as_array()) {
            let names: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            if !names.is_empty() {
                return Some(VariateName::Multiple(names));
            }
        }
        return None;
    }
    let field = variate_field(&d.kind);
    d.extra
        .get(field)
        .and_then(|v| v.as_str())
        .map(|s| VariateName::Single(s.to_string()))
}

/// Returns true if this distribution kind requires `hepphys` module to be in scope.
pub fn needs_hepphys(kind: &str) -> bool {
    matches!(kind, "crystalball_dist" | "argus_dist")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::Builder;
    use crate::model::Distribution;
    use flatppl_syntax::{Syntax, print_with};
    use std::collections::BTreeMap;

    fn dist(kind: &str, fields: &[(&str, serde_json::Value)]) -> Distribution {
        Distribution {
            name: "d".into(),
            kind: kind.into(),
            extra: fields
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect::<BTreeMap<_, _>>(),
        }
    }

    #[test]
    fn gaussian_maps_to_normal() {
        let d = dist(
            "gaussian_dist",
            &[
                ("mean", serde_json::json!("mu_param")),
                ("sigma", serde_json::json!("sigma_param")),
                ("x", serde_json::json!("mass_obs")),
            ],
        );
        let mut m = flatppl_core::Module::new();
        let node = {
            let mut b = Builder::new(&mut m);
            emit_distribution(&mut b, &d).unwrap()
        };
        {
            let mut b = Builder::new(&mut m);
            b.bind("d", node);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("Normal"), "got: {text}");
        assert!(
            text.contains("mu_param") && text.contains("sigma_param"),
            "got: {text}"
        );
    }

    #[test]
    fn normal_dist_alias() {
        let d = dist(
            "normal_dist",
            &[
                ("mean", serde_json::json!(0.0)),
                ("sigma", serde_json::json!(1.0)),
            ],
        );
        let mut m = flatppl_core::Module::new();
        let node = {
            let mut b = Builder::new(&mut m);
            emit_distribution(&mut b, &d).unwrap()
        };
        {
            let mut b = Builder::new(&mut m);
            b.bind("d", node);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("Normal"), "got: {text}");
    }

    #[test]
    fn poisson_maps_mean_to_rate() {
        let d = dist(
            "poisson_dist",
            &[
                ("mean", serde_json::json!("lambda_param")),
                ("x", serde_json::json!("n_obs")),
            ],
        );
        let mut m = flatppl_core::Module::new();
        let node = {
            let mut b = Builder::new(&mut m);
            emit_distribution(&mut b, &d).unwrap()
        };
        {
            let mut b = Builder::new(&mut m);
            b.bind("d", node);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("Poisson"), "got: {text}");
        assert!(text.contains("rate"), "got: {text}");
        assert!(text.contains("lambda_param"), "got: {text}");
    }

    #[test]
    fn exponential_maps_c_to_rate() {
        let d = dist(
            "exponential_dist",
            &[
                ("c", serde_json::json!("c_param")),
                ("x", serde_json::json!("t_obs")),
            ],
        );
        let mut m = flatppl_core::Module::new();
        let node = {
            let mut b = Builder::new(&mut m);
            emit_distribution(&mut b, &d).unwrap()
        };
        {
            let mut b = Builder::new(&mut m);
            b.bind("d", node);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("Exponential"), "got: {text}");
        assert!(text.contains("rate"), "got: {text}");
        assert!(text.contains("c_param"), "got: {text}");
    }

    #[test]
    fn lognormal_maps_correctly() {
        let d = dist(
            "lognormal_dist",
            &[
                ("mu", serde_json::json!("mu_param")),
                ("sigma", serde_json::json!("sigma_param")),
                ("x", serde_json::json!("x_obs")),
            ],
        );
        let mut m = flatppl_core::Module::new();
        let node = {
            let mut b = Builder::new(&mut m);
            emit_distribution(&mut b, &d).unwrap()
        };
        {
            let mut b = Builder::new(&mut m);
            b.bind("d", node);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("LogNormal"), "got: {text}");
        assert!(
            text.contains("mu_param") && text.contains("sigma_param"),
            "got: {text}"
        );
    }

    #[test]
    fn uniform_maps_correctly() {
        let d = dist("uniform_dist", &[("x", serde_json::json!("x_obs"))]);
        let mut m = flatppl_core::Module::new();
        let node = {
            let mut b = Builder::new(&mut m);
            emit_distribution(&mut b, &d).unwrap()
        };
        {
            let mut b = Builder::new(&mut m);
            b.bind("d", node);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("Uniform"), "got: {text}");
    }

    #[test]
    fn unknown_dist_type_errors() {
        let d = dist("no_such_dist", &[]);
        let mut m = flatppl_core::Module::new();
        let mut b = Builder::new(&mut m);
        let result = emit_distribution(&mut b, &d);
        assert!(matches!(result, Err(Error::UnknownDistType(_))));
    }

    #[test]
    fn histfactory_dist_unsupported() {
        let d = dist("histfactory_dist", &[]);
        let mut m = flatppl_core::Module::new();
        let mut b = Builder::new(&mut m);
        let result = emit_distribution(&mut b, &d);
        assert!(matches!(result, Err(Error::Unsupported(_))));
    }

    #[test]
    fn variate_name_extracts_x() {
        let d = dist("gaussian_dist", &[("x", serde_json::json!("mass_obs"))]);
        assert_eq!(
            variate_name(&d),
            Some(VariateName::Single("mass_obs".to_string()))
        );
    }

    #[test]
    fn variate_name_absent() {
        let d = dist("gaussian_dist", &[]);
        assert_eq!(variate_name(&d), None);
    }

    #[test]
    fn variate_name_crystalball_uses_m_field() {
        let d = dist("crystalball_dist", &[("m", serde_json::json!("m_obs"))]);
        assert_eq!(
            variate_name(&d),
            Some(VariateName::Single("m_obs".to_string()))
        );
    }

    #[test]
    fn variate_name_argus_uses_mass_field() {
        let d = dist("argus_dist", &[("mass", serde_json::json!("m_obs"))]);
        assert_eq!(
            variate_name(&d),
            Some(VariateName::Single("m_obs".to_string()))
        );
    }

    #[test]
    fn variate_name_mvnormal_multiple() {
        let d = dist(
            "multivariate_normal_dist",
            &[("x", serde_json::json!(["x0", "x1"]))],
        );
        assert_eq!(
            variate_name(&d),
            Some(VariateName::Multiple(vec![
                "x0".to_string(),
                "x1".to_string()
            ]))
        );
    }

    #[test]
    fn numeric_field_uses_lit_real() {
        let d = dist(
            "gaussian_dist",
            &[
                ("mean", serde_json::json!(0.0)),
                ("sigma", serde_json::json!(1.0)),
            ],
        );
        let mut m = flatppl_core::Module::new();
        let node = {
            let mut b = Builder::new(&mut m);
            emit_distribution(&mut b, &d).unwrap()
        };
        {
            let mut b = Builder::new(&mut m);
            b.bind("d", node);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("Normal"), "got: {text}");
        // numeric literals should appear in output
        assert!(text.contains("0") || text.contains("1"), "got: {text}");
    }
}
