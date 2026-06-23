//! Fundamental HS3 distribution `type` -> FlatPPL distribution call.
use crate::builder::Builder;
use crate::dist_spec::{self, Variate};
use crate::error::{Error, Result};
use crate::expr;
use crate::model::Distribution;
use flatppl_core::id::NodeId;
use flatppl_core::node::{Call, CallHead, Inputs, NamedArg, NamedKind, Node, Ref, RefNs};

/// Build a FlatPPL array node from a JSON array of scalars (numbers/strings).
fn array_of_values(b: &mut Builder, arr: &[serde_json::Value]) -> Result<NodeId> {
    let elems: Vec<NodeId> = arr
        .iter()
        .map(|v| field_node(b, v))
        .collect::<Result<_>>()?;
    Ok(b.array(&elems))
}

/// Build a FlatPPL 2-D array (vector of vectors) from a JSON 2-D array.
fn array2d_of_values(b: &mut Builder, arr: &[serde_json::Value]) -> Result<NodeId> {
    let rows: Vec<NodeId> = arr
        .iter()
        .map(|row| {
            if let Some(inner) = row.as_array() {
                array_of_values(b, inner)
            } else {
                field_node(b, row)
            }
        })
        .collect::<Result<_>>()?;
    Ok(b.array(&rows))
}

/// Context for conditional distribution parameter resolution.
///
/// When a parameter value names a `generic_function` of an observable axis,
/// `field_node_ctx` emits `func(axis)` (a `CallHead::User` call) rather than a
/// bare `self_ref`. Without this context the behaviour is identical to
/// `field_node`.
// `axes` is not yet consumed within this crate; Task 3 (emit_conditional) will use it.
#[allow(dead_code)]
pub struct CondCtx<'a> {
    /// Maps a `generic_function` name → its observable axis name.
    pub funcs: &'a std::collections::BTreeMap<&'a str, &'a str>,
    /// The set of dataset observable axis names.
    pub axes: &'a std::collections::BTreeSet<&'a str>,
}

/// Resolve a parameter value. With a conditional context, a value that names a
/// `generic_function` of an observable axis emits `func(axis)` (a user-call);
/// otherwise behaves exactly like `field_node`.
fn field_node_ctx(
    b: &mut Builder,
    v: &serde_json::Value,
    cond: Option<&CondCtx>,
) -> Result<NodeId> {
    if let (Some(ctx), Some(name)) = (cond, v.as_str()) {
        if let Some(axis) = ctx.funcs.get(name) {
            let func = b.self_ref(name);
            let arg = b.self_ref(axis);
            return Ok(b
                .m
                .alloc(flatppl_core::node::Node::Call(flatppl_core::node::Call {
                    head: flatppl_core::node::CallHead::User(func),
                    args: Box::new([arg]),
                    named: Box::new([]),
                    inputs: None,
                })));
        }
    }
    field_node(b, v)
}

/// Lower a scalar JSON field value to a FlatPPL node: numbers become real
/// literals, strings become self-refs (to a parameter/binding by name).
///
/// Any other JSON shape (object, array, bool, null) — or a number that does not
/// fit an `f64` — has no scalar lowering and is rejected with
/// [`Error::Unsupported`] rather than being silently coerced to `0.0`.
pub(crate) fn field_node(b: &mut Builder, v: &serde_json::Value) -> Result<NodeId> {
    match v {
        serde_json::Value::Number(n) => n.as_f64().map(|x| b.lit_real(x)).ok_or_else(|| {
            Error::Unsupported(format!("numeric field not representable as f64: {n}"))
        }),
        // A numeric literal written as a string (e.g. "1.0" — common in HS3/RooFit
        // where coefficient/parameter values are stringified) lowers to a real
        // literal; any other string is a reference to another binding (a
        // parameter, function, or distribution).
        serde_json::Value::String(s) => match s.parse::<f64>() {
            Ok(x) => Ok(b.lit_real(x)),
            Err(_) => Ok(b.self_ref(s)),
        },
        other => Err(Error::Unsupported(format!(
            "expected a numeric or string field value, got: {other}"
        ))),
    }
}

/// Lower the `key` field of `d` to a node, falling back to the real literal
/// `default` when the field is absent. Propagates the [`field_node`] error for a
/// present-but-unlowerable value (object/array/null/bool) rather than masking it
/// with the default.
fn field_or(b: &mut Builder, d: &Distribution, key: &str, default: f64) -> Result<NodeId> {
    match d.extra.get(key) {
        Some(v) => field_node(b, v),
        None => Ok(b.lit_real(default)),
    }
}

/// Bare distribution call (no relabel; caller wraps with the variate).
///
/// `domain` carries the `(min, max)` bounds resolved from the document's
/// `domains` block for this distribution's variate, when one is declared. It is
/// required for `uniform_dist` (whose support has no other source) and ignored
/// by all other kinds.
///
/// `generic_obs` is the observable variable a `generic_dist`'s inline expression
/// is a function of (inferred by the caller from the document). When `None`, the
/// `generic_dist` lowering falls back to the conventional `"x"`.
pub fn emit_distribution(
    b: &mut Builder,
    d: &Distribution,
    domain: Option<(f64, f64)>,
    generic_obs: Option<&str>,
    cond: Option<&CondCtx>,
) -> Result<NodeId> {
    match d.kind.as_str() {
        "gaussian_dist" | "normal_dist" => {
            let mut kws: Vec<(&str, NodeId)> = Vec::new();
            if let Some(v) = d.extra.get("mean") {
                kws.push(("mu", field_node_ctx(b, v, cond)?));
            }
            if let Some(v) = d.extra.get("sigma") {
                kws.push(("sigma", field_node_ctx(b, v, cond)?));
            }
            Ok(b.call_kw("Normal", &kws))
        }
        "poisson_dist" => {
            let mut kws: Vec<(&str, NodeId)> = Vec::new();
            if let Some(v) = d.extra.get("mean") {
                kws.push(("rate", field_node(b, v)?));
            }
            Ok(b.call_kw("Poisson", &kws))
        }
        // §08/§12: HS³ `exponential_dist` density is exp(−c·x), so the HS³ `c` is
        // a positive decay rate. FlatPPL `Exponential(rate)` is rate·exp(−rate·x),
        // so the rate maps directly: rate = c, no negation. (RooFit's internal
        // RooExponential slope is −rate, but HS³ stores the already-inverted,
        // positive c — e.g. fixtures bind c to a `-tau` function.)
        "exponential_dist" => {
            let mut kws: Vec<(&str, NodeId)> = Vec::new();
            if let Some(v) = d.extra.get("c") {
                kws.push(("rate", field_node(b, v)?));
            }
            Ok(b.call_kw("Exponential", &kws))
        }
        "lognormal_dist" => {
            let mut kws: Vec<(&str, NodeId)> = Vec::new();
            if let Some(v) = d.extra.get("mu") {
                kws.push(("mu", field_node(b, v)?));
            }
            if let Some(v) = d.extra.get("sigma") {
                kws.push(("sigma", field_node(b, v)?));
            }
            Ok(b.call_kw("LogNormal", &kws))
        }
        // §08: Uniform(S) == normalize(Lebesgue(S)). The support set is the
        // variate's declared domain interval; without one there is no finite
        // support to normalize over, so reject rather than emit a bare Uniform().
        "uniform_dist" => {
            let (min, max) = domain.ok_or_else(|| {
                Error::Unsupported(format!(
                    "uniform_dist `{}` has no declared domain for its variate; \
                     a `domains` entry giving (min, max) is required",
                    d.name
                ))
            })?;
            let lo = b.lit_real(min);
            let hi = b.lit_real(max);
            let support = b.call("interval", &[lo, hi]);
            Ok(b.call("Uniform", &[support]))
        }
        // §12: product_dist is lowered by `emit_product`, which needs each
        // factor's variate (resolved by the caller from the document) to choose
        // between an independent `joint` and a same-variate pointwise density
        // product. It is therefore dispatched from convert.rs, not here.
        "product_dist" => Err(Error::Unsupported(
            "product_dist is handled via emit_product".into(),
        )),
        "generalized_normal_dist" => {
            let mut kws: Vec<(&str, NodeId)> = Vec::new();
            if let Some(v) = d.extra.get("mean") {
                kws.push(("mean", field_node(b, v)?));
            }
            if let Some(v) = d.extra.get("alpha") {
                kws.push(("alpha", field_node(b, v)?));
            }
            if let Some(v) = d.extra.get("beta") {
                kws.push(("beta", field_node(b, v)?));
            }
            Ok(b.call_kw("GeneralizedNormal", &kws))
        }
        "multivariate_normal_dist" => {
            let mut kws: Vec<(&str, NodeId)> = Vec::new();
            if let Some(arr) = d.extra.get("mean").and_then(|v| v.as_array()) {
                kws.push(("mu", array_of_values(b, arr)?));
            }
            if let Some(arr) = d.extra.get("covariances").and_then(|v| v.as_array()) {
                kws.push(("cov", array2d_of_values(b, arr)?));
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
                let m0 = field_or(b, d, "m0", 0.0)?;
                let sigma_l = field_or(b, d, "sigma_L", 1.0)?;
                let sigma_r = field_or(b, d, "sigma_R", 1.0)?;
                let alpha_l = field_or(b, d, "alpha_L", 1.0)?;
                let n_l = field_or(b, d, "n_L", 1.0)?;
                let alpha_r = field_or(b, d, "alpha_R", 1.0)?;
                let n_r = field_or(b, d, "n_R", 1.0)?;
                Ok(b.module_user_call("hepphys", "DoubleSidedCrystalBall", &[m0, sigma_l, sigma_r, alpha_l, alpha_r, n_l, n_r]))
            } else {
                let m0 = field_or(b, d, "m0", 0.0)?;
                let sigma = field_or(b, d, "sigma", 1.0)?;
                let alpha = field_or(b, d, "alpha", 1.0)?;
                let n = field_or(b, d, "n", 1.0)?;
                Ok(b.module_user_call("hepphys", "CrystalBall", &[m0, sigma, alpha, n]))
            }
        }
        "argus_dist" => {
            let resonance = field_or(b, d, "resonance", 0.0)?;
            let slope = field_or(b, d, "slope", -1.0)?;
            let power = field_or(b, d, "power", 0.5)?;
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
            let mut coeff_nodes: Vec<NodeId> = coeff_vals
                .iter()
                .map(|v| field_node(b, v))
                .collect::<Result<_>>()?;

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
            if extended {
                // §12:140 — the extended mixture is the *unnormalized* superposition
                // (RooAddPdf in extended mode); coefficients are absolute yields, so
                // there is no outer normalize.
                Ok(superpose_node)
            } else {
                // Non-extended: coefficients are mixing fractions summing to 1, so
                // normalize the superposition.
                Ok(b.call("normalize", &[superpose_node]))
            }
        }

        // §12: generic_dist → normalize(weighted(<expr_fn>, Lebesgue(reals)))
        // When the observable's domain is declared, the normalization is restricted
        // to that interval: normalize(truncate(weighted(…, Lebesgue(reals)), interval(lo, hi))).
        // Without a declared domain the fallback is normalize(weighted(…, Lebesgue(reals))).
        "generic_dist" => {
            let expression = d
                .extra
                .get("expression")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Unsupported("generic_dist missing `expression` field".into()))?;
            let obs = generic_obs.unwrap_or("x");
            let weight_fn = expr::parse_expr_as_fn(b, expression, obs)?;
            let lebesgue = build_lebesgue_reals(b);
            let weighted = b.call("weighted", &[weight_fn, lebesgue]);
            let measure = match domain {
                Some((lo, hi)) => {
                    let lo_node = b.lit_real(lo);
                    let hi_node = b.lit_real(hi);
                    let itvl = b.call("interval", &[lo_node, hi_node]);
                    b.call("truncate", &[weighted, itvl])
                }
                None => weighted,
            };
            Ok(b.call("normalize", &[measure]))
        }

        // §12: density_function_dist → normalize(weighted(<named_fn>, Lebesgue(reals)))
        // When the observable's domain is declared, wraps with truncate over the interval.
        "density_function_dist" => {
            let fname = d
                .extra
                .get("function")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Unsupported("density_function_dist missing `function` field".into()))?;
            let fn_ref = b.self_ref(fname);
            let lebesgue = build_lebesgue_reals(b);
            let weighted = b.call("weighted", &[fn_ref, lebesgue]);
            let measure = match domain {
                Some((lo, hi)) => {
                    let lo_node = b.lit_real(lo);
                    let hi_node = b.lit_real(hi);
                    let itvl = b.call("interval", &[lo_node, hi_node]);
                    b.call("truncate", &[weighted, itvl])
                }
                None => weighted,
            };
            Ok(b.call("normalize", &[measure]))
        }

        // §12: log_density_function_dist → normalize(logweighted(<named_fn>, Lebesgue(reals)))
        // When the observable's domain is declared, wraps with truncate over the interval.
        "log_density_function_dist" => {
            let fname = d
                .extra
                .get("function")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Unsupported("log_density_function_dist missing `function` field".into()))?;
            let fn_ref = b.self_ref(fname);
            let lebesgue = build_lebesgue_reals(b);
            let logweighted = b.call("logweighted", &[fn_ref, lebesgue]);
            let measure = match domain {
                Some((lo, hi)) => {
                    let lo_node = b.lit_real(lo);
                    let hi_node = b.lit_real(hi);
                    let itvl = b.call("interval", &[lo_node, hi_node]);
                    b.call("truncate", &[logweighted, itvl])
                }
                None => logweighted,
            };
            Ok(b.call("normalize", &[measure]))
        }

        // §12: efficiency_product_pdf_dist (RooEffProd) → weighted(self_ref(<eff>), self_ref(<pdf>))
        // The efficiency function reweights the pdf measure pointwise. No own
        // variate (the inner pdf carries it); the product is integrable over the
        // pdf's support, so observable-range normalization is applied downstream
        // (range-normalized by the consumer), like a raw distribution.
        "efficiency_product_pdf_dist" => {
            let eff_name = d
                .extra
                .get("eff")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    Error::Unsupported("efficiency_product_pdf_dist missing `eff` field".into())
                })?;
            let pdf_name = d
                .extra
                .get("pdf")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    Error::Unsupported("efficiency_product_pdf_dist missing `pdf` field".into())
                })?;
            let eff_ref = b.self_ref(eff_name);
            let pdf_ref = b.self_ref(pdf_name);
            Ok(b.call("weighted", &[eff_ref, pdf_ref]))
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
            let rate_node = field_node(b, rate)?;
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
            let rate_node = field_node(b, rate)?;
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

        // §08/§12: polynomial_dist (RooPolynomial) → normalize(truncate(weighted(
        //   functionof(polynomial([c...], _x_), x = _x_), Lebesgue(reals)), interval(lo, hi)))
        // RooFit normalizes the polynomial over the observable's DECLARED range, so
        // (exactly like chebychev_dist) the weighted measure must be truncated to
        // interval(lo, hi) before normalize — over all of ℝ the integral of a
        // degree≥1 polynomial diverges and the normalizer is infinite.
        // Variate is the `x` field.
        "polynomial_dist" => {
            let (lo, hi) = domain.ok_or_else(|| {
                Error::Unsupported(format!(
                    "polynomial_dist `{}` has no declared domain for its variate; \
                     a `domains` entry giving (min, max) is required (RooPolynomial \
                     normalizes over the observable's range)",
                    d.name
                ))
            })?;
            let coeff_arr = d
                .extra
                .get("coefficients")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Unsupported("polynomial_dist missing `coefficients` field".into()))?
                .clone();
            // Build coefficient vector.
            let coeff_elems: Vec<NodeId> = coeff_arr
                .iter()
                .map(|v| field_node(b, v))
                .collect::<Result<_>>()?;
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
            let lo_node = b.lit_real(lo);
            let hi_node = b.lit_real(hi);
            let interval = b.call("interval", &[lo_node, hi_node]);
            let truncated = b.call("truncate", &[weighted, interval]);
            Ok(b.call("normalize", &[truncated]))
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
            let exp_elems: Vec<NodeId> = expected_arr
                .iter()
                .map(|v| field_node(b, v))
                .collect::<Result<_>>()?;
            let exp_vec = b.array(&exp_elems);
            // broadcast(Poisson, [expected...])
            let poisson_head = b.call_head("Poisson");
            let broadcasted = b.call("broadcast", &[poisson_head, exp_vec]);
            // relabel(..., ["x1", "x2", ...])
            let label_nodes: Vec<NodeId> = x_names.iter().map(|n| b.str_lit(n)).collect();
            let labels = b.array(&label_nodes);
            Ok(b.call("relabel", &[broadcasted, labels]))
        }

        // §09: chebychev_dist → RooChebychev convention.
        //
        // pdf(x) ∝ 1 + Σ_{k=1..N} a_k · T_k(t)
        //   t = (2x − lo − hi) / (hi − lo)   (rescale [lo,hi] → [−1,1])
        //   T_k = Chebyshev polynomial of the first kind (degree k)
        //   a_1..a_N = coefficients[0..N-1]  (degrees start at 1; T_0=1 is the implicit leading 1)
        // Emits: normalize(truncate(weighted(functionof(WEIGHT, x=_x_), Lebesgue(reals)), interval(lo,hi)))
        //
        // NUMERIC CONVENTION PENDING: Phase-4 rf207 end-to-end confirmation vs ROOT required.
        "chebychev_dist" => {
            let (lo, hi) = domain.ok_or_else(|| {
                Error::Unsupported(format!(
                    "chebychev_dist `{}` has no declared domain for its variate; \
                     a `domains` entry giving (min, max) is required",
                    d.name
                ))
            })?;
            let coeff_arr = d
                .extra
                .get("coefficients")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Unsupported("chebychev_dist missing `coefficients` field".into()))?
                .clone();
            let obs_name = d
                .extra
                .get("x")
                .and_then(|v| v.as_str())
                .unwrap_or("x");
            // Build the placeholder ref for the observable.
            let ph_name = format!("_{obs_name}_");
            let ph_sym = b.m.intern(&ph_name);
            let x_ref = b.m.alloc(Node::Ref(Ref {
                ns: RefNs::Local,
                name: ph_sym,
            }));
            // t = divide(sub(mul(2.0, _x_), lo+hi), hi-lo)  — real division.
            // (NOT `div`, which is integer floor division and would quantize t.)
            let two = b.lit_real(2.0);
            let lo_plus_hi = b.lit_real(lo + hi);
            let hi_minus_lo = b.lit_real(hi - lo);
            let two_x = b.call("mul", &[two, x_ref]);
            let numerator = b.call("sub", &[two_x, lo_plus_hi]);
            let t = b.call("divide", &[numerator, hi_minus_lo]);
            // Build the Chebyshev series: 1 + fold of add(acc, mul(coeff_k, poly.chebyshev(k, t)))
            let mut weight = b.lit_real(1.0);
            for (idx, coeff_val) in coeff_arr.iter().enumerate() {
                let degree = b.lit_int((idx + 1) as i64);
                let cheby = b.module_user_call("poly", "chebyshev", &[degree, t]);
                let coeff = field_node(b, coeff_val)?;
                let term = b.call("mul", &[coeff, cheby]);
                weight = b.call("add", &[weight, term]);
            }
            // functionof(weight, obs_name = _obs_name_)
            let name_sym = b.m.intern(obs_name);
            let functionof_sym = b.m.intern("functionof");
            let entry = (
                name_sym,
                Ref {
                    ns: RefNs::Local,
                    name: ph_sym,
                },
            );
            let weight_fn = b.m.alloc(Node::Call(Call {
                head: CallHead::Builtin(functionof_sym),
                args: Box::new([weight]),
                named: Box::new([]),
                inputs: Some(Inputs::Spec(Box::new([entry]))),
            }));
            let lebesgue = build_lebesgue_reals(b);
            let weighted = b.call("weighted", &[weight_fn, lebesgue]);
            let lo_node = b.lit_real(lo);
            let hi_node = b.lit_real(hi);
            let interval = b.call("interval", &[lo_node, hi_node]);
            let truncated = b.call("truncate", &[weighted, interval]);
            Ok(b.call("normalize", &[truncated]))
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
        let nodes: Vec<NodeId> = edges
            .iter()
            .map(|v| field_node(b, v))
            .collect::<Result<_>>()?;
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
pub(crate) fn build_polynomial_fn(b: &mut Builder, coeff_vec: NodeId, obs_name: &str) -> NodeId {
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

/// HS3 `product_dist` lowering (§12). The form depends on the factors' variates,
/// which the caller resolves from the document (in `factors` order):
///
/// * **shared single variate** — all factors are pdfs over the *same* observable
///   (RooProdPdf of same-observable pdfs): the normalized pointwise density
///   product. Take the first factor as the base measure and reweight it, in log
///   space, by the sum of the other factors' log-densities:
///   `normalize(logweighted(functionof(add(logdensityof(M2, _), …, logdensityof(Mₙ, _))), M1))`.
///   This is flat in the factor count (one `add`-fold of N−1 log-densities, no
///   nesting) and the reweighted base yields the unnormalized product `∏ᵢ gᵢ`,
///   which `normalize` renders a probability measure (§06).
/// * **otherwise** — an independent product over distinct variates:
///   `joint(f1 = M1, f2 = M2, …)`.
pub fn emit_product(
    b: &mut Builder,
    factors: &[String],
    factor_variates: &[Option<VariateName>],
) -> Result<NodeId> {
    if factors.is_empty() {
        return Err(Error::Unsupported("product_dist with no factors".into()));
    }
    if product_shared_variate(factor_variates) {
        // Normalized pointwise density product over the shared variate `var`.
        // weight(point) = exp(Σ_{i≥1} logdensityof(factorᵢ, point)); base = factor₀.
        let var = match &factor_variates[0] {
            Some(VariateName::Single(s)) => s.clone(),
            // product_shared_variate guarantees factor 0 is a Single variate.
            _ => unreachable!("product_shared_variate ensures a single variate"),
        };
        let ph_sym = b.m.intern(&format!("_{var}_"));
        let point = b.m.alloc(Node::Ref(Ref {
            ns: RefNs::Local,
            name: ph_sym,
        }));
        // add-fold of logdensityof(factorᵢ, point) for the non-base factors.
        let mut body: Option<NodeId> = None;
        for f in &factors[1..] {
            let mref = b.self_ref(f);
            let ld = b.call("logdensityof", &[mref, point]);
            body = Some(match body {
                None => ld,
                Some(prev) => b.call("add", &[prev, ld]),
            });
        }
        let body = body.expect("shared variate implies >= 2 factors");
        // functionof(body, var = _var_) — the single-argument weight function.
        let head = CallHead::Builtin(b.m.intern("functionof"));
        let entry = (
            b.m.intern(&var),
            Ref {
                ns: RefNs::Local,
                name: ph_sym,
            },
        );
        let weight_fn = b.m.alloc(Node::Call(Call {
            head,
            args: Box::new([body]),
            named: Box::new([]),
            inputs: Some(Inputs::Spec(Box::new([entry]))),
        }));
        let base = b.self_ref(&factors[0]);
        let lw = b.call("logweighted", &[weight_fn, base]);
        Ok(b.call("normalize", &[lw]))
    } else {
        // Independent product → joint keyed by each factor's VARIATE name, so the
        // joint's record fields match the observable names in the data / domain
        // (e.g. `joint(x = gx, y = gy)`). The keyword form is exactly
        // `joint(relabel(gx, ["x"]), …)` (§06), so the factor measures stay bare.
        // A multivariate or unnamed factor falls back to its binding name.
        let named: Vec<NamedArg> = factors
            .iter()
            .zip(factor_variates)
            .map(|(f, fv)| {
                let key = match fv {
                    Some(VariateName::Single(v)) => v.clone(),
                    _ => f.clone(),
                };
                NamedArg {
                    kind: NamedKind::Field,
                    name: b.m.intern(&key),
                    value: b.self_ref(f),
                }
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
}

/// Factor distribution names of a `product_dist`, in document order.
pub fn product_factors(d: &Distribution) -> Vec<String> {
    d.extra
        .get("factors")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// True when there are ≥2 factors and every one has the same single scalar
/// variate — the RooProdPdf-over-a-shared-observable case that lowers to a
/// pointwise density product rather than an independent `joint`.
pub fn product_shared_variate(factor_variates: &[Option<VariateName>]) -> bool {
    if factor_variates.len() < 2 {
        return false;
    }
    let mut names = factor_variates.iter().map(|v| match v {
        Some(VariateName::Single(s)) => Some(s.as_str()),
        _ => None,
    });
    let first = match names.next() {
        Some(Some(s)) => s,
        _ => return false,
    };
    names.all(|v| v == Some(first))
}

/// The reference (base) measure a distribution's density is taken against.
/// Used to guard the shared-variate `product_dist` lowering: a pointwise density
/// product is only meaningful when all factors share one reference measure (§12).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RefMeasure {
    /// Continuous — density w.r.t. Lebesgue measure.
    Lebesgue,
    /// Discrete — density (pmf) w.r.t. the counting measure.
    Counting,
    /// Composite or point-process distributions whose measure is not a simple
    /// pointwise density over a shared scalar observable — never eligible for the
    /// density-product form (treated as incompatible).
    Other,
}

/// Classify a distribution kind's reference measure (see [`RefMeasure`]).
pub fn reference_measure(kind: &str) -> RefMeasure {
    match kind {
        "gaussian_dist"
        | "normal_dist"
        | "exponential_dist"
        | "lognormal_dist"
        | "uniform_dist"
        | "generalized_normal_dist"
        | "crystalball_dist"
        | "argus_dist"
        | "multivariate_normal_dist"
        | "generic_dist"
        | "density_function_dist"
        | "log_density_function_dist"
        | "polynomial_dist"
        | "relativistic_breit_wigner_dist" => RefMeasure::Lebesgue,
        "poisson_dist" | "barlow_beeston_lite_poisson_constraint_dist" => RefMeasure::Counting,
        _ => RefMeasure::Other,
    }
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

/// Extract the per-instance variate name(s) from a distribution, consulting the
/// static [`dist_spec`] table for the kind's variate shape and field key.
/// Returns `None` when the kind carries no variate of its own
/// ([`Variate::None`]) or the variate field is absent.
pub fn variate_name(d: &Distribution) -> Option<VariateName> {
    match dist_spec::variate(&d.kind) {
        // No variate of its own (composites, expression kinds, Poisson-process
        // kinds, and barlow_beeston_lite which emits its own relabel).
        Variate::None => None,
        // Scalar variate: the named field holds a single observed-variable name.
        Variate::Scalar(field) => d
            .extra
            .get(field)
            .and_then(|v| v.as_str())
            .map(|s| VariateName::Single(s.to_string())),
        // Array variate (multivariate_normal_dist): the field holds a JSON array
        // of observed-variable names.
        Variate::MultiArray(field) => {
            let names: Vec<String> = d
                .extra
                .get(field)
                .and_then(|v| v.as_array())?
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            (!names.is_empty()).then_some(VariateName::Multiple(names))
        }
    }
}

/// Returns true if this distribution kind requires `hepphys` module to be in scope.
pub fn needs_hepphys(kind: &str) -> bool {
    dist_spec::needs_hepphys(kind)
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
            emit_distribution(&mut b, &d, None, None, None).unwrap()
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
            emit_distribution(&mut b, &d, None, None, None).unwrap()
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
            emit_distribution(&mut b, &d, None, None, None).unwrap()
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
            emit_distribution(&mut b, &d, None, None, None).unwrap()
        };
        {
            let mut b = Builder::new(&mut m);
            b.bind("d", node);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("Exponential"), "got: {text}");
        assert!(text.contains("rate"), "got: {text}");
        assert!(text.contains("c_param"), "got: {text}");
        // §08/§12: HS³ `c` IS the FlatPPL rate (density exp(−c·x)), so rate = c
        // directly — never negated.
        assert!(
            text.contains("rate = c_param") && !text.contains("neg"),
            "rate should be the bare c, not neg(c), got: {text}"
        );
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
            emit_distribution(&mut b, &d, None, None, None).unwrap()
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
    fn uniform_with_domain_emits_interval_support() {
        let d = dist("uniform_dist", &[("x", serde_json::json!("x_obs"))]);
        let mut m = flatppl_core::Module::new();
        let node = {
            let mut b = Builder::new(&mut m);
            emit_distribution(&mut b, &d, Some((0.0, 10.0)), None, None).unwrap()
        };
        {
            let mut b = Builder::new(&mut m);
            b.bind("d", node);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("Uniform"), "got: {text}");
        // Support is the variate's declared interval, not a bare Uniform().
        assert!(text.contains("interval"), "got: {text}");
        assert!(text.contains("10"), "got: {text}");
        assert!(
            !text.contains("Uniform()"),
            "must not be a bare Uniform(), got: {text}"
        );
    }

    #[test]
    fn uniform_without_domain_errors() {
        let d = dist("uniform_dist", &[("x", serde_json::json!("x_obs"))]);
        let mut m = flatppl_core::Module::new();
        let mut b = Builder::new(&mut m);
        let result = emit_distribution(&mut b, &d, None, None, None);
        assert!(
            matches!(result, Err(Error::Unsupported(_))),
            "got: {result:?}"
        );
    }

    #[test]
    fn unknown_dist_type_errors() {
        let d = dist("no_such_dist", &[]);
        let mut m = flatppl_core::Module::new();
        let mut b = Builder::new(&mut m);
        let result = emit_distribution(&mut b, &d, None, None, None);
        assert!(matches!(result, Err(Error::UnknownDistType(_))));
    }

    #[test]
    fn histfactory_dist_unsupported() {
        let d = dist("histfactory_dist", &[]);
        let mut m = flatppl_core::Module::new();
        let mut b = Builder::new(&mut m);
        let result = emit_distribution(&mut b, &d, None, None, None);
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
            emit_distribution(&mut b, &d, None, None, None).unwrap()
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

    /// Coherence guard for [`reference_measure`] — the second hand-maintained
    /// `kind`-match parallel to the `dist_spec` dispatch table. A genuine pdf
    /// (a kind with a single scalar variate) that is added to the dispatch but
    /// forgotten here silently falls to [`RefMeasure::Other`] and is then
    /// wrongly rejected from valid shared-variate `product_dist` lowerings (§12).
    ///
    /// Eligibility is derived from the table itself: every kind whose
    /// `dist_spec::variate` is a scalar is a pointwise-density candidate and MUST
    /// classify to a concrete reference measure. (Companion to the black-box
    /// `every_recognized_dist_kind_has_a_dispatch_arm` test in tests/coherence.rs;
    /// it lives here because `reference_measure` is `pub(crate)`.)
    // ── generic_dist domain normalization (§12) ──────────────────────────────

    #[test]
    fn generic_dist_with_domain_emits_truncate_interval() {
        let d = dist(
            "generic_dist",
            &[("expression", serde_json::json!("1.0 + 0.1*abs(x)"))],
        );
        let mut m = flatppl_core::Module::new();
        let node = {
            let mut b = Builder::new(&mut m);
            emit_distribution(&mut b, &d, Some((-20.0, 20.0)), None, None).unwrap()
        };
        {
            let mut b = Builder::new(&mut m);
            b.bind("d", node);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("normalize"), "got: {text}");
        assert!(
            text.contains("truncate"),
            "normalize over domain requires truncate, got: {text}"
        );
        assert!(
            text.contains("interval"),
            "truncate requires interval, got: {text}"
        );
        assert!(text.contains("-20"), "expected lo bound, got: {text}");
        assert!(text.contains("20"), "expected hi bound, got: {text}");
        assert!(text.contains("weighted"), "got: {text}");
        assert!(text.contains("Lebesgue"), "got: {text}");
    }

    #[test]
    fn generic_dist_without_domain_emits_lebesgue_reals_no_truncate() {
        let d = dist(
            "generic_dist",
            &[("expression", serde_json::json!("1.0 + 0.1*abs(x)"))],
        );
        let mut m = flatppl_core::Module::new();
        let node = {
            let mut b = Builder::new(&mut m);
            emit_distribution(&mut b, &d, None, None, None).unwrap()
        };
        {
            let mut b = Builder::new(&mut m);
            b.bind("d", node);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("normalize"), "got: {text}");
        assert!(text.contains("weighted"), "got: {text}");
        assert!(text.contains("Lebesgue"), "got: {text}");
        assert!(
            !text.contains("truncate"),
            "fallback must not emit truncate, got: {text}"
        );
    }

    #[test]
    fn density_function_dist_with_domain_emits_truncate_interval() {
        let d = dist(
            "density_function_dist",
            &[("function", serde_json::json!("my_pdf"))],
        );
        let mut m = flatppl_core::Module::new();
        let node = {
            let mut b = Builder::new(&mut m);
            emit_distribution(&mut b, &d, Some((-5.0, 5.0)), None, None).unwrap()
        };
        {
            let mut b = Builder::new(&mut m);
            b.bind("d", node);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("normalize"), "got: {text}");
        assert!(
            text.contains("truncate"),
            "normalize over domain requires truncate, got: {text}"
        );
        assert!(
            text.contains("interval"),
            "truncate requires interval, got: {text}"
        );
        assert!(text.contains("weighted"), "got: {text}");
        assert!(text.contains("Lebesgue"), "got: {text}");
    }

    #[test]
    fn log_density_function_dist_with_domain_emits_truncate_interval() {
        let d = dist(
            "log_density_function_dist",
            &[("function", serde_json::json!("my_logpdf"))],
        );
        let mut m = flatppl_core::Module::new();
        let node = {
            let mut b = Builder::new(&mut m);
            emit_distribution(&mut b, &d, Some((0.0, 10.0)), None, None).unwrap()
        };
        {
            let mut b = Builder::new(&mut m);
            b.bind("d", node);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("normalize"), "got: {text}");
        assert!(
            text.contains("truncate"),
            "normalize over domain requires truncate, got: {text}"
        );
        assert!(
            text.contains("interval"),
            "truncate requires interval, got: {text}"
        );
        assert!(text.contains("logweighted"), "got: {text}");
        assert!(text.contains("Lebesgue"), "got: {text}");
    }

    #[test]
    fn conditional_gaussian_mean_emits_func_applied_to_axis() {
        use std::collections::{BTreeMap, BTreeSet};
        let mut m = flatppl_core::Module::new();
        let text = {
            let mut b = Builder::new(&mut m);
            let funcs: BTreeMap<&str, &str> = [("fy", "y")].into_iter().collect();
            let axes: BTreeSet<&str> = ["x", "y"].into_iter().collect();
            let ctx = CondCtx {
                funcs: &funcs,
                axes: &axes,
            };
            let d = dist(
                "gaussian_dist",
                &[
                    ("x", serde_json::json!("x")),
                    ("mean", serde_json::json!("fy")),
                    ("sigma", serde_json::json!("sigma")),
                ],
            );
            let node = emit_distribution(&mut b, &d, None, None, Some(&ctx)).unwrap();
            b.bind("g", node);
            flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal)
        };
        assert!(text.contains("Normal(mu = fy(y)"), "got: {text}");
    }

    #[test]
    fn every_scalar_variate_kind_has_a_concrete_reference_measure() {
        // The recognized scalar-variate (pdf) kinds. `polynomial_dist` is omitted
        // deliberately: it has a scalar `x` for variate-naming but lowers to a
        // `functionof`/normalize form, so it is NOT a same-measure density-product
        // factor and is correctly `Other` (it is never paired in a shared product).
        const SCALAR_PDF_KINDS: &[&str] = &[
            "gaussian_dist",
            "normal_dist",
            "poisson_dist",
            "exponential_dist",
            "lognormal_dist",
            "uniform_dist",
            "generalized_normal_dist",
            "crystalball_dist",
            "argus_dist",
        ];
        for kind in SCALAR_PDF_KINDS {
            // The list must stay honest: each entry really is a scalar-variate kind.
            assert!(
                matches!(dist_spec::variate(kind), Variate::Scalar(_)),
                "{kind} is listed as a scalar-variate pdf but dist_spec says otherwise — \
                 update SCALAR_PDF_KINDS or the table"
            );
            assert_ne!(
                reference_measure(kind),
                RefMeasure::Other,
                "{kind} is a scalar-variate pdf but reference_measure classifies it as `Other`; \
                 it will be wrongly rejected from shared-variate product_dist lowerings. \
                 Add it to the reference_measure match."
            );
        }
    }
}
