//! Document -> Module orchestration.
use crate::builder::Builder;
use crate::distribution::{VariateName, emit_distribution, needs_hepphys, variate_name};
use crate::error::{Error, Result};
use crate::expr;
use crate::model::{Document, Function, HistFactory, Modifier};
use crate::presets::{emit_domain, emit_parameter_point};
use flatppl_core::Module;
use flatppl_core::id::NodeId;
use std::collections::{BTreeSet, HashMap};

pub fn document_to_module(doc: &Document) -> Result<Module> {
    let mut m = Module::new();
    let dist_names: BTreeSet<&str> = doc.distributions.iter().map(|d| d.name.as_str()).collect();
    // Names defined in the `functions` block — these are deterministic bindings, not free params.
    let fn_names: BTreeSet<&str> = doc.functions.iter().map(|f| f.name.as_str()).collect();
    let mut declared: BTreeSet<String> = BTreeSet::new();

    // Names of native histfactory_dist distributions; these are assembled by the
    // pyhf-shared channel path, not emitted as ordinary distributions or wrapped
    // by the generic likelihood emitter.
    let histfactory_names: BTreeSet<&str> = doc
        .distributions
        .iter()
        .filter(|d| d.kind == "histfactory_dist")
        .map(|d| d.name.as_str())
        .collect();

    // 1. Free-parameter declarations: string-valued distribution fields (except `x`)
    //    that aren't themselves distributions. Set = natural domain default.
    {
        let mut b = Builder::new(&mut m);
        for d in &doc.distributions {
            if d.kind == "histfactory_dist" {
                continue; // params declared by the channel assembler
            }
            // Variate fields: skip them as free params (they are observed variables).
            // Different distribution kinds use different variate field names.
            let variate_fields: &[&str] = match d.kind.as_str() {
                "crystalball_dist" => &["m"],
                "argus_dist" => &["mass"],
                "multivariate_normal_dist" => &["x"], // x is an array of obs names, handled separately
                _ => &["x"],
            };
            for (field, v) in &d.extra {
                if variate_fields.contains(&field.as_str()) {
                    continue;
                }
                // For multivariate_normal_dist, mean and covariances are arrays —
                // walk the array to find string-valued parameter references.
                if d.kind == "multivariate_normal_dist" {
                    if field == "mean" {
                        if let Some(arr) = v.as_array() {
                            for elem in arr {
                                if let Some(name) = elem.as_str() {
                                    if !dist_names.contains(name) && !declared.contains(name) {
                                        let set = b.call_head(param_domain(&d.kind, "mean"));
                                        b.bind_set(name, set);
                                        declared.insert(name.to_string());
                                    }
                                }
                            }
                        }
                        continue;
                    }
                    if field == "covariances" {
                        // 2-D array — skip; covariance entries are typically literals
                        continue;
                    }
                }
                // For mixture_dist, summands is an array of dist-names (not free params),
                // and coefficients is an array of weights — walk coefficients for symbolic names.
                if d.kind == "mixture_dist" {
                    if field == "summands" {
                        // summand names are distributions; skip (dist_names handles this)
                        continue;
                    }
                    if field == "coefficients" {
                        if let Some(arr) = v.as_array() {
                            for elem in arr {
                                if let Some(name) = elem.as_str() {
                                    if !dist_names.contains(name) && !declared.contains(name) {
                                        // Mixture weights are in [0,1]; use unitinterval if available,
                                        // otherwise reals (HS3 domain declarations override).
                                        let set = b.call_head("reals");
                                        b.bind_set(name, set);
                                        declared.insert(name.to_string());
                                    }
                                }
                            }
                        }
                        continue;
                    }
                    if field == "extended" {
                        // boolean flag, not a parameter
                        continue;
                    }
                }
                // Skip expression-based distribution special fields.
                if matches!(
                    d.kind.as_str(),
                    "generic_dist" | "density_function_dist" | "log_density_function_dist"
                ) {
                    // `expression` is a formula string, not a param name.
                    // `function` is a reference to the functions block, not a free param.
                    if field == "expression" || field == "function" {
                        continue;
                    }
                }
                // Poisson-process distributions: skip structural/non-param fields and
                // handle array-valued `coefficients` / `expected` inline.
                if matches!(
                    d.kind.as_str(),
                    "rate_extended_dist"
                        | "rate_density_dist"
                        | "bincounts_extended_dist"
                        | "bincounts_density_dist"
                ) {
                    // `distribution` is a dist-name ref, `function` a function ref, `axes` structural.
                    if field == "distribution" || field == "function" || field == "axes" {
                        continue;
                    }
                }
                if d.kind == "polynomial_dist" {
                    if field == "coefficients" {
                        // Walk coefficient array for symbolic names.
                        if let Some(arr) = v.as_array() {
                            for elem in arr {
                                if let Some(name) = elem.as_str() {
                                    if !dist_names.contains(name)
                                        && !fn_names.contains(name)
                                        && !declared.contains(name)
                                    {
                                        let set = b.call_head("reals");
                                        b.bind_set(name, set);
                                        declared.insert(name.to_string());
                                    }
                                }
                            }
                        }
                        continue;
                    }
                    // `x` is the variate, skip.
                    if field == "x" {
                        continue;
                    }
                }
                if d.kind == "barlow_beeston_lite_poisson_constraint_dist" {
                    if field == "x" {
                        // x is an array of observed variable names, not free params.
                        continue;
                    }
                    if field == "expected" {
                        // Walk expected array for symbolic names.
                        if let Some(arr) = v.as_array() {
                            for elem in arr {
                                if let Some(name) = elem.as_str() {
                                    if !dist_names.contains(name)
                                        && !fn_names.contains(name)
                                        && !declared.contains(name)
                                    {
                                        let set = b.call_head(param_domain(&d.kind, "expected"));
                                        b.bind_set(name, set);
                                        declared.insert(name.to_string());
                                    }
                                }
                            }
                        }
                        continue;
                    }
                }
                if let Some(name) = v.as_str() {
                    if dist_names.contains(name)
                        || fn_names.contains(name)
                        || declared.contains(name)
                    {
                        continue;
                    }
                    let set = b.call_head(param_domain(&d.kind, field)); // bare set constant
                    b.bind_set(name, set);
                    declared.insert(name.to_string());
                }
            }
        }
    }
    // 1b. Functions block → deterministic bindings.
    //     `product`, `sum`, and `generic_function` entries become plain `name = <expr>` bindings.
    {
        let mut b = Builder::new(&mut m);
        for f in &doc.functions {
            emit_function(&mut b, f)?;
        }
    }
    // 2. Distributions -> bindings (relabel with the variate).
    //    Also emit the hepphys module binding once if any native distribution
    //    requires it (crystalball_dist, argus_dist) and no histfactory path did it.
    {
        let mut b = Builder::new(&mut m);
        let needs_hp = doc
            .distributions
            .iter()
            .any(|d| d.kind != "histfactory_dist" && needs_hepphys(&d.kind));
        // Only bind hepphys here if no histfactory_dist is present (the histfactory path
        // binds it in step 2b).
        let has_histfactory = doc
            .distributions
            .iter()
            .any(|d| d.kind == "histfactory_dist");
        if needs_hp && !has_histfactory {
            let name_arg = b.str_lit("particle-physics");
            let ver_arg = b.str_lit("0.1");
            let module_call = b.call("standard_module", &[name_arg, ver_arg]);
            b.bind("hepphys", module_call);
        }
        for d in &doc.distributions {
            if d.kind == "histfactory_dist" {
                continue;
            }
            let dist = emit_distribution(&mut b, d)?;
            let bound = match variate_name(d) {
                Some(VariateName::Single(v)) => {
                    let label = b.str_lit(&v);
                    let labels = b.array(&[label]);
                    b.call("relabel", &[dist, labels])
                }
                Some(VariateName::Multiple(names)) => {
                    let label_nodes: Vec<_> = names.iter().map(|n| b.str_lit(n)).collect();
                    let labels = b.array(&label_nodes);
                    b.call("relabel", &[dist, labels])
                }
                None => dist,
            };
            if let Some(line) = dist_doc_line(&d.kind) {
                b.bind_doc(&d.name, bound, &[line]);
            } else {
                b.bind(&d.name, bound);
            }
        }
    }
    // 2b. Native histfactory_dist distributions -> channel assembly (pyhf-shared).
    {
        let mut b = Builder::new(&mut m);
        let mut bound_hepphys = false;
        for d in &doc.distributions {
            if d.kind != "histfactory_dist" {
                continue;
            }
            // Deserialize the histfactory body from the flattened `extra` map.
            let hf: HistFactory = serde_json::from_value(serde_json::to_value(&d.extra)?)?;

            // Observed bin contents: pair this distribution with its binned datum
            // through the likelihood that references it.
            let obs_vals = find_histfactory_observed(doc, &d.name)
                .ok_or_else(|| Error::NoObservation(d.name.clone()))?;

            // Bind hepphys once (the channel assembler emits hepphys.* calls).
            if !bound_hepphys {
                let name_arg = b.str_lit("particle-physics");
                let ver_arg = b.str_lit("0.1");
                let module_call = b.call("standard_module", &[name_arg, ver_arg]);
                b.bind("hepphys", module_call);
                bound_hepphys = true;
            }

            // Build owned per-sample modifier vectors, injecting each sample's
            // `errors` into its staterror modifier's `data` so the shared
            // assembler (which reads staterror errors from `modifier.data`) sees
            // them.  HS3 carries errors on the sample, not the modifier.
            let sample_mods: Vec<Vec<Modifier>> = hf
                .samples
                .iter()
                .map(|s| {
                    let errors = s.data.errors().to_vec();
                    s.modifiers
                        .iter()
                        .map(|mo| {
                            let mut mo = mo.clone();
                            if mo.kind == "staterror" && mo.data.is_none() {
                                mo.data = Some(serde_json::json!(errors));
                            }
                            mo
                        })
                        .collect()
                })
                .collect();

            let samples: Vec<(&str, &[f64], &[Modifier])> = hf
                .samples
                .iter()
                .zip(sample_mods.iter())
                .map(|(s, mods)| (s.name.as_str(), s.data.contents(), mods.as_slice()))
                .collect();

            // Observed array node.
            let obs_elems: Vec<_> = obs_vals.iter().map(|v| b.lit_real(*v)).collect();
            let observed = b.array(&obs_elems);

            crate::pyhf::assemble_channel(&mut b, &d.name, &samples, observed, None)?;
        }
    }
    // 3. Presets.
    {
        let mut b = Builder::new(&mut m);
        for d in &doc.domains {
            emit_domain(&mut b, d);
        }
        for pp in &doc.parameter_points {
            emit_parameter_point(&mut b, pp);
        }
    }
    // 4. Likelihoods.  Build a map from datum name -> flattened values so that
    //    the likelihood emitter can inline unbinned observations.
    {
        let data_map: HashMap<String, Vec<f64>> = doc
            .data
            .iter()
            .filter(|d| d.kind == "unbinned")
            .map(|d| {
                let vals: Vec<f64> = d
                    .entries
                    .iter()
                    .filter_map(|e| e.first())
                    .copied()
                    .collect();
                (d.name.clone(), vals)
            })
            .collect();
        let mut b = Builder::new(&mut m);
        for lk in &doc.likelihoods {
            // Skip likelihoods whose distributions are all native histfactory_dist;
            // those are assembled into `L_<channel>` by the channel path above.
            if !lk.distributions.is_empty()
                && lk
                    .distributions
                    .iter()
                    .all(|n| histfactory_names.contains(n.as_str()))
            {
                continue;
            }
            crate::likelihood::emit_likelihood(&mut b, lk, &data_map);
        }
    }
    Ok(m)
}

/// Resolve the observed bin contents for a histfactory distribution `dist_name`
/// by following the likelihood that lists it (`distributions[i]` ↔ `data[i]`)
/// to a binned datum, then reading that datum's `contents`.
fn find_histfactory_observed(doc: &Document, dist_name: &str) -> Option<Vec<f64>> {
    for lk in &doc.likelihoods {
        if let Some(idx) = lk.distributions.iter().position(|n| n == dist_name) {
            if let Some(serde_json::Value::String(data_name)) = lk.data.get(idx) {
                if let Some(datum) = doc
                    .data
                    .iter()
                    .find(|d| &d.name == data_name && d.kind == "binned")
                {
                    if let Some(contents) = &datum.contents {
                        return Some(contents.clone());
                    }
                }
            }
        }
    }
    None
}

fn param_domain(dist_kind: &str, field: &str) -> &'static str {
    match (dist_kind, field) {
        // Scale-like params always positive
        (_, "sigma") | (_, "sigma_L") | (_, "sigma_R") => "posreals",
        (_, "n") | (_, "n_L") | (_, "n_R") => "posreals",
        (_, "beta") => "posreals",
        // ARGUS slope is typically negative; only power is strictly positive.
        ("argus_dist", "slope") => "reals",
        (_, "power") => "posreals",
        // alpha is a scale only for generalized_normal_dist; for crystalball it is a tail cut (reals)
        ("generalized_normal_dist", "alpha") => "posreals",
        // Poisson/exponential rate-like params
        ("poisson_dist", "mean") | ("exponential_dist", "c") => "posreals",
        // Poisson-process rate (expected count ≥ 0)
        (_, "rate") => "posreals",
        // Barlow-Beeston expected counts are ≥ 0
        ("barlow_beeston_lite_poisson_constraint_dist", "expected") => "posreals",
        _ => "reals",
    }
}

/// Return the doc-comment line for a non-1:1 distribution lowering, or `None`
/// for 1:1 mappings that need no annotation.
fn dist_doc_line(kind: &str) -> Option<&'static str> {
    // Content stored WITHOUT leading `% ` — the printer prepends `% ` when
    // rendering a single-line Markdown doc-comment (see flatppl_syntax::print_doc).
    match kind {
        "product_dist" => Some("HS3 product_dist → joint over factor distributions"),
        "mixture_dist" => {
            Some("HS3 mixture_dist → normalize(superpose(weighted(coeff, summand)…))")
        }
        "generic_dist" => {
            Some("HS3 generic_dist → normalize(weighted(functionof(<expr>), Lebesgue(reals)))")
        }
        "density_function_dist" => {
            Some("HS3 density_function_dist → normalize(weighted(<function>, Lebesgue(reals)))")
        }
        "log_density_function_dist" => Some(
            "HS3 log_density_function_dist → normalize(logweighted(<function>, Lebesgue(reals)))",
        ),
        "rate_extended_dist" => {
            Some("HS3 rate_extended_dist → PoissonProcess(weighted(rate, shape))")
        }
        "rate_density_dist" => {
            Some("HS3 rate_density_dist → PoissonProcess(weighted(<function>, Lebesgue(reals)))")
        }
        "bincounts_extended_dist" => {
            Some("HS3 bincounts_extended_dist → BinnedPoissonProcess(bins, weighted(rate, shape))")
        }
        "bincounts_density_dist" => Some(
            "HS3 bincounts_density_dist → BinnedPoissonProcess(bins, weighted(<function>, Lebesgue(reals)))",
        ),
        "polynomial_dist" => Some(
            "HS3 polynomial_dist → normalize(weighted(functionof(polynomial(coefficients)), Lebesgue(reals)))",
        ),
        "barlow_beeston_lite_poisson_constraint_dist" => Some(
            "HS3 barlow_beeston_lite_poisson_constraint_dist → per-bin broadcast(Poisson, expected)",
        ),
        // 1:1 mappings — no annotation needed.
        _ => None,
    }
}

/// Emit a `functions` block entry as a deterministic FlatPPL binding.
///
/// - `product`: `name = mul(f1, mul(f2, ...))` (fold).
/// - `sum`:     `name = add(s1, add(s2, ...))` (fold).
/// - `generic_function`: `name = <parsed expression>` (the expression may
///   reference other bindings via `self_ref`; it is *not* wrapped in a lambda
///   here — the expression is a deterministic scalar/function-valued formula).
fn emit_function(b: &mut Builder, f: &Function) -> Result<()> {
    match f.kind.as_str() {
        "product" => {
            let factors = f
                .extra
                .get("factors")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    Error::Unsupported(format!("product function `{}` missing `factors`", f.name))
                })?;
            if factors.is_empty() {
                return Err(Error::Unsupported(format!(
                    "product function `{}` has no factors",
                    f.name
                )));
            }
            let nodes: Vec<_> = factors.iter().map(|v| fn_factor_node(b, v)).collect();
            let folded = nodes
                .into_iter()
                .reduce(|acc, x| b.call("mul", &[acc, x]))
                .unwrap();
            b.bind_doc(&f.name, folded, &["HS3 product function → fold of mul"]);
        }
        "sum" => {
            let summands = f
                .extra
                .get("summands")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    Error::Unsupported(format!("sum function `{}` missing `summands`", f.name))
                })?;
            if summands.is_empty() {
                return Err(Error::Unsupported(format!(
                    "sum function `{}` has no summands",
                    f.name
                )));
            }
            let nodes: Vec<_> = summands.iter().map(|v| fn_factor_node(b, v)).collect();
            let folded = nodes
                .into_iter()
                .reduce(|acc, x| b.call("add", &[acc, x]))
                .unwrap();
            b.bind_doc(&f.name, folded, &["HS3 sum function → fold of add"]);
        }
        "generic_function" => {
            let expression = f
                .extra
                .get("expression")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    Error::Unsupported(format!(
                        "generic_function `{}` missing `expression`",
                        f.name
                    ))
                })?;
            // Determine the observable variable name.  HS3 generic_function uses
            // a `variables` (or `x`) field to name the input(s); we default to `"x"`.
            let obs_name = f
                .extra
                .get("variables")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|v| v.as_str())
                .or_else(|| f.extra.get("x").and_then(|v| v.as_str()))
                .unwrap_or("x");
            // Emit as a lambda: `obs_name -> <expr>`, making it a callable weight.
            let fn_node = expr::parse_expr_as_fn(b, expression, obs_name)?;
            b.bind_doc(
                &f.name,
                fn_node,
                &["HS3 generic_function → lowered expression"],
            );
        }
        other => {
            return Err(Error::Unsupported(format!(
                "unknown function type `{other}` for function `{}`",
                f.name
            )));
        }
    }
    Ok(())
}

/// Map a `functions` factor/summand value to a FlatPPL node.
/// Numbers become `lit_real`; strings become `self_ref`.
fn fn_factor_node(b: &mut Builder, v: &serde_json::Value) -> NodeId {
    match v {
        serde_json::Value::Number(n) => b.lit_real(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => b.self_ref(s),
        _ => b.lit_real(0.0),
    }
}

#[cfg(test)]
mod tests {
    use flatppl_syntax::{Syntax, print_with};
    const MINIMAL: &str = r#"{
      "distributions": [
        {"name": "mass", "type": "gaussian_dist",
         "mean": "mu_param", "sigma": "sigma_param", "x": "mass_obs"}
      ],
      "parameter_points": [
        {"name": "nominal", "entries": [
          {"name": "mu_param", "value": 5.28},
          {"name": "sigma_param", "value": 0.003}
        ]}
      ]
    }"#;
    #[test]
    fn slice1_minimal_gaussian_matches_spec() {
        let m = crate::read(MINIMAL).unwrap();
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("relabel"), "got:\n{text}");
        assert!(text.contains("Normal"), "got:\n{text}");
        assert!(text.contains("mass_obs"), "got:\n{text}");
        assert!(
            text.contains("mu_param") && text.contains("sigma_param"),
            "got:\n{text}"
        );
        assert!(
            text.contains("elementof(reals)"),
            "expected bare set constant, got:\n{text}"
        );
        assert!(
            text.contains("elementof(posreals)"),
            "expected bare posreals constant, got:\n{text}"
        );
        assert!(
            !text.contains("reals()"),
            "must not emit nullary call, got:\n{text}"
        );
        assert!(
            text.contains("record") && text.contains("5.28"),
            "got:\n{text}"
        );
    }
}
