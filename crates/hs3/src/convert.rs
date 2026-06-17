//! Document -> Module orchestration.
//!
//! This is a model-only importer: it lowers an HS3 document's model blocks
//! (`distributions`, `functions`, `domains`, `parameter_points`, `data`,
//! `likelihoods`) to a FlatPPL module. An HS3 `analyses` / bayesupdate block
//! (inference configuration — POI, priors, calculator choice; §12:147) is
//! intentionally NOT imported: it is layered on top of the model rather than
//! part of it. Such a block is silently passed over rather than rejected, since
//! real HS3 files routinely bundle one alongside the model.
use crate::builder::Builder;
use crate::dist_spec;
use crate::distribution::{
    RefMeasure, VariateName, emit_distribution, emit_product, field_node, needs_hepphys,
    product_factors, product_shared_variate, reference_measure, variate_name,
};
use crate::error::{Error, Result};
use crate::expr;
use crate::model::{Distribution, Document, Function, HistFactory, Modifier};
use crate::presets::{emit_domain, emit_parameter_point};
use flatppl_core::Module;
use std::collections::{BTreeMap, BTreeSet};

pub fn document_to_module(doc: &Document) -> Result<Module> {
    reject_unsupported(doc)?;

    let mut m = Module::new();
    let dist_names: BTreeSet<&str> = doc.distributions.iter().map(|d| d.name.as_str()).collect();
    // Names defined in the `functions` block — these are deterministic bindings, not free params.
    let fn_names: BTreeSet<&str> = doc.functions.iter().map(|f| f.name.as_str()).collect();

    // Names of native histfactory_dist distributions; these are assembled by the
    // pyhf-shared channel path, not emitted as ordinary distributions or wrapped
    // by the generic likelihood emitter.
    let histfactory_names: BTreeSet<&str> = doc
        .distributions
        .iter()
        .filter(|d| d.kind == "histfactory_dist")
        .map(|d| d.name.as_str())
        .collect();

    // 1.  Free-parameter declarations.
    declare_free_params(&mut m, doc, &dist_names, &fn_names)?;
    // 1b. Functions block → deterministic bindings.
    emit_functions(&mut m, doc)?;
    // 2.  Distributions → bindings (relabel with the variate).
    emit_distributions(&mut m, doc)?;
    // 2b. Native histfactory_dist distributions → channel assembly (pyhf-shared).
    emit_histfactory_channels(&mut m, doc)?;
    // 3.  Presets (domains, parameter_points).
    emit_presets(&mut m, doc);
    // 4.  Likelihoods.
    emit_likelihoods(&mut m, doc, &histfactory_names)?;

    Ok(m)
}

/// Reject documents carrying constructs outside this importer's supported
/// subset before any lowering begins.
///
/// This guards duplicate distribution or function binding names (which would
/// silently shadow one another in the emitted module).
///
/// NOTE: a document's HS3 `analyses` / bayesupdate block (§12:147) is
/// intentionally NOT imported and NOT an error. `analyses` is inference
/// configuration (POI, priors, calculator choice) layered on top of the model,
/// not part of the model itself; this is a model-only importer, so an
/// `analyses` block — which real HS3 files routinely bundle alongside the model
/// (see the paper §A.1–A.3 examples) — is silently passed over. The
/// `Document::analyses` field is still parsed so its presence is observable, but
/// the rest of the document lowers normally regardless of it.
fn reject_unsupported(doc: &Document) -> Result<()> {
    // Distribution and function entries both become top-level bindings; a name
    // collision (within or across the two blocks) would silently drop a binding.
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for name in doc
        .distributions
        .iter()
        .map(|d| d.name.as_str())
        .chain(doc.functions.iter().map(|f| f.name.as_str()))
    {
        if !seen.insert(name) {
            return Err(Error::Unsupported(format!(
                "duplicate binding name `{name}` in distributions/functions"
            )));
        }
    }
    Ok(())
}

/// Build a lookup from observed-variable name → `(min, max)` over every axis of
/// every `domains` entry. Used to resolve a distribution variate's support
/// (e.g. for `uniform_dist`).
///
/// The same observable may appear in more than one `domains` entry; that is fine
/// as long as the bounds agree. Two axes naming the same observable with
/// *conflicting* `(min, max)` are contradictory — silently keeping the last
/// would pick an arbitrary support — so this is rejected.
fn domain_bounds(doc: &Document) -> Result<BTreeMap<&str, (f64, f64)>> {
    let mut map: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
    for d in &doc.domains {
        for ax in &d.axes {
            let bounds = (ax.min, ax.max);
            match map.insert(ax.name.as_str(), bounds) {
                Some(prev) if prev != bounds => {
                    return Err(Error::Unsupported(format!(
                        "observable `{}` has conflicting domain bounds: {:?} vs {:?}",
                        ax.name, prev, bounds
                    )));
                }
                _ => {}
            }
        }
    }
    Ok(map)
}

/// Walk an array-valued distribution field for string-valued parameter
/// references and declare each previously-undeclared one as a free parameter
/// `name = elementof(<set_name>)`. Names that already denote a distribution, a
/// function, or an already-declared parameter are skipped. Non-string array
/// entries (numeric literals) are left alone.
fn declare_array_params(
    b: &mut Builder,
    arr: &[serde_json::Value],
    set_name: &str,
    dist_names: &BTreeSet<&str>,
    fn_names: &BTreeSet<&str>,
    declared: &mut BTreeSet<String>,
) {
    for elem in arr {
        if let Some(name) = elem.as_str() {
            if dist_names.contains(name) || fn_names.contains(name) || declared.contains(name) {
                continue;
            }
            let set = b.call_head(set_name);
            b.bind_set(name, set);
            declared.insert(name.to_string());
        }
    }
}

/// Step 1: declare free parameters — string-valued distribution fields (other
/// than the variate) that name neither another distribution nor a function.
/// Each becomes `name = elementof(<natural-domain set>)`.
fn declare_free_params(
    m: &mut Module,
    doc: &Document,
    dist_names: &BTreeSet<&str>,
    fn_names: &BTreeSet<&str>,
) -> Result<()> {
    let mut declared: BTreeSet<String> = BTreeSet::new();
    {
        let mut b = Builder::new(m);
        for d in &doc.distributions {
            if d.kind == "histfactory_dist" {
                continue; // params declared by the channel assembler
            }
            // The variate field names an observed variable, not a free param.
            let variate_field = dist_spec::variate_field(&d.kind);
            for (field, v) in &d.extra {
                if field == variate_field {
                    continue;
                }
                // For multivariate_normal_dist, mean and covariances are arrays —
                // walk the array to find string-valued parameter references.
                if d.kind == "multivariate_normal_dist" {
                    if field == "mean" {
                        if let Some(arr) = v.as_array() {
                            let set = dist_spec::param_domain(&d.kind, "mean");
                            declare_array_params(
                                &mut b,
                                arr,
                                set,
                                dist_names,
                                fn_names,
                                &mut declared,
                            );
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
                            // Mixture weights are in [0,1]; use reals as the default
                            // (HS3 domain declarations override).
                            declare_array_params(
                                &mut b,
                                arr,
                                "reals",
                                dist_names,
                                fn_names,
                                &mut declared,
                            );
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
                            declare_array_params(
                                &mut b,
                                arr,
                                "reals",
                                dist_names,
                                fn_names,
                                &mut declared,
                            );
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
                            let set = dist_spec::param_domain(&d.kind, "expected");
                            declare_array_params(
                                &mut b,
                                arr,
                                set,
                                dist_names,
                                fn_names,
                                &mut declared,
                            );
                        }
                        continue;
                    }
                }
                // A string-valued field that is NOT a recognized parameter of
                // this kind is an unknown construct, not a free parameter — only
                // promote allowlisted fields (see dist_spec::is_known_field).
                if let Some(name) = v.as_str() {
                    if !dist_spec::is_known_field(&d.kind, field) {
                        return Err(Error::Unsupported(format!(
                            "distribution `{}` ({}) has unrecognized field `{field}`",
                            d.name, d.kind
                        )));
                    }
                    if dist_names.contains(name)
                        || fn_names.contains(name)
                        || declared.contains(name)
                    {
                        continue;
                    }
                    let set = b.call_head(dist_spec::param_domain(&d.kind, field)); // bare set constant
                    b.bind_set(name, set);
                    declared.insert(name.to_string());
                }
            }
        }
        // Free parameters referenced ONLY inside generic `expression` strings are
        // never seen by the field walk above (it explicitly skips the formula
        // fields). Identifiers in those expressions lower to `self_ref` nodes
        // resolved at module level, so each must have a module binding or the
        // emitted FlatPPL has an unresolved reference. Declare any such
        // identifier here.
        declare_generic_expr_params(&mut b, doc, dist_names, fn_names, &mut declared)?;
    }
    Ok(())
}

/// Declare free parameters that appear ONLY inside generic `expression` strings
/// (`generic_function`, `generic_dist`, and — defensively — the inline
/// `expression` of `density_function_dist`/`log_density_function_dist`).
///
/// An identifier is declared `name = elementof(<set>)` when it:
///   - is listed in some `parameter_points` entry (i.e. it is a real model
///     parameter, not a typo or an inlined math symbol),
///   - is not already declared,
///   - does not name a distribution, a function, or an observable/variate, and
///   - is not the generic lambda's bound variable (the observable name).
///
/// The set is `interval(lo, hi)` when the name has a `domains` axis, else
/// `reals`. Discovery is order-deterministic (distributions then functions, each
/// in document order; identifiers in first-occurrence order).
fn declare_generic_expr_params(
    b: &mut Builder,
    doc: &Document,
    dist_names: &BTreeSet<&str>,
    fn_names: &BTreeSet<&str>,
    declared: &mut BTreeSet<String>,
) -> Result<()> {
    let bounds = domain_bounds(doc)?;

    // Names that denote observables (a distribution's variate), never free params.
    let mut observables: BTreeSet<String> = BTreeSet::new();
    for d in &doc.distributions {
        match variate_name(d) {
            Some(VariateName::Single(v)) => {
                observables.insert(v);
            }
            Some(VariateName::Multiple(ns)) => observables.extend(ns),
            None => {}
        }
    }

    // Names declared in some parameter_points entry — the authoritative list of
    // real model parameters. An expression identifier not in here is either an
    // observable (handled above) or out of scope to declare.
    let param_point_names: BTreeSet<&str> = doc
        .parameter_points
        .iter()
        .flat_map(|pp| pp.entries.iter().map(|e| e.name.as_str()))
        .collect();

    // (expression, bound-variable name) pairs to scan, in deterministic order.
    let mut sources: Vec<(&str, &str)> = Vec::new();
    // generic_dist / density_function_dist / log_density_function_dist inline
    // expressions (the latter two normally carry only a `function` ref, but an
    // inline `expression`, if present, is scanned too).
    for d in &doc.distributions {
        if matches!(
            d.kind.as_str(),
            "generic_dist" | "density_function_dist" | "log_density_function_dist"
        ) {
            if let Some(expr) = d.extra.get("expression").and_then(|v| v.as_str()) {
                // generic_dist lowers over the hardcoded observable `x`.
                sources.push((expr, "x"));
            }
        }
    }
    // generic_function expressions, over their declared bound variable.
    for f in &doc.functions {
        if f.kind == "generic_function" {
            if let Some(expr) = f.extra.get("expression").and_then(|v| v.as_str()) {
                let obs_name = f
                    .extra
                    .get("variables")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|v| v.as_str())
                    .or_else(|| f.extra.get("x").and_then(|v| v.as_str()))
                    .unwrap_or("x");
                sources.push((expr, obs_name));
            }
        }
    }

    for (expr, bound_var) in sources {
        for name in expr::free_identifiers(expr) {
            if name == bound_var
                || dist_names.contains(name.as_str())
                || fn_names.contains(name.as_str())
                || observables.contains(&name)
                || declared.contains(&name)
            {
                continue;
            }
            // Only declare names the model actually lists as parameters.
            if !param_point_names.contains(name.as_str()) {
                continue;
            }
            let set = match bounds.get(name.as_str()) {
                Some(&(lo, hi)) => {
                    let lo = b.lit_real(lo);
                    let hi = b.lit_real(hi);
                    b.call("interval", &[lo, hi])
                }
                None => b.call_head("reals"),
            };
            b.bind_set(&name, set);
            declared.insert(name);
        }
    }
    Ok(())
}

/// Step 1b: lower each `functions` block entry to a deterministic binding.
fn emit_functions(m: &mut Module, doc: &Document) -> Result<()> {
    let mut b = Builder::new(m);
    for f in &doc.functions {
        emit_function(&mut b, f)?;
    }
    Ok(())
}

/// Step 2: emit each non-histfactory distribution as a binding, wrapping with a
/// `relabel` over its variate. Binds the `hepphys` standard module once up front
/// if any distribution needs it and no histfactory channel path will bind it.
fn emit_distributions(m: &mut Module, doc: &Document) -> Result<()> {
    let domains = domain_bounds(doc)?;
    {
        let mut b = Builder::new(m);
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
            crate::pyhf::emit_standard_module(&mut b);
        }
        // Resolve each distribution's variate once, so product_dist can classify
        // its factors (shared variate → density product, else independent joint).
        let dist_by_name: std::collections::BTreeMap<&str, &Distribution> = doc
            .distributions
            .iter()
            .map(|d| (d.name.as_str(), d))
            .collect();
        // Factors of a shared-variate product_dist are emitted as SCALAR measures
        // (no `relabel`): the pointwise density product is over the scalar
        // observable, and the §12 lowering scores it as `iid(prod, N)` over a bare
        // observation vector — a record-keyed (relabelled) factor cannot be
        // consumed by `iid` (which threads a flat value). The shared variate's
        // identity is still carried by the product's own classification.
        let shared_product_factors: std::collections::BTreeSet<&str> = doc
            .distributions
            .iter()
            .filter(|d| d.kind == "product_dist")
            .flat_map(|d| {
                let factors = product_factors(d);
                let fv: Vec<Option<VariateName>> = factors
                    .iter()
                    .map(|f| dist_by_name.get(f.as_str()).and_then(|fd| variate_name(fd)))
                    .collect();
                if product_shared_variate(&fv) {
                    factors
                } else {
                    Vec::new()
                }
            })
            .filter_map(|f| dist_by_name.get_key_value(f.as_str()).map(|(k, _)| *k))
            .collect();
        for d in &doc.distributions {
            if d.kind == "histfactory_dist" {
                continue;
            }
            // product_dist is composite: its form depends on the factors' variates.
            if d.kind == "product_dist" {
                // The factor list is immutable; build it once and reuse it for the
                // variate map, the measure map, and the emit (threaded below).
                let factors = product_factors(d);
                let factor_variates: Vec<Option<VariateName>> = factors
                    .iter()
                    .map(|f| dist_by_name.get(f.as_str()).and_then(|fd| variate_name(fd)))
                    .collect();
                let shared = product_shared_variate(&factor_variates);
                // A shared-observable product is a pointwise density product
                // (§12), well-defined only when all factors share one reference
                // measure. Reject mixed measures rather than emit a wrong one.
                if shared {
                    let measures: Vec<RefMeasure> = factors
                        .iter()
                        .map(|f| {
                            dist_by_name
                                .get(f.as_str())
                                .map_or(RefMeasure::Other, |fd| reference_measure(&fd.kind))
                        })
                        .collect();
                    let base = measures[0];
                    if base == RefMeasure::Other || measures.iter().any(|m| *m != base) {
                        return Err(Error::Unsupported(format!(
                            "product_dist `{}` multiplies factors over the same observable, but \
                             they do not share a known reference measure — a pointwise density \
                             product is undefined across mixed measures (§12)",
                            d.name
                        )));
                    }
                }
                let node = emit_product(&mut b, &factors, &factor_variates)?;
                let doc_line = if shared {
                    "HS3 product_dist (shared variate) → normalize(logweighted …): pointwise density product"
                } else {
                    "HS3 product_dist → joint over factor distributions"
                };
                b.bind_doc(&d.name, node, &[doc_line]);
                continue;
            }
            // Resolve the variate's declared domain (needed for uniform_dist).
            let domain = match variate_name(d) {
                Some(VariateName::Single(ref v)) => domains.get(v.as_str()).copied(),
                _ => None,
            };
            let dist = emit_distribution(&mut b, d, domain)?;
            // A shared-variate product factor stays scalar (see above).
            let bound = if shared_product_factors.contains(d.name.as_str()) {
                dist
            } else {
                match variate_name(d) {
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
                }
            };
            if let Some(line) = dist_spec::doc_line(&d.kind) {
                b.bind_doc(&d.name, bound, &[line]);
            } else {
                b.bind(&d.name, bound);
            }
        }
    }
    Ok(())
}

/// Step 2b: assemble each native `histfactory_dist` into a channel likelihood
/// via the pyhf-shared assembler. Binds `hepphys` once if any channel is present.
fn emit_histfactory_channels(m: &mut Module, doc: &Document) -> Result<()> {
    let mut b = Builder::new(m);
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

        // A `lumi` modifier needs a luminosity constraint (a Normal aux with a
        // sigma from the measurement's lumi-config). The native HS3 Document
        // carries no such config, so the only honest options are to fabricate a
        // constraint (silently wrong) or reject. Match the pyhf path and reject,
        // rather than passing `lumi: None` — which would emit `... .* lumi` with
        // NO constraint, a silently weaker model.
        if hf
            .samples
            .iter()
            .any(|s| s.modifiers.iter().any(|mo| mo.kind == "lumi"))
        {
            return Err(Error::Unsupported(format!(
                "channel `{}`: native histfactory lumi modifier requires a measurement \
                 lumi-config (sigma); not supported on the native HS3 path",
                d.name
            )));
        }

        // Bind hepphys once (the channel assembler emits hepphys.* calls).
        if !bound_hepphys {
            crate::pyhf::emit_standard_module(&mut b);
            bound_hepphys = true;
        }

        // Build owned per-sample modifier vectors, injecting each sample's
        // `errors` into its staterror modifier's `data` so the shared assembler
        // (which reads staterror errors from `modifier.data`) sees them. HS3
        // carries errors on the sample, not the modifier.
        let sample_mods: Vec<Vec<Modifier>> = hf
            .samples
            .iter()
            .map(|s| {
                let errors = s.data.errors().to_vec();
                s.modifiers
                    .iter()
                    .map(|mo| {
                        let mut mo = mo.clone();
                        if mo.kind == "shapesys" {
                            // HS3 `shapesys` `vals` are RELATIVE per-bin
                            // uncertainties (RooFit / HS3 convention), unlike
                            // pyhf's absolute `data`. Scale by this sample's
                            // nominal to absolute σ so the shared channel
                            // assembler's τ = (nominal/σ)² yields RooFit's
                            // τ = 1/vals². (The pyhf path passes absolute vals
                            // straight through and is untouched.)
                            let nominal = s.data.contents();
                            let vals = mo
                                .data
                                .as_ref()
                                .map(|v| v.get("vals").unwrap_or(v))
                                .and_then(serde_json::Value::as_array);
                            if let Some(vals) = vals {
                                let abs: Vec<f64> = vals
                                    .iter()
                                    .zip(nominal)
                                    .filter_map(|(rel, nom)| rel.as_f64().map(|r| r * nom))
                                    .collect();
                                mo.data = Some(serde_json::json!(abs));
                            }
                        }
                        if mo.kind == "staterror" {
                            // Normalize the per-bin uncertainty array the channel
                            // assembler reads from `modifier.data` to a bare array,
                            // accepting any source: a bare array (pyhf), a
                            // `{"uncertainties": [...]}` object (modern HS3 / pyhs3),
                            // or the sample's `data.errors` (spec form, no modifier
                            // data).
                            let arr: Option<Vec<f64>> = match &mo.data {
                                Some(v) if v.is_array() => None,
                                Some(v) => {
                                    v.get("uncertainties").and_then(|u| u.as_array()).map(|a| {
                                        a.iter().filter_map(serde_json::Value::as_f64).collect()
                                    })
                                }
                                None => Some(errors.clone()),
                            };
                            if let Some(a) = arr {
                                mo.data = Some(serde_json::json!(a));
                            }
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

        crate::pyhf::assemble_channel(&mut b, &d.name, &samples, observed, obs_vals.len(), None)?;
    }
    Ok(())
}

/// Step 3: emit `domains` and `parameter_points` presets.
fn emit_presets(m: &mut Module, doc: &Document) {
    let mut b = Builder::new(m);
    for d in &doc.domains {
        emit_domain(&mut b, d);
    }
    for pp in &doc.parameter_points {
        emit_parameter_point(&mut b, pp);
    }
}

/// Step 4: emit likelihood bindings. Builds a datum-name → flattened-values map
/// so the emitter can inline unbinned observations. Skips likelihoods whose
/// distributions are all native `histfactory_dist` (assembled in step 2b).
///
/// Unbinned data is lowered as a 1-D column of scalar observations; a
/// multidimensional entry (an event with more than one coordinate) cannot be
/// inlined this way and would silently drop columns, so it is rejected.
fn emit_likelihoods(
    m: &mut Module,
    doc: &Document,
    histfactory_names: &BTreeSet<&str>,
) -> Result<()> {
    let mut data_map: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for d in doc.data.iter().filter(|d| d.kind == "unbinned") {
        // NOTE: unbinned data is materialized into memory proportional to the input
        // document size (one f64 per entry), with no entry-count cap. Fine for the
        // current CLI-on-local-file threat model. If untrusted HS3 documents are ever
        // ingested non-interactively, add a hard limit on the total entry count here.
        let mut vals = Vec::with_capacity(d.entries.len());
        for e in &d.entries {
            if e.len() != 1 {
                return Err(Error::Unsupported(format!(
                    "unbinned datum `{}` has a {}-dimensional entry; only scalar \
                     (1-D) observations are supported",
                    d.name,
                    e.len()
                )));
            }
            vals.push(e[0]);
        }
        data_map.insert(d.name.clone(), vals);
    }
    // Per-distribution variate (axis) names. A distribution is lowered as
    // `relabel(<dist>, [names])`, i.e. a record-shaped measure keyed by those
    // names, so its observation must be a matching record (see emit_likelihood).
    let labels_by_dist: BTreeMap<String, Vec<String>> = doc
        .distributions
        .iter()
        .filter(|d| d.kind != "histfactory_dist")
        .filter_map(|d| match variate_name(d) {
            Some(VariateName::Single(v)) => Some((d.name.clone(), vec![v])),
            Some(VariateName::Multiple(ns)) => Some((d.name.clone(), ns)),
            None => None,
        })
        .collect();
    let mut b = Builder::new(m);
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
        crate::likelihood::emit_likelihood(&mut b, lk, &data_map, &labels_by_dist)?;
    }
    Ok(())
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

/// Emit a `functions` block entry as a deterministic FlatPPL binding.
///
/// - `product`: `name = mul(f1, mul(f2, ...))` (fold).
/// - `sum`:     `name = add(s1, add(s2, ...))` (fold).
/// - `generic_function`: `name = <parsed expression>` (the expression may
///   reference other bindings via `self_ref`; it is *not* wrapped in a lambda
///   here — the expression is a deterministic scalar/function-valued formula).
fn emit_function(b: &mut Builder, f: &Function) -> Result<()> {
    match f.kind.as_str() {
        "product" => fold_function(b, f, "factors", "mul", "HS3 product function → fold of mul")?,
        "sum" => fold_function(b, f, "summands", "add", "HS3 sum function → fold of add")?,
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

/// Lower a `product`/`sum` function entry to a left-fold of a binary scalar
/// builtin over its operands. `key` is the operand-array field (`factors` /
/// `summands`), `op` the fold builtin (`mul` / `add`), and `doc` the provenance
/// line. Errs if the operand array is missing or empty.
fn fold_function(b: &mut Builder, f: &Function, key: &str, op: &str, doc: &str) -> Result<()> {
    let operands = f.extra.get(key).and_then(|v| v.as_array()).ok_or_else(|| {
        Error::Unsupported(format!("{} function `{}` missing `{key}`", f.kind, f.name))
    })?;
    if operands.is_empty() {
        return Err(Error::Unsupported(format!(
            "{} function `{}` has no `{key}`",
            f.kind, f.name
        )));
    }
    let nodes: Vec<_> = operands
        .iter()
        .map(|v| field_node(b, v))
        .collect::<Result<_>>()?;
    let folded = nodes
        .into_iter()
        .reduce(|acc, x| b.call(op, &[acc, x]))
        .expect("non-empty operands checked above");
    b.bind_doc(&f.name, folded, &[doc]);
    Ok(())
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

    // A native histfactory_dist sample carrying a `lumi` modifier must be
    // rejected — the native HS3 path has no lumi-config (sigma) to build the
    // constraint, so passing lumi:None would emit a silently weaker model.
    const NATIVE_LUMI_JSON: &str = r#"{
      "distributions": [
        {"name": "ch", "type": "histfactory_dist",
         "axes": [{"name": "obs", "nbins": 1, "min": 0.0, "max": 1.0}],
         "samples": [
           {"name": "sig", "data": {"contents": [5.0]},
            "modifiers": [{"type": "lumi", "name": "Lumi"}]}
         ]}
      ],
      "likelihoods": [
        {"name": "L", "distributions": ["ch"], "data": ["obs_data"]}
      ],
      "data": [
        {"name": "obs_data", "type": "binned", "contents": [5.0]}
      ]
    }"#;

    #[test]
    fn native_histfactory_lumi_modifier_errors() {
        let err = crate::read(NATIVE_LUMI_JSON).expect_err("native lumi must be rejected");
        assert!(matches!(err, crate::Error::Unsupported(_)), "got: {err}");
        let msg = err.to_string();
        assert!(
            msg.contains("lumi") && msg.contains("lumi-config"),
            "error should mention the missing lumi-config: {msg}"
        );
    }

    // Two domains entries naming the same observable with different bounds are
    // contradictory and must be rejected (not last-wins).
    const CONFLICTING_DOMAIN_JSON: &str = r#"{
      "distributions": [
        {"name": "u", "type": "uniform_dist", "x": "x_obs"}
      ],
      "domains": [
        {"name": "d1", "axes": [{"name": "x_obs", "min": 0.0, "max": 1.0}]},
        {"name": "d2", "axes": [{"name": "x_obs", "min": 0.0, "max": 2.0}]}
      ]
    }"#;

    #[test]
    fn conflicting_domain_bounds_error() {
        let err =
            crate::read(CONFLICTING_DOMAIN_JSON).expect_err("conflicting domain bounds must error");
        assert!(matches!(err, crate::Error::Unsupported(_)), "got: {err}");
        assert!(
            err.to_string().contains("conflicting domain bounds"),
            "got: {err}"
        );
    }

    // Identical bounds repeated across domains entries are fine (not a conflict).
    const AGREEING_DOMAIN_JSON: &str = r#"{
      "distributions": [
        {"name": "u", "type": "uniform_dist", "x": "x_obs"}
      ],
      "domains": [
        {"name": "d1", "axes": [{"name": "x_obs", "min": 0.0, "max": 1.0}]},
        {"name": "d2", "axes": [{"name": "x_obs", "min": 0.0, "max": 1.0}]}
      ]
    }"#;

    #[test]
    fn agreeing_duplicate_domain_bounds_ok() {
        let m = crate::read(AGREEING_DOMAIN_JSON).expect("agreeing duplicate bounds must convert");
        let text = print_with(&m, Syntax::Minimal);
        assert!(
            text.contains("Uniform") && text.contains("interval"),
            "got:\n{text}"
        );
    }
}
