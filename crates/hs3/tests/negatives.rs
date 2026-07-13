//! Negative tests: every input here is malformed or unsupported and MUST return
//! `Err`. These guard the Stream B/C changes that replaced silent coercion with
//! explicit errors. Each test asserts both that conversion fails AND that the
//! error message names the offending construct (so the message stays useful).

/// Assert that `read_hs3(json)` errors and the message contains `needle`.
fn assert_err_hs3(label: &str, json: &str, needle: &str) {
    match flatppl_hs3::read_hs3(json) {
        Ok(_) => panic!("{label}: expected Err, got Ok"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains(needle),
                "{label}: error message should mention `{needle}`, got: {msg}"
            );
        }
    }
}

/// Assert that `read_hs3(json)` errors with the `Error::Unimplemented` prefix
/// (never `Unsupported`) AND that the message contains `needle`. Use this for
/// silently-dropped-field rejections: the testsuite classifies the
/// `unimplemented HS3 construct:` prefix as a clean skip, so a site that
/// flips to `Unsupported` must fail this assertion even if the body text is
/// unchanged.
fn assert_unimplemented_hs3(label: &str, json: &str, needle: &str) {
    match flatppl_hs3::read_hs3(json) {
        Ok(_) => panic!("{label}: expected Err, got Ok"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.starts_with("unimplemented HS3 construct:"),
                "{label}: error should use the `unimplemented HS3 construct:` prefix, got: {msg}"
            );
            assert!(
                msg.contains(needle),
                "{label}: error message should mention `{needle}`, got: {msg}"
            );
        }
    }
}

/// Assert that `read_pyhf(json)` errors and the message contains `needle`.
fn assert_err_pyhf(label: &str, json: &str, needle: &str) {
    match flatppl_hs3::read_pyhf(json) {
        Ok(_) => panic!("{label}: expected Err, got Ok"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains(needle),
                "{label}: error message should mention `{needle}`, got: {msg}"
            );
        }
    }
}

/// Assert that the auto-detecting `read(json)` errors and the message contains
/// every needle in `needles`.
fn assert_err_read(label: &str, json: &str, needles: &[&str]) {
    match flatppl_hs3::read(json) {
        Ok(_) => panic!("{label}: expected Err, got Ok"),
        Err(e) => {
            let msg = e.to_string();
            for needle in needles {
                assert!(
                    msg.contains(needle),
                    "{label}: error message should mention `{needle}`, got: {msg}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// uniform_dist without a declared domain → Err (Stream B).
// ---------------------------------------------------------------------------
#[test]
fn uniform_without_domain_errs() {
    assert_err_hs3(
        "uniform_no_domain",
        r#"{"distributions":[{"name":"u","type":"uniform_dist","x":"ux"}],"parameter_points":[]}"#,
        "has no declared domain",
    );
}

// ---------------------------------------------------------------------------
// Duplicate binding name across distributions → Err (would otherwise shadow).
// ---------------------------------------------------------------------------
#[test]
fn duplicate_binding_name_errs() {
    assert_err_hs3(
        "duplicate_name",
        r#"{"distributions":[
            {"name":"dup","type":"gaussian_dist","mean":"mu","sigma":"s","x":"x1"},
            {"name":"dup","type":"gaussian_dist","mean":"mu2","sigma":"s2","x":"x2"}],
            "parameter_points":[]}"#,
        "duplicate binding name",
    );
}

// NOTE: multi-dimensional unbinned data is no longer an error — it is embedded
// as a multi-column `table` (one column per observable axis). Positive coverage
// of the table/domain lowering lives in `tests/data_tables.rs`.

// ---------------------------------------------------------------------------
// Multi-axis bincounts → Err (only single-axis binning supported).
// ---------------------------------------------------------------------------
#[test]
fn multiaxis_bincounts_errs() {
    assert_err_hs3(
        "multiaxis_bincounts",
        r#"{"distributions":[
            {"name":"shp","type":"uniform_dist","x":"sx"},
            {"name":"proc","type":"bincounts_extended_dist","rate":"r","distribution":"shp",
             "axes":[{"nbins":2,"min":0.0,"max":1.0},{"nbins":3,"min":0.0,"max":3.0}]}],
            "domains":[{"name":"d","axes":[{"name":"sx","min":0.0,"max":3.0}]}],
            "parameter_points":[{"name":"n","entries":[{"name":"r","value":5.0}]}]}"#,
        "multi-axis",
    );
}

// ---------------------------------------------------------------------------
// histfactory modifier negatives (driven through the pyhf channel assembler).
// ---------------------------------------------------------------------------

/// A modifier missing its `parameter`/`name` field → Err (would emit a dangling ref).
#[test]
fn modifier_missing_parameter_errs() {
    assert_err_pyhf(
        "modifier_no_param",
        r#"{"channels":[{"name":"c","samples":[
            {"name":"sig","data":[10.0,12.0],"modifiers":[
              {"type":"normfactor","data":null}]}]}],
            "observations":[{"name":"c","data":[10.0,12.0]}],
            "measurements":[{"name":"m","config":{"poi":""}}]}"#,
        "missing its `parameter`",
    );
}

/// An unknown modifier type → Err (UnknownModifier).
#[test]
fn unknown_modifier_type_errs() {
    assert_err_pyhf(
        "unknown_modifier",
        r#"{"channels":[{"name":"c","samples":[
            {"name":"sig","data":[10.0,12.0],"modifiers":[
              {"name":"p","type":"frobnicate","data":null}]}]}],
            "observations":[{"name":"c","data":[10.0,12.0]}],
            "measurements":[{"name":"m","config":{"poi":""}}]}"#,
        "frobnicate",
    );
}

// ---------------------------------------------------------------------------
// histosys + shapefactor POSITIVE coverage.
//
// These modifier kinds are *supported* (not errors); the task asks for explicit
// path coverage so a regression that breaks them is caught. They run through the
// pyhf channel assembler.
// ---------------------------------------------------------------------------

#[test]
fn histosys_modifier_path_converts() {
    let json = r#"{"channels":[{"name":"c","samples":[
        {"name":"sig","data":[10.0,12.0],"modifiers":[
          {"name":"alpha_shape","type":"histosys",
           "data":{"hi":{"contents":[11.0,13.0]},"lo":{"contents":[9.0,11.0]}}}]}]}],
        "observations":[{"name":"c","data":[10.0,12.0]}],
        "measurements":[{"name":"m","config":{"poi":""}}]}"#;
    let m = flatppl_hs3::read_pyhf(json).expect("histosys must convert");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    // histosys interpolates the nominal between lo/hi via the interp module call
    // (the sample's expected yields), then the channel model is a reified Poisson.
    assert!(
        text.contains(
            "c_sig_expected = \
             hepphys.interp_poly6_lin([9.0, 11.0], c_sig_nominal, [11.0, 13.0], alpha_shape)"
        ),
        "histosys interpolation mismatch, got:\n{text}"
    );
    assert!(
        text.contains("c_model = functionof(broadcast(Poisson, c_expected))"),
        "histosys obs model mismatch, got:\n{text}"
    );
    // The shape parameter carries an N(alpha, 1) auxiliary constraint, reified
    // (functionof) and observed at 0, parameter-keyed.
    assert!(
        text.contains("alpha_shape_constraint = functionof(Normal(mu = alpha_shape, sigma = 1.0))")
            && text.contains(
                "alpha_shape_constraint_likelihood = \
                 likelihoodof(alpha_shape_constraint, 0.0)"
            ),
        "histosys aux constraint mismatch, got:\n{text}"
    );
    assert!(
        flatppl_syntax::parse(&text).is_ok(),
        "round-trip parse failed:\n{text}"
    );
}

#[test]
fn shapefactor_modifier_path_converts() {
    let json = r#"{"channels":[{"name":"c","samples":[
        {"name":"sig","data":[10.0,12.0],"modifiers":[
          {"name":"sf","type":"shapefactor","data":null}]}]}],
        "observations":[{"name":"c","data":[10.0,12.0]}],
        "measurements":[{"name":"m","config":{"poi":""}}]}"#;
    let m = flatppl_hs3::read_pyhf(json).expect("shapefactor must convert");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    // shapefactor is an unconstrained per-bin multiplicative scale: no aux term.
    assert!(
        text.contains("c_sig_expected = broadcast(mul, c_sig_nominal, sf)")
            && text.contains("c_model = functionof(broadcast(Poisson, c_expected))"),
        "shapefactor obs model mismatch, got:\n{text}"
    );
    assert!(
        text.contains("sf = elementof(cartpow(posreals, 2))"),
        "shapefactor parameter domain mismatch, got:\n{text}"
    );
    // Unconstrained: a single observation term, so the top-level likelihood is
    // just that term (no joint_likelihood, no constraint terms).
    assert!(
        text.contains("c_observed = [10.0, 12.0]")
            && text.contains("c_likelihood = likelihoodof(c_model, c_observed)")
            && text.contains("likelihood = c_likelihood"),
        "shapefactor likelihood should be the bare main term, got:\n{text}"
    );
    assert!(
        !text.contains("joint_likelihood"),
        "single unconstrained term must not wrap in joint_likelihood, got:\n{text}"
    );
    assert!(
        flatppl_syntax::parse(&text).is_ok(),
        "round-trip parse failed:\n{text}"
    );
}

// ---------------------------------------------------------------------------
// HS3 equality operators in a generic_dist expression → Err (Unimplemented).
//
// FlatPPL restricts equality to discrete domains (§07), and HS3 expression
// operands are untyped reals, so `==`/`!=` (approx-equal) and `===`/`!==`
// (exact) have no honest lowering. They are valid HS3, hence the
// `unimplemented HS3 construct:` prefix (testsuite SKIP, never hard-fail).
// Comparisons/booleans/ternary now LOWER (lt/le/gt/ge, land/lor/lnot,
// ifelse) — positive coverage lives in src/expr.rs tests and expr_dists.rs.
// ---------------------------------------------------------------------------

/// Build a generic_dist whose expression uses `expr`.
fn generic_dist_with(expr: &str) -> String {
    format!(
        r#"{{"distributions":[{{"name":"d","type":"generic_dist","expression":"{expr}"}}],"parameter_points":[]}}"#
    )
}

#[test]
fn equality_operators_err_unimplemented() {
    for (expr, op) in [
        ("x == 1.0", "=="),
        ("x != 1.0", "!="),
        ("x === 1.0", "==="),
        ("x !== 1.0", "!=="),
    ] {
        let json = generic_dist_with(expr);
        assert_unimplemented_hs3(&format!("equality `{op}`"), &json, op);
    }
}

// ---------------------------------------------------------------------------
// Native HS3 histfactory_dist with a `lumi` modifier → Err (Stream B / Iter2).
//
// The native HS3 path has no measurement lumi-config (the sigma needed to build
// the luminosity constraint), unlike the pyhf path. Importing it as an
// unconstrained scale would silently weaken the model, so it is rejected.
// ---------------------------------------------------------------------------
#[test]
fn native_histfactory_lumi_modifier_errs() {
    let json = r#"{
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
    // The message must name both the offending modifier and the missing config.
    assert_err_read("native_lumi", json, &["lumi", "lumi-config"]);
}

// ---------------------------------------------------------------------------
// Two `domains` axes naming the same observable with DIFFERENT bounds → Err.
//
// Contradictory bounds must be rejected rather than silently resolved last-wins.
// (Identical repeated bounds are fine — covered by the positive sibling test.)
// ---------------------------------------------------------------------------
#[test]
fn conflicting_domain_bounds_errs() {
    let json = r#"{
        "distributions": [
            {"name": "u", "type": "uniform_dist", "x": "x_obs"}
        ],
        "domains": [
            {"name": "d1", "axes": [{"name": "x_obs", "min": 0.0, "max": 1.0}]},
            {"name": "d2", "axes": [{"name": "x_obs", "min": 0.0, "max": 2.0}]}
        ]
    }"#;
    assert_err_read("conflicting_domain", json, &["conflicting domain bounds"]);
}

/// Sanity counterpart: the SAME observable repeated across `domains` entries
/// with IDENTICAL bounds must NOT error (agreement is not a conflict).
#[test]
fn agreeing_domain_bounds_ok() {
    let json = r#"{
        "distributions": [
            {"name": "u", "type": "uniform_dist", "x": "x_obs"}
        ],
        "domains": [
            {"name": "d1", "axes": [{"name": "x_obs", "min": 0.0, "max": 1.0}]},
            {"name": "d2", "axes": [{"name": "x_obs", "min": 0.0, "max": 1.0}]}
        ]
    }"#;
    let m = flatppl_hs3::read(json).expect("agreeing repeated domain bounds must NOT error");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    assert!(
        text.contains("Uniform(interval(0.0, 1.0))"),
        "uniform should carry the agreed bounds, got:\n{text}"
    );
}

// ---------------------------------------------------------------------------
// Native likelihood `data[i]` resolution is strict (Iter4-B): a string ref MUST
// name a `Document.data` datum. Anything else — a name that exists nowhere, or
// one that merely COLLIDES with a distribution / free-parameter binding — is
// rejected, because binding the observation to an unrelated top-level binding
// would be silently wrong and the round-trip gate (syntactically valid) cannot
// catch it.
// ---------------------------------------------------------------------------

/// A data ref that names nothing at all → Err.
#[test]
fn unresolved_likelihood_data_ref_errs() {
    let json = r#"{
        "distributions": [
            {"name": "g", "type": "gaussian_dist", "mean": "mu", "sigma": "s", "x": "gx"}
        ],
        "likelihoods": [
            {"name": "L", "distributions": ["g"], "data": ["mystery"]}
        ],
        "parameter_points": [
            {"name": "n", "entries": [{"name": "mu", "value": 0.0}, {"name": "s", "value": 1.0}]}
        ]
    }"#;
    assert_err_read(
        "unresolved_likelihood_data",
        json,
        &["mystery", "resolves to no datum"],
    );
}

/// A data ref that COLLIDES with a distribution name (but is not a datum) → Err.
/// Previously this silently bound the observation to the `g` distribution.
#[test]
fn likelihood_data_ref_colliding_with_dist_name_errs() {
    let json = r#"{
        "distributions": [
            {"name": "g", "type": "gaussian_dist", "mean": "mu", "sigma": "s", "x": "gx"}
        ],
        "likelihoods": [
            {"name": "L", "distributions": ["g"], "data": ["g"]}
        ],
        "parameter_points": [
            {"name": "n", "entries": [{"name": "mu", "value": 0.0}, {"name": "s", "value": 1.0}]}
        ]
    }"#;
    assert_err_read(
        "likelihood_data_collides_with_dist",
        json,
        &["g", "resolves to no datum"],
    );
}

/// A data ref that COLLIDES with a free-parameter name (but is not a datum) →
/// Err. Previously this silently bound the observation to the `mu` parameter.
#[test]
fn likelihood_data_ref_colliding_with_param_name_errs() {
    let json = r#"{
        "distributions": [
            {"name": "g", "type": "gaussian_dist", "mean": "mu", "sigma": "s", "x": "gx"}
        ],
        "likelihoods": [
            {"name": "L", "distributions": ["g"], "data": ["mu"]}
        ],
        "parameter_points": [
            {"name": "n", "entries": [{"name": "mu", "value": 0.0}, {"name": "s", "value": 1.0}]}
        ]
    }"#;
    assert_err_read(
        "likelihood_data_collides_with_param",
        json,
        &["mu", "resolves to no datum"],
    );
}

#[test]
fn same_variate_product_with_mixed_measures_errs() {
    // A product over one observable `obs` of a continuous (gaussian, Lebesgue)
    // and a discrete (poisson, counting) factor has no pointwise-density-product
    // meaning — must fail loud, not emit a wrong measure (§12).
    assert_err_hs3(
        "mixed_measure_product",
        r#"{"distributions":[
            {"name":"prod","type":"product_dist","factors":["g","p"]},
            {"name":"g","type":"gaussian_dist","mean":"mu","sigma":"s","x":"obs"},
            {"name":"p","type":"poisson_dist","mean":"lam","x":"obs"}
        ],"parameter_points":[{"name":"nom","entries":[
            {"name":"mu","value":0.0},{"name":"s","value":1.0},{"name":"lam","value":3.0}
        ]}]}"#,
        "reference measure",
    );
}

#[test]
fn same_variate_three_factor_product_mixed_measures_errs() {
    // g1, g2 (Lebesgue) then p (counting), all over `obs`: base measure is set by
    // g1, and the later poisson differs — the any()-over-non-first path.
    assert_err_hs3(
        "mixed_measure_product_3factor",
        r#"{"distributions":[
            {"name":"prod","type":"product_dist","factors":["g1","g2","p"]},
            {"name":"g1","type":"gaussian_dist","mean":"m1","sigma":"s1","x":"obs"},
            {"name":"g2","type":"gaussian_dist","mean":"m2","sigma":"s2","x":"obs"},
            {"name":"p","type":"poisson_dist","mean":"lam","x":"obs"}
        ],"parameter_points":[{"name":"nom","entries":[
            {"name":"m1","value":0.0},{"name":"s1","value":1.0},
            {"name":"m2","value":0.0},{"name":"s2","value":1.0},{"name":"lam","value":3.0}
        ]}]}"#,
        "reference measure",
    );
}

// ---------------------------------------------------------------------------
// Not-yet-implemented constructs use the `unimplemented HS3 construct:` prefix
// (Error::Unimplemented) so the testsuite can classify them as clean skips;
// invalid documents keep `unsupported HS3 construct:` (Error::Unsupported).
// ---------------------------------------------------------------------------
#[test]
fn multi_axis_bincounts_is_unimplemented_prefix() {
    assert_err_hs3(
        "multi_axis_bincounts_prefix",
        r#"{"distributions":[{"name":"b","type":"bincounts_extended_dist",
            "rate":"r","distribution":"inner",
            "axes":[{"nbins":2,"min":0.0,"max":1.0},{"nbins":2,"min":0.0,"max":1.0}]},
            {"name":"inner","type":"gaussian_dist","mean":"mu","sigma":"s","x":"x"}],
            "parameter_points":[]}"#,
        "unimplemented HS3 construct:",
    );
}

#[test]
fn invalid_document_keeps_unsupported_prefix() {
    assert_err_hs3(
        "duplicate_name_prefix",
        r#"{"distributions":[
            {"name":"dup","type":"gaussian_dist","mean":"mu","sigma":"s","x":"x1"},
            {"name":"dup","type":"gaussian_dist","mean":"mu2","sigma":"s2","x":"x2"}],
            "parameter_points":[]}"#,
        "unsupported HS3 construct:",
    );
}

// ---------------------------------------------------------------------------
// Fields the converter previously DESERIALIZED PAST silently — a weighted
// dataset became an unweighted model, aux likelihood terms vanished. Presence
// must now fail loud (Unimplemented) until actually lowered.
// ---------------------------------------------------------------------------
#[test]
fn aux_distributions_errs() {
    assert_unimplemented_hs3(
        "aux_distributions",
        r#"{"distributions":[{"name":"g","type":"gaussian_dist","mean":"mu","sigma":"s","x":"x"}],
            "data":[{"name":"d","type":"unbinned","entries":[[1.0]],
                     "axes":[{"name":"x","min":0.0,"max":5.0}]}],
            "likelihoods":[{"name":"L","distributions":["g"],"data":["d"],
                            "aux_distributions":["g"]}],
            "parameter_points":[]}"#,
        "aux_distributions",
    );
}

#[test]
fn weighted_unbinned_data_errs() {
    assert_unimplemented_hs3(
        "weighted_data",
        r#"{"distributions":[{"name":"g","type":"gaussian_dist","mean":"mu","sigma":"s","x":"x"}],
            "data":[{"name":"d","type":"unbinned","entries":[[1.0],[2.0]],
                     "weights":[0.5,0.5],"axes":[{"name":"x","min":0.0,"max":5.0}]}],
            "parameter_points":[]}"#,
        "weights",
    );
}

#[test]
fn entries_uncertainties_errs() {
    assert_unimplemented_hs3(
        "entries_uncertainties",
        r#"{"distributions":[{"name":"g","type":"gaussian_dist","mean":"mu","sigma":"s","x":"x"}],
            "data":[{"name":"d","type":"unbinned","entries":[[1.0]],
                     "entries_uncertainties":[[0.1]],
                     "axes":[{"name":"x","min":0.0,"max":5.0}]}],
            "parameter_points":[]}"#,
        "entries_uncertainties",
    );
}

#[test]
fn binned_uncertainty_block_errs() {
    assert_unimplemented_hs3(
        "binned_uncertainty",
        r#"{"distributions":[{"name":"g","type":"gaussian_dist","mean":"mu","sigma":"s","x":"x"}],
            "data":[{"name":"d","type":"binned","contents":[3.0,4.0],
                     "uncertainty":{"type":"gaussian_uncertainty","sigma":[0.5,0.5]},
                     "axes":[{"name":"x","min":0.0,"max":2.0}]}],
            "parameter_points":[]}"#,
        "uncertainty",
    );
}

/// A `point` datum carrying an `uncertainty` block must still be rejected: this
/// pins the datum_columns-before-value ordering in build_table/data_shapes (the
/// rejection must fire before the scalar `value` branch short-circuits it).
#[test]
fn point_uncertainty_errs() {
    assert_unimplemented_hs3(
        "point_uncertainty",
        r#"{"distributions":[{"name":"g","type":"gaussian_dist","mean":"mu","sigma":"s","x":"x"}],
            "data":[{"name":"d","type":"point","value":1.27,
                     "uncertainty":{"type":"gaussian_uncertainty","sigma":0.1}}],
            "likelihoods":[{"name":"L","distributions":["g"],"data":["d"]}],
            "parameter_points":[]}"#,
        "uncertainty",
    );
}
