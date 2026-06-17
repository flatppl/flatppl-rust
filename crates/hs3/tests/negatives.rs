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

// ---------------------------------------------------------------------------
// Multi-dimensional unbinned datum → Err (only scalar observations supported).
// ---------------------------------------------------------------------------
#[test]
fn multidim_unbinned_data_errs() {
    assert_err_hs3(
        "multidim_unbinned",
        r#"{"distributions":[{"name":"g","type":"gaussian_dist","mean":"mu","sigma":"s","x":"gx"}],
            "data":[{"name":"obs","type":"unbinned","entries":[[1.0,2.0]]}],
            "parameter_points":[{"name":"n","entries":[{"name":"mu","value":0.0},{"name":"s","value":1.0}]}]}"#,
        "dimensional",
    );
}

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
    // histosys interpolates the nominal between lo/hi via the interp module call,
    // with an N(alpha, 1) auxiliary constraint on the shape parameter.
    assert!(
        text.contains(
            "obs_model_c = broadcast(Poisson, hepphys.interp_poly6_lin([9.0, 11.0], [10.0, 12.0], [11.0, 13.0], alpha_shape))"
        ),
        "histosys obs_model mismatch, got:\n{text}"
    );
    assert!(
        text.contains("likelihoodof(Normal(mu = alpha_shape, sigma = 1.0), 0.0)"),
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
        text.contains("obs_model_c = broadcast(Poisson, broadcast(mul, [10.0, 12.0], sf))"),
        "shapefactor obs_model mismatch, got:\n{text}"
    );
    assert!(
        text.contains("sf = elementof(cartpow(posreals, 2))"),
        "shapefactor parameter domain mismatch, got:\n{text}"
    );
    // No auxiliary likelihood term (unconstrained).
    assert!(
        text.contains("L_c = likelihoodof(obs_model_c, [10.0, 12.0])"),
        "shapefactor likelihood should be the bare main term, got:\n{text}"
    );
    assert!(
        flatppl_syntax::parse(&text).is_ok(),
        "round-trip parse failed:\n{text}"
    );
}

// ---------------------------------------------------------------------------
// Unsupported expression operators in a generic_dist expression.
//
// FlatPPL's deterministic-expression sublanguage has no boolean/comparison
// operators; the importer must reject each rather than emit a bogus call. We
// parameterize over all eight comparison/logical binary operators plus the
// logical-not prefix. Each must produce an Err that names the operator.
// ---------------------------------------------------------------------------

/// Build a generic_dist whose expression uses `expr`.
fn generic_dist_with(expr: &str) -> String {
    format!(
        r#"{{"distributions":[{{"name":"d","type":"generic_dist","expression":"{expr}"}}],"parameter_points":[]}}"#
    )
}

#[test]
fn unsupported_binary_operators_err() {
    // (expression-snippet, operator-token-the-message-must-name)
    let cases = [
        ("x == 1.0", "=="),
        ("x != 1.0", "!="),
        ("x < 1.0", "<"),
        ("x <= 1.0", "<="),
        ("x > 1.0", ">"),
        ("x >= 1.0", ">="),
        ("x && y", "&&"),
        ("x || y", "||"),
    ];
    for (expr, op) in cases {
        let json = generic_dist_with(expr);
        match flatppl_hs3::read_hs3(&json) {
            Ok(_) => panic!("operator `{op}` (expr `{expr}`) must be rejected, got Ok"),
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains(op) && msg.contains("not supported"),
                    "operator `{op}`: message should name the op and say `not supported`, got: {msg}"
                );
            }
        }
    }
}

/// The logical-not prefix `!` is rejected too — it surfaces as a parse-level
/// error (unexpected atom) rather than the trailing-operator check, so we assert
/// only that conversion fails and the message references `!`.
#[test]
fn unsupported_logical_not_errs() {
    let json = generic_dist_with("!x");
    match flatppl_hs3::read_hs3(&json) {
        Ok(_) => panic!("logical-not `!` must be rejected, got Ok"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains('!'),
                "logical-not error should reference `!`, got: {msg}"
            );
        }
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
