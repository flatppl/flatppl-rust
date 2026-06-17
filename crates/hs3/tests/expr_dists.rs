//! Integration tests for HS3 `functions` block and expression-based
//! distributions (`generic_dist`, `density_function_dist`,
//! `log_density_function_dist`).
//!
//! Each test checks:
//!   1. `read_hs3` returns Ok.
//!   2. The printed FlatPPL (Minimal) contains the expected FlatPPL constructs.
//!   3. The emitted text round-trip-parses without error.
use flatppl_syntax::{Syntax, parse, print_with};

// ---------------------------------------------------------------------------
// generic_function + density_function_dist
//
// Defines a Gaussian-shape function in `functions`, then references it via
// `density_function_dist`.  Checks that:
//   - the `functions` entry emits a lambda binding,
//   - the `density_function_dist` emits `normalize(weighted(..., Lebesgue(reals)))`.
// ---------------------------------------------------------------------------

const DENSITY_FUNCTION_JSON: &str = r#"{
  "functions": [
    {
      "name": "my_gauss_fn",
      "type": "generic_function",
      "expression": "exp(-0.5 * ((x - mu) / sigma) ^ 2)",
      "variables": ["x"]
    }
  ],
  "distributions": [
    {
      "name": "gauss_dist",
      "type": "density_function_dist",
      "function": "my_gauss_fn"
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "mu",    "value": 0.0},
      {"name": "sigma", "value": 1.0}
    ]}
  ]
}"#;

#[test]
fn density_function_dist_converts() {
    let m = flatppl_hs3::read_hs3(DENSITY_FUNCTION_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== density_function_dist ===\n{text}\n=== end ===");

    // Exact body: the generic_function lowers each operator to its FlatPPL call
    // form over a fresh bound variable _x_; free names mu/sigma stay as refs.
    assert!(
        text.contains(
            "my_gauss_fn = functionof(exp(mul(neg(0.5), pow(divide(sub(_x_, mu), sigma), 2.0))), x = _x_)"
        ),
        "generic_function body mismatch, got:\n{text}"
    );
    // density (not log-density) → normalize(weighted(<fn>, Lebesgue(reals))).
    assert!(
        text.contains("gauss_dist = normalize(weighted(my_gauss_fn, Lebesgue(reals)))"),
        "density_function_dist body mismatch, got:\n{text}"
    );
    // density_function_dist must use weighted, never logweighted.
    assert!(
        !text.contains("logweighted"),
        "must not emit logweighted for density_function_dist, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// log_density_function_dist
//
// Uses a log-density formula (the exponent of a Gaussian) referenced via
// `log_density_function_dist`.  Checks `normalize(logweighted(...))`.
// ---------------------------------------------------------------------------

const LOG_DENSITY_FUNCTION_JSON: &str = r#"{
  "functions": [
    {
      "name": "log_gauss_fn",
      "type": "generic_function",
      "expression": "-0.5 * ((x - mu) / sigma) ^ 2",
      "variables": ["x"]
    }
  ],
  "distributions": [
    {
      "name": "log_gauss_dist",
      "type": "log_density_function_dist",
      "function": "log_gauss_fn"
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "mu",    "value": 0.0},
      {"name": "sigma", "value": 1.0}
    ]}
  ]
}"#;

#[test]
fn log_density_function_dist_converts() {
    let m = flatppl_hs3::read_hs3(LOG_DENSITY_FUNCTION_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== log_density_function_dist ===\n{text}\n=== end ===");

    // Exact body: same operator-lowering as the density variant, but the
    // distribution wraps the function in normalize(LOGweighted(...)) — the log
    // variant treats the function as a log-density, not a density.
    assert!(
        text.contains(
            "log_gauss_fn = functionof(mul(neg(0.5), pow(divide(sub(_x_, mu), sigma), 2.0)), x = _x_)"
        ),
        "log generic_function body mismatch, got:\n{text}"
    );
    assert!(
        text.contains("log_gauss_dist = normalize(logweighted(log_gauss_fn, Lebesgue(reals)))"),
        "log_density_function_dist body mismatch, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// generic_dist (inline expression)
//
// An inline density formula: no `functions` entry — the expression is
// embedded directly in the distribution's `expression` field.
// Checks `normalize(weighted(<lambda>, Lebesgue(reals)))`.
// ---------------------------------------------------------------------------

const GENERIC_DIST_JSON: &str = r#"{
  "distributions": [
    {
      "name": "inline_gauss",
      "type": "generic_dist",
      "expression": "exp(-0.5 * ((x - mu) / sigma) ^ 2)"
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "mu",    "value": 0.0},
      {"name": "sigma", "value": 1.0}
    ]}
  ]
}"#;

#[test]
fn generic_dist_converts() {
    let m = flatppl_hs3::read_hs3(GENERIC_DIST_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== generic_dist ===\n{text}\n=== end ===");

    // Exact body: the inline expression lowers into a functionof(...) over a
    // fresh _x_, wrapped directly in normalize(weighted(..., Lebesgue(reals)))
    // (no separate functions-block binding — the expression is embedded).
    assert!(
        text.contains(
            "inline_gauss = normalize(weighted(functionof(exp(mul(neg(0.5), pow(divide(sub(_x_, mu), sigma), 2.0))), x = _x_), Lebesgue(reals)))"
        ),
        "generic_dist body mismatch, got:\n{text}"
    );
    // generic_dist is a density, never a log-density.
    assert!(
        !text.contains("logweighted"),
        "must not emit logweighted for generic_dist, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// functions: product
//
// `product` folds its factors with `mul`.
// ---------------------------------------------------------------------------

const PRODUCT_FUNCTION_JSON: &str = r#"{
  "functions": [
    {
      "name": "prod_fn",
      "type": "product",
      "factors": ["a", "b", 2.0]
    }
  ],
  "distributions": [
    {
      "name": "obs_dist",
      "type": "gaussian_dist",
      "mean": "mu",
      "sigma": "sig",
      "x": "x_obs"
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "a",   "value": 1.0},
      {"name": "b",   "value": 0.5},
      {"name": "mu",  "value": 0.0},
      {"name": "sig", "value": 1.0}
    ]}
  ]
}"#;

#[test]
fn product_function_converts() {
    let m = flatppl_hs3::read_hs3(PRODUCT_FUNCTION_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== product function ===\n{text}\n=== end ===");

    // Exact body: the factors [a, b, 2.0] fold LEFT-ASSOCIATIVELY into
    // mul(mul(a, b), 2.0). Pinning the whole RHS catches both a wrong fold
    // direction (mul(a, mul(b, 2.0))) and any factor reordering.
    assert!(
        text.contains("prod_fn = mul(mul(a, b), 2.0)"),
        "product fold mismatch (expected left-assoc mul(mul(a, b), 2.0)), got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// functions: sum
//
// `sum` folds its summands with `add`.
// ---------------------------------------------------------------------------

const SUM_FUNCTION_JSON: &str = r#"{
  "functions": [
    {
      "name": "sum_fn",
      "type": "sum",
      "summands": ["c1", "c2", 1.0]
    }
  ],
  "distributions": [
    {
      "name": "obs_dist2",
      "type": "gaussian_dist",
      "mean": "mu2",
      "sigma": "sig2",
      "x": "x_obs2"
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "c1",   "value": 0.3},
      {"name": "c2",   "value": 0.5},
      {"name": "mu2",  "value": 0.0},
      {"name": "sig2", "value": 1.0}
    ]}
  ]
}"#;

#[test]
fn sum_function_converts() {
    let m = flatppl_hs3::read_hs3(SUM_FUNCTION_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== sum function ===\n{text}\n=== end ===");

    // Exact body: the summands [c1, c2, 1.0] fold LEFT-ASSOCIATIVELY into
    // add(add(c1, c2), 1.0).
    assert!(
        text.contains("sum_fn = add(add(c1, c2), 1.0)"),
        "sum fold mismatch (expected left-assoc add(add(c1, c2), 1.0)), got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// Free parameters referenced ONLY inside generic expressions
//
// `mean2` appears only in a generic_function expression (`sqrt(mean2)`), and
// `alpha` appears only in a generic_dist expression. Neither is used by a plain
// distribution field, so the field-walking declaration pass never sees them.
// They MUST still be declared as `elementof(...)` (with bounds from `domains`
// where present) or the emitted FlatPPL has unresolved module references.
// ---------------------------------------------------------------------------

const GENERIC_EXPR_FREE_PARAMS_JSON: &str = r#"{
  "distributions": [
    {"name": "g2", "type": "gaussian_dist", "mean": "mean", "sigma": "sigma", "x": "x"},
    {"name": "genpdf", "type": "generic_dist",
     "expression": "(1 + 0.1 * abs(x) + sin(sqrt(abs(x * alpha + 0.1))))"}
  ],
  "functions": [
    {"name": "mean", "type": "generic_function", "expression": "sqrt(mean2)"}
  ],
  "domains": [
    {"name": "default_domain", "type": "product_domain", "axes": [
      {"name": "alpha", "min": 0.1, "max": 10.0},
      {"name": "mean2", "min": 0.0, "max": 200.0},
      {"name": "sigma", "min": 0.1, "max": 10.0},
      {"name": "x",     "min": -20.0, "max": 20.0}
    ]}
  ],
  "parameter_points": [
    {"name": "default_values", "parameters": [
      {"name": "x",     "value": 0.0},
      {"name": "mean2", "value": 10.0},
      {"name": "sigma", "value": 3.0},
      {"name": "alpha", "value": 5.0}
    ]}
  ]
}"#;

#[test]
fn generic_expr_free_params_declared() {
    let m = flatppl_hs3::read_hs3(GENERIC_EXPR_FREE_PARAMS_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== generic_expr free params ===\n{text}\n=== end ===");

    // `mean2` is referenced only inside the generic_function `sqrt(mean2)`; it
    // must be declared with its `domains` bounds [0, 200].
    assert!(
        text.contains("mean2 = elementof(interval(0.0, 200.0))"),
        "mean2 (used only in generic_function expr) not declared with domain bounds, got:\n{text}"
    );
    // `alpha` is referenced only inside the generic_dist expression; it must be
    // declared with its `domains` bounds [0.1, 10].
    assert!(
        text.contains("alpha = elementof(interval(0.1, 10.0))"),
        "alpha (used only in generic_dist expr) not declared with domain bounds, got:\n{text}"
    );
    // The lambda bound variable `x` must NOT be promoted to a free param
    // declaration; it is bound by the generic expression's lambda / is the
    // observable, not a module-level parameter.
    assert!(
        !text.contains("x = elementof"),
        "the observable/bound variable x must not be declared as a free param, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// Generic-expr free param with NO domains entry falls back to `reals`.
// ---------------------------------------------------------------------------

const GENERIC_EXPR_NO_DOMAIN_JSON: &str = r#"{
  "distributions": [
    {"name": "genpdf", "type": "generic_dist",
     "expression": "exp(-0.5 * (x - shift) ^ 2)"}
  ],
  "parameter_points": [
    {"name": "nominal", "parameters": [
      {"name": "x",     "value": 0.0},
      {"name": "shift", "value": 1.0}
    ]}
  ]
}"#;

#[test]
fn generic_expr_free_param_defaults_to_reals() {
    let m = flatppl_hs3::read_hs3(GENERIC_EXPR_NO_DOMAIN_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== generic_expr no-domain ===\n{text}\n=== end ===");

    // `shift` has no `domains` entry, so it defaults to `reals`.
    assert!(
        text.contains("shift = elementof(reals)"),
        "shift (no domain) should default to elementof(reals), got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// PI constant inlining
//
// `generic_dist` with PI in the expression — PI must be inlined as a real
// literal, not left as an identifier (FlatPPL has no `pi` constant).
// ---------------------------------------------------------------------------

const PI_EXPR_JSON: &str = r#"{
  "distributions": [
    {
      "name": "pi_dist",
      "type": "generic_dist",
      "expression": "exp(-0.5 * ((x - mu) / sigma) ^ 2) / (sigma * PI)"
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "mu",    "value": 0.0},
      {"name": "sigma", "value": 1.0}
    ]}
  ]
}"#;

#[test]
fn pi_constant_inlined_in_generic_dist() {
    let m = flatppl_hs3::read_hs3(PI_EXPR_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== PI inlining ===\n{text}\n=== end ===");

    // PI must be inlined as a numeric literal, never as a bare identifier
    assert!(
        !text.contains("PI"),
        "PI must be inlined as literal, got:\n{text}"
    );
    // π ≈ 3.14159... — look for a recognizable prefix
    assert!(
        text.contains("3.14") || text.contains("3.1415"),
        "expected π literal (~3.14), got:\n{text}"
    );
    assert!(
        text.contains("normalize"),
        "missing normalize, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}
