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

// ---------------------------------------------------------------------------
// erf/erfc → special-functions module
//
// A generic_function that uses `erf`/`erfc` must:
//   1. Emit a call into the `specfun` alias (`specfun.erf(...)` / `.erfc(...)`).
//   2. Bind the special-functions standard module at exactly the spec name and
//      version — pin the full binding string, since a wrong module name or
//      version (e.g. `special_functions`/`0.2`) would silently break inference
//      resolution while a bare `standard_module` substring still matched.
//   3. Round-trip parse without error.
// ---------------------------------------------------------------------------

#[test]
fn expr_erf_lowers_to_special_functions() {
    const JSON: &str = r#"{
      "functions": [
        {"name": "eff", "type": "generic_function", "expression": "erf(x)", "x": "x"}
      ]
    }"#;
    let m = flatppl_hs3::read_hs3(JSON).expect("read_hs3");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== erf emit ===\n{text}\n=== end ===");
    assert!(text.contains(".erf("), "missing specfun.erf call: {text}");
    assert!(
        text.contains(r#"standard_module("special-functions", "0.1")"#),
        "missing/wrong special-functions binding: {text}"
    );
    flatppl_syntax::parse(&text).expect("re-parse");
}

#[test]
fn expr_erfc_lowers_to_special_functions() {
    const JSON: &str = r#"{
      "functions": [
        {"name": "eff", "type": "generic_function", "expression": "erfc(x)", "x": "x"}
      ]
    }"#;
    let m = flatppl_hs3::read_hs3(JSON).expect("read_hs3");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== erfc emit ===\n{text}\n=== end ===");
    assert!(text.contains(".erfc("), "missing specfun.erfc call: {text}");
    assert!(
        text.contains(r#"standard_module("special-functions", "0.1")"#),
        "missing/wrong special-functions binding: {text}"
    );
    flatppl_syntax::parse(&text).expect("re-parse");
}

// ---------------------------------------------------------------------------
// generic_dist: piecewise ternary (RooGenericPdf idiom)
//
// A piecewise density via the ternary — the RooGenericPdf idiom this operator
// support exists for. Structural + round-trip only: no corpus fixture
// exercises these operators, so there is no numeric gate.
// ---------------------------------------------------------------------------

const PIECEWISE_TERNARY_JSON: &str = r#"{
  "distributions": [
    {"name": "pw", "type": "generic_dist",
     "expression": "x > 0.0 ? exp(-x) : 0.0"}
  ],
  "domains": [
    {"name": "default_domain", "type": "product_domain",
     "axes": [{"name": "x", "min": -5.0, "max": 5.0}]}
  ],
  "parameter_points": []
}"#;

#[test]
fn generic_dist_piecewise_ternary_converts() {
    let m = flatppl_hs3::read_hs3(PIECEWISE_TERNARY_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== piecewise ternary generic_dist ===\n{text}\n=== end ===");

    // The ternary lowers to ifelse(gt(...), ..., ...); the comparison operand
    // lowers to gt(x, 0.0).
    assert!(
        text.contains("ifelse(gt("),
        "ternary must lower to ifelse(gt(...)), got:\n{text}"
    );
    // The "then" branch keeps the exp(...) call form.
    assert!(
        text.contains("exp("),
        "missing exp(...) branch, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// functions: operand names are free parameters, and a fold over an observable
// is a function OF it.
//
// HS3 §"Functions": a `polynomial`'s `coefficients` and a `sum`/`product`'s
// `summands`/`factors` name model quantities. A name there is a parameter of
// the model exactly as one in a distribution field is, so it needs a binding;
// and an entry whose operands include its observable denotes a function of
// that observable, not a scalar.
//
// Neither held before. `declare_free_params` walks distribution fields and
// `declare_generic_expr_params` walks `expression` strings, so a name appearing
// only in an operand array reached neither and emitted an unresolvable
// reference. And a fold ignored its observable, emitting a bare expression over
// a name unbound at module level.
//
// The two folds above keep their bare form on purpose: their operands are
// parameters only, so there is no observable to be a function of. That is why
// those goldens do not move, and it is also why the defect survived them.
// ---------------------------------------------------------------------------

const FUNCTION_OPERANDS_JSON: &str = r#"{
  "functions": [
    {"name": "poly_fn", "type": "polynomial", "coefficients": ["a0", "a1"], "x": "y"},
    {"name": "sum_over_obs", "type": "sum", "summands": ["a0", "y"]},
    {"name": "prod_over_obs", "type": "product", "factors": ["a1", "y"]},
    {"name": "sum_params_only", "type": "sum", "summands": ["a0", "a1"]}
  ],
  "distributions": [
    {"name": "obs_dist", "type": "gaussian_dist", "mean": "mu", "sigma": "sig", "x": "y"}
  ],
  "domains": [
    {"name": "default_domain", "type": "product_domain", "axes": [
      {"name": "a0", "min": -5.0, "max": 5.0},
      {"name": "a1", "min": -1.0, "max": 1.0}
    ]}
  ],
  "parameter_points": [
    {"name": "nominal", "parameters": [
      {"name": "a0", "value": -0.5},
      {"name": "a1", "value": -0.5},
      {"name": "y", "value": 0.0},
      {"name": "mu", "value": 0.0},
      {"name": "sig", "value": 1.0}
    ]}
  ]
}"#;

#[test]
fn function_operand_names_are_declared_free_parameters() {
    let m = flatppl_hs3::read_hs3(FUNCTION_OPERANDS_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);

    // Declared over the `domains` axis when there is one, matching the rule
    // `declare_generic_expr_params` already uses for expression identifiers.
    assert!(
        text.contains("a0 = elementof(interval(-5.0, 5.0))")
            && text.contains("a1 = elementof(interval(-1.0, 1.0))"),
        "operand names must be declared over their domain axis, got:\n{text}"
    );
    // `y` is the observable, listed in parameter_points as its reference-point
    // value. It is the lambda's bound variable, NOT a module binding.
    assert!(
        !text.contains("y = elementof"),
        "the observable must not be declared as a free parameter, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

#[test]
fn a_fold_over_an_observable_is_a_lambda() {
    let m = flatppl_hs3::read_hs3(FUNCTION_OPERANDS_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);

    // A fold whose operands include the observable is a function of it, the
    // same shape the sibling `polynomial` entry already emits.
    // Minimal syntax spells a lambda `functionof(<body>, <param> = <hole>)`;
    // canonical syntax prints the same node as `y -> <body>`.
    assert!(
        text.contains("sum_over_obs = functionof(add(a0, _y_), y = _y_)")
            && text.contains("prod_over_obs = functionof(mul(a1, _y_), y = _y_)"),
        "a fold over an observable must be a lambda over it, got:\n{text}"
    );
    assert!(
        text.contains("poly_fn = functionof(polynomial([a0, a1], _y_), y = _y_)"),
        "the polynomial sibling pins the shape being matched, got:\n{text}"
    );
    // A fold over parameters ONLY stays a bare scalar: wrapping it would make
    // it function-valued where a real is expected.
    assert!(
        text.contains("sum_params_only = add(a0, a1)"),
        "a parameter-only fold must stay bare, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// A distribution field naming a `functions` entry is that function APPLIED.
//
// HS3 §"Functions": a distribution parameter naming a function takes that
// function's VALUE. When the function is of another observable the distribution
// is conditional on it, which the importer lowers as a joint over both axes.
//
// The application and the conditional detection both key off one map from
// function name to observable axis, and that map covered `generic_function`
// only. So a `polynomial`, `sum` or `product` mean emitted the bare name into a
// record field, which §04 forbids ("a function may not appear inside a record"),
// and the model was never marked conditional. rf301, rf302, rf303 and rf305 all
// hit it; rf302 showed both halves at once, its generic_function mean lowering
// correctly beside three that did not.
//
// `function_observable` is now the single rule, shared by the emission that
// picks the lambda's bound variable and the map that picks the application
// argument. Were they to disagree the model would apply the function at the
// wrong axis and still type-check.
// ---------------------------------------------------------------------------

const APPLIED_FUNCTION_JSON: &str = r#"{
  "functions": [
    {"name": "fy_gen", "type": "generic_function", "expression": "a0 + a1 * y"},
    {"name": "fy_poly", "type": "polynomial", "coefficients": ["a0", "a1"], "x": "y"},
    {"name": "fy_sum", "type": "sum", "summands": ["a0", "y"]},
    {"name": "fy_prod", "type": "product", "factors": ["a1", "y"]}
  ],
  "distributions": [
    {"name": "m_gen", "type": "gaussian_dist", "mean": "fy_gen", "sigma": "sig", "x": "x"},
    {"name": "m_poly", "type": "gaussian_dist", "mean": "fy_poly", "sigma": "sig", "x": "x"},
    {"name": "m_sum", "type": "gaussian_dist", "mean": "fy_sum", "sigma": "sig", "x": "x"},
    {"name": "m_prod", "type": "gaussian_dist", "mean": "fy_prod", "sigma": "sig", "x": "x"}
  ],
  "data": [
    {"name": "d", "type": "unbinned",
     "axes": [{"name": "y"}, {"name": "x"}],
     "entries": [[0.1, 0.2], [0.3, 0.4]]}
  ],
  "domains": [
    {"name": "default_domain", "type": "product_domain", "axes": [
      {"name": "a0", "min": -5.0, "max": 5.0},
      {"name": "a1", "min": -1.0, "max": 1.0},
      {"name": "x", "min": -5.0, "max": 5.0},
      {"name": "y", "min": -5.0, "max": 5.0}
    ]}
  ],
  "parameter_points": [
    {"name": "nominal", "parameters": [
      {"name": "a0", "value": 0.0}, {"name": "a1", "value": 0.5},
      {"name": "x", "value": 0.0}, {"name": "y", "value": 0.0},
      {"name": "sig", "value": 1.0}
    ]}
  ]
}"#;

#[test]
fn every_function_kind_is_applied_at_its_observable() {
    let m = flatppl_hs3::read_hs3(APPLIED_FUNCTION_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);

    // Every kind applies, not just generic_function. The bare name would be a
    // function inside a record, which §04 forbids.
    // Minimal syntax names the lambda's bound observable `_y_`, so the
    // application reads `fy_x(_y_)` inside the conditional's `functionof`.
    for f in ["fy_gen", "fy_poly", "fy_sum", "fy_prod"] {
        assert!(
            text.contains(&format!("Normal(mu = {f}(_y_), sigma = sig)")),
            "`{f}` must be applied at its observable, got:\n{text}"
        );
        assert!(
            !text.contains(&format!("Normal(mu = {f}, sigma = sig)")),
            "`{f}` must not appear bare in a record field, got:\n{text}"
        );
    }
    // Applying it also makes the distribution conditional on that axis, so each
    // model is a joint over (y, x) rather than a bare 1-D Normal.
    // Count a code-only pattern: each conditional's doc comment also mentions
    // `logweighted`, so a bare count of that word doubles.
    assert_eq!(
        text.matches("normalize(logweighted(functionof(").count(),
        4,
        "each of the four models is a conditional joint, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}
