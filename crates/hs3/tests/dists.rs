//! Integration tests for newly-mapped HS3 distribution types.
//!
//! Each test checks:
//!   1. `read_hs3` returns Ok.
//!   2. The printed FlatPPL (Minimal) contains the right FlatPPL dist name and
//!      the right parameter identifiers.
//!   3. The emitted text round-trip-parses without error.
use flatppl_syntax::{Syntax, parse, print_with};

// ---------------------------------------------------------------------------
// generalized_normal_dist → GeneralizedNormal(mean=…, alpha=…, beta=…)
// ---------------------------------------------------------------------------

const GENNORMAL_JSON: &str = r#"{
  "distributions": [
    {
      "name": "gn_dist",
      "type": "generalized_normal_dist",
      "mean":  "gn_mu",
      "alpha": "gn_alpha",
      "beta":  "gn_beta",
      "x":     "x_obs"
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "gn_mu",    "value": 0.0},
      {"name": "gn_alpha", "value": 1.0},
      {"name": "gn_beta",  "value": 2.0}
    ]}
  ]
}"#;

#[test]
fn generalized_normal_converts() {
    let m = flatppl_hs3::read_hs3(GENNORMAL_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== GeneralizedNormal ===\n{text}\n=== end ===");

    // Exact body: each keyword must bind its matching HS3 field (mean=gn_mu,
    // alpha=gn_alpha, beta=gn_beta) — a swapped-kwarg lowering would still pass
    // bare-token checks, so pin the full call RHS and relabel.
    assert!(
        text.contains(
            r#"gn_dist = relabel(GeneralizedNormal(mean = gn_mu, alpha = gn_alpha, beta = gn_beta), ["x_obs"])"#
        ),
        "GeneralizedNormal body mismatch (kwarg→field binding), got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// multivariate_normal_dist → MvNormal(mu=[…], cov=[[…],…])
// ---------------------------------------------------------------------------

const MVNORMAL_JSON: &str = r#"{
  "distributions": [
    {
      "name": "mv_dist",
      "type": "multivariate_normal_dist",
      "mean": ["mv_mu0", "mv_mu1"],
      "covariances": [[1.0, 0.0], [0.0, 1.0]],
      "x":   ["obs0", "obs1"]
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "mv_mu0", "value": 0.0},
      {"name": "mv_mu1", "value": 0.0}
    ]}
  ]
}"#;

#[test]
fn multivariate_normal_converts() {
    let m = flatppl_hs3::read_hs3(MVNORMAL_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== MvNormal ===\n{text}\n=== end ===");

    // Exact body: mean vector [mv_mu0, mv_mu1], 2×2 identity covariance, both
    // observed names in the relabel, in order.
    assert!(
        text.contains(
            "mv_dist = relabel(MvNormal(mu = [mv_mu0, mv_mu1], cov = [[1.0, 0.0], [0.0, 1.0]]), \
             [\"obs0\", \"obs1\"])"
        ),
        "MvNormal body mismatch, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// crystalball_dist (single-sided) → hepphys.CrystalBall(m0, sigma, alpha, n)
// ---------------------------------------------------------------------------

const CB_JSON: &str = r#"{
  "distributions": [
    {
      "name": "cb_dist",
      "type": "crystalball_dist",
      "m0":    "cb_m0",
      "sigma": "cb_sigma",
      "alpha": "cb_alpha",
      "n":     "cb_n",
      "m":     "m_obs"
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "cb_m0",    "value": 5.28},
      {"name": "cb_sigma", "value": 0.003},
      {"name": "cb_alpha", "value": 1.5},
      {"name": "cb_n",     "value": 3.0}
    ]}
  ]
}"#;

#[test]
fn crystalball_single_sided_converts() {
    let m = flatppl_hs3::read_hs3(CB_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== CrystalBall ===\n{text}\n=== end ===");

    assert!(
        !text.contains("DoubleSided"),
        "must not emit DoubleSided, got:\n{text}"
    );
    // hepphys must be imported from the particle-physics standard module.
    assert!(
        text.contains("hepphys = standard_module(\"particle-physics\", \"0.1\")"),
        "missing hepphys standard_module import, got:\n{text}"
    );
    // Exact body: positional args (m0, sigma, alpha, n) in HS3 order, relabeled
    // onto the observed mass variate m_obs.
    assert!(
        text.contains(
            "cb_dist = relabel(hepphys.CrystalBall(cb_m0, cb_sigma, cb_alpha, cb_n), [\"m_obs\"])"
        ),
        "single-sided CrystalBall body mismatch, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// crystalball_dist (double-sided) → hepphys.DoubleSidedCrystalBall(…)
// ---------------------------------------------------------------------------

const DSCB_JSON: &str = r#"{
  "distributions": [
    {
      "name": "dscb_dist",
      "type": "crystalball_dist",
      "m0":      "dscb_m0",
      "sigma_L": "dscb_sigL",
      "sigma_R": "dscb_sigR",
      "alpha_L": "dscb_aL",
      "n_L":     "dscb_nL",
      "alpha_R": "dscb_aR",
      "n_R":     "dscb_nR",
      "m":       "m_obs2"
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "dscb_m0",   "value": 125.0},
      {"name": "dscb_sigL", "value": 1.5},
      {"name": "dscb_sigR", "value": 2.0},
      {"name": "dscb_aL",   "value": 1.2},
      {"name": "dscb_nL",   "value": 5.0},
      {"name": "dscb_aR",   "value": 1.8},
      {"name": "dscb_nR",   "value": 4.0}
    ]}
  ]
}"#;

#[test]
fn crystalball_double_sided_converts() {
    let m = flatppl_hs3::read_hs3(DSCB_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== DoubleSidedCrystalBall ===\n{text}\n=== end ===");

    assert!(
        !text.contains("hepphys.CrystalBall("),
        "must not emit single-sided CB, got:\n{text}"
    );
    // Exact body: the seven double-sided parameters in the spec §09 alternating order
    // (m0, sigma_L, sigma_R, alpha_L, alpha_R, n_L, n_R), relabeled onto m_obs2.
    assert!(
        text.contains(
            "dscb_dist = relabel(hepphys.DoubleSidedCrystalBall(dscb_m0, dscb_sigL, dscb_sigR, \
             dscb_aL, dscb_aR, dscb_nL, dscb_nR), [\"m_obs2\"])"
        ),
        "double-sided CrystalBall body mismatch, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// argus_dist → hepphys.Argus(resonance, slope, power)
// ---------------------------------------------------------------------------

const ARGUS_JSON: &str = r#"{
  "distributions": [
    {
      "name": "argus_d",
      "type": "argus_dist",
      "resonance": "arg_c",
      "slope":     "arg_chi",
      "power":     "arg_p",
      "mass":      "mass_obs"
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "arg_c",   "value": 5.29},
      {"name": "arg_chi", "value": -10.0},
      {"name": "arg_p",   "value": 0.5}
    ]}
  ]
}"#;

#[test]
fn argus_converts() {
    let m = flatppl_hs3::read_hs3(ARGUS_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== Argus ===\n{text}\n=== end ===");

    // Exact body: positional args (resonance, slope, power) in HS3 order,
    // relabeled onto the observed mass variate.
    assert!(
        text.contains("argus_d = relabel(hepphys.Argus(arg_c, arg_chi, arg_p), [\"mass_obs\"])"),
        "Argus body mismatch, got:\n{text}"
    );
    assert!(
        text.contains("hepphys = standard_module(\"particle-physics\", \"0.1\")"),
        "missing hepphys standard_module import, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// mixture_dist (extended=true) → normalize(superpose(weighted(c1, s1), weighted(c2, s2)))
// extended=true: N coefficients for N summands, all used directly.
// ---------------------------------------------------------------------------

const MIXTURE_EXTENDED_JSON: &str = r#"{
  "distributions": [
    {
      "name": "g1",
      "type": "gaussian_dist",
      "mean": "mu1",
      "sigma": "sig1",
      "x": "x_obs"
    },
    {
      "name": "g2",
      "type": "gaussian_dist",
      "mean": "mu2",
      "sigma": "sig2",
      "x": "x_obs"
    },
    {
      "name": "mix",
      "type": "mixture_dist",
      "summands": ["g1", "g2"],
      "coefficients": [0.3, 0.7],
      "extended": true
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "mu1",  "value": 0.0},
      {"name": "sig1", "value": 1.0},
      {"name": "mu2",  "value": 3.0},
      {"name": "sig2", "value": 1.0}
    ]}
  ]
}"#;

#[test]
fn mixture_extended_converts() {
    let m = flatppl_hs3::read_hs3(MIXTURE_EXTENDED_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== mixture_dist extended ===\n{text}\n=== end ===");

    // The whole `mix` binding is the bare superpose of two weighted summands,
    // with the declared coefficients (0.3, 0.7) bound to the matching summand
    // (g1, g2). extended=true uses the coefficients directly.
    assert!(
        text.contains("mix = superpose(weighted(0.3, g1), weighted(0.7, g2))"),
        "extended mixture body mismatch (coeff→summand binding), got:\n{text}"
    );
    // extended=true must NOT wrap the superposition in an outer `normalize(`
    // (the rate-weighted superposition is already an unnormalized measure —
    // normalizing would discard the rate information). Check the *binding* line,
    // not the provenance comment (which mentions normalize for the non-extended
    // form).
    let mix_line = text
        .lines()
        .find(|l| l.trim_start().starts_with("mix ="))
        .expect("mix binding line present");
    assert!(
        !mix_line.contains("normalize("),
        "extended mixture binding must NOT contain normalize(, got:\n{mix_line}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// mixture_dist (extended=false / absent) → normalize(superpose(...)) with
// implicit last coefficient = 1 - sum(given).
// Non-extended: N-1 explicit coefficients; last is computed.
// ---------------------------------------------------------------------------

const MIXTURE_NONEXTENDED_JSON: &str = r#"{
  "distributions": [
    {
      "name": "h1",
      "type": "gaussian_dist",
      "mean": "nu1",
      "sigma": "tau1",
      "x": "y_obs"
    },
    {
      "name": "h2",
      "type": "gaussian_dist",
      "mean": "nu2",
      "sigma": "tau2",
      "x": "y_obs"
    },
    {
      "name": "mix2",
      "type": "mixture_dist",
      "summands": ["h1", "h2"],
      "coefficients": [0.4],
      "extended": false
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "nu1",  "value": 0.0},
      {"name": "tau1", "value": 1.0},
      {"name": "nu2",  "value": 2.0},
      {"name": "tau2", "value": 0.5}
    ]}
  ]
}"#;

#[test]
fn mixture_nonextended_converts() {
    let m = flatppl_hs3::read_hs3(MIXTURE_NONEXTENDED_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== mixture_dist non-extended ===\n{text}\n=== end ===");

    // Non-extended: the full normalized superposition. The single explicit
    // coefficient 0.4 binds to the FIRST summand (h1); the implicit last
    // coefficient 0.6 (= 1 - 0.4) binds to the SECOND summand (h2). This exact
    // body pins both the normalize wrapper AND the coefficient→summand binding.
    assert!(
        text.contains("mix2 = normalize(superpose(weighted(0.4, h1), weighted(0.6, h2)))"),
        "non-extended mixture body mismatch (coeff→summand binding), got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// relativistic_breit_wigner_dist → still deferred (Unsupported)
// ---------------------------------------------------------------------------

const RBW_JSON: &str = r#"{
  "distributions": [
    {
      "name": "rbw",
      "type": "relativistic_breit_wigner_dist",
      "channels": [
        {"Gamma": "G", "m1": 0.14, "m2": 0.14, "l": 1, "R": 1.5}
      ],
      "x": "m_obs"
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "G", "value": 0.15}
    ]}
  ]
}"#;

#[test]
fn relativistic_breit_wigner_still_deferred() {
    let result = flatppl_hs3::read_hs3(RBW_JSON);
    assert!(
        result.is_err(),
        "relativistic_breit_wigner_dist must remain unsupported; got Ok"
    );
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("relativistic_breit_wigner_dist") || msg.contains("multi-channel"),
        "error should mention deferred status, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// rate_extended_dist → PoissonProcess(weighted(<rate>, self_ref(<distribution>)))
// ---------------------------------------------------------------------------

const RATE_EXTENDED_JSON: &str = r#"{
  "distributions": [
    {
      "name": "shape_dist",
      "type": "gaussian_dist",
      "mean": "mu_shape",
      "sigma": "sig_shape",
      "x": "x_obs"
    },
    {
      "name": "process",
      "type": "rate_extended_dist",
      "rate": "n_sig",
      "distribution": "shape_dist"
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "mu_shape",  "value": 0.0},
      {"name": "sig_shape", "value": 1.0},
      {"name": "n_sig",     "value": 100.0}
    ]}
  ]
}"#;

#[test]
fn rate_extended_dist_converts() {
    let m = flatppl_hs3::read_hs3(RATE_EXTENDED_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== rate_extended_dist ===\n{text}\n=== end ===");

    // Exact body: the rate (n_sig) weights a self-ref to the inner shape dist;
    // rate_extended has no own variate, so the `process` binding is the bare
    // PoissonProcess with no relabel. Pinning the whole RHS catches a swapped
    // weighted(shape_dist, n_sig) order or a stray relabel.
    assert!(
        text.contains("process = PoissonProcess(weighted(n_sig, shape_dist))"),
        "rate_extended_dist body mismatch, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// rate_density_dist → PoissonProcess(weighted(self_ref(<function>), Lebesgue(reals)))
// ---------------------------------------------------------------------------

const RATE_DENSITY_JSON: &str = r#"{
  "functions": [
    {
      "name": "my_density",
      "type": "generic_function",
      "expression": "exp(-0.5 * x * x)",
      "variables": ["x"]
    }
  ],
  "distributions": [
    {
      "name": "process2",
      "type": "rate_density_dist",
      "function": "my_density"
    }
  ],
  "parameter_points": []
}"#;

#[test]
fn rate_density_dist_converts() {
    let m = flatppl_hs3::read_hs3(RATE_DENSITY_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== rate_density_dist ===\n{text}\n=== end ===");

    // Exact body: the density function is weighted against Lebesgue(reals)
    // (no rate parameter — the function carries the intensity), wrapped in
    // PoissonProcess.
    assert!(
        text.contains("process2 = PoissonProcess(weighted(my_density, Lebesgue(reals)))"),
        "rate_density_dist body mismatch, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// bincounts_extended_dist → BinnedPoissonProcess([edges], weighted(rate, shape))
// Tests both {nbins,min,max} axis form and edge-expansion.
// ---------------------------------------------------------------------------

const BINCOUNTS_EXTENDED_JSON: &str = r#"{
  "distributions": [
    {
      "name": "bshape",
      "type": "gaussian_dist",
      "mean": "bmu",
      "sigma": "bsig",
      "x": "bx_obs"
    },
    {
      "name": "binned_proc",
      "type": "bincounts_extended_dist",
      "rate": "n_bkg",
      "distribution": "bshape",
      "axes": [{"nbins": 4, "min": 0.0, "max": 2.0}]
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "bmu",   "value": 1.0},
      {"name": "bsig",  "value": 0.5},
      {"name": "n_bkg", "value": 50.0}
    ]}
  ]
}"#;

#[test]
fn bincounts_extended_dist_converts() {
    let m = flatppl_hs3::read_hs3(BINCOUNTS_EXTENDED_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== bincounts_extended_dist ===\n{text}\n=== end ===");

    // Exact body: the {nbins:4, min:0, max:2} axis expands to the full edge
    // vector [0.0, 0.5, 1.0, 1.5, 2.0] (step = 0.5), and the rate (n_bkg)
    // weights a self-ref to the shape dist. Pinning the whole RHS is what makes
    // the edge expansion meaningful — bare contains("0.5") false-passes on the
    // `bsig` parameter value 0.5.
    assert!(
        text.contains(
            "binned_proc = BinnedPoissonProcess([0.0, 0.5, 1.0, 1.5, 2.0], weighted(n_bkg, bshape))"
        ),
        "bincounts_extended_dist body mismatch (edge expansion + weighted order), got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// bincounts_extended_dist with {edges:[...]} axis form
// ---------------------------------------------------------------------------

const BINCOUNTS_EDGES_JSON: &str = r#"{
  "distributions": [
    {
      "name": "eshape",
      "type": "uniform_dist",
      "x": "ex_obs"
    },
    {
      "name": "edge_proc",
      "type": "bincounts_extended_dist",
      "rate": "n_edge",
      "distribution": "eshape",
      "axes": [{"edges": [0.0, 1.0, 3.0, 6.0]}]
    }
  ],
  "domains": [
    {"name": "obs_domain", "axes": [{"name": "ex_obs", "min": 0.0, "max": 6.0}]}
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "n_edge", "value": 10.0}
    ]}
  ]
}"#;

#[test]
fn bincounts_extended_edges_form_converts() {
    let m = flatppl_hs3::read_hs3(BINCOUNTS_EDGES_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== bincounts_extended edges ===\n{text}\n=== end ===");

    // Exact body: the explicit edge vector [0.0, 1.0, 3.0, 6.0] from axes.edges
    // (not expanded — used verbatim), with the rate (n_edge) weighting a self-ref
    // to the uniform shape.
    assert!(
        text.contains(
            "edge_proc = BinnedPoissonProcess([0.0, 1.0, 3.0, 6.0], weighted(n_edge, eshape))"
        ),
        "bincounts edges-form body mismatch, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// bincounts multi-axis → Err(Unsupported)
// ---------------------------------------------------------------------------

const BINCOUNTS_MULTIAXIS_JSON: &str = r#"{
  "distributions": [
    {
      "name": "mshape",
      "type": "uniform_dist",
      "x": "mx_obs"
    },
    {
      "name": "multi_proc",
      "type": "bincounts_extended_dist",
      "rate": "n_multi",
      "distribution": "mshape",
      "axes": [
        {"nbins": 2, "min": 0.0, "max": 1.0},
        {"nbins": 3, "min": 0.0, "max": 3.0}
      ]
    }
  ],
  "domains": [
    {"name": "obs_domain", "axes": [{"name": "mx_obs", "min": 0.0, "max": 3.0}]}
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "n_multi", "value": 5.0}
    ]}
  ]
}"#;

#[test]
fn bincounts_multiaxis_returns_error() {
    let result = flatppl_hs3::read_hs3(BINCOUNTS_MULTIAXIS_JSON);
    assert!(result.is_err(), "multi-axis bincounts must error; got Ok");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("multi-axis"),
        "error should mention multi-axis, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// bincounts_density_dist → BinnedPoissonProcess([edges], weighted(fn_ref, Lebesgue(reals)))
// ---------------------------------------------------------------------------

const BINCOUNTS_DENSITY_JSON: &str = r#"{
  "functions": [
    {
      "name": "flat_fn",
      "type": "generic_function",
      "expression": "x * 0.1 + 1.0",
      "variables": ["x"]
    }
  ],
  "distributions": [
    {
      "name": "density_proc",
      "type": "bincounts_density_dist",
      "function": "flat_fn",
      "axes": [{"edges": [0.0, 2.0, 4.0, 6.0]}]
    }
  ],
  "parameter_points": []
}"#;

#[test]
fn bincounts_density_dist_converts() {
    let m = flatppl_hs3::read_hs3(BINCOUNTS_DENSITY_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== bincounts_density_dist ===\n{text}\n=== end ===");

    // Exact body: the explicit edge vector [0.0, 2.0, 4.0, 6.0] from axes.edges,
    // and the density function weighted against Lebesgue(reals) (no rate param).
    assert!(
        text.contains(
            "density_proc = BinnedPoissonProcess([0.0, 2.0, 4.0, 6.0], weighted(flat_fn, Lebesgue(reals)))"
        ),
        "bincounts_density_dist body mismatch, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// polynomial_dist → normalize(weighted(functionof(polynomial([c...], _x_), x = _x_), Lebesgue))
// ---------------------------------------------------------------------------

const POLYNOMIAL_JSON: &str = r#"{
  "distributions": [
    {
      "name": "poly_d",
      "type": "polynomial_dist",
      "coefficients": [1.0, "c1", 0.5],
      "x": "p_obs"
    }
  ],
  "domains": [
    {"name": "default_domain", "type": "product_domain",
     "axes": [{"name": "p_obs", "min": -5.0, "max": 5.0}]}
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "c1", "value": 0.3}
    ]}
  ]
}"#;

#[test]
fn polynomial_dist_converts() {
    let m = flatppl_hs3::read_hs3(POLYNOMIAL_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== polynomial_dist ===\n{text}\n=== end ===");

    // Exact body: the coefficient vector [1.0, c1, 0.5] (mixed literal/param) is
    // applied via polynomial over a fresh bound variable _p_obs_, normalized
    // against Lebesgue(reals), and relabeled onto the observed variate p_obs.
    assert!(
        text.contains(
            "poly_d = relabel(normalize(truncate(weighted(functionof(polynomial([1.0, c1, 0.5], _p_obs_), \
             p_obs = _p_obs_), Lebesgue(reals)), interval(-5.0, 5.0))), [\"p_obs\"])"
        ),
        "polynomial_dist body mismatch, got:\n{text}"
    );
    // c1 is a free parameter (the only non-literal coefficient).
    assert!(
        text.contains("c1 = elementof(reals)"),
        "missing c1 free-param declaration, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// barlow_beeston_lite_poisson_constraint_dist →
//   relabel(broadcast(Poisson, [expected...]), [x names...])
// ---------------------------------------------------------------------------

const BB_LITE_JSON: &str = r#"{
  "distributions": [
    {
      "name": "bb_constraint",
      "type": "barlow_beeston_lite_poisson_constraint_dist",
      "x": ["bb_obs0", "bb_obs1", "bb_obs2"],
      "expected": [10.0, "e1", 5.0]
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "e1", "value": 8.0}
    ]}
  ]
}"#;

#[test]
fn barlow_beeston_lite_poisson_constraint_converts() {
    let m = flatppl_hs3::read_hs3(BB_LITE_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== barlow_beeston_lite ===\n{text}\n=== end ===");

    // Exact body: per-bin Poisson broadcast over the expected vector
    // [10.0, e1, 5.0] (mixed literal/param), relabeled onto the three observed
    // bin names in order.
    assert!(
        text.contains(
            "bb_constraint = relabel(broadcast(Poisson, [10.0, e1, 5.0]), \
             [\"bb_obs0\", \"bb_obs1\", \"bb_obs2\"])"
        ),
        "barlow_beeston_lite body mismatch, got:\n{text}"
    );
    // e1 is the only free (non-literal) expected value; it must be a positive rate.
    assert!(
        text.contains("e1 = elementof(posreals)"),
        "missing e1 positive-rate declaration, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// chebychev_dist → normalize(truncate(weighted(functionof(WEIGHT, x=_x_), Lebesgue(reals)), interval(lo, hi)))
// WEIGHT = add(1.0, Σ mul(cᵢ, poly.chebyshev(i, t)))
// t = div(sub(mul(2.0, _x_), lo+hi), hi-lo)
// ---------------------------------------------------------------------------

const CHEBY_JSON: &str = r#"{
  "distributions": [
    {"name": "bkg", "type": "chebychev_dist", "coefficients": ["c0","c1"], "x": "m"}
  ],
  "domains": [
    {"name": "default_domain", "type": "product_domain",
     "axes": [{"name": "m", "min": 0.0, "max": 10.0}]}
  ]
}"#;

#[test]
fn chebychev_converts() {
    let m = flatppl_hs3::read_hs3(CHEBY_JSON).expect("read_hs3");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    eprintln!("{text}");
    // Must be the QUALIFIED module-member call `poly.chebyshev(`, never the bare
    // `chebyshev(` builtin — a bare call would collide with distances.ron's
    // `chebyshev` (L∞ distance) and silently mean the wrong function.
    assert!(
        text.contains("poly.chebyshev("),
        "must use qualified poly.chebyshev (not a bare chebyshev that collides with distances): {text}"
    );
    // Lowering is normalize(truncate(weighted(functionof(...)), interval(lo,hi))).
    // Assert the truncation to the observable interval, not just normalize — a
    // dropped truncate would otherwise pass silently.
    assert!(text.contains("normalize"), "must be normalized: {text}");
    assert!(
        text.contains("truncate(weighted(functionof("),
        "must truncate the weighted functionof over the observable interval: {text}"
    );
    assert!(
        text.contains("interval(0.0, 10.0)"),
        "truncation interval must carry the declared [min, max] bounds: {text}"
    );
    assert!(
        text.contains("c0") && text.contains("c1"),
        "missing coeffs: {text}"
    );
    flatppl_syntax::parse(&text).expect("re-parse");
}

// ---------------------------------------------------------------------------
// efficiency_product_pdf_dist → weighted(<eff>, <pdf>)  (RooEffProd)
// No own variate; the inner pdf carries it. Range-normalization is applied by
// the consumer (the harness assemble step), so the converter emits the raw
// pointwise reweighting — no premature normalize/truncate.
// ---------------------------------------------------------------------------
const EFFPROD_JSON: &str = r#"{
  "distributions": [
    {"name": "m", "type": "exponential_dist", "c": "lam", "x": "t"},
    {"name": "me", "type": "efficiency_product_pdf_dist", "eff": "effn", "pdf": "m"}
  ],
  "functions": [
    {"name": "effn", "type": "generic_function", "expression": "0.5 * t"}
  ],
  "domains": [
    {"name": "default_domain", "type": "product_domain",
     "axes": [{"name": "t", "min": 0.0, "max": 5.0}]}
  ],
  "parameter_points": [
    {"name": "n", "entries": [{"name": "lam", "value": 1.5}]}
  ]
}"#;

#[test]
fn efficiency_product_pdf_converts() {
    let m = flatppl_hs3::read_hs3(EFFPROD_JSON).expect("read_hs3");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    eprintln!("=== efficiency_product_pdf ===\n{text}\n=== end ===");
    // The effprod binding is the bare reweighting of the pdf measure by the
    // efficiency function — no own relabel (Variate::None), no premature
    // normalize/truncate (the consumer range-normalizes).
    assert!(
        text.contains("me = weighted(effn, m)"),
        "efficiency_product_pdf body mismatch, got:\n{text}"
    );
    // The wrapped exponential pdf still carries its own variate via relabel,
    // with the corrected rate = c (no negation).
    assert!(
        text.contains(r#"m = relabel(Exponential(rate = lam), ["t"])"#),
        "wrapped exponential pdf missing or rate mis-emitted, got:\n{text}"
    );
    flatppl_syntax::parse(&text).expect("re-parse");
}

// ---------------------------------------------------------------------------
// Regression: a coefficient written as a STRING numeric literal ("1.0") is a
// constant, not a free parameter. It must lower to a literal inside the
// polynomial and emit NO `1.0 = elementof(...)` binding — that statement is
// invalid FlatPPL and fails the print→reparse self-check. rf203_ranges hits
// this (its polynomial coefficients are ["1.0", "a"]).
// ---------------------------------------------------------------------------
const STRING_LITERAL_COEFF_JSON: &str = r#"{
  "distributions": [
    {"name": "px", "type": "polynomial_dist", "coefficients": ["1.0", "a"], "x": "x"}
  ],
  "domains": [
    {"name": "default_domain", "type": "product_domain",
     "axes": [{"name": "x", "min": -10.0, "max": 10.0}]}
  ],
  "parameter_points": [{"name": "n", "entries": [{"name": "a", "value": 0.1}]}]
}"#;

#[test]
fn string_literal_coefficient_lowers_to_literal_not_param() {
    // read_hs3 runs the print→reparse self-check, so an invalid `1.0 = elementof`
    // binding would make this `.expect` fail.
    let m = flatppl_hs3::read_hs3(STRING_LITERAL_COEFF_JSON).expect("read_hs3");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    // The "1.0" string coefficient becomes a literal inside the polynomial array.
    assert!(text.contains("polynomial([1.0, a]"), "got:\n{text}");
    // And is NOT declared as a free parameter binding.
    assert!(
        !text.contains("1.0 = elementof"),
        "numeric-literal coefficient must not become a param binding, got:\n{text}"
    );
    flatppl_syntax::parse(&text).expect("re-parse");
}

// ---------------------------------------------------------------------------
// M1: generic_dist / generic_function infer their observable from the document
// (the free identifier naming a data axis or distribution variate), instead of
// hardcoding "x". rf703's `eff` and rf210's angular generic_dist hit this.
// ---------------------------------------------------------------------------
const GENERIC_NONX_JSON: &str = r#"{
  "distributions": [
    {"name": "g", "type": "generic_dist", "expression": "1 + sin(2 * psi)"}
  ],
  "data": [
    {"name": "d", "type": "unbinned", "axes": [{"name": "psi", "min": 0.0, "max": 3.0}], "entries": [[0.5]]}
  ],
  "domains": [
    {"name": "default_domain", "type": "product_domain",
     "axes": [{"name": "psi", "min": 0.0, "max": 3.0}]}
  ]
}"#;

#[test]
fn generic_dist_infers_non_x_observable() {
    let m = flatppl_hs3::read_hs3(GENERIC_NONX_JSON).expect("read_hs3");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    eprintln!("=== generic_dist non-x observable ===\n{text}\n=== end ===");
    // Observable inferred as `psi` (a data axis): functionof binds psi, and the
    // truncation uses psi's declared domain — never the hardcoded "x".
    assert!(
        text.contains("functionof(") && text.contains("psi = _psi_"),
        "generic_dist must bind the inferred observable psi, got:\n{text}"
    );
    assert!(
        text.contains("interval(0.0, 3.0)"),
        "must truncate over psi's declared domain, got:\n{text}"
    );
    assert!(
        !text.contains("psi = elementof"),
        "psi is the observable, not a free param, got:\n{text}"
    );
    flatppl_syntax::parse(&text).expect("re-parse");
}

const GENERIC_FN_OBS_JSON: &str = r#"{
  "distributions": [
    {"name": "m", "type": "gaussian_dist", "mean": "mu", "sigma": 1.0, "x": "t"}
  ],
  "functions": [
    {"name": "eff", "type": "generic_function", "expression": "0.5 * t"}
  ],
  "parameter_points": [{"name": "n", "entries": [{"name": "mu", "value": 0.0}]}]
}"#;

#[test]
fn generic_function_of_observable_is_a_lambda() {
    let m = flatppl_hs3::read_hs3(GENERIC_FN_OBS_JSON).expect("read_hs3");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    eprintln!("=== generic_function of observable ===\n{text}\n=== end ===");
    // `t` is the gaussian's variate (an observable), so eff is a function of t:
    // emitted as functionof binding t, not a bare expr leaving t unbound.
    assert!(
        text.contains("eff = functionof(") && text.contains("t = _t_"),
        "generic_function of an observable must be a lambda over it, got:\n{text}"
    );
    assert!(
        !text.contains("t = elementof"),
        "t is the observable, not a free param, got:\n{text}"
    );
    flatppl_syntax::parse(&text).expect("re-parse");
}
