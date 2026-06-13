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

    assert!(
        text.contains("GeneralizedNormal"),
        "missing GeneralizedNormal, got:\n{text}"
    );
    assert!(text.contains("mean"), "missing mean kwarg, got:\n{text}");
    assert!(text.contains("alpha"), "missing alpha kwarg, got:\n{text}");
    assert!(text.contains("beta"), "missing beta kwarg, got:\n{text}");
    assert!(text.contains("gn_mu"), "missing gn_mu, got:\n{text}");
    assert!(text.contains("gn_alpha"), "missing gn_alpha, got:\n{text}");
    assert!(text.contains("gn_beta"), "missing gn_beta, got:\n{text}");
    // variate relabeled
    assert!(text.contains("relabel"), "missing relabel, got:\n{text}");
    assert!(text.contains("x_obs"), "missing x_obs, got:\n{text}");

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

    assert!(text.contains("MvNormal"), "missing MvNormal, got:\n{text}");
    assert!(text.contains("mu"), "missing mu kwarg, got:\n{text}");
    assert!(text.contains("cov"), "missing cov kwarg, got:\n{text}");
    assert!(text.contains("mv_mu0"), "missing mv_mu0, got:\n{text}");
    assert!(text.contains("mv_mu1"), "missing mv_mu1, got:\n{text}");
    // both observed variable names must appear in relabel
    assert!(text.contains("obs0"), "missing obs0, got:\n{text}");
    assert!(text.contains("obs1"), "missing obs1, got:\n{text}");
    assert!(text.contains("relabel"), "missing relabel, got:\n{text}");

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
        text.contains("hepphys.CrystalBall"),
        "missing hepphys.CrystalBall, got:\n{text}"
    );
    assert!(
        !text.contains("DoubleSided"),
        "must not emit DoubleSided, got:\n{text}"
    );
    assert!(
        text.contains("hepphys"),
        "missing hepphys binding, got:\n{text}"
    );
    assert!(text.contains("cb_m0"), "missing cb_m0, got:\n{text}");
    assert!(text.contains("cb_sigma"), "missing cb_sigma, got:\n{text}");
    assert!(text.contains("cb_alpha"), "missing cb_alpha, got:\n{text}");
    assert!(text.contains("cb_n"), "missing cb_n, got:\n{text}");
    assert!(
        text.contains("m_obs"),
        "missing m_obs variate, got:\n{text}"
    );
    assert!(text.contains("relabel"), "missing relabel, got:\n{text}");
    assert!(
        text.contains("standard_module"),
        "missing standard_module binding, got:\n{text}"
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
        text.contains("hepphys.DoubleSidedCrystalBall"),
        "missing hepphys.DoubleSidedCrystalBall, got:\n{text}"
    );
    assert!(
        !text.contains("hepphys.CrystalBall("),
        "must not emit single-sided CB, got:\n{text}"
    );
    assert!(text.contains("dscb_m0"), "missing dscb_m0, got:\n{text}");
    assert!(
        text.contains("dscb_sigL"),
        "missing dscb_sigL, got:\n{text}"
    );
    assert!(
        text.contains("dscb_sigR"),
        "missing dscb_sigR, got:\n{text}"
    );
    assert!(text.contains("dscb_aL"), "missing dscb_aL, got:\n{text}");
    assert!(text.contains("dscb_nL"), "missing dscb_nL, got:\n{text}");
    assert!(text.contains("dscb_aR"), "missing dscb_aR, got:\n{text}");
    assert!(text.contains("dscb_nR"), "missing dscb_nR, got:\n{text}");
    assert!(
        text.contains("m_obs2"),
        "missing m_obs2 variate, got:\n{text}"
    );
    assert!(
        text.contains("standard_module"),
        "missing standard_module binding, got:\n{text}"
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

    assert!(
        text.contains("hepphys.Argus"),
        "missing hepphys.Argus, got:\n{text}"
    );
    assert!(text.contains("arg_c"), "missing arg_c, got:\n{text}");
    assert!(text.contains("arg_chi"), "missing arg_chi, got:\n{text}");
    assert!(text.contains("arg_p"), "missing arg_p, got:\n{text}");
    assert!(
        text.contains("mass_obs"),
        "missing mass_obs variate, got:\n{text}"
    );
    assert!(text.contains("relabel"), "missing relabel, got:\n{text}");
    assert!(
        text.contains("standard_module"),
        "missing standard_module binding, got:\n{text}"
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

    assert!(
        text.contains("normalize(superpose("),
        "missing normalize(superpose(, got:\n{text}"
    );
    assert!(
        text.contains("weighted("),
        "missing weighted(, got:\n{text}"
    );
    // both summand self-refs must appear
    assert!(text.contains("g1"), "missing g1 summand ref, got:\n{text}");
    assert!(text.contains("g2"), "missing g2 summand ref, got:\n{text}");

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

    assert!(
        text.contains("normalize(superpose("),
        "missing normalize(superpose(, got:\n{text}"
    );
    assert!(
        text.contains("weighted("),
        "missing weighted(, got:\n{text}"
    );
    // both summand self-refs must appear
    assert!(text.contains("h1"), "missing h1 summand ref, got:\n{text}");
    assert!(text.contains("h2"), "missing h2 summand ref, got:\n{text}");
    // implicit second coefficient 0.6 (= 1 - 0.4) must appear
    assert!(
        text.contains("0.6"),
        "missing implicit coefficient 0.6, got:\n{text}"
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

    assert!(
        text.contains("PoissonProcess(weighted("),
        "missing PoissonProcess(weighted(, got:\n{text}"
    );
    assert!(text.contains("n_sig"), "missing n_sig, got:\n{text}");
    assert!(
        text.contains("shape_dist"),
        "missing shape_dist self-ref, got:\n{text}"
    );
    // rate_extended has no own variate — no relabel for 'process'
    // (the inner dist shape_dist carries a variate, but process itself does not)

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

    assert!(
        text.contains("PoissonProcess(weighted("),
        "missing PoissonProcess(weighted(, got:\n{text}"
    );
    assert!(
        text.contains("my_density"),
        "missing my_density ref, got:\n{text}"
    );
    assert!(text.contains("Lebesgue"), "missing Lebesgue, got:\n{text}");

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

    assert!(
        text.contains("BinnedPoissonProcess("),
        "missing BinnedPoissonProcess(, got:\n{text}"
    );
    assert!(text.contains("n_bkg"), "missing n_bkg rate, got:\n{text}");
    assert!(
        text.contains("bshape"),
        "missing bshape self-ref, got:\n{text}"
    );
    assert!(
        text.contains("weighted("),
        "missing weighted(, got:\n{text}"
    );
    // Edge expansion: 4 bins [0,2] → step=0.5 → edges 0.0, 0.5, 1.0, 1.5, 2.0
    assert!(
        text.contains("0.5"),
        "missing computed edge 0.5, got:\n{text}"
    );
    assert!(
        text.contains("1.5"),
        "missing computed edge 1.5, got:\n{text}"
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

    assert!(
        text.contains("BinnedPoissonProcess("),
        "missing BinnedPoissonProcess(, got:\n{text}"
    );
    // Edge vector from axes.edges
    assert!(text.contains("3.0"), "missing edge 3.0, got:\n{text}");
    assert!(text.contains("6.0"), "missing edge 6.0, got:\n{text}");

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

    assert!(
        text.contains("BinnedPoissonProcess("),
        "missing BinnedPoissonProcess(, got:\n{text}"
    );
    assert!(
        text.contains("flat_fn"),
        "missing flat_fn ref, got:\n{text}"
    );
    assert!(text.contains("Lebesgue"), "missing Lebesgue, got:\n{text}");
    assert!(text.contains("4.0"), "missing edge 4.0, got:\n{text}");

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

    assert!(
        text.contains("normalize(weighted(functionof(polynomial("),
        "missing normalize(weighted(functionof(polynomial(, got:\n{text}"
    );
    assert!(text.contains("c1"), "missing c1 param, got:\n{text}");
    assert!(text.contains("0.5"), "missing literal 0.5, got:\n{text}");
    assert!(text.contains("Lebesgue"), "missing Lebesgue, got:\n{text}");
    assert!(
        text.contains("p_obs"),
        "missing p_obs variate relabel, got:\n{text}"
    );
    assert!(text.contains("relabel"), "missing relabel, got:\n{text}");

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

    assert!(
        text.contains("broadcast(Poisson,"),
        "missing broadcast(Poisson,, got:\n{text}"
    );
    assert!(text.contains("relabel"), "missing relabel, got:\n{text}");
    assert!(text.contains("bb_obs0"), "missing bb_obs0, got:\n{text}");
    assert!(text.contains("bb_obs1"), "missing bb_obs1, got:\n{text}");
    assert!(text.contains("bb_obs2"), "missing bb_obs2, got:\n{text}");
    assert!(
        text.contains("e1"),
        "missing e1 expected param, got:\n{text}"
    );
    assert!(
        text.contains("10.0") || text.contains("10"),
        "missing 10.0 literal, got:\n{text}"
    );

    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}
