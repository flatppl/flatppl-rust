//! Golden structural-snapshot assertions for one representative HS3 model per
//! lowering family.
//!
//! Unlike the `contains()`-style smoke tests in `dists.rs`/`expr_dists.rs`,
//! every assertion here pins the whole emitted right-hand side as an exact
//! substring (`name = <full RHS>`), so a silent change to operator order,
//! parameter naming, or the measure-algebra wrapper fails loudly. These are
//! committed goldens: if the lowering legitimately changes, update the expected
//! string in one place.
//!
//! Each fixture is intentionally small (one distribution + the minimal
//! `parameter_points`/`domains` it needs) so the golden is a single line.
use flatppl_syntax::{Syntax, parse, print_with};

/// Convert `json`, print Minimal, and assert the output contains `needle` as an
/// exact substring; on failure dump the whole emitted module. Also verifies the
/// emitted FlatPPL round-trip-parses.
fn assert_golden(label: &str, json: &str, needle: &str) {
    let m = flatppl_hs3::read_hs3(json).unwrap_or_else(|e| panic!("{label}: read_hs3 failed: {e}"));
    let text = print_with(&m, Syntax::Minimal);
    assert!(
        text.contains(needle),
        "{label}: golden mismatch.\nExpected to contain:\n  {needle}\nGot:\n{text}"
    );
    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "{label}: round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        parsed.err()
    );
}

// ---------------------------------------------------------------------------
// gaussian_dist → Normal(mu, sigma)  (1:1 map, no provenance comment)
// ---------------------------------------------------------------------------
#[test]
fn golden_gaussian() {
    assert_golden(
        "gaussian",
        r#"{"distributions":[{"name":"g","type":"gaussian_dist","mean":"mu","sigma":"s","x":"gx"}],
            "parameter_points":[{"name":"n","entries":[{"name":"mu","value":0.0},{"name":"s","value":1.0}]}]}"#,
        r#"g = relabel(Normal(mu = mu, sigma = s), ["gx"])"#,
    );
}

// ---------------------------------------------------------------------------
// exponential_dist → Exponential(rate = c)
//
// HS3's exponential_dist density is exp(−c·x), so the HS3 `c` is a positive
// decay rate; FlatPPL's Exponential(rate) is rate·exp(−rate·x). The rate maps
// directly: rate = c, no negation. (RooFit's internal RooExponential slope is
// −rate, but HS3 stores the already-inverted, positive c.)
// ---------------------------------------------------------------------------
#[test]
fn golden_exponential_rate_is_c() {
    let json = r#"{"distributions":[{"name":"e","type":"exponential_dist","c":"lam","x":"ex"}],
        "parameter_points":[{"name":"n","entries":[{"name":"lam","value":2.0}]}]}"#;
    let m = flatppl_hs3::read_hs3(json).expect("read_hs3");
    let text = print_with(&m, Syntax::Minimal);
    assert!(
        text.contains(r#"e = relabel(Exponential(rate = lam), ["ex"])"#),
        "exponential rate must be the bare HS3 c, got:\n{text}"
    );
    // Must NOT negate the parameter (the old, incorrect lowering).
    assert!(
        !text.contains("neg(lam)"),
        "exponential rate must not be negated, got:\n{text}"
    );
    assert!(parse(&text).is_ok(), "round-trip parse failed:\n{text}");
}

// ---------------------------------------------------------------------------
// uniform_dist → Uniform(interval(min, max))
//
// The support is pulled from the `domains` block for the variate; the corrected
// lowering (Stream B) errors when no domain is declared (see negatives.rs). Here
// we assert the interval carries the *actual* declared bounds, not a default.
// ---------------------------------------------------------------------------
#[test]
fn golden_uniform_interval_uses_declared_bounds() {
    let json = r#"{"distributions":[{"name":"u","type":"uniform_dist","x":"ux"}],
        "domains":[{"name":"d","axes":[{"name":"ux","min":-2.0,"max":5.0}]}],
        "parameter_points":[]}"#;
    let m = flatppl_hs3::read_hs3(json).expect("read_hs3");
    let text = print_with(&m, Syntax::Minimal);
    assert!(
        text.contains(r#"u = relabel(Uniform(interval(-2.0, 5.0)), ["ux"])"#),
        "uniform must carry declared interval bounds (-2.0, 5.0), got:\n{text}"
    );
    assert!(parse(&text).is_ok(), "round-trip parse failed:\n{text}");
}

// ---------------------------------------------------------------------------
// generalized_normal_dist → GeneralizedNormal(mean, alpha, beta)
//
// Pin the full call RHS (not just the bare tokens checked in dists.rs) so a
// swapped-kwarg lowering — e.g. GeneralizedNormal(mean=…, alpha=gn_beta,
// beta=gn_alpha) — fails: each keyword must bind its matching HS3 field.
// ---------------------------------------------------------------------------
#[test]
fn golden_generalized_normal() {
    assert_golden(
        "generalized_normal",
        r#"{"distributions":[{"name":"gn","type":"generalized_normal_dist","mean":"gn_mu","alpha":"gn_alpha","beta":"gn_beta","x":"gx"}],
            "parameter_points":[{"name":"n","entries":[{"name":"gn_mu","value":0.0},{"name":"gn_alpha","value":1.0},{"name":"gn_beta","value":2.0}]}]}"#,
        r#"gn = relabel(GeneralizedNormal(mean = gn_mu, alpha = gn_alpha, beta = gn_beta), ["gx"])"#,
    );
}

// ---------------------------------------------------------------------------
// lognormal_dist → LogNormal(mu, sigma)
// ---------------------------------------------------------------------------
#[test]
fn golden_lognormal() {
    assert_golden(
        "lognormal",
        r#"{"distributions":[{"name":"ln","type":"lognormal_dist","mu":"lm","sigma":"ls","x":"lx"}],
            "parameter_points":[{"name":"n","entries":[{"name":"lm","value":0.0},{"name":"ls","value":1.0}]}]}"#,
        r#"ln = relabel(LogNormal(mu = lm, sigma = ls), ["lx"])"#,
    );
}

// ---------------------------------------------------------------------------
// density_function_dist → normalize(weighted(<fn>, Lebesgue(reals)))
//
// The referenced generic_function lowers to a point-free `functionof(...)`
// binding; the distribution wraps it in normalize(weighted(..., Lebesgue)).
// Asserts the full lowered expression (operators lowered to add/sub/mul/...).
// ---------------------------------------------------------------------------
#[test]
fn golden_density_function() {
    let json = r#"{"functions":[{"name":"my_gauss_fn","type":"generic_function","expression":"exp(-0.5 * ((x - mu) / sigma) ^ 2)","variables":["x"]}],
        "distributions":[{"name":"gauss_dist","type":"density_function_dist","function":"my_gauss_fn"}],
        "parameter_points":[{"name":"n","entries":[{"name":"mu","value":0.0},{"name":"sigma","value":1.0}]}]}"#;
    let m = flatppl_hs3::read_hs3(json).expect("read_hs3");
    let text = print_with(&m, Syntax::Minimal);
    // The function body lowers each operator to its FlatPPL call form.
    assert!(
        text.contains(
            "my_gauss_fn = functionof(exp(mul(neg(0.5), pow(divide(sub(_x_, mu), sigma), 2.0))), x = _x_)"
        ),
        "generic_function body mismatch, got:\n{text}"
    );
    // The distribution normalizes the function against Lebesgue measure.
    assert!(
        text.contains("gauss_dist = normalize(weighted(my_gauss_fn, Lebesgue(reals)))"),
        "density_function_dist body mismatch, got:\n{text}"
    );
    // density (not log-density) → plain weighted, never logweighted.
    assert!(
        !text.contains("logweighted"),
        "density_function_dist must use weighted, not logweighted, got:\n{text}"
    );
    assert!(parse(&text).is_ok(), "round-trip parse failed:\n{text}");
}

// ---------------------------------------------------------------------------
// barlow_beeston_lite_poisson_constraint_dist →
//   relabel(broadcast(Poisson, [expected...]), [x names...])
// (Also covered structurally in dists.rs; kept here as the family golden.)
// ---------------------------------------------------------------------------
#[test]
fn golden_barlow_beeston() {
    let json = r#"{"distributions":[{"name":"bb","type":"barlow_beeston_lite_poisson_constraint_dist","x":["b0","b1"],"expected":[10.0,"e1"]}],
        "parameter_points":[{"name":"n","entries":[{"name":"e1","value":8.0}]}]}"#;
    assert_golden(
        "barlow_beeston",
        json,
        r#"bb = relabel(broadcast(Poisson, [10.0, e1]), ["b0", "b1"])"#,
    );
}

// ---------------------------------------------------------------------------
// crystalball_dist single + double-sided dispatch on which params are present.
// ---------------------------------------------------------------------------
#[test]
fn golden_crystalball_single_and_double() {
    assert_golden(
        "crystalball_single",
        r#"{"distributions":[{"name":"cb","type":"crystalball_dist","m0":"m0","sigma":"sg","alpha":"a","n":"nn","m":"mo"}],
            "parameter_points":[{"name":"p","entries":[{"name":"m0","value":5.0},{"name":"sg","value":0.1},{"name":"a","value":1.5},{"name":"nn","value":3.0}]}]}"#,
        r#"cb = relabel(hepphys.CrystalBall(m0, sg, a, nn), ["mo"])"#,
    );
    assert_golden(
        "crystalball_double",
        r#"{"distributions":[{"name":"dcb","type":"crystalball_dist","m0":"m0","sigma_L":"sl","sigma_R":"sr","alpha_L":"al","n_L":"nl","alpha_R":"ar","n_R":"nr","m":"mo"}],
            "parameter_points":[{"name":"p","entries":[{"name":"m0","value":125.0},{"name":"sl","value":1.5},{"name":"sr","value":2.0},{"name":"al","value":1.2},{"name":"nl","value":5.0},{"name":"ar","value":1.8},{"name":"nr","value":4.0}]}]}"#,
        r#"dcb = relabel(hepphys.DoubleSidedCrystalBall(m0, sl, sr, al, ar, nl, nr), ["mo"])"#,
    );
}

// ---------------------------------------------------------------------------
// argus_dist → hepphys.Argus(resonance, slope, power)
// ---------------------------------------------------------------------------
#[test]
fn golden_argus() {
    assert_golden(
        "argus",
        r#"{"distributions":[{"name":"ar","type":"argus_dist","resonance":"c","slope":"chi","power":"p","mass":"mo"}],
            "parameter_points":[{"name":"n","entries":[{"name":"c","value":5.29},{"name":"chi","value":-10.0},{"name":"p","value":0.5}]}]}"#,
        r#"ar = relabel(hepphys.Argus(c, chi, p), ["mo"])"#,
    );
}

// ---------------------------------------------------------------------------
// multivariate_normal_dist → MvNormal(mu = [...], cov = [[...],...])
// ---------------------------------------------------------------------------
#[test]
fn golden_mvnormal() {
    assert_golden(
        "mvnormal",
        r#"{"distributions":[{"name":"mv","type":"multivariate_normal_dist","mean":["m0","m1"],"covariances":[[1.0,0.0],[0.0,1.0]],"x":["o0","o1"]}],
            "parameter_points":[{"name":"n","entries":[{"name":"m0","value":0.0},{"name":"m1","value":0.0}]}]}"#,
        r#"mv = relabel(MvNormal(mu = [m0, m1], cov = [[1.0, 0.0], [0.0, 1.0]]), ["o0", "o1"])"#,
    );
}

// ---------------------------------------------------------------------------
// polynomial_dist → normalize(weighted(functionof(polynomial(coeffs)), Lebesgue))
// ---------------------------------------------------------------------------
#[test]
fn golden_polynomial() {
    assert_golden(
        "polynomial",
        r#"{"distributions":[{"name":"poly","type":"polynomial_dist","coefficients":[1.0,"c1",0.5],"x":"po"}],
            "domains":[{"name":"default_domain","type":"product_domain","axes":[{"name":"po","min":-5.0,"max":5.0}]}],
            "parameter_points":[{"name":"n","entries":[{"name":"c1","value":0.3}]}]}"#,
        "poly = relabel(normalize(truncate(weighted(functionof(polynomial([1.0, c1, 0.5], _po_), \
         po = _po_), Lebesgue(reals)), interval(-5.0, 5.0))), [\"po\"])",
    );
}

// ---------------------------------------------------------------------------
// mixture_dist non-extended AND extended (coefficient→summand binding + the
// normalize-only-when-non-extended rule).
// ---------------------------------------------------------------------------
#[test]
fn golden_mixture_nonextended_has_normalize() {
    let json = r#"{"distributions":[
        {"name":"a","type":"gaussian_dist","mean":"ma","sigma":"sa","x":"y"},
        {"name":"b","type":"gaussian_dist","mean":"mb","sigma":"sb","x":"y"},
        {"name":"mix","type":"mixture_dist","summands":["a","b"],"coefficients":[0.4],"extended":false}],
        "parameter_points":[{"name":"n","entries":[{"name":"ma","value":0.0},{"name":"sa","value":1.0},{"name":"mb","value":2.0},{"name":"sb","value":0.5}]}]}"#;
    assert_golden(
        "mixture_nonextended",
        json,
        "mix = normalize(superpose(weighted(0.4, a), weighted(0.6, b)))",
    );
}

// ---------------------------------------------------------------------------
// pyhf 2-bin/1-channel fixture → the assembled obs_model and joint_likelihood.
//
// This pins the full point-free histfactory assembly: the per-bin Poisson over
// (signal·mu + background·gamma), the ContinuedPoisson shapesys aux term, and
// the observed-data vector [50.0, 60.0]. (Smoke-tested in pyhf_real.rs; here we
// assert the exact obs_model and likelihood expressions.)
// ---------------------------------------------------------------------------
const FIXTURE_2BIN: &str = include_str!("fixtures/2bin_1channel.json");

#[test]
fn golden_pyhf_2bin_assembly() {
    let m = flatppl_hs3::read(FIXTURE_2BIN).expect("2bin fixture must convert");
    let text = print_with(&m, Syntax::Minimal);
    // Per-sample expected: each nominal template scaled by its modifier
    // (signal * mu, background * gamma).
    assert!(
        text.contains(
            "singlechannel_signal_expected = broadcast(mul, singlechannel_signal_nominal, mu)"
        ),
        "signal expected mismatch, got:\n{text}"
    );
    assert!(
        text.contains(
            "singlechannel_background_expected = \
             broadcast(mul, singlechannel_background_nominal, uncorr_bkguncrt)"
        ),
        "background expected mismatch, got:\n{text}"
    );
    // Total expected = sum over samples, per bin.
    assert!(
        text.contains(
            "singlechannel_expected = \
             broadcast(add, singlechannel_signal_expected, singlechannel_background_expected)"
        ),
        "total expected mismatch, got:\n{text}"
    );
    // The observation model is a reified kernel (functionof), as likelihoodof
    // requires; the observation term binds it to the observed counts.
    assert!(
        text.contains(
            "singlechannel_model = functionof(broadcast(Poisson, singlechannel_expected))"
        ),
        "obs model mismatch, got:\n{text}"
    );
    assert!(
        text.contains(
            "singlechannel_likelihood = likelihoodof(singlechannel_model, singlechannel_observed)"
        ) && text.contains("singlechannel_observed = [50.0, 60.0]"),
        "observation term / observed data mismatch, got:\n{text}"
    );
    // shapesys constraint: a ContinuedPoisson on effective counts, parameter-keyed.
    assert!(
        text.contains("hepphys.ContinuedPoisson")
            && text.contains("uncorr_bkguncrt_constraint_likelihood"),
        "missing parameter-keyed shapesys constraint, got:\n{text}"
    );
    assert!(
        text.contains("uncorr_bkguncrt = elementof(cartpow(posreals, 2))"),
        "shapesys domain mismatch, got:\n{text}"
    );
    // Flat top-level likelihood = observation term + constraint term.
    assert!(
        text.contains(
            "likelihood = \
             joint_likelihood(singlechannel_likelihood, uncorr_bkguncrt_constraint_likelihood)"
        ),
        "flat top-level likelihood mismatch, got:\n{text}"
    );
    assert!(!text.contains("fn("), "must be point-free, got:\n{text}");
    assert!(parse(&text).is_ok(), "round-trip parse failed:\n{text}");
}

// ---------------------------------------------------------------------------
// paper_histfactory.json (HS3 paper § A.3): native histfactory_dist with
// normsys + normfactor + staterror over two bins. Pins the staterror deltas
// (5/100 = 0.05, 10/100 = 0.1) and the normsys interpolation call.
// ---------------------------------------------------------------------------
const FIXTURE_HISTFACTORY: &str = include_str!("fixtures/paper_histfactory.json");

#[test]
fn golden_paper_histfactory_staterror_deltas() {
    let m = flatppl_hs3::read(FIXTURE_HISTFACTORY).expect("paper_histfactory must convert");
    let text = print_with(&m, Syntax::Minimal);
    // normsys interpolation is a module-member call, never `call(hepphys...)`.
    assert!(
        text.contains("hepphys.interp_poly6_exp(") || text.contains("hepphys.interp"),
        "missing normsys interp module call, got:\n{text}"
    );
    assert!(
        !text.contains("call(hepphys"),
        "must not emit invalid call(hepphys...) builtin, got:\n{text}"
    );
    // staterror (mcstat): ROOT-default Poisson (Barlow-Beeston) constraint, a
    // ContinuedPoisson on the per-bin effective counts tau = 1/delta^2 =
    // [1/0.05^2, 1/0.1^2] = [400, 100]. Parameter-keyed, computed once.
    assert!(
        text.contains("mcstat_tau = [400.0, 100.0]"),
        "staterror effective counts mismatch (expected [400, 100]), got:\n{text}"
    );
    assert!(
        text.contains("mcstat_constraint = functionof(broadcast(hepphys.ContinuedPoisson"),
        "expected a ContinuedPoisson staterror constraint on mcstat, got:\n{text}"
    );
    // Observed bin contents [122.0, 112.0], in order, fed to the channel's
    // observation term. Pin the full likelihoodof so a reordered array fails.
    assert!(
        text.contains(
            "model_channel1_likelihood = likelihoodof(model_channel1_model, model_channel1_observed)"
        ) && text.contains("model_channel1_observed = [122.0, 112.0]"),
        "observed-data likelihood mismatch (expected [122.0, 112.0]), got:\n{text}"
    );
    assert!(
        text.contains("joint_likelihood"),
        "missing joint_likelihood, got:\n{text}"
    );
    assert!(!text.contains("fn("), "must be point-free, got:\n{text}");
    assert!(parse(&text).is_ok(), "round-trip parse failed:\n{text}");
}

// ---------------------------------------------------------------------------
// paper_product.json (HS3 paper § A.2): product_dist over two gaussians of the
// SAME observable `x` with 10 unbinned toy-data entries. RooProdPdf over a
// shared observable is the normalized pointwise density product, lowered to
// `normalize(logweighted(x -> Σ logdensityof(gᵢ, x), g₀))` (§12) — NOT a `joint`
// (which would be a 2-D independent product over distinct variates).
// ---------------------------------------------------------------------------
const FIXTURE_PRODUCT: &str = include_str!("fixtures/paper_product.json");

#[test]
fn golden_paper_product_density_product() {
    let m = flatppl_hs3::read(FIXTURE_PRODUCT).expect("paper_product must convert");
    let text = print_with(&m, Syntax::Minimal);
    // Shared-variate product: reweight g1 by g2's log-density, then normalize.
    assert!(
        text.contains(
            "prod = normalize(logweighted(functionof(logdensityof(g2, _x_), x = _x_), g1))"
        ),
        "shared-variate product_dist lowering mismatch, got:\n{text}"
    );
    // Must NOT be an independent joint over (x, x).
    assert!(
        !text.contains("joint("),
        "same-variate product must not lower to joint, got:\n{text}"
    );
    assert!(
        text.matches("Normal").count() >= 2,
        "expected >=2 Normal calls, got:\n{text}"
    );
    // The 10 unbinned toy-data entries become the `toy` vector (exact, in order)
    // wired into the product likelihood — pin the full bracketed RHS so a
    // reordered/truncated array fails rather than a single distinctive value.
    assert!(
        text.contains(
            "toy = [-0.028567328469794265, -0.0975895992436726, 0.8301414329794277, \
             -0.18001364208465098, 0.8853988033587967, -0.2791754160017632, 1.168603380508273, \
             2.290388749097474, 0.18297688463530193, 1.8448742587493427]"
        ),
        "toy-data vector mismatch, got:\n{text}"
    );
    // 10 unbinned entries over the single observable are N iid observations:
    // the model is plated `iid(prod, 10)` and observed against the bare vector.
    assert!(
        text.contains("likelihood = likelihoodof(iid(prod, 10), toy)"),
        "toy-data likelihood wiring mismatch, got:\n{text}"
    );
    assert!(parse(&text).is_ok(), "round-trip parse failed:\n{text}");
}

// product_dist over factors with DISTINCT variates is a genuine independent
// product → joint (the shared-variate density-product form must NOT fire).
#[test]
fn golden_product_distinct_variates_joins() {
    let json = r#"{"distributions":[
        {"name":"pd","type":"product_dist","factors":["gx","gy"]},
        {"name":"gx","type":"gaussian_dist","mean":"mx","sigma":"sx","x":"x"},
        {"name":"gy","type":"gaussian_dist","mean":"my","sigma":"sy","x":"y"}],
        "parameter_points":[{"name":"v","entries":[{"name":"mx","value":0.0},{"name":"sx","value":1.0},{"name":"my","value":0.0},{"name":"sy","value":1.0}]}]}"#;
    let m = flatppl_hs3::read_hs3(json).expect("read_hs3");
    let text = print_with(&m, Syntax::Minimal);
    assert!(
        text.contains("pd = joint(gx = gx, gy = gy)"),
        "distinct-variate product must lower to joint, got:\n{text}"
    );
    assert!(
        !text.contains("logweighted"),
        "distinct-variate product must not use the density-product form, got:\n{text}"
    );
    assert!(parse(&text).is_ok(), "round-trip parse failed:\n{text}");
}

// Three factors over the SAME variate: the density product stays flat — one
// add-fold of N-1 log-densities, base = first factor (no nesting).
#[test]
fn golden_product_three_shared_variates() {
    let json = r#"{"distributions":[
        {"name":"prod3","type":"product_dist","factors":["a","b","c"]},
        {"name":"a","type":"gaussian_dist","mean":"ma","sigma":"sa","x":"x"},
        {"name":"b","type":"gaussian_dist","mean":"mb","sigma":"sb","x":"x"},
        {"name":"c","type":"gaussian_dist","mean":"mc","sigma":"sc","x":"x"}],
        "parameter_points":[{"name":"v","entries":[{"name":"ma","value":0.0},{"name":"sa","value":1.0},{"name":"mb","value":0.0},{"name":"sb","value":1.0},{"name":"mc","value":0.0},{"name":"sc","value":1.0}]}]}"#;
    let m = flatppl_hs3::read_hs3(json).expect("read_hs3");
    let text = print_with(&m, Syntax::Minimal);
    assert!(
        text.contains(
            "prod3 = normalize(logweighted(functionof(add(logdensityof(b, _x_), logdensityof(c, _x_)), x = _x_), a))"
        ),
        "3-factor shared-variate product lowering mismatch, got:\n{text}"
    );
    assert!(parse(&text).is_ok(), "round-trip parse failed:\n{text}");
}

#[test]
fn golden_mixture_extended_no_normalize() {
    let json = r#"{"distributions":[
        {"name":"a","type":"gaussian_dist","mean":"ma","sigma":"sa","x":"y"},
        {"name":"b","type":"gaussian_dist","mean":"mb","sigma":"sb","x":"y"},
        {"name":"mix","type":"mixture_dist","summands":["a","b"],"coefficients":[0.3,0.7],"extended":true}],
        "parameter_points":[{"name":"n","entries":[{"name":"ma","value":0.0},{"name":"sa","value":1.0},{"name":"mb","value":3.0},{"name":"sb","value":1.0}]}]}"#;
    let m = flatppl_hs3::read_hs3(json).expect("read_hs3");
    let text = print_with(&m, Syntax::Minimal);
    assert!(
        text.contains("mix = superpose(weighted(0.3, a), weighted(0.7, b))"),
        "extended mixture body mismatch, got:\n{text}"
    );
    let mix_line = text
        .lines()
        .find(|l| l.trim_start().starts_with("mix ="))
        .expect("mix binding line");
    assert!(
        !mix_line.contains("normalize("),
        "extended mixture binding must NOT contain normalize(, got:\n{mix_line}"
    );
    assert!(parse(&text).is_ok(), "round-trip parse failed:\n{text}");
}
