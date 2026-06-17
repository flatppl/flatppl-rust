//! Integration tests for HS3 paper appendix examples A.1 and A.2.
use flatppl_syntax::{Syntax, parse, print_with};

const FIXTURE_GAUSSIAN: &str = include_str!("fixtures/paper_gaussian.json");
const FIXTURE_PRODUCT: &str = include_str!("fixtures/paper_product.json");
const FIXTURE_HISTFACTORY: &str = include_str!("fixtures/paper_histfactory.json");

/// HS3 paper A.1: single gaussian_dist + unbinned data (1 entry) + sigma const:true.
#[test]
fn paper_gaussian_converts() {
    let m =
        flatppl_hs3::read(FIXTURE_GAUSSIAN).expect("paper_gaussian.json must parse and convert");
    let text = print_with(&m, Syntax::Minimal);

    assert!(text.contains("Normal"), "missing Normal, got:\n{text}");
    assert!(text.contains("relabel"), "missing relabel, got:\n{text}");
    // The single unbinned observation 1.27 is observed as a record keyed by the
    // distribution's variate name `x` (the model is `relabel(Normal, ["x"])`, a
    // record-shaped measure, so the observation must match its axes).
    assert!(
        text.contains("obs_gaussian_channel = record(x = 1.27)"),
        "observed-data record mismatch (expected record(x = 1.27)), got:\n{text}"
    );
    assert!(
        text.contains("elementof"),
        "missing elementof (free param), got:\n{text}"
    );
    // sigma is const:true -> fixed(...)
    assert!(text.contains("fixed"), "missing fixed(sigma), got:\n{text}");
    // likelihoodof wiring
    assert!(
        text.contains("likelihoodof"),
        "missing likelihoodof, got:\n{text}"
    );

    // Round-trip: emitted FlatPPL must re-parse without error.
    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "emitted FlatPPL must re-parse cleanly: {:?}\n\nEmitted text:\n{text}",
        parsed.err()
    );
}

/// HS3 paper A.3: native histfactory_dist (3 samples, normsys + normfactor +
/// staterror) + binned observation.
#[test]
fn paper_histfactory_converts() {
    let module = flatppl_hs3::read(FIXTURE_HISTFACTORY).expect("read failed");
    let text = print_with(&module, Syntax::Minimal);
    eprintln!("=== A.3 conversion ===\n{text}\n=== end ===");

    assert!(text.contains("Poisson"), "missing Poisson, got:\n{text}");
    assert!(
        text.contains("hepphys.interp_poly6_exp(") || text.contains("interp_"),
        "missing normsys interp, got:\n{text}"
    );
    assert!(
        text.contains("Normal"),
        "missing Normal (aux), got:\n{text}"
    );
    assert!(
        text.contains("joint_likelihood"),
        "missing joint_likelihood, got:\n{text}"
    );
    // Observed bin contents [122.0, 112.0], in order, on the main Poisson term.
    // Pin the exact bracketed vector so a reordered observation array fails.
    assert!(
        text.contains("likelihoodof(obs_model_model_channel1, [122.0, 112.0])"),
        "observed-data likelihood mismatch (expected [122.0, 112.0]), got:\n{text}"
    );
    assert!(!text.contains("fn("), "not point-free, got:\n{text}");

    // staterror aux: ROOT-default Poisson (Barlow–Beeston) constraint on the
    // per-bin mcstat scales, emitted as a ContinuedPoisson. Numerical
    // conformance vs ROOT/pyhf is covered in the flatppl-js cross-engine suite.
    assert!(
        text.contains("hepphys.ContinuedPoisson") && text.contains("mcstat"),
        "expected a ContinuedPoisson staterror constraint on mcstat, got:\n{text}"
    );

    // Round-trip parse.
    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted text:\n{text}",
        parsed.err()
    );
}

/// HS3 paper A.2: product_dist (g1,g2) + unbinned data (10 entries).
#[test]
fn paper_product_converts() {
    let m = flatppl_hs3::read(FIXTURE_PRODUCT).expect("paper_product.json must parse and convert");
    let text = print_with(&m, Syntax::Minimal);

    // product_dist over the SAME observable x → normalized pointwise density
    // product (§12), not a joint.
    assert!(
        text.contains(
            "prod = normalize(logweighted(functionof(logdensityof(g2, _x_), x = _x_), g1))"
        ),
        "shared-variate product_dist lowering mismatch, got:\n{text}"
    );
    assert!(
        !text.contains("joint("),
        "same-variate product must not lower to joint, got:\n{text}"
    );
    // two Normal distributions (g1 and g2)
    let normal_count = text.matches("Normal").count();
    assert!(
        normal_count >= 2,
        "expected at least 2 Normal calls, got {normal_count}:\n{text}"
    );
    // likelihoodof from likelihood
    assert!(
        text.contains("likelihoodof"),
        "missing likelihoodof, got:\n{text}"
    );
    // The 10 unbinned toy-data entries become the `toy` vector, in order, fed to
    // the product likelihood. Pin the exact bracketed binding RHS (a reordered or
    // truncated array fails) plus the wiring into likelihoodof.
    assert!(
        text.contains(
            "toy = [-0.028567328469794265, -0.0975895992436726, 0.8301414329794277, \
             -0.18001364208465098, 0.8853988033587967, -0.2791754160017632, 1.168603380508273, \
             2.290388749097474, 0.18297688463530193, 1.8448742587493427]"
        ),
        "toy-data vector mismatch, got:\n{text}"
    );
    // 10 unbinned entries over one observable = N iid observations: the model is
    // plated `iid(prod, 10)`, observed against the bare toy vector.
    assert!(
        text.contains("likelihood = likelihoodof(iid(prod, 10), toy)"),
        "toy-data likelihood wiring mismatch, got:\n{text}"
    );

    // Round-trip: emitted FlatPPL must re-parse without error.
    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "emitted FlatPPL must re-parse cleanly: {:?}\n\nEmitted text:\n{text}",
        parsed.err()
    );
}
