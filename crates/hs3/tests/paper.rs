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
    // observed value from unbinned data entry
    assert!(
        text.contains("1.27"),
        "missing observed value 1.27, got:\n{text}"
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
    assert!(
        text.contains("122.0") || text.contains("122"),
        "missing observed bin0, got:\n{text}"
    );
    assert!(
        text.contains("112.0") || text.contains("112"),
        "missing observed bin1, got:\n{text}"
    );
    assert!(!text.contains("fn("), "not point-free, got:\n{text}");

    // staterror deltas: bin0 = 5/100 = 0.05, bin1 = 10/100 = 0.1
    assert!(
        text.contains("0.05"),
        "missing staterror delta 0.05, got:\n{text}"
    );
    assert!(
        text.contains("0.1"),
        "missing staterror delta 0.1, got:\n{text}"
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

    // joint(...) from product_dist
    assert!(
        text.contains("joint("),
        "missing joint( from product_dist, got:\n{text}"
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
    // toy data values should appear (at least one of the 10 entries)
    // the last entry is 1.8448742587493427; check for a distinctive value
    assert!(
        text.contains("1.8448742587493427") || text.contains("0.8301414"),
        "missing toy data values, got:\n{text}"
    );

    // Round-trip: emitted FlatPPL must re-parse without error.
    let parsed = parse(&text);
    assert!(
        parsed.is_ok(),
        "emitted FlatPPL must re-parse cleanly: {:?}\n\nEmitted text:\n{text}",
        parsed.err()
    );
}
