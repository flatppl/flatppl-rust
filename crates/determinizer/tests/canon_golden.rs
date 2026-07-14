use flatppl_determinizer::determinize;

fn determinize_src(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    determinize(&m).expect("must lower, not refuse")
}

// Constant subexpressions in the determinised FlatPDL fold to a literal:
// scoring a mixture with literal weights leaves `log(0.3 + 0.5)`; const-fold
// reduces the `add(0.3, 0.5)` to `0.8`. Buffy #263 Pass 1.
#[test]
fn const_fold_reduces_literal_sum_of_weights() {
    let src = "\
m = normalize(superpose(\
weighted(0.3, Normal(mu = 0.0, sigma = 1.0)), \
weighted(0.5, Gamma(shape = 2.0, rate = 1.0))))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let pir = flatppl_flatpir::write(&determinize_src(src));
    assert!(
        pir.contains("(log 0.8)"),
        "add(0.3, 0.5) folds to 0.8 under the outer log:\n{pir}"
    );
    assert!(
        !pir.contains("(add 0.3 0.5)"),
        "no un-folded literal sum remains:\n{pir}"
    );
}

// const_fold is idempotent: determinizing the same source twice produces
// identical FlatPIR (canonicalize already ran to a fixpoint inside the first
// determinize).
#[test]
fn const_fold_is_idempotent() {
    let src = "\
m = normalize(superpose(\
weighted(0.3, Normal(mu = 0.0, sigma = 1.0)), \
weighted(0.5, Gamma(shape = 2.0, rate = 1.0))))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let once = flatppl_flatpir::write(&determinize_src(src));
    let twice = flatppl_flatpir::write(&determinize_src(src));
    assert_eq!(
        once, twice,
        "determinize output is a canonicalization fixpoint"
    );
}
