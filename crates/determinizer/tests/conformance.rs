use flatppl_determinizer::is_flatpdl;

fn infer_module(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

#[test]
fn deterministic_model_is_flatpdl() {
    // pure deterministic arithmetic — no measure, no draw
    let m = infer_module("x = elementof(reals)\ny = add(x, 1.0)");
    assert!(is_flatpdl(&m).is_ok(), "{:?}", is_flatpdl(&m));
}

#[test]
fn a_draw_is_not_flatpdl() {
    // a stochastic node violates the predicate (Stochastic phase / Measure-typed law)
    let m = infer_module("z = draw(Normal(mu = 0.0, sigma = 1.0))");
    let v = is_flatpdl(&m).unwrap_err();
    assert!(
        v.iter().any(|n| matches!(
            n.kind,
            flatppl_determinizer::NonConformKind::StochasticPhase
                | flatppl_determinizer::NonConformKind::MeasureTyped
        )),
        "expected a stochastic/measure violation; got: {v:?}"
    );
}
