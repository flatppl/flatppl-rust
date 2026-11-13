//! Spec §06: `weighted(f, M)` has density `f(x)` times `M`'s density, which is zero
//! outside `M`'s support, so the product is zero there whatever `f` is. A weight
//! that is NaN outside the support (a square root of a negative argument) must not
//! turn the −∞ log-density into NaN: the weight is applied inside the support gate,
//! at a point inside the support.
use flatppl_determinizer::determinize_with_roots;
use flatppl_infer::ModuleBundle;

fn lp(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let root = m.intern("lp");
    let out = determinize_with_roots(&m, &ModuleBundle::new(), Some(&[root])).expect("must lower");
    let text = flatppl_flatpir::write(&out);
    let start = text.find("(%bind lp").expect("lp binding");
    text[start..].lines().next().unwrap().to_string()
}

#[test]
fn a_weight_outside_the_support_is_not_evaluated_there() {
    for op in ["weighted", "logweighted"] {
        let outside = lp(&format!(
            "M = {op}(x -> sqrt(x - 1.0), Lebesgue(support = interval(1.0, 2.0)))\n\
             lp = logdensityof(M, 0.5)"
        ));
        assert!(
            !outside.contains("(sqrt (sub 0.5"),
            "{op}: weight read the raw point: {outside}"
        );
        assert!(
            outside.contains("(neg inf)"),
            "{op}: no -inf gate: {outside}"
        );
    }
}

#[test]
fn a_record_weight_is_gated_by_the_box_and_reads_a_point_inside_it() {
    let outside = lp(
        "S = cartprod(x = interval(1.0, 2.0), y = interval(0.0, 1.0))\n\
         M = weighted((x, y) -> sqrt(x - 1.0) + y, Lebesgue(support = S))\n\
         lp = logdensityof(M, record(x = 0.5, y = 0.25))",
    );
    assert!(
        !outside.contains("(sqrt (sub 0.5"),
        "weight read the raw point: {outside}"
    );
    assert!(outside.contains("(neg inf)"), "no -inf gate: {outside}");
}
