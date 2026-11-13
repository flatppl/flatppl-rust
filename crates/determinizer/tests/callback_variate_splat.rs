//! Spec §04 "Explicit callback parameters" in the lowering rules: `weighted(f, M)`,
//! `logweighted(f, M)` and `pushfwd(f, M)` apply `f(x)` to the whole variate, and
//! a record variate splats into an `f` of several inputs (§04 "Calling
//! conventions"). Each splatted weight equals the same weight reading the record
//! fields by name. The query keeps only the ancestors of `lp` (§13 "Retained
//! subgraph"), so a named callback and its body bindings drop on either path.
use flatppl_determinizer::determinize_with_roots;
use flatppl_infer::ModuleBundle;

mod common;
use common::renumber_points;

const BOX: &str = "x = elementof(interval(0.0, 1.0))\n\
                   y = elementof(interval(0.0, 2.0))\n\
                   S = cartprod(x = interval(0.0, 1.0), y = interval(0.0, 2.0))\n\
                   B = Lebesgue(support = S)\n";
const POINT: &str = "lp = logdensityof(W, record(x = 0.5, y = 1.5))";

fn pir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let lp = m.intern("lp");
    renumber_points(&flatppl_flatpir::write(
        &determinize_with_roots(&m, &ModuleBundle::new(), Some(&[lp])).expect("must lower"),
    ))
}

#[test]
fn the_splatted_weight_reads_each_input_from_its_field() {
    for op in ["weighted", "logweighted"] {
        let whole = pir(&format!("{BOX}W = {op}(r -> r.x * r.y, B)\n{POINT}"));
        for weight in [
            "r = elementof(S)\nrxy = r.x * r.y\nh = functionof(rxy, r)\nW = {op}(h, B)",
            "xy = x * y\nf = functionof(xy, x, y)\nW = {op}(f, B)",
            "W = {op}((x, y) -> x * y, B)",
            "W = {op}((y, x) -> x * y, B)",
        ] {
            let splat = pir(&format!("{BOX}{}\n{POINT}", weight.replace("{op}", op)));
            assert_eq!(splat, whole, "{op}: {weight}");
        }
    }
}
