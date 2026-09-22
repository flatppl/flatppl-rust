//! Observation count must not grow the density body or its expression depth.

use flatppl_determinizer::determinize;

fn lower(n: usize) -> String {
    let src = format!(
        "g = Normal(0.0, 1.0)\n\
         obs = elementof(cartpow(reals, {n}))\n\
         lk = likelihoodof(iid(g, {n}), obs)\n\
         lp = logdensityof(lk, record())"
    );
    let mut m = flatppl_syntax::parse(&src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let out = determinize(&m).expect("must lower");
    flatppl_determinizer::is_flatpdl(&out).expect("output must be FlatPDL");
    flatppl_flatpir::write(&out)
}

#[test]
fn iid_body_stays_constant_across_observation_counts() {
    for n in [1, 2, 3, 17, 1023, 2000, 20000] {
        let pir = lower(n);
        assert_eq!(pir.matches("builtin_logdensityof").count(), 1, "{pir}");
        assert_eq!(pir.matches("(get0 ").count(), 0, "{pir}");
        assert_eq!(pir.matches("(add ").count(), 0, "{pir}");
        assert_eq!(pir.matches("(sum ").count(), 1, "{pir}");
        assert_eq!(pir.matches("(broadcast ").count(), 1, "{pir}");
    }
}
