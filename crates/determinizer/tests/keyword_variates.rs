//! Spec §06 keyword forms of `joint` and `jointchain`: each name holds its component's
//! variate whole, so an array component stays an array and a record component nests
//! under its name, from which a later kernel reads it by that name.
//!
//! The oracles are the closed-form sums of the Normal log-densities, scored through
//! flatppl-js on the determinized module: -3.9265252801739634 (nested record),
//! -2.8268155996140187 (array under `joint`) and -3.7507541328186917 (array under
//! `jointchain`), against -3.9265252801739634, -2.826815599614018 and
//! -3.7507541328186904 computed directly.
use flatppl_determinizer::determinize;

fn pir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    flatppl_flatpir::write(&determinize(&m).expect("must lower"))
}

#[test]
fn a_record_component_nests_and_the_kernel_reads_it_by_its_label() {
    // log N(0.5; 0, 1) + log N(0.25; 1, 2) + log N(1.5; 0.5 + 0.25, 1).
    let out = pir(
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\nb = draw(Normal(mu = 1.0, sigma = 2.0))\n\
         M = lawof(record(a = a, b = b))\ntheta = elementof(cartprod(a = reals, b = reals))\n\
         K = functionof(Normal(mu = theta.a + theta.b, sigma = 1.0), theta = theta)\n\
         j = jointchain(theta = M, o = K)\n\
         lp = logdensityof(j, record(theta = record(a = 0.5, b = 0.25), o = 1.5))",
    );
    assert_eq!(out.matches("builtin_logdensityof").count(), 3, "{out}");
    assert!(
        out.contains("(%field mu 0.0) (%field sigma 1.0))) 0.5)")
            && out.contains("(%field mu 1.0) (%field sigma 2.0))) 0.25)")
            && out.contains("(%field mu 0.75) (%field sigma 1.0))) 1.5)"),
        "{out}"
    );
}

#[test]
fn keyword_joint_scores_an_array_component_whole() {
    // The three cells of `obs` are one component: a sum of three Normal log-densities.
    let out = pir("obs = joint(obs = iid(Normal(mu = 0.0, sigma = 1.0), 3))\n\
         lp = logdensityof(obs, record(obs = [0.1, 0.2, 0.3]))");
    assert_eq!(out.matches("builtin_logdensityof").count(), 1, "{out}");
    assert!(out.contains("(sum "), "{out}");
}

#[test]
fn keyword_jointchain_passes_an_array_component_whole_to_the_kernel() {
    // log N(0.1) + log N(0.2) + log N(0.3), then log N(0.7; sum(w) = 0.6, 1).
    let out = pir(
        "v = draw(iid(Normal(mu = 0.0, sigma = 1.0), 3))\nM = lawof(v)\n\
         w = elementof(cartpow(reals, 3))\n\
         K = functionof(Normal(mu = sum(w), sigma = 1.0), w = w)\n\
         j = jointchain(w = M, o = K)\n\
         lp = logdensityof(j, record(w = [0.1, 0.2, 0.3], o = 0.7))",
    );
    assert_eq!(out.matches("builtin_logdensityof").count(), 2, "{out}");
    assert!(out.contains("(sum "), "{out}");
    assert!(out.contains(" 0.7)"), "{out}");
}
