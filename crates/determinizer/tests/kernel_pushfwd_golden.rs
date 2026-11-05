//! A pushforward inside a kernel body lowers through every kernel spelling.
//!
//! §04 reifies the same kernel as `functionof(M, mu = mu)` over a module
//! input, as `functionof(M, mu = _m_)` over a placeholder, as a lambda
//! `mu -> M` or as a named function `K(mu) = M`. A local boundary (placeholder
//! or lambda parameter) is substituted before lowering, which rebuilds the base
//! measure without its annotations. The change of variables still needs the
//! base's variate type and support (§06 "Density convention", "Engine contract
//! for pushfwd density evaluation").

use std::f64::consts::{LN_2, PI};

use flatppl_determinizer::determinize;

mod common;

fn lowered(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let out = determinize(&m).unwrap_or_else(|e| panic!("must lower: {e:?}\n{src}"));
    common::pir_binding(&flatppl_flatpir::write(&out), "lp")
}

/// The four spellings of one kernel, with `mu` its boundary in `body`.
fn spellings(body: &str) -> [(&'static str, String); 4] {
    [
        (
            "functionof",
            format!("mu = elementof(reals)\nK = functionof({body}, mu = mu)\n"),
        ),
        (
            "placeholder",
            format!("K = functionof({}, mu = _m_)\n", body.replace("mu", "_m_")),
        ),
        ("lambda", format!("K = mu -> {body}\n")),
        ("named function", format!("K(mu) = {body}\n")),
    ]
}

const SCALAR_MAPS: [(&str, &str); 5] = [
    (
        "b = bijection(fn(2.0 * _), fn(_ / 2.0), log(2.0))\n",
        "pushfwd(b, Normal(mu, 2.0))",
    ),
    (
        "",
        "pushfwd(bijection(fn(2.0 * _), fn(_ / 2.0), log(2.0)), Normal(mu, 2.0))",
    ),
    ("", "pushfwd(fn(2.0 * _), Normal(mu, 2.0))"),
    ("", "pushfwd(exp, Normal(mu, 1.5))"),
    (
        "h = identity\nb = bijection(exp, log, h)\n",
        "pushfwd(b, Normal(mu, 1.5))",
    ),
];

#[test]
fn every_kernel_spelling_lowers_a_scalar_pushforward_alike() {
    for (prelude, body) in SCALAR_MAPS {
        for use_ in [
            "lp = logdensityof(likelihoodof(K, 3.0), record(mu = 1.0))",
            "lp = logdensityof(K(1.0), 3.0)",
        ] {
            let [(_, reference), rest @ ..] = spellings(body);
            let expected = lowered(&format!("{prelude}{reference}{use_}"));
            for (label, kernel) in rest {
                assert_eq!(
                    lowered(&format!("{prelude}{kernel}{use_}")),
                    expected,
                    "{label}: {body} / {use_}"
                );
            }
        }
    }
}

#[test]
fn every_kernel_spelling_lowers_an_iid_base_pushforward_alike() {
    // f and f_inv map the whole variate, and logvolume is the log-volume
    // element of the whole map: 3 log 2 for y = 2x on R^3.
    let prelude = "b = bijection(fn(2.0 * _), fn(_ / 2.0), 3.0 * log(2.0))\n";
    let body = "pushfwd(b, iid(Normal(mu, 2.0), 3))";
    let use_ = "lp = logdensityof(likelihoodof(K, [3.0, 1.0, -2.0]), record(mu = 1.0))";
    let [(_, reference), rest @ ..] = spellings(body);
    let expected = lowered(&format!("{prelude}{reference}{use_}"));
    for (label, kernel) in rest {
        assert_eq!(
            lowered(&format!("{prelude}{kernel}{use_}")),
            expected,
            "{label}"
        );
    }
}

#[test]
fn a_rebuilt_base_keeps_its_support_and_reference_measure() {
    // `log` needs the base's support inside the positive reals, and the
    // Poisson base has a counting reference: no volume element, a snapped atom.
    for (body, use_) in [
        (
            "pushfwd(log, Gamma(mu, 1.0))",
            "lp = logdensityof(likelihoodof(K, 0.5), record(mu = 2.0))",
        ),
        (
            "pushfwd(fn(_ + 1), Poisson(mu))",
            "lp = logdensityof(likelihoodof(K, 3), record(mu = 2.0))",
        ),
    ] {
        let reference =
            format!("mu = elementof(posreals)\nK = functionof({body}, mu = mu)\n{use_}");
        let expected = lowered(&reference);
        assert_eq!(
            lowered(&format!("K = mu -> {body}\n{use_}")),
            expected,
            "{body}"
        );
        assert_eq!(
            lowered(&format!("K(mu) = {body}\n{use_}")),
            expected,
            "{body}"
        );
    }
}

#[test]
fn a_lambda_kernel_pushforward_scores_the_doubled_normal() {
    // The doubling map on Normal(1, 2) is Normal(2, 4). At y = 3 the change of
    // variables reads the base at f_inv(3) = 1.5 and subtracts logvolume = log 2.
    let out = lowered(
        "K = mu -> pushfwd(bijection(fn(2.0 * _), fn(_ / 2.0), log(2.0)), Normal(mu, 2.0))\n\
         lp = logdensityof(likelihoodof(K, 3.0), record(mu = 1.0))",
    );
    assert!(
        out.contains("(record (%field mu 1.0) (%field sigma 2.0))) 1.5)")
            && out.contains("(log 2.0)")
            && out.contains("(sub "),
        "{out}"
    );
    let base = -0.5 * (2.0 * PI).ln() - 2.0_f64.ln() - 0.5 * ((1.5 - 1.0) / 2.0_f64).powi(2);
    // scipy.stats.norm(2, 4).logpdf(3.0)
    assert!((base - LN_2 - (-2.336_482_894_324_563)).abs() < 1e-14);
}
