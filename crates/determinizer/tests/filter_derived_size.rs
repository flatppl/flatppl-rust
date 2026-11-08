//! §06's region-restricted idiom: `filter` the data to a window, then `iid` over
//! `lengthof` of the filtered data. With fixed data and a fixed predicate the
//! kept count is known, so the `iid` lowers at that size; a count of 0 is the
//! empty product measure, log-density 0 (§06 `iid`), even though a derived 0
//! types as `%dynamic` (the 2026-08-20 zero-size ruling). Before, the size stayed
//! `%dynamic` and the determiniser refused ("iid size is not a
//! statically-resolved shape").
//!
//! The numbers (scipy, checked when this landed): window [-3, 3], prior
//! N(0, 2), truncated N(mu, 1) per kept observation, at mu = 0.5. Data
//! [8.2, 9.4, 12.1] keeps none → `norm.logpdf(0.5, 0, 2)` = -1.643335713764618;
//! [8.2, 0.4, -1.1, 12.1] keeps two → -4.753286509091051. They are scored in
//! flatppl-testsuite (`corpora/coverage/out_of_window`); here the structure.

use flatppl_determinizer::determinize;

fn lowered(data: &str) -> String {
    let src = format!(
        "obs_values = {data}\nwindow = interval(-3.0, 3.0)\n\
         obs_w = filter(fn(_ in window), obs_values)\nn = lengthof(obs_w)\n\
         mu = elementof(reals)\n\
         model_w = normalize(truncate(Normal(mu = mu, sigma = 1.0), window))\n\
         L = likelihoodof(functionof(iid(model_w, n)), obs_w)\n\
         prior = joint(mu = Normal(mu = 0.0, sigma = 2.0))\n\
         posterior = bayesupdate(L, prior)\n\
         lp = logdensityof(posterior, record(mu = mu))\n\
         inputs = (mu)\noutputs = (lp)\n"
    );
    let mut m = flatppl_syntax::parse(&src).unwrap();
    let diags = flatppl_infer::infer(&mut m);
    assert!(
        diags
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "{diags:?}"
    );
    let out = determinize(&m).expect("the filtered iid lowers");
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
    flatppl_flatpir::write(&out)
}

#[test]
fn an_empty_window_is_the_empty_product() {
    let pir = lowered("[8.2, 9.4, 12.1]");
    // The likelihood term is the literal 0.0; only the prior density remains.
    assert!(
        pir.contains("(add 0.0 (%meta") && pir.contains("(builtin_logdensityof Normal"),
        "{pir}"
    );
    assert_eq!(pir.matches("builtin_logdensityof").count(), 1, "{pir}");
}

#[test]
fn a_nonempty_window_lowers_at_the_kept_count() {
    let pir = lowered("[8.2, 0.4, -1.1, 12.1]");
    // One truncated-normal density mapped over the filtered data, plus the prior.
    assert!(
        pir.contains("(%ref self obs_w)") && pir.contains("builtin_touniform"),
        "{pir}"
    );
}
