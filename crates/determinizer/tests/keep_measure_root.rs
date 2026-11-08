//! A determinized module is measure-free. A requested output that is a measure,
//! a likelihood or a kernel refuses with its kind named, and a dead measure
//! binding or unread draw is removed, never given a placeholder value.

use flatppl_determinizer::determinize_with_roots;
use flatppl_infer::ModuleBundle;

const MODEL: &str = "m = Normal(0, 1)\nx ~ m\nlp = logdensityof(m, 0.5)\n";

fn lower(src: &str, roots: Option<&[&str]>) -> Result<String, String> {
    let mut module = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut module);
    let syms: Option<Vec<flatppl_core::Symbol>> =
        roots.map(|names| names.iter().map(|n| module.intern(n)).collect());
    determinize_with_roots(&module, &ModuleBundle::new(), syms.as_deref())
        .map(|out| flatppl_flatpir::write(&out))
        .map_err(|e| e.reason)
}

#[test]
fn a_requested_measure_output_refuses_naming_its_kind() {
    for (src, root, kind, example) in [
        (MODEL, "m", "is a measure", "logdensityof(m, x)"),
        (
            "mu = elementof(reals)\nL = likelihoodof(functionof(Normal(mu, 1.0), mu = mu), 2.0)\n\
             lp = logdensityof(L, record(mu = 0.5))\n",
            "L",
            "is a likelihood",
            "logdensityof(L, theta)",
        ),
        (
            "mu = elementof(reals)\nK = functionof(Normal(mu, 1.0), mu = mu)\n\
             lp = logdensityof(K(0.5), 2.0)\n",
            "K",
            "is a kernel",
            "logdensityof(K(theta), x)",
        ),
    ] {
        let reason = lower(src, Some(&[root])).expect_err("a measure-layer root must refuse");
        for needle in [
            format!("`{root}` {kind}"),
            "a determinized module holds no measures".to_string(),
            example.to_string(),
        ] {
            assert!(reason.contains(&needle), "`{needle}` not in: {reason}");
        }
    }
}

#[test]
fn dead_measure_bindings_and_unread_draws_are_removed_not_zeroed() {
    for roots in [None, Some(&["lp"][..])] {
        let pir = lower(MODEL, roots).expect("the query lowers");
        assert!(pir.contains("(%bind lp "), "lp must lower:\n{pir}");
        assert!(
            !pir.contains("(%bind m ") && !pir.contains("(%bind x "),
            "a dead measure binding and an unread draw are removed:\n{pir}"
        );
        assert!(!pir.contains(" 0.0)"), "no placeholder value:\n{pir}");
    }
}
