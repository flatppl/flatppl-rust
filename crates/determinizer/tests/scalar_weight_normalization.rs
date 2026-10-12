use flatppl_core::Module;
use flatppl_determinizer::{LoweringOptions, determinize_with_options};

mod common;

fn lower(source: &str) -> Module {
    let mut module = flatppl_syntax::parse(source).unwrap();
    assert!(flatppl_infer::infer(&mut module).is_empty());
    determinize_with_options(
        &module,
        &flatppl_infer::ModuleBundle::new(),
        None,
        &LoweringOptions {
            preserve_rand_tuple: true,
            ..Default::default()
        },
    )
    .unwrap()
}

fn output(source: &str) -> String {
    common::pir_binding(&flatppl_flatpir::write(&lower(source)), "outputs")
}

#[test]
fn constant_weight_cancels_only_under_a_valid_mass_guard() {
    let prefix = "w = external(reals)\nx = external(reals)\ninputs = (w, x)\n";
    let base = "locscale(StudentT(4), 2, 3)";
    for (operator, valid) in [
        ("weighted", "isfinite(w) && w > 0.0"),
        ("logweighted", "isfinite(w)"),
    ] {
        let source = format!("{prefix}outputs = logdensityof(normalize({operator}(w, {base})), x)");
        let expected =
            format!("{prefix}outputs = ifelse({valid}, logdensityof({base}, x), divide(0.0, 0.0))");
        assert_eq!(output(&source), output(&expected));
        let lowered = lower(&source);
        for mut decoded in [
            flatppl_syntax::parse(&flatppl_syntax::print(&lowered)).unwrap(),
            flatppl_flatpir::read(&flatppl_flatpir::write(&lowered)).unwrap(),
        ] {
            assert!(flatppl_infer::infer(&mut decoded).is_empty());
            flatppl_determinizer::is_flatpdl(&decoded).unwrap();
        }
    }
}

#[test]
fn positive_constant_weight_preserves_sample_and_successor_state() {
    let prefix = "state = rnginit(19)\n";
    for base in ["Normal(2, 3)", "Poisson(2)"] {
        let expected = output(&format!("{prefix}outputs = rand({base}, state)"));
        for operator in ["weighted", "logweighted"] {
            assert_eq!(
                output(&format!(
                    "{prefix}outputs = rand(normalize({operator}(2, {base})), state)"
                )),
                expected
            );
        }
    }

    let shared = lower(
        "w = external(reals)\ninputs = w\nstate = rnginit(19)\n\
         mu ~ Normal(0, 1)\nx ~ normalize(weighted(w, Normal(mu, 3)))\n\
         outputs = rand(lawof(record(mu = mu, a = x, b = x)), state)",
    );
    let pir = flatppl_flatpir::write(&shared);
    assert_eq!(pir.matches("builtin_sample").count(), 2, "{pir}");
    assert!(
        pir.contains(
            "(record (%field mu (%ref self mu)) (%field a (%ref self x)) (%field b (%ref self x)))"
        ),
        "{pir}"
    );
}

#[test]
fn normalized_samples_never_hide_undefined_mass_in_integer_values() {
    for (weight, base) in [("0", "Normal(0, 1)"), ("w", "Poisson(2)")] {
        let source = format!(
            "w = external(reals)\ninputs = w\nstate = rnginit(19)\noutputs = rand(normalize(weighted({weight}, {base})), state)"
        );
        let mut module = flatppl_syntax::parse(&source).unwrap();
        assert!(flatppl_infer::infer(&mut module).is_empty());
        assert!(flatppl_determinizer::determinize(&module).is_err());
    }
}
