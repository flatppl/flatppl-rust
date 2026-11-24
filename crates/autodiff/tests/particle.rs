use flatppl_autodiff::{NUMERICAL_SOURCE_OPTIONS, determinize_query_with_options};
use flatppl_infer::{ModuleBundle, Severity};

fn check_roundtrip(source: &str, selected: &[&str], numerical: bool) {
    let options = if numerical {
        NUMERICAL_SOURCE_OPTIONS
    } else {
        Default::default()
    };
    let mut module = flatppl_syntax::parse(source).unwrap();
    assert!(
        flatppl_infer::infer(&mut module)
            .iter()
            .all(|d| d.severity != Severity::Error)
    );
    let module = determinize_query_with_options(&module, &ModuleBundle::new(), &options).unwrap();
    for transform in [
        flatppl_autodiff::gradient_with_options,
        flatppl_autodiff::jvp_with_options,
        flatppl_autodiff::vjp_with_options,
    ] {
        let mut result = transform(&module, selected, &options).unwrap();
        flatppl_autodiff::prepare_source_with_options(&mut result, &options).unwrap();
        let source = flatppl_syntax::print(&result);
        let mut reparsed = flatppl_syntax::parse(&source).unwrap();
        let diagnostics = flatppl_infer::infer(&mut reparsed);
        assert!(
            diagnostics.iter().all(|d| d.severity != Severity::Error),
            "{diagnostics:?}"
        );
    }
}

#[test]
fn split_crystal_ball_differentiates_both_widths_under_a_conditional() {
    check_roundtrip(
        r#"hep = standard_module("particle-physics", "0.1")
x = elementof(reals)
sl = elementof(posreals)
sr = elementof(posreals)
inputs = (x, sl, sr)
l = logdensityof(hep.DoubleSidedCrystalBall(0.4, sl, sr, 1.1, 1.6, 3.25, 4.2), x)
outputs = ifelse(x < 4.0, l, 0.25 * x * x)
"#,
        &["x", "sl", "sr"],
        false,
    );
}

#[test]
fn argus_normalizer_scores_replay_with_numerical_integrals() {
    check_roundtrip(
        r#"hep = standard_module("particle-physics", "0.1")
slope = elementof(reals)
power = elementof(posreals)
x = elementof(reals)
inputs = (slope, power, x)
outputs = logdensityof(hep.Argus(4.0, slope, power), x)
"#,
        &["slope", "power", "x"],
        true,
    );
}

#[test]
fn voigtian_width_and_resolution_scores_reparse_bounded_source() {
    check_roundtrip(
        r#"hep = standard_module("particle-physics", "0.1")
width = elementof(posreals)
sigma = elementof(posreals)
x = elementof(reals)
inputs = (width, sigma, x)
outputs = logdensityof(hep.Voigtian(0.3, width, sigma), x)
"#,
        &["width", "sigma", "x"],
        false,
    );
}
