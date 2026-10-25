//! §07 `linspace(from, to, n)` and `extlinspace(from, to, n)` as emitter heads.
//! Their point count is static from the result type, so they emit as an index
//! vector scaled into `[from, to]`, with `extlinspace`'s infinite end points
//! around it.

use flatppl_stablehlo::{Mode, emit};

fn emit_outputs(outputs: &str) -> String {
    emit_with_inputs("lo = elementof(reals)\ninputs = (lo)", outputs)
}

fn emit_with_inputs(inputs: &str, outputs: &str) -> String {
    let src = format!("flatppl_compat = \"0.1\"\n{inputs}\noutputs = {outputs}");
    let mut m = flatppl_syntax::parse(&src).unwrap();
    let diags = flatppl_infer::infer(&mut m);
    assert!(
        diags
            .iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "inference errors: {diags:?}"
    );
    let d = flatppl_determinizer::determinize(&m).expect("must lower, not refuse");
    emit(&d, Mode::LogDensity, &Default::default()).expect("must emit")
}

#[test]
fn linspace_emits_a_static_vector() {
    let out = emit_outputs("linspace(lo, 2.0, 5)");
    assert!(out.contains("-> tensor<5xf32>"), "five points:\n{out}");
    assert_eq!(
        out.matches("stablehlo.select").count(),
        2,
        "both ends selected exactly:\n{out}"
    );
}

#[test]
fn extlinspace_adds_the_infinite_end_points() {
    let out = emit_outputs("extlinspace(lo, 2.0, 3)");
    assert!(
        out.contains("-> tensor<5xf32>"),
        "three points and two ends:\n{out}"
    );
    assert!(
        out.contains("0x7F800000") && out.contains("0xFF800000"),
        "both infinities:\n{out}"
    );
}

// §07: "When `n = 1`, `from` and `to` must be equal. The result contains that
// endpoint as its sole real value."
#[test]
fn a_single_point_linspace_is_its_endpoint() {
    let out = emit_outputs("linspace(lo, lo, 1)");
    assert!(out.contains("-> tensor<1xf32>"), "one point:\n{out}");
}

// Runtime ends only show at run time whether §07's `from == to` holds for `n = 1`,
// so the single point is NaN unless they are equal, for `extlinspace` too.
#[test]
fn a_single_point_between_runtime_ends_is_nan_unless_they_are_equal() {
    let inputs = "lo = elementof(reals)\nhi = elementof(reals)\ninputs = (lo, hi)";
    for outputs in ["linspace(lo, hi, 1)", "extlinspace(lo, hi, 1)"] {
        let out = emit_with_inputs(inputs, outputs);
        assert!(
            out.contains("stablehlo.compare EQ") && out.contains("0x7FC00000"),
            "{outputs}: equal-ends gate over NaN:\n{out}"
        );
    }
}
