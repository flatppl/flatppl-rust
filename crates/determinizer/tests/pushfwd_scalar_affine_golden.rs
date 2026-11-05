//! A scalar-scaled affine forward map over an array variate (§06 case 1).
//!
//! §07 gives `mul` a scalar-vector and a scalar-matrix domain, `divide` an
//! array-scalar domain and `add`/`sub` arrays of one shape. So `a * x + c` over
//! an n-cell variate has the Jacobian `a·I`: `f_inv(y) = (y - c) / a` and the
//! log-volume is `n log|a|`, one scalar for the whole map.

use std::f64::consts::{LN_2, PI};

use flatppl_determinizer::determinize;

mod common;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn lowered(src: &str) -> String {
    let out = determinize(&parse_infer(src)).unwrap_or_else(|e| panic!("must lower: {e:?}\n{src}"));
    common::pir_binding(&flatppl_flatpir::write(&out), "lp")
}

fn refusal(src: &str) -> String {
    match determinize(&parse_infer(src)) {
        Ok(out) => panic!("must refuse:\n{}", flatppl_syntax::print(&out)),
        Err(e) => format!("{e:?}"),
    }
}

const BASE: &str = "iid(Normal(1.0, 2.0), 3)";
const POINT: &str = "[3.0, 1.0, -2.0]";

fn normal_logpdf(x: f64, mu: f64, sigma: f64) -> f64 {
    -0.5 * (2.0 * PI).ln() - sigma.ln() - 0.5 * ((x - mu) / sigma).powi(2)
}

#[test]
fn a_scalar_times_a_vector_inverts_cellwise_and_counts_every_cell() {
    let out = lowered(&format!(
        "lp = logdensityof(pushfwd(fn(2.0 * _), {BASE}), {POINT})"
    ));
    // f_inv(y) = y / 2.0, and logvol = 3 · log|2|.
    assert!(out.contains("(divide "), "{out}");
    assert!(
        out.contains("(mul 3.0 (%meta") && out.contains("(log (%meta") && out.contains("(abs 2.0)"),
        "{out}"
    );
    // The emitted change of variables, by hand: Σ log N(y_i / 2; 1, 2) - 3 log 2.
    let y = [3.0, 1.0, -2.0];
    let value: f64 = y
        .iter()
        .map(|&v| normal_logpdf(v / 2.0, 1.0, 2.0))
        .sum::<f64>()
        - 3.0 * LN_2;
    // scipy.stats.norm(2, 4).logpdf([3, 1, -2]).sum()
    assert!((value - (-7.478_198_682_973_689)).abs() < 1e-14, "{value}");
}

#[test]
fn both_operand_orders_of_the_scale_lower_alike() {
    let left = lowered(&format!(
        "lp = logdensityof(pushfwd(fn(2.0 * _), {BASE}), {POINT})"
    ));
    let right = lowered(&format!(
        "lp = logdensityof(pushfwd(fn(_ * 2.0), {BASE}), {POINT})"
    ));
    assert_eq!(left, right);
}

#[test]
fn a_vector_shift_and_a_scalar_divisor_lower() {
    // (x - m) / 2: f_inv(y) = y * 2 + m, logvol = 3 · (-log|2|).
    let out = lowered(&format!(
        "m = [1.0, 0.0, -1.0]\nlp = logdensityof(pushfwd(fn((_ - m) / 2.0), {BASE}), {POINT})"
    ));
    assert!(out.contains("(add ") && out.contains("(neg "), "{out}");
    // c - x reflects: log|-1| = 0.
    let reflect = lowered(&format!(
        "c = [1.0, 0.0, -1.0]\nlp = logdensityof(pushfwd(fn(c - _), {BASE}), {POINT})"
    ));
    assert!(
        reflect.contains("(sub ") && !reflect.contains("(log "),
        "{reflect}"
    );
}

#[test]
fn a_negative_literal_scale_is_nonzero() {
    // `-3.0` is `neg(3.0)`, a nonzero constant, on a scalar and on a vector base.
    let scalar = lowered("lp = logdensityof(pushfwd(fn(-3.0 * _), Normal(1.0, 2.0)), 3.0)");
    assert!(
        scalar.contains("(abs (%meta") || scalar.contains("(abs (neg"),
        "{scalar}"
    );
    lowered(&format!(
        "lp = logdensityof(pushfwd(fn(-3.0 * _), {BASE}), {POINT})"
    ));
}

#[test]
fn a_kernel_parameter_scale_lowers() {
    // The scale reaches the map as an additional argument (§04).
    for kernel in [
        "K = a -> pushfwd((x, s) -> s * x, iid(Normal(1.0, 2.0), 3), a)",
        "K(a) = pushfwd((x, s) -> s * x, iid(Normal(1.0, 2.0), 3), a)",
    ] {
        lowered(&format!(
            "{kernel}\nlp = logdensityof(likelihoodof(K, {POINT}), record(a = 2.0))"
        ));
    }
}

#[test]
fn a_zero_or_unproven_scale_refuses_as_on_a_scalar_base() {
    // §06: "affine maps ... (with nonzero scaling)".
    for src in [
        format!("lp = logdensityof(pushfwd(fn(0.0 * _), {BASE}), {POINT})"),
        format!(
            "a = elementof(reals)\nK = functionof(pushfwd((x, s) -> s * x, {BASE}, a), a = a)\n\
             lp = logdensityof(likelihoodof(K, {POINT}), record(a = 2.0))"
        ),
    ] {
        let e = refusal(&src);
        assert!(e.contains("provably nonzero"), "{e}");
    }
}

#[test]
fn a_scaled_input_plus_a_vector_shift_lowers_in_every_spelling() {
    // `2 x + c` with c = [1, 0, -1]: f_inv(y) = (y - c) / 2, logvol = 3 log 2. A
    // scalar factor is not the matrix of `mu + L * x`.
    let spell = |map: &str| {
        lowered(&format!(
            "c = [1.0, 0.0, -1.0]\nlp = logdensityof(pushfwd({map}, {BASE}), {POINT})"
        ))
    };
    let fn_form = spell("fn(2.0 * _ + c)");
    assert!(
        fn_form.contains("(mul 3.0 ") && !fn_form.contains("linsolve"),
        "{fn_form}"
    );
    assert_eq!(spell("x -> 2.0 * x + c"), fn_form);
    lowered(&format!(
        "c = [1.0, 0.0, -1.0]\nlp = logdensityof(pushfwd(fn(c + 2.0 * _), {BASE}), {POINT})"
    ));
    let y = [3.0, 1.0, -2.0];
    let c = [1.0, 0.0, -1.0];
    let value: f64 = y
        .iter()
        .zip(c)
        .map(|(&v, c)| normal_logpdf((v - c) / 2.0, 1.0, 2.0))
        .sum::<f64>()
        - 3.0 * LN_2;
    // scipy.stats.multivariate_normal(2 + c, 16 I).logpdf([3, 1, -2])
    assert!((value - (-7.228_198_682_973_689)).abs() < 1e-14, "{value}");
}
