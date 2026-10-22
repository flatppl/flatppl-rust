//! An elementwise affine map `x .* a .+ b` over a vector variate, with `a` and `b`
//! fixed scalars or vectors, has the diagonal Jacobian `diag(a)`:
//! `f_inv(y) = (y .- b) ./ a` and `logvol = Σᵢ log|aᵢ|`. Structural only
//! (flatppl-rust is not a density engine). flatppl-testsuite scores the numbers.
use flatppl_determinizer::{determinize, is_flatpdl};

mod common;
use common::pir_binding;

const IID: &str = "iid(Normal(mu = 0.0, sigma = 1.0), 3)";
const MVN: &str = "MvNormal(mu = [0.2, -0.1, 0.4], cov = rowstack([[1.0, 0.3, 0.1], [0.3, 2.0, -0.4], [0.1, -0.4, 1.5]]))";

fn model(prelude: &str, map: &str, base: &str) -> String {
    format!("{prelude}\nd = pushfwd({map}, {base})\nlp = logdensityof(d, [0.7, -1.2, 2.5])")
}

fn lp(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let out = determinize(&m).unwrap_or_else(|e| panic!("must lower, not refuse: {e:?}\n{src}"));
    let pir = flatppl_flatpir::write(&out);
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl failed:\n{pir}");
    pir_binding(&pir, "lp")
}

fn refusal(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    determinize(&m).expect_err("must refuse, not lower").reason
}

#[test]
fn vector_scale_divides_cellwise_and_sums_its_log_volume() {
    let out = lp(&model("", "x -> (x .* [2.0, -0.4, 1.5]) .+ 0.5", IID));
    assert!(
        out.contains("(broadcast divide (") && out.contains("(broadcast sub "),
        "the inverse is (y .- b) ./ a:\n{out}"
    );
    assert!(
        out.contains("(sum ")
            && out.contains("(broadcast log (")
            && out.contains("(broadcast abs "),
        "the log-volume is sum(log.(abs.(a))):\n{out}"
    );
}

#[test]
fn operand_order_does_not_change_the_lowering() {
    let prelude = "a = [2.0, -0.4, 1.5]\nb = [0.3, -1.0, 2.0]";
    let expected = lp(&model(prelude, "x -> (x .* a) .+ b", IID));
    for map in [
        "x -> b .+ (a .* x)",
        "x -> (a .* x) .+ b",
        "x -> b .+ (x .* a)",
    ] {
        assert_eq!(expected, lp(&model(prelude, map, IID)), "`{map}`");
    }
}

#[test]
fn scale_only_and_shift_only_lower() {
    let scale = lp(&model("", "x -> x .* [2.0, -0.4, 1.5]", IID));
    assert!(
        !scale.contains("(broadcast sub "),
        "no shift to undo:\n{scale}"
    );
    let shift = lp(&model("", "x -> x .+ [0.3, -1.0, 2.0]", IID));
    assert!(
        !shift.contains("(broadcast divide ") && !shift.contains("(sum (broadcast log"),
        "a pure shift preserves volume:\n{shift}"
    );
}

#[test]
fn a_coupled_base_keeps_its_own_density() {
    let out = lp(&model(
        "a = [2.0, -0.4, 1.5]\nb = [0.3, -1.0, 2.0]",
        "x -> (x .* a) .+ b",
        MVN,
    ));
    assert_eq!(
        out.matches("builtin_logdensityof").count(),
        1,
        "one MvNormal density at the preimage:\n{out}"
    );
    assert!(out.contains("MvNormal"), "{out}");
}

#[test]
fn a_symbolic_coefficient_is_trusted() {
    // As the scalar arm trusts a symbolic scale: only a static zero is provable.
    let out = lp(&model(
        "a = elementof(cartpow(reals, 3))",
        "x -> x .* a",
        IID,
    ));
    assert!(out.contains("(broadcast divide "), "{out}");
}

#[test]
fn a_coefficient_of_another_length_refuses() {
    for (map, what) in [
        ("x -> x .* [2.0, 0.4]", "a vector of length 2, not 3"),
        ("x -> x .* [2.0]", "a vector of length 1, not 3"),
        (
            "x -> x .+ [0.3, 1.0, 2.0, 4.0]",
            "a vector of length 4, not 3",
        ),
    ] {
        let reason = refusal(&model("", map, IID));
        assert!(reason.contains(what), "`{map}`: {reason}");
    }
}

#[test]
fn a_dynamic_length_coefficient_refuses() {
    let reason = refusal(
        "n = external(posintegers)\n\
         a = external(cartpow(reals, n))\n\
         d = pushfwd(x -> x .* a, iid(Normal(mu = 0.0, sigma = 1.0), 3))\n\
         lp = logdensityof(d, [0.7, -1.2, 2.5])",
    );
    assert!(reason.contains("not statically"), "{reason}");
}

#[test]
fn a_zero_scale_entry_refuses() {
    for (prelude, map) in [
        ("", "x -> (x .* [2.0, 0.0, 1.5]) .+ 0.5"),
        ("a = [2.0, -0.0, 1.5]", "x -> x .* a"),
        ("", "x -> x .* 0.0"),
    ] {
        let reason = refusal(&model(prelude, map, IID));
        assert!(reason.contains("zero scale entry"), "`{map}`: {reason}");
    }
}
