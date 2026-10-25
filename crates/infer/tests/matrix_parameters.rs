//! A vector of vectors is not a matrix argument.
//!
//! §08 declares `MvNormal`'s `cov` and the `scale` of `Wishart` and
//! `InverseWishart` as $n \times n$ matrices. §03 "Arrays": "Vectors of vectors
//! are not interpreted as matrices implicitly, but can be turned into matrices
//! explicitly using `rowstack` or `colstack`". These calls typed at exit 0
//! before, so the determiniser and the StableHLO emitter ran on them.

use flatppl_infer::{Diagnostic, Severity, infer};

fn errors(src: &str) -> Vec<Diagnostic> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .collect()
}

fn assert_refuses(src: &str, param: &str) {
    let errs = errors(src);
    assert!(
        errs.iter()
            .any(|d| d.message.contains(&format!("`{param}`"))
                && d.message
                    .contains("a vector of vectors, but spec §08 declares it a matrix")
                && d.message.contains("rowstack")
                && d.node.is_some()),
        "`{src}` must refuse `{param}` with a located message naming rowstack; got: {errs:?}"
    );
}

fn assert_clean(src: &str) {
    let errs = errors(src);
    assert!(errs.is_empty(), "`{src}` must type; got: {errs:?}");
}

#[test]
fn a_nested_covariance_literal_is_refused() {
    let point = "lp = logdensityof(m, [0.7, -1.2])";
    assert_refuses(
        &format!("m = MvNormal(mu = [0.2, -0.1], cov = [[1.0, 0.3], [0.3, 2.0]])\n{point}"),
        "cov",
    );
    // Positional arguments bind to the same parameter.
    assert_refuses(
        &format!("m = MvNormal([0.2, -0.1], [[1.0, 0.3], [0.3, 2.0]])\n{point}"),
        "cov",
    );
    // A binding holding the literal is the same value.
    assert_refuses(
        &format!("c = [[1.0, 0.3], [0.3, 2.0]]\nm = MvNormal(mu = [0.2, -0.1], cov = c)\n{point}"),
        "cov",
    );
}

#[test]
fn a_nested_scale_literal_is_refused() {
    for dist in ["Wishart", "InverseWishart"] {
        assert_refuses(
            &format!(
                "w = {dist}(nu = 4.0, scale = [[1.0, 0.0], [0.0, 1.0]])\n\
                 lp = logdensityof(w, rowstack([[1.0, 0.2], [0.2, 1.0]]))"
            ),
            "scale",
        );
    }
}

#[test]
fn a_rowstack_or_colstack_matrix_types() {
    for build in ["rowstack", "colstack"] {
        assert_clean(&format!(
            "m = MvNormal(mu = [0.2, -0.1], cov = {build}([[1.0, 0.3], [0.3, 2.0]]))\n\
             lp = logdensityof(m, [0.7, -1.2])"
        ));
    }
    assert_clean(
        "w = Wishart(nu = 4.0, scale = rowstack([[1.0, 0.0], [0.0, 1.0]]))\n\
         lp = logdensityof(w, rowstack([[1.0, 0.2], [0.2, 1.0]]))",
    );
}

#[test]
fn a_broadcast_element_must_be_a_matrix_too() {
    // Each element of a broadcast argument is an argument to the constructor.
    let mus = "[[0.0, 0.0], [1.0, 1.0]]";
    assert_refuses(
        &format!("m = MvNormal.(mu = {mus}, cov = [[[1.0, 0.0], [0.0, 1.0]]])"),
        "cov",
    );
    assert_refuses(
        &format!("m = broadcast(MvNormal, {mus}, [[[1.0, 0.0], [0.0, 1.0]]])"),
        "cov",
    );
    assert_refuses(
        "covs = [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]]\n\
         m = MvNormal.(mu = [[0.0, 0.0], [1.0, 1.0]], cov = covs)",
        "cov",
    );
    assert_refuses(
        "w = Wishart.(nu = [3.0, 4.0], scale = [[[1.0, 0.0], [0.0, 1.0]]])",
        "scale",
    );
    assert_clean(&format!(
        "m = MvNormal.(mu = {mus}, cov = [rowstack([[1.0, 0.0], [0.0, 1.0]])])"
    ));
}
