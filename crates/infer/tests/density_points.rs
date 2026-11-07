//! Spec §06: `densityof(M, x)` and `logdensityof(M, x)` evaluate the density of `M`
//! at a point of its variate space, and `logdensityof.(M, grid)` does so at each grid
//! element (§04 holds the measure fixed). A point that no variate of `M` can be is a
//! static error that names both types. Scalar kinds follow §03's inclusions, so an
//! integer point fits a real variate.

use flatppl_infer::{Severity, infer};

fn errors(src: &str) -> Vec<String> {
    let mut module = flatppl_syntax::parse(src).unwrap();
    infer(&mut module)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

fn assert_clean(src: &str) {
    let found = errors(src);
    assert!(found.is_empty(), "unexpected errors: {found:?}\n{src}");
}

fn assert_refused(src: &str, needles: &[&str]) {
    let found = errors(src);
    assert_eq!(found.len(), 1, "expected one error, got {found:?}\n{src}");
    for needle in needles {
        assert!(found[0].contains(needle), "`{needle}` not in: {}", found[0]);
    }
}

const MU: &str = "mu = elementof(cartpow(posreals, 3))\n";

#[test]
fn a_point_of_another_shape_is_refused_naming_both_types() {
    for (expr, point, variate) in [
        (
            "logdensityof(Poisson.(mu), 7)",
            "integer",
            "array of integer [3]",
        ),
        (
            "densityof(Poisson.(mu), 7)",
            "integer",
            "array of integer [3]",
        ),
        (
            "logdensityof(Normal(0.0, 1.0), [1.0, 2.0])",
            "array of real [2]",
            "real",
        ),
        (
            "logdensityof(iid(Normal(0.0, 1.0), 3), [1.0, 2.0])",
            "array of real [2]",
            "array of real [3]",
        ),
        (
            "logdensityof(Normal.(mu, 1.0), 2.5)",
            "real",
            "array of real [3]",
        ),
    ] {
        assert_refused(
            &format!("{MU}lp = {expr}\n"),
            &[
                "the point is",
                &format!("is {point}, but"),
                &format!("variate is {variate}"),
                "spec §06 `densityof`",
            ],
        );
    }
}

#[test]
fn each_grid_element_must_be_a_variate() {
    // The measure is held fixed, so each element of the grid is a point of it.
    for expr in [
        "logdensityof.(Poisson.(mu), [1, 7, 30])",
        "broadcast(logdensityof, Poisson.(mu), [1, 7, 30])",
    ] {
        assert_refused(
            &format!("{MU}lp = {expr}\n"),
            &["is integer, but", "variate is array of integer [3]"],
        );
    }
}

#[test]
fn points_of_the_variate_type_stay_clean() {
    for expr in [
        "logdensityof(Poisson.(mu), [1, 7, 30])",
        "logdensityof(Normal.(mu, 1.0), [0.5, 1.0, 1.5])",
        // §03: integers ⊂ reals.
        "logdensityof(Normal(0.0, 1.0), 1)",
        "logdensityof.(Normal(0.0, 1.0), [0.1, 0.2, 0.3])",
        "logdensityof(joint(a = Normal(0.0, 1.0), b = Poisson(2.0)), record(a = 0.3, b = 2))",
        "logdensityof(iid(Normal(0.0, 1.0), 3), [1.0, 2.0, 3.0])",
    ] {
        assert_clean(&format!("{MU}lp = {expr}\n"));
    }
}

#[test]
fn an_undecided_extent_is_not_refused() {
    // A filtered vector has a data-dependent length, so nothing is judged.
    assert_clean(&format!(
        "{MU}xs = filter(fn(_ > 1), [1, 2, 3, 4])\nlp = logdensityof(iid(Poisson(2.0), 3), xs)\n"
    ));
}
