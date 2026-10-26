//! A bare built-in map over a VECTOR variate is an elementwise map, and its
//! Jacobian is DIAGONAL. The bare spelling used to take the SCALAR derivation and
//! emit that op's scalar log-volume against an n-vector, so the two spellings §06
//! declares equivalent disagreed:
//!
//! ```text
//! pushfwd(exp, MvNormal(…))            → sub(builtin_logdensityof(MvNormal, …, log(y)), log(y))
//! pushfwd(x -> broadcast(exp, x), …)   → sub(…, sum(broadcast(logvol, broadcast(log, y))))
//! ```
//!
//! The first subtracts a VECTOR from a scalar where the diagonal log-det is
//! `Σᵢ log yᵢ`; `is_flatpdl` does not flag `sub(scalar, vector)`, so it exited 0 on
//! a wrong number. The vector dispatch now precedes the bare/lambda split, and both
//! spellings emit through one builder (`invert::wrap_elementwise`).
//!
//! The load-bearing assertion is that the two spellings are BYTE-IDENTICAL — the
//! failure mode is them drifting, which a pair of separately hand-written
//! expectations would not catch. Structural only (flatppl-rust is not a density
//! engine).
use flatppl_determinizer::{determinize, is_flatpdl};

mod common;
use common::{call_arg, pir_binding, pir_head};

fn determinize_src(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    determinize(&m).expect("must lower, not refuse")
}

/// The lowered `lp` binding's FlatPIR text, conformance-checked on the way through.
fn lp(src: &str) -> String {
    let out = determinize_src(src);
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
fn bare_and_broadcast_spellings_are_byte_identical_over_a_vector_base() {
    // Over an `iid` base and over `MvNormal` — the second confirms the dispatch keys
    // on the variate domain being a vector, not on how the product was built.
    for (base, point) in [
        ("iid(Normal(mu = 0.0, sigma = 1.0), 3)", "[0.5, 0.6, 0.7]"),
        (
            "MvNormal(mu = [0.0, 0.0], cov = rowstack([[1.0, 0.0], [0.0, 1.0]]))",
            "[0.5, 0.5]",
        ),
    ] {
        let bare = lp(&format!(
            "d = pushfwd(exp, {base})\nlp = logdensityof(d, {point})"
        ));
        let broadcast = lp(&format!(
            "d = pushfwd(x -> broadcast(exp, x), {base})\nlp = logdensityof(d, {point})"
        ));
        assert_eq!(
            bare, broadcast,
            "`{base}`: §06 makes the bare and broadcast spellings the same map"
        );
        // And that the shared form is the DIAGONAL log-det, not a scalar Jacobian:
        // the volume term reduces the per-cell terms with `sum`.
        assert!(
            bare.contains("(sum ") && bare.contains("(sub "),
            "`{base}`: expected sub(…, sum(broadcast(logvol, …))):\n{bare}"
        );
    }
}

#[test]
fn the_scalar_inverse_is_never_applied_to_the_whole_vector() {
    // The defect in its own terms: it applied the scalar inverse to the WHOLE
    // n-vector, which prints as `log` at an array-typed node (and left the log-det a
    // vector). Neither spelling may do that — the per-cell application is a
    // `broadcast`. `MvNormal` keeps the base density a single
    // `builtin_logdensityof`, so the only `sum` in the emission is the log-det.
    for map in ["exp", "x -> broadcast(exp, x)"] {
        let out = lp(&format!(
            "d = pushfwd({map}, MvNormal(mu = [0.0, 0.0], cov = rowstack([[1.0, 0.0], [0.0, 1.0]])))\n\
             lp = logdensityof(d, [0.5, 0.5])"
        ));
        assert!(
            !out.contains("(log (%meta ((%array"),
            "`{map}`: the scalar inverse must not be applied to the whole vector:\n{out}"
        );
        assert_eq!(
            out.matches("(sum ").count(),
            1,
            "`{map}`: the volume term is the summed diagonal log-det:\n{out}"
        );
    }
}

#[test]
fn scalar_base_still_takes_the_scalar_derivation() {
    // The regression half: the vector dispatch must not capture a scalar variate.
    let out = lp("d = pushfwd(exp, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)");
    assert!(
        !out.contains("(sum ") && !out.contains("(broadcast "),
        "a scalar base keeps the plain scalar change of variables:\n{out}"
    );
    assert_eq!(
        out.matches("(log ").count(),
        2,
        "preimage AND volume term, `logdensityof(M, log y) − log y`:\n{out}"
    );
}

#[test]
fn bare_domain_restricted_map_reads_the_per_cell_support() {
    // The bare spelling's §06 domain restriction is now checked against the ELEMENT
    // support, which is what the map actually sees: `log` over a vector of positive
    // cells is defined per cell. Both spellings must agree on that too.
    let base = "iid(Gamma(shape = 2.0, rate = 1.0), 3)";
    let bare = lp(&format!(
        "d = pushfwd(log, {base})\nlp = logdensityof(d, [0.5, 0.6, 0.7])"
    ));
    let broadcast = lp(&format!(
        "d = pushfwd(x -> broadcast(log, x), {base})\nlp = logdensityof(d, [0.5, 0.6, 0.7])"
    ));
    assert_eq!(
        bare, broadcast,
        "the domain guard must read the same support"
    );
    assert!(bare.contains("(sum "), "diagonal log-det:\n{bare}");

    // And an out-of-domain element support still refuses in both spellings (§06
    // case 1: "refused rather than yielding a silently sub-probability measure").
    for map in ["log", "x -> broadcast(log, x)"] {
        let mut m = flatppl_syntax::parse(&format!(
            "d = pushfwd({map}, iid(Normal(mu = 0.0, sigma = 1.0), 3))\n\
             lp = logdensityof(d, [0.5, 0.6, 0.7])"
        ))
        .unwrap();
        let _ = flatppl_infer::infer(&mut m);
        determinize(&m).expect_err(&format!("`{map}` over a real-support base must refuse"));
    }
}

#[test]
fn discrete_vector_base_gate_applies_every_scalar_function_per_cell() {
    // The lattice gate round-trips the preimage through the forward map. Over a vector
    // variate `round`, `real`, `exp` and `abs` are scalar-domain (§07), so each is applied
    // with `broadcast`; applying one to the vector itself is outside its domain, and the
    // emitted FlatPDL must infer clean. Scored through flatppl-js, the gate gives
    // -0.8925742052568393 at exp([1, 4]) and -inf at the off-lattice point [1, 4].
    for map in ["exp", "x -> broadcast(exp, x)"] {
        let out = lp(&format!(
            "d = pushfwd({map}, Multinomial(n = 5, p = [0.2, 0.8]))\n\
             lp = logdensityof(d, [1.0, 4.0])"
        ));
        for head in [
            "(broadcast exp ",
            "(broadcast real ",
            "(broadcast round ",
            "(broadcast abs ",
        ] {
            assert!(out.contains(head), "`{map}`: missing `{head}` in:\n{out}");
        }
        let mut module = determinize_src(&format!(
            "d = pushfwd({map}, Multinomial(n = 5, p = [0.2, 0.8]))\n\
             lp = logdensityof(d, [1.0, 4.0])"
        ));
        let errors: Vec<_> = flatppl_infer::infer(&mut module)
            .into_iter()
            .filter(|d| d.severity == flatppl_infer::Severity::Error)
            .collect();
        assert!(
            errors.is_empty(),
            "`{map}`: the emitted IR must infer clean: {errors:?}"
        );
    }
}

#[test]
fn discrete_vector_base_carries_no_volume_term_either_way() {
    // `Multinomial`'s variate is a vector of integers, so the reference is counting
    // and there is no volume element at all (§06 "Density convention"). That is why
    // the discrete goldens did not expose this defect — the term they would have
    // compared is absent. The preimage itself must still be elementwise.
    for map in ["exp", "x -> broadcast(exp, x)"] {
        let out = lp(&format!(
            "d = pushfwd({map}, Multinomial(n = 5, p = [0.2, 0.8]))\n\
             lp = logdensityof(d, [1.0, 4.0])"
        ));
        assert!(
            out.contains("(broadcast log "),
            "`{map}`: elementwise preimage:\n{out}"
        );
        assert_eq!(
            pir_head(&call_arg(&out, "ifelse", 1)),
            "builtin_logdensityof",
            "`{map}`: no volume term over a discrete vector variate:\n{out}"
        );
    }
}

#[test]
fn a_matrix_variate_refuses_rather_than_taking_the_scalar_derivation() {
    // `domain_is_vector` tests `shape.len() == 1`, so a rank-2 variate (`Wishart`,
    // `LKJ`) fell through to the SCALAR derivation: a bare `exp` emitted
    // `log(<matrix>)` with one scalar volume term where the diagonal log-det sums
    // over every cell, and the image gate emitted `in(<matrix>, posreals)` — §07 `in`
    // requires "The type of `x` must match the element type of set `S`". `is_flatpdl`
    // is phase/type-based and accepted both. §06 case 1 names no matrix-variate
    // forward map, so refuse.
    for base in [
        "Wishart(nu = 5.0, scale = rowstack([[1.0, 0.0], [0.0, 1.0]]))",
        "LKJ(n = 2, eta = 1.5)",
    ] {
        let reason = refusal(&format!(
            "d = pushfwd(exp, {base})\nlp = logdensityof(d, [[1.0, 0.0], [0.0, 1.0]])"
        ));
        assert!(
            reason.contains("variate is a matrix"),
            "`{base}` must refuse on the matrix variate, got: {reason}"
        );
    }
}

#[test]
fn the_matrix_refusal_leaves_the_scalar_and_vector_ranks_alone() {
    // The regression half: the guard keys on rank 2, so rank 0 and rank 1 must still
    // lower. Without this the refusal could be unconditional and the test above would
    // still pass.
    for (base, point) in [
        ("Normal(mu = 0.0, sigma = 1.0)", "0.5"),
        ("iid(Normal(mu = 0.0, sigma = 1.0), 3)", "[0.5, 0.6, 0.7]"),
    ] {
        let out = lp(&format!(
            "d = pushfwd(exp, {base})\nlp = logdensityof(d, {point})"
        ));
        assert!(
            out.contains("builtin_logdensityof Normal"),
            "`{base}` must still lower:\n{out}"
        );
    }
}

// `x .^ k` is `broadcast(pow, x, k)`: the exponent is a second operand, not part
// of the operator, so the elementwise arm took it for a coupled or unknown map and
// refused. Its cell map is `y -> y ^ k`, which the explicit per-cell spelling names.
#[test]
fn a_dotted_power_is_the_per_cell_power() {
    for (k, base, point) in [
        (
            "3",
            "iid(Normal(mu = 0.0, sigma = 1.0), 3)",
            "[0.125, -1.0, 8.0]",
        ),
        ("2", "iid(Exponential(rate = 1.0), 3)", "[0.25, 1.0, 4.0]"),
        (
            "-1",
            "iid(Gamma(shape = 3.0, rate = 2.0), 3)",
            "[0.5, 1.0, 3.0]",
        ),
    ] {
        let dotted = lp(&format!(
            "d = pushfwd(x -> x .^ {k}, {base})\nlp = logdensityof(d, {point})"
        ));
        let per_cell = lp(&format!(
            "d = pushfwd(x -> broadcast(y -> y ^ {k}, x), {base})\nlp = logdensityof(d, {point})"
        ));
        assert_eq!(dotted, per_cell, "`.^ {k}` over `{base}`");
        assert!(
            dotted.contains("(sum "),
            "`.^ {k}`: diagonal log-det:\n{dotted}"
        );
    }
}

#[test]
fn a_dotted_power_keeps_the_per_cell_domain_and_literal_rules() {
    let normal = "iid(Normal(mu = 0.0, sigma = 1.0), 3)";
    let exponential = "iid(Exponential(rate = 1.0), 3)";
    for (map, base, want) in [
        (
            "x .^ 2",
            normal,
            "pow(_, 2) requires M's support to lie within its domain",
        ),
        (
            "x .^ 0",
            exponential,
            "pow with exponent 0 is not invertible",
        ),
        (
            "x .^ k",
            exponential,
            "pow(_, k) is recognised here only when k is a numeric literal",
        ),
        (
            "x .^ [1.0, 2.0, 3.0]",
            exponential,
            "is recognised here only when k is a numeric literal",
        ),
    ] {
        let mut m = flatppl_syntax::parse(&format!(
            "k = 2.0\nd = pushfwd(x -> {map}, {base})\nlp = logdensityof(d, [0.25, 1.0, 4.0])"
        ))
        .unwrap();
        let _ = flatppl_infer::infer(&mut m);
        let got = determinize(&m).expect_err("must refuse, not lower");
        assert!(
            got.reason.contains(want),
            "`{map}` over `{base}`: got {}",
            got.reason
        );
        // The synthesised cell map has no location, so the refusal names the
        // broadcast the user wrote.
        assert_eq!(got.construct, "broadcast", "`{map}`");
    }
}
