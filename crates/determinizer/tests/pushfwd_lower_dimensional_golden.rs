//! `pushfwd` of a law on a lower-dimensional support (`stdsimplex(n)`, `zerosum(n)`,
//! `LKJ`, `LKJCholesky`) and of a point mass.
//!
//! * A uniform scale and shift `a .* x .+ b` (scalar `a`) of a `stdsimplex(n)` or
//!   `zerosum(n)` support scales each of the n − 1 free coordinates of §03's
//!   coordinate Lebesgue measure by `a`: volume `(n − 1)·log|a|`.
//! * Every other map of such a law has no density (§06 fixes no reference measure
//!   for its image), and neither has any map of `LKJ`/`LKJCholesky` (§03 gives
//!   correlation sets none).
//! * `pushfwd(f, Dirac(c))` is `Dirac(f(c))`: no volume term, the point gated on
//!   `f(c) = y`.
//!
//! flatppl-rust is not a density engine: the volume term is evaluated here from its
//! literal arithmetic and compared with mpmath 1.4.1; the base Dirichlet term is
//! scored by an engine in the testsuite.
use flatppl_determinizer::is_flatpdl;

mod common;
use common::{call_arg, determinize, pir_binding, pir_head};

/// 2·log 2, mpmath 1.4.1 at 50 digits: 1.3862943611198906188.
const TWO_LOG_2: f64 = 1.386_294_361_119_890_6;

fn lp(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let errors: Vec<_> = flatppl_infer::infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "inference errors in:\n{src}\n{errors:?}");
    let out = determinize(&m).expect("must lower, not refuse");
    let pir = flatppl_flatpir::write(&out);
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl failed:\n{pir}");
    // The bound value, without its `(%bind lp …)` wrapper.
    call_arg(&pir_binding(&pir, "lp"), "%bind", 1)
}

fn refusal(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    determinize(&m).expect_err("must refuse, not lower").reason
}

/// The value of a FlatPIR expression of literals through `mul`, `log`, `abs`, `neg`.
fn eval(expr: &str) -> f64 {
    let e = expr.trim();
    let Some(body) = e.strip_prefix('(') else {
        return match e {
            "inf" => f64::INFINITY,
            _ => e.parse().unwrap_or_else(|_| panic!("not a literal: {e}")),
        };
    };
    let head = pir_head(e);
    if body.starts_with("%meta") {
        return eval(&call_arg(e, "%meta", 1));
    }
    let arg = |n| eval(&call_arg(e, &head, n));
    match head.as_str() {
        "mul" => arg(0) * arg(1),
        "log" => arg(0).ln(),
        "abs" => arg(0).abs(),
        "neg" => -arg(0),
        _ => panic!("no literal evaluation for `{head}` in {e}"),
    }
}

const LOWER_DIMENSIONAL: &str = "the image of a law on a lower-dimensional support has no density";

#[test]
fn a_uniform_scale_of_a_dirichlet_takes_n_minus_one_cells() {
    // Dirichlet([2, 3, 4]) at x = [0.2, 0.3, 0.5] through 2x + 1: logDirichlet(x) −
    // 2·log 2 = 0.63657682907155101 (mpmath); main subtracted 3·log 2.
    for map in ["fn(2.0 .* _ .+ 1.0)", "fn(2.0 * _ + [1.0, 1.0, 1.0])"] {
        let out = lp(&format!(
            "d = pushfwd({map}, Dirichlet(alpha = [2.0, 3.0, 4.0]))\n\
             lp = logdensityof(d, [1.4, 1.6, 2.0])"
        ));
        assert_eq!(pir_head(&out), "sub", "`{map}`:\n{out}");
        assert_eq!(
            pir_head(&call_arg(&out, "sub", 0)),
            "builtin_logdensityof",
            "`{map}`:\n{out}"
        );
        let volume = eval(&call_arg(&out, "sub", 1));
        assert!(
            (volume - TWO_LOG_2).abs() <= 4.0 * f64::EPSILON,
            "`{map}`: {volume}"
        );
    }
}

#[test]
fn a_base_behind_a_binding_keeps_its_deficit() {
    let out = lp(
        "D = Dirichlet(alpha = [2.0, 3.0, 4.0])\nd = pushfwd(fn(2.0 .* _ .+ 1.0), D)\n\
         lp = logdensityof(d, [1.4, 1.6, 2.0])",
    );
    let volume = eval(&call_arg(&out, "sub", 1));
    assert!(
        (volume - TWO_LOG_2).abs() <= 4.0 * f64::EPSILON,
        "{volume}\n{out}"
    );
}

#[test]
fn a_matrix_base_refuses_first_lower_dimensional_or_not() {
    // `iid(Dirichlet(…), 2)` is a matrix variate, which takes the matrix refusal
    // first, as a full-dimensional matrix base does for these maps.
    for map in ["fn(2.0 .* _)", "fn(broadcast(fn(exp.(_)), _))"] {
        let reason = refusal(&format!(
            "d = pushfwd({map}, iid(Dirichlet(alpha = [2.0, 3.0, 4.0]), 2))\n\
             lp = logdensityof(d, [[0.4, 0.6, 1.0], [0.4, 0.6, 1.0]])"
        ));
        assert!(
            reason.contains("the pushforward base variate is a matrix"),
            "`{map}`: {reason}"
        );
    }
}

#[test]
fn a_uniform_scale_of_lebesgue_zerosum_takes_n_minus_one_cells() {
    // Lebesgue(zerosum(3)) through 2x: density 1 on the image's chart, so −2·log 2
    // there (mpmath −1.3862943611198906188); main gave −3·log 2.
    let out = lp("d = pushfwd(fn(2.0 .* _), Lebesgue(zerosum(3)))\n\
         lp = logdensityof(d, [0.2, 0.4, -0.6])");
    assert_eq!(call_arg(&out, "ifelse", 1), "0.0", "in the image:\n{out}");
    let volume = eval(&call_arg(&out, "sub", 1));
    assert!((volume - TWO_LOG_2).abs() <= 4.0 * f64::EPSILON, "{volume}");
}

#[test]
fn neg_of_a_dirichlet_keeps_a_zero_volume_term() {
    // The uniform scale −1.
    let out = lp("d = pushfwd(neg, Dirichlet(alpha = [2.0, 3.0, 4.0]))\n\
         lp = logdensityof(d, [-0.2, -0.3, -0.5])");
    assert!(out.contains("builtin_logdensityof Dirichlet"), "{out}");
}

#[test]
fn every_other_map_of_a_lower_dimensional_law_refuses() {
    let dirichlet = "Dirichlet(alpha = [2.0, 3.0, 4.0])";
    let zerosum = "Lebesgue(zerosum(3))";
    let vector = "[1.4, 1.6, 2.0]";
    let matrix = "rowstack([[2.0, 0.0], [0.0, 2.0]])";
    for (pre, map, base, point) in [
        ("", "fn(exp.(_))", dirichlet, vector),
        ("", "fn(broadcast(exp, _))", dirichlet, vector),
        ("", "fn([2.0, 3.0, 4.0] .* _)", dirichlet, vector),
        (
            "L = rowstack([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 1.0, 4.0]])\n",
            "fn([1.0, 0.0, 0.0] + L * _)",
            dirichlet,
            vector,
        ),
        ("", "reverse", dirichlet, vector),
        ("", "fn(exp.(_))", zerosum, vector),
        (
            "D = Dirichlet(alpha = [2.0, 3.0, 4.0])\n",
            "fn(exp.(_))",
            "D",
            vector,
        ),
        ("", "fn(2.0 * _)", "LKJ(n = 2, eta = 2.0)", matrix),
        ("", "fn(2.0 * _)", "LKJCholesky(n = 2, eta = 2.0)", matrix),
    ] {
        let reason = refusal(&format!(
            "{pre}d = pushfwd({map}, {base})\nlp = logdensityof(d, {point})"
        ));
        assert!(
            reason.contains(LOWER_DIMENSIONAL),
            "`{map}` of `{base}`: {reason}"
        );
    }
    // `locscale` over a vector takes a matrix scale.
    let reason = refusal(
        "L = rowstack([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 1.0, 4.0]])\n\
         d = locscale(Dirichlet(alpha = [2.0, 3.0, 4.0]), [1.0, 0.0, 0.0], L)\n\
         lp = logdensityof(d, [1.4, 1.6, 2.0])",
    );
    assert!(reason.contains(LOWER_DIMENSIONAL), "{reason}");
}

#[test]
fn an_explicit_bijection_of_a_dirichlet_keeps_its_stated_volume() {
    // §06: implementations "are not required to verify" an annotation, so its
    // logvol stands as written.
    let out = lp(
        "d = pushfwd(bijection(fn(2.0 .* _), fn(_ ./ 2.0), 3.0 * log(2.0)), \
         Dirichlet(alpha = [2.0, 3.0, 4.0]))\n\
         lp = logdensityof(d, [0.4, 0.6, 1.0])",
    );
    let volume = eval(&call_arg(&out, "sub", 1));
    assert!(
        (volume - 1.5 * TWO_LOG_2).abs() <= 8.0 * f64::EPSILON,
        "{volume}"
    );
}

#[test]
fn a_point_mass_takes_no_volume_term_and_gates_on_its_image() {
    // `Dirac(f(c))`: the base is scored at its atom, and `iszero(y − f(c))` decides.
    for src in [
        "d = pushfwd(fn(2.0 * _ + 1.0), Dirac(0.5))\nlp = logdensityof(d, 2.0)",
        "d = locscale(Dirac(0.5), 1.0, 2.0)\nlp = logdensityof(d, 2.0)",
        "d = pushfwd(exp, Dirac(0.5))\nlp = logdensityof(d, 1.6487212707001282)",
        "d = pushfwd(fn(exp.(_)), Dirac([0.5, 1.0]))\n\
         lp = logdensityof(d, [1.6487212707001282, 2.718281828459045])",
    ] {
        let out = lp(src);
        assert_eq!(pir_head(&out), "ifelse", "{src}:\n{out}");
        assert!(
            call_arg(&out, "ifelse", 0).contains("(iszero "),
            "{src}: the forward round trip:\n{out}"
        );
        let arm = call_arg(&out, "ifelse", 1);
        assert_eq!(
            pir_head(&arm),
            "builtin_logdensityof",
            "{src}: no volume term:\n{out}"
        );
        assert!(!out.contains("(log "), "{src}: no log-volume:\n{out}");
    }
}

#[test]
fn a_counting_base_still_takes_no_volume_term() {
    let out = lp("d = pushfwd(fn(2.0 * _ + 1.0), Poisson(rate = 3.0))\nlp = logdensityof(d, 7.0)");
    assert_eq!(
        pir_head(&call_arg(&out, "ifelse", 1)),
        "builtin_logdensityof",
        "{out}"
    );
}

/// 2·log 3, mpmath 1.4.1 at 50 digits: 2.1972245773362193828.
const TWO_LOG_3: f64 = 2.197_224_577_336_219_4;

const D: &str = "D = Dirichlet(alpha = [2.0, 3.0, 4.0])\n";

/// The volume terms of nested `sub`s, outermost first.
fn volumes(mut out: String, depth: usize) -> Vec<f64> {
    let mut v = Vec::new();
    for _ in 0..depth {
        v.push(eval(&call_arg(&out, "sub", 1)));
        out = call_arg(&out, "sub", 0);
    }
    v
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 8.0 * f64::EPSILON * b.abs().max(1.0)
}

#[test]
fn a_map_of_a_mapped_simplex_law_keeps_the_free_dimension() {
    // A bijection keeps the dimension of its support, so the outer map sees the
    // simplex's n − 1 free coordinates too: 2·log 2 + 2·log 3 for 3x of 2x
    // (mpmath 3.5835189384561100016), where a full-dimensional reading takes 3·log 3.
    let out = lp(&format!(
        "{D}d = pushfwd(fn(3.0 .* _), pushfwd(fn(2.0 .* _), D))\n\
         lp = logdensityof(d, [1.2, 1.8, 3.0])"
    ));
    let v = volumes(out, 2);
    assert!(close(v[0], TWO_LOG_3) && close(v[1], TWO_LOG_2), "{v:?}");
    assert!(close(v[0] + v[1], 3.583_518_938_456_11), "{v:?}");
    // A scale over a shift: 2·log 2 outside, nothing inside.
    let out = lp(&format!(
        "{D}d = pushfwd(fn(2.0 .* _), pushfwd(fn(_ .+ 1.0), D))\n\
         lp = logdensityof(d, [2.4, 2.6, 3.0])"
    ));
    let v = volumes(out, 2);
    assert!(close(v[0], TWO_LOG_2) && v[1] == 0.0, "{v:?}");
    // Three levels deep.
    let out = lp(&format!(
        "{D}d = pushfwd(fn(2.0 .* _), pushfwd(fn(_ .+ 1.0), pushfwd(fn(3.0 .* _), D)))\n\
         lp = logdensityof(d, [3.2, 3.8, 5.0])"
    ));
    let v = volumes(out, 3);
    assert!(
        close(v[0], TWO_LOG_2) && v[1] == 0.0 && close(v[2], TWO_LOG_3),
        "{v:?}"
    );
}

#[test]
fn a_uniform_scale_over_an_explicit_bijection_counts_the_free_cells() {
    // The outer term takes cells − deficit; the inner one keeps its author's value.
    let out = lp(&format!(
        "{D}b = bijection(fn(2.0 .* _), fn(_ ./ 2.0), 3.0 * log(2.0))\n\
         d = pushfwd(fn(3.0 .* _), pushfwd(b, D))\n\
         lp = logdensityof(d, [1.2, 1.8, 3.0])"
    ));
    let v = volumes(out, 2);
    assert!(
        close(v[0], TWO_LOG_3) && close(v[1], 1.5 * TWO_LOG_2),
        "{v:?}"
    );
}

#[test]
fn every_composed_uniform_affine_map_lowers() {
    // The one set the Stan emitter reads too: `./` and composed steps included.
    for (map, want) in [
        ("fn(_ ./ 2.0)", -TWO_LOG_2),
        ("fn((_ .+ 1.0) .* 2.0)", TWO_LOG_2),
        ("fn(2.0 .- _)", 0.0),
    ] {
        let out = lp(&format!(
            "{D}d = pushfwd({map}, D)\nlp = logdensityof(d, [0.1, 0.15, 0.25])"
        ));
        let v = eval(&call_arg(&out, "sub", 1));
        assert!(close(v, want), "`{map}`: {v}\n{out}");
    }
}

#[test]
fn a_non_uniform_map_of_a_mapped_simplex_law_refuses() {
    let point = "[1.2, 1.8, 3.0]";
    for m in [
        "pushfwd(fn(exp.(_)), pushfwd(fn(2.0 .* _), D))",
        "pushfwd(fn(exp.(_)), pushfwd(neg, D))",
        "pushfwd(fn(exp.(_)), pushfwd(fn(_ .+ 1.0), Lebesgue(zerosum(3))))",
        "locscale(pushfwd(fn(2.0 .* _), D), [0.0, 0.0, 0.0], \
         rowstack([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]))",
    ] {
        let reason = refusal(&format!("{D}d = {m}\nlp = logdensityof(d, {point})"));
        assert!(reason.contains(LOWER_DIMENSIONAL), "`{m}`: {reason}");
    }
}

#[test]
fn a_map_of_a_base_holding_a_point_mass_refuses() {
    // Over a continuous variate the inverse does not locate the atom exactly,
    // whatever the other components are, in a superposition or a product.
    let continuous =
        "normalize(superpose(weighted(0.5, Dirac(1.0)), weighted(0.5, Normal(0.0, 1.0))))";
    let atomic = "superpose(weighted(0.5, pushfwd(x -> x + 0.0, Bernoulli(0.3))), \
                  weighted(0.5, Dirac(1.2)))";
    for (src, point) in [
        (format!("pushfwd(fn(2.0 * _), {continuous})"), "2.0"),
        (format!("locscale({continuous}, 0.1, 0.3)"), "0.4"),
        (format!("locscale({atomic}, 0.1, 0.3)"), "0.4"),
        (
            "pushfwd(fn(2.0 .* _), joint(Dirac(0.5), Normal(0.0, 1.0)))".to_string(),
            "[1.0, 0.4]",
        ),
    ] {
        let reason = refusal(&format!("d = {src}\nlp = logdensityof(d, {point})"));
        assert!(
            reason.contains("holds a point mass beside other components"),
            "`{src}`: {reason}"
        );
    }
    // A superposition holding the law of a `Dirac` draw.
    let reason = refusal(
        "x ~ Dirac(1.0)\nd = pushfwd(fn(2.0 * _), normalize(superpose(weighted(0.5, lawof(x)), \
         weighted(0.5, Normal(0.0, 1.0)))))\nlp = logdensityof(d, 2.0)",
    );
    assert!(
        reason.contains("holds a point mass beside other components"),
        "{reason}"
    );
    // A point mass not written as `Dirac(c)`: a tilt onto a `Dirac`, an applied
    // kernel whose body is a `Dirac`, the law of a `Dirac` draw. None takes a
    // volume term.
    for (pre, src) in [
        (
            "",
            "pushfwd(fn(2.0 * _), tilt(Normal(0.0, 1.0), Dirac(0.5)))",
        ),
        (
            "c = elementof(reals)\nK = functionof(Dirac(c), c = c)\n",
            "pushfwd(fn(2.0 * _), K(0.3))",
        ),
        ("x ~ Dirac(0.5)\nL = lawof(x)\n", "pushfwd(fn(2.0 * _), L)"),
        (
            "x ~ Dirac(0.5)\nz ~ Normal(x, 1.0)\n",
            "pushfwd(fn(2.0 * _), lawof(x))",
        ),
    ] {
        let reason = refusal(&format!("{pre}d = {src}\nlp = logdensityof(d, 0.6)"));
        assert!(
            reason.contains("point mass that is not written as `Dirac(c)`"),
            "`{src}`: {reason}"
        );
    }
    // A draw whose measure has a stochastic ancestor: its law marginalises the
    // ancestor (`lawof(x)` of `x ~ Dirac(mu)`, `mu ~ Normal` is Normal), so the walk
    // keeps the value set and the marginal refuses as on main.
    let reason = refusal(
        "mu ~ Normal(0.0, 1.0)\nx ~ Dirac(mu)\nd = pushfwd(fn(2.0 * _), lawof(x))\n\
         lp = logdensityof(d, 1.0)",
    );
    assert!(
        reason.contains("no closed form covers this marginal"),
        "{reason}"
    );
    // The joint law of such a draw and its ancestor is a point mass in the draw
    // given the ancestor, beside a continuous part.
    let reason = refusal(
        "y = elementof(cartpow(reals, 2))\nmu ~ Normal(0.0, 1.0)\nx ~ Dirac(mu)\n\
         d = pushfwd(fn(2.0 .* _), lawof([x, mu]))\nlp = logdensityof(d, y)",
    );
    assert!(
        reason.contains("holds a point mass beside other components"),
        "{reason}"
    );
    // The law of a continuous draw keeps main's change of variables.
    let out =
        lp("x ~ Normal(0.0, 1.0)\nd = pushfwd(fn(2.0 * _), lawof(x))\nlp = logdensityof(d, 1.0)");
    assert!(out.contains("builtin_logdensityof Normal"), "{out}");
    assert!(
        out.contains("(abs 2.0)"),
        "the volume term log|2| stays:\n{out}"
    );
    // A Dirichlet behind a kernel keeps its free dimension.
    let out = lp(
        "a = elementof(cartpow(posreals, 3))\nK = functionof(Dirichlet(a), a = a)\n\
         d = pushfwd(fn(2.0 .* _), K([2.0, 3.0, 4.0]))\nlp = logdensityof(d, [0.4, 0.6, 1.0])",
    );
    let v = eval(&call_arg(&out, "sub", 1));
    assert!(close(v, TWO_LOG_2), "{v}\n{out}");
    // The unmapped mixture keeps its density.
    let out = lp(&format!("d = {continuous}\nlp = logdensityof(d, 1.0)"));
    assert!(out.contains("builtin_logdensityof Normal"), "{out}");
    // Over a discrete variate the counting path snaps the preimage, so the map
    // of a count mixture still lowers.
    let count = "superpose(weighted(0.3, Dirac(0)), weighted(0.7, Poisson(2.0)))";
    let out = lp(&format!(
        "d = pushfwd(fn(2 * _), {count})\nlp = logdensityof(d, 4)"
    ));
    assert!(out.contains("builtin_logdensityof Poisson"), "{out}");
}

#[test]
fn a_superposition_of_different_dimensions_refuses() {
    let mixed = "superpose(weighted(0.5, iid(Normal(0.0, 1.0), 2)), \
                 weighted(0.5, Dirichlet(alpha = [1.0, 1.0])))";
    for m in [
        mixed.to_string(),
        format!("pushfwd(fn(2.0 .* _ .+ 3.0), {mixed})"),
    ] {
        let reason = refusal(&format!("d = {m}\nlp = logdensityof(d, [0.25, 0.75])"));
        assert!(
            reason.contains("reference measures of different dimension"),
            "`{m}`: {reason}"
        );
    }
    // Two simplex laws of one length share a reference measure and still lower.
    let same = "superpose(weighted(0.5, Dirichlet(alpha = [1.0, 2.0])), \
                weighted(0.5, Dirichlet(alpha = [2.0, 1.0])))";
    let out = lp(&format!("d = {same}\nlp = logdensityof(d, [0.25, 0.75])"));
    assert!(out.contains("builtin_logdensityof Dirichlet"), "{out}");
}
