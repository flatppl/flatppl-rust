//! `locscale(m, shift, scale)` with a scalar `scale` and a vector `shift` over a
//! vector law.
//!
//! §06 makes it the map x ↦ scale·x + shift, and §07 `mul` scales every cell by a
//! scalar, so it is the uniform affine map `scale .* x .+ shift`. Each of the
//! cells − deficit free coordinates scales by `scale`: volume `(n − 1)·log|scale|`
//! over a `stdsimplex(n)` or `zerosum(n)` support, `n·log|scale|` over a
//! full-dimensional one.
//!
//! flatppl-rust is not a density engine: the volume term is evaluated here from its
//! literal arithmetic and compared with mpmath 1.4.1; the base term is scored by an
//! engine in the testsuite.
use flatppl_determinizer::is_flatpdl;

mod common;
use common::{call_arg, determinize, pir_binding, pir_head};

/// 2·log 2, mpmath 1.4.1 at 50 digits: 1.3862943611198906188.
const TWO_LOG_2: f64 = 1.386_294_361_119_890_6;
/// 2·log 3, mpmath 1.4.1 at 50 digits: 2.1972245773362193828.
const TWO_LOG_3: f64 = 2.197_224_577_336_219_4;

const LOWER_DIMENSIONAL: &str = "the image of a law on a lower-dimensional support has no density";

fn module(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let errors: Vec<_> = flatppl_infer::infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "inference errors in:\n{src}\n{errors:?}");
    m
}

/// The bound value of `lp`, without its `(%bind lp …)` wrapper.
fn bound_lp(out: &flatppl_core::Module) -> String {
    let pir = flatppl_flatpir::write(out);
    assert!(is_flatpdl(out).is_ok(), "is_flatpdl failed:\n{pir}");
    call_arg(&pir_binding(&pir, "lp"), "%bind", 1)
}

fn lp(src: &str) -> String {
    bound_lp(&determinize(&module(src)).expect("must lower, not refuse"))
}

/// The same lowering with every egglog rule disabled.
fn lp_unoptimised(src: &str) -> String {
    let out = flatppl_determinizer::determinize_with_options_and_rules(
        &module(src),
        &flatppl_infer::ModuleBundle::new(),
        None,
        &flatppl_determinizer::LoweringOptions {
            disable_sufficient_statistics: true,
            ..Default::default()
        },
        &flatppl_determinizer::RulePolicy::all_disabled(),
    )
    .expect("must lower, not refuse");
    bound_lp(&out)
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
        return e.parse().unwrap_or_else(|_| panic!("not a literal: {e}"));
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

/// The volume term of a lowered `sub(<base density>, <volume>)`. An iid or
/// MvNormal base scores as a `sum` of per-cell builtin densities.
fn volume(out: &str) -> f64 {
    assert_eq!(pir_head(out), "sub", "{out}");
    assert!(
        call_arg(out, "sub", 0).contains("builtin_logdensityof"),
        "{out}"
    );
    eval(&call_arg(out, "sub", 1))
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 4.0 * f64::EPSILON, "{what}: {got}");
}

const DIRICHLET_234: &str = "d = locscale(Dirichlet(alpha = [2.0, 3.0, 4.0]), [1.0, 1.0, 1.0], 2.0)\n\
                             lp = logdensityof(d, [1.4, 1.6, 2.0])";

#[test]
fn a_simplex_law_takes_n_minus_one_cells() {
    // Dirichlet([2, 3, 4]) at x = [0.2, 0.3, 0.5]: logDirichlet(x) − 2·log 2 =
    // 0.63657682907155101126 (mpmath); main refused.
    close(volume(&lp(DIRICHLET_234)), TWO_LOG_2, "Dirichlet(2, 3, 4)");
    // Dirichlet([1, 1]) has density 1 on its simplex: −log 2 in all.
    close(
        volume(&lp(
            "d = locscale(Dirichlet(alpha = [1.0, 1.0]), [3.0, 3.0], 2.0)\n\
             lp = logdensityof(d, [3.5, 4.5])",
        )),
        // log 2, mpmath 1.4.1 at 50 digits: 0.69314718055994530942.
        std::f64::consts::LN_2,
        "Dirichlet(1, 1)",
    );
}

#[test]
fn the_volume_is_unchanged_with_the_optimiser_off() {
    close(
        volume(&lp_unoptimised(DIRICHLET_234)),
        TWO_LOG_2,
        "optimiser off",
    );
}

#[test]
fn locscale_and_the_written_map_lower_alike() {
    let written = lp(
        "d = pushfwd(fn(2.0 .* _ .+ [1.0, 1.0, 1.0]), Dirichlet(alpha = [2.0, 3.0, 4.0]))\n\
         lp = logdensityof(d, [1.4, 1.6, 2.0])",
    );
    assert_eq!(lp(DIRICHLET_234), written);
}

#[test]
fn a_full_dimensional_law_takes_every_cell() {
    for base in [
        "iid(Normal(0.0, 1.0), 2)",
        "MvNormal(mu = [0.0, 0.0], cov = rowstack([[1.0, 0.0], [0.0, 1.0]]))",
    ] {
        // The flatppl-js engine scores the MvNormal case at −4.173990532634454 against
        // scipy `multivariate_normal.logpdf((y − [1, 2]) / 3, 0, I) − 2·log 3`.
        let out = lp(&format!(
            "d = locscale({base}, [1.0, 2.0], 3.0)\nlp = logdensityof(d, [0.5, 0.5])"
        ));
        close(volume(&out), TWO_LOG_3, base);
    }
}

#[test]
fn a_query_scale_is_a_live_coefficient() {
    // The non-centred `locscale(iid(Normal(0, 1), J), mu, tau)` with `tau` an input.
    let out = lp(
        "tau = elementof(posreals)\nmu = elementof(cartpow(reals, 3))\n\
         inputs = (mu, tau)\n\
         lp = logdensityof(locscale(iid(Normal(0.0, 1.0), 3), mu, tau), [1.5, -0.5, 2.0])",
    );
    // The volume term is `%meta`-wrapped: read its `mul` through `call_arg`.
    let vol = call_arg(&out, "sub", 1);
    assert_eq!(call_arg(&vol, "mul", 0), "3.0", "{out}");
    assert!(
        call_arg(&vol, "mul", 1).contains("(abs (%ref self tau))"),
        "{out}"
    );
}

#[test]
fn a_point_mass_takes_no_volume_term() {
    // 0.3·[1.2, 2.3] + 0.1 = [0.46, 0.79]: the atom, gated on the forward image.
    let out = lp("d = locscale(Dirac([1.2, 2.3]), [0.1, 0.1], 0.3)\n\
         lp = logdensityof(d, [0.46, 0.79])");
    assert_eq!(pir_head(&out), "ifelse", "{out}");
    assert!(!out.contains("(log (abs 0.3))"), "{out}");
}

#[test]
fn shapes_without_a_density_refuse() {
    for (measure, point, reason) in [
        // §07 `sub` takes operands of one shape.
        (
            "locscale(Dirichlet(alpha = [2.0, 3.0, 4.0]), 1.0, 2.0)",
            "[1.4, 1.6, 2.0]",
            "requires a `shift` of the SAME shape as the variate, and this one is a scalar",
        ),
        (
            "locscale(Dirichlet(alpha = [2.0, 3.0, 4.0]), [1.0, 1.0], 2.0)",
            "[1.4, 1.6, 2.0]",
            "a vector of length 2, not 3",
        ),
        (
            "locscale(Dirichlet(alpha = [2.0, 3.0, 4.0]), [1.0, 1.0, 1.0], 0.0)",
            "[1.0, 1.0, 1.0]",
            "zero scale entry",
        ),
        // A matrix scale over a chart, and any map of a correlation law.
        (
            "locscale(Dirichlet(alpha = [2.0, 3.0, 4.0]), [1.0, 1.0, 1.0], \
             rowstack([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]))",
            "[1.4, 1.6, 2.0]",
            LOWER_DIMENSIONAL,
        ),
        (
            "locscale(LKJ(2, 1.0), rowstack([[0.0, 0.0], [0.0, 0.0]]), 2.0)",
            "rowstack([[2.0, 0.6], [0.6, 2.0]])",
            LOWER_DIMENSIONAL,
        ),
    ] {
        let got = refusal(&format!("d = {measure}\nlp = logdensityof(d, {point})"));
        assert!(got.contains(reason), "`{measure}`: {got}");
    }
}
