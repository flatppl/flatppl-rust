//! The nullary form of the measure-kernel evaluation primitives (spec §07): a
//! measure is a nullary kernel, so `builtin_logdensityof(M, x)` and the others
//! omit `kernel_input`. A closed built-in measure `K(args)` lowers to the same
//! FlatPDL as the kernel form `builtin_*(K, record(args), ...)`.

use flatppl_determinizer::determinize_with_roots;

fn lower(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = flatppl_infer::infer(&mut m);
    assert!(diags.is_empty(), "{src}\n{diags:?}");
    let roots = [m.intern("outputs")];
    let out = determinize_with_roots(&m, &flatppl_infer::ModuleBundle::new(), Some(&roots))
        .unwrap_or_else(|e| panic!("{src}\n{e:?}"));
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
    flatppl_flatpir::write(&out)
}

fn assert_same(nullary: &str, kernel: &str) {
    let head = "mu = elementof(reals)\nx = elementof(reals)\nxs = elementof(cartpow(reals, 3))\n\
                v = elementof(cartpow(reals, 2))\nstate = rnginit(7)\n";
    let tail = "inputs = (mu, x, xs, v)\noutputs = (q)\n";
    assert_eq!(
        lower(&format!("{head}{nullary}\n{tail}")),
        lower(&format!("{head}{kernel}\n{tail}")),
        "{nullary}"
    );
}

#[test]
fn a_closed_builtin_measure_lowers_like_its_kernel_form() {
    for (nullary, kernel) in [
        (
            "q = builtin_logdensityof(Normal(mu, 2.0), x)",
            "q = builtin_logdensityof(Normal, record(mu = mu, sigma = 2.0), x)",
        ),
        (
            "M = Normal(mu = mu, sigma = 2.0)\nN = M\nq = builtin_logdensityof(N, x)",
            "q = builtin_logdensityof(Normal, record(mu = mu, sigma = 2.0), x)",
        ),
        (
            "q = builtin_logdensityof(Poisson(rate = 3.0), 2)",
            "q = builtin_logdensityof(Poisson, record(rate = 3.0), 2)",
        ),
        (
            "q = builtin_logdensityof(MvNormal([0.0, 1.0], eye(2)), v)",
            "q = builtin_logdensityof(MvNormal, record(mu = [0.0, 1.0], cov = eye(2)), v)",
        ),
        (
            "q = builtin_logdensityof(Lebesgue(reals), x)",
            "q = builtin_logdensityof(Lebesgue, record(support = reals), x)",
        ),
        (
            "q = builtin_touniform(Exponential(rate = 2.0), x)",
            "q = builtin_touniform(Exponential, record(rate = 2.0), x)",
        ),
        (
            "q = builtin_fromnormal(Normal(mu, 2.0), x)",
            "q = builtin_fromnormal(Normal, record(mu = mu, sigma = 2.0), x)",
        ),
        (
            "q = builtin_sample(state, Exponential(2.0), 3)",
            "q = builtin_sample(state, Exponential, record(rate = 2.0), 3)",
        ),
        (
            "q = builtin_sample(state, Normal(mu, 2.0))",
            "q = builtin_sample(state, Normal, record(mu = mu, sigma = 2.0))",
        ),
    ] {
        assert_same(nullary, kernel);
    }
}

/// A broadcast holds the measure constant, as for `logdensityof.(M, xs)`.
#[test]
fn a_broadcast_holds_the_measure_constant() {
    assert_same(
        "q = builtin_logdensityof.(Normal(mu, 2.0), xs)",
        "q = logdensityof.(Normal(mu, 2.0), xs)",
    );
    // A transport takes the same per-cell form as the density.
    let src = |head: &str| {
        format!(
            "mu = elementof(reals)\nxs = elementof(cartpow(reals, 3))\n\
             q = {head}.(Normal(mu, 2.0), xs)\ninputs = (mu, xs)\noutputs = (q)\n"
        )
    };
    assert_eq!(
        lower(&src("builtin_touniform")),
        lower(&src("builtin_logdensityof")).replace("builtin_logdensityof", "builtin_touniform")
    );
}

/// The kernel of a measure-input kernel stays a measure, which FlatPDL refuses.
#[test]
fn a_measure_input_kernel_stays_refused() {
    let src = "x = elementof(cartpow(reals, 3))\n\
               q = builtin_logdensityof(PoissonProcess(Normal(0.0, 1.0)), x)\n\
               inputs = (x)\noutputs = (q)\n";
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let roots = [m.intern("outputs")];
    assert!(
        determinize_with_roots(&m, &flatppl_infer::ModuleBundle::new(), Some(&roots)).is_err(),
        "a measure-input kernel must refuse"
    );
}
