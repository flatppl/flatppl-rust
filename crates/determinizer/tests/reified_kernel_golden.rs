//! Determiniser lowering of reified-kernel *applications* — `k(input)` where
//! `k = kernelof(...)`. The measure is a `%call(User(k), [input])` node; the
//! reduction substitutes `input`'s fields into the kernel body, yielding the
//! underlying law the existing density rules already lower.
use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

#[test]
fn kernel_application_over_fixed_input_lowers_to_builtin_logdensityof() {
    // A kernel whose ONLY stochastic ancestor is the boundary input `mu`.
    // Applying it at `mu = 1.5` fully pins the law → a plain `Normal(1.5, 2.0)`
    // that `build_density_term` turns into one `builtin_logdensityof`.
    let src = "\
mu = elementof(reals)
y = draw(Normal(mu = mu, sigma = 2.0))
k = kernelof(y, mu = mu)
dist = k(record(mu = 1.5))
lp = logdensityof(dist, 0.7)";
    let pir = flatppl_flatpir::write(&determinize(&parse_infer(src)).expect("must lower"));
    assert!(pir.contains("builtin_logdensityof"), "got:\n{pir}");
    assert!(
        pir.contains("(record (%field mu 1.5) (%field sigma 2.0))")
            || pir.contains("(%field sigma 2.0)"),
        "kernel body's Normal(mu=1.5, sigma=2.0) must survive reduction; got:\n{pir}"
    );
}
