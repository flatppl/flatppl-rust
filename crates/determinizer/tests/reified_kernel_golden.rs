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

#[test]
fn non_kernel_user_call_does_not_reduce() {
    // `f` is a plain function, not a kernel → application must NOT be treated
    // as a measure; the query refuses (a function is not a measure).
    let src = "\
f = x -> x + 1.0
z = f(2.0)
lp = logdensityof(z, 0.5)";
    let err = determinize(&parse_infer(src)).expect_err("a function value is not a measure");
    assert!(
        format!("{err:?}").contains("primitive measure"),
        "got: {err:?}"
    );
}

#[test]
fn kernel_marginalizing_internal_stochastic_refuses() {
    // `k = kernelof(y, mu = mu)` bounds ONLY `mu`; `sigma` traces to a stochastic
    // `s ~ Exponential`, an internal latent the kernel would have to marginalize.
    // Applying `k` at `mu = 1.5` reduces the body to `Normal(mu = 1.5, sigma = s)`,
    // whose density still depends on the unpinned latent `s` — a kchain-style
    // marginal over an internal stochastic latent → intractable per §06 "Density
    // of composed measures". The determiniser must REFUSE, never emit a
    // `builtin_logdensityof(Normal, record(mu = 1.5, sigma = (%ref self s)), v)`
    // that silently treats the latent as a fixed kwarg.
    //
    // CHARACTERIZATION TEST (not RED→GREEN): the invariant already holds without
    // any dedicated guard. `reduce_kernel_application` does substitute only `mu`,
    // so the reduced measure carries `(%ref self s)`, and `build_density_term`
    // does build a `builtin_logdensityof` referencing it — but that emitted
    // density KEEPS the `s = draw(Exponential(1.0))` binding live, so the driver's
    // next `find_measure_node` scan hits the residual `draw` (a `MEASURE_VOCAB`
    // op) and refuses in `apply_rule`'s "no determinization rule for this
    // measure-layer construct" fallthrough — `determinize` returns `Err`, never an
    // `Ok(module)` carrying the mislowered density. This test pins that so a
    // future change to the sweep / dispatch cannot let the marginal escape.
    let src = "\
mu = elementof(reals)
s = draw(Exponential(1.0))
y = draw(Normal(mu = mu, sigma = s))
k = kernelof(y, mu = mu)
dist = k(record(mu = 1.5))
lp = logdensityof(dist, 0.7)";
    let err =
        determinize(&parse_infer(src)).expect_err("marginal over an internal latent must refuse");
    let msg = format!("{err:?}");
    // Accept the current refuse wording (names the residual stochastic `draw` /
    // measure-layer construct) OR a future sharpened guard message that names the
    // intractable-marginal / stochastic cause directly.
    assert!(
        msg.contains("draw")
            || msg.contains("marginal")
            || msg.contains("stochastic")
            || msg.contains("kchain"),
        "refuse reason must name the intractable-marginal / residual-draw cause; got: {msg}"
    );
}
