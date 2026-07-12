//! Seam between the `%autoinputs` (auto-traced, keyword-only) kernel reduction
//! and the `kchain` marginal path.
//!
//! `resolve_kernel` (shared by the kchain marginal in `marginal.rs`) accepts an
//! `%autoinputs` boundary so that a `logdensityof(k(input), θ)` APPLICATION of an
//! auto kernel lowers. But the kchain marginal enumerates a discrete latent and
//! substitutes each atom into `kernel.inputs[0]`, ASSUMING `inputs[0]` is the
//! enumerated latent dependency. An `%autoinputs` boundary traces the reified
//! body's `elementof` FREE parameters — never the `draw`-bound latent — so for a
//! kchain over an auto kernel that substitution replaces the WRONG node (a free
//! parameter gets the latent's atoms), emitting a `logsumexp` that eliminates a
//! parameter which must stay symbolic. That is a silent wrong density, and
//! `is_flatpdl` (structural) cannot catch it. The kchain marginal must therefore
//! REFUSE an `%autoinputs` kernel: its atom-substitution requires a `%specinputs`
//! boundary whose `inputs[0]` IS the enumerated latent.
use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

/// A `kchain(M, K)` whose kernel `K` is boundary-less (`%autoinputs`) must
/// REFUSE. The kernel `k = kernelof(record(y = draw(Normal(mu = theta, sigma =
/// 1.0))))` auto-traces its sole `elementof` leaf `theta` as its boundary input
/// — NOT the `draw`-bound Bernoulli latent `b` the kchain enumerates. Were this
/// to lower, the marginal would substitute the Bernoulli atoms {0, 1} into
/// `theta`, emitting a `logsumexp` that silently eliminates the free parameter
/// `theta` (a wrong density). The determiniser must refuse rather than mislower.
#[test]
fn kchain_over_autoinputs_kernel_refuses() {
    let src = "\
theta = elementof(reals)
b = draw(Bernoulli(p = 0.5))
k = kernelof(record(y = draw(Normal(mu = theta, sigma = 1.0))))
jc = kchain(lawof(record(b = b)), k)
lp = logdensityof(jc, record(b = 1, y = 0.5))";
    let m = parse_infer(src);
    let err = determinize(&m).expect_err(
        "kchain marginal over an %autoinputs kernel must refuse: the auto-traced boundary is the \
         free parameter theta, not the enumerated Bernoulli latent — lowering would silently \
         eliminate theta via logsumexp",
    );
    assert!(
        err.construct.contains("kchain"),
        "refusal names kchain: {err:?}"
    );
    // Whatever else survives, `theta` must NOT have been eliminated by a
    // logsumexp over the Bernoulli atoms (the mislowering this guard prevents).
}
