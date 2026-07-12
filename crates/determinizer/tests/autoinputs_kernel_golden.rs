//! Determiniser lowering of `%autoinputs` (keyword/auto-traced) kernel
//! *applications*. A boundary-less `kernelof(body)` / `functionof(body)` traces
//! its inputs to the body's `elementof` parametric leaves (spec §04 "Reification
//! to functions and kernels"); with no boundary spec, "no argument order can be
//! inferred", so the reified callable supports **keyword arguments only**.
//!
//! These pin that `reduce_kernel_application` β-reduces such a keyword
//! application by binding each supplied `k(name = value)` to the auto-traced
//! boundary input of the same name — and REFUSES a positional application or a
//! name mismatch (a wrong keyword-boundary bind would be a silent wrong density).
use flatppl_determinizer::{determinize, determinize_with};
use flatppl_infer::ModuleBundle;
use std::sync::Arc;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn parse(src: &str) -> flatppl_core::Module {
    flatppl_syntax::parse(src).unwrap()
}

/// Same-module auto-traced kernel, keyword application. `k = kernelof(y)` has no
/// boundary spec, so inference auto-traces the sole `elementof` leaf `mu` as its
/// boundary input (`%autoinputs ((mu (%ref self mu)))`). Applying `k(mu = 1.5)`
/// binds `mu` by keyword into the underlying law `Normal(mu = mu, sigma = 2.0)`,
/// pinning it to `Normal(mu = 1.5, sigma = 2.0)` — one `builtin_logdensityof`.
#[test]
fn same_module_autoinputs_kernel_application_lowers() {
    let src = "\
mu = elementof(reals)
y = draw(Normal(mu = mu, sigma = 2.0))
k = kernelof(y)
dist = k(mu = 1.5)
lp = logdensityof(dist, 0.7)";
    let pir = flatppl_flatpir::write(&determinize(&parse_infer(src)).expect("must lower"));
    assert!(pir.contains("builtin_logdensityof"), "got:\n{pir}");
    assert!(
        pir.contains("(%field mu 1.5)"),
        "keyword-bound `mu = 1.5` must β-reduce into the law's `mu`; got:\n{pir}"
    );
}

/// Cross-module auto-traced kernel application. The submodule's `k =
/// functionof(Normal(mu = center, sigma = 1.0))` is boundary-less, so its input
/// `center` is auto-traced. The host `load_module`s it and scores the applied
/// kernel `logdensityof(m.k(center = 0.0), 0.5)`. The graft carries `Inputs::Auto`
/// through, the driver re-infers (repopulating `auto_inputs_of` on the grafted
/// node), and `reduce_kernel_application` binds `center = 0.0` by keyword into
/// the kernel body — a fully-determined `Normal(mu = 0.0, sigma = 1.0)`.
#[test]
fn crossmodule_autoinputs_kernel_application_lowers() {
    let helpers = "\
flatppl_compat = \"0.1\"
center = elementof(reals)
k = functionof(Normal(mu = center, sigma = 1.0))";
    let model = "\
flatppl_compat = \"0.1\"
m = load_module(\"helpers.flatppl\")
lp = logdensityof(m.k(center = 0.0), 0.5)";

    let mut hmod = parse(helpers);
    let _ = flatppl_infer::infer(&mut hmod);
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(hmod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let lowered = determinize_with(&mmod, &bundle)
        .expect("cross-module %autoinputs kernel application must lower, not refuse");
    let pir = flatppl_flatpir::write(&lowered);
    assert!(
        pir.contains("builtin_logdensityof"),
        "cross-module auto-traced kernel application did not lower to builtin_logdensityof; \
         got:\n{pir}"
    );
    // The keyword-bound `center = 0.0` β-reduces into `mu`, and the submodule's
    // `sigma = 1.0` survives the graft — a fully-determined law.
    assert!(
        pir.contains("(%field mu 0.0)") && pir.contains("(%field sigma 1.0)"),
        "applied `center = 0.0` did not β-reduce into Normal(mu = 0.0, sigma = 1.0); got:\n{pir}"
    );
}

/// A POSITIONAL application of an `%autoinputs` kernel must REFUSE. Spec §04:
/// without a boundary spec "no argument order can be inferred", so the callable
/// is keyword-only. Binding `k(1.5)` by position would silently attach `1.5` to
/// an arbitrarily-ordered traced input — a wrong-density hazard — so it refuses.
#[test]
fn positional_application_of_autoinputs_kernel_refuses() {
    let src = "\
mu = elementof(reals)
y = draw(Normal(mu = mu, sigma = 2.0))
k = kernelof(y)
dist = k(1.5)
lp = logdensityof(dist, 0.7)";
    let err = determinize(&parse_infer(src))
        .expect_err("positional application of a keyword-only %autoinputs kernel must refuse");
    let _ = format!("{err:?}");
}

/// A keyword application whose name does not match the auto-traced boundary must
/// REFUSE. Here the boundary input is `mu`, but the application supplies `nu`;
/// there is no `mu` keyword to bind (and `nu` matches nothing), so the reduction
/// refuses rather than leave the boundary input free (a wrong density).
#[test]
fn autoinputs_application_name_mismatch_refuses() {
    let src = "\
mu = elementof(reals)
y = draw(Normal(mu = mu, sigma = 2.0))
k = kernelof(y)
dist = k(nu = 1.5)
lp = logdensityof(dist, 0.7)";
    let err = determinize(&parse_infer(src))
        .expect_err("a keyword name not matching the auto-traced boundary must refuse");
    let _ = format!("{err:?}");
}
