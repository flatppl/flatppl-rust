//! Cross-module measure-ref lowering: a `logdensityof`/`likelihoodof` whose
//! measure resolves through a `(%ref <loaded-module> member)` into a loaded
//! submodule graph carried by a [`flatppl_infer::ModuleBundle`].
use flatppl_determinizer::determinize_with;
use flatppl_infer::ModuleBundle;
use std::sync::Arc;

fn parse(src: &str) -> flatppl_core::Module {
    flatppl_syntax::parse(src).unwrap()
}

/// T6 characterization: `determinize_with(&m, &empty_bundle)` produces
/// byte-identical FlatPIR to `determinize(&m)` for a self-contained
/// same-module model — the delegation keeps every existing caller's behaviour.
#[test]
fn determinize_with_empty_bundle_matches_determinize() {
    let src = "x = draw(Normal(mu = 0.0, sigma = 1.0))\n\
               lp = logdensityof(lawof(record(x = x)), record(x = 0.5))";
    let mut m = parse(src);
    let _ = flatppl_infer::infer(&mut m);
    let a = flatppl_determinizer::determinize(&m);
    let b = determinize_with(&m, &ModuleBundle::new());
    assert_eq!(
        a.map(|x| flatppl_flatpir::write(&x)).ok(),
        b.map(|x| flatppl_flatpir::write(&x)).ok()
    );
}

/// T7: a cross-module likelihood over a `functionof`-reified kernel defined in a
/// loaded submodule lowers to a fully-formed `builtin_logdensityof`. Spec §04
/// "Reification and module scope": a measure crosses module boundaries freely
/// (`lawof(draw(m)) ≡ m`), so resolving a cross-module measure ref is spec-legal.
///
/// This uses the brief's `load_module("helpers.flatppl", center = a)` form: the
/// submodule's `center` parameter is substituted at the load boundary with the
/// host's `a` (spec §04 "Load-time substitution"). The determiniser honors that
/// `%assign`, so the kernel's `center` references resolve to host `a`; inference
/// accordingly reports the likelihood's free input as `a` (`%inputs a`), and the
/// θ point names `a`. The strengthened assertions check the lowering is a REAL
/// density: the θ value (`mu = 0.0`) is inlined into the distribution and the
/// observed data (`input_data`) is the variate — not merely that the op name is
/// present.
#[test]
fn cross_module_likelihood_lowers() {
    let helpers = "\
flatppl_compat = \"0.1\"
center = elementof(reals)
obs_kernel = functionof(Normal(mu = center, sigma = 1.0), center = center)";
    let model = "\
flatppl_compat = \"0.1\"
a = elementof(reals)
helpers = load_module(\"helpers.flatppl\", center = a)
input_data = 2.5
L = likelihoodof(helpers.obs_kernel, input_data)
lp = logdensityof(L, record(a = 0.0))";

    let mut hmod = parse(helpers);
    let _ = flatppl_infer::infer(&mut hmod);
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(hmod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let lowered =
        determinize_with(&mmod, &bundle).expect("cross-module likelihood must lower, not refuse");
    let pir = flatppl_flatpir::write(&lowered);
    assert!(
        pir.contains("builtin_logdensityof"),
        "cross-module kernel density did not lower to builtin_logdensityof; got:\n{pir}"
    );
    // θ field `a = 0.0` is inlined into the distribution's `mu` (honoring the
    // load-time `center = a` substitution), so the density is `Normal(mu = 0.0,
    // sigma = 1.0)` — a fully-determined kernel, not a free parameter.
    assert!(
        pir.contains("(%field mu 0.0)"),
        "θ value did not inline into the distribution `mu`; got:\n{pir}"
    );
    // The observed data baked into the likelihood is the variate.
    assert!(
        pir.contains("input_data"),
        "observed data (input_data) is not referenced as the variate; got:\n{pir}"
    );
}

/// Safety property the whole feature rests on: a `logdensityof` over a
/// cross-module kernel ref whose submodule is ABSENT from the bundle must refuse
/// cleanly (return `Err`), never panic and never lower against a missing
/// dependency. This is refuse-don't-mislower for the resolution path itself.
#[test]
fn cross_module_missing_bundle_refuses() {
    let model = "\
flatppl_compat = \"0.1\"
center = elementof(reals)
helpers = load_module(\"helpers.flatppl\", center = center)
input_data = 2.5
L = likelihoodof(helpers.obs_kernel, input_data)
lp = logdensityof(L, record(center = 0.0))";

    // Empty bundle: the `helpers.flatppl` dependency is not present.
    let bundle = ModuleBundle::new();
    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let result = determinize_with(&mmod, &bundle);
    assert!(
        result.is_err(),
        "cross-module likelihood over a missing bundle entry must refuse, not lower; got:\n{}",
        result
            .map(|l| flatppl_flatpir::write(&l))
            .unwrap_or_default()
    );
}

/// GAP C — a cross-module ref that is ITSELF the direct `logdensityof` target: a
/// submodule LIKELIHOOD handle `m.L`. The whole `likelihoodof(K, obs)` lives in
/// the submodule; the host merely `load_module`s it and queries `logdensityof(m.L,
/// θ)`. Spec §04 "Reification and module scope": a likelihood handle crosses
/// module boundaries freely, so this SHOULD lower. Before the fix the bare
/// `(%ref helpers L)` target refused ("primitive measure must be a built-in
/// constructor call"); now the entry grafts the referenced likelihood subtree and
/// dispatches on the grafted node through the existing likelihood path.
///
/// Mirrors [`cross_module_likelihood_lowers`], but the cross-module ref is the
/// TARGET (`logdensityof(helpers.L, …)`), not a kernel inside a host-built
/// `likelihoodof`. Same `load_module("helpers.flatppl", center = a)` %assign, so
/// θ names host `a` and the value `a = 0.0` inlines into the distribution `mu`.
#[test]
fn cross_module_likelihood_handle_lowers() {
    let helpers = "\
flatppl_compat = \"0.1\"
center = elementof(reals)
input_data = 2.5
L = likelihoodof(functionof(Normal(mu = center, sigma = 1.0), center = center), input_data)";
    let model = "\
flatppl_compat = \"0.1\"
a = elementof(reals)
helpers = load_module(\"helpers.flatppl\", center = a)
lp = logdensityof(helpers.L, record(a = 0.0))";

    let mut hmod = parse(helpers);
    let _ = flatppl_infer::infer(&mut hmod);
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(hmod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let lowered = determinize_with(&mmod, &bundle)
        .expect("cross-module likelihood HANDLE target must lower, not refuse");
    let pir = flatppl_flatpir::write(&lowered);
    assert!(
        pir.contains("builtin_logdensityof"),
        "cross-module likelihood-handle target did not lower to builtin_logdensityof; got:\n{pir}"
    );
    // θ field `a = 0.0` inlines into the distribution's `mu` (honoring the
    // load-time `center = a` substitution): a fully-determined kernel.
    assert!(
        pir.contains("(%field mu 0.0)"),
        "θ value did not inline into the distribution `mu`; got:\n{pir}"
    );
    // The observed data baked into the submodule likelihood is the variate.
    assert!(
        pir.contains("input_data"),
        "submodule observed data (input_data) is not referenced as the variate; got:\n{pir}"
    );
}

/// GAP C — a cross-module bare MEASURE handle `m.d` as the direct `logdensityof`
/// target (no likelihood layer, no θ). The submodule defines a fully-concrete
/// `d = Normal(mu = 0.0, sigma = 1.0)`; the host scores it at a scalar variate:
/// `logdensityof(m.d, 0.5)`. Spec §04: a measure crosses module boundaries freely
/// (`lawof(draw(m)) ≡ m`). The entry grafts the `Normal` constructor into the host
/// and the existing measure path emits one `builtin_logdensityof`.
#[test]
fn cross_module_measure_handle_lowers() {
    let helpers = "\
flatppl_compat = \"0.1\"
d = Normal(mu = 0.0, sigma = 1.0)";
    let model = "\
flatppl_compat = \"0.1\"
helpers = load_module(\"helpers.flatppl\")
lp = logdensityof(helpers.d, 0.5)";

    let mut hmod = parse(helpers);
    let _ = flatppl_infer::infer(&mut hmod);
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(hmod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let lowered = determinize_with(&mmod, &bundle)
        .expect("cross-module bare-measure HANDLE target must lower, not refuse");
    let pir = flatppl_flatpir::write(&lowered);
    assert!(
        pir.contains("builtin_logdensityof"),
        "cross-module measure-handle target did not lower to builtin_logdensityof; got:\n{pir}"
    );
    // The concrete submodule kernel `Normal(mu = 0.0, sigma = 1.0)` survives the
    // graft into the emitted density term.
    assert!(
        pir.contains("(%field mu 0.0)") && pir.contains("(%field sigma 1.0)"),
        "submodule Normal(mu = 0.0, sigma = 1.0) did not survive the graft; got:\n{pir}"
    );
}

/// GAP C, refuse-don't-mislower: a cross-module direct query TARGET whose graft
/// hits a submodule-dependency name collision with an unrelated host binding still
/// refuses cleanly (mirrors [`cross_module_name_collision_refuses`], but the
/// colliding submodule likelihood is the `logdensityof` target `helpers.L`, not a
/// kernel inside a host `likelihoodof`). The submodule kernel depends on an
/// internal `scale = 2.0` whose name collides with an unrelated host `scale =
/// 10.0`; grafting must NOT reuse the host binding — it refuses.
#[test]
fn cross_module_target_name_collision_refuses() {
    let helpers = "\
flatppl_compat = \"0.1\"
center = elementof(reals)
scale = 2.0
obs_kernel = functionof(Normal(mu = center, sigma = scale), center = center)
input_data = 2.5
L = likelihoodof(obs_kernel, input_data)";
    let model = "\
flatppl_compat = \"0.1\"
center = elementof(reals)
scale = 10.0
helpers = load_module(\"helpers.flatppl\", center = center)
lp = logdensityof(helpers.L, record(center = 0.0))";

    let mut hmod = parse(helpers);
    let _ = flatppl_infer::infer(&mut hmod);
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(hmod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let result = determinize_with(&mmod, &bundle);
    assert!(
        result.is_err(),
        "a cross-module TARGET whose graft collides with an unrelated host binding must refuse, \
         not silently reuse the host binding; got:\n{}",
        result
            .map(|l| flatppl_flatpir::write(&l))
            .unwrap_or_default()
    );
}

/// FIX for silent mislowering on a binding-name collision (refuse-don't-mislower):
/// a submodule kernel depends on an INTERNAL binding (`scale = 2.0`) whose name
/// collides with an UNRELATED host binding (`scale = 10.0`). Modules are
/// independent namespaces, so grafting must NOT reuse the host binding (which
/// would silently score `sigma = 10.0` instead of the submodule's `2.0`). The
/// determiniser refuses rather than emit a wrong density with no diagnostic.
#[test]
fn cross_module_name_collision_refuses() {
    let helpers = "\
flatppl_compat = \"0.1\"
center = elementof(reals)
scale = 2.0
obs_kernel = functionof(Normal(mu = center, sigma = scale), center = center)";
    let model = "\
flatppl_compat = \"0.1\"
center = elementof(reals)
scale = 10.0
helpers = load_module(\"helpers.flatppl\", center = center)
input_data = 2.5
L = likelihoodof(helpers.obs_kernel, input_data)
lp = logdensityof(L, record(center = 0.0))";

    let mut hmod = parse(helpers);
    let _ = flatppl_infer::infer(&mut hmod);
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(hmod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let result = determinize_with(&mmod, &bundle);
    assert!(
        result.is_err(),
        "a submodule dependency colliding with an unrelated host binding must refuse, \
         not silently reuse the host binding; got:\n{}",
        result
            .map(|l| flatppl_flatpir::write(&l))
            .unwrap_or_default()
    );
}
