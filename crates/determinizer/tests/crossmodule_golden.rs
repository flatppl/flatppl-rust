//! Cross-module measure-ref lowering: a `logdensityof`/`likelihoodof` whose
//! measure resolves through a `(%ref <loaded-module> member)` into a loaded
//! submodule graph carried by a [`flatppl_infer::ModuleBundle`].
use flatppl_determinizer::{determinize_with, determinize_with_roots};
use flatppl_infer::ModuleBundle;
use std::sync::Arc;

fn parse(src: &str) -> flatppl_core::Module {
    flatppl_syntax::parse(src).unwrap()
}

fn assert_outputs(model: &str, bundle: &ModuleBundle, expected: &str) {
    let mut model = parse(model);
    let (_, bundle) = flatppl_infer::infer_module_with_inputs(
        &mut model,
        bundle.clone(),
        flatppl_infer::Level::Shape,
    );
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let actual = determinize_with_roots(&model, &bundle, Some(&roots)).unwrap();
    let mut expected = parse(expected);
    let roots = [expected.intern("inputs"), expected.intern("outputs")];
    let expected = determinize_with_roots(&expected, &ModuleBundle::new(), Some(&roots)).unwrap();
    assert_eq!(
        flatppl_flatpir::write(&actual),
        flatppl_flatpir::write(&expected)
    );
}

fn assert_normal_score(
    mut model: flatppl_core::Module,
    bundle: &ModuleBundle,
    x: f64,
    mu: f64,
    sigma: f64,
) {
    let roots = [model.intern("lp")];
    let actual = determinize_with_roots(&model, bundle, Some(&roots)).unwrap();
    let mut expected = parse(&format!(
        "lp = builtin_logdensityof(Normal, record(mu = {mu:?}, sigma = {sigma:?}), {x:?})"
    ));
    let roots = [expected.intern("lp")];
    let expected = determinize_with_roots(&expected, &ModuleBundle::new(), Some(&roots)).unwrap();
    assert_eq!(
        flatppl_flatpir::write(&actual),
        flatppl_flatpir::write(&expected)
    );
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

/// An imported likelihood keeps its scale when the host defines the same name.
#[test]
fn cross_module_target_name_collision_preserves_scale() {
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

    assert_normal_score(mmod, &bundle, 2.5, 0.0, 2.0);
}

/// GAP D (Symptom 1 — iid size on a freshly-grafted target): a cross-module
/// likelihood HANDLE `m.L` whose kernel wraps a STATIC-size `iid(Normal, 3)`,
/// with the kernel bound SEPARATELY in the submodule (`k = functionof(iid(...),
/// center = center)`). Scoring `logdensityof(m.L, θ)` must lower the iid to its
/// axis-native broadcast density (a PRIMITIVE `Normal` kernel — see
/// `lower_iid`'s primitive-kernel fast path). Before the fix, grafting `m.L`
/// inlined the iid subtree UNTYPED and the very same lowering call read
/// `iid_static_size` off the (still type-less) iid node → `None` → refuse ("iid
/// size is not a statically-resolved 1-D count"), even though the identical
/// model lowers fine same-module (where inference had already typed the iid).
/// The defer-and-reloop fix grafts first, reloops so inference types the
/// grafted iid domain, then lowers with a resolved static size.
#[test]
fn cross_module_iid_kernel_handle_lowers() {
    let helpers = "\
flatppl_compat = \"0.1\"
center = elementof(reals)
obs_data = [1.0, 2.0, 3.0]
k = functionof(iid(Normal(mu = center, sigma = 1.0), 3), center = center)
L = likelihoodof(k, obs_data)";
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
        .expect("cross-module iid-kernel HANDLE target must lower, not refuse on iid size");
    let pir = flatppl_flatpir::write(&lowered);
    assert!(
        pir.contains("builtin_logdensityof"),
        "cross-module iid-kernel handle did not lower to builtin_logdensityof; got:\n{pir}"
    );
    // The static-size-3 iid over a primitive Normal kernel lowers to ONE
    // axis-native broadcast head, not 3 unrolled terms.
    let n_terms = pir.matches("builtin_logdensityof").count();
    assert_eq!(
        n_terms, 1,
        "iid(Normal, 3) must lower to one broadcast head; got {n_terms}:\n{pir}"
    );
    assert!(
        pir.contains("(broadcast builtin_logdensityof Normal") && pir.contains("(sum "),
        "iid density is sum(broadcast(builtin_logdensityof, Normal, …)):\n{pir}"
    );
}

/// GAP D (Symptom 2 — dead grafted kernel binding): a cross-module likelihood
/// HANDLE `m.L` whose kernel is a SEPARATELY-NAMED submodule binding
/// (`obs_kernel = functionof(Normal(...), center = center)`; `L =
/// likelihoodof(obs_kernel, obs)`). Grafting `m.L` pulls `obs_kernel` in as its
/// own standalone host binding; after the query lowers, that binding is dead but
/// (before the fix) survived the sweep — the sweep ran inside the same
/// graft+lower call, BEFORE re-inference typed `obs_kernel` as `Kernel`, so the
/// type-based sweep arm did not catch it, and `is_flatpdl` then refused it as
/// `KernelNotBuiltinArg`. The defer-and-reloop fix reloops so inference types the
/// grafted kernel binding before the post-lowering sweep, which then removes it.
#[test]
fn cross_module_named_kernel_handle_lowers() {
    let helpers = "\
flatppl_compat = \"0.1\"
center = elementof(reals)
input_data = 2.5
obs_kernel = functionof(Normal(mu = center, sigma = 1.0), center = center)
L = likelihoodof(obs_kernel, input_data)";
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

    let lowered = determinize_with(&mmod, &bundle).expect(
        "cross-module named-kernel HANDLE target must lower, not refuse (KernelNotBuiltinArg)",
    );
    let pir = flatppl_flatpir::write(&lowered);
    assert!(
        pir.contains("builtin_logdensityof"),
        "cross-module named-kernel handle did not lower to builtin_logdensityof; got:\n{pir}"
    );
    // θ field `a = 0.0` inlines into the distribution's `mu` (honoring the
    // load-time `center = a` substitution): a fully-determined kernel.
    assert!(
        pir.contains("(%field mu 0.0)"),
        "θ value did not inline into the distribution `mu`; got:\n{pir}"
    );
}

/// GAP D deferred minor (refuse-don't-mislower, TARGET-ref form): a bare
/// `logdensityof(m.MissingThing, θ)` whose module ref is the DIRECT target and
/// whose member is absent from the bundle must refuse cleanly (`Err`), never
/// panic. Mirrors [`cross_module_missing_bundle_refuses`] (which puts the missing
/// ref inside a host `likelihoodof`) but exercises the direct-target graft path.
#[test]
fn cross_module_missing_dependency_target_refuses() {
    // The submodule exists in the bundle but has NO `MissingThing` member.
    let helpers = "\
flatppl_compat = \"0.1\"
d = Normal(mu = 0.0, sigma = 1.0)";
    let model = "\
flatppl_compat = \"0.1\"
helpers = load_module(\"helpers.flatppl\")
lp = logdensityof(helpers.MissingThing, record(a = 0.0))";

    let mut hmod = parse(helpers);
    let _ = flatppl_infer::infer(&mut hmod);
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(hmod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let result = determinize_with(&mmod, &bundle);
    assert!(
        result.is_err(),
        "a cross-module direct TARGET whose member is absent from the bundle must refuse, \
         not lower; got:\n{}",
        result
            .map(|l| flatppl_flatpir::write(&l))
            .unwrap_or_default()
    );
}

/// A likelihood built from an imported kernel uses that module's scale.
#[test]
fn cross_module_name_collision_preserves_scale() {
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

    assert_normal_score(mmod, &bundle, 2.5, 0.0, 2.0);
}

/// Refuse-don't-mislower for a TWO-LEVEL nested cross-module ref whose second
/// module is ABSENT from the bundle: the host loads `mid` (`middle.flatppl`),
/// whose queried handle `d` is itself just a reference `nested.val` into a
/// SECOND loaded module (`middle.flatppl`'s own `nested =
/// load_module("leaf.flatppl")`). The host's bundle carries `middle.flatppl`
/// and `unrelated.flatppl` but deliberately NOT `leaf.flatppl` — so when the
/// recursive graft resolves `middle.flatppl`'s OWN `nested` alias against the
/// bundle it finds no entry and refuses (`resolve_src_module_ref` → `None`).
///
/// Critically, the host ALSO happens to define its own, wholly UNRELATED
/// binding named `nested` (`load_module("unrelated.flatppl")`). The recursive
/// graft reads `middle.flatppl`'s OWN `nested` binding (not the host's), so it
/// can NEVER be fooled into scoring against `unrelated.flatppl`'s `val =
/// Normal(mu = 999.0, …)` — the missing `leaf.flatppl` makes it refuse cleanly,
/// with no diagnostic-free mislowering possible. (This is the nested analogue of
/// [`cross_module_missing_bundle_refuses`]: a nested `load_module` whose target
/// module is not in the bundle refuses rather than lower against a wrong or a
/// same-named-but-unrelated module.)
#[test]
fn cross_module_nested_load_module_ref_refuses() {
    let leaf = "\
flatppl_compat = \"0.1\"
val = Normal(mu = 0.0, sigma = 1.0)";
    let unrelated = "\
flatppl_compat = \"0.1\"
val = Normal(mu = 999.0, sigma = 1.0)";
    let middle = "\
flatppl_compat = \"0.1\"
nested = load_module(\"leaf.flatppl\")
d = nested.val";
    let model = "\
flatppl_compat = \"0.1\"
nested = load_module(\"unrelated.flatppl\")
mid = load_module(\"middle.flatppl\")
lp = logdensityof(mid.d, 0.5)";

    let mut leaf_mod = parse(leaf);
    let _ = flatppl_infer::infer(&mut leaf_mod);
    let mut unrelated_mod = parse(unrelated);
    let _ = flatppl_infer::infer(&mut unrelated_mod);
    let mut middle_mod = parse(middle);
    let _ = flatppl_infer::infer(&mut middle_mod);

    // The bundle is flat and keyed by path: it carries `middle.flatppl` and
    // `unrelated.flatppl` (both direct host dependencies) but deliberately NOT
    // `leaf.flatppl` — `middle.flatppl`'s own nested dependency is invisible
    // to the host's bundle, exactly as it would be from a real loader that
    // only resolves one `load_module` hop per bundle.
    let mut bundle = ModuleBundle::new();
    bundle.insert("middle.flatppl", Arc::new(middle_mod));
    bundle.insert("unrelated.flatppl", Arc::new(unrelated_mod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let result = determinize_with(&mmod, &bundle);
    assert!(
        result.is_err(),
        "a queried handle whose grafted body itself references a SECOND loaded module \
         (a 2-level-nested load_module) must refuse, never silently lower against an \
         unrelated same-named host binding and never panic; got:\n{}",
        result
            .map(|l| flatppl_flatpir::write(&l))
            .unwrap_or_default()
    );
}

/// A TWO-LEVEL nested cross-module ref RESOLVES and lowers when every module on
/// the chain is present in the bundle (spec §04 "Reification and module scope":
/// a measure crosses module boundaries transitively). The host loads `A`
/// (`a.flatppl`); `A`'s queried handle `x` is itself a cross-module ref
/// `inner.d` into `A`'s OWN loaded module `B` (`b.flatppl`); `B` defines a
/// concrete `d = Normal(mu = 0.0, sigma = 1.0)`. Scoring `logdensityof(A.x, 0.5)`
/// must chase both hops — graft `A`'s subtree into the host, and when that graft
/// meets the nested `inner.d` ref, recursively graft `B`'s `d` — so the query
/// lowers to a fully-formed `builtin_logdensityof`. Before the fix the graft
/// walk refused outright at the nested `RefNs::Module` ref ("nested load_module
/// … not supported"); the recursive graft resolves it instead.
#[test]
fn nested_cross_module_ref_lowers() {
    let leaf = "\
flatppl_compat = \"0.1\"
d = Normal(mu = 0.0, sigma = 1.0)";
    let mid = "\
flatppl_compat = \"0.1\"
inner = load_module(\"b.flatppl\")
x = inner.d";
    let model = "\
flatppl_compat = \"0.1\"
outer = load_module(\"a.flatppl\")
lp = logdensityof(outer.x, 0.5)";

    let mut leaf_mod = parse(leaf);
    let _ = flatppl_infer::infer(&mut leaf_mod);
    let mut mid_mod = parse(mid);
    let _ = flatppl_infer::infer(&mut mid_mod);

    // Both modules on the chain are present: `a.flatppl` (the host's direct
    // dependency) and `b.flatppl` (`a.flatppl`'s own nested dependency).
    let mut bundle = ModuleBundle::new();
    bundle.insert("a.flatppl", Arc::new(mid_mod));
    bundle.insert("b.flatppl", Arc::new(leaf_mod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let lowered = determinize_with(&mmod, &bundle).expect(
        "a 2-level-nested cross-module ref with every module present must lower, not refuse",
    );
    let pir = flatppl_flatpir::write(&lowered);
    assert!(
        pir.contains("builtin_logdensityof"),
        "nested cross-module ref did not lower to builtin_logdensityof; got:\n{pir}"
    );
    // The leaf module's concrete kernel `Normal(mu = 0.0, sigma = 1.0)` survives
    // both graft hops into the emitted density term.
    assert!(
        pir.contains("(%field mu 0.0)") && pir.contains("(%field sigma 1.0)"),
        "leaf Normal(mu = 0.0, sigma = 1.0) did not survive the recursive graft; got:\n{pir}"
    );
}

/// Refuse-don't-mislower: a nested cross-module ref whose SECOND module is absent
/// from the bundle refuses. Same chain as [`nested_cross_module_ref_lowers`] but
/// the bundle carries only `a.flatppl`; when the recursive graft resolves
/// `a.flatppl`'s own `inner = load_module("b.flatppl")` it finds no `b.flatppl`
/// entry and refuses rather than lower against a missing dependency.
#[test]
fn nested_ref_missing_module_refuses() {
    let mid = "\
flatppl_compat = \"0.1\"
inner = load_module(\"b.flatppl\")
x = inner.d";
    let model = "\
flatppl_compat = \"0.1\"
outer = load_module(\"a.flatppl\")
lp = logdensityof(outer.x, 0.5)";

    let mut mid_mod = parse(mid);
    let _ = flatppl_infer::infer(&mut mid_mod);

    // Only `a.flatppl` is present; its own nested `b.flatppl` dependency is not.
    let mut bundle = ModuleBundle::new();
    bundle.insert("a.flatppl", Arc::new(mid_mod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let result = determinize_with(&mmod, &bundle);
    assert!(
        result.is_err(),
        "a nested cross-module ref whose second module is absent from the bundle must refuse, \
         not lower; got:\n{}",
        result
            .map(|l| flatppl_flatpir::write(&l))
            .unwrap_or_default()
    );
}

/// Nested imports keep their own scale despite a same-named host binding.
#[test]
fn nested_graft_name_collision_preserves_scale() {
    let leaf = "\
flatppl_compat = \"0.1\"
scale = 2.0
d = Normal(mu = 0.0, sigma = scale)";
    let mid = "\
flatppl_compat = \"0.1\"
inner = load_module(\"b.flatppl\")
x = inner.d";
    let model = "\
flatppl_compat = \"0.1\"
scale = 10.0
outer = load_module(\"a.flatppl\")
lp = logdensityof(outer.x, 0.5)";

    let mut leaf_mod = parse(leaf);
    let _ = flatppl_infer::infer(&mut leaf_mod);
    let mut mid_mod = parse(mid);
    let _ = flatppl_infer::infer(&mut mid_mod);

    let mut bundle = ModuleBundle::new();
    bundle.insert("a.flatppl", Arc::new(mid_mod));
    bundle.insert("b.flatppl", Arc::new(leaf_mod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    assert_normal_score(mmod, &bundle, 0.5, 0.0, 2.0);
}

/// Same-named dependencies from different modules retain both values.
#[test]
fn nested_submodule_names_preserve_each_value() {
    let leaf = "\
flatppl_compat = \"0.1\"
scale = 99.0
val = scale";
    let mid = "\
flatppl_compat = \"0.1\"
scale = 2.0
inner = load_module(\"leaf.flatppl\")
d = Normal(mu = inner.val, sigma = scale)";
    let model = "\
flatppl_compat = \"0.1\"
outer = load_module(\"mid.flatppl\")
lp = logdensityof(outer.d, 0.5)";

    let mut leaf_mod = parse(leaf);
    let _ = flatppl_infer::infer(&mut leaf_mod);
    let mut mid_mod = parse(mid);
    let _ = flatppl_infer::infer(&mut mid_mod);

    let mut bundle = ModuleBundle::new();
    bundle.insert("mid.flatppl", Arc::new(mid_mod));
    bundle.insert("leaf.flatppl", Arc::new(leaf_mod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    assert_normal_score(mmod, &bundle, 0.5, 99.0, 2.0);
}

/// Two paths into one nested module share the same dependency.
#[test]
fn nested_same_module_diamond_still_lowers() {
    let leaf = "\
flatppl_compat = \"0.1\"
val = 1.0";
    let mid = "\
flatppl_compat = \"0.1\"
inner = load_module(\"leaf.flatppl\")
d = Normal(mu = inner.val, sigma = inner.val)";
    let model = "\
flatppl_compat = \"0.1\"
outer = load_module(\"mid.flatppl\")
lp = logdensityof(outer.d, 0.5)";

    let mut leaf_mod = parse(leaf);
    let _ = flatppl_infer::infer(&mut leaf_mod);
    let mut mid_mod = parse(mid);
    let _ = flatppl_infer::infer(&mut mid_mod);

    let mut bundle = ModuleBundle::new();
    bundle.insert("mid.flatppl", Arc::new(mid_mod));
    bundle.insert("leaf.flatppl", Arc::new(leaf_mod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    assert_normal_score(mmod, &bundle, 0.5, 1.0, 1.0);
}

/// Termination: a CYCLIC module graph refuses rather than recursing forever.
/// `A` (`a.flatppl`) loads `B` and defines `x = b.d`; `B` (`b.flatppl`) loads `A`
/// and defines `d = a.x` — so resolving `A.x` chases `x → B.d → A.x → …`. The
/// source/member cycle guard detects re-entry even as each load creates a new
/// instance.
#[test]
fn cyclic_module_graph_refuses() {
    let a = "\
flatppl_compat = \"0.1\"
b = load_module(\"b.flatppl\")
x = b.d";
    let b = "\
flatppl_compat = \"0.1\"
a = load_module(\"a.flatppl\")
d = a.x";
    let model = "\
flatppl_compat = \"0.1\"
outer = load_module(\"a.flatppl\")
lp = logdensityof(outer.x, 0.5)";

    let mut a_mod = parse(a);
    let _ = flatppl_infer::infer(&mut a_mod);
    let mut b_mod = parse(b);
    let _ = flatppl_infer::infer(&mut b_mod);

    let mut bundle = ModuleBundle::new();
    bundle.insert("a.flatppl", Arc::new(a_mod));
    bundle.insert("b.flatppl", Arc::new(b_mod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let result = determinize_with(&mmod, &bundle);
    assert!(
        result.is_err(),
        "a cyclic module graph (A loads B, B loads A) must refuse (terminate with Err), \
         not recurse forever; got:\n{}",
        result
            .map(|l| flatppl_flatpir::write(&l))
            .unwrap_or_default()
    );
}

/// GAP #194 — a cross-module kernel APPLICATION as the `logdensityof` target:
/// the submodule defines a reified SCALAR kernel `k = functionof(Normal(mu =
/// center, sigma = 1.0), center = center)`, the host `load_module`s it and
/// scores the APPLIED kernel `logdensityof(m.k(record(center = 0.0)), 0.5)`.
/// The target is `%call { head: User((%ref m k)), args: [record(center = 0.0)] }`
/// — a `%call` whose CALLEE is a cross-module ref. Spec §04: a kernelof
/// application and a cross-module measure both cross module boundaries, so this
/// SHOULD lower.
///
/// Before the fix the bare cross-module callee `(%ref m k)` could not be
/// resolved by `reduce_kernel_application`'s same-module `resolve_reified`
/// (`resolve_ref_one` only resolves `RefNs::SelfMod`), so the reduction returned
/// `None` and the query refused ("primitive measure must be a built-in
/// constructor"). The fix grafts the callee into the host FIRST (via the same
/// cross-module graft used for a direct target), rebuilds the `%call` with the
/// grafted LOCAL callee, and defers: the driver re-infers (typing the grafted
/// kernel) and `reduce_kernel_application` β-reduces the now-local application
/// on the next iteration, binding the applied `center = 0.0` into the law.
#[test]
fn cross_module_kernel_application_lowers() {
    let helpers = "\
flatppl_compat = \"0.1\"
center = elementof(reals)
k = functionof(Normal(mu = center, sigma = 1.0), center = center)";
    let model = "\
flatppl_compat = \"0.1\"
m = load_module(\"helpers.flatppl\")
lp = logdensityof(m.k(record(center = 0.0)), 0.5)";

    let mut hmod = parse(helpers);
    let _ = flatppl_infer::infer(&mut hmod);
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(hmod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let lowered = determinize_with(&mmod, &bundle)
        .expect("cross-module kernel APPLICATION target must lower, not refuse");
    let pir = flatppl_flatpir::write(&lowered);
    assert!(
        pir.contains("builtin_logdensityof"),
        "cross-module kernel application did not lower to builtin_logdensityof; got:\n{pir}"
    );
    // The applied `center = 0.0` β-reduces into the kernel body's `mu`, and the
    // submodule's `sigma = 1.0` survives the graft — a fully-determined law.
    assert!(
        pir.contains("(%field mu 0.0)") && pir.contains("(%field sigma 1.0)"),
        "applied `center = 0.0` did not β-reduce into the kernel body's Normal(mu = 0.0, \
         sigma = 1.0); got:\n{pir}"
    );
}

/// Applying an imported kernel also preserves its private dependency values.
#[test]
fn cross_module_kernel_application_collision_preserves_scale() {
    let helpers = "\
flatppl_compat = \"0.1\"
center = elementof(reals)
scale = 2.0
k = functionof(Normal(mu = center, sigma = scale), center = center)";
    let model = "\
flatppl_compat = \"0.1\"
scale = 10.0
m = load_module(\"helpers.flatppl\")
lp = logdensityof(m.k(record(center = 0.0)), 0.5)";

    let mut hmod = parse(helpers);
    let _ = flatppl_infer::infer(&mut hmod);
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(hmod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    assert_normal_score(mmod, &bundle, 0.5, 0.0, 2.0);
}

/// Re-exporting a measure under the same name preserves its density.
#[test]
fn cross_module_reexport_same_name_lowers() {
    let leaf = "\
flatppl_compat = \"0.1\"
d = Normal(mu = 0.0, sigma = 1.0)";
    let mid = "\
flatppl_compat = \"0.1\"
inner = load_module(\"b.flatppl\")
d = inner.d
m = d";
    let model = "\
flatppl_compat = \"0.1\"
outer = load_module(\"a.flatppl\")
lp = logdensityof(outer.m, 0.5)";

    let mut leaf_mod = parse(leaf);
    let _ = flatppl_infer::infer(&mut leaf_mod);
    let mut mid_mod = parse(mid);
    let _ = flatppl_infer::infer(&mut mid_mod);

    let mut bundle = ModuleBundle::new();
    bundle.insert("a.flatppl", Arc::new(mid_mod));
    bundle.insert("b.flatppl", Arc::new(leaf_mod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let lowered = determinize_with(&mmod, &bundle).expect(
        "a pure same-name cross-module re-export IS the target's binding (§04), so it must \
         resolve through and lower, not refuse as a name collision",
    );
    let pir = flatppl_flatpir::write(&lowered);
    assert!(
        pir.contains("builtin_logdensityof"),
        "same-name cross-module re-export did not lower to builtin_logdensityof; got:\n{pir}"
    );
    // The re-export resolves through to the leaf's concrete kernel.
    assert!(
        pir.contains("(%field mu 0.0)") && pir.contains("(%field sigma 1.0)"),
        "re-exported leaf Normal(mu = 0.0, sigma = 1.0) did not survive the resolve-through; \
         got:\n{pir}"
    );
}

/// A RENAMED pure cross-module re-export (`A.m = inner.leaf_d`, different name from
/// the target `B.leaf_d`) also resolves through and lowers. Different names never
/// collide either way, but this guards that the resolve-through binds the host
/// name (`m`) to the grafted target correctly (not to a dangling ref).
#[test]
fn cross_module_reexport_renamed_lowers() {
    let leaf = "\
flatppl_compat = \"0.1\"
leaf_d = Normal(mu = 0.0, sigma = 1.0)";
    let mid = "\
flatppl_compat = \"0.1\"
inner = load_module(\"b.flatppl\")
m = inner.leaf_d";
    let model = "\
flatppl_compat = \"0.1\"
outer = load_module(\"a.flatppl\")
lp = logdensityof(outer.m, 0.5)";

    let mut leaf_mod = parse(leaf);
    let _ = flatppl_infer::infer(&mut leaf_mod);
    let mut mid_mod = parse(mid);
    let _ = flatppl_infer::infer(&mut mid_mod);

    let mut bundle = ModuleBundle::new();
    bundle.insert("a.flatppl", Arc::new(mid_mod));
    bundle.insert("b.flatppl", Arc::new(leaf_mod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let lowered = determinize_with(&mmod, &bundle)
        .expect("a renamed pure cross-module re-export must resolve through and lower");
    let pir = flatppl_flatpir::write(&lowered);
    assert!(
        pir.contains("builtin_logdensityof")
            && pir.contains("(%field mu 0.0)")
            && pir.contains("(%field sigma 1.0)"),
        "renamed cross-module re-export did not resolve through to the leaf kernel; got:\n{pir}"
    );
}

/// A local value and a nested imported value remain distinct despite their name.
#[test]
fn two_distinct_same_name_bindings_preserve_each_value() {
    let leaf = "\
flatppl_compat = \"0.1\"
d = 99.0";
    let mid = "\
flatppl_compat = \"0.1\"
d = 2.0
inner = load_module(\"leaf.flatppl\")
combo = Normal(mu = inner.d, sigma = d)";
    let model = "\
flatppl_compat = \"0.1\"
outer = load_module(\"mid.flatppl\")
lp = logdensityof(outer.combo, 0.5)";

    let mut leaf_mod = parse(leaf);
    let _ = flatppl_infer::infer(&mut leaf_mod);
    let mut mid_mod = parse(mid);
    let _ = flatppl_infer::infer(&mut mid_mod);

    let mut bundle = ModuleBundle::new();
    bundle.insert("mid.flatppl", Arc::new(mid_mod));
    bundle.insert("leaf.flatppl", Arc::new(leaf_mod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    assert_normal_score(mmod, &bundle, 0.5, 99.0, 2.0);
}

/// GAP #194 FIX 1, refuse-don't-mislower (CRITICAL — reproduces a silent wrong
/// density): a cross-module kernel APPLICATION whose ARGUMENT is itself a
/// cross-module ref (`logdensityof(m.k(m.rec), pt)`, where `m.rec` is a record
/// defined in the submodule) must refuse rather than lower. The doc comment on
/// `graft_kernel_application_callee` claims the application's args are
/// host-local and carries them over unchanged when rebuilding the `%call` — but
/// `m.rec` is NOT host-local. Only the callee (`m.k`) is grafted; the raw
/// unresolved `(%ref m rec)` argument would be spliced as-is into the rebuilt
/// call. `reduce_kernel_application` (structural; `resolve_ref_one` follows
/// only `SelfMod` refs) cannot see through it, so it splices the dangling ref
/// into the kernel body — `determinize_with` must NOT return `Ok` with that
/// dangling, `Type::Failed("cross-module resolution")`-tagged node standing in
/// for a resolved value.
#[test]
fn cross_module_kernel_application_with_module_arg_refuses() {
    let helpers = "\
flatppl_compat = \"0.1\"
center = elementof(reals)
k = functionof(Normal(mu = center, sigma = 1.0), center = center)
rec = record(center = 0.0)";
    let model = "\
flatppl_compat = \"0.1\"
m = load_module(\"helpers.flatppl\")
lp = logdensityof(m.k(m.rec), 0.5)";

    let mut hmod = parse(helpers);
    let _ = flatppl_infer::infer(&mut hmod);
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(hmod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let result = determinize_with(&mmod, &bundle);
    assert!(
        result.is_err(),
        "a cross-module kernel APPLICATION whose ARGUMENT is itself a cross-module ref must \
         refuse (only the callee is grafted; the argument crosses the module boundary too) \
         rather than splice an unresolved dangling ref into the kernel body — a silent wrong \
         density; got:\n{}",
        result
            .map(|l| flatppl_flatpir::write(&l))
            .unwrap_or_default()
    );
}

/// Different members re-exported under one spelling retain their own values.
#[test]
fn two_reexports_with_same_name_preserve_each_value() {
    let priors = "\
flatppl_compat = \"0.1\"
x = 1.0
y = 9.0";
    let a = "\
flatppl_compat = \"0.1\"
p = load_module(\"priors.flatppl\")
foo = p.x
m = foo";
    let b = "\
flatppl_compat = \"0.1\"
p = load_module(\"priors.flatppl\")
foo = p.y
n = foo";
    let mid = "\
flatppl_compat = \"0.1\"
amod = load_module(\"a.flatppl\")
bmod = load_module(\"b.flatppl\")
combo = Normal(mu = amod.m, sigma = bmod.n)";
    let model = "\
flatppl_compat = \"0.1\"
outer = load_module(\"mid.flatppl\")
lp = logdensityof(outer.combo, 0.5)";

    let mut priors_mod = parse(priors);
    let _ = flatppl_infer::infer(&mut priors_mod);
    let mut a_mod = parse(a);
    let _ = flatppl_infer::infer(&mut a_mod);
    let mut b_mod = parse(b);
    let _ = flatppl_infer::infer(&mut b_mod);
    let mut mid_mod = parse(mid);
    let _ = flatppl_infer::infer(&mut mid_mod);

    let mut bundle = ModuleBundle::new();
    bundle.insert("priors.flatppl", Arc::new(priors_mod));
    bundle.insert("a.flatppl", Arc::new(a_mod));
    bundle.insert("b.flatppl", Arc::new(b_mod));
    bundle.insert("mid.flatppl", Arc::new(mid_mod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    assert_normal_score(mmod, &bundle, 0.5, 1.0, 9.0);
}

/// A re-export and a direct reference reach one loaded instance's value.
#[test]
fn diamond_reexport_plus_direct_lowers() {
    let leaf = "\
flatppl_compat = \"0.1\"
d = 3.0";
    let mid = "\
flatppl_compat = \"0.1\"
inner = load_module(\"b.flatppl\")
d = inner.d
combo = Normal(mu = d, sigma = inner.d)";
    let model = "\
flatppl_compat = \"0.1\"
outer = load_module(\"a.flatppl\")
lp = logdensityof(outer.combo, 0.5)";

    let mut leaf_mod = parse(leaf);
    let _ = flatppl_infer::infer(&mut leaf_mod);
    let mut mid_mod = parse(mid);
    let _ = flatppl_infer::infer(&mut mid_mod);

    let mut bundle = ModuleBundle::new();
    bundle.insert("a.flatppl", Arc::new(mid_mod));
    bundle.insert("b.flatppl", Arc::new(leaf_mod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    assert_normal_score(mmod, &bundle, 0.5, 3.0, 3.0);
}

/// A chain of re-exports preserves the terminal measure.
#[test]
fn chained_reexport_lowers() {
    let c = "\
flatppl_compat = \"0.1\"
d = Normal(mu = 0.0, sigma = 1.0)";
    let b = "\
flatppl_compat = \"0.1\"
innerC = load_module(\"c.flatppl\")
d = innerC.d";
    let a = "\
flatppl_compat = \"0.1\"
innerB = load_module(\"b.flatppl\")
d = innerB.d
m = d";
    let model = "\
flatppl_compat = \"0.1\"
outer = load_module(\"a.flatppl\")
lp = logdensityof(outer.m, 0.5)";

    let mut c_mod = parse(c);
    let _ = flatppl_infer::infer(&mut c_mod);
    let mut b_mod = parse(b);
    let _ = flatppl_infer::infer(&mut b_mod);
    let mut a_mod = parse(a);
    let _ = flatppl_infer::infer(&mut a_mod);

    let mut bundle = ModuleBundle::new();
    bundle.insert("a.flatppl", Arc::new(a_mod));
    bundle.insert("b.flatppl", Arc::new(b_mod));
    bundle.insert("c.flatppl", Arc::new(c_mod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let lowered = determinize_with(&mmod, &bundle).expect(
        "a chained re-export (a.d = b.d, b.d = c.d, c.d = Normal) is transitively the same value, \
         so it must resolve through and lower, not refuse as a name collision",
    );
    let pir = flatppl_flatpir::write(&lowered);
    assert!(
        pir.contains("builtin_logdensityof"),
        "chained re-export did not lower to builtin_logdensityof; got:\n{pir}"
    );
    // The chain resolves through to the ultimate leaf's concrete kernel.
    assert!(
        pir.contains("(%field mu 0.0)") && pir.contains("(%field sigma 1.0)"),
        "chained re-export did not resolve through to c's Normal(mu = 0.0, sigma = 1.0); \
         got:\n{pir}"
    );
}

/// A cyclic re-export chain refuses rather than recursing
/// forever. `a.d = b.d` and `b.d = a.d` form a re-export loop with no terminal
/// binding. The source/member cycle guard detects re-entry across instances.
#[test]
fn cyclic_reexport_chain_refuses() {
    let a = "\
flatppl_compat = \"0.1\"
innerB = load_module(\"b.flatppl\")
d = innerB.d
m = d";
    let b = "\
flatppl_compat = \"0.1\"
innerA = load_module(\"a.flatppl\")
d = innerA.d";
    let model = "\
flatppl_compat = \"0.1\"
outer = load_module(\"a.flatppl\")
lp = logdensityof(outer.m, 0.5)";

    let mut a_mod = parse(a);
    let _ = flatppl_infer::infer(&mut a_mod);
    let mut b_mod = parse(b);
    let _ = flatppl_infer::infer(&mut b_mod);

    let mut bundle = ModuleBundle::new();
    bundle.insert("a.flatppl", Arc::new(a_mod));
    bundle.insert("b.flatppl", Arc::new(b_mod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let result = determinize_with(&mmod, &bundle);
    assert!(
        result.is_err(),
        "a cyclic re-export chain (a.d = b.d, b.d = a.d) must refuse (terminate with Err), \
         not recurse forever; got:\n{}",
        result
            .map(|l| flatppl_flatpir::write(&l))
            .unwrap_or_default()
    );
}

// ===========================================================================
// Buffy #359 — cross-module QUERY MODULE resolution (the pre-pass).
//
// A query module `load_module`s a model and scores its `prior`/`posterior` at a
// cross-module VALUE (`model.default_pars`) via top-level ALIAS bindings
// (`prior = model.prior`, `x = model.default_pars`). The determiniser pre-pass
// resolves every such alias IN PLACE before the measure-reduction loop, so the
// host is self-contained and both `flatppl determinize` (no roots) and the
// with-roots path lower it. These are the four gaps closed:
//   1. the VARIATE (arg2, `x`) is grafted, not just the measure — the variate
//      destructuring sees a local `record` instead of `joint value must be a
//      record`;
//   2. the alias binding is rewritten in place, not orphaned — no dangling
//      module ref survives under no roots;
//   3. a dependency shared across two queries (`theta1_dist`, and `prior`
//      reached from both `prior` and `posterior`) is grafted ONCE (shared dedup
//      registry), not re-grafted into a collision;
//   4. an alias whose name equals its submodule member (`prior = model.prior`)
//      is that member's local slot — no self-collision.
// ===========================================================================

/// The scored model, mirroring `flatppl-examples/bayesian_inference_1.flatppl`:
/// two independent priors assembled with `joint`, an IID-Normal forward model,
/// a likelihood/posterior, and a `default_pars` record used as the query point.
const BI1_SUB: &str = "flatppl_compat = \"0.1\"
theta1_dist = Normal(0, 1)
theta2_dist = Exponential(1)
prior = joint(theta1 = theta1_dist, theta2 = theta2_dist)
theta1 = elementof(reals)
theta2 = elementof(reals)
c = 5
f_a = par -> c * par
f_b = fn(abs(_) * _)
a = f_a(theta2)
b = f_b(theta1, theta2)
obs ~ iid(Normal(mu = a, sigma = b), 10)
forward_kernel = kernelof(record(obs = obs))
observed_data = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]
L = likelihoodof(forward_kernel, record(obs = observed_data))
posterior = bayesupdate(L, prior)
default_pars = record(theta1 = 0.5, theta2 = 1.0)";

/// Build the `m.flatppl`-keyed bundle for [`BI1_SUB`] and the inferred host.
fn bi1_host(host_src: &str) -> (flatppl_core::Module, ModuleBundle) {
    let mut sub = parse(BI1_SUB);
    let _ = flatppl_infer::infer(&mut sub);
    let mut bundle = ModuleBundle::new();
    bundle.insert("m.flatppl", Arc::new(sub));
    let mut host = parse(host_src);
    let _ = flatppl_infer::infer_module(&mut host, &bundle, flatppl_infer::Level::Shape);
    (host, bundle)
}

/// Gaps 1, 2, 4 + numeric: a query module that scores the loaded model's `prior`
/// at its `default_pars` via same-name alias bindings determinizes with NO roots
/// (plain `flatppl determinize`) — the variate is grafted (no "joint value must
/// be a record" refuse) and the alias is rewritten in place (no orphaned module
/// ref survives the conformance check).
///
/// NUMERIC: the lowered `l1` is
///   add(builtin_logdensityof(Normal, {mu=0,sigma=1}, 0.5),
///       builtin_logdensityof(Exponential, {rate=1}, 1.0))
/// which an INDEPENDENT scipy/closed-form oracle evaluates to
///   Normal.logpdf(0.5;0,1) + Exponential.logpdf(1.0;rate=1) = -2.0439385…
/// (the determiniser is symbolic-only; the oracle value pins the emitted terms).
#[test]
fn crossmodule_query_module_alias_lowers_no_roots() {
    let host = "flatppl_compat = \"0.1\"
model = load_module(\"m.flatppl\")
prior = model.prior
x = model.default_pars
l1 = logdensityof(prior, x)";
    let (h, bundle) = bi1_host(host);

    let lowered = determinize_with(&h, &bundle).expect(
        "a cross-module query module (prior = model.prior, x = model.default_pars) must \
         determinize with no roots",
    );
    let pir = flatppl_flatpir::write(&lowered);

    // Gap 1: the VARIATE `x` was grafted to a local record (not a dangling module
    // ref), so the variate destructuring saw a record.
    assert!(
        pir.contains("(%field theta1 0.5)") && pir.contains("(%field theta2 1.0)"),
        "the cross-module variate `x = model.default_pars` was not grafted to a local \
         record; got:\n{pir}"
    );
    // Numeric structure: the two prior terms scored at the default point (oracle
    // -2.0439385…). `mu=0,sigma=1` scored at 0.5; `rate=1` scored at 1.0.
    assert!(
        pir.contains("(builtin_logdensityof Normal")
            && pir.contains("(%field mu 0) (%field sigma 1))) 0.5)"),
        "l1's Normal prior term did not lower to builtin_logdensityof at 0.5; got:\n{pir}"
    );
    assert!(
        pir.contains("(builtin_logdensityof Exponential") && pir.contains("(%field rate 1))) 1.0)"),
        "l1's Exponential prior term did not lower to builtin_logdensityof at 1.0; got:\n{pir}"
    );
    // Gap 2 (no orphan): no residual cross-module ref — success already implies
    // conformance, but assert the alias RHS is a local `joint`/record, not a
    // module ref, by checking the module carries no `(%ref <alias> …)` form.
    // (A surviving module ref renders with the loaded-module alias name.)
    assert!(
        !pir.contains("(%ref model "),
        "an orphaned cross-module ref into `model` survived the pre-pass; got:\n{pir}"
    );
}

/// Gap 3 (the crux): TWO queries sharing dependencies. `prior` and `posterior`
/// are both cross-module aliases into the SAME loaded model; `posterior`'s body
/// (`bayesupdate(L, prior)`) references the submodule `prior`, and both aliases
/// transitively reach `theta1_dist`/`theta2_dist`. The shared per-pass dedup
/// registry grafts each shared dependency ONCE and dedups the second reach — the
/// pre-fix per-call registry re-grafted and collided (an "unrelated host binding"
/// refuse). Determinizes with no roots, and the shared `theta1_dist` is bound
/// exactly once.
#[test]
fn crossmodule_multi_query_shared_dep_dedups() {
    let host = "flatppl_compat = \"0.1\"
model = load_module(\"m.flatppl\")
prior = model.prior
posterior = model.posterior
x = model.default_pars
l1 = logdensityof(prior, x)
l2 = logdensityof(posterior, x)";
    let (h, bundle) = bi1_host(host);

    let lowered = determinize_with(&h, &bundle).expect(
        "two cross-module queries sharing dependencies (prior + posterior over the same \
         loaded model) must determinize — the shared dep is grafted once, not collided",
    );
    let pir = flatppl_flatpir::write(&lowered);

    // Both queries lowered.
    assert!(
        pir.contains("(%bind l1 ") && pir.contains("(%bind l2 "),
        "both l1 and l2 must lower; got:\n{pir}"
    );
    assert!(
        pir.contains("(builtin_logdensityof Normal")
            && pir.contains("(builtin_logdensityof Exponential"),
        "the shared prior terms must lower to builtin_logdensityof; got:\n{pir}"
    );
    // Shared-dep dedup: `theta1_dist`, reached via both `prior` and `posterior`,
    // is grafted into the host EXACTLY ONCE (a re-graft would have collided/refused).
    assert_eq!(
        pir.matches("(%bind theta1_dist ").count(),
        1,
        "the shared submodule dependency `theta1_dist` must be grafted exactly once \
         (deduped across the two queries); got:\n{pir}"
    );
}

/// The same query module determinizes WITH roots too (root-based DCE): keeping
/// only `l1`, the `load_module` handle and all unreferenced grafted scaffolding
/// are dropped, leaving the self-contained scored density.
#[test]
fn crossmodule_query_module_alias_lowers_with_roots() {
    let host = "flatppl_compat = \"0.1\"
model = load_module(\"m.flatppl\")
prior = model.prior
posterior = model.posterior
x = model.default_pars
l1 = logdensityof(prior, x)
l2 = logdensityof(posterior, x)";
    let (mut h, bundle) = bi1_host(host);
    let l1 = h.intern("l1");

    let lowered = determinize_with_roots(&h, &bundle, Some(&[l1]))
        .expect("the cross-module query module must determinize with roots (keep l1)");
    let pir = flatppl_flatpir::write(&lowered);

    // DCE dropped the load_module handle and every binding unreachable from l1.
    assert!(
        !pir.contains("load_module"),
        "root-based DCE must drop the dead `model = load_module(…)` handle; got:\n{pir}"
    );
    assert!(
        !pir.contains("(%bind l2 ") && !pir.contains("(%bind posterior "),
        "only l1 and its dependencies should survive; got:\n{pir}"
    );
    // l1 still carries the two scored prior terms (oracle -2.0439385…).
    assert!(
        pir.contains("(builtin_logdensityof Normal")
            && pir.contains("(%field mu 0) (%field sigma 1))) 0.5)")
            && pir.contains("(builtin_logdensityof Exponential")
            && pir.contains("(%field rate 1))) 1.0)"),
        "l1 must retain both builtin_logdensityof prior terms under roots; got:\n{pir}"
    );
}

// Regression (Buffy #379): a `~` whose primitive measure is a cross-module
// distribution ref — `x ~ h.d` (`x = draw((%ref h d))`) — must graft the
// submodule distribution and determinize, not refuse. The module ref is NESTED
// inside `draw(...)`, so the top-level alias-binding rewrite doesn't reach it;
// `resolve_crossmodule_aliases`'s phase 5 (draw-measure-scoped) grafts it. Only
// MEASURE-position refs graft this way — a cross-module VALUE argument still
// refuses (see `cross_module_kernel_application_with_module_arg_refuses`).
#[test]
fn cross_module_draw_measure_ref_grafts_and_determinizes() {
    let helpers = "flatppl_compat = \"0.1\"\nd = Normal(mu = 0.0, sigma = 1.0)";
    let model = "flatppl_compat = \"0.1\"\n\
h = load_module(\"helpers.flatppl\")\n\
x ~ h.d\n\
lp = logdensityof(lawof(record(x = x)), record(x = 0.5))";

    let mut hmod = parse(helpers);
    let _ = flatppl_infer::infer(&mut hmod);
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(hmod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let lowered = determinize_with(&mmod, &bundle)
        .expect("cross-module draw-measure ref must graft + determinize, not refuse");
    let ir = flatppl_flatpir::write(&lowered);
    assert!(
        ir.contains("builtin_logdensityof") && ir.contains("Normal"),
        "expected a grafted Normal density term:\n{ir}"
    );
}

// Regression (Buffy #379, function-callee): a REGULAR binding whose RHS applies
// a cross-module FUNCTION to a HOST value — `shifted = h.g(mu0)` — must graft
// the callee `h.g` and determinize (the applied function feeds a distribution
// parameter). Previously the module-ref callee nested in the application (not a
// top-level alias, not a draw measure) was left dangling → a
// `Type::Failed("cross-module resolution")` node. The driver pre-pass now grafts
// such callees via `graft_kernel_application_callee` — which still REFUSES a
// cross-module value ARGUMENT (`cross_module_kernel_application_with_module_arg_refuses`).
#[test]
fn cross_module_function_callee_in_binding_grafts() {
    let helpers = "flatppl_compat = \"0.1\"\ng = par -> par + 1.0";
    let model = "flatppl_compat = \"0.1\"\n\
h = load_module(\"helpers.flatppl\")\n\
mu0 = elementof(reals)\n\
shifted = h.g(mu0)\n\
x ~ Normal(mu = shifted, sigma = 1.0)\n\
lp = logdensityof(lawof(record(x = x)), record(x = 0.5))";

    let mut hmod = parse(helpers);
    let _ = flatppl_infer::infer(&mut hmod);
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(hmod));

    let mut mmod = parse(model);
    let _ = flatppl_infer::infer_module(&mut mmod, &bundle, flatppl_infer::Level::Shape);

    let lowered = determinize_with(&mmod, &bundle)
        .expect("a cross-module function callee applied to a host value must graft, not refuse");
    let ir = flatppl_flatpir::write(&lowered);
    assert!(
        ir.contains("builtin_logdensityof") && ir.contains("Normal"),
        "expected a Normal density term with the grafted-function-derived mean:\n{ir}"
    );
}

#[test]
fn nested_loads_keep_their_parent_instance() {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "leaf.flatppl",
        Arc::new(parse("k = external(reals)\nval = k + 1.0")),
    );
    bundle.insert("wrapper.flatppl", Arc::new(parse(
        "k = external(reals)\nchild = load_module(\"leaf.flatppl\", k = k)\nf(x) = x + child.val",
    )));
    assert_outputs(
        "a = load_module(\"wrapper.flatppl\", k = 1.0)\n\
         b = load_module(\"wrapper.flatppl\", k = 10.0)\n\
         x = elementof(reals)\ninputs = x\noutputs = [a.f(x), b.f(x)]",
        &bundle,
        "x = elementof(reals)\ninputs = x\noutputs = [x + 2.0, x + 11.0]",
    );
}

#[test]
fn imported_values_inside_expressions_keep_load_substitutions() {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "values.flatppl",
        Arc::new(parse(
            "offset = 3.0\nx = external(reals)\nvalue = x + offset",
        )),
    );
    assert_outputs(
        "p = elementof(cartprod(z = reals, a = cartpow(reals, 2)))\ninputs = p\n\
         left = load_module(\"values.flatppl\", x = 1.0)\n\
         right = load_module(\"values.flatppl\", x = 10.0)\n\
         outputs = record(total = p.z + sum(p.a) + left.offset,\n\
                          values = [left.value + 1.0, right.value + 1.0])",
        &bundle,
        "p = elementof(cartprod(z = reals, a = cartpow(reals, 2)))\ninputs = p\n\
         outputs = record(total = p.z + sum(p.a) + 3.0,\n\
                          values = [5.0, 14.0])",
    );
}

#[test]
fn reexported_laws_preserve_shared_stochastic_ancestors() {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "leaf.flatppl",
        Arc::new(parse(
            "z ~ Normal(0.0, 1.0)\na ~ Normal(z, 1.0)\nb ~ Normal(z, 1.0)\n\
             left = lawof(a)\nright = lawof(b)",
        )),
    );
    bundle.insert(
        "wrapper.flatppl",
        Arc::new(parse(
            "child = load_module(\"leaf.flatppl\")\nleft = child.left\nright = child.right",
        )),
    );
    assert_outputs(
        "w = load_module(\"wrapper.flatppl\")\nleft = w.left\nright = w.right\n\
         outputs = logdensityof(joint(a = left, b = right), record(a = 0.0, b = 1.0))",
        &bundle,
        "z ~ Normal(0.0, 1.0)\na ~ Normal(z, 1.0)\nb ~ Normal(z, 1.0)\n\
         outputs = logdensityof(lawof(record(a = a, b = b)), record(a = 0.0, b = 1.0))",
    );
}

#[test]
fn reexported_kernel_applies_through_aliases() {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "leaf.flatppl",
        Arc::new(parse(
            "center = elementof(reals)\n\
             k = functionof(Normal(center, 1.0), center = center)\nalias = k",
        )),
    );
    bundle.insert(
        "wrapper.flatppl",
        Arc::new(parse(
            "inner = load_module(\"leaf.flatppl\")\nk = inner.alias",
        )),
    );
    assert_outputs(
        "w = load_module(\"wrapper.flatppl\")\n\
         outputs = logdensityof(w.k(record(center = 0.0)), 0.5)",
        &bundle,
        "outputs = logdensityof(Normal(0.0, 1.0), 0.5)",
    );
}

#[test]
fn reexported_measure_preserves_sample_and_rng_state() {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "helper.flatppl",
        Arc::new(parse("d = Normal(0.0, 2.0)\nalias = d")),
    );
    assert_outputs(
        "h = load_module(\"helper.flatppl\")\nm = h.alias\n\
         s = rnginit(0)\nv, next = rand(s, m)\noutputs = (v, next)",
        &bundle,
        "s = rnginit(0)\nv, next = rand(s, Normal(0.0, 2.0))\noutputs = (v, next)",
    );
}

#[test]
fn sibling_load_assignments_are_not_import_cycles() {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "leaf.flatppl",
        Arc::new(parse("k = external(reals)\nval = k + 1.0")),
    );
    let chain = "a = load_module(\"leaf.flatppl\", k = 1.0)\n\
                 b = load_module(\"leaf.flatppl\", k = a.val)\nval = b.val";
    bundle.insert("wrapper.flatppl", Arc::new(parse(chain)));
    assert_outputs(
        &format!("{chain}\nw = load_module(\"wrapper.flatppl\")\nv = w.val\noutputs = [val, v]"),
        &bundle,
        "outputs = [3.0, 3.0]",
    );
}

#[test]
fn imported_automatic_inputs_keep_labels_through_simplification() {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "helper.flatppl",
        Arc::new(parse(
            "z = elementof(reals)\nc = 2.0\nf = functionof(z + c)\n\
         second(a, b) = b\ng = functionof(second(z, 1.0))",
        )),
    );
    assert_outputs(
        "h = load_module(\"helper.flatppl\")\nz = elementof(reals)\ninputs = z\n\
         outputs = [h.f(z = 2.0), h.g(z = 5.0), z]",
        &bundle,
        "z = elementof(reals)\ninputs = z\noutputs = [4.0, 1.0, z]",
    );
}

#[test]
fn imported_boundary_cuts_assigned_expression() {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "helper.flatppl",
        Arc::new(parse(
            "p = elementof(reals)\nf = functionof(p + 1.0, p = p)",
        )),
    );
    assert_outputs(
        "raw = elementof(reals)\ninputs = raw\n\
         h = load_module(\"helper.flatppl\", p = raw + 9.0)\noutputs = h.f(p = 2.0)",
        &bundle,
        "raw = elementof(reals)\ninputs = raw\noutputs = 3.0",
    );
}

#[test]
fn import_names_do_not_overwrite_host_names() {
    let mut bundle = ModuleBundle::new();
    bundle.insert("helper.flatppl", Arc::new(parse("c = 3.0\nf(x) = x + c")));
    assert_outputs(
        "h = load_module(\"helper.flatppl\")\nc = 9.0\n__import_c_0 = 50.0\n\
         x = elementof(reals)\ninputs = x\noutputs = [h.f(x), c, __import_c_0]",
        &bundle,
        "x = elementof(reals)\ninputs = x\noutputs = [x + 3.0, 9.0, 50.0]",
    );
}

#[test]
fn imported_likelihood_keeps_public_input_labels() {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "helper.flatppl",
        Arc::new(parse(
            "center = elementof(reals)\n\
         k = functionof(Normal(center, 1.0), center = center)",
        )),
    );
    let mut model = parse(
        "center = 99.0\nh = load_module(\"helper.flatppl\")\n\
         L = likelihoodof(h.k, 0.5)\nlp = logdensityof(L, record(center = 0.0))",
    );
    let _ = flatppl_infer::infer_module(&mut model, &bundle, flatppl_infer::Level::Shape);
    assert_normal_score(model, &bundle, 0.5, 0.0, 1.0);
}

#[test]
fn imported_likelihood_accepts_renamed_parameter_alias() {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "helper.flatppl",
        Arc::new(parse(
            "p = elementof(reals)\nk = functionof(Normal(p, 1.0))",
        )),
    );
    let mut model = parse(
        "h = load_module(\"helper.flatppl\")\nq = h.p\n\
         L = likelihoodof(h.k, 0.5)\nlp = logdensityof(L, record(p = 0.0))",
    );
    let _ = flatppl_infer::infer_module(&mut model, &bundle, flatppl_infer::Level::Shape);
    assert_normal_score(model, &bundle, 0.5, 0.0, 1.0);
}

#[test]
fn repeated_imported_queries_restore_discarded_measure() {
    let mut bundle = ModuleBundle::new();
    bundle.insert("helper.flatppl", Arc::new(parse("d = Normal(0.0, 2.0)")));
    assert_outputs(
        "h = load_module(\"helper.flatppl\")\na = logdensityof(h.d, 0.0)\n\
         b = logdensityof(h.d, 1.0)\noutputs = [a, b]",
        &bundle,
        "a = logdensityof(Normal(0.0, 2.0), 0.0)\n\
         b = logdensityof(Normal(0.0, 2.0), 1.0)\noutputs = [a, b]",
    );
}

#[test]
fn configured_imports_keep_separate_dependencies() {
    let mut bundle = ModuleBundle::new();
    bundle.insert(
        "helpers.flatppl",
        Arc::new(parse("k = external(reals)\nc = k + 1.0\nf(x) = x + c")),
    );
    assert_outputs(
        "a = load_module(\"helpers.flatppl\", k = 1.0)\n\
         b = load_module(\"helpers.flatppl\", k = 10.0)\n\
         fa = a.f\nfb = b.f\nx = elementof(reals)\ninputs = x\n\
         outputs = [fa(x), fb(x)]",
        &bundle,
        "x = elementof(reals)\ninputs = x\noutputs = [x + 2.0, x + 11.0]",
    );
}

#[test]
fn repeated_imported_calls_reuse_dependencies_across_phases() {
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(parse("c = 3.0\nf(x) = x + c")));
    assert_outputs(
        "h = load_module(\"helpers.flatppl\")\nx = elementof(reals)\ninputs = x\n\
         a = h.f(x)\nb = h.f(x + 1.0)\noutputs = [a, b, h.f(x + 2.0)]",
        &bundle,
        "x = elementof(reals)\ninputs = x\na = x + 3.0\n\
         b = (x + 1.0) + 3.0\noutputs = [a, b, (x + 2.0) + 3.0]",
    );
}

#[test]
fn nested_imported_calls_preserve_arguments_and_result_shape() {
    let helpers = parse("step = external(reals)\nf(x) = x + step");
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(helpers));
    let mut model = parse(
        "raw = external(reals)\nstep = raw + 1\n\
         h = load_module(\"helpers.flatppl\", step = step)\n\
         x = elementof(reals)\ninputs = (x, raw)\n\
         outputs = record(values = [h.f(x), h.f(h.f(x))])",
    );
    let _ = flatppl_infer::infer_module(&mut model, &bundle, flatppl_infer::Level::Shape);
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let actual = determinize_with_roots(&model, &bundle, Some(&roots)).unwrap();
    let mut expected = parse(
        "raw = external(reals)\nstep = raw + 1\n\
         x = elementof(reals)\ninputs = (x, raw)\n\
         outputs = record(values = [x + step, (x + step) + step])",
    );
    let roots = [expected.intern("inputs"), expected.intern("outputs")];
    let expected = determinize_with_roots(&expected, &ModuleBundle::new(), Some(&roots)).unwrap();
    assert_eq!(
        flatppl_flatpir::write(&actual),
        flatppl_flatpir::write(&expected)
    );
}

#[test]
fn unused_nested_imported_calls_do_not_block_outputs() {
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(parse("c = 1\nf(x) = x + c")));
    let mut model = parse(
        "h = load_module(\"helpers.flatppl\")\nc = 9\n\
         x = elementof(reals)\ninputs = x\nfirst(a, b) = a\n\
         unused = [h.f(x)]\noutputs = first(x, [h.f(x)])",
    );
    let _ = flatppl_infer::infer_module(&mut model, &bundle, flatppl_infer::Level::Shape);
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let actual = determinize_with_roots(&model, &bundle, Some(&roots)).unwrap();
    let mut expected = parse("x = elementof(reals)\ninputs = x\noutputs = x");
    let roots = [expected.intern("inputs"), expected.intern("outputs")];
    let expected = determinize_with_roots(&expected, &ModuleBundle::new(), Some(&roots)).unwrap();
    assert_eq!(
        flatppl_flatpir::write(&actual),
        flatppl_flatpir::write(&expected)
    );
}

/// Lower local members before graft checks and imported members after grafts.
#[test]
fn standard_functions_lower_before_and_after_module_grafts() {
    let helpers = r#"
flatppl_compat = "0.1"
hep = standard_module("particle-physics", "0.1")
x = 1.0
a = elementof(reals)
g(p) = hep.interp_poly6_exp(0.8, 1.0, 1.2, p)
k = functionof(Normal(mu = hep.interp_poly6_exp(0.8, 1.0, 1.2, a), sigma = 1.0))
L = likelihoodof(k, 2.0)
"#;
    let mut hmod = parse(helpers);
    let _ = flatppl_infer::infer(&mut hmod);
    let mut bundle = ModuleBundle::new();
    bundle.insert("helpers.flatppl", Arc::new(hmod));

    for query in [
        "lp = logdensityof(h.L, record(a = theta))",
        "mu = h.g(theta)\nlp = logdensityof(Normal(mu, 1.0), 2.0)",
        "hep = standard_module(\"particle-physics\", \"0.1\")\n\
         unused = hep.kallen(h.x, 1.0, 1.0)\n\
         lp = logdensityof(Normal(0.0, 1.0), 0.0)",
    ] {
        let source = format!(
            "flatppl_compat = \"0.1\"\nh = load_module(\"helpers.flatppl\")\n\
             theta = elementof(reals)\n{query}"
        );
        let mut model = parse(&source);
        let _ = flatppl_infer::infer_module(&mut model, &bundle, flatppl_infer::Level::Shape);
        let roots = [model.intern("lp")];
        let lowered = flatppl_determinizer::determinize_with_roots(&model, &bundle, Some(&roots))
            .expect("imported standard functions must lower after their graft");
        assert!(flatppl_determinizer::is_flatpdl(&lowered).is_ok());
    }
}
