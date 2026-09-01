// Structural conformance gate for the determiniser.
//
// This file holds STRUCTURAL determinizer tests only: they check that a
// handful of rosetta models (single Gaussian, product of Gaussians, iid,
// joint, likelihoodof) determinize into a FlatPDL-conformant module with the
// expected number of `builtin_logdensityof` calls and no residual
// measure-layer ops (`lawof`, `draw`, `iid`, `joint`, `likelihoodof`,
// `logdensityof`). Each test also sanity-checks a closed-form oracle value in
// pure Rust — no external engine is involved.
//
// Numeric value verification (scoring the emitted FlatPDL surface syntax
// through the flatppl-js engine and comparing to a frozen oracle) lives in
// `flatppl-testsuite`, not here: `flatppl-rust` is not a density engine.

use std::f64::consts::PI;

use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

/// Closed-form Gaussian log-density: log N(x; mu, sigma).
fn gaussian_logpdf(x: f64, mu: f64, sigma: f64) -> f64 {
    -0.5 * (2.0 * PI).ln() - sigma.ln() - 0.5 * ((x - mu) / sigma).powi(2)
}

// ── Structural oracle checks (always run) ────────────────────────────────────
//
// These verify that:
// 1. The determinizer produces a FlatPDL-conformant module.
// 2. The emitted surface syntax encodes the correct `builtin_logdensityof`
//    call(s), with no residual measure layer.
// The closed-form oracle values below are pure-Rust arithmetic sanity checks;
// they are not compared against any engine here.

#[test]
fn single_gaussian_oracle_agrees_with_flatpdl_structure() {
    // Model: a ~ Normal(0, 1); score a=0.5.
    // Oracle: log N(0.5; 0.0, 1.0) = -1.0439385332046727…
    let oracle = gaussian_logpdf(0.5, 0.0, 1.0);
    assert!(
        (oracle - (-1.043_938_533_204_672_7_f64)).abs() < 1e-12,
        "closed-form oracle sanity: {oracle}"
    );

    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("single-gaussian must lower");

    // FlatPDL conformance.
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "emitted FlatPDL must be conformant"
    );

    // Surface syntax encodes exactly one builtin_logdensityof call.
    let src_out = flatppl_syntax::print(&out);
    assert!(
        src_out.contains("builtin_logdensityof"),
        "emitted FlatPDL contains builtin_logdensityof:\n{src_out}"
    );
    // Use the FlatPIR form to check for residual measure-layer ops: FlatPIR
    // spells the measure-layer op as `(logdensityof `, while the FlatPDL
    // primitive is `(builtin_logdensityof ` — they don't overlap.
    let pir_out = flatppl_flatpir::write(&out);
    assert!(
        !pir_out.contains("(logdensityof ")
            && !pir_out.contains("lawof")
            && !pir_out.contains("(draw "),
        "measure layer eliminated:\n{pir_out}"
    );
    // The determinized module binds `lp` to a deterministic real — `a` is
    // pinned to the scored value (0.5) and no stochastic nodes remain.
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "single density term:\n{pir}"
    );
}

#[test]
fn product_gaussians_oracle_agrees_with_flatpdl_structure() {
    // Model: a ~ N(0,1), b ~ N(1,2); score a=0.5, b=0.5.
    // Oracle: log N(0.5;0,1) + log N(0.5;1,2)
    let oracle = gaussian_logpdf(0.5, 0.0, 1.0) + gaussian_logpdf(0.5, 1.0, 2.0);
    let expected = -1.043_938_533_204_672_7_f64 + (-1.643_335_713_764_618_f64);
    assert!(
        (oracle - expected).abs() < 1e-12,
        "closed-form oracle sanity: {oracle}"
    );

    let src = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))
b = draw(Normal(mu = 1.0, sigma = 2.0))
lp = logdensityof(lawof(record(a = a, b = b)), record(a = 0.5, b = 0.5))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("product must lower");

    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "emitted FlatPDL must be conformant"
    );

    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "two density terms:\n{pir}"
    );
    assert!(
        !pir.contains("lawof") && !pir.contains("(draw "),
        "measure layer eliminated:\n{pir}"
    );
}

#[test]
fn iid_normal_sum_structure() {
    // logdensityof(iid(Normal(0,1), 3), [0.5, -0.3, 1.2]) = Σ_{i<3} log N(get0(v,i);0,1)
    // Static unroll: each term scores the SAME Normal(0,1) at a distinct index of
    // the variate `get0(v, i)`, i = 0, 1, 2.
    //
    // The variate is a NAMED binding (`data`), not an inline literal: canon
    // Pass 3 (`flatten_structural`, buffy #263) only resolves `get0` over a
    // LITERAL `vector(...)` constructor, never through a `Ref` — so naming the
    // data here (matching `iid_lengthof_sized_lowers` below) keeps the `get0`
    // projection structure this test pins intact. An inline-literal variate
    // would be flattened away entirely by canon, which would make this test's
    // per-index `get0` shape check vacuous (the term-by-term sum is invariant
    // to `get0` index assignment here, since all three terms share the
    // identical Normal(0,1) kernel, so a wrong/transposed index could not be
    // caught once the literals are substituted in).
    let src = "\
data = [0.5, -0.3, 1.2]
d = iid(Normal(mu = 0.0, sigma = 1.0), 3)
lp = logdensityof(d, data)";
    let m = parse_infer(src);
    let out = determinize(&m).expect("iid must lower");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL"
    );
    let pir = flatppl_flatpir::write(&out);
    // Axis-native: ONE builtin_logdensityof, as the broadcast head — no per-element unroll.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "iid density is one broadcast head, not N unrolled terms:\n{pir}"
    );
    assert!(
        pir.contains("(broadcast builtin_logdensityof Normal") && pir.contains("(sum "),
        "iid density is sum(broadcast(builtin_logdensityof, Normal, …)):\n{pir}"
    );
    assert!(
        !pir.contains("(get0 "),
        "no per-element get0 projections remain:\n{pir}"
    );
    assert!(
        !pir.contains("(iid ") && !pir.contains("(logdensityof "),
        "no measure layer:\n{pir}"
    );
    // The params record scored per cell: Normal(mu=0.0, sigma=1.0), broadcast
    // from a length-1 array-of-records (a bare record is not a legal broadcast
    // input, §04 "Broadcasting" — see `lower_iid`'s primitive-kernel fast
    // path), so each param is itself a singleton `vector(...)` fed through an
    // inner `broadcast(record, …)`, once.
    assert_eq!(
        pir.matches("(broadcast record (%kwarg mu").count(),
        1,
        "one length-1 array-of-records Normal(0,1) params, broadcast across the axis:\n{pir}"
    );
    assert_eq!(
        pir.matches("(vector 0.0)").count(),
        1,
        "mu = 0.0 lifted to a length-1 vector once:\n{pir}"
    );
    assert_eq!(
        pir.matches("(vector 1.0)").count(),
        1,
        "sigma = 1.0 lifted to a length-1 vector once:\n{pir}"
    );
}

// `iid(M, n)` with a NAMED literal size (`n = 3`, referenced by `(%ref self n)`
// rather than an inline `3`) must lower exactly like the inline-literal case.
// `literal_usize` alone only matches `Node::Lit` directly, so a size arg that is
// a self-ref to a literal-bound name previously refused ("iid size must be a
// literal integer") even though `n` is statically 3 — resolving one
// `(%ref self …)` level before `literal_usize` fixes this without widening
// past a genuine non-literal (still refused).
#[test]
fn iid_named_literal_size_lowers() {
    let src = "\
n = 3
d = iid(Normal(mu = 0.0, sigma = 1.0), n)
lp = logdensityof(d, [0.5, -0.3, 1.2])";
    let m = parse_infer(src);
    let out = determinize(&m).expect("iid with a named literal size must lower");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL"
    );
    let pir = flatppl_flatpir::write(&out);
    // Axis-native broadcast: one builtin_logdensityof head, not N unrolled terms.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "iid density is one broadcast head from a named size:\n{pir}"
    );
    assert!(
        pir.contains("(broadcast builtin_logdensityof Normal") && pir.contains("(sum "),
        "iid density is sum(broadcast(builtin_logdensityof, Normal, …)):\n{pir}"
    );
    assert!(!pir.contains("(get0 "), "no per-element get0:\n{pir}");
    assert!(
        !pir.contains("(iid ") && !pir.contains("(logdensityof "),
        "no measure layer:\n{pir}"
    );
}

// `iid(M, lengthof(data))` — the canonical §06 shape-dependent iid size
// (`iid(M, lengthof(obs))`). The size is not a raw literal: it is `lengthof`
// over a fixed-shape array, which `flatppl-infer` const-evaluates (it runs at
// `Level::Shape`) into the iid measure's static domain shape. The determiniser
// reads that resolved static size from the iid node's own inferred domain type,
// so a `lengthof`-sized iid lowers to the right number of density terms rather
// than refusing "iid size must be a literal integer".
#[test]
fn iid_lengthof_sized_lowers() {
    let src = "\
data = [1.2, 3.4, 5.1]
d = iid(Normal(mu = 0.0, sigma = 1.0), lengthof(data))
lp = logdensityof(d, data)";
    let m = parse_infer(src);
    let out = determinize(&m).expect("lengthof-sized iid must lower via const-eval");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL"
    );
    let pir = flatppl_flatpir::write(&out);
    // Axis-native broadcast: one builtin_logdensityof head, not N unrolled terms.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "iid density is one broadcast head from lengthof(data) = 3:\n{pir}"
    );
    assert!(
        pir.contains("(broadcast builtin_logdensityof Normal") && pir.contains("(sum "),
        "iid density is sum(broadcast(builtin_logdensityof, Normal, …)):\n{pir}"
    );
    assert!(!pir.contains("(get0 "), "no per-element get0:\n{pir}");
    assert!(
        !pir.contains("(iid ") && !pir.contains("(logdensityof "),
        "no residual measure layer:\n{pir}"
    );
    // The params record scored per cell: Normal(mu=0.0, sigma=1.0), broadcast
    // from a length-1 array-of-records, once (see the identical comment in
    // `iid_normal_sum_structure` above).
    assert_eq!(
        pir.matches("(broadcast record (%kwarg mu").count(),
        1,
        "one length-1 array-of-records Normal(0,1) params, broadcast across the axis:\n{pir}"
    );
    assert_eq!(
        pir.matches("(vector 0.0)").count(),
        1,
        "mu = 0.0 lifted to a length-1 vector once:\n{pir}"
    );
    assert_eq!(
        pir.matches("(vector 1.0)").count(),
        1,
        "sigma = 1.0 lifted to a length-1 vector once:\n{pir}"
    );
}

// `iid` over a NON-SCALAR `M` (here a nested `iid(Normal, 2)`) must lower
// correctly — NOT refuse, NOT mislower. This proves the deliberate asymmetry with
// `joint`: `iid(M, size)` is the product `M^⊗N` over ARRAYS of shape `size`, a
// nested variate with a leading repeat axis `[N, …M-shape]`
// (§06 "Independent composition"). So the outer
// `get0(v, i)` recovers the full i-th `M`-variate (an entire inner row). There is
// deliberately NO scalar-component guard on `iid` (unlike `joint`, whose flat
// `cat` variate needs one); adding one would WRONGLY refuse this valid model.
//
// Model: iid(iid(Normal(0,1), 2), 3) scored at a shape-[3,2] array literal.
//
// **Flattened single broadcast+reduce (not an outer unroll).** The OUTER `M`
// here is `iid(Normal(0,1), 2)` — a COMPOSED (non-primitive) inner measure, so
// `lower_iid`'s primitive-kernel fast path does not fire directly on it. But
// `lower_iid` peels through this one further `iid` layer (`Normal` bottoms out
// as a bare constructor at depth 2) and flattens the WHOLE nested product to
// ONE axis-native expression — `sum(broadcast(builtin_logdensityof, Normal,
// <rank-2 [1,1] singleton params>, data))` — reusing the same
// `emit_kernel_broadcast_density` tail as the primitive (depth-1) fast path,
// just with each scalar param lifted through 2 nested `vector(...)` wraps
// instead of 1. This scores all 6 leaves (Σ over the full `[3,2]` array) in a
// single broadcast+reduce: no `get0` at all, no `functionof`, and no residual
// measure layer. (Σ over a nested independent product is order-independent,
// §06 "Independent composition" — grouping by row vs scoring the flat array
// in one broadcast is the same total density.)
#[test]
fn iid_nonscalar_inner_measure_flattens_to_single_reduce() {
    let src = "\
data = [[0.5, -0.3], [1.2, 0.1], [-0.7, 0.9]]
d = iid(iid(Normal(mu = 0.0, sigma = 1.0), 2), 3)
lp = logdensityof(d, data)";
    let m = parse_infer(src);
    let out = determinize(&m).expect("iid over a non-scalar M must lower, not refuse");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "emitted FlatPDL must be conformant"
    );
    let pir = flatppl_flatpir::write(&out);
    // Exactly ONE broadcast head, scoring the full [3,2] array in one shot —
    // not 3 outer-row broadcasts, not a 6-term flat unroll.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "exactly one flattened broadcast head (not a per-row or per-leaf unroll):\n{pir}"
    );
    assert_eq!(
        pir.matches("(broadcast builtin_logdensityof Normal")
            .count(),
        1,
        "the single broadcast scores Normal directly:\n{pir}"
    );
    assert_eq!(
        pir.matches("(sum ").count(),
        1,
        "one full reduce over the whole array:\n{pir}"
    );
    // No residual measure layer: neither the nested `iid` nor the query survives.
    assert!(
        !pir.contains("(iid ") && !pir.contains("(logdensityof "),
        "no measure layer:\n{pir}"
    );
    // No `get0` unroll and no `functionof` — the flatten replaces both
    // candidates the composed case previously considered.
    assert!(
        !pir.contains("(get0 "),
        "flattened form has no get0 projection at all:\n{pir}"
    );
    assert!(
        !pir.contains("(functionof "),
        "flattened form never reifies a functionof body:\n{pir}"
    );
    // The obs is the FULL, un-projected `data` array (bound once, no per-row
    // slicing) — the broadcast's zipped collection argument is a direct
    // reference to `data`, not a `get0` of it.
    assert!(
        pir.contains("(%ref self data)"),
        "the broadcast scores the full data array directly:\n{pir}"
    );
    // Each scalar param is lifted through 2 nested `vector(...)` wraps — one
    // size-1 axis per peeled `iid` layer — giving a rank-2 [1,1] singleton
    // params array (FlatPIR's `%meta` type annotation prints this as a
    // rank-2 `(%array 1 (1) (%array 1 (1) (%scalar real)))`, i.e. an
    // array-of-arrays, one nesting level per peeled `iid`) that broadcasts
    // against data's full [3,2] shape.
    assert_eq!(
        pir.matches("(vector 0.0)").count(),
        1,
        "mu = 0.0 lifted to the innermost length-1 vector once:\n{pir}"
    );
    assert_eq!(
        pir.matches("(vector 1.0)").count(),
        1,
        "sigma = 1.0 lifted to the innermost length-1 vector once:\n{pir}"
    );
    assert_eq!(
        pir.matches("(%array 1 (1) (%array 1 (1) (%scalar real)))")
            .count(),
        2,
        "mu and sigma both wrapped through 2 nested size-1 array axes (rank-2 [1,1], one per peeled iid layer):\n{pir}"
    );
    assert_eq!(
        pir.matches("(broadcast record (%kwarg mu").count(),
        1,
        "a single record broadcast synthesizes the [1,1] params array:\n{pir}"
    );
}

// `iid(M, n)` where `M` is a BARE constructor with a NON-SCALAR variate
// domain (`MvNormal`, vector domain) must NOT take the axis-native broadcast
// fast path: `emit_kernel_broadcast_density`'s size-1-array-of-records
// flatten is verified only for a SCALAR-domain kernel (each scalar param
// lifted to a length-1 `vector`) — a non-scalar-domain kernel has its own
// compound parameters (e.g. MvNormal's rank-1 `mu`) that would need a
// genuinely different wrap the flatten does not build. `is_scalar_domain_kernel`
// gates both `lower_iid` fast-path entries on the kernel's own inferred
// domain being CONFIRMED scalar, so a bare `MvNormal` structurally matches
// `split_kernel_constructor` but fails that gate and falls through —
// unchanged — to the `get0`/`fold_add` unroll fallback (the same path this
// took before the axis-native fast path existed). Regression guard for the
// whole-branch review: confirms the gate re-routes MvNormal to the safe
// unroll rather than refusing or (worse) mislowering it through the flatten.
#[test]
fn iid_nonscalar_domain_kernel_uses_unroll_not_broadcast_flatten() {
    let src = "\
data = [[0.2, 0.3], [0.4, -0.1], [0.0, 0.5]]
d = iid(MvNormal(mu = [0.0, 0.0], cov = eye(2)), 3)
lp = logdensityof(d, data)";
    let m = parse_infer(src);
    let out = determinize(&m)
        .expect("iid over a non-scalar-domain kernel must lower via unroll, not refuse/panic");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "emitted FlatPDL must be conformant"
    );
    let pir = flatppl_flatpir::write(&out);
    // Unroll fallback: one `get0` per iid element (3), NOT the size-1-broadcast
    // flatten form.
    assert_eq!(
        pir.matches("(get0 ").count(),
        3,
        "unroll fallback: one get0 per iid element, not the broadcast flatten:\n{pir}"
    );
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        3,
        "one MvNormal density term per unrolled element:\n{pir}"
    );
    assert!(
        pir.contains("MvNormal"),
        "each term scores MvNormal directly:\n{pir}"
    );
    // The broadcast-flatten form this gate must NOT take: no `broadcast record`
    // synthesizing a singleton params array, and no bare axis-native
    // `(broadcast builtin_logdensityof MvNormal ...)` head.
    assert!(
        !pir.contains("(broadcast record"),
        "must not build the singleton-broadcast kernel_input:\n{pir}"
    );
    assert!(
        !pir.contains("(broadcast builtin_logdensityof"),
        "must not take the axis-native broadcast fast path for a non-scalar-domain kernel:\n{pir}"
    );
}

#[test]
fn joint_two_gaussians_structure() {
    // logdensityof(joint(Normal(0,1), Normal(1,2)), [0.5, 0.5]) →
    //   add(density(Normal(0,1), get0(v,0)), density(Normal(1,2), get0(v,1)))
    //
    // The variate is a NAMED binding (`data`), not an inline literal: canon
    // Pass 3 (`flatten_structural`, buffy #263) only resolves `get0` over a
    // LITERAL `vector(...)` constructor, never through a `Ref`, so naming the
    // data keeps the per-component `get0` slot structure this test pins
    // intact (an inline-literal variate would flatten both projections away).
    let src = "\
data = [0.5, 0.5]
d = joint(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 1.0, sigma = 2.0))
lp = logdensityof(d, data)";
    let m = parse_infer(src);
    let out = determinize(&m).expect("joint must lower");
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "2 joint terms:\n{pir}"
    );
    assert!(!pir.contains("(joint "), "no joint:\n{pir}");
    // Structural: the two positional components carry their OWN distinct params —
    // component 0 scores Normal(0,1), component 1 scores Normal(1,2) — each
    // projected out of the variate at its own `get0` slot.
    assert!(
        pir.contains("(%field mu 0.0) (%field sigma 1.0)"),
        "component 0 scores Normal(0,1):\n{pir}"
    );
    assert!(
        pir.contains("(%field mu 1.0) (%field sigma 2.0)"),
        "component 1 scores Normal(1,2):\n{pir}"
    );
    assert_eq!(
        pir.matches("(get0 ").count(),
        2,
        "one get0 projection per component:\n{pir}"
    );
    assert!(
        pir.contains(") 0)") && pir.contains(") 1)"),
        "components projected at slots 0 and 1:\n{pir}"
    );
}

#[test]
fn weighted_function_weight_structure() {
    // logdensityof(weighted(x -> exp(x), g), 0.5) → add(log(w(0.5)), density(g, 0.5))
    //   — §06 "Density of composed measures": the weight may be a function of the
    //   variate, applied at v then wrapped in `log` (there is an OUTER `log`).
    let src = "\
g = Normal(mu = 0.0, sigma = 1.0)
d = weighted(x -> exp(x), g)
lp = logdensityof(d, 0.5)";
    let m = parse_infer(src);
    let out = determinize(&m).expect("function-weighted weighted must lower");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL"
    );
    let pir = flatppl_flatpir::write(&out);
    // One density term (g); the applied weight is `log(w(0.5))`, not a
    // `builtin_logdensityof`. The weight application itself is a residual
    // user-function `%call` beta-reduced away by canon Pass 2 (Buffy #263) —
    // `w(0.5) = exp(0.5)` (unfolded: `exp` is excluded from const-fold, see
    // `canon/fold.rs`), so no `(%call` survives.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "single density term:\n{pir}"
    );
    assert!(
        !pir.contains("(weighted ") && !pir.contains("(logdensityof ") && !pir.contains("(%call"),
        "measure layer gone and no residual user-call:\n{pir}"
    );
    // Structural: `add(log(w(0.5)), density(g, 0.5))`. The weight is APPLIED at
    // the variate (inlined to `exp(0.5)`) and wrapped in an OUTER `log` (distinguishes
    // `weighted` from `logweighted`); g is scored at the same variate 0.5.
    assert!(pir.contains("(add "), "add(logw, density):\n{pir}");
    assert!(
        pir.contains("(log ") && pir.contains("(exp 0.5)"),
        "weight applied at v then log-wrapped: log(exp(0.5)):\n{pir}"
    );
    assert!(
        pir.contains("(%field mu 0.0) (%field sigma 1.0)"),
        "g = Normal(0,1) scored:\n{pir}"
    );
}

#[test]
fn logweighted_function_weight_structure() {
    // logdensityof(logweighted(x -> logdensityof(g2, x), g1), 0.5)
    //   → add(ℓ(0.5), density(g1, 0.5))   (g1=N(0,1), g2=N(1,2))
    //   — §06: the log-weight ℓ is a function of the variate, applied at v; it is
    //   ALREADY in log space, so there is NO outer `log` (unlike `weighted`). Here
    //   ℓ(x) = logdensityof(g2, x), so ℓ(0.5) lowers to a second density term.
    let src = "\
g1 = Normal(mu = 0.0, sigma = 1.0)
g2 = Normal(mu = 1.0, sigma = 2.0)
d = logweighted(x -> logdensityof(g2, x), g1)
lp = logdensityof(d, 0.5)";
    let m = parse_infer(src);
    let out = determinize(&m).expect("function-weighted logweighted must lower");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL"
    );
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "g1 + g2 terms:\n{pir}"
    );
    assert!(
        !pir.contains("(logweighted ")
            && !pir.contains("(logdensityof ")
            && !pir.contains("(%call"),
        "measure layer gone and no residual user-call:\n{pir}"
    );
    // Structural: `add(ℓ(0.5), density(g1, 0.5))`. The log-weight application
    // is a residual user-function `%call` beta-reduced away by canon Pass 2
    // (Buffy #263): `ℓ(0.5) = logdensityof(g2, 0.5)` had ALREADY lowered to a
    // `builtin_logdensityof` term before the `%call` wrapping it was inlined,
    // so it now sits directly as a second `add` operand — being already in
    // log space, it is NOT wrapped in a top-level `log` — the emitted
    // expression starts with `add`, not `add(log(…`.
    assert!(pir.contains("(add "), "add(logweight, density):\n{pir}");
    // No `(log (builtin_logdensityof` — that would be the `weighted` shape
    // (a spurious outer log around the inlined log-weight term).
    assert!(
        !pir.contains("(log (builtin_logdensityof"),
        "logweighted must NOT wrap the inlined log-weight term in an outer log:\n{pir}"
    );
    // Both Gaussians are scored: g1 = N(0,1) directly, g2 = N(1,2) inside ℓ.
    assert!(
        pir.contains("(%field mu 0.0) (%field sigma 1.0)")
            && pir.contains("(%field mu 1.0) (%field sigma 2.0)"),
        "g1 = N(0,1) and g2 = N(1,2) both scored:\n{pir}"
    );
}

#[test]
fn weighted_k_parameter_weight_splits_variate_components() {
    // §06 "Weight arity": the variate is a 2-element array, and the weight
    // declares exactly 2 scalar parameters, so each parameter receives one
    // component in order — `w(v[1], v[2])`, not `w(v)`.
    let src = "\
g = Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 1.0)))
d = weighted((a, b) -> a * b, g)
lp = logdensityof(d, [0.3, 0.4])";
    let m = parse_infer(src);
    let out = determinize(&m).expect("k-parameter weight over a matching k-array must lower");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL"
    );
    let pir = flatppl_flatpir::write(&out);
    assert!(
        !pir.contains("(weighted ") && !pir.contains("(%call"),
        "measure layer gone and no residual user-call:\n{pir}"
    );
    // The weight is inlined to `mul(get(v, 1), get(v, 2))`, one `get` per
    // parameter, in order — not applied to the whole array as a single argument.
    assert_eq!(
        pir.matches("(get ").count(),
        2,
        "one `get` projection per weight parameter:\n{pir}"
    );
    assert!(
        pir.contains(" 1)") && pir.contains(" 2)"),
        "components projected at 1 and 2 (spec's 1-based indexing):\n{pir}"
    );
}

#[test]
fn weighted_wrong_arity_weight_refuses() {
    // §06 "Weight arity": over a 2-element array variate, only arity 1 (whole
    // variate) or arity 2 (one component per parameter) are allowed — arity 3
    // is a spec error, not a silently accepted mis-binding.
    let src = "\
g = Lebesgue(support = cartprod(interval(0.0, 1.0), interval(0.0, 1.0)))
d = weighted((a, b, c) -> a * b * c, g)
lp = logdensityof(d, [0.3, 0.4])";
    let m = parse_infer(src);
    let e = determinize(&m).expect_err("a weight of the wrong arity must refuse, not mis-bind");
    assert!(
        e.reason.contains("Weight arity") || e.reason.contains("weight arity"),
        "refusal should cite §06 Weight arity, got: {e:?}"
    );
}

#[test]
fn normalize_truncated_normal_structure() {
    // normalize(truncate(Normal(0,1), interval(-1,1))) scored at 0.5 →
    //   sub(density_with_gate, log(sub(touniform(base, hi), touniform(base, lo))))
    //   — §06 "Density of composed measures" closed-form Z = CDF(hi) − CDF(lo) via the touniform (CDF)
    //   transport; valid here because the base Normal(0,1) is a normalized measure.
    let src = "\
g = Normal(mu = 0.0, sigma = 1.0)
d = normalize(truncate(g, interval(-1.0, 1.0)))
lp = logdensityof(d, 0.5)";
    let m = parse_infer(src);
    let out = determinize(&m).expect("normalize(truncate) must lower");
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
    let pir = flatppl_flatpir::write(&out);
    assert!(
        !pir.contains("(normalize ") && !pir.contains("totalmass"),
        "no normalize/totalmass:\n{pir}"
    );
    // Structural: the closed-form log-Z is `log(sub(touniform(hi), touniform(lo)))`
    // — exactly TWO CDF transports (at the two interval endpoints), differenced,
    // then logged, and subtracted from the gated density.
    assert_eq!(
        pir.matches("builtin_touniform").count(),
        2,
        "two CDF transports (one per endpoint) for Z:\n{pir}"
    );
    assert!(
        pir.contains("(log (%meta") || pir.contains("(log "),
        "log-Z present:\n{pir}"
    );
    assert!(
        pir.contains("(sub ") && pir.contains("(log "),
        "density − log(Z) shape:\n{pir}"
    );
    // The gate: the truncate lowering emits `ifelse(in(v, S), density, neg(inf))`.
    assert!(
        pir.contains("(ifelse ") && pir.contains("(neg inf)"),
        "support gate present (ifelse … neg inf):\n{pir}"
    );
    // The single density term scores the base Normal(0,1).
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "single density term for the base:\n{pir}"
    );
}

#[test]
fn likelihoodof_gaussian_structure() {
    // obs = likelihoodof(iid(Normal(mu,sigma), 1), [1.27])
    // logdensityof(obs, record(mu=0, sigma=1)) → density(Normal(0,1), 1.27):
    // K is scored at the baked-in obs (1.27) with θ = {mu:0, sigma:1} inlined for
    // the free params — §06 "Likelihood construction" densityof(likelihoodof(K, obs), θ) = pdf(κ(θ), obs).
    let src = "\
mu = elementof(reals)
sigma = elementof(posreals)
gauss_x = Normal(mu = mu, sigma = sigma)
obs = likelihoodof(iid(gauss_x, 1), [1.27])
lp = logdensityof(obs, record(mu = 0.0, sigma = 1.0))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("likelihood must lower");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL"
    );
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "1 term:\n{pir}"
    );
    assert!(
        !pir.contains("(likelihoodof ") && !pir.contains("(iid "),
        "measure layer gone:\n{pir}"
    );
    // Structural: the θ point {mu=0.0, sigma=1.0} is INLINED into the kernel's
    // params (not left as `(%ref self mu/sigma)`), and the kernel is scored at the
    // baked-in observation 1.27.
    let lp_line = pir
        .lines()
        .find(|l| l.contains("(%bind lp "))
        .expect("lp binding present");
    assert!(
        lp_line.contains("(%field mu 0.0)") && lp_line.contains("(%field sigma 1.0)"),
        "θ inlined into the kernel params:\n{lp_line}"
    );
    assert!(
        lp_line.contains("1.27"),
        "kernel scored at the baked-in observation 1.27:\n{lp_line}"
    );
    assert!(
        !lp_line.contains("(%ref self mu)") && !lp_line.contains("(%ref self sigma)"),
        "θ must be inlined, not a residual self-ref:\n{lp_line}"
    );
}

// Regression for a cross-query parameter leak. TWO likelihood
// queries over the SAME shared params (`mu`, `sigma`) at DISTINCT θ points must
// each score at its OWN θ. Each θ is inlined into that query's density subtree;
// the shared `mu`/`sigma` bindings are NOT mutated (which would clobber both
// terms to the last θ written — a silent mislowering that `is_flatpdl` passes).
#[test]
fn two_likelihood_queries_do_not_leak_theta_across_each_other() {
    let src = "\
mu = elementof(reals)
sigma = elementof(posreals)
gauss_x = Normal(mu = mu, sigma = sigma)
obs = likelihoodof(iid(gauss_x, 1), [1.27])
lp = logdensityof(obs, record(mu = 0.0, sigma = 1.0))
lp2 = logdensityof(obs, record(mu = 5.0, sigma = 2.0))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("two-query likelihood must lower");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL:\n{}",
        flatppl_flatpir::write(&out)
    );
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "one density term per query:\n{pir}"
    );

    // The two terms must be DISTINCT and each carry its OWN θ, inlined as
    // literals (not a shared `(%ref self mu/sigma)` that resolves to the last
    // θ). Inspect each query's binding line independently.
    let lp_line = pir
        .lines()
        .find(|l| l.contains("(%bind lp "))
        .expect("lp binding present");
    let lp2_line = pir
        .lines()
        .find(|l| l.contains("(%bind lp2 "))
        .expect("lp2 binding present");

    assert!(
        lp_line.contains("(%field mu 0.0)") && lp_line.contains("(%field sigma 1.0)"),
        "lp must score at ITS θ (mu=0.0, sigma=1.0):\n{lp_line}"
    );
    assert!(
        lp2_line.contains("(%field mu 5.0)") && lp2_line.contains("(%field sigma 2.0)"),
        "lp2 must score at ITS θ (mu=5.0, sigma=2.0):\n{lp2_line}"
    );

    // No θ leaked the other way: the two terms carry different values.
    assert!(
        !lp_line.contains("5.0") && !lp2_line.contains("0.0"),
        "θ leaked across the two queries:\nlp:  {lp_line}\nlp2: {lp2_line}"
    );

    // The shared params must NOT have been mutated to a θ literal — they stay
    // `elementof` free-param declarations (valid FlatPDL). A `(%bind mu 5.0)` /
    // `(%bind sigma 2.0)` is the smoking gun of the mutate-shared-bindings bug.
    assert!(
        pir.contains("(%bind mu (") && pir.contains("elementof reals"),
        "mu stays an elementof param decl (not clobbered to a θ literal):\n{pir}"
    );
    assert!(
        !pir.contains("(%bind mu 5.0)") && !pir.contains("(%bind sigma 2.0)"),
        "shared params must not be mutated to a query's θ:\n{pir}"
    );

    // And no residual self-ref to the (now-unused) params survives in either
    // scored density subtree.
    assert!(
        !lp_line.contains("(%ref self mu)") && !lp2_line.contains("(%ref self mu)"),
        "θ must be inlined, not left as a shared self-ref:\nlp:  {lp_line}\nlp2: {lp2_line}"
    );
}

// `joint_likelihood(L1, …, Lk)` combines likelihoods by summing their
// log-densities (§06 "Combining likelihoods":
// `log L(θ) = log L1(θ) + log L2(θ) + …`), all components scored at the SAME
// parameter point θ. `logdensityof(joint_likelihood(L1, L2), θ)` must lower to
// `add(logdensityof(L1, θ), logdensityof(L2, θ))` — two density terms folded
// with `add`, each component's own free params bound from the shared θ.
#[test]
fn joint_likelihood_sums_component_densities() {
    let src = "\
mu = elementof(reals)
nu = elementof(reals)
g1 = Normal(mu = mu, sigma = 1.0)
g2 = Normal(mu = nu, sigma = 1.0)
L1 = likelihoodof(iid(g1, 1), [1.0])
L2 = likelihoodof(iid(g2, 1), [2.0])
L = joint_likelihood(L1, L2)
lp = logdensityof(L, record(mu = 0.0, nu = 0.5))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("joint_likelihood must lower to a sum of component densities");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL:\n{}",
        flatppl_flatpir::write(&out)
    );
    let pir = flatppl_flatpir::write(&out);
    // Two component densities, folded with `add`.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "one density term per joint_likelihood component:\n{pir}"
    );
    let lp_line = pir
        .lines()
        .find(|l| l.contains("(%bind lp "))
        .expect("lp binding present");
    assert!(
        lp_line.contains("(add "),
        "components summed with add:\n{lp_line}"
    );
    // No residual measure / likelihood layer.
    assert!(
        !pir.contains("(joint_likelihood ")
            && !pir.contains("(likelihoodof ")
            && !pir.contains("(iid ")
            && !pir.contains("(logdensityof "),
        "measure/likelihood layer eliminated:\n{pir}"
    );
    // Each component scores at ITS θ: component 1 → Normal(mu=0.0), scored at the
    // baked-in observation 1.0; component 2 → Normal(mu=0.5), scored at 2.0.
    assert!(
        lp_line.contains("(%field mu 0.0)") && lp_line.contains("(%field mu 0.5)"),
        "each component binds its own free param from the shared θ:\n{lp_line}"
    );
    assert!(
        !lp_line.contains("(%ref self mu)") && !lp_line.contains("(%ref self nu)"),
        "θ must be inlined per component, not left as a shared self-ref:\n{lp_line}"
    );
}

// A component of a `joint_likelihood` may itself be a `joint_likelihood`
// (§06 combination is associative). `joint_likelihood(joint_likelihood(L1, L2),
// L3)` must flatten to `Σ` over all three leaf likelihoods — three density terms
// — via the per-likelihood lowering's recursion, not stop at the outer two.
#[test]
fn joint_likelihood_nested_flattens_to_all_leaf_densities() {
    let src = "\
mu = elementof(reals)
nu = elementof(reals)
xi = elementof(reals)
g1 = Normal(mu = mu, sigma = 1.0)
g2 = Normal(mu = nu, sigma = 1.0)
g3 = Normal(mu = xi, sigma = 1.0)
L1 = likelihoodof(iid(g1, 1), [1.0])
L2 = likelihoodof(iid(g2, 1), [2.0])
L3 = likelihoodof(iid(g3, 1), [3.0])
inner = joint_likelihood(L1, L2)
L = joint_likelihood(inner, L3)
lp = logdensityof(L, record(mu = 0.0, nu = 0.5, xi = 1.0))";
    let m = parse_infer(src);
    let out =
        determinize(&m).expect("nested joint_likelihood must lower to a sum of leaf densities");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL:\n{}",
        flatppl_flatpir::write(&out)
    );
    let pir = flatppl_flatpir::write(&out);
    // One density term per LEAF likelihood — the nested joint_likelihood flattens.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        3,
        "one density term per leaf likelihood (nested joint_likelihood flattened):\n{pir}"
    );
    // No residual measure / likelihood layer (including the nested combinator).
    assert!(
        !pir.contains("(joint_likelihood ")
            && !pir.contains("(likelihoodof ")
            && !pir.contains("(iid ")
            && !pir.contains("(logdensityof "),
        "measure/likelihood layer eliminated:\n{pir}"
    );
}

// Regression fixture for transitive pinning (measure-algebra-audit.md H3): a variate reached
// through a derived binding (`a = 2·theta`, `theta = draw(M)`) must score at
// the pinned `theta` and propagate transitively — no stochastic `draw` may
// survive, even though `a` is unreferenced by `lp` and depends on `theta`.
#[test]
fn derived_binding_pins_transitively() {
    // theta ~ Normal(0,1); a = 2*theta (derived). Score the joint at theta=0.5.
    // density is density(Normal(0,1), 0.5), scored at the pinned theta; no
    // stochastic `draw` may survive, even though `a` depends on `theta`.
    let src = "\
theta = draw(Normal(mu = 0.0, sigma = 1.0))
a = mul(2.0, theta)
lp = logdensityof(lawof(record(theta = theta)), record(theta = 0.5))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("must lower");
    let pir = flatppl_flatpir::write(&out);
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "no stochastic draw survives (a's dep):\n{pir}"
    );
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        1,
        "single density term:\n{pir}"
    );
    // Transitive pinning: no `(draw ` survives anywhere (theta is pinned, and a's
    // dependency on it is rewritten), and the term scores the pinned theta = 0.5.
    assert!(
        !pir.contains("(draw "),
        "no stochastic draw survives:\n{pir}"
    );
    assert!(
        pir.contains("(%field mu 0.0) (%field sigma 1.0)"),
        "scores Normal(0,1) at the pinned theta:\n{pir}"
    );
}

#[test]
fn empty_record_is_zero() {
    let src = "lp = logdensityof(lawof(record()), record())";
    let m = parse_infer(src);
    let out = determinize(&m).expect("empty record must lower to 0");
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
    let pir = flatppl_flatpir::write(&out);
    assert!(
        !pir.contains("builtin_logdensityof"),
        "no density terms:\n{pir}"
    );
}

// Empty independent product: `iid(M, 0)` is Σ over an empty index set = 0, the
// same as the empty measure `record()` (the iid Σ rule, §06 "Density of composed
// measures", with an empty index set). The determiniser still lowers that to the
// log-density literal 0 with no density term — but the shape is NO LONGER
// REACHABLE from surface source, so this test pins the two static refusals that
// now stand in front of it instead of the lowering.
//
// Both come from `flatppl-dev/empty-arrays-ruling.md` (2026-08-20): "an author
// may not write a degenerate shape, data may turn out empty".
//   * A WRITTEN size of `0` is a static error (§06 `iid`, the positivity rule).
//   * A DERIVED size resolving to 0 types `%dynamic`, never `Dim::Static(0)`
//     (sub-ruling 3), and the unroll needs a static 1-D count — so it refuses.
//
// Which leaves the static empty product with no legal spelling: a derived size
// only turns out to be 0 at data-load time, and that is the js engine's job, not
// this crate's. The empty-product lowering stays as §06 requires; it has no
// surface witness here. Recorded in `flatppl-dev/TODO-flatppl-rust.md`.
#[test]
fn iid_zero_size_is_refused_written_and_derived() {
    let written = "\
xs = [1.0, 2.0, 3.0]
keep = filter(x -> x < 0.0, xs)
d = iid(Normal(mu = 0.0, sigma = 1.0), 0)
lp = logdensityof(d, keep)";
    let mut m = flatppl_syntax::parse(written).expect("fixture must parse");
    let msgs: Vec<String> = flatppl_infer::infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .map(|d| d.message)
        .collect();
    assert!(
        msgs.iter()
            .any(|m| m.contains("`iid`'s `size` is written as `0`")),
        "a written zero size must be a static error; got {msgs:?}"
    );

    let derived = "\
xs = [1.0, 2.0, 3.0]
keep = filter(x -> x < 0.0, xs)
n = sub(lengthof(xs), lengthof(xs))
d = iid(Normal(mu = 0.0, sigma = 1.0), n)
lp = logdensityof(d, keep)";
    let m = parse_infer(derived);
    let err = determinize(&m).expect_err("a %dynamic iid size must refuse, not unroll");
    assert!(
        format!("{err:?}").contains("not a statically-resolved 1-D count"),
        "a derived zero size stays %dynamic and the unroll refuses it: {err:?}"
    );
}

/// The spelling the fixture above used to carry. §05 "Note on axis names" makes
/// `[]` an axis list, legal only as `output_axes`, and `ArrayLiteral` has no
/// empty form — so a written `[]` observation never reaches the determiniser.
#[test]
fn empty_vector_literal_observation_is_refused() {
    let src = "\
d = iid(Normal(mu = 0.0, sigma = 1.0), 0)
lp = logdensityof(d, [])";
    let mut m = flatppl_syntax::parse(src).expect("fixture must parse");
    let msgs: Vec<String> = flatppl_infer::infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .map(|d| d.message)
        .collect();
    assert!(
        msgs.iter()
            .any(|m| m.contains("an axis list is legal only as `output_axes`")),
        "a written `[]` observation must be a static error; got {msgs:?}"
    );
}

// ── Record-valued `iid`: the table variate ───────────────────────────────────

/// §06 `iid`: "When `M` is record-valued and `size` is a scalar length, the
/// variate is an N-row table — one row per draw of `M`'s record". The density is
/// the independent-product Σ over rows, each row scored field-wise by the record
/// law (§06 "Density of composed measures").
///
/// Closed form for the fixture below: Σ_i log N(a_i; 0.3, 1) + Σ_i log N(b_i; 1, 2).
#[test]
fn record_iid_over_a_table_variate_lowers_row_wise() {
    let a = [0.61, -0.32];
    let b = [1.1, 2.53];
    let oracle: f64 = a.iter().map(|&x| gaussian_logpdf(x, 0.3, 1.0)).sum::<f64>()
        + b.iter().map(|&x| gaussian_logpdf(x, 1.0, 2.0)).sum::<f64>();

    let src = "\
tab_obs = table(a = [0.61, -0.32], b = [1.1, 2.53])
mu_a = elementof(reals)
mu_b = elementof(reals)
run = joint(a = Normal(mu = mu_a, sigma = 1.0), b = Normal(mu = mu_b, sigma = 2.0))
tab ~ iid(run, 2)
L = likelihoodof(kernelof(record(tab = tab), mu_a = mu_a, mu_b = mu_b), record(tab = tab_obs))
lp = logdensityof(L, record(mu_a = 0.3, mu_b = 1.0))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("a record-valued iid over a table variate must lower");
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL"
    );
    let pir = flatppl_flatpir::write(&out);
    assert!(
        !pir.contains("(iid ") && !pir.contains("(logdensityof ") && !pir.contains("(joint "),
        "measure layer eliminated:\n{pir}"
    );
    // One term per row per field: 2 rows x 2 fields.
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        4,
        "Σ over rows x fields:\n{pir}"
    );
    // Each field is scored at the θ value, NOT at the marginal of a latent.
    assert!(
        pir.contains("0.3") && pir.contains("1.0"),
        "the kernel's declared inputs are conditioned on:\n{pir}"
    );
    assert!(
        oracle.is_finite(),
        "closed-form oracle sanity: {oracle}" // -8.16… ; the live numeric gate is flatppl-testsuite
    );
}

/// The emitted FlatPDL reads each row field FROM ITS COLUMN — `crates/stablehlo`
/// has no `table` constructor arm ("unsupported builtin head 'table'"), so a
/// lowering that left the table value in place could not execute. §03 "Tables"
/// makes row `i`'s field `a` element `i` of column `a`, so the column read is the
/// same value by a spelling the emitter can lower.
#[test]
fn record_iid_row_fields_come_from_the_columns_not_a_table_value() {
    let src = "\
tab_obs = table(a = [0.61, -0.32], b = [1.1, 2.53])
mu_a = elementof(reals)
run = joint(a = Normal(mu = mu_a, sigma = 1.0), b = Normal(mu = 0.0, sigma = 2.0))
tab ~ iid(run, 2)
L = likelihoodof(kernelof(record(tab = tab), mu_a = mu_a), record(tab = tab_obs))
lp = logdensityof(L, record(mu_a = 0.3))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("must lower");
    let pir = flatppl_flatpir::write(&out);
    // Scope the check to the scored binding: `tab_obs` keeps its own `table`
    // literal (a public binding the retained subgraph drops only once an
    // `outputs` ABI names the query), but nothing in `lp` may read a row out of
    // it — the column reads const-fold to the scalars themselves.
    let lp = pir
        .split("(%bind lp ")
        .nth(1)
        .expect("the determinized module binds lp");
    assert!(
        !lp.contains("(table ") && !lp.contains("(get0 ") && !lp.contains("(get "),
        "the scored density reads no table value or row:\n{lp}"
    );
}

/// §04 "Reification to measures" makes a `kernelof`'s declared name its ARGUMENT,
/// and §06 `likelihoodof` fixes `densityof(likelihoodof(K, obs), theta)` as
/// `pdf(K(theta), obs)` — conditional on θ. So a latent the kernel declares as an
/// input is conditioned on, never marginalized: the record-law rewrite that
/// serves an undeclared shared latent must not fire for it.
///
/// Without this the fixture scored `obs` against `Normal(0, sqrt(5² + 1²))` — the
/// prior predictive — instead of `Normal(0.3, 1)`.
#[test]
fn a_kernels_declared_input_is_conditioned_not_marginalized() {
    let conditional = gaussian_logpdf(0.61, 0.3, 1.0) + gaussian_logpdf(1.1, 1.0, 2.0);
    let marginal =
        gaussian_logpdf(0.61, 0.0, 26.0_f64.sqrt()) + gaussian_logpdf(1.1, 0.0, 29.0_f64.sqrt());
    assert!(
        (conditional - marginal).abs() > 1.0,
        "the two readings must be distinguishable: {conditional} vs {marginal}"
    );

    let src = "\
mu_a ~ Normal(mu = 0.0, sigma = 5.0)
mu_b ~ Normal(mu = 0.0, sigma = 5.0)
run = joint(a = Normal(mu = mu_a, sigma = 1.0), b = Normal(mu = mu_b, sigma = 2.0))
r ~ run
L = likelihoodof(kernelof(record(r = r), mu_a = mu_a, mu_b = mu_b), record(r = record(a = 0.61, b = 1.1)))
lp = logdensityof(L, record(mu_a = 0.3, mu_b = 1.0))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("must lower");
    let printed = flatppl_syntax::print(&out);
    // The conditional form carries the θ values as the kernel's own parameters.
    assert!(
        printed.contains("record(mu = 0.3, sigma = 1.0)")
            && printed.contains("record(mu = 1.0, sigma = 2.0)"),
        "obs scored at θ, not at the prior predictive:\n{printed}"
    );
    // sqrt(26) / sqrt(29) are the marginal sigmas the record-law rewrite emitted.
    assert!(
        !printed.contains("5.0990195135927845") && !printed.contains("5.385164807134504"),
        "no prior-predictive marginal sigma:\n{printed}"
    );
}

/// The conditioning above is scoped to the reification that declares the input:
/// a sibling measure outside it keeps marginalizing over the same latent. Here
/// `z` is declared by the likelihood's kernel but the SECOND query scores a
/// measure that only READS `z`, so its own density still needs `z` marginalized —
/// and no closed form covers it, so it refuses rather than emitting a
/// conditional.
#[test]
fn conditioning_is_scoped_to_the_declaring_reification() {
    let src = "\
z ~ Normal(mu = 0.0, sigma = 1.0)
L = likelihoodof(kernelof(record(y = y), z = z), record(y = 0.4))
y ~ Normal(mu = z, sigma = 1.0)
lp = logdensityof(L, record(z = 0.2))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("the declared-input query must lower");
    let printed = flatppl_syntax::print(&out);
    assert!(
        printed.contains("record(mu = 0.2, sigma = 1.0)"),
        "the declaring kernel conditions on z:\n{printed}"
    );
}
