//! Applied-reification boundary substitution: `k(z = v)` must score at `v`, not at
//! `z`.
//!
//! §04 *Specifying reification boundaries* makes the applied value the boundary node's
//! only content inside the reified graph — "A specified boundary node `a` can be thought
//! of as being substituted with a new node, generated via `elementof(valueset(a))`, in
//! the reified graph". `kernel::substitute_ref` is syntactic, so it delivered that only
//! where the reference was a literal descendant of the body; a boundary reached through
//! a derived binding or through a `record` field's own ref survived, and the query
//! lowered to a function of the pinned parameter instead of refusing.
//!
//! Oracles are closed-form and were cross-checked against Distributions.jl. No engine
//! output was used as a target — the assertions read the emitted FlatPDL's structure and
//! the report carries the arithmetic.
use flatppl_determinizer::{determinize, determinize_with_roots};
use flatppl_infer::ModuleBundle;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

/// Lower with `lp` as the only requested output, so root-based DCE drops the bindings
/// nothing reaches. A surviving `elementof` parameter then means the SCORED density
/// still reads it, which is the claim these tests make — without the roots every dead
/// binding stays and the assertion would be vacuous.
fn lower(src: &str) -> String {
    let mut m = parse_infer(src);
    let lp = m.intern("lp");
    let out = determinize_with_roots(&m, &ModuleBundle::new(), Some(&[lp])).expect("must lower");
    flatppl_syntax::print(&out)
}

/// The boundary is reached through a derived binding, so the syntactic substitution
/// stopped at `(%ref self mu2)`.
///
/// Oracle: `mu2 = 2.0 * 0.5 = 1.0`, so the law is `Normal(1.0, 1.0)` and the density at
/// `1.0` is `-0.5 * log(2 * pi) = -0.9189385332046727`. Before the fix the emitted
/// density read `mu = mu2` with `z = elementof(reals)` surviving as a determinized
/// input — a function of the parameter the application pinned.
#[test]
fn applied_reification_substitutes_a_boundary_reached_through_a_derived_binding() {
    let out = lower(
        "\
z   = elementof(reals)
mu2 = 2.0 * z
b1 ~ Normal(mu = mu2, sigma = 1.0)
K   = kernelof(b1, z = z)
lp  = logdensityof(K(z = 0.5), 1.0)",
    );
    assert!(
        out.contains("record(mu = 1.0, sigma = 1.0)"),
        "the pinned boundary must reach the derived mean (2.0 * 0.5 = 1.0); got:\n{out}"
    );
    assert!(
        !out.contains("elementof") && !out.contains("mu2"),
        "no unpinned parameter may survive in the scored density; got:\n{out}"
    );
}

/// A `record` body's fields are refs, so the syntactic substitution never reached the
/// draws' own constructors.
///
/// Oracle: both fields are `Normal(0.0, 1.0)` after the pin, and the independent product
/// at `(1.0, -1.0)` is `-log(2 * pi) - 1 = -2.8378770664093453`.
#[test]
fn applied_reification_substitutes_a_boundary_reached_through_record_fields() {
    let out = lower(
        "\
z = elementof(reals)
b1 ~ Normal(mu = z, sigma = 1.0)
b2 ~ Normal(mu = z, sigma = 1.0)
KR = kernelof(record(p = b1, q = b2), z = z)
lp = logdensityof(KR(z = 0.0), record(p = 1.0, q = -1.0))",
    );
    assert_eq!(
        out.matches("record(mu = 0.0, sigma = 1.0)").count(),
        2,
        "both factors must read the pinned boundary; got:\n{out}"
    );
    assert!(
        !out.contains("elementof"),
        "no unpinned parameter may survive in the scored density; got:\n{out}"
    );
}

/// One derived binding feeding SEVERAL record fields is substituted once and reaches
/// both — the deterministic half of the node-identity requirement the shared-latent case
/// below carries at the stochastic level.
///
/// Oracle: `mu2 = 2.0 * 0.5 = 1.0` in both factors, so the product at `(1.0, -1.0)` is
/// `log N(1; 1, 1) + log N(-1; 1, 1) = -log(2 * pi) - 2 = -3.8378770664093453`.
#[test]
fn applied_reification_reaches_every_field_through_one_shared_derived_binding() {
    let out = lower(
        "\
z   = elementof(reals)
mu2 = 2.0 * z
b1 ~ Normal(mu = mu2, sigma = 1.0)
b2 ~ Normal(mu = mu2, sigma = 1.0)
KR  = kernelof(record(p = b1, q = b2), z = z)
lp  = logdensityof(KR(z = 0.5), record(p = 1.0, q = -1.0))",
    );
    assert_eq!(
        out.matches("record(mu = 1.0, sigma = 1.0)").count(),
        2,
        "the shared derived mean must be pinned in both factors; got:\n{out}"
    );
    assert!(!out.contains("elementof"), "got:\n{out}");
}

/// THE RETAIN-not-COPY regression, and the reason the substitution is finished AFTER the
/// lowering rather than deepened before it.
///
/// `u` is an interior latent of the reification, shared by both record fields. §06's
/// `joint` ancestry rule keeps it ONE node, so the law at `z = 0` is
/// `MvNormal([0,0], [[2,1],[1,2]])` and the density at `(1, -1)` is
/// `-log(2*pi) - 0.5*log(3) - 1 = -3.3871832107434003`. Cloning `u` per field — what an
/// unmemoized deep pre-substitution does — gives the `iid`-style COPY product of two
/// `Normal(0, sqrt 2)` marginals, `-3.0310242469693`.
///
/// The discriminator is structural and exact: RETAIN emits ONE closed-form record-law
/// term carrying the shared-latent covariance factor (`log1p(2.0)`, the `det` of the
/// correlated covariance), while COPY emits a SUM of two per-field factors. Both oracle
/// numbers are closed-form and were cross-checked against Distributions.jl.
///
/// The applied form must also agree with the unapplied `lawof(record(...))` twin at the
/// same point, which is the shape `marginal::shared_latent_record_law` already lowered
/// correctly before this wave. Asserting the two emissions are IDENTICAL pins the
/// commuting agreement rather than re-deriving the arithmetic here.
#[test]
fn an_applied_reification_marginalizes_a_shared_interior_latent_as_the_record_law() {
    let applied = lower(
        "\
z  = elementof(reals)
u  ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
a2 ~ Normal(mu = u, sigma = 1.0)
KR = kernelof(record(p = a1, q = a2), z = z)
lp = logdensityof(KR(z = 0.0), record(p = 1.0, q = -1.0))",
    );
    // The unapplied twin: `u`'s prior is written at the pinned value directly.
    let unapplied = lower(
        "\
u  ~ Normal(mu = 0.0, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
a2 ~ Normal(mu = u, sigma = 1.0)
lp = logdensityof(lawof(record(p = a1, q = a2)), record(p = 1.0, q = -1.0))",
    );
    assert!(
        applied.contains("log1p(2.0)"),
        "the correlated record law's covariance factor must be emitted (RETAIN), not a \
         product of per-field marginals (COPY); got:\n{applied}"
    );
    assert!(
        !applied.contains("builtin_logdensityof"),
        "the record law is ONE closed-form term; a per-field product would emit \
         `builtin_logdensityof` factors (COPY); got:\n{applied}"
    );
    assert_eq!(
        applied, unapplied,
        "the applied reification at z = 0 must lower exactly as its unapplied twin"
    );
}

/// The pinned boundary really reaches the interior latent's own prior: moving the
/// application point moves the emitted law.
///
/// At `z = 5`, `(1, -1)` the law is `MvNormal([5,5], [[2,1],[1,2]])`. The quadratic form
/// is `(y - mu)' Sigma^-1 (y - mu)` with `y - mu = (-4, -6)`:
/// `(1/3) * (2*16 - 2*24 + 2*36) = 56/3 = 18.666666666666664`. The covariance factor is
/// unchanged, so only that term moves — which is what a boundary that reaches the prior's
/// MEAN and nothing else must do.
#[test]
fn an_applied_reification_boundary_reaches_the_interior_latents_prior() {
    let out = lower(
        "\
z  = elementof(reals)
u  ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
a2 ~ Normal(mu = u, sigma = 1.0)
KR = kernelof(record(p = a1, q = a2), z = z)
lp = logdensityof(KR(z = 5.0), record(p = 1.0, q = -1.0))",
    );
    assert!(
        out.contains("18.666666666666664"),
        "the quadratic form must move with the application point (56/3 at z = 5); \
         got:\n{out}"
    );
    assert!(
        out.contains("log1p(2.0)") && !out.contains("elementof"),
        "the covariance factor is unchanged and no parameter survives; got:\n{out}"
    );
}

/// An applied value that READS the boundary node is substituted exactly ONCE.
///
/// `K(z = z + 1.0)` applies `K` at the ambient `z` plus one, which is legal and which
/// `a86437d` lowered correctly. Two substitution passes over the same entry is a wrong
/// number rather than a no-op: the syntactic pass writes `z + 1.0` into the body, and a
/// second pass cannot tell its own output from source, so it yields `z + 1.0 + 1.0` — at
/// `z = 0` that scores `-1.4189385332046727` where the truth is `-0.9189385332046727`.
///
/// The fix is that a same-module boundary target is substituted by the FINISH alone
/// (`kernel::Substitute::LocalOnly`), never by both. Dropping such an entry from the
/// finish's map instead would restore the original boundary-drop bug for the
/// derived-binding spelling, so the three shapes below are pinned together with the
/// derived-binding test above rather than in place of it.
#[test]
fn a_self_referential_applied_value_is_substituted_exactly_once() {
    for (applied, expected) in [
        ("z + 1.0", "record(mu = z + 1.0, sigma = 1.0)"),
        ("2.0 * z", "record(mu = 2.0 * z, sigma = 1.0)"),
    ] {
        let out = lower(&format!(
            "\
z  = elementof(reals)
b1 ~ Normal(mu = z, sigma = 1.0)
K  = kernelof(b1, z = z)
lp = logdensityof(K(z = {applied}), 1.0)"
        ));
        assert!(
            out.contains(expected),
            "`K(z = {applied})` must substitute once, giving `{expected}`; got:\n{out}"
        );
    }
    // The same hazard one indirection away: the applied value is a BINDING whose own
    // RHS reads the boundary node.
    let out = lower(
        "\
z  = elementof(reals)
v  = z + 1.0
b1 ~ Normal(mu = z, sigma = 1.0)
K  = kernelof(b1, z = z)
lp = logdensityof(K(z = v), 1.0)",
    );
    assert!(
        out.contains("record(mu = v, sigma = 1.0)"),
        "the applied binding must stand as written, not `v + 1.0`; got:\n{out}"
    );
}

/// A CROSS-NAMED applied value lowers, keeping the ambient sibling.
///
/// `K(z = w, w = 0.5)` binds input `z` to the ambient `w` and input `w` to `0.5`, so over
/// the body `mu = z + w` the reified graph reads `mu = w + 0.5` with the ambient `w`
/// surviving as a determinized input. §04 is what makes that the answer rather than
/// `0.5 + 0.5`: an applied value is evaluated in the AMBIENT scope and is not part of the
/// reified graph, so the `w` input's pin does not reach the `w` that input `z`'s value
/// names.
///
/// Oracle at ambient `w = 0`: `log N(1; 0.5, 1) = -1.0439385332046727`. Both substitution
/// passes are simultaneous, so the density has always been right to emit; what changed is
/// the residual guard, which flagged the legitimately-present `w` as a possibly-missed
/// occurrence and refused. It now excludes any target that ANY applied value reads.
#[test]
fn a_cross_named_applied_value_keeps_the_ambient_sibling() {
    let out = lower(
        "\
z  = elementof(reals)
w  = elementof(reals)
b1 ~ Normal(mu = z + w, sigma = 1.0)
K  = kernelof(b1, z = z, w = w)
lp = logdensityof(K(z = w, w = 0.5), 1.0)",
    );
    assert!(
        out.contains("record(mu = w + 0.5, sigma = 1.0)"),
        "input `z` binds the ambient `w` and input `w` binds 0.5; got:\n{out}"
    );
    assert!(
        out.contains("w = elementof(reals)"),
        "the ambient `w` must survive as a determinized input; got:\n{out}"
    );
}

/// The cyclic swap, where no substitution ORDER is correct.
///
/// `K(z = w, w = z)` over `mu = 2.0 * z + w` is `mu = 2.0 * w + z`, so BOTH ambient
/// parameters survive. Oracle at ambient `w = 1`, `z = 0.5`:
/// `log N(1; 2.5, 1) = -2.0439385332046727`, against `log N(1; 1.5, 1) =
/// -1.0439385332046727` for the sequential `2.0 * z + z`.
#[test]
fn a_cyclic_swap_of_applied_values_exchanges_both_exactly_once() {
    let out = lower(
        "\
z  = elementof(reals)
w  = elementof(reals)
b1 ~ Normal(mu = 2.0 * z + w, sigma = 1.0)
K  = kernelof(b1, z = z, w = w)
lp = logdensityof(K(z = w, w = z), 1.0)",
    );
    assert!(
        out.contains("record(mu = 2.0 * w + z, sigma = 1.0)"),
        "a swap exchanges the two parameters exactly once; got:\n{out}"
    );
    assert!(
        out.contains("z = elementof(reals)") && out.contains("w = elementof(reals)"),
        "both ambient parameters survive a swap; got:\n{out}"
    );
}

/// The residual guard's REAL job, kept: a boundary occurrence no applied value explains
/// still refuses.
///
/// `h = functionof(z + 1.0, z = z)` inside the scored body carries `z` in its own
/// `%specinputs` boundary, which the finish's `children()` walk cannot reach and must not
/// rewrite (it belongs to `h`'s scope). The application pins `z = 0.5`, no applied value
/// reads `z`, so the surviving occurrence is a genuinely missed one and the density would
/// score at the unpinned parameter.
///
/// The second shape is the one that pins the exclusion as per-TARGET rather than a
/// blanket disarming: it holds a cross-named pair (`z = w`) AND a third input `v` that
/// leaks the same way. `w` is excluded because input `z`'s value reads it; `v` is not,
/// because nothing reads `v` — so the refusal must still fire, and must name `v`.
#[test]
fn a_boundary_occurrence_no_applied_value_reads_still_refuses() {
    let leaked_only = "\
z  = elementof(reals)
h  = functionof(z + 1.0, z = z)
b1 ~ Normal(mu = h(2.0), sigma = 1.0)
K  = kernelof(b1, z = z)
lp = logdensityof(K(z = 0.5), 1.0)";
    let err = determinize(&parse_infer(leaked_only)).expect_err("the boundary leaks through `h`");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("could not reach") && msg.contains("`z`"),
        "the refusal must name the unreachable occurrence of `z`; got: {msg}"
    );

    let leaked_beside_a_cross_name = "\
z  = elementof(reals)
w  = elementof(reals)
v  = elementof(reals)
h  = functionof(v + 1.0, v = v)
b1 ~ Normal(mu = z + w + h(2.0), sigma = 1.0)
K  = kernelof(b1, z = z, w = w, v = v)
lp = logdensityof(K(z = w, w = 0.5, v = 3.0), 1.0)";
    let err = determinize(&parse_infer(leaked_beside_a_cross_name))
        .expect_err("`v` leaks even though `w` is legitimately present");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("`v`") && !msg.contains("`w`"),
        "a cross-named sibling must not excuse a target nothing reads; got: {msg}"
    );
}

/// A query point that names the boundary node refuses.
///
/// §04 scopes the substitution to the REIFIED graph, so a point written as the ambient
/// `z` must keep reading the ambient `z`. The substitution that finishes the boundary
/// binding rewrites the whole emitted density, the point included, so the two cannot
/// share a name — refuse rather than score at the applied value.
#[test]
fn an_applied_reification_refuses_a_query_point_naming_the_boundary() {
    let src = "\
z  = elementof(reals)
b1 ~ Normal(mu = z, sigma = 1.0)
K  = kernelof(b1, z = z)
lp = logdensityof(K(z = 0.5), z)";
    let err = determinize(&parse_infer(src)).expect_err("the point and the boundary collide");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("boundary input"),
        "the refusal must name the boundary collision; got: {msg}"
    );
}
