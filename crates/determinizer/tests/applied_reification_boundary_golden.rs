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

/// A nested reification that RE-DECLARES a pinned boundary name refuses, and refusing it
/// is what keeps the cross-naming relaxation honest.
///
/// `substitute_refs_by_name` runs on `driver::map_tree`, which has no shadow guard, so it
/// descends into `h` and overwrites `h`'s OWN input reference with the outer pin —
/// `h(2.0)`'s argument is discarded and `h`'s body folds to a constant. §04 forbids that
/// directly: "The resulting function `h` now has arguments named `a` and `d`, but these
/// are local to the function and decoupled from the original nodes `a` and `d`."
///
/// The error is UNBOUNDED, and it is the cross-naming exclusion that exposed it: a
/// cross-read target is excluded from the residual check by design, so the residual check
/// can no longer be this hazard's backstop. `subtree_capturing_reification_input` refuses
/// it first instead. See `a_nested_reification_redeclaring_a_pinned_name_refuses` for the
/// wrong number this prevents, and the alpha-renamed and lambda controls that must lower.
///
/// The second shape pins the exclusion as per-TARGET rather than a blanket disarming: it
/// holds a cross-named pair (`z = w`) AND a third input `v` whose name `h` re-declares.
/// The refusal must still fire and must name `v`, not the legitimately-present `w`.
#[test]
fn a_nested_reification_declaring_a_pinned_boundary_name_refuses() {
    let redeclared_only = "\
z  = elementof(reals)
h  = functionof(z + 1.0, z = z)
b1 ~ Normal(mu = h(2.0), sigma = 1.0)
K  = kernelof(b1, z = z)
lp = logdensityof(K(z = 0.5), 1.0)";
    let err =
        determinize(&parse_infer(redeclared_only)).expect_err("`h` re-declares the pinned `z`");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("OWN boundary inputs") && msg.contains("`z`"),
        "the refusal must name the nested re-declaration of `z`; got: {msg}"
    );

    let redeclared_beside_a_cross_name = "\
z  = elementof(reals)
w  = elementof(reals)
v  = elementof(reals)
h  = functionof(v + 1.0, v = v)
b1 ~ Normal(mu = z + w + h(2.0), sigma = 1.0)
K  = kernelof(b1, z = z, w = w, v = v)
lp = logdensityof(K(z = w, w = 0.5, v = 3.0), 1.0)";
    let err = determinize(&parse_infer(redeclared_beside_a_cross_name))
        .expect_err("`v` is re-declared even though `w` is legitimately present");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("`v`") && !msg.contains("`w`"),
        "a cross-named sibling must not excuse a re-declared name; got: {msg}"
    );
}

/// The wrong number the nested-re-declaration refusal prevents, pinned from both sides:
/// the hazardous spelling must refuse, and the two spellings that CANNOT capture must
/// lower to the truth.
///
/// `h = functionof(w * 10.0, w = w)` inside the body of a `K(z = w, w = 0.5)` application
/// scored `record(mu = w + 0.5 + 5.0, …)` — `h`'s own input overwritten with the outer pin
/// `0.5`, its argument `2.0` thrown away. At ambient `w = 0`, `y = 1.0` that is
/// `log N(1; 5.5, 1) = -11.043938533204672` where the truth is
/// `log N(1; 20.5, 1) = -191.04393853320468`. The error is unbounded in `h`'s body, not a
/// perturbation.
///
/// The two controls prove the defect was pure NAME capture rather than anything about the
/// nesting, because head produces the truth for both:
///
/// - alpha-renaming the nested input (`functionof(t * 10.0, t = t)`) — §04 makes the two
///   the same function, so the two models must lower identically;
/// - the lambda spelling (`w -> w * 10.0`), where `lower_lambda` mints a `%local`
///   placeholder that cannot collide with a same-module name.
///
/// The `%autoinputs` spelling is the fourth row. It keeps its boundary in the module's
/// side-table rather than inline, so reading `Inputs::Spec` alone missed it: the finish
/// clobbered the auto-input and the reduction then could not bind it, refusing at a later
/// site with a false internal message ("user call declares 0 parameters, got 1" — `g`
/// declares one parameter, `w`). It must reach the same honest refusal as the `%specinputs`
/// spelling.
#[test]
fn a_nested_reification_redeclaring_a_pinned_name_refuses() {
    let hazard = |nested: &str, applied: &str| {
        format!(
            "\
z  = elementof(reals)
w  = elementof(reals)
t  = elementof(reals)
{nested}
b1 ~ Normal(mu = z + w + {applied}, sigma = 1.0)
K  = kernelof(b1, z = z, w = w)
lp = logdensityof(K(z = w, w = 0.5), 1.0)"
        )
    };

    for (nested, applied, what) in [
        ("h  = functionof(w * 10.0, w = w)", "h(2.0)", "%specinputs"),
        ("g  = functionof(w * 10.0)", "g(w = 2.0)", "%autoinputs"),
    ] {
        let src = hazard(nested, applied);
        let err = determinize(&parse_infer(&src)).err().unwrap_or_else(|| {
            panic!("{what}: must refuse, not score at the clobbered `mu = w + 0.5 + 5.0`")
        });
        let msg = format!("{err:?}");
        assert!(
            msg.contains("OWN boundary inputs") && msg.contains("`w`"),
            "{what}: the refusal must name the nested re-declaration of `w`; got: {msg}"
        );
        assert!(
            !msg.contains("declares 0 parameters"),
            "{what}: the refusal must not be the downstream arity message; got: {msg}"
        );
    }

    for (nested, applied, what) in [
        (
            "h  = functionof(t * 10.0, t = t)",
            "h(2.0)",
            "alpha-renamed",
        ),
        ("h  = w -> w * 10.0", "h(2.0)", "lambda"),
    ] {
        let mut m = parse_infer(&hazard(nested, applied));
        let lp = m.intern("lp");
        let out = determinize_with_roots(&m, &ModuleBundle::new(), Some(&[lp]))
            .unwrap_or_else(|e| panic!("{what}: must lower; got {e:?}"));
        let printed = flatppl_syntax::print(&out);
        assert!(
            printed.contains("record(mu = w + 0.5 + 20.0, sigma = 1.0)"),
            "{what}: a nested input that cannot collide keeps its own argument, so `h(2.0)` \
             is 20.0; got:\n{printed}"
        );
    }
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

/// The same hazard through the PRE-EXISTING self-reference exclusion, which was a silent
/// wrong number on `origin/main` independently of the cross-naming relaxation.
///
/// `K(z = z + 1.0)` is self-referential, so `z` was already excluded from the residual
/// check before this wave — and `h = functionof(z * 10.0, z = z)` in the body was clobbered
/// exactly the same way. At `9d6f526` this emitted `record(mu = z + 1.0 + 30.0, …)`, which
/// is `h`'s body evaluated after the finish wrote `z + 1.0` into it and the inline then
/// bound `z := 2.0`: `(2.0 + 1.0) * 10.0`. At ambient `z = 0`, `y = 1.0` that scores
/// `log N(1; 31, 1) = -450.91893853320465` where the truth is
/// `log N(1; 21, 1) = -200.91893853320468`.
///
/// The nested-re-declaration refusal keys on the SUBSTITUTION MAP rather than on which
/// targets the residual check excluded, so it covers this route too and closes a
/// pre-existing hole rather than only containing the new one.
#[test]
fn a_self_referential_application_with_a_redeclared_nested_name_refuses() {
    let src = "\
z  = elementof(reals)
h  = functionof(z * 10.0, z = z)
b1 ~ Normal(mu = z + h(2.0), sigma = 1.0)
K  = kernelof(b1, z = z)
lp = logdensityof(K(z = z + 1.0), 1.0)";
    let err = determinize(&parse_infer(src))
        .expect_err("must refuse, not score at the clobbered `mu = z + 1.0 + 30.0`");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("OWN boundary inputs") && msg.contains("`z`"),
        "the refusal must name the nested re-declaration of `z`; got: {msg}"
    );
}
