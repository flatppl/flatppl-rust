//! A §07 collection-domain head under a `broadcast` refuses instead of emitting the
//! whole-array reduction.
//!
//! The bug this file closes: `Emitter::lower_broadcast` handed any non-density head
//! straight to `ops::lower_builtin`, which lowers the head's WHOLE-ARRAY form. The
//! `broadcast` wrapper was discarded, so `sum.(v)` emitted exactly the undotted
//! `sum(v)`'s reduce and answered with a NUMBER at exit 0 — the worst failure mode this
//! emitter has, since refuse-don't-mislower is the whole discipline. Witnessed at
//! `9701877` over `v = elementof(cartpow(reals, [4]))`:
//!
//! ```text
//! $ flatppl infer     # y = sum.(v)
//! (%bind y (%meta (%deferred %parameterized %unknown) (broadcast sum (%ref self v))))
//! $ flatppl stablehlo
//! %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [0]
//!      : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
//! ```
//!
//! `sum.(M)` over a `[2, 3]` matrix was worse — two reduces contracting it to a scalar.
//! Both of those are now refused by `flatppl_infer` before the emitter is reached (§07
//! denies every one of these heads a scalar, and §04 applies the head per element), so
//! the emitter witness below uses the ONE spelling that types clean: §03's vector of
//! vectors, whose cells really are arrays. At `9701877` that answered with a single
//! scalar too, when the correct answer is the vector of per-inner-vector sums.
//!
//! No golden is added. Every test here asserts a REFUSAL, so there is no `.mlir` to
//! execute and no number to oracle.

use flatppl_core::Module;

fn parse_infer(src: &str) -> Module {
    let mut m = flatppl_syntax::parse(src).expect("parse");
    let diags = flatppl_infer::infer(&mut m);
    assert!(diags.is_empty(), "infer diagnostics: {diags:?}");
    m
}

fn determinize_abi(src: &str) -> Module {
    let mut m = parse_infer(src);
    let syms: Vec<flatppl_core::Symbol> =
        ["inputs", "outputs"].iter().map(|r| m.intern(r)).collect();
    flatppl_determinizer::determinize_with_roots(
        &m,
        &flatppl_infer::ModuleBundle::new(),
        Some(&syms),
    )
    .expect("must determinize, not refuse")
}

fn emit_err(src: &str) -> String {
    flatppl_stablehlo::emit(
        &determinize_abi(src),
        flatppl_stablehlo::Mode::LogDensity,
        &flatppl_stablehlo::EmitOptions::default(),
    )
    .expect_err("must refuse in the emitter")
    .msg
}

/// A vector of vectors, whose cells are arrays — the one operand shape for which a
/// dotted reduction is well-typed, so the only one that reaches the emitter.
fn nested_src(head: &str, set: &str) -> String {
    format!(
        "a = elementof(cartpow({set}, [3]))\nb = elementof(cartpow({set}, [3]))\n\
         vv = [a, b]\ny = {head}.(vv)\ninputs = (a, b)\noutputs = (y)\n"
    )
}

/// The witness. `sum.(vv)` is `[sum(a), sum(b)]`, a `[2]` vector — at `9701877` this
/// emitted a single `tensor<f32>` scalar, exit 0.
#[test]
fn a_dotted_sum_over_nested_arrays_refuses_instead_of_reducing_the_whole_thing() {
    let err = emit_err(&nested_src("sum", "reals"));
    assert!(
        err.contains("`sum` under a broadcast has no tensor form"),
        "must refuse the broadcast, not lower the bare reduction: {err}"
    );
    assert!(
        err.contains("§04 \"Broadcasting\"") && err.contains("§07 \"Reductions\""),
        "must cite the per-element rule and the head's domain: {err}"
    );
    assert!(
        err.contains("`sum(v)`") && err.contains("aggregate(sum, [.i]"),
        "must name the whole-array reduction and the per-axis one: {err}"
    );
    // A norm is not one of §04's ten eligible built-ins, so its remedy must not offer
    // an `aggregate` the reader would only be refused for.
    let norm = emit_err(&nested_src("l2norm", "reals"));
    assert!(
        norm.contains("`l2norm(v)`") && !norm.contains("aggregate"),
        "an ineligible head must not be sent to `aggregate`: {norm}"
    );
}

/// The whole family, not one head. Each of these lowered to its undotted form's value.
#[test]
fn every_collection_domain_head_refuses_under_a_broadcast() {
    for head in [
        "sum",
        "mean",
        "var",
        "std",
        "prod",
        "maximum",
        "minimum",
        "l1norm",
        "l2norm",
        "linfnorm",
        "logsumexp",
        "lengthof",
    ] {
        let err = emit_err(&nested_src(head, "reals"));
        assert!(
            err.contains(&format!("`{head}` under a broadcast has no tensor form")),
            "`{head}.(vv)` must refuse: {err}"
        );
    }
    for head in ["lany", "lall"] {
        let err = emit_err(&nested_src(head, "booleans"));
        assert!(
            err.contains(&format!("`{head}` under a broadcast has no tensor form")),
            "`{head}.(bb)` must refuse: {err}"
        );
    }
}

/// The §07 citation follows the head into the message, so a reader is sent to the table
/// that actually governs it.
#[test]
fn the_refusal_cites_the_heads_own_section() {
    assert!(
        emit_err(&nested_src("lany", "booleans")).contains("§07 \"Boolean reductions\""),
        "`lany` belongs to §07's boolean-reduction table"
    );
    assert!(
        emit_err(&nested_src("l2norm", "reals")).contains("§07 \"Norms and normalization\""),
        "`l2norm` belongs to §07's norms table"
    );
}

/// The gate must not spill onto the elementwise broadcasts, which are the emitter's
/// bread and butter — `.+` and the dotted comparisons all route through the same
/// `lower_broadcast` arm this refusal was added to.
#[test]
fn elementwise_broadcasts_still_emit() {
    for src in [
        "v = elementof(cartpow(reals, [4]))\ny = sum(v .+ 1.0)\ninputs = (v)\noutputs = (y)\n",
        "v = elementof(cartpow(reals, [4]))\ny = sum(exp.(v))\ninputs = (v)\noutputs = (y)\n",
        "v = elementof(cartpow(reals, [4]))\nb = v .> 3.0\ny = ifelse(lany(b), 1.0, 2.0)\n\
         inputs = (v)\noutputs = (y)\n",
    ] {
        let out = flatppl_stablehlo::emit(
            &determinize_abi(src),
            flatppl_stablehlo::Mode::LogDensity,
            &flatppl_stablehlo::EmitOptions::default(),
        );
        assert!(out.is_ok(), "must still emit: {src}\n{:?}", out.err());
    }
}
