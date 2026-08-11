//! A sole positional `table(...)` auto-splats its COLUMNS, exactly as a `record(...)`
//! splats its fields.
//!
//! §04 "Calling conventions" names both in one breath: "`f(record(a = x, b = y, ...))` and
//! `f(table(a = x, b = y, ...))` are equivalent to `f(a = x, b = y, ...)`. The order of
//! fields or columns is not relevant." The amendment (flatppl-design#74, merged) makes the
//! splat unconditional: "A sole positional record or table therefore always splats: whether
//! its field or column names match the callable's argument names decides only whether the
//! call is valid, never whether the splat occurs."
//!
//! `reduce_kernel_application` recognised `record` only, so a sole positional TABLE fell
//! through to the POSITIONAL arm — where a single-input reification bound the whole table to
//! that one input. That is the whole-value-when-names-mismatch reading #74 rules out, and it
//! was wrong in both directions: a MATCHING table refused (a capability §04 grants), and a
//! MISMATCHING one lowered (a call §04 makes a static error).
//!
//! flatppl-design#78 (OPEN, owner-accepted, PENDING owner review) exempts "a callable with
//! exactly one input whose documented domain admits records or tables", so `sum(t)` and
//! `lengthof(t)` reduce over a table rather than splatting. It cannot reach this site: the
//! callee here is always a user `functionof`/`kernelof` reification, which has no documented
//! domain — §07's "Domains" column covers built-ins. The exemption is decided by the
//! callee's arity AND declared domain, so a single-input USER reification stays splatting.
//! `crates/infer` reached the same conclusion independently in #144
//! (`a_user_callable_is_never_exempt_from_the_splat`).
//!
//! **Division of labour with #144.** Inference now rejects a MISMATCHED splat, so the
//! determiniser is no longer the gate for that direction and this file does not re-assert the
//! diagnostic — `crates/infer/tests/arity.rs` owns it. What remains uniquely the
//! determiniser's, and what these tests are for, is the MATCHING direction: performing the
//! splat so the body actually reduces. Verified still load-bearing on this base by reverting
//! `kernel.rs` to d6dfe31, which reddens every matching test below.
use flatppl_determinizer::determinize;

/// Does NOT assert `infer` is clean, deliberately: `refusal` below expects a module inference
/// has already flagged (#144 marks a mismatched splat `Type::Failed`), so asserting clean
/// diagnostics would contradict the very cases this file covers.
fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn pir(src: &str) -> String {
    let out = determinize(&parse_infer(src)).expect("must lower, not refuse");
    flatppl_flatpir::write(&out)
}

/// The determinised module printed as SURFACE syntax. Used where the assertion is about a
/// synthesized column access, which reads as `t.a` here but as
/// `(get (%ref self t) "a")` wrapped in `%meta` in FlatPIR — the surface form states the
/// intent without the assertion having to tolerate meta nesting.
fn printed(src: &str) -> String {
    let out = determinize(&parse_infer(src)).expect("must lower, not refuse");
    flatppl_syntax::print(&out)
}

fn refusal(src: &str) -> String {
    determinize(&parse_infer(src))
        .expect_err("a name mismatch is a §04 static error — refuse, never bind the whole value")
        .reason
}

/// The capability §04 grants and the record-only check withheld: each column binds the
/// like-named input, so the body reduces to `sum(xs) + sum(ys)` with no residual call.
#[test]
fn a_matching_table_literal_splats_its_columns() {
    let text = pir("\
xs = elementof(cartpow(reals, 4))
ys = elementof(cartpow(reals, 4))
g = functionof(sum(_p_) + sum(_r_), a = _p_, b = _r_)
z = g(table(a = xs, b = ys))");
    assert!(
        text.contains("(sum (%ref self xs))") && text.contains("(sum (%ref self ys))"),
        "each column binds its like-named input:\n{text}"
    );
}

/// A SINGLE-input reification splats too — it does not receive the table whole. This is the
/// #74 ruling at the arity #78's exemption talks about, and the case that separates the two:
/// a user reification has no documented domain, so it is not exempt.
#[test]
fn a_single_input_reification_splats_rather_than_taking_the_table_whole() {
    let text = pir("\
xs = elementof(cartpow(reals, 4))
g = functionof(sum(_p_), a = _p_)
z = g(table(a = xs))");
    assert!(
        text.contains("(sum (%ref self xs))"),
        "the sole column binds the sole input by NAME:\n{text}"
    );
    // Binding the table whole would have left the table node under `sum` instead.
    assert!(
        !text.contains("(sum (table"),
        "the table must not reach `sum` whole:\n{text}"
    );
}

/// The direction that was accepts-invalid — column `a` against an input named `zz`, which
/// used to bind the whole table and produce a plausible number. It now refuses, but the
/// refusal has MOVED LAYER: #144 marks the call `Type::Failed` in inference, and the
/// determiniser refuses on the residual failed node rather than through `record_field`'s
/// `None`. So this asserts the end-to-end verdict and names the layer, which is the part no
/// single-crate test covers.
///
/// The infer-side goldens own the diagnostic itself — `arity.rs`'s
/// `a_sole_positional_table_splats_by_column_name` pins this exact message and its matching
/// counterpart, and `a_user_callable_is_never_exempt_from_the_splat` pins the reasoning
/// `is_splattable` relies on. Nothing here re-asserts either.
///
/// The determiniser's own mismatch arm is consequently a BACKSTOP, not the gate: it is
/// unreachable through surface syntax while inference flags the call first. It still matters
/// for a hand-built or FlatPIR-loaded module that carries no such diagnostic.
#[test]
fn a_mismatched_splat_refuses_though_inference_now_catches_it_first() {
    let reason = refusal(
        "\
xs = elementof(cartpow(reals, 4))
g = functionof(sum(_p_), zz = _p_)
z = g(table(a = xs))",
    );
    assert!(
        reason.contains("has no parameter `a`"),
        "the refusal is inference's §04 name check, surfaced through the failed node: {reason}"
    );
}

/// Column ORDER is not relevant (§04), so the same call with the columns transposed reduces
/// to the same body. A positional fallback would silently swap the two operands.
#[test]
fn table_column_order_does_not_matter() {
    let src = |cols: &str| {
        format!(
            "\
xs = elementof(cartpow(reals, 4))
ys = elementof(cartpow(reals, 4))
g = functionof(sum(_p_) - sum(_r_), a = _p_, b = _r_)
z = g(table({cols}))"
        )
    };
    let a_first = pir(&src("a = xs, b = ys"));
    let b_first = pir(&src("b = ys, a = xs"));
    assert!(
        a_first.contains("(sub (%meta ((%scalar real) %parameterized reals) (sum (%ref self xs)))"),
        "`a` binds the minuend:\n{a_first}"
    );
    assert_eq!(
        a_first, b_first,
        "column order must not change the reduction"
    );
}

/// The record path this shares its implementation with is unchanged — a matching record
/// literal still splats, including at a single input (`k(record(mu = 1.5))`).
#[test]
fn a_matching_record_literal_still_splats_at_a_single_input() {
    let text = pir("\
mk = kernelof(draw(Normal(mu = _mm_, sigma = 1.0)), mu = _mm_)
lp = logdensityof(lawof(record(y = draw(mk(record(mu = 1.5))))), record(y = 0.5))");
    assert!(
        text.contains("(%field mu 1.5)"),
        "the record's `mu` field binds the kernel's `mu` input:\n{text}"
    );
}

// Two tests stood here and are DELETED rather than adapted, because #144 moved what they
// asserted into inference and `crates/infer/tests/arity.rs` already owns it:
//
// - `infer_is_currently_silent_on_a_mismatched_splat` was written to fail by design once the
//   splat landed. It has done its job: inference now reports "`g` has no parameter `a`" with
//   the splat hint. Re-asserting that here would duplicate
//   `a_sole_positional_table_splats_by_column_name`.
// - `a_mismatched_record_field_still_refuses` asserted the RECORD mismatch refuses through the
//   FlatPDL exit check. Same story — inference flags it first now, and
//   `a_single_parameter_constructor_name_checks_a_splatted_record` plus
//   `a_record_that_does_not_splat_is_not_name_checked` cover the record side.
//
// The end-to-end verdict and the layer handoff survive in
// `a_mismatched_splat_refuses_though_inference_now_catches_it_first` above. What this file
// still uniquely covers is the MATCHING direction — the capability only the determiniser can
// provide, verified still load-bearing on this base by reverting `kernel.rs` to d6dfe31 and
// watching all three matching tests redden.

// ---- opaque tables ---------------------------------------------------------------
//
// D6 left this open and said so: "An OPAQUE table — a `load_data` result rather than a
// `table(...)` literal — is still bound whole", because `splat_head` needs a syntactic
// `table(...)` node to destructure and a `load_data` result has none. §04 draws no such
// distinction, and §13 `sec:determinization-signature` makes a `load_data`'s shape come from
// its declared `valueset`, so its columns are as statically known as a literal's — they are in
// the inferred type, `(%table (%columns (a …) (b …)) (%nrows 4))`.
//
// Two things changed since D6, in different layers. #144 made inference reject the MISMATCH
// direction from the type, which closed D6's headline symptom (the call no longer lowers with
// the table bound whole). But that left the MATCHING direction — a §04-legal call — passing
// inference and then refusing here with the generic "residual user call", while its literal
// twin lowered. These tests cover the matching direction, which is the half that is uniquely
// the determiniser's: `record_field` now synthesizes the column access §03 defines
// (`t.a` → `get(t, "a")`) so the splat is constructible from an opaque value.

/// The opaque twin of `a_matching_table_literal_splats_its_columns`, asserted to reduce to the
/// same shape: each column binds its like-named input and no residual call survives. The
/// column values arrive as `get`, not as the literal's inline nodes, which is the only
/// difference §03's column access permits.
#[test]
fn a_matching_opaque_table_splats_its_columns() {
    let src = "\
t = load_data(\"x.csv\", cartpow(cartprod(a = reals, b = reals), 4))
g = functionof(sum(_p_) + sum(_r_), a = _p_, b = _r_)
z = g(t)";
    let text = printed(src);
    assert!(
        text.contains("z = sum(t.a) + sum(t.b)"),
        "each column binds its like-named input as a column access:\n{text}"
    );
    assert!(
        !pir(src).contains("%call"),
        "no residual user call may survive the splat:\n{}",
        pir(src)
    );
}

/// §04: "The order of fields or columns is not relevant." An opaque table whose columns are
/// declared in the opposite order to the callee's inputs binds by NAME and reduces
/// identically — the same property the literal case pins, now for the type-derived columns.
#[test]
fn an_opaque_table_splats_by_name_not_by_column_order() {
    let forward = printed(
        "\
t = load_data(\"x.csv\", cartpow(cartprod(a = reals, b = reals), 4))
g = functionof(sum(_p_) + sum(_r_), a = _p_, b = _r_)
z = g(t)",
    );
    let reversed = printed(
        "\
t = load_data(\"x.csv\", cartpow(cartprod(b = reals, a = reals), 4))
g = functionof(sum(_p_) + sum(_r_), a = _p_, b = _r_)
z = g(t)",
    );
    for text in [&forward, &reversed] {
        assert!(
            text.contains("z = sum(t.a) + sum(t.b)"),
            "column order must not change which input a column binds:\n{text}"
        );
    }
}

/// The D6 repro itself: a two-column opaque table against a ONE-input reification. D6
/// measured this as lowering with the table bound whole (`(lengthof (%ref self t))`). It is
/// now a static error, caught by inference from the type — so the determiniser never sees a
/// well-formed module and this asserts the diagnostic rather than a refusal reason.
///
/// Kept here rather than only in `crates/infer/tests/arity.rs` because the *opaque* table is
/// what D6 flagged, and this is the case that would silently regress if `table_columns` ever
/// stopped reading the type.
#[test]
fn the_d6_opaque_table_repro_is_a_static_error_not_a_whole_value_bind() {
    let mut m = flatppl_syntax::parse(
        "t = load_data(\"x.csv\", cartpow(cartprod(a = reals, b = reals), 4))\n\
         g = functionof(lengthof(_q_), tt = _q_)\n\
         z = g(t)",
    )
    .unwrap();
    let errors: Vec<String> = flatppl_infer::infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .map(|d| d.message)
        .collect();
    assert!(
        errors
            .iter()
            .any(|e| e.contains("`g` declares 1 parameter, got 2 arguments")),
        "the two columns splat onto one input, which is a §04 static error, got: {errors:?}"
    );
    // And the old symptom is gone: nothing binds the table whole.
    assert!(
        !flatppl_flatpir::write(&m).contains("(lengthof (%ref self t))"),
        "the table must not be bound whole to the single input"
    );
}

/// PERMISSIVE where the type does not say: a sole positional argument that is NOT a table
/// keeps the positional binding it had, so nothing is refused or splatted on a guess.
/// `table_columns` keys on `Type::Table`, so a vector — or any value whose type is deferred —
/// simply is not splattable.
#[test]
fn a_non_table_sole_argument_still_binds_positionally() {
    let text = pir("\
v = elementof(cartpow(reals, 4))
g = functionof(sum(_q_), q = _q_)
z = g(v)");
    assert!(
        text.contains("(sum (%ref self v))"),
        "a vector argument binds positionally, unsplatted:\n{text}"
    );
}
