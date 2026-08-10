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
use flatppl_determinizer::determinize;

/// Does NOT assert `infer` is clean. The subject here is the DETERMINISER's binding
/// decision, and `infer`'s verdict on a name mismatch is actively changing: the
/// `infer-always-splat` branch (`TODO-flatppl-rust.md`, "RESOLVED and ENFORCED
/// (flatppl-design#74 …)") makes a mismatched splat a static error in inference, tables
/// included. These tests must keep passing across that landing, so the silence is pinned
/// once, deliberately, in [`infer_is_currently_silent_on_a_mismatched_splat`] instead of
/// being an incidental precondition of every case.
fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn pir(src: &str) -> String {
    let out = determinize(&parse_infer(src)).expect("must lower, not refuse");
    flatppl_flatpir::write(&out)
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

/// The direction that was accepts-invalid: column `a` against an input named `zz`. §04 makes
/// this a static error, and the whole-table bind previously produced a plausible number for
/// it. `infer` reports nothing, so the determiniser is the gate.
#[test]
fn a_mismatched_table_column_refuses_instead_of_binding_the_table_whole() {
    let reason = refusal(
        "\
xs = elementof(cartpow(reals, 4))
g = functionof(sum(_p_), zz = _p_)
z = g(table(a = xs))",
    );
    assert!(
        reason.contains("residual user call"),
        "the unreduced application refuses at the FlatPDL exit check: {reason}"
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

/// Why the determiniser is the gate today: `infer` reports NOTHING for a mismatched splat,
/// even though §04 makes it a static error.
///
/// **Invert or delete this test when `infer-always-splat` lands** — that branch makes the
/// mismatch a static error in inference, at which point this assertion is the one that
/// should fail, pointing a reader at the TODO entry rather than at a mystery.
#[test]
fn infer_is_currently_silent_on_a_mismatched_splat() {
    let mut m = flatppl_syntax::parse(
        "\
xs = elementof(cartpow(reals, 4))
g = functionof(sum(_p_), zz = _p_)
z = g(table(a = xs))",
    )
    .unwrap();
    let diags = flatppl_infer::infer(&mut m);
    assert!(
        diags.is_empty(),
        "if inference now diagnoses this, the determiniser is no longer the only gate — \
         update this test and the TODO entry: {diags:?}"
    );
}

/// A mismatched RECORD field still refuses, as it did before — the guard was already correct
/// for records and must stay so.
#[test]
fn a_mismatched_record_field_still_refuses() {
    let reason = refusal(
        "\
r0 = record(aa = 1.5)
g = functionof(_p_ + 1.0, zz = _p_)
z = g(r0)",
    );
    assert!(
        reason.contains("residual user call"),
        "the unreduced application refuses: {reason}"
    );
}
