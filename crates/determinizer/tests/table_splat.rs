//! Table application follows the callable interface, including opaque column types.
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
        a_first.contains("(sub (%meta ((%scalar real) %parametric reals) (sum (%ref self xs)))"),
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
    // The body is NON-COMMUTATIVE (`-`, not `+`) and the second module declares its columns
    // in the REVERSED order. A commutative body would pass either way round, so it could not
    // tell name binding from declaration order; subtraction can.
    let forward = printed(
        "\
t = load_data(\"x.csv\", cartpow(cartprod(a = reals, b = reals), 4))
g = functionof(sum(_p_) - sum(_r_), a = _p_, b = _r_)
z = g(t)",
    );
    let reversed = printed(
        "\
t = load_data(\"x.csv\", cartpow(cartprod(b = reals, a = reals), 4))
g = functionof(sum(_p_) - sum(_r_), a = _p_, b = _r_)
z = g(t)",
    );
    for text in [&forward, &reversed] {
        assert!(
            text.contains("z = sum(t.a) - sum(t.b)"),
            "column `a` must bind input `a` whatever order the columns are declared in:\n{text}"
        );
    }
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

#[test]
fn unary_interfaces_keep_literal_and_opaque_tables_whole() {
    for value in [
        "table(a = [1.0, 2.0])",
        "load_data(\"x.csv\", cartpow(cartprod(a = reals, b = reals), 4))",
    ] {
        let source = format!("t = {value}\ng(r) = sum(r)\nz = g(t)");
        let mut module = flatppl_syntax::parse(&source).unwrap();
        assert!(flatppl_infer::infer(&mut module).is_empty());
        let text = printed(&source);
        assert!(
            text.contains("z = sum(t)") || text.contains("z = sum(table("),
            "{text}"
        );
    }
}

#[test]
fn a_multi_input_table_must_name_its_inputs() {
    let mut module =
        flatppl_syntax::parse("g(a,b)=sum(a)+sum(b)\nz=g(table(a=[1.0],wrong=[2.0]))").unwrap();
    assert!(
        flatppl_infer::infer(&mut module)
            .iter()
            .any(|d| d.severity == flatppl_infer::Severity::Error)
    );
}
