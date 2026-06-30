//! Nested data structures (spec §03/§04, flatppl-design commit ee232b4):
//! records may contain records, tuples may nest, and table columns may be
//! tables (or vectors whose elements are arrays). These exercise the inference
//! side — types and value-sets — through the annotated-FlatPIR rendering.
//!
//! Stimuli are inlined (NOT added to `fixtures/flatppl/`, the cross-engine
//! corpus) so they don't run through a flatppl-js that has not yet landed the
//! same change — mirroring `syntax/tests/roundtrip.rs::full_syntax_wraps_*`.

use flatppl_infer::infer;

fn ir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = infer(&mut m);
    flatppl_flatpir::write(&m)
}

// ── Nested records (record-in-record) ────────────────────────────────────────

/// A record field may itself be a record (spec §03): the type and value-set
/// nest, and dotted access `r.a.b` chains down to the leaf scalar.
#[test]
fn nested_record_type_and_valueset() {
    let out = ir("r = record(a = record(b = 1.0, c = 2), d = 3.0)");
    assert!(
        out.contains(
            "(%record (a (%record (b (%scalar real)) (c (%scalar integer)))) (d (%scalar real)))"
        ),
        "record-of-record should nest in the type, got:\n{out}"
    );
    assert!(
        out.contains("(record (a (record (b reals) (c integers))) (d reals))"),
        "record-of-record should nest in the value-set, got:\n{out}"
    );
}

/// `r.a.b` (nested field chaining, spec §03) resolves to the inner field type.
#[test]
fn nested_record_field_chains() {
    let out = ir("r = record(a = record(b = 1.0))\nx = r.a.b");
    // `x` is the leaf real; the binding for x is annotated scalar real.
    assert!(
        out.contains("(%bind x (%meta ((%scalar real)"),
        "r.a.b should resolve to the inner real, got:\n{out}"
    );
}

// ── Nested tuples (tuple-in-tuple) ───────────────────────────────────────────

/// A tuple may contain another tuple (spec §04 — tuples nest): the type and
/// value-set nest, and `t[1][2]` reaches the inner component.
#[test]
fn nested_tuple_type_and_valueset() {
    let out = ir("p = tuple(tuple(1.0, 2), true)");
    assert!(
        out.contains("(%tuple (%tuple (%scalar real) (%scalar integer)) (%scalar boolean))"),
        "tuple-of-tuple should nest in the type, got:\n{out}"
    );
    assert!(
        out.contains("(cartprod (cartprod reals integers) booleans)"),
        "tuple-of-tuple should nest in the value-set, got:\n{out}"
    );
}

/// `t[1][2]` indexes the inner tuple's second component (spec §04, 1-based).
#[test]
fn nested_tuple_index_chains() {
    let out = ir("p = tuple(tuple(1.0, 2), true)\nx = p[1][2]");
    assert!(
        out.contains("(%bind x (%meta ((%scalar integer)"),
        "p[1][2] should resolve to the inner integer, got:\n{out}"
    );
}

// ── Table columns that are arrays (3-vector per row) ─────────────────────────

/// A vector column whose elements are arrays — a 3-vector per row (spec §03,
/// the §07 `load_data` example). The stored column element is the per-row
/// array, and the row value-set carries that array, not a stripped scalar.
#[test]
fn table_with_array_valued_column() {
    let out = ir("t = table(a = [1.0, 2.0], b = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])");
    assert!(
        out.contains(
            "(%table (%columns (a (%scalar real)) (b (%array 1 (3) (%scalar real)))) (%nrows 2))"
        ),
        "array-valued column should keep its 3-vector element, got:\n{out}"
    );
    assert!(
        out.contains("(record (a reals) (b (cartpow reals 3)))"),
        "row value-set should carry the 3-vector column, got:\n{out}"
    );
}

// ── Table columns that are tables (nested table) ─────────────────────────────

/// A table column may itself be a table (spec §03): the parent stores the
/// sub-table's per-row record as the column element, and the row value-set
/// nests a record-of-records.
#[test]
fn table_with_table_valued_column() {
    let out = ir("t = table(id = [1, 2], hits = table(x = [1.0, 2.0], y = [3.0, 4.0]))");
    assert!(
        out.contains(
            "(%table (%columns (id (%scalar integer)) (hits (%record (x (%scalar real)) (y (%scalar real))))) (%nrows 2))"
        ),
        "table-valued column should store the sub-table row record, got:\n{out}"
    );
    assert!(
        out.contains("(record (id integers) (hits (record (x reals) (y reals))))"),
        "row value-set should nest a record-of-records, got:\n{out}"
    );
}

// ── Table element access (row record, column vector / sub-table) ─────────────

/// Row access `t[i]` yields the row record (spec §03 "Each row of a table is a
/// record"); for a nested-table column the row entry is itself a record.
#[test]
fn table_row_access_is_a_record() {
    let out = ir("t = table(id = [1, 2], hits = table(x = [1.0, 2.0]))\nrow = t[1]");
    assert!(
        out.contains("(id (%scalar integer))")
            && out.contains("(hits (%record (x (%scalar real))))"),
        "t[1] should be a nested row record, got:\n{out}"
    );
    // Chained access through the nested row reaches the leaf.
    let out = ir("t = table(id = [1, 2], hits = table(x = [1.0, 2.0]))\nv = t[1].hits.x");
    assert!(
        out.contains("(%bind v (%meta ((%scalar real)"),
        "t[1].hits.x should resolve to the inner real, got:\n{out}"
    );
}

/// Column access `t.col` returns the column as a vector (spec §03); a
/// table-valued column returns the sub-table.
#[test]
fn table_column_access_vector_or_subtable() {
    let out =
        ir("t = table(a = [1.0, 2.0], hits = table(x = [1.0, 2.0]))\ncol = t.a\nsub = t.hits");
    assert!(
        out.contains("(%bind col (%meta ((%array 1 (2) (%scalar real))"),
        "t.a should be a length-2 real column vector, got:\n{out}"
    );
    assert!(
        out.contains("(%bind sub (%meta ((%table (%columns (x (%scalar real))) (%nrows 2))"),
        "t.hits should be the sub-table, got:\n{out}"
    );
}
