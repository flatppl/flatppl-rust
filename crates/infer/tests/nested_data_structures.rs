//! Nested data structures (spec §03/§04, flatppl-design commit ee232b4):
//! records may contain records, tuples may nest, and table columns may be
//! tables (or vectors whose elements are arrays). These exercise the inference
//! side — types and value-sets — through the annotated-FlatPIR rendering.
//!
//! Stimuli are inlined (NOT added to `fixtures/flatppl/`, the cross-engine
//! corpus) so they don't run through a flatppl-js that has not yet landed the
//! same change — mirroring `syntax/tests/roundtrip.rs::full_syntax_wraps_*`.

use flatppl_infer::{Severity, infer};

fn ir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = infer(&mut m);
    flatppl_flatpir::write(&m)
}

/// Inference error messages for `src` (spec §03 well-formedness diagnostics).
fn errors(src: &str) -> Vec<String> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
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

/// A tuple may contain another tuple (spec §04 — tuples nest): the type nests.
/// A tuple has NO value-set (spec §04: tuples are objects, not values), so the
/// value-set slot is `%unknown` — no `cartprod` leaks onto a tuple.
#[test]
fn nested_tuple_type_and_valueset() {
    let out = ir("p = tuple(tuple(1.0, 2), true)");
    assert!(
        out.contains("(%tuple (%tuple (%scalar real) (%scalar integer)) (%scalar boolean))"),
        "tuple-of-tuple should nest in the type, got:\n{out}"
    );
    assert!(
        out.contains(
            "(%tuple (%tuple (%scalar real) (%scalar integer)) (%scalar boolean)) %fixed %unknown)"
        ),
        "a tuple has no value-set (§04) — expect %unknown, no cartprod; got:\n{out}"
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
        out.contains("(cartpow (record (a reals) (b (cartpow reals 3))) 2)"),
        "table value-set is cartpow(rowRecord, nrows); got:\n{out}"
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
        out.contains("(cartpow (record (id integers) (hits (record (x reals) (y reals)))) 2)"),
        "table value-set is cartpow(rowRecord, nrows), row nests a record-of-records; got:\n{out}"
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

// ── §03 well-formedness: equal column lengths + no objects-in-arrays ──────────

/// All table columns must have the same length (spec §03). table_type took the
/// row count from the first column only; a table-valued column made this
/// concretely wrong (its sub-table nrows became the parent nrows). A length
/// mismatch must now be reported, not silently accepted. (Regression for the
/// F2 finding of the nested-data review.)
#[test]
fn table_unequal_column_lengths_is_an_error() {
    let errs = errors("t = table(hits = table(x = [1.0, 2.0, 3.0, 4.0, 5.0]), id = [10, 20])");
    assert!(
        errs.iter()
            .any(|m| m.contains("equal length") || m.contains("rows")),
        "unequal column lengths must be reported, got: {errs:?}"
    );
}

/// A plain (non-nested) length mismatch is reported too — the rule is general.
#[test]
fn table_unequal_plain_columns_is_an_error() {
    let errs = errors("t = table(a = [1.0, 2.0], b = [1.0, 2.0, 3.0])");
    assert!(
        errs.iter()
            .any(|m| m.contains("equal length") || m.contains("rows")),
        "unequal plain column lengths must be reported, got: {errs:?}"
    );
}

/// An array element must be a scalar, string, or array — never a record (spec
/// §03). A vector-of-records would otherwise masquerade as a table-valued
/// column (indistinguishable types, phantom sub-table on column access). It
/// must be reported at construction. (Regression for the F1 finding.)
#[test]
fn array_of_records_is_rejected() {
    let errs = errors("c = [record(x = 1.0), record(x = 2.0)]");
    assert!(
        errs.iter().any(|m| m.contains("array element")),
        "an array of records must be reported, got: {errs:?}"
    );
    // The same rejection applies inside a table column position.
    let errs = errors("t = table(pts = [record(x = 1.0), record(x = 2.0)])");
    assert!(
        errs.iter().any(|m| m.contains("array element")),
        "a vector-of-records column must be reported, got: {errs:?}"
    );
}

/// Measures / tuples are likewise forbidden as array elements (spec §02/§03).
#[test]
fn array_of_objects_is_rejected() {
    let errs = errors("m = Normal(mu = 0.0, sigma = 1.0)\na = [m, m]");
    assert!(
        errs.iter().any(|m| m.contains("array element")),
        "an array of measures must be reported, got: {errs:?}"
    );
}

/// Strings and arrays remain valid array elements (selector vectors, vec-of-vec).
#[test]
fn array_of_strings_and_arrays_is_ok() {
    assert!(
        errors("names = [\"a\", \"b\", \"c\"]").is_empty(),
        "a string vector must be accepted"
    );
    assert!(
        errors("m = [[1.0, 2.0], [3.0, 4.0]]").is_empty(),
        "a vector-of-vectors must be accepted"
    );
}

/// With equal columns, a nested table reconstructs correctly at a row count
/// OTHER than 2 (the earlier tests all used nrows = 2, masking F1/F2): the
/// outer table, the sub-table from column access, and a scalar column vector
/// all carry nrows = 3.
#[test]
fn nested_table_reconstruction_with_distinct_nrows() {
    let out = ir(
        "t = table(id = [1, 2, 3], hits = table(x = [1.0, 2.0, 3.0]))\nsub = t.hits\ncol = t.id",
    );
    assert!(
        out.contains(
            "(%table (%columns (id (%scalar integer)) (hits (%record (x (%scalar real))))) (%nrows 3))"
        ),
        "outer table should be 3 rows, got:\n{out}"
    );
    assert!(
        out.contains("(%bind sub (%meta ((%table (%columns (x (%scalar real))) (%nrows 3))"),
        "t.hits should reconstruct a 3-row sub-table, got:\n{out}"
    );
    assert!(
        out.contains("(%bind col (%meta ((%array 1 (3) (%scalar integer))"),
        "t.id should be a length-3 column vector, got:\n{out}"
    );
}

/// A table-valued column placed FIRST sets the shared row count from its
/// sub-table (spec §03), with the remaining equal-length columns accepted.
#[test]
fn table_valued_column_first_sets_nrows() {
    let out = ir("t = table(hits = table(x = [1.0, 2.0, 3.0]), id = [1, 2, 3])");
    assert!(
        out.contains(
            "(%table (%columns (hits (%record (x (%scalar real)))) (id (%scalar integer))) (%nrows 3))"
        ),
        "a table-valued first column should set nrows = 3, got:\n{out}"
    );
}

// ── §06 mixing shape classes is a static error (cartprod / joint) ─────────────

/// Positional `joint` of components with different shape classes (a scalar and
/// a vector) is a static error (spec §06: "Mixing shape classes is a static
/// error"), not a silently-deferred domain.
#[test]
fn positional_joint_mixing_shape_classes_is_an_error() {
    let errs =
        errors("j = joint(Normal(mu = 0.0, sigma = 1.0), iid(Normal(mu = 0.0, sigma = 1.0), 2))");
    assert!(
        errs.iter().any(|m| m.contains("shape class")),
        "joint of a scalar and a vector measure must be a static error, got: {errs:?}"
    );
}

/// Positional `cartprod` mixing shape classes (a scalar set and a vector set)
/// is a static error too — §03 cartprod mirrors §06 joint; §07 `cat` forbids
/// concatenating a scalar with a vector.
#[test]
fn positional_cartprod_mixing_shape_classes_is_an_error() {
    let errs = errors("p = elementof(cartprod(reals, cartpow(reals, 3)))");
    assert!(
        errs.iter().any(|m| m.contains("shape class")),
        "cartprod of a scalar and a vector set must be a static error, got: {errs:?}"
    );
}

// ── table(r) / record(t) auto-splat duality (spec §03) ───────────────────────

/// `table(r)` builds a table from a record of equal-length vectors (spec §03
/// auto-splat duality).
#[test]
fn table_from_record_splats() {
    let out = ir("r = record(a = [1.0, 2.0], b = [3, 4])\nt = table(r)");
    assert!(
        out.contains(
            "(%bind t (%meta ((%table (%columns (a (%scalar real)) (b (%scalar integer))) (%nrows 2))"
        ),
        "table(record-of-vectors) should build a 2-row table; got:\n{out}"
    );
}

/// `record(t)` is the inverse: a table splats into a record of its column
/// vectors.
#[test]
fn record_from_table_splats() {
    let out = ir("t = table(a = [1.0, 2.0], b = [3, 4])\nr = record(t)");
    assert!(
        out.contains(
            "(%bind r (%meta ((%record (a (%array 1 (2) (%scalar real))) (b (%array 1 (2) (%scalar integer))))"
        ),
        "record(table) should build a record of column vectors; got:\n{out}"
    );
}
