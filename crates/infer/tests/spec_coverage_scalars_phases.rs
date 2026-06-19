//! Spec-coverage golden tests: §03 scalar types, §04 phases, §07 norms, §11 %meta.
//!
//! Each test is self-contained. Assertions are grounded against actual rendered
//! output, not guessed (see task brief §CRITICAL METHOD).

use flatppl_infer::infer;

fn ir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = infer(&mut m);
    flatppl_flatpir::write(&m)
}

fn diags(src: &str) -> Vec<flatppl_infer::Diagnostic> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m)
}

// ---- §03 Scalar types ----

/// Spec §03: every comparison operator produces a boolean scalar.
/// `>`, `>=`, `!=`, and `(…) ==` each render with `(%scalar boolean)` in their
/// %meta type slot.
#[test]
fn comparison_ops_return_boolean() {
    // `1 > 2` → lowers to `gt`
    let out = ir("x = 1 > 2");
    let line = out
        .lines()
        .find(|l| l.contains("gt"))
        .expect("gt in output");
    assert!(
        line.contains("(%scalar boolean)"),
        "gt must be boolean; got: {line}"
    );

    // `1.0 >= 1.0` → lowers to `ge`
    let out = ir("x = 1.0 >= 1.0");
    let line = out
        .lines()
        .find(|l| l.contains("ge"))
        .expect("ge in output");
    assert!(
        line.contains("(%scalar boolean)"),
        "ge must be boolean; got: {line}"
    );

    // `1 != 2` → lowers to `unequal`
    let out = ir("x = 1 != 2");
    let line = out
        .lines()
        .find(|l| l.contains("unequal"))
        .expect("unequal in output");
    assert!(
        line.contains("(%scalar boolean)"),
        "unequal must be boolean; got: {line}"
    );

    // `(1 < 2) == true` → outer node is `equal`, inner is `lt`
    let out = ir("x = (1 < 2) == true");
    let line = out
        .lines()
        .find(|l| l.contains("(equal"))
        .expect("equal in output");
    assert!(
        line.contains("(%scalar boolean)"),
        "equal must be boolean; got: {line}"
    );
}

/// Spec §03: boolean operands in arithmetic are promoted — boolean ⊔ boolean
/// joins to integer (the common integer supertype of `{0,1}`), and integer
/// dominated by real promotes to real.
#[test]
fn boolean_in_arithmetic_promotes_to_integer() {
    // `true + true` → add node type is (%scalar integer)
    let out = ir("x = true + true");
    let line = out
        .lines()
        .find(|l| l.contains("(add"))
        .expect("add in output");
    assert!(
        line.contains("(%scalar integer)"),
        "true+true add must be integer; got: {line}"
    );

    // `3.0 * false` → mul node type is (%scalar real)
    let out = ir("y = 3.0 * false");
    let line = out
        .lines()
        .find(|l| l.contains("(mul"))
        .expect("mul in output");
    assert!(
        line.contains("(%scalar real)"),
        "3.0*false mul must be real; got: {line}"
    );
}

/// Spec §03: logical operators `land`, `lor`, `lnot` produce boolean scalars.
/// (`lxor` is a known deferred gap and is not tested here.)
#[test]
fn logical_ops_return_boolean() {
    let out = ir("x = land(true, false)");
    let line = out
        .lines()
        .find(|l| l.contains("(land"))
        .expect("land in output");
    assert!(
        line.contains("(%scalar boolean)"),
        "land must be boolean; got: {line}"
    );

    let out = ir("x = lor(true, false)");
    let line = out
        .lines()
        .find(|l| l.contains("(lor"))
        .expect("lor in output");
    assert!(
        line.contains("(%scalar boolean)"),
        "lor must be boolean; got: {line}"
    );

    let out = ir("x = lnot(true)");
    let line = out
        .lines()
        .find(|l| l.contains("(lnot"))
        .expect("lnot in output");
    assert!(
        line.contains("(%scalar boolean)"),
        "lnot must be boolean; got: {line}"
    );
}

// ---- §04 Phases ----

/// Spec §04: `external(posintegers)` is phase %fixed, element type (%scalar integer),
/// and its value set is `posintegers`. A derived binding `y = n + 1` inherits %fixed
/// because its only free ancestor is fixed.
#[test]
fn external_is_fixed_with_element_type() {
    let out = ir("n = external(posintegers)\ny = n + 1");

    // The `external` node: scalar integer, fixed, posintegers
    assert!(
        out.contains("(%meta ((%scalar integer) %fixed posintegers) (external posintegers))"),
        "external(posintegers) must be ((%scalar integer) %fixed posintegers); got:\n{out}"
    );

    // `y = n + 1` derives from a fixed ancestor → still %fixed
    let add_line = out
        .lines()
        .find(|l| l.contains("(add") && l.contains("(%ref self n)"))
        .expect("add line with ref n not found");
    assert!(
        add_line.contains("%fixed"),
        "derived add must be %fixed; got: {add_line}"
    );
    assert!(
        add_line.contains("(%scalar integer)"),
        "n + 1 type must be (%scalar integer); got: {add_line}"
    );
}

/// Spec §04 / §07: `rnginit` returns an rngstate; `rand(s, M)` returns a tuple
/// of the variate type and a fresh rngstate. The tuple renders as
/// `(%tuple (%scalar real) %rngstate)`.
#[test]
fn rand_returns_variate_and_rngstate() {
    let out = ir("s = rnginit(42)\nr = rand(s, Normal(0.0, 1.0))");

    // rnginit node
    assert!(
        out.contains("(%meta (%rngstate %fixed rngstates) (rnginit 42))"),
        "rnginit must be (%rngstate %fixed rngstates); got:\n{out}"
    );

    // rand node: tuple of (scalar real, rngstate)
    let rand_line = out
        .lines()
        .find(|l| l.contains("(rand"))
        .expect("rand line not found");
    assert!(
        rand_line.contains("(%tuple (%scalar real) %rngstate)"),
        "rand must return (%tuple (%scalar real) %rngstate); got: {rand_line}"
    );
}

// ---- §07 Norms and normalization ----

/// Spec §07: `logsoftmax` maps a real array to a real array of the same shape —
/// its value-set is NOT the standard simplex (unlike `softmax`, which lands on
/// the simplex). Both are rank-1 real arrays; only the value-set differs.
#[test]
fn logsoftmax_is_real_array_not_simplex() {
    let out_ls = ir("y = logsoftmax([0.0, 1.0, 2.0])");
    let line_ls = out_ls
        .lines()
        .find(|l| l.contains("(logsoftmax"))
        .expect("logsoftmax in output");

    // Type: rank-1 array of scalar real
    assert!(
        line_ls.contains("(%array 1 (3) (%scalar real))"),
        "logsoftmax must be a rank-1 real array; got: {line_ls}"
    );
    // Value-set: NOT the standard simplex
    assert!(
        !line_ls.contains("stdsimplex"),
        "logsoftmax value-set must not be a simplex; got: {line_ls}"
    );

    // Confirm softmax IS on the simplex (contrast)
    let out_sm = ir("z = softmax([0.0, 1.0, 2.0])");
    let line_sm = out_sm
        .lines()
        .find(|l| l.contains("(softmax"))
        .expect("softmax in output");
    assert!(
        line_sm.contains("stdsimplex"),
        "softmax must be on the simplex (contrast check); got: {line_sm}"
    );
}

/// Spec §07: `l2unit` maps a real array to a rank-1 real array — its value-set
/// is a Cartesian power of reals, not the standard simplex.
#[test]
fn l2unit_is_real_array_not_simplex() {
    let out = ir("y = l2unit([3.0, 4.0])");
    let line = out
        .lines()
        .find(|l| l.contains("(l2unit"))
        .expect("l2unit in output");

    // Type: rank-1 array of scalar real
    assert!(
        line.contains("(%array 1 (2) (%scalar real))"),
        "l2unit must be a rank-1 real array; got: {line}"
    );
    // Value-set: not the simplex
    assert!(
        !line.contains("stdsimplex"),
        "l2unit value-set must not be a simplex; got: {line}"
    );
}

// ---- §11 %meta annotations ----

/// Spec §11: `%meta` wrappers are never nested — the rendered FlatPIR must not
/// contain the substring `%meta (%meta`, even when the IR chains multiple
/// annotated bindings.
#[test]
fn meta_never_nests() {
    let src = "a = elementof(reals)\nb ~ Normal(a,1.0)\nc = b + 1.0";
    let out = ir(src);
    assert!(
        !out.contains("%meta (%meta"),
        "rendered FlatPIR must not contain nested %meta; got:\n{out}"
    );
}

/// Spec §11: when `functionof` wraps a concrete measure body (e.g. `Normal`)
/// with no explicit boundary, its value-set slot in the %meta triple is
/// `%unknown` — not `%deferred`. The kernel itself IS fully typed (not deferred).
#[test]
fn unknown_not_deferred_on_callable_valueset() {
    let src = "f = functionof(Normal(0.0, 1.0))";
    let out = ir(src);
    let d = diags(src);

    // No errors
    assert!(
        d.iter()
            .all(|d| d.severity != flatppl_infer::Severity::Error),
        "functionof(Normal) must not error; got: {d:?}"
    );

    let line = out
        .lines()
        .find(|l| l.contains("(functionof"))
        .expect("functionof line not found");

    // Value-set is %unknown (not %deferred)
    assert!(
        line.contains("%unknown"),
        "functionof value-set must be %unknown; got: {line}"
    );
    assert!(
        !line.contains("%deferred"),
        "functionof must not be %deferred; got: {line}"
    );
}
