//! A sole positional record or table onto a row with UNNAMED variadic inputs is a §04 static
//! error, not a silent positional bind.
//!
//! §04 "Calling conventions": "Special operations have zero to three distinguished,
//! **unnamed**, ordered inputs of fixed arity", then, naming these rows outright:
//!
//! > - `vector`: Unnamed variadic inputs with significant order.
//! > - `tuple`: Unnamed variadic inputs with significant order (minimum two).
//! > - `cat`, `fchain`, `kchain`: Variadic unnamed inputs with significant order.
//! > - `get`: One distinguished input plus variadic unnamed input with significant order.
//! > - `superpose`: Variadic unnamed inputs with no significant order.
//!
//! An unnamed input offers no name for a splatted field to bind to, so NO field or column name
//! can ever match one, and §04's "A call with field or column names that do not match the
//! callable's argument names is a static error" applies to every such call. `get0` inherits via
//! §07's "zero-based variant of `get`".
//!
//! The B76 audit left these as the last three accepting names (`cat`, `get`, `get0`) after the
//! NAMES wave closed the other 42; they bound the columns by ORDER with no diagnostic. Closed
//! here on the owner's ruling — "we do not want hidden magic".

use flatppl_infer::{Diagnostic, Severity, builtin_catalogue, infer};

fn errors(src: &str) -> Vec<String> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m)
        .into_iter()
        .filter(|d: &Diagnostic| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

/// A sole positional 4-column table — four columns so no row's ARITY can incidentally reject
/// it, which is what used to mask `builtin_sample`.
fn table_splat(builtin: &str) -> Vec<String> {
    errors(&format!(
        "xs = elementof(cartpow(reals, 4))\n\
         t = table(zzq = xs, zzr = xs, zzs = xs, zzt = xs)\n\
         z = {builtin}(t)\n"
    ))
}

fn record_splat(builtin: &str) -> Vec<String> {
    errors(&format!(
        "z = {builtin}(record(zzq = 1.0, zzr = 2.0, zzs = 3.0, zzt = 4.0))\n"
    ))
}

/// Every base row with an unnamed variadic tail refuses a splatted aggregate, in BOTH the
/// table and record spellings — §04 draws no distinction between them ("`f(record(…))` and
/// `f(table(…))` are equivalent to `f(a = x, b = y, ...)`").
///
/// The five rows are enumerated from the catalogue rather than listed, so a variadic row added
/// later is covered without editing this test.
#[test]
fn every_unnamed_variadic_row_refuses_a_splatted_aggregate() {
    let cat = builtin_catalogue();
    let variadic: Vec<&str> = cat
        .base_names()
        .filter(|n| {
            // Same structural property the implementation reads, re-derived here from the
            // public API so the test is not just the implementation restated.
            cat.base_arity(n).is_some_and(|a| a.max.is_none())
        })
        .collect();
    assert!(
        variadic.len() >= 5,
        "expected the known variadic rows, found {variadic:?}"
    );
    for n in &variadic {
        for (spelling, errs) in [("table", table_splat(n)), ("record", record_splat(n))] {
            assert!(
                errs.iter()
                    .any(|e| e.contains("variadic inputs are UNNAMED")),
                "`{n}` must refuse a {spelling} splat: {errs:?}"
            );
        }
    }
}

/// The three names the B76 audit left open, pinned individually so a regression names the row.
#[test]
fn the_three_names_b76_left_open_are_closed() {
    for n in ["cat", "get", "get0"] {
        assert!(
            !table_splat(n).is_empty(),
            "`{n}` was the last accepting row and must now refuse"
        );
    }
}

/// `builtin_sample` was masked: a 2-column table failed its ARITY (at least 3 arguments), so
/// the splat looked handled. With four columns the arity fits and it used to be accepted —
/// which is why the refusal precedes the arity comparison rather than following it.
#[test]
fn builtin_sample_is_not_saved_by_its_arity() {
    let two_col = errors(
        "xs = elementof(cartpow(reals, 4))\n\
         t = table(zzq = xs, zzr = xs)\n\
         z = builtin_sample(t)\n",
    );
    assert!(
        two_col
            .iter()
            .any(|e| e.contains("variadic inputs are UNNAMED")),
        "even where arity would also complain, the §04 reason is the one reported: {two_col:?}"
    );
    assert!(
        !table_splat("builtin_sample").is_empty(),
        "and a fitting column count must not slip through"
    );
}

/// The diagnostic must name the honest spelling, per row. `cat`'s whole argument list is the
/// variadic tail, so every field becomes its own argument; §07 "Field and element access" gives
/// the concise form: "`r.a` ≡ `get(r, "a")`".
#[test]
fn the_cat_diagnostic_names_each_field_as_its_own_argument() {
    let errs = table_splat("cat");
    let msg = errs
        .iter()
        .find(|e| e.contains("variadic inputs are UNNAMED"))
        .expect("the §04 refusal");
    assert!(
        msg.contains("`cat(t.zzq, t.zzr, t.zzs, t.zzt)`"),
        "spells out the fix with the actual column names: {msg}"
    );
    // The columns are named in the message too, so the author can see what was splatted.
    assert!(
        msg.contains("`zzq`, `zzr`, `zzs`, `zzt`"),
        "names the splatted columns: {msg}"
    );
}

/// `get`/`get0` differ: the aggregate is almost certainly the intended `container`, and the
/// SELECTOR is what is missing. Advising `get(t.zzq, …)` there would be wrong.
#[test]
fn the_get_diagnostic_points_at_the_container_and_selectors() {
    for n in ["get", "get0"] {
        let errs = table_splat(n);
        let msg = errs
            .iter()
            .find(|e| e.contains("variadic inputs are UNNAMED"))
            .expect("the §04 refusal");
        assert!(
            msg.contains(&format!("`{n}(t, \"zzq\")`")) && msg.contains("`container` argument"),
            "advises container + selector, not field-by-field: {msg}"
        );
    }
}

/// **The keyword spelling must NOT be suggested here.** Every other splat diagnostic ends with
/// §04's "to pass it as one ordinary argument use the keyword spelling, as in
/// `f(pars = record(...))`" — but an UNNAMED input has no keyword to address it by, so that
/// advice cannot work on these rows and would send the author down a dead end.
#[test]
fn the_refusal_does_not_advise_the_unusable_keyword_spelling() {
    for n in ["cat", "get", "get0", "vector", "builtin_sample"] {
        for errs in [table_splat(n), record_splat(n)] {
            for e in errs.iter().filter(|e| e.contains("UNNAMED")) {
                assert!(
                    !e.contains("keyword spelling"),
                    "`{n}` must not advise a keyword that cannot exist: {e}"
                );
            }
        }
    }
}

/// The diagnostic cites the section that documents the row, through the same
/// `base_param_section` mapping the name and arity checks use.
#[test]
fn the_refusal_cites_the_documenting_section() {
    let msg = table_splat("cat")
        .into_iter()
        .find(|e| e.contains("UNNAMED"))
        .expect("the §04 refusal");
    assert!(msg.contains("(spec §07)"), "cat is a §07 row: {msg}");
    // And §04 is cited for the rule itself, since that is where the static error comes from.
    assert!(msg.contains("spec §04"), "cites §04 for the rule: {msg}");
}

/// **Prove-it-is-wrong.** Only a CONFIRMED sole-positional record/table refuses. An argument
/// whose type inference has not resolved must keep the path it has — `arg_reading` answers
/// `None` for `Deferred`/`Var`/`Any`/`Failed`, so the whole arity check bails and nothing is
/// reported. Pinned so the guard cannot drift into fail-closed.
#[test]
fn an_unresolved_argument_stays_permissive() {
    assert!(
        errors("f = functionof(cat(_p_), q = _p_)\nz = f(1.0)\n").is_empty(),
        "a deferred argument must not be refused"
    );
    // A non-aggregate sole argument is an ordinary positional call and is untouched.
    assert!(
        errors("v = elementof(cartpow(reals, 3))\nz = cat(v)\n").is_empty(),
        "a sole VECTOR argument is not a splat and must still bind positionally"
    );
}

/// The nine `#78`-exempt names are untouched: they take the aggregate WHOLE, so no splat
/// happens and this refusal must never reach them. `sum` and `lengthof` are the normative
/// examples in #78's own sentence.
#[test]
fn the_exempt_carve_out_is_untouched() {
    for n in [
        "sum",
        "mean",
        "var",
        "std",
        "lengthof",
        "reverse",
        "indicesof",
        "indicesof0",
        "identity",
    ] {
        assert!(
            table_splat(n).is_empty(),
            "`{n}` is #78-exempt and must still take the table whole"
        );
    }
}

/// A NAMED row is untouched too — it keeps the name-checked splat the NAMES wave gave it, so
/// this change is scoped to the unnamed-variadic shape and did not become a blanket
/// splat-refusal.
#[test]
fn a_named_row_keeps_its_name_checked_splat() {
    assert!(
        errors("z = atan2(record(y = 1.0, x = 2.0))\n").is_empty(),
        "a name-MATCHED splat onto a named row still binds"
    );
    let mismatch = errors("z = atan2(record(zzq = 1.0, zzr = 2.0))\n");
    assert!(
        mismatch
            .iter()
            .any(|e| e.contains("has no parameter `zzq`")),
        "and a mismatch still reports the NAME check, not the variadic one: {mismatch:?}"
    );
}
