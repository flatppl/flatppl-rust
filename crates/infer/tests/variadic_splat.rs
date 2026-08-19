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
            // The implementation's real predicate, re-derived here from the public API so the
            // test is not just the implementation restated: variadic AND declaring no names.
            // `vector` is excluded on §07/§03 grounds (see
            // `vector_keeps_the_sec03_element_diagnosis`), and a row that DECLARES names — like
            // `builtin_sample` — is not an "unnamed" row at all, so the ordinary name check
            // decides it (see `the_guard_keys_on_declared_names_not_on_a_hardcoded_list`).
            *n != "vector"
                && cat.base_param_names(n).is_none()
                && cat.base_arity(n).is_some_and(|a| a.max.is_none())
        })
        .collect();
    let mut variadic = variadic;
    variadic.sort_unstable();
    assert_eq!(
        variadic,
        ["cat", "get", "get0"],
        "exactly the nameless variadic rows"
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

/// The refusal precedes the ARITY comparison, so the §04 reason is reported even where the count
/// would also be wrong. `get` takes at least 2 arguments, so a ONE-column table fails arity too —
/// and must still report the binding problem, which is the actionable one.
///
/// This property was found through `builtin_sample`, whose 2-column probe failed its "at least 3
/// arguments" check and so looked handled in both the B76 audit and the NAMES sweep; four columns
/// fit and it was accepted outright. `builtin_sample` no longer belongs to this guard at all (it
/// declares §07 names — `wave-CATADJ-report.md` §6), so the property is pinned here on a row that
/// does. The lesson that produced it stands: probe with more columns than any row's arity.
#[test]
fn the_refusal_precedes_the_arity_check() {
    let one_col = errors(
        "xs = elementof(cartpow(reals, 4))\n\
         t = table(zzq = xs)\n\
         z = get(t)\n",
    );
    assert!(
        one_col
            .iter()
            .any(|e| e.contains("variadic inputs are UNNAMED")),
        "the §04 reason is reported, not a bare count complaint: {one_col:?}"
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
    for n in ["cat", "get", "get0", "builtin_sample"] {
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

/// **`checked` is the one NAMED row that does NOT keep the name-checked splat** — owner ruling
/// on design PR #78 (decisions-log 2026-08-18): "§07's keyword form owns `checked`; no special
/// operation splats a sole record/table argument." Before this ruling landed,
/// `checked(record(value = 1.0, condition = true))` splatted and SUCCEEDED, silently narrowing
/// the result to the `value` field's type — the red case `designpass2-report.md`'s "PR 78"
/// section calls out by name. `cat`/`get`/`get0` already refused (via the unnamed-variadic
/// guard above); `checked` needed its own carve-out because its row DOES declare names.
#[test]
fn checked_never_splats_a_sole_record_even_on_a_name_match() {
    let matched = errors("z = checked(record(value = 1.0, condition = true))\n");
    assert!(
        matched
            .iter()
            .any(|e| e.contains("has no special-operation splat")),
        "a name-MATCHED splat must still refuse: {matched:?}"
    );
    assert!(
        matched.iter().any(|e| e.contains("checked(value = ...")),
        "the refusal must point at the keyword form: {matched:?}"
    );
    // The two spellings §07 actually grants stay legal.
    for spelling in [
        "z = checked(1.0, true)\n",
        "z = checked(value = 1.0, condition = true)\n",
    ] {
        assert!(
            errors(spelling).is_empty(),
            "positional and keyword must both keep working: {spelling}"
        );
    }
}

/// **`vector` keeps §03's element diagnosis** — the one variadic row where a sole aggregate has
/// a plausible NON-splat reading, and the one place the §04 message would be a net loss.
///
/// §07 gives `vector` the arguments `x1, x2, ...` over the domain "scalars": its arguments ARE
/// the elements of the result. So `vector(pars)` reads naturally as "a one-element array holding
/// `pars`" — which is exactly what `[pars]` means in the two corpus models below — and §03's
/// element rule reports that accurately. The §04 splat message would instead advise
/// `vector(t.a, t.b, t.mu)`: a THREE-element array of reals, a different value from the
/// one-element array the author asked for.
///
/// This pins the corpus shape directly: an array literal holding a record, as written in
/// `flatppl-js/packages/engine/test/fixtures/simple-transport1.flatppl:21` and
/// `packages/web/demo/transport-model.flatppl:21` (`ys ~ transport.(xs, [pars])`).
#[test]
fn vector_keeps_the_sec03_element_diagnosis() {
    // The transport shape: a record, then an array literal holding it.
    let errs = errors(
        "a = elementof(reals)\n\
         b = elementof(reals)\n\
         mu = elementof(reals)\n\
         pars = record(a = a, b = b, mu = mu)\n\
         z = [pars]\n",
    );
    assert!(
        errs.iter().any(
            |e| e.contains("array elements must be scalars, strings, or arrays")
                && e.contains("got a record")
        ),
        "§03's element rule is the accurate diagnosis for a one-element array: {errs:?}"
    );
    assert!(
        !errs.iter().any(|e| e.contains("UNNAMED")),
        "the §04 splat message must NOT fire — it would advise a different array shape: {errs:?}"
    );
}

// ---------------------------------------------------------------------------
// `builtin_sample` — an ORDINARY callable, so a name-matched splat must bind
// ---------------------------------------------------------------------------

/// A model prefix supplying a real rngstate, kernel and kernel input, so the only thing under
/// test is how the arguments are SPELLED.
const PRIMS: &str = "s = rnginit([42, 0, 0, 0])\n\
                     k = kernelof(draw(Normal(mu = _m_, sigma = 1.0)), mu = _m_)\n\
                     x = record(mu = 0.5)\n";

/// The adjudicated non-conformance (`wave-CATADJ-report.md` §6), now fixed.
///
/// §04's special-operations list omits `builtin_sample` entirely, so it is an ORDINARY callable:
/// "All built-in ordinary callables have a defined input order and accept both positional and
/// keyword arguments." §07 "Measure kernel evaluation primitives" documents its three
/// distinguished inputs by name — "`builtin_sample(rngstate, kernel, kernel_input, n, m, ...)`"
/// — so a record carrying exactly those splats to a valid three-argument call and §04's
/// auto-splat bullet applies with no scoping dispute.
///
/// Before the fix its row alone among the six primitives carried no `names`, so the structural
/// guard caught its trailing `Variadic(Scalar(Integer))` (the optional sample shape) and refused.
#[test]
fn a_name_matched_record_splats_onto_builtin_sample() {
    assert!(
        errors(&format!(
            "{PRIMS}r = record(rngstate = s, kernel = k, kernel_input = x)\n\
             z = builtin_sample(r)\n"
        ))
        .is_empty(),
        "§07 names these three inputs, so the splat binds"
    );
    // The two spellings §04 grants an ordinary callable, as controls.
    for spelling in [
        "z = builtin_sample(s, k, x)\n",
        "z = builtin_sample(rngstate = s, kernel = k, kernel_input = x)\n",
    ] {
        assert!(
            errors(&format!("{PRIMS}{spelling}")).is_empty(),
            "positional and keyword must both keep working: {spelling}"
        );
    }
}

/// A MISMATCHED record errors through the ordinary NAME check, citing §07's parameter list —
/// **not** the variadic-splat message. The row is no longer in that guard at all.
#[test]
fn a_mismatched_record_on_builtin_sample_errors_via_the_name_check() {
    let errs = errors(&format!(
        "{PRIMS}r = record(zzq = s, zzr = k, zzs = x)\nz = builtin_sample(r)\n"
    ));
    assert!(
        errs.iter()
            .any(|e| e.contains("`builtin_sample` has no parameter `zzq`")
                && e.contains("spec §07 parameters: `rngstate`, `kernel`, `kernel_input`")),
        "the NAME check decides: {errs:?}"
    );
    assert!(
        !errs
            .iter()
            .any(|e| e.contains("variadic inputs are UNNAMED")),
        "the variadic guard must not fire on a row that declares names: {errs:?}"
    );
}

/// An EXTRA field errors, and the reason is worth recording because §07's Arguments cell does
/// mention `n`.
///
/// Only the three DISTINGUISHED inputs are declared; the sample-shape tail (`n, m, ...`) stays
/// variadic and nameless, because naming a prefix of an unbounded tail would refuse a legitimate
/// further shape argument. So a record carrying `n` fails §04's mismatch clause: "A call with
/// field or column names that do not match the callable's argument names is a static error".
///
/// **Residual ambiguity, flagged not resolved:** §07 spells the signature `builtin_sample(rngstate,
/// kernel, kernel_input, n, m, ...)`, so `n` IS a name the spec writes down — yet the `...`
/// leaves the tail unbounded and unnamed beyond `m`, and §04 gives no rule for binding a splatted
/// field to a variadic slot. Refusing is the conservative side: it rejects a spelling the spec
/// arguably permits rather than accepting one it may not, and the keyword form
/// `builtin_sample(rngstate = …, kernel = …, kernel_input = …, 4)` remains available.
#[test]
fn an_extra_shape_field_errors_because_only_the_distinguished_inputs_are_named() {
    let errs = errors(&format!(
        "{PRIMS}r = record(rngstate = s, kernel = k, kernel_input = x, n = 4)\n\
         z = builtin_sample(r)\n"
    ));
    assert!(
        errs.iter()
            .any(|e| e.contains("`builtin_sample` has no parameter `n`")),
        "the shape tail is nameless, so `n` binds nothing: {errs:?}"
    );
}

/// A TABLE whose columns match the three names **splats correctly and then fails on argument
/// TYPES** — and both halves of that are the point.
///
/// The BINDING is valid: §04 draws no distinction between a record and a table splat, and the
/// names match. What is wrong is the types — a table's columns are equal-length vectors, so
/// `rngstate` receives a vector of reals where §07's Domains cell wants `rngstates`, and `kernel`
/// a vector where it wants a kernel.
///
/// **FLIPPED by wave KWTYPE.** BSFIX pinned this as ACCEPTED and recorded why: the kernel-type
/// check read `args` positionally, so it fired on `builtin_sample(1.0, 2.0, 3.0)` and stayed
/// silent on the keyword form — and a splat lowers to the keyword form, so it inherited that hole.
/// Verified pre-existing at `af1d92d` at the time. KWTYPE normalizes keyword arguments into their
/// declared positions before the per-op rules run, so all three spellings — positional, keyword,
/// and splatted — now reach the same check. The assertions below are the inverse of BSFIX's.
#[test]
fn a_name_matched_table_splat_now_reaches_the_argument_type_check() {
    let table = errors(
        "xs = elementof(cartpow(reals, 4))\n\
         t = table(rngstate = xs, kernel = xs, kernel_input = xs)\n\
         z = builtin_sample(t)\n",
    );
    assert!(
        table
            .iter()
            .any(|e| e.contains("must be a distribution kernel")),
        "the splat binds by name and then the type check fires: {table:?}"
    );
    // All three spellings of the same call now agree — this is the asymmetry KWTYPE closed.
    for spelling in [
        "z = builtin_sample(1.0, 2.0, 3.0)\n",
        "z = builtin_sample(rngstate = 1.0, kernel = 2.0, kernel_input = 3.0)\n",
        "z = builtin_sample(1.0, kernel = 2.0, kernel_input = 3.0)\n",
    ] {
        assert!(
            errors(spelling)
                .iter()
                .any(|e| e.contains("must be a distribution kernel")),
            "every §04-equivalent spelling must reach the check: {spelling}"
        );
    }
}

/// The guard steps aside for `builtin_sample` because it DECLARES names, not because it is listed
/// somewhere — so the mechanism is self-maintaining, and `cat`/`get`/`get0` are unaffected.
#[test]
fn the_guard_keys_on_declared_names_not_on_a_hardcoded_list() {
    let cat = builtin_catalogue();
    assert_eq!(
        cat.base_param_names("builtin_sample")
            .expect("builtin_sample declares its §07 names"),
        ["rngstate", "kernel", "kernel_input"],
        "exactly §07's three distinguished inputs"
    );
    for n in ["cat", "get", "get0"] {
        assert!(
            cat.base_param_names(n).is_none(),
            "`{n}` still declares no names, so it stays in the guard"
        );
        assert!(
            !table_splat(n).is_empty(),
            "`{n}` must still refuse a splatted aggregate"
        );
    }
}
