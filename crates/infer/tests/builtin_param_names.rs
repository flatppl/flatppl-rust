//! §07/§06 function rows carry their documented parameter names, so §04's name-binding
//! rule reaches them — not just §08 constructors.
//!
//! §04 "Calling conventions": "**Auto-splatting** (of records and table columns):
//! `f(record(a = x, b = y, ...))` and `f(table(a = x, b = y, ...))` are equivalent to
//! `f(a = x, b = y, ...)`. … **A call with field or column names that do not match the
//! callable's argument names is a static error.**" The splat is defined as equivalence to
//! the KEYWORD form, and §04's general rule is "Arguments are bound to inputs by name".
//!
//! Only `Sig::Distribution` rows declared names, so for every §07 function row the name
//! check had nothing to compare and the splat fell through to binding by column ORDER. The
//! B76 audit measured the consequence: 45 builtins accepted a sole positional 2-column table
//! whose column names matched nothing, 23 of them returning a CONFIRMED type, so a wrong
//! binding looked correct and flowed on. `logdensityof(t)` typed as `(%scalar real)`/`reals`
//! with a table bound where §06's measure argument belongs.
//!
//! Names come from §07's "Arguments" column and §06's prose, entered per row and
//! cross-checked against each row's declared parameter COUNT — all 142 rows with a §07 table
//! entry agreed, so no arity was invented to fit a name list.

use flatppl_infer::{Diagnostic, Severity, builtin_catalogue, infer};

fn errors(src: &str) -> Vec<String> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m)
        .into_iter()
        .filter(|d: &Diagnostic| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

/// A sole positional 2-column table whose column names match nothing.
fn mismatched(builtin: &str) -> Vec<String> {
    errors(&format!(
        "xs = elementof(cartpow(reals, 4))\n\
         ys = elementof(cartpow(reals, 4))\n\
         t = table(zzq = xs, zzr = ys)\n\
         z = {builtin}(t)\n"
    ))
}

/// The nine names §04's single-input carve-out exempts (flatppl-design#78) — they take the
/// aggregate WHOLE, so no splat and no name check applies to them.
const EXEMPT: &[&str] = &[
    "identity",
    "indicesof",
    "indicesof0",
    "lengthof",
    "mean",
    "reverse",
    "std",
    "sum",
    "var",
];

/// The audit's headline case, and the reason this wave exists. §06: "`densityof(M, x)` and
/// `logdensityof(M, x)` evaluate the density of a measure at a point with respect to an
/// implicit reference measure." Before names, `logdensityof(t)` typed as a clean real
/// log-density with a table where `M` belongs.
///
/// The citation must say **§06**, not §07: these are measure operations described in prose,
/// and §07's tables do not list them, so citing §07 would send a reader to the wrong page.
#[test]
fn logdensityof_names_its_measure_argument_and_cites_sec06() {
    let errs = mismatched("logdensityof");
    // One diagnostic per unbindable column, not one for the call — the splat names every
    // field that binds nothing, so the author fixes them in one pass.
    assert_eq!(errs.len(), 2, "one per unbindable column: {errs:?}");
    for (col, e) in ["zzq", "zzr"].iter().zip(&errs) {
        assert!(
            e.contains(&format!("`logdensityof` has no parameter `{col}`"))
                && e.contains("spec §06 parameters: `M`, `x`"),
            "names the measure parameter from §06: {e}"
        );
    }
    assert!(
        errors("z = densityof(record(zzq = 1.0, zzr = 2.0))\n")[0]
            .contains("spec §06 parameters: `M`, `x`"),
        "densityof names the same two"
    );
}

/// An ordinary §07 row cites §07 and its own Arguments column. `atan2` is the sharpest case:
/// its arguments are `y`, `x` **in that order**, so a positional bind silently swapped them —
/// `atan2(table(x = …, y = …))` used to be accepted and bound `y` to the `x` column.
#[test]
fn a_sec07_row_names_its_arguments_in_spec_order() {
    let errs = mismatched("atan2");
    assert!(
        errs[0].contains("`atan2` has no parameter `zzq`")
            && errs[0].contains("spec §07 parameters: `y`, `x`"),
        "cites §07 and the documented order: {}",
        errs[0]
    );
    // The swap that used to pass silently is now caught, because `x` and `y` ARE declared —
    // so this reports the ARITY-free name path, not a count mismatch.
    assert!(
        errors("z = atan2(record(x = 1.0, y = 2.0))\n").is_empty(),
        "both names are declared, so a reordered record is valid — §04: \
         \"The order of fields or columns is not relevant\""
    );
}

/// §04 says column order is irrelevant once binding is by name, which is the property the
/// positional path violated. Both orders must type identically.
#[test]
fn column_order_is_irrelevant_once_binding_is_by_name() {
    for src in [
        "z = atan2(record(y = 1.0, x = 2.0))\n",
        "z = atan2(record(x = 2.0, y = 1.0))\n",
    ] {
        assert!(errors(src).is_empty(), "must be valid either way: {src}");
    }
}

/// The whole audit surface, swept: every base name against a name-mismatched sole positional
/// 2-column table. Only the nine exempt names and the deliberately-nameless VARIADIC rows may
/// still accept it.
#[test]
fn no_builtin_accepts_a_name_mismatched_splat_except_the_documented_exclusions() {
    let cat = builtin_catalogue();
    let mut accepted: Vec<&str> = cat
        .base_names()
        .filter(|n| mismatched(n).is_empty())
        .collect();
    accepted.sort_unstable();
    let unexpected: Vec<&&str> = accepted
        .iter()
        .filter(|n| !EXEMPT.contains(n) && !["cat", "get", "get0"].contains(n))
        .collect();
    assert!(
        unexpected.is_empty(),
        "these accept a name-mismatched splat and should not: {unexpected:?}"
    );
    // Pin the exclusions positively too, so shrinking the set is also a visible change.
    for n in EXEMPT {
        assert!(
            accepted.contains(n),
            "`{n}` is #78-exempt and must still take the table whole"
        );
    }
}

/// The name-MATCHED direction: a splat whose column names ARE the declared ones must not
/// raise a §04 name error. Swept over every row declaring exactly two names, so the fix is
/// shown to enable the valid spelling rather than merely reject the invalid one.
///
/// A domain or type complaint is allowed through here — the columns are real vectors and many
/// rows want scalars or matrices. Only a NAME diagnostic is a failure.
#[test]
fn a_name_matched_splat_raises_no_name_error() {
    let cat = builtin_catalogue();
    let mut offenders = vec![];
    let mut checked = 0;
    for n in cat.base_names() {
        let Some(p) = cat.base_param_names(n) else {
            continue;
        };
        if p.len() != 2 {
            continue;
        }
        checked += 1;
        let errs = errors(&format!(
            "xs = elementof(cartpow(reals, 4))\n\
             ys = elementof(cartpow(reals, 4))\n\
             t = table({} = xs, {} = ys)\n\
             z = {n}(t)\n",
            p[0], p[1]
        ));
        if errs.iter().any(|m| m.contains("has no parameter")) {
            offenders.push(format!("{n}: {}", errs.join(" ¦ ")));
        }
    }
    assert!(checked > 50, "swept a meaningful number of rows: {checked}");
    assert!(
        offenders.is_empty(),
        "a name-matched splat must bind: {offenders:?}"
    );
}

/// §08 constructors are untouched — they already name-checked, and must keep citing §08.
#[test]
fn distribution_rows_are_unchanged() {
    let errs = errors("z = Normal(record(zzz = 0.5, qqq = 1.0))\n");
    assert!(
        errs[0].contains("`Normal` has no parameter `zzz`")
            && errs[0].contains("spec §08 parameters: `mu`, `sigma`"),
        "still §08: {}",
        errs[0]
    );
}

/// **Prove-it-is-wrong.** A row documenting no names answers `None` from
/// `base_param_names`, which the arity check reads as "accept" rather than "reject". `length`
/// has no §07 entry at all, so it declares no names — and the call is still caught, by the
/// ARITY rule, which is the backstop that makes the permissive default safe.
#[test]
fn an_undocumented_row_stays_permissive_on_names_and_relies_on_arity() {
    let cat = builtin_catalogue();
    assert!(
        cat.base_param_names("length").is_none(),
        "`length` has no §07 entry, so no documented names"
    );
    let errs = mismatched("length");
    assert!(
        errs[0].contains("`length` takes 1 argument (spec §07), got 2"),
        "the arity rule still catches it: {}",
        errs[0]
    );
}

/// The variadic rows are deliberately nameless, and this pins the boundary rather than
/// hiding it. §07 spells their tails `x, y, ...` / `container, selectors...`, naming a prefix
/// but giving the repeated arguments no individual names, and §04 states no rule for binding
/// a splatted field to a variadic slot. Enforcing the prefix alone would reject a column
/// legitimately destined for an unnamed tail argument, so this waits on a spec answer.
#[test]
fn variadic_rows_are_deliberately_nameless_pending_a_spec_rule() {
    let cat = builtin_catalogue();
    for n in ["cat", "get", "get0"] {
        assert!(
            cat.base_param_names(n).is_none(),
            "`{n}` is variadic and must stay nameless until §04 rules on variadic splat"
        );
        assert!(
            mismatched(n).is_empty(),
            "`{n}` therefore still accepts a name-mismatched splat — the known gap"
        );
    }
}

/// Names are only useful if they are the SPEC's. Spot-check a spread of rows whose argument
/// order or naming is easy to get wrong, each read from its §07 Arguments cell.
#[test]
fn spot_checked_names_match_the_spec_arguments_column() {
    let cat = builtin_catalogue();
    for (n, want) in [
        ("atan2", &["y", "x"][..]),
        ("pow", &["base", "exponent"][..]),
        ("linsolve", &["A", "b"][..]),
        ("quadform", &["A", "x"][..]),
        ("in", &["x", "S"][..]),
        ("conv", &["v", "kernel"][..]),
        ("bincounts", &["bins", "data"][..]),
        ("complex", &["re", "im"][..]),
        ("checked", &["value", "condition"][..]),
        ("load_data", &["source", "valueset"][..]),
        ("rand", &["rstate", "m"][..]),
        ("filter", &["pred", "data"][..]),
        ("diag", &["A", "k"][..]),
        ("onehot", &["i", "n"][..]),
        ("sum", &["xs"][..]),
    ] {
        let got = cat
            .base_param_names(n)
            .unwrap_or_else(|| panic!("`{n}` should declare names"));
        assert_eq!(got, want, "`{n}` names must be the spec's");
    }
}
