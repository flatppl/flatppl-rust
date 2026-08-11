//! Keyword arguments reach the same per-argument checks as positional ones.
//!
//! §04 "Calling conventions": "All built-in ordinary callables have a defined input order and
//! accept both positional and keyword arguments." So `f(x, y)`, `f(a = x, b = y)` and the mixed
//! `f(x, b = y)` are the same call, and every rule reading argument positions must see the same
//! vector for all of them. §04 also makes the splat a third spelling of the keyword form:
//! "`f(record(a = x, b = y, ...))` … are equivalent to `f(a = x, b = y, ...)`".
//!
//! Before `normalize_keyword_args`, the per-op rules indexed `args` positionally
//! (`args.get(1)`, `args.first()`), so a keyword call reached them with an EMPTY `args` and every
//! position-indexed check silently deferred. Measured width of the asymmetry — 7 rows, two
//! distinct checks, no row erroring on the keyword path alone:
//!
//! | check | rows |
//! |---|---|
//! | kernel-type (`non_kernel_or_defer`) | `builtin_sample`, `builtin_touniform`, `builtin_fromuniform`, `builtin_tonormal`, `builtin_fromnormal` |
//! | integer-domain | `div`, `mod` |
//!
//! Distribution rows are NOT affected: they check no argument types on *either* path, which is a
//! symmetric gap of the same family as the unchecked operand domains the B76 audit found, and is
//! deliberately out of scope here. `builtin_logdensityof` likewise has no kernel check on either
//! path — the sixth primitive, symmetric.

use flatppl_infer::{Diagnostic, Severity, infer};

fn errors(src: &str) -> Vec<String> {
    let mut m = flatppl_syntax::parse(src).unwrap();
    infer(&mut m)
        .into_iter()
        .filter(|d: &Diagnostic| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

/// The five kernel-consuming primitives: every §04-equivalent spelling of a wrong-typed call must
/// reach the kernel-type check. §07 "Measure kernel evaluation primitives" is what the check
/// enforces — each "operates directly on a FlatPPL kernel object and a valid kernel input value".
#[test]
fn every_spelling_reaches_the_kernel_type_check() {
    // `builtin_sample(rngstate, kernel, kernel_input, …)` — kernel is argument 2.
    for spelling in [
        "z = builtin_sample(1.0, 2.0, 3.0)\n",
        "z = builtin_sample(rngstate = 1.0, kernel = 2.0, kernel_input = 3.0)\n",
        // Keyword order is irrelevant once binding is by name.
        "z = builtin_sample(kernel = 2.0, kernel_input = 3.0, rngstate = 1.0)\n",
        // Mixed: §04 binds the positional prefix in order, then the keywords by name.
        "z = builtin_sample(1.0, kernel = 2.0, kernel_input = 3.0)\n",
    ] {
        assert!(
            errors(spelling)
                .iter()
                .any(|e| e.contains("argument 2 must be a distribution kernel")),
            "must reach the check: {spelling}"
        );
    }
    // The four transports take the kernel as argument 1.
    for op in [
        "builtin_touniform",
        "builtin_fromuniform",
        "builtin_tonormal",
        "builtin_fromnormal",
    ] {
        let kw = errors(&format!(
            "z = {op}(kernel = 1.0, kernel_input = 2.0, x = 3.0)\n"
        ));
        let ku = errors(&format!(
            "z = {op}(kernel = 1.0, kernel_input = 2.0, u = 3.0)\n"
        ));
        let kz = errors(&format!(
            "z = {op}(kernel = 1.0, kernel_input = 2.0, z = 3.0)\n"
        ));
        assert!(
            [kw, ku, kz].iter().any(|errs| errs
                .iter()
                .any(|e| e.contains("argument 1 must be a distribution kernel"))),
            "`{op}` must reach the kernel check through its keyword spelling"
        );
    }
}

/// An ordinary §07 function: the integer-domain check on `div`/`mod`. §07 "Operator-equivalent
/// functions" gives both the domain `integers`, and the diagnostic points at `divide` for the real
/// case — advice a keyword-spelling author needs just as much as a positional one.
#[test]
fn a_sec07_functions_domain_check_reaches_the_keyword_spelling() {
    for op in ["div", "mod"] {
        for spelling in [
            format!("z = {op}(1.0, 1.0)\n"),
            format!("z = {op}(a = 1.0, b = 1.0)\n"),
            format!("z = {op}(1.0, b = 1.0)\n"),
        ] {
            assert!(
                errors(&spelling)
                    .iter()
                    .any(|e| e.contains("is integer-domain (spec §07)")),
                "must reach the integer-domain check: {spelling}"
            );
        }
    }
}

/// **Positive controls.** Valid keyword calls still type clean — the normalization must not turn a
/// well-formed call into an error, which is the failure mode a positional-vector rewrite invites.
#[test]
fn valid_keyword_calls_still_pass() {
    const PRIMS: &str = "s = rnginit([42, 0, 0, 0])\n\
                         k = kernelof(draw(Normal(mu = _m_, sigma = 1.0)), mu = _m_)\n\
                         x = record(mu = 0.5)\n";
    for src in [
        // The primitives, in all three spellings.
        format!("{PRIMS}z = builtin_sample(s, k, x)\n"),
        format!("{PRIMS}z = builtin_sample(rngstate = s, kernel = k, kernel_input = x)\n"),
        format!(
            "{PRIMS}r = record(rngstate = s, kernel = k, kernel_input = x)\nz = builtin_sample(r)\n"
        ),
        // A shape argument still lands after the three distinguished inputs.
        format!("{PRIMS}z = builtin_sample(s, k, x, 4)\n"),
        // `div`/`mod` on their actual domain.
        "z = div(a = 7, b = 2)\n".to_string(),
        "z = mod(a = 7, b = 2)\n".to_string(),
        // An ordinary two-argument function by keyword.
        "z = atan2(y = 1.0, x = 2.0)\n".to_string(),
        // A distribution by keyword — the spelling the whole corpus uses.
        "z = Normal(mu = 0.0, sigma = 1.0)\n".to_string(),
        // `load_data`, which reads its own keywords through a dual-spelling helper.
        "d = load_data(source = \"x.csv\", valueset = cartpow(reals, 4))\n".to_string(),
    ] {
        assert!(
            errors(&src).is_empty(),
            "a valid keyword call must stay clean: {src}\n{:?}",
            errors(&src)
        );
    }
}

/// Rows whose variadic inputs are genuinely NAMED are never normalized, because they declare no
/// parameter names for a keyword to map onto — so `record`/`table`/`joint`/`jointchain`/`broadcast`
/// keep reading `named` themselves and are untouched. This is what makes leaving `named` intact
/// safe rather than merely convenient.
#[test]
fn named_variadic_rows_are_untouched() {
    for src in [
        "z = record(a = 1.0, b = 2.0)\n",
        "xs = elementof(cartpow(reals, 4))\nz = table(a = xs, b = xs)\n",
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\nb = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         z = joint(a = lawof(a), b = lawof(b))\n",
    ] {
        assert!(
            errors(src).is_empty(),
            "a named-variadic row must keep working: {src}\n{:?}",
            errors(src)
        );
    }
}

/// **Normalization refuses to guess.** An ambiguous or incomplete mapping is handed back
/// unchanged so the existing checks report it — not silently patched into a positional vector.
///
/// This also PINS A PRE-EXISTING GAP this wave measured but did not fix: a **double-bound**
/// parameter goes unreported. `atan2(1.0, y = 2.0)` supplies `y` positionally *and* by keyword,
/// and neither existing check sees it — `arity_check` counts `args.len() + named.len()` = 2, which
/// matches `atan2`'s arity, and the name check only verifies that each supplied name IS declared,
/// never that it is supplied once. Verified unchanged at base `52dde93` and at this head, so the
/// normalizer's `None` is not what hides it. Pinned as measured, so closing it shows up here as a
/// deliberate change.
#[test]
fn an_ambiguous_mapping_is_left_to_the_existing_checks() {
    // Double-bound: NOT reported today, by either check. Pre-existing.
    assert!(
        errors("z = atan2(1.0, y = 2.0)\n").is_empty(),
        "pre-existing gap: a double-bound parameter is not detected — see this test's comment"
    );
    // An undeclared keyword name IS reported, by the name check.
    let unknown = errors("z = atan2(y = 1.0, zzq = 2.0)\n");
    assert!(
        unknown.iter().any(|e| e.contains("has no parameter `zzq`")),
        "the NAME check reports an undeclared keyword: {unknown:?}"
    );
    // Under-supplied IS reported, by the arity check — the normalizer declines to fabricate the
    // missing position, which is what leaves this visible.
    let gap = errors("z = atan2(y = 1.0)\n");
    assert!(
        gap.iter()
            .any(|e| e.contains("`atan2` takes 2 arguments (spec §07), got 1")),
        "an under-supplied call is reported on count: {gap:?}"
    );
}
