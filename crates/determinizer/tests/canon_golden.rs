use flatppl_determinizer::determinize;

fn determinize_src(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    determinize(&m).expect("must lower, not refuse")
}

// Constant subexpressions in the determinised FlatPDL fold to a literal:
// scoring a mixture with literal weights leaves `log(0.3 + 0.5)`; const-fold
// reduces the `add(0.3, 0.5)` to `0.8`. Buffy #263 Pass 1.
#[test]
fn const_fold_reduces_literal_sum_of_weights() {
    let src = "\
m = normalize(superpose(\
weighted(0.3, Normal(mu = 0.0, sigma = 1.0)), \
weighted(0.5, Gamma(shape = 2.0, rate = 1.0))))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let pir = flatppl_flatpir::write(&determinize_src(src));
    assert!(
        pir.contains("(log 0.8)"),
        "add(0.3, 0.5) folds to 0.8 under the outer log:\n{pir}"
    );
    assert!(
        !pir.contains("(add 0.3 0.5)"),
        "no un-folded literal sum remains:\n{pir}"
    );
}

// const_fold is idempotent: determinizing the same source twice produces
// identical FlatPIR (canonicalize already ran to a fixpoint inside the first
// determinize).
#[test]
fn const_fold_is_idempotent() {
    let src = "\
m = normalize(superpose(\
weighted(0.3, Normal(mu = 0.0, sigma = 1.0)), \
weighted(0.5, Gamma(shape = 2.0, rate = 1.0))))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let once = flatppl_flatpir::write(&determinize_src(src));
    let twice = flatppl_flatpir::write(&determinize_src(src));
    assert_eq!(
        once, twice,
        "determinize output is a canonicalization fixpoint"
    );
}

// `resolve_alias_refs`, isolated from `const_fold`: `w` is a literal-alias
// mixture weight (a top-level `%bind` whose RHS is a bare `Lit`, referenced
// via `(%ref self w)` from `weighted(w, ...)`), not a bare literal like the
// two tests above. The determinized FlatPDL must resolve every `w` reference
// to its literal value — the alias itself (0.4) or its downstream fold (the
// mixture-weight sum 0.4 + 0.5 = 0.9, once `resolve_alias_refs` exposes the
// literal to `const_fold`) — never a residual `(%ref self w)`. Buffy #263
// Pass 1, Step 12 golden.
#[test]
fn resolve_alias_refs_inlines_literal_weight_alias() {
    let src = "\
w = 0.4
m = normalize(superpose(\
weighted(w, Normal(mu = 0.0, sigma = 1.0)), \
weighted(0.5, Gamma(shape = 2.0, rate = 1.0))))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        !pir.contains("(%ref self w)"),
        "the alias to `w` must be resolved, not left as a residual ref:\n{pir}"
    );
    assert!(
        pir.contains("0.4") || pir.contains("0.9"),
        "the resolved literal (0.4) or its downstream fold (0.9) must appear:\n{pir}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
}

// `sweep_dead_bindings`, isolated: `_ = exp(1.0)` lowers to a synthetic,
// unreferenced binding (`__0x1`, per the parser's discard-name convention);
// the sweep must zero it. A scored output named with a leading underscore
// (`__score__`, the flatppl-testsuite/CLI scoring convention) is NOT
// synthetic — it is a plain user-chosen name — so it must survive untouched
// even though `!public` alone (name starts with `_`) would also match it.
// Pins the `synthetic`-vs-`!public` predicate fix the numeric det-js
// equivalence gate caught (see task-1-report.md). Buffy #263 Pass 1, Step 12
// golden.
#[test]
fn sweep_dead_bindings_removes_synthetic_but_preserves_scored_binding() {
    let src = "\
_ = exp(1.0)
a = draw(Normal(mu = 0.0, sigma = 1.0))
__score__ = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("(%bind __0x1 0.0)"),
        "the dead synthetic discard binding must be swept to the zero sentinel:\n{pir}"
    );
    assert!(
        !pir.contains("exp(1.0)") && !pir.contains("(exp 1.0)"),
        "the swept binding's original RHS must not survive:\n{pir}"
    );
    assert!(
        pir.contains("__score__") && pir.contains("builtin_logdensityof"),
        "the non-synthetic __score__ binding must survive with its real expression, not be swept:\n{pir}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
}

// A user-defined function used deterministically must be INLINED in FlatPDL:
// no residual `(%call ...)` reaches consumers (flatppl-js can't evaluate one —
// Buffy #261). Buffy #263 Pass 2.
#[test]
fn inline_user_calls_eliminates_residual_user_call() {
    let src = "\
scale(x) = mul(x, 2.0)
s = scale(1.5)
a = draw(Normal(mu = 0.0, sigma = s))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        !pir.contains("(%call"),
        "no residual user-function call in FlatPDL:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl:\n{pir}"
    );
}

// inline_user_calls (Pass 2) is idempotent: determinizing the same source
// twice produces identical FlatPIR (canonicalize already ran to a fixpoint
// inside the first determinize). Same fixture as
// `inline_user_calls_eliminates_residual_user_call` — a genuine user-function
// call Pass 2 rewrites, not a vacuous no-op source — so this also pins that
// the capture-avoiding `substitute_ref` (kernel.rs `shadows_name` guard)
// reaches a stable fixpoint rather than re-triggering on a second pass.
#[test]
fn inline_user_calls_is_idempotent() {
    let src = "\
scale(x) = mul(x, 2.0)
s = scale(1.5)
a = draw(Normal(mu = 0.0, sigma = s))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let once = flatppl_flatpir::write(&determinize_src(src));
    assert!(
        !once.contains("(%call"),
        "sanity: fixture must actually contain a user call Pass 2 inlines \
         (else this test is vacuous):\n{once}"
    );
    let twice = flatppl_flatpir::write(&determinize_src(src));
    assert_eq!(
        once, twice,
        "determinize output is a canonicalization fixpoint"
    );
}
