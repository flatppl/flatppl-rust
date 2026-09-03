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

// Pass 4-B completes the folder over the EXACT arithmetic Pass 1 skipped:
// int/real-mixed (`1 - 0.4`, where `1` is an Int) and integer-exponent `pow`
// (`2.0 ^ 2`). Both IEEE-exact, structural-around-density (a weight and a
// distribution parameter). The mixed `sub` folds to 0.6, the integer `pow` to
// 4.0 — no residual `(sub 1 0.4)` / `(pow ` remains. Buffy #263 Pass 4-B.
#[test]
fn const_fold_completes_int_real_mixed_and_integer_pow() {
    let src = "\
w = 1 - 0.4
k = 2.0 ^ 2
m = normalize(superpose(\
weighted(0.4, Normal(mu = 0.0, sigma = 1.0)), \
weighted(w, Gamma(shape = k, rate = 1.0))))
a = draw(m)
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("0.6"),
        "int/real-mixed `1 - 0.4` folds to 0.6:\n{pir}"
    );
    assert!(
        pir.contains("(%field shape 4.0)"),
        "integer `2.0 ^ 2` folds to the Gamma shape 4.0:\n{pir}"
    );
    assert!(
        !pir.contains("(sub 1 0.4)") && !pir.contains("(pow "),
        "no un-folded mixed sub / integer pow remains:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl:\n{pir}"
    );
}

// Int^Int with a non-negative exponent folds to an Int literal (not Real),
// matching `promote2`'s `Integer ⊔ Integer = Integer` — so a genuinely-Int
// parameter like Binomial's `n` keeps its integer literal (`8`, not `8.0`).
// Pins the Pass 4-B pow type-fidelity fix.
#[test]
fn const_fold_integer_pow_of_ints_stays_int() {
    let src = "\
a = draw(Binomial(n = 2 ^ 3, p = 0.5))
lp = logdensityof(lawof(record(a = a)), record(a = 4))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("(%field n 8)"),
        "2 ^ 3 (Int^Int) folds to the Int literal 8:\n{pir}"
    );
    assert!(
        !pir.contains("(%field n 8.0)"),
        "not widened to Real 8.0:\n{pir}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
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

// A positional `joint(M1, M2)` scored at a literal vector variate lowers each
// component's density term via `get0(v, i)` where `v` IS the literal
// `vector(0.5, 0.5)` variate itself (density.rs `lower_joint`). Since the
// container is a literal constructor and the index is a literal int,
// `flatten_structural` must resolve each `get0` to the projected element
// directly — no residual `(get0 ` over a literal vector may remain. Buffy
// #263 Pass 3.
#[test]
fn flatten_structural_resolves_static_get0_vector_projection() {
    let src = "\
d = joint(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 1.0, sigma = 2.0))
lp = logdensityof(d, [0.5, 0.5])";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        !pir.contains("(get0 "),
        "static get0(vector(...), i) must resolve to the projected element:\n{pir}"
    );
    assert!(
        pir.contains("(%field mu 0.0) (%field sigma 1.0)") && pir.matches("0.5").count() >= 2,
        "both density terms score 0.5 directly, not via a residual get0:\n{pir}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
}

// `flatten_structural` (isolated on the get0/vector fixture) is idempotent:
// determinizing the same source twice produces identical FlatPIR.
#[test]
fn flatten_structural_get0_is_idempotent() {
    let src = "\
d = joint(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 1.0, sigma = 2.0))
lp = logdensityof(d, [0.5, 0.5])";
    let once = flatppl_flatpir::write(&determinize_src(src));
    assert!(
        !once.contains("(get0 "),
        "sanity: fixture must actually contain a static get0 flatten_structural \
         resolves (else this test is vacuous):\n{once}"
    );
    let twice = flatppl_flatpir::write(&determinize_src(src));
    assert_eq!(
        once, twice,
        "determinize output is a canonicalization fixpoint"
    );
}

// §04 auto-splat (buffy #247) pulls each field of an opaque multi-output
// record call via `get(arg, "field")`. Once Pass 2 (`inline_user_calls`)
// beta-reduces the call away, the call site becomes a LITERAL
// `record(shape = 2.0, rate = 1.0)` — a static `get(record(...), "field")`
// that `flatten_structural` must resolve to the literal field value directly.
// Buffy #263 Pass 3.
#[test]
fn flatten_structural_resolves_static_get_record_projection() {
    let src = "\
gamma_shape_rate(mu, sigma) = record(shape = mu, rate = sigma)
a = draw(Gamma(gamma_shape_rate(2.0, 1.0)))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        !pir.contains("(get "),
        "static get(record(...), \"field\") must resolve to the literal field value:\n{pir}"
    );
    assert!(
        pir.contains("(%field shape 2.0)") && pir.contains("(%field rate 1.0)"),
        "Gamma's shape/rate are bound directly to the resolved literals:\n{pir}"
    );
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok());
}

// `flatten_structural` (isolated on the get/record fixture) is idempotent:
// determinizing the same source twice produces identical FlatPIR.
#[test]
fn flatten_structural_get_record_is_idempotent() {
    let src = "\
gamma_shape_rate(mu, sigma) = record(shape = mu, rate = sigma)
a = draw(Gamma(gamma_shape_rate(2.0, 1.0)))
lp = logdensityof(lawof(record(a = a)), record(a = 0.5))";
    let once = flatppl_flatpir::write(&determinize_src(src));
    assert!(
        !once.contains("(get "),
        "sanity: fixture must actually contain a static get flatten_structural \
         resolves (else this test is vacuous):\n{once}"
    );
    let twice = flatppl_flatpir::write(&determinize_src(src));
    assert_eq!(
        once, twice,
        "determinize output is a canonicalization fixpoint"
    );
}

// §11 "Literal values" forbids a signed atom. Folding `sub` to a negative
// real (here, inside `locscale`'s log-volume term) must emit `(neg 0.35)`,
// never a bare `-0.35`, or the printed FlatPIR fails to round-trip through
// `flatppl_flatpir::read` (the `bc7a1a3` reader refusal). Open [S] surfaced
// while closing M16.
#[test]
fn const_fold_negative_real_result_emits_neg_call_and_round_trips() {
    let src = "\
z ~ Normal(0.0, 1.0)
y ~ locscale(lawof(z), 1.0, 2.0)
lp = logdensityof(lawof(y), 0.3)";
    let pir = flatppl_flatpir::write(&determinize_src(src));
    assert!(
        pir.contains("(neg 0.35)"),
        "the folded sub(0.3, ...) result must be the canonical neg call:\n{pir}"
    );
    assert!(
        !pir.contains("-0.35"),
        "no signed atom may reach the printed FlatPIR:\n{pir}"
    );
    flatppl_flatpir::read(&pir).expect("printed FlatPIR must round-trip through the reader");
}

// Same defect, integer arithmetic: `sub` on two literal ints folding negative
// (`3 - 5`) must emit `(neg 2)`, not a bare `-2`.
#[test]
fn const_fold_negative_int_result_emits_neg_call_and_round_trips() {
    let src = "lp = 3 - 5";
    let pir = flatppl_flatpir::write(&determinize_src(src));
    assert!(
        pir.contains("(neg 2)"),
        "the folded sub(3, 5) result must be the canonical neg call:\n{pir}"
    );
    assert!(
        !pir.contains("-2"),
        "no signed atom may reach the printed FlatPIR:\n{pir}"
    );
    flatppl_flatpir::read(&pir).expect("printed FlatPIR must round-trip through the reader");
}

// A `neg` fold that would itself resurrect a signed atom must not be
// re-folded: `collect_folds`'s unary path skips replacing `(neg <positive>)`
// when the arithmetic result is negative (the exact shape this whole fold
// pass must produce), so a fixpoint is reached in one sweep instead of the
// pass rebuilding an identical `(neg <mag>)` node forever. Pin this directly
// against a source-level `-0.35` literal, which the parser already lowers to
// `neg(0.35)` before const-fold ever runs.
#[test]
fn const_fold_does_not_resurrect_a_signed_atom_from_a_source_neg_literal() {
    let src = "lp = -0.35";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("(neg 0.35)"),
        "a source-level -0.35 literal stays the canonical neg call:\n{pir}"
    );
    assert!(
        !pir.contains("-0.35"),
        "no signed atom may reach the printed FlatPIR:\n{pir}"
    );
    flatppl_flatpir::read(&pir).expect("printed FlatPIR must round-trip through the reader");
    // And determinizing twice must not loop or diverge: canonicalize already
    // ran to its fixpoint inside the first determinize.
    let twice = flatppl_flatpir::write(&determinize_src(src));
    assert_eq!(
        pir, twice,
        "determinize output is a canonicalization fixpoint"
    );
}

// A binding the `inputs` ABI promotes to a function argument keeps its
// references. §13's phase table maps the row *fixed / derived (e.g.
// `rnginit(seed)`) / listed in `inputs`* to "function argument, replacing the
// computed value", so `inputs = c` for `c = 2.0 * 3.0` declares a one-argument
// function. `resolve_alias_refs` used to inline the folded `6.0` into every
// consumer, which erased the argument: the emitter then saw `inputs = 6.0`,
// dropped the element, and baked the value at exit 0 (NEW-C1, the corrected
// root cause of RUST-STABLEHLO-001).
#[test]
fn a_promoted_input_is_not_inlined_into_its_consumers() {
    let src = "\
c = 2.0 * 3.0
m = lawof(record(a = draw(Normal(mu = c, sigma = 1.0))))
q = logdensityof(m, record(a = 0.5))
inputs = c
outputs = q";
    let pir = flatppl_flatpir::write(&determinize_src(src));
    assert!(
        pir.contains("(%ref self c)"),
        "the density must read the promoted binding, not its value:\n{pir}"
    );
    assert!(
        pir.contains("(%bind inputs (%ref self c))"),
        "the `inputs` declaration itself must survive as a reference:\n{pir}"
    );
}

// The same exemption is scoped to `inputs`: a trivial binding NOT promoted is
// still inlined, so the exemption does not disable alias resolution generally.
#[test]
fn an_unpromoted_trivial_binding_is_still_inlined() {
    let src = "\
p = elementof(reals)
c = 2.0 * 3.0
m = lawof(record(a = draw(Normal(mu = c, sigma = 1.0))))
q = logdensityof(m, record(a = p))
inputs = p
outputs = q";
    let pir = flatppl_flatpir::write(&determinize_src(src));
    assert!(
        !pir.contains("(%ref self c)"),
        "an unpromoted literal binding is still inlined:\n{pir}"
    );
}
