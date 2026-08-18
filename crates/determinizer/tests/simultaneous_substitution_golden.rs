//! Boundary substitution is SIMULTANEOUS: applying `k(z = w, w = 0.5)` must not let the
//! `w := 0.5` entry rewrite the `w` the `z := w` entry just inserted.
//!
//! §04 *Specifying reification boundaries* substitutes each boundary node with its
//! applied value **inside the reified graph**, and an applied value is an ordinary
//! expression of the AMBIENT scope — it is not part of that graph, so nothing inside it
//! is itself a substitution target. Applying the entries one after another violates
//! that: the second pass cannot tell the first pass's insertion from source text.
//!
//! `kernel::reduce_kernel_application_bound` ran one `substitute_ref` per entry, so
//! every `Substitute::All` caller — `canon::inline`, the sampler's forward `pushfwd`
//! map, and the change-of-variables sites — could capture. The three repros below cover
//! the first two; each emitted a wrong value at exit 0 before the fix. The
//! change-of-variables site takes the same reduction and the same fix, but its own
//! record-variate reference-measure gate refuses every shape reached while composing a
//! repro, so no test stands on it.
//!
//! Oracles are closed-form and independently derived (scipy); the assertions read the
//! emitted FlatPDL's structure, never engine output.
use flatppl_determinizer::{determinize, determinize_with_roots};
use flatppl_infer::ModuleBundle;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

/// Lower with one requested output, so root-based DCE drops what nothing reaches and a
/// surviving `elementof` means the KEPT expression genuinely still reads it.
fn lower(src: &str, root: &str) -> String {
    let mut m = parse_infer(src);
    let root = m.intern(root);
    let out = determinize_with_roots(&m, &ModuleBundle::new(), Some(&[root])).expect("must lower");
    flatppl_syntax::print(&out)
}

/// `canon::inline` beta-reduces a residual user call through `Substitute::All`, and a
/// cross-named application is the plain deterministic shape of the defect.
///
/// `f = functionof(z + w, z = z, w = w)` applied at `f(z = w, w = 0.5)` is `w + 0.5`
/// with the ambient `w` surviving as a determinized input. Sequentially, `z := w` writes
/// `w` into the body and `w := 0.5` then rewrites that insertion, folding the whole
/// expression to the constant `1.0` — at ambient `w = 0` the query scores
/// `-0.9189385332046727` where the truth is `log N(1; 0.5, 1) = -1.0439385332046727`,
/// and `w` vanishes from the module entirely.
#[test]
fn canon_inline_does_not_capture_a_cross_named_applied_value() {
    let out = lower(
        "\
z = elementof(reals)
w = elementof(reals)
f = functionof(z + w, z = z, w = w)
r = f(z = w, w = 0.5)
lp = logdensityof(Normal(mu = r, sigma = 1.0), 1.0)",
        "lp",
    );
    assert!(
        out.contains("w + 0.5"),
        "`f(z = w, w = 0.5)` over `z + w` is `w + 0.5`, not a constant; got:\n{out}"
    );
    assert!(
        out.contains("w = elementof(reals)"),
        "the ambient `w` the applied value names must survive as a determinized input; \
         got:\n{out}"
    );
}

/// The cyclic swap, where no substitution ORDER is correct and only simultaneity is.
///
/// `f = functionof(2.0 * z + w, z = z, w = w)` at `f(z = w, w = z)` is `2.0 * w + z`, so
/// BOTH ambient parameters survive. Sequentially, `z := w` gives `2.0 * w + w` and
/// `w := z` then rewrites both, giving `2.0 * z + z` — `w` disappears and `z` is read
/// three times. At ambient `w = 1`, `z = 0.5` the truth is
/// `log N(1; 2.5, 1) = -2.0439385332046727` against the captured
/// `log N(1; 1.5, 1) = -1.0439385332046727`.
#[test]
fn canon_inline_does_not_capture_a_cyclic_swap() {
    let out = lower(
        "\
z = elementof(reals)
w = elementof(reals)
f = functionof(2.0 * z + w, z = z, w = w)
r = f(z = w, w = z)
lp = logdensityof(Normal(mu = r, sigma = 1.0), 1.0)",
        "lp",
    );
    assert!(
        out.contains("2.0 * w + z"),
        "a cyclic swap exchanges the two parameters exactly once; got:\n{out}"
    );
    assert!(
        out.contains("z = elementof(reals)") && out.contains("w = elementof(reals)"),
        "both ambient parameters survive a swap; got:\n{out}"
    );
}

/// The SAMPLER's `Substitute::All` site: `pushfwd`'s forward map, applied at the sampled
/// record by §04 auto-splat.
///
/// The map is `functionof(p * q, p = p, q = q)` and the base is
/// `lawof(record(p = d1, q = d2))` with `d1 ~ Normal(mu = q, sigma = 1.0)` — so the
/// value bound to input `p` (the `d1` sample) READS the module binding input `q`
/// targets. Sequentially, `p := <d1 sample>` inserts that read and `q := <d2 sample>`
/// then rewrites it, so `d1` is sampled at a mean of `d2`'s realized value instead of
/// the ambient `q`: a different measure, hence a wrong sample, at exit 0.
///
/// The discriminator is the sample COUNT. Two logical samples re-expand to three
/// syntactic occurrences here (the writer has no CSE and `d1`'s advanced rng seeds
/// `d2`), exactly as `pushfwd_over_independent_record_law_applies_map_to_both_draws`
/// documents. Capture nests a whole extra `d2` sample inside `d1`'s own mean, giving
/// five.
#[test]
fn a_pushfwd_sample_map_does_not_capture_a_cross_named_field() {
    let m = parse_infer(
        "\
s  = rnginit([42, 0, 0, 0])
p  = elementof(reals)
q  = elementof(reals)
d1 ~ Normal(mu = q, sigma = 1.0)
d2 ~ Normal(mu = 0.0, sigma = 1.0)
f  = functionof(p * q, p = p, q = q)
draws = rand(s, pushfwd(f, lawof(record(p = d1, q = d2))))",
    );
    let out = determinize(&m).expect("a record-splatting pushfwd map must lower");
    let printed = flatppl_syntax::print(&out);
    assert_eq!(
        printed.matches("builtin_sample").count(),
        3,
        "two logical samples, three occurrences (no CSE; d1's advanced rng seeds d2). \
         Capture nests a second d2 sample inside d1's own mean; got:\n{printed}"
    );
    assert!(
        printed.contains("record(mu = q, sigma = 1.0)") && !printed.contains("mu = get0("),
        "`d1` must be sampled at the AMBIENT `q`, not at `d2`'s realized value; \
         got:\n{printed}"
    );
}
