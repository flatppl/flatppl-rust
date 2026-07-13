//! Determiniser lowering of the §06 case-2 STRUCTURAL PROJECTION
//! `pushfwd(fn(get(_, [names])), M)` over INDEPENDENT INDEX-KEYED products — an
//! `iid`, a positional `joint`, or either wrapped in `relabel(_, [labels])` that
//! names their components. The marginal is closed-form: the sum of just the
//! SELECTED components' densities at the projected point; the unselected
//! (independent, normalized) components integrate to 1 and drop (§06 "joint and
//! iid (independent products)"). `relabel` supplies the field names an index-keyed
//! product lacks, so the projection reuses the existing keyword-joint marginal
//! machinery. `jointchain` (a DEPENDENT product) must still refuse — dropping a
//! component a downstream kernel depends on is a `kchain` integral, not a free
//! drop. Structural only (flatppl-rust is not a density engine): assert the
//! emitted FlatPDL term structure.
use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}
fn pir(src: &str) -> String {
    flatppl_flatpir::write(&determinize(&parse_infer(src)).expect("must lower"))
}

#[test]
fn relabel_iid_projection_marginalizes_dropped_component() {
    // The §06 canonical example (06-measure-algebra.md line 397):
    //   mu = relabel(iid(Normal(0,1), 3), ["a", "b", "c"])
    //   pushfwd(fn(get(_, ["a", "c"])), mu)   # marginalizes out b
    // The relabel names the 3 iid copies a, b, c; projecting {a, c} keeps the a
    // and c copies and marginalizes out the MIDDLE copy b. The marginal is
    // logdensityof(Normal, 0.1) + logdensityof(Normal, 0.3) — exactly TWO scored
    // terms (the iid copies are identical Normal(0,1), so the kept pair is
    // discriminated by the projected point's a=0.1 / c=0.3 values, and the
    // dropped b is discriminated by there being no third term / no field b).
    let p = pir(
        "mu = relabel(iid(Normal(mu = 0.0, sigma = 1.0), 3), [\"a\", \"b\", \"c\"])\n\
         p = pushfwd(fn(get(_, [\"a\", \"c\"])), mu)\n\
         lp = logdensityof(p, record(a = 0.1, c = 0.3))",
    );
    // Exactly TWO scored components (a and c), never three — b is marginalized out.
    assert_eq!(
        p.matches("builtin_logdensityof").count(),
        2,
        "marginal keeps exactly the two selected copies (a, c), b dropped:\n{p}"
    );
    // The kept copies are scored at the a=0.1 and c=0.3 projected-point values.
    assert!(p.contains("0.1"), "a-component scored at 0.1:\n{p}");
    assert!(p.contains("0.3"), "c-component scored at 0.3:\n{p}");
    // b never enters: had the wrong (shifted) pair been kept, scoring would have
    // demanded a field b from the {a, c} point and there would be a stray value
    // that isn't 0.1/0.3 — the two-term count above already fails a wrong keep.
}

#[test]
fn positional_joint_projection_lowers() {
    // A POSITIONAL `joint` (index-keyed, no field labels) named by `relabel`, then
    // projected to a non-adjacent subset {a, c}. Distinct component distributions
    // discriminate a correct keep from an off-by-one/positional shift: keeping
    // {a, c} must score the Normal (a) and the Gamma (c) and drop the MIDDLE
    // Exponential (b) — a shift would keep the Exponential or the wrong pair.
    //
    // The brief's integer-selector spelling `get(_, [0, 2])` is deliberately NOT
    // used: FlatPPL `get` integer indices are 1-BASED (`get0` is 0-based, §07), so
    // `[0, 2]` would be an off-by-one over a positional product — the required and
    // unambiguous §06 form is the relabel-named projection used here.
    let p = pir("j = joint(Normal(mu = 0.0, sigma = 1.0), \
                   Exponential(rate = 1.0), \
                   Gamma(shape = 2.0, rate = 1.0))\n\
         rj = relabel(j, [\"a\", \"b\", \"c\"])\n\
         pr = pushfwd(fn(get(_, [\"a\", \"c\"])), rj)\n\
         lp = logdensityof(pr, record(a = 0.1, c = 0.3))");
    assert_eq!(
        p.matches("builtin_logdensityof").count(),
        2,
        "marginal keeps the two selected positional components (a, c):\n{p}"
    );
    // The kept components are the Normal (a) and the Gamma (c); the dropped middle
    // is the Exponential (b) — assert the RIGHT components survived, not a shift.
    assert!(
        p.contains("Normal"),
        "kept component a (Normal) present:\n{p}"
    );
    assert!(
        p.contains("Gamma"),
        "kept component c (Gamma) present:\n{p}"
    );
    assert!(
        !p.contains("Exponential"),
        "middle component b (Exponential) marginalized out:\n{p}"
    );
}

#[test]
fn relabel_projection_keeps_correct_nonadjacent_indices() {
    // Non-adjacent multi-drop over a 4-component product: keep {a, c}, drop the
    // interior b AND the trailing d. Distinct distributions at every position
    // lock the index remap: a wrong (shifted) keep would surface Exponential (b)
    // or Beta (d), or miss Normal (a) / Gamma (c).
    let p = pir("j = joint(Normal(mu = 0.0, sigma = 1.0), \
                   Exponential(rate = 1.0), \
                   Gamma(shape = 2.0, rate = 1.0), \
                   Beta(alpha = 2.0, beta = 3.0))\n\
         rj = relabel(j, [\"a\", \"b\", \"c\", \"d\"])\n\
         pr = pushfwd(fn(get(_, [\"a\", \"c\"])), rj)\n\
         lp = logdensityof(pr, record(a = 0.1, c = 0.3))");
    assert_eq!(
        p.matches("builtin_logdensityof").count(),
        2,
        "marginal keeps exactly the two selected components (a, c):\n{p}"
    );
    assert!(p.contains("Normal"), "kept a (Normal) present:\n{p}");
    assert!(p.contains("Gamma"), "kept c (Gamma) present:\n{p}");
    assert!(
        !p.contains("Exponential"),
        "dropped interior b (Exponential) absent:\n{p}"
    );
    assert!(
        !p.contains("Beta"),
        "dropped trailing d (Beta) absent:\n{p}"
    );
}

#[test]
fn jointchain_projection_refuses() {
    // A `jointchain` relabeled to NAMES THAT DIFFER from its internal variate
    // fields (`x`/`y` vs the chain's `a`/`b`) is an ill-formed rename — inference
    // keeps the domain `record(a, b)`, so a point keyed by `x` does not match, and
    // the relabel cannot be aligned to the chain's dependency structure. Refuse
    // (naming jointchain) rather than guess a remap: only an IDENTITY relabel (or
    // a bare jointchain) exposes a well-defined prefix keep.
    let e = determinize(&parse_infer(
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         k = kernelof(record(b = draw(Normal(mu = a, sigma = 0.5))), a = a)\n\
         jc = jointchain(lawof(record(a = a)), k)\n\
         rj = relabel(jc, [\"x\", \"y\"])\n\
         pr = pushfwd(fn(get(_, [\"x\"])), rj)\n\
         lp = logdensityof(pr, record(x = 0.3))",
    ))
    .expect_err("projection over a jointchain (dependent product) must refuse");
    let msg = format!("{e:?}");
    assert!(msg.contains("refuse"), "must be a refusal: {msg}");
    assert!(
        msg.contains("jointchain"),
        "message must name jointchain as the unsupported dependent product: {msg}"
    );
}

#[test]
fn jointchain_prefix_keep_projection_lowers() {
    // A 2-stage jointchain `a → b` (b's kernel reads the base variate a). Keeping
    // the LEADING prefix {a} and dropping the trailing b is a dependency-
    // respecting prefix keep: b's kernel is a normalized Markov kernel that
    // integrates to 1 and drops cleanly, so the marginal is just the base density
    // `logdensityof(Normal(0,1), 0.3)`. The relabel is the IDENTITY (labels
    // `[a, b]` = the chain's variate fields), so it re-dispatches to the bare
    // jointchain prefix keep. Distinct distributions (base Normal, kernel
    // Exponential) make a wrong keep detectable — only the Normal term may
    // survive; the trailing Exponential (b) term must be absent.
    let p = pir("a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         k = kernelof(record(b = draw(Exponential(rate = a))), a = a)\n\
         jc = jointchain(lawof(record(a = a)), k)\n\
         m = relabel(jc, [\"a\", \"b\"])\n\
         lp = logdensityof(pushfwd(fn(get(_, [\"a\"])), m), record(a = 0.3))");
    assert_eq!(
        p.matches("builtin_logdensityof").count(),
        1,
        "prefix keep {{a}} scores only the base term; trailing b dropped:\n{p}"
    );
    assert!(p.contains("Normal"), "kept base (Normal) present:\n{p}");
    assert!(
        !p.contains("Exponential"),
        "dropped trailing b (Exponential) marginalized out:\n{p}"
    );
    assert!(
        p.contains("0.3"),
        "base scored at the projected value 0.3:\n{p}"
    );
}

#[test]
fn jointchain_prefix_keep_improper_trailing_kernel_refuses() {
    // A 2-stage jointchain a → b whose trailing kernel's BODY is an IMPROPER
    // (infinite-mass) measure `Lebesgue(reals)` — a reference measure, NOT a
    // probability measure. Keeping the leading prefix {a} and dropping the
    // trailing b is closed-form ONLY if the dropped kernel integrates to 1; here
    // ∫ Lebesgue(reals) = ∞, so the true marginal is φ(a)·∞, NOT φ(a). Inference
    // types EVERY kernelof(...) as Mass::Normalized regardless of body, so the
    // drop guard must NOT trust the kernel-TYPE mass — it must read the kernel
    // BODY's own measure mass and refuse an improper body rather than silently
    // lower to a finite WRONG density.
    let e = determinize(&parse_infer(
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         kb = kernelof(record(b = draw(Lebesgue(support = reals))), a = a)\n\
         jc = jointchain(lawof(record(a = a)), kb)\n\
         lp = logdensityof(pushfwd(fn(get(_, [\"a\"])), jc), record(a = 0.3))",
    ))
    .expect_err("dropping a trailing kernel with an improper (infinite-mass) body must refuse");
    let msg = format!("{e:?}");
    assert!(msg.contains("refuse"), "must be a refusal: {msg}");
    assert!(msg.contains("jointchain"), "names jointchain: {msg}");
}

#[test]
fn jointchain_nonprefix_keep_refuses() {
    // Same 2-stage chain, but keep only {b}, DROPPING the leading a. b's kernel
    // READS a, so marginalizing a out is the intractable `kchain` integral
    // `∫ densityof(K(a), b) dM(a)`, not a free trailing-suffix drop. {b} is not a
    // leading prefix — refuse rather than mislower.
    let e = determinize(&parse_infer(
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         k = kernelof(record(b = draw(Exponential(rate = a))), a = a)\n\
         jc = jointchain(lawof(record(a = a)), k)\n\
         m = relabel(jc, [\"a\", \"b\"])\n\
         lp = logdensityof(pushfwd(fn(get(_, [\"b\"])), m), record(b = 0.5))",
    ))
    .expect_err("dropping the depended-upon leading variate must refuse");
    let msg = format!("{e:?}");
    assert!(msg.contains("refuse"), "must be a refusal: {msg}");
    assert!(msg.contains("jointchain"), "names jointchain: {msg}");
    assert!(msg.contains("kchain"), "names the kchain integral: {msg}");
}

#[test]
fn jointchain_three_stage_prefix_keep_lowers() {
    // 3-stage chain `a → b(reads a) → c(reads b)`, bare (no relabel). Keep the
    // 2-prefix {a, b}, drop the trailing c: c's kernel is a normalized Markov
    // kernel integrating to 1, so the marginal is the sub-jointchain density over
    // {a, b} — the base Normal term + the b|a Exponential term; the trailing c
    // (Gamma) term is absent. Distinct distributions at every stage lock the keep.
    let p = pir("a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         b = draw(Exponential(rate = a))\n\
         kb = kernelof(record(b = b), a = a)\n\
         kc = kernelof(record(c = draw(Gamma(shape = 2.0, rate = b))), b = b)\n\
         jc = jointchain(lawof(record(a = a)), kb, kc)\n\
         lp = logdensityof(pushfwd(fn(get(_, [\"a\", \"b\"])), jc), record(a = 0.3, b = 0.5))");
    assert_eq!(
        p.matches("builtin_logdensityof").count(),
        2,
        "2-prefix {{a, b}} keeps the base + b|a terms; trailing c dropped:\n{p}"
    );
    assert!(p.contains("Normal"), "base (Normal) present:\n{p}");
    assert!(
        p.contains("Exponential"),
        "kept b|a (Exponential) present:\n{p}"
    );
    assert!(
        !p.contains("Gamma"),
        "dropped trailing c (Gamma) marginalized out:\n{p}"
    );
}

#[test]
fn jointchain_three_stage_middle_drop_refuses() {
    // Same 3-stage chain. Keep {a, c}, dropping the MIDDLE b. c's kernel READS b,
    // so dropping b is the intractable kchain integral, not a trailing-suffix
    // drop. {a, c} is not a leading prefix (b is interior) — refuse.
    let e = determinize(&parse_infer(
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         b = draw(Exponential(rate = a))\n\
         kb = kernelof(record(b = b), a = a)\n\
         kc = kernelof(record(c = draw(Gamma(shape = 2.0, rate = b))), b = b)\n\
         jc = jointchain(lawof(record(a = a)), kb, kc)\n\
         lp = logdensityof(pushfwd(fn(get(_, [\"a\", \"c\"])), jc), record(a = 0.3, c = 0.7))",
    ))
    .expect_err("dropping the depended-upon interior variate must refuse");
    let msg = format!("{e:?}");
    assert!(msg.contains("refuse"), "must be a refusal: {msg}");
    assert!(msg.contains("jointchain"), "names jointchain: {msg}");
}
