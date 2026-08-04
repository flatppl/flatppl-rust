//! Determiniser lowering of the §06 case-2 STRUCTURAL PROJECTION
//! `pushfwd(fn(get(_, [names])), M)` over INDEPENDENT INDEX-KEYED products — an
//! `iid`, a positional `joint`, or either wrapped in `relabel(_, [labels])` that
//! names their components. The marginal is closed-form: the sum of just the
//! SELECTED components' densities at the projected point; the unselected
//! (independent, normalized) components integrate to 1 and drop (§06 "joint and
//! iid (independent products)"). `relabel` supplies the field names an index-keyed
//! product lacks, so the projection reuses the existing keyword-joint marginal
//! machinery. `jointchain` is a DEPENDENT product, so only a dependency-respecting
//! LEADING PREFIX keep is closed-form; any other keep drops a component a later
//! kernel depends on, which is a `kchain` integral and refuses. Structural only
//! (flatppl-rust is not a density engine): assert the emitted FlatPDL term
//! structure.
//!
//! Two selector spellings reach the same marginal. A FIELD-NAME selector
//! `get(_, ["a", "c"])` addresses a field-keyed product (keyword `joint`,
//! record-of-draws, record-family `jointchain`) or an index-keyed product named by
//! `relabel`. An INDEX selector `get(_, [1, 3])` — `get` is 1-BASED, `get0` 0-based
//! (§07 `get0`) — addresses an index-keyed product's slots: a bare `iid`, a positional
//! `joint`, or a scalar-cat `jointchain`. Each spelling refuses over the other's keying.
mod common;
use common::pir_binding;
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
    // The by-LABEL spelling is the subject here. The equivalent integer spelling
    // over the same bare positional joint is `get(_, [1, 3])` — 1-based (§05) —
    // covered by `positional_joint_index_projection_keeps_nonadjacent_slots`.
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

// ---------------------------------------------------------------------------
// INDEX selectors over bare index-keyed products (§06 case 2's `joint` / `iid` /
// `jointchain` MUST, without a `relabel` to name the slots).
// ---------------------------------------------------------------------------

#[test]
fn iid_index_projection_keeps_only_the_selected_copy() {
    // `pushfwd(fn(get(_, [1])), iid(Normal(0,1), 3))` scored at the 1-vector [0.5].
    // The marginal is ONE copy's own law: logpdf(Normal(0,1), 0.5) =
    // -1.0439385332046727. Failing to marginalize would score all three copies,
    // 3 × that = -3.131815599614018 — the three-vs-one term count is what
    // discriminates, so assert the term count, not just presence.
    let p = pir("m = iid(Normal(mu = 0.0, sigma = 1.0), 3)\n\
         p = pushfwd(fn(get(_, [1])), m)\n\
         lp = logdensityof(p, [0.5])");
    let lp = pir_binding(&p, "lp");
    assert_eq!(
        lp.matches("builtin_logdensityof").count(),
        1,
        "one selected copy scored, the other two marginalized out:\n{lp}"
    );
    assert!(
        lp.contains("0.5"),
        "scored at the projected value 0.5:\n{lp}"
    );
}

#[test]
fn positional_joint_index_projection_keeps_the_selected_component() {
    // `pushfwd(fn(get(_, [1])), joint(Normal(0,1), Normal(1,2)))` at [0.5]. The
    // marginal is the FIRST component's own law, logpdf(Normal(0,1), 0.5) =
    // -1.0439385332046727. The two ways to get this wrong are distinguishable
    // here: scoring the WRONG component (Normal(1,2)) gives -1.643335713764618,
    // and failing to marginalize at all (the full joint) gives
    // -2.6872742469692907. Assert the count AND which component survived.
    let p = pir(
        "m = joint(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 1.0, sigma = 2.0))\n\
         p = pushfwd(fn(get(_, [1])), m)\n\
         lp = logdensityof(p, [0.5])",
    );
    let lp = pir_binding(&p, "lp");
    assert_eq!(
        lp.matches("builtin_logdensityof").count(),
        1,
        "only the selected component is scored:\n{lp}"
    );
    // Match the KEPT component's whole parameter record: a bare `"1.0"` would also
    // match the dropped component's `mu = 1.0`, so it discriminates nothing.
    assert!(
        lp.contains("(%field mu 0.0) (%field sigma 1.0)"),
        "the kept component is Normal(mu = 0.0, sigma = 1.0):\n{lp}"
    );
    assert!(
        !lp.contains("(%field mu 1.0) (%field sigma 2.0)"),
        "the dropped component Normal(mu = 1.0, sigma = 2.0) is marginalized out:\n{lp}"
    );
}

#[test]
fn positional_joint_index_projection_keeps_nonadjacent_slots() {
    // Keep slots {1, 3} of a 3-component positional joint, dropping the MIDDLE
    // Exponential. Distinct distributions lock the slot remap: the projected point
    // [0.1, 0.3] is scored at ITS OWN slots 0 and 1, so the kept Normal takes 0.1
    // and the kept Gamma takes 0.3 — logpdf(Normal(0,1), 0.1) +
    // logpdf(Gamma(shape=2, rate=1), 0.3) = -2.4279113375306087.
    let p = pir("j = joint(Normal(mu = 0.0, sigma = 1.0), \
                   Exponential(rate = 1.0), \
                   Gamma(shape = 2.0, rate = 1.0))\n\
         p = pushfwd(fn(get(_, [1, 3])), j)\n\
         lp = logdensityof(p, [0.1, 0.3])");
    let lp = pir_binding(&p, "lp");
    assert_eq!(
        lp.matches("builtin_logdensityof").count(),
        2,
        "exactly the two selected slots are scored:\n{lp}"
    );
    assert!(lp.contains("Normal"), "kept slot 1 (Normal) present:\n{lp}");
    assert!(lp.contains("Gamma"), "kept slot 3 (Gamma) present:\n{lp}");
    assert!(
        !lp.contains("Exponential"),
        "dropped middle slot 2 (Exponential) marginalized out:\n{lp}"
    );
}

#[test]
fn get0_index_projection_matches_the_one_based_get_spelling() {
    // `get` is 1-based, `get0` 0-based (§07 `get0`), so `get0(_, [0, 2])` selects
    // the same slots as `get(_, [1, 3])` and must emit the identical marginal.
    // Comparing the two lowerings pins the offset in ONE place: a recogniser that
    // ignored the head would make these differ by a slot.
    let one_based = pir("j = joint(Normal(mu = 0.0, sigma = 1.0), \
                   Exponential(rate = 1.0), \
                   Gamma(shape = 2.0, rate = 1.0))\n\
         p = pushfwd(fn(get(_, [1, 3])), j)\n\
         lp = logdensityof(p, [0.1, 0.3])");
    let zero_based = pir("j = joint(Normal(mu = 0.0, sigma = 1.0), \
                   Exponential(rate = 1.0), \
                   Gamma(shape = 2.0, rate = 1.0))\n\
         p = pushfwd(fn(get0(_, [0, 2])), j)\n\
         lp = logdensityof(p, [0.1, 0.3])");
    assert_eq!(
        pir_binding(&one_based, "lp"),
        pir_binding(&zero_based, "lp"),
        "get(_, [1, 3]) and get0(_, [0, 2]) select the same slots"
    );
}

#[test]
fn iid_index_projection_of_a_nonscalar_component_keeps_the_whole_row() {
    // `iid`'s variate has a LEADING REPEAT AXIS, so slot `j` is a whole
    // `M`-variate row — the projection of `iid(MvNormal(...), 3)` onto one slot is
    // that MvNormal itself, scored at the row [0.5, 0.5]. (A positional `joint`
    // has no such axis, which is why its own arm requires scalar components.)
    let p = pir(
        "m = iid(MvNormal(mu = [0.0, 1.0], sigma = [[1.0, 0.0], [0.0, 1.0]]), 3)\n\
         p = pushfwd(fn(get(_, [1])), m)\n\
         lp = logdensityof(p, [[0.5, 0.5]])",
    );
    let lp = pir_binding(&p, "lp");
    assert_eq!(
        lp.matches("builtin_logdensityof").count(),
        1,
        "one selected row scored:\n{lp}"
    );
    assert!(lp.contains("MvNormal"), "the row's own law:\n{lp}");
}

#[test]
fn scalar_cat_jointchain_index_prefix_keep_lowers() {
    // A scalar-cat `jointchain` (variate = the `cat` of its scalar draws, so it IS
    // index-keyed) with a 2-prefix keep {1, 2} over a 3-stage chain a → b(reads a)
    // → c(reads b). The dropped trailing c is a normalized Markov kernel that
    // integrates to 1, so the marginal is the sub-chain: logpdf(Normal(0,1), 0.3)
    // + logpdf(Exponential(rate = 0.3), 0.5) = -2.317911337530609. The kept
    // kernel's rate must bind to the kept slot's own value 0.3.
    let p = pir("a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         b = draw(Exponential(rate = a))\n\
         kb = kernelof(b, a = a)\n\
         kc = kernelof(draw(Gamma(shape = 2.0, rate = b)), b = b)\n\
         jc = jointchain(lawof(a), kb, kc)\n\
         lp = logdensityof(pushfwd(fn(get(_, [1, 2])), jc), [0.3, 0.5])");
    let lp = pir_binding(&p, "lp");
    assert_eq!(
        lp.matches("builtin_logdensityof").count(),
        2,
        "the 2-prefix keeps the base + b|a terms; trailing c dropped:\n{lp}"
    );
    assert!(lp.contains("Normal"), "base (Normal) present:\n{lp}");
    assert!(
        lp.contains("Exponential"),
        "kept b|a (Exponential) present:\n{lp}"
    );
    assert!(
        !lp.contains("Gamma"),
        "dropped trailing c (Gamma) marginalized out:\n{lp}"
    );
}

#[test]
fn scalar_cat_jointchain_index_nonprefix_keep_refuses() {
    // Same chain shape, keeping only slot {2} — dropping the leading a that b's
    // kernel READS. Marginalizing a out is the intractable `kchain` integral, not
    // a free trailing drop.
    let e = determinize(&parse_infer(
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         k = kernelof(draw(Exponential(rate = a)), a = a)\n\
         jc = jointchain(lawof(a), k)\n\
         lp = logdensityof(pushfwd(fn(get(_, [2])), jc), [0.3])",
    ))
    .expect_err("dropping the depended-upon leading slot must refuse");
    let msg = format!("{e:?}");
    assert!(msg.contains("refuse"), "must be a refusal: {msg}");
    assert!(msg.contains("kchain"), "names the kchain integral: {msg}");
}

#[test]
fn scalar_cat_jointchain_index_prefix_keep_improper_trailing_kernel_refuses() {
    // The improper-body guard applies to the index spelling too: dropping a
    // trailing kernel whose body is `Lebesgue(reals)` (∫ = ∞) would lower a finite
    // WRONG density. Inference types every `kernelof` as normalized regardless of
    // body, so the guard reads the kernel BODY's own mass.
    let e = determinize(&parse_infer(
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         kb = kernelof(draw(Lebesgue(support = reals)), a = a)\n\
         jc = jointchain(lawof(a), kb)\n\
         lp = logdensityof(pushfwd(fn(get(_, [1])), jc), [0.3])",
    ))
    .expect_err("dropping a trailing kernel with an improper body must refuse");
    let msg = format!("{e:?}");
    assert!(
        msg.contains("normalized"),
        "names the un-normalized dropped body: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Conformant refusals. §06 case 2 makes the closed form a MUST only over a
// measure with explicit product structure; everything else refuses, and the
// message must name what was missing.
// ---------------------------------------------------------------------------

#[test]
fn index_projection_over_a_nonproduct_measure_names_the_missing_product_structure() {
    // `MvNormal` has a vector variate but NO explicit product structure, so §06
    // case 2 permits a static error. The message must say the projection was
    // RECOGNISED and the base lacked product structure — not that the forward map
    // was unrecognised, which would send a reader looking for a `bijection`
    // annotation (the wrong fix for a non-invertible projection).
    let e = determinize(&parse_infer(
        "m = MvNormal(mu = [0.0, 1.0], sigma = [[1.0, 0.0], [0.0, 1.0]])\n\
         p = pushfwd(fn(get(_, [1])), m)\n\
         lp = logdensityof(p, [0.5])",
    ))
    .expect_err("a projection off a measure with no product structure must refuse");
    let msg = format!("{e:?}");
    assert!(
        msg.contains("product"),
        "message must name the missing product structure: {msg}"
    );
    assert!(
        !msg.contains("bijection") && !msg.contains("matrix-affine"),
        "must NOT be reported as an unrecognised forward map: {msg}"
    );
}

#[test]
fn get_index_zero_refuses_as_out_of_range() {
    // `get` is 1-BASED (§05 "FlatPPL uses 1-based indexing"), so `get(_, [0])`
    // selects no slot. Refuse naming the convention rather than silently reading it
    // as the first component — the 0-based spelling is `get0`.
    let e = determinize(&parse_infer(
        "m = iid(Normal(mu = 0.0, sigma = 1.0), 3)\n\
         p = pushfwd(fn(get(_, [0])), m)\n\
         lp = logdensityof(p, [0.5])",
    ))
    .expect_err("a 0 index under the 1-based `get` must refuse");
    let msg = format!("{e:?}");
    assert!(
        msg.contains("1-based"),
        "message must name the 1-based convention: {msg}"
    );
}

#[test]
fn duplicate_index_selection_refuses() {
    // `get(_, [1, 1])` would score the same component twice, double-counting its
    // density term.
    let e = determinize(&parse_infer(
        "j = joint(Normal(mu = 0.0, sigma = 1.0), Exponential(rate = 1.0))\n\
         p = pushfwd(fn(get(_, [1, 1])), j)\n\
         lp = logdensityof(p, [0.1, 0.1])",
    ))
    .expect_err("a duplicate selected index must refuse");
    let msg = format!("{e:?}");
    assert!(msg.contains("twice"), "names the double count: {msg}");
}

#[test]
fn index_projection_dropping_an_unnormalized_component_refuses() {
    // §06 case 2's free drop is an identity of a product of PROBABILITY measures:
    // `∫ M_dropped = 1`. A dropped `weighted(3.0, Normal(1,2))` integrates to 3,
    // so the marginal carries a log(3) factor the free drop would silently omit.
    let e = determinize(&parse_infer(
        "j = joint(Normal(mu = 0.0, sigma = 1.0), weighted(3.0, Normal(mu = 1.0, sigma = 2.0)))\n\
         p = pushfwd(fn(get(_, [1])), j)\n\
         lp = logdensityof(p, [0.5])",
    ))
    .expect_err("dropping a non-normalized component must refuse");
    let msg = format!("{e:?}");
    assert!(
        msg.contains("non-normalized"),
        "names the un-normalized dropped component: {msg}"
    );
}

#[test]
fn index_projection_over_a_keyword_joint_points_at_the_field_spelling() {
    // A keyword `joint` has a RECORD variate keyed by field name; an integer slot
    // does not address it. Refuse pointing at the by-name spelling.
    let e = determinize(&parse_infer(
        "j = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Normal(mu = 1.0, sigma = 2.0))\n\
         p = pushfwd(fn(get(_, [1])), j)\n\
         lp = logdensityof(p, [0.5])",
    ))
    .expect_err("an index selector over a field-keyed product must refuse");
    let msg = format!("{e:?}");
    assert!(
        msg.contains("record") && msg.contains("name"),
        "names the record variate and the by-name spelling: {msg}"
    );
}

#[test]
fn index_projection_over_a_relabeled_product_points_at_the_label_spelling() {
    // `relabel` gives an index-keyed product a RECORD variate (§06), so after a
    // relabel the projection is by LABEL, not by slot.
    let e = determinize(&parse_infer(
        "m = relabel(iid(Normal(mu = 0.0, sigma = 1.0), 3), [\"a\", \"b\", \"c\"])\n\
         p = pushfwd(fn(get(_, [1])), m)\n\
         lp = logdensityof(p, [0.5])",
    ))
    .expect_err("an index selector over a relabeled product must refuse");
    let msg = format!("{e:?}");
    assert!(
        msg.contains("record") && msg.contains("name"),
        "names the record variate and the by-name spelling: {msg}"
    );
}

#[test]
fn index_projection_over_a_nonscalar_joint_component_refuses() {
    // A positional `joint`'s variate is the flat `cat` of its components, so a slot
    // index names a component only when EVERY component is scalar. Two vector
    // components share a shape class (so inference accepts the joint) but each
    // occupies several `cat` slots — an index no longer names a component.
    let e = determinize(&parse_infer(
        "j = joint(iid(Normal(mu = 0.0, sigma = 1.0), 2), \
                   iid(Normal(mu = 1.0, sigma = 2.0), 2))\n\
         p = pushfwd(fn(get(_, [1])), j)\n\
         lp = logdensityof(p, [[0.1, 0.2]])",
    ))
    .expect_err("an index selector over a non-scalar-component joint must refuse");
    let msg = format!("{e:?}");
    assert!(
        msg.contains("scalar"),
        "names the scalar requirement: {msg}"
    );
}

#[test]
fn single_element_selector_refuses_naming_the_subset_form() {
    // §06 case 2's pattern is the SUBSET selector `get(_, [...])`. Single-element
    // access `get(_, 1)` projects onto one component and changes the variate KIND
    // (the point is the component's own variate, not a one-slot sub-product), so it
    // needs a different point mapping. Recognising it here buys the diagnostic: it
    // would otherwise be reported as an unrecognised forward map.
    let e = determinize(&parse_infer(
        "m = iid(Normal(mu = 0.0, sigma = 1.0), 3)\n\
         p = pushfwd(fn(get(_, 1)), m)\n\
         lp = logdensityof(p, 0.5)",
    ))
    .expect_err("single-element access is not the case-2 subset pattern");
    let msg = format!("{e:?}");
    assert!(
        msg.contains("SUBSET"),
        "message must point at the subset form: {msg}"
    );
}

#[test]
fn scalar_cat_jointchain_permuted_prefix_keep_refuses() {
    // The kept SET is the leading prefix {1, 2} but the ORDER is permuted. The
    // sub-chain is scored positionally — slot j of the point feeds component j — so
    // honouring the written order is not expressible here, and §07's subset
    // selection never states whether the result follows selector order or container
    // order. Measured before this refusal existed: `get(_, [2, 1])` at [0.5, 0.3]
    // emitted logpdf(N(0,1), 0.5) + logpdf(Exp(rate 0.5), 0.3) =
    // -1.887085713764618, where the ascending spelling at [0.3, 0.5] gives
    // -2.317911337530609. Two different numbers from the same selection — refuse.
    let e = determinize(&parse_infer(
        "a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         b = draw(Exponential(rate = a))\n\
         kb = kernelof(b, a = a)\n\
         kc = kernelof(draw(Gamma(shape = 2.0, rate = b)), b = b)\n\
         jc = jointchain(lawof(a), kb, kc)\n\
         lp = logdensityof(pushfwd(fn(get(_, [2, 1])), jc), [0.5, 0.3])",
    ))
    .expect_err("a permuted prefix keep has no settled marginal and must refuse");
    let msg = format!("{e:?}");
    assert!(
        msg.contains("PERMUTED"),
        "message must name the permuted order, not a non-prefix set: {msg}"
    );
}

#[test]
fn index_projection_at_a_scalar_point_refuses() {
    // The brief's own headline model with a SCALAR point. Every slot the index arm
    // emits is a `get0(v, j)`, so a scalar point would emit `get0(0.5, 0)` — and
    // inference raises nothing on the projected law, making the determiniser the
    // last gate. Refuse rather than emit ill-typed FlatPDL.
    let e = determinize(&parse_infer(
        "m = iid(Normal(mu = 0.0, sigma = 1.0), 3)\n\
         p = pushfwd(fn(get(_, [1])), m)\n\
         lp = logdensityof(p, 0.5)",
    ))
    .expect_err("a scalar point off a vector-variate projection must refuse");
    let msg = format!("{e:?}");
    assert!(
        !msg.contains("get0"),
        "must refuse, not report an emitted get0: {msg}"
    );
}

#[test]
fn index_projection_at_an_overlong_point_refuses() {
    // Selecting one component and scoring at a 2-vector: the second entry has no
    // slot in the sub-product, so lowering it would drop `0.7` with no diagnostic.
    let e = determinize(&parse_infer(
        "m = iid(Normal(mu = 0.0, sigma = 1.0), 3)\n\
         p = pushfwd(fn(get(_, [1])), m)\n\
         lp = logdensityof(p, [0.5, 0.7])",
    ))
    .expect_err("a point longer than the selection must refuse");
    let msg = format!("{e:?}");
    assert!(
        msg.contains("selects 1 component(s) but the query point has 2"),
        "message must name both lengths: {msg}"
    );
}

#[test]
fn unsupported_selector_forms_refuse_as_projections_not_as_unrecognised_maps() {
    // A pure `get` on the lambda's placeholder is ALWAYS a projection, never a
    // bijection, so a selector form this arm does not lower must still refuse AS a
    // projection. Reporting it as an unrecognised forward map would send a reader
    // after a `bijection` annotation, which no projection can carry.
    //
    // §07's multi-axis subset `get(A, [1, 3, 4], 2)`, the `all` / `only` axis
    // keywords, and a vector mixing integers with field names each reach here.
    let cases = [
        (
            "m = iid(Normal(mu = 0.0, sigma = 1.0), 3)\n\
             p = pushfwd(fn(get(_, [1], 2)), m)\n\
             lp = logdensityof(p, [0.5])",
            "multi-axis",
        ),
        (
            "m = iid(Normal(mu = 0.0, sigma = 1.0), 3)\n\
             p = pushfwd(fn(get(_, all)), m)\n\
             lp = logdensityof(p, [0.5, 0.7, 0.9])",
            "`all` axis keyword",
        ),
        (
            "m = iid(Normal(mu = 0.0, sigma = 1.0), 3)\n\
             p = pushfwd(fn(get(_, [1, \"a\"])), m)\n\
             lp = logdensityof(p, [0.5, 0.7])",
            "not all field names or all integer indices",
        ),
        (
            // Zero selector args is the opposite of multi-axis; the message
            // must not claim "more than one selector argument".
            "m = iid(Normal(mu = 0.0, sigma = 1.0), 3)\n\
             p = pushfwd(fn(get(_)), m)\n\
             lp = logdensityof(p, [0.5, 0.7, 0.9])",
            "no selector argument",
        ),
    ];
    for (src, expected) in cases {
        let e = format!(
            "{:?}",
            determinize(&parse_infer(src)).expect_err("an unsupported selector form must refuse")
        );
        assert!(
            e.contains("structural projection"),
            "must refuse AS a projection ({expected}): {e}"
        );
        assert!(
            e.contains(expected),
            "message must name the unsupported selector form: {e}"
        );
        assert!(
            !e.contains("matrix-affine") && !e.contains("bijection"),
            "must NOT be reported as an unrecognised forward map ({expected}): {e}"
        );
    }
}

// The REIFIED spelling of the same projection. `functionof(v -> get(v, [1]))` has no
// boundary input of its own — a CLOSED reification — and §04 forbids a nullary callable
// "as this would make them equivalent to known values", so it IS the lambda it wraps.
// The recogniser above reads a bare `fn` lambda head, so before the unwrap this spelling
// missed it and refused with a bijection-annotation misdiagnosis: `pushfwd bijection arg
// must be a bijection(f, f_inv, logvol) node`, which sends a reader after an annotation
// no projection can carry. No wrong number was ever emitted — the split was in spelling.
//
// Pinned as byte-identical emission against the plain spelling, plus the term count and
// the value, so neither the routing nor the marginal can drift alone.
#[test]
fn closed_reified_projection_lowers_identically_to_the_lambda_spelling() {
    let reified = "m = joint(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 1.0, sigma = 2.0))\n\
                   p = pushfwd(functionof(v -> get(v, [1])), m)\n\
                   lp = logdensityof(p, [0.5])";
    let plain = "m = joint(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 1.0, sigma = 2.0))\n\
                 p = pushfwd(v -> get(v, [1]), m)\n\
                 lp = logdensityof(p, [0.5])";
    let p_reified = pir(reified);
    let p_plain = pir(plain);
    // ONE term: component 1 is kept, component 2 integrates to 1 and drops.
    assert_eq!(
        pir_binding(&p_reified, "lp")
            .matches("builtin_logdensityof")
            .count(),
        1,
        "keeps exactly the selected component:\n{p_reified}"
    );
    // The kept component is `Normal(0, 1)` scored at 0.5. Its density is
    // -1.0439385332046727 (Distributions.jl: `logpdf(Normal(0.0, 1.0), 0.5)`), which the
    // dropped `Normal(1, 2)` discriminates against — scoring the WRONG component would
    // give -1.7370857137646181, and scoring both -3.351... .
    assert!(
        pir_binding(&p_reified, "lp").contains("(%field sigma 1.0)"),
        "the kept component is Normal(0, 1), not Normal(1, 2):\n{p_reified}"
    );
    assert_eq!(
        p_reified, p_plain,
        "reified and lambda spellings lower to identical FlatPDL:\nreified:\n{p_reified}\nlambda:\n{p_plain}"
    );

    // NESTED wrappers unwrap to a fixpoint. §04's rationale applies at every level, so
    // stopping after one layer would leave the doubly-reified spelling refusing with the
    // same map misdiagnosis one level up.
    let nested = "m = joint(Normal(mu = 0.0, sigma = 1.0), Normal(mu = 1.0, sigma = 2.0))\n\
                  p = pushfwd(functionof(functionof(v -> get(v, [1]))), m)\n\
                  lp = logdensityof(p, [0.5])";
    assert_eq!(
        pir(nested),
        p_plain,
        "a doubly-reified projection lowers identically too"
    );
}

// The unwrap is not projection-specific: it serves the BIJECTION path the same way. An
// affine forward map is the shape that shows it, since it exercises the
// change-of-variables lowering (the `- log(abs(2.0))` volume term) rather than the
// marginal. Refused at `48899b0`, byte-identical to the plain lambda now.
#[test]
fn closed_reified_affine_forward_map_lowers_identically_to_the_lambda_spelling() {
    let reified = "m = Normal(mu = 0.0, sigma = 1.0)\n\
                   p = pushfwd(functionof(x -> 2.0 * x + 1.0), m)\n\
                   lp = logdensityof(p, 0.5)";
    let plain = "m = Normal(mu = 0.0, sigma = 1.0)\n\
                 p = pushfwd(x -> 2.0 * x + 1.0, m)\n\
                 lp = logdensityof(p, 0.5)";
    let p_reified = pir(reified);
    assert!(
        pir_binding(&p_reified, "lp").contains("log"),
        "the change-of-variables volume term must be present:\n{p_reified}"
    );
    assert_eq!(
        p_reified,
        pir(plain),
        "reified and lambda affine spellings lower to identical FlatPDL:\n{p_reified}"
    );
}
