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
    // `jointchain` is a DEPENDENT product: kernel `k` reads the base variate `a`,
    // so marginalizing a component is a `kchain` integral, not the free drop the
    // independent-product identity allows. Projection over it (even via `relabel`)
    // must REFUSE — only dependency-respecting prefix keeps are closed-form, a
    // bounded follow-up (§06 case 2 "report a static error").
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
