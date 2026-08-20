//! The seven §07 heads the `missing-reductions` spec draft adds are DETERMINISTIC
//! ops, so the determiniser has nothing to do with them but carry them through.
//!
//! Pinned rather than assumed. §13 "Determinization" removes the measure layer and
//! leaves "deterministic ops plus the six `builtin_*` primitives", so a new
//! deterministic head needs no rule — but a head the determiniser silently DROPPED
//! or rewrote would show up nowhere else: the emitter tests all start from a
//! determinized module, so they would pass on whatever survived. These tests read
//! the emitted FlatPDL and check the head is still there, spelled the same, and
//! still carrying the type `infer` gave it.
//!
//! Normative source: flatppl-design branch `missing-reductions` @ `ee4c6fb`.

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let diags = flatppl_infer::infer(&mut m);
    assert!(diags.is_empty(), "infer diagnostics: {diags:?}");
    m
}

fn pdl(src: &str) -> String {
    let out = flatppl_determinizer::determinize(&parse_infer(src)).expect("must lower, not refuse");
    flatppl_flatpir::write(&out)
}

/// A model that reduces `head(v)` into a Normal's location, so the head sits inside
/// a real density query rather than as a bare output — the shape a determiniser rule
/// would rewrite if one existed.
fn model(head: &str, set: &str, call: &str) -> String {
    format!(
        "v = elementof(cartpow({set}, [4]))
z = {call}
lp = logdensityof(lawof(record(y = draw(Normal(mu = sum(z), sigma = 1.0)))), record(y = 1.0))
"
    )
    .replace("{head}", head)
}

/// All seven survive determinization by name. `median`/`quantile` included: the
/// StableHLO emitter refuses them, but that is a BACKEND limit — the determiniser
/// must still produce well-formed FlatPDL, or the refusal would come from the wrong
/// layer and a future backend with a sort would have nothing to lower.
#[test]
fn every_new_head_survives_determinization() {
    for (head, set, call) in [
        ("cummax", "reals", "cummax(v)"),
        ("cummin", "reals", "cummin(v)"),
        ("linfnorm", "reals", "linfnorm(v)"),
        ("median", "reals", "median(v)"),
        ("quantile", "reals", "quantile(v, 0.5)"),
    ] {
        let out = pdl(&model(head, set, call));
        assert!(
            out.contains(&format!("({head} ")),
            "`{head}` must survive determinization:\n{out}"
        );
    }
}

/// The boolean pair, whose result feeds `ifelse` rather than arithmetic — a boolean
/// cannot be a Normal's `mu` without §03's promotion, and this keeps the model in
/// §07's own domains.
#[test]
fn the_boolean_reductions_survive_determinization() {
    for head in ["lany", "lall"] {
        let src = format!(
            "b = elementof(cartpow(booleans, [4]))
z = ifelse({head}(b), 1.0, 2.0)
lp = logdensityof(lawof(record(y = draw(Normal(mu = z, sigma = 1.0)))), record(y = 1.0))
"
        );
        let out = pdl(&src);
        assert!(
            out.contains(&format!("({head} ")),
            "`{head}` must survive determinization:\n{out}"
        );
    }
}

/// The type must survive too, not just the name. A head that reached FlatPDL with
/// its `%meta` slot dropped would emit against the wrong ABI type — which is
/// exactly the disagreement `infer::ops::refuse_array_comparison` was added to
/// prevent on the comparison side.
#[test]
fn the_new_heads_carry_their_inferred_type_into_flatpdl() {
    let out = pdl(&model("cummax", "reals", "cummax(v)"));
    let line = out
        .lines()
        .find(|l| l.contains("(cummax "))
        .expect("a cummax line");
    assert!(
        line.contains("(%array 1 (4) (%scalar real))"),
        "`cummax` must keep its shape-preserving type in FlatPDL; got: {line}"
    );

    let src = "b = elementof(cartpow(booleans, [4]))
z = ifelse(lany(b), 1.0, 2.0)
lp = logdensityof(lawof(record(y = draw(Normal(mu = z, sigma = 1.0)))), record(y = 1.0))
";
    let out = pdl(src);
    let line = out
        .lines()
        .find(|l| l.contains("(lany "))
        .expect("a lany line");
    assert!(
        line.contains("(%scalar boolean)"),
        "`lany` must keep its boolean type in FlatPDL; got: {line}"
    );
}

/// A DOTTED comparison mask reaching `lany` must also survive — that is the input
/// §07 "Boolean reductions" gives the pair, and the determiniser's broadcast
/// handling is what carries it. The bare `gt(v, 3.0)` is not an alternative:
/// `infer` refuses it, because §07 gives the comparisons a scalar domain.
#[test]
fn a_dotted_mask_reaches_the_boolean_reduction_in_flatpdl() {
    let src = "v = elementof(cartpow(reals, [4]))
z = ifelse(lany(v .> 3.0), 1.0, 2.0)
lp = logdensityof(lawof(record(y = draw(Normal(mu = z, sigma = 1.0)))), record(y = 1.0))
";
    let out = pdl(src);
    // The dotted spelling reaches FlatPDL as `(broadcast gt …)` — §04's
    // higher-order form, with `gt` in the head SLOT rather than as its own call.
    assert!(
        out.contains("(lany ") && out.contains("(broadcast gt "),
        "the mask and its reduction must both survive:\n{out}"
    );
    // And the mask keeps the boolean-ARRAY type, so the emitter reduces `i1`
    // rather than the float dtype.
    assert!(
        out.contains("(%array 1 (4) (%scalar boolean)) %parameterized (cartpow booleans 4)"),
        "the mask must stay a boolean array in FlatPDL:\n{out}"
    );
}
