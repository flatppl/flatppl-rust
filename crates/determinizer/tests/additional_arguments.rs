//! Spec §04 "Explicit callback parameters" in the lowering rules: `weighted`,
//! `logweighted`, `pushfwd` and `markovchain` call their callback with the additional
//! arguments, bound positionally or by name, a sole record or table by its fields.
//! `kscan` traverses its sequence `xs` and passes a final `p` whole. Each case equals
//! the same model with the arguments written into the callback.
use flatppl_determinizer::determinize;

mod common;
use common::renumber_points;

fn pir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    renumber_points(&flatppl_flatpir::write(
        &determinize(&m).expect("must lower"),
    ))
}

#[test]
fn weighted_binds_several_positional_additional_arguments() {
    // log w(0.5) = log(2.0 * 0.5 + 1.0)
    let extras = pir(
        "W = weighted((x, a, b) -> a * x + b, Normal(0.0, 1.0), 2.0, 1.0)\nlp = logdensityof(W, 0.5)",
    );
    let inline =
        pir("W = weighted(x -> 2.0 * x + 1.0, Normal(0.0, 1.0))\nlp = logdensityof(W, 0.5)");
    assert_eq!(extras, inline);
}

#[test]
fn weighted_binds_keyword_and_record_additional_arguments() {
    let inline =
        pir("W = weighted(x -> 2.0 * x + 1.0, Normal(0.0, 1.0))\nlp = logdensityof(W, 0.5)");
    for extras in [
        "a = 2.0, b = 1.0",
        "b = 1.0, a = 2.0",
        "record(b = 1.0, a = 2.0)",
    ] {
        let got = pir(&format!(
            "W = weighted((x, a, b) -> a * x + b, Normal(0.0, 1.0), {extras})\nlp = logdensityof(W, 0.5)"
        ));
        assert_eq!(got, inline, "{extras}");
    }
}

#[test]
fn logweighted_binds_keyword_additional_arguments() {
    let extras = pir(
        "W = logweighted((x, a) -> a * x, Normal(0.0, 1.0), a = 2.0)\nlp = logdensityof(W, 0.5)",
    );
    let inline = pir("W = logweighted(x -> 2.0 * x, Normal(0.0, 1.0))\nlp = logdensityof(W, 0.5)");
    assert_eq!(extras, inline);
}

#[test]
fn pushfwd_binds_keyword_additional_arguments() {
    // The synthesised inverse and log-volume see the same additional arguments.
    let extras =
        pir("W = pushfwd((x, a) -> a * x, Normal(0.0, 1.0), a = 2.0)\nlp = logdensityof(W, 0.5)");
    let inline = pir("W = pushfwd(x -> 2.0 * x, Normal(0.0, 1.0))\nlp = logdensityof(W, 0.5)");
    assert_eq!(extras, inline);
}

#[test]
fn pushfwd_bijection_passes_the_additional_arguments_to_inverse_and_volume() {
    let extras = pir(
        "W = pushfwd(bijection((x, a, b) -> exp(x) * a + b, (y, a, b) -> log((y - b) / a), \
         (y, a, b) -> log(a)), Normal(0.0, 1.0), 2.0, 1.0)\nlp = logdensityof(W, 4.0)",
    );
    // log N(log(3 / 2); 0, 1) - log(2): the preimage is f_inv(4.0, 2.0, 1.0), the volume logvol(.., 2.0, 1.0).
    assert!(extras.contains("(log 1.5)"), "{extras}");
    assert!(extras.contains("(log 2.0)"), "{extras}");
}

#[test]
fn markovchain_binds_a_record_parameter() {
    // Step i is Normal(prev + drift, scale), prev_0 = 0.0.
    let want = pir(
        "T = markovchain(s -> Normal(s + 0.1, 0.5), 0.0, 3)\nlp = logdensityof(T, [0.1, 0.2, 0.3])",
    );
    // One record parameter, splatted onto the step's inputs or read whole.
    let got = pir(
        "T = markovchain((state, drift, scale) -> Normal(state + drift, scale), 0.0, 3, \
         record(scale = 0.5, drift = 0.1))\nlp = logdensityof(T, [0.1, 0.2, 0.3])",
    );
    assert_eq!(got, want);
    let got = pir(
        "T = markovchain((state, p) -> Normal(state + p.drift, p.scale), 0.0, 3, \
         p = record(scale = 0.5, drift = 0.1))\nlp = logdensityof(T, [0.1, 0.2, 0.3])",
    );
    assert_eq!(got, want);
}

#[test]
fn kscan_passes_a_vector_parameter_whole() {
    // §04 "Explicit callback parameters": `w` is the whole vector at every step, also
    // when it has the trajectory length or length one.
    for (w, sum) in [("[0.5, 0.25]", "sum([0.5, 0.25])"), ("[0.5]", "sum([0.5])")] {
        let got = pir(&format!(
            "T = kscan((state, x, w) -> Normal(state + x, sum(w)), 0.0, [1.0, 2.0], {w})\n\
             lp = logdensityof(T, [0.1, 0.2])"
        ));
        let inline = pir(&format!(
            "T = kscan((state, x) -> Normal(state + x, {sum}), 0.0, [1.0, 2.0])\n\
             lp = logdensityof(T, [0.1, 0.2])"
        ));
        assert_eq!(got, inline, "{w}");
    }
}

#[test]
fn kscan_reads_a_table_row_whole() {
    let out = pir(
        "T = kscan((state, row) -> Normal(state + row.x, row.w), 0.0, \
         table(x = [1.0, 2.0], w = [0.5, 0.25]))\nlp = logdensityof(T, [0.1, 0.2])",
    );
    assert!(
        out.contains("(vector 1.0 2.0)") && out.contains("(vector 0.5 0.25)"),
        "{out}"
    );
}

#[test]
fn fchain_passes_the_additional_arguments_to_every_step() {
    // ((1 + 2) * 2) - 2 = 4, and (1 + 2) * 2 = 6.
    for (source, value) in [
        (
            "f1(x, p) = x + p\nf2(y, p) = y * p\nf3(y, p) = y - p\nc = fchain(f1, f2, f3)\nr = c(1.0, 2.0)",
            "4.0",
        ),
        (
            "f1(x, p) = x + p\nf2(y, p) = y * p\nc = fchain(f1, f2)\nr = c(1.0, p = 2.0)",
            "6.0",
        ),
    ] {
        let out = pir(source);
        assert!(
            out.contains(&format!("(%bind r {value})")),
            "{source}\n{out}"
        );
    }
}

#[test]
fn kscan_holds_scalar_inputs_constant_at_every_step() {
    // Step i is Normal(prev + xs[i], w) with the constant w, prev_0 = 0.0.
    let inline = pir(
        "T = kscan((state, x) -> Normal(state + x, 0.5), 0.0, [1.0, 2.0])\n\
         lp = logdensityof(T, [0.1, 0.2])",
    );
    for extras in ["[1.0, 2.0], 0.5", "[1.0, 2.0], w = 0.5"] {
        let got = pir(&format!(
            "T = kscan((state, x, w) -> Normal(state + x, w), 0.0, {extras})\n\
             lp = logdensityof(T, [0.1, 0.2])"
        ));
        assert_eq!(got, inline, "{extras}");
    }
    // A record parameter reaches the kernel whole.
    let got = pir(
        "T = kscan((state, x, p) -> Normal(state + x + p.q, p.s), 0.0, [1.0, 2.0], \
         record(s = 0.5, q = 10.0))\nlp = logdensityof(T, [0.1, 0.2])",
    );
    let inline = pir(
        "T = kscan((state, x) -> Normal(state + x + 10.0, 0.5), 0.0, [1.0, 2.0])\n\
         lp = logdensityof(T, [0.1, 0.2])",
    );
    assert_eq!(got, inline);
}

#[test]
fn fchain_passes_a_fed_record_whole_when_it_has_additional_arguments() {
    // Spec §06: each later step "receives its fed value whole as its first argument".
    // f1 feeds record(a = 1, b = 10); f2(r, p) reads r.a + r.b + p = 1 + 10 + 10.
    let out = pir(
        "f1(x, p) = record(a = x, b = p)\nf2(r, p) = r.a + r.b + p\n\
         c = fchain(f1, f2)\nr = c(1.0, 10.0)",
    );
    assert!(out.contains("(%bind r") && out.contains("21.0"), "{out}");
    // Without additional arguments the fed record still splats onto `f2(a, b)`: 1 + 2.
    let out =
        pir("f1(x) = record(a = x, b = 2.0)\nf2(a, b) = a + b\nc = fchain(f1, f2)\nr = c(1.0)");
    assert!(out.contains("(%bind r") && out.contains("3.0"), "{out}");
}
