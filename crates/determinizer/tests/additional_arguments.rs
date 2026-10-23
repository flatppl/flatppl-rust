//! Spec §04 "Additional arguments" in the lowering rules: `weighted`, `logweighted`,
//! `pushfwd`, `markovchain` and `kscan` call their callback with the additional
//! arguments, bound positionally or by name, a sole record or table by its fields.
//! Each case equals the same model with the arguments written into the callback.
use flatppl_determinizer::determinize;

fn pir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    flatppl_flatpir::write(&determinize(&m).expect("must lower"))
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
fn markovchain_binds_several_additional_arguments() {
    // Step i is Normal(prev + drift, scale), prev_0 = 0.0.
    let want = pir(
        "T = markovchain(s -> Normal(s + 0.1, 0.5), 0.0, 3)\nlp = logdensityof(T, [0.1, 0.2, 0.3])",
    );
    for extras in [
        "0.1, 0.5",
        "drift = 0.1, scale = 0.5",
        "record(scale = 0.5, drift = 0.1)",
    ] {
        let got = pir(&format!(
            "T = markovchain((state, drift, scale) -> Normal(state + drift, scale), 0.0, 3, {extras})\n\
             lp = logdensityof(T, [0.1, 0.2, 0.3])"
        ));
        assert_eq!(got, want, "{extras}");
    }
}

#[test]
fn kscan_traverses_every_additional_sequence_together() {
    // Step i is Normal(prev + xs[i], ws[i]) with prev_0 = 0.0.
    let out = pir(
        "T = kscan((state, x, w) -> Normal(state + x, w), 0.0, [1.0, 2.0], [0.5, 0.25])\n\
         lp = logdensityof(T, [0.1, 0.2])",
    );
    assert!(
        out.contains("(%field mu 1.0) (%field sigma 0.5))) 0.1)"),
        "{out}"
    );
    assert!(
        out.contains("(%field mu 2.1) (%field sigma 0.25))) 0.2)"),
        "{out}"
    );
}

#[test]
fn kscan_repeats_a_length_one_sequence() {
    let out = pir(
        "T = kscan((state, x, w) -> Normal(state + x, w), 0.0, [1.0, 2.0], [0.5])\n\
         lp = logdensityof(T, [0.1, 0.2])",
    );
    assert!(
        out.contains("(%field mu 1.0) (%field sigma 0.5))) 0.1)"),
        "{out}"
    );
    assert!(
        out.contains("(%field mu 2.1) (%field sigma 0.5))) 0.2)"),
        "{out}"
    );
}

#[test]
fn kscan_binds_keyword_sequences_and_a_sole_table_by_name() {
    let positional = pir(
        "T = kscan((state, x, w) -> Normal(state + x, w), 0.0, [1.0, 2.0], [0.5, 0.25])\n\
         lp = logdensityof(T, [0.1, 0.2])",
    );
    for extras in [
        "w = [0.5, 0.25], x = [1.0, 2.0]",
        "table(x = [1.0, 2.0], w = [0.5, 0.25])",
    ] {
        let got = pir(&format!(
            "T = kscan((state, x, w) -> Normal(state + x, w), 0.0, {extras})\n\
             lp = logdensityof(T, [0.1, 0.2])"
        ));
        assert_eq!(got, positional, "{extras}");
    }
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
fn kscan_holds_scalar_and_record_inputs_constant_at_every_step() {
    // Step i is Normal(prev + xs[i], w) with the constant w, prev_0 = 0.0.
    for extras in [
        "[1.0, 2.0], 0.5",
        "[1.0, 2.0], w = 0.5",
        "x = [1.0, 2.0], w = 0.5",
    ] {
        let got = pir(&format!(
            "T = kscan((state, x, w) -> Normal(state + x, w), 0.0, {extras})\n\
             lp = logdensityof(T, [0.1, 0.2])"
        ));
        assert!(
            got.contains("(%field mu 1.0) (%field sigma 0.5))) 0.1)"),
            "{extras}\n{got}"
        );
        assert!(
            got.contains("(%field mu 2.1) (%field sigma 0.5))) 0.2)"),
            "{extras}\n{got}"
        );
    }
    // A constant record stays whole, and two constants beside one sequence bind in order.
    let got = pir(
        "T = kscan((state, x, p, q) -> Normal(state + x + q, p.w), 0.0, [1.0, 2.0], \
         record(w = 0.5), 10.0)\nlp = logdensityof(T, [0.1, 0.2])",
    );
    assert!(
        got.contains("(%field mu 11.0) (%field sigma 0.5))) 0.1)"),
        "{got}"
    );
    assert!(
        got.contains("(%field mu 12.1) (%field sigma 0.5))) 0.2)"),
        "{got}"
    );
}
