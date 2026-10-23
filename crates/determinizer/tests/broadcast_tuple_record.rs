//! Spec §04 "Broadcasting": records are held constant for every call; tuple
//! operands are iterated component-wise, so `broadcast(f, (a, b), c)` is
//! `(f(a, c), f(b, c))`.
use flatppl_determinizer::determinize;

fn pir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    flatppl_flatpir::write(&determinize(&m).expect("must lower"))
}

#[test]
fn a_tuple_operand_unrolls_into_a_tuple_of_calls() {
    let f = "f(a, c) = a * c\n";
    // Literal components: (2*10, 3*10).
    for call in [
        "broadcast(f, (2.0, 3.0), 10.0)",
        "broadcast(f, a = (2.0, 3.0), c = 10.0)",
        "broadcasted(f)((2.0, 3.0), 10.0)",
    ] {
        let out = pir(&format!("{f}r = {call}\n"));
        assert!(out.contains("(tuple 20.0 30.0)"), "{call}\n{out}");
    }
    // Two tuple operands pair their components: (2*5, 3*7).
    let out = pir(&format!("{f}r = broadcast(f, (2.0, 3.0), (5.0, 7.0))\n"));
    assert!(out.contains("(tuple 10.0 21.0)"), "{out}");
}

#[test]
fn a_named_tuple_operand_is_read_component_by_component() {
    let source = "f(a, c) = a * c\nx = elementof(reals)\ny = elementof(reals)\n\
                  t = (x, y)\nr = broadcast(f, t, 10.0)\n";
    let explicit = "f(a, c) = a * c\nx = elementof(reals)\ny = elementof(reals)\n\
                    t = (x, y)\nr = (x * 10.0, y * 10.0)\n";
    assert_eq!(pir(source), pir(explicit));
}

#[test]
fn a_record_operand_is_held_constant_for_every_call() {
    let rec = "f(a, p) = a + p.s\nrec = record(s = 1.0, u = 2.0)\n";
    // Scoring and sampling both accept the record next to an array operand.
    let density = pir(
        "K(m, p) = Normal(mu = m, sigma = p.s)\nrec = record(s = 0.5, u = 2.0)\n\
         M = broadcast(K, [1.0, 2.0], rec)\nlp = logdensityof(M, [1.0, 2.0])
",
    );
    assert!(density.contains("builtin_logdensityof"), "{density}");
    let sample = pir(
        "K(m, p) = Normal(mu = m, sigma = p.s)\nrec = record(s = 0.5, u = 2.0)\n\
         M = broadcast(K, [1.0, 2.0], rec)\nout = rand(M, rnginit(7))\n",
    );
    assert!(sample.contains("builtin_sample"), "{sample}");
    pir(&format!("{rec}r = broadcast(f, [1.0, 2.0], rec)\n"));
}

#[test]
fn a_builtin_callee_unrolls_like_a_user_function() {
    // The callee is a bare builtin, not a user function: dot operators, a named
    // builtin and a dotted call all become a tuple of direct builtin calls.
    for (call, want) in [
        ("(2.0, 3.0) .+ 10.0", "(tuple 12.0 13.0)"),
        ("broadcast(add, (2.0, 3.0), (5.0, 7.0))", "(tuple 7.0 10.0)"),
        ("exp.((0.0, 1.0))", "(tuple (%meta"),
    ] {
        let out = pir(&format!("r = {call}\n"));
        assert!(out.contains(want), "{call}\n{out}");
        assert!(!out.contains("%call"), "{call}: residual user call\n{out}");
    }
    let out = pir("r = exp.((0.0, 1.0))\n");
    assert!(
        out.contains("(exp 0.0)") && out.contains("(exp 1.0)"),
        "{out}"
    );
}
