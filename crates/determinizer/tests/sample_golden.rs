use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

// rand(rng, lawof(record(x = draw(Normal)))) samples the one draw via
// builtin_sample, threads the rng, and eliminates the measure/stochastic layer.
#[test]
fn single_draw_samples_via_builtin_sample() {
    let src = "\
s = rnginit(0)
x = draw(Normal(mu = 0.0, sigma = 1.0))
draws = rand(s, lawof(record(x = x)))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("single-draw rand must lower to builtin_sample");
    let pir = flatppl_flatpir::write(&out);
    assert!(
        pir.contains("builtin_sample"),
        "emits builtin_sample:\n{pir}"
    );
    assert_eq!(
        pir.matches("builtin_sample").count(),
        1,
        "one sample per draw:\n{pir}"
    );
    assert!(
        !pir.contains("(draw ") && !pir.contains("(lawof ") && !pir.contains("(rand "),
        "measure/sample-surface layer eliminated:\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "must be FlatPDL:\n{pir}"
    );
    // Strengthened per the {pir} dump: the rng is threaded in as-is (no fresh
    // rngstate is fabricated), the constructor symbol survives bare (`Normal`,
    // not re-wrapped), and the sampled value is read out via `get0(sample, 0)`
    // (there is no separate `get1` primitive in this codebase — see
    // `sample::build_sample_term`).
    assert!(
        pir.contains("(builtin_sample (%ref self s) Normal ("),
        "builtin_sample threads the rng and carries the bare Normal ctor:\n{pir}"
    );
    assert!(
        pir.contains("(get0"),
        "sampled value is projected via get0(sample, 0):\n{pir}"
    );
    // The draw-binding `x` is now dead (the sampled value is a fresh inline
    // node, not a ref to `x`) and swept by `sweep_dead_measure_bindings`.
    assert!(
        pir.contains("(%bind x 0.0)"),
        "orphaned draw binding x is swept to a harmless literal:\n{pir}"
    );
}

// Two independent draws: two builtin_sample calls (field `a`, field `b`), the
// second consuming the first's advanced rng — sequential threading, not two
// fresh streams. There is no separate `get1` primitive in this codebase (see
// `sample::build_sample_term`); the advanced-rng slot is `get0(sample, 1)`.
//
// `a`'s `builtin_sample` node is referenced twice in the graph — once for its
// own sampled value (`get0(sample_a, 0)`, field `a`'s value) and once for the
// rng it hands to `b` (`get0(sample_a, 1)`, threaded into `sample_b`'s first
// arg). The FlatPIR writer is a plain tree printer with no common-subexpression
// sharing (`flatpir::writer` renders full subtrees at every reference site —
// see its module doc), so that shared node's `(builtin_sample (%ref self s)
// Normal ...)` text is re-expanded both places: 3 syntactic "builtin_sample"
// occurrences for 2 logical sample calls. That duplication is a printing
// artifact, not evidence of two independent (unthreaded) rng reads.
#[test]
fn two_independent_draws_thread_the_rng() {
    let src = "\
s = rnginit(0)
a = draw(Normal(mu = 0.0, sigma = 1.0))
b = draw(Normal(mu = 5.0, sigma = 1.0))
draws = rand(s, lawof(record(a = a, b = b)))";
    let m = parse_infer(src);
    let out = determinize(&m).expect("two draws must lower");
    let pir = flatppl_flatpir::write(&out);
    assert_eq!(
        pir.matches("builtin_sample").count(),
        3,
        "2 logical samples; the first's shared (value, rng) node is re-expanded \
         once more where its rng feeds the second sample (no CSE in the writer):\n{pir}"
    );
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "FlatPDL:\n{pir}"
    );
    // `a`'s sample reads the raw `s` rng directly — its call text
    // `(builtin_sample (%ref self s) Normal ...)` appears twice (its own
    // value and, re-expanded, inside `b`'s threaded rng argument).
    assert_eq!(
        pir.matches("(builtin_sample (%ref self s) Normal").count(),
        2,
        "a's sample seeds from the raw rng `s`, appearing at both its reference sites:\n{pir}"
    );
    // `b`'s sample does NOT read a bare/atomic rng argument (which would
    // render unwrapped, e.g. `(builtin_sample (%ref ...` or a literal) — its
    // first argument is the composite `get0(...)` projection of `a`'s
    // advanced rng, which the writer wraps in `%meta` because it is a
    // composite (non-atomic) expression (see `flatpir::writer`'s render_node
    // doc: atomic leaves render bare, composites get a `%meta` wrapper).
    assert_eq!(
        pir.matches("(builtin_sample (%meta").count(),
        1,
        "b's sample takes a composite (threaded) rng argument, not a bare ref:\n{pir}"
    );
    // And that composite argument is specifically a `get0` projection whose
    // own subject is `a`'s `builtin_sample` — i.e. `get0(sample_a, 1)`,
    // exactly the advanced-rng slot `build_sample_term` returns.
    assert!(
        pir.contains("(builtin_sample (%meta (%rngstate %fixed rngstates) (get0"),
        "b's rng argument is a get0(...) projection:\n{pir}"
    );
    let threaded_rng_start = pir
        .find("(builtin_sample (%meta (%rngstate %fixed rngstates) (get0")
        .expect("threaded rng argument located above");
    assert!(
        pir[threaded_rng_start..].contains("(builtin_sample (%ref self s) Normal"),
        "the get0(...) threaded into b's sample projects a's builtin_sample(s, ...):\n{pir}"
    );
}

// `lower_record_of_draws_sample`'s M1 guard: a POSITIONAL measure record
// (`record(a)`, no field name) has no key to fold field-by-field under, so it
// is not a field-keyed product — mirrors `density::match_independent_record`'s
// identical guard on the density side (see
// `refuse.rs::measure_record_with_positional_args_refuses`). Unlike the
// density-side companion guard for a non-`%field` named arg (which the parser
// can never produce inside `record(...)` — see `syntax::parser`'s hardcoded
// `NamedKind::Field` for `record`/`table`/`joint`/`jointchain`/`cartprod`
// heads — and so is untested here), this one IS reachable via valid surface
// syntax on the sample path too.
#[test]
fn positional_measure_record_sample_refuses() {
    let src = "\
s = rnginit(0)
a = draw(Normal(mu = 0.0, sigma = 1.0))
draws = rand(s, lawof(record(a)))";
    let m = parse_infer(src);
    let err = determinize(&m)
        .expect_err("a positional measure record is not a field-keyed product — refuse");
    assert!(
        err.reason.contains("field-keyed product"),
        "refusal explains the record is not field-keyed: {err:?}"
    );
}
