//! `markovchain` / `kscan` sampling (spec §06 "Dependent composition", §07
//! "Random number generation"). Step `i` draws from the state step `i - 1`
//! drew, so the lowering is one `scan` whose carry holds the state and the rng
//! state. The trajectory is the state column, without `init`, and the successor
//! rng state is the carry after the last step.

use flatppl_determinizer::determinize;

fn lower(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    let out = determinize(&m).expect("must lower, not refuse");
    let pir = flatppl_flatpir::write(&out);
    assert!(flatppl_determinizer::is_flatpdl(&out).is_ok(), "{pir}");
    assert!(
        !pir.contains("markovchain") && !pir.contains("kscan") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    pir
}

fn refusal(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    determinize(&m).expect_err("must refuse").reason
}

#[test]
fn kscan_sample_carries_the_rng_through_one_scan() {
    let pir = lower(
        "D = elementof(posreals)\ndts = [0.01, 0.02, 0.015]\n\
         W = kscan((x, dt, D) -> Normal(x, sqrt(2.0 * D * dt)), 0.0, dts, D)\n\
         d, s2 = rand(W, rnginit(0))\ninputs = D\noutputs = (d, s2)",
    );
    assert_eq!(pir.matches("(scan ").count(), 1, "{pir}");
    assert!(pir.contains("builtin_sample"), "{pir}");
    // The trajectory length is 3, so the successor state is row 2.
    assert!(pir.contains("\"rng\")) 2)"), "{pir}");
}

#[test]
fn markovchain_sample_walks_an_index_vector() {
    let pir = lower(
        "rho = elementof(interval(-1.0, 1.0))\n\
         mc = markovchain((x, rho) -> Normal(rho * x, 1.0), 0.5, 6, rho)\n\
         d, s2 = rand(mc, rnginit(0))\ninputs = rho\noutputs = (d, s2)",
    );
    assert_eq!(pir.matches("(scan ").count(), 1, "{pir}");
    assert!(pir.contains("(zeros "), "{pir}");
}

/// A drawn init is sampled first, and the chain starts from its successor state.
#[test]
fn a_chain_in_a_record_law_starts_from_the_init_draw_state() {
    let pir = lower(
        "a ~ Normal(0.0, 1.0)\ntraj ~ markovchain(x -> Normal(0.9 * x, 1.0), a, 4)\n\
         d, s2 = rand(lawof(record(a = a, traj = traj)), rnginit(0))\noutputs = (d, s2)",
    );
    assert_eq!(pir.matches("(scan ").count(), 1, "{pir}");
    assert!(
        pir.contains(
            "(%field rng (%meta (%rngstate %fixed rngstates) (get0 (%ref self __sample_a) 1)))"
        ),
        "{pir}"
    );
}

#[test]
fn a_record_state_chain_sample_refuses() {
    let err = refusal(
        "step = s -> lawof(record(a = draw(Normal(s.a, 1.0))))\n\
         mc = markovchain(step, record(a = 0.0), 3)\n\
         d, s2 = rand(mc, rnginit(0))\noutputs = (d, s2)",
    );
    assert!(
        err.contains("trajectory length is not a statically-resolved"),
        "{err}"
    );
}
