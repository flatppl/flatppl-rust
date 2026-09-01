//! `markovchain` / `kscan` trajectory density lowering (spec §06 "Dependent
//! composition"). The trajectory density is the product of the step conditionals
//! from `init`, and `init` itself carries no density — it is a VALUE in the state
//! space, excluded from the trajectory, so a length-`n` chain gives exactly `n`
//! terms.

use flatppl_determinizer::determinize;

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn lower(src: &str) -> String {
    let out = determinize(&parse_infer(src)).expect("must lower, not refuse");
    let pir = flatppl_flatpir::write(&out);
    assert!(
        flatppl_determinizer::is_flatpdl(&out).is_ok(),
        "is_flatpdl:\n{pir}"
    );
    assert!(
        !pir.contains("markovchain") && !pir.contains("kscan") && !pir.contains("(draw "),
        "measure layer gone:\n{pir}"
    );
    pir
}

// A 3-step random walk. Step i is `Normal(prevᵢ, 1.0)` with prev₀ = init = 0.0
// and prevᵢ = the previous trajectory slot, so the terms are
// Normal(0.0, 1)@0.5, Normal(0.5, 1)@1.5, Normal(1.5, 1)@1.0 — THREE terms for
// n = 3, never four: §06 excludes `init` from the trajectory.
#[test]
fn markovchain_three_step_walk() {
    let src = "\
f = s -> Normal(s, 1.0)
t = draw(markovchain(f, 0.0, 3))
lp = logdensityof(lawof(t), [0.5, 1.5, 1.0])";
    let pir = lower(src);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        3,
        "one term per trajectory element, none for init:\n{pir}"
    );
    // The first step's mean is `init`, not a trajectory slot.
    assert!(
        pir.contains("(%field mu 0.0)"),
        "step 1 mean is init = 0.0:\n{pir}"
    );
    // Later steps read the PREVIOUS slot.
    assert!(
        pir.contains("(%field mu 0.5)") && pir.contains("(%field mu 1.5)"),
        "steps 2 and 3 read the previous trajectory values:\n{pir}"
    );
}

// `kscan` threads a per-step exogenous input: step i is
// `κ(trajᵢ₋₁, xsᵢ)`. Here `dt` scales the step variance, so each term must pick
// up its OWN `dts` element in order — a swapped or reused index would give the
// same term count and a wrong density.
#[test]
fn kscan_threads_per_step_input() {
    let src = "\
dts = [0.5, 2.0, 8.0]
f = (s, dt) -> Normal(s, dt)
t = draw(kscan(f, 0.25, dts))
lp = logdensityof(lawof(t), [0.5, 1.5, 1.0])";
    let pir = lower(src);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        3,
        "one term per trajectory element:\n{pir}"
    );
    assert!(
        pir.contains("(%field mu 0.25)"),
        "step 1 mean is init = 0.25:\n{pir}"
    );
    // Each step reads its OWN `dts` slot, in order — a swapped or reused index
    // would keep the term count and change the density.
    for i in 0..3 {
        assert_eq!(
            pir.matches(&format!("(get0 (%ref self dts) {i})")).count(),
            1,
            "step {i} reads dts[{i}] exactly once:\n{pir}"
        );
    }
}

// A named length resolves through inference's shape fold (`Level::Shape`), so a
// `markovchain` whose `n` is a binding unrolls exactly as a literal one does.
#[test]
fn markovchain_named_length_unrolls() {
    let src = "\
n = 4
f = s -> Normal(s, 1.0)
t = draw(markovchain(f, 0.0, n))
lp = logdensityof(lawof(t), [0.5, 1.5, 1.0, 2.0])";
    let pir = lower(src);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        4,
        "a named length folds to 4 steps:\n{pir}"
    );
}

// The step kernel may be a COMPOSED measure, not just a bare constructor: each
// step is handed back to the density dispatcher as a kernel application, so the
// body reaches its own combinator rule. `logweighted(ℓ, M)` adds `ℓ` to the base
// log-density (§06 "Density of composed measures"), so each of the two steps
// contributes one constructor density plus its own weight term.
#[test]
fn markovchain_composed_step_kernel_lowers() {
    let src = "\
f = s -> logweighted(0.25, Normal(s, 1.0))
t = draw(markovchain(f, 1.0, 2))
lp = logdensityof(lawof(t), [0.5, 1.5])";
    let pir = lower(src);
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        2,
        "the composed step body lowers through its own logweighted rule:\n{pir}"
    );
    assert!(
        !pir.contains("logweighted"),
        "measure layer gone from the step body too:\n{pir}"
    );
    assert_eq!(
        pir.matches("0.25").count(),
        2,
        "each step picks up its own log-weight:\n{pir}"
    );
}

// A step kernel with the wrong arity is refused on the §06 signature rather than
// bound by a guessed position: `markovchain`'s kernel takes ONE input (the
// state), `kscan`'s takes TWO (state, exogenous input).
#[test]
fn markovchain_two_input_kernel_refuses() {
    let src = "\
f = (s, x) -> Normal(s + x, 1.0)
t = draw(markovchain(f, 0.0, 3))
lp = logdensityof(lawof(t), [0.5, 1.5, 1.0])";
    let err = determinize(&parse_infer(src)).expect_err("arity mismatch must refuse");
    assert!(
        err.construct.contains("markovchain"),
        "refusal names markovchain: {err:?}"
    );
    assert!(
        err.reason.contains("boundary input") && err.reason.contains("§06"),
        "refusal names the §06 signature: {err:?}"
    );
}

// A RECORD state makes the trajectory a table (§06: "If `init` and `traj[i]` are
// records, then the trajectories are tables, not arrays"), which inference leaves
// with a deferred domain — so there is no static array length to unroll and no
// `get0` slice to take. Refuse rather than treat the table as an array.
#[test]
fn markovchain_record_state_refuses() {
    let src = "\
f = s -> joint(a = Normal(s.a, 1.0))
t = draw(markovchain(f, record(a = 0.0), 3))
lp = logdensityof(lawof(t), [record(a = 0.5), record(a = 1.5), record(a = 1.0)])";
    let err = determinize(&parse_infer(src)).expect_err("a record-state chain must refuse");
    assert!(
        err.construct.contains("markovchain"),
        "refusal names markovchain: {err:?}"
    );
}
