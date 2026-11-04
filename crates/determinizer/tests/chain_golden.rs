//! `markovchain` / `kscan` trajectory density lowering (spec §06 "Dependent
//! composition"). The trajectory density is the product of the step conditionals
//! from `init`, and `init` itself carries no density — it is a VALUE in the state
//! space, excluded from the trajectory, so a length-`n` chain gives exactly `n`
//! terms.

use flatppl_determinizer::determinize;

mod common;
use common::{pir_binding, renumber_points, strip_meta};

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn lower(src: &str) -> String {
    let out = determinize(&parse_infer(src)).expect("must lower, not refuse");
    let pir = renumber_points(&flatppl_flatpir::write(&out));
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

/// The `lp` binding with type annotations stripped.
fn lp(src: &str) -> String {
    strip_meta(&pir_binding(&lower(src), "lp"))
}

/// One density body over the chain's step pairs. Input 0 is the previous state,
/// read from `cat([init], v[0..n-2])`, input 1 is the current state `v`.
fn step_sum(
    body: &str,
    extra_inputs: &str,
    init: &str,
    v: &str,
    n: usize,
    extra_args: &str,
) -> String {
    format!(
        "(%bind lp (sum (broadcast (functionof {body} %specinputs \
         ((density_point_0 (%ref %local _density_point_0_)) \
         (density_point_1 (%ref %local _density_point_1_)){extra_inputs})) \
         (cat (vector {init}) (get0 {v} (indicesof0 (zeros {})))) {v}{extra_args})))",
        n - 1
    )
}

// A 3-step random walk. Step i is `Normal(prevᵢ, 1.0)` with prev₀ = init = 0.0
// and prevᵢ = the previous trajectory slot. The shifted array holds THREE previous
// states for n = 3, never four: §06 excludes `init` from the trajectory.
#[test]
fn markovchain_three_step_walk() {
    let src = "\
f = s -> Normal(s, 1.0)
t = draw(markovchain(f, 0.0, 3))
lp = logdensityof(lawof(t), [0.5, 1.5, 1.0])";
    let body = "(builtin_logdensityof Normal (record (%field mu (%ref %local _density_point_0_)) \
                (%field sigma 1.0)) (%ref %local _density_point_1_))";
    assert_eq!(
        lp(src),
        step_sum(body, "", "0.0", "(vector 0.5 1.5 1.0)", 3, "")
    );
}

// `kscan` threads a per-step exogenous input: step i is `κ(trajᵢ₋₁, xsᵢ)`. Here
// `dt` scales the step, so `dts` must map cell by cell beside the trajectory as
// input 2 — a held-constant or reordered `dts` would change the density.
#[test]
fn kscan_threads_per_step_input() {
    let src = "\
dts = [0.5, 2.0, 8.0]
f = (s, dt) -> Normal(s, dt)
t = draw(kscan(f, 0.25, dts))
lp = logdensityof(lawof(t), [0.5, 1.5, 1.0])";
    let body = "(builtin_logdensityof Normal (record (%field mu (%ref %local _density_point_0_)) \
                (%field sigma (%ref %local _density_point_2_))) (%ref %local _density_point_1_))";
    let extra = " (density_point_2 (%ref %local _density_point_2_))";
    assert_eq!(
        lp(src),
        step_sum(
            body,
            extra,
            "0.25",
            "(vector 0.5 1.5 1.0)",
            3,
            " (%ref self dts)"
        )
    );
}

// A named length resolves through inference's shape fold (`Level::Shape`), so a
// `markovchain` whose `n` is a binding lowers exactly as a literal one does.
#[test]
fn markovchain_named_length_lowers() {
    let named = "\
n = 4
f = s -> Normal(s, 1.0)
t = draw(markovchain(f, 0.0, n))
lp = logdensityof(lawof(t), [0.5, 1.5, 1.0, 2.0])";
    assert_eq!(lp(named), lp(&named.replace(", n))", ", 4))")));
    assert!(lp(named).contains("(zeros 3)"), "{}", lp(named));
}

// The step kernel may be a COMPOSED measure, not just a bare constructor: the step
// is handed back to the density dispatcher as a kernel application, so the body
// reaches its own combinator rule. `logweighted(ℓ, M)` adds `ℓ` to the base
// log-density (§06 "Density of composed measures"), once per step through the map.
#[test]
fn markovchain_composed_step_kernel_lowers() {
    let src = "\
f = s -> logweighted(0.25, Normal(s, 1.0))
t = draw(markovchain(f, 1.0, 2))
lp = logdensityof(lawof(t), [0.5, 1.5])";
    let body = "(add 0.25 (builtin_logdensityof Normal (record (%field mu \
                (%ref %local _density_point_0_)) (%field sigma 1.0)) (%ref %local _density_point_1_)))";
    assert_eq!(
        lp(src),
        step_sum(body, "", "1.0", "(vector 0.5 1.5)", 2, "")
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

// The supplied parameter stays constant across the trajectory. Its prior
// contributes once, while each transition scores at the supplied scale.
#[test]
fn a_chain_parameter_is_one_shared_value() {
    let src = "\
sigma_step ~ normalize(truncate(Cauchy(0.0, 1.0), interval(0.0, inf)))
step_kernel(prev, sigma) = Normal(mu = prev, sigma = sigma)
x ~ markovchain(step_kernel, 0.0, 3, sigma_step)
prior = lawof(record(sigma_step = sigma_step))
forward_kernel = kernelof(record(x = x), sigma_step = sigma_step)
L = likelihoodof(forward_kernel, record(x = [0.1, 0.2, 0.3]))
posterior = bayesupdate(L, prior)
lp = logdensityof(posterior, record(sigma_step = 0.7))";
    let lp = lp(src);
    // One step body, at the supplied scale. A per-step resample would need a
    // mapped scale input.
    let body = "(functionof (builtin_logdensityof Normal (record (%field mu \
                (%ref %local _density_point_0_)) (%field sigma 0.7)) (%ref %local _density_point_1_))";
    assert!(lp.contains(body), "{lp}");
    assert!(
        lp.contains("(cat (vector 0.0) (get0 (vector 0.1 0.2 0.3) (indicesof0 (zeros 2))))"),
        "{lp}"
    );
    // The scale prior is scored once, not per step.
    assert_eq!(
        lp.matches("builtin_logdensityof Cauchy").count(),
        1,
        "the scale prior is scored once:\n{lp}"
    );
}

// A chain inside `iid` sees its point as the map's bare cell. The previous
// states select a prefix of that cell before the concatenation, so `cat` joins
// two visible vectors (Mb_model, posteriordb).
#[test]
fn a_chain_inside_iid_concatenates_a_prefix_of_its_cell() {
    let src = "\
f = s -> Normal(s, 1.0)
t = draw(iid(markovchain(f, 0.0, 3), 2))
lp = logdensityof(lawof(t), [[0.5, 1.5, 1.0], [0.1, 0.2, 0.3]])";
    let lp = lp(src);
    assert!(
        lp.contains(
            "(cat (vector 0.0) (get0 (%ref %local _density_point_0_) (indicesof0 (zeros 2))))"
        ),
        "{lp}"
    );
}
