//! `markovchain` / `kscan` sampling lowers to one `stablehlo.while` whose carry
//! holds the rng key. Only the key after the last step is kept (spec §07
//! "Random number generation").

use flatppl_stablehlo::{EmitOptions, Mode};

fn emit(source: &str) -> Result<String, String> {
    let mut model = flatppl_syntax::parse(source).unwrap();
    let diagnostics = flatppl_infer::infer(&mut model);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let roots = [model.intern("inputs"), model.intern("outputs")];
    let model = flatppl_determinizer::determinize_with_options(
        &model,
        &flatppl_infer::ModuleBundle::new(),
        Some(&roots),
        &flatppl_stablehlo::LOWERING_OPTIONS,
    )
    .map_err(|e| e.reason)?;
    flatppl_stablehlo::emit(
        &model,
        Mode::Sample,
        &EmitOptions {
            restrict_enzyme_compatible: false,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())
}

fn assert_one_keyed_while(hlo: &str) {
    let loops: Vec<&str> = hlo
        .lines()
        .filter(|l| l.contains("stablehlo.while("))
        .collect();
    assert_eq!(loops.len(), 1, "{hlo}");
    assert!(loops[0].contains("tensor<2xui64>"), "{hlo}");
}

#[test]
fn kscan_sample_is_one_while_carrying_the_key() {
    let hlo = emit(
        "D = elementof(posreals)\ndts = [0.01, 0.02, 0.015, 0.018, 0.012]\n\
         W = kscan((x, dt, D) -> Normal(x, sqrt(2.0 * D * dt)), 0.0, dts, D)\n\
         s = rnginit(0)\nd, s2 = rand(W, s)\ninputs = (s, D)\noutputs = (d, s2)",
    )
    .unwrap();
    assert_one_keyed_while(&hlo);
    let signature = hlo
        .lines()
        .find(|l| l.contains("func.func @sample"))
        .unwrap();
    assert!(
        signature.contains("-> (tensor<5xf32>, tensor<2xui64>"),
        "{hlo}"
    );
}

/// The draw after the chain reads the final carried key, not a key from
/// inside the loop body.
#[test]
fn the_next_draw_reads_the_final_key() {
    let hlo = emit(
        "rho = elementof(interval(-1.0, 1.0))\n\
         mc = markovchain((x, rho) -> Normal(rho * x, 1.0), 0.5, 6, rho)\n\
         s = rnginit(0)\nd, s2 = rand(mc, s)\ny, s3 = rand(Normal(0.0, 1.0), s2)\n\
         inputs = (s, rho)\noutputs = (d, y, s3)",
    )
    .unwrap();
    assert_one_keyed_while(&hlo);
    let results = hlo
        .lines()
        .find(|l| l.contains("stablehlo.while("))
        .and_then(|l| l.trim().split(' ').next())
        .map(|name| name.split(':').next().unwrap().to_owned())
        .unwrap();
    let after = hlo.split("stablehlo.while(").nth(1).unwrap();
    let after = &after[after.rfind("stablehlo.return").unwrap()..];
    assert!(
        after
            .lines()
            .any(|l| l.contains("rng_bit_generator") && l.contains(&format!("{results}#"))),
        "{hlo}"
    );
}

#[test]
fn an_earlier_rng_state_of_a_scan_refuses() {
    let err = emit(
        "dts = [0.1, 0.2, 0.3]\ns = rnginit(0)\n\
         step = (c, dt) -> record(x = get0(rand(Normal(c.x, dt), c.r), 0), r = get0(rand(Normal(c.x, dt), c.r), 1))\n\
         traj = scan(step, record(x = 0.0, r = s), dts)\nfirst = get0(traj.r, 0)\n\
         inputs = s\noutputs = (traj.x, first)",
    )
    .unwrap_err();
    assert!(
        err.contains("a scan keeps only its last rng state"),
        "{err}"
    );
}

/// A fixed `rnginit` that seeds only the chain sits in the scan's initial
/// record, and still becomes the `%key` argument.
#[test]
fn a_fixed_seed_in_the_scan_carry_becomes_the_key_argument() {
    let hlo = emit(
        "rho = elementof(interval(-1.0, 1.0))\n\
         traj ~ markovchain((x, rho) -> Normal(rho * x, 1.0), 2.0, 6, rho)\nx = traj[6]\n\
         s = rnginit([0])\ndraws = rand(lawof(x), s)\ninputs = (rho)\noutputs = (draws)",
    )
    .unwrap();
    let signature = hlo
        .lines()
        .find(|l| l.contains("func.func @sample"))
        .unwrap();
    assert!(
        signature.contains("@sample(%key: tensor<2xui64>, %arg0: tensor<f32>)"),
        "{hlo}"
    );
}

/// The step reads the drawn value and the successor key from one sample. The
/// emitted body repeats that sample (see TODO-flatppl-rust.md, R9c), so every
/// draw in the loop body must read the same carried key: both slots then come
/// from one draw.
#[test]
fn the_value_slot_and_the_key_slot_read_one_draw() {
    let hlo = emit(
        "rho = elementof(interval(-1.0, 1.0))\n\
         mc = markovchain((x, rho) -> Normal(rho * x, 1.0), 0.5, 6, rho)\n\
         s = rnginit(0)\nd, s2 = rand(mc, s)\ninputs = (s, rho)\noutputs = (d, s2)",
    )
    .unwrap();
    let body = hlo.split("stablehlo.while(").nth(1).unwrap();
    let keys: std::collections::BTreeSet<&str> = body
        .lines()
        .filter(|l| l.contains("rng_bit_generator"))
        .map(|l| {
            l.split("rng_bit_generator ")
                .nth(1)
                .unwrap()
                .split(',')
                .next()
                .unwrap()
        })
        .collect();
    assert_eq!(keys.len(), 1, "{hlo}");
}
