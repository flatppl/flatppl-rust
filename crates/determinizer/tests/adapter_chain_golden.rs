//! A lambda adapter is a legal `jointchain` step (spec §04 "Calling conventions"
//! recommends `pars -> K(x = pars.x)` for a one-input kernel fed a record, and
//! spec §06 "Dependent composition" feeds a step the `cat` of every variate to
//! its left). Its output measure is the underlying kernel's with the inputs bound
//! to other arguments, so a prefix keep drops it like any normalized trailing
//! kernel, and the kept steps score at the slot values the adapter reads.
//!
//! The terms asserted are the closed form of the chain at the point, read off the
//! emitted FlatPDL: `Normal(0, 1)` at `a`, then `Exponential(rate = a)` at `b`,
//! then `Gamma(shape = 2, rate = b)` at `c`.
mod common;
use common::pir_binding;
use flatppl_determinizer::determinize;

fn pir(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    flatppl_flatpir::write(&determinize(&m).expect("must lower"))
}

const RECORD_CHAIN: &str = "\
z = draw(Normal(mu = 0.0, sigma = 1.0))\n\
k = kernelof(record(b = draw(Exponential(rate = z))), z = z)\n\
jc = jointchain(lawof(record(a = z)), pars -> k(z = pars.a))\n";

const SCALAR_CHAIN: &str = "\
a = draw(Normal(mu = 0.0, sigma = 1.0))\n\
b = draw(Exponential(rate = a))\n\
kb = kernelof(b, a = a)\n\
kc = kernelof(draw(Gamma(shape = 2.0, rate = b)), b = b)\n\
jc = jointchain(lawof(a), kb, pars -> kc(b = pars[2]))\n";

#[test]
fn a_record_chain_with_an_adapter_step_scores_each_component_at_its_slot() {
    let p = pir(&format!(
        "{RECORD_CHAIN}lp = logdensityof(jc, record(a = 0.3, b = 0.5))"
    ));
    let lp = pir_binding(&p, "lp");
    assert_eq!(lp.matches("builtin_logdensityof").count(), 2, "{lp}");
    assert!(
        lp.contains("builtin_logdensityof Normal"),
        "base term: {lp}"
    );
    assert!(lp.contains("(%field mu 0.0) (%field sigma 1.0)"), "{lp}");
    assert!(
        lp.contains("builtin_logdensityof Exponential") && lp.contains("(%field rate 0.3)"),
        "the adapter binds the kernel's input to the base slot 0.3: {lp}"
    );
}

#[test]
fn a_record_chain_prefix_keep_drops_an_adapter_step() {
    let p = pir(&format!(
        "{RECORD_CHAIN}lp = logdensityof(pushfwd(fn(get(_, [\"a\"])), jc), record(a = 0.3))"
    ));
    let lp = pir_binding(&p, "lp");
    assert_eq!(lp.matches("builtin_logdensityof").count(), 1, "{lp}");
    assert!(lp.contains("(%field mu 0.0) (%field sigma 1.0)"), "{lp}");
    assert!(
        !lp.contains("Exponential"),
        "dropped step must be absent: {lp}"
    );
}

#[test]
fn a_scalar_cat_chain_adapter_reads_its_slot_of_the_cat() {
    // The third step is fed the `cat` `[a, b]` whole; the adapter reads `b`.
    let p = pir(&format!(
        "{SCALAR_CHAIN}lp = logdensityof(jc, [0.3, 0.5, 0.7])"
    ));
    let lp = pir_binding(&p, "lp");
    assert_eq!(lp.matches("builtin_logdensityof").count(), 3, "{lp}");
    assert!(
        lp.contains("(%field rate 0.3)"),
        "Exponential at a = 0.3: {lp}"
    );
    assert!(lp.contains("Gamma"), "{lp}");
    assert!(
        lp.contains("(get "),
        "Gamma's rate reads a slot of the cat: {lp}"
    );
}

#[test]
fn a_scalar_cat_prefix_keep_drops_an_adapter_step() {
    let p = pir(&format!(
        "{SCALAR_CHAIN}lp = logdensityof(pushfwd(fn(get(_, [1, 2])), jc), [0.3, 0.5])"
    ));
    let lp = pir_binding(&p, "lp");
    assert_eq!(lp.matches("builtin_logdensityof").count(), 2, "{lp}");
    assert!(lp.contains("Normal") && lp.contains("Exponential"), "{lp}");
    assert!(
        !lp.contains("Gamma"),
        "dropped adapter step must be absent: {lp}"
    );
}
