//! Density lowering of a §09 STANDARD-module distribution member reached as a
//! constructor CALL — `hep.CrystalBall(m0 = …, sigma = …, alpha = …, n = …)`.
//!
//! A standard-module member is a catalogue entry, not a submodule binding: there
//! is no subtree in the `ModuleBundle` to graft, so the cross-module graft paths
//! must DECLINE such a ref and leave the call to the primitive-constructor
//! lowering, which emits the BARE member name as the kernel tag. That is the same
//! tag `broadcast_golden.rs`'s module-qualified broadcast head already emits, and
//! the one both engines' kernel registries key.
//!
//! Before this, every standard-module member on the determinise path refused with
//! "cross-module ref could not be resolved against the module bundle", because
//! the alias's rhs is `standard_module(...)` rather than `load_module(...)`.

use flatppl_determinizer::{determinize, is_flatpdl};

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

fn determinize_src(src: &str) -> flatppl_core::Module {
    determinize(&parse_infer(src)).expect("must lower, not refuse")
}

// The direct shape: a §09 member constructor call bound to a name, scored through
// `logdensityof`. The kernel tag is the bare `CrystalBall`, its kernel input the
// member's own §09 parameter names.
#[test]
fn std_module_member_call_lowers_to_bare_member_kernel() {
    let src = "\
hep = standard_module(\"particle-physics\", \"0.1\")
sig = hep.CrystalBall(m0 = 5.279, sigma = 0.003, alpha = 1.5, n = 3.0)
lp = logdensityof(lawof(sig), 5.28)";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);

    assert!(
        pir.contains("builtin_logdensityof CrystalBall"),
        "kernel tag is the bare member name:\n{pir}"
    );
    assert!(
        pir.contains("%field m0") && pir.contains("%field sigma"),
        "kernel input keyed by the §09 parameter names:\n{pir}"
    );
    assert!(
        !pir.contains("(%ref hep CrystalBall)"),
        "the module-qualified ref must not survive as the kernel tag:\n{pir}"
    );
    assert!(!pir.contains("lawof"), "measure layer gone:\n{pir}");
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl:\n{pir}");
}

// Two DIFFERENT §09 members inside a mixture, each carrying its own parameter
// roster — the b_mass_peak corpus shape (`corpora/coverage/b_mass_peak` in
// flatppl-testsuite). `normalize` over weights summing to 1 is the identity, so
// each observation scores as `logsumexp(log f + logpdf_CB, log(1−f) + logpdf_Argus)`.
#[test]
fn std_module_members_lower_inside_a_normalized_mixture() {
    let src = "\
hep = standard_module(\"particle-physics\", \"0.1\")
f = elementof(unitinterval)
sig = hep.CrystalBall(m0 = 5.279, sigma = 0.003, alpha = 1.5, n = 3.0)
bkg = hep.Argus(resonance = 5.29, slope = -20.0, power = 0.5)
mix = normalize(superpose(weighted(f, sig), weighted(1.0 - f, bkg)))
lp = logdensityof(lawof(mix), 5.28)";
    let out = determinize_src(src);
    let pir = flatppl_flatpir::write(&out);

    assert!(
        pir.contains("builtin_logdensityof CrystalBall")
            && pir.contains("builtin_logdensityof Argus"),
        "both members lower to their own bare tag:\n{pir}"
    );
    assert!(
        pir.contains("logsumexp"),
        "the mixture is a log-sum-exp over the two weighted components:\n{pir}"
    );
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl:\n{pir}");
}

// Refuse-don't-mislower: a §09 module FUNCTION member is not a distribution
// constructor, so a call to one used as a measure must NOT become a density term.
// The same gate `broadcast_module_function_head_still_refuses` locks for the
// broadcast head, applied to the constructor-call shape.
#[test]
fn std_module_function_member_as_a_measure_still_refuses() {
    let src = "\
hep = standard_module(\"particle-physics\", \"0.1\")
k = hep.interp_pwlin(0.9, 1.0, 1.1, 0.5)
lp = logdensityof(lawof(k), 1.0)";
    let err = determinize(&parse_infer(src)).expect_err(
        "a §09 module FUNCTION member is not a distribution constructor and must refuse",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("primitive measure must be a built-in constructor call"),
        "the refusal should name that this is not a constructor call: {msg}"
    );
}
