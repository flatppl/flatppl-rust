//! Determiniser synthesis of (f_inv, logvol) for known/invertible forward
//! functions in pushfwd density lowering (§06 case 1 + 3-bounded). Structural
//! only: assert the emitted change-of-variables FlatPIR, cross-checked against
//! the explicit-bijection form.
use flatppl_determinizer::determinize;
fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}
fn pir(src: &str) -> String {
    flatppl_flatpir::write(&determinize(&parse_infer(src)).expect("must lower"))
}

#[test]
fn pushfwd_bare_exp_lowers_like_explicit_bijection() {
    // Canonical LogNormal (§06 line 382). Must equal the explicit exp_bijection form.
    let synth = pir("ln = pushfwd(exp, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(ln, 0.5)");
    // Inline `bijection(...)` rather than a named `b = bijection(...)` binding:
    // the latter survives determinization as a dead `b = 0.0` binding, which
    // would break byte-equality for a reason unrelated to the change-of-variables.
    // The inline form is semantically identical and keeps binding structure equal,
    // so `assert_eq!` verifies the *whole* synthesized change-of-variables (incl.
    // the forward log-volume convention: exp ⇒ f_inv = log, logvol = identity).
    let explicit = pir(
        "ln = pushfwd(bijection(exp, log, x -> x), Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(ln, 0.5)",
    );
    assert!(synth.contains("builtin_logdensityof"), "got:\n{synth}");
    assert_eq!(
        synth, explicit,
        "synthesized exp must match explicit bijection(exp, log, id)"
    );
}
#[test]
fn pushfwd_eta_lambda_exp_lowers() {
    let p =
        pir("ln = pushfwd(x -> exp(x), Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(ln, 0.5)");
    assert!(
        p.contains("builtin_logdensityof") && p.contains("log"),
        "got:\n{p}"
    );
}
