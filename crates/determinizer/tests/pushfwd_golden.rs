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

#[test]
fn pushfwd_affine_lambda_lowers() {
    // x -> 2*x + 1 : f_inv(y) = (y-1)/2, logvol = log(2) (constant).
    let p = pir(
        "d = pushfwd(x -> 2.0 * x + 1.0, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    );
    assert!(p.contains("builtin_logdensityof"), "got:\n{p}");
    // f_inv is the composed affine inverse (y-1)/2 = divide(sub(y, 1.0), 2.0)
    // (leaf substrings; the printer wedges `%meta` type annotations between ops):
    assert!(
        p.contains("(divide") && p.contains("(sub (%ref %local _x_) 1.0)"),
        "affine inverse (y-1)/2 present:\n{p}"
    );
    // logvol is the constant log|2| = log(abs(2)):
    assert!(p.contains("(abs 2.0)"), "logvol log(2) present:\n{p}");
}

#[test]
fn pushfwd_composition_exp_affine_lowers() {
    // x -> exp(2*x) : chain. f_inv(y) = log(y)/2 ; logvol = log(2) + 2x  (chain rule).
    let p = pir(
        "d = pushfwd(x -> exp(2.0 * x), Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    );
    assert!(
        p.contains("builtin_logdensityof") && p.contains("log"),
        "got:\n{p}"
    );
    // Composed inverse log(y)/2 = divide(log(y), 2.0):
    assert!(
        p.contains("(divide") && p.contains("(log (%ref %local _x_))"),
        "inverse log(y)/2 present:\n{p}"
    );
    // Chain-rule logvol: the exp term contributes the partial-forward 2x =
    // mul(2.0, x); the affine term contributes log|2| = log(abs(2)).
    assert!(
        p.contains("(mul 2.0 (%ref %local _x_))") && p.contains("(abs 2.0)"),
        "chain-rule logvol (2x + log 2) present:\n{p}"
    );
}

#[test]
fn pushfwd_noninvertible_lambda_refuses() {
    // x -> x*x is NOT injective on reals → refuse (recognized op, non-invertible here).
    let e = determinize(&parse_infer(
        "d = pushfwd(x -> x * x, Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(d, 0.5)",
    ))
    .expect_err("non-injective must refuse");
    assert!(format!("{e:?}").contains("refuse"), "got: {e:?}");
}
