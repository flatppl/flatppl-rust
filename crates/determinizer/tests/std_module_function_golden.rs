//! Lowering of a §09 STANDARD-module FUNCTION member call in a VALUE position.
//!
//! A function member is a catalogue entry with a closed form in §09, not a
//! distribution constructor, so it cannot take the kernel-tag path
//! `std_module_ctor_golden.rs` locks for a DISTRIBUTION member. Before
//! `determinizer::stdfn` it survived as a `CallHead::User` application and the
//! conformance gate refused it with the generic "residual user call".
//!
//! These are STRUCTURAL assertions: they check that the emitted subtree is base
//! ops and that FlatPDL conformance holds. flatppl-rust evaluates nothing, so the
//! NUMBERS are gated separately — each lowering in `stdfn.rs` was checked point by
//! point against an independent oracle (scipy.special for the `polynomials`
//! members, a 6×6 linear solve of §09's own C² conditions for the two degree-6
//! interpolators, closed form for the rest).

use flatppl_determinizer::{determinize, determinize_with_roots, is_flatpdl};

fn parse_infer(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let _ = flatppl_infer::infer(&mut m);
    m
}

/// Determinize `src` and return the emitted FlatPIR.
fn lower(src: &str) -> String {
    let out = determinize(&parse_infer(src)).expect("must lower, not refuse");
    let pir = flatppl_flatpir::write(&out);
    assert!(is_flatpdl(&out).is_ok(), "is_flatpdl:\n{pir}");
    pir
}

/// A model binding `probe` to `expr`, scored through a `Normal` so the module has a
/// query to determinize (a bare value binding gives the driver no measure to
/// eliminate).
fn probe_model(loads: &str, expr: &str) -> String {
    format!(
        "{loads}
probe = {expr}
sig = Normal(mu = probe, sigma = 1.0)
lp = logdensityof(lawof(sig), 0.0)"
    )
}

const HEP: &str = "hep = standard_module(\"particle-physics\", \"0.1\")";
const POLY: &str = "poly = standard_module(\"polynomials\", \"0.1\")";
const DIST: &str = "dist = standard_module(\"distances\", \"0.1\")";

// Every §09 `particle-physics` interpolator lowers, at a SYMBOLIC alpha so the
// branch structure survives constant folding. All five switch on alpha, so each
// emits an `ifelse`; none may leave a module-qualified ref behind.
#[test]
fn all_five_interpolators_lower_to_base_ops() {
    for member in [
        "interp_pwlin",
        "interp_pwexp",
        "interp_poly2_lin",
        "interp_poly6_lin",
        "interp_poly6_exp",
    ] {
        let src = probe_model(
            &format!("{HEP}\na = elementof(reals)"),
            &format!("hep.{member}(0.8, 1.0, 1.3, a)"),
        );
        let pir = lower(&src);
        assert!(
            pir.contains("ifelse"),
            "{member} must switch on alpha:\n{pir}"
        );
        assert!(
            !pir.contains(&format!("(%ref hep {member})")),
            "{member} must not survive as a module ref:\n{pir}"
        );
    }
}

// The two exponential-extrapolation interpolators reach `exp` and `log`; the
// linear-extrapolation ones must not (a stray transcendental would mean the wrong
// branch formula was emitted).
#[test]
fn only_the_exponential_interpolators_emit_exp_and_log() {
    let exp_members = ["interp_pwexp", "interp_poly6_exp"];
    let lin_members = ["interp_pwlin", "interp_poly2_lin", "interp_poly6_lin"];
    for member in exp_members {
        let src = probe_model(
            &format!("{HEP}\na = elementof(reals)"),
            &format!("hep.{member}(0.8, 1.0, 1.3, a)"),
        );
        let pir = lower(&src);
        assert!(pir.contains("(exp "), "{member} needs exp:\n{pir}");
        assert!(pir.contains("(log "), "{member} needs log:\n{pir}");
    }
    for member in lin_members {
        let src = probe_model(
            &format!("{HEP}\na = elementof(reals)"),
            &format!("hep.{member}(0.8, 1.0, 1.3, a)"),
        );
        let pir = lower(&src);
        assert!(
            !pir.contains("(exp ") && !pir.contains("(log "),
            "{member} extrapolates linearly and must stay transcendental-free:\n{pir}"
        );
    }
}

// §09 kinematics: `kallen` is a polynomial, `breakup_momentum` a pair of square
// roots over a product — both fully symbolic, no branch.
#[test]
fn kinematics_members_lower_to_base_ops() {
    let src = probe_model(
        &format!("{HEP}\nx = elementof(reals)\nmass = elementof(posreals)"),
        "hep.kallen(x, 1.0, 2.0) + hep.breakup_momentum(mass, 1.0, 0.5)",
    );
    let pir = lower(&src);
    assert!(
        !pir.contains("(%ref hep kallen)") && !pir.contains("(%ref hep breakup_momentum)"),
        "both members must be inlined:\n{pir}"
    );
    assert!(
        pir.contains("(sqrt "),
        "breakup_momentum needs sqrt:\n{pir}"
    );
}

// `blatt_weisskopf` selects its barrier polynomial by the LITERAL angular
// momentum, over §09's full declared range 0..=7. Only a literal (or a literal
// reached through one binding hop) lowers.
#[test]
fn blatt_weisskopf_lowers_for_every_declared_angular_momentum() {
    for ell in 0..=7 {
        let src = probe_model(
            &format!("{HEP}\np = elementof(posreals)"),
            &format!("hep.blatt_weisskopf({ell}, p, 1.5)"),
        );
        let pir = lower(&src);
        assert!(
            !pir.contains("(%ref hep blatt_weisskopf)"),
            "l = {ell} must lower:\n{pir}"
        );
    }
}

// §09 declares `blatt_weisskopf` for l <= 7 only, and FlatPPL has no control flow
// to pick a barrier polynomial at run time. Both an out-of-range literal and a
// genuinely non-constant l must REFUSE, naming the member.
#[test]
fn blatt_weisskopf_refuses_an_undefined_or_non_constant_angular_momentum() {
    for (label, prelude, arg) in [
        (
            "out of range",
            format!("{HEP}\np = elementof(posreals)"),
            "8",
        ),
        (
            "non-constant",
            format!("{HEP}\np = elementof(posreals)\nl = integer(p)"),
            "l",
        ),
    ] {
        let src = probe_model(&prelude, &format!("hep.blatt_weisskopf({arg}, p, 1.5)"));
        let err = determinize(&parse_infer(&src))
            .expect_err("an l this pass cannot read must refuse, not lower");
        assert!(
            format!("{err:?}").contains("blatt_weisskopf"),
            "the {label} refusal must name the member: {err:?}"
        );
    }
}

// The four `polynomials` members lower at a literal degree, emitted as the
// unrolled three-term recursion. Degree 0 collapses to the constant 1.
#[test]
fn polynomials_members_lower_at_a_literal_degree() {
    for member in ["legendre", "hermite", "laguerre", "chebyshev"] {
        for degree in [0, 1, 4] {
            let src = probe_model(
                &format!("{POLY}\nx = elementof(reals)"),
                &format!("poly.{member}({degree}, x)"),
            );
            let pir = lower(&src);
            assert!(
                !pir.contains(&format!("(%ref poly {member})")),
                "{member} degree {degree} must lower:\n{pir}"
            );
        }
    }
}

// A `polynomials` degree that is not a literal cannot select a recursion depth, so
// it must refuse rather than lower to some default.
#[test]
fn polynomials_member_refuses_a_non_literal_degree() {
    let src = probe_model(
        &format!("{POLY}\nx = elementof(reals)\nn = integer(x)"),
        "poly.legendre(n, x)",
    );
    let err = determinize(&parse_infer(&src)).expect_err("a non-literal degree must refuse");
    assert!(
        format!("{err:?}").contains("legendre"),
        "the refusal must name the member: {err:?}"
    );
}

// The five `distances` members with a §07 base-op form. §07 lists `mul` on two
// plain vectors as elementwise-free, so the inner products go through
// `transpose(u) * v`.
#[test]
fn scalar_distance_members_lower_to_norms_and_inner_products() {
    for (member, expect) in [
        ("euclidean", "l2norm"),
        ("squared_euclidean", "transpose"),
        ("manhattan", "l1norm"),
        ("chebyshev", "linfnorm"),
        ("cosine", "l2norm"),
    ] {
        let src = probe_model(
            &format!("{DIST}\nu = [1.0, 2.0, 3.0]\nv = [0.5, 1.5, 2.5]"),
            &format!("dist.{member}(u, v)"),
        );
        let pir = lower(&src);
        assert!(
            pir.contains(expect),
            "{member} should lower through {expect}:\n{pir}"
        );
        assert!(
            !pir.contains(&format!("(%ref dist {member})")),
            "{member} must not survive as a module ref:\n{pir}"
        );
    }
}

// A member call nested in another member call's ARGUMENT must lower too — the
// replacement carries arguments over unchanged, so the inner call only surfaces on
// the next pass of the fixed-point loop.
#[test]
fn a_member_call_nested_in_a_member_argument_lowers() {
    let src = probe_model(
        &format!("{HEP}\nx = elementof(reals)"),
        "hep.kallen(hep.kallen(x, 1.0, 2.0), 1.0, 2.0)",
    );
    let pir = lower(&src);
    assert!(
        !pir.contains("(%ref hep kallen)"),
        "both nesting levels must lower:\n{pir}"
    );
}

// A member reached through a binding hop (`f = hep.kallen; f(...)`) is the same
// callee shape `density::std_module_ctor_sym` follows for a distribution member.
// The now-dead `f` binding still holds the ref (nothing calls it), so the check is
// that the CALL is gone — `lower`'s `is_flatpdl` assertion is what proves that.
#[test]
fn a_member_reached_through_a_binding_hop_lowers() {
    let src = probe_model(
        &format!("{HEP}\nx = elementof(reals)\nf = hep.kallen"),
        "f(x, 1.0, 2.0)",
    );
    let pir = lower(&src);
    assert!(
        pir.contains("(mul "),
        "the hopped callee must inline to arithmetic:\n{pir}"
    );
}

// Refuse-don't-mislower, LOCATED: a §09 function member with no base-op form must
// refuse naming the member, not with the generic "residual user call". These are
// the members that need an engine primitive. The name is MODULE-qualified because
// `determinize` keeps every binding, so the alias's `standard_module` call is still
// there to read the module name off.
#[test]
fn a_member_without_a_base_op_form_refuses_naming_itself() {
    for (loads, expr, member) in [
        (
            "sp = standard_module(\"special-functions\", \"0.1\")",
            "sp.erf(0.5)",
            "special-functions.erf",
        ),
        (
            "sp = standard_module(\"special-functions\", \"0.1\")",
            "sp.bessel_j(1.0, 2.0)",
            "special-functions.bessel_j",
        ),
        (HEP, "hep.wignerd(1, 0, 0, 0.3)", "particle-physics.wignerd"),
        (
            HEP,
            "real(hep.resonance_breitwigner(1.0, 1.0, 0.1, 0.0, 0.0, 0, 1.5))",
            "particle-physics.resonance_breitwigner",
        ),
        (
            "dist = standard_module(\"distances\", \"0.1\")",
            "dist.minkowski([1.0, 2.0], [0.5, 1.0], 3.0)",
            "distances.minkowski",
        ),
    ] {
        let src = probe_model(loads, expr);
        let err =
            determinize(&parse_infer(&src)).expect_err("a member with no base-op form must refuse");
        let msg = format!("{err:?}");
        assert!(
            msg.contains(member),
            "the refusal must name `{member}`: {msg}"
        );
        assert!(
            msg.contains("needs an engine primitive"),
            "the refusal must say what is missing: {msg}"
        );
    }
}

// Root-based DCE runs between the lowering pass and the conformance check, so by
// then the alias's `standard_module` binding can be gone and the module name with
// it. The refusal must still name the member, falling back to the alias spelling
// the user wrote.
#[test]
fn the_located_refusal_survives_dce_dropping_the_alias_binding() {
    let src = probe_model(
        "sp = standard_module(\"special-functions\", \"0.1\")",
        "sp.erf(0.5)",
    );
    let mut m = parse_infer(&src);
    let root = m.intern("lp");
    let err = determinize_with_roots(&m, &flatppl_infer::ModuleBundle::new(), Some(&[root]))
        .expect_err("erf has no base-op form and must refuse");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("sp.erf") && msg.contains("needs an engine primitive"),
        "the refusal must still name the member after DCE: {msg}"
    );
}

// Numeric regression for the coefficient algebra, pinned as the CONSTANT-FOLDED
// literals the emitted tree collapses to at all-literal arguments. Each value was
// checked against an independent oracle before being pinned: `interp_poly6_lin`'s
// against a 6×6 linear solve of §09's own C² conditions, `blatt_weisskopf`'s
// against the spherical-Bessel identity behind §09's barrier polynomials, the rest
// against §09's closed forms evaluated directly. A change to the algebra that
// keeps the shape intact still moves these.
#[test]
fn folded_values_match_the_oracle_checked_constants() {
    let src = "\
hep = standard_module(\"particle-physics\", \"0.1\")
a = hep.interp_poly2_lin(0.8, 1.0, 1.3, 0.37)
d = hep.interp_poly6_lin(0.8, 1.0, 1.3, 0.37)
e = hep.kallen(3.0, 1.0, 2.0)
f = hep.breakup_momentum(3.0, 1.0, 0.5)
g = hep.blatt_weisskopf(2, 0.3, 1.5)
sig = Normal(mu = a + d + e + f + g, sigma = 1.0)
lp = logdensityof(lawof(sig), 0.0)";
    let pir = lower(src);
    // The two interpolators keep their (literal-conditioned) branch switch, so the
    // in-range value shows up as the innermost `ifelse` arm.
    for (label, value) in [
        ("interp_poly2_lin at alpha = 0.37", "1.099345"),
        ("interp_poly6_lin at alpha = 0.37", "1.1042111317451688"),
        // §11 "Literal values": no signed literal atom, so a negative folds to
        // `neg(<magnitude>)`.
        ("kallen(3, 1, 2)", "(neg 8.0)"),
        ("breakup_momentum(3, 1, 0.5)", "1.2808688457449497"),
        ("blatt_weisskopf(2, 0.3, 1.5)", "0.06519210230048154"),
    ] {
        assert!(
            pir.contains(value),
            "{label} should fold to {value}:\n{pir}"
        );
    }
}

// §11 "Literal values" gives a scalar literal no leading sign — a negated literal
// is the call `(neg <magnitude>)` — and the FlatPIR reader enforces it. Several of
// these lowerings carry negative constants (the `alpha < -1` bound, poly6_lin's
// `-10`, poly6_exp's `-24`/`-8`/`-7`/`-9`/`-5`/`-12`/`-3`), so every emitted module
// has to READ BACK, not just print.
#[test]
fn every_lowering_round_trips_through_the_flatpir_reader() {
    let cases = [
        (HEP, "hep.interp_pwlin(0.8, 1.0, 1.3, a)"),
        (HEP, "hep.interp_pwexp(0.8, 1.0, 1.3, a)"),
        (HEP, "hep.interp_poly2_lin(0.8, 1.0, 1.3, a)"),
        (HEP, "hep.interp_poly6_lin(0.8, 1.0, 1.3, a)"),
        (HEP, "hep.interp_poly6_exp(0.8, 1.0, 1.3, a)"),
        (HEP, "hep.kallen(a, 1.0, 2.0)"),
        (HEP, "hep.breakup_momentum(3.0 + a, 1.0, 0.5)"),
        (HEP, "hep.blatt_weisskopf(3, 0.3 + a, 1.5)"),
        (POLY, "poly.legendre(5, a)"),
        (POLY, "poly.hermite(4, a)"),
        (POLY, "poly.laguerre(4, a)"),
        (POLY, "poly.chebyshev(6, a)"),
        (DIST, "dist.euclidean([a, 2.0], [0.5, 1.5])"),
        (DIST, "dist.squared_euclidean([a, 2.0], [0.5, 1.5])"),
        (DIST, "dist.manhattan([a, 2.0], [0.5, 1.5])"),
        (DIST, "dist.chebyshev([a, 2.0], [0.5, 1.5])"),
        (DIST, "dist.cosine([a, 2.0], [0.5, 1.5])"),
    ];
    for (loads, expr) in cases {
        let src = probe_model(&format!("{loads}\na = elementof(reals)"), expr);
        let pir = lower(&src);
        flatppl_flatpir::read(&pir)
            .unwrap_or_else(|e| panic!("{expr} must read back:\n{e:?}\n{pir}"));
    }
}

// A §09 DISTRIBUTION member must keep taking the kernel-tag path: the function
// lowering must not claim it. Guards the gate `std_module_ctor_golden.rs` locks
// from the other side.
#[test]
fn a_distribution_member_is_untouched_by_the_function_lowering() {
    let src = "\
hep = standard_module(\"particle-physics\", \"0.1\")
sig = hep.CrystalBall(m0 = 5.279, sigma = 0.003, alpha = 1.5, n = 3.0)
lp = logdensityof(lawof(sig), 5.28)";
    let pir = lower(src);
    assert!(
        pir.contains("builtin_logdensityof CrystalBall"),
        "the distribution member still becomes a kernel tag:\n{pir}"
    );
}
