//! Spec §06 "Joint measures": `joint(K1, K2, ...)` of kernels is a kernel that fans its input
//! out to every component and takes the union of their inputs by name, keyword-only. As a
//! `jointchain` step it receives the record of the preceding variates by the ordinary
//! calling convention, which splats over those inputs. The density is the product of the
//! base density and every component's conditional density.
//!
//! Oracles (closed-form Normal log-densities, scored through flatppl-js on the determinized
//! module): one input at (a = 0.5, p = 0.4, q = 0.9) is -2.2136684190540725
//! (-2.213668419054073 scored); two inputs at (a = 0.5, b = 0.25, p = 0.4, q = 0.9) is
//! -3.9210666328186905 (-3.921066632818691 scored).
use flatppl_determinizer::{determinize, is_flatpdl};

fn lower(src: &str) -> String {
    let mut m = flatppl_syntax::parse(src).unwrap();
    let errors: Vec<_> = flatppl_infer::infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == flatppl_infer::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    let out = determinize(&m).unwrap_or_else(|e| panic!("must lower: {e:?}\n{src}"));
    is_flatpdl(&out).unwrap();
    flatppl_flatpir::write(&out)
}

/// The terms `Normal(mu, sigma)` at `point`, as printed.
fn term(mu: &str, sigma: &str, point: &str) -> String {
    format!("(%field mu {mu}) (%field sigma {sigma}))) {point})")
}

fn assert_terms(pir: &str, terms: &[String]) {
    assert_eq!(
        pir.matches("builtin_logdensityof").count(),
        terms.len(),
        "{pir}"
    );
    for term in terms {
        assert!(pir.contains(term), "missing {term} in:\n{pir}");
    }
    assert!(
        !pir.contains("jointchain") && !pir.contains("joint "),
        "{pir}"
    );
}

#[test]
fn a_joint_kernel_of_one_input_is_a_chain_step() {
    // log N(0.5; 0, 1) + log N(0.4; a, 1) + log N(0.9; 2a, 0.5) with a = 0.5.
    let pir = lower(
        "x = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         a = elementof(reals)\n\
         K1 = kernelof(draw(Normal(mu = a, sigma = 1.0)))\n\
         K2 = kernelof(draw(Normal(mu = 2.0 * a, sigma = 0.5)))\n\
         J = joint(p = K1, q = K2)\n\
         jc = jointchain(lawof(record(a = x)), J)\n\
         lp = logdensityof(jc, record(a = 0.5, p = 0.4, q = 0.9))",
    );
    assert_terms(
        &pir,
        &[
            term("0.0", "1.0", "0.5"),
            term("0.5", "1.0", "0.4"),
            term("1.0", "0.5", "0.9"),
        ],
    );
}

#[test]
fn a_joint_kernel_of_two_inputs_is_a_chain_step() {
    // The components declare `a` and `a, b`; the joint takes both, keyword-only, and each
    // component reads only its own. log N(0.5; 0, 1) + log N(0.25; 1, 2) + log N(0.4; a, 1)
    // + log N(0.9; a + b, 0.5).
    let pir = lower(
        "x = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         y = draw(Normal(mu = 1.0, sigma = 2.0))\n\
         a = elementof(reals)\n\
         b = elementof(reals)\n\
         K1 = kernelof(draw(Normal(mu = a, sigma = 1.0)))\n\
         K2 = kernelof(draw(Normal(mu = a + b, sigma = 0.5)))\n\
         J = joint(p = K1, q = K2)\n\
         jc = jointchain(lawof(record(a = x, b = y)), J)\n\
         lp = logdensityof(jc, record(a = 0.5, b = 0.25, p = 0.4, q = 0.9))",
    );
    assert_terms(
        &pir,
        &[
            term("0.0", "1.0", "0.5"),
            term("1.0", "2.0", "0.25"),
            term("0.5", "1.0", "0.4"),
            term("0.75", "0.5", "0.9"),
        ],
    );
}

#[test]
fn a_joint_kernel_is_a_step_of_the_keyword_chain() {
    // The step's variate nests under its label: record(a = .., jj = record(p = .., q = ..)).
    let pir = lower(
        "x = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         a = elementof(reals)\n\
         K1 = kernelof(draw(Normal(mu = a, sigma = 1.0)))\n\
         K2 = kernelof(draw(Normal(mu = 2.0 * a, sigma = 0.5)))\n\
         J = joint(p = K1, q = K2)\n\
         jc = jointchain(a = lawof(x), jj = J)\n\
         lp = logdensityof(jc, record(a = 0.5, jj = record(p = 0.4, q = 0.9)))",
    );
    assert_terms(
        &pir,
        &[
            term("0.0", "1.0", "0.5"),
            term("0.5", "1.0", "0.4"),
            term("1.0", "0.5", "0.9"),
        ],
    );
}

#[test]
fn a_positional_joint_of_record_kernels_is_a_chain_step() {
    // The positional joint's variate is the `cat` of its components' records.
    let pir = lower(
        "x = draw(Normal(mu = 0.0, sigma = 1.0))\n\
         a = elementof(reals)\n\
         K1 = kernelof(record(p = draw(Normal(mu = a, sigma = 1.0))))\n\
         K2 = kernelof(record(q = draw(Normal(mu = 2.0 * a, sigma = 0.5))))\n\
         J = joint(K1, K2)\n\
         jc = jointchain(lawof(record(a = x)), J)\n\
         lp = logdensityof(jc, record(a = 0.5, p = 0.4, q = 0.9))",
    );
    assert_terms(
        &pir,
        &[
            term("0.0", "1.0", "0.5"),
            term("0.5", "1.0", "0.4"),
            term("1.0", "0.5", "0.9"),
        ],
    );
}
