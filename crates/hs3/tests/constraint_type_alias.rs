//! ROOT's native-HS3 writer records a staterror's constraint *type* under the
//! key `constraint_type` (the pyhf / paper-appendix dialect uses `constraint`).
//! The importer must read both — else a Gaussian-constrained staterror is
//! silently mislowered to the Poisson default.
//!
//! End-to-end: a native-HS3 `staterror` with `constraint_type: "Gaussian"` must
//! yield a Normal auxiliary likelihood term, not a (Poisson) ContinuedPoisson.

const GAUSS_CONSTRAINT_TYPE: &str = r#"{
  "distributions": [
    { "name": "ch_model", "type": "histfactory_dist",
      "axes": [{ "name": "obs", "edges": [0.0, 1.0, 2.0] }],
      "samples": [
        { "name": "sig", "data": { "contents": [12.0, 11.0] },
          "modifiers": [{ "parameter": "mu", "type": "normfactor" }] },
        { "name": "bkg", "data": { "contents": [100.0, 100.0] },
          "modifiers": [{ "parameter": "gamma_stat", "type": "staterror",
                          "constraint_type": "Gaussian", "data": [5.0, 5.0] }] }
      ] }
  ],
  "data": [ { "name": "obs_data", "type": "binned", "contents": [110.0, 109.0] } ],
  "likelihoods": [ { "name": "lk", "distributions": ["ch_model"], "data": ["obs_data"] } ]
}"#;

#[test]
fn staterror_constraint_type_gaussian_lowers_to_normal() {
    let m = flatppl_hs3::read_hs3(GAUSS_CONSTRAINT_TYPE)
        .expect("native-HS3 `constraint_type` should be read like `constraint`");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    // The staterror aux uses Normal(gamma, delta) (Gaussian), not ContinuedPoisson.
    assert!(
        text.contains("Normal"),
        "Gaussian-constrained staterror must lower to a Normal aux, got:\n{text}"
    );
    assert!(
        !text.contains("ContinuedPoisson"),
        "must NOT fall back to the Poisson default, got:\n{text}"
    );
}
