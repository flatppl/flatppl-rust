//! The staterror constraint form when the modifier names none.
//!
//! pyhf: always Gaussian. Its staterror paramset is `constrained_by_normal` and
//! its workspace schema forbids a `constraint` field on the modifier, so there
//! is nothing else it could be.
//!
//! Native HS3: Poisson (Barlow-Beeston), ROOT's default. An explicit
//! `constraint` / `constraint_type` still wins on both paths
//! (`constraint_type_alias.rs`).

const PYHF: &str = r#"{
  "channels": [
    { "name": "c",
      "samples": [
        { "name": "sig", "data": [20.0, 10.0],
          "modifiers": [{ "name": "mu", "type": "normfactor", "data": null }] },
        { "name": "bkg", "data": [50.0, 60.0],
          "modifiers": [{ "name": "st", "type": "staterror", "data": [5.0, 6.0] }] }
      ] }
  ],
  "observations": [{ "name": "c", "data": [72.0, 68.0] }],
  "measurements": [{ "name": "m", "config": { "poi": "mu", "parameters": [] } }]
}"#;

const HS3: &str = r#"{
  "distributions": [
    { "name": "ch_model", "type": "histfactory_dist",
      "axes": [{ "name": "obs", "edges": [0.0, 1.0, 2.0] }],
      "samples": [
        { "name": "sig", "data": { "contents": [20.0, 10.0] },
          "modifiers": [{ "parameter": "mu", "type": "normfactor" }] },
        { "name": "bkg", "data": { "contents": [50.0, 60.0] },
          "modifiers": [{ "parameter": "st", "type": "staterror", "data": [5.0, 6.0] }] }
      ] }
  ],
  "data": [ { "name": "obs_data", "type": "binned", "contents": [72.0, 68.0] } ],
  "likelihoods": [ { "name": "lk", "distributions": ["ch_model"], "data": ["obs_data"] } ]
}"#;

#[test]
fn pyhf_staterror_defaults_to_gaussian() {
    let m = flatppl_hs3::read(PYHF).unwrap();
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    // sigma = sqrt(err^2)/nominal, per bin: 5/50 and 6/60.
    assert!(
        text.contains("st_delta = [0.1, 0.1]")
            && text.contains("functionof(broadcast(Normal, st, st_delta))"),
        "expected Normal(st, [0.1, 0.1]), got:\n{text}"
    );
    assert!(!text.contains("ContinuedPoisson"), "got:\n{text}");
}

#[test]
fn native_hs3_staterror_defaults_to_poisson() {
    let m = flatppl_hs3::read_hs3(HS3).unwrap();
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    // tau = nominal^2/err^2, per bin: 50^2/25 = 100 and 60^2/36 = 100.
    assert!(
        text.contains("st_tau = [100.0, 100.0]") && text.contains("hepphys.ContinuedPoisson"),
        "expected a ContinuedPoisson staterror aux, got:\n{text}"
    );
}
