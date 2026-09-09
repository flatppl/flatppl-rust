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

fn shared_staterror(first: Option<&str>, second: Option<&str>) -> String {
    let mut source: serde_json::Value = serde_json::from_str(HS3).unwrap();
    let samples = source["distributions"][0]["samples"]
        .as_array_mut()
        .unwrap();
    let mut other = samples[1].clone();
    other["name"] = "other".into();
    samples.push(other);
    for (sample, constraint) in samples[1..].iter_mut().zip([first, second]) {
        if let Some(constraint) = constraint {
            sample["modifiers"][0]["constraint_type"] = constraint.into();
        }
    }
    source.to_string()
}

#[test]
fn shared_staterror_rejects_incompatible_constraint_families() {
    for (first, second) in [("Gauss", "Poisson"), ("Poisson", "Gauss")] {
        flatppl_hs3::read_hs3(&shared_staterror(Some(first), Some(second)))
            .expect_err("a shared parameter cannot have incompatible auxiliary families");
    }
}

#[test]
fn shared_staterror_accepts_equivalent_constraint_families() {
    for (first, second, family) in [
        (Some("Gauss"), "Gaussian", "broadcast(Normal, st, st_delta)"),
        (None, "Poisson", "hepphys.ContinuedPoisson"),
    ] {
        let module = flatppl_hs3::read_hs3(&shared_staterror(first, Some(second))).unwrap();
        let text = flatppl_syntax::print_with(&module, flatppl_syntax::Syntax::Minimal);
        assert!(text.contains(family), "{text}");
    }
}

/// `HS3` with `constraint_type` set on its one staterror modifier.
fn constrained_staterror(constraint: &str) -> String {
    let mut source: serde_json::Value = serde_json::from_str(HS3).unwrap();
    source["distributions"][0]["samples"][1]["modifiers"][0]["constraint_type"] = constraint.into();
    source.to_string()
}

#[test]
fn an_unknown_staterror_constraint_type_is_refused() {
    // A misspelling used to fall through to the Poisson family, so `Poison`
    // emitted a ContinuedPoisson aux and exited 0.
    for constraint in ["Poison", "Gaus", "gauss", "poisson", "Normal", ""] {
        let err = flatppl_hs3::read_hs3(&constrained_staterror(constraint))
            .expect_err("an unlisted constraint type must be refused, not defaulted");
        let message = err.to_string();
        assert!(
            message.contains("unknown staterror constraint type") && message.contains(constraint),
            "the error must name the rejected spelling: {message}"
        );
    }
}

#[test]
fn a_known_but_unlowered_staterror_constraint_type_is_refused() {
    // HS3 defines LogNormal (ROOT spells it Lognormal) and ROOT adds Const.
    // Both are valid source documents this importer does not lower, so they
    // take the `Unimplemented` marker the conversion testsuite reads as a skip.
    for constraint in ["LogNormal", "Lognormal", "Const"] {
        let err = flatppl_hs3::read_hs3(&constrained_staterror(constraint))
            .expect_err("an unlowered constraint type must be refused, not defaulted");
        let message = err.to_string();
        assert!(
            message.starts_with("unimplemented HS3 construct: ") && message.contains(constraint),
            "the error must mark the gap and name the type: {message}"
        );
    }
}

#[test]
fn both_known_staterror_constraint_families_still_import() {
    for (constraint, family, other) in [
        (
            "Gauss",
            "broadcast(Normal, st, st_delta)",
            "ContinuedPoisson",
        ),
        (
            "Gaussian",
            "broadcast(Normal, st, st_delta)",
            "ContinuedPoisson",
        ),
        (
            "Poisson",
            "hepphys.ContinuedPoisson",
            "broadcast(Normal, st,",
        ),
    ] {
        let module = flatppl_hs3::read_hs3(&constrained_staterror(constraint)).unwrap();
        let text = flatppl_syntax::print_with(&module, flatppl_syntax::Syntax::Minimal);
        assert!(text.contains(family), "{constraint}:\n{text}");
        assert!(!text.contains(other), "{constraint}:\n{text}");
    }
}
