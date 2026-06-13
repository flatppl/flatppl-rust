const PYHF: &str = r#"{
  "channels": [
    { "name": "singlechannel",
      "samples": [
        { "name": "signal", "data": [12.0, 11.0],
          "modifiers": [{ "name": "mu", "type": "normfactor", "data": null }] },
        { "name": "background", "data": [50.0, 52.0],
          "modifiers": [{ "name": "uncorr_bkguncrt", "type": "shapesys", "data": [3.0, 7.0] }] }
      ] }
  ],
  "observations": [{ "name": "singlechannel", "data": [51.0, 48.0] }],
  "measurements": [{ "name": "m", "config": { "poi": "mu" } }]
}"#;

#[test]
fn pyhf_uncorrelated_background_assembles() {
    let m = flatppl_hs3::read(PYHF).unwrap();
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    assert!(text.contains("broadcast(Poisson"), "got:\n{text}");
    assert!(text.contains("ContinuedPoisson"), "got:\n{text}");
    assert!(text.contains("joint_likelihood("), "got:\n{text}");
    assert!(text.contains("likelihoodof("), "got:\n{text}");
    assert!(!text.contains("fn("), "must be point-free, got:\n{text}");
    // observed data [51.0, 48.0] on the main Poisson term, as the exact in-order
    // vector (a bare contains("51")/("48") would false-pass on substrings and
    // miss a reordered observation).
    assert!(
        text.contains("likelihoodof(obs_model_singlechannel, [51.0, 48.0])"),
        "observed data [51.0, 48.0] not on main term, got:\n{text}"
    );
    // shapesys domain uses integer size, not real:
    assert!(
        text.contains("cartpow(posreals, 2)"),
        "expected cartpow(posreals, 2), got:\n{text}"
    );
}
