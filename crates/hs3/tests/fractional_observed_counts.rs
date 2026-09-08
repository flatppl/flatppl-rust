//! Per-channel count-likelihood selection from the observed data.
//!
//! §08 gives `Poisson` support `nonnegintegers`, so an integer channel keeps the
//! samplable `Poisson`. §09 says `ContinuedPoisson` matches the Poisson density
//! at non-negative integers but is not a probability measure and has no `rand`,
//! so it is used only where the data forces it: a channel with a fractional
//! observed count (an Asimov dataset). Both importers go through
//! `pyhf::assemble_channel`, but each resolves its own observed data, so both
//! are covered here.

/// The channel's own count likelihood, isolated from the staterror and shapesys
/// constraint terms (which emit `hepphys.ContinuedPoisson` regardless).
fn model_binding(text: &str) -> &str {
    text.lines()
        .find(|l| l.trim_start().starts_with("ch_model ="))
        .unwrap_or_else(|| panic!("no `ch_model` binding, got:\n{text}"))
}

fn hs3_document(observed: &str) -> String {
    format!(
        r#"{{
  "distributions": [
    {{ "name": "ch", "type": "histfactory_dist",
      "samples": [
        {{ "name": "sig", "data": {{ "contents": [12.0, 11.0] }},
          "modifiers": [{{ "parameter": "mu", "type": "normfactor" }}] }}
      ] }}
  ],
  "data": [{{ "name": "obs_data", "type": "binned", "contents": [{observed}] }}],
  "likelihoods": [
    {{ "name": "main", "distributions": ["ch"], "data": ["obs_data"] }}
  ]
}}"#
    )
}

fn pyhf_workspace(observed: &str) -> String {
    format!(
        r#"{{
  "channels": [
    {{ "name": "ch",
      "samples": [
        {{ "name": "sig", "data": [12.0, 11.0],
          "modifiers": [{{ "name": "mu", "type": "normfactor", "data": null }}] }}
      ] }}
  ],
  "observations": [{{ "name": "ch", "data": [{observed}] }}],
  "measurements": [{{ "name": "m", "config": {{ "poi": "mu" }} }}]
}}"#
    )
}

fn printed(m: &flatppl_core::Module) -> String {
    flatppl_syntax::print_with(m, flatppl_syntax::Syntax::Minimal)
}

#[test]
fn native_hs3_fractional_counts_emit_continued_poisson() {
    let m = flatppl_hs3::read_hs3(&hs3_document("51.5, 48.0")).unwrap();
    let text = printed(&m);
    let model = model_binding(&text);
    assert!(
        model.contains("broadcast(hepphys.ContinuedPoisson"),
        "fractional channel must score the continued density, got:\n{model}"
    );
    assert!(
        text.contains("ch_observed = [51.5, 48.0]"),
        "fractional counts must survive as written, got:\n{text}"
    );
}

#[test]
fn native_hs3_integer_counts_keep_poisson() {
    let m = flatppl_hs3::read_hs3(&hs3_document("51.0, 48.0")).unwrap();
    let text = printed(&m);
    let model = model_binding(&text);
    assert!(
        model.contains("broadcast(Poisson") && !model.contains("ContinuedPoisson"),
        "integer channel must keep the samplable Poisson, got:\n{model}"
    );
}

#[test]
fn pyhf_fractional_counts_emit_continued_poisson() {
    let m = flatppl_hs3::read_pyhf(&pyhf_workspace("51.5, 48.0")).unwrap();
    let text = printed(&m);
    let model = model_binding(&text);
    assert!(
        model.contains("broadcast(hepphys.ContinuedPoisson"),
        "fractional channel must score the continued density, got:\n{model}"
    );
    assert!(
        text.contains("ch_observed = [51.5, 48.0]"),
        "fractional counts must survive as written, got:\n{text}"
    );
}

#[test]
fn pyhf_integer_counts_keep_poisson() {
    let m = flatppl_hs3::read_pyhf(&pyhf_workspace("51.0, 48.0")).unwrap();
    let text = printed(&m);
    let model = model_binding(&text);
    assert!(
        model.contains("broadcast(Poisson") && !model.contains("ContinuedPoisson"),
        "integer channel must keep the samplable Poisson, got:\n{model}"
    );
}

/// The rule is per channel, not per workspace: one Asimov channel must not drag
/// an integer channel off `Poisson`.
#[test]
fn pyhf_fractional_channel_does_not_infect_its_sibling() {
    let json = r#"{
  "channels": [
    { "name": "ch",
      "samples": [
        { "name": "sig", "data": [12.0],
          "modifiers": [{ "name": "mu", "type": "normfactor", "data": null }] }
      ] },
    { "name": "asimov",
      "samples": [
        { "name": "sig", "data": [12.0],
          "modifiers": [{ "name": "mu", "type": "normfactor", "data": null }] }
      ] }
  ],
  "observations": [
    { "name": "ch", "data": [51.0] },
    { "name": "asimov", "data": [51.5] }
  ],
  "measurements": [{ "name": "m", "config": { "poi": "mu" } }]
}"#;
    let text = printed(&flatppl_hs3::read_pyhf(json).unwrap());
    let integer = model_binding(&text);
    assert!(
        integer.contains("broadcast(Poisson") && !integer.contains("ContinuedPoisson"),
        "integer channel must keep the samplable Poisson, got:\n{integer}"
    );
    let asimov = text
        .lines()
        .find(|l| l.trim_start().starts_with("asimov_model ="))
        .unwrap_or_else(|| panic!("no `asimov_model` binding, got:\n{text}"));
    assert!(
        asimov.contains("broadcast(hepphys.ContinuedPoisson"),
        "fractional channel must score the continued density, got:\n{asimov}"
    );
}

/// Negative and non-finite counts have no Poisson-like density and stay refused.
#[test]
fn pyhf_negative_counts_still_refused() {
    let err = flatppl_hs3::read_pyhf(&pyhf_workspace("-1.0, 48.0"))
        .expect_err("a negative observed count must be refused")
        .to_string();
    assert!(
        err.contains("observed event count") && err.contains("finite and nonnegative"),
        "error must name the offending count, got: {err}"
    );
}
