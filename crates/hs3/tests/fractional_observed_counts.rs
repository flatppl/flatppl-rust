//! The count likelihood of every channel is `hepphys.ContinuedPoisson`.
//!
//! §08 gives `Poisson` the domain `integers`, but pyhf and ROOT score integer and
//! fractional (Asimov) counts alike with the continuous lgamma extension, which
//! §09 `ContinuedPoisson` defines. The family never depends on the data values.
//! Both importers go through `pyhf::assemble_channel`, but each resolves its own
//! observed data, so both are covered here.

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
fn native_hs3_integer_counts_emit_continued_poisson() {
    let m = flatppl_hs3::read_hs3(&hs3_document("51.0, 48.0")).unwrap();
    let text = printed(&m);
    let model = model_binding(&text);
    assert!(
        model.contains("broadcast(hepphys.ContinuedPoisson"),
        "integer counts use the same family as fractional ones, got:\n{model}"
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
fn pyhf_integer_counts_emit_continued_poisson() {
    let m = flatppl_hs3::read_pyhf(&pyhf_workspace("51.0, 48.0")).unwrap();
    let text = printed(&m);
    let model = model_binding(&text);
    assert!(
        model.contains("broadcast(hepphys.ContinuedPoisson"),
        "integer counts use the same family as fractional ones, got:\n{model}"
    );
}

/// An integer channel and an Asimov channel in one workspace use one family.
#[test]
fn pyhf_integer_and_fractional_channels_share_one_family() {
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
        integer.contains("broadcast(hepphys.ContinuedPoisson"),
        "integer channel must score the continued density, got:\n{integer}"
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

/// Negative and non-finite counts have no `ContinuedPoisson` density and stay refused.
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
