//! Tests that non-1:1 HS3 lowerings emit the expected doc-comment lines and
//! that the resulting FlatPPL text round-trip-parses cleanly.
//!
//! Rule: a binding gets a `%`-doc-comment IFF its HS3 source lowered to a
//! composite or measure-algebra expression (more than a single distribution
//! call).  A direct 1:1 function→function map gets NO comment.
use flatppl_syntax::{Syntax, parse, print_with};

// ---------------------------------------------------------------------------
// mixture_dist — normalize(superpose(weighted(…)))
// ---------------------------------------------------------------------------

const MIXTURE_JSON: &str = r#"{
  "distributions": [
    {
      "name": "sig",
      "type": "gaussian_dist",
      "mean": "mu_sig",
      "sigma": "sigma_sig",
      "x": "x_obs"
    },
    {
      "name": "bkg",
      "type": "gaussian_dist",
      "mean": "mu_bkg",
      "sigma": "sigma_bkg",
      "x": "x_obs"
    },
    {
      "name": "mix_dist",
      "type": "mixture_dist",
      "summands": ["sig", "bkg"],
      "coefficients": [0.3],
      "extended": false
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "mu_sig",    "value": 1.0},
      {"name": "sigma_sig", "value": 0.5},
      {"name": "mu_bkg",    "value": 0.0},
      {"name": "sigma_bkg", "value": 1.0}
    ]}
  ]
}"#;

#[test]
fn mixture_dist_has_doc_comment_and_roundtrips() {
    let m = flatppl_hs3::read_hs3(MIXTURE_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== mixture_dist doc-comment ===\n{text}\n=== end ===");

    // The binding for mix_dist must carry the HS3 provenance comment.
    assert!(
        text.contains("% HS3 mixture_dist"),
        "missing mixture_dist doc-comment, got:\n{text}"
    );
    // The output must still contain the FlatPPL expression.
    assert!(
        text.contains("normalize"),
        "missing normalize, got:\n{text}"
    );
    assert!(
        text.contains("superpose"),
        "missing superpose, got:\n{text}"
    );

    // Round-trip: the doc-comment must not break re-parsing.
    let rt = parse(&text);
    assert!(
        rt.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        rt.err()
    );
}

// ---------------------------------------------------------------------------
// generic_dist — normalize(weighted(functionof(<expr>), Lebesgue(reals)))
// ---------------------------------------------------------------------------

const GENERIC_DIST_JSON: &str = r#"{
  "distributions": [
    {
      "name": "my_generic",
      "type": "generic_dist",
      "expression": "exp(-0.5 * x * x)",
      "x": "x_obs"
    }
  ]
}"#;

#[test]
fn generic_dist_has_doc_comment_and_roundtrips() {
    let m = flatppl_hs3::read_hs3(GENERIC_DIST_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== generic_dist doc-comment ===\n{text}\n=== end ===");

    assert!(
        text.contains("% HS3 generic_dist"),
        "missing generic_dist doc-comment, got:\n{text}"
    );
    assert!(
        text.contains("normalize"),
        "missing normalize, got:\n{text}"
    );
    assert!(text.contains("Lebesgue"), "missing Lebesgue, got:\n{text}");

    let rt = parse(&text);
    assert!(
        rt.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        rt.err()
    );
}

// ---------------------------------------------------------------------------
// histfactory channel (pyhf path) — obs_model and L_ bindings
// ---------------------------------------------------------------------------

const PYHF_CHAN_JSON: &str = r#"{
  "channels": [
    {
      "name": "chan1",
      "samples": [
        {
          "name": "bkg",
          "data": [50.0, 52.0],
          "modifiers": [
            {
              "name": "mu_bkg",
              "type": "normfactor",
              "data": null
            }
          ]
        }
      ]
    }
  ],
  "observations": [
    {"name": "chan1", "data": [51.0, 48.0]}
  ],
  "measurements": [
    {"name": "m", "config": {"poi": "mu_bkg", "parameters": []}}
  ],
  "version": "1.0.0"
}"#;

#[test]
fn histfactory_channel_has_doc_comments_and_roundtrips() {
    let m = flatppl_hs3::read_pyhf(PYHF_CHAN_JSON).expect("read_pyhf must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== histfactory channel doc-comment ===\n{text}\n=== end ===");

    // obs_model binding must have the channel-level doc-comment.
    assert!(
        text.contains("% HS3 histfactory channel"),
        "missing histfactory channel doc-comment, got:\n{text}"
    );
    assert!(
        text.contains("broadcast(Poisson"),
        "missing Poisson broadcast, got:\n{text}"
    );

    // L_ binding must have the likelihood doc-comment.
    assert!(
        text.contains("% HS3 histfactory likelihood"),
        "missing histfactory likelihood doc-comment, got:\n{text}"
    );

    // Round-trip.
    let rt = parse(&text);
    assert!(
        rt.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        rt.err()
    );
}

// ---------------------------------------------------------------------------
// 1:1 mapping (gaussian_dist → Normal) must NOT get a doc-comment.
// ---------------------------------------------------------------------------

const GAUSSIAN_JSON: &str = r#"{
  "distributions": [
    {
      "name": "gauss",
      "type": "gaussian_dist",
      "mean": "mu",
      "sigma": "sigma",
      "x": "x_obs"
    }
  ],
  "parameter_points": [
    {"name": "nominal", "entries": [
      {"name": "mu",    "value": 0.0},
      {"name": "sigma", "value": 1.0}
    ]}
  ]
}"#;

#[test]
fn direct_mapping_has_no_doc_comment() {
    let m = flatppl_hs3::read_hs3(GAUSSIAN_JSON).expect("read_hs3 must succeed");
    let text = print_with(&m, Syntax::Minimal);
    eprintln!("=== gaussian no doc-comment ===\n{text}\n=== end ===");

    // Must contain the distribution but no HS3 provenance comment.
    assert!(text.contains("Normal"), "missing Normal, got:\n{text}");
    assert!(
        !text.contains("% HS3"),
        "unexpected doc-comment on 1:1 mapping, got:\n{text}"
    );

    let rt = parse(&text);
    assert!(
        rt.is_ok(),
        "round-trip parse failed: {:?}\n\nEmitted:\n{text}",
        rt.err()
    );
}
