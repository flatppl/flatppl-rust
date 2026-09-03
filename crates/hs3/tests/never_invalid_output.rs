//! Documents that used to convert to exit 0 and write FlatPPL that is not a
//! valid module.
//!
//! Every case here is a source construct with no valid FlatPPL image: a name the
//! grammar does not admit, a literal the language has no spelling for, a second
//! binding of an existing name, or a block that vanished. All of them are
//! refused at the reader, so no caller has to inspect the output to learn the
//! conversion failed.

fn err_hs3(json: &str) -> String {
    match flatppl_hs3::read_hs3(json) {
        Ok(m) => panic!(
            "expected Err, got a module:\n{}",
            flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal)
        ),
        Err(e) => e.to_string(),
    }
}

fn assert_err_hs3(json: &str, needles: &[&str]) {
    let msg = err_hs3(json);
    for needle in needles {
        assert!(
            msg.contains(needle),
            "error should mention `{needle}`, got: {msg}"
        );
    }
}

// ---------------------------------------------------------------------------
// Names the FlatPPL grammar does not admit (spec §05
// `Name ::= (Letter | "_") (Letter | Digit | "_")*`).
// ---------------------------------------------------------------------------

/// A distribution named `bad name`. This emitted `bad name = Normal(…)`, which
/// is two statements' worth of tokens on one line: a syntax error. It failed in
/// a debug build and succeeded in release, because the print-then-reparse
/// self-check was `#[cfg(debug_assertions)]`.
#[test]
fn a_name_with_a_space_errs() {
    assert_err_hs3(
        r#"{"distributions":[{"type":"gaussian_dist","name":"bad name",
                             "x":"x","mean":"mu","sigma":"s"}]}"#,
        &["bad name", "` `", "§05"],
    );
}

/// A name starting with a digit lexes as a number followed by a name.
#[test]
fn a_name_starting_with_a_digit_errs() {
    assert_err_hs3(
        r#"{"distributions":[{"type":"gaussian_dist","name":"2mass",
                             "x":"x","mean":"mu","sigma":"s"}]}"#,
        &["`2mass`", "does not start with a letter"],
    );
}

/// A reserved word cannot be bound (spec §05 "Note on reserved words").
#[test]
fn a_reserved_word_name_errs() {
    assert_err_hs3(
        r#"{"distributions":[{"type":"gaussian_dist","name":"in",
                             "x":"x","mean":"mu","sigma":"s"}]}"#,
        &["`in`", "reserved word"],
    );
}

/// `inputs` and `outputs` are reserved for the determinization signature
/// (spec §05, §13), so a document may not bind either.
#[test]
fn a_determinization_signature_name_errs() {
    assert_err_hs3(
        r#"{"distributions":[{"type":"gaussian_dist","name":"inputs",
                             "x":"x","mean":"mu","sigma":"s"}]}"#,
        &["`inputs`", "determinization signature"],
    );
}

/// The invalid name may also arrive as a free parameter reference, not just as a
/// block's own name.
#[test]
fn an_invalid_free_parameter_name_errs() {
    assert_err_hs3(
        r#"{"distributions":[{"type":"gaussian_dist","name":"g",
                             "x":"x","mean":"bad mu","sigma":"s"}]}"#,
        &["free parameter", "bad mu"],
    );
}

// ---------------------------------------------------------------------------
// Literals FlatPPL cannot spell.
// ---------------------------------------------------------------------------

/// A `"NaN"` bin edge became a real literal that the printer writes as the bare
/// token `NaN`. FlatPPL has no NaN literal (spec §03 "Predefined constants"
/// gives `inf` and `pi`, no NaN), so the parser reads `NaN` as a NAME and lint
/// reports an unresolvable name. The print-then-reparse self-check cannot see
/// this: `NaN` parses fine as a name.
#[test]
fn a_nan_bin_edge_errs() {
    assert_err_hs3(
        r#"{"distributions":[
             {"name":"bshape","type":"gaussian_dist","mean":"bmu","sigma":"bsig","x":"bx"},
             {"name":"proc","type":"bincounts_extended_dist","rate":"n","distribution":"bshape",
              "axes":[{"edges":[0.0,"NaN",2.0]}]}]}"#,
        &["bin edge 1", "not a finite number"],
    );
}

/// An `"inf"` bin edge is legal FlatPPL (`inf` is a predefined constant), which
/// is exactly why it slipped through. It is still not a bin edge: bin widths are
/// edge differences, so an infinite edge leaves a bin with no width.
#[test]
fn an_infinite_bin_edge_errs() {
    assert_err_hs3(
        r#"{"distributions":[
             {"name":"bshape","type":"gaussian_dist","mean":"bmu","sigma":"bsig","x":"bx"},
             {"name":"proc","type":"bincounts_extended_dist","rate":"n","distribution":"bshape",
              "axes":[{"edges":[0.0,1.0,"inf"]}]}]}"#,
        &["bin edge 2", "not a finite number"],
    );
}

/// `nbins: 0` divides the axis range by zero, so every edge came out NaN and the
/// emitted vector was `[NaN]`.
#[test]
fn zero_bins_errs() {
    assert_err_hs3(
        r#"{"distributions":[
             {"name":"bshape","type":"gaussian_dist","mean":"bmu","sigma":"bsig","x":"bx"},
             {"name":"proc","type":"bincounts_extended_dist","rate":"n","distribution":"bshape",
              "axes":[{"nbins":0,"min":0.0,"max":2.0}]}]}"#,
        &["`nbins` is 0"],
    );
}

/// A NaN written into an ordinary scalar field, not a bin edge.
#[test]
fn a_nan_scalar_field_errs() {
    assert_err_hs3(
        r#"{"distributions":[{"type":"gaussian_dist","name":"g",
                             "x":"x","mean":"NaN","sigma":"s"}]}"#,
        &["`NaN`", "no NaN literal"],
    );
}

// ---------------------------------------------------------------------------
// One output namespace, one binding per name.
// ---------------------------------------------------------------------------

/// A distribution and a parameter point both named `g`. Every block becomes a
/// top-level binding, so this emitted `g = Normal(…)` and `g = record(…)` in one
/// module. The duplicate check used to cover only distributions and functions.
#[test]
fn a_distribution_and_a_parameter_point_sharing_a_name_errs() {
    assert_err_hs3(
        r#"{"distributions":[{"name":"g","type":"gaussian_dist","mean":"m","sigma":"s","x":"x"}],
            "parameter_points":[{"name":"g","entries":[{"name":"m","value":1.0}]}]}"#,
        &["duplicate binding name `g`"],
    );
}

/// The same for a dataset and a domain.
#[test]
fn a_dataset_and_a_domain_sharing_a_name_errs() {
    assert_err_hs3(
        r#"{"data":[{"name":"d","type":"unbinned",
                     "axes":[{"name":"x","min":-5,"max":5}],"entries":[[0.5]]}],
            "domains":[{"name":"d","type":"product_domain",
                        "axes":[{"name":"x","min":-5,"max":5}]}]}"#,
        &["duplicate binding name `d`"],
    );
}

/// A dataset emits `<name>_domain` alongside `<name>`, so a block literally
/// named `d_domain` collides with dataset `d`.
#[test]
fn a_dataset_colliding_with_a_generated_domain_name_errs() {
    assert_err_hs3(
        r#"{"data":[{"name":"d","type":"unbinned",
                     "axes":[{"name":"x","min":-5,"max":5}],"entries":[[0.5]]}],
            "domains":[{"name":"d_domain","type":"product_domain",
                        "axes":[{"name":"x","min":-5,"max":5}]}]}"#,
        &["duplicate binding name `d_domain`"],
    );
}

// ---------------------------------------------------------------------------
// A malformed array must fail, not be repaired by deletion.
// ---------------------------------------------------------------------------

/// `"factors": ["a", 7, "b"]` converted to a two-factor product: the numeric
/// entry was dropped by a `filter_map`, repairing the input by removing model
/// structure.
#[test]
fn a_non_name_product_factor_errs() {
    assert_err_hs3(
        r#"{"distributions":[
             {"name":"a","type":"gaussian_dist","mean":"m","sigma":"s","x":"x"},
             {"name":"b","type":"gaussian_dist","mean":"m2","sigma":"s2","x":"y"},
             {"name":"p","type":"product_dist","factors":["a",7,"b"]}]}"#,
        &["`p`", "`factors` entry 1 is not a name"],
    );
}

/// The same shape in a `mixture_dist`'s `summands`.
#[test]
fn a_non_name_mixture_summand_errs() {
    assert_err_hs3(
        r#"{"distributions":[
             {"name":"a","type":"gaussian_dist","mean":"m","sigma":"s","x":"x"},
             {"name":"b","type":"gaussian_dist","mean":"m2","sigma":"s2","x":"x"},
             {"name":"mx","type":"mixture_dist","summands":["a",null,"b"],
              "coefficients":[0.3,0.7]}]}"#,
        &["`mx`", "`summands` entry 1 is not a name"],
    );
}

/// A `mixture_dist`'s `coefficients` legitimately mixes weights and parameter
/// names, so the name-array check must NOT fire on it.
#[test]
fn mixture_coefficients_may_mix_numbers_and_names() {
    let json = r#"{"distributions":[
         {"name":"a","type":"gaussian_dist","mean":"m","sigma":"s","x":"x"},
         {"name":"b","type":"gaussian_dist","mean":"m2","sigma":"s2","x":"x"},
         {"name":"mx","type":"mixture_dist","summands":["a","b"],
          "coefficients":["w",0.7],"extended":true}]}"#;
    flatppl_hs3::read_hs3(json).expect("a symbolic mixture weight is not malformed");
}

// ---------------------------------------------------------------------------
// A block that vanished.
// ---------------------------------------------------------------------------

/// A misspelled top-level block used to disappear: `{"distrubutions": […]}`
/// converted to exit 0 and emitted a model with no distribution and no
/// likelihood, with no warning. The HS3 spec §"Top-level components" lists a
/// closed set of nine components, so any other key is not HS3.
#[test]
fn a_misspelled_top_level_block_errs() {
    assert_err_hs3(
        r#"{"distrubutions":[{"type":"gaussian_dist","name":"g",
                              "x":"x","mean":"mu","sigma":"s"}],
            "domains":[{"type":"product_domain","name":"default_domain",
                        "axes":[{"name":"x","min":-5,"max":5}]}]}"#,
        &["distrubutions", "distributions"],
    );
}

/// The two non-model components the spec does list, `metadata` and `misc`, must
/// still be accepted and ignored.
#[test]
fn metadata_and_misc_are_accepted() {
    let json = r#"{"metadata":{"hs3_version":"0.2.9","authors":["a"]},
                    "misc":{"ROOT_internal":{"colors":[1,2]}},
                    "distributions":[{"type":"gaussian_dist","name":"g",
                                      "x":"x","mean":"mu","sigma":"s"}]}"#;
    let m = flatppl_hs3::read_hs3(json).expect("metadata and misc are HS3 top-level components");
    let text = flatppl_syntax::print_with(&m, flatppl_syntax::Syntax::Minimal);
    assert!(text.contains("g = Normal("), "got:\n{text}");
}

/// A stray top-level key in a pyhf workspace. pyhf's own `workspace.json` sets
/// `additionalProperties: false` and raises
/// `pyhf.exceptions.InvalidSpecification` — "Additional properties are not
/// allowed ('chanels' was unexpected)."
#[test]
fn a_misspelled_pyhf_key_errs() {
    let json = r#"{"channels":[{"name":"ch","samples":[
         {"name":"s","data":[10.0],"modifiers":[{"name":"mu","type":"normfactor","data":null}]}]}],
       "observations":[{"name":"ch","data":[10.0]}],
       "measurments":[{"name":"m","config":{"poi":"mu","parameters":[]}}],
       "version":"1.0.0"}"#;
    let msg = match flatppl_hs3::read_pyhf(json) {
        Ok(_) => panic!("expected Err, got Ok"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("measurments") && msg.contains("measurements"),
        "the error should name the stray key and the expected one, got: {msg}"
    );
}
