//! Round-trip tests for the FlatPIR JSON encoding over the workspace
//! `fixtures/flatpir/*.flatpir` corpus.
//!
//! Oracle (spec §11: comments/whitespace are not part of the data model): the
//! JSON path must agree with the *already-trusted* canonical text round-trip
//! (`roundtrip.rs`). Any semantic loss in the JSON encoding surfaces as a
//! `write` mismatch in property 1.

use std::fs;
use std::path::PathBuf;

use flatppl_flatpir::{from_json, read, to_json, write};
use serde_json::json;

fn fixture(name: &str) -> String {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "../../fixtures/flatpir", name]
        .iter()
        .collect();
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// Property 1 — the JSON path equals the canonical text path:
/// `write(m)  ==  write(from_json(to_json(m)))`.
/// Property 2 — JSON idempotency: `to_json` is stable under a JSON round-trip.
fn assert_json_roundtrip(name: &str) {
    let m1 = read(&fixture(name)).unwrap_or_else(|e| panic!("{name}: read failed: {e}"));
    let j = to_json(&m1);

    let m2 = from_json(&j)
        .unwrap_or_else(|e| panic!("{name}: from_json failed: {e}\n--- json ---\n{j:#}"));
    assert_eq!(
        write(&m1),
        write(&m2),
        "{name}: JSON path diverged from the canonical text path"
    );

    // JSON-level oracle: re-encoding the decoded module reproduces the original
    // JSON exactly (a strictly stronger check than text equality, and it doubles
    // as the idempotency assertion — reuse the already-decoded `m2`).
    assert_eq!(j, to_json(&m2), "{name}: to_json is not idempotent");
}

macro_rules! json_tests {
    ($($test:ident => $file:literal),* $(,)?) => {
        $(#[test] fn $test() { assert_json_roundtrip($file); })*
    };
}

json_tests! {
    json_helpers => "helpers.flatpir",
    json_model => "model.flatpir",
    json_helpers_annotated => "helpers-annotated.flatpir",
    json_model_annotated => "model-annotated.flatpir",
    json_user_call => "user-call.flatpir",
    json_values => "values.flatpir",
    json_aggregate => "aggregate.flatpir",
    json_docs => "docs.flatpir",
    json_reified => "reified.flatpir",
}

/// Every `.flatpir` file in the corpus round-trips through the JSON encoding —
/// a glob guard so any fixture added later is covered without editing the
/// `json_tests!` list above.
#[test]
fn every_corpus_fixture_roundtrips() {
    let dir: PathBuf = [env!("CARGO_MANIFEST_DIR"), "../../fixtures/flatpir"]
        .iter()
        .collect();
    let mut count = 0;
    for entry in fs::read_dir(&dir).expect("read fixtures/flatpir") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) == Some("flatpir") {
            let name = path.file_name().unwrap().to_str().unwrap();
            assert_json_roundtrip(name);
            count += 1;
        }
    }
    assert!(
        count >= 9,
        "expected at least the 9 corpus fixtures, found {count}"
    );
}

/// Targeted shape checks: confirm the encoding matches the documented
/// `.flatpir.json` shape, not merely an internally-consistent round-trip.
#[test]
fn reified_inputs_shape() {
    let j = to_json(&read(&fixture("reified.flatpir")).unwrap());
    let binds = j["%module"]["binds"].as_array().unwrap();
    let bind = |name: &str| {
        binds
            .iter()
            .find(|b| b["name"] == name)
            .unwrap_or_else(|| panic!("no binding `{name}`"))
    };

    // `f` is a boundary-less, not-yet-inferred reification → list is "%deferred".
    let f = &bind("f")["expr"];
    assert_eq!(f[0], "functionof");
    let f_inputs = f.as_array().unwrap().last().unwrap();
    assert_eq!(f_inputs["%inputs"]["origin"], "%autoinputs");
    assert_eq!(f_inputs["%inputs"]["list"], "%deferred");

    // `g` carries a filled auto-inputs list (a JSON array of [name, ref] pairs).
    let g = &bind("g")["expr"];
    let g_inputs = g.as_array().unwrap().last().unwrap();
    assert!(g_inputs["%inputs"]["list"].is_array());
    assert_eq!(g_inputs["%inputs"]["list"][0][0], "a");
}

#[test]
fn user_call_and_specinputs_shape() {
    let j = to_json(&read(&fixture("user-call.flatpir")).unwrap());
    let binds = j["%module"]["binds"].as_array().unwrap();

    // `result` is a user-defined call → head element is the literal "%call".
    let result = binds.iter().find(|b| b["name"] == "result").unwrap();
    assert_eq!(result["expr"][0], "%call");

    // `scaled` is a %specinputs reification (ordered, positionally callable).
    let scaled = binds.iter().find(|b| b["name"] == "scaled").unwrap();
    let inputs = scaled["expr"].as_array().unwrap().last().unwrap();
    assert_eq!(inputs["%inputs"]["origin"], "%specinputs");
}

#[test]
fn annotated_meta_shape() {
    let j = to_json(&read(&fixture("model-annotated.flatpir")).unwrap());
    let binds = j["%module"]["binds"].as_array().unwrap();

    // `b ~ Normal(...)` — the draw carries a stochastic %meta on the call array.
    let b = &binds.iter().find(|b| b["name"] == "b").unwrap()["expr"];
    assert_eq!(b["%meta"]["phase"], "%stochastic");
    assert_eq!(b["%meta"]["expr"][0], "draw");
}

#[test]
#[allow(clippy::approx_constant)] // 3.14 is the fixture's literal value, not π
fn literals_shape() {
    let j = to_json(&read(&fixture("values.flatpir")).unwrap());
    let binds = j["%module"]["binds"].as_array().unwrap();
    let expr = |name: &str| binds.iter().find(|b| b["name"] == name).unwrap()["expr"].clone();
    assert_eq!(expr("n")["int"], 42);
    assert_eq!(expr("x")["real"], 3.14);
    assert_eq!(expr("flag")["bool"], true);
    assert_eq!(expr("label")["str"], "hello, world");
}

/// Exercise the `%meta` type sub-grammar — including `%array`, whose shape
/// `(<dim>…)` is a *headless* list — and confirm dims encode as tagged
/// integers, not bare head strings.
#[test]
fn type_grammar_roundtrip_and_headless_shape() {
    let src = "\
(%module
  (%public a b c d e)
  (%bind a (%meta ((%array 2 (2 3) (%scalar real)) %parameterized reals) (elementof reals)))
  (%bind b (%meta ((%tvector 3 (%scalar real)) %fixed reals) (elementof reals)))
  (%bind c (%meta ((%record (mu (%scalar real)) (sigma (%scalar real))) %fixed %unknown) (elementof reals)))
  (%bind d (%meta ((%tuple (%scalar real) (%scalar integer)) %fixed %unknown) (elementof reals)))
  (%bind e (%meta ((%table (%columns (x (%scalar real))) (%nrows 10)) %fixed %unknown) (elementof reals))))";

    let m1 = read(src).unwrap();
    let j = to_json(&m1);
    let m2 = from_json(&j).unwrap();
    assert_eq!(write(&m1), write(&m2), "type-grammar round-trip");

    // `a`'s %array type: ["%array", {int:2}, [{int:2},{int:3}], ["%scalar",…]]
    let ty = &j["%module"]["binds"][0]["expr"]["%meta"]["type"];
    assert_eq!(ty[0], "%array");
    assert_eq!(ty[1]["int"], 2); // ndims is a tagged integer
    assert!(ty[2].is_array()); // the shape is a headless array …
    assert_eq!(ty[2][0]["int"], 2); // … of tagged-integer dims
    assert_eq!(ty[2][1]["int"], 3);
}

// ---- from_json error paths (malformed input must error, not panic) ----

#[test]
fn rejects_missing_module() {
    assert!(from_json(&json!({})).is_err());
    assert!(from_json(&json!({ "public": [], "binds": [] })).is_err());
}

#[test]
fn rejects_unwrapped_literal_expr() {
    // A raw JSON number is not a FlatPIR expression (literals must be wrapped).
    let bad = json!({ "%module": { "public": [], "binds": [
        { "name": "x", "expr": 42 }
    ]}});
    assert!(from_json(&bad).is_err());
}

#[test]
fn rejects_unknown_node_object() {
    let bad = json!({ "%module": { "public": [], "binds": [
        { "name": "x", "expr": { "bogus": 1 } }
    ]}});
    assert!(from_json(&bad).is_err());
}

#[test]
fn rejects_empty_call_array() {
    let bad = json!({ "%module": { "public": [], "binds": [
        { "name": "x", "expr": [] }
    ]}});
    assert!(from_json(&bad).is_err());
}

// ---- coverage of node kinds / forms not present in the fixed corpus ----

/// Small helper: round-trip an inline module and return its JSON.
fn rt(src: &str) -> serde_json::Value {
    let m1 = read(src).unwrap_or_else(|e| panic!("read failed: {e}\n{src}"));
    let j = to_json(&m1);
    let m2 = from_json(&j).unwrap_or_else(|e| panic!("from_json failed: {e}"));
    assert_eq!(write(&m1), write(&m2), "round-trip diverged for:\n{src}");
    // JSON-level oracle (stronger than text equality, and doubles as the
    // idempotency check): re-encode the already-decoded `m2`, no second decode.
    assert_eq!(j, to_json(&m2), "not idempotent:\n{src}");
    j
}

#[test]
fn hole_roundtrips() {
    let j = rt("(%module (%public h) (%bind h (add _ 1)))");
    let expr = &j["%module"]["binds"][0]["expr"];
    assert_eq!(expr[0], "add");
    assert_eq!(expr[1]["hole"], true);
}

#[test]
fn partial_meta_deferred_slots() {
    // Only the phase slot is known; type and value-set are %deferred.
    let j = rt("(%module (%public a)\
        (%bind a (%meta (%deferred %parameterized %deferred) (elementof reals))))");
    let meta = &j["%module"]["binds"][0]["expr"]["%meta"];
    assert_eq!(meta["type"]["const"], "%deferred");
    assert_eq!(meta["phase"], "%parameterized");
    assert_eq!(meta["valueset"]["const"], "%deferred");
}

#[test]
fn mass_classes_and_valueset_forms() {
    let j = rt("(%module (%public p q r s)\
        (%bind p (%meta ((%measure (%domain (%scalar real)) (%mass %null)) %fixed reals) (elementof reals)))\
        (%bind q (%meta ((%scalar real) %fixed (stdsimplex 3)) (elementof reals)))\
        (%bind r (%meta ((%scalar real) %fixed (interval 0 1)) (elementof reals)))\
        (%bind s (%meta ((%array 1 (%dynamic) (%scalar real)) %fixed (cartpow reals 4)) (elementof reals))))");
    let b = |n: usize| j["%module"]["binds"][n]["expr"]["%meta"].clone();
    // mass class inside a %measure type
    assert_eq!(b(0)["type"][2], json!(["%mass", { "const": "%null" }]));
    // value-set slot forms
    assert_eq!(b(1)["valueset"], json!(["stdsimplex", { "int": 3 }]));
    assert_eq!(
        b(2)["valueset"],
        json!(["interval", { "real": 0.0 }, { "real": 1.0 }])
    );
    // (cartpow <set> <dim>) — a set and a single dimension, not a shape vector
    assert_eq!(
        b(3)["valueset"],
        json!(["cartpow", { "const": "reals" }, { "int": 4 }])
    );
    // %dynamic dim inside the %array shape: a headless list of one dynamic dim,
    // encoded as the dedicated tagged atom `{"%dynamic": true}` (unambiguous in
    // any position — never mistaken for a call head).
    assert_eq!(b(3)["type"][2], json!([{ "%dynamic": true }]));
}

#[test]
fn composite_constructors_shape() {
    let j = to_json(&read(&fixture("values.flatpir")).unwrap());
    let binds = j["%module"]["binds"].as_array().unwrap();
    let expr = |name: &str| binds.iter().find(|b| b["name"] == name).unwrap()["expr"].clone();
    // vector of reals
    assert_eq!(
        expr("v"),
        json!(["vector", {"real":1.0}, {"real":2.0}, {"real":3.0}])
    );
    // vector of vectors (headless inner lists? no — `vector` is the head)
    assert_eq!(expr("nested")[0], "vector");
    assert_eq!(expr("nested")[1], json!(["vector", {"int":1}, {"int":2}]));
    // record uses %field entries
    assert_eq!(
        expr("r"),
        json!(["record",
            {"%field": {"name":"mu","value":{"real":0.0}}},
            {"%field": {"name":"sigma","value":{"real":1.0}}}])
    );
    // tuple mixes a ref and a ref; complex is a constructor call
    assert_eq!(expr("pair")[0], "tuple");
    assert_eq!(expr("z"), json!(["complex", {"real":0.5}, {"real":2.0}]));
}

#[test]
fn metricsum_variance_axes_shape() {
    let j = to_json(&read(&fixture("aggregate.flatpir")).unwrap());
    let binds = j["%module"]["binds"].as_array().unwrap();
    let m = binds.iter().find(|b| b["name"] == "M").unwrap();
    // metricsum carries (vector (%uaxis a) (%laxis c)) — upper/lower axis labels.
    let axes = &m["expr"][2];
    assert_eq!(axes[0], "vector");
    assert_eq!(axes[1], json!({"%uaxis": "a"}));
    assert_eq!(axes[2], json!({"%laxis": "c"}));
}

/// Decode a hand-authored document (not produced by `to_json`), with keys in a
/// non-canonical order, and confirm it reads to the expected module. Exercises
/// the decode path independently of encode and pins key-order insensitivity.
#[test]
fn decodes_handauthored_document() {
    let doc = json!({ "%module": {
        // binds before public; bind fields out of "natural" order
        "binds": [
            { "expr": ["elementof", {"const": "reals"}], "name": "a" },
            { "expr": ["draw", ["Normal",
                {"%kwarg": {"value": {"real": 0.0}, "name": "mu"}},
                {"%kwarg": {"name": "sigma", "value": {"real": 1.0}}}]],
              "name": "b" }
        ],
        "public": ["a", "b"]
    }});
    let m = from_json(&doc).expect("hand-authored doc must decode");
    let canonical = write(&m);
    // Re-encoding reaches the same JSON the canonical module would produce.
    assert_eq!(to_json(&m), to_json(&read(&canonical).unwrap()));
    assert!(canonical.contains("(%bind a (elementof reals))"));
    assert!(canonical.contains("draw (Normal"));
}

// ---- M5 / error conditions (SPEC §14): malformed input must Err, not panic --

#[test]
fn rejects_malformed_input_pair() {
    // A one-element entry is not a [name, ref] pair.
    let one_elem = json!({ "%module": { "public": [], "binds": [
        { "name": "f", "expr": ["functionof", { "int": 1 },
            { "%inputs": { "origin": "%specinputs", "list": [["x"]] } }] }
    ]}});
    assert!(from_json(&one_elem).is_err());

    // A zero-element entry is likewise rejected.
    let zero_elem = json!({ "%module": { "public": [], "binds": [
        { "name": "f", "expr": ["functionof", { "int": 1 },
            { "%inputs": { "origin": "%specinputs", "list": [[]] } }] }
    ]}});
    assert!(from_json(&zero_elem).is_err());
}

#[test]
fn rejects_reader_rejected_expr() {
    // Structurally-valid JSON, but the `%meta` phase is one the reader rejects.
    // The reader error must surface as an `Err`, not a panic.
    let bad = json!({ "%module": { "public": ["a"], "binds": [
        { "name": "a", "expr": ["elementof",
            { "%meta": { "type": { "const": "%deferred" },
                         "phase": "%bogus",
                         "valueset": { "const": "reals" } } },
            { "const": "reals" }] }
    ]}});
    assert!(from_json(&bad).is_err());
}

#[test]
fn rejects_bad_inputs_list_scalar() {
    // `%inputs.list` is neither a "%deferred" string nor an array of pairs.
    let bad = json!({ "%module": { "public": [], "binds": [
        { "name": "f", "expr": ["functionof", { "int": 1 },
            { "%inputs": { "origin": "%specinputs", "list": 5 } }] }
    ]}});
    assert!(from_json(&bad).is_err());
}

#[test]
fn rejects_empty_inputs_list() {
    // SPEC §9: a reified input list cannot be empty (callables cannot be nullary).
    // Enforced at the JSON layer, not just by the reader.
    let bad = json!({ "%module": { "public": [], "binds": [
        { "name": "f", "expr": ["functionof", { "int": 1 },
            { "%inputs": { "origin": "%specinputs", "list": [] } }] }
    ]}});
    assert!(from_json(&bad).is_err());
}

#[test]
fn rejects_non_ref_input_entry() {
    // SPEC §9: each entry is `[name, ref]` where `ref` is a `{"%ref"}` object.
    let bad = json!({ "%module": { "public": [], "binds": [
        { "name": "f", "expr": ["functionof", { "int": 1 },
            { "%inputs": { "origin": "%specinputs", "list": [["x", { "int": 1 }]] } }] }
    ]}});
    assert!(from_json(&bad).is_err());
}

// ---- M1 / atom validation: un-representable / ambiguous literal objects -----

#[test]
fn rejects_reclassifying_const() {
    // A `const` whose text would re-classify on re-read (hole/bool/number) is
    // not representable as a bare atom and must be rejected, not silently mutated.
    for sym in ["_", "true", "42"] {
        let bad = json!({ "%module": { "public": ["x"], "binds": [
            { "name": "x", "expr": { "const": sym } }
        ]}});
        assert!(
            from_json(&bad).is_err(),
            "expected rejection for const {sym:?}"
        );
    }
}

#[test]
fn rejects_ambiguous_literal() {
    // An atom object with more than one value-wrapper key is ambiguous.
    let bad = json!({ "%module": { "public": ["x"], "binds": [
        { "name": "x", "expr": { "int": 3, "real": 4 } }
    ]}});
    assert!(from_json(&bad).is_err());
}

// ---- H2 / empty public is always emitted and preserved ----------------------

#[test]
fn empty_public_roundtrips() {
    // An authored empty `public` array must decode to a module with NO public
    // bindings (it must not fall through to the name-convention fallback, which
    // would make `vis` public).
    let doc = json!({ "%module": { "public": [], "binds": [
        { "name": "vis", "expr": ["elementof", { "const": "reals" }] }
    ]}});
    let m = from_json(&doc).expect("empty-public doc must decode");

    // The decoded module has NO public bindings: `vis` is private. (Were the
    // empty `public` array dropped, the reader's name-convention fallback would
    // have made the non-underscore `vis` public — this is what the json layer's
    // always-emit `(%public)` prevents.)
    assert_eq!(
        to_json(&m)["%module"]["public"],
        json!([]),
        "an authored empty public list must be preserved, not name-convention'd"
    );
}

// ---- M4 / variant coverage: forms absent from the fixed corpus --------------

/// One round-trip exercising every otherwise-untested type / value-set / mass
/// variant, plus a `bool:false` literal and a neutral `%axis`. The `%meta`
/// slots need not agree with the bound expression — the reader does not
/// typecheck meta against the expr — so each bind is a trivially-valid
/// `(%meta (…) (elementof <set>))`.
#[test]
fn all_type_valueset_mass_variants() {
    let src = "\
(%module
  (%public fn lk rg vr fl an cx bf)
  (%bind fn (%meta ((%function (%inputs a)) %fixed %unknown) (elementof reals)))
  (%bind lk (%meta ((%likelihood (%inputs a) (%obstype (%scalar boolean))) %fixed %unknown) (elementof reals)))
  (%bind rg (%meta (%rngstate %fixed rngstates) (elementof rngstates)))
  (%bind vr (%meta (%var3 %deferred %deferred) (elementof reals)))
  (%bind fl (%meta ((%failed \"boom\") %fixed %unknown) (elementof reals)))
  (%bind an (%meta (%any %fixed anything) (elementof reals)))
  (%bind cx (%meta ((%scalar complex) %fixed complexes) (elementof complexes)))
  (%bind bf false)
  (%bind vs1 (%meta ((%scalar real) %fixed posreals) (elementof posreals)))
  (%bind vs2 (%meta ((%scalar real) %fixed nonnegreals) (elementof nonnegreals)))
  (%bind vs3 (%meta ((%scalar real) %fixed unitinterval) (elementof unitinterval)))
  (%bind vs4 (%meta ((%scalar integer) %fixed integers) (elementof integers)))
  (%bind vs5 (%meta ((%scalar integer) %fixed posintegers) (elementof posintegers)))
  (%bind vs6 (%meta ((%scalar integer) %fixed nonnegintegers) (elementof nonnegintegers)))
  (%bind vs7 (%meta ((%scalar boolean) %fixed booleans) (elementof booleans)))
  (%bind vs8 (%meta ((%scalar complex) %fixed complexes) (elementof complexes)))
  (%bind vs9 (%meta (%rngstate %fixed rngstates) (elementof rngstates)))
  (%bind vs10 (%meta (%any %fixed anything) (elementof anything)))
  (%bind ms1 (%meta ((%measure (%domain (%scalar real)) (%mass %deferred)) %fixed reals) (elementof reals)))
  (%bind ms2 (%meta ((%measure (%domain (%scalar real)) (%mass %normalized)) %fixed reals) (elementof reals)))
  (%bind ms3 (%meta ((%measure (%domain (%scalar real)) (%mass %finite)) %fixed reals) (elementof reals)))
  (%bind ms4 (%meta ((%measure (%domain (%scalar real)) (%mass %locallyfinite)) %fixed reals) (elementof reals)))
  (%bind ms5 (%meta ((%measure (%domain (%scalar real)) (%mass %unknown)) %fixed reals) (elementof reals))))";

    let j = rt(src);
    let binds = j["%module"]["binds"].as_array().unwrap();
    let meta = |name: &str| {
        binds
            .iter()
            .find(|b| b["name"] == name)
            .unwrap_or_else(|| panic!("no bind {name}"))["expr"]["%meta"]
            .clone()
    };
    let expr = |name: &str| {
        binds
            .iter()
            .find(|b| b["name"] == name)
            .unwrap_or_else(|| panic!("no bind {name}"))["expr"]
            .clone()
    };

    // Type encodings.
    assert_eq!(
        meta("fn")["type"],
        json!(["%function", ["%inputs", { "const": "a" }]])
    );
    assert_eq!(
        meta("lk")["type"],
        json!([
            "%likelihood",
            ["%inputs", { "const": "a" }],
            ["%obstype", ["%scalar", { "const": "boolean" }]]
        ])
    );
    assert_eq!(meta("rg")["type"], json!({ "const": "%rngstate" }));
    assert_eq!(meta("vr")["type"], json!({ "const": "%var3" }));
    assert_eq!(meta("fl")["type"], json!(["%failed", { "str": "boom" }]));
    assert_eq!(meta("an")["type"], json!({ "const": "%any" }));
    assert_eq!(
        meta("cx")["type"],
        json!(["%scalar", { "const": "complex" }])
    );

    // Mass classes inside %measure types.
    assert_eq!(
        meta("ms2")["type"][2],
        json!(["%mass", { "const": "%normalized" }])
    );
    assert_eq!(
        meta("ms3")["type"][2],
        json!(["%mass", { "const": "%finite" }])
    );
    assert_eq!(
        meta("ms4")["type"][2],
        json!(["%mass", { "const": "%locallyfinite" }])
    );
    assert_eq!(
        meta("ms5")["type"][2],
        json!(["%mass", { "const": "%unknown" }])
    );

    // A `false` literal and a bare value-set.
    assert_eq!(expr("bf"), json!({ "bool": false }));
    assert_eq!(meta("vs10")["valueset"], json!({ "const": "anything" }));
    assert_eq!(meta("vs1")["valueset"], json!({ "const": "posreals" }));

    // Neutral `%axis` shape from the aggregate fixture (a plain axis label,
    // distinct from the %uaxis/%laxis covered elsewhere).
    let agg = to_json(&read(&fixture("aggregate.flatpir")).unwrap());
    let text = serde_json::to_string(&agg).unwrap();
    assert!(
        text.contains(r#"{"%axis":"i"}"#),
        "expected a neutral {{\"%axis\":\"i\"}} in the aggregate fixture"
    );
}

// ---- L3 / edge inputs (round-trip via rt) -----------------------------------

#[test]
fn empty_module_roundtrips() {
    // The reader accepts a truly empty module; `to_json` always emits the
    // (empty) public form, and the writer collapses it back to `(%module)`.
    let j = rt("(%module)");
    assert_eq!(j["%module"]["public"], json!([]));
    assert_eq!(j["%module"]["binds"], json!([]));
}

#[test]
fn no_public_uses_name_convention() {
    // No `(%public …)`: the name convention makes `x` public and `_y` private.
    // `rt` asserts the JSON path matches the canonical text path either way.
    let j = rt("(%module (%bind x 1) (%bind _y 2))");
    assert_eq!(j["%module"]["public"], json!(["x"]));
}

#[test]
fn string_escapes_roundtrip() {
    let j = rt(r#"(%module (%public s) (%bind s "a\"b\nc\td\\e"))"#);
    // The decoded string value (quotes, newline, tab, backslash) survives.
    assert_eq!(
        j["%module"]["binds"][0]["expr"],
        json!({ "str": "a\"b\nc\td\\e" })
    );
}

#[test]
fn negative_and_big_numbers() {
    let j = rt("(%module (%public n m) (%bind n -42) (%bind m -3.5))");
    let binds = j["%module"]["binds"].as_array().unwrap();
    let expr = |name: &str| binds.iter().find(|b| b["name"] == name).unwrap()["expr"].clone();
    assert_eq!(expr("n"), json!({ "int": -42 }));
    assert_eq!(expr("m"), json!({ "real": -3.5 }));
}

#[test]
fn interval_inf_bounds() {
    // `inf` / `-inf` bounds parse (via parse_bound) and round-trip. They encode
    // as bare `const` symbols inside an `["interval", …]` value-set form.
    let j = rt("(%module (%public a)\
        (%bind a (%meta ((%scalar real) %fixed (interval -inf inf)) (elementof reals))))");
    let vs = &j["%module"]["binds"][0]["expr"]["%meta"]["valueset"];
    assert_eq!(vs[0], "interval");
    assert_eq!(
        *vs,
        json!(["interval", { "const": "-inf" }, { "const": "inf" }])
    );
}

/// A node object carrying a literal key AND a structural key (or two structural
/// keys) is ambiguous: the decode ladder would pick one and silently drop the
/// rest. `from_json` must reject it.
#[test]
fn rejects_literal_plus_structural_key() {
    let bad = |expr: serde_json::Value| json!({ "%module": { "public": [], "binds": [{ "name": "x", "expr": expr }] } });
    assert!(
        from_json(&bad(
            json!({ "int": 1, "%ref": { "ns": "self", "name": "m" } })
        ))
        .is_err()
    );
    assert!(from_json(&bad(json!({ "const": "foo", "%axis": "i" }))).is_err());
    assert!(
        from_json(&bad(
            json!({ "%ref": { "ns": "self", "name": "m" }, "%axis": "i" })
        ))
        .is_err()
    );
}

/// Deeply-nested JSON must return `Err`, never recurse until the native stack
/// overflows (which would abort the process uncatchably). The depth guard trips
/// well before that.
#[test]
fn rejects_deeply_nested_expr() {
    // 300 > MAX_DEPTH (128) so the guard trips and `from_json` returns Err well
    // before any stack exhaustion. (Kept modest: building/dropping a far deeper
    // serde_json::Value would itself overflow the test thread's stack.)
    let mut expr = json!({ "int": 1 });
    for _ in 0..300 {
        expr = json!([expr]); // each layer is a headless list → +1 emit depth
    }
    let doc = json!({ "%module": { "public": [], "binds": [{ "name": "x", "expr": expr }] } });
    assert!(from_json(&doc).is_err()); // returns Err, does not abort
}

/// `%dynamic` dims encode the SAME way regardless of position in an `%array`
/// shape — always the tagged atom `{"%dynamic": true}`, never a bare call head.
/// This is the M2 fix: the shape is unambiguous for a direct JSON consumer.
#[test]
fn dynamic_dim_is_position_independent() {
    let j = rt("(%module (%public a)\
        (%bind a (%meta ((%array 2 (%dynamic 3) (%scalar real)) %fixed %unknown) (elementof reals))))");
    let shape = &j["%module"]["binds"][0]["expr"]["%meta"]["type"][2];
    // headless list: dynamic-first dim and the static dim both tagged explicitly
    assert_eq!(*shape, json!([{ "%dynamic": true }, { "int": 3 }]));
}

/// `try_to_json` returns `Ok` for an in-contract module (the common case);
/// `to_json` is the infallible wrapper over it.
#[test]
fn try_to_json_ok_for_valid_module() {
    let m = read("(%module (%public x) (%bind x (add 1 2)))").unwrap();
    let j = flatppl_flatpir::try_to_json(&m).expect("valid module encodes");
    assert_eq!(j, to_json(&m));
}
