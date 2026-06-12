//! FlatPIR round-trip tests over the workspace `fixtures/flatpir/*.flatpir`
//! corpus.
//!
//! Contract (canonical, not byte-preserving): `read → write → read → write`
//! reaches a byte-stable fixpoint, and the canonical text always re-reads. The
//! fixtures are spec-§11-sourced (no oracle contamination from any engine).

use std::fs;
use std::path::PathBuf;

use flatppl_core::{Phase, ScalarType, Type};
use flatppl_flatpir::{read, write};

fn fixture(name: &str) -> String {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "../../fixtures/flatpir", name]
        .iter()
        .collect();
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// `read → write → read → write` must reach a byte-stable canonical fixpoint,
/// and the canonical form must itself re-read (proving `write` emits valid
/// FlatPIR that the reader accepts).
fn assert_canonical_fixpoint(name: &str) {
    let src = fixture(name);
    let m1 = read(&src).unwrap_or_else(|e| panic!("{name}: initial read failed: {e}"));
    let s1 = write(&m1);
    let m2 = read(&s1).unwrap_or_else(|e| {
        panic!("{name}: re-read of canonical form failed: {e}\n--- canonical ---\n{s1}")
    });
    let s2 = write(&m2);
    assert_eq!(s1, s2, "{name}: canonical form is not idempotent");
}

macro_rules! roundtrip_tests {
    ($($test:ident => $file:literal),* $(,)?) => {
        $(#[test] fn $test() { assert_canonical_fixpoint($file); })*
    };
}

roundtrip_tests! {
    rt_helpers => "helpers.flatpir",
    rt_model => "model.flatpir",
    rt_helpers_annotated => "helpers-annotated.flatpir",
    rt_model_annotated => "model-annotated.flatpir",
    rt_user_call => "user-call.flatpir",
    rt_values => "values.flatpir",
    rt_aggregate => "aggregate.flatpir",
    rt_docs => "docs.flatpir",
    rt_reified => "reified.flatpir",
}

/// The reified-callable input forms (spec §11): `%specinputs` lands in
/// IR-proper ([`flatppl_core::Inputs::Spec`]); a filled `%autoinputs` list is
/// inference metadata and lands in the auto-inputs side-table; `%deferred`
/// leaves the side-table empty.
#[test]
fn reified_inputs_route_to_ir_and_side_table() {
    use flatppl_core::{Inputs, Node, RefNs};
    let m = read(&fixture("reified.flatpir")).unwrap();
    let rhs = |name: &str| {
        let (id, _) = m
            .bindings()
            .find(|(_, b)| m.resolve(b.name) == name)
            .unwrap_or_else(|| panic!("no binding {name}"));
        m.binding(id).rhs
    };

    // f: %autoinputs %deferred — Auto inputs, no side-table entry.
    let f = rhs("f");
    let Node::Call(fc) = m.node(f) else { panic!() };
    assert_eq!(fc.inputs, Some(Inputs::Auto));
    assert!(m.auto_inputs_of(f).is_none());

    // g: filled %autoinputs — Auto inputs + side-table entry (a -> self a).
    let g = rhs("g");
    let Node::Call(gc) = m.node(g) else { panic!() };
    assert_eq!(gc.inputs, Some(Inputs::Auto));
    let entries = m.auto_inputs_of(g).expect("g auto-inputs filled");
    assert_eq!(entries.len(), 1);
    assert_eq!(m.resolve(entries[0].0), "a");
    assert_eq!(entries[0].1.ns, RefNs::SelfMod);

    // helpers obs_kernel: %specinputs — Spec entries in IR-proper, placeholder
    // entry carries the %local namespace.
    let m2 = read(&fixture("helpers.flatpir")).unwrap();
    let (kid, _) = m2
        .bindings()
        .find(|(_, b)| m2.resolve(b.name) == "obs_kernel")
        .unwrap();
    let Node::Call(kc) = m2.node(m2.binding(kid).rhs) else {
        panic!()
    };
    let Some(Inputs::Spec(entries)) = &kc.inputs else {
        panic!("obs_kernel must have %specinputs");
    };
    let names: Vec<&str> = entries.iter().map(|(n, _)| m2.resolve(*n)).collect();
    assert_eq!(names, ["center", "spread", "x"]);
    assert_eq!(entries[2].1.ns, RefNs::Local);
    assert_eq!(m2.resolve(entries[2].1.name), "_x_");
}

/// Pins the exact canonical layout so accidental formatting drift is caught.
#[test]
fn canonical_format_golden() {
    let src = "(%module (%public y) (%bind _x 1) (%bind y (add (%ref self _x) 2.0)))";
    let got = write(&read(src).unwrap());
    let expected = "\
(%module
  (%public y)

  (%bind _x 1)

  (%bind y (add (%ref self _x) 2.0)))";
    assert_eq!(got, expected);
}

/// `%public` is the authored interface, not "all non-underscore names": the
/// spec §11 model omits `helpers` (a load_module binding) and `_combined`, so
/// both must round-trip as private.
#[test]
fn public_list_is_authored_not_derived() {
    let m = read(&fixture("model.flatpir")).unwrap();
    let all: Vec<&str> = m.bindings().map(|(_, b)| m.resolve(b.name)).collect();
    assert_eq!(all, ["helpers", "a", "b", "_combined", "input_data", "L"]);
    let public: Vec<&str> = m
        .public_bindings()
        .map(|(_, b)| m.resolve(b.name))
        .collect();
    assert_eq!(public, ["a", "b", "input_data", "L"]);
}

/// `%meta` annotations flow into the per-node side-tables (and back out on
/// write — covered by the fixpoint test).
#[test]
fn meta_annotations_populate_side_tables() {
    let m = read(&fixture("helpers-annotated.flatpir")).unwrap();
    let sv = m
        .bindings()
        .find(|(_, b)| m.resolve(b.name) == "shifted_value")
        .map(|(id, _)| id)
        .expect("shifted_value binding");
    let rhs = m.binding(sv).rhs;
    assert_eq!(m.type_of(rhs), Some(&Type::Scalar(ScalarType::Real)));
    assert_eq!(m.binding_phase(sv), Some(Phase::Parameterized));
}

/// A bare (unannotated) module carries no `%meta` side-table entries.
#[test]
fn bare_module_has_no_annotations() {
    let m = read(&fixture("model.flatpir")).unwrap();
    for (id, _) in m.bindings() {
        let rhs = m.binding(id).rhs;
        assert_eq!(m.type_of(rhs), None);
        assert_eq!(m.binding_phase(id), None);
    }
}

#[test]
fn reports_malformed_input() {
    assert!(read("(%module (%bind x))").is_err()); // missing expression
    assert!(read("(%bind x 1)").is_err()); // no (%module …) wrapper
    assert!(read("(%module (%public ghost))").is_err()); // public names unknown binding
}

/// The §11 strictness contract for the new forms: a massless measure/kernel
/// type, a two-slot `%meta`, and malformed mass/set values are parse errors
/// (early days — no tolerant reading of incomplete forms).
#[test]
fn mass_and_valueset_strictness() {
    let cases = [
        // %mass is a required sub-form.
        "(%module (%bind x (draw (%meta (%measure (%domain (%scalar real))) %fixed reals) (Normal 0 1))))",
        "(%module (%bind k (functionof (%meta (%kernel (%inputs a)) %fixed %unknown) (%ref %local _a_) %specinputs ((a (%ref %local _a_))))))",
        // %meta takes exactly three slots.
        "(%module (%bind x (add (%meta (%scalar real) %fixed) 1 2)))",
        // Unknown mass class / value set.
        "(%module (%bind x (draw (%meta (%measure (%domain (%scalar real)) (%mass %tiny)) %fixed reals) (Normal 0 1))))",
        "(%module (%bind x (add (%meta (%scalar real) %fixed %sometimes) 1 2)))",
        // Malformed set forms.
        "(%module (%bind x (add (%meta (%scalar real) %fixed (stdsimplex)) 1 2)))",
        "(%module (%bind x (add (%meta (%scalar real) %fixed (cartpow reals)) 1 2)))",
    ];
    for src in cases {
        assert!(read(src).is_err(), "should be rejected: {src}");
    }
}

/// The value-set slot round-trips every form class.
#[test]
fn valueset_forms_roundtrip() {
    let src = r#"(%module
  (%bind a (add (%meta (%scalar real) %fixed (interval 0.0 inf)) 1.0 2.0))
  (%bind b (add (%meta (%scalar real) %fixed (stdsimplex %dynamic)) 1.0 2.0))
  (%bind c (add (%meta (%scalar real) %fixed (cartpow posintegers 4)) 1.0 2.0))
  (%bind d (add (%meta (%scalar real) %fixed anything) 1.0 2.0)))"#;
    let m = read(src).unwrap();
    let out = write(&m);
    for needle in [
        "(interval 0.0 inf)",
        "(stdsimplex %dynamic)",
        "(cartpow posintegers 4)",
        " anything)",
    ] {
        assert!(out.contains(needle), "missing {needle} in:\n{out}");
    }
    assert_eq!(write(&read(&out).unwrap()), out);
}
