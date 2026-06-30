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
        "(%module (%bind x (%meta ((%measure (%domain (%scalar real))) %fixed reals) (draw (Normal 0 1)))))",
        "(%module (%bind k (%meta ((%kernel (%inputs a)) %fixed %unknown) (functionof (%ref %local _a_) %specinputs ((a (%ref %local _a_)))))))",
        // %meta takes exactly three slots.
        "(%module (%bind x (%meta ((%scalar real) %fixed) (add 1 2))))",
        // Unknown mass class / value set.
        "(%module (%bind x (%meta ((%measure (%domain (%scalar real)) (%mass %tiny)) %fixed reals) (draw (Normal 0 1)))))",
        "(%module (%bind x (%meta ((%scalar real) %fixed %sometimes) (add 1 2))))",
        // Malformed set forms.
        "(%module (%bind x (%meta ((%scalar real) %fixed (stdsimplex)) (add 1 2))))",
        "(%module (%bind x (%meta ((%scalar real) %fixed (cartpow reals)) (add 1 2))))",
    ];
    for src in cases {
        assert!(read(src).is_err(), "should be rejected: {src}");
    }
}

/// The value-set slot round-trips every form class.
#[test]
fn valueset_forms_roundtrip() {
    let src = r#"(%module
  (%bind a (%meta ((%scalar real) %fixed (interval 0.0 inf)) (add 1.0 2.0)))
  (%bind b (%meta ((%scalar real) %fixed (stdsimplex %dynamic)) (add 1.0 2.0)))
  (%bind c (%meta ((%scalar real) %fixed (cartpow posintegers 4)) (add 1.0 2.0)))
  (%bind d (%meta ((%scalar real) %fixed anything) (add 1.0 2.0))))"#;
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

use flatppl_flatpir::{read as fp_read, write as fp_write};

/// The new heterogeneous value-set forms survive read → write → read.
#[test]
fn cartprod_and_record_valuesets_roundtrip() {
    // positional cartprod, multi-axis cartpow, and named record value-sets
    let src = "(%module \
        (%bind p (%meta ((%array 1 (2) (%scalar real)) %fixed (cartprod reals posreals)) (elementof reals))) \
        (%bind m (%meta ((%array 2 (2 3) (%scalar real)) %fixed (cartpow (cartpow reals 3) 2)) (elementof reals))) \
        (%bind r (%meta ((%record (a (%scalar real)) (b (%scalar real))) %fixed (record (a reals) (b unitinterval))) (elementof reals))))";
    let m1 = fp_read(src).expect("initial read");
    let s1 = fp_write(&m1);
    let m2 = fp_read(&s1).expect("re-read of canonical form");
    let s2 = fp_write(&m2);
    assert_eq!(s1, s2, "value-set forms not idempotent:\n{s1}");
    assert!(
        s1.contains("(cartprod reals posreals)"),
        "positional cartprod missing:\n{s1}"
    );
    assert!(
        s1.contains("(record (a reals) (b unitinterval))"),
        "record set missing:\n{s1}"
    );
    assert!(
        s1.contains("(cartpow (cartpow reals 3) 2)"),
        "nested cartpow missing:\n{s1}"
    );
}

/// Nested `%record` / `%tuple` / `%table` type annotations — and the matching
/// nested value-sets — survive read → write → read (spec §03/§04, flatppl-design
/// commit ee232b4: records nest, tuples nest, a table column may be a table).
/// The RHS is an inert `(elementof reals)`; only the `%meta` forms are exercised.
#[test]
fn nested_record_tuple_table_types_roundtrip() {
    let src = "(%module \
        (%bind r (%meta ((%record (a (%record (b (%scalar real)))) (d (%scalar real))) %fixed (record (a (record (b reals))) (d reals))) (elementof reals))) \
        (%bind p (%meta ((%tuple (%tuple (%scalar real) (%scalar integer)) (%scalar boolean)) %fixed (cartprod (cartprod reals integers) booleans)) (elementof reals))) \
        (%bind t (%meta ((%table (%columns (id (%scalar integer)) (hits (%record (x (%scalar real))))) (%nrows 2)) %fixed (record (id integers) (hits (record (x reals))))) (elementof reals))))";
    let m1 = fp_read(src).expect("initial read");
    let s1 = fp_write(&m1);
    let m2 = fp_read(&s1).expect("re-read of canonical form");
    let s2 = fp_write(&m2);
    assert_eq!(s1, s2, "nested type/value-set forms not idempotent:\n{s1}");
    assert!(
        s1.contains("(%record (a (%record (b (%scalar real)))) (d (%scalar real)))"),
        "nested record type missing:\n{s1}"
    );
    assert!(
        s1.contains("(%tuple (%tuple (%scalar real) (%scalar integer)) (%scalar boolean))"),
        "nested tuple type missing:\n{s1}"
    );
    assert!(
        s1.contains(
            "(%table (%columns (id (%scalar integer)) (hits (%record (x (%scalar real))))) (%nrows 2))"
        ),
        "nested table type missing:\n{s1}"
    );
    assert!(
        s1.contains("(record (id integers) (hits (record (x reals))))"),
        "nested table value-set missing:\n{s1}"
    );
}

/// A nested `%deferred` type slot — a record field / tuple element / array
/// element whose type inference is still a gap (e.g. an op with no rule) — is
/// emitted by the writer, so the reader must accept it; read → write → read must
/// round-trip rather than error (review finding F3). `Type::Deferred` is a legal
/// occupant of every nested slot.
#[test]
fn nested_deferred_type_roundtrips() {
    let src = "(%module \
        (%bind r (%meta ((%record (a %deferred) (b (%scalar real))) %fixed %unknown) (elementof reals))) \
        (%bind tup (%meta ((%tuple %deferred (%scalar integer)) %fixed %unknown) (elementof reals))) \
        (%bind arr (%meta ((%array 1 (2) %deferred) %fixed %unknown) (elementof reals))))";
    let m1 = fp_read(src).expect("read nested %deferred");
    let s1 = fp_write(&m1);
    let m2 = fp_read(&s1).expect("re-read of canonical form must succeed (F3)");
    let s2 = fp_write(&m2);
    assert_eq!(s1, s2, "nested %deferred not idempotent:\n{s1}");
    assert!(
        s1.contains("(%record (a %deferred) (b (%scalar real)))"),
        "nested record %deferred missing:\n{s1}"
    );
    assert!(
        s1.contains("(%tuple %deferred (%scalar integer))"),
        "nested tuple %deferred missing:\n{s1}"
    );
    assert!(
        s1.contains("(%array 1 (2) %deferred)"),
        "nested array %deferred missing:\n{s1}"
    );
}

/// A module with an explicit *empty* `(%public)` — no public bindings — must
/// survive `write → read`. The writer emits `(%public)` so re-reading uses the
/// explicit (empty) interface rather than the name-convention fallback (which
/// would otherwise flip the non-underscore `vis` to public). Regression test
/// for the empty-public writer lossiness.
#[test]
fn empty_public_interface_is_faithful() {
    let src = "(%module (%public) (%bind vis (elementof reals)))";
    let m = read(src).unwrap();
    assert_eq!(
        m.public_bindings().count(),
        0,
        "an explicit empty (%public) means nothing is public"
    );

    let text = write(&m);
    assert!(
        text.contains("(%public)"),
        "writer must emit an explicit empty (%public):\n{text}"
    );

    let m2 = read(&text).unwrap();
    assert_eq!(
        m2.public_bindings().count(),
        0,
        "empty public interface must round-trip, not name-convention `vis` to public"
    );
    assert_eq!(write(&m), write(&m2), "canonical fixpoint");
}
