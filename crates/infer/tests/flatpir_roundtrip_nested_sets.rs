//! Print-then-read agreement for deeply annotated FlatPIR.
//!
//! Inference annotations nest as deep as the value they describe: spec §11 says
//! "A value-set expression's shape parameters ... must agree with the node's
//! structural shape", so a rank-64 array's value set is 64 nested `cartpow`
//! forms, and every enclosing record or tuple adds its own type and value-set
//! wrapper on top. §11 fixes no nesting bound of its own, so the printer and
//! the reader must agree: the checked writer must refuse exactly the text the
//! reader refuses, and never hand back text no tool can read.
//!
//! Stimuli are inlined rather than added to `fixtures/flatppl/`: they are
//! resource-guard probes, not cross-engine corpus models.

use flatppl_infer::{Severity, infer};

/// A rank-64 array — the `addaxes` structural-rank cap, so this is the deepest
/// value set inference will build from a single call.
const RANK_CAP_ARRAY: &str = "addaxes(v, 0, 63)";

fn record_level(inner: String) -> String {
    format!("record(f = {inner})")
}

fn tuple_level(inner: String) -> String {
    format!("({inner}, 0.0)")
}

/// A module binding `w` to [`RANK_CAP_ARRAY`] under `levels` wrappers.
fn model(wrap: fn(String) -> String, levels: usize) -> String {
    let mut inner = RANK_CAP_ARRAY.to_string();
    for _ in 0..levels {
        inner = wrap(inner);
    }
    format!("v = [1.0, 2.0, 3.0]\nw = {inner}\n")
}

/// Infer `src`, requiring it to be free of errors.
fn inferred(src: &str) -> flatppl_core::Module {
    let mut m = flatppl_syntax::parse(src).expect("stimulus parses");
    let errors: Vec<String> = infer(&mut m)
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect();
    assert!(
        errors.is_empty(),
        "stimulus should infer cleanly: {errors:?}"
    );
    m
}

/// Walk `levels` of nesting, asserting the checked writer and the reader agree
/// at every level. Returns `(accepted, refused)` so a caller can require that
/// the sweep actually crossed the boundary rather than staying on one side.
fn agreement_over(
    wrap: fn(String) -> String,
    levels: std::ops::RangeInclusive<usize>,
) -> (u32, u32) {
    let (mut accepted, mut refused) = (0, 0);
    for n in levels {
        let module = inferred(&model(wrap, n));
        match flatppl_flatpir::try_write(&module) {
            Ok(text) => {
                accepted += 1;
                if let Err(e) = flatppl_flatpir::read(&text) {
                    panic!("level {n}: try_write accepted text the reader rejects: {e}");
                }
            }
            Err(refusal) => {
                refused += 1;
                assert!(
                    refusal.message.contains("cannot be read back"),
                    "level {n}: refusal should name the reason, got: {}",
                    refusal.message
                );
                // The unchecked writer is what the CLI used to emit. Confirm the
                // refusal is real rather than the check being over-eager.
                let unchecked = flatppl_flatpir::write(&module);
                assert!(
                    flatppl_flatpir::read(&unchecked).is_err(),
                    "level {n}: try_write refused text the reader accepts"
                );
            }
        }
    }
    (accepted, refused)
}

/// The bare rank-cap array is shallow enough to survive the round trip, so the
/// rank cap alone keeps the unwrapped witness readable.
#[test]
fn a_rank_cap_array_prints_and_reads_back() {
    let module = inferred(&model(record_level, 0));
    let text = flatppl_flatpir::try_write(&module).expect("rank-64 annotation renders");
    assert!(
        text.contains("(%array 64 "),
        "the stimulus should reach the rank cap, got:\n{text}"
    );
    flatppl_flatpir::read(&text).expect("rank-64 annotation reads back");
}

/// A record wrapper adds a type wrapper, a value-set wrapper and a `%meta` per
/// level, so nesting records over the rank-cap array crosses the reader's
/// guard. Before the fix inference exited 0 here and the output would not read.
#[test]
fn record_wrapped_rank_cap_arrays_print_only_when_they_read_back() {
    let (accepted, refused) = agreement_over(record_level, 0..=24);
    assert!(
        accepted > 0 && refused > 0,
        "{accepted} accepted, {refused} refused"
    );
}

/// The tuple wrapper is one level cheaper per step than the record wrapper, so
/// it crosses the same boundary later; the invariant is the same.
#[test]
fn tuple_wrapped_rank_cap_arrays_print_only_when_they_read_back() {
    let (accepted, refused) = agreement_over(tuple_level, 0..=34);
    assert!(
        accepted > 0 && refused > 0,
        "{accepted} accepted, {refused} refused"
    );
}

/// The JSON encoder renders through the same checked path, so it refuses rather
/// than panicking on a module the text writer cannot represent.
#[test]
fn json_encoding_refuses_an_unreadable_module() {
    let module = inferred(&model(record_level, 24));
    let err = flatppl_flatpir::try_to_json(&module).expect_err("should refuse");
    assert!(
        err.message.contains("cannot be read back"),
        "got: {}",
        err.message
    );
}
