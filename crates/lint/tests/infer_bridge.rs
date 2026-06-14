//! Integration tests for the `flatppl-lint` infer-bridge rules:
//! `inference-cycle`, `unresolved-name`, and (incidentally) `inference-gap`.
//!
//! These rules are wired in `infer_bridge()` in `crates/lint/src/lib.rs`:
//!   - infer `Error` with "cycle" in message  → `RuleId::InferenceCycle`  (Deny)
//!   - infer `Error` without "cycle"          → `RuleId::UnresolvedName`  (Deny)
//!   - infer `Note`                           → `RuleId::InferenceGap`    (Warn)

use flatppl_core::{Binding, Module, Node, Ref, RefNs};
use flatppl_lint::{Config, Diagnostic, RuleId, Severity, lint};

// ---- helpers ---------------------------------------------------------------

/// Parse surface FlatPPL into a `Module`; panics on error (inputs are known-good).
fn parse(src: &str) -> Module {
    flatppl_syntax::parse(src).expect("parse")
}

/// Run lint with the default config and return all diagnostics.
fn diags(src: &str) -> Vec<Diagnostic> {
    let mut m = parse(src);
    lint(&mut m, &Config::default())
}

// ---- Test 1: inference-cycle -----------------------------------------------
//
// `a = b` / `b = a` is a mutually-referential pair.  The surface parser emits
// `(%ref self a)` / `(%ref self b)` nodes, so the infer trace hits the cycle
// guard in `infer_binding` and emits an Error whose message contains "cycle".
// The bridge maps that Error to `RuleId::InferenceCycle` at `Severity::Deny`.

#[test]
fn cycle_maps_to_inference_cycle_deny() {
    let ds = diags("a = b\nb = a\n");
    let cyc = ds.iter().find(|d| d.rule == RuleId::InferenceCycle);
    assert!(
        cyc.is_some(),
        "expected InferenceCycle diagnostic, got: {ds:?}"
    );
    assert_eq!(cyc.unwrap().severity, Severity::Deny);
}

// Perf gate: when all three infer-bridge rules are `Allow`, `lint` skips the
// (dominant-cost) infer pass entirely. Verify that path is correct — a cyclic
// model produces no infer-bridge diagnostics and does not panic.
#[test]
fn all_infer_rules_allowed_skips_infer_and_reports_nothing() {
    let mut m = parse("a = b\nb = a\n");
    let mut cfg = Config::default();
    for rule in [
        RuleId::UnresolvedName,
        RuleId::InferenceCycle,
        RuleId::InferenceGap,
    ] {
        cfg.set(rule, Severity::Allow);
    }
    let ds = lint(&mut m, &cfg);
    assert!(
        !ds.iter().any(|d| d.rule == RuleId::InferenceCycle
            || d.rule == RuleId::UnresolvedName
            || d.rule == RuleId::InferenceGap),
        "infer-bridge rules were all allowed but a diagnostic surfaced: {ds:?}"
    );
}

// ---- Test 2: unresolved-name (raw IR) --------------------------------------
//
// The surface parser's binding pre-pass only emits `(%ref self X)` for names
// that already appear as binding LHS-es in the same source text, so a dangling
// self-ref that resolves to nothing cannot be written from surface FlatPPL.
// We construct the IR directly:
//
//   x = does_not_exist   (as a `(%ref self does_not_exist)` node)
//
// The infer trace hits `binding_by_name("does_not_exist") == None` (line ~153
// of crates/infer/src/trace.rs) and emits Error("unresolved reference …").
// No "cycle" in that message, so the bridge routes it to `RuleId::UnresolvedName`.

#[test]
fn dangling_self_ref_maps_to_unresolved_name_deny() {
    let mut m = Module::new();

    // Name that is NOT a binding in the module — infer will fail to resolve it.
    let missing = m.intern("does_not_exist");
    let rhs = m.alloc(Node::Ref(Ref {
        ns: RefNs::SelfMod,
        name: missing,
    }));

    let name = m.intern("x");
    m.add_binding(Binding {
        name,
        rhs,
        doc: None,
        public: true,
        synthetic: false,
    });

    let ds = lint(&mut m, &Config::default());
    let unres = ds.iter().find(|d| d.rule == RuleId::UnresolvedName);
    assert!(
        unres.is_some(),
        "expected UnresolvedName diagnostic, got: {ds:?}"
    );
    assert_eq!(unres.unwrap().severity, Severity::Deny);
}
