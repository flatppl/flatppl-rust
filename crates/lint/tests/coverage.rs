//! Coverage for rule/config edge paths that the surface-text tests don't reach:
//! the all-suppressed early return, the synthetic-binding skip, DAG re-visit in
//! the reference walk, `Inputs::Auto` boundary refs, and the infer-gap (Note)
//! bridge plus its `push` suppression.

use flatppl_core::{Binding, Call, CallHead, Inputs, Module, Node, Ref, RefNs, Scalar};
use flatppl_lint::{Config, RuleId, Severity, lint};

fn parse(src: &str) -> Module {
    flatppl_syntax::parse(src).expect("parse")
}

#[test]
fn all_native_rules_allowed_short_circuits() {
    // `sum` shadows a built-in and `_x` is unused, but with all three native
    // rules allowed, `native()` returns early and emits none of them.
    let mut m = parse("sum = 1.0\n_x = 2.0\n");
    let mut cfg = Config::default();
    for r in [
        RuleId::UnusedBinding,
        RuleId::ShadowsBuiltin,
        RuleId::MissingDoc,
    ] {
        cfg.set(r, Severity::Allow);
    }
    let ds = lint(&mut m, &cfg);
    assert!(ds.iter().all(|d| !matches!(
        d.rule,
        RuleId::UnusedBinding | RuleId::ShadowsBuiltin | RuleId::MissingDoc
    )));
}

#[test]
fn synthetic_bindings_are_skipped() {
    // `_ = 1.0` lowers to a synthetic discard binding; no rule must reference it.
    let mut m = parse("_ = 1.0\nmu = 0.0\n");
    let ds = lint(&mut m, &Config::default());
    assert!(ds.iter().all(|d| !d.message.contains("__0x")));
}

#[test]
fn shared_node_visited_once() {
    // One literal reused as both args of a call: the reference walk must guard
    // against re-visiting it (the arena is a DAG).
    let mut m = Module::new();
    let lit = m.alloc(Node::Lit(Scalar::Real(1.0)));
    let add = m.intern("add");
    let call = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(add),
        args: Box::new([lit, lit]),
        named: Box::new([]),
        inputs: None,
    }));
    let x = m.intern("x");
    m.add_binding(Binding {
        name: x,
        rhs: call,
        doc: None,
        public: true,
        synthetic: false,
    });
    // Exercises the walk; a public `x` referencing a built-in is otherwise clean.
    let _ = lint(&mut m, &Config::default());
}

#[test]
fn auto_inputs_boundary_ref_counts_as_use() {
    // `_k` is referenced only via a reification's filled `%autoinputs` side
    // table; the walk must read it so `_k` is not flagged unused.
    let mut m = Module::new();
    let k_lit = m.alloc(Node::Lit(Scalar::Real(2.0)));
    let k = m.intern("_k");
    m.add_binding(Binding {
        name: k,
        rhs: k_lit,
        doc: None,
        public: false,
        synthetic: false,
    });

    let body = m.alloc(Node::Lit(Scalar::Real(0.0)));
    let functionof = m.intern("functionof");
    let f_call = m.alloc(Node::Call(Call {
        head: CallHead::Builtin(functionof),
        args: Box::new([body]),
        named: Box::new([]),
        inputs: Some(Inputs::Auto),
    }));
    let kw = m.intern("k");
    m.set_auto_inputs(
        f_call,
        Box::new([(
            kw,
            Ref {
                ns: RefNs::SelfMod,
                name: k,
            },
        )]),
    );
    let f = m.intern("f");
    m.add_binding(Binding {
        name: f,
        rhs: f_call,
        doc: None,
        public: true,
        synthetic: false,
    });

    let ds = lint(&mut m, &Config::default());
    assert!(
        !ds.iter().any(|d| d.rule == RuleId::UnusedBinding),
        "_k is used via auto-inputs but was flagged: {ds:?}"
    );
}

#[test]
fn unknown_op_yields_inference_gap_note() {
    let mut m = parse("y = somethingweird(1.0)\n");
    let ds = lint(&mut m, &Config::default());
    assert!(ds.iter().any(|d| d.rule == RuleId::InferenceGap));
}

#[test]
fn allowed_inference_gap_is_suppressed_while_infer_still_runs() {
    // inference-gap allowed, but unresolved/cycle stay active → the infer pass
    // still runs and the gap note is dropped by `push`.
    let mut m = parse("y = somethingweird(1.0)\n");
    let mut cfg = Config::default();
    cfg.set(RuleId::InferenceGap, Severity::Allow);
    let ds = lint(&mut m, &cfg);
    assert!(!ds.iter().any(|d| d.rule == RuleId::InferenceGap));
}
