//! `emit_query` builds its own emitter, so it also points alias references at
//! the end of their chain before the emitter's one-hop reads. Canonicalization
//! does that too, so only `FLATPPL_DETERMINIZE_NO_CANON` (the det-js
//! equivalence gate) lets an alias reach it. One test per binary: the variable
//! is process-wide.

use flatppl_host::{Context, EmitOptions};

fn compile(source: &str) -> String {
    let mut context = Context::default();
    let module = context.parse(source, None, None).unwrap();
    module
        .compile(&EmitOptions::default())
        .unwrap_or_else(|e| panic!("{e:?}"))
        .stablehlo
}

#[test]
fn an_uncanonicalized_alias_chain_emits_like_the_direct_set() {
    // SAFETY: the only test in this binary, so no other thread reads the environment.
    unsafe { std::env::set_var("FLATPPL_DETERMINIZE_NO_CANON", "1") };
    let tail = "x = elementof(reals)\nv = ifelse(x in S, 1.0, 0.0)\n\
                inputs = (x)\noutputs = (v)\n";
    assert_eq!(
        compile(&format!("S0 = interval(0.0, 2.0)\nS1 = S0\nS = S1\n{tail}")),
        compile(&format!("S = interval(0.0, 2.0)\n{tail}")),
    );
}
