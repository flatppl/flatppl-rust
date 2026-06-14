//! Native lint rules over the [`flatppl_core`] IR.

use std::collections::HashSet;

use flatppl_core::{Idx, Inputs, Module, Node, NodeId, RefNs, Symbol};

use crate::builtins::BUILTINS;
use crate::{Config, Diagnostic, RuleId, Severity, push};

/// Run the native rules in a single pass over the module's bindings.
///
/// Each rule is skipped when its configured severity is [`Severity::Allow`], so
/// a suppressed rule pays no cost — in particular the `unused-binding` reference
/// walk (the expensive one) runs only when that rule is active. Synthetic
/// bindings are skipped for every rule, so they are filtered once up front.
pub(crate) fn native(module: &Module, cfg: &Config) -> Vec<Diagnostic> {
    let unused_on = cfg.level(RuleId::UnusedBinding) != Severity::Allow;
    let shadow_on = cfg.level(RuleId::ShadowsBuiltin) != Severity::Allow;
    let doc_on = cfg.level(RuleId::MissingDoc) != Severity::Allow;

    let mut out = Vec::new();
    if !(unused_on || shadow_on || doc_on) {
        return out;
    }

    // The referenced-name set is computed once, and only when `unused-binding`
    // is active.
    let referenced = unused_on.then(|| referenced_self_names(module));

    for (_, binding) in module.bindings() {
        if binding.synthetic {
            continue;
        }
        // Resolved once and shared by every rule's check and message.
        let name = module.resolve(binding.name);

        // `shadows-builtin`: name collides with a built-in. `BUILTINS` is sorted
        // (generated from `keyword-lists.json`), so binary search beats a linear
        // scan — see the `builtins_sorted_and_unique` test.
        if shadow_on && BUILTINS.binary_search(&name).is_ok() {
            push(
                &mut out,
                cfg,
                RuleId::ShadowsBuiltin,
                format!("binding `{name}` shadows a built-in of the same name"),
                module.span_of(binding.rhs),
            );
        }

        // `unused-binding`: a private binding referenced by no other binding.
        if unused_on
            && !binding.public
            && !referenced
                .as_ref()
                .expect("referenced set is built when unused-binding is active")
                .contains(&binding.name)
        {
            push(
                &mut out,
                cfg,
                RuleId::UnusedBinding,
                format!("binding `{name}` is never used"),
                module.span_of(binding.rhs),
            );
        }

        // `missing-doc`: a public binding with no doc comment.
        if doc_on && binding.public && binding.doc.is_none() {
            push(
                &mut out,
                cfg,
                RuleId::MissingDoc,
                format!("public binding `{name}` has no doc comment"),
                module.span_of(binding.rhs),
            );
        }
    }
    out
}

/// Every self-module name referenced anywhere in the module's live nodes
/// (DFS from each binding's RHS). Covers ordinary refs and reification boundary
/// inputs (`Inputs::Spec` / filled `Inputs::Auto`), which `for_each_child` does
/// not visit.
///
/// `seen` is a dense `Vec<bool>` keyed by [`NodeId`] index (the arena is a DAG,
/// so a node can be reached more than once) — no per-node hashing.
fn referenced_self_names(module: &Module) -> HashSet<Symbol> {
    let mut refs = HashSet::with_capacity(module.binding_count());
    let mut seen = vec![false; module.node_count()];
    let mut stack: Vec<NodeId> = module.bindings().map(|(_, b)| b.rhs).collect();
    while let Some(id) = stack.pop() {
        let idx = id.index();
        if seen[idx] {
            continue;
        }
        seen[idx] = true;
        match module.node(id) {
            Node::Ref(r) if matches!(r.ns, RefNs::SelfMod) => {
                refs.insert(r.name);
            }
            Node::Call(call) => match &call.inputs {
                Some(Inputs::Spec(entries)) => {
                    for (_, r) in entries.iter() {
                        if matches!(r.ns, RefNs::SelfMod) {
                            refs.insert(r.name);
                        }
                    }
                }
                Some(Inputs::Auto) => {
                    if let Some(entries) = module.auto_inputs_of(id) {
                        for (_, r) in entries {
                            if matches!(r.ns, RefNs::SelfMod) {
                                refs.insert(r.name);
                            }
                        }
                    }
                }
                None => {}
            },
            _ => {}
        }
        module.for_each_child(id, |child| stack.push(child));
    }
    refs
}
