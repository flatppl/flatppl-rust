//! `textDocument/references`, `textDocument/prepareRename` and
//! `textDocument/rename`.
//!
//! Find-references is the inverse of [`resolve_ref_def`]: instead of walking one
//! reference to its definition, it walks every `Ref` node in every file of the
//! bundle and keeps the ones that resolve to the target binding. Because the
//! resolver is shared, references follow the same alias chains and module hops
//! that hover and go-to-definition already follow.
//!
//! Rename adds the definition site to that set and rewrites all of them. What it
//! refuses, and on which normative rule, is in [`rename_edits`].
//!
//! [`resolve_ref_def`]: crate::capabilities::resolve_ref_def

use std::collections::HashSet;

use flatppl_core::{BindingId, Call, CallHead, Idx, Inputs, Node, NodeId, Ref, RefNs};

use crate::capabilities::resolve_ref_def;
use crate::db::{Catalogues, FileSet, SourceFile};
use crate::names::{self, RefPart, Refusal};
use crate::queries::{SpanIndex, node_at_offset_indexed, parse};

/// One occurrence of a name: the file's stored path plus the half-open byte
/// range of the identifier itself (not of the enclosing expression).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct NameLoc {
    pub path: String,
    pub start: u32,
    pub end: u32,
}

/// What a rename or a references request is about.
enum Target {
    /// A module binding, possibly defined in another file of the bundle.
    Binding(BindingTarget),
    /// A function/kernel argument name. §04 "Objects, expressions, names and
    /// modules" keeps argument names out of the module namespace, and §04
    /// "Placeholders and holes" scopes a placeholder to "the nearest enclosing
    /// `functionof` or `kernelof`", so body occurrences are confined to one
    /// reification, including when several reifications share a binding.
    Argument(ArgumentTarget),
}

/// A module binding target.
struct BindingTarget {
    /// The file that defines the binding — not necessarily the requesting file.
    file: SourceFile,
    bid: BindingId,
    name: String,
}

/// A callable-argument target, in the file the request came from.
struct ArgumentTarget {
    file: SourceFile,
    /// The reification that declares this argument, possibly a nested lambda.
    reification: NodeId,
    declaration: (u32, u32),
    /// The surface argument name, e.g. `a`.
    name: String,
    /// The lowered placeholder the body references, e.g. `_a_`.
    placeholder: String,
    /// The boundary value of a keyword boundary `a = _a_`, when the rename also
    /// rewrites the placeholder, which it does when that is spelled `_a_`.
    boundary_value: Option<(u32, u32)>,
}

/// Resolve what the cursor at `byte_offset` designates, or refuse with the
/// reason to show the user.
///
/// Definition sites are checked first: a `~` binding's RHS node span starts at
/// the binding name, so the node under the cursor there is the `draw` call, not
/// a reference. Only then is the node at the offset consulted.
fn resolve_target(
    db: &dyn salsa::Database,
    file: SourceFile,
    fs: FileSet,
    cats: Catalogues,
    byte_offset: u32,
    index: &SpanIndex,
) -> Result<Target, Refusal> {
    let unresolved = || {
        Refusal(
            "no FlatPPL binding under the cursor: built-in names, standard-module \
             members, record fields and literals are not module bindings and cannot \
             be renamed (spec §04 \"Name resolution\", §04 \"Objects, expressions, \
             names and modules\")"
                .to_string(),
        )
    };
    let module = parse(db, file).module(db).ok_or_else(|| {
        Refusal("the file does not parse, so no binding can be resolved".to_string())
    })?;
    let text = file.text(db);

    // A definition site: the cursor is on a binding's own name.
    for (bid, binding) in module.bindings() {
        if binding.synthetic {
            continue;
        }
        let Some(span) = module.span_of(binding.rhs) else {
            continue;
        };
        let name = module.resolve(binding.name);
        let Some((start, end)) = names::def_name_range(text, span.start, name) else {
            continue;
        };
        if start <= byte_offset && byte_offset < end {
            return Ok(Target::Binding(BindingTarget {
                file,
                bid,
                name: name.to_string(),
            }));
        }
        // An argument declaration site: the cursor is inside the callable's
        // argument list, which carries no node of its own.
        if let Some(t) = argument_target_at(
            module,
            text,
            file,
            binding.rhs,
            Some((start, end)),
            byte_offset,
            None,
        ) {
            return Ok(t);
        }
    }

    // Named-function declarations can precede their RHS span, and were handled
    // above. Within an expression, the nearest reification owns the placeholder.
    let owner = (0..module.node_count())
        .map(NodeId::from_usize)
        .filter(|&id| matches!(module.node(id), Node::Call(c) if c.inputs.is_some()))
        .filter_map(|id| {
            let span = module.span_of(id)?;
            (span.start <= byte_offset && byte_offset < span.end)
                .then_some((id, span.end - span.start))
        })
        .min_by_key(|(_, width)| *width)
        .map(|(id, _)| id);
    let def_name = owner.and_then(|id| {
        let span = module.span_of(id)?;
        module
            .bindings()
            .find(|(_, b)| b.rhs == id)
            .and_then(|(_, b)| names::def_name_range(text, span.start, module.resolve(b.name)))
    });
    if let Some(id) = owner
        && let Some(t) = argument_target_at(module, text, file, id, def_name, byte_offset, None)
    {
        return Ok(t);
    }

    let node_id = node_at_offset_indexed(index, byte_offset).ok_or_else(unresolved)?;
    let Node::Ref(r) = module.node(node_id) else {
        return Err(unresolved());
    };
    let span = module.span_of(node_id).ok_or_else(unresolved)?;
    match r.ns {
        // A function/kernel argument. §04 keeps argument names out of the module
        // namespace, so this is not a binding rename — but nothing in the docs
        // forbids renaming one, so it is supported as its own scoped operation.
        RefNs::Local => {
            let placeholder = module.resolve(r.name).to_string();
            let reification = owner.ok_or_else(|| {
                Refusal(format!(
                    "could not find the callable that declares `{placeholder}`"
                ))
            })?;
            argument_target_at(
                module,
                text,
                file,
                reification,
                def_name,
                byte_offset,
                Some(&placeholder),
            )
            .ok_or_else(|| {
                Refusal(format!(
                    "`{placeholder}` is declared by a bare-name `functionof`/`kernelof` \
                     boundary, which has no keyword to rename. Renaming the argument of \
                     a keyword boundary (`a = _a_`), a `f(a) = …` definition or a lambda \
                     is supported."
                ))
            })
        }
        RefNs::Module(alias) => {
            // The node spans `alias.member`. When the cursor is on the alias half
            // the target is the alias binding in this file; the member half
            // resolves through the bundle. (The parser also emits a bare `Ref` for
            // the alias, which usually wins the offset lookup, so this is the
            // defensive path for the alias half.)
            let alias_end = names::head_ident_end(text, span.start).unwrap_or(span.start);
            if byte_offset < alias_end {
                let alias_ref = Ref {
                    ns: RefNs::SelfMod,
                    name: alias,
                };
                let (def_file, bid) = resolve_ref_def(db, file, fs, cats, module, &alias_ref)
                    .ok_or_else(unresolved)?;
                let def_mod = parse(db, def_file).module(db).ok_or_else(unresolved)?;
                let name = def_mod.resolve(def_mod.binding(bid).name).to_string();
                return Ok(Target::Binding(BindingTarget {
                    file: def_file,
                    bid,
                    name,
                }));
            }
            // A `standard_module` member has no workspace file to edit, so
            // `resolve_ref_def` finds nothing and the rename refuses.
            let (def_file, bid) =
                resolve_ref_def(db, file, fs, cats, module, r).ok_or_else(|| {
                    Refusal(
                        "this name is a standard-module member (spec §09), defined by \
                         the catalogue rather than by a workspace file, and cannot be \
                         renamed"
                            .to_string(),
                    )
                })?;
            let def_mod = parse(db, def_file).module(db).ok_or_else(unresolved)?;
            let name = def_mod.resolve(def_mod.binding(bid).name).to_string();
            Ok(Target::Binding(BindingTarget {
                file: def_file,
                bid,
                name,
            }))
        }
        RefNs::SelfMod => {
            let (def_file, bid) =
                resolve_ref_def(db, file, fs, cats, module, r).ok_or_else(unresolved)?;
            let def_mod = parse(db, def_file).module(db).ok_or_else(unresolved)?;
            let name = def_mod.resolve(def_mod.binding(bid).name).to_string();
            Ok(Target::Binding(BindingTarget {
                file: def_file,
                bid,
                name,
            }))
        }
    }
}

/// Build an [`ArgumentTarget`] for one reification.
///
/// Selects the argument either by `placeholder` (when the cursor was on a body
/// reference) or by the cursor falling inside a declared argument's range (when it
/// was on the declaration list). The declaration is the keyword of a keyword
/// boundary `functionof(…, a = _a_)`, or the surface argument of the two sugar
/// spellings. Returns `None` when the node is not a reification with an ordered
/// boundary, when the declarations cannot be located, or when the requested
/// argument is not among them — every one a reason to fail closed rather than edit.
fn argument_target_at(
    module: &flatppl_core::Module,
    text: &str,
    file: SourceFile,
    reification: NodeId,
    def_name: Option<(u32, u32)>,
    byte_offset: u32,
    placeholder: Option<&str>,
) -> Option<Target> {
    let Node::Call(call) = module.node(reification) else {
        return None;
    };
    let CallHead::Builtin(head) = call.head else {
        return None;
    };
    let head_name = module.resolve(head);
    if head_name != "functionof" && head_name != "kernelof" {
        return None;
    }
    // `%autoinputs` has no ordered surface argument list to rename.
    let Inputs::Spec(entries) = call.inputs.as_ref()? else {
        return None;
    };
    let boundary = module.boundary_spans_of(reification);
    let declared: Vec<(String, u32, u32)> = match boundary {
        Some(spans) => entries
            .iter()
            .zip(spans)
            .map(|((sym, _), (keyword, _))| {
                (module.resolve(*sym).to_string(), keyword.start, keyword.end)
            })
            .collect(),
        None => names::argument_decl_ranges(text, module.span_of(reification)?.start, def_name),
    };
    if declared.len() != entries.len() {
        return None;
    }

    let index = match placeholder {
        // A body reference: the entry that declares this placeholder.
        Some(ph) => entries
            .iter()
            .position(|(_, r)| matches!(r.ns, RefNs::Local) && module.resolve(r.name) == ph)?,
        // A declaration site: the argument whose range contains the cursor.
        None => declared
            .iter()
            .position(|(_, s, e)| *s <= byte_offset && byte_offset < *e)?,
    };
    let (sym, entry) = &entries[index];
    let (name, start, end) = &declared[index];
    // The source text and the IR must agree, so a shape this code does not
    // model, or a stale span table, fails closed.
    if module.resolve(*sym) != name || text.get(*start as usize..*end as usize) != Some(name) {
        return None;
    }
    let placeholder = module.resolve(entry.name).to_string();
    let boundary_value = boundary
        .filter(|_| matches!(entry.ns, RefNs::Local) && placeholder == format!("_{name}_"))
        .map(|spans| (spans[index].1.start, spans[index].1.end));
    Some(Target::Argument(ArgumentTarget {
        file,
        reification,
        declaration: (*start, *end),
        name: name.clone(),
        placeholder,
        boundary_value,
    }))
}

/// Every occurrence of the target binding's name across the bundle: each `Ref`
/// node in each file of `fs` that resolves to it, plus the definition site when
/// `include_declaration`.
///
/// The walk uses `parse`, not `analyze`: reference resolution reads only
/// structure (binding names, node kinds, directive paths), so a references
/// request over a large workspace does not force type inference on every file.
///
/// Every node index is visited rather than only the ones reachable from a
/// binding RHS. The parser emits a bare `Ref` for the alias half of
/// `alias.member` that no binding owns, and that node is a genuine occurrence of
/// the alias name. Results are deduplicated by range, so a name reached twice
/// yields one edit.
fn occurrences(
    db: &dyn salsa::Database,
    fs: FileSet,
    cats: Catalogues,
    target: &Target,
    include_declaration: bool,
) -> Vec<NameLoc> {
    match target {
        Target::Binding(t) => binding_occurrences(db, fs, cats, t, include_declaration),
        Target::Argument(t) => argument_occurrences(db, fs, cats, t, include_declaration),
    }
}

/// Every occurrence of an argument name: each `%local` reference to its
/// placeholder inside the owning reification, each `name = value` keyword at a
/// call of the callable, plus the declaration site in the argument list.
///
/// Confined to one reification in one file. §04 "Placeholders and holes" gives the
/// scoping rule — "The scope of a placeholder is the nearest enclosing
/// `functionof` or `kernelof`" — and adds that "The same placeholder name may
/// appear in different scopes without conflict", so traversal stops at nested
/// reifications even when they declare the same placeholder name.
fn argument_occurrences(
    db: &dyn salsa::Database,
    fs: FileSet,
    cats: Catalogues,
    target: &ArgumentTarget,
    include_declaration: bool,
) -> Vec<NameLoc> {
    let Some(module) = parse(db, target.file).module(db) else {
        return Vec::new();
    };
    let text = target.file.text(db);
    let path = target.file.path(db).clone();
    let mut out: Vec<NameLoc> = Vec::new();
    let mut seen: HashSet<NameLoc> = HashSet::new();
    let mut push = |loc: NameLoc| {
        if seen.insert(loc.clone()) {
            out.push(loc);
        }
    };
    let at = |start: u32, end: u32| NameLoc {
        path: path.clone(),
        start,
        end,
    };

    for id in scope_nodes(module, target.reification, false) {
        let Node::Ref(r) = module.node(id) else {
            continue;
        };
        if !matches!(r.ns, RefNs::Local) || module.resolve(r.name) != target.placeholder {
            continue;
        }
        let Some(span) = module.span_of(id) else {
            continue;
        };
        // The body writes the SURFACE name for a sugar argument, so the range must
        // match that, not the placeholder. A mismatch fails closed.
        if let Some((s, e)) = names::ref_name_range(text, span.start, RefPart::Head, &target.name) {
            push(at(s, e));
        } else if target.boundary_value.is_some()
            && let Some((s, e)) =
                names::ref_name_range(text, span.start, RefPart::Head, &target.placeholder)
        {
            push(at(s + 1, e - 1));
        }
    }
    // The inner name of `_a_`, so the rename keeps the placeholder spelling.
    if let Some((s, e)) = target.boundary_value
        && text.get(s as usize..e as usize) == Some(target.placeholder.as_str())
    {
        push(at(s + 1, e - 1));
    }

    for loc in callable_uses(db, fs, cats, target).keywords {
        push(loc);
    }
    if include_declaration {
        push(at(target.declaration.0, target.declaration.1));
    }
    out
}

/// The nodes of `reification`'s body, without the bodies of nested
/// reifications unless `nested`. §04 "Placeholders and holes" scopes a
/// placeholder to its nearest enclosing reification.
fn scope_nodes(module: &flatppl_core::Module, reification: NodeId, nested: bool) -> Vec<NodeId> {
    let mut out = Vec::new();
    let mut pending = vec![reification];
    let mut visited = HashSet::new();
    while let Some(id) = pending.pop() {
        if !visited.insert(id)
            || (!nested
                && id != reification
                && matches!(module.node(id), Node::Call(c) if c.inputs.is_some()))
        {
            continue;
        }
        module.for_each_child(id, |child| pending.push(child));
        out.push(id);
    }
    out
}

/// Refuse an argument rename that would change what the callable means or
/// leave a call behind.
fn check_argument_rename(
    db: &dyn salsa::Database,
    fs: FileSet,
    cats: Catalogues,
    t: &ArgumentTarget,
    new_name: &str,
) -> Result<(), Refusal> {
    let module = parse(db, t.file)
        .module(db)
        .ok_or_else(|| Refusal("the defining file does not parse".to_string()))?;
    let Node::Call(Call {
        inputs: Some(Inputs::Spec(entries)),
        ..
    }) = module.node(t.reification)
    else {
        return Err(Refusal("the callable has no ordered inputs".to_string()));
    };
    if entries
        .iter()
        .any(|(sym, _)| module.resolve(*sym) == new_name)
    {
        return Err(Refusal(format!(
            "`{new_name}` is already an input of this callable; a repeated input name is \
             a static error (spec §04 \"Reification to functions and kernels\")"
        )));
    }
    if t.boundary_value.is_some() {
        let placeholder = format!("_{new_name}_");
        if !names::is_placeholder_form(&placeholder) {
            return Err(Refusal(format!(
                "the placeholder would become `{placeholder}`, which is not the spec §04 \
                 placeholder form `_name_`"
            )));
        }
        let is_placeholder =
            |r: &Ref| matches!(r.ns, RefNs::Local) && module.resolve(r.name) == placeholder;
        if entries.iter().any(|(_, r)| is_placeholder(r))
            || scope_nodes(module, t.reification, false)
                .into_iter()
                .any(|id| matches!(module.node(id), Node::Ref(r) if is_placeholder(r)))
        {
            return Err(Refusal(format!(
                "`{placeholder}` is already a placeholder of this callable, so the rename \
                 would merge two inputs (spec §04 \"Placeholders and holes\")"
            )));
        }
    } else if module.boundary_spans_of(t.reification).is_none() {
        // A sugar body spells the argument by its surface name, so the new name
        // would capture a free reference or a called builtin, nested bodies
        // included. A head lowered from an operator also counts, which is safe.
        let captures = scope_nodes(module, t.reification, true)
            .into_iter()
            .any(|id| match module.node(id) {
                Node::Ref(Ref {
                    ns: RefNs::SelfMod,
                    name,
                })
                | Node::Ref(Ref {
                    ns: RefNs::Module(name),
                    ..
                })
                | Node::Const(name)
                | Node::Call(Call {
                    head: CallHead::Builtin(name),
                    ..
                }) => module.resolve(*name) == new_name,
                _ => false,
            });
        if captures {
            return Err(Refusal(format!(
                "the callable's body already refers to `{new_name}`, which the renamed \
                 argument would capture (spec §05 \"Lambda syntax\": arguments shadow \
                 module-level bindings)"
            )));
        }
    }
    if let Some(at) = callable_uses(db, fs, cats, t).escape {
        let line = line_of(db, fs, &at);
        return Err(Refusal(format!(
            "the callable is used as a value at {}:{line}, where the rename cannot follow \
             calls that pass `{}` by name",
            at.path, t.name
        )));
    }
    Ok(())
}

/// The 1-based line of `loc` in its file of the bundle.
fn line_of(db: &dyn salsa::Database, fs: FileSet, loc: &NameLoc) -> usize {
    fs.files(db)
        .iter()
        .find(|f| *f.path(db) == loc.path)
        .map_or(0, |f| {
            f.text(db)[..loc.start as usize].matches('\n').count() + 1
        })
}

/// Builtins that apply their first argument, a callable, and expose none of
/// its input names (spec §04 "Explicit callback parameters"). `pushfwd`,
/// `weighted` and `logweighted` pass the whole variate, and `reduce`, `scan`
/// and `kscan` pass two or more positional arguments.
const POSITIONAL_CONSUMERS: &[&str] = &[
    "pushfwd",
    "weighted",
    "logweighted",
    "reduce",
    "scan",
    "kscan",
];

/// Builtins that apply the callables in arguments `from..to` to one value.
/// §04 "Calling conventions" passes a lone positional argument whole to a
/// one-input callable, but auto-splats a record into a callable with more
/// inputs by name. A first `fchain` or `kchain` argument is excluded: it gives
/// the composite its input names. Other `filter` and `markovchain` arguments
/// are data, or the §04 callback parameter `p` that the callback receives.
const ONE_INPUT_APPLIERS: &[(&str, usize, Option<usize>)] = &[
    ("filter", 0, Some(1)),
    ("markovchain", 0, Some(1)),
    ("fchain", 1, None),
    ("kchain", 1, None),
    ("jointchain", 1, None),
];

/// How the target's callable is used across the bundle.
struct CallableUses {
    /// The `name = value` keywords that pass the target argument by name.
    keywords: Vec<NameLoc>,
    /// The first use as a value that a rename cannot follow.
    escape: Option<NameLoc>,
}

/// Find the keywords that pass the target argument by name, and any use of
/// the callable as a value.
///
/// §04 lets a call, and `broadcast(f, name = array)`, pass an input by its
/// name, so a rename that left `f(a = 2)` behind would leave `f` without a
/// parameter `a`. The callable is each binding whose RHS is the reification,
/// each binding that aliases one (`g = f`), across files, and the reification
/// itself as an inline callee. Any other use, such as a record field or a user
/// function argument, can reach a keyword call this walk cannot see.
fn callable_uses(
    db: &dyn salsa::Database,
    fs: FileSet,
    cats: Catalogues,
    target: &ArgumentTarget,
) -> CallableUses {
    let mut uses = CallableUses {
        keywords: Vec::new(),
        escape: None,
    };
    let Some(module) = parse(db, target.file).module(db) else {
        return uses;
    };
    let mut callables: HashSet<(SourceFile, BindingId)> = module
        .bindings()
        .filter(|(_, b)| b.rhs == target.reification)
        .map(|(bid, _)| (target.file, bid))
        .collect();
    let one_input = matches!(
        module.node(target.reification),
        Node::Call(Call { inputs: Some(Inputs::Spec(entries)), .. }) if entries.len() == 1
    );
    // The part of the reference that names the callable, when it is one.
    let callable_ref =
        |f: SourceFile, module: &flatppl_core::Module, id: NodeId, set: &HashSet<_>| {
            let Node::Ref(r) = module.node(id) else {
                return None;
            };
            let part = match r.ns {
                RefNs::Local => return None,
                RefNs::Module(_) => RefPart::Member,
                RefNs::SelfMod => RefPart::Head,
            };
            resolve_ref_def(db, f, fs, cats, module, r)
                .filter(|def| set.contains(def))
                .map(|_| part)
        };
    loop {
        let mut aliases = Vec::new();
        for &f in fs.files(db) {
            let Some(module) = parse(db, f).module(db) else {
                continue;
            };
            for (bid, b) in module.bindings() {
                if !callables.contains(&(f, bid))
                    && callable_ref(f, module, b.rhs, &callables).is_some()
                {
                    aliases.push((f, bid));
                }
            }
        }
        if aliases.is_empty() {
            break;
        }
        callables.extend(aliases);
    }

    for &f in fs.files(db) {
        let Some(module) = parse(db, f).module(db) else {
            continue;
        };
        let text = f.text(db);
        let path = f.path(db);
        // The end of the callee text when `id` is the callable.
        let callee_end = |id: NodeId| {
            let span = module.span_of(id)?;
            if f == target.file && id == target.reification {
                return Some(span.end);
            }
            let part = callable_ref(f, module, id, &callables)?;
            names::callee_ref_end(text, span.start, part)
        };
        let mut followed: HashSet<NodeId> = module.bindings().map(|(_, b)| b.rhs).collect();
        for i in 0..module.node_count() {
            let id = NodeId::from_usize(i);
            let Node::Call(call) = module.node(id) else {
                continue;
            };
            let (callee, forwards) = match call.head {
                CallHead::User(callee) => (callee, true),
                CallHead::Builtin(head) => {
                    let head = module.resolve(head);
                    let Some(&first) = call.args.first() else {
                        continue;
                    };
                    if head == "broadcast" {
                        (first, true)
                    } else if POSITIONAL_CONSUMERS.contains(&head) {
                        (first, false)
                    } else {
                        if one_input
                            && let Some(&(_, from, to)) =
                                ONE_INPUT_APPLIERS.iter().find(|(name, ..)| *name == head)
                        {
                            let to = to.unwrap_or(call.args.len()).min(call.args.len());
                            followed.extend(call.args.get(from..to).unwrap_or_default());
                        }
                        continue;
                    }
                }
            };
            followed.insert(callee);
            if !forwards
                || !call
                    .named
                    .iter()
                    .any(|arg| module.resolve(arg.name) == target.name)
            {
                continue;
            }
            let Some(end) = callee_end(callee) else {
                continue;
            };
            // `broadcast(f, a = …)` opens its list after the head, `f.(a = …)`
            // and `f(a = …)` after the callee.
            let explicit_head = match call.head {
                CallHead::Builtin(_) => module.span_of(id).and_then(|span| {
                    let head_end = names::head_ident_end(text, span.start)?;
                    (&text[span.start as usize..head_end as usize] == "broadcast")
                        .then_some(head_end)
                }),
                CallHead::User(_) => None,
            };
            let open_after = explicit_head.unwrap_or(end);
            for (s, e) in names::call_keyword_ranges(text, open_after) {
                if text[s as usize..e as usize] == target.name {
                    uses.keywords.push(NameLoc {
                        path: path.clone(),
                        start: s,
                        end: e,
                    });
                }
            }
        }
        if uses.escape.is_none() {
            uses.escape = (0..module.node_count())
                .map(NodeId::from_usize)
                .filter(|id| !followed.contains(id))
                .find(|&id| {
                    (f == target.file && id == target.reification)
                        || callable_ref(f, module, id, &callables).is_some()
                })
                .and_then(|id| module.span_of(id))
                .map(|span| NameLoc {
                    path: path.clone(),
                    start: span.start,
                    end: span.end,
                });
        }
    }
    uses
}

/// Every occurrence of a module binding's name across the bundle.
fn binding_occurrences(
    db: &dyn salsa::Database,
    fs: FileSet,
    cats: Catalogues,
    target: &BindingTarget,
    include_declaration: bool,
) -> Vec<NameLoc> {
    use flatppl_core::Idx;
    let mut out: Vec<NameLoc> = Vec::new();
    let mut seen: HashSet<NameLoc> = HashSet::new();
    let mut push = |loc: NameLoc| {
        if seen.insert(loc.clone()) {
            out.push(loc);
        }
    };

    for &f in fs.files(db) {
        let Some(module) = parse(db, f).module(db) else {
            continue;
        };
        let text = f.text(db);
        let path = f.path(db).clone();

        for i in 0..module.node_count() {
            let id = flatppl_core::NodeId::from_usize(i);
            let Node::Ref(r) = module.node(id) else {
                continue;
            };
            if matches!(r.ns, RefNs::Local) {
                continue;
            }
            let Some(span) = module.span_of(id) else {
                continue;
            };
            if resolve_ref_def(db, f, fs, cats, module, r) != Some((target.file, target.bid)) {
                continue;
            }
            let part = if matches!(r.ns, RefNs::Module(_)) {
                RefPart::Member
            } else {
                RefPart::Head
            };
            if let Some((start, end)) = names::ref_name_range(text, span.start, part, &target.name)
            {
                push(NameLoc {
                    path: path.clone(),
                    start,
                    end,
                });
            }
        }

        if include_declaration && f == target.file {
            let binding = module.binding(target.bid);
            if let Some(span) = module.span_of(binding.rhs)
                && let Some((start, end)) = names::def_name_range(text, span.start, &target.name)
            {
                push(NameLoc {
                    path: path.clone(),
                    start,
                    end,
                });
            }
        }
    }
    out
}

/// All references to the binding under the cursor, for `textDocument/references`.
///
/// Returns an empty vec when the cursor is not on a renameable binding — the
/// protocol has no error channel for "nothing here", and an empty list is the
/// correct answer.
pub fn references(
    db: &dyn salsa::Database,
    file: SourceFile,
    fs: FileSet,
    cats: Catalogues,
    byte_offset: u32,
    index: &SpanIndex,
    include_declaration: bool,
) -> Vec<NameLoc> {
    match resolve_target(db, file, fs, cats, byte_offset, index) {
        Ok(target) => occurrences(db, fs, cats, &target, include_declaration),
        Err(_) => Vec::new(),
    }
}

/// The byte range of the name the cursor sits on, for
/// `textDocument/prepareRename`, or the refusal to report.
///
/// A `None`/error result tells the client the position cannot be renamed, which
/// is how a built-in, a standard-module member and a record field are declined
/// before the user types a new name. A function/kernel argument IS renameable and
/// returns its range here.
pub fn prepare_rename(
    db: &dyn salsa::Database,
    file: SourceFile,
    fs: FileSet,
    cats: Catalogues,
    byte_offset: u32,
    index: &SpanIndex,
) -> Result<(u32, u32), Refusal> {
    let target = resolve_target(db, file, fs, cats, byte_offset, index)?;
    let text = file.text(db).as_bytes();
    // The cursor may sit on an underscore of a `_a_` whose inner name is the
    // occurrence.
    let on_underscores = |l: &NameLoc| {
        l.start > 0
            && text.get(l.start as usize - 1) == Some(&b'_')
            && text.get(l.end as usize) == Some(&b'_')
            && l.start - 1 <= byte_offset
            && byte_offset <= l.end
    };
    let placeholder_target = matches!(&target, Target::Argument(t) if t.boundary_value.is_some());
    // The renameable range is the occurrence under the cursor itself, so the
    // editor highlights exactly the identifier it will replace.
    occurrences(db, fs, cats, &target, true)
        .into_iter()
        .find(|l| {
            l.path == *file.path(db)
                && ((l.start <= byte_offset && byte_offset < l.end)
                    || (placeholder_target && on_underscores(l)))
        })
        .map(|l| (l.start, l.end))
        .ok_or_else(|| Refusal("the name under the cursor has no editable range".to_string()))
}

/// The edits that rename the binding under the cursor to `new_name`, or the
/// refusal to report.
///
/// Refusals, each on its normative rule:
///
/// - **Not a binding.** §04 "Name resolution" resolves a name to a module
///   binding first and "otherwise […] to the FlatPPL built-in of that name", so a
///   built-in has no definition site in the workspace to rewrite. §09 members and
///   record fields likewise: §04 "Objects, expressions, names and modules" states
///   that "Record field names and table column names are local to their object
///   and not part of the global module namespace". Reported by [`resolve_target`].
/// - **Illegal new name.** For a module binding, §04 "Binding names" and §05
///   "Note on reserved words" via [`names::check_new_name`]. For an argument, the
///   looser §05 `Name` production via [`names::check_new_argument_name`] — §04's
///   binding regexes do NOT govern argument names, so an argument may take a shape
///   a binding may not.
/// - **Collision with an existing binding.** §04 "Objects, expressions, names and
///   modules": "A FlatPPL **module** is an unordered set of bindings of names to
///   expressions." A second binding of the same name in one module is not such a
///   set, so a rename onto an existing name in the defining module refuses.
/// - **An argument rename that changes the callable or leaves a call behind**, in
///   [`check_argument_rename`]: a new name that is already an input (§04
///   "Reification to functions and kernels" makes a repeated input a static
///   error) or already a placeholder of the reification (§04 "Placeholders and
///   holes"); a placeholder `_new_` outside the §04 form; a sugar body that
///   already refers to the new name, which the argument would capture (§05
///   "Lambda syntax"); and a callable used as a value that a keyword call could
///   reach unseen.
///
/// Deliberately NOT refused, because no rule forbids them, or because `infer`
/// reports the consequence precisely:
///
/// - **Shadowing a built-in with a binding.** §04 "Name resolution": "This makes
///   built-in names shadowable: a module may bind any name except for `self` and
///   `base`."
/// - **An argument shadowing a module binding or built-in.** §05 "Lambda syntax":
///   "the argument names refer to the lambda's inputs and shadow any module-level
///   binding of the same name."
/// - **Making a cross-module-referenced name private.** The spec is silent on the
///   mechanism for `module._private` (§04 "Binding names" says only that private
///   bindings "are not part of the module's public interface"), and `infer`
///   already reports the consequence precisely — "`_s` is private to
///   `d.flatppl`" — at the offending reference. So the rename is performed and the
///   existing diagnostic flags the result, rather than the server blocking a
///   legitimate "make this private, then fix the callers" edit.
pub fn rename_edits(
    db: &dyn salsa::Database,
    file: SourceFile,
    fs: FileSet,
    cats: Catalogues,
    byte_offset: u32,
    index: &SpanIndex,
    new_name: &str,
) -> Result<Vec<NameLoc>, Refusal> {
    let target = resolve_target(db, file, fs, cats, byte_offset, index)?;
    match &target {
        Target::Binding(t) => {
            names::check_new_name(new_name)?;
            if new_name == t.name {
                return Ok(Vec::new());
            }
            let def_mod = parse(db, t.file).module(db).ok_or_else(|| {
                Refusal("the defining file does not parse, so it cannot be edited".to_string())
            })?;
            if def_mod
                .bindings()
                .any(|(bid, b)| bid != t.bid && def_mod.resolve(b.name) == new_name)
            {
                return Err(Refusal(format!(
                    "`{new_name}` is already bound in {}; a module is a set of bindings of \
                     names to expressions (spec §04), so the name cannot be bound twice",
                    t.file.path(db)
                )));
            }
        }
        Target::Argument(t) => {
            names::check_new_argument_name(new_name)?;
            if new_name == t.name {
                return Ok(Vec::new());
            }
            check_argument_rename(db, fs, cats, t, new_name)?;
        }
    }

    let locs = occurrences(db, fs, cats, &target, true);

    // A definition site we could not locate textually means the rewrite would be
    // partial. This is a fail-closed implementation guard, not a doctrine refusal:
    // no rule forbids the rename, we simply cannot compute a correct edit for it.
    let (def_path, name, declaration) = match &target {
        Target::Binding(t) => {
            let declaration = parse(db, t.file).module(db).and_then(|module| {
                let span = module.span_of(module.binding(t.bid).rhs)?;
                names::def_name_range(t.file.text(db), span.start, &t.name)
            });
            (t.file.path(db).clone(), &t.name, declaration)
        }
        Target::Argument(t) => (t.file.path(db).clone(), &t.name, Some(t.declaration)),
    };
    if !locs
        .iter()
        .any(|l| l.path == def_path && Some((l.start, l.end)) == declaration)
    {
        return Err(Refusal(format!(
            "could not locate the declaration of `{name}` in {def_path}"
        )));
    }

    Ok(locs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;
    use crate::queries::node_span_index;

    /// Byte offset of `needle`'s Nth (0-based) occurrence in `src`.
    fn nth_offset(src: &str, needle: &str, n: usize) -> u32 {
        src.match_indices(needle)
            .nth(n)
            .expect("occurrence present")
            .0 as u32
    }

    fn single_file(src: &str) -> (Database, SourceFile, FileSet, Catalogues) {
        let db = Database::default();
        let f = SourceFile::new(&db, "m.flatppl".to_string(), src.to_string());
        let fs = FileSet::new(&db, vec![f]);
        let cats = Catalogues::new(&db, vec![]);
        (db, f, fs, cats)
    }

    /// The renamed source, applying `locs` to `src` (descending, so earlier
    /// offsets stay valid).
    fn apply(src: &str, locs: &[NameLoc], new_name: &str) -> String {
        let mut sorted: Vec<&NameLoc> = locs.iter().collect();
        sorted.sort_by_key(|l| std::cmp::Reverse(l.start));
        let mut out = src.to_string();
        for l in sorted {
            out.replace_range(l.start as usize..l.end as usize, new_name);
        }
        out
    }

    // ── references ───────────────────────────────────────────────────────────

    #[test]
    fn references_finds_every_use_and_the_declaration() {
        let src = "x = 1\ny = add(x, 2)\nz = mul(x, x)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        // Cursor on the declaration of `x`.
        let locs = references(&db, f, fs, cats, 0, index, true);
        let texts: Vec<&str> = locs
            .iter()
            .map(|l| &src[l.start as usize..l.end as usize])
            .collect();
        assert_eq!(
            texts.len(),
            4,
            "3 uses + 1 declaration; got {locs:?} ({texts:?})"
        );
        assert!(texts.iter().all(|t| *t == "x"), "every range is `x`");
        assert!(
            locs.iter().any(|l| l.start == 0),
            "the declaration at offset 0 must be included; got {locs:?}"
        );
    }

    #[test]
    fn references_excludes_the_declaration_when_not_requested() {
        let src = "x = 1\ny = add(x, 2)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = references(&db, f, fs, cats, 0, index, false);
        assert_eq!(locs.len(), 1, "only the use in `add(x, 2)`; got {locs:?}");
        assert!(locs.iter().all(|l| l.start != 0));
    }

    #[test]
    fn references_does_not_match_a_longer_name() {
        let src = "x = 1\nxy = 2\nz = add(x, xy)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = references(&db, f, fs, cats, 0, index, true);
        assert_eq!(locs.len(), 2, "`x` decl + one use, not `xy`; got {locs:?}");
        for l in &locs {
            assert_eq!(&src[l.start as usize..l.end as usize], "x");
        }
    }

    #[test]
    fn references_on_a_tilde_binding_declaration() {
        // A `~` binding's RHS span starts at the name, so the declaration is
        // found by the def-name pass rather than by a node lookup.
        let src = "mu = 0.0\nx ~ Normal(mu, 1.0)\ny = add(x, 1.0)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = references(&db, f, fs, cats, 9, index, true);
        assert_eq!(locs.len(), 2, "`x` decl + its use; got {locs:?}");
    }

    #[test]
    fn references_on_a_function_definition_name() {
        let src = "f(a) = add(a, 1)\nz = f(3)\nw = f(4)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = references(&db, f, fs, cats, 0, index, true);
        assert_eq!(locs.len(), 3, "`f` decl + two calls; got {locs:?}");
    }

    #[test]
    fn references_span_the_multi_file_bundle() {
        let db = Database::default();
        let helpers = SourceFile::new(
            &db,
            "helpers.flatppl".to_string(),
            "shifted = 1.0\nlocal_use = add(shifted, 1.0)".to_string(),
        );
        let model = SourceFile::new(
            &db,
            "model.flatppl".to_string(),
            "h = load_module(\"helpers.flatppl\")\nv = h.shifted\nw = add(h.shifted, 1.0)"
                .to_string(),
        );
        let fs = FileSet::new(&db, vec![helpers, model]);
        let cats = Catalogues::new(&db, vec![]);

        // Ask from the DEFINING file, on the declaration of `shifted`.
        let index = node_span_index(&db, helpers, fs, cats);
        let locs = references(&db, helpers, fs, cats, 0, index, true);
        let by_file = |p: &str| locs.iter().filter(|l| l.path == p).count();
        assert_eq!(
            by_file("helpers.flatppl"),
            2,
            "declaration + local use in helpers; got {locs:?}"
        );
        assert_eq!(
            by_file("model.flatppl"),
            2,
            "both `h.shifted` member refs in the importer; got {locs:?}"
        );
        // The member ranges must cover `shifted`, not `h.shifted`.
        let model_text = model.text(&db);
        for l in locs.iter().filter(|l| l.path == "model.flatppl") {
            assert_eq!(&model_text[l.start as usize..l.end as usize], "shifted");
        }
    }

    #[test]
    fn references_from_the_importer_reach_the_definition() {
        let db = Database::default();
        let helpers = SourceFile::new(&db, "helpers.flatppl".to_string(), "s = 1.0".to_string());
        let model = SourceFile::new(
            &db,
            "model.flatppl".to_string(),
            "h = load_module(\"helpers.flatppl\")\nv = h.s".to_string(),
        );
        let fs = FileSet::new(&db, vec![helpers, model]);
        let cats = Catalogues::new(&db, vec![]);
        let index = node_span_index(&db, model, fs, cats);
        // Cursor on the `s` of `h.s`.
        let off = nth_offset(model.text(&db), "h.s", 0) + 2;
        let locs = references(&db, model, fs, cats, off, index, true);
        assert!(
            locs.iter().any(|l| l.path == "helpers.flatppl"),
            "the definition in the loaded module must be found; got {locs:?}"
        );
        assert!(
            locs.iter().any(|l| l.path == "model.flatppl"),
            "the member ref in the importer must be found; got {locs:?}"
        );
    }

    #[test]
    fn references_on_a_module_alias_find_the_alias_uses() {
        let db = Database::default();
        let helpers = SourceFile::new(
            &db,
            "helpers.flatppl".to_string(),
            "a = 1.0\nb = 2.0".to_string(),
        );
        let src = "h = load_module(\"helpers.flatppl\")\nv = h.a\nw = h.b";
        let model = SourceFile::new(&db, "model.flatppl".to_string(), src.to_string());
        let fs = FileSet::new(&db, vec![helpers, model]);
        let cats = Catalogues::new(&db, vec![]);
        let index = node_span_index(&db, model, fs, cats);
        // Cursor on the alias declaration `h`.
        let locs = references(&db, model, fs, cats, 0, index, true);
        assert_eq!(
            locs.len(),
            3,
            "alias declaration + the `h` of each member access; got {locs:?}"
        );
        for l in &locs {
            assert_eq!(l.path, "model.flatppl");
            assert_eq!(&src[l.start as usize..l.end as usize], "h");
        }
    }

    #[test]
    fn references_empty_over_a_builtin() {
        let src = "x = add(1, 2)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let off = nth_offset(src, "add", 0);
        assert!(
            references(&db, f, fs, cats, off, index, true).is_empty(),
            "a built-in is not a binding, so it has no references to report"
        );
    }

    // ── prepareRename ────────────────────────────────────────────────────────

    #[test]
    fn prepare_rename_returns_the_identifier_range() {
        let src = "x = 1\ny = add(x, 2)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let off = nth_offset(src, "add(x", 0) + 4;
        assert_eq!(prepare_rename(&db, f, fs, cats, off, index), Ok((14, 15)));
    }

    #[test]
    fn prepare_rename_range_is_the_member_not_the_whole_access() {
        let db = Database::default();
        let helpers = SourceFile::new(&db, "helpers.flatppl".to_string(), "s = 1.0".to_string());
        let src = "h = load_module(\"helpers.flatppl\")\nv = h.s";
        let model = SourceFile::new(&db, "model.flatppl".to_string(), src.to_string());
        let fs = FileSet::new(&db, vec![helpers, model]);
        let cats = Catalogues::new(&db, vec![]);
        let index = node_span_index(&db, model, fs, cats);
        let dot = nth_offset(src, "h.s", 0);
        let range = prepare_rename(&db, model, fs, cats, dot + 2, index).expect("renameable");
        assert_eq!(
            &src[range.0 as usize..range.1 as usize],
            "s",
            "the editable range covers `s`, not `h.s`"
        );
    }

    #[test]
    fn prepare_rename_refuses_a_builtin() {
        let src = "x = add(1, 2)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let off = nth_offset(src, "add", 0);
        let err = prepare_rename(&db, f, fs, cats, off, index).expect_err("builtin must refuse");
        assert!(
            err.0.contains("built-in"),
            "refusal must name the reason; got {}",
            err.0
        );
    }

    /// A function argument IS renameable. Nothing in the docs forbids it — §04
    /// only says argument names live outside the module namespace — so
    /// prepareRename must return a range rather than declining the category.
    #[test]
    fn prepare_rename_allows_a_function_argument() {
        let src = "f(a) = add(a, 1)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let off = nth_offset(src, "add(a", 0) + 4;
        let range = prepare_rename(&db, f, fs, cats, off, index)
            .expect("a function argument is renameable");
        assert_eq!(&src[range.0 as usize..range.1 as usize], "a");
    }

    #[test]
    fn prepare_rename_allows_an_argument_declaration_site() {
        // Offset 2 is the `a` inside `f(a)`, which carries no node of its own.
        let src = "f(a) = add(a, 1)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let range = prepare_rename(&db, f, fs, cats, 2, index).expect("declaration renameable");
        assert_eq!(range, (2, 3));
    }

    #[test]
    fn prepare_rename_refuses_a_standard_module_member() {
        let db = Database::default();
        let ron = r#"Catalogue(base: [], modules: [Module(name:"myext",version:"0.1",bindings:[Binding(name:"MyDist", sig: Distribution(domain: Scalar(Real), support: Reals, mass: Normalized))])])"#;
        let src = "e = standard_module(\"myext\",\"0.1\")\nx = e.MyDist(0.0)";
        let f = SourceFile::new(&db, "m.flatppl".to_string(), src.to_string());
        let fs = FileSet::new(&db, vec![f]);
        let cats = Catalogues::new(&db, vec![ron.to_string()]);
        let index = node_span_index(&db, f, fs, cats);
        let off = nth_offset(src, "e.MyDist", 0) + 3;
        let err = prepare_rename(&db, f, fs, cats, off, index)
            .expect_err("a §09 member must refuse rename");
        assert!(
            err.0.contains("standard-module"),
            "refusal must cite the standard-module rule; got {}",
            err.0
        );
    }

    #[test]
    fn prepare_rename_refuses_a_record_field() {
        // §04: record field names are local to their object, not module bindings.
        let src = "s = record(a = 1.0)\nt = s.a";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let off = nth_offset(src, "s.a", 0) + 2;
        assert!(
            prepare_rename(&db, f, fs, cats, off, index).is_err(),
            "a record field must not be renameable"
        );
    }

    // ── rename ───────────────────────────────────────────────────────────────

    #[test]
    fn rename_rewrites_every_occurrence_in_one_file() {
        let src = "x = 1\ny = add(x, 2)\nz = mul(x, x)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = rename_edits(&db, f, fs, cats, 0, index, "theta").expect("rename allowed");
        assert_eq!(
            apply(src, &locs, "theta"),
            "theta = 1\ny = add(theta, 2)\nz = mul(theta, theta)"
        );
    }

    #[test]
    fn rename_rewrites_across_the_bundle() {
        let db = Database::default();
        let helpers_src = "shifted = 1.0\nu = add(shifted, 1.0)";
        let model_src = "h = load_module(\"helpers.flatppl\")\nv = h.shifted";
        let helpers = SourceFile::new(&db, "helpers.flatppl".to_string(), helpers_src.to_string());
        let model = SourceFile::new(&db, "model.flatppl".to_string(), model_src.to_string());
        let fs = FileSet::new(&db, vec![helpers, model]);
        let cats = Catalogues::new(&db, vec![]);
        let index = node_span_index(&db, model, fs, cats);
        // Rename from the IMPORTER, on the member half of `h.shifted`.
        let off = nth_offset(model_src, "h.shifted", 0) + 2;
        let locs = rename_edits(&db, model, fs, cats, off, index, "offset").expect("allowed");

        let h_edits: Vec<NameLoc> = locs
            .iter()
            .filter(|l| l.path == "helpers.flatppl")
            .cloned()
            .collect();
        let m_edits: Vec<NameLoc> = locs
            .iter()
            .filter(|l| l.path == "model.flatppl")
            .cloned()
            .collect();
        assert_eq!(
            apply(helpers_src, &h_edits, "offset"),
            "offset = 1.0\nu = add(offset, 1.0)",
            "the definition and its local uses are rewritten"
        );
        assert_eq!(
            apply(model_src, &m_edits, "offset"),
            "h = load_module(\"helpers.flatppl\")\nv = h.offset",
            "the member half is rewritten and the alias is untouched"
        );
    }

    #[test]
    fn rename_refuses_a_colliding_name() {
        let src = "x = 1\ny = 2\nz = add(x, y)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let err = rename_edits(&db, f, fs, cats, 0, index, "y").expect_err("collision must refuse");
        assert!(
            err.0.contains("already bound"),
            "refusal must say the name is taken; got {}",
            err.0
        );
    }

    #[test]
    fn rename_refuses_an_illegal_name() {
        let src = "x = 1";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        for bad in ["in", "1x", "_x_", "__gen", "self", "x-y"] {
            assert!(
                rename_edits(&db, f, fs, cats, 0, index, bad).is_err(),
                "`{bad}` must be refused as a binding name"
            );
        }
    }

    #[test]
    fn rename_refuses_a_builtin() {
        let src = "x = add(1, 2)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let off = nth_offset(src, "add", 0);
        assert!(rename_edits(&db, f, fs, cats, off, index, "plus").is_err());
    }

    #[test]
    fn rename_allows_shadowing_a_builtin_name() {
        // §04 "Name resolution": built-in names are shadowable.
        let src = "x = 1\ny = add(x, 2)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = rename_edits(&db, f, fs, cats, 0, index, "sqrt")
            .expect("shadowing a built-in is legal");
        assert_eq!(apply(src, &locs, "sqrt"), "sqrt = 1\ny = add(sqrt, 2)");
    }

    /// Making a cross-module-referenced name private is PERFORMED, not refused.
    ///
    /// The spec is silent on the mechanism for `module._private`, and `infer`
    /// already reports the consequence precisely at the offending reference (see
    /// `cross_module_private_access_is_already_diagnosed`), so blocking the edit
    /// would only deny a legitimate "make this private, then fix the callers"
    /// workflow. The rename also rewrites the cross-module reference, so the bundle
    /// stays internally consistent and the diagnostic lands on the visibility, not
    /// on a dangling name.
    #[test]
    fn rename_public_to_private_is_performed_not_refused() {
        let db = Database::default();
        let helpers_src = "s = 1.0";
        let helpers = SourceFile::new(&db, "helpers.flatppl".to_string(), helpers_src.to_string());
        let model_src = "h = load_module(\"helpers.flatppl\")\nv = h.s";
        let model = SourceFile::new(&db, "model.flatppl".to_string(), model_src.to_string());
        let fs = FileSet::new(&db, vec![helpers, model]);
        let cats = Catalogues::new(&db, vec![]);
        let index = node_span_index(&db, helpers, fs, cats);
        let locs = rename_edits(&db, helpers, fs, cats, 0, index, "_s")
            .expect("no rule forbids making a binding private");
        let pick =
            |p: &str| -> Vec<NameLoc> { locs.iter().filter(|l| l.path == p).cloned().collect() };
        assert_eq!(
            apply(helpers_src, &pick("helpers.flatppl"), "_s"),
            "_s = 1.0"
        );
        assert_eq!(
            apply(model_src, &pick("model.flatppl"), "_s"),
            "h = load_module(\"helpers.flatppl\")\nv = h._s",
            "the cross-module reference is rewritten too"
        );
    }

    /// The measured basis for the decision above: `infer` already flags
    /// cross-module access to a private binding, so allowing that rename does not
    /// hide an error. If this ever fails, the rename should refuse instead of
    /// relying on the diagnostic.
    #[test]
    fn cross_module_private_access_is_already_diagnosed() {
        let db = Database::default();
        let helpers = SourceFile::new(&db, "helpers.flatppl".to_string(), "_s = 1.0".to_string());
        let model = SourceFile::new(
            &db,
            "model.flatppl".to_string(),
            "h = load_module(\"helpers.flatppl\")\nv = h._s".to_string(),
        );
        let fs = FileSet::new(&db, vec![helpers, model]);
        let cats = Catalogues::new(&db, vec![]);
        let diags = crate::capabilities::diagnostics(&db, model, fs, cats);
        assert!(
            diags.iter().any(|d| d.message.contains("is private to")
                && d.severity == Some(lsp_types::DiagnosticSeverity::ERROR)),
            "infer must flag `h._s` as private; got {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn rename_public_to_private_allowed_without_a_cross_module_reference() {
        let src = "s = 1.0\nu = add(s, 1.0)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = rename_edits(&db, f, fs, cats, 0, index, "_s")
            .expect("no cross-module reference, so the visibility change is safe");
        assert_eq!(apply(src, &locs, "_s"), "_s = 1.0\nu = add(_s, 1.0)");
    }

    #[test]
    fn rename_private_to_public_is_allowed() {
        let src = "_s = 1.0\nu = add(_s, 1.0)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs =
            rename_edits(&db, f, fs, cats, 0, index, "s").expect("widening visibility is legal");
        assert_eq!(apply(src, &locs, "s"), "s = 1.0\nu = add(s, 1.0)");
    }

    // ── argument (%local) rename ──────────────────────────────────────────────

    #[test]
    fn references_on_an_argument_cover_the_declaration_and_body_uses() {
        let src = "f(a, b) = add(mul(a, a), b)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = references(&db, f, fs, cats, 2, index, true);
        assert_eq!(locs.len(), 3, "declaration + two body uses; got {locs:?}");
        for l in &locs {
            assert_eq!(&src[l.start as usize..l.end as usize], "a");
        }
    }

    #[test]
    fn rename_a_named_function_argument() {
        let src = "f(a, b) = add(mul(a, a), b)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = rename_edits(&db, f, fs, cats, 2, index, "scale").expect("allowed");
        assert_eq!(
            apply(src, &locs, "scale"),
            "f(scale, b) = add(mul(scale, scale), b)"
        );
    }

    /// §04 lets a call pass an input by name, so the keyword at each call site
    /// is an occurrence of the argument. Leaving it would make the clean input
    /// fail with "`f` has no parameter `a`".
    #[test]
    fn rename_an_argument_rewrites_named_call_site_arguments() {
        let src =
            "f(a, b) = add(a, b)\ny = f(b = 1, a = 2)\nz = f(b = 1, a = sqrt(f(a = 3, b = 4)))";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = rename_edits(&db, f, fs, cats, 2, index, "scale").expect("allowed");
        let renamed = apply(src, &locs, "scale");
        assert_eq!(
            renamed,
            "f(scale, b) = add(scale, b)\ny = f(b = 1, scale = 2)\n\
             z = f(b = 1, scale = sqrt(f(scale = 3, b = 4)))"
        );
        let mut m = flatppl_syntax::parse(&renamed).expect("still parses");
        let diags = flatppl_infer::infer_with(&mut m, flatppl_infer::Level::Shape);
        assert!(
            diags.is_empty(),
            "the renamed input stays clean; got {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    /// Calls through an alias and from an importing module are call sites too.
    /// A same-named keyword of another callable or a record is not.
    #[test]
    fn rename_an_argument_rewrites_named_arguments_across_the_bundle() {
        let db = Database::default();
        let helpers_src = "g = (a, b) -> add(a, b)\nh = g\nu = h(a = 1, b = 2)";
        let model_src = "m = load_module(\"helpers.flatppl\")\nk(a) = a\n\
                         v = m.g(b = \"a = 1\", # a = 2\n a = [k(a = 1)])\nw = record(a = 1)";
        let helpers = SourceFile::new(&db, "helpers.flatppl".to_string(), helpers_src.to_string());
        let model = SourceFile::new(&db, "model.flatppl".to_string(), model_src.to_string());
        let fs = FileSet::new(&db, vec![helpers, model]);
        let cats = Catalogues::new(&db, vec![]);
        let index = node_span_index(&db, helpers, fs, cats);
        let off = nth_offset(helpers_src, "(a, b)", 0) + 1;
        let locs = rename_edits(&db, helpers, fs, cats, off, index, "x").expect("allowed");
        let edits_in = |path: &str| -> Vec<NameLoc> {
            locs.iter().filter(|l| l.path == path).cloned().collect()
        };
        assert_eq!(
            apply(helpers_src, &edits_in("helpers.flatppl"), "x"),
            "g = (x, b) -> add(x, b)\nh = g\nu = h(x = 1, b = 2)"
        );
        assert_eq!(
            apply(model_src, &edits_in("model.flatppl"), "x"),
            "m = load_module(\"helpers.flatppl\")\nk(a) = a\n\
             v = m.g(b = \"a = 1\", # a = 2\n x = [k(a = 1)])\nw = record(a = 1)"
        );
    }

    #[test]
    fn rename_a_lambda_argument() {
        // §05 "Lambda syntax": `(a, b) -> expr`. The reification span starts at the
        // argument list rather than after a binding name.
        let src = "g = (a, b) -> add(a, b)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let off = nth_offset(src, "(a, b)", 0) + 1;
        let locs = rename_edits(&db, f, fs, cats, off, index, "x").expect("allowed");
        assert_eq!(apply(src, &locs, "x"), "g = (x, b) -> add(x, b)");
    }

    #[test]
    fn rename_a_single_bare_lambda_argument() {
        // §05: `arg -> expr` with no parentheses in the single-argument form.
        let src = "s = a -> add(a, 1)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = rename_edits(&db, f, fs, cats, 4, index, "v").expect("allowed");
        assert_eq!(apply(src, &locs, "v"), "s = v -> add(v, 1)");
    }

    /// §04 "Placeholders and holes": "The same placeholder name may appear in
    /// different scopes without conflict." So renaming `a` in one callable must not
    /// touch the `a` of another.
    #[test]
    fn rename_an_argument_leaves_a_same_named_argument_of_another_callable_alone() {
        let src = "f(a) = add(a, 1)\ng(a) = mul(a, 2)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = rename_edits(&db, f, fs, cats, 2, index, "z").expect("allowed");
        assert_eq!(apply(src, &locs, "z"), "f(z) = add(z, 1)\ng(a) = mul(a, 2)");
    }

    #[test]
    fn rename_outer_argument_preserves_nested_scope() {
        let src = "f(a) = add(a, sum(broadcast(a -> add(a, 1), [1, 2])))\ny = f(3)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = rename_edits(&db, f, fs, cats, 2, index, "scale").unwrap();
        assert_eq!(
            apply(src, &locs, "scale"),
            "f(scale) = add(scale, sum(broadcast(a -> add(a, 1), [1, 2])))\ny = f(3)"
        );
    }

    #[test]
    fn rename_binding_ignores_comment_declaration_text() {
        let src = "x = (\n # x = ignored\n 1)\ny = x";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = rename_edits(&db, f, fs, cats, src.len() as u32 - 1, index, "renamed").unwrap();
        assert_eq!(
            apply(src, &locs, "renamed"),
            "renamed = (\n # x = ignored\n 1)\ny = renamed"
        );
    }

    #[test]
    fn rename_binding_ignores_quoted_declaration_text() {
        let src = "text = \"x = ignored\"\nx = 1\ny = x";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = rename_edits(&db, f, fs, cats, src.len() as u32 - 1, index, "renamed").unwrap();
        assert_eq!(
            apply(src, &locs, "renamed"),
            "text = \"x = ignored\"\nrenamed = 1\ny = renamed"
        );
    }

    #[test]
    fn rename_refuses_when_decomposition_declaration_is_unavailable() {
        for name in ["a", "b"] {
            let src = format!("a, b = [1, 2]\ny = {name}");
            let (db, f, fs, cats) = single_file(&src);
            let index = node_span_index(&db, f, fs, cats);
            assert!(
                rename_edits(&db, f, fs, cats, src.len() as u32 - 1, index, "renamed").is_err(),
                "a rename must include the declaration"
            );
        }
    }

    #[test]
    fn rename_argument_preserves_comment_trivia() {
        let src = "f(\n # argument\n a) = add(a, 1)\ny = f(3)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let cursor = nth_offset(src, "add(a", 0) + 4;
        let edits = rename_edits(&db, f, fs, cats, cursor, index, "value").unwrap();
        assert_eq!(
            apply(src, &edits, "value"),
            "f(\n # argument\n value) = add(value, 1)\ny = f(3)"
        );
    }

    #[test]
    fn rename_function_with_same_named_identity_argument() {
        let src = "f(f) = f\ny = f(3)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let cursor = nth_offset(src, "f(3)", 0);
        let locs = rename_edits(&db, f, fs, cats, cursor, index, "identity").unwrap();
        assert_eq!(
            apply(src, &locs, "identity"),
            "identity(f) = f\ny = identity(3)"
        );
    }

    #[test]
    fn rename_distinguishes_function_header_from_returned_lambda() {
        let src = "f(a) = a -> add(a, 1)\ny = f(3)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let outer = rename_edits(&db, f, fs, cats, 2, index, "outer").unwrap();
        assert_eq!(
            apply(src, &outer, "outer"),
            "f(outer) = a -> add(a, 1)\ny = f(3)"
        );
        // The returned lambda is a value that `f(3)(a = 2)` could call by
        // keyword, so its rename refuses. Its references stay in its own scope.
        for cursor in [nth_offset(src, "a ->", 0), nth_offset(src, "add(a", 0) + 4] {
            let inner = references(&db, f, fs, cats, cursor, index, true);
            assert_eq!(
                apply(src, &inner, "inner"),
                "f(a) = inner -> add(inner, 1)\ny = f(3)"
            );
            let err = rename_edits(&db, f, fs, cats, cursor, index, "inner").unwrap_err();
            assert!(err.0.contains("used as a value"), "{}", err.0);
        }
    }

    #[test]
    fn rename_nested_argument_selects_its_own_scope() {
        let src = "f(a) = add(a, sum(broadcast(a -> add(a, 1), [1, 2])))\ny = f(3)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let declaration = nth_offset(src, "a ->", 0);
        let body_use = nth_offset(src, "add(a, 1)", 0) + 4;
        for cursor in [declaration, body_use] {
            let locs = rename_edits(&db, f, fs, cats, cursor, index, "inner").unwrap();
            assert_eq!(
                apply(src, &locs, "inner"),
                "f(a) = add(a, sum(broadcast(inner -> add(inner, 1), [1, 2])))\ny = f(3)"
            );
        }
    }

    /// Rename at `cursor`, then check that the result parses and infers clean.
    fn rename_clean(src: &str, cursor: u32, new_name: &str) -> String {
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = rename_edits(&db, f, fs, cats, cursor, index, new_name)
            .unwrap_or_else(|e| panic!("rename refused: {}", e.0));
        let renamed = apply(src, &locs, new_name);
        let mut m = flatppl_syntax::parse(&renamed).expect("still parses");
        let diags = flatppl_infer::infer_with(&mut m, flatppl_infer::Level::Shape);
        assert!(
            diags.is_empty(),
            "{renamed}: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
        renamed
    }

    /// The refusal for a rename at `cursor`.
    fn rename_refusal(src: &str, cursor: u32, new_name: &str) -> String {
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        match rename_edits(&db, f, fs, cats, cursor, index, new_name) {
            Ok(locs) => panic!("rename allowed: {}", apply(src, &locs, new_name)),
            Err(e) => e.0,
        }
    }

    /// §04 "Reification to functions and kernels" makes a repeated input name a
    /// static error. Once call-site keywords follow the rename, a merged input
    /// would also rewrite every call into a duplicate keyword.
    #[test]
    fn rename_onto_a_sibling_input_refuses() {
        for (src, cursor) in [
            ("f(a, b) = add(a, b)\ny = f(a = 1, b = 2)", 2),
            (
                "f = functionof(sub(_a_, _x_), a = _a_, b = _x_)\ny = f(a = 1, b = 2)",
                30,
            ),
        ] {
            let err = rename_refusal(src, cursor, "b");
            assert!(err.contains("already an input"), "{err}");
        }
    }

    /// Renaming `a` to `x` beside `b = _x_` would merge two inputs into one
    /// placeholder, and the result would still infer clean.
    #[test]
    fn rename_onto_an_existing_placeholder_refuses() {
        let src = "f = functionof(sub(_a_, _x_), a = _a_, b = _x_)\ny = f(a = 1, b = 2)";
        let err = rename_refusal(src, nth_offset(src, "a = _a_", 0), "x");
        assert!(err.contains("`_x_`"), "{err}");
    }

    /// §04 "Placeholders and holes" scopes a placeholder to its nearest
    /// reification, so a nested `_z_` does not block the outer rename.
    #[test]
    fn rename_onto_a_nested_scope_placeholder_is_allowed() {
        let src = "f = functionof(add(_a_, sum(broadcast(functionof(mul(_z_, 2.0), z = _z_), \
                   [1.0]))), a = _a_)\ny = f(a = 1.0)";
        assert_eq!(
            rename_clean(src, nth_offset(src, "a = _a_", 0), "z"),
            "f = functionof(add(_z_, sum(broadcast(functionof(mul(_z_, 2.0), z = _z_), \
             [1.0]))), z = _z_)\ny = f(z = 1.0)"
        );
    }

    #[test]
    fn rename_to_an_illegal_placeholder_refuses() {
        let src = "f = functionof(add(_a_, 1.0), a = _a_)\ny = f(a = 2.0)";
        let err = rename_refusal(src, nth_offset(src, "a = _a_", 0), "_q");
        assert!(err.contains("`__q_`"), "{err}");
    }

    /// The body reads `z` free, so an argument renamed to `z` would capture it.
    #[test]
    fn rename_onto_a_free_body_reference_refuses() {
        for (src, cursor, new_name) in [
            ("z = 1.0\nf(a) = add(a, z)\ny = f(a = 2.0)", 10, "z"),
            ("z = 1.0\nw = z\ng = a -> add(a, w)", 0, "w"),
            (
                "z = 1.0\nf(a) = add(a, sum(broadcast(b -> add(b, z), [1.0])))\ny = f(a = 2.0)",
                10,
                "z",
            ),
            ("f(a) = mul(a, pi)\ny = f(a = 2.0)", 2, "pi"),
            // The body would call the renamed argument instead of `add`.
            ("f(a) = add(a, 1.0)\ny = f(a = 2.0)", 2, "add"),
            (
                "f(a) = mul(a, sum(broadcast(b -> add(b, 1.0), [1.0])))\ny = f(a = 2.0)",
                2,
                "add",
            ),
        ] {
            let cursor = if cursor == 0 {
                nth_offset(src, "a ->", 0)
            } else {
                cursor
            };
            let err = rename_refusal(src, cursor, new_name);
            assert!(err.contains("capture"), "{src}: {err}");
        }
    }

    #[test]
    fn rename_onto_an_imported_alias_refuses() {
        let db = Database::default();
        let helpers = SourceFile::new(&db, "helpers.flatppl".to_string(), "s = 1.0".to_string());
        let model_src = "m = load_module(\"helpers.flatppl\")\nf(a) = add(a, m.s)";
        let model = SourceFile::new(&db, "model.flatppl".to_string(), model_src.to_string());
        let fs = FileSet::new(&db, vec![helpers, model]);
        let cats = Catalogues::new(&db, vec![]);
        let index = node_span_index(&db, model, fs, cats);
        let cursor = nth_offset(model_src, "f(a)", 0) + 2;
        let err = rename_edits(&db, model, fs, cats, cursor, index, "m").unwrap_err();
        assert!(err.0.contains("capture"), "{}", err.0);
    }

    /// §04 "Broadcasting" binds `broadcast(f, name = array)` keywords to `f`.
    #[test]
    fn rename_an_argument_rewrites_broadcast_keywords() {
        let src = "f(a) = mul(a, 2.0)\ny = broadcast(f, a = [1.0, 2.0])\nw = f.(a = [3.0])";
        assert_eq!(
            rename_clean(src, 2, "z"),
            "f(z) = mul(z, 2.0)\ny = broadcast(f, z = [1.0, 2.0])\nw = f.(z = [3.0])"
        );
    }

    /// A callable stored in a record can be called by keyword through the field,
    /// which the rename cannot follow, so it refuses instead of leaving the call.
    #[test]
    fn rename_refuses_when_the_callable_escapes_into_a_value() {
        let src = "f(a) = mul(a, 2.0)\ng = record(h = f)\ny = g.h(a = 2.0)";
        let err = rename_refusal(src, 2, "z");
        assert!(err.contains("used as a value"), "{err}");
    }

    /// §04 "Calling conventions": a one-input callable receives a lone
    /// positional argument whole, so `filter` and `markovchain` expose no name.
    #[test]
    fn rename_allows_a_one_input_callback() {
        for (src, cursor_at, expected) in [
            (
                "d = [1.0, -1.0]\ny = filter(x -> gt(x, 0.0), d)",
                "x ->",
                "d = [1.0, -1.0]\ny = filter(v -> gt(v, 0.0), d)",
            ),
            (
                "m = markovchain(s -> Normal(s, 1.0), 0.0, 5)",
                "s ->",
                "m = markovchain(v -> Normal(v, 1.0), 0.0, 5)",
            ),
        ] {
            assert_eq!(
                rename_clean(src, nth_offset(src, cursor_at, 0), "v"),
                expected
            );
        }
    }

    /// §04 "Explicit callback parameters" passes `p` whole to the callback,
    /// which can call it by keyword, so a callable passed as `p` escapes.
    #[test]
    fn rename_refuses_a_callable_passed_as_a_callback_parameter() {
        for src in [
            "f(a) = gt(a, 0.0)\nd = [1.0, -1.0]\nk = filter((x, p) -> p(a = x), d, f)",
            "f(a) = mul(a, 2.0)\nm = markovchain((s, p) -> Normal(p(a = s), 1.0), 0.0, 5, f)",
        ] {
            let err = rename_refusal(src, 2, "z");
            assert!(err.contains("used as a value"), "{src}: {err}");
        }
    }

    /// `fchain` can auto-splat a record into a multi-input callable by name.
    #[test]
    fn rename_refuses_a_multi_input_callable_in_fchain() {
        let src = "g(x) = record(a = x, b = x)\nf(a, b) = add(a, b)\nh = fchain(g, f)";
        let cursor = nth_offset(src, "f(a", 0) + 2;
        let err = rename_refusal(src, cursor, "z");
        assert!(err.contains("used as a value"), "{err}");
    }

    /// `pushfwd` applies its function positionally and exposes no input name.
    #[test]
    fn rename_allows_a_callable_consumed_by_pushfwd() {
        let src = "f(a) = mul(a, 2.0)\nm = pushfwd(f, Normal(0.0, 1.0))";
        assert_eq!(
            rename_clean(src, 2, "z"),
            "f(z) = mul(z, 2.0)\nm = pushfwd(f, Normal(0.0, 1.0))"
        );
    }

    /// An argument is governed by §05's plain `Name`, not §04's binding regexes, so
    /// a `__`-prefixed argument name — illegal for a module binding — is allowed
    /// here, as is shadowing a built-in (§05 "Lambda syntax").
    #[test]
    fn rename_an_argument_allows_shapes_a_binding_may_not_take() {
        let src = "f(a) = add(a, 1)";
        for candidate in ["__gen", "sqrt", "_priv"] {
            let (db, f, fs, cats) = single_file(src);
            let index = node_span_index(&db, f, fs, cats);
            let locs = rename_edits(&db, f, fs, cats, 2, index, candidate)
                .unwrap_or_else(|e| panic!("`{candidate}` must be a legal argument name: {}", e.0));
            assert_eq!(
                apply(src, &locs, candidate),
                format!("f({candidate}) = add({candidate}, 1)")
            );
        }
    }

    /// The parser rejects the `_name_` placeholder form as an argument name, so
    /// accepting it would produce a file that no longer parses.
    #[test]
    fn rename_an_argument_to_the_placeholder_form_refuses() {
        let src = "f(a) = add(a, 1)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let err = rename_edits(&db, f, fs, cats, 2, index, "_x_")
            .expect_err("the placeholder form is not a legal argument name");
        assert!(err.0.contains("placeholder"), "got {}", err.0);
        // Confirm the premise: the parser really does reject it.
        assert!(flatppl_syntax::parse("f(_x_) = add(_x_, 1)").is_err());
    }

    /// The brief's form: a keyword boundary whose placeholder spells the input
    /// name. The keyword, the placeholder and the call-site keywords move
    /// together, from any of the three cursors.
    #[test]
    fn rename_a_keyword_boundary_argument() {
        let src = "f = functionof(add(_a_, 1), a = _a_)\ny = f(a = 2)";
        let expected = "f = functionof(add(_z_, 1), z = _z_)\ny = f(z = 2)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let keyword = nth_offset(src, "a = _a_", 0);
        let body = nth_offset(src, "_a_", 0);
        let value = nth_offset(src, "_a_", 1) + 1;
        for cursor in [keyword, body, value] {
            let locs = rename_edits(&db, f, fs, cats, cursor, index, "z").expect("allowed");
            assert_eq!(apply(src, &locs, "z"), expected, "cursor {cursor}");
        }
        assert_eq!(
            prepare_rename(&db, f, fs, cats, body, index),
            Ok((body + 1, body + 2)),
            "the inner name of `_a_` is the editable range"
        );
        let mut m = flatppl_syntax::parse(expected).expect("still parses");
        let diags = flatppl_infer::infer_with(&mut m, flatppl_infer::Level::Shape);
        assert!(
            diags.is_empty(),
            "got {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    /// A keyword that does not spell its value renames alone: the value is a
    /// different placeholder or a module binding.
    #[test]
    fn rename_a_keyword_boundary_argument_leaves_an_unrelated_value() {
        for (src, expected) in [
            (
                "f = functionof(add(_x_, 1), a = _x_)\ny = f(a = 2)",
                "f = functionof(add(_x_, 1), z = _x_)\ny = f(z = 2)",
            ),
            (
                "a = 1\nf = functionof(add(a, 1), a = a)\ny = f(a = 2)",
                "a = 1\nf = functionof(add(a, 1), z = a)\ny = f(z = 2)",
            ),
        ] {
            let (db, f, fs, cats) = single_file(src);
            let index = node_span_index(&db, f, fs, cats);
            let cursor = nth_offset(src, ", a =", 0) + 2;
            let locs = rename_edits(&db, f, fs, cats, cursor, index, "z").expect("allowed");
            assert_eq!(apply(src, &locs, "z"), expected);
        }
    }

    /// An immediately invoked callable is its own call site.
    #[test]
    fn rename_an_inline_callee_argument_rewrites_its_keyword() {
        for (src, cursor_at, expected) in [
            (
                "y = (a -> add(a, 1))(a = 2)",
                "a ->",
                "y = (z -> add(z, 1))(z = 2)",
            ),
            (
                "y = functionof(add(_a_, 1), a = _a_)(a = 2)",
                "a = _a_",
                "y = functionof(add(_z_, 1), z = _z_)(z = 2)",
            ),
        ] {
            let (db, f, fs, cats) = single_file(src);
            let index = node_span_index(&db, f, fs, cats);
            let cursor = nth_offset(src, cursor_at, 0);
            let locs = rename_edits(&db, f, fs, cats, cursor, index, "z").expect("allowed");
            assert_eq!(apply(src, &locs, "z"), expected);
        }
    }

    /// A bare-name boundary derives the input name from its entry, so there is
    /// no keyword to rename.
    #[test]
    fn rename_a_bare_boundary_argument_fails_closed() {
        let src = "k = kernelof(Normal(_mu_, 1.0), _mu_)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let err = rename_edits(&db, f, fs, cats, nth_offset(src, "_mu_", 0), index, "nu")
            .expect_err("no keyword to rename");
        assert!(err.0.contains("bare-name"), "got {}", err.0);
    }

    #[test]
    fn rename_to_the_same_name_is_a_no_op() {
        let src = "x = 1\ny = add(x, 2)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        assert!(
            rename_edits(&db, f, fs, cats, 0, index, "x")
                .expect("allowed")
                .is_empty()
        );
    }

    #[test]
    fn rename_a_module_alias_leaves_the_member_alone() {
        let db = Database::default();
        let helpers = SourceFile::new(&db, "helpers.flatppl".to_string(), "s = 1.0".to_string());
        let model_src = "h = load_module(\"helpers.flatppl\")\nv = h.s\nw = add(h.s, 1.0)";
        let model = SourceFile::new(&db, "model.flatppl".to_string(), model_src.to_string());
        let fs = FileSet::new(&db, vec![helpers, model]);
        let cats = Catalogues::new(&db, vec![]);
        let index = node_span_index(&db, model, fs, cats);
        let locs = rename_edits(&db, model, fs, cats, 0, index, "helpers").expect("allowed");
        assert!(
            locs.iter().all(|l| l.path == "model.flatppl"),
            "an alias is local to its module; got {locs:?}"
        );
        assert_eq!(
            apply(model_src, &locs, "helpers"),
            "helpers = load_module(\"helpers.flatppl\")\nv = helpers.s\nw = add(helpers.s, 1.0)"
        );
    }
}
