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

use flatppl_core::{BindingId, CallHead, Inputs, Node, Ref, RefNs};

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
    /// `functionof` or `kernelof`", so the rename is confined to one binding's
    /// RHS in one file — never cross-file.
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
    /// The binding whose RHS reifies the callable that declares this argument.
    bid: BindingId,
    /// The surface argument name, e.g. `a`.
    name: String,
    /// The lowered placeholder the body references, e.g. `_a_`.
    placeholder: String,
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
        if let Some(t) = argument_target_at(module, text, file, bid, byte_offset, None) {
            return Ok(t);
        }
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
            // Find the binding whose RHS reifies the callable declaring it.
            let owner = module
                .bindings()
                .filter(|(_, b)| {
                    module
                        .span_of(b.rhs)
                        .is_some_and(|s| s.start <= span.start && span.end <= s.end)
                })
                .min_by_key(|(_, b)| {
                    let s = module.span_of(b.rhs).expect("filtered to spanned RHS");
                    s.end - s.start
                });
            let (bid, _) = owner.ok_or_else(|| {
                Refusal(format!(
                    "could not find the callable that declares `{placeholder}`"
                ))
            })?;
            argument_target_at(module, text, file, bid, byte_offset, Some(&placeholder)).ok_or_else(
                || {
                    Refusal(format!(
                        "`{placeholder}` is a placeholder whose declaration site has no \
                         recorded span: an explicit `functionof`/`kernelof` boundary \
                         declares it as a keyword argument, which the IR stores without \
                         a source span, so the rename cannot be applied safely. Renaming \
                         the argument of a `f(a) = …` definition or a lambda is supported."
                    ))
                },
            )
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

/// Build an [`ArgumentTarget`] for the callable reified by binding `bid`.
///
/// Selects the argument either by `placeholder` (when the cursor was on a body
/// reference) or by the cursor falling inside a declared argument's range (when it
/// was on the declaration list). Returns `None` when `bid`'s RHS is not a
/// reification with an ordered boundary, when the declaration list cannot be
/// located in the source, or when the requested argument is not among the
/// declared ones — every one a reason to fail closed rather than edit.
fn argument_target_at(
    module: &flatppl_core::Module,
    text: &str,
    file: SourceFile,
    bid: BindingId,
    byte_offset: u32,
    placeholder: Option<&str>,
) -> Option<Target> {
    let binding = module.binding(bid);
    let Node::Call(call) = module.node(binding.rhs) else {
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
    let reif_span = module.span_of(binding.rhs)?;
    let def_name = names::def_name_range(text, reif_span.start, module.resolve(binding.name));
    let declared = names::argument_decl_ranges(text, reif_span.start, def_name);
    if declared.is_empty() {
        return None;
    }
    let siblings: Vec<String> = declared.iter().map(|(n, _, _)| n.clone()).collect();

    // The surface name we are after.
    let name = match placeholder {
        // A body reference: invert the parser's `_{name}_` lowering. The explicit
        // boundary spelling (where the body writes `_mu_` itself) does not invert,
        // and its declaration is an unspanned keyword argument, so it yields None.
        Some(ph) => {
            let surface = names::placeholder_surface_name(ph)?;
            if !siblings.iter().any(|s| s == surface) {
                return None;
            }
            surface.to_string()
        }
        // A declaration site: the argument whose range contains the cursor.
        None => declared
            .iter()
            .find(|(_, s, e)| *s <= byte_offset && byte_offset < *e)
            .map(|(n, _, _)| n.clone())?,
    };
    // The boundary must actually declare it, so a source list that disagrees with
    // the IR (a shape this code does not model) fails closed.
    let entry = entries
        .iter()
        .find(|(sym, _)| module.resolve(*sym) == name)?;
    Some(Target::Argument(ArgumentTarget {
        file,
        bid,
        placeholder: module.resolve(entry.1.name).to_string(),
        name,
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
        Target::Argument(t) => argument_occurrences(db, t, include_declaration),
    }
}

/// Every occurrence of an argument name: each `%local` reference to its
/// placeholder inside the owning binding's RHS, plus the declaration site in the
/// argument list.
///
/// Confined to one binding in one file. §04 "Placeholders and holes" gives the
/// scoping rule — "The scope of a placeholder is the nearest enclosing
/// `functionof` or `kernelof`" — and adds that "The same placeholder name may
/// appear in different scopes without conflict", so the containment test below is
/// what keeps a same-named argument of a *different* callable out of the set.
fn argument_occurrences(
    db: &dyn salsa::Database,
    target: &ArgumentTarget,
    include_declaration: bool,
) -> Vec<NameLoc> {
    use flatppl_core::Idx;
    let Some(module) = parse(db, target.file).module(db) else {
        return Vec::new();
    };
    let text = target.file.text(db);
    let path = target.file.path(db).clone();
    let binding = module.binding(target.bid);
    let Some(rhs_span) = module.span_of(binding.rhs) else {
        return Vec::new();
    };

    let mut out: Vec<NameLoc> = Vec::new();
    let mut seen: HashSet<NameLoc> = HashSet::new();
    let mut push = |start: u32, end: u32| {
        let loc = NameLoc {
            path: path.clone(),
            start,
            end,
        };
        if seen.insert(loc.clone()) {
            out.push(loc);
        }
    };

    for i in 0..module.node_count() {
        let id = flatppl_core::NodeId::from_usize(i);
        let Node::Ref(r) = module.node(id) else {
            continue;
        };
        if !matches!(r.ns, RefNs::Local) || module.resolve(r.name) != target.placeholder {
            continue;
        }
        let Some(span) = module.span_of(id) else {
            continue;
        };
        // Only references inside the owning reification's RHS.
        if span.start < rhs_span.start || span.end > rhs_span.end {
            continue;
        }
        // The body writes the SURFACE name for a sugar argument, so the range must
        // match that, not the placeholder. A mismatch fails closed.
        if let Some((s, e)) = names::ref_name_range(text, span.start, RefPart::Head, &target.name) {
            push(s, e);
        }
    }

    if include_declaration {
        let def_name = names::def_name_range(text, rhs_span.start, module.resolve(binding.name));
        for (n, s, e) in names::argument_decl_ranges(text, rhs_span.start, def_name) {
            if n == target.name {
                push(s, e);
            }
        }
    }
    out
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
            if let Some(span) = module.span_of(binding.rhs) {
                if let Some((start, end)) = names::def_name_range(text, span.start, &target.name) {
                    push(NameLoc {
                        path: path.clone(),
                        start,
                        end,
                    });
                }
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
    // The renameable range is the occurrence under the cursor itself, so the
    // editor highlights exactly the identifier it will replace.
    occurrences(db, fs, cats, &target, true)
        .into_iter()
        .find(|l| l.path == *file.path(db) && l.start <= byte_offset && byte_offset < l.end)
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
///
/// Deliberately NOT refused, because no rule forbids them, or because `infer`
/// reports the consequence precisely:
///
/// - **Renaming an argument onto a sibling.** §04 "Reification to functions and
///   kernels" makes a repeated boundary input name a static error, and `infer` now
///   reports it at the reification — "boundary input `b` is declared more than
///   once" (measured by `duplicate_argument_name_is_diagnosed`). So the rename is
///   performed and the diagnostic flags the result, rather than the server blocking
///   a legitimate "rename this, then fix the other argument" edit.
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
        }
    }

    let locs = occurrences(db, fs, cats, &target, true);

    // A definition site we could not locate textually means the rewrite would be
    // partial. This is a fail-closed implementation guard, not a doctrine refusal:
    // no rule forbids the rename, we simply cannot compute a correct edit for it.
    let (def_path, name) = match &target {
        Target::Binding(t) => (t.file.path(db).clone(), &t.name),
        Target::Argument(t) => (t.file.path(db).clone(), &t.name),
    };
    if !locs.iter().any(|l| l.path == def_path) {
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
        let locs = references(&db, f, fs, cats, 0, &index, true);
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
        let locs = references(&db, f, fs, cats, 0, &index, false);
        assert_eq!(locs.len(), 1, "only the use in `add(x, 2)`; got {locs:?}");
        assert!(locs.iter().all(|l| l.start != 0));
    }

    #[test]
    fn references_does_not_match_a_longer_name() {
        let src = "x = 1\nxy = 2\nz = add(x, xy)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = references(&db, f, fs, cats, 0, &index, true);
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
        let locs = references(&db, f, fs, cats, 9, &index, true);
        assert_eq!(locs.len(), 2, "`x` decl + its use; got {locs:?}");
    }

    #[test]
    fn references_on_a_function_definition_name() {
        let src = "f(a) = add(a, 1)\nz = f(3)\nw = f(4)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = references(&db, f, fs, cats, 0, &index, true);
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
        let locs = references(&db, helpers, fs, cats, 0, &index, true);
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
        let locs = references(&db, model, fs, cats, off, &index, true);
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
        let locs = references(&db, model, fs, cats, 0, &index, true);
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
            references(&db, f, fs, cats, off, &index, true).is_empty(),
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
        assert_eq!(prepare_rename(&db, f, fs, cats, off, &index), Ok((14, 15)));
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
        let range = prepare_rename(&db, model, fs, cats, dot + 2, &index).expect("renameable");
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
        let err = prepare_rename(&db, f, fs, cats, off, &index).expect_err("builtin must refuse");
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
        let range = prepare_rename(&db, f, fs, cats, off, &index)
            .expect("a function argument is renameable");
        assert_eq!(&src[range.0 as usize..range.1 as usize], "a");
    }

    #[test]
    fn prepare_rename_allows_an_argument_declaration_site() {
        // Offset 2 is the `a` inside `f(a)`, which carries no node of its own.
        let src = "f(a) = add(a, 1)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let range = prepare_rename(&db, f, fs, cats, 2, &index).expect("declaration renameable");
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
        let err = prepare_rename(&db, f, fs, cats, off, &index)
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
            prepare_rename(&db, f, fs, cats, off, &index).is_err(),
            "a record field must not be renameable"
        );
    }

    // ── rename ───────────────────────────────────────────────────────────────

    #[test]
    fn rename_rewrites_every_occurrence_in_one_file() {
        let src = "x = 1\ny = add(x, 2)\nz = mul(x, x)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = rename_edits(&db, f, fs, cats, 0, &index, "theta").expect("rename allowed");
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
        let locs = rename_edits(&db, model, fs, cats, off, &index, "offset").expect("allowed");

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
        let err =
            rename_edits(&db, f, fs, cats, 0, &index, "y").expect_err("collision must refuse");
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
                rename_edits(&db, f, fs, cats, 0, &index, bad).is_err(),
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
        assert!(rename_edits(&db, f, fs, cats, off, &index, "plus").is_err());
    }

    #[test]
    fn rename_allows_shadowing_a_builtin_name() {
        // §04 "Name resolution": built-in names are shadowable.
        let src = "x = 1\ny = add(x, 2)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = rename_edits(&db, f, fs, cats, 0, &index, "sqrt")
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
        let locs = rename_edits(&db, helpers, fs, cats, 0, &index, "_s")
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
        let locs = rename_edits(&db, f, fs, cats, 0, &index, "_s")
            .expect("no cross-module reference, so the visibility change is safe");
        assert_eq!(apply(src, &locs, "_s"), "_s = 1.0\nu = add(_s, 1.0)");
    }

    #[test]
    fn rename_private_to_public_is_allowed() {
        let src = "_s = 1.0\nu = add(_s, 1.0)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs =
            rename_edits(&db, f, fs, cats, 0, &index, "s").expect("widening visibility is legal");
        assert_eq!(apply(src, &locs, "s"), "s = 1.0\nu = add(s, 1.0)");
    }

    // ── argument (%local) rename ──────────────────────────────────────────────

    #[test]
    fn references_on_an_argument_cover_the_declaration_and_body_uses() {
        let src = "f(a, b) = add(mul(a, a), b)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = references(&db, f, fs, cats, 2, &index, true);
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
        let locs = rename_edits(&db, f, fs, cats, 2, &index, "scale").expect("allowed");
        assert_eq!(
            apply(src, &locs, "scale"),
            "f(scale, b) = add(mul(scale, scale), b)"
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
        let locs = rename_edits(&db, f, fs, cats, off, &index, "x").expect("allowed");
        assert_eq!(apply(src, &locs, "x"), "g = (x, b) -> add(x, b)");
    }

    #[test]
    fn rename_a_single_bare_lambda_argument() {
        // §05: `arg -> expr` with no parentheses in the single-argument form.
        let src = "s = a -> add(a, 1)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = rename_edits(&db, f, fs, cats, 4, &index, "v").expect("allowed");
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
        let locs = rename_edits(&db, f, fs, cats, 2, &index, "z").expect("allowed");
        assert_eq!(apply(src, &locs, "z"), "f(z) = add(z, 1)\ng(a) = mul(a, 2)");
    }

    /// §04 "Reification to functions and kernels" makes a repeated argument name a
    /// static error, and `infer` reports it (see
    /// `duplicate_argument_name_is_diagnosed`), so the server performs the edit and
    /// lets the diagnostic flag the result instead of refusing.
    #[test]
    fn rename_an_argument_onto_a_sibling_is_allowed_and_diagnosed() {
        let src = "f(a, b) = add(a, b)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let locs = rename_edits(&db, f, fs, cats, 2, &index, "b").expect("allowed");
        let renamed = apply(src, &locs, "b");
        assert_eq!(renamed, "f(b, b) = add(b, b)");
        let mut m = flatppl_syntax::parse(&renamed).expect("still parses");
        let diags = flatppl_infer::infer_with(&mut m, flatppl_infer::Level::Shape);
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("declared more than once")),
            "the result must be diagnosed; got {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    /// The measured basis for allowing the rename above: `infer` flags a repeated
    /// argument name, so the LSP need not refuse to keep the file legal.
    #[test]
    fn duplicate_argument_name_is_diagnosed() {
        let mut m = flatppl_syntax::parse("f(a, a) = add(a, a)").expect("parses");
        let diags = flatppl_infer::infer_with(&mut m, flatppl_infer::Level::Shape);
        assert!(
            diags.iter().any(|d| d
                .message
                .contains("boundary input `a` is declared more than once")),
            "got {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
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
            let locs = rename_edits(&db, f, fs, cats, 2, &index, candidate)
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
        let err = rename_edits(&db, f, fs, cats, 2, &index, "_x_")
            .expect_err("the placeholder form is not a legal argument name");
        assert!(err.0.contains("placeholder"), "got {}", err.0);
        // Confirm the premise: the parser really does reject it.
        assert!(flatppl_syntax::parse("f(_x_) = add(_x_, 1)").is_err());
    }

    /// An explicit `functionof`/`kernelof` boundary declares its argument as a
    /// keyword argument, which the IR stores without a span. That is an
    /// implementation limit, not a doctrine refusal, and the message says so.
    #[test]
    fn rename_an_explicit_boundary_argument_fails_closed() {
        let src = "k = kernelof(Normal(_mu_, 1.0), mu = _mu_)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let off = nth_offset(src, "_mu_", 0);
        let err = rename_edits(&db, f, fs, cats, off, &index, "nu")
            .expect_err("no recorded span for the boundary keyword");
        assert!(
            err.0.contains("recorded span"),
            "the message must state the implementation limit; got {}",
            err.0
        );
    }

    #[test]
    fn rename_to_the_same_name_is_a_no_op() {
        let src = "x = 1\ny = add(x, 2)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        assert!(
            rename_edits(&db, f, fs, cats, 0, &index, "x")
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
        let locs = rename_edits(&db, model, fs, cats, 0, &index, "helpers").expect("allowed");
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
