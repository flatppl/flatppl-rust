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

use flatppl_core::{BindingId, Node, Ref, RefNs};

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

/// The binding a rename or a references request is about.
struct Target {
    /// The file that defines the binding — not necessarily the requesting file.
    file: SourceFile,
    bid: BindingId,
    name: String,
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
            return Ok(Target {
                file,
                bid,
                name: name.to_string(),
            });
        }
    }

    let node_id = node_at_offset_indexed(index, byte_offset).ok_or_else(unresolved)?;
    let Node::Ref(r) = module.node(node_id) else {
        return Err(unresolved());
    };
    let span = module.span_of(node_id).ok_or_else(unresolved)?;
    match r.ns {
        // §04 "Objects, expressions, names and modules": the "argument names of
        // functions and kernels" are not part of the module namespace. A `%local`
        // placeholder is scoped to its reification, so renaming one is a
        // different operation from renaming a binding; it is not supported.
        RefNs::Local => Err(Refusal(
            "`_`-scoped placeholders (function and kernel argument names) are local \
             to their callable, not module bindings (spec §04 \"Objects, expressions, \
             names and modules\"), and are not renameable"
                .to_string(),
        )),
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
                return Ok(Target {
                    file: def_file,
                    bid,
                    name,
                });
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
            Ok(Target {
                file: def_file,
                bid,
                name,
            })
        }
        RefNs::SelfMod => {
            let (def_file, bid) =
                resolve_ref_def(db, file, fs, cats, module, r).ok_or_else(unresolved)?;
            let def_mod = parse(db, def_file).module(db).ok_or_else(unresolved)?;
            let name = def_mod.resolve(def_mod.binding(bid).name).to_string();
            Ok(Target {
                file: def_file,
                bid,
                name,
            })
        }
    }
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
/// is how a built-in, a standard-module member, a record field and a `%local`
/// placeholder are all declined before the user types a new name.
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
/// - **Illegal new name.** §04 "Binding names" and §05 "Note on reserved words",
///   enforced by [`names::check_new_name`].
/// - **Collision.** §04 "Objects, expressions, names and modules": "A FlatPPL
///   **module** is an unordered set of bindings of names to expressions." A second
///   binding of the same name in one module is not such a set, so a rename onto an
///   existing name in the defining module refuses.
/// - **Public → private with a cross-module reference.** §04 "Binding names":
///   public names "form the interface of a FlatPPL module", while private ones
///   "are not part of the module's public interface". §04 "Module composition"
///   gives the loading module access to "names in the loaded module" — i.e. to
///   that interface. Prefixing an underscore therefore removes a name another
///   file still reaches through `alias.name`, so the rename refuses instead of
///   writing a bundle that no longer resolves.
///
/// Shadowing a built-in is deliberately NOT refused: §04 "Name resolution" says
/// "This makes built-in names shadowable: a module may bind any name except for
/// `self` and `base`."
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
    names::check_new_name(new_name)?;
    if new_name == target.name {
        return Ok(Vec::new());
    }

    let def_mod = parse(db, target.file).module(db).ok_or_else(|| {
        Refusal("the defining file does not parse, so it cannot be edited".to_string())
    })?;
    if def_mod
        .bindings()
        .any(|(bid, b)| bid != target.bid && def_mod.resolve(b.name) == new_name)
    {
        return Err(Refusal(format!(
            "`{new_name}` is already bound in {}; a module is a set of bindings of \
             names to expressions (spec §04), so the name cannot be bound twice",
            target.file.path(db)
        )));
    }

    let locs = occurrences(db, fs, cats, &target, true);

    if names::is_private(new_name) && !names::is_private(&target.name) {
        let def_path = target.file.path(db).clone();
        if let Some(other) = locs.iter().find(|l| l.path != def_path) {
            return Err(Refusal(format!(
                "`{}` is referenced from {} across a module boundary; making it \
                 private (spec §04 \"Binding names\") would drop it from this \
                 module's public interface and leave that reference unresolvable",
                target.name, other.path
            )));
        }
    }

    // A definition site we could not locate textually means the rewrite would be
    // partial. Refuse rather than emit a broken edit.
    if !locs.iter().any(|l| l.path == *target.file.path(db)) {
        return Err(Refusal(format!(
            "could not locate the definition of `{}` in {}",
            target.name,
            target.file.path(db)
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

    #[test]
    fn prepare_rename_refuses_a_function_parameter() {
        // §04: argument names of functions and kernels are local to the callable.
        let src = "f(a) = add(a, 1)";
        let (db, f, fs, cats) = single_file(src);
        let index = node_span_index(&db, f, fs, cats);
        let off = nth_offset(src, "add(a", 0) + 4;
        let err = prepare_rename(&db, f, fs, cats, off, &index).expect_err("parameter must refuse");
        assert!(
            err.0.contains("placeholder"),
            "refusal must cite the placeholder rule; got {}",
            err.0
        );
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

    #[test]
    fn rename_public_to_private_refuses_when_referenced_across_a_module_boundary() {
        let db = Database::default();
        let helpers = SourceFile::new(&db, "helpers.flatppl".to_string(), "s = 1.0".to_string());
        let model_src = "h = load_module(\"helpers.flatppl\")\nv = h.s";
        let model = SourceFile::new(&db, "model.flatppl".to_string(), model_src.to_string());
        let fs = FileSet::new(&db, vec![helpers, model]);
        let cats = Catalogues::new(&db, vec![]);
        let index = node_span_index(&db, helpers, fs, cats);
        let err = rename_edits(&db, helpers, fs, cats, 0, &index, "_s")
            .expect_err("public → private with a cross-module reference must refuse");
        assert!(
            err.0.contains("private") && err.0.contains("model.flatppl"),
            "refusal must name the rule and the referring file; got {}",
            err.0
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
