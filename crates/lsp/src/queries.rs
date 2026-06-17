//! Derived salsa queries over the source inputs.

use crate::capabilities::LspDiag;
use crate::db::{Catalogues, FileSet, SourceFile};
use flatppl_core::{CallHead, Module, Node, Scalar};
use std::sync::Arc;

// ── Salsa field-compatibility wrapper ───────────────────────────────────────
//
// salsa 0.18 tracked-struct fields (even `#[return_ref]` ones) must satisfy
// `Hash + Update` (the latter is either the `salsa::Update` trait or the
// fallback `'static + PartialEq`).  `Module` derives only `Clone + Debug +
// Default`; it has neither `Hash` nor `PartialEq`.
//
// `ArcModule` wraps `Arc<Module>` and provides pointer-identity `Hash`,
// `PartialEq`, and `Eq`.  Pointer identity is sound for salsa's purposes:
// two separately-parsed arcs are *never* pointer-equal, so salsa will always
// see a change, which is the conservatively-correct behaviour (over-recomputes
// rather than under-recomputes).  `LspDiag` has structural `Eq`, so the
// diagnostics field gates actual short-circuit reuse correctly.
#[derive(Clone, Debug)]
pub struct ArcModule(Arc<Module>);

impl ArcModule {
    fn new(m: Module) -> Self {
        ArcModule(Arc::new(m))
    }

    fn get(&self) -> &Module {
        &self.0
    }
}

impl PartialEq for ArcModule {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ArcModule {}

impl std::hash::Hash for ArcModule {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Arc::as_ptr(&self.0), state);
    }
}

// ── Parsed tracked struct ────────────────────────────────────────────────────

/// The result of parsing a single source file: an optional module (present on
/// success) and a list of diagnostics (empty on success, one error on failure).
///
/// `module` is stored as `Option<ArcModule>` — a pointer-identity-comparable
/// wrapper around `Arc<Module>` — so that the field satisfies salsa's
/// `Hash + PartialEq` requirements without requiring `Module: Hash + PartialEq`.
/// Callers access the module via [`Parsed::module`] which returns `Option<&Module>`.
#[salsa::tracked]
pub struct Parsed<'db> {
    #[return_ref]
    module_arc: Option<ArcModule>,
    #[return_ref]
    pub diagnostics: Vec<LspDiag>,
}

impl<'db> Parsed<'db> {
    /// Return a reference to the parsed `Module`, or `None` on parse error.
    pub fn module(self, db: &'db dyn salsa::Database) -> Option<&'db Module> {
        self.module_arc(db).as_ref().map(|a| a.get())
    }
}

// ── parse tracked query ──────────────────────────────────────────────────────

/// Parse a [`SourceFile`] into a [`Parsed`] result.
///
/// On success the module is `Some` and diagnostics are empty; on failure the
/// module is `None` and diagnostics contain a single error mapped via
/// [`LspDiag::from_parse_error`].
#[salsa::tracked]
pub fn parse<'db>(db: &'db dyn salsa::Database, file: SourceFile) -> Parsed<'db> {
    match flatppl_syntax::parse(file.text(db)) {
        Ok(module) => Parsed::new(db, Some(ArcModule::new(module)), Vec::new()),
        Err(e) => Parsed::new(db, None, vec![LspDiag::from_parse_error(&e)]),
    }
}

// ── Cross-file load_module resolution ────────────────────────────────────────

/// Extract the literal path string of every `load_module` / `standard_module`
/// directive in `module`: iterate bindings, and for each binding whose RHS is a
/// `Call` with a builtin head named `load_module` or `standard_module`, take
/// `args[0]` when it is a `Scalar::Str`.
///
/// `standard_module` paths are module *names*, not workspace files; they will
/// not resolve to a `SourceFile` (so `resolve_path` returns `None` and they are
/// correctly skipped from the bundle). Standard-module resolution happens inside
/// `infer` via the catalogue, not the bundle.
pub(crate) fn load_module_paths(module: &Module) -> Vec<String> {
    let mut paths = Vec::new();
    for (_, binding) in module.bindings() {
        let Node::Call(call) = module.node(binding.rhs) else {
            continue;
        };
        let CallHead::Builtin(head) = call.head else {
            continue;
        };
        let head_name = module.resolve(head);
        if head_name != "load_module" && head_name != "standard_module" {
            continue;
        }
        if let Some(Node::Lit(Scalar::Str(s))) = call.args.first().map(|&a| module.node(a)) {
            paths.push(s.to_string());
        }
    }
    paths
}

/// Normalize a slash-separated path: drop `.` segments and collapse `..` against
/// the preceding non-`..` segment. Leading `..` segments that cannot be
/// collapsed are preserved. No filesystem access — purely lexical, matching how
/// directive paths are compared against workspace `SourceFile` paths.
fn normalize_path(path: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                if matches!(out.last(), Some(&s) if s != "..") {
                    out.pop();
                } else {
                    out.push("..");
                }
            }
            other => out.push(other),
        }
    }
    out.join("/")
}

/// Resolve `directive_path` (from a `load_module` directive in `importer`) to a
/// workspace [`SourceFile`] in `fs`.
///
/// Resolution joins the directive path onto the importer's parent directory and
/// normalizes it, comparing the result against each `SourceFile`'s (normalized)
/// path. A direct match where the directive path equals a `SourceFile` path is
/// also accepted (so paths that are already workspace-relative resolve without a
/// parent prefix). Returns `None` when nothing matches (the common case for
/// `standard_module` names).
pub(crate) fn resolve_path(
    db: &dyn salsa::Database,
    importer: SourceFile,
    directive_path: &str,
    fs: FileSet,
) -> Option<SourceFile> {
    let importer_path = importer.path(db);
    let parent = match importer_path.rfind('/') {
        Some(i) => &importer_path[..i],
        None => "",
    };
    let joined = if parent.is_empty() {
        directive_path.to_string()
    } else {
        format!("{parent}/{directive_path}")
    };
    let joined_norm = normalize_path(&joined);
    let direct_norm = normalize_path(directive_path);
    // Prefer the relative-`joined` interpretation (deterministic in ambiguous
    // workspaces where both `a/b/x.flatppl` and `b/x.flatppl` exist); only fall
    // back to a direct path match when no joined match exists.
    let by = |target: &str| {
        fs.files(db)
            .iter()
            .copied()
            .find(|f| normalize_path(&f.path(db)) == target)
    };
    by(&joined_norm).or_else(|| by(&direct_norm))
}

/// Build the **transitively closed** cross-file
/// [`ModuleBundle`](flatppl_infer::ModuleBundle) for `file`.
///
/// `flatppl_infer`'s cross-module resolution reuses the *same* bundle to resolve
/// a dependency's own `load_module` refs while inferring it. So a chain
/// model → helpers → utils requires `utils` to be in the bundle too, even though
/// `model` only directly imports `helpers`. A direct-only bundle would leave
/// `utils` unresolved when infer walks `helpers`.
///
/// This therefore runs a breadth-first worklist over `SourceFile`s: seed it with
/// `file`, and for each file pop'd, scan its `load_module` directives, resolve
/// each directive path against *that file's* directory (via `resolve_path`),
/// `parse(db, dep)` the resolved dependency — keeping the salsa edge — insert it
/// into the bundle under its directive-literal path, and enqueue it. A `visited`
/// set keyed on `SourceFile` (a `Copy + Eq + Hash` salsa id) terminates the walk
/// on cyclic imports rather than spinning; the engine handles module cycles at
/// inference time, the bundle build only needs every reachable dep present once.
///
/// Reading every dependency's `parse` here is what records the cross-file salsa
/// dependency edges: because `analyze` calls `import_bundle`, `analyze`'s read
/// set transitively includes every (direct or transitive) dependency's `parse`,
/// so editing any dependency's text invalidates the importer's analysis.
///
/// Each dependency is inserted under the *same* literal directive path string
/// the directive used, so `infer`'s `(%ref alias X)` resolution (which keys the
/// bundle by the directive path) finds it.
///
/// This is a plain `fn`, not a `#[salsa::tracked]` query: it cannot be tracked
/// because `ModuleBundle` is neither `Hash`/`Eq` nor `Update` (it owns
/// `Module`s). Tracking is unnecessary — the salsa edges it creates are recorded
/// in its caller's (`analyze`'s) read set regardless.
pub fn import_bundle(
    db: &dyn salsa::Database,
    file: SourceFile,
    fs: FileSet,
) -> flatppl_infer::ModuleBundle {
    let mut bundle = flatppl_infer::ModuleBundle::new();
    let mut visited: std::collections::HashSet<SourceFile> = std::collections::HashSet::new();
    let mut queue: std::collections::VecDeque<SourceFile> = std::collections::VecDeque::new();
    visited.insert(file);
    queue.push_back(file);

    while let Some(current) = queue.pop_front() {
        let Some(module) = parse(db, current).module(db) else {
            continue;
        };
        for path in load_module_paths(module) {
            // Resolve each directive relative to the file that declares it.
            let Some(dep_file) = resolve_path(db, current, &path, fs) else {
                continue;
            };
            if let Some(dep_mod) = parse(db, dep_file).module(db) {
                // Key by the directive-literal path so infer's `(%ref alias X)`
                // lookup (keyed by directive path) matches.
                bundle.insert(path, dep_mod.clone());
            }
            // Enqueue once; the visited set is keyed on the resolved file so two
            // directives that resolve to the same file (or an import cycle) do
            // not re-walk it.
            if visited.insert(dep_file) {
                queue.push_back(dep_file);
            }
        }
    }
    bundle
}

// ── Analyzed tracked struct ──────────────────────────────────────────────────

/// The result of analyzing (parsing + inferring) a single source file against
/// the workspace: an optional inferred module (the `Module` carries its inferred
/// type/phase side-tables) and the combined parse + inference diagnostics.
///
/// `module` reuses the [`ArcModule`] wrapper (pointer-identity `Hash`/`Eq`) so
/// the field satisfies salsa's storage bounds without `Module: Hash + Eq`.
/// Access the inferred module via [`Analyzed::module`].
#[salsa::tracked]
pub struct Analyzed<'db> {
    #[return_ref]
    module_arc: Option<ArcModule>,
    #[return_ref]
    pub diagnostics: Vec<LspDiag>,
}

impl<'db> Analyzed<'db> {
    /// Return a reference to the inferred `Module`, or `None` on parse error.
    pub fn module(self, db: &'db dyn salsa::Database) -> Option<&'db Module> {
        self.module_arc(db).as_ref().map(|a| a.get())
    }
}

/// Analyze `file`: parse it, build the cross-file `load_module` bundle from
/// `fs`, parse `cats`' RON catalogue sources, and run inference at
/// [`Level::Normalization`](flatppl_infer::Level::Normalization).
///
/// Diagnostics combine the parse diagnostics, any catalogue RON parse errors,
/// and the inference diagnostics. On parse failure the module is `None` and only
/// the parse diagnostics are returned.
#[salsa::tracked]
pub fn analyze<'db>(
    db: &'db dyn salsa::Database,
    file: SourceFile,
    fs: FileSet,
    cats: Catalogues,
) -> Analyzed<'db> {
    let parsed = parse(db, file);
    let Some(module) = parsed.module(db) else {
        return Analyzed::new(db, None, parsed.diagnostics(db).clone());
    };
    let mut m = module.clone();

    // Parse the external catalogues; RON failures become Error diagnostics
    // anchored at offset 0 rather than aborting analysis.
    let mut diags = parsed.diagnostics(db).clone();
    let mut catalogues = Vec::new();
    for src in cats.sources(db) {
        match flatppl_infer::parse_catalogue(src) {
            Ok(cat) => catalogues.push(cat),
            Err(e) => diags.push(LspDiag {
                start: 0,
                end: 0,
                severity: crate::capabilities::DiagSeverity::Error,
                message: format!("catalogue parse error: {e}"),
            }),
        }
    }

    let bundle = import_bundle(db, file, fs);
    let infer_diags = flatppl_infer::infer_module_ext(
        &mut m,
        &bundle,
        &catalogues,
        flatppl_infer::Level::Normalization,
    );
    diags.extend(infer_diags.iter().map(|d| LspDiag::from_infer(d, &m)));

    Analyzed::new(db, Some(ArcModule::new(m)), diags)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{Catalogues, Database, FileSet, SourceFile};

    #[test]
    fn parse_returns_module_for_valid_source() {
        let db = Database::default();
        let f = SourceFile::new(&db, "m.flatppl".to_string(), "x = add(1, 2)".to_string());
        let parsed = parse(&db, f);
        assert!(parsed.module(&db).is_some());
        assert!(parsed.diagnostics(&db).is_empty());
    }

    #[test]
    fn parse_reports_error_for_invalid_source() {
        let db = Database::default();
        let f = SourceFile::new(&db, "bad.flatppl".to_string(), "x = (((".to_string());
        let parsed = parse(&db, f);
        assert!(parsed.module(&db).is_none());
        assert!(!parsed.diagnostics(&db).is_empty());
    }

    // ── analyze tests ─────────────────────────────────────────────────────────

    use flatppl_core::Type;

    /// The inferred type of the RHS of the binding named `name` in `module`.
    fn rhs_type<'db>(module: &'db Module, name: &str) -> Option<&'db Type> {
        let (_, b) = module
            .bindings()
            .find(|(_, b)| module.resolve(b.name) == name)?;
        module.type_of(b.rhs)
    }

    fn empty_fs(db: &Database) -> FileSet {
        FileSet::new(db, Vec::new())
    }

    fn empty_cats(db: &Database) -> Catalogues {
        Catalogues::new(db, Vec::new())
    }

    #[test]
    fn analyze_infers_types_single_file() {
        let db = Database::default();
        let f = SourceFile::new(&db, "m.flatppl".to_string(), "x = add(1, 2)".to_string());
        let fs = empty_fs(&db);
        let cats = empty_cats(&db);

        let analyzed = analyze(&db, f, fs, cats);
        assert!(
            !analyzed
                .diagnostics(&db)
                .iter()
                .any(|d| d.severity == crate::capabilities::DiagSeverity::Error),
            "expected no error diagnostics; got {:?}",
            analyzed.diagnostics(&db)
        );
        let module = analyzed.module(&db).expect("module present on success");
        let ty = rhs_type(module, "x").expect("x has an inferred rhs type");
        assert!(
            matches!(ty, Type::Scalar(_)),
            "x's rhs should be a scalar type; got {ty:?}"
        );
    }

    #[test]
    fn analyze_resolves_external_catalogue_std_module() {
        let db = Database::default();
        let cat_src = r#"Catalogue(base: [], modules: [Module(name:"myext",version:"0.1",bindings:[Binding(name:"MyDist", sig: Distribution(domain: Scalar(Real), support: Reals, mass: Normalized))])])"#;
        let f = SourceFile::new(
            &db,
            "m.flatppl".to_string(),
            "e = standard_module(\"myext\",\"0.1\")\nx = e.MyDist(0.0)".to_string(),
        );
        let fs = empty_fs(&db);
        let cats = Catalogues::new(&db, vec![cat_src.to_string()]);

        let analyzed = analyze(&db, f, fs, cats);
        assert!(
            !analyzed
                .diagnostics(&db)
                .iter()
                .any(|d| d.message.contains("not found")),
            "no `not found` diag for a resolved std-module ref; got {:?}",
            analyzed.diagnostics(&db)
        );
        let module = analyzed.module(&db).expect("module present");
        let ty = rhs_type(module, "x").expect("x has an inferred rhs type");
        assert!(
            matches!(ty, Type::Measure { .. }),
            "x = e.MyDist(0.0) should infer a measure type; got {ty:?}"
        );
    }

    #[test]
    fn cross_file_load_module_resolves() {
        let db = Database::default();
        let helpers = SourceFile::new(
            &db,
            "helpers.flatppl".to_string(),
            "center = elementof(reals)\nshifted = add(center, 1.0)".to_string(),
        );
        let model = SourceFile::new(
            &db,
            "model.flatppl".to_string(),
            "h = load_module(\"helpers.flatppl\")\nv = add(h.shifted, 2.0)".to_string(),
        );
        let fs = FileSet::new(&db, vec![helpers, model]);
        let cats = empty_cats(&db);

        let analyzed = analyze(&db, model, fs, cats);
        let diags = analyzed.diagnostics(&db);
        assert!(
            !diags
                .iter()
                .any(|d| d.message.contains("not found") || d.message.contains("deferred")),
            "no `not found`/`deferred` diag for a resolved cross-file ref; got {diags:?}"
        );
        let module = analyzed.module(&db).expect("module present");
        let ty = rhs_type(module, "v").expect("v has an inferred rhs type");
        assert_eq!(
            ty,
            &Type::Scalar(flatppl_core::ScalarType::Real),
            "v = add(h.shifted, 2.0) should resolve through the bundle to Scalar(Real); got {ty:?}"
        );
    }

    /// Transitive closure: a two-level import chain model → helpers → utils.
    /// `helpers` re-exports `utils`'s `base` via its own `load_module`; `model`
    /// consumes `helpers`'s re-export. `import_bundle(model)` must include `utils`
    /// (a transitive, not direct, dependency) so `infer` — which resolves
    /// `helpers`'s `load_module("utils.flatppl")` ref against the *same* bundle —
    /// finds it. With a direct-only bundle this fails (utils absent → unresolved).
    #[test]
    fn transitive_load_module_resolves() {
        let db = Database::default();
        let utils = SourceFile::new(
            &db,
            "utils.flatppl".to_string(),
            "seed = add(elementof(reals), 1.0)".to_string(),
        );
        let helpers = SourceFile::new(
            &db,
            "helpers.flatppl".to_string(),
            "u = load_module(\"utils.flatppl\")\nreexport = add(u.seed, 1.0)".to_string(),
        );
        let model = SourceFile::new(
            &db,
            "model.flatppl".to_string(),
            "h = load_module(\"helpers.flatppl\")\nv = add(h.reexport, 2.0)".to_string(),
        );
        let fs = FileSet::new(&db, vec![utils, helpers, model]);
        let cats = empty_cats(&db);

        let analyzed = analyze(&db, model, fs, cats);
        let diags = analyzed.diagnostics(&db);
        assert!(
            !diags
                .iter()
                .any(|d| d.message.contains("not found") || d.message.contains("deferred")),
            "no `not found`/`deferred` diag for a transitively-resolved ref; got {diags:?}"
        );
        let module = analyzed.module(&db).expect("module present");
        let ty = rhs_type(module, "v").expect("v has an inferred rhs type");
        assert_eq!(
            ty,
            &Type::Scalar(flatppl_core::ScalarType::Real),
            "v should resolve transitively through helpers→utils to Scalar(Real); got {ty:?}"
        );
    }

    /// The cross-file salsa-edge guard: editing a dependency's text must
    /// invalidate (and recompute) the importer's analysis. We assert the
    /// observable type change — after `helpers.shifted` becomes complex-valued,
    /// the importer's `v` type changes accordingly. If the edge were not
    /// recorded, salsa would hand back the stale `Analyzed` and the type would
    /// still read `Real`.
    #[test]
    fn incrementality_dep_edit_reanalyzes_importer() {
        let mut db = Database::default();
        let helpers = SourceFile::new(
            &db,
            "helpers.flatppl".to_string(),
            "center = elementof(reals)\nshifted = add(center, 1.0)".to_string(),
        );
        let model = SourceFile::new(
            &db,
            "model.flatppl".to_string(),
            "h = load_module(\"helpers.flatppl\")\nv = add(h.shifted, 2.0)".to_string(),
        );
        let fs = FileSet::new(&db, vec![helpers, model]);
        let cats = empty_cats(&db);

        let analyzed = analyze(&db, model, fs, cats);
        let module = analyzed.module(&db).expect("module present");
        let ty = rhs_type(module, "v").expect("v has a type");
        assert_eq!(
            ty,
            &Type::Scalar(flatppl_core::ScalarType::Real),
            "baseline: v is Scalar(Real); got {ty:?}"
        );

        // Edit the dependency: make `shifted` complex-valued so `v` becomes
        // complex too. This mutates only the helpers input.
        use salsa::Setter;
        helpers
            .set_text(&mut db)
            .to("center = elementof(reals)\nshifted = add(center, im)".to_string());

        let analyzed2 = analyze(&db, model, fs, cats);
        let module2 = analyzed2
            .module(&db)
            .expect("module present after dep edit");
        let ty2 = rhs_type(module2, "v").expect("v has a type after dep edit");
        assert_eq!(
            ty2,
            &Type::Scalar(flatppl_core::ScalarType::Complex),
            "editing the dependency must re-analyze the importer: v should now be \
             Scalar(Complex); got {ty2:?}. If still Real, the cross-file salsa edge \
             was not recorded."
        );
    }
}
