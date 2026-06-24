//! Derived salsa queries over the source inputs.

use crate::capabilities::LspDiag;
use crate::db::{Catalogues, FileSet, SourceFile};
use crate::line_index::LineIndex;
use flatppl_core::{CallHead, Idx, Module, Node, NodeId, Scalar};
use flatppl_fileaccess::Location;
use std::sync::Arc;

// ── ArcCatalogues wrapper ────────────────────────────────────────────────────
//
// `flatppl_infer::Catalogue` derives only `Clone + Debug` — not `PartialEq`,
// `Eq`, `Hash`, or `salsa::Update` — so `Arc<Vec<Catalogue>>` satisfies none
// of the bounds salsa's `#[salsa::tracked]` function return type requires
// (`Eq` for backdating, `Update` for the update dispatch).
//
// `ArcCatalogues` wraps `Arc<Vec<Catalogue>>` and provides pointer-identity
// `PartialEq`/`Eq` (two separately-created arcs are never pointer-equal, so
// salsa will always re-propagate; over-recomputes rather than under-computes —
// the same conservative policy used by `ArcModule`). The `Update` impl
// likewise falls back to pointer identity: if the arc pointer changed the
// value definitely changed, so we always overwrite and return `true`.
//
// `errors` carries per-source parse diagnostics produced at catalogue-parse time
// so `analyze` does not re-parse the sources a second time just to emit them.
#[derive(Clone, Debug)]
pub struct ArcCatalogues {
    cats: Arc<Vec<flatppl_infer::Catalogue>>,
    errors: Arc<Vec<LspDiag>>,
}

impl ArcCatalogues {
    /// Return a reference to the successfully-parsed catalogues, for passing to
    /// `infer_module_ext` and the completion builder.
    pub fn as_slice(&self) -> &[flatppl_infer::Catalogue] {
        &self.cats
    }

    /// Return the parse-error diagnostics for sources that failed to parse.
    pub fn errors(&self) -> &[LspDiag] {
        &self.errors
    }
}

// Pointer-identity on `cats` only — `errors` is derived from the same parse
// pass and changes exactly when `cats` changes.
impl PartialEq for ArcCatalogues {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.cats, &other.cats)
    }
}

impl Eq for ArcCatalogues {}

impl std::hash::Hash for ArcCatalogues {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Arc::as_ptr(&self.cats), state);
    }
}

// SAFETY: pointer-identity equality is conservative (may over-recompute but
// never under-computes). `maybe_update` always overwrites `old_pointer` with
// `new_value`; returning `true` tells salsa the value changed, triggering
// downstream recomputation. This is sound because we never suppress a genuine
// change.
unsafe impl salsa::Update for ArcCatalogues {
    unsafe fn maybe_update(old_pointer: *mut Self, new_value: Self) -> bool {
        let old: &mut Self = unsafe { &mut *old_pointer };
        if Arc::ptr_eq(&old.cats, &new_value.cats) {
            return false;
        }
        *old = new_value;
        true
    }
}

// ── parsed_catalogues tracked query ─────────────────────────────────────────

/// Parse the host-supplied external RON catalogues once per `Catalogues`
/// revision. Successful sources are collected into the `cats` vec; failing
/// sources produce `LspDiag` entries in `errors`. Both are stored in the
/// returned `ArcCatalogues` so `analyze` can consume the errors without
/// re-parsing.
#[salsa::tracked]
pub fn parsed_catalogues(db: &dyn salsa::Database, cats: Catalogues) -> ArcCatalogues {
    #[cfg(test)]
    PARSED_CATALOGUES_RUNS.with(|c| c.set(c.get() + 1));
    let mut parsed = Vec::new();
    let mut errors = Vec::new();
    for s in cats.sources(db) {
        match flatppl_infer::parse_catalogue(s) {
            Ok(c) => parsed.push(c),
            Err(e) => errors.push(LspDiag {
                start: 0,
                end: 0,
                severity: crate::capabilities::DiagSeverity::Error,
                message: format!("catalogue parse error: {e}"),
            }),
        }
    }
    ArcCatalogues {
        cats: Arc::new(parsed),
        errors: Arc::new(errors),
    }
}

// Per-thread execution counter for `parsed_catalogues`. Thread-local so
// concurrent tests do not interfere with each other's measurements.
#[cfg(test)]
thread_local! {
    pub static PARSED_CATALOGUES_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

// ── SpanIndex: offset→node lookup (salsa-memoized) ──────────────────────────

/// Sorted span table for offset→node lookup. Entries are `(start, end, node)`
/// sorted by `start` ascending, then `end` descending, so a binary search to
/// the last `start <= offset` gives a short candidate prefix.
#[derive(Clone, Debug, PartialEq, Eq, salsa::Update)]
pub struct SpanIndex {
    spans: Vec<(u32, u32, NodeId)>,
}

// Test-only execution counter for `node_span_index` (proves salsa memoizes it).
// Thread-local so concurrent tests do not interfere with each other's measurements.
#[cfg(test)]
thread_local! {
    pub static SPAN_INDEX_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Build the offset→node span index for the analyzed module, once per revision.
#[salsa::tracked]
pub fn node_span_index(
    db: &dyn salsa::Database,
    file: SourceFile,
    fs: FileSet,
    cats: Catalogues,
) -> SpanIndex {
    #[cfg(test)]
    SPAN_INDEX_RUNS.with(|c| c.set(c.get() + 1));
    let analyzed = analyze(db, file, fs, cats);
    let mut spans: Vec<(u32, u32, NodeId)> = Vec::new();
    if let Some(module) = analyzed.module(db) {
        for i in 0..module.node_count() {
            let id = NodeId::from_usize(i);
            if let Some(span) = module.span_of(id) {
                spans.push((span.start, span.end, id));
            }
        }
    }
    spans.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    SpanIndex { spans }
}

/// The innermost (smallest-width) node whose span contains `offset`, ties broken
/// to the LOWEST NodeId — byte-for-byte equivalent to `Module::node_at_offset`'s
/// linear scan (which keeps the first minimal-width hit in NodeId order).
pub fn node_at_offset_indexed(index: &SpanIndex, offset: u32) -> Option<NodeId> {
    let mut best: Option<(u32 /*width*/, NodeId)> = None;
    // All spans with start <= offset live in the prefix up to this point.
    let upper = index
        .spans
        .partition_point(|&(start, _, _)| start <= offset);
    for &(start, end, id) in &index.spans[..upper] {
        debug_assert!(start <= offset);
        if offset < end {
            let width = end - start;
            // Strict `<` keeps the first (lowest-NodeId) hit on a width tie,
            // matching Module::node_at_offset's `is_none_or(|(_, w)| width < w)`.
            let better = match best {
                None => true,
                Some((w, prev_id)) => width < w || (width == w && id < prev_id),
            };
            if better {
                best = Some((width, id));
            }
        }
    }
    best.map(|(_, id)| id)
}

// ── LineIndex tracked query ──────────────────────────────────────────────────

/// The UTF-8↔UTF-16 line index for a file, computed once per revision and
/// shared across every capability/handler (was rebuilt per request).
#[salsa::tracked]
pub fn line_index(db: &dyn salsa::Database, file: SourceFile) -> LineIndex {
    #[cfg(test)]
    LINE_INDEX_RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    LineIndex::new(file.text(db))
}

/// Test-only execution counter (proves the query is memoized per revision).
#[cfg(test)]
pub static LINE_INDEX_RUNS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

// ── Salsa field-compatibility wrapper ───────────────────────────────────────
//
// salsa tracked-struct fields (even `#[returns(ref)]` ones) must satisfy
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

// ── ImportBundle wrapper ─────────────────────────────────────────────────────
//
// `flatppl_infer::ModuleBundle` holds `Arc<Module>` deps but is neither `Eq`
// nor `salsa::Update` (its `Module`s have f64 literals, so no structural `Eq`).
// `import_bundle` is a `#[salsa::tracked]` query, whose return type must satisfy
// `Eq + Update + Debug + Send + Sync`. `ImportBundle` wraps `Arc<ModuleBundle>`
// and supplies the same pointer-identity `Eq`/`Hash`/`Update` policy as
// `ArcModule`/`ArcCatalogues`: two separately-assembled bundles are never
// pointer-equal, so salsa always re-propagates on a genuine recompute
// (over-recomputes rather than under-computes — conservatively correct).
//
// Within a single revision the memoized query returns the *same* `Arc`, so the
// per-dependency `Arc<Module>` is shared across every `analyze` of that revision
// (verified by `dependency_module_is_shared_not_recloned`).
//
// `dep_files` records the RESOLVED `SourceFile` handles for every transitive
// dependency. `affected_files` (server.rs) matches importers by `SourceFile`
// identity rather than directive-literal path strings, so a relative import
// `"../helpers.flatppl"` whose directive differs lexically from the stored path
// still identifies the correct importer.
#[derive(Clone, Debug)]
pub struct ImportBundle {
    bundle: Arc<flatppl_infer::ModuleBundle>,
    dep_files: Arc<std::collections::HashSet<SourceFile>>,
}

impl ImportBundle {
    fn new(
        bundle: flatppl_infer::ModuleBundle,
        dep_files: std::collections::HashSet<SourceFile>,
    ) -> Self {
        ImportBundle {
            bundle: Arc::new(bundle),
            dep_files: Arc::new(dep_files),
        }
    }

    /// Borrow the assembled bundle, for passing to `infer_module_ext`.
    pub fn as_bundle(&self) -> &flatppl_infer::ModuleBundle {
        &self.bundle
    }

    /// The shared `Arc<Module>` for the dependency keyed by `path`, if present.
    /// Cloning the returned `Arc` is a refcount bump, not a deep clone.
    pub fn module_for(&self, path: &str) -> Option<Arc<Module>> {
        self.bundle.get_arc(path).cloned()
    }

    /// Return `true` when `dep` is a (direct or transitive) dependency of this
    /// file, matched by `SourceFile` identity (salsa input id), not by path
    /// string. This is the correct predicate for `affected_files`: a relative
    /// import `"../helpers.flatppl"` resolves to the same `SourceFile` id as
    /// the absolute stored path, so the match is exact regardless of the literal
    /// directive text.
    pub fn imports(&self, dep: SourceFile) -> bool {
        self.dep_files.contains(&dep)
    }
}

impl PartialEq for ImportBundle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.bundle, &other.bundle)
    }
}

impl Eq for ImportBundle {}

impl std::hash::Hash for ImportBundle {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Arc::as_ptr(&self.bundle), state);
    }
}

// SAFETY: pointer-identity equality is conservative (may over-recompute but
// never under-computes). `maybe_update` overwrites `old_pointer` with
// `new_value` on a pointer difference and returns `true`, telling salsa the
// value changed. Sound because we never suppress a genuine change.
unsafe impl salsa::Update for ImportBundle {
    unsafe fn maybe_update(old_pointer: *mut Self, new_value: Self) -> bool {
        let old: &mut Self = unsafe { &mut *old_pointer };
        if Arc::ptr_eq(&old.bundle, &new_value.bundle) {
            return false;
        }
        *old = new_value;
        true
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
    #[returns(ref)]
    module_arc: Option<ArcModule>,
    #[returns(ref)]
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

/// Resolve `directive_path` (from a `load_module` directive in `importer`) to a
/// workspace [`SourceFile`] in `fs`.
///
/// Resolution goes through [`flatppl_fileaccess::Location`] — the ecosystem's one
/// §04 path/URL resolver — so a local importer's relative dep joins against its
/// directory, while an `http`/`https` directive (or a URL importer's relative
/// dep) resolves as a URL, all uniformly. The LSP only *resolves* here; it never
/// fetches (URL-dep content arrives as client-fed `SourceFile`s — see
/// `server.rs`).
///
/// Two interpretations are tried, preferring the importer-relative `join`
/// (deterministic in ambiguous workspaces where both `a/b/x.flatppl` and
/// `b/x.flatppl` exist) over taking the directive as-is. Each candidate file's
/// stored path is canonicalized the same way ([`Location::normalized`]), so the
/// comparison is symmetric. Returns `None` when nothing matches: the common case
/// for `standard_module` names, and for an as-yet-unfed URL dep — a soft "source
/// not available", like a missing local dep.
pub(crate) fn resolve_path(
    db: &dyn salsa::Database,
    importer: SourceFile,
    directive_path: &str,
    fs: FileSet,
) -> Option<SourceFile> {
    let joined = Location::parse(&importer.path(db)).join(directive_path);
    let direct = Location::parse(directive_path).normalized();
    let find = |target: &Location| {
        fs.files(db)
            .iter()
            .copied()
            .find(|f| &Location::parse(&f.path(db)).normalized() == target)
    };
    find(&joined).or_else(|| find(&direct))
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
/// This is a `#[salsa::tracked]` query so the transitive bundle is assembled
/// **once per revision** rather than rebuilt on every `analyze` call (per
/// keystroke, per open doc). Each dependency `Module` is held behind an
/// `Arc<Module>` (see [`ImportBundle`]); the single `Arc::new(parsed.clone())`
/// per dependency happens inside this memoized body, so repeat `analyze`s reuse
/// the same shared `Arc`s instead of deep-cloning every dependency anew.
///
/// The query reads `parse(db, dep)` for every transitive dependency, so the
/// cross-file salsa edges are recorded in the query's own read set — and, via
/// `analyze` calling `import_bundle`, in `analyze`'s read set too. Editing any
/// dependency's text therefore invalidates this query and the importer's
/// `analyze`.
#[salsa::tracked]
pub fn import_bundle(db: &dyn salsa::Database, file: SourceFile, fs: FileSet) -> ImportBundle {
    #[cfg(test)]
    IMPORT_BUNDLE_RUNS.with(|c| c.set(c.get() + 1));

    let mut bundle = flatppl_infer::ModuleBundle::new();
    // Tracks the RESOLVED SourceFile for every transitive dependency. Used by
    // `affected_files` (server.rs) to match importers by SourceFile identity
    // instead of directive-literal path strings — so a relative import such as
    // `"../helpers.flatppl"` correctly identifies the same dependency as its
    // absolute stored path.
    let mut dep_files: std::collections::HashSet<SourceFile> = std::collections::HashSet::new();
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
            // Record the resolved SourceFile identity so affected_files can
            // match by id rather than by the directive's literal path string.
            dep_files.insert(dep_file);
            if let Some(dep_mod) = parse(db, dep_file).module(db) {
                // Key by the directive-literal path so infer's `(%ref alias X)`
                // lookup (keyed by directive path) matches. The single deep
                // clone of the parsed dependency lives here, behind an `Arc`,
                // computed once per revision rather than per `analyze` call.
                bundle.insert(path, Arc::new(dep_mod.clone()));
            }
            // Enqueue once; the visited set is keyed on the resolved file so two
            // directives that resolve to the same file (or an import cycle) do
            // not re-walk it.
            if visited.insert(dep_file) {
                queue.push_back(dep_file);
            }
        }
    }
    ImportBundle::new(bundle, dep_files)
}

// Per-thread execution counter for `import_bundle` (proves the query is
// memoized once per revision, not recomputed per `analyze` call). Thread-local
// so concurrent tests do not interfere with each other's measurements (mirrors
// `PARSED_CATALOGUES_RUNS`).
#[cfg(test)]
thread_local! {
    pub static IMPORT_BUNDLE_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

// Per-thread execution counter for `analyze`. Thread-local so concurrent tests
// do not interfere with each other's measurements (mirrors `IMPORT_BUNDLE_RUNS`).
// Reset with `ANALYZE_RUNS.with(|c| c.set(0))` before measuring; read with
// `ANALYZE_RUNS.with(|c| c.get())`.
#[cfg(test)]
thread_local! {
    pub static ANALYZE_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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
    #[returns(ref)]
    module_arc: Option<ArcModule>,
    #[returns(ref)]
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
/// [`Level::Shape`](flatppl_infer::Level::Shape).
///
/// `Level::Shape` is the maximal (additive) level: it does everything
/// `Normalization` does plus demand-driven shape resolution, so fixed-phase
/// integer shape expressions fold to concrete dims. This is what editor
/// surfaces want — an `iid(M, lengthof(data))` binding shows `…[5]` in its
/// inlay hint / hover instead of `…[?]`. Resolution is lazy (consulted only at
/// shape positions, depth-capped), so it adds no blanket cost.
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
    #[cfg(test)]
    ANALYZE_RUNS.with(|c| c.set(c.get() + 1));
    let parsed = parse(db, file);
    let Some(module) = parsed.module(db) else {
        return Analyzed::new(db, None, parsed.diagnostics(db).clone());
    };
    let mut m = module.clone();

    // Obtain the external catalogues and their parse errors via the tracked
    // `parsed_catalogues` query (parsed once per `Catalogues` revision, not per
    // analyze call). The query carries both the parsed catalogues and any
    // per-source diagnostics, so no second parse pass is needed here.
    let mut diags = parsed.diagnostics(db).clone();
    let catalogues = parsed_catalogues(db, cats);
    diags.extend(catalogues.errors().iter().cloned());

    let bundle = import_bundle(db, file, fs);
    let infer_diags = flatppl_infer::infer_module_ext(
        &mut m,
        bundle.as_bundle(),
        catalogues.as_slice(),
        flatppl_infer::Level::Shape,
    );
    diags.extend(infer_diags.iter().map(|d| LspDiag::from_infer(d, &m)));

    Analyzed::new(db, Some(ArcModule::new(m)), diags)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{Catalogues, Database, FileSet, SourceFile};

    #[test]
    fn line_index_is_memoized_per_revision() {
        use std::sync::atomic::Ordering::Relaxed;
        let db = Database::default();
        let f = SourceFile::new(&db, "m.flatppl".to_string(), "a\nbb\nc".to_string());
        LINE_INDEX_RUNS.store(0, Relaxed);
        let _ = line_index(&db, f);
        let _ = line_index(&db, f);
        let _ = line_index(&db, f); // 3 calls, same revision
        assert_eq!(
            LINE_INDEX_RUNS.load(Relaxed),
            1,
            "computed once per revision, not per call"
        );
    }

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

    /// `resolve_path` resolves URL `load_module` deps through `fileaccess::Location`:
    /// (1) a local importer with an absolute-URL directive → the URL SourceFile;
    /// (2) a URL importer with a *relative* directive → the §04 URL-relative join
    /// (the load-bearing case for client-fed URL modules whose own deps are
    /// relative); (3) a `..` relative directive normalizes against the URL.
    /// Locks resolution through `fileaccess::Location` (the §04 resolver).
    #[test]
    fn resolve_path_resolves_url_deps() {
        let db = Database::default();
        let common = SourceFile::new(
            &db,
            "https://h.example/examples/common.flatppl".to_string(),
            "x = elementof(reals)".to_string(),
        );
        let priors = SourceFile::new(
            &db,
            "https://h.example/examples/priors.flatppl".to_string(),
            "p = elementof(reals)".to_string(),
        );
        let shared = SourceFile::new(
            &db,
            "https://h.example/shared/util.flatppl".to_string(),
            "u = elementof(reals)".to_string(),
        );
        let model = SourceFile::new(
            &db,
            "/proj/model.flatppl".to_string(),
            "c = load_module(\"https://h.example/examples/common.flatppl\")".to_string(),
        );
        let fs = FileSet::new(&db, vec![common, priors, shared, model]);

        // (1) local importer + absolute-URL directive.
        assert_eq!(
            resolve_path(&db, model, "https://h.example/examples/common.flatppl", fs),
            Some(common),
            "absolute-URL directive resolves to the URL SourceFile"
        );
        // (2) URL importer + relative directive → URL-relative join.
        assert_eq!(
            resolve_path(&db, common, "priors.flatppl", fs),
            Some(priors),
            "relative directive resolves against the importer URL's directory"
        );
        // (3) URL importer + `..` directive normalizes.
        assert_eq!(
            resolve_path(&db, common, "../shared/util.flatppl", fs),
            Some(shared),
            "`..` in a URL-relative directive normalizes across path segments"
        );
    }

    /// End-to-end: a local model loads a URL module which itself has a *relative*
    /// `load_module` dep; both URL sources are present in the FileSet (as the
    /// editor client would feed them). `import_bundle` must include the
    /// transitively-resolved relative URL dep.
    #[test]
    fn import_bundle_walks_url_relative_deps() {
        let db = Database::default();
        let priors = SourceFile::new(
            &db,
            "https://h.example/examples/priors.flatppl".to_string(),
            "theta = elementof(reals)".to_string(),
        );
        let common = SourceFile::new(
            &db,
            "https://h.example/examples/common.flatppl".to_string(),
            "pr = load_module(\"priors.flatppl\")\nm = add(pr.theta, 1.0)".to_string(),
        );
        let model = SourceFile::new(
            &db,
            "/proj/model.flatppl".to_string(),
            "c = load_module(\"https://h.example/examples/common.flatppl\")\nv = add(c.m, 2.0)"
                .to_string(),
        );
        let fs = FileSet::new(&db, vec![priors, common, model]);

        let bundle = import_bundle(&db, model, fs);
        // The transitive URL-relative dep is reached and recorded.
        assert!(
            bundle.imports(priors),
            "import_bundle(model) must include the URL-relative transitive dep `priors`"
        );
        assert!(
            bundle.imports(common),
            "and the directly-loaded URL `common`"
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

    #[test]
    fn external_catalogues_parsed_once_per_revision() {
        let db = Database::default();
        // A trivial valid catalogue source.
        let cats = Catalogues::new(&db, vec!["Catalogue(base:[],modules:[])".to_string()]);
        let f = SourceFile::new(&db, "m.flatppl".to_string(), "x = 1".to_string());
        let fs = FileSet::new(&db, vec![f]);
        // Thread-local counter: reset to 0, exercise analyze + a direct
        // parsed_catalogues call (3 calls total), assert the body ran exactly once.
        PARSED_CATALOGUES_RUNS.with(|c| c.set(0));
        let _ = analyze(&db, f, fs, cats);
        let _ = analyze(&db, f, fs, cats);
        let _ = parsed_catalogues(&db, cats);
        let runs = PARSED_CATALOGUES_RUNS.with(|c| c.get());
        assert_eq!(
            runs, 1,
            "external catalogues parsed once per revision, not per analyze/completion"
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

    /// Editing the `Catalogues` input must cause `analyze` to recompute for a
    /// file that reads from it — proving the salsa edge `analyze` → `Catalogues`
    /// is recorded. The proof is that `ANALYZE_RUNS` increments after a catalogue
    /// edit even though the source file itself was not changed.
    ///
    /// Approach: warm the cache with a catalogue that contains a module binding,
    /// reset the counter, swap in a new `Catalogues` value (different `Arc`, so
    /// pointer-identity equality detects the change), call `analyze` again, and
    /// assert the body re-ran (counter > 0).
    #[test]
    fn catalogue_edit_reanalyzes_dependents() {
        let mut db = Database::default();
        // A catalogue exposing one module with one distribution binding.
        let cat_v1 = r#"Catalogue(base:[],modules:[Module(name:"myext",version:"0.1",bindings:[Binding(name:"MyDist",sig:Distribution(domain:Scalar(Real),support:Reals,mass:Normalized))])])"#;
        let f = SourceFile::new(
            &db,
            "m.flatppl".to_string(),
            "e = standard_module(\"myext\",\"0.1\")\nx = e.MyDist(0.0)".to_string(),
        );
        let fs = FileSet::new(&db, vec![f]);
        let cats = Catalogues::new(&db, vec![cat_v1.to_string()]);

        // Warm: first analysis populates salsa's memoization table.
        let _ = analyze(&db, f, fs, cats);

        // Reset the counter after the warm-up run.
        ANALYZE_RUNS.with(|c| c.set(0));

        // Edit the Catalogues input: a new Vec is a new Arc (pointer-identity
        // change), so ArcCatalogues::maybe_update returns `true` and salsa marks
        // all dependents stale. The source file `f` did not change.
        use salsa::Setter;
        let cat_v2 = r#"Catalogue(base:[],modules:[])"#; // module removed
        cats.set_sources(&mut db).to(vec![cat_v2.to_string()]);

        let _ = analyze(&db, f, fs, cats);
        let runs = ANALYZE_RUNS.with(|c| c.get());
        assert!(
            runs > 0,
            "editing Catalogues must re-analyze dependents (ANALYZE_RUNS={runs}); \
             if 0, the salsa edge analyze → Catalogues is not recorded"
        );
    }

    // ── import_bundle memoization tests ───────────────────────────────────────

    /// A two-file workspace: B (`helpers`) defines a binding; A (`model`) does
    /// `load_module` of B and uses it. Returns `(db, a, b, fs, cats)`.
    fn import_workspace() -> (Database, SourceFile, SourceFile, FileSet, Catalogues) {
        let db = Database::default();
        let b = SourceFile::new(
            &db,
            "helpers.flatppl".to_string(),
            "center = elementof(reals)\nshifted = add(center, 1.0)".to_string(),
        );
        let a = SourceFile::new(
            &db,
            "model.flatppl".to_string(),
            "h = load_module(\"helpers.flatppl\")\nv = add(h.shifted, 2.0)".to_string(),
        );
        let fs = FileSet::new(&db, vec![b, a]);
        let cats = empty_cats(&db);
        (db, a, b, fs, cats)
    }

    #[test]
    fn import_bundle_memoized_per_revision() {
        let (db, a, _b, fs, cats) = import_workspace();
        IMPORT_BUNDLE_RUNS.with(|c| c.set(0));
        let _ = import_bundle(&db, a, fs);
        let _ = import_bundle(&db, a, fs);
        let _ = analyze(&db, a, fs, cats); // also reads the bundle
        assert_eq!(
            IMPORT_BUNDLE_RUNS.with(|c| c.get()),
            1,
            "import_bundle computed once per revision, not per analyze"
        );
    }

    /// `ImportBundle::imports` matches by SourceFile identity, not by
    /// directive-literal path string.
    ///
    /// A builds a `load_module` of B using B's stored path as the directive
    /// (the simplest case where the directive equals the stored path). We verify:
    ///   1. `import_bundle(A).imports(B)` returns `true`.
    ///   2. `import_bundle(A).imports(A)` (self) returns `false` — the self node
    ///      is the root and NOT in dep_files.
    ///   3. `import_bundle(A).imports(C)` (independent file) returns `false`.
    ///
    /// These assertions prove the identity-based match is correct and that the
    /// set does not accidentally include unrelated files.
    #[test]
    fn import_bundle_imports_matches_by_sourcefile_identity() {
        let db = Database::default();
        let b = SourceFile::new(
            &db,
            "helpers.flatppl".to_string(),
            "center = elementof(reals)\n".to_string(),
        );
        let a = SourceFile::new(
            &db,
            "model.flatppl".to_string(),
            "h = load_module(\"helpers.flatppl\")\nv = add(h.center, 1.0)\n".to_string(),
        );
        let c = SourceFile::new(
            &db,
            "independent.flatppl".to_string(),
            "x = add(1, 2)\n".to_string(),
        );
        let fs = FileSet::new(&db, vec![a, b, c]);

        let bundle_a = import_bundle(&db, a, fs);

        assert!(
            bundle_a.imports(b),
            "import_bundle(A).imports(B) must be true: A has load_module(\"helpers.flatppl\") \
             which resolves to B by SourceFile identity"
        );
        assert!(
            !bundle_a.imports(a),
            "import_bundle(A).imports(A) must be false: A is the root, not a dependency"
        );
        assert!(
            !bundle_a.imports(c),
            "import_bundle(A).imports(C) must be false: C is independent"
        );
    }

    /// `affected_files` with SourceFile-identity matching correctly includes
    /// the importer when the dependency is changed.
    ///
    /// This mirrors the server-side `affected_files_excludes_non_importers` test
    /// at the queries layer: using `imports(changed)` rather than
    /// `module_for(&changed.path(db)).is_some()`, the importer is found even
    /// when the directive literal does not byte-match the stored path.
    #[test]
    fn import_bundle_dep_files_excludes_self_and_non_deps() {
        let db = Database::default();
        let helpers = SourceFile::new(
            &db,
            "helpers.flatppl".to_string(),
            "leaf = elementof(reals)\n".to_string(),
        );
        let model = SourceFile::new(
            &db,
            "model.flatppl".to_string(),
            "h = load_module(\"helpers.flatppl\")\nv = add(h.leaf, 1.0)\n".to_string(),
        );
        let other = SourceFile::new(
            &db,
            "other.flatppl".to_string(),
            "z = add(2, 3)\n".to_string(),
        );
        let fs = FileSet::new(&db, vec![helpers, model, other]);

        // bundle for `model`: deps = {helpers}
        let bundle_model = import_bundle(&db, model, fs);
        assert!(bundle_model.imports(helpers), "model imports helpers");
        assert!(!bundle_model.imports(model), "model is the root, not a dep");
        assert!(!bundle_model.imports(other), "model does not import other");

        // bundle for `helpers` (a leaf): no deps
        let bundle_helpers = import_bundle(&db, helpers, fs);
        assert!(
            !bundle_helpers.imports(model),
            "helpers does not import model"
        );
        assert!(
            !bundle_helpers.imports(other),
            "helpers does not import other"
        );
    }

    #[test]
    fn dependency_module_is_shared_not_recloned() {
        // The Arc<Module> for B inside two import_bundle results in the same
        // revision must be pointer-equal (shared, not deep-cloned each call).
        let (db, a, _b, fs, _cats) = import_workspace();
        let b1 = import_bundle(&db, a, fs)
            .module_for("helpers.flatppl")
            .expect("dep B present in bundle");
        let b2 = import_bundle(&db, a, fs)
            .module_for("helpers.flatppl")
            .expect("dep B present in bundle");
        assert!(
            std::sync::Arc::ptr_eq(&b1, &b2),
            "dep module shared across calls"
        );
    }

    // ── 3a: span index tests ──────────────────────────────────────────────────

    #[test]
    fn span_index_matches_linear_node_at_offset() {
        let db = Database::default();
        let src = "x = add(mul(2, 3), sqrt(4.0))\ny ~ Normal(x, 1.0)";
        let f = SourceFile::new(&db, "m.flatppl".to_string(), src.to_string());
        let fs = FileSet::new(&db, Vec::new());
        let cats = Catalogues::new(&db, Vec::new());
        let analyzed = analyze(&db, f, fs, cats);
        let module = analyzed.module(&db).expect("module");
        let index = node_span_index(&db, f, fs, cats);
        for off in 0..(src.len() as u32) {
            assert_eq!(
                node_at_offset_indexed(&index, off),
                module.node_at_offset(off),
                "mismatch at offset {off}"
            );
        }
    }

    #[test]
    fn span_index_memoized_per_revision() {
        let db = Database::default();
        let f = SourceFile::new(&db, "m.flatppl".to_string(), "x = 1".to_string());
        let fs = FileSet::new(&db, Vec::new());
        let cats = Catalogues::new(&db, Vec::new());
        SPAN_INDEX_RUNS.with(|c| c.set(0));
        let _ = node_span_index(&db, f, fs, cats);
        let _ = node_span_index(&db, f, fs, cats);
        assert_eq!(SPAN_INDEX_RUNS.with(|c| c.get()), 1);
    }

    // ── 3b: catalogue single-parse test ──────────────────────────────────────

    #[test]
    fn analyze_does_not_reparse_catalogues() {
        let db = Database::default();
        let f = SourceFile::new(&db, "m.flatppl".to_string(), "x = 1".to_string());
        let fs = FileSet::new(&db, Vec::new());
        let cats = Catalogues::new(&db, vec!["(((".to_string()]);
        PARSED_CATALOGUES_RUNS.with(|c| c.set(0));
        let analyzed = analyze(&db, f, fs, cats);
        // analyze must obtain catalogues + their errors solely via the memoized
        // query — exactly one parse pass.
        assert_eq!(PARSED_CATALOGUES_RUNS.with(|c| c.get()), 1);
        // Errors from the invalid catalogue source must surface in analyze's
        // diagnostics — proving they travel through the memoized query's
        // carried errors, not a re-parse loop in analyze.
        assert!(
            analyzed
                .diagnostics(&db)
                .iter()
                .any(|d| d.message.contains("catalogue parse error")),
            "catalogue parse errors must reach analyze diagnostics via the memoized query"
        );
    }
}
