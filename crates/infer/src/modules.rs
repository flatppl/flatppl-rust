//! Cross-module (`load_module`) resolution: the dependency bundle, the
//! inference session that spans it, substitution seeding, the per-import-site
//! memo, and cross-module cycle detection. Single-module inference lives in
//! `trace.rs`; everything that crosses a module boundary lives here.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use flatppl_core::{
    BindingId, CallHead, Module, NamedKind, Node, NodeId, Phase, Scalar, Symbol, Type, ValueSet,
};

use crate::Diagnostic;
use crate::catalogue::CatalogueSet;

/// Does `source` name an `http`/`https` URL (case-insensitive)? Used only to
/// phrase an unresolved-dependency diagnostic: a remote source absent from the
/// bundle hasn't been *fetched*, which is different from a local file that isn't
/// *found*. Mirrors the scheme check in `flatppl-fileaccess` — kept inline so
/// `flatppl-infer` stays dependency-free and wasm-targetable.
fn is_remote_source(source: &str) -> bool {
    let b = source.as_bytes();
    (b.len() >= 7 && source[..7].eq_ignore_ascii_case("http://"))
        || (b.len() >= 8 && source[..8].eq_ignore_ascii_case("https://"))
}

/// Parsed dependency modules, keyed by the `load_module` path string. Supplied
/// by the host (the engine does no file I/O).
///
/// Dependencies are held behind `Arc<Module>` so a host that assembles the same
/// bundle repeatedly (e.g. the LSP, once per keystroke) shares one parsed copy
/// rather than deep-cloning each dependency `Module` on every assembly. Inserts
/// and lookups move/borrow the `Arc`; the only deep clone of a dependency is the
/// per-import-site working copy `infer_dep` mutates, which is genuinely needed
/// (inference annotates it in place).
#[derive(Debug, Default, Clone)]
pub struct ModuleBundle {
    modules: HashMap<String, Arc<Module>>,
}

impl ModuleBundle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a parsed dependency (shared `Arc`) under its `load_module` path.
    pub fn insert(&mut self, path: impl Into<String>, module: Arc<Module>) {
        self.modules.insert(path.into(), module);
    }

    /// The dependency parsed for `path`, if present.
    pub fn get(&self, path: &str) -> Option<&Module> {
        self.modules.get(path).map(|a| a.as_ref())
    }

    /// The shared `Arc` for `path`, if present (refcount bump, no deep clone).
    pub fn get_arc(&self, path: &str) -> Option<&Arc<Module>> {
        self.modules.get(path)
    }
}

/// The outcome of resolving `(%ref alias X)` across module boundaries.
#[derive(Debug, Clone)]
pub(crate) struct Resolved {
    pub(crate) ty: Type,
    pub(crate) phase: Phase,
    pub(crate) vset: ValueSet,
    /// For a reified-callable binding (`%function` / `%kernel`), the inferred
    /// type of the callable's reified BODY in the dependency. Applying the
    /// callable (via `likelihoodof` or a user call) reads this body-result
    /// type exactly as the single-module machinery reads a local callable's
    /// body — except the body lives across the module/interner boundary, so it
    /// rides over here instead of being looked up by node. `None` for
    /// non-callable bindings.
    pub(crate) result: Option<Type>,
    /// For a §09 *standard-module* reference resolved against the built-in
    /// catalogue, the catalogue signature of the referenced binding (plus its
    /// honest-degrade note). The `ty`/`vset` above carry the *bare-reference*
    /// type (matching a bare base-distribution/function name); the actual
    /// measure/result type is lowered at the APPLICATION site, where the
    /// concrete argument types are known. `None` for cross-module
    /// (`load_module`) references — those resolve to a dependency binding, not
    /// a catalogue sig.
    pub(crate) catalogue: Option<CatalogueRef>,
}

/// A §09 standard-module binding resolved against the built-in catalogue: its
/// signature (cloned so the application site can lower it with concrete call
/// args) and its honest-degrade note.
#[derive(Debug, Clone)]
pub(crate) struct CatalogueRef {
    pub(crate) sig: crate::catalogue::Sig,
    pub(crate) degraded: Option<String>,
}

/// Parsed directive from a `load_module` / `standard_module` call.
struct LoadDirective {
    path: String,
    /// `true` when the head was `standard_module` (resolve against the built-in
    /// catalogue rather than the host bundle).
    standard: bool,
    /// The requested version (the second `standard_module` positional arg), if
    /// present. `None` for `load_module` or a malformed call.
    version: Option<String>,
    /// (dependency input-name, substitution value node in the importer).
    substitutions: Vec<(String, NodeId)>,
}

/// Spans a `ModuleBundle` for one `infer_module` run. Holds the per-import-site
/// dependency memo and the active-import stack. Interior mutability lets the
/// per-module `Inferencer` borrow `&InferSession` while recursing into a child
/// `Inferencer` over a cloned dependency.
pub(crate) struct InferSession<'b> {
    pub(crate) bundle: &'b ModuleBundle,
    /// Merged catalogue set (built-in + host-supplied external catalogues).
    /// `standard_module` resolution consults this instead of `builtin()` directly
    /// so that host-supplied external catalogues are visible.
    pub(crate) catalogues: CatalogueSet<'b>,
    /// (path, substitution-signature) -> the dependency's inferred (annotated) Module.
    memo: RefCell<HashMap<(String, String), Module>>,
    /// Active import paths on the current resolution chain (cycle detection).
    stack: RefCell<Vec<String>>,
    /// Diagnostics accumulated from dependency inference runs. The root
    /// `Inferencer::run` drains this into its own diagnostic list so that
    /// cycle errors and other dep-level errors reach the caller.
    dep_diags: RefCell<Vec<Diagnostic>>,
}

impl<'b> InferSession<'b> {
    pub(crate) fn new(bundle: &'b ModuleBundle) -> Self {
        InferSession {
            bundle,
            catalogues: CatalogueSet::builtin_only(),
            memo: RefCell::new(HashMap::new()),
            stack: RefCell::new(Vec::new()),
            dep_diags: RefCell::new(Vec::new()),
        }
    }

    /// Like `new`, but also wires in host-supplied external catalogues.
    /// The `CatalogueSet` holds a `&'b [Catalogue]` so the external slice must
    /// live at least as long as `'b` (the bundle).
    pub(crate) fn with_external_catalogues(
        bundle: &'b ModuleBundle,
        external: &'b [crate::catalogue::Catalogue],
    ) -> Self {
        InferSession {
            bundle,
            catalogues: CatalogueSet::with_external(external),
            memo: RefCell::new(HashMap::new()),
            stack: RefCell::new(Vec::new()),
            dep_diags: RefCell::new(Vec::new()),
        }
    }

    /// Drain all diagnostics accumulated from dependency inference runs and
    /// return them. Called once per `Inferencer::run` to propagate dep-level
    /// errors (cycle errors, child errors) up to the root caller.
    pub(crate) fn drain_dep_diags(&self) -> Vec<Diagnostic> {
        self.dep_diags.borrow_mut().drain(..).collect()
    }

    /// Append `diags` to the dependency-diagnostic accumulator. Called by
    /// `infer_dep` after each child inference walk.
    pub(crate) fn push_dep_diags(&self, diags: Vec<Diagnostic>) {
        self.dep_diags.borrow_mut().extend(diags);
    }

    /// Extract the `load_module` / `standard_module` directive from the binding
    /// whose LHS is `alias` in `importer`.
    fn load_directive(&self, importer: &Module, alias: Symbol) -> Result<LoadDirective, String> {
        let alias_name = importer.resolve(alias).to_string();
        let bid: BindingId = importer
            .binding_by_name(alias)
            .ok_or_else(|| format!("`{alias_name}` is not a module"))?;
        let rhs = importer.binding(bid).rhs;
        let Node::Call(call) = importer.node(rhs) else {
            return Err(format!("`{alias_name}` is not a module"));
        };
        let CallHead::Builtin(head) = call.head else {
            return Err(format!("`{alias_name}` is not a module"));
        };
        let head_name = importer.resolve(head);
        if head_name != "load_module" && head_name != "standard_module" {
            return Err(format!("`{alias_name}` is not a module"));
        }
        let standard = head_name == "standard_module";
        let path = match call.args.first().map(|&a| importer.node(a)) {
            Some(Node::Lit(Scalar::Str(s))) => s.to_string(),
            _ => return Err(format!("`{alias_name}` load is missing a path string")),
        };
        // `standard_module(name, version)` carries the requested version as the
        // second positional arg; `load_module(path)` has no version.
        let version = match call.args.get(1).map(|&a| importer.node(a)) {
            Some(Node::Lit(Scalar::Str(s))) => Some(s.to_string()),
            _ => None,
        };
        let substitutions = call
            .named
            .iter()
            .filter(|n| n.kind == NamedKind::Assign)
            .map(|n| (importer.resolve(n.name).to_string(), n.value))
            .collect();
        Ok(LoadDirective {
            path,
            standard,
            version,
            substitutions,
        })
    }

    /// Infer (and memo) the dependency at `path`, seeding substitution inputs
    /// with `seeds` before the walk. Pushes/pops the active-import stack and
    /// inserts the annotated clone into `self.memo` under `key` when done.
    /// The caller (`resolve`) is responsible for the cycle check before calling.
    /// Diagnostics from the child run are accumulated in `self.dep_diags` so
    /// that cycle errors and other dep-level errors reach the root caller.
    fn infer_dep(
        &self,
        dep: &Module,
        path: &str,
        key: &(String, String),
        seeds: &[(NodeId, Resolved)],
        level: crate::Level,
    ) {
        self.stack.borrow_mut().push(path.to_string());
        let mut dep_clone = dep.clone();
        let child_diags =
            crate::trace::Inferencer::new_seeded(&mut dep_clone, level, self, seeds).run();
        self.stack.borrow_mut().pop();
        self.push_dep_diags(child_diags);
        // Two-phase memo access: `contains_key` above released the `Ref` so we
        // can call `borrow_mut()` here without conflicting borrows.
        self.memo.borrow_mut().insert(key.clone(), dep_clone);
    }

    /// Returns the `%assign` substitutions of the `load_module` call bound to
    /// `alias` in `importer`: `(input-name, value-node-in-importer)` pairs.
    /// Returns an empty vec when the directive is missing or malformed — the
    /// hard error is re-reported by `resolve`.
    pub(crate) fn substitutions_of(
        &self,
        importer: &Module,
        alias: Symbol,
    ) -> Vec<(String, NodeId)> {
        match self.load_directive(importer, alias) {
            Ok(d) => d.substitutions,
            Err(_) => vec![],
        }
    }

    /// Resolve `(%ref alias binding_name)` from `importer`. `subst_annos` are
    /// importer-context inferred annotations for substitution inputs. On
    /// failure returns `Err(message)`; the caller emits an anchored error +
    /// `Type::Failed`.
    pub(crate) fn resolve(
        &self,
        importer: &Module,
        alias: Symbol,
        binding_name: &str,
        subst_annos: &[(String, Resolved)],
        level: crate::Level,
    ) -> Result<Resolved, String> {
        let directive = self.load_directive(importer, alias)?;

        // §09 standard modules resolve against the merged catalogue set
        // (built-in + host-supplied external catalogues). The bare-reference
        // type matches a bare base name; the measure/result type is lowered at
        // the application site (where the concrete arg types are known), so the
        // catalogue sig rides over here.
        if directive.standard {
            return resolve_standard(&self.catalogues, &directive, binding_name);
        }

        let dep = self.bundle.get(&directive.path).ok_or_else(|| {
            // Distinguish a remote (URL) directive from a local one. A URL that
            // isn't in the module set is a fine reference whose source simply
            // hasn't been fetched yet — say so, rather than a bare "not found"
            // that reads like a 404 / "use a filename instead". The remedy
            // (fetch the deps) is host-neutral: `flatppl prepare` on the CLI, the
            // editor's download-dependencies action in an LSP client.
            let p = &directive.path;
            if is_remote_source(p) {
                format!(
                    "remote module `{p}` is not available — fetch the model's dependencies first"
                )
            } else {
                format!("module file `{p}` not found")
            }
        })?;

        // Memo key: path + Debug signature of substitution annotations
        // (ValueSet is not Hash/Eq, so we use Debug strings).
        let sig = subst_annos
            .iter()
            .map(|(name, r)| format!("{name}={:?}/{:?}/{:?}", r.ty, r.phase, r.vset))
            .collect::<Vec<_>>()
            .join(",");
        let key = (directive.path.clone(), sig);

        // Two-phase memo access: check first without holding a `Ref`, then
        // infer+insert (which needs `borrow_mut`), then re-borrow to read.
        // Holding a `Ref` across `borrow_mut` would panic at runtime.
        if !self.memo.borrow().contains_key(&key) {
            if self.stack.borrow().contains(&directive.path) {
                let mut chain = self.stack.borrow().clone();
                chain.push(directive.path.clone());
                return Err(format!("module cycle: {}", chain.join(" → ")));
            }
            let seeds = seed_plan(dep, subst_annos);
            self.infer_dep(dep, &directive.path, &key, &seeds, level);
        }

        let memo = self.memo.borrow();
        let dep_annotated = memo.get(&key).expect("just inserted");
        let dep_path = &directive.path;
        // CROSS-INTERNER: resolve by string, not Symbol — the importer and
        // dependency have separate interners.
        let (_, b) = dep_annotated
            .bindings()
            .find(|(_, b)| dep_annotated.resolve(b.name) == binding_name)
            .ok_or_else(|| format!("module `{dep_path}` has no binding `{binding_name}`"))?;
        if !b.public {
            return Err(format!("`{binding_name}` is private to `{dep_path}`"));
        }
        let rhs = b.rhs;
        // Spec §04 stochastic boundary: only `fixed`/`parameterized` bindings of
        // the loaded module are accessible. A `stochastic`-phase binding — a
        // `draw` (or `draw` descendant) not reified via `lawof`/`kernelof` — is
        // invisible across the load boundary (preserving referential
        // transparency). `lawof`/`kernelof` absorb stochasticity, so a reified
        // measure/kernel is fixed/parameterized and stays visible.
        if dep_annotated.phase_of(rhs) == Some(Phase::Stochastic) {
            return Err(format!(
                "`{binding_name}` is stochastic and not accessible from module `{dep_path}` \
                 (spec §04: stochastic bindings are invisible across the load boundary; \
                 reify it with `lawof`/`kernelof` to export it)"
            ));
        }
        Ok(Resolved {
            ty: dep_annotated
                .type_of(rhs)
                .cloned()
                .unwrap_or(Type::Deferred),
            phase: dep_annotated.phase_of(rhs).unwrap_or(Phase::Fixed),
            vset: dep_annotated
                .valueset_of(rhs)
                .cloned()
                .unwrap_or(ValueSet::Unknown),
            result: callable_body_result(dep_annotated, rhs),
            catalogue: None,
        })
    }
}

/// Resolve a §09 standard-module reference against the merged catalogue set.
/// Validates the requested version, distinguishes a missing module ("not
/// found") from a missing binding ("has no binding"), and returns a `Resolved`
/// whose `ty`/`vset` are the *bare-reference* values (matching a bare base
/// name) with the catalogue sig carried for lowering at the application site.
fn resolve_standard(
    catalogues: &CatalogueSet<'_>,
    directive: &LoadDirective,
    binding_name: &str,
) -> Result<Resolved, String> {
    let path = &directive.path;

    // Module-name miss → "not found"; module present but binding absent →
    // "has no binding".
    let known_version = catalogues
        .module_version(path)
        .ok_or_else(|| format!("standard module `{path}` not found"))?;

    // Validate the requested version (when the call supplied one).
    if let Some(requested) = &directive.version {
        if requested != known_version {
            return Err(format!(
                "standard module `{path}` has unknown version `{requested}` (catalogue provides `{known_version}`)"
            ));
        }
    }

    let (sig, degraded) = catalogues
        .module(path, binding_name)
        .ok_or_else(|| format!("standard module `{path}` has no binding `{binding_name}`"))?;

    Ok(Resolved {
        // Bare reference: matches a bare base name (`Normal` referenced bare is
        // `Type::Any`). The real type is lowered when the ref is applied.
        ty: Type::Any,
        phase: Phase::Fixed,
        vset: ValueSet::Unknown,
        result: None,
        catalogue: Some(CatalogueRef {
            sig: sig.clone(),
            degraded: degraded.map(str::to_string),
        }),
    })
}

/// For a binding whose RHS is a reified callable (`functionof` / `kernelof`,
/// i.e. a call carrying an inputs boundary), the dependency's inferred type of
/// its reified BODY (the first positional argument). `None` when the RHS is not
/// a reification. This is the cross-module analogue of `ops::reified_result_type`
/// — the local machinery looks the body type up by node, but the body lives in
/// the dependency's interner, so we read it here and ride it over in `Resolved`.
///
/// **Mirror note:** the body-probe (`call.inputs.is_some()` + `args.first()`)
/// mirrors the identical probe in `ops::reified_body` (the single-module twin).
/// If the probe shape ever changes there, update this function in lock-step.
fn callable_body_result(dep: &Module, rhs: NodeId) -> Option<Type> {
    let Node::Call(call) = dep.node(rhs) else {
        return None;
    };
    call.inputs.as_ref()?;
    let body = *call.args.first()?;
    dep.type_of(body).cloned()
}

/// Map each substitution to the dependency's input binding RHS node, paired
/// with the importer-context annotation to seed there. Names not found in the
/// dependency are silently skipped (unknown-input validation happens at the
/// load_module call site in trace.rs).
pub(crate) fn seed_plan(
    dep: &Module,
    subst_annos: &[(String, Resolved)],
) -> Vec<(NodeId, Resolved)> {
    let mut seeds = Vec::new();
    for (name, res) in subst_annos {
        if let Some((_, b)) = dep
            .bindings()
            .find(|(_, b)| dep.resolve(b.name) == name.as_str())
        {
            // Substitution seeds carry no body-result (inputs are plain
            // values, not reified callables); clone the rest from `res`.
            seeds.push((
                b.rhs,
                Resolved {
                    result: None,
                    catalogue: None,
                    ..res.clone()
                },
            ));
        }
    }
    seeds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_inserts_and_looks_up() {
        let mut b = ModuleBundle::new();
        b.insert("helpers.flatppl", Arc::new(Module::new()));
        assert!(b.get("helpers.flatppl").is_some());
        assert!(b.get("missing.flatppl").is_none());
    }
}
