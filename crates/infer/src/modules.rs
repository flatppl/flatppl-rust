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

/// Parsed dependency modules, keyed by **resolved file identity** — the host's
/// canonical spelling of the file or URL, not the `load_module` literal.
/// Supplied by the host (the engine does no file I/O).
///
/// Spec §04 "Path resolution" resolves a relative `load_module` path against
/// the directory of the file that declares it, so the same literal in two
/// importers names two different files. Keying by the literal collapses them,
/// which hands a reference the wrong module's type with no diagnostic. The
/// literal is therefore only a lookup input, resolved per importer through
/// `resolutions`.
///
/// Dependencies are held behind `Arc<Module>` so a host that assembles the same
/// bundle repeatedly (e.g. the LSP, once per keystroke) shares one parsed copy
/// rather than deep-cloning each dependency `Module` on every assembly. Inserts
/// and lookups move/borrow the `Arc`; the only deep clone of a dependency is the
/// per-import-site working copy `infer_dep` mutates, which is genuinely needed
/// (inference annotates it in place).
#[derive(Debug, Default, Clone)]
pub struct ModuleBundle {
    /// Dependencies by resolved identity.
    by_id: HashMap<String, Arc<Module>>,
    /// importer identity -> directive literal -> resolved identity.
    resolutions: HashMap<String, HashMap<String, String>>,
    /// Directive literal -> identity, kept only while that literal denotes ONE
    /// identity across the whole bundle. `None` records a literal used for two
    /// files: a lookup with no importer context must then refuse rather than
    /// pick one of them.
    by_literal: HashMap<String, Option<String>>,
    /// Identity of the module handed to `infer_module`. Its own directives
    /// resolve against this key.
    root: String,
}

impl ModuleBundle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a parsed dependency whose resolved identity **is** its
    /// `load_module` literal. For a host that does no path resolution of its
    /// own, and for a single-importer graph.
    pub fn insert(&mut self, path: impl Into<String>, module: Arc<Module>) {
        let path = path.into();
        self.insert_by_id(path.clone(), path, module);
    }

    /// Insert a parsed dependency under its resolved `identity`, and record that
    /// `literal`, as spelled in the module identified by `importer`, resolves
    /// to it.
    pub fn insert_resolved(
        &mut self,
        importer: impl Into<String>,
        literal: impl Into<String>,
        identity: impl Into<String>,
        module: Arc<Module>,
    ) {
        let literal = literal.into();
        let identity = identity.into();
        self.resolutions
            .entry(importer.into())
            .or_default()
            .insert(literal.clone(), identity.clone());
        self.insert_by_id(literal, identity, module);
    }

    /// Record the identity of the module `infer_module` is called on, so its own
    /// directives resolve against the right importer.
    pub fn set_root(&mut self, identity: impl Into<String>) {
        self.root = identity.into();
    }

    /// Identity of the module `infer_module` is called on.
    pub fn root(&self) -> &str {
        &self.root
    }

    /// The identity `literal` denotes when spelled in `importer`.
    ///
    /// Falls back to the unambiguous literal alias when the host recorded no
    /// resolution for this site, which keeps a literal-keyed bundle
    /// ([`insert`](Self::insert)) working. An ambiguous literal with no
    /// resolution yields `None`: refusing beats returning an arbitrary file.
    pub fn identity_of(&self, importer: &str, literal: &str) -> Option<&str> {
        if let Some(id) = self
            .resolutions
            .get(importer)
            .and_then(|per_importer| per_importer.get(literal))
        {
            return Some(id.as_str());
        }
        self.by_literal.get(literal)?.as_deref()
    }

    /// The dependency `literal` denotes when spelled in `importer`.
    pub fn get_from(&self, importer: &str, literal: &str) -> Option<&Module> {
        self.get_by_id(self.identity_of(importer, literal)?)
    }

    /// The shared `Arc` for the dependency `literal` denotes in `importer`.
    pub fn get_arc_from(&self, importer: &str, literal: &str) -> Option<&Arc<Module>> {
        self.get_arc_by_id(self.identity_of(importer, literal)?)
    }

    /// The dependency stored under resolved `identity`.
    pub fn get_by_id(&self, identity: &str) -> Option<&Module> {
        self.by_id.get(identity).map(|a| a.as_ref())
    }

    /// The shared `Arc` stored under resolved `identity`.
    pub fn get_arc_by_id(&self, identity: &str) -> Option<&Arc<Module>> {
        self.by_id.get(identity)
    }

    /// The dependency for `path` with no importer context: the literal alias
    /// route. `None` when `path` denotes two different files in this bundle.
    pub fn get(&self, path: &str) -> Option<&Module> {
        self.get_from("", path)
    }

    /// The shared `Arc` for `path` with no importer context (refcount bump, no
    /// deep clone). `None` when `path` denotes two different files.
    pub fn get_arc(&self, path: &str) -> Option<&Arc<Module>> {
        self.get_arc_from("", path)
    }

    fn insert_by_id(&mut self, literal: String, identity: String, module: Arc<Module>) {
        match self.by_literal.get(&literal) {
            // Already bound to a different file: the literal stops being a
            // usable key for importer-free lookups.
            Some(Some(prev)) if *prev != identity => {
                self.by_literal.insert(literal, None);
            }
            Some(_) => {}
            None => {
                self.by_literal.insert(literal, Some(identity.clone()));
            }
        }
        self.by_id.insert(identity, module);
    }
}

// ── Cross-interner symbol translation ────────────────────────────────────────
//
// A `Symbol` is an index into ONE module's interner (`core::id`), and the
// importer and each dependency intern independently. Several `Type` and
// `ValueSet` variants carry `Symbol`s as *names*: `Kernel`/`Function`/
// `Likelihood` inputs, `Record` fields, `Table` columns, `ValueSet::RecordSet`
// fields. Handing such a value across the load boundary unchanged re-reads every
// name as whatever the receiving interner holds at that index, which violates
// §11's "The `%inputs` names are the callable's input names" and silently
// rewrites a record domain. The determiniser's graft already re-interns
// (`determinizer::crossmodule`); these walkers give inference the same
// discipline.

/// `ty` with every interned name replaced by `f(name)`.
fn map_type_symbols(ty: &Type, f: &mut dyn FnMut(Symbol) -> Symbol) -> Type {
    fn map_fields(
        fields: &[(Symbol, Type)],
        f: &mut dyn FnMut(Symbol) -> Symbol,
    ) -> Box<[(Symbol, Type)]> {
        fields
            .iter()
            .map(|(n, t)| (f(*n), map_type_symbols(t, f)))
            .collect()
    }
    fn map_inputs(inputs: &[Symbol], f: &mut dyn FnMut(Symbol) -> Symbol) -> Box<[Symbol]> {
        inputs.iter().map(|s| f(*s)).collect()
    }
    match ty {
        Type::Array { shape, elem } => Type::Array {
            shape: shape.clone(),
            elem: Box::new(map_type_symbols(elem, f)),
        },
        Type::TVector { len, elem } => Type::TVector {
            len: *len,
            elem: Box::new(map_type_symbols(elem, f)),
        },
        Type::Record(fields) => Type::Record(map_fields(fields, f)),
        Type::Tuple(parts) => Type::Tuple(parts.iter().map(|t| map_type_symbols(t, f)).collect()),
        Type::Table { columns, nrows } => Type::Table {
            columns: map_fields(columns, f),
            nrows: *nrows,
        },
        Type::Measure { domain, mass } => Type::Measure {
            domain: Box::new(map_type_symbols(domain, f)),
            mass: *mass,
        },
        Type::Kernel { inputs, mass } => Type::Kernel {
            inputs: map_inputs(inputs, f),
            mass: *mass,
        },
        Type::Function { inputs } => Type::Function {
            inputs: map_inputs(inputs, f),
        },
        Type::Likelihood { inputs, obstype } => Type::Likelihood {
            inputs: map_inputs(inputs, f),
            obstype: Box::new(map_type_symbols(obstype, f)),
        },
        // Symbol-free leaves.
        Type::Deferred
        | Type::Failed(_)
        | Type::Any
        | Type::Scalar(_)
        | Type::RngState
        | Type::Module
        | Type::Var(_) => ty.clone(),
    }
}

/// `vs` with every interned `RecordSet` field name replaced by `f(name)`.
fn map_valueset_symbols(vs: &ValueSet, f: &mut dyn FnMut(Symbol) -> Symbol) -> ValueSet {
    match vs {
        ValueSet::CartPow(elem, d) => {
            ValueSet::CartPow(Box::new(map_valueset_symbols(elem, f)), *d)
        }
        ValueSet::CartProd(parts) => {
            ValueSet::CartProd(parts.iter().map(|s| map_valueset_symbols(s, f)).collect())
        }
        ValueSet::RecordSet(fields) => ValueSet::RecordSet(
            fields
                .iter()
                .map(|(n, s)| (f(*n), map_valueset_symbols(s, f)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Re-intern every name in `ty` from `from`'s interner into `into`'s.
fn reintern_type(into: &mut Module, from: &Module, ty: &Type) -> Type {
    map_type_symbols(ty, &mut |s| {
        let name = from.resolve(s).to_string();
        into.intern(&name)
    })
}

/// Re-intern every name in `vs` from `from`'s interner into `into`'s.
fn reintern_valueset(into: &mut Module, from: &Module, vs: &ValueSet) -> ValueSet {
    map_valueset_symbols(vs, &mut |s| {
        let name = from.resolve(s).to_string();
        into.intern(&name)
    })
}

/// The substitution half of the per-import-site memo key.
///
/// `Debug` pins the structure but renders a name as its bare `Symbol` index,
/// which two modules assign independently — so two import sites in DIFFERENT
/// modules can substitute records with different field names and produce the
/// same string, and the second site would read the first site's names out of the
/// memo. The interned names are therefore spelled out alongside the structure.
/// (`ValueSet` is not `Hash`/`Eq`, which is why the key is a string at all.)
fn subst_signature(importer: &Module, annos: &[(String, Resolved)]) -> String {
    annos
        .iter()
        .map(|(name, r)| {
            let mut names: Vec<String> = Vec::new();
            let mut record = |s: Symbol| {
                names.push(importer.resolve(s).to_string());
                s
            };
            map_type_symbols(&r.ty, &mut record);
            map_valueset_symbols(&r.vset, &mut record);
            format!(
                "{name}={:?}/{:?}/{:?}[{}]",
                r.ty,
                r.phase,
                r.vset,
                names.join("+")
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// `res` with `ty`, `vset` and `result` re-interned from `from` into `into`.
fn reintern_resolved(into: &mut Module, from: &Module, res: Resolved) -> Resolved {
    Resolved {
        ty: reintern_type(into, from, &res.ty),
        vset: reintern_valueset(into, from, &res.vset),
        result: res.result.map(|t| reintern_type(into, from, &t)),
        phase: res.phase,
        catalogue: res.catalogue,
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
    /// (resolved identity, substitution-signature) -> the dependency's inferred
    /// (annotated) Module.
    memo: RefCell<HashMap<(String, String), Module>>,
    /// Resolved identities of the modules on the current resolution chain
    /// (cycle detection). The top is the module being inferred right now, so it
    /// is also the importer whose directives resolve next.
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

    /// Resolved identity of the module being inferred right now: the innermost
    /// dependency on the chain, or the bundle root at the top level. A
    /// `load_module` literal read out of that module resolves against this key.
    pub(crate) fn importer_id(&self) -> String {
        self.stack
            .borrow()
            .last()
            .cloned()
            .unwrap_or_else(|| self.bundle.root().to_string())
    }

    /// The dependency a `load_module` literal denotes in the module being
    /// inferred right now.
    pub(crate) fn dep_for_literal(&self, literal: &str) -> Option<&Module> {
        self.bundle.get_from(&self.importer_id(), literal)
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

    /// Infer (and memo) the dependency with resolved identity `id`, seeding
    /// substitution inputs
    /// with `seeds` before the walk. Pushes/pops the active-import stack and
    /// inserts the annotated clone into `self.memo` under `key` when done.
    /// The caller (`resolve`) is responsible for the cycle check before calling.
    /// Diagnostics from the child run are accumulated in `self.dep_diags` so
    /// that cycle errors and other dep-level errors reach the root caller.
    ///
    /// The seeds carry `importer`-context types, so their names are re-interned
    /// into the dependency clone before the walk — the mirror of the outbound
    /// translation in `resolve`. Without it the dependency's annotation table
    /// holds foreign `Symbol` indices, which the outbound translation then reads
    /// against the wrong interner.
    fn infer_dep(
        &self,
        dep: &Module,
        importer: &Module,
        id: &str,
        key: &(String, String),
        seeds: &[(NodeId, Resolved)],
        level: crate::Level,
    ) {
        self.stack.borrow_mut().push(id.to_string());
        let mut dep_clone = dep.clone();
        let seeds: Vec<(NodeId, Resolved)> = seeds
            .iter()
            .map(|(n, r)| (*n, reintern_resolved(&mut dep_clone, importer, r.clone())))
            .collect();
        let child_diags =
            crate::trace::Inferencer::new_seeded(&mut dep_clone, level, self, &seeds).run();
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
    ///
    /// `importer` is `&mut` because the returned `Resolved` is re-interned into
    /// the importer's interner before it leaves the boundary
    /// (`reintern_resolved`) — a dependency `Symbol` is meaningless here.
    pub(crate) fn resolve(
        &self,
        importer: &mut Module,
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

        // The literal resolves against the module that declares it (spec §04
        // "Path resolution"), so the same spelling in two importers can denote
        // two different files. `importer_id` is that declaring module.
        let importer_id = self.importer_id();
        // Distinguish a remote (URL) directive from a local one. A URL that
        // isn't in the module set is a fine reference whose source simply
        // hasn't been fetched yet — say so, rather than a bare "not found"
        // that reads like a 404 / "use a filename instead". The remedy
        // (fetch the deps) is host-neutral: `flatppl prepare` on the CLI, the
        // editor's download-dependencies action in an LSP client.
        let unresolved = || {
            let p = &directive.path;
            if is_remote_source(p) {
                format!(
                    "remote module `{p}` is not available — fetch the model's dependencies first"
                )
            } else {
                format!("module file `{p}` not found")
            }
        };
        let dep_id = self
            .bundle
            .identity_of(&importer_id, &directive.path)
            .ok_or_else(unresolved)?
            .to_string();
        let dep = self.bundle.get_by_id(&dep_id).ok_or_else(unresolved)?;

        let key = (dep_id.clone(), subst_signature(importer, subst_annos));

        // Two-phase memo access: check first without holding a `Ref`, then
        // infer+insert (which needs `borrow_mut`), then re-borrow to read.
        // Holding a `Ref` across `borrow_mut` would panic at runtime.
        if !self.memo.borrow().contains_key(&key) {
            if self.stack.borrow().contains(&dep_id) {
                let mut chain = self.stack.borrow().clone();
                chain.push(dep_id.clone());
                return Err(format!("module cycle: {}", chain.join(" → ")));
            }
            let seeds = seed_plan(dep, subst_annos);
            self.infer_dep(dep, importer, &dep_id, &key, &seeds, level);
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
        let raw = Resolved {
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
        };
        Ok(reintern_resolved(importer, dep_annotated, raw))
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
