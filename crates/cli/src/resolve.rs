//! Source resolution for the CLI: read the main input and (for `infer`) the
//! transitive `load_module` dependency graph through the `flatppl-fileaccess`
//! abstraction — local paths pass through, `http`/`https` URLs are fetched and
//! cached (spec §sec:url-cache).
//!
//! This is the host-side counterpart to `load_module`: the engine stays
//! I/O-free and consumes a pre-assembled [`ModuleBundle`]; this module builds
//! that bundle. Trust is batched per discovery wave — each level of the
//! dependency tree is approved together (interactive prompt; non-interactive
//! refuses untrusted URLs, per spec §sec:url-cache).

use std::cell::RefCell;
use std::collections::HashSet;
use std::io::IsTerminal;

use flatppl_core::{CallHead, Idx, Module, Node, NodeId, Scalar};
use flatppl_fileaccess::{Cache, HttpFetcher, Location};

use crate::Failure;

/// The CLI's file-access layer: a [`Cache`] + HTTP fetcher plus the interactive
/// trust policy. Reads go through here so every `source` (local or URL) is
/// handled uniformly.
pub struct CliResolver {
    cache: Cache,
    fetcher: HttpFetcher,
    /// Whether we may prompt for trust (stdin + stderr are both TTYs).
    interactive: bool,
    /// URLs approved this session (so a batch-prompted wave is not re-prompted
    /// when each member is then read).
    approved: RefCell<HashSet<String>>,
}

impl CliResolver {
    /// Configure from the environment (`FLATPPL_CACHEDIR` / `_OFFLINE` /
    /// `FLATPPL_TRUST`; spec §sec:url-cache). Interactive iff stdin and stderr
    /// are terminals.
    pub fn from_env() -> Self {
        CliResolver {
            cache: Cache::from_env(),
            fetcher: HttpFetcher,
            interactive: std::io::stdin().is_terminal() && std::io::stderr().is_terminal(),
            approved: RefCell::new(HashSet::new()),
        }
    }

    /// Batch-approve every not-yet-trusted URL among `locs` (one prompt for the
    /// whole wave). Non-interactive tooling errors instead of prompting. Local
    /// locations and already-trusted/cached URLs need nothing.
    pub fn ensure_trusted(&self, locs: &[&Location]) -> Result<(), Failure> {
        let mut need: Vec<String> = Vec::new();
        for loc in locs {
            if let Location::Remote(url) = loc {
                if !self.approved.borrow().contains(url) && self.cache.needs_approval(url) {
                    need.push(url.clone());
                }
            }
        }
        need.sort();
        need.dedup();
        if need.is_empty() {
            return Ok(());
        }
        if !self.interactive {
            return Err(Failure::Plain(format!(
                "refusing to fetch untrusted URL(s) — non-interactive; set FLATPPL_TRUST to \
                 allow, or pre-trust them:\n  {}",
                need.join("\n  ")
            )));
        }
        if prompt_trust(&need)? {
            let mut approved = self.approved.borrow_mut();
            for url in need {
                approved.insert(url);
            }
            Ok(())
        } else {
            Err(Failure::Plain(
                "aborted: declined to trust the listed URL(s)".to_string(),
            ))
        }
    }

    /// Resolve `loc` to a local file and read it as UTF-8 text. A URL is fetched
    /// and cached, gated on trust (which a prior `ensure_trusted` call may
    /// already have granted for the whole wave, so this does not re-prompt).
    pub fn read_string(&self, loc: &Location) -> Result<String, Failure> {
        self.ensure_trusted(&[loc])?;
        let path = match loc {
            Location::Local(p) => {
                if !p.exists() {
                    return Err(Failure::Plain(format!("file not found: {}", p.display())));
                }
                p.clone()
            }
            Location::Remote(url) => {
                let oracle = |u: &str| self.approved.borrow().contains(u);
                self.cache
                    .get(url, &self.fetcher, &oracle)
                    .map_err(|e| Failure::Plain(e.to_string()))?
            }
        };
        std::fs::read_to_string(&path)
            .map_err(|e| Failure::Plain(format!("reading `{}`: {e}", loc.display())))
    }
}

/// Prompt (on stderr) to trust + fetch a wave of URLs; read the answer from
/// stdin. Returns `true` on `y`/`yes`.
fn prompt_trust(urls: &[String]) -> Result<bool, Failure> {
    use std::io::Write;
    eprintln!("flatppl: the following URL source(s) are not yet trusted:");
    for url in urls {
        eprintln!("  {url}");
    }
    eprint!("Fetch and trust them? [y/N]: ");
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    std::io::stdin()
        .read_line(&mut line)
        .map_err(|e| Failure::Plain(format!("reading approval: {e}")))?;
    Ok(matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

/// The `load_module` directives in `module`: each `(source-string, resolved
/// Location)`, the source resolved relative to `base`. `standard_module` is a
/// catalogue reference (not a file) and is excluded; a non-literal source is
/// skipped (it cannot be resolved statically).
pub fn directives_of(module: &Module, base: &Location) -> Vec<(String, Location)> {
    let mut out = Vec::new();
    for i in 0..module.node_count() {
        let id = NodeId::from_usize(i);
        let Node::Call(call) = module.node(id) else {
            continue;
        };
        let CallHead::Builtin(head) = call.head else {
            continue;
        };
        if module.resolve(head) != "load_module" {
            continue;
        }
        if let Some(&arg0) = call.args.first() {
            if let Node::Lit(Scalar::Str(source)) = module.node(arg0) {
                out.push((source.to_string(), base.join(source)));
            }
        }
    }
    out
}

/// Build the inference [`ModuleBundle`] for `root` (located at `root_loc`):
/// breadth-first over the `load_module` graph, resolving + parsing each
/// dependency through `resolver` and keying the bundle by the directive string
/// the engine looks up. Trust is batched per BFS level.
///
/// The bundle is keyed by directive string (the engine's lookup key), so a
/// given string must denote one file across the whole graph — a conflict is a
/// hard error rather than a silent last-writer-wins.
#[cfg(feature = "infer")]
pub fn build_bundle(
    root: &Module,
    root_loc: &Location,
    resolver: &CliResolver,
) -> Result<flatppl_infer::ModuleBundle, Failure> {
    use std::collections::HashMap;
    use std::sync::Arc;

    let mut bundle = flatppl_infer::ModuleBundle::new();
    // Parse each distinct file once, keyed by its resolved location.
    let mut parsed: HashMap<String, Arc<Module>> = HashMap::new();
    // Locations whose own dependencies have already been queued (cycle guard).
    let mut walked: HashSet<String> = HashSet::new();
    // directive string → resolved location, to catch a string used for two files.
    let mut key_loc: HashMap<String, String> = HashMap::new();

    walked.insert(root_loc.display());
    let mut level = directives_of(root, root_loc);

    while !level.is_empty() {
        let locs: Vec<&Location> = level.iter().map(|(_, l)| l).collect();
        resolver.ensure_trusted(&locs)?;

        let mut next: Vec<(String, Location)> = Vec::new();
        for (directive, loc) in &level {
            let ld = loc.display();
            match key_loc.get(directive) {
                Some(prev) if prev != &ld => {
                    return Err(Failure::Plain(format!(
                        "load_module(\"{directive}\") refers to two different files \
                         ({prev} and {ld}); the inference bundle is keyed by the directive \
                         string, so a name must denote one file across the module graph"
                    )));
                }
                _ => {
                    key_loc.insert(directive.clone(), ld.clone());
                }
            }

            let module = match parsed.get(&ld) {
                Some(m) => m.clone(),
                None => {
                    let source = resolver.read_string(loc)?;
                    let format = crate::Format::from_location(loc).map_err(Failure::Plain)?;
                    let module =
                        crate::read_module(format, &source).map_err(|(message, line, span)| {
                            Failure::Diagnostic {
                                path: std::path::PathBuf::from(loc.display()),
                                source: source.clone(),
                                message,
                                line,
                                span,
                            }
                        })?;
                    let module = Arc::new(module);
                    parsed.insert(ld.clone(), module.clone());
                    module
                }
            };
            bundle.insert(directive.clone(), module.clone());

            // Queue this file's own dependencies exactly once.
            if walked.insert(ld.clone()) {
                next.extend(directives_of(&module, loc));
            }
        }
        level = next;
    }
    Ok(bundle)
}

#[cfg(all(test, feature = "infer"))]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The `bayesian_inference_3` fixture loads `common`, which loads `priors`
    /// (two-level nesting). Building the bundle from local fixtures and inferring
    /// must resolve every cross-module reference (no "not found"/"deferred"),
    /// proving the CLI walker assembles the same graph the engine expects.
    #[test]
    fn build_bundle_resolves_local_two_level_nesting() {
        let dir: PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            "../../fixtures/flatppl/bayesian_inference",
        ]
        .iter()
        .collect();
        let root_path = dir.join("bayesian_inference_3.flatppl");
        let root_loc = Location::Local(root_path.clone());

        let resolver = CliResolver::from_env(); // local-only: no env/net/trust needed
        let source = std::fs::read_to_string(&root_path).unwrap();
        let mut root = flatppl_syntax::parse(&source).expect("root parses");

        let bundle = build_bundle(&root, &root_loc, &resolver).expect("bundle builds");
        let diags = flatppl_infer::infer_module(&mut root, &bundle, flatppl_infer::Level::Shape);

        let resolution_errors: Vec<_> = diags
            .iter()
            .filter(|d| {
                d.severity == flatppl_infer::Severity::Error
                    && (d.message.contains("not found")
                        || d.message.contains("has no binding")
                        || d.message.contains("deferred"))
            })
            .collect();
        assert!(
            resolution_errors.is_empty(),
            "nested local deps must resolve via the CLI walker; got {resolution_errors:?}"
        );
    }
}
