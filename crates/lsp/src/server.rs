//! The FlatPPL stdio message loop.
//!
//! [`run`] drives the main LSP event loop after the initialize handshake has
//! already completed. It owns the salsa [`Database`], the open-document map,
//! the workspace [`FileSet`], and the external [`Catalogues`]; it processes
//! `didOpen`/`didChange` notifications (full-sync), `hover` requests, and
//! `shutdown`.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::str::FromStr;
use std::time::{Duration, Instant};

use crossbeam_channel::select;
use lsp_server::{Connection, Message, Response};
use lsp_types::{
    CompletionOptions, HoverProviderCapability, OneOf, PublishDiagnosticsParams,
    ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind, Uri,
    notification::{
        DidChangeTextDocument, DidChangeWatchedFiles, DidCloseTextDocument, DidOpenTextDocument,
        Notification as _, PublishDiagnostics,
    },
    request::{
        Completion, DocumentSymbolRequest, GotoDefinition, HoverRequest, InlayHintRequest,
        Request as _, WorkspaceSymbolRequest,
    },
};

use crate::db::{Catalogues, Database, FileSet, SourceFile};
use crate::line_index::Pos;
use crate::queries::{import_bundle, line_index, node_span_index};

// ── run ─────────────────────────────────────────────────────────────────────

/// Drive the FlatPPL LSP event loop.
///
/// `connection` is the live [`Connection`] returned by `Connection::stdio()`;
/// the initialize handshake must have been completed **before** calling this
/// function (i.e. `connection.initialize` or the start/finish pair have already
/// been called).  `init_params` is the raw `serde_json::Value` returned from
/// the handshake.
pub fn run(
    connection: Connection,
    init_params: serde_json::Value,
) -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
    let mut db = Database::default();

    // ── Parse InitializeParams ───────────────────────────────────────────────

    #[allow(deprecated)] // root_uri is deprecated but still the most-portable field
    let params: lsp_types::InitializeParams =
        serde_json::from_value(init_params).unwrap_or_default();

    // ── External catalogues from initializationOptions ───────────────────────
    //
    // Clients may supply: `"initializationOptions": { "catalogues": ["...ron...", ...] }`
    let cat_sources: Vec<String> = catalogue_sources_from_params(&params);
    let cats = Catalogues::new(&db, cat_sources);

    // ── Workspace scan ───────────────────────────────────────────────────────
    //
    // Collect workspace roots from rootUri / workspaceFolders, then recursively
    // find every `*.flatppl` file, read it, and build the initial SourceFile
    // map and FileSet.

    let mut uri_to_file: HashMap<String, SourceFile> = HashMap::new();

    #[allow(deprecated)]
    let roots: Vec<String> = {
        let mut v = Vec::new();
        if let Some(folders) = &params.workspace_folders {
            for f in folders {
                v.push(f.uri.as_str().to_owned());
            }
        } else if let Some(uri) = &params.root_uri {
            v.push(uri.as_str().to_owned());
        }
        v
    };

    for root_uri_str in &roots {
        if let Some(path) = file_uri_to_path(root_uri_str) {
            scan_dir(Path::new(&path), &mut db, &mut uri_to_file);
        }
    }

    let fs = build_fileset(&db, &uri_to_file);

    // URIs of files currently open in the editor (via didOpen/didClose).
    // Files added from disk by didChangeWatchedFiles are NOT in this set, so
    // the watched-file handler can distinguish editor-managed from disk-only
    // files and update disk-only files on CHANGED/CREATED without clobbering
    // unsaved editor edits.
    let mut editor_open_uris: HashSet<String> = HashSet::new();

    // Last document version reported by the editor (didOpen / didChange).
    // Used to (a) drop stale / out-of-order edits and (b) stamp the published
    // diagnostics with the version they were computed against.
    let mut doc_versions: HashMap<String, i32> = HashMap::new();

    // Publish initial diagnostics for all workspace files. Startup is not part
    // of an edit burst, so these go out immediately (no debounce).
    for (uri_str, &file) in &uri_to_file {
        publish_diagnostics(&connection, &db, file, fs, cats, uri_str, None)?;
    }

    // ── Concurrency + debounce machinery ─────────────────────────────────────
    //
    // Requests run off the main thread on a worker pool holding cloned salsa
    // `Database` handles; worker responses come back on `result_rx`. Diagnostics
    // are debounced: notification arms mark affected URIs `dirty` and arm a
    // deadline; once the burst settles (`DEBOUNCE` of quiescence) we flush.
    let (result_tx, result_rx) = crossbeam_channel::unbounded::<Message>();
    let pool = crate::pool::Pool::new(
        std::thread::available_parallelism()
            .map(|n| n.get().min(4))
            .unwrap_or(2),
        result_tx.clone(),
    );
    const DEBOUNCE: Duration = Duration::from_millis(200);
    let mut diag_deadline: Option<Instant> = None;
    // URIs whose diagnostics need (re)publishing once the burst settles.
    let mut dirty: HashSet<String> = HashSet::new();

    // ── Main loop ────────────────────────────────────────────────────────────
    //
    // `select!` over three sources: the client connection, the worker results
    // channel, and (when armed) the debounce timeout. A `None` from the match
    // means "handled internally, no client message to process this iteration".

    loop {
        let timeout = diag_deadline.map(|d| d.saturating_duration_since(Instant::now()));
        let selected: Option<Result<Message, crossbeam_channel::RecvError>> = match timeout {
            Some(t) => select! {
                recv(connection.receiver) -> m => Some(m),
                recv(result_rx) -> r => {
                    if let Ok(msg) = r { connection.sender.send(msg)?; }
                    None
                }
                default(t) => {
                    // Debounce fired: publish the coalesced dirty set.
                    flush_dirty(
                        &connection, &db, fs, cats, &uri_to_file, &doc_versions, &mut dirty,
                    )?;
                    diag_deadline = None;
                    None
                }
            },
            None => select! {
                recv(connection.receiver) -> m => Some(m),
                recv(result_rx) -> r => {
                    if let Ok(msg) = r { connection.sender.send(msg)?; }
                    None
                }
            },
        };
        let Some(msg) = selected else { continue };
        let msg = match msg {
            Ok(m) => m,
            Err(_) => break, // client connection closed
        };
        match msg {
            Message::Notification(note) => {
                match note.method.as_str() {
                    DidOpenTextDocument::METHOD => {
                        let p: lsp_types::DidOpenTextDocumentParams =
                            match serde_json::from_value(note.params) {
                                Ok(v) => v,
                                Err(e) => {
                                    eprintln!(
                                        "flatppl-lsp: malformed didOpen params, skipping: {e}"
                                    );
                                    continue;
                                }
                            };
                        let uri_str = p.text_document.uri.as_str().to_owned();
                        let text = p.text_document.text;
                        // Mark as editor-managed so watched-file CHANGED events
                        // skip it (editor content takes precedence over on-disk).
                        editor_open_uris.insert(uri_str.clone());
                        doc_versions.insert(uri_str.clone(), p.text_document.version);
                        upsert_file(&mut db, &mut uri_to_file, uri_str, text);
                        // Update the shared FileSet only when the file SET membership
                        // changes (a new open always adds a file, so this fires).
                        sync_file_set(&mut db, fs, &uri_to_file);
                        // Re-publish diagnostics for ALL open docs: a newly-opened
                        // file can satisfy a previously-unresolved import in any
                        // already-open doc, so the full set must be refreshed.
                        // Mark them dirty and arm the debounce instead of
                        // publishing inline.
                        for doc_uri_str in uri_to_file.keys() {
                            dirty.insert(doc_uri_str.clone());
                        }
                        diag_deadline = Some(Instant::now() + DEBOUNCE);
                    }
                    DidChangeTextDocument::METHOD => {
                        let p: lsp_types::DidChangeTextDocumentParams =
                            match serde_json::from_value(note.params) {
                                Ok(v) => v,
                                Err(e) => {
                                    eprintln!(
                                        "flatppl-lsp: malformed didChange params, skipping: {e}"
                                    );
                                    continue;
                                }
                            };
                        let uri_str = p.text_document.uri.as_str().to_owned();
                        // Drop stale / out-of-order edits: an editor may deliver a
                        // didChange whose version predates one we already applied
                        // (network reordering, replayed buffers). Applying it would
                        // resurrect older text; ignore it entirely.
                        let new_version = p.text_document.version;
                        if let Some(&prev) = doc_versions.get(&uri_str) {
                            if new_version < prev {
                                continue;
                            }
                        }
                        doc_versions.insert(uri_str.clone(), new_version);
                        // Full sync — take last content change.
                        if let Some(change) = p.content_changes.into_iter().last() {
                            upsert_file(&mut db, &mut uri_to_file, uri_str.clone(), change.text);
                        }
                        // Guard the FileSet salsa input: a pure text edit leaves
                        // membership unchanged, so no revision bump is needed.
                        sync_file_set(&mut db, fs, &uri_to_file);
                        // Republish diagnostics only for the changed doc and the
                        // open docs that (transitively) import it — the only docs
                        // whose diagnostics can change on this edit. Mark them
                        // dirty and (re)arm the debounce; a rapid edit burst thus
                        // coalesces into a single publish per affected doc.
                        if let Some(&changed) = uri_to_file.get(&uri_str) {
                            for (doc_uri_str, _file) in
                                affected_files(&db, fs, &uri_to_file, changed)
                            {
                                dirty.insert(doc_uri_str);
                            }
                            diag_deadline = Some(Instant::now() + DEBOUNCE);
                        }
                    }
                    DidCloseTextDocument::METHOD => {
                        // The editor closed its buffer for this file. Drop it from
                        // the editor-managed set and forget its version so on-disk
                        // `didChangeWatchedFiles` events take over again (the file
                        // is no longer authoritatively owned by the editor).
                        let p: lsp_types::DidCloseTextDocumentParams =
                            match serde_json::from_value(note.params) {
                                Ok(v) => v,
                                Err(e) => {
                                    eprintln!("flatppl-lsp: malformed didClose, skipping: {e}");
                                    continue;
                                }
                            };
                        let uri_str = p.text_document.uri.as_str().to_owned();
                        editor_open_uris.remove(&uri_str);
                        doc_versions.remove(&uri_str);
                    }
                    DidChangeWatchedFiles::METHOD => {
                        // Clients (e.g. VS Code) register their own glob watchers and
                        // push `workspace/didChangeWatchedFiles` for on-disk changes to
                        // files that are NOT open in the editor (e.g. a `load_module`
                        // dependency edited by another tool, or a git checkout).
                        // lsp-types 0.97's `WorkspaceServerCapabilities` has no static
                        // field for `didChangeWatchedFiles` registration options, so we
                        // handle the notification here and rely on the client's own
                        // watcher registration (dynamic `client/registerCapability`).
                        let p: lsp_types::DidChangeWatchedFilesParams = match serde_json::from_value(
                            note.params,
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                eprintln!(
                                    "flatppl-lsp: malformed didChangeWatchedFiles, skipping: {e}"
                                );
                                continue;
                            }
                        };
                        for change in p.changes {
                            let uri_str = change.uri.as_str().to_owned();
                            // Only .flatppl files; skip anything else.
                            if !uri_str.ends_with(".flatppl") {
                                continue;
                            }
                            // Skip files currently open in the editor — the editor's
                            // didChange is the source of truth for those (avoid
                            // clobbering unsaved edits).
                            if editor_open_uris.contains(&uri_str)
                                && change.typ != lsp_types::FileChangeType::DELETED
                            {
                                continue;
                            }
                            match change.typ {
                                lsp_types::FileChangeType::CREATED
                                | lsp_types::FileChangeType::CHANGED => {
                                    if let Some(path) = file_uri_to_path(&uri_str) {
                                        if let Ok(text) = std::fs::read_to_string(&path) {
                                            upsert_file(&mut db, &mut uri_to_file, uri_str, text);
                                        }
                                    }
                                }
                                lsp_types::FileChangeType::DELETED => {
                                    uri_to_file.remove(&uri_str);
                                }
                                _ => {}
                            }
                        }
                        sync_file_set(&mut db, fs, &uri_to_file);
                        // Republish diagnostics for all tracked docs: a watched-file
                        // change can affect any open importer. Mark them dirty and
                        // arm the debounce.
                        for doc_uri_str in uri_to_file.keys() {
                            dirty.insert(doc_uri_str.clone());
                        }
                        diag_deadline = Some(Instant::now() + DEBOUNCE);
                    }
                    _ => {} // ignore other notifications
                }
            }
            Message::Request(req) => {
                // Handle shutdown first.
                if connection.handle_shutdown(&req)? {
                    break;
                }
                // Dispatch the request to a worker thread on the pool. The
                // worker snapshots a salsa handle on the main thread (so a later
                // edit's `cancel_others` waits for it) and replies on
                // `result_tx`; cancelled jobs drop silently.
                dispatch_request(&pool, &result_tx, &db, &uri_to_file, fs, cats, req);
            }
            Message::Response(_) => {} // ignore server-originated response echoes
        }
    }

    Ok(())
}

// ── Off-main-thread request dispatch ─────────────────────────────────────────

/// Snapshot the salsa database + file map on the **main thread** and hand a
/// request job to the worker pool.
///
/// The `Database::clone` MUST happen here, on the main thread, before the job
/// is enqueued: salsa's `Storage::clone` bumps the live-clone count, and a later
/// input write (`set_text` on an edit) calls `cancel_others`, which sets the
/// cancellation flag and blocks until every outstanding clone drops. Cloning on
/// the worker would race that wait. The clone drops when the job returns,
/// releasing the handle so a pending write can proceed.
///
/// On the worker the query body runs under `salsa::Cancelled::catch`: if a
/// concurrent write cancels this revision the body unwinds with
/// `salsa::Cancelled` and we reply nothing (the client re-requests against the
/// new state; a stale reply computed against pre-edit text would be wrong).
fn dispatch_request(
    pool: &crate::pool::Pool,
    result_tx: &crossbeam_channel::Sender<Message>,
    db: &Database,
    uri_to_file: &HashMap<String, SourceFile>,
    fs: FileSet,
    cats: Catalogues,
    req: lsp_server::Request,
) {
    let db: Database = db.clone();
    let files: HashMap<String, SourceFile> = uri_to_file.clone();
    let result_tx = result_tx.clone();
    pool.spawn(move || {
        // `AssertUnwindSafe`: `&Database`/`&Request` are not auto-`UnwindSafe`,
        // but this is sound — on cancel we discard all captured state and reply
        // nothing, so no observer sees a half-updated value.
        let outcome = salsa::Cancelled::catch(std::panic::AssertUnwindSafe(|| {
            handle_request_on_worker(&db, &files, fs, cats, &req)
        }));
        match outcome {
            Ok(resp) => {
                let _ = result_tx.send(Message::Response(resp));
            }
            // Cancelled by a newer revision: drop silently, send no response.
            Err(_cancelled) => {}
        }
        // `db` (the clone) drops here, releasing the salsa handle.
    });
}

/// Run a single LSP request to a `Response` on a worker thread.
///
/// Dispatches over `req.method` to the existing capability handlers. An unknown
/// method yields a `MethodNotFound` error response. This runs inside
/// `salsa::Cancelled::catch`, so any salsa query it touches may unwind with
/// `salsa::Cancelled` when a concurrent edit invalidates the revision.
fn handle_request_on_worker(
    db: &Database,
    uri_to_file: &HashMap<String, SourceFile>,
    fs: FileSet,
    cats: Catalogues,
    req: &lsp_server::Request,
) -> Response {
    match req.method.as_str() {
        HoverRequest::METHOD => handle_hover(db, uri_to_file, fs, cats, req),
        DocumentSymbolRequest::METHOD => handle_document_symbols(db, uri_to_file, req),
        WorkspaceSymbolRequest::METHOD => handle_workspace_symbols(db, fs, req),
        InlayHintRequest::METHOD => handle_inlay_hints(db, uri_to_file, fs, cats, req),
        GotoDefinition::METHOD => handle_goto_definition(db, uri_to_file, fs, cats, req),
        Completion::METHOD => handle_completion(db, uri_to_file, fs, cats, req),
        _ => Response::new_err(
            req.id.clone(),
            lsp_server::ErrorCode::MethodNotFound as i32,
            format!("unsupported method: {}", req.method),
        ),
    }
}

/// Publish diagnostics for every URI accumulated in `dirty`, draining the set.
///
/// Diagnostics are published synchronously on the main thread: they are cheap
/// relative to a full edit burst and publishing here keeps ordering simple
/// (no interleaving with worker responses mid-flush). URIs no longer tracked
/// (e.g. a file deleted between dirtying and flush) are skipped.
fn flush_dirty(
    connection: &Connection,
    db: &Database,
    fs: FileSet,
    cats: Catalogues,
    uri_to_file: &HashMap<String, SourceFile>,
    doc_versions: &HashMap<String, i32>,
    dirty: &mut HashSet<String>,
) -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
    for uri_str in dirty.drain() {
        if let Some(&file) = uri_to_file.get(&uri_str) {
            publish_diagnostics(
                connection,
                db,
                file,
                fs,
                cats,
                &uri_str,
                doc_versions.get(&uri_str).copied(),
            )?;
        }
    }
    Ok(())
}

// ── Capability advertisement ─────────────────────────────────────────────────

/// Build the `ServerCapabilities` value we advertise during `initialize`.
pub fn server_capabilities() -> ServerCapabilities {
    ServerCapabilities {
        // We index positions in UTF-16 code units (the LSP default and what our
        // LineIndex computes), so advertise it explicitly rather than relying on
        // the client's assumed default.
        position_encoding: Some(lsp_types::PositionEncodingKind::UTF16),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        document_symbol_provider: Some(OneOf::Left(true)),
        workspace_symbol_provider: Some(OneOf::Left(true)),
        inlay_hint_provider: Some(OneOf::Left(true)),
        definition_provider: Some(OneOf::Left(true)),
        completion_provider: Some(CompletionOptions {
            trigger_characters: Some(vec![
                ".".to_string(),
                "~".to_string(),
                "=".to_string(),
                "(".to_string(),
                ",".to_string(),
            ]),
            ..Default::default()
        }),
        ..Default::default()
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Extract catalogue RON source strings from `initializationOptions.catalogues`.
fn catalogue_sources_from_params(params: &lsp_types::InitializeParams) -> Vec<String> {
    params
        .initialization_options
        .as_ref()
        .and_then(|v| v.get("catalogues"))
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default()
}

/// Convert a `file://` URI string to a filesystem path string, or `None` for
/// non-`file:` schemes.
///
/// The path portion is percent-decoded so that workspace roots containing
/// spaces or other special characters (e.g. `file:///Users/me/My%20Project`)
/// resolve correctly on disk.
fn file_uri_to_path(uri_str: &str) -> Option<String> {
    let path = uri_str.strip_prefix("file://")?;
    // strip_prefix leaves "//host/..." on Windows UNC file URIs; on Unix the
    // authority is always empty so `path` is now the absolute path (possibly
    // percent-encoded).
    Some(percent_decode(path))
}

/// Percent-encode a filesystem path into a `file://` URI body (encodes spaces
/// and other reserved bytes; leaves `/` and unreserved chars). Symmetric with
/// `file_uri_to_path`'s decode.
pub(crate) fn path_to_file_uri(path: &str) -> String {
    let mut out = String::from("file://");
    for b in path.bytes() {
        match b {
            b'/' | b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Percent-decode a URI path component (`%XX` → byte, then UTF-8).
///
/// Invalid sequences (`%` not followed by two hex digits, or non-UTF-8 byte
/// runs) are passed through unchanged.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (
                (bytes[i + 1] as char).to_digit(16),
                (bytes[i + 2] as char).to_digit(16),
            ) {
                out.push((hi * 16 + lo) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}

/// Recursively walk `dir`, read every `*.flatppl` file found, and insert it
/// into `uri_to_file`.  Unreadable files and non-UTF-8 content are skipped.
fn scan_dir(dir: &Path, db: &mut Database, uri_to_file: &mut HashMap<String, SourceFile>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_dir(&path, db, uri_to_file);
        } else if path.extension().and_then(|e| e.to_str()) == Some("flatppl") {
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let path_str = path.to_string_lossy().into_owned();
            let uri_str = path_to_file_uri(&path_str);
            let file = SourceFile::new(db, path_str, text);
            uri_to_file.insert(uri_str, file);
        }
    }
}

/// Build (or rebuild) a [`FileSet`] from the current `uri_to_file` map.
///
/// Called any time a file is added or modified so salsa sees a fresh input.
fn build_fileset(db: &Database, uri_to_file: &HashMap<String, SourceFile>) -> FileSet {
    let files: Vec<SourceFile> = uri_to_file.values().copied().collect();
    FileSet::new(db, files)
}

/// Update the `FileSet` salsa input only when the file SET membership changes.
///
/// A pure text edit of an already-open file leaves membership unchanged — the
/// edit flows through `SourceFile::set_text` in `upsert_file`, not through
/// `FileSet`. Bumping the `FileSet` input on every keystroke causes unnecessary
/// salsa revision churn; this guard skips the setter when the set of `SourceFile`
/// handles is identical to what is already stored.
///
/// Membership is compared structurally (sorted by stored path) rather than by
/// count. A `didChangeWatchedFiles` batch that deletes one file and creates
/// another leaves the count unchanged but changes membership — the count-based
/// guard would wrongly skip the update, leaving the `FileSet` salsa input stale
/// (keeping the deleted `SourceFile`, missing the new one). Comparing the actual
/// member handles catches this case correctly.
fn sync_file_set(db: &mut Database, fs: FileSet, uri_to_file: &HashMap<String, SourceFile>) {
    use salsa::Setter;
    let mut new_files: Vec<SourceFile> = uri_to_file.values().copied().collect();
    new_files.sort_by_key(|f| f.path(db).clone());
    let mut current: Vec<SourceFile> = fs.files(db).to_vec();
    current.sort_by_key(|f| f.path(db).clone());
    if new_files == current {
        return; // membership + identity unchanged → no salsa input churn
    }
    fs.set_files(db).to(new_files);
}

/// Return the subset of `uri_to_file` whose diagnostics can change when
/// `changed` is edited: `changed` itself, plus every open file whose transitive
/// import bundle includes `changed` as a resolved dependency.
///
/// `import_bundle` is a memoized salsa query, so the bundle lookups here are
/// cache hits for every file whose inputs have not changed.  Independent files
/// (those that do not import `changed`) are excluded, avoiding spurious
/// `analyze` recomputation.
///
/// Matching is by `SourceFile` identity (salsa input id) rather than by the
/// directive's literal path string.  This matters when a relative import such
/// as `"../helpers.flatppl"` resolves to a `SourceFile` whose stored path is
/// the absolute `/abs/helpers.flatppl` — the literal and the path differ, so a
/// string comparison would miss the importer and leave its diagnostics stale.
fn affected_files(
    db: &dyn salsa::Database,
    fs: FileSet,
    uri_to_file: &HashMap<String, SourceFile>,
    changed: SourceFile,
) -> Vec<(String, SourceFile)> {
    uri_to_file
        .iter()
        .filter(|(_, f)| **f == changed || import_bundle(db, **f, fs).imports(changed))
        .map(|(u, f)| (u.clone(), *f))
        .collect()
}

/// Insert or update a [`SourceFile`] in the map.
///
/// If the URI already has an entry, the `text` input is updated via the salsa
/// setter so downstream queries are incrementally recomputed.  Otherwise a new
/// `SourceFile` is created and inserted.  Returns the (new or existing) file.
fn upsert_file(
    db: &mut Database,
    uri_to_file: &mut HashMap<String, SourceFile>,
    uri_str: String,
    text: String,
) -> SourceFile {
    use salsa::Setter;
    if let Some(&existing) = uri_to_file.get(&uri_str) {
        existing.set_text(db).to(text);
        existing
    } else {
        let path = file_uri_to_path(&uri_str).unwrap_or_else(|| uri_str.clone());
        let file = SourceFile::new(db, path, text);
        uri_to_file.insert(uri_str, file);
        file
    }
}

/// Send a `textDocument/publishDiagnostics` notification for `file`.
///
/// `uri_str` must be a valid URI string; the send is best-effort (a send
/// failure is returned as an error to the caller).
fn publish_diagnostics(
    connection: &Connection,
    db: &Database,
    file: SourceFile,
    fs: FileSet,
    cats: Catalogues,
    uri_str: &str,
    version: Option<i32>,
) -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
    let diagnostics = crate::capabilities::diagnostics(db, file, fs, cats);
    let uri = Uri::from_str(uri_str)?;
    let params = PublishDiagnosticsParams {
        uri,
        diagnostics,
        version,
    };
    let note = lsp_server::Notification::new(PublishDiagnostics::METHOD.to_owned(), params);
    connection.sender.send(Message::Notification(note))?;
    Ok(())
}

/// Handle a `textDocument/hover` request.  Returns a `Response` (result or
/// null) without sending it — the caller dispatches the message.
fn handle_hover(
    db: &Database,
    uri_to_file: &HashMap<String, SourceFile>,
    fs: FileSet,
    cats: Catalogues,
    req: &lsp_server::Request,
) -> Response {
    let result = (|| -> Option<lsp_types::Hover> {
        let params: lsp_types::HoverParams = serde_json::from_value(req.params.clone()).ok()?;
        let uri_str = params
            .text_document_position_params
            .text_document
            .uri
            .as_str()
            .to_owned();
        let lsp_pos = params.text_document_position_params.position;
        let file = *uri_to_file.get(&uri_str)?;
        let li = line_index(db, file);
        let byte_offset = li.offset(Pos {
            line: lsp_pos.line,
            character: lsp_pos.character,
        });
        let index = node_span_index(db, file, fs, cats);
        let markdown = crate::capabilities::hover(db, file, fs, cats, byte_offset, &index)?;
        Some(lsp_types::Hover {
            contents: lsp_types::HoverContents::Markup(lsp_types::MarkupContent {
                kind: lsp_types::MarkupKind::Markdown,
                value: markdown,
            }),
            range: None,
        })
    })();

    match result {
        Some(hover) => Response::new_ok(req.id.clone(), hover),
        None => Response::new_ok(req.id.clone(), serde_json::Value::Null),
    }
}

/// Handle a `textDocument/documentSymbol` request.  Returns a `Response`
/// (result or null) without sending it — the caller dispatches the message.
fn handle_document_symbols(
    db: &Database,
    uri_to_file: &HashMap<String, SourceFile>,
    req: &lsp_server::Request,
) -> Response {
    let syms: Vec<lsp_types::DocumentSymbol> = (|| {
        let params: lsp_types::DocumentSymbolParams =
            serde_json::from_value(req.params.clone()).ok()?;
        let uri_str = params.text_document.uri.as_str().to_owned();
        let file = *uri_to_file.get(&uri_str)?;
        Some(crate::capabilities::document_symbols(db, file))
    })()
    .unwrap_or_default();

    let response = lsp_types::DocumentSymbolResponse::Nested(syms);
    Response::new_ok(req.id.clone(), response)
}

/// Handle a `workspace/symbol` request.  Returns a `Response` (result or
/// null) without sending it — the caller dispatches the message.
fn handle_workspace_symbols(db: &Database, fs: FileSet, req: &lsp_server::Request) -> Response {
    let query = (|| -> Option<String> {
        let params: lsp_types::WorkspaceSymbolParams =
            serde_json::from_value(req.params.clone()).ok()?;
        Some(params.query)
    })()
    .unwrap_or_default();

    let syms = crate::capabilities::workspace_symbols(db, fs, &query);
    let response = lsp_types::WorkspaceSymbolResponse::Flat(syms);
    Response::new_ok(req.id.clone(), response)
}

/// Handle a `textDocument/inlayHint` request.  Returns a `Response`
/// (result or null) without sending it — the caller dispatches the message.
fn handle_inlay_hints(
    db: &Database,
    uri_to_file: &HashMap<String, SourceFile>,
    fs: FileSet,
    cats: Catalogues,
    req: &lsp_server::Request,
) -> Response {
    let hints: Vec<lsp_types::InlayHint> = (|| {
        let params: lsp_types::InlayHintParams = serde_json::from_value(req.params.clone()).ok()?;
        let uri_str = params.text_document.uri.as_str().to_owned();
        let file = *uri_to_file.get(&uri_str)?;
        let li = line_index(db, file);
        let start_byte = li.offset(Pos {
            line: params.range.start.line,
            character: params.range.start.character,
        });
        let end_byte = li.offset(Pos {
            line: params.range.end.line,
            character: params.range.end.character,
        });
        Some(crate::capabilities::inlay_hints(
            db, file, fs, cats, start_byte, end_byte,
        ))
    })()
    .unwrap_or_default();

    Response::new_ok(req.id.clone(), hints)
}

/// Handle a `textDocument/definition` request.  Returns a `Response`
/// (scalar `Location` or null) without sending it — the caller dispatches.
fn handle_goto_definition(
    db: &Database,
    uri_to_file: &HashMap<String, SourceFile>,
    fs: FileSet,
    cats: Catalogues,
    req: &lsp_server::Request,
) -> Response {
    let result = (|| -> Option<lsp_types::GotoDefinitionResponse> {
        let params: lsp_types::GotoDefinitionParams =
            serde_json::from_value(req.params.clone()).ok()?;
        let uri_str = params
            .text_document_position_params
            .text_document
            .uri
            .as_str()
            .to_owned();
        let lsp_pos = params.text_document_position_params.position;
        let file = *uri_to_file.get(&uri_str)?;
        let li = line_index(db, file);
        let byte_offset = li.offset(Pos {
            line: lsp_pos.line,
            character: lsp_pos.character,
        });
        let index = node_span_index(db, file, fs, cats);
        let def_loc =
            crate::capabilities::goto_definition(db, file, fs, cats, byte_offset, &index)?;
        // Build the target URI from the DefLoc path.
        let target_uri_str = if def_loc.path.starts_with("file://") {
            def_loc.path.clone()
        } else {
            path_to_file_uri(&def_loc.path)
        };
        let target_uri = Uri::from_str(&target_uri_str).ok()?;
        // Build the target range: find the dep SourceFile and use its cached
        // line index (avoids a per-request LineIndex::new rebuild).
        let dep_file = fs
            .files(db)
            .iter()
            .copied()
            .find(|f| f.path(db) == def_loc.path);
        let target_li = dep_file
            .map(|f| line_index(db, f))
            .unwrap_or_else(|| crate::line_index::LineIndex::new(""));
        let start = target_li.position(def_loc.start);
        let end = target_li.position(def_loc.end);
        let range = lsp_types::Range::new(
            lsp_types::Position::new(start.line, start.character),
            lsp_types::Position::new(end.line, end.character),
        );
        let location = lsp_types::Location {
            uri: target_uri,
            range,
        };
        Some(lsp_types::GotoDefinitionResponse::Scalar(location))
    })();

    match result {
        Some(resp) => Response::new_ok(req.id.clone(), resp),
        None => Response::new_ok(req.id.clone(), serde_json::Value::Null),
    }
}

/// Handle a `textDocument/completion` request.  Returns a `Response`
/// (a `CompletionResponse::Array` of items or null) without sending it — the
/// caller dispatches the message.
fn handle_completion(
    db: &Database,
    uri_to_file: &HashMap<String, SourceFile>,
    fs: FileSet,
    cats: Catalogues,
    req: &lsp_server::Request,
) -> Response {
    let result = (|| -> Option<lsp_types::CompletionResponse> {
        let params: lsp_types::CompletionParams =
            serde_json::from_value(req.params.clone()).ok()?;
        let uri_str = params
            .text_document_position
            .text_document
            .uri
            .as_str()
            .to_owned();
        let lsp_pos = params.text_document_position.position;
        let file = *uri_to_file.get(&uri_str)?;
        let li = line_index(db, file);
        let byte_offset = li.offset(Pos {
            line: lsp_pos.line,
            character: lsp_pos.character,
        });
        let text = file.text(db);
        let prefix = member_prefix_at(text, byte_offset);
        let items = crate::capabilities::completion(db, file, fs, cats, byte_offset, prefix);
        Some(lsp_types::CompletionResponse::Array(items))
    })();

    match result {
        Some(resp) => Response::new_ok(req.id.clone(), resp),
        None => Response::new_ok(req.id.clone(), serde_json::Value::Null),
    }
}

/// Scan backwards from `byte` in `text` to detect a member-access prefix.
///
/// Returns `Some(ident)` when the character immediately before `byte` is `.`
/// and the characters before the `.` form a non-empty ASCII identifier
/// (`[A-Za-z0-9_]+` ending with `[A-Za-z_]`). Returns `None` otherwise (e.g.
/// bare identifier, start of line, or the `.` is not preceded by an ident).
///
/// Only the ASCII identifier characters are recognized; Unicode identifiers
/// are not supported by the current FlatPPL surface syntax.
pub(crate) fn member_prefix_at(text: &str, byte: u32) -> Option<String> {
    let byte = byte as usize;
    // There must be at least one byte before the cursor.
    if byte == 0 {
        return None;
    }
    let bytes = text.as_bytes();
    // The byte immediately before the cursor must be `.`.
    if bytes[byte - 1] != b'.' {
        return None;
    }
    // Scan backwards from the `.` to collect identifier bytes.
    let dot_pos = byte - 1;
    if dot_pos == 0 {
        return None;
    }
    let mut end = dot_pos;
    // Walk backwards while we see ASCII identifier chars.
    while end > 0 && is_ident_byte(bytes[end - 1]) {
        end -= 1;
    }
    let start = end;
    if start == dot_pos {
        // Nothing before the dot — no identifier.
        return None;
    }
    let ident = std::str::from_utf8(&bytes[start..dot_pos]).ok()?;
    if ident.is_empty() {
        return None;
    }
    Some(ident.to_string())
}

/// Cursor context for a completion request, derived textually (no parse, since
/// completion fires on often-unparseable mid-edit text).
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum CompletionContext {
    /// Immediately after `alias.` — member completion (unchanged behavior).
    Member(String),
    /// The nearest significant char left of the in-progress identifier is `~`,
    /// i.e. the cursor is in a tilde-binding RHS (a measure expression, §05).
    AfterTilde,
    /// Anything else: `=` RHS, call args, line start, fallback. Full set.
    Other,
}

/// Classify the completion context at `byte` in `text`.
#[allow(dead_code)]
pub(crate) fn completion_context(text: &str, byte: u32) -> CompletionContext {
    if let Some(alias) = member_prefix_at(text, byte) {
        return CompletionContext::Member(alias);
    }
    let bytes = text.as_bytes();
    let mut i = byte as usize;
    // Skip the in-progress identifier directly left of the cursor.
    while i > 0 && is_ident_byte(bytes[i - 1]) {
        i -= 1;
    }
    // Skip whitespace and newlines back to the nearest significant char.
    while i > 0 && matches!(bytes[i - 1], b' ' | b'\t' | b'\r' | b'\n') {
        i -= 1;
    }
    if i > 0 && bytes[i - 1] == b'~' {
        return CompletionContext::AfterTilde;
    }
    CompletionContext::Other
}

/// Return `true` for bytes that may appear in a FlatPPL identifier
/// (`[A-Za-z0-9_]`).
#[inline]
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::line_index::LineIndex;

    // ── member_prefix_at ─────────────────────────────────────────────────────

    #[test]
    fn member_prefix_at_detects_ident_before_dot() {
        // "x = e." — cursor at byte 6 (after '.'), ident is "e".
        assert_eq!(
            member_prefix_at("x = e.", 6),
            Some("e".to_string()),
            "cursor right after 'e.' must yield Some(\"e\")"
        );
    }

    #[test]
    fn member_prefix_at_no_dot_returns_none() {
        // "x = add" — cursor at byte 7, no dot.
        assert_eq!(
            member_prefix_at("x = add", 7),
            None,
            "cursor after plain ident must yield None"
        );
    }

    #[test]
    fn member_prefix_at_dot_at_start_returns_none() {
        // ".foo" — dot at byte 0, nothing before it.
        assert_eq!(member_prefix_at(".foo", 1), None);
    }

    #[test]
    fn member_prefix_at_multi_char_ident() {
        // "mymod.x" — cursor at byte 7.
        assert_eq!(
            member_prefix_at("a = mymod.", 10),
            Some("mymod".to_string()),
        );
    }

    // ── completion_context ────────────────────────────────────────────────────
    #[test]
    fn completion_context_after_dot_is_member() {
        // "a = mymod." — cursor at byte 10.
        assert!(matches!(
            completion_context("a = mymod.", 10),
            CompletionContext::Member(ref s) if s == "mymod"
        ));
    }

    #[test]
    fn completion_context_after_tilde_empty() {
        // "x ~ " — cursor at byte 4, right after "~ ".
        assert!(matches!(
            completion_context("x ~ ", 4),
            CompletionContext::AfterTilde
        ));
    }

    #[test]
    fn completion_context_after_tilde_partial_ident() {
        // "x ~ Nor" — cursor at byte 7, mid-distribution-name.
        assert!(matches!(
            completion_context("x ~ Nor", 7),
            CompletionContext::AfterTilde
        ));
    }

    #[test]
    fn completion_context_after_eq_is_other() {
        // "x = " — cursor at byte 4. v1 keeps `=` as Other (full set).
        assert!(matches!(
            completion_context("x = ", 4),
            CompletionContext::Other
        ));
    }

    #[test]
    fn completion_context_line_start_is_other() {
        // "x" — cursor at byte 1, typing a binding name.
        assert!(matches!(
            completion_context("x", 1),
            CompletionContext::Other
        ));
    }

    #[test]
    fn completion_context_tilde_across_newline() {
        // multi-line: "obs ~\n  Nor" — cursor at byte 10, ident "Nor" after newline+indent.
        let text = "obs ~\n  Nor";
        assert!(matches!(
            completion_context(text, text.len() as u32),
            CompletionContext::AfterTilde
        ));
    }

    // ── position_to_byte ──────────────────────────────────────────────────────

    fn position_to_byte(text: &str, line: u32, character: u32) -> u32 {
        let li = LineIndex::new(text);
        li.offset(Pos { line, character })
    }

    #[test]
    fn position_to_byte_first_line() {
        // Single-line text: character maps directly to byte offset.
        let text = "hello world";
        assert_eq!(position_to_byte(text, 0, 0), 0);
        assert_eq!(position_to_byte(text, 0, 5), 5);
        assert_eq!(position_to_byte(text, 0, 11), 11); // EOF
    }

    #[test]
    fn position_to_byte_second_line() {
        // "ab\ncde": line 1 starts at byte 3.
        let text = "ab\ncde";
        assert_eq!(position_to_byte(text, 1, 0), 3); // 'c'
        assert_eq!(position_to_byte(text, 1, 2), 5); // 'e'
    }

    #[test]
    fn position_to_byte_utf16() {
        // é (U+00E9): 2 UTF-8 bytes, 1 UTF-16 code unit.
        // "éx": 'x' is at byte 2, UTF-16 column 1.
        let text = "éx";
        assert_eq!(position_to_byte(text, 0, 0), 0); // 'é' at byte 0
        assert_eq!(position_to_byte(text, 0, 1), 2); // 'x' at byte 2
    }

    // ── catalogue_sources_from_params ─────────────────────────────────────────

    fn parse_catalogue_sources(raw: serde_json::Value) -> Vec<String> {
        let params: lsp_types::InitializeParams = serde_json::from_value(raw).unwrap_or_default();
        catalogue_sources_from_params(&params)
    }

    #[test]
    fn catalogue_strings_parsed_from_init_options() {
        let raw = serde_json::json!({
            "capabilities": {},
            "initializationOptions": {
                "catalogues": ["Catalogue(base:[],modules:[])", "Catalogue(base:[],modules:[])"]
            }
        });
        let cats = parse_catalogue_sources(raw);
        assert_eq!(cats.len(), 2);
        assert!(cats[0].contains("Catalogue"));
    }

    #[test]
    fn catalogue_strings_absent_gives_empty() {
        let raw = serde_json::json!({ "capabilities": {} });
        let cats = parse_catalogue_sources(raw);
        assert!(cats.is_empty());
    }

    // ── server_capabilities ───────────────────────────────────────────────────

    #[test]
    fn capabilities_advertise_hover_and_full_sync() {
        let caps = server_capabilities();
        assert_eq!(
            caps.hover_provider,
            Some(HoverProviderCapability::Simple(true))
        );
        assert_eq!(
            caps.text_document_sync,
            Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL))
        );
        assert_eq!(
            caps.document_symbol_provider,
            Some(OneOf::Left(true)),
            "server must advertise documentSymbol capability"
        );
        assert_eq!(
            caps.workspace_symbol_provider,
            Some(OneOf::Left(true)),
            "server must advertise workspaceSymbol capability"
        );
        assert_eq!(
            caps.inlay_hint_provider,
            Some(OneOf::Left(true)),
            "server must advertise inlayHint capability"
        );
        assert_eq!(
            caps.definition_provider,
            Some(OneOf::Left(true)),
            "server must advertise definition capability"
        );
        assert!(
            caps.completion_provider.is_some(),
            "server must advertise completion capability"
        );
        let comp_opts = caps.completion_provider.as_ref().unwrap();
        assert_eq!(
            comp_opts.trigger_characters.as_deref(),
            Some(
                [
                    ".".to_string(),
                    "~".to_string(),
                    "=".to_string(),
                    "(".to_string(),
                    ",".to_string(),
                ]
                .as_slice()
            ),
            "completion trigger characters must be '.', '~', '=', '(', ','"
        );
    }

    // ── file_uri_to_path ──────────────────────────────────────────────────────

    #[test]
    fn file_uri_to_path_plain() {
        assert_eq!(
            file_uri_to_path("file:///tmp/a.flatppl"),
            Some("/tmp/a.flatppl".to_owned())
        );
    }

    #[test]
    fn file_uri_to_path_percent_decoded() {
        // Spaces encoded as %20 must be decoded to real spaces.
        assert_eq!(
            file_uri_to_path("file:///tmp/My%20Project/a.flatppl"),
            Some("/tmp/My Project/a.flatppl".to_owned())
        );
    }

    #[test]
    fn file_uri_to_path_rejects_non_file() {
        assert_eq!(file_uri_to_path("https://example.com/foo"), None);
    }

    // ── In-memory round-trip: didOpen → publishDiagnostics ───────────────────
    //
    // Uses `Connection::memory()` to drive a minimal interaction: open a
    // FlatPPL file with a parse error and verify that `publishDiagnostics`
    // carries at least one diagnostic.

    #[test]
    fn did_open_triggers_publish_diagnostics() {
        use lsp_server::{Connection, Message};
        use lsp_types::notification::{DidOpenTextDocument, Notification as _};

        // lsp_server::Connection::memory() gives two connected ends.
        let (client_conn, server_conn) = Connection::memory();

        // We need to bypass the initialize handshake and call `run` directly
        // with a minimal params value. Run the server in a thread.
        let server_thread = std::thread::spawn(move || {
            // Minimal init params: no workspace folders, no catalogues.
            let init_params = serde_json::json!({ "capabilities": {} });
            run(server_conn, init_params).expect("server loop failed");
        });

        // Send didOpen with a parse-error FlatPPL file.
        let did_open_params = lsp_types::DidOpenTextDocumentParams {
            text_document: lsp_types::TextDocumentItem {
                uri: Uri::from_str("file:///tmp/test.flatppl").unwrap(),
                language_id: "flatppl".into(),
                version: 1,
                text: "x = (((   -- syntax error".into(),
            },
        };
        let note =
            lsp_server::Notification::new(DidOpenTextDocument::METHOD.to_owned(), did_open_params);
        client_conn
            .sender
            .send(Message::Notification(note))
            .unwrap();

        // Receive the publishDiagnostics notification from the server.
        let msg = client_conn
            .receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("timed out waiting for publishDiagnostics");

        let Message::Notification(publish) = msg else {
            panic!("expected a Notification, got: {msg:?}");
        };
        assert_eq!(
            publish.method,
            lsp_types::notification::PublishDiagnostics::METHOD
        );
        let params: PublishDiagnosticsParams =
            serde_json::from_value(publish.params).expect("valid PublishDiagnosticsParams");
        assert!(
            !params.diagnostics.is_empty(),
            "parse error must produce at least one diagnostic"
        );

        // Send shutdown + exit to stop the server thread.
        let shutdown_req = lsp_server::Request::new(
            lsp_server::RequestId::from(1i32),
            "shutdown".into(),
            serde_json::Value::Null,
        );
        client_conn
            .sender
            .send(Message::Request(shutdown_req))
            .unwrap();
        // Wait for the shutdown response.
        let _resp = client_conn
            .receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .ok();
        // Send exit notification.
        let exit_note = lsp_server::Notification::new("exit".into(), serde_json::Value::Null);
        client_conn
            .sender
            .send(Message::Notification(exit_note))
            .unwrap();

        server_thread.join().expect("server thread panicked");
    }

    // ── cross-file hover resolves after mid-session didOpen ──────────────────
    //
    // Regression guard for the stale-FileSet bug: the server starts with an
    // EMPTY workspace (no root, no initial scan). Two files are opened via
    // `didOpen` mid-session — `helpers.flatppl` first, then `model.flatppl`
    // which loads it. A hover on `model`'s cross-file reference (`h.center`)
    // must resolve to a non-null response that contains a type token.
    //
    // Without the in-place `set_files` fix the outer `fs` stays empty, so
    // `import_bundle` finds no files to resolve against and the cross-file ref
    // remains unresolved → hover returns null.

    #[test]
    fn cross_file_hover_resolves_after_did_open() {
        use lsp_server::{Connection, Message};
        use lsp_types::notification::{DidOpenTextDocument, Notification as _};
        use lsp_types::request::{HoverRequest, Request as _};

        let (client_conn, server_conn) = Connection::memory();

        let server_thread = std::thread::spawn(move || {
            // Empty workspace: no rootUri, no workspace folders.
            let init_params = serde_json::json!({ "capabilities": {} });
            run(server_conn, init_params).expect("server loop failed");
        });

        let send_open = |uri: &str, text: &str| {
            let params = lsp_types::DidOpenTextDocumentParams {
                text_document: lsp_types::TextDocumentItem {
                    uri: Uri::from_str(uri).unwrap(),
                    language_id: "flatppl".into(),
                    version: 1,
                    text: text.into(),
                },
            };
            let note =
                lsp_server::Notification::new(DidOpenTextDocument::METHOD.to_owned(), params);
            client_conn
                .sender
                .send(Message::Notification(note))
                .unwrap();
        };

        // Open helpers first so it is registered in uri_to_file before model.
        send_open("file:///tmp/helpers.flatppl", "center = elementof(reals)\n");
        // Drain the publishDiagnostics notification for helpers.
        let _ = client_conn
            .receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("expected publishDiagnostics for helpers");

        // Open model, which loads helpers.
        send_open(
            "file:///tmp/model.flatppl",
            "h = load_module(\"helpers.flatppl\")\nv = add(h.center, 1.0)\n",
        );
        // Drain publishDiagnostics for helpers (re-emitted) and model.
        for _ in 0..2 {
            let _ = client_conn
                .receiver
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("expected publishDiagnostics after model open");
        }

        // Send a hover on line 1 of model.flatppl.
        // Text: "h = load_module(\"helpers.flatppl\")\nv = add(h.center, 1.0)\n"
        // Line 1: "v = add(h.center, 1.0)\n" — the `add(...)` call starts at
        // char 4 (byte 39 in the file). The whole `add(h.center, 1.0)` expression
        // infers as Scalar(Real) via cross-file resolution, so any character inside
        // it returns a typed hover. We use char 4 (`a` of `add`) which is reliably
        // typed as the call's result.
        let hover_params = lsp_types::HoverParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: Uri::from_str("file:///tmp/model.flatppl").unwrap(),
                },
                position: lsp_types::Position {
                    line: 1,
                    character: 4, // 'a' of 'add' — within the typed call expression
                },
            },
            work_done_progress_params: Default::default(),
        };
        let hover_req = lsp_server::Request::new(
            lsp_server::RequestId::from(42i32),
            HoverRequest::METHOD.to_owned(),
            serde_json::to_value(hover_params).unwrap(),
        );
        client_conn
            .sender
            .send(Message::Request(hover_req))
            .unwrap();

        let resp_msg = client_conn
            .receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("timed out waiting for hover response");

        let Message::Response(resp) = resp_msg else {
            panic!("expected a Response, got: {resp_msg:?}");
        };
        assert!(
            resp.error.is_none(),
            "hover response must not be an error: {:?}",
            resp.error
        );
        let result = resp.result.expect("hover result must be non-null");
        assert!(
            result != serde_json::Value::Null,
            "hover on cross-file ref must return non-null (FileSet was stale without fix)"
        );
        // The hover markdown must mention "type".
        let result_str = result.to_string().to_lowercase();
        assert!(
            result_str.contains("type"),
            "hover result must contain 'type'; got: {result_str}"
        );

        // Shutdown.
        let shutdown_req = lsp_server::Request::new(
            lsp_server::RequestId::from(99i32),
            "shutdown".into(),
            serde_json::Value::Null,
        );
        client_conn
            .sender
            .send(Message::Request(shutdown_req))
            .unwrap();
        let _ = client_conn
            .receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .ok();
        let exit_note = lsp_server::Notification::new("exit".into(), serde_json::Value::Null);
        client_conn
            .sender
            .send(Message::Notification(exit_note))
            .unwrap();

        server_thread.join().expect("server thread panicked");
    }

    // ── didChangeWatchedFiles: on-disk create/change picked up ───────────────
    //
    // Scenario: the server starts with an empty workspace. A `.flatppl` file is
    // written to a temp path. The client sends `workspace/didChangeWatchedFiles`
    // with a CREATED event for that file's `file://` URI. The test then sends a
    // `documentSymbol` request for that URI and asserts the server returns at
    // least one symbol — proving it read the file from disk.
    //
    // A CHANGED event for the same URI (with updated content) is then sent, and
    // a second `documentSymbol` request asserts the updated symbol name is
    // visible, proving the disk-reload path works.

    #[test]
    fn watched_file_created_and_changed_picked_up() {
        use lsp_server::{Connection, Message};
        use lsp_types::notification::Notification as _;
        use lsp_types::request::{DocumentSymbolRequest, Request as _};

        // Write a temp .flatppl file with a known binding.
        let tmp_path = std::env::temp_dir().join(format!(
            "flatppl_lsp_watched_{}.flatppl",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&tmp_path, "alpha = elementof(reals)\n").unwrap();
        let tmp_uri_str = format!("file://{}", tmp_path.display());

        let (client_conn, server_conn) = Connection::memory();
        let server_thread = std::thread::spawn(move || {
            let init_params = serde_json::json!({ "capabilities": {} });
            run(server_conn, init_params).expect("server loop failed");
        });

        // Helper: send a notification.
        let send_note = |method: &str, params: serde_json::Value| {
            let note = lsp_server::Notification::new(method.to_owned(), params);
            client_conn
                .sender
                .send(Message::Notification(note))
                .unwrap();
        };

        // Helper: drain messages until a non-publishDiagnostics message arrives,
        // returning that message. Discards any publishDiagnostics notifications
        // that the server emits after a watched-file event.
        let drain_to_response = || loop {
            let msg = client_conn
                .receiver
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("timed out waiting for response");
            match &msg {
                Message::Notification(n)
                    if n.method == lsp_types::notification::PublishDiagnostics::METHOD =>
                {
                    continue;
                }
                _ => return msg,
            }
        };

        // Send a CREATED watched-file event. The server reads the file from disk
        // and emits a publishDiagnostics notification for it (empty, valid file).
        let dcwf_params = serde_json::json!({
            "changes": [{ "uri": tmp_uri_str, "type": 1 }]  // 1 = CREATED
        });
        send_note(DidChangeWatchedFiles::METHOD, dcwf_params);

        // Send a documentSymbol request; drain any publishDiagnostics first.
        let ds_params = serde_json::json!({
            "textDocument": { "uri": tmp_uri_str }
        });
        // Enqueue the request then drain to get the response (past any diagnostics).
        {
            let req = lsp_server::Request::new(
                lsp_server::RequestId::from(10i32),
                DocumentSymbolRequest::METHOD.to_owned(),
                ds_params.clone(),
            );
            client_conn.sender.send(Message::Request(req)).unwrap();
        }
        let resp_msg = drain_to_response();
        let Message::Response(resp) = resp_msg else {
            panic!("expected Response, got: {resp_msg:?}");
        };
        assert!(
            resp.error.is_none(),
            "documentSymbol must not error: {:?}",
            resp.error
        );
        let result = resp.result.expect("documentSymbol result must be present");
        // The server should have loaded "alpha = elementof(reals)" → at least one symbol.
        let syms = result.to_string();
        assert!(
            syms.contains("alpha"),
            "symbol 'alpha' must appear after CREATED watched-file event; got: {syms}"
        );

        // Now update the file on disk with a new binding name.
        std::fs::write(&tmp_path, "beta = elementof(reals)\n").unwrap();

        // Send a CHANGED watched-file event.
        let dcwf_changed = serde_json::json!({
            "changes": [{ "uri": tmp_uri_str, "type": 2 }]  // 2 = CHANGED
        });
        send_note(DidChangeWatchedFiles::METHOD, dcwf_changed);

        // Query symbols again — must now show "beta".
        {
            let req = lsp_server::Request::new(
                lsp_server::RequestId::from(11i32),
                DocumentSymbolRequest::METHOD.to_owned(),
                ds_params,
            );
            client_conn.sender.send(Message::Request(req)).unwrap();
        }
        let resp_msg2 = drain_to_response();
        let Message::Response(resp2) = resp_msg2 else {
            panic!("expected Response, got: {resp_msg2:?}");
        };
        assert!(
            resp2.error.is_none(),
            "second documentSymbol must not error: {:?}",
            resp2.error
        );
        let result2 = resp2
            .result
            .expect("second documentSymbol result must be present");
        let syms2 = result2.to_string();
        assert!(
            syms2.contains("beta"),
            "symbol 'beta' must appear after CHANGED watched-file event; got: {syms2}"
        );

        // Cleanup temp file.
        let _ = std::fs::remove_file(&tmp_path);

        // Shutdown.
        let shutdown_req = lsp_server::Request::new(
            lsp_server::RequestId::from(99i32),
            "shutdown".into(),
            serde_json::Value::Null,
        );
        client_conn
            .sender
            .send(Message::Request(shutdown_req))
            .unwrap();
        let _ = client_conn
            .receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .ok();
        let exit_note = lsp_server::Notification::new("exit".into(), serde_json::Value::Null);
        client_conn
            .sender
            .send(Message::Notification(exit_note))
            .unwrap();
        server_thread.join().expect("server thread panicked");
    }

    // ── affected_files / ANALYZE_RUNS tests ──────────────────────────────────
    //
    // 3-file workspace: B is a leaf; A does `load_module` of B; C is
    // independent (imports neither A nor B).

    fn make_abc_workspace() -> (
        crate::db::Database,
        SourceFile,
        SourceFile,
        SourceFile,
        crate::db::FileSet,
        HashMap<String, SourceFile>,
    ) {
        let db = crate::db::Database::default();
        let b = SourceFile::new(
            &db,
            "/tmp/b.flatppl".to_string(),
            "leaf = elementof(reals)\n".to_string(),
        );
        let a = SourceFile::new(
            &db,
            "/tmp/a.flatppl".to_string(),
            "b = load_module(\"/tmp/b.flatppl\")\nv = add(b.leaf, 1.0)\n".to_string(),
        );
        let c = SourceFile::new(
            &db,
            "/tmp/c.flatppl".to_string(),
            "x = add(1, 2)\n".to_string(),
        );
        let fs = crate::db::FileSet::new(&db, vec![a, b, c]);
        let mut uri_to_file: HashMap<String, SourceFile> = HashMap::new();
        uri_to_file.insert("file:///tmp/a.flatppl".to_string(), a);
        uri_to_file.insert("file:///tmp/b.flatppl".to_string(), b);
        uri_to_file.insert("file:///tmp/c.flatppl".to_string(), c);
        (db, a, b, c, fs, uri_to_file)
    }

    // ── sync_file_set membership guard tests ─────────────────────────────────

    /// A delete+create batch with unchanged count must still update the FileSet.
    ///
    /// Start with FileSet = {A, B}. Build a `uri_to_file` map representing {A, C}
    /// (B removed, C added — same count 2). Call `sync_file_set` and assert
    /// `fs.files(db)` now contains A and C and NOT B. This is the MEASURED proof
    /// that equal count but changed membership still triggers the update.
    #[test]
    fn sync_file_set_delete_create_batch_updates_membership() {
        use crate::db::{Database, FileSet, SourceFile};

        let mut db = Database::default();
        let a = SourceFile::new(&db, "/tmp/a.flatppl".to_string(), "a = 1".to_string());
        let b = SourceFile::new(&db, "/tmp/b.flatppl".to_string(), "b = 2".to_string());
        let fs = FileSet::new(&db, vec![a, b]);

        // Simulate a didChangeWatchedFiles batch: B deleted, C created (same count).
        let c = SourceFile::new(&db, "/tmp/c.flatppl".to_string(), "c = 3".to_string());
        let mut uri_to_file: HashMap<String, SourceFile> = HashMap::new();
        uri_to_file.insert("file:///tmp/a.flatppl".to_string(), a);
        uri_to_file.insert("file:///tmp/c.flatppl".to_string(), c);

        // Before the call, fs still holds {A, B}.
        assert_eq!(
            fs.files(&db).len(),
            2,
            "initial FileSet must have 2 members"
        );

        sync_file_set(&mut db, fs, &uri_to_file);

        let current: Vec<SourceFile> = fs.files(&db).to_vec();
        assert_eq!(
            current.len(),
            2,
            "FileSet must still have 2 members after delete+create"
        );
        assert!(
            current.contains(&a),
            "A must be in the updated FileSet; got {current:?}"
        );
        assert!(
            current.contains(&c),
            "C (newly created) must be in the updated FileSet; got {current:?}"
        );
        assert!(
            !current.contains(&b),
            "B (deleted) must NOT be in the updated FileSet; got {current:?}"
        );
    }

    /// A pure text edit must NOT cause `sync_file_set` to bump the FileSet input.
    ///
    /// After warming the FileSet with {A}, update A's text via `set_text` (a
    /// membership-unchanged edit), then call `sync_file_set`. The FileSet must
    /// still contain exactly A. This is the guard against per-keystroke revision
    /// churn: membership is identical, so the salsa setter is skipped.
    #[test]
    fn sync_file_set_skips_update_on_pure_text_edit() {
        use crate::db::{Database, FileSet, SourceFile};
        use salsa::Setter;

        let mut db = Database::default();
        let a = SourceFile::new(&db, "/tmp/a.flatppl".to_string(), "a = 1".to_string());
        let fs = FileSet::new(&db, vec![a]);
        let mut uri_to_file: HashMap<String, SourceFile> = HashMap::new();
        uri_to_file.insert("file:///tmp/a.flatppl".to_string(), a);

        // Pure text edit: membership unchanged.
        a.set_text(&mut db).to("a = 99".to_string());

        // sync_file_set must not panic and must leave the membership intact.
        sync_file_set(&mut db, fs, &uri_to_file);

        let current: Vec<SourceFile> = fs.files(&db).to_vec();
        assert_eq!(current, vec![a], "FileSet must still contain only A");
    }

    /// `affected_files(changed=B)` must include A and B (A imports B) but must
    /// exclude C (C imports neither A nor B).
    #[test]
    fn affected_files_excludes_non_importers() {
        let (db, _a, b, _c, fs, uri_to_file) = make_abc_workspace();

        let affected: std::collections::HashSet<String> = affected_files(&db, fs, &uri_to_file, b)
            .into_iter()
            .map(|(u, _)| u)
            .collect();

        assert!(
            affected.contains("file:///tmp/b.flatppl"),
            "changed file B must be in affected set; got {affected:?}"
        );
        assert!(
            affected.contains("file:///tmp/a.flatppl"),
            "A imports B, so A must be in affected set; got {affected:?}"
        );
        assert!(
            !affected.contains("file:///tmp/c.flatppl"),
            "C is independent; must NOT be in affected set; got {affected:?}"
        );
    }

    /// Editing B (via `set_text`) must NOT invalidate C's `analyze` cache.
    ///
    /// After warming A, B, C, we reset `ANALYZE_RUNS`, edit B's text, and
    /// re-run `analyze` for only the affected set (A, B). Running `analyze(C)`
    /// afterward must not increment the counter — C's inputs are unchanged so
    /// salsa serves it from cache.
    #[test]
    fn editing_a_file_does_not_reanalyze_independent_files() {
        use crate::queries::{ANALYZE_RUNS, analyze};
        use salsa::Setter;

        let (mut db, a, b, c, fs, _uri_to_file) = make_abc_workspace();
        let cats = crate::db::Catalogues::new(&db, Vec::new());

        // Warm: analyze all three so the revision is established.
        let _ = analyze(&db, a, fs, cats);
        let _ = analyze(&db, b, fs, cats);
        let _ = analyze(&db, c, fs, cats);

        // Reset the counter, then edit B's text (a pure text change, not a
        // membership change).
        ANALYZE_RUNS.with(|c| c.set(0));
        b.set_text(&mut db)
            .to("leaf = elementof(reals)\nextra = add(leaf, 2.0)\n".to_string());

        // Simulate what the fixed didChange arm does: re-analyze only the
        // affected set (B and A, which imports B).
        let _ = analyze(&db, b, fs, cats);
        let _ = analyze(&db, a, fs, cats);
        let runs_after_ab = ANALYZE_RUNS.with(|c| c.get());
        assert_eq!(
            runs_after_ab, 2,
            "editing B should recompute analyze for B and A (its importer); got {runs_after_ab}"
        );

        // Now run analyze(C) — C's inputs are unchanged, so salsa must serve it
        // from cache without running the body again.
        ANALYZE_RUNS.with(|c| c.set(0));
        let _ = analyze(&db, c, fs, cats);
        let runs_c = ANALYZE_RUNS.with(|c| c.get());
        assert_eq!(
            runs_c, 0,
            "C is independent of B; editing B must NOT invalidate C's analyze cache \
             (ANALYZE_RUNS incremented {runs_c} times for C, expected 0)"
        );
    }
}
