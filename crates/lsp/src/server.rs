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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, select};
use lsp_server::{Connection, ErrorCode, Message, RequestId, Response};
use lsp_types::{
    CompletionOptions, HoverProviderCapability, OneOf, PublishDiagnosticsParams,
    ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind, Uri,
    notification::{
        DidChangeTextDocument, DidChangeWatchedFiles, DidCloseTextDocument, DidOpenTextDocument,
        Notification as _, PublishDiagnostics,
    },
    request::{
        Completion, DocumentSymbolRequest, GotoDefinition, HoverRequest, InlayHintRequest,
        PrepareRenameRequest, References, Rename, Request as _, SignatureHelpRequest,
        WorkspaceSymbolRequest,
    },
};

use crate::db::{Catalogues, Database, FileSet, SourceFile};
use crate::line_index::Pos;
use crate::outbound::{Outbound, Outgoing};
use crate::queries::{import_bundle, line_index, node_span_index};

// ── run ─────────────────────────────────────────────────────────────────────

/// The client connection: the receive half, and the writer this server owns.
///
/// The write half is an [`Outbound`] rather than `lsp_server`'s sender so a
/// response body that is already JSON can go straight to the wire — see the
/// [`crate::outbound`] module for why that matters.
pub struct Wire {
    pub receiver: Receiver<Message>,
    pub out: Outbound,
}

/// Answer the initialize handshake on `wire`, returning the client's
/// `InitializeParams` for [`run_on`].
///
/// Replaces `Connection::initialize`, which writes through `lsp_server`'s own
/// sender. Only one writer may hold the output stream, so a server whose
/// [`Outbound`] owns stdout must answer the handshake through it too.
pub fn handshake(
    wire: &Wire,
    server_capabilities: serde_json::Value,
) -> Result<serde_json::Value, Box<dyn std::error::Error + Sync + Send>> {
    loop {
        let msg = wire.receiver.recv()?;
        match msg {
            Message::Request(req) if req.method == "initialize" => {
                let result = serde_json::json!({ "capabilities": server_capabilities });
                wire.out
                    .send(Message::Response(Response::new_ok(req.id, result)))?;
                // The client must acknowledge before any other traffic. A
                // request before `initialized` is a client bug, and answering
                // it against a half-built server is worse than refusing it.
                match wire.receiver.recv()? {
                    Message::Notification(n) if n.method == "initialized" => return Ok(req.params),
                    other => {
                        return Err(
                            format!("expected initialized notification, got {other:?}").into()
                        );
                    }
                }
            }
            // LSP 3.17: every request before `initialize` is answered
            // `ServerNotInitialized`, and every notification but `exit` is
            // dropped.
            Message::Request(req) => {
                wire.out.send(Message::Response(Response::new_err(
                    req.id,
                    ErrorCode::ServerNotInitialized as i32,
                    format!("expected an initialize request, got {}", req.method),
                )))?;
            }
            Message::Notification(n) if n.method == "exit" => {
                return Err("client exited before initialize".into());
            }
            Message::Notification(_) | Message::Response(_) => {}
        }
    }
}

/// Answer a `shutdown` request and wait for the `exit` notification.
///
/// `Ok(false)` for any other request. Replaces `Connection::handle_shutdown`
/// for the same reason [`handshake`] replaces `Connection::initialize`.
fn handle_shutdown(
    wire: &Wire,
    req: &lsp_server::Request,
) -> Result<bool, Box<dyn std::error::Error + Sync + Send>> {
    if req.method != "shutdown" {
        return Ok(false);
    }
    wire.out
        .send(Message::Response(Response::new_ok(req.id.clone(), ())))?;
    // LSP 3.17 has the client send `exit` after the shutdown response. The
    // bound is `lsp_server`'s: a client that never sends it must not wedge the
    // process.
    match wire
        .receiver
        .recv_timeout(std::time::Duration::from_secs(30))
    {
        Ok(Message::Notification(n)) if n.method == "exit" => Ok(true),
        // Anything else means the client kept talking after shutdown. Exit
        // anyway: the server has promised to stop.
        Ok(_) | Err(_) => Ok(true),
    }
}

/// Drive the FlatPPL LSP event loop over an `lsp_server::Connection`.
///
/// For a host that drives the server over channels, including the integration
/// tests. The initialize handshake must have completed before this call, and
/// `init_params` is the raw `serde_json::Value` it returned. A `documentSymbol`
/// answer is materialised back into a `serde_json::Value` for the channel,
/// which the stdio server avoids — see [`crate::outbound`].
pub fn run(
    connection: Connection,
    init_params: serde_json::Value,
) -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
    let Connection { sender, receiver } = connection;
    let (out, forwarder) = Outbound::to_messages(sender);
    let result = run_on(Wire { receiver, out }, init_params);
    // The forwarder ends when the last `Outbound` clone drops, which `run_on`
    // has already done by returning.
    let _ = forwarder.join();
    result
}

/// Drive the FlatPPL LSP event loop.
///
/// The initialize handshake must have completed before this call (see
/// [`handshake`]), and `init_params` is the raw `serde_json::Value` it
/// returned.
pub fn run_on(
    connection: Wire,
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

    // Directory names pruned from the workspace scan and from disk-created
    // files (`didChangeWatchedFiles`). `node_modules` is always excluded
    // regardless of client config — a dependency's test fixtures are never
    // diagnostics material. The client may add more via
    // `initializationOptions.diagnosticsExclude` (e.g. `"fixtures"` — many
    // fixture corpora are deliberately invalid models, not user errors).
    let mut excluded_dir_names = excluded_dir_names_from_params(&params);
    excluded_dir_names.insert("node_modules".to_owned());

    let mut uri_to_file: HashMap<String, SourceFile> = HashMap::new();

    // Remote `load_module` / `load_data` URL sources, fed by the editor client
    // via the `flatppl/urlSources` notification and keyed by the URL string.
    // The LSP NEVER fetches: it can't obtain URL-trust approval over the protocol
    // (spec §sec:url-cache requires interactive approval / non-interactive
    // refusal), and the client is already the sole fetcher+truster. These are
    // merged into the `FileSet` for resolution but kept OUT of `uri_to_file`, so
    // they participate in cross-module inference yet never get diagnostics
    // published (they are dependency content, not editor buffers).
    let mut url_to_file: HashMap<String, SourceFile> = HashMap::new();

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
            scan_dir(
                Path::new(&path),
                &mut db,
                &mut uri_to_file,
                &excluded_dir_names,
            );
        }
    }

    let fs = build_fileset(&db, &uri_to_file, &url_to_file);

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
    // Worker replies. Unbounded is safe because a queued reply is a shared
    // handle or a small error, never a serialised result — see `Reply`.
    let (result_tx, result_rx) = crossbeam_channel::unbounded::<Reply>();
    // Substituted for `result_rx` while the writer's backlog is full, so the
    // `select!` below waits on the client and the debounce alone.
    let no_results: crossbeam_channel::Receiver<Reply> = crossbeam_channel::never();
    let pool = crate::pool::Pool::new(
        std::thread::available_parallelism()
            .map(|n| n.get().min(4))
            .unwrap_or(2),
        crate::pool::QUEUE_CAPACITY,
    );
    // Requests dispatched and not yet answered. JSON-RPC 2.0 §5 requires
    // exactly one response per request, and four paths can produce one (the
    // worker, a `$/cancelRequest`, a queue drain, a full queue), so the main
    // thread arbitrates: a response goes out only if its id is still here.
    let mut pending: HashSet<RequestId> = HashSet::new();
    // Ids a client `$/cancelRequest` named. A job takes its own id out of this
    // before doing any work, so a cancelled request costs nothing; the main
    // thread has already answered it `RequestCancelled`. Each id is removed by
    // whichever side reads it, so the set holds only cancels still waiting for
    // their request — see `note_cancelled` for the bound on those.
    let cancelled: Arc<Mutex<HashSet<RequestId>>> = Arc::new(Mutex::new(HashSet::new()));
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
        // A reply that is not a `Reply::Raw` still carries a
        // `serde_json::Value`, so hold off while the writer has
        // `OUTBOUND_BACKLOG` messages to get through. An untaken reply sits in
        // the channel, and the request behind it in the pool's queue.
        let throttled = connection.out.backlog() >= OUTBOUND_BACKLOG;
        let results = if throttled { &no_results } else { &result_rx };
        let debounce_wait = diag_deadline.map(|d| d.saturating_duration_since(Instant::now()));
        let wait = match (throttled, debounce_wait) {
            (true, Some(t)) => Some(t.min(THROTTLE_POLL)),
            (true, None) => Some(THROTTLE_POLL),
            (false, d) => d,
        };
        let selected: Option<Result<Message, crossbeam_channel::RecvError>> = match wait {
            Some(t) => select! {
                recv(connection.receiver) -> m => Some(m),
                recv(results) -> r => {
                    if let Ok(reply) = r { answer(&connection, &mut pending, reply)?; }
                    None
                }
                default(t) => {
                    // Either the debounce fired or the throttle poll came round.
                    // Publish only for the former.
                    if diag_deadline.is_some_and(|d| Instant::now() >= d) {
                        flush_dirty(
                            &connection, &db, fs, cats, &uri_to_file, &doc_versions, &mut dirty,
                        )?;
                        diag_deadline = None;
                    }
                    None
                }
            },
            None => select! {
                recv(connection.receiver) -> m => Some(m),
                recv(results) -> r => {
                    if let Ok(reply) = r { answer(&connection, &mut pending, reply)?; }
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
                        release_stale_requests(&connection, &pool, &mut pending)?;
                        // Mark as editor-managed so watched-file CHANGED events
                        // skip it (editor content takes precedence over on-disk).
                        editor_open_uris.insert(uri_str.clone());
                        doc_versions.insert(uri_str.clone(), p.text_document.version);
                        upsert_file(&mut db, &mut uri_to_file, uri_str, text);
                        // Update the shared FileSet only when the file SET membership
                        // changes (a new open always adds a file, so this fires).
                        sync_file_set(&mut db, fs, &uri_to_file, &url_to_file);
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
                        release_stale_requests(&connection, &pool, &mut pending)?;
                        // Full sync — take last content change.
                        if let Some(change) = p.content_changes.into_iter().last() {
                            upsert_file(&mut db, &mut uri_to_file, uri_str.clone(), change.text);
                        }
                        // Guard the FileSet salsa input: a pure text edit leaves
                        // membership unchanged, so no revision bump is needed.
                        sync_file_set(&mut db, fs, &uri_to_file, &url_to_file);
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
                        // LSP 3.17 `textDocument/didClose`: "The document's truth
                        // now exists where the document's uri points to (e.g. if
                        // the document's uri is a file uri the truth now exists on
                        // disk)." So the unsaved buffer must go: re-read the file
                        // and put the disk text back into the salsa input.
                        //
                        // Dropping only `editor_open_uris` and `doc_versions` left
                        // the abandoned text in the database, and every importing
                        // module kept inferring against it until an unrelated
                        // watched-file event happened to arrive. Close-without-save
                        // is an ordinary editor action, not an edge case.
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
                        release_stale_requests(&connection, &pool, &mut pending)?;
                        let disk = file_uri_to_path(&uri_str)
                            .and_then(|path| std::fs::read_to_string(path).ok());
                        match disk {
                            // The file exists: its content is the truth again.
                            Some(text) => {
                                upsert_file(&mut db, &mut uri_to_file, uri_str.clone(), text);
                            }
                            // No file behind the URI (never saved, deleted while
                            // open, or a non-`file:` scheme). There is no truth to
                            // fall back to, so stop tracking it entirely rather
                            // than keep analysing text that exists nowhere.
                            None => {
                                uri_to_file.remove(&uri_str);
                            }
                        }
                        sync_file_set(&mut db, fs, &uri_to_file, &url_to_file);
                        // Every open importer's inference can change, and the
                        // closed file's own diagnostics are withdrawn if it is
                        // gone. Publish an empty set for it in that case: the
                        // client keeps showing the last set otherwise.
                        if !uri_to_file.contains_key(&uri_str) {
                            let note = lsp_server::Notification::new(
                                PublishDiagnostics::METHOD.to_owned(),
                                PublishDiagnosticsParams {
                                    uri: p.text_document.uri.clone(),
                                    diagnostics: Vec::new(),
                                    version: None,
                                },
                            );
                            connection.out.send(Message::Notification(note))?;
                        }
                        for doc_uri_str in uri_to_file.keys() {
                            dirty.insert(doc_uri_str.clone());
                        }
                        diag_deadline = Some(Instant::now() + DEBOUNCE);
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
                        release_stale_requests(&connection, &pool, &mut pending)?;
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
                                        let path = Path::new(&path);
                                        let under_excluded_dir = path
                                            .ancestors()
                                            .skip(1)
                                            .any(|anc| is_excluded_dir(anc, &excluded_dir_names));
                                        if under_excluded_dir {
                                            continue;
                                        }
                                        if let Ok(text) = std::fs::read_to_string(path) {
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
                        sync_file_set(&mut db, fs, &uri_to_file, &url_to_file);
                        // Republish diagnostics for all tracked docs: a watched-file
                        // change can affect any open importer. Mark them dirty and
                        // arm the debounce.
                        for doc_uri_str in uri_to_file.keys() {
                            dirty.insert(doc_uri_str.clone());
                        }
                        diag_deadline = Some(Instant::now() + DEBOUNCE);
                    }
                    "flatppl/urlSources" => {
                        // The editor client is the sole fetcher+truster of remote
                        // `load_module` / `load_data` URLs (the LSP cannot prompt
                        // for URL approval over the protocol). It pushes the
                        // content it already fetched as `{ sources: [{uri, text}] }`;
                        // we merge it into the FileSet as read-only source entries
                        // so resolution finds it. No network, no trust, no fetch
                        // on this side — the content is simply a salsa input.
                        let entries = note
                            .params
                            .get("sources")
                            .and_then(|v| v.as_array())
                            .cloned()
                            .unwrap_or_default();
                        let mut changed = false;
                        if !entries.is_empty() {
                            release_stale_requests(&connection, &pool, &mut pending)?;
                        }
                        for entry in &entries {
                            let (Some(url), Some(text)) = (
                                entry.get("uri").and_then(|v| v.as_str()),
                                entry.get("text").and_then(|v| v.as_str()),
                            ) else {
                                continue;
                            };
                            upsert_url_source(
                                &mut db,
                                &mut url_to_file,
                                url.to_string(),
                                text.to_string(),
                            );
                            changed = true;
                        }
                        if changed {
                            // A fed URL can satisfy a previously-unresolved import
                            // in any open doc, so refresh them all (the URL sources
                            // themselves are not in `uri_to_file`, so they get no
                            // diagnostics). A pure-text re-feed leaves membership
                            // unchanged but still invalidates importers via the
                            // dep's `parse` edge.
                            sync_file_set(&mut db, fs, &uri_to_file, &url_to_file);
                            for doc_uri_str in uri_to_file.keys() {
                                dirty.insert(doc_uri_str.clone());
                            }
                            diag_deadline = Some(Instant::now() + DEBOUNCE);
                        }
                    }
                    "$/cancelRequest" => {
                        // LSP 3.17 `$/cancelRequest`: the client asks the server
                        // to stop working on a request. The server still owes
                        // that request a response, and the protocol reserves
                        // `RequestCancelled` (-32800) for exactly this.
                        let Ok(p) = serde_json::from_value::<lsp_types::CancelParams>(note.params)
                        else {
                            continue;
                        };
                        let id = match p.id {
                            lsp_types::NumberOrString::Number(n) => RequestId::from(n),
                            lsp_types::NumberOrString::String(s) => RequestId::from(s),
                        };
                        // Recorded first: a job still queued, or one a worker is
                        // about to start, reads this and abandons its query
                        // instead of computing a result nobody will receive.
                        note_cancelled(&cancelled, id.clone());
                        answer(
                            &connection,
                            &mut pending,
                            Reply::Ready(Message::Response(Response::new_err(
                                id,
                                ErrorCode::RequestCanceled as i32,
                                "request cancelled by the client".to_owned(),
                            ))),
                        )?;
                    }
                    _ => {} // ignore other notifications
                }
            }
            Message::Request(req) => {
                // Handle shutdown first. `handle_shutdown` answers it itself, so
                // it never enters `pending`.
                if handle_shutdown(&connection, &req)? {
                    break;
                }
                pending.insert(req.id.clone());
                // A cancel that arrived BEFORE its request (message reordering,
                // a replayed buffer) would otherwise leave the request with no
                // response at all: the main thread had nothing to answer, and
                // the job sees the id cancelled and returns silently.
                if take_cancelled(&cancelled, &req.id) {
                    answer(
                        &connection,
                        &mut pending,
                        Reply::Ready(Message::Response(Response::new_err(
                            req.id,
                            ErrorCode::RequestCanceled as i32,
                            "request cancelled by the client".to_owned(),
                        ))),
                    )?;
                    continue;
                }
                // Dispatch the request to a worker thread on the pool. The
                // worker snapshots a salsa handle on the main thread (so a later
                // edit's `cancel_others` waits for it) and replies on
                // `result_tx`.
                if let Err(id) = dispatch_request(
                    &pool,
                    &result_tx,
                    &db,
                    &uri_to_file,
                    fs,
                    cats,
                    &cancelled,
                    req,
                ) {
                    // The queue is full. Answer now rather than block the main
                    // loop, which must stay free to take the next edit and the
                    // `$/cancelRequest` that would drain the backlog.
                    answer(
                        &connection,
                        &mut pending,
                        Reply::Ready(Message::Response(Response::new_err(
                            id,
                            ErrorCode::RequestFailed as i32,
                            format!(
                                "server request queue is full ({} outstanding); \
                                 cancel or retry",
                                crate::pool::QUEUE_CAPACITY
                            ),
                        ))),
                    )?;
                }
            }
            Message::Response(_) => {} // ignore server-originated response echoes
        }
    }

    Ok(())
}

// ── Response arbitration ─────────────────────────────────────────────────────

/// Cancelled ids retained before the set is cleared.
///
/// An id leaves the set as soon as either side reads it, so it only accumulates
/// for a cancel whose request never arrives — which a client should not send at
/// all. Clearing past the cap costs nothing but a missed optimisation: a queued
/// job then runs work whose response `answer` discards anyway, because the
/// request is no longer `pending`.
const MAX_CANCELLED_IDS: usize = 4096;

/// Record `id` as cancelled, clearing the set if it has grown past
/// [`MAX_CANCELLED_IDS`].
fn note_cancelled(cancelled: &Arc<Mutex<HashSet<RequestId>>>, id: RequestId) {
    if let Ok(mut set) = cancelled.lock() {
        if set.len() >= MAX_CANCELLED_IDS {
            set.clear();
        }
        set.insert(id);
    }
}

/// Was `id` cancelled? Removes it, so each cancel is read once and the set does
/// not grow with every cancel a long session sees.
fn take_cancelled(cancelled: &Arc<Mutex<HashSet<RequestId>>>, id: &RequestId) -> bool {
    cancelled
        .lock()
        .map(|mut set| set.remove(id))
        .unwrap_or(false)
}

/// What a worker hands back to the main thread.
///
/// A large answer travels as shared JSON text, memoized per revision, so a
/// queued response costs a refcount rather than a `serde_json::Value` — see
/// [`crate::outbound`] for the 38x that saves.
enum Reply {
    /// An already-formed message: an error, or a result whose `Value` is small.
    Ready(Message),
    /// A response body that is already JSON.
    Raw { id: RequestId, result: Arc<str> },
}

impl Reply {
    /// The request id this reply answers, for the `pending` arbiter.
    fn response_id(&self) -> Option<&RequestId> {
        match self {
            Reply::Ready(Message::Response(resp)) => Some(&resp.id),
            Reply::Ready(_) => None,
            Reply::Raw { id, .. } => Some(id),
        }
    }
}

/// Responses handed to the writer and not yet on the wire.
///
/// A queued `Reply::Raw` is cheap, but the handlers that still build a
/// `serde_json::Value` on the worker are not, so the writer's backlog is
/// capped: while it is at least this deep the main thread stops taking worker
/// replies, and the excess waits as a queued job (a `Database` clone and a
/// `Request`) instead of as a `Value`. It refuses nothing — a delayed request
/// is still answered as soon as the client reads.
const OUTBOUND_BACKLOG: usize = 4;

/// How often to re-check the writer's backlog while throttled.
///
/// Only armed while the backlog is full, so an idle server still blocks in
/// `select!` and burns no CPU.
const THROTTLE_POLL: Duration = Duration::from_millis(2);

/// Send `reply`, dropping a duplicate response.
///
/// JSON-RPC 2.0 §5 requires exactly one response per request, and several paths
/// can produce one for the same id: the worker that ran it, a
/// `$/cancelRequest`, a queue drain, a full queue. `pending` is the arbiter —
/// the first path to claim the id wins and the rest are dropped. Notifications
/// pass through untouched. A duplicate is dropped *before* its JSON is built,
/// so a cancelled request costs no serialisation.
fn answer(
    connection: &Wire,
    pending: &mut HashSet<RequestId>,
    reply: Reply,
) -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
    if let Some(id) = reply.response_id() {
        if !pending.remove(id) {
            return Ok(()); // already answered by another path
        }
    }
    match reply {
        Reply::Ready(msg) => connection.out.send(msg)?,
        Reply::Raw { id, result } => connection.out.send(Outgoing::RawResult { id, result })?,
    }
    Ok(())
}

/// Drain every queued request job and answer each one `ContentModified`.
///
/// Called immediately before an input write. Each queued job holds a
/// `Database` clone taken before enqueue, and salsa's `cancel_others` blocks
/// until every outstanding clone drops — so executing a client's backlog is
/// what delays the edit. Dropping the jobs releases those clones at once.
///
/// LSP 3.17 reserves `ContentModified` (-32801) for a result a content change
/// invalidated. The spec's caveat — do not send it merely because a change sits
/// *unprocessed* — does not apply here: the change is being applied on this
/// very iteration, so the revision these jobs were snapshotted against is
/// gone. Jobs a worker already picked up are untouched; they finish, or unwind
/// with `salsa::Cancelled`, which answers the same code.
fn release_stale_requests(
    connection: &Wire,
    pool: &crate::pool::Pool,
    pending: &mut HashSet<RequestId>,
) -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
    for id in pool.drain_queued() {
        answer(
            connection,
            pending,
            Reply::Ready(Message::Response(Response::new_err(
                id,
                ErrorCode::ContentModified as i32,
                "the document changed before this request ran; re-request against \
                 the new version"
                    .to_owned(),
            ))),
        )?;
    }
    Ok(())
}

/// The `InvalidParams` (-32602) answer for a request whose params did not
/// deserialize.
///
/// JSON-RPC 2.0 §5.1 defines -32602 as "Invalid method parameter(s)" and LSP
/// 3.17 inherits it. Eight handlers used to fold this failure into the same
/// `Option` that means "nothing to report at this position" and answered an
/// empty success, so a client bug or a protocol mismatch looked like a file
/// with no symbols, no hints and no definition.
fn invalid_params(req: &lsp_server::Request, what: &str, error: serde_json::Error) -> Response {
    Response::new_err(
        req.id.clone(),
        ErrorCode::InvalidParams as i32,
        format!("malformed {what} params: {error}"),
    )
}

// ── Off-main-thread request dispatch ─────────────────────────────────────────

/// Snapshot the salsa database + file map on the **main thread** and hand a
/// request job to the worker pool. `Err(id)` when the queue is full — the
/// caller owes that request a response.
///
/// The `Database::clone` MUST happen here, on the main thread, before the job
/// is enqueued: salsa's `Storage::clone` bumps the live-clone count, and a later
/// input write (`set_text` on an edit) calls `cancel_others`, which sets the
/// cancellation flag and blocks until every outstanding clone drops. Cloning on
/// the worker would race that wait — a worker cloning as fast as another drops
/// could hold the count above one indefinitely and livelock the write. Taking
/// the clone here closes the set of clones at write time, so the wait is
/// finite; the queue bound is what keeps it short.
///
/// On the worker the query body runs under `salsa::Cancelled::catch`: if a
/// concurrent write cancels this revision the body unwinds with
/// `salsa::Cancelled` and we answer `ContentModified` (-32801). A stale reply
/// computed against pre-edit text would be wrong, but silence is not the
/// alternative the protocol allows — JSON-RPC requires a response, and LSP 3.17
/// defines that code for this case.
///
/// A request the client cancelled while it sat in the queue is not run at all:
/// the main thread has already answered it `RequestCancelled` (-32800), so the
/// job returns without touching a query.
#[allow(clippy::too_many_arguments)] // one message-loop seam, not a public API
fn dispatch_request(
    pool: &crate::pool::Pool,
    result_tx: &crossbeam_channel::Sender<Reply>,
    db: &Database,
    uri_to_file: &HashMap<String, SourceFile>,
    fs: FileSet,
    cats: Catalogues,
    cancelled: &Arc<Mutex<HashSet<RequestId>>>,
    req: lsp_server::Request,
) -> Result<(), RequestId> {
    let db: Database = db.clone();
    let files: HashMap<String, SourceFile> = uri_to_file.clone();
    let result_tx = result_tx.clone();
    let cancelled = Arc::clone(cancelled);
    let id = req.id.clone();
    pool.try_spawn(id.clone(), move || {
        if take_cancelled(&cancelled, &id) {
            return; // already answered `RequestCancelled` on the main thread
        }
        // `AssertUnwindSafe`: `&Database`/`&Request` are not auto-`UnwindSafe`,
        // but this is sound — on cancel we discard all captured state and answer
        // an error, so no observer sees a half-updated value.
        let outcome = salsa::Cancelled::catch(std::panic::AssertUnwindSafe(|| {
            handle_request_on_worker(&db, &files, fs, cats, &req)
        }));
        let reply = match outcome {
            Ok(reply) => reply,
            Err(_cancelled) => Reply::Ready(Message::Response(Response::new_err(
                id,
                ErrorCode::ContentModified as i32,
                "the document changed while this request was running; re-request \
                 against the new version"
                    .to_owned(),
            ))),
        };
        let _ = result_tx.send(reply);
        // `db` (the clone) drops here, releasing the salsa handle.
    })
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
) -> Reply {
    // `documentSymbol` is the one method whose answer is large enough that the
    // deferred serialisation in `Reply::Symbols` is worth the extra variant.
    if req.method.as_str() == DocumentSymbolRequest::METHOD {
        return handle_document_symbols(db, uri_to_file, req);
    }
    Reply::Ready(Message::Response(handle_request_response(
        db,
        uri_to_file,
        fs,
        cats,
        req,
    )))
}

/// The methods whose answer is a `Response` built on the worker.
fn handle_request_response(
    db: &Database,
    uri_to_file: &HashMap<String, SourceFile>,
    fs: FileSet,
    cats: Catalogues,
    req: &lsp_server::Request,
) -> Response {
    match req.method.as_str() {
        HoverRequest::METHOD => handle_hover(db, uri_to_file, fs, cats, req),
        WorkspaceSymbolRequest::METHOD => handle_workspace_symbols(db, fs, req),
        InlayHintRequest::METHOD => handle_inlay_hints(db, uri_to_file, fs, cats, req),
        GotoDefinition::METHOD => handle_goto_definition(db, uri_to_file, fs, cats, req),
        Completion::METHOD => handle_completion(db, uri_to_file, fs, cats, req),
        References::METHOD => handle_references(db, uri_to_file, fs, cats, req),
        PrepareRenameRequest::METHOD => handle_prepare_rename(db, uri_to_file, fs, cats, req),
        Rename::METHOD => handle_rename(db, uri_to_file, fs, cats, req),
        SignatureHelpRequest::METHOD => handle_signature_help(db, uri_to_file, fs, cats, req),
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
    connection: &Wire,
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
        references_provider: Some(OneOf::Left(true)),
        // `prepare_provider` so the client asks whether a position is renameable
        // before prompting for a new name: built-ins, standard-module members,
        // record fields and `%local` placeholders are declined up front.
        rename_provider: Some(OneOf::Right(lsp_types::RenameOptions {
            prepare_provider: Some(true),
            work_done_progress_options: Default::default(),
        })),
        // `(` opens a signature, `,` advances the active parameter, and `=`
        // retriggers because it turns the current argument into a §05 keyword
        // argument, which moves the highlight to that name's declared slot.
        signature_help_provider: Some(lsp_types::SignatureHelpOptions {
            trigger_characters: Some(vec!["(".to_string(), ",".to_string()]),
            retrigger_characters: Some(vec![",".to_string(), "=".to_string()]),
            work_done_progress_options: Default::default(),
        }),
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

/// Extract directory names to prune from the workspace scan, from
/// `initializationOptions.diagnosticsExclude`. Defaults to `{"fixtures"}`
/// when the client sends nothing — `node_modules` is added unconditionally
/// by the caller, not here, since it must never be overridable.
fn excluded_dir_names_from_params(params: &lsp_types::InitializeParams) -> HashSet<String> {
    let configured: Option<Vec<String>> = params
        .initialization_options
        .as_ref()
        .and_then(|v| v.get("diagnosticsExclude"))
        .and_then(|v| serde_json::from_value(v.clone()).ok());
    match configured {
        Some(names) => names.into_iter().collect(),
        None => HashSet::from(["fixtures".to_owned()]),
    }
}

/// Whether `path` (a directory) should be pruned from the scan: its own
/// final component name matches one of `excluded_dir_names` exactly.
fn is_excluded_dir(path: &Path, excluded_dir_names: &HashSet<String>) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| excluded_dir_names.contains(name))
}

/// Convert a `file://` URI string to a filesystem path string, or `None` for
/// non-`file:` schemes.
///
/// Three shapes, per RFC 8089 "The 'file' URI Scheme":
///
/// - `file:///Users/me/My%20Project` — an empty authority and a POSIX path.
///   Percent-decoded so a root containing spaces resolves on disk.
/// - `file:///C:/proj/m.flatppl` — a Windows local path. The leading `/` before
///   the drive letter belongs to the URI, not to the path, and the separators
///   come back as `\`.
/// - `file://host/share/m.flatppl` — a non-empty authority, i.e. a Windows UNC
///   path `\\host\share\m.flatppl`.
///
/// The old implementation handled only the first: it stripped `file://` and
/// percent-decoded, with no authority and no drive handling, so a Windows URI
/// came back as `/C:/proj/m.flatppl` — a path that does not exist — and a UNC
/// URI lost its leading separators. `build.yml` ships `flatppl-lsp` for
/// `x86_64-pc-windows-msvc`, so both shapes are reachable.
fn file_uri_to_path(uri_str: &str) -> Option<String> {
    let body = uri_str.strip_prefix("file://")?;
    // A non-empty authority: `file://host/share/...` is a UNC path. (An empty
    // authority leaves `body` starting with the path's own `/`.)
    if !body.is_empty() && !body.starts_with('/') {
        let decoded = percent_decode(body);
        return Some(format!("\\\\{}", decoded.replace('/', "\\")));
    }
    let decoded = percent_decode(body);
    if let Some(rest) = windows_drive_path(&decoded) {
        return Some(rest);
    }
    Some(decoded)
}

/// `\`-separated Windows path for a decoded URI body of the form
/// `/C:/proj/m.flatppl`, else `None`.
///
/// A POSIX absolute path cannot begin with a drive letter followed by `:`, so
/// the shapes do not collide. (`/c:/x` as a literal POSIX path is legal and
/// would be misread; every `file:` URI implementation makes that same trade,
/// because the URI itself carries no platform tag.)
fn windows_drive_path(body: &str) -> Option<String> {
    let rest = body.strip_prefix('/')?;
    let mut chars = rest.chars();
    let drive = chars.next()?;
    if !drive.is_ascii_alphabetic() || chars.next()? != ':' {
        return None;
    }
    // A bare `/C:` is the drive root; anything after the colon must be a
    // separator, not more text.
    match rest.get(2..3) {
        None | Some("/") | Some("\\") => Some(rest.replace('/', "\\")),
        _ => None,
    }
}

/// Percent-encode a filesystem path into a `file://` URI (encodes spaces and
/// other reserved bytes; leaves `/` and unreserved chars). Symmetric with
/// [`file_uri_to_path`].
///
/// Windows paths get their own shapes, matching RFC 8089 and what every LSP
/// client emits:
///
/// - `C:\proj\m.flatppl` -> `file:///C:/proj/m.flatppl`
/// - `\\host\share\m.flatppl` -> `file://host/share/m.flatppl`
///
/// Byte-encoding every non-unreserved byte, as the old implementation did, gave
/// `file://C%3A%5Cproj%5Cm.flatppl`: no scheme-relative `/`, the drive colon and
/// every separator escaped. No client resolves that, and it is what the
/// workspace-root scan, the watched-file handler, the `load_module` base path
/// and the definition URIs handed back to the client all went through.
pub(crate) fn path_to_file_uri(path: &str) -> String {
    // UNC: the host and share become the URI authority.
    if let Some(rest) = path.strip_prefix("\\\\") {
        let mut out = String::from("file://");
        push_encoded(&mut out, &rest.replace('\\', "/"));
        return out;
    }
    let mut out = String::from("file://");
    let windows_local = {
        let mut chars = path.chars();
        matches!(
            (chars.next(), chars.next(), chars.next()),
            (Some(d), Some(':'), Some('\\') | Some('/') | None) if d.is_ascii_alphabetic()
        )
    };
    if windows_local {
        // The `/` before the drive letter is the URI's, so the authority stays
        // empty and the path is absolute.
        out.push('/');
        push_encoded(&mut out, &path.replace('\\', "/"));
    } else {
        push_encoded(&mut out, path);
    }
    out
}

/// Percent-encode `s` into `out`, leaving `/`, `:` and the RFC 3986 unreserved
/// characters. `:` is legal in a URI path (RFC 3986 §3.3 `pchar`) and a Windows
/// drive letter needs it intact.
fn push_encoded(out: &mut String, s: &str) {
    for b in s.bytes() {
        match b {
            b'/' | b':' | b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
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
/// A directory whose name is in `excluded_dir_names` is pruned entirely —
/// none of its descendants are visited.
///
/// The walk resolves each entry to its canonical path and stays **inside the
/// canonical root**, keeping a visited set of directories it has already
/// entered. `Path::is_dir` follows symlinks, so without both guards a
/// workspace holding `link -> ../outside` and `selfloop -> .` produced 65
/// published files from 2 real ones — including a file outside the configured
/// root, read and analysed, and 32 nested repetitions of the loop until the
/// OS path limit stopped the walk. Each duplicate was a separate salsa
/// `SourceFile` with its own diagnostics.
///
/// A depth cap alone is not enough: it bounds the loop but still reads outside
/// the root. Containment is what stops that, and canonicalisation is what makes
/// containment decidable.
fn scan_dir(
    dir: &Path,
    db: &mut Database,
    uri_to_file: &mut HashMap<String, SourceFile>,
    excluded_dir_names: &HashSet<String>,
) {
    // The root defines containment, so it must be canonical too: comparing a
    // canonical child against a symlinked root would reject everything.
    let Ok(root) = std::fs::canonicalize(dir) else {
        return;
    };
    let mut visited: HashSet<std::path::PathBuf> = HashSet::new();
    visited.insert(root.clone());
    scan_dir_within(
        &root,
        &root,
        db,
        uri_to_file,
        excluded_dir_names,
        &mut visited,
    );
}

/// One level of [`scan_dir`]'s walk. `root` is the canonical workspace root;
/// `visited` holds the canonical directories already entered.
fn scan_dir_within(
    dir: &Path,
    root: &Path,
    db: &mut Database,
    uri_to_file: &mut HashMap<String, SourceFile>,
    excluded_dir_names: &HashSet<String>,
    visited: &mut HashSet<std::path::PathBuf>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        // Resolve before deciding anything: the entry may be a symlink, and
        // both the containment test and the loop test are about the target.
        let Ok(canonical) = std::fs::canonicalize(&path) else {
            continue; // a broken symlink, or a path we cannot resolve
        };
        if !canonical.starts_with(root) {
            continue; // outside the configured workspace root
        }
        if canonical.is_dir() {
            // Prune on the name the walk reached it by AND on the resolved
            // name, so a link cannot smuggle an excluded directory back in.
            if is_excluded_dir(&path, excluded_dir_names)
                || is_excluded_dir(&canonical, excluded_dir_names)
            {
                continue;
            }
            if !visited.insert(canonical.clone()) {
                continue; // already walked: a loop, or two links to one directory
            }
            scan_dir_within(
                &canonical,
                root,
                db,
                uri_to_file,
                excluded_dir_names,
                visited,
            );
        } else if canonical.extension().and_then(|e| e.to_str()) == Some("flatppl") {
            let Ok(text) = std::fs::read_to_string(&canonical) else {
                continue;
            };
            // Store the canonical path, so two spellings of one file are one
            // `SourceFile` with one set of diagnostics.
            let path_str = canonical.to_string_lossy().into_owned();
            let uri_str = path_to_file_uri(&path_str);
            let file = SourceFile::new(db, path_str, text);
            uri_to_file.insert(uri_str, file);
        }
    }
}

/// Build (or rebuild) a [`FileSet`] from the editor buffers (`uri_to_file`) plus
/// the client-fed URL sources (`url_to_file`).
///
/// Called any time the membership changes so salsa sees a fresh input. URL
/// sources are included for resolution (their stored path is the URL, which
/// `resolve_path` matches via `Location`) but never get diagnostics published.
fn build_fileset(
    db: &Database,
    uri_to_file: &HashMap<String, SourceFile>,
    url_to_file: &HashMap<String, SourceFile>,
) -> FileSet {
    let files: Vec<SourceFile> = uri_to_file
        .values()
        .chain(url_to_file.values())
        .copied()
        .collect();
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
fn sync_file_set(
    db: &mut Database,
    fs: FileSet,
    uri_to_file: &HashMap<String, SourceFile>,
    url_to_file: &HashMap<String, SourceFile>,
) {
    use salsa::Setter;
    let mut new_files: Vec<SourceFile> = uri_to_file
        .values()
        .chain(url_to_file.values())
        .copied()
        .collect();
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

/// Insert or update a client-fed URL source (from `flatppl/urlSources`), keyed
/// by the URL.
///
/// Unlike [`upsert_file`] there is no URI→path conversion: the stored
/// `SourceFile.path` **is** the URL, so [`resolve_path`](crate::queries) (via
/// `Location`) matches a `load_module(url)` directive — or a URL-relative
/// directive joined against an importer URL — straight against it. An existing
/// entry is updated in place via the salsa setter so importers recompute.
fn upsert_url_source(
    db: &mut Database,
    url_to_file: &mut HashMap<String, SourceFile>,
    url: String,
    text: String,
) -> SourceFile {
    use salsa::Setter;
    if let Some(&existing) = url_to_file.get(&url) {
        existing.set_text(db).to(text);
        existing
    } else {
        let file = SourceFile::new(db, url.clone(), text);
        url_to_file.insert(url, file);
        file
    }
}

/// Send a `textDocument/publishDiagnostics` notification for `file`.
///
/// `uri_str` must be a valid URI string; the send is best-effort (a send
/// failure is returned as an error to the caller).
fn publish_diagnostics(
    connection: &Wire,
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
    connection.out.send(Message::Notification(note))?;
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
    let params: lsp_types::HoverParams = match serde_json::from_value(req.params.clone()) {
        Ok(p) => p,
        Err(e) => return invalid_params(req, "hover", e),
    };
    let result = (|| -> Option<lsp_types::Hover> {
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

/// Handle a `textDocument/documentSymbol` request.  Returns the response body
/// as shared JSON text, memoized per revision, without building a
/// `serde_json::Value` for it (see [`crate::outbound`]).
fn handle_document_symbols(
    db: &Database,
    uri_to_file: &HashMap<String, SourceFile>,
    req: &lsp_server::Request,
) -> Reply {
    let params: lsp_types::DocumentSymbolParams = match serde_json::from_value(req.params.clone()) {
        Ok(p) => p,
        Err(e) => return Reply::Ready(Message::Response(invalid_params(req, "documentSymbol", e))),
    };
    let result = (|| {
        let uri_str = params.text_document.uri.as_str().to_owned();
        let file = *uri_to_file.get(&uri_str)?;
        Some(crate::queries::document_symbol_json(db, file).text())
    })()
    // An untracked URI has no symbols, which is the empty tree, not an error.
    .unwrap_or_else(|| Arc::from("[]"));

    Reply::Raw {
        id: req.id.clone(),
        result,
    }
}

/// Handle a `workspace/symbol` request.  Returns a `Response` (result or
/// null) without sending it — the caller dispatches the message.
fn handle_workspace_symbols(db: &Database, fs: FileSet, req: &lsp_server::Request) -> Response {
    let params: lsp_types::WorkspaceSymbolParams = match serde_json::from_value(req.params.clone())
    {
        Ok(p) => p,
        Err(e) => return invalid_params(req, "workspace/symbol", e),
    };
    let syms = crate::capabilities::workspace_symbols(db, fs, &params.query);
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
    let params: lsp_types::InlayHintParams = match serde_json::from_value(req.params.clone()) {
        Ok(p) => p,
        Err(e) => return invalid_params(req, "inlayHint", e),
    };
    let hints: Vec<lsp_types::InlayHint> = (|| {
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
    let params: lsp_types::GotoDefinitionParams = match serde_json::from_value(req.params.clone()) {
        Ok(p) => p,
        Err(e) => return invalid_params(req, "definition", e),
    };
    let result = (|| -> Option<lsp_types::GotoDefinitionResponse> {
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
    let params: lsp_types::CompletionParams = match serde_json::from_value(req.params.clone()) {
        Ok(p) => p,
        Err(e) => return invalid_params(req, "completion", e),
    };
    let result = (|| -> Option<lsp_types::CompletionResponse> {
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
        let ctx = completion_context(text, byte_offset);
        let lead_space = tight_after_operator(text, byte_offset);
        let items = crate::capabilities::completion(db, file, fs, cats, ctx, lead_space);
        Some(lsp_types::CompletionResponse::Array(items))
    })();

    match result {
        Some(resp) => Response::new_ok(req.id.clone(), resp),
        None => Response::new_ok(req.id.clone(), serde_json::Value::Null),
    }
}

/// Turn a `SourceFile`'s stored path into the URI to report it under.
///
/// A stored path that is already a URI (a client-fed URL source, or a `file://`
/// path) is used as-is; a bare filesystem path is percent-encoded.
fn file_uri_for(path: &str) -> Option<Uri> {
    let s = if path.contains("://") {
        path.to_string()
    } else {
        path_to_file_uri(path)
    };
    Uri::from_str(&s).ok()
}

/// The `SourceFile` in `fs` whose stored path is `path`, for reusing its cached
/// line index when converting a byte range in another file.
fn file_by_path(db: &Database, fs: FileSet, path: &str) -> Option<SourceFile> {
    fs.files(db).iter().copied().find(|f| f.path(db) == path)
}

/// Convert a byte range in the file stored at `path` to an LSP `Range`.
fn range_in(db: &Database, fs: FileSet, path: &str, start: u32, end: u32) -> lsp_types::Range {
    let li = file_by_path(db, fs, path)
        .map(|f| line_index(db, f))
        .unwrap_or_else(|| crate::line_index::LineIndex::new(""));
    let s = li.position(start);
    let e = li.position(end);
    lsp_types::Range::new(
        lsp_types::Position::new(s.line, s.character),
        lsp_types::Position::new(e.line, e.character),
    )
}

/// Resolve a request's text-document-position params to `(file, byte_offset)`.
fn position_target(
    db: &Database,
    uri_to_file: &HashMap<String, SourceFile>,
    tdp: &lsp_types::TextDocumentPositionParams,
) -> Option<(SourceFile, u32)> {
    let file = *uri_to_file.get(tdp.text_document.uri.as_str())?;
    let li = line_index(db, file);
    let byte = li.offset(Pos {
        line: tdp.position.line,
        character: tdp.position.character,
    });
    Some((file, byte))
}

/// Handle a `textDocument/references` request.
///
/// Returns the locations of every reference to the binding under the cursor
/// across the whole file set. A position that is not on a binding yields an
/// empty array, which is the protocol's "nothing to report".
fn handle_references(
    db: &Database,
    uri_to_file: &HashMap<String, SourceFile>,
    fs: FileSet,
    cats: Catalogues,
    req: &lsp_server::Request,
) -> Response {
    let params: lsp_types::ReferenceParams = match serde_json::from_value(req.params.clone()) {
        Ok(p) => p,
        Err(e) => return invalid_params(req, "references", e),
    };
    let locations: Vec<lsp_types::Location> = (|| {
        let (file, byte) = position_target(db, uri_to_file, &params.text_document_position)?;
        let index = node_span_index(db, file, fs, cats);
        let locs = crate::rename::references(
            db,
            file,
            fs,
            cats,
            byte,
            &index,
            params.context.include_declaration,
        );
        Some(
            locs.into_iter()
                .filter_map(|l| {
                    Some(lsp_types::Location {
                        uri: file_uri_for(&l.path)?,
                        range: range_in(db, fs, &l.path, l.start, l.end),
                    })
                })
                .collect(),
        )
    })()
    .unwrap_or_default();

    Response::new_ok(req.id.clone(), locations)
}

/// Handle a `textDocument/prepareRename` request.
///
/// A `null` result tells the client the position cannot be renamed, so it never
/// prompts for a new name over a built-in, a standard-module member, a record
/// field or a `%local` placeholder.
fn handle_prepare_rename(
    db: &Database,
    uri_to_file: &HashMap<String, SourceFile>,
    fs: FileSet,
    cats: Catalogues,
    req: &lsp_server::Request,
) -> Response {
    let params: lsp_types::TextDocumentPositionParams =
        match serde_json::from_value(req.params.clone()) {
            Ok(p) => p,
            Err(e) => return invalid_params(req, "prepareRename", e),
        };
    let result = (|| -> Option<lsp_types::Range> {
        let (file, byte) = position_target(db, uri_to_file, &params)?;
        let index = node_span_index(db, file, fs, cats);
        let (start, end) = crate::rename::prepare_rename(db, file, fs, cats, byte, &index).ok()?;
        Some(range_in(db, fs, &file.path(db).clone(), start, end))
    })();

    match result {
        Some(range) => Response::new_ok(req.id.clone(), range),
        None => Response::new_ok(req.id.clone(), serde_json::Value::Null),
    }
}

/// Handle a `textDocument/rename` request.
///
/// A refusal comes back as a `RequestFailed` error carrying the normative reason
/// (see `crate::rename::rename_edits`), so the editor shows why the rename did
/// not happen instead of silently doing nothing.
fn handle_rename(
    db: &Database,
    uri_to_file: &HashMap<String, SourceFile>,
    fs: FileSet,
    cats: Catalogues,
    req: &lsp_server::Request,
) -> Response {
    let params: lsp_types::RenameParams = match serde_json::from_value(req.params.clone()) {
        Ok(p) => p,
        Err(e) => return invalid_params(req, "rename", e),
    };
    let Some((file, byte)) = position_target(db, uri_to_file, &params.text_document_position)
    else {
        return Response::new_err(
            req.id.clone(),
            lsp_server::ErrorCode::InvalidParams as i32,
            "rename position is not in a tracked document".to_string(),
        );
    };
    let index = node_span_index(db, file, fs, cats);
    let locs = match crate::rename::rename_edits(db, file, fs, cats, byte, &index, &params.new_name)
    {
        Ok(locs) => locs,
        Err(refusal) => {
            return Response::new_err(
                req.id.clone(),
                lsp_server::ErrorCode::RequestFailed as i32,
                refusal.0,
            );
        }
    };

    // Group the edits per file: a WorkspaceEdit is keyed by document URI, and a
    // rename through a `load_module` boundary touches at least two files.
    //
    // Grouping happens on the stored path, not on `Uri`: `WorkspaceEdit::changes`
    // is a `HashMap<Uri, _>` imposed by lsp-types, and `Uri` carries a lazily
    // filled cache, so it trips `clippy::mutable_key_type`. Keying on the path
    // and converting once at the end keeps the interior-mutable type out of every
    // hash lookup; the allow covers only the unavoidable final map.
    let mut by_path: HashMap<String, Vec<lsp_types::TextEdit>> = HashMap::new();
    for l in locs {
        by_path
            .entry(l.path.clone())
            .or_default()
            .push(lsp_types::TextEdit {
                range: range_in(db, fs, &l.path, l.start, l.end),
                new_text: params.new_name.clone(),
            });
    }
    #[allow(clippy::mutable_key_type)] // lsp-types keys WorkspaceEdit by `Uri`
    let changes = by_path
        .into_iter()
        .filter_map(|(path, edits)| Some((file_uri_for(&path)?, edits)))
        .collect();
    let edit = lsp_types::WorkspaceEdit {
        changes: Some(changes),
        ..Default::default()
    };
    Response::new_ok(req.id.clone(), edit)
}

/// Handle a `textDocument/signatureHelp` request.  Returns a `Response`
/// (a `SignatureHelp` or null) without sending it — the caller dispatches.
fn handle_signature_help(
    db: &Database,
    uri_to_file: &HashMap<String, SourceFile>,
    fs: FileSet,
    cats: Catalogues,
    req: &lsp_server::Request,
) -> Response {
    let params: lsp_types::SignatureHelpParams = match serde_json::from_value(req.params.clone()) {
        Ok(p) => p,
        Err(e) => return invalid_params(req, "signatureHelp", e),
    };
    let result = (|| -> Option<lsp_types::SignatureHelp> {
        let (file, byte) = position_target(db, uri_to_file, &params.text_document_position_params)?;
        crate::signature::signature_help(db, file, fs, cats, byte)
    })();

    match result {
        Some(help) => Response::new_ok(req.id.clone(), help),
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

/// Whether the cursor sits *tight* against a `~` or `=` operator: the character
/// immediately before the in-progress identifier — with **no** whitespace
/// between — is `~` or `=`.
///
/// Unlike [`completion_context`], this does NOT skip whitespace: it is true only
/// at `x ~|` / `mu =|`, not at `x ~ |`. The language server uses it to prepend a
/// space to a completion's inserted text (`x ~` + `Normal` → `x ~ Normal`) only
/// when the user has not already typed the space, so an existing space is never
/// doubled.
pub(crate) fn tight_after_operator(text: &str, byte: u32) -> bool {
    let bytes = text.as_bytes();
    let mut i = byte as usize;
    while i > 0 && is_ident_byte(bytes[i - 1]) {
        i -= 1;
    }
    i > 0 && (bytes[i - 1] == b'~' || bytes[i - 1] == b'=')
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

    // ── tight_after_operator ──────────────────────────────────────────────────
    #[test]
    fn tight_after_operator_immediately_after_tilde() {
        // "x ~" — cursor at byte 3, right after `~`, no space yet.
        assert!(tight_after_operator("x ~", 3));
    }

    #[test]
    fn tight_after_operator_immediately_after_eq() {
        // "mu =" — cursor at byte 4, right after `=`.
        assert!(tight_after_operator("mu =", 4));
    }

    #[test]
    fn tight_after_operator_tight_with_partial_ident() {
        // "x ~Nor" — cursor at byte 6; ident "Nor" sits directly on `~`, no space.
        assert!(tight_after_operator("x ~Nor", 6));
    }

    #[test]
    fn tight_after_operator_false_when_space_present() {
        // "x ~ " — a space already separates `~` from the cursor: not tight.
        assert!(!tight_after_operator("x ~ ", 4));
        // "x ~ Nor" — space before the ident: not tight.
        assert!(!tight_after_operator("x ~ Nor", 7));
    }

    #[test]
    fn tight_after_operator_false_for_non_operator() {
        assert!(!tight_after_operator("x", 1));
        assert!(!tight_after_operator("f(a,", 4)); // comma is not ~ or =
    }

    #[test]
    fn tight_after_operator_false_at_start_of_buffer() {
        // Nothing before the cursor: the `i > 0` guard must short-circuit, no panic.
        assert!(!tight_after_operator("", 0));
        assert!(!tight_after_operator("x", 0));
    }

    // ── excluded_dir_names_from_params / scan_dir pruning ────────────────────

    #[test]
    fn excluded_dir_names_defaults_to_fixtures() {
        let params: lsp_types::InitializeParams =
            serde_json::from_value(serde_json::json!({ "capabilities": {} })).unwrap();
        let names = excluded_dir_names_from_params(&params);
        assert_eq!(names, HashSet::from(["fixtures".to_owned()]));
    }

    #[test]
    fn excluded_dir_names_read_from_init_options() {
        let raw = serde_json::json!({
            "capabilities": {},
            "initializationOptions": { "diagnosticsExclude": ["fixtures", "demo"] }
        });
        let params: lsp_types::InitializeParams = serde_json::from_value(raw).unwrap();
        let names = excluded_dir_names_from_params(&params);
        assert_eq!(
            names,
            HashSet::from(["fixtures".to_owned(), "demo".to_owned()])
        );
    }

    #[test]
    fn excluded_dir_names_override_replaces_the_default() {
        // A client-supplied list that omits "fixtures" drops it entirely —
        // the default is a fallback for an ABSENT key, not a floor unioned
        // into whatever the client sends.
        let raw = serde_json::json!({
            "capabilities": {},
            "initializationOptions": { "diagnosticsExclude": ["demo"] }
        });
        let params: lsp_types::InitializeParams = serde_json::from_value(raw).unwrap();
        let names = excluded_dir_names_from_params(&params);
        assert_eq!(names, HashSet::from(["demo".to_owned()]));
    }

    #[test]
    fn is_excluded_dir_matches_final_component_only() {
        let excluded = HashSet::from(["fixtures".to_owned()]);
        assert!(is_excluded_dir(Path::new("/a/b/fixtures"), &excluded));
        // "fixtures" appears as an ancestor, but the dir itself is "b" — no match.
        assert!(!is_excluded_dir(Path::new("/a/fixtures/b"), &excluded));
        assert!(!is_excluded_dir(Path::new("/a/b/other"), &excluded));
    }

    #[test]
    fn scan_dir_prunes_excluded_subtrees() {
        // <tmp>/{good.flatppl, node_modules/leaked.flatppl, test/fixtures/bad.flatppl}
        let root = std::env::temp_dir().join(format!(
            "flatppl_lsp_scan_dir_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let node_modules = root.join("node_modules");
        let fixtures = root.join("test").join("fixtures");
        std::fs::create_dir_all(&node_modules).unwrap();
        std::fs::create_dir_all(&fixtures).unwrap();
        std::fs::write(root.join("good.flatppl"), "x ~ Normal(0, 1);").unwrap();
        std::fs::write(node_modules.join("leaked.flatppl"), "bad syntax !!!").unwrap();
        std::fs::write(fixtures.join("bad.flatppl"), "bad syntax !!!").unwrap();

        let mut db = Database::default();
        let mut uri_to_file: HashMap<String, SourceFile> = HashMap::new();
        let excluded_dir_names = HashSet::from(["node_modules".to_owned(), "fixtures".to_owned()]);
        scan_dir(&root, &mut db, &mut uri_to_file, &excluded_dir_names);

        let scanned: Vec<String> = uri_to_file.keys().cloned().collect();
        assert_eq!(
            scanned.len(),
            1,
            "expected only good.flatppl; got: {scanned:?}"
        );
        assert!(scanned[0].ends_with("good.flatppl"), "got: {scanned:?}");

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A symlink pointing out of the workspace is not followed, and a symlink
    /// pointing back at the workspace does not loop.
    ///
    /// The audit's workspace — `link -> ../outside` plus `selfloop -> .` — gave
    /// 65 published files from 2 real ones, one of them outside the configured
    /// root, plus 32 nested repetitions of the loop.
    #[test]
    fn scan_dir_refuses_an_outward_symlink_and_a_self_loop() {
        let base = std::env::temp_dir().join(format!(
            "flatppl_lsp_symlink_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let ws = base.join("ws");
        let outside = base.join("outside").join("secret");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(ws.join("m.flatppl"), "a = 1.5\n").unwrap();
        std::fs::write(outside.join("leak.flatppl"), "b = 2.5\n").unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(base.join("outside"), ws.join("link")).unwrap();
            std::os::unix::fs::symlink(&ws, ws.join("selfloop")).unwrap();
        }

        let mut db = Database::default();
        let mut uri_to_file: HashMap<String, SourceFile> = HashMap::new();
        scan_dir(&ws, &mut db, &mut uri_to_file, &HashSet::new());

        let scanned: Vec<String> = uri_to_file.keys().cloned().collect();
        assert_eq!(
            scanned.len(),
            1,
            "one real file in the root; got: {scanned:?}"
        );
        assert!(
            scanned[0].ends_with("m.flatppl"),
            "the one file is the root's own; got: {scanned:?}"
        );
        assert!(
            !scanned.iter().any(|u| u.contains("leak")),
            "a file outside the root must never be read: {scanned:?}"
        );
        assert!(
            !scanned.iter().any(|u| u.contains("selfloop")),
            "the loop must not produce duplicate entries: {scanned:?}"
        );

        std::fs::remove_dir_all(&base).ok();
    }

    /// Two links to one real directory yield one entry each file, not two: the
    /// visited set is keyed on the canonical path, and the stored path is
    /// canonical too, so a file has one `SourceFile` and one set of
    /// diagnostics.
    #[test]
    fn scan_dir_deduplicates_two_links_to_one_directory() {
        let ws = std::env::temp_dir().join(format!(
            "flatppl_lsp_dedupe_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let real = ws.join("real");
        std::fs::create_dir_all(&real).unwrap();
        std::fs::write(real.join("m.flatppl"), "a = 1.5\n").unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&real, ws.join("alias_one")).unwrap();
            std::os::unix::fs::symlink(&real, ws.join("alias_two")).unwrap();
        }

        let mut db = Database::default();
        let mut uri_to_file: HashMap<String, SourceFile> = HashMap::new();
        scan_dir(&ws, &mut db, &mut uri_to_file, &HashSet::new());

        let scanned: Vec<String> = uri_to_file.keys().cloned().collect();
        assert_eq!(
            scanned.len(),
            1,
            "one real file behind three names; got: {scanned:?}"
        );

        std::fs::remove_dir_all(&ws).ok();
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

    // ── Windows-shaped URIs ──────────────────────────────────────────────────
    //
    // `build.yml` ships `flatppl` and `flatppl-lsp` for
    // `x86_64-pc-windows-msvc`, while `test.yml` runs `ubuntu-latest` alone, so
    // these are the only coverage the Windows shapes have. They are pure string
    // conversions, so they run on any host. A Windows CI job is still missing:
    // nothing here exercises a real Windows filesystem.
    #[test]
    fn windows_local_path_round_trips() {
        // RFC 8089: the `/` before the drive letter is the URI's, and `:` is a
        // legal path character that must stay unescaped.
        assert_eq!(
            path_to_file_uri("C:\\proj\\m.flatppl"),
            "file:///C:/proj/m.flatppl"
        );
        assert_eq!(
            file_uri_to_path("file:///C:/proj/m.flatppl"),
            Some("C:\\proj\\m.flatppl".to_owned())
        );
        // A drive root, and a lower-case drive letter.
        assert_eq!(path_to_file_uri("C:\\"), "file:///C:/");
        assert_eq!(
            file_uri_to_path("file:///d:/x.flatppl"),
            Some("d:\\x.flatppl".to_owned())
        );
    }

    #[test]
    fn windows_local_path_with_a_space_round_trips() {
        assert_eq!(
            path_to_file_uri("C:\\My Proj\\m.flatppl"),
            "file:///C:/My%20Proj/m.flatppl"
        );
        assert_eq!(
            file_uri_to_path("file:///C:/My%20Proj/m.flatppl"),
            Some("C:\\My Proj\\m.flatppl".to_owned())
        );
    }

    #[test]
    fn windows_unc_path_round_trips() {
        // A UNC host is the URI authority, so there is no third slash.
        assert_eq!(
            path_to_file_uri("\\\\host\\share\\m.flatppl"),
            "file://host/share/m.flatppl"
        );
        assert_eq!(
            file_uri_to_path("file://host/share/m.flatppl"),
            Some("\\\\host\\share\\m.flatppl".to_owned())
        );
    }

    #[test]
    fn the_old_encoding_is_gone() {
        // The shape that no client could resolve: no scheme-relative `/`, the
        // drive colon escaped, every separator escaped.
        let uri = path_to_file_uri("C:\\proj\\m.flatppl");
        assert!(
            !uri.contains("%3A"),
            "the drive colon must stay literal: {uri}"
        );
        assert!(!uri.contains("%5C"), "separators must become `/`: {uri}");
        assert!(
            uri.starts_with("file:///"),
            "a local path needs an empty authority: {uri}"
        );
    }

    #[test]
    fn posix_paths_are_unaffected() {
        assert_eq!(
            path_to_file_uri("/Users/me/m.flatppl"),
            "file:///Users/me/m.flatppl"
        );
        assert_eq!(
            file_uri_to_path("file:///Users/me/m.flatppl"),
            Some("/Users/me/m.flatppl".to_owned())
        );
        // A relative path keeps its old shape (no leading `/` invented).
        assert_eq!(
            path_to_file_uri("helpers.flatppl"),
            "file://helpers.flatppl"
        );
        // A POSIX name that merely contains a colon is not a drive path.
        assert_eq!(
            file_uri_to_path("file:///Users/odd:name/m.flatppl"),
            Some("/Users/odd:name/m.flatppl".to_owned())
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

    // ── flatppl/urlSources: client-fed URL deps resolve (server never fetches) ─
    //
    // A model loads a URL module which itself has a *relative* `load_module` dep.
    // The server has no network; the editor client pushes the fetched URL content
    // via `flatppl/urlSources`. After the push, a hover on the model's cross-URL
    // reference resolves through the whole chain (model → common → priors, the
    // last reached by URL-relative join) — proving URL deps resolve from fed
    // content alone, with zero fetching.

    #[test]
    fn url_sources_feed_resolves_cross_url_reference() {
        use lsp_server::{Connection, Message};
        use lsp_types::notification::{DidOpenTextDocument, Notification as _};
        use lsp_types::request::{HoverRequest, Request as _};

        let (client_conn, server_conn) = Connection::memory();
        let server_thread = std::thread::spawn(move || {
            let init_params = serde_json::json!({ "capabilities": {} });
            run(server_conn, init_params).expect("server loop failed");
        });

        // Open the model: it loads a URL `common` module and uses `c.m`.
        let open = lsp_types::DidOpenTextDocumentParams {
            text_document: lsp_types::TextDocumentItem {
                uri: Uri::from_str("file:///tmp/model.flatppl").unwrap(),
                language_id: "flatppl".into(),
                version: 1,
                text: "c = load_module(\"https://h.example/ex/common.flatppl\")\n\
                       v = add(c.m, 1.0)\n"
                    .into(),
            },
        };
        client_conn
            .sender
            .send(Message::Notification(lsp_server::Notification::new(
                DidOpenTextDocument::METHOD.to_owned(),
                open,
            )))
            .unwrap();
        // Drain the model's initial diagnostics (URL dep not yet fed).
        let _ = client_conn
            .receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("expected initial publishDiagnostics for model");

        // Push the fetched URL sources: `common` (which loads `priors.flatppl`
        // RELATIVE to its own URL) and `priors`.
        let url_sources = serde_json::json!({
            "sources": [
                {
                    "uri": "https://h.example/ex/common.flatppl",
                    "text": "pr = load_module(\"priors.flatppl\")\nm = add(pr.theta, 1.0)\n"
                },
                {
                    "uri": "https://h.example/ex/priors.flatppl",
                    "text": "theta = elementof(reals)\n"
                }
            ]
        });
        client_conn
            .sender
            .send(Message::Notification(lsp_server::Notification::new(
                "flatppl/urlSources".to_owned(),
                url_sources,
            )))
            .unwrap();
        // Drain the model's re-published diagnostics after the feed.
        let _ = client_conn
            .receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("expected re-published diagnostics after urlSources feed");

        // Hover on `add(c.m, 1.0)` (line 1, char 4 = 'a' of 'add'). Resolves to
        // Scalar(Real) only if the whole URL chain resolved from fed content.
        let hover_params = lsp_types::HoverParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: Uri::from_str("file:///tmp/model.flatppl").unwrap(),
                },
                position: lsp_types::Position {
                    line: 1,
                    character: 4,
                },
            },
            work_done_progress_params: Default::default(),
        };
        client_conn
            .sender
            .send(Message::Request(lsp_server::Request::new(
                lsp_server::RequestId::from(42i32),
                HoverRequest::METHOD.to_owned(),
                serde_json::to_value(hover_params).unwrap(),
            )))
            .unwrap();
        let resp_msg = client_conn
            .receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("timed out waiting for hover response");
        let Message::Response(resp) = resp_msg else {
            panic!("expected a Response, got: {resp_msg:?}");
        };
        let result = resp.result.expect("hover result present");
        assert!(
            result != serde_json::Value::Null && result.to_string().to_lowercase().contains("real"),
            "hover on `c.m` must resolve through the fed URL chain to a real type; got: {result}"
        );

        // Shutdown.
        client_conn
            .sender
            .send(Message::Request(lsp_server::Request::new(
                lsp_server::RequestId::from(99i32),
                "shutdown".into(),
                serde_json::Value::Null,
            )))
            .unwrap();
        let _ = client_conn
            .receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .ok();
        client_conn
            .sender
            .send(Message::Notification(lsp_server::Notification::new(
                "exit".into(),
                serde_json::Value::Null,
            )))
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

        sync_file_set(&mut db, fs, &uri_to_file, &HashMap::new());

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
        sync_file_set(&mut db, fs, &uri_to_file, &HashMap::new());

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
