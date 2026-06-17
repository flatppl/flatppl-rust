//! The FlatPPL stdio message loop.
//!
//! [`run`] drives the main LSP event loop after the initialize handshake has
//! already completed. It owns the salsa [`Database`], the open-document map,
//! the workspace [`FileSet`], and the external [`Catalogues`]; it processes
//! `didOpen`/`didChange` notifications (full-sync), `hover` requests, and
//! `shutdown`.

use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

use lsp_server::{Connection, Message, Response};
use lsp_types::{
    CompletionOptions, HoverProviderCapability, OneOf, PublishDiagnosticsParams,
    ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind, Uri,
    notification::{
        DidChangeTextDocument, DidOpenTextDocument, Notification as _, PublishDiagnostics,
    },
    request::{
        Completion, DocumentSymbolRequest, GotoDefinition, HoverRequest, InlayHintRequest,
        Request as _, WorkspaceSymbolRequest,
    },
};

use crate::db::{Catalogues, Database, FileSet, SourceFile};
use crate::line_index::{LineIndex, Pos};

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

    // Publish initial diagnostics for all workspace files.
    for (uri_str, &file) in &uri_to_file {
        publish_diagnostics(&connection, &db, file, fs, cats, uri_str)?;
    }

    // ── Main loop ────────────────────────────────────────────────────────────

    for msg in &connection.receiver {
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
                        upsert_file(&mut db, &mut uri_to_file, uri_str, text);
                        // Update the single shared FileSet so hover and future
                        // analyses see the newly-opened file.
                        {
                            use salsa::Setter;
                            let new_files: Vec<SourceFile> =
                                uri_to_file.values().copied().collect();
                            fs.set_files(&mut db).to(new_files);
                        }
                        // Re-publish diagnostics for ALL open docs so that
                        // files that import the newly-opened one also refresh
                        // (symmetric with didChange).
                        let open_files: Vec<(String, SourceFile)> =
                            uri_to_file.iter().map(|(u, &f)| (u.clone(), f)).collect();
                        for (doc_uri_str, file) in open_files {
                            publish_diagnostics(&connection, &db, file, fs, cats, &doc_uri_str)?;
                        }
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
                        // Full sync — take last content change.
                        if let Some(change) = p.content_changes.into_iter().last() {
                            upsert_file(&mut db, &mut uri_to_file, uri_str.clone(), change.text);
                        }
                        // Update the single shared FileSet in place so hover
                        // always sees the current document set.
                        {
                            use salsa::Setter;
                            let new_files: Vec<SourceFile> =
                                uri_to_file.values().copied().collect();
                            fs.set_files(&mut db).to(new_files);
                        }
                        // Re-publish diagnostics for ALL open docs so cross-file
                        // dependency edges are re-evaluated.
                        let open_files: Vec<(String, SourceFile)> =
                            uri_to_file.iter().map(|(u, &f)| (u.clone(), f)).collect();
                        for (doc_uri_str, file) in open_files {
                            publish_diagnostics(&connection, &db, file, fs, cats, &doc_uri_str)?;
                        }
                    }
                    _ => {} // ignore other notifications
                }
            }
            Message::Request(req) => {
                // Handle shutdown first.
                if connection.handle_shutdown(&req)? {
                    break;
                }

                match req.method.as_str() {
                    HoverRequest::METHOD => {
                        let hover_resp = handle_hover(&db, &uri_to_file, fs, cats, &req);
                        connection.sender.send(Message::Response(hover_resp))?;
                    }
                    DocumentSymbolRequest::METHOD => {
                        let sym_resp = handle_document_symbols(&db, &uri_to_file, fs, cats, &req);
                        connection.sender.send(Message::Response(sym_resp))?;
                    }
                    WorkspaceSymbolRequest::METHOD => {
                        let sym_resp = handle_workspace_symbols(&db, fs, cats, &req);
                        connection.sender.send(Message::Response(sym_resp))?;
                    }
                    InlayHintRequest::METHOD => {
                        let hints_resp = handle_inlay_hints(&db, &uri_to_file, fs, cats, &req);
                        connection.sender.send(Message::Response(hints_resp))?;
                    }
                    GotoDefinition::METHOD => {
                        let def_resp = handle_goto_definition(&db, &uri_to_file, fs, cats, &req);
                        connection.sender.send(Message::Response(def_resp))?;
                    }
                    Completion::METHOD => {
                        let comp_resp = handle_completion(&db, &uri_to_file, fs, cats, &req);
                        connection.sender.send(Message::Response(comp_resp))?;
                    }
                    _ => {
                        // Unknown request: reply with MethodNotFound.
                        let resp = Response::new_err(
                            req.id.clone(),
                            lsp_server::ErrorCode::MethodNotFound as i32,
                            format!("unsupported method: {}", req.method),
                        );
                        connection.sender.send(Message::Response(resp))?;
                    }
                }
            }
            Message::Response(_) => {} // ignore server-originated response echoes
        }
    }

    Ok(())
}

// ── Capability advertisement ─────────────────────────────────────────────────

/// Build the `ServerCapabilities` value we advertise during `initialize`.
pub fn server_capabilities() -> ServerCapabilities {
    ServerCapabilities {
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        document_symbol_provider: Some(OneOf::Left(true)),
        workspace_symbol_provider: Some(OneOf::Left(true)),
        inlay_hint_provider: Some(OneOf::Left(true)),
        definition_provider: Some(OneOf::Left(true)),
        completion_provider: Some(CompletionOptions {
            trigger_characters: Some(vec![".".to_string()]),
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
            let uri_str = format!("file://{path_str}");
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
) -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
    let diagnostics = crate::capabilities::diagnostics(db, file, fs, cats);
    let uri = Uri::from_str(uri_str)?;
    let params = PublishDiagnosticsParams {
        uri,
        diagnostics,
        version: None,
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
        let text = file.text(db);
        let li = LineIndex::new(text);
        let byte_offset = li.offset(Pos {
            line: lsp_pos.line,
            character: lsp_pos.character,
        });
        let markdown = crate::capabilities::hover(db, file, fs, cats, byte_offset)?;
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
    fs: FileSet,
    cats: Catalogues,
    req: &lsp_server::Request,
) -> Response {
    let syms: Vec<lsp_types::DocumentSymbol> = (|| {
        let params: lsp_types::DocumentSymbolParams =
            serde_json::from_value(req.params.clone()).ok()?;
        let uri_str = params.text_document.uri.as_str().to_owned();
        let file = *uri_to_file.get(&uri_str)?;
        Some(crate::capabilities::document_symbols(db, file, fs, cats))
    })()
    .unwrap_or_default();

    let response = lsp_types::DocumentSymbolResponse::Nested(syms);
    Response::new_ok(req.id.clone(), response)
}

/// Handle a `workspace/symbol` request.  Returns a `Response` (result or
/// null) without sending it — the caller dispatches the message.
fn handle_workspace_symbols(
    db: &Database,
    fs: FileSet,
    cats: Catalogues,
    req: &lsp_server::Request,
) -> Response {
    let query = (|| -> Option<String> {
        let params: lsp_types::WorkspaceSymbolParams =
            serde_json::from_value(req.params.clone()).ok()?;
        Some(params.query)
    })()
    .unwrap_or_default();

    let syms = crate::capabilities::workspace_symbols(db, fs, cats, &query);
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
        let text = file.text(db);
        let li = LineIndex::new(text);
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
        let text = file.text(db);
        let li = LineIndex::new(text);
        let byte_offset = li.offset(Pos {
            line: lsp_pos.line,
            character: lsp_pos.character,
        });
        let def_loc = crate::capabilities::goto_definition(db, file, fs, cats, byte_offset)?;
        // Build the target URI from the DefLoc path.
        let target_uri_str = if def_loc.path.starts_with("file://") {
            def_loc.path.clone()
        } else {
            format!("file://{}", def_loc.path)
        };
        let target_uri = Uri::from_str(&target_uri_str).ok()?;
        // Build the target range: need the text of the target file to do
        // byte→position conversion.
        let target_text = fs
            .files(db)
            .iter()
            .copied()
            .find(|f| f.path(db) == def_loc.path)
            .map(|f| f.text(db).to_string())
            .unwrap_or_default();
        let target_li = LineIndex::new(&target_text);
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
        let text = file.text(db);
        let li = LineIndex::new(text);
        let byte_offset = li.offset(Pos {
            line: lsp_pos.line,
            character: lsp_pos.character,
        });
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
            Some([".".to_string()].as_slice()),
            "completion trigger character must be '.'"
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
}
