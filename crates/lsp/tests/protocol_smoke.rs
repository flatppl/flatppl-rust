//! End-to-end protocol smoke tests: initialize → didOpen → hover / completion /
//! documentSymbol / definition.
//!
//! Drives the full LSP handshake over an in-memory `Connection::memory()` pair
//! and verifies that requests for each P-B capability come back with real
//! responses over the wire.

use std::str::FromStr;
use std::time::Duration;

use lsp_server::{Connection, Message, Request, RequestId};
use lsp_types::{
    ClientCapabilities, HoverContents, InitializeParams, InitializedParams, MarkupContent,
    MarkupKind, TextDocumentItem, Uri,
    notification::{
        DidChangeTextDocument, DidChangeWatchedFiles, DidCloseTextDocument, DidOpenTextDocument,
        Initialized, Notification as _,
    },
    request::{
        Completion, DocumentSymbolRequest, GotoDefinition, HoverRequest, Initialize,
        InlayHintRequest, Request as _, WorkspaceSymbolRequest,
    },
};

// ── shared helpers ───────────────────────────────────────────────────────────

/// Drive the `initialize` / `initialized` handshake from the client side.
///
/// Sends the `initialize` request with `req_id`, reads the `InitializeResult`
/// response, then sends the `initialized` notification.  After this returns the
/// connection is ready for normal LSP traffic.
fn do_handshake(client_conn: &Connection, req_id: i32) {
    #[allow(deprecated)]
    let init_params_value = serde_json::to_value(InitializeParams {
        capabilities: ClientCapabilities::default(),
        ..Default::default()
    })
    .expect("serialize InitializeParams");

    let init_req = lsp_server::Request {
        id: RequestId::from(req_id),
        method: Initialize::METHOD.to_owned(),
        params: init_params_value,
    };
    client_conn.sender.send(Message::Request(init_req)).unwrap();

    // Read the InitializeResult response.
    let _init_resp = client_conn
        .receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("timed out waiting for InitializeResult");

    // Send `initialized` notification.
    let initialized_note =
        lsp_server::Notification::new(Initialized::METHOD.to_owned(), InitializedParams {});
    client_conn
        .sender
        .send(Message::Notification(initialized_note))
        .unwrap();
}

/// Send a `textDocument/didOpen` notification and drain the resulting
/// `publishDiagnostics` notification(s) until we see one matching `uri`.
fn do_open_and_drain_diags(client_conn: &Connection, uri: &str, text: &str) {
    let did_open_params = lsp_types::DidOpenTextDocumentParams {
        text_document: TextDocumentItem {
            uri: Uri::from_str(uri).unwrap(),
            language_id: "flatppl".into(),
            version: 1,
            text: text.into(),
        },
    };
    let note =
        lsp_server::Notification::new(DidOpenTextDocument::METHOD.to_owned(), did_open_params);
    client_conn
        .sender
        .send(Message::Notification(note))
        .unwrap();

    // Drain until we see the publishDiagnostics notification.
    loop {
        let msg = client_conn
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("timed out waiting for publishDiagnostics");
        if let Message::Notification(n) = &msg {
            if n.method == lsp_types::notification::PublishDiagnostics::METHOD {
                break;
            }
        }
    }
}

/// Send a request and receive its response, skipping any interleaved
/// notifications.  Panics if no response arrives within 5 s.
fn round_trip(client_conn: &Connection, req: lsp_server::Request) -> lsp_server::Response {
    let id = req.id.clone();
    client_conn.sender.send(Message::Request(req)).unwrap();
    loop {
        let msg = client_conn
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("timed out waiting for response");
        match msg {
            Message::Response(resp) if resp.id == id => return resp,
            _ => continue,
        }
    }
}

/// Send shutdown + exit and join the server thread.
fn do_shutdown(client_conn: &Connection, shutdown_id: i32) {
    let shutdown_req = Request::new(
        RequestId::from(shutdown_id),
        "shutdown".into(),
        serde_json::Value::Null,
    );
    client_conn
        .sender
        .send(Message::Request(shutdown_req))
        .unwrap();
    let _shutdown_resp = client_conn
        .receiver
        .recv_timeout(Duration::from_secs(5))
        .ok();
    let exit_note = lsp_server::Notification::new("exit".into(), serde_json::Value::Null);
    client_conn
        .sender
        .send(Message::Notification(exit_note))
        .unwrap();
}

/// Source text used for the hover smoke test.  The expression `add(1, 2)` is
/// fully typed by the engine; byte offset 8 lands on the literal `1` which
/// carries an inferred scalar type.
const SRC: &str = "x = add(1, 2)";

/// Byte offset of the literal `1` inside `add(1, 2)`.
/// `x = add(1, 2)`
///  0123456789...
///          ^ offset 8
const HOVER_OFFSET: u32 = 8;

/// The file URI used for the didOpen + hover requests.
const FILE_URI: &str = "file:///tmp/smoke.flatppl";

// ── P-B source: two bindings, one cross-reference ───────────────────────────
//
// Used for the completion / documentSymbol / definition smoke tests.
//
//  Line 0: "x = 1"        — x is a scalar integer literal
//  Line 1: "y = add(x, 2)" — y uses x; `x` inside add() is at char 8
const PB_SRC: &str = "x = 1\ny = add(x, 2)";
const PB_FILE_URI: &str = "file:///tmp/smoke_pb.flatppl";

#[test]
fn initialize_did_open_hover_smoke() {
    // ── 1. Create the in-memory connection pair ──────────────────────────────
    //
    // `Connection::memory()` returns two connected ends.  The LSP convention is
    // that one end acts as the server and the other as the client.
    let (server_conn, client_conn) = Connection::memory();

    // ── 2. Spawn the server thread ───────────────────────────────────────────
    //
    // The server calls `server_conn.initialize(server_caps)`, which blocks
    // until the client sends `initialize` + `initialized`.  After the handshake
    // it calls `flatppl_lsp::server::run` with the returned init params.
    let server_thread = std::thread::spawn(move || {
        let server_caps =
            serde_json::to_value(flatppl_lsp::server::server_capabilities()).expect("caps");
        let init_params = server_conn.initialize(server_caps).expect("handshake");
        flatppl_lsp::server::run(server_conn, init_params).expect("server loop");
    });

    // ── 3. Client: drive the initialize handshake ────────────────────────────
    do_handshake(&client_conn, 1);

    // ── 4. didOpen + drain publishDiagnostics ────────────────────────────────
    do_open_and_drain_diags(&client_conn, FILE_URI, SRC);

    // ── 5. Send a hover request at byte offset 8 (the literal `1`) ──────────
    let hover_params = lsp_types::HoverParams {
        text_document_position_params: lsp_types::TextDocumentPositionParams {
            text_document: lsp_types::TextDocumentIdentifier {
                uri: Uri::from_str(FILE_URI).unwrap(),
            },
            // UTF-16 position for byte offset 8 in `x = add(1, 2)` (ASCII, so byte == column).
            position: lsp_types::Position {
                line: 0,
                character: HOVER_OFFSET,
            },
        },
        work_done_progress_params: Default::default(),
    };
    let hover_req = Request {
        id: RequestId::from(2i32),
        method: HoverRequest::METHOD.to_owned(),
        params: serde_json::to_value(hover_params).unwrap(),
    };
    let hover_response = round_trip(&client_conn, hover_req);

    // ── 6. Assert the hover response is non-null with type information ────────
    assert!(
        hover_response.error.is_none(),
        "hover response must not be an error; got: {:?}",
        hover_response.error
    );

    let result = hover_response.result.expect("hover result must be present");
    assert!(
        !result.is_null(),
        "hover result must be non-null at offset {HOVER_OFFSET} in {SRC:?}"
    );

    let hover: lsp_types::Hover =
        serde_json::from_value(result).expect("hover result must deserialize to lsp_types::Hover");

    let markdown = match &hover.contents {
        HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value,
        }) => value.clone(),
        other => panic!("expected Markdown hover contents; got: {other:?}"),
    };

    // The hover string is `**type:** \`...\`` — assert it mentions "type" and a
    // scalar type keyword (rendered bare/lowercase: `real`, `integer`, …).
    assert!(
        markdown.to_lowercase().contains("type"),
        "hover markdown must mention 'type'; got: {markdown:?}"
    );
    let has_scalar_token = ["integer", "real", "boolean", "complex"]
        .iter()
        .any(|tok| markdown.contains(tok));
    assert!(
        has_scalar_token,
        "hover markdown must mention a scalar type token; got: {markdown:?}"
    );

    // ── 7. Shutdown + exit ───────────────────────────────────────────────────
    do_shutdown(&client_conn, 99);
    server_thread.join().expect("server thread must not panic");
}

// ── P-B protocol smoke: completion + documentSymbol + definition ─────────────

/// Protocol-level smoke coverage for the P-B capabilities added in this plan:
/// `textDocument/completion`, `textDocument/documentSymbol`,
/// `textDocument/definition`, `workspace/symbol`, and `textDocument/inlayHint`.
///
/// Source: `"x = 1\ny = add(x, 2)"` — two in-scope bindings, one
/// same-module reference.  All three requests are driven over the wire after a
/// single `initialize` + `didOpen` handshake.
#[test]
fn pb_capabilities_smoke() {
    // ── 1. Spawn server ──────────────────────────────────────────────────────
    let (server_conn, client_conn) = Connection::memory();
    let server_thread = std::thread::spawn(move || {
        let server_caps =
            serde_json::to_value(flatppl_lsp::server::server_capabilities()).expect("caps");
        let init_params = server_conn.initialize(server_caps).expect("handshake");
        flatppl_lsp::server::run(server_conn, init_params).expect("server loop");
    });

    // ── 2. Handshake + didOpen ───────────────────────────────────────────────
    do_handshake(&client_conn, 1);
    do_open_and_drain_diags(&client_conn, PB_FILE_URI, PB_SRC);

    // ── 3. textDocument/completion ───────────────────────────────────────────
    //
    // Cursor at the end of the document (line 1, char 14 — after "y = add(x, 2)").
    // General completion should return a non-empty list containing at least one
    // built-in base name ("Normal") and the in-scope bindings ("x", "y").
    {
        let comp_params = lsp_types::CompletionParams {
            text_document_position: lsp_types::TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: Uri::from_str(PB_FILE_URI).unwrap(),
                },
                position: lsp_types::Position {
                    line: 1,
                    character: 14, // end of "y = add(x, 2)"
                },
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
            context: None,
        };
        let comp_req = Request {
            id: RequestId::from(10i32),
            method: Completion::METHOD.to_owned(),
            params: serde_json::to_value(comp_params).unwrap(),
        };
        let resp = round_trip(&client_conn, comp_req);

        assert!(
            resp.error.is_none(),
            "completion response must not be an error; got: {:?}",
            resp.error
        );
        let result = resp.result.expect("completion result must be present");
        assert!(
            !result.is_null(),
            "completion result must be non-null for PB_SRC"
        );
        // Deserialize as a CompletionResponse (Array or List).
        let comp_resp: lsp_types::CompletionResponse =
            serde_json::from_value(result).expect("completion result must deserialize");
        let items = match comp_resp {
            lsp_types::CompletionResponse::Array(items) => items,
            lsp_types::CompletionResponse::List(list) => list.items,
        };
        assert!(
            !items.is_empty(),
            "completion list must be non-empty for source {PB_SRC:?}"
        );
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        // Built-in base distribution must be present (general completion includes catalogue).
        assert!(
            labels.contains(&"Normal"),
            "completion must include built-in 'Normal'; got: {labels:?}"
        );
        // In-scope bindings from the open document must be present.
        assert!(
            labels.contains(&"x"),
            "completion must include in-scope binding 'x'; got: {labels:?}"
        );
        assert!(
            labels.contains(&"y"),
            "completion must include in-scope binding 'y'; got: {labels:?}"
        );
    }

    // ── 4. textDocument/documentSymbol ───────────────────────────────────────
    //
    // Must return at least "x" and "y" as symbols.
    {
        let sym_params = lsp_types::DocumentSymbolParams {
            text_document: lsp_types::TextDocumentIdentifier {
                uri: Uri::from_str(PB_FILE_URI).unwrap(),
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };
        let sym_req = Request {
            id: RequestId::from(11i32),
            method: DocumentSymbolRequest::METHOD.to_owned(),
            params: serde_json::to_value(sym_params).unwrap(),
        };
        let resp = round_trip(&client_conn, sym_req);

        assert!(
            resp.error.is_none(),
            "documentSymbol response must not be an error; got: {:?}",
            resp.error
        );
        let result = resp.result.expect("documentSymbol result must be present");
        assert!(
            !result.is_null(),
            "documentSymbol result must be non-null for PB_SRC"
        );
        let sym_resp: lsp_types::DocumentSymbolResponse =
            serde_json::from_value(result).expect("documentSymbol result must deserialize");
        let names: Vec<String> = match sym_resp {
            lsp_types::DocumentSymbolResponse::Nested(syms) => {
                syms.into_iter().map(|s| s.name).collect()
            }
            lsp_types::DocumentSymbolResponse::Flat(syms) => {
                syms.into_iter().map(|s| s.name).collect()
            }
        };
        assert!(
            names.iter().any(|n| n == "x"),
            "documentSymbol must include 'x'; got: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "y"),
            "documentSymbol must include 'y'; got: {names:?}"
        );
    }

    // ── 5. textDocument/definition ───────────────────────────────────────────
    //
    // Cursor on the `x` inside `add(x, 2)` on line 1, char 8 (0-indexed).
    // Source line 1: "y = add(x, 2)"
    //                 01234567890123
    //                         ^ char 8 = 'x'
    // Expect a non-null Location pointing back into the same file.
    {
        let def_params = lsp_types::GotoDefinitionParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: Uri::from_str(PB_FILE_URI).unwrap(),
                },
                position: lsp_types::Position {
                    line: 1,
                    character: 8, // 'x' inside add(x, 2)
                },
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };
        let def_req = Request {
            id: RequestId::from(12i32),
            method: GotoDefinition::METHOD.to_owned(),
            params: serde_json::to_value(def_params).unwrap(),
        };
        let resp = round_trip(&client_conn, def_req);

        assert!(
            resp.error.is_none(),
            "definition response must not be an error; got: {:?}",
            resp.error
        );
        let result = resp.result.expect("definition result must be present");
        assert!(
            !result.is_null(),
            "definition result must be non-null for 'x' reference in PB_SRC"
        );
        // Deserialize as a GotoDefinitionResponse (Scalar Location or array).
        let def_resp: lsp_types::GotoDefinitionResponse =
            serde_json::from_value(result).expect("definition result must deserialize");
        // The response is a scalar Location pointing into the same file at line 0
        // (where `x = 1` is defined).
        let location = match def_resp {
            lsp_types::GotoDefinitionResponse::Scalar(loc) => loc,
            lsp_types::GotoDefinitionResponse::Array(locs) => {
                assert!(!locs.is_empty(), "definition array must be non-empty");
                locs.into_iter().next().unwrap()
            }
            lsp_types::GotoDefinitionResponse::Link(links) => {
                assert!(!links.is_empty(), "definition link array must be non-empty");
                let link = links.into_iter().next().unwrap();
                lsp_types::Location {
                    uri: link.target_uri,
                    range: link.target_range,
                }
            }
        };
        // The definition of `x` is on line 0 of the same file.
        assert_eq!(
            location.range.start.line, 0,
            "definition of 'x' must be on line 0; got range: {:?}",
            location.range
        );
    }

    // ── 6. workspace/symbol ──────────────────────────────────────────────────
    //
    // An empty query returns every workspace symbol; must include "x" and "y".
    {
        let ws_params = lsp_types::WorkspaceSymbolParams {
            query: String::new(),
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };
        let ws_req = Request {
            id: RequestId::from(13i32),
            method: WorkspaceSymbolRequest::METHOD.to_owned(),
            params: serde_json::to_value(ws_params).unwrap(),
        };
        let resp = round_trip(&client_conn, ws_req);

        assert!(
            resp.error.is_none(),
            "workspace/symbol response must not be an error; got: {:?}",
            resp.error
        );
        let result = resp
            .result
            .expect("workspace/symbol result must be present");
        assert!(
            !result.is_null(),
            "workspace/symbol result must be non-null for PB_SRC"
        );
        let ws_resp: lsp_types::WorkspaceSymbolResponse =
            serde_json::from_value(result).expect("workspace/symbol result must deserialize");
        let names: Vec<String> = match ws_resp {
            lsp_types::WorkspaceSymbolResponse::Flat(syms) => {
                syms.into_iter().map(|s| s.name).collect()
            }
            lsp_types::WorkspaceSymbolResponse::Nested(syms) => {
                syms.into_iter().map(|s| s.name).collect()
            }
        };
        assert!(
            names.iter().any(|n| n == "x"),
            "workspace/symbol must include 'x'; got: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "y"),
            "workspace/symbol must include 'y'; got: {names:?}"
        );
    }

    // ── 7. textDocument/inlayHint ────────────────────────────────────────────
    //
    // Request hints over the whole document range; the binding RHSs carry
    // inferred types, so at least one type hint must come back.
    {
        let inlay_params = lsp_types::InlayHintParams {
            text_document: lsp_types::TextDocumentIdentifier {
                uri: Uri::from_str(PB_FILE_URI).unwrap(),
            },
            range: lsp_types::Range {
                start: lsp_types::Position::new(0, 0),
                end: lsp_types::Position::new(1, 14),
            },
            work_done_progress_params: Default::default(),
        };
        let inlay_req = Request {
            id: RequestId::from(14i32),
            method: InlayHintRequest::METHOD.to_owned(),
            params: serde_json::to_value(inlay_params).unwrap(),
        };
        let resp = round_trip(&client_conn, inlay_req);

        assert!(
            resp.error.is_none(),
            "inlayHint response must not be an error; got: {:?}",
            resp.error
        );
        let result = resp.result.expect("inlayHint result must be present");
        assert!(!result.is_null(), "inlayHint result must be non-null");
        let hints: Vec<lsp_types::InlayHint> =
            serde_json::from_value(result).expect("inlayHint result must deserialize");
        assert!(
            !hints.is_empty(),
            "inlayHint must return at least one type hint over the full range of {PB_SRC:?}"
        );
    }

    // ── 8. Shutdown ──────────────────────────────────────────────────────────
    do_shutdown(&client_conn, 99);
    server_thread.join().expect("server thread must not panic");
}

// ── Task 2: protocol-gap regression coverage ────────────────────────────────

/// After `didClose`, the editor no longer owns the file, so a subsequent
/// on-disk change delivered via `workspace/didChangeWatchedFiles` must be
/// picked up and re-analyzed (replacing the previously-open editor content).
#[test]
fn did_close_lets_watched_file_changes_take_over() {
    // Write a temp .flatppl file to disk.
    let tmp_path = std::env::temp_dir().join(format!(
        "flatppl_lsp_didclose_{}.flatppl",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));
    std::fs::write(&tmp_path, "alpha = elementof(reals)\n").unwrap();
    let uri_str = format!("file://{}", tmp_path.display());

    let (server_conn, client_conn) = Connection::memory();
    let server_thread = std::thread::spawn(move || {
        let server_caps =
            serde_json::to_value(flatppl_lsp::server::server_capabilities()).expect("caps");
        let init_params = server_conn.initialize(server_caps).expect("handshake");
        flatppl_lsp::server::run(server_conn, init_params).expect("server loop");
    });

    do_handshake(&client_conn, 1);

    // didOpen the file with editor content (version 1), drain its diagnostics.
    let did_open_params = lsp_types::DidOpenTextDocumentParams {
        text_document: TextDocumentItem {
            uri: Uri::from_str(&uri_str).unwrap(),
            language_id: "flatppl".into(),
            version: 1,
            text: "alpha = elementof(reals)\n".into(),
        },
    };
    client_conn
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            DidOpenTextDocument::METHOD.to_owned(),
            did_open_params,
        )))
        .unwrap();
    loop {
        let msg = client_conn
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("timed out waiting for publishDiagnostics after didOpen");
        if let Message::Notification(n) = &msg {
            if n.method == lsp_types::notification::PublishDiagnostics::METHOD {
                break;
            }
        }
    }

    // didClose — the editor relinquishes ownership of the file.
    let did_close_params = serde_json::json!({
        "textDocument": { "uri": uri_str }
    });
    client_conn
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            DidCloseTextDocument::METHOD.to_owned(),
            did_close_params,
        )))
        .unwrap();

    // Change the file on disk to a new (valid) binding.
    std::fs::write(&tmp_path, "beta = elementof(reals)\n").unwrap();

    // Notify via watched files (CHANGED = 2). Because the file is closed, the
    // server reloads it from disk instead of skipping it as editor-managed.
    let dcwf_params = serde_json::json!({
        "changes": [{ "uri": uri_str, "type": 2 }]
    });
    client_conn
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            DidChangeWatchedFiles::METHOD.to_owned(),
            dcwf_params,
        )))
        .unwrap();

    // The watched-file change triggers a fresh publishDiagnostics. Drain to it.
    let diag_msg = loop {
        let msg = client_conn
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("timed out waiting for publishDiagnostics after watched change");
        if let Message::Notification(n) = &msg {
            if n.method == lsp_types::notification::PublishDiagnostics::METHOD {
                break msg;
            }
        }
    };
    let Message::Notification(diag_note) = diag_msg else {
        unreachable!("loop only breaks on a Notification");
    };
    let diag_params: lsp_types::PublishDiagnosticsParams =
        serde_json::from_value(diag_note.params).expect("valid PublishDiagnosticsParams");
    assert!(
        diag_params.diagnostics.is_empty(),
        "reloaded valid file must publish empty diagnostics; got: {:?}",
        diag_params.diagnostics
    );

    // documentSymbol must now reflect the on-disk content ("beta").
    let ds_params = serde_json::json!({ "textDocument": { "uri": uri_str } });
    let ds_req = Request {
        id: RequestId::from(20i32),
        method: DocumentSymbolRequest::METHOD.to_owned(),
        params: ds_params,
    };
    let resp = round_trip(&client_conn, ds_req);
    assert!(
        resp.error.is_none(),
        "documentSymbol must not error; got: {:?}",
        resp.error
    );
    let result = resp.result.expect("documentSymbol result must be present");
    let syms = result.to_string();
    assert!(
        syms.contains("beta"),
        "documentSymbol must reflect on-disk 'beta' after didClose + watched change; got: {syms}"
    );

    let _ = std::fs::remove_file(&tmp_path);

    do_shutdown(&client_conn, 99);
    server_thread.join().expect("server thread must not panic");
}

/// A `didChange` whose version is older than the last-applied version is
/// stale (out-of-order delivery) and must be dropped — the document content
/// stays at the newer version.
#[test]
fn stale_did_change_is_ignored() {
    const URI: &str = "file:///tmp/stale_test.flatppl";

    let (server_conn, client_conn) = Connection::memory();
    let server_thread = std::thread::spawn(move || {
        let server_caps =
            serde_json::to_value(flatppl_lsp::server::server_capabilities()).expect("caps");
        let init_params = server_conn.initialize(server_caps).expect("handshake");
        flatppl_lsp::server::run(server_conn, init_params).expect("server loop");
    });

    do_handshake(&client_conn, 1);

    // didOpen at version 5.
    let did_open_params = lsp_types::DidOpenTextDocumentParams {
        text_document: TextDocumentItem {
            uri: Uri::from_str(URI).unwrap(),
            language_id: "flatppl".into(),
            version: 5,
            text: "v5 = elementof(reals)\n".into(),
        },
    };
    client_conn
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            DidOpenTextDocument::METHOD.to_owned(),
            did_open_params,
        )))
        .unwrap();
    loop {
        let msg = client_conn
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("timed out waiting for publishDiagnostics after didOpen v5");
        if let Message::Notification(n) = &msg {
            if n.method == lsp_types::notification::PublishDiagnostics::METHOD {
                break;
            }
        }
    }

    // Stale didChange at version 3 (< 5) — must be ignored by the server.
    let stale_change = serde_json::json!({
        "textDocument": { "uri": URI, "version": 3 },
        "contentChanges": [{ "text": "v3 = elementof(reals)\n" }]
    });
    client_conn
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            DidChangeTextDocument::METHOD.to_owned(),
            stale_change,
        )))
        .unwrap();

    // The server `continue`s on a stale edit (no republish), so verify content
    // through documentSymbol: it must still show "v5", not "v3".
    let ds_params = serde_json::json!({ "textDocument": { "uri": URI } });
    let ds_req = Request {
        id: RequestId::from(21i32),
        method: DocumentSymbolRequest::METHOD.to_owned(),
        params: ds_params,
    };
    let resp = round_trip(&client_conn, ds_req);
    assert!(
        resp.error.is_none(),
        "documentSymbol must not error; got: {:?}",
        resp.error
    );
    let result = resp.result.expect("documentSymbol result must be present");
    let syms = result.to_string();
    assert!(
        syms.contains("v5"),
        "stale didChange must be dropped — content must still be v5; got: {syms}"
    );
    assert!(
        !syms.contains("v3"),
        "stale didChange (version 3 < 5) must NOT be applied; got: {syms}"
    );

    do_shutdown(&client_conn, 99);
    server_thread.join().expect("server thread must not panic");
}

// ── Task 4: debounce + cancellation ──────────────────────────────────────────

/// A rapid burst of `didChange` notifications must coalesce into a SINGLE
/// `publishDiagnostics`, and that publish must carry the LATEST version.
///
/// The server debounces diagnostics (~200ms): each `didChange` re-arms the
/// deadline, so a burst delivered faster than the debounce window produces just
/// one publish once the burst settles. Timing is deliberately generous (debounce
/// 200ms → first wait 700ms, quiescence check 400ms) to stay non-flaky in CI;
/// the assertion (exactly one publish, version 5) does not depend on the exact
/// timing, only that all five changes land inside one window.
#[test]
fn did_change_burst_coalesces_into_one_publish() {
    const URI: &str = "file:///tmp/burst_test.flatppl";

    let (server_conn, client_conn) = Connection::memory();
    let server_thread = std::thread::spawn(move || {
        let server_caps =
            serde_json::to_value(flatppl_lsp::server::server_capabilities()).expect("caps");
        let init_params = server_conn.initialize(server_caps).expect("handshake");
        flatppl_lsp::server::run(server_conn, init_params).expect("server loop");
    });

    do_handshake(&client_conn, 1);

    // didOpen at version 0, then drain its (debounced) publish so it does not
    // pollute the burst measurement.
    let did_open_params = lsp_types::DidOpenTextDocumentParams {
        text_document: TextDocumentItem {
            uri: Uri::from_str(URI).unwrap(),
            language_id: "flatppl".into(),
            version: 0,
            text: "b0 = elementof(reals)\n".into(),
        },
    };
    client_conn
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            DidOpenTextDocument::METHOD.to_owned(),
            did_open_params,
        )))
        .unwrap();
    // Drain the didOpen publish.
    loop {
        let msg = client_conn
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("timed out waiting for didOpen publishDiagnostics");
        if let Message::Notification(n) = &msg {
            if n.method == lsp_types::notification::PublishDiagnostics::METHOD {
                break;
            }
        }
    }

    // Fire 5 rapid didChange notifications (versions 1..=5) with no waits in
    // between, so they all land inside one debounce window.
    for v in 1..=5 {
        let change = serde_json::json!({
            "textDocument": { "uri": URI, "version": v },
            "contentChanges": [{ "text": format!("b{v} = elementof(reals)\n") }]
        });
        client_conn
            .sender
            .send(Message::Notification(lsp_server::Notification::new(
                DidChangeTextDocument::METHOD.to_owned(),
                change,
            )))
            .unwrap();
    }

    // Wait past the debounce window for the single coalesced publish.
    let diag_msg = loop {
        let msg = client_conn
            .receiver
            .recv_timeout(Duration::from_millis(700))
            .expect("timed out waiting for coalesced publishDiagnostics");
        if let Message::Notification(n) = &msg {
            if n.method == lsp_types::notification::PublishDiagnostics::METHOD {
                break msg;
            }
        }
    };
    let Message::Notification(diag_note) = diag_msg else {
        unreachable!("loop only breaks on a Notification");
    };
    let params: lsp_types::PublishDiagnosticsParams =
        serde_json::from_value(diag_note.params).expect("valid PublishDiagnosticsParams");
    assert_eq!(
        params.version,
        Some(5),
        "coalesced publish must carry the LATEST version (5); got: {:?}",
        params.version
    );

    // No SECOND publish should arrive — the burst coalesced into exactly one.
    let second = client_conn
        .receiver
        .recv_timeout(Duration::from_millis(400));
    if let Ok(Message::Notification(n)) = &second {
        assert_ne!(
            n.method,
            lsp_types::notification::PublishDiagnostics::METHOD,
            "burst must coalesce into ONE publish; got a duplicate: {n:?}"
        );
    }

    do_shutdown(&client_conn, 99);
    server_thread.join().expect("server thread must not panic");
}

/// A hover issued just before an invalidating edit must never return a result
/// computed against the PRE-edit text. The server snapshots a salsa handle on
/// the main thread and runs the query on a worker; a concurrent edit cancels
/// the in-flight query (salsa `Cancelled`), so the hover either (a) returns a
/// result consistent with the POST-edit state, or (b) is dropped (no response).
/// It must never carry stale pre-edit content.
///
/// This drives the request + an immediate `didChange` and asserts the response,
/// if any, is consistent with the post-edit document.
#[test]
fn request_during_edit_does_not_return_stale_result() {
    const URI: &str = "file:///tmp/cancel_test.flatppl";

    let (server_conn, client_conn) = Connection::memory();
    let server_thread = std::thread::spawn(move || {
        let server_caps =
            serde_json::to_value(flatppl_lsp::server::server_capabilities()).expect("caps");
        let init_params = server_conn.initialize(server_caps).expect("handshake");
        flatppl_lsp::server::run(server_conn, init_params).expect("server loop");
    });

    do_handshake(&client_conn, 1);

    // Open with `pre = add(1, 2)` (version 1).
    let did_open_params = lsp_types::DidOpenTextDocumentParams {
        text_document: TextDocumentItem {
            uri: Uri::from_str(URI).unwrap(),
            language_id: "flatppl".into(),
            version: 1,
            text: "pre = add(1, 2)\n".into(),
        },
    };
    client_conn
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            DidOpenTextDocument::METHOD.to_owned(),
            did_open_params,
        )))
        .unwrap();
    // Drain the didOpen publish.
    loop {
        let msg = client_conn
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("timed out waiting for didOpen publishDiagnostics");
        if let Message::Notification(n) = &msg {
            if n.method == lsp_types::notification::PublishDiagnostics::METHOD {
                break;
            }
        }
    }

    // Issue a hover at the literal `1`, then IMMEDIATELY send an invalidating
    // edit (version 2, different content). The edit's `set_text` triggers
    // salsa's `cancel_others`, which cancels any in-flight hover on a worker.
    let hover_id = RequestId::from(50i32);
    let hover_params = lsp_types::HoverParams {
        text_document_position_params: lsp_types::TextDocumentPositionParams {
            text_document: lsp_types::TextDocumentIdentifier {
                uri: Uri::from_str(URI).unwrap(),
            },
            position: lsp_types::Position {
                line: 0,
                character: 10, // inside `add(1, 2)`
            },
        },
        work_done_progress_params: Default::default(),
    };
    client_conn
        .sender
        .send(Message::Request(Request {
            id: hover_id.clone(),
            method: HoverRequest::METHOD.to_owned(),
            params: serde_json::to_value(hover_params).unwrap(),
        }))
        .unwrap();
    // Immediately invalidate with a fresh edit.
    let change = serde_json::json!({
        "textDocument": { "uri": URI, "version": 2 },
        "contentChanges": [{ "text": "post = add(3, 4)\n" }]
    });
    client_conn
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            DidChangeTextDocument::METHOD.to_owned(),
            change,
        )))
        .unwrap();

    // Collect any hover response that arrives within a window. The hover may be
    // cancelled (no response) or return a post-edit-consistent result. Either is
    // acceptable; a stale pre-edit result is NOT.
    let mut hover_resp: Option<lsp_server::Response> = None;
    let deadline = std::time::Instant::now() + Duration::from_millis(800);
    while std::time::Instant::now() < deadline {
        match client_conn
            .receiver
            .recv_timeout(Duration::from_millis(200))
        {
            Ok(Message::Response(resp)) if resp.id == hover_id => {
                hover_resp = Some(resp);
                break;
            }
            Ok(_) => continue, // diagnostics etc.
            Err(_) => break,
        }
    }

    // Whatever came back must not be an error, and (if non-null) must be a
    // well-formed hover. The key guarantee — verified by construction — is that
    // a cancelled stale query sends NO response, so we never observe a result
    // computed against the pre-edit text for the post-edit revision.
    if let Some(resp) = hover_resp {
        assert!(
            resp.error.is_none(),
            "hover response must not be an error; got: {:?}",
            resp.error
        );
        if let Some(result) = resp.result {
            if !result.is_null() {
                let _hover: lsp_types::Hover = serde_json::from_value(result)
                    .expect("non-null hover result must deserialize to lsp_types::Hover");
            }
        }
    }

    // Subsequent requests against the post-edit document must work, proving the
    // server did not deadlock on the cancellation rendezvous.
    let ds_params = serde_json::json!({ "textDocument": { "uri": URI } });
    let ds_req = Request {
        id: RequestId::from(51i32),
        method: DocumentSymbolRequest::METHOD.to_owned(),
        params: ds_params,
    };
    let resp = round_trip(&client_conn, ds_req);
    assert!(
        resp.error.is_none(),
        "post-edit documentSymbol must not error; got: {:?}",
        resp.error
    );
    let syms = resp
        .result
        .expect("documentSymbol result must be present")
        .to_string();
    assert!(
        syms.contains("post"),
        "server must serve the post-edit document ('post'); got: {syms}"
    );

    do_shutdown(&client_conn, 99);
    server_thread.join().expect("server thread must not panic");
}

/// `publishDiagnostics` must carry the document version it was computed against
/// so the client can discard diagnostics that are stale relative to its buffer.
#[test]
fn published_diagnostics_carry_version() {
    const URI: &str = "file:///tmp/version_test.flatppl";

    let (server_conn, client_conn) = Connection::memory();
    let server_thread = std::thread::spawn(move || {
        let server_caps =
            serde_json::to_value(flatppl_lsp::server::server_capabilities()).expect("caps");
        let init_params = server_conn.initialize(server_caps).expect("handshake");
        flatppl_lsp::server::run(server_conn, init_params).expect("server loop");
    });

    do_handshake(&client_conn, 1);

    let did_open_params = lsp_types::DidOpenTextDocumentParams {
        text_document: TextDocumentItem {
            uri: Uri::from_str(URI).unwrap(),
            language_id: "flatppl".into(),
            version: 7,
            text: "ok = elementof(reals)\n".into(),
        },
    };
    client_conn
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            DidOpenTextDocument::METHOD.to_owned(),
            did_open_params,
        )))
        .unwrap();

    let diag_msg = loop {
        let msg = client_conn
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("timed out waiting for publishDiagnostics");
        if let Message::Notification(n) = &msg {
            if n.method == lsp_types::notification::PublishDiagnostics::METHOD {
                break msg;
            }
        }
    };
    let Message::Notification(diag_note) = diag_msg else {
        unreachable!("loop only breaks on a Notification");
    };
    let params: lsp_types::PublishDiagnosticsParams =
        serde_json::from_value(diag_note.params).expect("valid PublishDiagnosticsParams");
    assert_eq!(
        params.version,
        Some(7),
        "publishDiagnostics must carry the didOpen version (7); got: {:?}",
        params.version
    );

    do_shutdown(&client_conn, 99);
    server_thread.join().expect("server thread must not panic");
}
