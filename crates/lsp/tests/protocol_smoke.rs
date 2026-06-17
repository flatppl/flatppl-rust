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
    notification::{DidOpenTextDocument, Initialized, Notification as _},
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
    // scalar type token ("Integer", "Real", "Scalar", or "scalar").
    assert!(
        markdown.to_lowercase().contains("type"),
        "hover markdown must mention 'type'; got: {markdown:?}"
    );
    let has_scalar_token = ["Integer", "Real", "Scalar", "scalar", "Complex"]
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
