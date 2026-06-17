//! End-to-end protocol smoke test: initialize → didOpen → hover.
//!
//! Drives the full LSP handshake over an in-memory `Connection::memory()` pair
//! and verifies that a `textDocument/hover` response comes back with a non-null
//! result containing type information.

use std::str::FromStr;

use lsp_server::{Connection, Message, Request, RequestId};
use lsp_types::{
    ClientCapabilities, HoverContents, InitializeParams, InitializedParams, MarkupContent,
    MarkupKind, TextDocumentItem, Uri,
    notification::{DidOpenTextDocument, Initialized, Notification as _},
    request::{HoverRequest, Initialize, Request as _},
};

/// Source text used throughout.  The expression `add(1, 2)` is fully typed by
/// the engine; byte offset 8 lands on the literal `1` which carries an inferred
/// scalar type.
const SRC: &str = "x = add(1, 2)";

/// Byte offset of the literal `1` inside `add(1, 2)`.
/// `x = add(1, 2)`
///  0123456789...
///          ^ offset 8
const HOVER_OFFSET: u32 = 8;

/// The file URI used for the didOpen + hover requests.
const FILE_URI: &str = "file:///tmp/smoke.flatppl";

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

    // ── 3. Client side: drive the initialize handshake ───────────────────────
    //
    // `server_conn.initialize` waits for:
    //   a) an `initialize` Request from the client,
    //   b) then replies with `InitializeResult`,
    //   c) then waits for an `initialized` Notification.
    //
    // So we must send (a), read (b), then send (c).
    let init_req_id = RequestId::from(1i32);

    #[allow(deprecated)]
    let init_params_value = serde_json::to_value(InitializeParams {
        capabilities: ClientCapabilities::default(),
        ..Default::default()
    })
    .expect("serialize InitializeParams");

    let init_req = lsp_server::Request {
        id: init_req_id.clone(),
        method: Initialize::METHOD.to_owned(),
        params: init_params_value,
    };
    client_conn.sender.send(Message::Request(init_req)).unwrap();

    // Read the InitializeResult response.
    let _init_resp = client_conn
        .receiver
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("timed out waiting for InitializeResult");

    // Send `initialized` notification to complete the handshake.
    let initialized_note =
        lsp_server::Notification::new(Initialized::METHOD.to_owned(), InitializedParams {});
    client_conn
        .sender
        .send(Message::Notification(initialized_note))
        .unwrap();

    // ── 4. Send didOpen ──────────────────────────────────────────────────────
    let did_open_params = lsp_types::DidOpenTextDocumentParams {
        text_document: TextDocumentItem {
            uri: Uri::from_str(FILE_URI).unwrap(),
            language_id: "flatppl".into(),
            version: 1,
            text: SRC.into(),
        },
    };
    let did_open_note =
        lsp_server::Notification::new(DidOpenTextDocument::METHOD.to_owned(), did_open_params);
    client_conn
        .sender
        .send(Message::Notification(did_open_note))
        .unwrap();

    // ── 5. Skip the publishDiagnostics notification ──────────────────────────
    //
    // The server sends `publishDiagnostics` after every didOpen.  Drain
    // messages until we see it, then move on.
    loop {
        let msg = client_conn
            .receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("timed out waiting for publishDiagnostics");
        if let Message::Notification(n) = &msg {
            if n.method == lsp_types::notification::PublishDiagnostics::METHOD {
                break;
            }
        }
    }

    // ── 6. Send a hover request at byte offset 8 (the literal `1`) ──────────
    let hover_req_id = RequestId::from(2i32);
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
        id: hover_req_id.clone(),
        method: HoverRequest::METHOD.to_owned(),
        params: serde_json::to_value(hover_params).unwrap(),
    };
    client_conn
        .sender
        .send(Message::Request(hover_req))
        .unwrap();

    // ── 7. Read messages until the hover Response arrives ────────────────────
    let hover_response = loop {
        let msg = client_conn
            .receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("timed out waiting for hover response");
        match msg {
            Message::Response(resp) if resp.id == hover_req_id => break resp,
            // Skip any intervening notifications (there should be none, but be safe).
            _ => continue,
        }
    };

    // ── 8. Assert the hover response is non-null with type information ────────
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

    // ── 9. Shutdown + exit ───────────────────────────────────────────────────
    let shutdown_req = Request::new(
        RequestId::from(99i32),
        "shutdown".into(),
        serde_json::Value::Null,
    );
    client_conn
        .sender
        .send(Message::Request(shutdown_req))
        .unwrap();
    // Read the shutdown response.
    let _shutdown_resp = client_conn
        .receiver
        .recv_timeout(std::time::Duration::from_secs(5))
        .ok();
    // Send exit.
    let exit_note = lsp_server::Notification::new("exit".into(), serde_json::Value::Null);
    client_conn
        .sender
        .send(Message::Notification(exit_note))
        .unwrap();

    server_thread.join().expect("server thread must not panic");
}
