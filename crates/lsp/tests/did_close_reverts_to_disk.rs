//! `didClose` puts the document's truth back on disk.
//!
//! LSP 3.17 `textDocument/didClose`: "The document's truth now exists where the
//! document's uri points to (e.g. if the document's uri is a file uri the truth
//! now exists on disk)."
//!
//! The handler used to drop only the editor-open set and the version map, so an
//! unsaved buffer stayed in the salsa database and every importing module kept
//! inferring against abandoned text until an unrelated watched-file event
//! happened to arrive. Measured over stdio: a file holding `a = 1.5`, edited to
//! `a = true` and closed without saving, still hovered as `boolean` while the
//! disk said `1.5`.
//!
//! The existing protocol test never saw this because it always sends the
//! watched-file event.

use std::str::FromStr;
use std::time::Duration;

use lsp_server::{Connection, Message, Request, RequestId};
use lsp_types::{
    TextDocumentItem, Uri,
    notification::{
        DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, Notification as _,
    },
    request::{HoverRequest, Request as _},
};

/// A scratch directory unique to this test, removed on drop.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(label: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!(
            "flatppl-lsp-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

fn hover_type(client: &Connection, id: i32, uri: &str) -> Option<String> {
    let params = lsp_types::HoverParams {
        text_document_position_params: lsp_types::TextDocumentPositionParams {
            text_document: lsp_types::TextDocumentIdentifier {
                uri: Uri::from_str(uri).unwrap(),
            },
            // `a = <value>`: character 4 is the right-hand side.
            position: lsp_types::Position {
                line: 0,
                character: 4,
            },
        },
        work_done_progress_params: Default::default(),
    };
    let want = RequestId::from(id);
    client
        .sender
        .send(Message::Request(Request {
            id: want.clone(),
            method: HoverRequest::METHOD.to_owned(),
            params: serde_json::to_value(params).unwrap(),
        }))
        .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while std::time::Instant::now() < deadline {
        match client.receiver.recv_timeout(Duration::from_millis(200)) {
            Ok(Message::Response(resp)) if resp.id == want => {
                let value = resp.result?;
                if value.is_null() {
                    return None;
                }
                let hover: lsp_types::Hover = serde_json::from_value(value).ok()?;
                let lsp_types::HoverContents::Markup(m) = hover.contents else {
                    return None;
                };
                return Some(m.value);
            }
            Ok(_) => continue,
            Err(_) => continue,
        }
    }
    None
}

#[test]
fn closing_an_unsaved_buffer_restores_the_disk_text() {
    let dir = Scratch::new("didclose");
    let path = dir.0.join("m.flatppl");
    std::fs::write(&path, "a = 1.5\n").unwrap();
    let uri = format!("file://{}", path.to_string_lossy());

    let (server_conn, client) = Connection::memory();
    let server = std::thread::spawn(move || {
        let caps = serde_json::to_value(flatppl_lsp::server::server_capabilities()).expect("caps");
        let init = server_conn.initialize(caps).expect("handshake");
        flatppl_lsp::server::run(server_conn, init).expect("server loop");
    });
    client
        .sender
        .send(Message::Request(Request {
            id: RequestId::from(1i32),
            method: lsp_types::request::Initialize::METHOD.to_owned(),
            params: serde_json::json!({ "capabilities": {} }),
        }))
        .unwrap();
    client
        .receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("InitializeResult");
    client
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            lsp_types::notification::Initialized::METHOD.to_owned(),
            lsp_types::InitializedParams {},
        )))
        .unwrap();

    // Open at the disk content.
    client
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            DidOpenTextDocument::METHOD.to_owned(),
            lsp_types::DidOpenTextDocumentParams {
                text_document: TextDocumentItem {
                    uri: Uri::from_str(&uri).unwrap(),
                    language_id: "flatppl".into(),
                    version: 1,
                    text: "a = 1.5\n".into(),
                },
            },
        )))
        .unwrap();
    let at_open = hover_type(&client, 10, &uri).expect("hover at open");
    assert!(
        at_open.contains("`real`"),
        "the disk text binds a real; got:\n{at_open}"
    );

    // Edit the buffer to a different type WITHOUT saving.
    client
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            DidChangeTextDocument::METHOD.to_owned(),
            serde_json::json!({
                "textDocument": { "uri": uri, "version": 2 },
                "contentChanges": [{ "text": "a = true\n" }]
            }),
        )))
        .unwrap();
    let after_edit = hover_type(&client, 11, &uri).expect("hover after the edit");
    assert!(
        after_edit.contains("`boolean`"),
        "the unsaved buffer is the truth while the document is open; got:\n{after_edit}"
    );

    // Close without saving. The disk still holds `a = 1.5`.
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "a = 1.5\n");
    client
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            DidCloseTextDocument::METHOD.to_owned(),
            lsp_types::DidCloseTextDocumentParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: Uri::from_str(&uri).unwrap(),
                },
            },
        )))
        .unwrap();
    let after_close = hover_type(&client, 12, &uri).expect("hover after the close");
    assert!(
        after_close.contains("`real`"),
        "on close the truth is the disk again, so the type is `real` and not the \
         abandoned buffer's `boolean`; got:\n{after_close}"
    );

    client
        .sender
        .send(Message::Request(Request::new(
            RequestId::from(99i32),
            "shutdown".into(),
            serde_json::Value::Null,
        )))
        .unwrap();
    let _ = client.receiver.recv_timeout(Duration::from_secs(5));
    client
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            "exit".into(),
            serde_json::Value::Null,
        )))
        .unwrap();
    drop(client);
    server.join().expect("server thread");
}

#[test]
fn closing_a_buffer_with_no_file_behind_it_stops_tracking_it() {
    // An untitled or already-deleted document has no truth to fall back to, so
    // the server must stop analysing it rather than keep the abandoned text.
    let dir = Scratch::new("didclose-gone");
    let path = dir.0.join("gone.flatppl");
    let uri = format!("file://{}", path.to_string_lossy());

    let (server_conn, client) = Connection::memory();
    let server = std::thread::spawn(move || {
        let caps = serde_json::to_value(flatppl_lsp::server::server_capabilities()).expect("caps");
        let init = server_conn.initialize(caps).expect("handshake");
        flatppl_lsp::server::run(server_conn, init).expect("server loop");
    });
    client
        .sender
        .send(Message::Request(Request {
            id: RequestId::from(1i32),
            method: lsp_types::request::Initialize::METHOD.to_owned(),
            params: serde_json::json!({ "capabilities": {} }),
        }))
        .unwrap();
    client
        .receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("InitializeResult");
    client
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            lsp_types::notification::Initialized::METHOD.to_owned(),
            lsp_types::InitializedParams {},
        )))
        .unwrap();
    client
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            DidOpenTextDocument::METHOD.to_owned(),
            lsp_types::DidOpenTextDocumentParams {
                text_document: TextDocumentItem {
                    uri: Uri::from_str(&uri).unwrap(),
                    language_id: "flatppl".into(),
                    version: 1,
                    text: "a = 1.5\n".into(),
                },
            },
        )))
        .unwrap();
    assert!(
        hover_type(&client, 20, &uri)
            .expect("hover while open")
            .contains("`real`"),
        "the buffer is the truth while open"
    );

    client
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            DidCloseTextDocument::METHOD.to_owned(),
            lsp_types::DidCloseTextDocumentParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: Uri::from_str(&uri).unwrap(),
                },
            },
        )))
        .unwrap();
    assert!(
        hover_type(&client, 21, &uri).is_none(),
        "with no file on disk there is no document left to hover"
    );

    client
        .sender
        .send(Message::Request(Request::new(
            RequestId::from(99i32),
            "shutdown".into(),
            serde_json::Value::Null,
        )))
        .unwrap();
    let _ = client.receiver.recv_timeout(Duration::from_secs(5));
    client
        .sender
        .send(Message::Notification(lsp_server::Notification::new(
            "exit".into(),
            serde_json::Value::Null,
        )))
        .unwrap();
    drop(client);
    server.join().expect("server thread");
}
