//! Hover reports each importer's own `common.flatppl`.
//!
//! Spec §04 "Path resolution": "Relative file paths in `load_module(...)` are
//! resolved relative to the directory of the FlatPPL file containing that
//! `load_module(...)` call". `a/model.flatppl` and `b/model.flatppl` each load
//! their own `common.flatppl`, so `ma.x` is a real and `mb.x` a boolean.
//!
//! Keyed by the directive literal, the second `common.flatppl` overwrote the
//! first in the inference bundle and BOTH refs hovered as `boolean`, with zero
//! diagnostics — a confidently wrong type with no signal. The bundle is now
//! keyed by resolved `SourceFile` path, with the declaring file recorded per
//! directive.

use std::str::FromStr;
use std::time::Duration;

use lsp_server::{Connection, Message, Request, RequestId};
use lsp_types::{
    TextDocumentItem, Uri,
    notification::{DidOpenTextDocument, Notification as _},
    request::{HoverRequest, Request as _},
};

const TOP: &str = "file:///ws/top.flatppl";

/// `(uri, source)` for the audit's `a`/`b` graph. The two `model.flatppl`
/// files are byte-identical, and so are the two directives inside them.
const GRAPH: &[(&str, &str)] = &[
    ("file:///ws/a/common.flatppl", "val = 1.5\n"),
    ("file:///ws/b/common.flatppl", "val = true\n"),
    (
        "file:///ws/a/model.flatppl",
        "c = load_module(\"common.flatppl\")\nx = c.val\n",
    ),
    (
        "file:///ws/b/model.flatppl",
        "c = load_module(\"common.flatppl\")\nx = c.val\n",
    ),
    (
        TOP,
        "ma = load_module(\"a/model.flatppl\")\n\
         mb = load_module(\"b/model.flatppl\")\n\
         ya = ma.x\n\
         yb = mb.x\n",
    ),
];

fn hover_text(client: &Connection, id: i32, line: u32, character: u32) -> String {
    let params = lsp_types::HoverParams {
        text_document_position_params: lsp_types::TextDocumentPositionParams {
            text_document: lsp_types::TextDocumentIdentifier {
                uri: Uri::from_str(TOP).unwrap(),
            },
            position: lsp_types::Position { line, character },
        },
        work_done_progress_params: Default::default(),
    };
    let req = Request {
        id: RequestId::from(id),
        method: HoverRequest::METHOD.to_owned(),
        params: serde_json::to_value(params).unwrap(),
    };
    let want = req.id.clone();
    client.sender.send(Message::Request(req)).unwrap();
    loop {
        let msg = client
            .receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("timed out waiting for the hover response");
        if let Message::Response(resp) = msg {
            if resp.id != want {
                continue;
            }
            let value = resp.result.expect("hover result");
            let hover: lsp_types::Hover = serde_json::from_value(value).expect("a hover, not null");
            let lsp_types::HoverContents::Markup(m) = hover.contents else {
                panic!("expected markup hover contents");
            };
            return m.value;
        }
    }
}

#[test]
fn hover_reads_each_importers_own_common_module() {
    let (client, server_conn) = Connection::memory();
    let server = std::thread::spawn(move || {
        // No rootUri: every file arrives through didOpen, so the test needs no
        // real workspace on disk.
        let init_params = serde_json::json!({ "capabilities": {} });
        flatppl_lsp::server::run(server_conn, init_params).expect("server loop");
    });

    for (uri, text) in GRAPH {
        let params = lsp_types::DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri: Uri::from_str(uri).unwrap(),
                language_id: "flatppl".into(),
                version: 1,
                text: (*text).into(),
            },
        };
        let note = lsp_server::Notification::new(DidOpenTextDocument::METHOD.to_owned(), params);
        client.sender.send(Message::Notification(note)).unwrap();
    }

    // `ya = ma.x` and `yb = mb.x`: character 8 is the member name `x`.
    let ya = hover_text(&client, 100, 2, 8);
    let yb = hover_text(&client, 101, 3, 8);
    assert!(
        ya.contains("`real`"),
        "`ma.x` must read a/common.flatppl's real `val`; got:\n{ya}"
    );
    assert!(
        yb.contains("`boolean`"),
        "`mb.x` must read b/common.flatppl's boolean `val`; got:\n{yb}"
    );

    let shutdown = Request::new(
        RequestId::from(200i32),
        "shutdown".into(),
        serde_json::Value::Null,
    );
    client.sender.send(Message::Request(shutdown)).unwrap();
    let _ = client.receiver.recv_timeout(Duration::from_secs(5));
    let exit = lsp_server::Notification::new("exit".into(), serde_json::Value::Null);
    client.sender.send(Message::Notification(exit)).unwrap();
    drop(client);
    server.join().expect("server thread");
}
