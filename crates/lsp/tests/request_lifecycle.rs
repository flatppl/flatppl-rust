//! Every request gets exactly one response, with the code LSP 3.17 defines.
//!
//! JSON-RPC 2.0 §5: "When a rpc call is made, the Server MUST reply with a
//! Response, except for in the case of Notifications." The server used to drop
//! a request three ways: a salsa cancellation replied nothing, `$/cancelRequest`
//! fell into the unknown-notification arm, and eight handlers folded a params
//! parse failure into an empty success. Measured over stdio, 500 requests
//! followed by one edit produced 4 responses in 60 seconds.
//!
//! The codes, all from LSP 3.17 "Response Message" / `LSPErrorCodes`:
//!
//!   -32602 InvalidParams     the params did not deserialize
//!   -32800 RequestCancelled  the client cancelled it
//!   -32801 ContentModified   an edit invalidated the revision
//!   -32803 RequestFailed     the server queue is full

use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::time::{Duration, Instant};

use lsp_server::{Connection, ErrorCode, Message, Request, RequestId, Response};
use lsp_types::{
    TextDocumentItem, Uri,
    notification::{DidChangeTextDocument, DidOpenTextDocument, Notification as _},
    request::{DocumentSymbolRequest, HoverRequest, Request as _},
};

const URI: &str = "file:///ws/lifecycle.flatppl";

/// A source big enough that a `documentSymbol` is not instant, so a flood
/// actually queues.
fn source(bindings: usize) -> String {
    (0..bindings)
        .map(|i| format!("b{i} = {}.0 + 1.0\n", i))
        .collect()
}

struct Server {
    client: Connection,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Server {
    fn start() -> Server {
        let (server_conn, client) = Connection::memory();
        let thread = std::thread::spawn(move || {
            let caps = serde_json::to_value(flatppl_lsp::server::server_capabilities())
                .expect("capabilities");
            let init = server_conn.initialize(caps).expect("handshake");
            flatppl_lsp::server::run(server_conn, init).expect("server loop");
        });
        // Client side of the handshake.
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
        Server {
            client,
            thread: Some(thread),
        }
    }

    fn notify(&self, method: &str, params: serde_json::Value) {
        self.client
            .sender
            .send(Message::Notification(lsp_server::Notification {
                method: method.to_owned(),
                params,
            }))
            .unwrap();
    }

    fn request(&self, id: i32, method: &str, params: serde_json::Value) -> RequestId {
        let id = RequestId::from(id);
        self.client
            .sender
            .send(Message::Request(Request {
                id: id.clone(),
                method: method.to_owned(),
                params,
            }))
            .unwrap();
        id
    }

    fn open(&self, text: &str) {
        self.notify(
            DidOpenTextDocument::METHOD,
            serde_json::to_value(lsp_types::DidOpenTextDocumentParams {
                text_document: TextDocumentItem {
                    uri: Uri::from_str(URI).unwrap(),
                    language_id: "flatppl".into(),
                    version: 1,
                    text: text.into(),
                },
            })
            .unwrap(),
        );
    }

    /// Collect responses until every id in `want` has one, or `budget` expires.
    /// Panics on a second response for the same id.
    fn collect(&self, want: &HashSet<RequestId>, budget: Duration) -> HashMap<RequestId, Response> {
        let mut got: HashMap<RequestId, Response> = HashMap::new();
        let deadline = Instant::now() + budget;
        while got.len() < want.len() && Instant::now() < deadline {
            match self
                .client
                .receiver
                .recv_timeout(Duration::from_millis(200))
            {
                Ok(Message::Response(resp)) => {
                    if !want.contains(&resp.id) {
                        continue;
                    }
                    assert!(
                        got.insert(resp.id.clone(), resp.clone()).is_none(),
                        "two responses for {:?}; JSON-RPC allows exactly one",
                        resp.id
                    );
                }
                Ok(_) => continue, // diagnostics
                Err(_) => continue,
            }
        }
        got
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.client
            .sender
            .send(Message::Request(Request::new(
                RequestId::from(9999i32),
                "shutdown".into(),
                serde_json::Value::Null,
            )))
            .ok();
        let _ = self.client.receiver.recv_timeout(Duration::from_secs(5));
        self.client
            .sender
            .send(Message::Notification(lsp_server::Notification::new(
                "exit".into(),
                serde_json::Value::Null,
            )))
            .ok();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn code_of(resp: &Response) -> i32 {
    resp.error.as_ref().map(|e| e.code).unwrap_or(0)
}

/// A flood of requests followed by an edit: every single request is answered,
/// and every answer is a success or one of the four defined codes. Measured
/// over stdio before this change, 496 of 500 were never answered at all.
#[test]
fn a_request_flood_across_an_edit_answers_every_request() {
    let server = Server::start();
    server.open(&source(400));
    let ids: HashSet<RequestId> = (100..500i32)
        .map(|i| {
            server.request(
                i,
                DocumentSymbolRequest::METHOD,
                serde_json::json!({ "textDocument": { "uri": URI } }),
            )
        })
        .collect();
    server.notify(
        DidChangeTextDocument::METHOD,
        serde_json::json!({
            "textDocument": { "uri": URI, "version": 2 },
            "contentChanges": [{ "text": source(401) }]
        }),
    );

    let got = server.collect(&ids, Duration::from_secs(60));
    assert_eq!(
        got.len(),
        ids.len(),
        "{} of {} requests were never answered",
        ids.len() - got.len(),
        ids.len()
    );
    let allowed = [
        0, // success
        ErrorCode::ContentModified as i32,
        ErrorCode::RequestCanceled as i32,
        ErrorCode::RequestFailed as i32,
    ];
    for resp in got.values() {
        let code = code_of(resp);
        assert!(
            allowed.contains(&code),
            "unexpected code {code} for {:?}: {:?}",
            resp.id,
            resp.error
        );
    }
}

/// `$/cancelRequest` is honoured with `RequestCancelled` (-32800). Cancelling
/// before the request arrives is the deterministic ordering: the cancel is
/// recorded, and the request is answered from it instead of being dispatched.
/// Without that check the request would be dropped entirely — the main thread
/// had nothing to answer and the job would see the id cancelled and return.
#[test]
fn a_cancelled_request_is_answered_request_cancelled() {
    let server = Server::start();
    server.open(&source(10));
    server.notify("$/cancelRequest", serde_json::json!({ "id": 300 }));
    let id = server.request(
        300,
        DocumentSymbolRequest::METHOD,
        serde_json::json!({ "textDocument": { "uri": URI } }),
    );
    let got = server.collect(&HashSet::from([id.clone()]), Duration::from_secs(10));
    let resp = got
        .get(&id)
        .expect("a cancelled request still gets a response");
    assert_eq!(
        code_of(resp),
        ErrorCode::RequestCanceled as i32,
        "expected RequestCancelled (-32800); got {:?}",
        resp.error
    );
}

/// A malformed payload is `InvalidParams` (-32602), not an empty success.
/// All eight handlers that used to swallow the parse failure are covered; the
/// old behaviour was indistinguishable from a file with nothing to report.
#[test]
fn malformed_params_are_invalid_params_not_an_empty_success() {
    let server = Server::start();
    server.open(&source(10));
    // `textDocument` is required by every one of these; a bare object fails to
    // deserialize for all of them.
    let bad = serde_json::json!({ "nonsense": true });
    let methods = [
        HoverRequest::METHOD,
        DocumentSymbolRequest::METHOD,
        lsp_types::request::InlayHintRequest::METHOD,
        lsp_types::request::GotoDefinition::METHOD,
        lsp_types::request::Completion::METHOD,
        lsp_types::request::References::METHOD,
        lsp_types::request::PrepareRenameRequest::METHOD,
        lsp_types::request::SignatureHelpRequest::METHOD,
        lsp_types::request::Rename::METHOD,
        lsp_types::request::WorkspaceSymbolRequest::METHOD,
    ];
    let mut ids = HashSet::new();
    let mut by_method = HashMap::new();
    for (i, method) in methods.iter().enumerate() {
        let id = server.request(700 + i as i32, method, bad.clone());
        ids.insert(id.clone());
        by_method.insert(id, *method);
    }
    let got = server.collect(&ids, Duration::from_secs(20));
    assert_eq!(got.len(), ids.len(), "every handler must answer");
    for (id, resp) in &got {
        let method = by_method[id];
        assert_eq!(
            code_of(resp),
            ErrorCode::InvalidParams as i32,
            "{method} must answer InvalidParams (-32602); got {:?} / {:?}",
            resp.error,
            resp.result
        );
    }
}
